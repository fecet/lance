// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Video decoding: raw H264 Annex B → RGB24 frames.
//!
//! Blob format: raw H264 Annex B byte stream (SPS + PPS + IDR + P-frames).
//! No MP4 container — avformat is not used.
//! Codec context is cached per-thread for zero init overhead.

use std::any::Any;
use std::cell::RefCell;
use std::sync::{Arc, LazyLock};

/// Dedicated rayon thread pool for H264 decoding.
pub static DECODE_POOL: LazyLock<rayon::ThreadPool> = LazyLock::new(|| {
    let n = std::env::var("LANCE_DECODE_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("lance-decode-{i}"))
        .build()
        .expect("failed to create decode thread pool")
});

use arrow_array::{Array, ArrayRef, LargeBinaryArray, StructArray};
use arrow_schema::DataType;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use datafusion::error::Result as DFResult;
use rayon::prelude::*;

use ffmpeg_next as ffmpeg;

// =============================================================================
// Thread-local cached resources
// =============================================================================

struct CachedSws {
    ctx: *mut ffmpeg_sys_next::SwsContext,
    w: i32,
    h: i32,
    src_fmt: ffmpeg_sys_next::AVPixelFormat,
}
impl Drop for CachedSws {
    fn drop(&mut self) {
        if !self.ctx.is_null() { unsafe { ffmpeg_sys_next::sws_freeContext(self.ctx); } }
    }
}

struct CachedFrame {
    frame: *mut ffmpeg_sys_next::AVFrame,
}
impl Drop for CachedFrame {
    fn drop(&mut self) {
        if !self.frame.is_null() { unsafe { ffmpeg_sys_next::av_frame_free(&mut self.frame); } }
    }
}

struct CachedCodec {
    ctx: *mut ffmpeg_sys_next::AVCodecContext,
}
impl Drop for CachedCodec {
    fn drop(&mut self) {
        if !self.ctx.is_null() { unsafe { ffmpeg_sys_next::avcodec_free_context(&mut self.ctx); } }
    }
}

unsafe impl Send for CachedSws {}
unsafe impl Send for CachedFrame {}
unsafe impl Send for CachedCodec {}

thread_local! {
    static TL_SWS: RefCell<Option<CachedSws>> = const { RefCell::new(None) };
    static TL_FRAME: RefCell<Option<CachedFrame>> = const { RefCell::new(None) };
    static TL_CODEC: RefCell<Option<CachedCodec>> = const { RefCell::new(None) };
}

unsafe fn get_or_create_sws(
    w: i32, h: i32, src_fmt: ffmpeg_sys_next::AVPixelFormat,
) -> *mut ffmpeg_sys_next::SwsContext {
    TL_SWS.with(|cell| {
        let mut opt = cell.borrow_mut();
        if let Some(ref cached) = *opt {
            if cached.w == w && cached.h == h && cached.src_fmt == src_fmt {
                return cached.ctx;
            }
        }
        let ctx = ffmpeg_sys_next::sws_getContext(
            w, h, src_fmt, w, h,
            ffmpeg_sys_next::AVPixelFormat::AV_PIX_FMT_RGB24,
            0, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null(),
        );
        *opt = Some(CachedSws { ctx, w, h, src_fmt });
        ctx
    })
}

unsafe fn get_or_create_frame() -> *mut ffmpeg_sys_next::AVFrame {
    TL_FRAME.with(|cell| {
        let mut opt = cell.borrow_mut();
        if let Some(ref cached) = *opt { return cached.frame; }
        let f = ffmpeg_sys_next::av_frame_alloc();
        *opt = Some(CachedFrame { frame: f });
        f
    })
}

unsafe fn get_or_create_h264_ctx()
    -> std::result::Result<*mut ffmpeg_sys_next::AVCodecContext, String>
{
    TL_CODEC.with(|cell| {
        let mut opt = cell.borrow_mut();
        if let Some(ref cached) = *opt {
            ffmpeg_sys_next::avcodec_flush_buffers(cached.ctx);
            return Ok(cached.ctx);
        }
        let codec = ffmpeg_sys_next::avcodec_find_decoder(
            ffmpeg_sys_next::AVCodecID::AV_CODEC_ID_H264,
        );
        if codec.is_null() { return Err("H264 decoder not found".into()); }
        let ctx = ffmpeg_sys_next::avcodec_alloc_context3(codec);
        if ctx.is_null() { return Err("avcodec_alloc_context3 failed".into()); }
        (*ctx).thread_count = 1;
        let ret = ffmpeg_sys_next::avcodec_open2(ctx, codec, std::ptr::null_mut());
        if ret < 0 {
            ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
            return Err(format!("avcodec_open2: {ret}"));
        }
        *opt = Some(CachedCodec { ctx });
        Ok(ctx)
    })
}

// =============================================================================
// Core decode: raw H264 Annex B → RGB24
// =============================================================================

static FFMPEG_INIT: std::sync::Once = std::sync::Once::new();

/// Scale one decoded AVFrame to RGB24 and append to rgb_data.
/// Returns the frame_size (bytes per frame).
unsafe fn scale_frame_rgb(
    frame: *mut ffmpeg_sys_next::AVFrame,
    sws_ctx: *mut ffmpeg_sys_next::SwsContext,
    w: i32, h: i32,
    rgb_data: &mut Vec<u8>,
) -> usize {
    let width_bytes = (w * 3) as usize;
    let frame_size = width_bytes * h as usize;
    let dst_linesize: [i32; 4] = [width_bytes as i32, 0, 0, 0];
    let old_len = rgb_data.len();
    rgb_data.resize(old_len + frame_size, 0);
    let dst_ptr = rgb_data.as_mut_ptr().add(old_len);
    let dst_slices: [*mut u8; 4] = [dst_ptr, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut()];
    ffmpeg_sys_next::sws_scale(
        sws_ctx,
        (*frame).data.as_ptr() as *const *const u8,
        (*frame).linesize.as_ptr(),
        0, h,
        dst_slices.as_ptr(),
        dst_linesize.as_ptr(),
    );
    frame_size
}

pub fn decode_h264_frames(
    blob: &[u8],
    target_frame: Option<usize>,
) -> std::result::Result<Vec<u8>, String> {
    FFMPEG_INIT.call_once(|| { let _ = ffmpeg::init(); });

    unsafe {
        let codec_ctx = get_or_create_h264_ctx()?;
        let frame = get_or_create_frame();
        let mut packet = ffmpeg_sys_next::av_packet_alloc();

        let parser = ffmpeg_sys_next::av_parser_init(
            ffmpeg_sys_next::AVCodecID::AV_CODEC_ID_H264 as std::ffi::c_int,
        );
        if parser.is_null() {
            ffmpeg_sys_next::av_packet_free(&mut packet);
            return Err("av_parser_init failed".into());
        }

        let mut data_ptr = blob.as_ptr();
        let mut data_remaining = blob.len() as std::ffi::c_int;

        let mut rgb_data: Vec<u8> = Vec::new();
        let mut frame_count: usize = 0;
        let mut done = false;

        // sws lazy init state
        let mut sws_ctx: *mut ffmpeg_sys_next::SwsContext = std::ptr::null_mut();
        let mut w: i32 = 0;
        let mut h: i32 = 0;
        let mut frame_size: usize = 0;

        // Process one decoded frame: scale if needed, advance counter
        macro_rules! handle_frame {
            () => {
                // Lazy init sws on first frame
                if sws_ctx.is_null() {
                    w = (*codec_ctx).width;
                    h = (*codec_ctx).height;
                    sws_ctx = get_or_create_sws(w, h, (*codec_ctx).pix_fmt);
                    frame_size = (w * 3) as usize * h as usize;
                    let cap = if target_frame.is_some() { frame_size } else { frame_size * 8 };
                    rgb_data.reserve(cap);
                }
                match target_frame {
                    Some(tf) => {
                        if frame_count == tf {
                            scale_frame_rgb(frame, sws_ctx, w, h, &mut rgb_data);
                            done = true;
                        }
                        frame_count += 1;
                    }
                    None => {
                        scale_frame_rgb(frame, sws_ctx, w, h, &mut rgb_data);
                        frame_count += 1;
                    }
                }
            };
        }

        // Parse loop
        while data_remaining > 0 && !done {
            let mut pkt_data: *mut u8 = std::ptr::null_mut();
            let mut pkt_size: std::ffi::c_int = 0;

            let consumed = ffmpeg_sys_next::av_parser_parse2(
                parser, codec_ctx,
                &mut pkt_data, &mut pkt_size,
                data_ptr, data_remaining,
                ffmpeg_sys_next::AV_NOPTS_VALUE,
                ffmpeg_sys_next::AV_NOPTS_VALUE,
                0,
            );
            if consumed < 0 { break; }

            data_ptr = data_ptr.add(consumed as usize);
            data_remaining -= consumed;

            if pkt_size > 0 {
                (*packet).data = pkt_data;
                (*packet).size = pkt_size;
                ffmpeg_sys_next::avcodec_send_packet(codec_ctx, packet);
                while !done && ffmpeg_sys_next::avcodec_receive_frame(codec_ctx, frame) == 0 {
                    handle_frame!();
                }
            }
        }

        // Flush parser
        if !done {
            let mut pkt_data: *mut u8 = std::ptr::null_mut();
            let mut pkt_size: std::ffi::c_int = 0;
            ffmpeg_sys_next::av_parser_parse2(
                parser, codec_ctx,
                &mut pkt_data, &mut pkt_size,
                std::ptr::null(), 0,
                ffmpeg_sys_next::AV_NOPTS_VALUE,
                ffmpeg_sys_next::AV_NOPTS_VALUE,
                0,
            );
            if pkt_size > 0 {
                (*packet).data = pkt_data;
                (*packet).size = pkt_size;
                ffmpeg_sys_next::avcodec_send_packet(codec_ctx, packet);
                while !done && ffmpeg_sys_next::avcodec_receive_frame(codec_ctx, frame) == 0 {
                    handle_frame!();
                }
            }
        }

        // Flush decoder
        if !done {
            ffmpeg_sys_next::avcodec_send_packet(codec_ctx, std::ptr::null());
            while !done && ffmpeg_sys_next::avcodec_receive_frame(codec_ctx, frame) == 0 {
                handle_frame!();
            }
        }

        // Clamp: target_frame > actual frames — return last frame
        if target_frame.is_some() && rgb_data.is_empty() && frame_count > 0 {
            ffmpeg_sys_next::av_parser_close(parser);
            ffmpeg_sys_next::av_packet_free(&mut packet);
            let all_rgb = decode_h264_frames(blob, None)?;
            if frame_size > 0 && !all_rgb.is_empty() {
                let start = all_rgb.len().saturating_sub(frame_size);
                return Ok(all_rgb[start..].to_vec());
            }
            return Ok(all_rgb);
        }

        ffmpeg_sys_next::av_parser_close(parser);
        ffmpeg_sys_next::av_packet_free(&mut packet);

        Ok(rgb_data)
    }
}

// =============================================================================
// DataFusion UDF
// =============================================================================

#[derive(Debug, Hash, Eq, PartialEq)]
struct DecodeH264Udf {
    signature: Signature,
}
impl DecodeH264Udf {
    fn new() -> Self {
        Self { signature: Signature::any(1, Volatility::Immutable) }
    }
}

impl ScalarUDFImpl for DecodeH264Udf {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "decode_h264" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _: &[DataType]) -> DFResult<DataType> { Ok(DataType::LargeBinary) }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let arr = match &args.args[0] {
            ColumnarValue::Array(a) => a.clone(),
            ColumnarValue::Scalar(s) => s.to_array()?,
        };
        let binary_arr: &LargeBinaryArray = if let Some(s) = arr.as_any().downcast_ref::<StructArray>() {
            s.column_by_name("data")
                .ok_or_else(|| datafusion::error::DataFusionError::Execution("blob struct missing 'data'".into()))?
                .as_any().downcast_ref::<LargeBinaryArray>()
                .ok_or_else(|| datafusion::error::DataFusionError::Execution("'data' not LargeBinary".into()))?
        } else {
            arr.as_any().downcast_ref::<LargeBinaryArray>()
                .ok_or_else(|| datafusion::error::DataFusionError::Execution("expected LargeBinary".into()))?
        };

        let results: Vec<Option<Result<Vec<u8>, String>>> = DECODE_POOL.install(|| {
            (0..binary_arr.len()).into_par_iter().map(|i| {
                if binary_arr.is_null(i) { None }
                else { Some(decode_h264_frames(binary_arr.value(i), None)) }
            }).collect()
        });

        let mut builder = arrow_array::builder::LargeBinaryBuilder::new();
        for (i, r) in results.into_iter().enumerate() {
            match r {
                None => builder.append_null(),
                Some(Ok(rgb)) => builder.append_value(&rgb),
                Some(Err(e)) => return Err(datafusion::error::DataFusionError::Execution(
                    format!("decode_h264 row {i}: {e}")
                )),
            }
        }
        Ok(ColumnarValue::Array(Arc::new(builder.finish()) as ArrayRef))
    }
}

pub fn decode_h264_udf() -> ScalarUDF {
    ScalarUDF::new_from_impl(DecodeH264Udf::new())
}
