// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Video decoding UDF: decode_h264(blob_column) -> LargeBinary (RGB frames)

use std::any::Any;
use std::cell::RefCell;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, LargeBinaryArray, StructArray};
use arrow_schema::DataType;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use datafusion::error::Result as DFResult;
use rayon::prelude::*;

use ffmpeg_next as ffmpeg;

// =============================================================================
// AVIO memory reader
// =============================================================================

struct BlobReader {
    data: *const u8,
    len: usize,
    pos: usize,
}

unsafe extern "C" fn read_blob(
    opaque: *mut std::ffi::c_void,
    buf: *mut u8,
    buf_size: std::ffi::c_int,
) -> std::ffi::c_int {
    let ctx = &mut *(opaque as *mut BlobReader);
    let remaining = ctx.len - ctx.pos;
    let to_read = std::cmp::min(remaining, buf_size as usize);
    if to_read == 0 {
        return ffmpeg_sys_next::AVERROR_EOF;
    }
    std::ptr::copy_nonoverlapping(ctx.data.add(ctx.pos), buf, to_read);
    ctx.pos += to_read;
    to_read as std::ffi::c_int
}

unsafe extern "C" fn seek_blob(
    opaque: *mut std::ffi::c_void,
    offset: i64,
    whence: std::ffi::c_int,
) -> i64 {
    let ctx = &mut *(opaque as *mut BlobReader);
    match whence {
        ffmpeg_sys_next::AVSEEK_SIZE => ctx.len as i64,
        0 => { ctx.pos = offset as usize; offset }
        1 => { ctx.pos = (ctx.pos as i64 + offset) as usize; ctx.pos as i64 }
        2 => { ctx.pos = (ctx.len as i64 + offset) as usize; ctx.pos as i64 }
        _ => -1,
    }
}

// =============================================================================
// Thread-local cached resources: SwsContext + AVFrame
// =============================================================================

struct CachedSws {
    ctx: *mut ffmpeg_sys_next::SwsContext,
    w: i32,
    h: i32,
    src_fmt: ffmpeg_sys_next::AVPixelFormat,
}

impl Drop for CachedSws {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { ffmpeg_sys_next::sws_freeContext(self.ctx); }
        }
    }
}

struct CachedFrame {
    frame: *mut ffmpeg_sys_next::AVFrame,
}

impl Drop for CachedFrame {
    fn drop(&mut self) {
        if !self.frame.is_null() {
            unsafe { ffmpeg_sys_next::av_frame_free(&mut self.frame); }
        }
    }
}

// SAFETY: these are only accessed from the thread that created them (thread_local)
unsafe impl Send for CachedSws {}
unsafe impl Send for CachedFrame {}

thread_local! {
    static TL_SWS: RefCell<Option<CachedSws>> = const { RefCell::new(None) };
    static TL_FRAME: RefCell<Option<CachedFrame>> = const { RefCell::new(None) };
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
        // Create new
        let ctx = ffmpeg_sys_next::sws_getContext(
            w, h, src_fmt,
            w, h, ffmpeg_sys_next::AVPixelFormat::AV_PIX_FMT_RGB24,
            0, // no flags for pure color conversion (like torchcodec)
            std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null(),
        );
        *opt = Some(CachedSws { ctx, w, h, src_fmt });
        ctx
    })
}

unsafe fn get_or_create_frame() -> *mut ffmpeg_sys_next::AVFrame {
    TL_FRAME.with(|cell| {
        let mut opt = cell.borrow_mut();
        if let Some(ref cached) = *opt {
            return cached.frame;
        }
        let f = ffmpeg_sys_next::av_frame_alloc();
        *opt = Some(CachedFrame { frame: f });
        f
    })
}

// =============================================================================
// Core decode function
// =============================================================================

static FFMPEG_INIT: std::sync::Once = std::sync::Once::new();

pub fn decode_h264_frames(blob: &[u8]) -> std::result::Result<Vec<u8>, String> {
    FFMPEG_INIT.call_once(|| { let _ = ffmpeg::init(); });

    unsafe {
        // AVIO setup
        let buf_size: usize = 64 * 1024;
        let avio_buf = ffmpeg_sys_next::av_malloc(buf_size) as *mut u8;
        if avio_buf.is_null() {
            return Err("av_malloc failed".into());
        }

        let mut reader = Box::new(BlobReader {
            data: blob.as_ptr(),
            len: blob.len(),
            pos: 0,
        });
        let reader_ptr = &mut *reader as *mut BlobReader as *mut std::ffi::c_void;

        let avio_ctx = ffmpeg_sys_next::avio_alloc_context(
            avio_buf, buf_size as std::ffi::c_int,
            0, reader_ptr,
            Some(read_blob), None, Some(seek_blob),
        );
        if avio_ctx.is_null() {
            ffmpeg_sys_next::av_free(avio_buf as *mut std::ffi::c_void);
            return Err("avio_alloc_context failed".into());
        }

        // Format context
        let mut fmt_ctx = ffmpeg_sys_next::avformat_alloc_context();
        if fmt_ctx.is_null() {
            ffmpeg_sys_next::av_free(avio_buf as *mut std::ffi::c_void);
            return Err("avformat_alloc_context failed".into());
        }
        (*fmt_ctx).pb = avio_ctx;
        (*fmt_ctx).probesize = 1024;
        (*fmt_ctx).max_analyze_duration = 0;

        let fmt_name = std::ffi::CString::new("mp4").unwrap();
        let input_fmt = ffmpeg_sys_next::av_find_input_format(fmt_name.as_ptr());

        let ret = ffmpeg_sys_next::avformat_open_input(
            &mut fmt_ctx, std::ptr::null(), input_fmt, std::ptr::null_mut(),
        );
        if ret < 0 {
            ffmpeg_sys_next::avio_context_free(&mut (avio_ctx as *mut _));
            return Err(format!("avformat_open_input: {ret}"));
        }

        let ret = ffmpeg_sys_next::avformat_find_stream_info(fmt_ctx, std::ptr::null_mut());
        if ret < 0 {
            ffmpeg_sys_next::avformat_close_input(&mut fmt_ctx);
            return Err(format!("avformat_find_stream_info: {ret}"));
        }

        // Find video stream
        let mut video_stream_idx: i32 = -1;
        for i in 0..(*fmt_ctx).nb_streams as usize {
            let stream = *(*fmt_ctx).streams.add(i);
            if (*(*stream).codecpar).codec_type == ffmpeg_sys_next::AVMediaType::AVMEDIA_TYPE_VIDEO {
                video_stream_idx = i as i32;
                break;
            }
        }
        if video_stream_idx < 0 {
            ffmpeg_sys_next::avformat_close_input(&mut fmt_ctx);
            return Err("no video stream".into());
        }

        let stream = *(*fmt_ctx).streams.add(video_stream_idx as usize);
        let codecpar = (*stream).codecpar;

        // Open decoder
        let codec = ffmpeg_sys_next::avcodec_find_decoder((*codecpar).codec_id);
        if codec.is_null() {
            ffmpeg_sys_next::avformat_close_input(&mut fmt_ctx);
            return Err("codec not found".into());
        }
        let codec_ctx = ffmpeg_sys_next::avcodec_alloc_context3(codec);
        if codec_ctx.is_null() {
            ffmpeg_sys_next::avformat_close_input(&mut fmt_ctx);
            return Err("avcodec_alloc_context3 failed".into());
        }
        ffmpeg_sys_next::avcodec_parameters_to_context(codec_ctx, codecpar);
        let ret = ffmpeg_sys_next::avcodec_open2(codec_ctx, codec, std::ptr::null_mut());
        if ret < 0 {
            ffmpeg_sys_next::avcodec_free_context(&mut (codec_ctx as *mut _));
            ffmpeg_sys_next::avformat_close_input(&mut fmt_ctx);
            return Err(format!("avcodec_open2: {ret}"));
        }

        let w = (*codec_ctx).width;
        let h = (*codec_ctx).height;
        let src_fmt = (*codec_ctx).pix_fmt;

        // Get cached SwsContext and AVFrame
        let sws_ctx = get_or_create_sws(w, h, src_fmt);
        if sws_ctx.is_null() {
            ffmpeg_sys_next::avcodec_free_context(&mut (codec_ctx as *mut _));
            ffmpeg_sys_next::avformat_close_input(&mut fmt_ctx);
            return Err("sws_getContext failed".into());
        }

        let frame = get_or_create_frame();
        let mut packet = ffmpeg_sys_next::av_packet_alloc();

        let width_bytes = (w * 3) as usize;
        let frame_size = width_bytes * h as usize;
        let mut rgb_data = Vec::with_capacity(frame_size * 8);
        let dst_linesize: [i32; 4] = [width_bytes as i32, 0, 0, 0];

        let mut scale_frame = |frame: *mut ffmpeg_sys_next::AVFrame| {
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
        };

        // Decode loop
        while ffmpeg_sys_next::av_read_frame(fmt_ctx, packet) >= 0 {
            if (*packet).stream_index == video_stream_idx {
                ffmpeg_sys_next::avcodec_send_packet(codec_ctx, packet);
                while ffmpeg_sys_next::avcodec_receive_frame(codec_ctx, frame) == 0 {
                    scale_frame(frame);
                }
            }
            ffmpeg_sys_next::av_packet_unref(packet);
        }

        // Flush
        ffmpeg_sys_next::avcodec_send_packet(codec_ctx, std::ptr::null());
        while ffmpeg_sys_next::avcodec_receive_frame(codec_ctx, frame) == 0 {
            scale_frame(frame);
        }

        // Cleanup (frame and sws_ctx are thread-local cached, don't free)
        ffmpeg_sys_next::av_packet_free(&mut packet);
        ffmpeg_sys_next::avcodec_free_context(&mut (codec_ctx as *mut _));
        // Save pb pointer before avformat_close_input frees fmt_ctx
        let pb = (*fmt_ctx).pb;
        ffmpeg_sys_next::avformat_close_input(&mut fmt_ctx);
        // Manually free user-allocated AVIO context and its buffer.
        // avformat_close_input does NOT free a user-supplied pb.
        if !pb.is_null() {
            // Free the internal read buffer (allocated via av_malloc)
            if !(*pb).buffer.is_null() {
                ffmpeg_sys_next::av_freep(
                    &mut (*pb).buffer as *mut *mut u8 as *mut std::ffi::c_void,
                );
            }
            ffmpeg_sys_next::avio_context_free(&mut (pb as *mut _));
        }
        drop(reader);

        Ok(rgb_data)
    }
}

// =============================================================================
// DataFusion UDF (unchanged)
// =============================================================================

#[derive(Debug, Hash, Eq, PartialEq)]
struct DecodeH264Udf {
    signature: Signature,
}

impl DecodeH264Udf {
    fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for DecodeH264Udf {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "decode_h264" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::LargeBinary)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let arr = match &args.args[0] {
            ColumnarValue::Array(a) => a.clone(),
            ColumnarValue::Scalar(s) => s.to_array()?,
        };

        let binary_arr: &LargeBinaryArray = if let Some(struct_arr) = arr.as_any().downcast_ref::<StructArray>() {
            struct_arr.column_by_name("data")
                .ok_or_else(|| datafusion::error::DataFusionError::Execution("blob struct missing 'data' field".into()))?
                .as_any().downcast_ref::<LargeBinaryArray>()
                .ok_or_else(|| datafusion::error::DataFusionError::Execution("blob 'data' field is not LargeBinary".into()))?
        } else {
            arr.as_any().downcast_ref::<LargeBinaryArray>()
                .ok_or_else(|| datafusion::error::DataFusionError::Execution("decode_h264 expects LargeBinary or blob Struct input".into()))?
        };

        let results: Vec<Option<Result<Vec<u8>, String>>> = (0..binary_arr.len())
            .into_par_iter()
            .map(|i| {
                if binary_arr.is_null(i) { None }
                else { Some(decode_h264_frames(binary_arr.value(i))) }
            })
            .collect();

        let mut builder = arrow_array::builder::LargeBinaryBuilder::new();
        for (i, result) in results.into_iter().enumerate() {
            match result {
                None => builder.append_null(),
                Some(Ok(rgb)) => builder.append_value(&rgb),
                Some(Err(e)) => {
                    return Err(datafusion::error::DataFusionError::Execution(
                        format!("decode_h264 failed on row {i}: {e}")
                    ));
                }
            }
        }

        Ok(ColumnarValue::Array(Arc::new(builder.finish()) as ArrayRef))
    }
}

pub fn decode_h264_udf() -> ScalarUDF {
    ScalarUDF::new_from_impl(DecodeH264Udf::new())
}
