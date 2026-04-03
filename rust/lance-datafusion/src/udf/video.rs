// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Video decoding: raw H264 Annex B → RGB24 frames.
//!
//! Blob format: raw H264 Annex B byte stream (SPS + PPS + IDR + P-frames).
//! No MP4 container — avformat is not used.
//!
//! Two decode paths:
//!   CPU (default): software H264 decoder, thread-local cached, rayon-parallel.
//!   GPU (LANCE_GPU_DECODE=1): h264_cuvid via NVDEC, single session behind Mutex.
//!
//! GPU path uses one NVDEC session (serial) — the hardware itself pipelines
//! internally, so multiple sessions don't help.  CPU fallback is automatic if
//! h264_cuvid is unavailable.

use std::any::Any;
use std::cell::RefCell;
use std::sync::{Arc, LazyLock, Mutex};

/// Whether GPU decode is requested via `LANCE_GPU_DECODE=1`.
pub static USE_GPU_DECODE: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("LANCE_GPU_DECODE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
});

/// Dedicated rayon thread pool for CPU H264 decoding.
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
// Thread-local cached resources (CPU path)
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
// GPU (h264_cuvid) cached resources — single session behind Mutex
// =============================================================================

struct GpuCodecState {
    hw_device_ctx: *mut ffmpeg_sys_next::AVBufferRef,
    frame: *mut ffmpeg_sys_next::AVFrame,
    sw_frame: *mut ffmpeg_sys_next::AVFrame,
    sws_ctx: *mut ffmpeg_sys_next::SwsContext,
    sws_w: i32,
    sws_h: i32,
    sws_src_fmt: ffmpeg_sys_next::AVPixelFormat,
}

unsafe impl Send for GpuCodecState {}

impl Drop for GpuCodecState {
    fn drop(&mut self) {
        unsafe {
            if !self.sws_ctx.is_null() { ffmpeg_sys_next::sws_freeContext(self.sws_ctx); }
            if !self.sw_frame.is_null() { ffmpeg_sys_next::av_frame_free(&mut self.sw_frame); }
            if !self.frame.is_null() { ffmpeg_sys_next::av_frame_free(&mut self.frame); }
            if !self.hw_device_ctx.is_null() { ffmpeg_sys_next::av_buffer_unref(&mut self.hw_device_ctx); }
        }
    }
}

/// Global GPU codec — `None` if cuvid unavailable.
static GPU_CODEC: LazyLock<Option<Mutex<GpuCodecState>>> = LazyLock::new(|| {
    if !*USE_GPU_DECODE { return None; }
    unsafe { init_gpu_codec() }
});

unsafe fn init_gpu_codec() -> Option<Mutex<GpuCodecState>> {
    FFMPEG_INIT.call_once(|| { let _ = ffmpeg::init(); });

    // Check h264_cuvid availability
    let codec_name = b"h264_cuvid\0";
    let codec = ffmpeg_sys_next::avcodec_find_decoder_by_name(codec_name.as_ptr() as *const _);
    if codec.is_null() {
        eprintln!("[lance-video] h264_cuvid not found, falling back to CPU decode");
        return None;
    }

    // Create CUDA hw device context (shared across all batch decodes)
    let mut hw_device_ctx: *mut ffmpeg_sys_next::AVBufferRef = std::ptr::null_mut();
    let ret = ffmpeg_sys_next::av_hwdevice_ctx_create(
        &mut hw_device_ctx,
        ffmpeg_sys_next::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA,
        std::ptr::null(),
        std::ptr::null_mut(),
        0,
    );
    if ret < 0 {
        eprintln!("[lance-video] av_hwdevice_ctx_create(CUDA) failed: {ret}, falling back to CPU");
        return None;
    }

    let frame = ffmpeg_sys_next::av_frame_alloc();
    let sw_frame = ffmpeg_sys_next::av_frame_alloc();

    eprintln!("[lance-video] GPU decode enabled (h264_cuvid)");
    Some(Mutex::new(GpuCodecState {
        hw_device_ctx,
        frame,
        sw_frame,
        sws_ctx: std::ptr::null_mut(),
        sws_w: 0,
        sws_h: 0,
        sws_src_fmt: ffmpeg_sys_next::AVPixelFormat::AV_PIX_FMT_NONE,
    }))
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
// GPU decode: h264_cuvid → GPU frame → av_hwframe_transfer → CPU NV12 → sws RGB24
// =============================================================================

/// Batch decode: GPU concatenated stream or CPU parallel.
pub fn decode_h264_batch(
    blobs: &[&[u8]],
    target_frames: Option<&[usize]>,
) -> Vec<std::result::Result<Vec<u8>, String>> {
    if blobs.is_empty() { return vec![]; }

    if let Some(ref gpu_mutex) = *GPU_CODEC {
        let mut state = match gpu_mutex.lock() {
            Ok(s) => s,
            Err(e) => return vec![Err(format!("GPU mutex poisoned: {e}")); blobs.len()],
        };
        unsafe { gpu_decode_batch_concat(&mut state, blobs, target_frames) }
    } else {
        DECODE_POOL.install(|| {
            blobs.into_par_iter().enumerate().map(|(i, blob)| {
                let tf = target_frames.map(|tfs| tfs[i]);
                decode_h264_frames(blob, tf)
            }).collect()
        })
    }
}

/// Concatenate all GOP blobs into one stream and decode in a single cuvid
/// session.  Each GOP starts with SPS+PPS+IDR so the hardware resets
/// naturally without an expensive flush_buffers call.
///
/// GOP boundaries are detected by key_frame (IDR) in the decoded output.
unsafe fn gpu_decode_batch_concat(
    state: &mut GpuCodecState,
    blobs: &[&[u8]],
    target_frames: Option<&[usize]>,
) -> Vec<std::result::Result<Vec<u8>, String>> {
    let n = blobs.len();

    // Concatenate all blobs
    let total_bytes: usize = blobs.iter().map(|b| b.len()).sum();
    let mut stream = Vec::with_capacity(total_bytes);
    for blob in blobs {
        stream.extend_from_slice(blob);
    }

    // Fresh codec context sharing the CUDA device (one init cost, no per-GOP flush)
    let codec_name = b"h264_cuvid\0";
    let codec = ffmpeg_sys_next::avcodec_find_decoder_by_name(codec_name.as_ptr() as *const _);
    if codec.is_null() {
        return vec![Err("h264_cuvid not found".into()); n];
    }
    let ctx = ffmpeg_sys_next::avcodec_alloc_context3(codec);
    if ctx.is_null() {
        return vec![Err("alloc_context3 failed".into()); n];
    }
    (*ctx).hw_device_ctx = ffmpeg_sys_next::av_buffer_ref(state.hw_device_ctx);
    let ret = ffmpeg_sys_next::avcodec_open2(ctx, codec, std::ptr::null_mut());
    if ret < 0 {
        ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
        return vec![Err(format!("avcodec_open2: {ret}")); n];
    }

    let frame = state.frame;
    let sw_frame = state.sw_frame;
    let mut packet = ffmpeg_sys_next::av_packet_alloc();
    let parser = ffmpeg_sys_next::av_parser_init(
        ffmpeg_sys_next::AVCodecID::AV_CODEC_ID_H264 as std::ffi::c_int,
    );
    if parser.is_null() {
        ffmpeg_sys_next::av_packet_free(&mut packet);
        ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));
        return vec![Err("av_parser_init failed".into()); n];
    }

    let mut cur_gop_rgb: Vec<u8> = Vec::new();
    let mut cur_gop_idx: usize = 0;
    let mut frame_in_gop: usize = 0;
    let mut w: i32 = 0;
    let mut h: i32 = 0;
    let mut frame_size: usize = 0;
    let mut first_frame_seen = false;
    let mut results: Vec<std::result::Result<Vec<u8>, String>> = Vec::with_capacity(n);

    let mut data_ptr = stream.as_ptr();
    let mut data_remaining = stream.len() as std::ffi::c_int;

    // Process one decoded frame, splitting GOPs on IDR (key_frame).
    macro_rules! process_frame {
        () => {
            // Detect GOP boundary: key_frame && not the very first frame
            let is_key = ((*frame).flags & ffmpeg_sys_next::AV_FRAME_FLAG_KEY) != 0;
            if is_key && first_frame_seen {
                // Finalize previous GOP
                results.push(Ok(std::mem::take(&mut cur_gop_rgb)));
                cur_gop_idx += 1;
                frame_in_gop = 0;
            }
            first_frame_seen = true;

            // GPU → CPU transfer
            ffmpeg_sys_next::av_frame_unref(sw_frame);
            let ret = ffmpeg_sys_next::av_hwframe_transfer_data(sw_frame, frame, 0);
            if ret < 0 {
                eprintln!("[lance-video] av_hwframe_transfer_data failed: {ret}, gop={cur_gop_idx} frame={frame_in_gop}");
                frame_in_gop += 1;
            } else {
                let sw_fmt = (*sw_frame).format as u32;
                let sw_pix_fmt: ffmpeg_sys_next::AVPixelFormat = std::mem::transmute(sw_fmt);
                if w == 0 {
                    w = (*sw_frame).width;
                    h = (*sw_frame).height;
                    frame_size = (w * 3) as usize * h as usize;
                }
                if state.sws_ctx.is_null() || state.sws_w != w || state.sws_h != h || state.sws_src_fmt != sw_pix_fmt {
                    if !state.sws_ctx.is_null() { ffmpeg_sys_next::sws_freeContext(state.sws_ctx); }
                    state.sws_ctx = ffmpeg_sys_next::sws_getContext(
                        w, h, sw_pix_fmt, w, h,
                        ffmpeg_sys_next::AVPixelFormat::AV_PIX_FMT_RGB24,
                        0, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null(),
                    );
                    state.sws_w = w;
                    state.sws_h = h;
                    state.sws_src_fmt = sw_pix_fmt;
                }

                // target_frame: only scale the requested frame index within this GOP
                let tf = if cur_gop_idx < n { target_frames.map(|tfs| tfs[cur_gop_idx]) } else { None };
                match tf {
                    Some(t) => {
                        if frame_in_gop == t {
                            cur_gop_rgb.clear();
                            cur_gop_rgb.reserve(frame_size);
                            scale_frame_rgb(sw_frame, state.sws_ctx, w, h, &mut cur_gop_rgb);
                        }
                    }
                    None => {
                        scale_frame_rgb(sw_frame, state.sws_ctx, w, h, &mut cur_gop_rgb);
                    }
                }
                frame_in_gop += 1;
            }
        };
    }

    // Parse the entire concatenated stream
    while data_remaining > 0 {
        let mut pkt_data: *mut u8 = std::ptr::null_mut();
        let mut pkt_size: std::ffi::c_int = 0;
        let consumed = ffmpeg_sys_next::av_parser_parse2(
            parser, ctx, &mut pkt_data, &mut pkt_size,
            data_ptr, data_remaining,
            ffmpeg_sys_next::AV_NOPTS_VALUE, ffmpeg_sys_next::AV_NOPTS_VALUE, 0,
        );
        if consumed < 0 { break; }
        data_ptr = data_ptr.add(consumed as usize);
        data_remaining -= consumed;

        if pkt_size > 0 {
            (*packet).data = pkt_data;
            (*packet).size = pkt_size;
            ffmpeg_sys_next::avcodec_send_packet(ctx, packet);
            while ffmpeg_sys_next::avcodec_receive_frame(ctx, frame) == 0 {
                process_frame!();
            }
        }
    }

    // Flush parser
    {
        let mut pkt_data: *mut u8 = std::ptr::null_mut();
        let mut pkt_size: std::ffi::c_int = 0;
        ffmpeg_sys_next::av_parser_parse2(
            parser, ctx, &mut pkt_data, &mut pkt_size,
            std::ptr::null(), 0,
            ffmpeg_sys_next::AV_NOPTS_VALUE, ffmpeg_sys_next::AV_NOPTS_VALUE, 0,
        );
        if pkt_size > 0 {
            (*packet).data = pkt_data;
            (*packet).size = pkt_size;
            ffmpeg_sys_next::avcodec_send_packet(ctx, packet);
            while ffmpeg_sys_next::avcodec_receive_frame(ctx, frame) == 0 {
                process_frame!();
            }
        }
    }

    // Flush decoder
    ffmpeg_sys_next::avcodec_send_packet(ctx, std::ptr::null());
    while ffmpeg_sys_next::avcodec_receive_frame(ctx, frame) == 0 {
        process_frame!();
    }

    // Push the last GOP
    results.push(Ok(std::mem::take(&mut cur_gop_rgb)));

    while results.len() < n {
        results.push(Ok(Vec::new()));
    }

    ffmpeg_sys_next::av_parser_close(parser);
    ffmpeg_sys_next::av_packet_free(&mut packet);
    ffmpeg_sys_next::avcodec_free_context(&mut (ctx as *mut _));

    results
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

        let mut non_null_data: Vec<&[u8]> = Vec::new();
        let mut null_mask: Vec<bool> = Vec::with_capacity(binary_arr.len());
        for i in 0..binary_arr.len() {
            if binary_arr.is_null(i) {
                null_mask.push(true);
            } else {
                null_mask.push(false);
                non_null_data.push(binary_arr.value(i));
            }
        }
        let decoded = decode_h264_batch(&non_null_data, None);
        let mut dec_iter = decoded.into_iter();
        let results: Vec<Option<Result<Vec<u8>, String>>> = null_mask.into_iter()
            .map(|is_null| if is_null { None } else { Some(dec_iter.next().unwrap()) })
            .collect();

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
