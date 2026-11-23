//! Basic geometry and frame types shared by every stage.

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BBox {
    pub fn from_cxcywh(cx: f32, cy: f32, w: f32, h: f32) -> Self {
        Self { x1: cx - w * 0.5, y1: cy - h * 0.5, x2: cx + w * 0.5, y2: cy + h * 0.5 }
    }
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }
    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }
    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)
    }
    pub fn iou(&self, o: &BBox) -> f32 {
        let ix = (self.x2.min(o.x2) - self.x1.max(o.x1)).max(0.0);
        let iy = (self.y2.min(o.y2) - self.y1.max(o.y1)).max(0.0);
        let inter = ix * iy;
        let union = self.area() + o.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    /// Box in *source* (captured image) pixel coordinates.
    pub bbox: BBox,
    pub score: f32,
    pub class: u16,
}

/// Mapping between the captured image and the network input:
/// `input = source * scale + pad`. The native module computes it and reports it per frame,
/// so the CUDA kernel and the decoder can never disagree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Letterbox {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub src_w: f32,
    pub src_h: f32,
}

impl Letterbox {
    pub fn compute(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Self {
        let (sw, sh, dw, dh) = (src_w as f32, src_h as f32, dst_w as f32, dst_h as f32);
        let scale = (dw / sw).min(dh / sh);
        Self {
            scale,
            pad_x: (dw - sw * scale) * 0.5,
            pad_y: (dh - sh * scale) * 0.5,
            src_w: sw,
            src_h: sh,
        }
    }
}

/// Per-stage GPU timings reported by the native module (microseconds). Zero for skipped frames.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeTimings {
    pub preprocess_us: f32,
    pub infer_us: f32,
    pub d2h_us: f32,
}

/// One frame as delivered by a [`crate::backend::Backend`].
#[derive(Clone, Debug)]
pub struct FrameResult {
    pub frame_id: u64,
    /// Monotonic ns, same clock as `Backend::now_ns`.
    pub capture_ts_ns: i64,
    /// When the native side finished (raw output on host).
    pub ready_ts_ns: i64,
    /// False when the detector skipped this frame (`detect_every_n > 1`); tracker only predicts.
    pub ran_inference: bool,
    /// Frames the backend discarded (latest-wins) since the previous delivered frame.
    pub dropped_before: u32,
    pub letterbox: Letterbox,
    /// Raw head output, channel-major `[channels][anchors]` (YOLOv8/11 layout: cx,cy,w,h,cls...).
    pub channels: usize,
    pub anchors: usize,
    pub raw: Vec<f32>,
    pub native: NativeTimings,
}

impl FrameResult {
    pub fn empty() -> Self {
        Self {
            frame_id: 0,
            capture_ts_ns: 0,
            ready_ts_ns: 0,
            ran_inference: false,
            dropped_before: 0,
            letterbox: Letterbox { scale: 1.0, pad_x: 0.0, pad_y: 0.0, src_w: 1.0, src_h: 1.0 },
            channels: 0,
            anchors: 0,
            raw: Vec::new(),
            native: NativeTimings::default(),
        }
    }
}
