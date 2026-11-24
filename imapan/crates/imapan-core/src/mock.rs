//! Synthetic backend: bouncing objects rendered straight into a YOLOv8-shaped output tensor.
//! Lets the whole Rust pipeline (decode, NMS, tracking, timing, logging) run and be tested on
//! any machine, with the same pacing / frame-skipping / latest-wins behavior as the real thing.

use crate::backend::{Backend, NextFrame};
use crate::mailbox::LatestSlot;
use crate::types::{BBox, FrameResult, Letterbox, NativeTimings};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct MockConfig {
    pub fps: f64,
    pub detect_every_n: u64,
    pub src_w: u32,
    pub src_h: u32,
    pub input_w: u32,
    pub input_h: u32,
    pub num_classes: usize,
    pub objects: usize,
    /// Probability a ground-truth object is missed on an inference frame.
    pub miss_prob: f32,
}

impl Default for MockConfig {
    fn default() -> Self {
        Self {
            fps: 240.0,
            detect_every_n: 1,
            src_w: 2560,
            src_h: 1440,
            input_w: 640,
            input_h: 384,
            num_classes: 80,
            objects: 5,
            miss_prob: 0.03,
        }
    }
}

pub struct MockBackend {
    slot: Arc<LatestSlot<FrameResult>>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    epoch: Instant,
}

fn tri(x: f32, range: f32) -> f32 {
    // Triangle wave in [0, range]: bouncing motion without state.
    let p = x.rem_euclid(2.0 * range);
    if p > range {
        2.0 * range - p
    } else {
        p
    }
}

/// Ground truth box of object `i` at time `t` seconds.
pub fn truth(cfg: &MockConfig, i: usize, t: f32) -> (BBox, u16) {
    let (w, h) = (100.0 + 20.0 * i as f32, 180.0 + 10.0 * i as f32);
    let (vx, vy) = (350.0 + 140.0 * i as f32, 220.0 + 90.0 * i as f32);
    let cx = w * 0.5 + tri(120.0 * i as f32 + vx * t, cfg.src_w as f32 - w);
    let cy = h * 0.5 + tri(80.0 * i as f32 + vy * t, cfg.src_h as f32 - h);
    (BBox::from_cxcywh(cx, cy, w, h), (i % cfg.num_classes) as u16)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn anchors_for(w: u32, h: u32) -> usize {
    [8u32, 16, 32].iter().map(|s| ((w / s) * (h / s)) as usize).sum()
}

impl MockBackend {
    pub fn new(cfg: MockConfig) -> Self {
        let slot = Arc::new(LatestSlot::new());
        let stop = Arc::new(AtomicBool::new(false));
        let epoch = Instant::now();
        let (s2, st2) = (slot.clone(), stop.clone());
        let handle = std::thread::spawn(move || producer(cfg, s2, st2, epoch));
        Self { slot, stop, handle: Some(handle), epoch }
    }
}

fn producer(cfg: MockConfig, slot: Arc<LatestSlot<FrameResult>>, stop: Arc<AtomicBool>, epoch: Instant) {
    let period = Duration::from_secs_f64(1.0 / cfg.fps);
    let lb = Letterbox::compute(cfg.src_w, cfg.src_h, cfg.input_w, cfg.input_h);
    let anchors = anchors_for(cfg.input_w, cfg.input_h);
    let channels = 4 + cfg.num_classes;
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let mut next = epoch;
    let mut frame_id = 0u64;
    while !stop.load(Ordering::Relaxed) {
        next += period;
        // sleep most of the way, spin the last 200us for stable pacing
        loop {
            let now = Instant::now();
            if now >= next {
                break;
            }
            let left = next - now;
            if left > Duration::from_micros(300) {
                std::thread::sleep(left - Duration::from_micros(200));
            } else {
                std::hint::spin_loop();
            }
        }
        let cap = epoch.elapsed().as_nanos() as i64;
        let t = cap as f32 * 1e-9;
        let infer = frame_id % cfg.detect_every_n == 0;

        let mut raw = Vec::new();
        let mut native = NativeTimings::default();
        if infer {
            raw = vec![0.0f32; channels * anchors];
            for i in 0..cfg.objects {
                if rng.next() < cfg.miss_prob {
                    continue;
                }
                let (b, class) = truth(&cfg, i, t);
                let (cx, cy) = b.center();
                let jx = (rng.next() - 0.5) * 4.0;
                let jy = (rng.next() - 0.5) * 4.0;
                let (icx, icy) = ((cx + jx) * lb.scale + lb.pad_x, (cy + jy) * lb.scale + lb.pad_y);
                let (iw, ih) = (b.width() * lb.scale, b.height() * lb.scale);
                // Anchor + a near-duplicate anchor so NMS has something to do.
                for (k, score) in [(0usize, 0.9f32), (1, 0.8)] {
                    let a = (i * 997 + (frame_id as usize % 40) * 7 + k) % anchors;
                    raw[a] = icx + k as f32;
                    raw[anchors + a] = icy;
                    raw[2 * anchors + a] = iw;
                    raw[3 * anchors + a] = ih;
                    raw[(4 + class as usize) * anchors + a] = score;
                }
            }
            native = NativeTimings { preprocess_us: 150.0, infer_us: 1800.0, d2h_us: 100.0 };
        }
        let ready = epoch.elapsed().as_nanos() as i64;
        slot.put(FrameResult {
            frame_id,
            capture_ts_ns: cap,
            ready_ts_ns: ready,
            ran_inference: infer,
            dropped_before: 0,
            letterbox: lb,
            channels: if infer { channels } else { 0 },
            anchors: if infer { anchors } else { 0 },
            raw,
            native,
        });
        frame_id += 1;
    }
    slot.close();
}

impl Backend for MockBackend {
    fn name(&self) -> &str {
        "mock"
    }
    fn now_ns(&self) -> i64 {
        self.epoch.elapsed().as_nanos() as i64
    }
    fn next_frame(&mut self, timeout: Duration, out: &mut FrameResult) -> Result<NextFrame, String> {
        match self.slot.take(timeout) {
            Some((f, dropped)) => {
                *out = f;
                out.dropped_before = dropped;
                Ok(NextFrame::Frame)
            }
            None => Ok(NextFrame::Timeout),
        }
    }
}

impl Drop for MockBackend {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
