//! Rust-side per-frame processing: decode -> track -> timing. Backend-agnostic.

use crate::decode::{DecodeConfig, DecodeError, Decoder};
use crate::stats::Hist;
use crate::tracker::{TrackOut, Tracker, TrackerConfig};
use crate::types::{Detection, FrameResult};

#[derive(Default)]
pub struct PipelineStats {
    /// Capture timestamp -> tracks available to the application.
    pub end_to_end: Hist,
    /// Native side: capture timestamp -> raw output on host.
    pub native_total: Hist,
    pub native_preprocess: Hist,
    pub native_infer: Hist,
    pub native_d2h: Hist,
    pub decode: Hist,
    pub track: Hist,
    pub frames: u64,
    pub inferred: u64,
    pub dropped: u64,
}

impl PipelineStats {
    pub fn report(&self) -> String {
        format!(
            "frames={} inferred={} dropped={}\n  end-to-end  {}\n  native      {}\n    preprocess {}\n    infer      {}\n    d2h        {}\n  decode+nms  {}\n  track       {}",
            self.frames,
            self.inferred,
            self.dropped,
            self.end_to_end.summary(),
            self.native_total.summary(),
            self.native_preprocess.summary(),
            self.native_infer.summary(),
            self.native_d2h.summary(),
            self.decode.summary(),
            self.track.summary(),
        )
    }
}

pub struct Pipeline {
    decoder: Decoder,
    tracker: Tracker,
    pub dets: Vec<Detection>,
    pub tracks: Vec<TrackOut>,
    pub stats: PipelineStats,
}

impl Pipeline {
    pub fn new(decode: DecodeConfig, tracker: TrackerConfig) -> Self {
        Self {
            decoder: Decoder::new(decode),
            tracker: Tracker::new(tracker),
            dets: Vec::new(),
            tracks: Vec::new(),
            stats: PipelineStats::default(),
        }
    }

    pub fn tracker(&self) -> &Tracker {
        &self.tracker
    }

    /// Processes one frame; `now_ns` is the backend clock. Results land in `self.tracks`.
    pub fn process(&mut self, f: &FrameResult, now_ns: impl Fn() -> i64) -> Result<(), DecodeError> {
        let t0 = now_ns();
        if f.ran_inference {
            self.decoder.decode(&f.raw, f.channels, f.anchors, &f.letterbox, &mut self.dets)?;
            let t1 = now_ns();
            self.tracker.update(f.capture_ts_ns, &self.dets);
            let t2 = now_ns();
            self.stats.decode.record_ns(t1 - t0);
            self.stats.track.record_ns(t2 - t1);
            self.stats.native_preprocess.record_us(f.native.preprocess_us);
            self.stats.native_infer.record_us(f.native.infer_us);
            self.stats.native_d2h.record_us(f.native.d2h_us);
            self.stats.inferred += 1;
        } else {
            self.dets.clear();
            self.tracker.predict_only(f.capture_ts_ns);
            self.stats.track.record_ns(now_ns() - t0);
        }
        self.tracker.outputs(&mut self.tracks);
        self.stats.native_total.record_ns(f.ready_ts_ns - f.capture_ts_ns);
        self.stats.end_to_end.record_ns(now_ns() - f.capture_ts_ns);
        self.stats.frames += 1;
        self.stats.dropped += f.dropped_before as u64;
        Ok(())
    }
}
