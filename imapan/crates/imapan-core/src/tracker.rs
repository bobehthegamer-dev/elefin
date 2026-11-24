//! ByteTrack-style multi-object tracker.
//!
//! * Two-stage association: high-confidence detections first, then low-confidence ones against
//!   the leftover tracks (recovers occluded / blurred objects without spawning false tracks).
//! * Class-aware, IoU-based, greedy assignment (fast, allocation-light; fine for tens of objects).
//! * `predict_only` advances the filters on frames where the detector was skipped, so the output
//!   rate can be higher than the detector rate.

use crate::kalman::{BoxFilter, FilterParams};
use crate::types::{BBox, Detection};

#[derive(Clone, Debug)]
pub struct TrackerConfig {
    pub high_thresh: f32,
    pub low_thresh: f32,
    /// A new track is only started from a detection at least this confident.
    pub new_track_thresh: f32,
    pub match_iou: f32,
    pub low_match_iou: f32,
    /// Detector frames a track must be matched in before it is reported.
    pub min_hits: u32,
    /// How long (seconds, from frame timestamps) a confirmed track coasts without a match.
    pub max_lost_s: f32,
    pub filter: FilterParams,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            high_thresh: 0.5,
            low_thresh: 0.1,
            new_track_thresh: 0.6,
            match_iou: 0.3,
            low_match_iou: 0.5,
            min_hits: 3,
            max_lost_s: 0.5,
            filter: FilterParams::default(),
        }
    }
}

#[derive(Clone, Debug)]
struct Track {
    id: u64,
    class: u16,
    score: f32,
    filter: BoxFilter,
    hits: u32,
    confirmed: bool,
    last_update_ts: i64,
}

/// A reportable track state.
#[derive(Clone, Copy, Debug)]
pub struct TrackOut {
    pub id: u64,
    pub class: u16,
    pub score: f32,
    pub bbox: BBox,
    /// Center velocity, px/s.
    pub vx: f32,
    pub vy: f32,
    /// True when this state is a pure prediction (no detection matched this frame).
    pub coasting: bool,
}

pub struct Tracker {
    cfg: TrackerConfig,
    tracks: Vec<Track>,
    next_id: u64,
    last_ts: Option<i64>,
    // scratch
    pairs: Vec<(f32, usize, usize)>,
}

const NS: f32 = 1e-9;

impl Tracker {
    pub fn new(cfg: TrackerConfig) -> Self {
        Self { cfg, tracks: Vec::new(), next_id: 1, last_ts: None, pairs: Vec::new() }
    }

    pub fn ids_created(&self) -> u64 {
        self.next_id - 1
    }

    /// Move all filters forward to `ts_ns` and drop tracks that coasted too long.
    pub fn predict_only(&mut self, ts_ns: i64) {
        if let Some(last) = self.last_ts {
            let dt = ((ts_ns - last) as f32 * NS).clamp(0.0, 0.5);
            for t in &mut self.tracks {
                t.filter.predict(dt);
            }
        }
        self.last_ts = Some(ts_ns);
        let max_lost_ns = (self.cfg.max_lost_s / NS) as i64;
        self.tracks.retain(|t| ts_ns - t.last_update_ts <= max_lost_ns);
    }

    pub fn update(&mut self, ts_ns: i64, dets: &[Detection]) {
        self.predict_only(ts_ns);

        let high: Vec<usize> =
            (0..dets.len()).filter(|&i| dets[i].score >= self.cfg.high_thresh).collect();
        let low: Vec<usize> = (0..dets.len())
            .filter(|&i| dets[i].score >= self.cfg.low_thresh && dets[i].score < self.cfg.high_thresh)
            .collect();

        let mut track_free = vec![true; self.tracks.len()];
        let mut det_used = vec![false; dets.len()];

        // Stage 1: all tracks vs high-confidence detections.
        let all: Vec<usize> = (0..self.tracks.len()).collect();
        let m1 = self.associate(&all, &high, dets, self.cfg.match_iou);
        for &(ti, di) in &m1 {
            track_free[ti] = false;
            det_used[di] = true;
        }
        // Stage 2: leftover confirmed tracks vs low-confidence detections.
        let rest: Vec<usize> = (0..self.tracks.len())
            .filter(|&i| track_free[i] && self.tracks[i].confirmed)
            .collect();
        let m2 = self.associate(&rest, &low, dets, self.cfg.low_match_iou);
        for &(ti, di) in &m2 {
            track_free[ti] = false;
            det_used[di] = true;
        }

        for &(ti, di) in m1.iter().chain(m2.iter()) {
            let t = &mut self.tracks[ti];
            t.filter.update(&dets[di].bbox);
            t.score = dets[di].score;
            t.hits += 1;
            t.last_update_ts = ts_ns;
            if t.hits >= self.cfg.min_hits {
                t.confirmed = true;
            }
        }

        // Tentative tracks die on their first miss; confirmed ones coast (see predict_only).
        let mut idx = 0;
        self.tracks.retain(|t| {
            let keep = !track_free[idx] || t.confirmed;
            idx += 1;
            keep
        });

        for &di in &high {
            if det_used[di] || dets[di].score < self.cfg.new_track_thresh {
                continue;
            }
            self.tracks.push(Track {
                id: self.next_id,
                class: dets[di].class,
                score: dets[di].score,
                filter: BoxFilter::new(&dets[di].bbox, self.cfg.filter),
                hits: 1,
                confirmed: self.cfg.min_hits <= 1,
                last_update_ts: ts_ns,
            });
            self.next_id += 1;
        }
    }

    /// Fills `out` with confirmed tracks at the most recent timestamp.
    pub fn outputs(&self, out: &mut Vec<TrackOut>) {
        out.clear();
        let now = self.last_ts.unwrap_or(0);
        for t in self.tracks.iter().filter(|t| t.confirmed) {
            let (vx, vy) = t.filter.velocity();
            out.push(TrackOut {
                id: t.id,
                class: t.class,
                score: t.score,
                bbox: t.filter.bbox(),
                vx,
                vy,
                coasting: t.last_update_ts != now,
            });
        }
    }

    pub fn active_tracks(&self) -> usize {
        self.tracks.iter().filter(|t| t.confirmed).count()
    }

    fn associate(
        &mut self,
        tracks: &[usize],
        dets_idx: &[usize],
        dets: &[Detection],
        thr: f32,
    ) -> Vec<(usize, usize)> {
        self.pairs.clear();
        for &ti in tracks {
            let tb = self.tracks[ti].filter.bbox();
            for &di in dets_idx {
                if dets[di].class != self.tracks[ti].class {
                    continue;
                }
                let iou = tb.iou(&dets[di].bbox);
                if iou >= thr {
                    self.pairs.push((iou, ti, di));
                }
            }
        }
        self.pairs.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let mut matches = Vec::new();
        let mut t_used = vec![false; self.tracks.len()];
        let mut d_used = vec![false; dets.len()];
        for &(_, ti, di) in &self.pairs {
            if !t_used[ti] && !d_used[di] {
                t_used[ti] = true;
                d_used[di] = true;
                matches.push((ti, di));
            }
        }
        matches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT_NS: i64 = 4_166_667; // 240 fps

    fn det(cx: f32, cy: f32, class: u16, score: f32) -> Detection {
        Detection { bbox: BBox::from_cxcywh(cx, cy, 80.0, 160.0), score, class }
    }

    #[test]
    fn keeps_one_id_with_detector_every_4th_frame() {
        let mut tr = Tracker::new(TrackerConfig::default());
        let mut out = Vec::new();
        let (vx, vy) = (500.0f32, 200.0f32);
        let mut max_err = 0.0f32;
        for f in 0..480i64 {
            let ts = f * DT_NS;
            let t = ts as f32 * 1e-9;
            let (cx, cy) = (200.0 + vx * t, 300.0 + vy * t);
            if f % 4 == 0 {
                tr.update(ts, &[det(cx, cy, 0, 0.9)]);
            } else {
                tr.predict_only(ts);
            }
            tr.outputs(&mut out);
            if f > 60 {
                assert_eq!(out.len(), 1, "frame {f}");
                let c = out[0].bbox.center();
                max_err = max_err.max((c.0 - cx).abs()).max((c.1 - cy).abs());
            }
        }
        assert_eq!(tr.ids_created(), 1, "id switched / respawned");
        assert!(max_err < 6.0, "prediction error {max_err}px");
    }

    #[test]
    fn coasts_then_expires() {
        let mut tr = Tracker::new(TrackerConfig::default());
        let mut out = Vec::new();
        for f in 0..10 {
            tr.update(f * DT_NS * 4, &[det(300.0, 300.0, 1, 0.9)]);
        }
        tr.outputs(&mut out);
        assert_eq!(out.len(), 1);
        // No detections for 0.2 s: still coasting.
        let t0 = 10 * DT_NS * 4;
        tr.update(t0 + 200_000_000, &[]);
        tr.outputs(&mut out);
        assert_eq!(out.len(), 1);
        assert!(out[0].coasting);
        // ...and gone after max_lost_s.
        tr.update(t0 + 800_000_000, &[]);
        tr.outputs(&mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn never_matches_across_classes_and_needs_confidence() {
        let mut tr = Tracker::new(TrackerConfig::default());
        let mut out = Vec::new();
        for f in 0..6 {
            tr.update(f * DT_NS, &[det(300.0, 300.0, 0, 0.9)]);
        }
        tr.update(6 * DT_NS, &[det(300.0, 300.0, 5, 0.9)]);
        assert_eq!(tr.ids_created(), 2, "different class must start its own track");
        // Detections below new_track_thresh never start a track.
        let mut tr2 = Tracker::new(TrackerConfig::default());
        for f in 0..10 {
            tr2.update(f * DT_NS, &[det(100.0, 100.0, 0, 0.55)]);
        }
        tr2.outputs(&mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn low_confidence_detections_keep_an_existing_track() {
        let mut tr = Tracker::new(TrackerConfig::default());
        for f in 0..10 {
            tr.update(f * DT_NS, &[det(300.0, 300.0, 0, 0.9)]);
        }
        for f in 10..30 {
            tr.update(f * DT_NS, &[det(300.0, 300.0, 0, 0.3)]);
        }
        let mut out = Vec::new();
        tr.outputs(&mut out);
        assert_eq!(out.len(), 1);
        assert!(!out[0].coasting);
        assert_eq!(tr.ids_created(), 1);
    }
}
