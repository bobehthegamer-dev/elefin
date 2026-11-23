//! YOLOv8 / YOLO11 head decoding + class-aware NMS.
//!
//! Expected tensor: `[1, 4 + num_classes, anchors]`, flattened channel-major. Rows 0..4 are
//! `cx, cy, w, h` in *network input* pixels; rows 4.. are per-class scores (already sigmoided,
//! no separate objectness). This is what `yolo export format=onnx` produces for detect models.
//! It is NOT the layout of YOLOv5 (has objectness) or YOLOv10 / end-to-end exports.

use crate::types::{BBox, Detection, Letterbox};

#[derive(Clone, Debug)]
pub struct DecodeConfig {
    pub conf: f32,
    pub nms_iou: f32,
    /// Cap before NMS so a low `conf` can't make NMS quadratic on thousands of boxes.
    pub max_pre_nms: usize,
    pub max_det: usize,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self { conf: 0.25, nms_iou: 0.5, max_pre_nms: 300, max_det: 100 }
    }
}

#[derive(Debug, PartialEq)]
pub enum DecodeError {
    BadShape { channels: usize, anchors: usize, len: usize },
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::BadShape { channels, anchors, len } => {
                write!(f, "bad tensor shape: channels={channels} anchors={anchors} len={len}")
            }
        }
    }
}
impl std::error::Error for DecodeError {}

/// Reusable decoder; keeps scratch buffers so the hot path doesn't allocate.
pub struct Decoder {
    pub cfg: DecodeConfig,
    best_score: Vec<f32>,
    best_cls: Vec<u16>,
    cands: Vec<Detection>,
}

impl Decoder {
    pub fn new(cfg: DecodeConfig) -> Self {
        Self { cfg, best_score: Vec::new(), best_cls: Vec::new(), cands: Vec::new() }
    }

    pub fn decode(
        &mut self,
        raw: &[f32],
        channels: usize,
        anchors: usize,
        lb: &Letterbox,
        out: &mut Vec<Detection>,
    ) -> Result<(), DecodeError> {
        out.clear();
        if channels <= 4 || anchors == 0 || raw.len() < channels * anchors {
            return Err(DecodeError::BadShape { channels, anchors, len: raw.len() });
        }
        let nc = channels - 4;

        // Channel-outer loop: sequential memory access instead of 80 strided reads per anchor.
        self.best_score.clear();
        self.best_score.resize(anchors, 0.0);
        self.best_cls.clear();
        self.best_cls.resize(anchors, 0);
        for c in 0..nc {
            let row = &raw[(4 + c) * anchors..(5 + c) * anchors];
            for i in 0..anchors {
                if row[i] > self.best_score[i] {
                    self.best_score[i] = row[i];
                    self.best_cls[i] = c as u16;
                }
            }
        }

        self.cands.clear();
        let inv = 1.0 / lb.scale;
        for i in 0..anchors {
            let s = self.best_score[i];
            if s < self.cfg.conf {
                continue;
            }
            let (cx, cy, w, h) =
                (raw[i], raw[anchors + i], raw[2 * anchors + i], raw[3 * anchors + i]);
            if w <= 0.0 || h <= 0.0 {
                continue;
            }
            let x1 = ((cx - w * 0.5 - lb.pad_x) * inv).clamp(0.0, lb.src_w);
            let y1 = ((cy - h * 0.5 - lb.pad_y) * inv).clamp(0.0, lb.src_h);
            let x2 = ((cx + w * 0.5 - lb.pad_x) * inv).clamp(0.0, lb.src_w);
            let y2 = ((cy + h * 0.5 - lb.pad_y) * inv).clamp(0.0, lb.src_h);
            if x2 - x1 < 1.0 || y2 - y1 < 1.0 {
                continue;
            }
            self.cands.push(Detection {
                bbox: BBox { x1, y1, x2, y2 },
                score: s,
                class: self.best_cls[i],
            });
        }

        let by_score_desc = |a: &Detection, b: &Detection| b.score.total_cmp(&a.score);
        if self.cands.len() > self.cfg.max_pre_nms {
            self.cands.select_nth_unstable_by(self.cfg.max_pre_nms, by_score_desc);
            self.cands.truncate(self.cfg.max_pre_nms);
        }
        self.cands.sort_unstable_by(by_score_desc);

        for d in &self.cands {
            if out.len() >= self.cfg.max_det {
                break;
            }
            let suppressed =
                out.iter().any(|k| k.class == d.class && k.bbox.iou(&d.bbox) > self.cfg.nms_iou);
            if !suppressed {
                out.push(*d);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes one anchor into a channel-major tensor.
    fn put(raw: &mut [f32], anchors: usize, i: usize, b: [f32; 4], class: usize, score: f32) {
        for (k, v) in b.iter().enumerate() {
            raw[k * anchors + i] = *v;
        }
        raw[(4 + class) * anchors + i] = score;
    }

    fn lb() -> Letterbox {
        Letterbox::compute(2560, 1440, 640, 384)
    }

    #[test]
    fn letterbox_matches_expected_geometry() {
        let l = lb();
        assert_eq!(l.scale, 0.25);
        assert_eq!(l.pad_x, 0.0);
        assert_eq!(l.pad_y, 12.0);
    }

    #[test]
    fn maps_back_to_source_coordinates() {
        let (ch, n) = (4 + 3, 10);
        let mut raw = vec![0.0; ch * n];
        // Source box center (1000, 700) size 200x100 -> input center (250, 187) size 50x25.
        put(&mut raw, n, 3, [250.0, 187.0, 50.0, 25.0], 2, 0.9);
        let mut d = Decoder::new(DecodeConfig::default());
        let mut out = Vec::new();
        d.decode(&raw, ch, n, &lb(), &mut out).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].class, 2);
        let b = out[0].bbox;
        assert!((b.x1 - 900.0).abs() < 0.01 && (b.x2 - 1100.0).abs() < 0.01, "{b:?}");
        assert!((b.y1 - 650.0).abs() < 0.01 && (b.y2 - 750.0).abs() < 0.01, "{b:?}");
    }

    #[test]
    fn nms_is_class_aware() {
        let (ch, n) = (4 + 3, 10);
        let mut raw = vec![0.0; ch * n];
        put(&mut raw, n, 0, [100.0, 100.0, 40.0, 40.0], 0, 0.9);
        put(&mut raw, n, 1, [101.0, 100.0, 40.0, 40.0], 0, 0.8); // duplicate, same class
        put(&mut raw, n, 2, [100.0, 100.0, 40.0, 40.0], 1, 0.7); // overlaps but other class
        let mut d = Decoder::new(DecodeConfig::default());
        let mut out = Vec::new();
        d.decode(&raw, ch, n, &lb(), &mut out).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].class, out[1].class), (0, 1));
    }

    #[test]
    fn confidence_filter_and_bad_shape() {
        let (ch, n) = (4 + 2, 5);
        let mut raw = vec![0.0; ch * n];
        put(&mut raw, n, 0, [100.0, 100.0, 40.0, 40.0], 0, 0.1);
        let mut d = Decoder::new(DecodeConfig::default());
        let mut out = Vec::new();
        d.decode(&raw, ch, n, &lb(), &mut out).unwrap();
        assert!(out.is_empty());
        assert!(d.decode(&raw[..10], ch, n, &lb(), &mut out).is_err());
    }
}
