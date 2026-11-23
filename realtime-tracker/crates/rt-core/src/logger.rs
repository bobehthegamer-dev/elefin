//! Asynchronous JSON-lines logger. The frame loop only does a non-blocking `try_send`; if the
//! disk can't keep up, records are dropped (and counted) rather than stalling the pipeline.

use crate::tracker::TrackOut;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::JoinHandle;

pub struct LogRecord {
    pub frame_id: u64,
    pub capture_ts_ns: i64,
    pub ran_inference: bool,
    pub n_dets: usize,
    pub e2e_us: f32,
    pub tracks: Vec<TrackOut>,
}

pub struct Logger {
    tx: Option<SyncSender<LogRecord>>,
    handle: Option<JoinHandle<()>>,
    dropped: Arc<AtomicU64>,
}

impl Logger {
    pub fn create(path: &Path, capacity: usize) -> std::io::Result<Self> {
        let mut w = BufWriter::with_capacity(1 << 20, File::create(path)?);
        let (tx, rx) = sync_channel::<LogRecord>(capacity);
        let handle = std::thread::spawn(move || {
            for r in rx {
                let _ = write!(
                    w,
                    "{{\"frame\":{},\"ts_ns\":{},\"infer\":{},\"dets\":{},\"e2e_us\":{:.0},\"tracks\":[",
                    r.frame_id, r.capture_ts_ns, r.ran_inference, r.n_dets, r.e2e_us
                );
                for (i, t) in r.tracks.iter().enumerate() {
                    if i > 0 {
                        let _ = w.write_all(b",");
                    }
                    let _ = write!(
                        w,
                        "{{\"id\":{},\"cls\":{},\"score\":{:.3},\"box\":[{:.1},{:.1},{:.1},{:.1}],\"v\":[{:.0},{:.0}],\"coast\":{}}}",
                        t.id, t.class, t.score, t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2, t.vx, t.vy, t.coasting
                    );
                }
                let _ = w.write_all(b"]}\n");
            }
            let _ = w.flush();
        });
        Ok(Self { tx: Some(tx), handle: Some(handle), dropped: Arc::new(AtomicU64::new(0)) })
    }

    pub fn log(&self, r: LogRecord) {
        if let Some(tx) = &self.tx {
            if let Err(TrySendError::Full(_)) = tx.try_send(r) {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

impl Drop for Logger {
    fn drop(&mut self) {
        self.tx.take(); // closes the channel; writer drains and flushes
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
