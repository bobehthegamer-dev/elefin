//! The seam between the GPU side (capture + preprocessing + TensorRT) and the Rust side.

use crate::types::FrameResult;
use std::time::Duration;

pub enum NextFrame {
    /// `out` has been filled.
    Frame,
    Timeout,
}

pub trait Backend {
    fn name(&self) -> &str;
    /// Monotonic nanoseconds on the same clock as `FrameResult::{capture,ready}_ts_ns`.
    fn now_ns(&self) -> i64;
    /// Blocks up to `timeout` for the newest frame (latest-wins; older ones are dropped).
    fn next_frame(&mut self, timeout: Duration, out: &mut FrameResult) -> Result<NextFrame, String>;
}
