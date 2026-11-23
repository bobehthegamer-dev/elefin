//! Real-time detection post-processing and tracking.
//!
//! The GPU side (capture, preprocessing, TensorRT) lives in `native/` and reaches this crate
//! through the [`backend::Backend`] trait; [`mock::MockBackend`] stands in when it isn't built.

pub mod backend;
pub mod decode;
pub mod kalman;
pub mod logger;
pub mod mailbox;
pub mod mock;
pub mod pipeline;
pub mod stats;
pub mod tracker;
pub mod types;

#[cfg(feature = "native")]
pub mod native;
