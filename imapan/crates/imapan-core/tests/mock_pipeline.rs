use rt_core::backend::{Backend, NextFrame};
use rt_core::decode::DecodeConfig;
use rt_core::mock::{MockBackend, MockConfig};
use rt_core::pipeline::Pipeline;
use rt_core::tracker::TrackerConfig;
use rt_core::types::FrameResult;
use std::time::{Duration, Instant};

#[test]
fn tracks_all_objects_with_stable_ids_through_the_full_pipeline() {
    let cfg = MockConfig { fps: 240.0, detect_every_n: 4, objects: 5, ..Default::default() };
    let mut be = MockBackend::new(cfg);
    let mut pipe = Pipeline::new(DecodeConfig::default(), TrackerConfig::default());
    let mut f = FrameResult::empty();
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(1500) {
        if let NextFrame::Frame = be.next_frame(Duration::from_millis(50), &mut f).unwrap() {
            let clock = || be.now_ns();
            pipe.process(&f, clock).unwrap();
        }
    }
    assert!(pipe.stats.frames > 200, "only {} frames", pipe.stats.frames);
    assert_eq!(pipe.tracks.len(), 5, "expected all 5 objects tracked");
    // 3% simulated misses must not cause id churn.
    assert!(pipe.tracker().ids_created() <= 7, "ids created: {}", pipe.tracker().ids_created());
}
