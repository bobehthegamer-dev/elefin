//! Fixed-bucket latency histogram (10 µs buckets up to 100 ms): O(1) record, no allocation.

#[derive(Clone)]
pub struct Hist {
    buckets: Vec<u32>,
    count: u64,
    sum_us: f64,
    max_us: f32,
}

const BUCKET_US: f32 = 10.0;
const N_BUCKETS: usize = 10_000;

impl Default for Hist {
    fn default() -> Self {
        Self::new()
    }
}

impl Hist {
    pub fn new() -> Self {
        Self { buckets: vec![0; N_BUCKETS + 1], count: 0, sum_us: 0.0, max_us: 0.0 }
    }
    pub fn record_us(&mut self, us: f32) {
        let us = us.max(0.0);
        let i = ((us / BUCKET_US) as usize).min(N_BUCKETS);
        self.buckets[i] += 1;
        self.count += 1;
        self.sum_us += us as f64;
        self.max_us = self.max_us.max(us);
    }
    pub fn record_ns(&mut self, ns: i64) {
        self.record_us(ns as f32 / 1000.0);
    }
    pub fn count(&self) -> u64 {
        self.count
    }
    pub fn mean_us(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            (self.sum_us / self.count as f64) as f32
        }
    }
    pub fn max_us(&self) -> f32 {
        self.max_us
    }
    /// Upper edge of the bucket holding the q-quantile (q in 0..=1).
    pub fn percentile_us(&self, q: f32) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        let target = ((self.count as f64 * q as f64).ceil() as u64).max(1);
        let mut acc = 0u64;
        for (i, &c) in self.buckets.iter().enumerate() {
            acc += c as u64;
            if acc >= target {
                return ((i + 1) as f32 * BUCKET_US).min(self.max_us.max(BUCKET_US));
            }
        }
        self.max_us
    }
    pub fn reset(&mut self) {
        self.buckets.iter_mut().for_each(|b| *b = 0);
        self.count = 0;
        self.sum_us = 0.0;
        self.max_us = 0.0;
    }
    pub fn summary(&self) -> String {
        format!(
            "p50 {:>7.0}us  p99 {:>7.0}us  max {:>7.0}us  mean {:>7.0}us  n={}",
            self.percentile_us(0.5),
            self.percentile_us(0.99),
            self.max_us,
            self.mean_us(),
            self.count
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn percentiles() {
        let mut h = Hist::new();
        for i in 1..=100 {
            h.record_us(i as f32 * 100.0); // 100us .. 10ms
        }
        assert!((h.percentile_us(0.5) - 5000.0).abs() <= 20.0, "{}", h.percentile_us(0.5));
        assert!((h.percentile_us(0.99) - 9900.0).abs() <= 20.0);
        assert_eq!(h.max_us(), 10000.0);
        assert_eq!(h.count(), 100);
    }
}
