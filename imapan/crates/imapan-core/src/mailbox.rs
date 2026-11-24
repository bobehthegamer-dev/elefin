//! Latest-wins single-slot mailbox: a slow consumer sees the newest item and the number of items
//! it missed, instead of an ever-growing queue (which is latency in a real-time pipeline).

use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

struct State<T> {
    item: Option<T>,
    dropped: u32,
    closed: bool,
}

pub struct LatestSlot<T> {
    st: Mutex<State<T>>,
    cv: Condvar,
}

impl<T> Default for LatestSlot<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> LatestSlot<T> {
    pub fn new() -> Self {
        Self { st: Mutex::new(State { item: None, dropped: 0, closed: false }), cv: Condvar::new() }
    }

    /// Stores `v`, replacing (and counting) any unconsumed item.
    pub fn put(&self, v: T) {
        let mut s = self.st.lock().unwrap();
        if s.item.replace(v).is_some() {
            s.dropped += 1;
        }
        self.cv.notify_one();
    }

    /// Blocks up to `timeout`. Returns the item and how many were dropped since the last take.
    pub fn take(&self, timeout: Duration) -> Option<(T, u32)> {
        let deadline = Instant::now() + timeout;
        let mut s = self.st.lock().unwrap();
        loop {
            if let Some(v) = s.item.take() {
                let d = std::mem::take(&mut s.dropped);
                return Some((v, d));
            }
            if s.closed {
                return None;
            }
            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            s = self.cv.wait_timeout(s, deadline - now).unwrap().0;
        }
    }

    pub fn close(&self) {
        self.st.lock().unwrap().closed = true;
        self.cv.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn latest_wins_and_counts_drops() {
        let s = LatestSlot::new();
        s.put(1);
        s.put(2);
        s.put(3);
        assert_eq!(s.take(Duration::from_millis(1)), Some((3, 2)));
        assert_eq!(s.take(Duration::from_millis(1)), None);
    }
}
