//! Box state filter. The constant-velocity model with diagonal noise decouples into four
//! independent 1-D position/velocity filters (cx, cy, w, h), which is exact and avoids matrix code.
//! Time steps come from frame timestamps (seconds), never from wall-clock arrival.

use crate::types::BBox;

#[derive(Clone, Copy, Debug)]
pub struct Kf1 {
    pub x: f32,
    pub v: f32,
    p00: f32,
    p01: f32,
    p11: f32,
}

impl Kf1 {
    pub fn new(x: f32, r: f32, vel_var: f32) -> Self {
        Self { x, v: 0.0, p00: r, p01: 0.0, p11: vel_var }
    }

    /// `q` is the acceleration noise spectral density (px^2 / s^3).
    pub fn predict(&mut self, dt: f32, q: f32) {
        if dt <= 0.0 {
            return;
        }
        self.x += self.v * dt;
        self.p00 += dt * (2.0 * self.p01 + dt * self.p11);
        self.p01 += dt * self.p11;
        // continuous white-noise acceleration
        self.p00 += q * dt * dt * dt / 3.0;
        self.p01 += q * dt * dt / 2.0;
        self.p11 += q * dt;
    }

    pub fn update(&mut self, z: f32, r: f32) {
        let s = self.p00 + r;
        let (k0, k1) = (self.p00 / s, self.p01 / s);
        let y = z - self.x;
        self.x += k0 * y;
        self.v += k1 * y;
        let (p00, p01, p11) = (self.p00, self.p01, self.p11);
        self.p00 = (1.0 - k0) * p00;
        self.p01 = (1.0 - k0) * p01;
        self.p11 = p11 - k1 * p01;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FilterParams {
    /// Measurement noise std-dev as a fraction of box size.
    pub meas_frac: f32,
    /// Initial velocity std-dev, in box sizes per second.
    pub vel_init_frac: f32,
    /// Acceleration noise std-dev, in box sizes per second^2 (position filters).
    pub accel_frac: f32,
}

impl Default for FilterParams {
    fn default() -> Self {
        Self { meas_frac: 0.05, vel_init_frac: 3.0, accel_frac: 20.0 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BoxFilter {
    f: [Kf1; 4], // cx, cy, w, h
    params: FilterParams,
}

impl BoxFilter {
    pub fn new(b: &BBox, params: FilterParams) -> Self {
        let (cx, cy) = b.center();
        let (w, h) = (b.width().max(1.0), b.height().max(1.0));
        let mk = |x: f32, size: f32| {
            let r = (params.meas_frac * size).powi(2);
            let vv = (params.vel_init_frac * size).powi(2);
            Kf1::new(x, r, vv)
        };
        Self { f: [mk(cx, w), mk(cy, h), mk(w, w), mk(h, h)], params }
    }

    fn sizes(&self) -> [f32; 4] {
        let (w, h) = (self.f[2].x.max(1.0), self.f[3].x.max(1.0));
        [w, h, w, h]
    }

    pub fn predict(&mut self, dt: f32) {
        let s = self.sizes();
        for k in 0..4 {
            // Size (w, h) is assumed to change much more slowly than position.
            let a = self.params.accel_frac * s[k] * if k >= 2 { 0.1 } else { 1.0 };
            self.f[k].predict(dt, a * a);
        }
        self.f[2].x = self.f[2].x.max(1.0);
        self.f[3].x = self.f[3].x.max(1.0);
    }

    pub fn update(&mut self, b: &BBox) {
        let (cx, cy) = b.center();
        let z = [cx, cy, b.width().max(1.0), b.height().max(1.0)];
        let s = self.sizes();
        for k in 0..4 {
            let r = (self.params.meas_frac * s[k]).powi(2);
            self.f[k].update(z[k], r);
        }
    }

    pub fn bbox(&self) -> BBox {
        BBox::from_cxcywh(self.f[0].x, self.f[1].x, self.f[2].x, self.f[3].x)
    }

    /// Velocity of the box center in px/s.
    pub fn velocity(&self) -> (f32, f32) {
        (self.f[0].v, self.f[1].v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_constant_velocity() {
        let dt = 1.0 / 240.0;
        let (vx, vy) = (600.0f32, -240.0f32); // px/s
        let mut b = BBox::from_cxcywh(100.0, 100.0, 80.0, 160.0);
        let mut kf = BoxFilter::new(&b, FilterParams::default());
        for _ in 0..120 {
            b = BBox::from_cxcywh(b.center().0 + vx * dt, b.center().1 + vy * dt, 80.0, 160.0);
            kf.predict(dt);
            kf.update(&b);
        }
        let (ev, fv) = kf.velocity();
        assert!((ev - vx).abs() < 30.0, "vx {ev}");
        assert!((fv - vy).abs() < 30.0, "vy {fv}");
        // Coast 3 frames without measurements: prediction must stay near truth.
        for _ in 0..3 {
            b = BBox::from_cxcywh(b.center().0 + vx * dt, b.center().1 + vy * dt, 80.0, 160.0);
            kf.predict(dt);
        }
        let (px, py) = kf.bbox().center();
        let (tx, ty) = b.center();
        assert!((px - tx).abs() < 3.0 && (py - ty).abs() < 3.0, "({px},{py}) vs ({tx},{ty})");
    }
}
