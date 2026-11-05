use wasm_bindgen::prelude::*;

const MAX_SEQ: usize = 256;

// ---------- PRNG (deterministic init) ----------
struct XorShift { state: u32 }
impl XorShift {
    fn new(seed: u32) -> Self { Self { state: seed.max(1) } }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        self.state = x;
        x
    }
    fn rand(&mut self) -> f32 { (self.next_u32() as f32) / (u32::MAX as f32) }
    fn fill_norm(&mut self, dst: &mut [f32], scale: f32) {
        for v in dst.iter_mut() { *v = (self.rand() * 2.0 - 1.0) * scale; }
    }
}

// ---------- RNN ----------
#[wasm_bindgen]
pub struct Rnn {
    v: usize,
    h: usize,
    // weights
    we: Vec<f32>, wh: Vec<f32>, wy: Vec<f32>, bh: Vec<f32>, by: Vec<f32>,
    // grads
    dwe: Vec<f32>, dwh: Vec<f32>, dwy: Vec<f32>, dbh: Vec<f32>, dby: Vec<f32>,
    // adam
    m_we: Vec<f32>, m_wh: Vec<f32>, m_wy: Vec<f32>, m_bh: Vec<f32>, m_by: Vec<f32>,
    v_we: Vec<f32>, v_wh: Vec<f32>, v_wy: Vec<f32>, v_bh: Vec<f32>, v_by: Vec<f32>,
    t: u32,
    lr: f32,
    // prealloc scratch
    hs: Vec<f32>,      // [MAX_SEQ+1][H]
    probs: Vec<f32>,   // [MAX_SEQ][V]
    y: Vec<f32>,       // [V]
    dy: Vec<f32>,      // [V]
    dh: Vec<f32>,      // [H]
    dh_new: Vec<f32>,  // [H]
    dh_raw: Vec<f32>,  // [H]
    dh_prev: Vec<f32>, // [H]
    h_new: Vec<f32>,   // [H]
    h_last: Vec<f32>,  // [H]
}

#[wasm_bindgen]
impl Rnn {
    #[wasm_bindgen(constructor)]
    pub fn new(v: usize, h: usize, seed: u32) -> Rnn {
        let mut rng = XorShift::new(seed);
        let mut we = vec![0.0; v * h];
        let mut wh = vec![0.0; h * h];
        let mut wy = vec![0.0; h * v];
        rng.fill_norm(&mut we, 0.1);
        rng.fill_norm(&mut wh, 0.1);
        rng.fill_norm(&mut wy, 0.1);
        Rnn {
            v, h,
            we, wh, wy,
            bh: vec![0.0; h],
            by: vec![0.0; v],
            dwe: vec![0.0; v * h], dwh: vec![0.0; h * h], dwy: vec![0.0; h * v],
            dbh: vec![0.0; h], dby: vec![0.0; v],
            m_we: vec![0.0; v * h], m_wh: vec![0.0; h * h], m_wy: vec![0.0; h * v],
            m_bh: vec![0.0; h], m_by: vec![0.0; v],
            v_we: vec![0.0; v * h], v_wh: vec![0.0; h * h], v_wy: vec![0.0; h * v],
            v_bh: vec![0.0; h], v_by: vec![0.0; v],
            t: 0, lr: 0.01,
            hs: vec![0.0; (MAX_SEQ + 1) * h],
            probs: vec![0.0; MAX_SEQ * v],
            y: vec![0.0; v],
            dy: vec![0.0; v],
            dh: vec![0.0; h],
            dh_new: vec![0.0; h],
            dh_raw: vec![0.0; h],
            dh_prev: vec![0.0; h],
            h_new: vec![0.0; h],
            h_last: vec![0.0; h],
        }
    }

    pub fn set_lr(&mut self, lr: f32) { self.lr = lr; }
    pub fn get_v(&self) -> usize { self.v }
    pub fn get_h(&self) -> usize { self.h }

    pub fn zero_grad(&mut self) {
        self.dwe.fill(0.0);
        self.dwh.fill(0.0);
        self.dwy.fill(0.0);
        self.dbh.fill(0.0);
        self.dby.fill(0.0);
    }

    /// forward + BPTT. grads accumulate. writes last hidden to self.h_last.
    pub fn forward_backward(&mut self, inputs: &[u32], targets: &[u32], hprev: &[f32]) -> f32 {
        let v = self.v;
        let h = self.h;
        let t_len = inputs.len().min(targets.len()).min(MAX_SEQ);
        if t_len == 0 || hprev.len() < h { return 0.0; }

        self.hs[..h].copy_from_slice(&hprev[..h]);

        // ---------- forward ----------
        let mut loss = 0.0f32;
        for t in 0..t_len {
            let x = inputs[t] as usize;
            let target = targets[t] as usize;
            let x_off = x * h;
            let hp_off = t * h;
            for i in 0..h {
                let mut s = self.we[x_off + i] + self.bh[i];
                let row_off = i * h;
                for j in 0..h {
                    s += self.wh[row_off + j] * self.hs[hp_off + j];
                }
                self.h_new[i] = s.tanh();
            }
            self.hs[(t + 1) * h..(t + 2) * h].copy_from_slice(&self.h_new);

            for k in 0..v {
                let mut s = self.by[k];
                for i in 0..h {
                    s += self.wy[i * v + k] * self.h_new[i];
                }
                self.y[k] = s;
            }

            let mut max_y = f32::NEG_INFINITY;
            for k in 0..v { if self.y[k] > max_y { max_y = self.y[k]; } }
            let mut sum = 0.0f32;
            let p_off = t * v;
            for k in 0..v {
                let e = (self.y[k] - max_y).exp();
                self.probs[p_off + k] = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for k in 0..v { self.probs[p_off + k] *= inv; }
            loss -= self.probs[p_off + target].max(1e-12).ln();
        }

        // ---------- backward ----------
        self.dh.fill(0.0);
        for t in (0..t_len).rev() {
            let target = targets[t] as usize;
            let x = inputs[t] as usize;
            let x_off = x * h;
            let p_off = t * v;
            let ht_off = (t + 1) * h;
            let hp_off = t * h;

            for k in 0..v { self.dy[k] = self.probs[p_off + k]; }
            self.dy[target] -= 1.0;

            // dWy += h_{t+1} ⊗ dy ; dby += dy
            for i in 0..h {
                let hi = self.hs[ht_off + i];
                let off = i * v;
                for k in 0..v {
                    self.dwy[off + k] += hi * self.dy[k];
                }
            }
            for k in 0..v { self.dby[k] += self.dy[k]; }

            // dh_new = dh + Wy^T·dy
            for i in 0..h {
                let mut s = self.dh[i];
                let off = i * v;
                for k in 0..v {
                    s += self.wy[off + k] * self.dy[k];
                }
                self.dh_new[i] = s;
            }

            // dh_raw = (1 - h²) ⊙ dh_new
            for i in 0..h {
                let hi = self.hs[ht_off + i];
                self.dh_raw[i] = (1.0 - hi * hi) * self.dh_new[i];
            }

            // dbh += dh_raw ; dWe[x] += dh_raw
            for i in 0..h {
                self.dbh[i] += self.dh_raw[i];
                self.dwe[x_off + i] += self.dh_raw[i];
            }

            // dWh += dh_raw ⊗ h_prev
            for i in 0..h {
                let dhi = self.dh_raw[i];
                let off = i * h;
                for j in 0..h {
                    self.dwh[off + j] += dhi * self.hs[hp_off + j];
                }
            }

            // dh = Wh^T·dh_raw
            for j in 0..h {
                let mut s = 0.0;
                for i in 0..h {
                    s += self.wh[i * h + j] * self.dh_raw[i];
                }
                self.dh_prev[j] = s;
            }
            self.dh.copy_from_slice(&self.dh_prev);
        }

        self.h_last.copy_from_slice(&self.hs[t_len * h..(t_len + 1) * h]);

        loss / (t_len as f32)
    }

    pub fn get_h_last(&self) -> Vec<f32> { self.h_last.clone() }

    pub fn adam_step(&mut self) {
        self.t += 1;
        let t = self.t as f32;
        let b1 = 0.9f32;
        let b2 = 0.999f32;
        let eps = 1e-8f32;
        let c1 = 1.0 - b1.powf(t);
        let c2 = 1.0 - b2.powf(t);
        let step = self.lr * c2.sqrt() / c1;

        adam_update(&mut self.we, &self.dwe, &mut self.m_we, &mut self.v_we, b1, b2, step, eps);
        adam_update(&mut self.wh, &self.dwh, &mut self.m_wh, &mut self.v_wh, b1, b2, step, eps);
        adam_update(&mut self.wy, &self.dwy, &mut self.m_wy, &mut self.v_wy, b1, b2, step, eps);
        adam_update(&mut self.bh, &self.dbh, &mut self.m_bh, &mut self.v_bh, b1, b2, step, eps);
        adam_update(&mut self.by, &self.dby, &mut self.m_by, &mut self.v_by, b1, b2, step, eps);
    }

    pub fn scale_grads(&mut self, s: f32) {
        for a in [&mut self.dwe, &mut self.dwh, &mut self.dwy, &mut self.dbh, &mut self.dby] {
            for x in a.iter_mut() { *x *= s; }
        }
    }

    pub fn sample(&self, seed: &[u32], len: usize, temp: f32) -> Vec<u32> {
        let h = self.h;
        let v = self.v;
        let mut h_state = vec![0.0f32; h];
        let mut new_h = vec![0.0f32; h];
        let mut y = vec![0.0f32; v];

        for &c in seed {
            self.forward_single(c as usize, &mut h_state, &mut new_h, &mut y);
        }

        let mut out = Vec::with_capacity(seed.len() + len);
        out.extend_from_slice(seed);
        for _ in 0..len {
            let pick = sample_from_logits(&y, temp, v);
            out.push(pick);
            self.forward_single(pick as usize, &mut h_state, &mut new_h, &mut y);
        }
        out
    }

    pub fn export_weights(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(
            self.we.len() + self.wh.len() + self.wy.len() + self.bh.len() + self.by.len()
        );
        out.extend_from_slice(&self.we);
        out.extend_from_slice(&self.wh);
        out.extend_from_slice(&self.wy);
        out.extend_from_slice(&self.bh);
        out.extend_from_slice(&self.by);
        out
    }

    pub fn import_weights(&mut self, w: &[f32]) -> Result<(), JsValue> {
        let total = self.we.len() + self.wh.len() + self.wy.len() + self.bh.len() + self.by.len();
        if w.len() != total {
            return Err(JsValue::from_str(&format!(
                "weights len mismatch: expected {}, got {}", total, w.len()
            )));
        }
        let mut off = 0usize;
        macro_rules! take {
            ($dst:expr) => {{
                let n = $dst.len();
                $dst.copy_from_slice(&w[off..off + n]);
                off += n;
            }};
        }
        take!(self.we);
        take!(self.wh);
        take!(self.wy);
        take!(self.bh);
        take!(self.by);
        Ok(())
    }

    fn forward_single(&self, x: usize, h_state: &mut [f32], new_h: &mut [f32], y: &mut [f32]) {
        let v = self.v;
        let h = self.h;
        let x_off = x * h;
        for i in 0..h {
            let mut s = self.we[x_off + i] + self.bh[i];
            let row_off = i * h;
            for j in 0..h {
                s += self.wh[row_off + j] * h_state[j];
            }
            new_h[i] = s.tanh();
        }
        h_state.copy_from_slice(new_h);
        for k in 0..v {
            let mut s = self.by[k];
            for i in 0..h {
                s += self.wy[i * v + k] * h_state[i];
            }
            y[k] = s;
        }
    }
}

fn sample_from_logits(y: &[f32], temp: f32, v: usize) -> u32 {
    let mut max_y = f32::NEG_INFINITY;
    for k in 0..v { if y[k] > max_y { max_y = y[k]; } }
    let mut probs = vec![0.0f32; v];
    let mut sum = 0.0f32;
    let inv_temp = 1.0 / temp.max(1e-3);
    for k in 0..v {
        let e = ((y[k] - max_y) * inv_temp).exp();
        probs[k] = e;
        sum += e;
    }
    let mut r = (js_sys::Math::random() as f32) * sum;
    for k in 0..v {
        r -= probs[k];
        if r <= 0.0 { return k as u32; }
    }
    (v - 1) as u32
}

fn adam_update(p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32],
               b1: f32, b2: f32, step: f32, eps: f32) {
    for i in 0..p.len() {
        m[i] = b1 * m[i] + (1.0 - b1) * g[i];
        v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
        p[i] -= step * m[i] / (v[i].sqrt() + eps);
    }
}
