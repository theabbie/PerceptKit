//! Adam optimizer with bias correction and optional weight decay.
//! Optimizer state is lazily sized to the parameter list on first step.

use crate::tensor::Tensor;

pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    step: usize,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            step: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn step(&mut self, params: &mut [(&mut Tensor, &Tensor)]) {
        // lazily size state
        if self.m.len() != params.len() {
            self.m = params.iter().map(|(p, _)| vec![0.0; p.len()]).collect();
            self.v = params.iter().map(|(p, _)| vec![0.0; p.len()]).collect();
        }
        self.step += 1;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let bc1 = 1.0 - b1.powi(self.step as i32);
        let bc2 = 1.0 - b2.powi(self.step as i32);
        for (idx, (p, g)) in params.iter_mut().enumerate() {
            let m = &mut self.m[idx];
            let v = &mut self.v[idx];
            for i in 0..p.data.len() {
                let gi = g.data[i] + self.weight_decay * p.data[i];
                m[i] = b1 * m[i] + (1.0 - b1) * gi;
                v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
                let mh = m[i] / bc1;
                let vh = v[i] / bc2;
                p.data[i] -= self.lr * mh / (vh.sqrt() + self.eps);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adam_minimizes_quadratic() {
        // minimize f(x) = (x - 3)^2 elementwise
        let mut w = Tensor::from_rows(vec![vec![0.0, 0.0, 0.0]]);
        let mut opt = Adam::new(0.1);
        for _ in 0..500 {
            let g = Tensor::from_rows(vec![w.data.iter().map(|x| 2.0 * (x - 3.0)).collect()]);
            opt.step(&mut [(&mut w, &g)]);
        }
        for x in &w.data {
            assert!((x - 3.0).abs() < 0.01, "x = {x}");
        }
    }

    #[test]
    fn state_resizes_on_new_shape() {
        let mut w = Tensor::from_rows(vec![vec![0.0; 8]]);
        let mut opt = Adam::new(0.01);
        let g = Tensor::from_rows(vec![vec![1.0; 8]]);
        opt.step(&mut [(&mut w, &g)]);
        let mut w2 = Tensor::from_rows(vec![vec![0.0; 4], vec![0.0; 4]]);
        opt.step(&mut [(&mut w2, &g)]);
        assert!(w2.data.iter().any(|x| *x != 0.0));
    }
}
