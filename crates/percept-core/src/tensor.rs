//! Minimal dense tensor: row-major `Vec<f32>` with shape. Every op is an
//! explicit forward/backward pair — the training loop composes them by hand.
//! This is deliberately *not* a general autodiff framework; it is the smallest
//! auditable surface that the transformer needs.

use crate::rng::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Tensor { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn from_rows(rows: Vec<Vec<f32>>) -> Self {
        let r = rows.len();
        let c = rows.first().map(|x| x.len()).unwrap_or(0);
        assert!(rows.iter().all(|x| x.len() == c), "ragged rows");
        Tensor { rows: r, cols: c, data: rows.into_iter().flatten().collect() }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Matrix multiply: self (r×k) * other (k×c) -> (r×c).
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.cols, other.rows, "matmul shape mismatch");
        let (r, k, c) = (self.rows, self.cols, other.cols);
        let mut out = Tensor::zeros(r, c);
        for i in 0..r {
            let a_row = &self.data[i * k..(i + 1) * k];
            let out_row = &mut out.data[i * c..(i + 1) * c];
            for (p, &a) in a_row.iter().enumerate() {
                if a == 0.0 {
                    continue;
                }
                let b_row = &other.data[p * c..(p + 1) * c];
                for (o, &b) in out_row.iter_mut().zip(b_row.iter()) {
                    *o += a * b;
                }
            }
        }
        out
    }

    /// Backward of matmul. Given upstream grad G (r×c), accumulate
    /// dA = G * B^T into `grad_a` and dB = A^T * G into `grad_b`.
    pub fn matmul_backward(
        &self,
        other: &Tensor,
        g: &Tensor,
        grad_a: &mut Tensor,
        grad_b: &mut Tensor,
    ) {
        assert_eq!(g.rows, self.rows);
        assert_eq!(g.cols, other.cols);
        // grad_a[i,p] += sum_c g[i,c] * b[p,c]
        for i in 0..self.rows {
            let g_row = &g.data[i * g.cols..(i + 1) * g.cols];
            for p in 0..self.cols {
                let b_row = &other.data[p * g.cols..(p + 1) * g.cols];
                let mut acc = 0.0f32;
                for (gv, bv) in g_row.iter().zip(b_row.iter()) {
                    acc += gv * bv;
                }
                grad_a.data[i * self.cols + p] += acc;
            }
        }
        // grad_b[p,c] += sum_i a[i,p] * g[i,c]
        for i in 0..self.rows {
            let g_row = &g.data[i * g.cols..(i + 1) * g.cols];
            for p in 0..self.cols {
                let a = self.data[i * self.cols + p];
                if a == 0.0 {
                    continue;
                }
                let gb_row =
                    &mut grad_b.data[p * g.cols..(p + 1) * g.cols];
                for (gb, gv) in gb_row.iter_mut().zip(g_row.iter()) {
                    *gb += a * gv;
                }
            }
        }
    }

    /// Add `other` broadcast along rows (other is 1×cols).
    pub fn add_row_broadcast(&self, other: &Tensor) -> Tensor {
        assert_eq!(other.rows, 1);
        assert_eq!(other.cols, self.cols);
        let mut out = self.clone();
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.data[i * self.cols + j] += other.data[j];
            }
        }
        out
    }

    /// Transpose.
    pub fn t(&self) -> Tensor {
        let mut out = Tensor::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        out
    }

    /// GELU (tanh approximation), elementwise. No clamp: tanh saturates
    /// gracefully, and clamping would break gelu(x) ≈ x for large x.
    pub fn gelu(&self) -> Tensor {
        Tensor {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .map(|&x| {
                    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
                })
                .collect(),
        }
    }

    pub fn gelu_backward(&self, g: &Tensor) -> Tensor {
        Tensor {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(g.data.iter())
                .map(|(&x, &gv)| {
                    let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
                    let tanh = inner.tanh();
                    let d_inner = 0.7978845608 * (1.0 + 3.0 * 0.044715 * x * x);
                    let d = 0.5 * (1.0 + tanh) + 0.5 * x * (1.0 - tanh * tanh) * d_inner;
                    gv * d
                })
                .collect(),
        }
    }

    /// LayerNorm over each row. Returns (normalized, mean, rstd) for backward.
    pub fn layer_norm(&self, eps: f32) -> (Tensor, Tensor, Tensor) {
        let mut norm = Tensor::zeros(self.rows, self.cols);
        let mut means = Tensor::zeros(self.rows, 1);
        let mut rstds = Tensor::zeros(self.rows, 1);
        for i in 0..self.rows {
            let row = &self.data[i * self.cols..(i + 1) * self.cols];
            let mean = row.iter().sum::<f32>() / self.cols as f32;
            let var = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>()
                / self.cols as f32;
            let rstd = 1.0 / (var + eps).sqrt();
            let out_row = &mut norm.data[i * self.cols..(i + 1) * self.cols];
            for (o, &x) in out_row.iter_mut().zip(row.iter()) {
                *o = (x - mean) * rstd;
            }
            means.data[i] = mean;
            rstds.data[i] = rstd;
        }
        (norm, means, rstds)
    }

    pub fn layer_norm_backward(
        &self,
        g: &Tensor,
        means: &Tensor,
        rstds: &Tensor,
    ) -> Tensor {
        let n = self.cols as f32;
        let mut out = Tensor::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            let mean = means.data[i];
            let rstd = rstds.data[i];
            let row = &self.data[i * self.cols..(i + 1) * self.cols];
            let g_row = &g.data[i * self.cols..(i + 1) * self.cols];
            let out_row = &mut out.data[i * self.cols..(i + 1) * self.cols];
            let sum_g = g_row.iter().sum::<f32>();
            // sum over k of g_k * x_hat_k, where x_hat = (x - mean) * rstd
            let sum_gx: f32 = g_row
                .iter()
                .zip(row.iter())
                .map(|(gv, x)| gv * (x - mean) * rstd)
                .sum();
            for j in 0..self.cols {
                out_row[j] = (g_row[j] - sum_g / n
                    - (x_hat(row[j], mean, rstd) * sum_gx / n))
                    * rstd;
            }
        }
        out
    }

    /// Softmax over each row, numerically stable.
    pub fn softmax(&self) -> Tensor {
        let mut out = Tensor::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            let row = &self.data[i * self.cols..(i + 1) * self.cols];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            let out_row = &mut out.data[i * self.cols..(i + 1) * self.cols];
            for (o, &x) in out_row.iter_mut().zip(row.iter()) {
                let e = (x - max).exp();
                *o = e;
                sum += e;
            }
            for o in out_row.iter_mut() {
                *o /= sum;
            }
        }
        out
    }

    /// Softmax backward given (softmax output, upstream grad).
    pub fn softmax_backward(&self, probs: &Tensor, g: &Tensor) -> Tensor {
        let mut out = Tensor::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            let p_row = &probs.data[i * self.cols..(i + 1) * self.cols];
            let g_row = &g.data[i * self.cols..(i + 1) * self.cols];
            let dot: f32 = p_row.iter().zip(g_row.iter()).map(|(p, gv)| p * gv).sum();
            let out_row = &mut out.data[i * self.cols..(i + 1) * self.cols];
            for (o, (p, gv)) in out_row.iter_mut().zip(p_row.iter().zip(g_row.iter())) {
                *o = p * (gv - dot);
            }
        }
        out
    }

    pub fn scale(&self, s: f32) -> Tensor {
        Tensor {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|x| x * s).collect(),
        }
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        Tensor {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    pub fn into_shape(self, rows: usize, cols: usize) -> Tensor {
        assert_eq!(rows * cols, self.data.len());
        Tensor { rows, cols, data: self.data }
    }

    /// Xavier/Glorot init.
    pub fn xavier(rows: usize, cols: usize, rng: &mut Rng) -> Self {
        let bound = (6.0 / (rows + cols) as f32).sqrt();
        let data = (0..rows * cols)
            .map(|_| {
                let u = rng.next_f32();
                (u * 2.0 - 1.0) * bound
            })
            .collect();
        Tensor { rows, cols, data }
    }

    /// Zero-mean normal init with given std.
    pub fn normal(rows: usize, cols: usize, std: f32, rng: &mut Rng) -> Self {
        let data = (0..rows * cols).map(|_| rng.next_normal() * std).collect();
        Tensor { rows, cols, data }
    }
}

#[inline]
fn x_hat(x: f32, mean: f32, rstd: f32) -> f32 {
    (x - mean) * rstd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_matches_naive() {
        let a = Tensor::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = Tensor::from_rows(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let c = a.matmul(&b);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_rectangular() {
        let a = Tensor::from_rows(vec![vec![1.0, 0.0, 2.0]]);
        let b = Tensor::from_rows(vec![vec![1.0], vec![2.0], vec![3.0]]);
        let c = a.matmul(&b);
        assert_eq!(c.rows, 1);
        assert_eq!(c.cols, 1);
        assert!((c.data[0] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn transpose_roundtrip() {
        let a = Tensor::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let t = a.t();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        let back = t.t();
        assert_eq!(back, a);
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let a = Tensor::from_rows(vec![vec![1.0, 2.0, 3.0], vec![1000.0, 1000.0, 1000.0]]);
        let s = a.softmax();
        for i in 0..2 {
            let sum: f32 = s.data[i * 3..(i + 1) * 3].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
        // stable under large logits
        assert!((s.data[3] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn gelu_at_zero_and_far() {
        let a = Tensor::from_rows(vec![vec![0.0, 10.0, -10.0]]);
        let g = a.gelu();
        assert!(g.data[0].abs() < 1e-6);
        assert!((g.data[1] - 10.0).abs() < 1e-3);
        assert!(g.data[2].abs() < 1e-3);
    }

    #[test]
    fn layer_norm_zero_mean_unit_var() {
        let a = Tensor::from_rows(vec![vec![1.0, 2.0, 3.0, 6.0]]);
        let (n, _, _) = a.layer_norm(1e-5);
        let mean: f32 = n.data.iter().sum::<f32>() / 4.0;
        let var: f32 = n.data.iter().map(|x| x * x).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
        assert!((var - 1.0).abs() < 1e-2);
    }
}
