//! Losses for the two task families:
//! - **Regression/scoring**: MSE and L1 on the global head.
//! - **Ranking**: pairwise Bradley-Terry / margin loss — robust when the
//!   absolute score scale is arbitrary (as with synthetic corruption
//!   severities).
//! - **Dense**: masked MSE/L1 on the dense head (only tokens with a target
//!   mask contribute), used for colorization/inpainting/transform tasks.

use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LossKind {
    /// Mean squared error on global score.
    MSE,
    /// Mean absolute error on global score.
    L1,
    /// Pairwise ranking: see [PairwiseInput].
    PairwiseMargin,
    /// Masked MSE on dense outputs (mask: 1.0 = supervise).
    DenseMaskedMSE,
    /// Masked L1 on dense outputs.
    DenseMaskedL1,
}

/// Input for pairwise ranking: model scores for two variants of the same
/// document (better, worse) plus a margin that must separate them.
pub struct PairwiseInput {
    pub score_better: f32,
    pub score_worse: f32,
    /// Required gap: score_better should exceed score_worse + margin.
    pub margin: f32,
}

/// A loss value plus the gradient wrt the model output(s) that produced it.
pub struct LossAndGrad {
    pub value: f32,
    /// Gradient wrt global score (shape (1,1)).
    pub g_global: Tensor,
    /// Gradient wrt dense outputs (shape (t, out_channels)), or `None` for
    /// scalar-only losses (MSE/L1/pairwise) that don't touch the dense head.
    pub g_dense: Option<Tensor>,
}

impl LossAndGrad {
    fn global_only(value: f32, g: f32) -> Self {
        LossAndGrad {
            value,
            g_global: Tensor::from_rows(vec![vec![g]]),
            g_dense: None,
        }
    }

    fn dense_only(value: f32, g_dense: Tensor) -> Self {
        LossAndGrad {
            value,
            g_global: Tensor::zeros(1, 1),
            g_dense: Some(g_dense),
        }
    }
}

/// MSE on global score. `pred` is the model's (1,1) global output.
pub fn mse(pred: f32, target: f32) -> LossAndGrad {
    let diff = pred - target;
    LossAndGrad::global_only(diff * diff, 2.0 * diff)
}

/// L1 on global score.
pub fn l1(pred: f32, target: f32) -> LossAndGrad {
    let diff = pred - target;
    let sign = if diff > 0.0 { 1.0 } else if diff < 0.0 { -1.0 } else { 0.0 };
    LossAndGrad::global_only(diff.abs(), sign)
}

/// Pairwise margin (hinge) ranking loss:
///   L = max(0, margin - (s_better - s_worse))
/// The caller applies -1/+1 gradients to the better/worse scores.
pub fn pairwise_margin(input: &PairwiseInput) -> LossAndGrad {
    let gap = input.score_better - input.score_worse;
    let viol = input.margin - gap;
    if viol > 0.0 {
        LossAndGrad::global_only(viol, -1.0)
    } else {
        LossAndGrad::global_only(0.0, 0.0)
    }
}

/// Masked dense MSE. `dense` is (t, out_channels), `target` same shape,
/// `mask` is (t, 1) with 1.0 = supervise this token.
/// Returns loss and grad wrt dense output (zeros where mask == 0).
pub fn dense_masked_mse(dense: &Tensor, target: &Tensor, mask: &Tensor) -> LossAndGrad {
    assert_eq!((dense.rows, dense.cols), (target.rows, target.cols));
    assert_eq!(mask.rows, dense.rows);
    let mut g = Tensor::zeros(dense.rows, dense.cols);
    let mut loss = 0.0f32;
    let mut count = 0.0f32;
    for i in 0..dense.rows {
        if mask.data[i] == 0.0 {
            continue;
        }
        count += 1.0;
        for j in 0..dense.cols {
            let diff = dense.data[i * dense.cols + j] - target.data[i * dense.cols + j];
            loss += diff * diff;
            g.data[i * dense.cols + j] = 2.0 * diff;
        }
    }
    let n = if count > 0.0 { count * dense.cols as f32 } else { 1.0 };
    for v in g.data.iter_mut() {
        *v /= n;
    }
    LossAndGrad::dense_only(loss / n, g)
}

/// Masked dense L1.
pub fn dense_masked_l1(dense: &Tensor, target: &Tensor, mask: &Tensor) -> LossAndGrad {
    assert_eq!((dense.rows, dense.cols), (target.rows, target.cols));
    assert_eq!(mask.rows, dense.rows);
    let mut g = Tensor::zeros(dense.rows, dense.cols);
    let mut loss = 0.0f32;
    let mut count = 0.0f32;
    for i in 0..dense.rows {
        if mask.data[i] == 0.0 {
            continue;
        }
        count += 1.0;
        for j in 0..dense.cols {
            let diff = dense.data[i * dense.cols + j] - target.data[i * dense.cols + j];
            loss += diff.abs();
            g.data[i * dense.cols + j] =
                if diff > 0.0 { 1.0 } else if diff < 0.0 { -1.0 } else { 0.0 };
        }
    }
    let n = if count > 0.0 { count * dense.cols as f32 } else { 1.0 };
    for v in g.data.iter_mut() {
        *v /= n;
    }
    LossAndGrad::dense_only(loss / n, g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_grad_matches_finite_diff() {
        let pred = 0.7;
        let target = 0.2;
        let l = mse(pred, target);
        let eps = 1e-4;
        let lp = mse(pred + eps, target).value;
        let lm = mse(pred - eps, target).value;
        let fd = (lp - lm) / (2.0 * eps);
        assert!((l.g_global.data[0] - fd).abs() < 1e-3);
        assert!(l.g_dense.is_none());
    }

    #[test]
    fn pairwise_loss_is_zero_when_margin_satisfied() {
        let l = pairwise_margin(&PairwiseInput {
            score_better: 1.0,
            score_worse: 0.0,
            margin: 0.5,
        });
        assert_eq!(l.value, 0.0);
    }

    #[test]
    fn pairwise_loss_positive_when_violated() {
        let l = pairwise_margin(&PairwiseInput {
            score_better: 0.0,
            score_worse: 1.0,
            margin: 0.5,
        });
        assert_eq!(l.value, 1.5);
    }

    #[test]
    fn dense_masked_mse_ignores_unmasked() {
        let dense = Tensor::from_rows(vec![vec![1.0, 1.0], vec![9.0, 9.0]]);
        let target = Tensor::from_rows(vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        let mask = Tensor::from_rows(vec![vec![1.0], vec![0.0]]);
        let l = dense_masked_mse(&dense, &target, &mask);
        // only first row supervised: mean = ((1)^2 + (1)^2)/2 = 1
        assert!((l.value - 1.0).abs() < 1e-6);
        let g = l.g_dense.unwrap();
        assert_eq!(g.data[2], 0.0);
        assert_eq!(g.data[3], 0.0);
    }
}
