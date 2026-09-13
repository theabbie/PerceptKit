//! Training loop: batches, shuffling, loss selection, gradient zeroing,
//! Adam step, and metric reporting. The caller supplies data; this owns the
//! loop mechanics. Designed to be called step-by-step from wasm too (yield
//! to the JS event loop between steps to keep the UI responsive).

use crate::loss::{self, LossKind, LossAndGrad, PairwiseInput};
use crate::model::PerceptModel;
use crate::optim::Adam;
use crate::rng::Rng;
use crate::tensor::Tensor;

#[derive(Clone, Debug)]
pub struct TrainerConfig {
    pub lr: f32,
    pub batch_size: usize,
    pub seed: u64,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        TrainerConfig { lr: 3e-4, batch_size: 16, seed: 42 }
    }
}

/// One training example for the global (scoring/ranking) path.
pub struct ScoreSample {
    pub tokens: Tensor,  // (t, in_channels)
    pub target: f32,     // score target (MSE/L1)
}

/// One training example for the pairwise ranking path.
pub struct PairSample {
    pub better_tokens: Tensor,
    pub worse_tokens: Tensor,
    pub margin: f32,
}

/// One training example for the dense (transform) path.
pub struct DenseSample {
    pub tokens: Tensor,       // (t, in_channels) model input
    pub target: Tensor,       // (t, out_channels) desired dense output
    pub mask: Tensor,         // (t, 1) supervision mask
}

pub struct Trainer {
    pub config: TrainerConfig,
    pub optimizer: Adam,
    pub rng: Rng,
}

impl Trainer {
    pub fn new(config: TrainerConfig) -> Self {
        let lr = config.lr;
        let seed = config.seed;
        Trainer {
            config,
            optimizer: Adam::new(lr),
            rng: Rng::new(seed),
        }
    }

    /// One epoch over score samples (MSE or L1). Returns mean loss.
    pub fn train_epoch_scores(
        &mut self,
        model: &mut PerceptModel,
        samples: &mut Vec<ScoreSample>,
        kind: LossKind,
    ) -> Result<f32, String> {
        if samples.is_empty() {
            return Err("no samples".into());
        }
        if !matches!(kind, LossKind::MSE | LossKind::L1) {
            return Err("train_epoch_scores requires MSE or L1".into());
        }
        self.rng.shuffle(samples);
        let mut total = 0.0f32;
        let mut n = 0usize;
        for chunk in samples.chunks_mut(self.config.batch_size.max(1)) {
            model.zero_grads();
            let mut batch_loss = 0.0f32;
            for s in chunk.iter_mut() {
                let out = model.forward(&s.tokens)?;
                let l: LossAndGrad = match kind {
                    LossKind::MSE => loss::mse(out.global_score.data[0], s.target),
                    LossKind::L1 => loss::l1(out.global_score.data[0], s.target),
                    _ => unreachable!(),
                };
                batch_loss += l.value;
                model.backward(&l.g_global, l.g_dense.as_ref(), &out.cache)?;
            }
            // average grads over batch, then step
            model.scale_grads(1.0 / chunk.len() as f32);
            let mut params = model.params();
            self.optimizer.step(&mut params);
            total += batch_loss;
            n += chunk.len();
        }
        Ok(total / n.max(1) as f32)
    }

    /// One epoch over pairwise ranking samples. Returns mean loss.
    pub fn train_epoch_pairs(
        &mut self,
        model: &mut PerceptModel,
        samples: &mut Vec<PairSample>,
    ) -> Result<f32, String> {
        if samples.is_empty() {
            return Err("no samples".into());
        }
        self.rng.shuffle(samples);
        let mut total = 0.0f32;
        let mut n = 0usize;
        for chunk in samples.chunks_mut(self.config.batch_size.max(1)) {
            model.zero_grads();
            let mut batch_loss = 0.0f32;
            for s in chunk.iter_mut() {
                let out_b = model.forward(&s.better_tokens)?;
                let out_w = model.forward(&s.worse_tokens)?;
                let input = PairwiseInput {
                    score_better: out_b.global_score.data[0],
                    score_worse: out_w.global_score.data[0],
                    margin: s.margin,
                };
                let l = loss::pairwise_margin(&input);
                batch_loss += l.value;
                // grad wrt better score is -1, wrt worse score is +1 (when violated)
                if l.value > 0.0 {
                    let g_better = Tensor::from_rows(vec![vec![-1.0]]);
                    let g_worse = Tensor::from_rows(vec![vec![1.0]]);
                    model.backward(&g_better, None, &out_b.cache)?;
                    model.backward(&g_worse, None, &out_w.cache)?;
                }
            }
            let b = chunk.len() as f32;
            model.scale_grads(1.0 / b);
            let mut params = model.params();
            self.optimizer.step(&mut params);
            total += batch_loss;
            n += chunk.len();
        }
        Ok(total / n.max(1) as f32)
    }

    /// One epoch over dense samples. Returns mean loss.
    pub fn train_epoch_dense(
        &mut self,
        model: &mut PerceptModel,
        samples: &mut Vec<DenseSample>,
        kind: LossKind,
    ) -> Result<f32, String> {
        if samples.is_empty() {
            return Err("no samples".into());
        }
        if !matches!(kind, LossKind::DenseMaskedMSE | LossKind::DenseMaskedL1) {
            return Err("train_epoch_dense requires DenseMaskedMSE or DenseMaskedL1".into());
        }
        self.rng.shuffle(samples);
        let mut total = 0.0f32;
        let mut n = 0usize;
        for chunk in samples.chunks_mut(self.config.batch_size.max(1)) {
            model.zero_grads();
            let mut batch_loss = 0.0f32;
            for s in chunk.iter_mut() {
                let out = model.forward(&s.tokens)?;
                let l: LossAndGrad = match kind {
                    LossKind::DenseMaskedMSE => loss::dense_masked_mse(&out.dense, &s.target, &s.mask),
                    LossKind::DenseMaskedL1 => loss::dense_masked_l1(&out.dense, &s.target, &s.mask),
                    _ => unreachable!(),
                };
                batch_loss += l.value;
                model.backward(&l.g_global, l.g_dense.as_ref(), &out.cache)?;
            }
            let b = chunk.len() as f32;
            model.scale_grads(1.0 / b);
            let mut params = model.params();
            self.optimizer.step(&mut params);
            total += batch_loss;
            n += chunk.len();
        }
        Ok(total / n.max(1) as f32)
    }

    /// Mean loss over score samples without training (evaluation).
    pub fn eval_scores(
        &self,
        model: &PerceptModel,
        samples: &[ScoreSample],
        kind: LossKind,
    ) -> Result<f32, String> {
        if samples.is_empty() {
            return Err("no samples".into());
        }
        let mut total = 0.0f32;
        for s in samples {
            let out = model.forward(&s.tokens)?;
            let l: LossAndGrad = match kind {
                LossKind::MSE => loss::mse(out.global_score.data[0], s.target),
                LossKind::L1 => loss::l1(out.global_score.data[0], s.target),
                _ => return Err("eval_scores requires MSE or L1".into()),
            };
            total += l.value;
        }
        Ok(total / samples.len() as f32)
    }

    /// Pairwise accuracy: fraction of pairs where better scores above worse.
    pub fn eval_pair_accuracy(
        &self,
        model: &PerceptModel,
        samples: &[PairSample],
    ) -> Result<f32, String> {
        if samples.is_empty() {
            return Err("no samples".into());
        }
        let mut correct = 0usize;
        for s in samples {
            let b = model.forward(&s.better_tokens)?.global_score.data[0];
            let w = model.forward(&s.worse_tokens)?.global_score.data[0];
            if b > w {
                correct += 1;
            }
        }
        Ok(correct as f32 / samples.len() as f32)
    }
}
