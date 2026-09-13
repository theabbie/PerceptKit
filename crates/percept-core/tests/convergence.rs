//! Convergence tests: the model must actually LEARN synthetic tasks that
//! exercise the exact failure modes PerceptKit targets:
//!
//! 1. **Scoring**: score a token sequence by how many "corrupted" tokens it
//!    contains (the document-scoring pattern).
//! 2. **Ranking**: order documents by corruption severity (the corruption-
//!    generator pattern).
//! 3. **Dense transform**: invert a corruption applied to tokens (the
//!    colorization/inpainting pattern).

use percept_core::loss::LossKind;
use percept_core::model::{ModelConfig, PerceptModel};
use percept_core::rng::Rng;
use percept_core::tensor::Tensor;
use percept_core::train::{DenseSample, PairSample, ScoreSample, Trainer, TrainerConfig};

fn small_config(in_ch: usize, out_ch: usize) -> ModelConfig {
    ModelConfig {
        in_channels: in_ch,
        d_model: 16,
        n_heads: 2,
        n_layers: 2,
        d_ff: 32,
        out_channels: out_ch,
        max_seq_len: 32,
    }
}

/// Task 1: score = fraction of corrupted tokens.
/// Token = [value, is_corrupted_flag]; corrupted tokens have value pushed
/// toward extremes. Model must regress the corruption fraction.
#[test]
fn learns_to_score_corruption_fraction() {
    let mut rng = Rng::new(10);
    let mut model = PerceptModel::new(small_config(2, 1), &mut rng).unwrap();

    let mut samples = Vec::new();
    for _ in 0..200 {
        let t = 16;
        let n_corrupt = rng.next_usize(t + 1);
        let mut tokens = Vec::with_capacity(t);
        for i in 0..t {
            let corrupted = i < n_corrupt;
            let base = rng.next_f32() * 2.0 - 1.0;
            let value = if corrupted { base * 3.0 } else { base };
            tokens.push(vec![value, if corrupted { 1.0 } else { 0.0 }]);
        }
        samples.push(ScoreSample {
            tokens: Tensor::from_rows(tokens),
            target: n_corrupt as f32 / t as f32,
        });
    }

    let cfg = TrainerConfig { lr: 3e-3, batch_size: 16, seed: 1 };
    let mut trainer = Trainer::new(cfg);

    let mut first_loss = None;
    for epoch in 0..60 {
        let loss = trainer
            .train_epoch_scores(&mut model, &mut samples, LossKind::MSE)
            .unwrap();
        if epoch == 0 {
            first_loss = Some(loss);
        }
    }
    let final_loss = trainer
        .train_epoch_scores(&mut model, &mut samples, LossKind::MSE)
        .unwrap();

    assert!(
        final_loss < 0.01,
        "scoring: first {first_loss:?} final {final_loss}"
    );
    assert!(
        final_loss < first_loss.unwrap() / 10.0,
        "scoring: loss didn't drop 10x: {first_loss:?} -> {final_loss}"
    );
}

/// Task 2: pairwise ranking by severity.
/// Sequences with more corrupted tokens must score higher (we train
/// "worse = more corrupt scores higher" — direction is arbitrary but must
/// be consistent).
#[test]
fn learns_to_rank_by_severity() {
    let mut rng = Rng::new(11);
    let mut model = PerceptModel::new(small_config(2, 1), &mut rng).unwrap();

    let mut samples = Vec::new();
    for _ in 0..150 {
        let t = 12;
        let sev_better = rng.next_usize(4); // 0..3 corruptions
        let sev_worse = 4 + rng.next_usize(4); // 4..7 corruptions

        let make = |n_corrupt: usize, rng: &mut Rng| -> Tensor {
            let mut tokens = Vec::with_capacity(t);
            for i in 0..t {
                let corrupted = i < n_corrupt;
                let base = rng.next_f32() * 2.0 - 1.0;
                tokens.push(vec![if corrupted { base * 3.0 } else { base }, if corrupted { 1.0 } else { 0.0 }]);
            }
            Tensor::from_rows(tokens)
        };

        samples.push(PairSample {
            better_tokens: make(sev_better, &mut rng),
            worse_tokens: make(sev_worse, &mut rng),
            margin: 0.5,
        });
    }

    let cfg = TrainerConfig { lr: 3e-3, batch_size: 16, seed: 2 };
    let mut trainer = Trainer::new(cfg);

    for _ in 0..50 {
        trainer.train_epoch_pairs(&mut model, &mut samples).unwrap();
    }
    let acc = trainer.eval_pair_accuracy(&model, &samples).unwrap();
    assert!(acc > 0.9, "ranking accuracy only {acc}");
}

/// Task 3: dense transform — restore corrupted token values.
/// Input token = [corrupted_value, flag]; target output = [original_value].
/// The model must learn to "undo" the corruption (x/3 when flagged), i.e.
/// the colorization/inpainting pattern in miniature.
#[test]
fn learns_to_undo_corruption() {
    let mut rng = Rng::new(12);
    let mut model = PerceptModel::new(small_config(2, 1), &mut rng).unwrap();

    let mut samples = Vec::new();
    for _ in 0..150 {
        let t = 12;
        let mut tokens = Vec::with_capacity(t);
        let mut targets = Vec::with_capacity(t);
        let mut mask = Vec::with_capacity(t);
        for _ in 0..t {
            let base = rng.next_f32() * 2.0 - 1.0;
            let corrupted = rng.next_f32() < 0.5;
            let value = if corrupted { base * 3.0 } else { base };
            tokens.push(vec![value, if corrupted { 1.0 } else { 0.0 }]);
            targets.push(vec![base]);
            mask.push(vec![if corrupted { 1.0 } else { 0.0 }]); // only learn to fix corrupted ones
        }
        samples.push(DenseSample {
            tokens: Tensor::from_rows(tokens),
            target: Tensor::from_rows(targets),
            mask: Tensor::from_rows(mask),
        });
    }

    let cfg = TrainerConfig { lr: 5e-3, batch_size: 16, seed: 3 };
    let mut trainer = Trainer::new(cfg);

    let mut first_loss = None;
    for epoch in 0..80 {
        let loss = trainer
            .train_epoch_dense(&mut model, &mut samples, LossKind::DenseMaskedMSE)
            .unwrap();
        if epoch == 0 {
            first_loss = Some(loss);
        }
    }
    let final_loss = trainer
        .train_epoch_dense(&mut model, &mut samples, LossKind::DenseMaskedMSE)
        .unwrap();
    assert!(
        final_loss < 0.02,
        "dense transform: {first_loss:?} -> {final_loss}"
    );
}

/// Sanity: model rejects wrong-shaped input loudly instead of UB.
#[test]
fn model_rejects_bad_shapes() {
    let mut rng = Rng::new(13);
    let model = PerceptModel::new(small_config(2, 1), &mut rng).unwrap();
    let bad = Tensor::from_rows(vec![vec![1.0, 2.0, 3.0]]); // 3 channels, expects 2
    assert!(model.forward(&bad).is_err());
    let too_long = Tensor::from_rows((0..40).map(|_| vec![0.0, 0.0]).collect());
    assert!(model.forward(&too_long).is_err());
    let empty = Tensor::zeros(0, 2);
    assert!(model.forward(&empty).is_err());
}
