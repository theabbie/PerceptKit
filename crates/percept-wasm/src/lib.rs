//! WebAssembly bindings for PerceptKit. This is the API surface the browser
//! app (and the planning LLM) programs against.
//!
//! Usage sketch (JS):
//! ```js
//! import init, { PerceptModelWasm, TrainerWasm } from "percept_wasm.js";
//! await init();
//! const model = new PerceptModelWasm(inChannels, dModel, nHeads, nLayers, dFf, outChannels, maxSeqLen, seed);
//! const trainer = new TrainerWasm(lr, batchSize, seed);
//! const loss = trainer.trainScoreEpoch(model, tokensFlat, seqLen, nTokens, targets);
//! const score = model.score(tokensFlat, seqLen);
//! const bytes = model.save(); // Uint8Array — persist anywhere
//! const loaded = PerceptModelWasm.load(bytes);
//! ```

use percept_core::loss::LossKind;
use percept_core::model::{ModelConfig, PerceptModel};
use percept_core::rng::Rng;
use percept_core::serialize;
use percept_core::tensor::Tensor;
use percept_core::train::{DenseSample, PairSample, ScoreSample, Trainer, TrainerConfig};
use wasm_bindgen::prelude::*;

fn loss_kind_from_str(s: &str) -> Result<LossKind, JsError> {
    match s {
        "mse" => Ok(LossKind::MSE),
        "l1" => Ok(LossKind::L1),
        "pairwise" => Ok(LossKind::PairwiseMargin),
        "dense_mse" => Ok(LossKind::DenseMaskedMSE),
        "dense_l1" => Ok(LossKind::DenseMaskedL1),
        _ => Err(JsError::new(&format!("unknown loss kind: {s}"))),
    }
}

#[wasm_bindgen]
pub struct PerceptModelWasm {
    inner: PerceptModel,
}

#[wasm_bindgen]
impl PerceptModelWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(
        in_channels: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        d_ff: usize,
        out_channels: usize,
        max_seq_len: usize,
        seed: f64,
    ) -> Result<PerceptModelWasm, JsError> {
        let config = ModelConfig {
            in_channels,
            d_model,
            n_heads,
            n_layers,
            d_ff,
            out_channels,
            max_seq_len,
        };
        let mut rng = Rng::new(seed.to_bits());
        let inner = PerceptModel::new(config, &mut rng).map_err(|e| JsError::new(&e))?;
        Ok(PerceptModelWasm { inner })
    }

    /// Load a model from bytes produced by `save`.
    #[wasm_bindgen(js_name = "load")]
    pub fn load(bytes: &[u8]) -> Result<PerceptModelWasm, JsError> {
        let inner = serialize::from_bytes(bytes).map_err(|e| JsError::new(&e))?;
        Ok(PerceptModelWasm { inner })
    }

    /// Serialize the model (config + weights) to bytes.
    #[wasm_bindgen(js_name = "save")]
    pub fn save(&self) -> Vec<u8> {
        serialize::to_bytes(&self.inner)
    }

    /// Score a token sequence: flat f32 array, row-major (n_tokens x in_channels).
    /// Returns the global scalar score.
    #[wasm_bindgen(js_name = "score")]
    pub fn score(&self, tokens_flat: &[f32], n_tokens: usize) -> Result<f32, JsError> {
        let tokens = flat_to_tensor(tokens_flat, n_tokens, self.inner.config.in_channels)?;
        let out = self.inner.forward(&tokens).map_err(|e| JsError::new(&e))?;
        Ok(out.global_score.data[0])
    }

    /// Dense (per-token) output: returns flat f32 array (n_tokens x out_channels).
    #[wasm_bindgen(js_name = "transform")]
    pub fn transform(&self, tokens_flat: &[f32], n_tokens: usize) -> Result<Vec<f32>, JsError> {
        let tokens = flat_to_tensor(tokens_flat, n_tokens, self.inner.config.in_channels)?;
        let out = self.inner.forward(&tokens).map_err(|e| JsError::new(&e))?;
        Ok(out.dense.data)
    }

    /// Number of trainable parameters.
    #[wasm_bindgen(js_name = "nParams")]
    pub fn n_params(&self) -> usize {
        self.inner.n_params()
    }

    /// The model's config as a JSON string.
    #[wasm_bindgen(js_name = "configJson")]
    pub fn config_json(&self) -> String {
        serde_json::to_string(&self.inner.config).unwrap_or_default()
    }
}

fn flat_to_tensor(flat: &[f32], rows: usize, cols: usize) -> Result<Tensor, JsError> {
    if flat.len() != rows * cols {
        return Err(JsError::new(&format!(
            "flat len {} != rows {} x cols {}",
            flat.len(),
            rows,
            cols
        )));
    }
    Ok(Tensor { rows, cols, data: flat.to_vec() })
}

#[wasm_bindgen]
pub struct TrainerWasm {
    inner: Trainer,
}

#[wasm_bindgen]
impl TrainerWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(lr: f32, batch_size: usize, seed: f64) -> TrainerWasm {
        // f64 seed keeps the JS API number-friendly (no BigInt friction);
        // to_bits gives a full-entropy u64 from it.
        let config = TrainerConfig { lr, batch_size, seed: seed.to_bits() };
        TrainerWasm { inner: Trainer::new(config) }
    }

    /// One epoch of score regression. `tokens_flat` is all samples
    /// concatenated; `seq_lens[i]` is the token count of sample i;
    /// `targets[i]` is the target score of sample i. Returns mean loss.
    #[wasm_bindgen(js_name = "trainScoreEpoch")]
    pub fn train_score_epoch(
        &mut self,
        model: &mut PerceptModelWasm,
        tokens_flat: &[f32],
        seq_lens: &[usize],
        targets: &[f32],
        loss_kind: &str,
    ) -> Result<f32, JsError> {
        let kind = loss_kind_from_str(loss_kind)?;
        if !matches!(kind, LossKind::MSE | LossKind::L1) {
            return Err(JsError::new("trainScoreEpoch requires 'mse' or 'l1'"));
        }
        if seq_lens.len() != targets.len() {
            return Err(JsError::new("seq_lens and targets must have equal length"));
        }
        let in_ch = model.inner.config.in_channels;
        let mut samples = Vec::with_capacity(seq_lens.len());
        let mut offset = 0usize;
        for (len, target) in seq_lens.iter().zip(targets.iter()) {
            let end = offset + len * in_ch;
            if end > tokens_flat.len() {
                return Err(JsError::new("tokens_flat too short for given seq_lens"));
            }
            samples.push(ScoreSample {
                tokens: Tensor { rows: *len, cols: in_ch, data: tokens_flat[offset..end].to_vec() },
                target: *target,
            });
            offset = end;
        }
        self.inner
            .train_epoch_scores(&mut model.inner, &mut samples, kind)
            .map_err(|e| JsError::new(&e))
    }

    /// One epoch of pairwise ranking. `better_flat`/`worse_flat` are
    /// concatenated token sequences; `seq_lens[i]` gives each pair's length
    /// (pairs must have equal lengths). Returns mean loss.
    #[wasm_bindgen(js_name = "trainPairEpoch")]
    pub fn train_pair_epoch(
        &mut self,
        model: &mut PerceptModelWasm,
        better_flat: &[f32],
        worse_flat: &[f32],
        seq_lens: &[usize],
        margins: &[f32],
    ) -> Result<f32, JsError> {
        if seq_lens.len() != margins.len() {
            return Err(JsError::new("seq_lens and margins must have equal length"));
        }
        let in_ch = model.inner.config.in_channels;
        let mut samples = Vec::with_capacity(seq_lens.len());
        let mut off_b = 0usize;
        let mut off_w = 0usize;
        for (len, margin) in seq_lens.iter().zip(margins.iter()) {
            let eb = off_b + len * in_ch;
            let ew = off_w + len * in_ch;
            if eb > better_flat.len() || ew > worse_flat.len() {
                return Err(JsError::new("flat arrays too short for given seq_lens"));
            }
            samples.push(PairSample {
                better_tokens: Tensor { rows: *len, cols: in_ch, data: better_flat[off_b..eb].to_vec() },
                worse_tokens: Tensor { rows: *len, cols: in_ch, data: worse_flat[off_w..ew].to_vec() },
                margin: *margin,
            });
            off_b = eb;
            off_w = ew;
        }
        self.inner
            .train_epoch_pairs(&mut model.inner, &mut samples)
            .map_err(|e| JsError::new(&e))
    }

    /// One epoch of dense transform training. `tokens_flat` and
    /// `targets_flat` are (n_tokens x in_channels / out_channels); `masks_flat`
    /// is (n_tokens x 1). All samples must have equal seq length.
    #[wasm_bindgen(js_name = "trainDenseEpoch")]
    pub fn train_dense_epoch(
        &mut self,
        model: &mut PerceptModelWasm,
        tokens_flat: &[f32],
        targets_flat: &[f32],
        masks_flat: &[f32],
        n_samples: usize,
        seq_len: usize,
        loss_kind: &str,
    ) -> Result<f32, JsError> {
        let kind = loss_kind_from_str(loss_kind)?;
        if !matches!(kind, LossKind::DenseMaskedMSE | LossKind::DenseMaskedL1) {
            return Err(JsError::new("trainDenseEpoch requires 'dense_mse' or 'dense_l1'"));
        }
        let in_ch = model.inner.config.in_channels;
        let out_ch = model.inner.config.out_channels;
        if tokens_flat.len() != n_samples * seq_len * in_ch {
            return Err(JsError::new("tokens_flat size mismatch"));
        }
        if targets_flat.len() != n_samples * seq_len * out_ch {
            return Err(JsError::new("targets_flat size mismatch"));
        }
        if masks_flat.len() != n_samples * seq_len {
            return Err(JsError::new("masks_flat size mismatch"));
        }
        let mut samples = Vec::with_capacity(n_samples);
        for s in 0..n_samples {
            let t0 = s * seq_len * in_ch;
            let g0 = s * seq_len * out_ch;
            let m0 = s * seq_len;
            samples.push(DenseSample {
                tokens: Tensor { rows: seq_len, cols: in_ch, data: tokens_flat[t0..t0 + seq_len * in_ch].to_vec() },
                target: Tensor { rows: seq_len, cols: out_ch, data: targets_flat[g0..g0 + seq_len * out_ch].to_vec() },
                mask: Tensor { rows: seq_len, cols: 1, data: masks_flat[m0..m0 + seq_len].to_vec() },
            });
        }
        self.inner
            .train_epoch_dense(&mut model.inner, &mut samples, kind)
            .map_err(|e| JsError::new(&e))
    }

    /// Pairwise accuracy on the given pairs (evaluation, no training).
    #[wasm_bindgen(js_name = "evalPairAccuracy")]
    pub fn eval_pair_accuracy(
        &self,
        model: &PerceptModelWasm,
        better_flat: &[f32],
        worse_flat: &[f32],
        seq_lens: &[usize],
    ) -> Result<f32, JsError> {
        let in_ch = model.inner.config.in_channels;
        let mut samples = Vec::with_capacity(seq_lens.len());
        let mut off_b = 0usize;
        let mut off_w = 0usize;
        for len in seq_lens {
            let eb = off_b + len * in_ch;
            let ew = off_w + len * in_ch;
            if eb > better_flat.len() || ew > worse_flat.len() {
                return Err(JsError::new("flat arrays too short"));
            }
            samples.push(PairSample {
                better_tokens: Tensor { rows: *len, cols: in_ch, data: better_flat[off_b..eb].to_vec() },
                worse_tokens: Tensor { rows: *len, cols: in_ch, data: worse_flat[off_w..ew].to_vec() },
                margin: 0.0,
            });
            off_b = eb;
            off_w = ew;
        }
        self.inner
            .eval_pair_accuracy(&model.inner, &samples)
            .map_err(|e| JsError::new(&e))
    }
}

/// Patchify an image into token sequences for the model.
/// `img` is a flat f32 array (w*h*channels, channel-interleaved).
/// Returns flat tokens (n_tokens x channels*patch*patch).
#[wasm_bindgen(js_name = "patchifyImage")]
pub fn patchify_image(
    img: &[f32],
    w: usize,
    h: usize,
    patch: usize,
    channels: usize,
) -> Result<Vec<f32>, JsError> {
    let grid = percept_core::patchify::PatchGrid::new(w, h, patch).map_err(|e| JsError::new(&e))?;
    let tokens = percept_core::patchify::patchify_image(img, &grid, channels)
        .map_err(|e| JsError::new(&e))?;
    Ok(tokens.data)
}

/// Inverse of patchifyImage.
#[wasm_bindgen(js_name = "unpatchifyImage")]
pub fn unpatchify_image(
    tokens_flat: &[f32],
    w: usize,
    h: usize,
    patch: usize,
    channels: usize,
) -> Result<Vec<f32>, JsError> {
    let grid = percept_core::patchify::PatchGrid::new(w, h, patch).map_err(|e| JsError::new(&e))?;
    let n_tokens = grid.n_tokens();
    let token_dim = channels * patch * patch;
    let tokens = Tensor { rows: n_tokens, cols: token_dim, data: tokens_flat.to_vec() };
    percept_core::patchify::unpatchify_image(&tokens, &grid, channels)
        .map_err(|e| JsError::new(&e))
}

/// Library version string.
#[wasm_bindgen(js_name = "version")]
pub fn version() -> String {
    "0.1.0".to_string()
}
