//! The PerceptKit model: token embedding + positional embedding + N pre-LN
//! encoder blocks + two heads:
//! - **Global head** (mean-pool -> Linear): one scalar per sequence, for
//!   scoring/ranking tasks.
//! - **Dense head** (per-token Linear): one vector per token, for
//!   transformation tasks (colorization, inpainting, layout repair).
//!
//! Either head can be trained; the other is simply unused. This one model
//! covers "score an image/document" and "transform an image" alike.

use crate::layers::{EncoderBlock, Linear};
use crate::rng::Rng;
use crate::tensor::Tensor;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    /// Number of input channels per token (e.g. 3 for RGB patches,
    /// or feature dims for layout elements).
    pub in_channels: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    /// Output channels of the dense head (e.g. 3 for colorization).
    pub out_channels: usize,
    /// Maximum sequence length (positional embedding table size).
    pub max_seq_len: usize,
}

impl ModelConfig {
    pub fn sanity_check(&self) -> Result<(), String> {
        if self.in_channels == 0 {
            return Err("in_channels must be > 0".into());
        }
        if self.d_model == 0 || self.d_model % self.n_heads != 0 {
            return Err("d_model must be > 0 and divisible by n_heads".into());
        }
        if self.n_layers == 0 {
            return Err("n_layers must be > 0".into());
        }
        if self.max_seq_len == 0 {
            return Err("max_seq_len must be > 0".into());
        }
        Ok(())
    }
}

pub struct PerceptModel {
    pub config: ModelConfig,
    pub in_embed: Linear,      // in_channels -> d_model
    pub pos_embed: Tensor,     // (max_seq_len, d_model)
    pub grad_pos_embed: Tensor,
    pub blocks: Vec<EncoderBlock>,
    pub global_head: Linear,   // d_model -> 1
    pub dense_head: Linear,    // d_model -> out_channels
}

/// Everything needed to run backward after a forward pass.
pub struct ForwardCache {
    pub tokens: Tensor,     // input (t, in_channels)
    pub embedded: Tensor,   // (t, d_model) after in_embed + pos
    pub in_proj: Tensor,    // (t, d_model) output of in_embed (pre-pos)
    pub block_caches: Vec<crate::layers::BlockCache>,
    pub encoded: Tensor,    // (t, d_model) final
}

/// Result of a forward pass.
pub struct ForwardOutput {
    pub global_score: Tensor, // (1, 1)
    pub dense: Tensor,        // (t, out_channels)
    pub cache: ForwardCache,
}

impl PerceptModel {
    pub fn new(config: ModelConfig, rng: &mut Rng) -> Result<Self, String> {
        config.sanity_check()?;
        Ok(PerceptModel {
            in_embed: Linear::new(config.in_channels, config.d_model, rng),
            pos_embed: Tensor::normal(config.max_seq_len, config.d_model, 0.02, rng),
            grad_pos_embed: Tensor::zeros(config.max_seq_len, config.d_model),
            blocks: (0..config.n_layers)
                .map(|_| EncoderBlock::new(config.d_model, config.n_heads, config.d_ff, rng))
                .collect(),
            global_head: Linear::new(config.d_model, 1, rng),
            dense_head: Linear::new(config.d_model, config.out_channels, rng),
            config,
        })
    }

    pub fn seq_len(&self, tokens: &Tensor) -> Result<usize, String> {
        if tokens.cols != self.config.in_channels {
            return Err(format!(
                "token width {} != config.in_channels {}",
                tokens.cols, self.config.in_channels
            ));
        }
        if tokens.rows == 0 {
            return Err("empty sequence".into());
        }
        if tokens.rows > self.config.max_seq_len {
            return Err(format!(
                "sequence length {} exceeds max_seq_len {}",
                tokens.rows, self.config.max_seq_len
            ));
        }
        Ok(tokens.rows)
    }

    pub fn forward(&self, tokens: &Tensor) -> Result<ForwardOutput, String> {
        let t = self.seq_len(tokens)?;
        let in_proj = self.in_embed.forward(tokens); // (t, d)
        let pos = Tensor {
            rows: t,
            cols: self.config.d_model,
            data: self.pos_embed.data[..t * self.config.d_model].to_vec(),
        };
        let mut h = in_proj.add(&pos);

        let mut block_caches = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (out, cache) = block.forward(&h);
            block_caches.push(cache);
            h = out;
        }

        // global head: mean pool over tokens
        let mut pooled = Tensor::zeros(1, self.config.d_model);
        for i in 0..t {
            for j in 0..self.config.d_model {
                pooled.data[j] += h.data[i * self.config.d_model + j] / t as f32;
            }
        }
        let global_score = self.global_head.forward(&pooled);

        let dense = self.dense_head.forward(&h);

        Ok(ForwardOutput {
            global_score,
            dense,
            cache: ForwardCache { tokens: tokens.clone(), embedded: h.clone(), in_proj, block_caches, encoded: h },
        })
    }

    /// Backward from a combined gradient signal: `g_global` is grad wrt the
    /// global score (shape (1,1)), `g_dense` is grad wrt the dense output
    /// (shape (t, out_channels)) or `None` to skip the dense path. Accumulates
    /// param grads. Returns grad wrt input tokens (useful for analysis;
    /// usually discarded).
    pub fn backward(
        &mut self,
        g_global: &Tensor,
        g_dense: Option<&Tensor>,
        cache: &ForwardCache,
    ) -> Result<Tensor, String> {
        let t = cache.tokens.rows;
        let d = self.config.d_model;

        // ----- dense head -----
        let mut g_enc = match g_dense {
            Some(gd) => {
                if (gd.rows, gd.cols) != (t, self.config.out_channels) {
                    return Err(format!(
                        "g_dense shape ({}, {}) != ({t}, {})",
                        gd.rows, gd.cols, self.config.out_channels
                    ));
                }
                self.dense_head.backward(&cache.encoded, gd) // (t, d)
            }
            None => Tensor::zeros(t, d),
        };

        // ----- global head (mean pool) -----
        if g_global.data.iter().any(|x| x.abs() > 0.0) {
            let g_pooled = self.global_head.backward(
                &Tensor {
                    rows: 1,
                    cols: d,
                    data: {
                        let mut p = vec![0.0f32; d];
                        for i in 0..t {
                            for j in 0..d {
                                p[j] += cache.encoded.data[i * d + j] / t as f32;
                            }
                        }
                        p
                    },
                },
                g_global,
            );
            // scatter mean-pool grad back to every token
            for i in 0..t {
                for j in 0..d {
                    g_enc.data[i * d + j] += g_pooled.data[j] / t as f32;
                }
            }
        }

        // ----- blocks (reverse order) -----
        for (block, bc) in self.blocks.iter_mut().zip(cache.block_caches.iter()).rev() {
            g_enc = block.backward(&g_enc, bc);
        }

        // ----- embedding: embedded = in_proj + pos -----
        // grad wrt pos (first t rows)
        for i in 0..t {
            for j in 0..d {
                self.grad_pos_embed.data[i * d + j] += g_enc.data[i * d + j];
            }
        }
        // grad wrt in_proj = grad wrt embedded
        let g_in = self.in_embed.backward(&cache.tokens, &g_enc);
        Ok(g_in)
    }

    /// Zero all gradient buffers.
    pub fn zero_grads(&mut self) {
        for g in self.grads_mut() {
            for v in g.data.iter_mut() {
                *v = 0.0;
            }
        }
    }

    /// Iterate over (param, grad) pairs for the optimizer.
    pub fn params(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        let mut p: Vec<(&mut Tensor, &Tensor)> = Vec::new();
        p.extend(self.in_embed.params());
        p.push((&mut self.pos_embed, &self.grad_pos_embed));
        for b in &mut self.blocks {
            p.extend(b.params());
        }
        p.extend(self.global_head.params());
        p.extend(self.dense_head.params());
        p
    }

    /// Read-only view of all parameters, in the same order as `params()`.
    /// Used for serialization and parameter counting.
    pub fn param_refs(&self) -> Vec<&Tensor> {
        let mut p: Vec<&Tensor> = Vec::new();
        p.push(&self.in_embed.weight);
        p.push(&self.in_embed.bias);
        p.push(&self.pos_embed);
        for b in &self.blocks {
            p.push(&b.attn.qkv.weight);
            p.push(&b.attn.qkv.bias);
            p.push(&b.attn.proj.weight);
            p.push(&b.attn.proj.bias);
            p.push(&b.mlp.fc1.weight);
            p.push(&b.mlp.fc1.bias);
            p.push(&b.mlp.fc2.weight);
            p.push(&b.mlp.fc2.bias);
        }
        p.push(&self.global_head.weight);
        p.push(&self.global_head.bias);
        p.push(&self.dense_head.weight);
        p.push(&self.dense_head.bias);
        p
    }

    /// Scale all gradient buffers by `s` (e.g. 1/batch_size to average).
    pub fn scale_grads(&mut self, s: f32) {
        for g in self.grads_mut() {
            for v in g.data.iter_mut() {
                *v *= s;
            }
        }
    }

    /// Mutable access to all gradient buffers, same order as `params()`.
    pub fn grads_mut(&mut self) -> Vec<&mut Tensor> {
        let mut g: Vec<&mut Tensor> = Vec::new();
        g.extend(self.in_embed.grads_mut());
        g.push(&mut self.grad_pos_embed);
        for b in &mut self.blocks {
            g.extend(b.grads_mut());
        }
        g.extend(self.global_head.grads_mut());
        g.extend(self.dense_head.grads_mut());
        g
    }

    pub fn n_params(&self) -> usize {
        self.param_refs().iter().map(|p| p.len()).sum()
    }
}
