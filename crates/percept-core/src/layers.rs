//! Transformer building blocks. Each layer struct owns its parameters and
//! gradient buffers, and exposes `forward` plus `backward` (which accumulates
//! parameter grads and returns the input gradient). All backward passes are
//! gradient-checked in tests.
//!
//! The pre-LN block is:
//!   h = x + Attn(LN(x))
//!   y = h + MLP(LN(h))
//! so backward is, for grad G wrt y:
//!   G_h   = G + MLP.backward(...)          (residual + mlp branch)
//!   G_x   = G_h + Attn.backward(...)      (residual + attn branch)
//! with the LN backward applied on each branch input.

use crate::rng::Rng;
use crate::tensor::Tensor;

/// Linear layer: y = x W + b.
pub struct Linear {
    pub weight: Tensor, // (in, out)
    pub bias: Tensor,   // (1, out)
    pub grad_weight: Tensor,
    pub grad_bias: Tensor,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize, rng: &mut Rng) -> Self {
        Linear {
            weight: Tensor::xavier(in_dim, out_dim, rng),
            bias: Tensor::zeros(1, out_dim),
            grad_weight: Tensor::zeros(in_dim, out_dim),
            grad_bias: Tensor::zeros(1, out_dim),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.matmul(&self.weight).add_row_broadcast(&self.bias)
    }

    /// Accumulate grads; return dL/dx.
    pub fn backward(&mut self, x: &Tensor, g: &Tensor) -> Tensor {
        let mut dx = Tensor::zeros(x.rows, x.cols);
        x.matmul_backward(&self.weight, g, &mut dx, &mut self.grad_weight);
        for j in 0..self.bias.cols {
            let mut acc = 0.0f32;
            for i in 0..g.rows {
                acc += g.data[i * g.cols + j];
            }
            self.grad_bias.data[j] += acc;
        }
        dx
    }

    pub fn params(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        vec![
            (&mut self.weight, &self.grad_weight),
            (&mut self.bias, &self.grad_bias),
        ]
    }

    /// Mutable access to gradient buffers only.
    pub fn grads_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.grad_weight, &mut self.grad_bias]
    }
}

/// Multi-head self-attention with fused QKV projection.
pub struct MultiHeadSelfAttention {
    pub n_heads: usize,
    pub d_model: usize,
    pub qkv: Linear, // d_model -> 3*d_model
    pub proj: Linear, // d_model -> d_model
}

pub struct AttnCache {
    pub qkv_out: Tensor,            // (t, 3d)
    pub attn_weights: Vec<Tensor>,   // per head: (t, t) softmax probs
    pub head_ctx: Tensor,           // concatenated head outputs (t, d_model)
}

impl MultiHeadSelfAttention {
    pub fn new(d_model: usize, n_heads: usize, rng: &mut Rng) -> Self {
        assert!(d_model % n_heads == 0, "d_model must be divisible by n_heads");
        MultiHeadSelfAttention {
            n_heads,
            d_model,
            qkv: Linear::new(d_model, 3 * d_model, rng),
            proj: Linear::new(d_model, d_model, rng),
        }
    }

    pub fn forward(&self, x: &Tensor) -> (Tensor, AttnCache) {
        let t = x.rows;
        let d = self.d_model;
        let hd = d / self.n_heads;
        let qkv = self.qkv.forward(x); // (t, 3d)
        let mut head_out = Tensor::zeros(t, d);
        let mut weights = Vec::with_capacity(self.n_heads);

        for h in 0..self.n_heads {
            let (q, k, v) = split_head(&qkv, h, self.n_heads, d, hd);
            let scores = q.matmul(&k.t()).scale(1.0 / (hd as f32).sqrt());
            let probs = scores.softmax();
            weights.push(probs.clone());
            let ctx = probs.matmul(&v); // (t, hd)
            for i in 0..t {
                for j in 0..hd {
                    head_out.data[i * d + h * hd + j] = ctx.data[i * hd + j];
                }
            }
        }

        let out = self.proj.forward(&head_out);
        (out.clone(), AttnCache { qkv_out: qkv, attn_weights: weights, head_ctx: head_out })
    }

    pub fn backward(
        &mut self,
        x: &Tensor,
        cache: &AttnCache,
        g: &Tensor,
    ) -> Tensor {
        let t = x.rows;
        let d = self.d_model;
        let hd = d / self.n_heads;

        // back through output projection
        let g_head = self.proj.backward(&cache.head_ctx, g); // (t, d)

        let mut g_qkv = Tensor::zeros(t, 3 * d);
        for h in 0..self.n_heads {
            let probs = &cache.attn_weights[h];
            let (_, _, v) = split_head(&cache.qkv_out, h, self.n_heads, d, hd);

            // g_ctx (t, hd)
            let mut g_ctx = Tensor::zeros(t, hd);
            for i in 0..t {
                for j in 0..hd {
                    g_ctx.data[i * hd + j] = g_head.data[i * d + h * hd + j];
                }
            }
            // dP = g_ctx V^T ; dV = P^T g_ctx
            let g_probs = g_ctx.matmul(&v.t()); // (t, t)
            let g_scores = Tensor::softmax_backward(probs, probs, &g_probs);
            let g_scores = g_scores.scale(1.0 / (hd as f32).sqrt());

            let (q, k, _) = split_head(&cache.qkv_out, h, self.n_heads, d, hd);
            let g_q = g_scores.matmul(&k);      // (t, hd)
            let g_k = g_scores.t().matmul(&q);  // (t, hd)
            // dV = P^T g_ctx  (NOT g_ctx directly)
            let g_v = probs.t().matmul(&g_ctx); // (t, hd)
            for i in 0..t {
                for j in 0..hd {
                    g_qkv.data[i * 3 * d + h * hd + j] += g_q.data[i * hd + j];
                    g_qkv.data[i * 3 * d + d + h * hd + j] += g_k.data[i * hd + j];
                    g_qkv.data[i * 3 * d + 2 * d + h * hd + j] += g_v.data[i * hd + j];
                }
            }
        }
        self.qkv.backward(x, &g_qkv)
    }

    pub fn params(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        let mut p = self.qkv.params();
        p.extend(self.proj.params());
        p
    }

    pub fn grads_mut(&mut self) -> Vec<&mut Tensor> {
        let mut g = self.qkv.grads_mut();
        g.extend(self.proj.grads_mut());
        g
    }
}

#[inline]
fn split_head(qkv: &Tensor, h: usize, _n_heads: usize, d: usize, hd: usize) -> (Tensor, Tensor, Tensor) {
    let t = qkv.rows;
    let mut q = Tensor::zeros(t, hd);
    let mut k = Tensor::zeros(t, hd);
    let mut v = Tensor::zeros(t, hd);
    for i in 0..t {
        for j in 0..hd {
            q.data[i * hd + j] = qkv.data[i * 3 * d + h * hd + j];
            k.data[i * hd + j] = qkv.data[i * 3 * d + d + h * hd + j];
            v.data[i * hd + j] = qkv.data[i * 3 * d + 2 * d + h * hd + j];
        }
    }
    (q, k, v)
}

/// FFN: Linear(d, d_ff) -> GELU -> Linear(d_ff, d)
pub struct Mlp {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl Mlp {
    pub fn new(d_model: usize, d_ff: usize, rng: &mut Rng) -> Self {
        Mlp {
            fc1: Linear::new(d_model, d_ff, rng),
            fc2: Linear::new(d_ff, d_model, rng),
        }
    }

    /// Returns (output, pre_gelu) for backward.
    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
        let pre = self.fc1.forward(x);
        (self.fc2.forward(&pre.gelu()), pre)
    }

    pub fn backward(&mut self, x: &Tensor, pre: &Tensor, g: &Tensor) -> Tensor {
        let g_act = pre.gelu();
        let g_pre_out = self.fc2.backward(&g_act, g);
        let g_pre = pre.gelu_backward(&g_pre_out);
        self.fc1.backward(x, &g_pre)
    }

    pub fn params(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        let mut p = self.fc1.params();
        p.extend(self.fc2.params());
        p
    }

    pub fn grads_mut(&mut self) -> Vec<&mut Tensor> {
        let mut g = self.fc1.grads_mut();
        g.extend(self.fc2.grads_mut());
        g
    }
}

/// Pre-LN transformer encoder block:
///   h = x + Attn(LN(x))
///   y = h + MLP(LN(h))
pub struct EncoderBlock {
    pub attn: MultiHeadSelfAttention,
    pub mlp: Mlp,
}

/// Cached forward intermediates for one block.
pub struct BlockCache {
    pub x: Tensor,
    pub n1: Tensor,
    pub m1: Tensor,
    pub r1: Tensor,
    pub attn_out: Tensor,
    pub attn_cache: AttnCache,
    pub h: Tensor,
    pub n2: Tensor,
    pub m2: Tensor,
    pub r2: Tensor,
    pub ff_pre: Tensor,
}

impl EncoderBlock {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize, rng: &mut Rng) -> Self {
        EncoderBlock {
            attn: MultiHeadSelfAttention::new(d_model, n_heads, rng),
            mlp: Mlp::new(d_model, d_ff, rng),
        }
    }

    pub fn forward(&self, x: &Tensor) -> (Tensor, BlockCache) {
        // attention sublayer: h = x + Attn(LN(x))
        let (n1, m1, r1) = x.layer_norm(1e-5);
        let (attn_out, attn_cache) = self.attn.forward(&n1);
        let h = x.add(&attn_out);

        // mlp sublayer: y = h + MLP(LN(h))
        let (n2, m2, r2) = h.layer_norm(1e-5);
        let (ff, ff_pre) = self.mlp.forward(&n2);
        let out = h.add(&ff);

        (
            out,
            BlockCache {
                x: x.clone(),
                n1, m1, r1,
                attn_out,
                attn_cache,
                h,
                n2, m2, r2,
                ff_pre,
            },
        )
    }

    /// Given grad G wrt block output, accumulate param grads and return
    /// grad wrt block input.
    pub fn backward(&mut self, g: &Tensor, cache: &BlockCache) -> Tensor {
        // ---- MLP sublayer: out = h + MLP(LN(h)) ----
        // g_mlp_in = grad wrt MLP input (= LN output)
        let g_mlp_in = self.mlp.backward(&cache.n2, &cache.ff_pre, g);
        // dL/dh = G (residual) + LN_backward(g_mlp_in)
        let g_h = g.add(&cache.h.layer_norm_backward(&g_mlp_in, &cache.m2, &cache.r2));

        // ---- Attention sublayer: h = x + Attn(LN(x)) ----
        // g_attn_in = grad wrt Attn input (= LN output)
        let g_attn_in = self.attn.backward(&cache.n1, &cache.attn_cache, &g_h);
        // dL/dx = g_h (residual) + LN_backward(g_attn_in)
        g_h.add(&cache.x.layer_norm_backward(&g_attn_in, &cache.m1, &cache.r1))
    }

    pub fn params(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        let mut p = self.attn.params();
        p.extend(self.mlp.params());
        p
    }

    pub fn grads_mut(&mut self) -> Vec<&mut Tensor> {
        let mut g = self.attn.grads_mut();
        g.extend(self.mlp.grads_mut());
        g
    }
}
