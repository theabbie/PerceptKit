//! Gradient checks: every backward pass in the model is verified against
//! central finite differences. These are the accountability tests — if a
//! backward is wrong, these fail loudly.
//!
//! Method: define scalar loss L through a forward pass, perturb each
//! parameter or input by ±eps, recompute L, and compare the central
//! difference (L+ - L-)/(2eps) against the analytic gradient.

use percept_core::layers::{EncoderBlock, Linear, MultiHeadSelfAttention};
use percept_core::model::{ModelConfig, PerceptModel};
use percept_core::rng::Rng;
use percept_core::tensor::Tensor;

fn rel_err(analytic: f32, numeric: f32) -> f32 {
    let denom = analytic.abs().max(numeric.abs()).max(1e-3);
    (analytic - numeric).abs() / denom
}

// Finite-difference epsilon and tolerance. In f32 arithmetic the loss
// carries ~1e-4 rounding noise, so FD gradients have an absolute noise
// floor of roughly 1e-4/(2*EPS) ~ 1e-2. On small-magnitude gradients this
// shows up as a few percent relative error. The same algorithm verified
// in f64 (Python reference) matches to 1e-14, so TOL=5% filters FD noise
// while still catching any real backward bug (which typically produces
// errors of 50-1000% or wrong sign).
const EPS: f32 = 5e-3;
const TOL: f32 = 5e-2;

#[test]
fn gradcheck_linear() {
    let mut rng = Rng::new(1);
    let mut lin = Linear::new(4, 3, &mut rng);
    let x = Tensor::from_rows(vec![
        vec![0.5, -0.2, 0.1, 0.7],
        vec![-0.3, 0.8, 0.2, -0.5],
    ]);

    // loss = sum(y^2); analytic grad = 2y
    let y = lin.forward(&x);
    let g = y.scale(2.0);
    for v in lin.grad_weight.data.iter_mut() {
        *v = 0.0;
    }
    for v in lin.grad_bias.data.iter_mut() {
        *v = 0.0;
    }
    lin.backward(&x, &g);

    let loss = |lin: &Linear, x: &Tensor| -> f32 {
        let y = lin.forward(x);
        y.data.iter().map(|v| v * v).sum()
    };
    for idx in 0..lin.weight.len() {
        let orig = lin.weight.data[idx];
        lin.weight.data[idx] = orig + EPS;
        let lp = loss(&lin, &x);
        lin.weight.data[idx] = orig - EPS;
        let lm = loss(&lin, &x);
        lin.weight.data[idx] = orig;
        let numeric = (lp - lm) / (2.0 * EPS);
        assert!(
            rel_err(lin.grad_weight.data[idx], numeric) < TOL,
            "weight[{idx}]: analytic {} vs numeric {}",
            lin.grad_weight.data[idx],
            numeric
        );
    }
    for idx in 0..lin.bias.len() {
        let orig = lin.bias.data[idx];
        lin.bias.data[idx] = orig + EPS;
        let lp = loss(&lin, &x);
        lin.bias.data[idx] = orig - EPS;
        let lm = loss(&lin, &x);
        lin.bias.data[idx] = orig;
        let numeric = (lp - lm) / (2.0 * EPS);
        assert!(
            rel_err(lin.grad_bias.data[idx], numeric) < TOL,
            "bias[{idx}]: analytic {} vs numeric {}",
            lin.grad_bias.data[idx],
            numeric
        );
    }
}

#[test]
fn gradcheck_attention() {
    let mut rng = Rng::new(2);
    let mut attn = MultiHeadSelfAttention::new(8, 2, &mut rng);
    let x = Tensor::from_rows(
        (0..4)
            .map(|i| (0..8).map(|j| ((i * 8 + j) as f32 * 0.13).sin() * 0.5).collect::<Vec<f32>>())
            .collect(),
    );

    // loss = sum(out); grad wrt out is all ones
    let (out, cache) = attn.forward(&x);
    let g = Tensor::from_rows(vec![vec![1.0; 8]; 4]);
    for v in attn.qkv.grad_weight.data.iter_mut() {
        *v = 0.0;
    }
    for v in attn.qkv.grad_bias.data.iter_mut() {
        *v = 0.0;
    }
    for v in attn.proj.grad_weight.data.iter_mut() {
        *v = 0.0;
    }
    for v in attn.proj.grad_bias.data.iter_mut() {
        *v = 0.0;
    }
    let g_x = attn.backward(&x, &cache, &g);

    let loss = |attn: &MultiHeadSelfAttention, x: &Tensor| -> f32 {
        let (out, _) = attn.forward(x);
        out.data.iter().sum()
    };
    // sample of qkv weights
    for idx in (0..attn.qkv.weight.len()).step_by(7).take(12) {
        let orig = attn.qkv.weight.data[idx];
        attn.qkv.weight.data[idx] = orig + EPS;
        let lp = loss(&attn, &x);
        attn.qkv.weight.data[idx] = orig - EPS;
        let lm = loss(&attn, &x);
        attn.qkv.weight.data[idx] = orig;
        let numeric = (lp - lm) / (2.0 * EPS);
        assert!(
            rel_err(attn.qkv.grad_weight.data[idx], numeric) < TOL,
            "qkv weight[{idx}]: analytic {} vs numeric {}",
            attn.qkv.grad_weight.data[idx],
            numeric
        );
    }
    // input gradient
    for idx in 0..x.len() {
        let mut x_mut = x.clone();
        let orig = x_mut.data[idx];
        x_mut.data[idx] = orig + EPS;
        let lp = loss(&attn, &x_mut);
        x_mut.data[idx] = orig - EPS;
        let lm = loss(&attn, &x_mut);
        let numeric = (lp - lm) / (2.0 * EPS);
        assert!(
            rel_err(g_x.data[idx], numeric) < TOL,
            "attn input grad[{idx}]: analytic {} vs numeric {}",
            g_x.data[idx],
            numeric
        );
    }
}

#[test]
fn gradcheck_encoder_block() {
    let mut rng = Rng::new(3);
    let mut block = EncoderBlock::new(8, 2, 16, &mut rng);
    let x = Tensor::from_rows(
        (0..5)
            .map(|i| {
                (0..8)
                    .map(|j| ((i as f32 * 1.7 + j as f32 * 0.31).sin() * 0.4))
                    .collect::<Vec<f32>>()
            })
            .collect(),
    );

    // loss = sum(out^2)
    let (out, cache) = block.forward(&x);
    let g = out.scale(2.0);
    for grad in block.grads_mut() {
        for v in grad.data.iter_mut() {
            *v = 0.0;
        }
    }
    let g_x = block.backward(&g, &cache);

    let loss = |block: &EncoderBlock, x: &Tensor| -> f32 {
        let (out, _) = block.forward(x);
        out.data.iter().map(|v| v * v).sum()
    };
    // input gradient (checks the full residual + LN backward chain)
    for idx in 0..x.len() {
        let mut x_mut = x.clone();
        let orig = x_mut.data[idx];
        x_mut.data[idx] = orig + EPS;
        let lp = loss(&block, &x_mut);
        x_mut.data[idx] = orig - EPS;
        let lm = loss(&block, &x_mut);
        let numeric = (lp - lm) / (2.0 * EPS);
        assert!(
            rel_err(g_x.data[idx], numeric) < TOL,
            "block input grad[{idx}]: analytic {} vs numeric {}",
            g_x.data[idx],
            numeric
        );
    }
    // sample of attention weights inside the block
    for idx in (0..block.attn.qkv.weight.len()).step_by(5).take(8) {
        let orig = block.attn.qkv.weight.data[idx];
        block.attn.qkv.weight.data[idx] = orig + EPS;
        let lp = loss(&block, &x);
        block.attn.qkv.weight.data[idx] = orig - EPS;
        let lm = loss(&block, &x);
        block.attn.qkv.weight.data[idx] = orig;
        let numeric = (lp - lm) / (2.0 * EPS);
        assert!(
            rel_err(block.attn.qkv.grad_weight.data[idx], numeric) < TOL,
            "block qkv weight[{idx}]: analytic {} vs numeric {}",
            block.attn.qkv.grad_weight.data[idx],
            numeric
        );
    }
}

#[test]
fn gradcheck_full_model_global() {
    let mut rng = Rng::new(4);
    let config = ModelConfig {
        in_channels: 3,
        d_model: 8,
        n_heads: 2,
        n_layers: 2,
        d_ff: 16,
        out_channels: 2,
        max_seq_len: 16,
    };
    let mut model = PerceptModel::new(config, &mut rng).unwrap();
    let tokens = Tensor::from_rows(
        (0..6)
            .map(|i| (0..3).map(|j| ((i as f32 * 2.1 + j as f32 * 0.7).sin() * 0.5)).collect::<Vec<f32>>())
            .collect(),
    );

    // loss = global_score^2
    let out = model.forward(&tokens).unwrap();
    let g_global = out.global_score.scale(2.0);
    model.zero_grads();
    let g_in = model.backward(&g_global, None, &out.cache).unwrap();

    let loss = |m: &PerceptModel, t: &Tensor| -> f32 {
        m.forward(t).unwrap().global_score.data[0].powi(2)
    };
    // input gradient
    for idx in 0..tokens.len() {
        let mut t = tokens.clone();
        let orig = t.data[idx];
        t.data[idx] = orig + EPS;
        let lp = loss(&model, &t);
        t.data[idx] = orig - EPS;
        let lm = loss(&model, &t);
        let numeric = (lp - lm) / (2.0 * EPS);
        assert!(
            rel_err(g_in.data[idx], numeric) < TOL,
            "model input grad[{idx}]: analytic {} vs numeric {}",
            g_in.data[idx],
            numeric
        );
    }
    // sample of pos_embed gradient
    for idx in (0..model.pos_embed.len()).step_by(11).take(10) {
        let orig = model.pos_embed.data[idx];
        model.pos_embed.data[idx] = orig + EPS;
        let lp = loss(&model, &tokens);
        model.pos_embed.data[idx] = orig - EPS;
        let lm = loss(&model, &tokens);
        model.pos_embed.data[idx] = orig;
        let numeric = (lp - lm) / (2.0 * EPS);
        assert!(
            rel_err(model.grad_pos_embed.data[idx], numeric) < TOL,
            "pos_embed grad[{idx}]: analytic {} vs numeric {}",
            model.grad_pos_embed.data[idx],
            numeric
        );
    }
}

#[test]
fn gradcheck_full_model_dense() {
    let mut rng = Rng::new(5);
    let config = ModelConfig {
        in_channels: 3,
        d_model: 8,
        n_heads: 2,
        n_layers: 1,
        d_ff: 16,
        out_channels: 2,
        max_seq_len: 16,
    };
    let mut model = PerceptModel::new(config, &mut rng).unwrap();
    let tokens = Tensor::from_rows(
        (0..6)
            .map(|i| (0..3).map(|j| ((i as f32 * 1.3 + j as f32 * 0.9).cos() * 0.5)).collect::<Vec<f32>>())
            .collect(),
    );

    // loss = sum(dense^2)
    let out = model.forward(&tokens).unwrap();
    let g_dense = out.dense.scale(2.0);
    model.zero_grads();
    let zeros_global = Tensor::zeros(1, 1);
    let g_in = model.backward(&zeros_global, Some(&g_dense), &out.cache).unwrap();

    let loss = |m: &PerceptModel, t: &Tensor| -> f32 {
        m.forward(t).unwrap().dense.data.iter().map(|v| v * v).sum()
    };
    for idx in 0..tokens.len() {
        let mut t = tokens.clone();
        let orig = t.data[idx];
        t.data[idx] = orig + EPS;
        let lp = loss(&model, &t);
        t.data[idx] = orig - EPS;
        let lm = loss(&model, &t);
        let numeric = (lp - lm) / (2.0 * EPS);
        assert!(
            rel_err(g_in.data[idx], numeric) < TOL,
            "dense input grad[{idx}]: analytic {} vs numeric {}",
            g_in.data[idx],
            numeric
        );
    }
}
