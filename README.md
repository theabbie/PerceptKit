# PerceptKit

A WebAssembly transformer library for building self-supervised perception
systems in the browser: **scorers, rankers, and dense transformers** for
documents, images, and any token-sequence data.

No training data bottleneck: you generate your own supervision (corrupt
documents and score them, grayscale images and learn colorization back,
mask image regions and learn to fill them) — PerceptKit is the brain that
learns from it, entirely client-side.

## Architecture

```
crates/
  percept-core/   Pure Rust: tensors, transformer, trainer, serialization
  percept-wasm/   wasm-bindgen API surface for JS
pkg/              Generated JS + wasm bindings (nodejs target; regenerable)
tests/e2e.test.js End-to-end test: full train/save/load cycle in Node
```

**One model, three task families** — a pre-LN transformer encoder with two heads:

- **Global head** (mean-pool → linear): scalar score per sequence.
  Use for document/image scoring, quality ranking.
- **Dense head** (per-token linear): vector per token.
  Use for colorization, inpainting, layout repair — any token-to-token transform.
- **Pairwise ranking loss**: train "better-than" preferences without
  needing an absolute score scale.

**Multimodal by unification**: everything is a token sequence. Images are
patchified into tokens (`patchifyImage`); documents/layouts are already
sequences. Same encoder for both.

**Model persistence**: `save()` → bytes → `load()` anywhere. Versioned
binary format with a JSON config header.

## JS API

```js
const { PerceptModelWasm, TrainerWasm, patchifyImage } = require("./pkg/percept_wasm.js");

// inChannels, dModel, nHeads, nLayers, dFf, outChannels, maxSeqLen, seed
const model = new PerceptModelWasm(2, 16, 2, 2, 32, 1, 32, 42);

// lr, batchSize, seed
const trainer = new TrainerWasm(3e-3, 16, 1);

// Score regression: tokensFlat = concatenated samples (row-major),
// seqLens[i] = token count of sample i, targets[i] = target score.
const loss = trainer.trainScoreEpoch(model, tokensFlat, seqLens, targets, "mse");

// Pairwise ranking: better/worse variants of the same item.
const loss = trainer.trainPairEpoch(model, betterFlat, worseFlat, seqLens, margins);
const acc  = trainer.evalPairAccuracy(model, betterFlat, worseFlat, seqLens);

// Dense transform: per-token targets with a supervision mask.
const loss = trainer.trainDenseEpoch(model, tokens, targets, masks, n, seqLen, "dense_mse");

// Inference
const score = model.score(tokensFlat, nTokens);        // scalar
const out   = model.transform(tokensFlat, nTokens);    // per-token vector

// Persistence
const bytes = model.save();                            // Uint8Array
const clone = PerceptModelWasm.load(bytes);

// Images <-> tokens
const toks = patchifyImage(imgFlat, w, h, patchSize, channels);
const img  = unpatchifyImage(toks, w, h, patchSize, channels);
```

## Testing (the accountability layer)

- `crates/percept-core`: 22 unit tests + 5 finite-difference gradient
  checks (every backward pass verified against numeric gradients) +
  4 convergence tests (the model must actually learn scoring, ranking,
  and corruption-undoing tasks — not just run).
- `tests/e2e.test.js`: full lifecycle in Node — create → generate
  synthetic corrupted data → train to convergence → probe predictions →
  save/load roundtrip → dense transform → patchify roundtrip → error handling.

```bash
cargo test -p percept-core          # Rust tests
node tests/e2e.test.js               # wasm end-to-end
```

## Build

```bash
rustup target add wasm32-unknown-unknown
cargo build -p percept-wasm --release --target wasm32-unknown-unknown
wasm-bindgen --target nodejs --out-dir pkg target/wasm32-unknown-unknown/release/percept_wasm.wasm
# for the browser app use --target web or bundler instead of nodejs
```

Binary size: ~210 KB unoptimized (no wasm-opt yet).

## Design notes

- **Autodiff-lite**: every op is an explicit forward/backward pair — no
  graph, no tape. Keeps the wasm binary small and the numerics auditable
  (every backward is gradient-checked).
- **Pre-LN blocks** with parameter-free LayerNorm; affine params live in
  the following linear layers.
- **f32 throughout**; the training loop yields to the JS event loop
  between epochs (call one epoch per tick to keep the UI responsive).
- Seeds are plain JS numbers (f64) to avoid BigInt friction.

## Roadmap

- [ ] wasm-opt / size reduction pass
- [ ] Browser (`--target web`) package build
- [ ] Corruption-generator helpers (overlap/overflow/dead-space for layout docs)
- [ ] Attention-map export for interpretability (which tokens made the score bad)
- [ ] Batched inference
