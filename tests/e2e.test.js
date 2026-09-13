/**
 * End-to-end test of the PerceptKit wasm package in Node.
 * Exercises the exact workflow the browser app will use:
 *   1. create model
 *   2. generate synthetic corruption-scored data
 *   3. train (score regression)
 *   4. verify learning happened
 *   5. save -> bytes -> load -> identical outputs
 *   6. dense transform path + patchify roundtrip
 */
const assert = require("assert");
const {
  PerceptModelWasm,
  TrainerWasm,
  patchifyImage,
  unpatchifyImage,
  version,
} = require("../pkg/percept_wasm.js");

function test() {
  assert.strictEqual(version(), "0.1.0");

  // ---- 1. create model: 2-channel tokens, tiny transformer ----
  const model = new PerceptModelWasm(2, 16, 2, 2, 32, 1, 32, 42); // seed as plain number
  assert.ok(model.nParams() > 0, "model should have params");
  const cfg = JSON.parse(model.configJson());
  assert.strictEqual(cfg.in_channels, 2);
  assert.strictEqual(cfg.d_model, 16);

  // ---- 2. synthetic data: corruption fraction scoring ----
  // token = [value, corrupted_flag]; corrupted values are amplified 3x.
  // target = fraction of corrupted tokens.
  const N = 200, T = 16, IN = 2;
  let tokensFlat = new Float32Array(N * T * IN);
  let seqLens = new Array(N).fill(T);
  let targets = new Float32Array(N);
  let mulberry = 1234;
  const rand = () => {
    // deterministic LCG for reproducibility
    mulberry = (mulberry * 1664525 + 1013904223) >>> 0;
    return mulberry / 0xffffffff;
  };
  for (let s = 0; s < N; s++) {
    const nCorrupt = Math.floor(rand() * (T + 1));
    targets[s] = nCorrupt / T;
    // corrupted positions are random (not always at the front) so the
    // model must read the flags, not memorize position patterns
    const flags = new Array(T).fill(0);
    const idx = Array.from({ length: T }, (_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    for (let k = 0; k < nCorrupt; k++) flags[idx[k]] = 1;
    for (let i = 0; i < T; i++) {
      const corrupted = flags[i];
      const base = rand() * 2 - 1;
      const v = corrupted ? base * 3 : base;
      const off = (s * T + i) * IN;
      tokensFlat[off] = v;
      tokensFlat[off + 1] = corrupted;
    }
  }

  // ---- 3. train ----
  const trainer = new TrainerWasm(3e-3, 16, 1);
  let firstLoss = null, lastLoss = null;
  for (let epoch = 0; epoch < 60; epoch++) {
    const loss = trainer.trainScoreEpoch(model, tokensFlat, seqLens, targets, "mse");
    if (epoch === 0) firstLoss = loss;
    lastLoss = loss;
  }
  console.log(`score training: first loss ${firstLoss.toFixed(4)} -> final ${lastLoss.toFixed(4)}`);
  assert.ok(lastLoss < 0.01, `final loss too high: ${lastLoss}`);
  assert.ok(lastLoss < firstLoss / 10, "loss should drop 10x");

  // ---- 4. spot-check predictions (in-distribution probes) ----
  // Use the same generative process as training data: clean tokens have
  // values in [-1,1]; corrupted tokens are amplified 3x with flag=1.
  const clean = new Float32Array(T * IN);
  const corrupt = new Float32Array(T * IN);
  for (let i = 0; i < T; i++) {
    const base = rand() * 2 - 1;
    clean[i * IN] = base;
    clean[i * IN + 1] = 0;
    const base2 = rand() * 2 - 1;
    corrupt[i * IN] = base2 * 3;
    corrupt[i * IN + 1] = 1;
  }
  const sClean = model.score(clean, T);
  const sCorrupt = model.score(corrupt, T);
  console.log(`scores: clean ${sClean.toFixed(3)} (target 0), corrupt ${sCorrupt.toFixed(3)} (target 1)`);
  assert.ok(Math.abs(sClean - 0) < 0.15, `clean score should be ~0, got ${sClean}`);
  assert.ok(Math.abs(sCorrupt - 1) < 0.15, `corrupt score should be ~1, got ${sCorrupt}`);

  // ---- 5. save / load roundtrip ----
  const bytes = model.save();
  console.log(`model saved: ${bytes.length} bytes`);
  const loaded = PerceptModelWasm.load(bytes);
  const s1 = model.score(tokensFlat.subarray(0, T * IN), T);
  const s2 = loaded.score(tokensFlat.subarray(0, T * IN), T);
  assert.strictEqual(s1, s2, "loaded model must produce identical scores");

  // ---- 6. dense transform path ----
  const denseModel = new PerceptModelWasm(2, 16, 2, 2, 32, 1, 32, 7);
  const denseTrainer = new TrainerWasm(5e-3, 16, 2);
  const ND = 150;
  let dTokens = new Float32Array(ND * T * IN);
  let dTargets = new Float32Array(ND * T * 1);
  let dMasks = new Float32Array(ND * T);
  for (let s = 0; s < ND; s++) {
    for (let i = 0; i < T; i++) {
      const base = rand() * 2 - 1;
      const corrupted = rand() < 0.5 ? 1 : 0;
      const off = (s * T + i) * IN;
      dTokens[off] = corrupted ? base * 3 : base;
      dTokens[off + 1] = corrupted;
      dTargets[s * T + i] = base; // learn to output the original value
      dMasks[s * T + i] = corrupted; // only supervised on corrupted tokens
    }
  }
  let dFirst = null, dLast = null;
  for (let epoch = 0; epoch < 80; epoch++) {
    const loss = denseTrainer.trainDenseEpoch(
      denseModel, dTokens, dTargets, dMasks, ND, T, "dense_mse");
    if (epoch === 0) dFirst = loss;
    dLast = loss;
  }
  console.log(`dense training: ${dFirst.toFixed(4)} -> ${dLast.toFixed(4)}`);
  assert.ok(dLast < 0.02, `dense final loss too high: ${dLast}`);

  // transform() should now approximately undo corruption
  const out = denseModel.transform(dTokens.subarray(0, T * IN), T);
  assert.strictEqual(out.length, T);
  // Error on supervised (corrupted) tokens is what training optimized;
  // unsupervised tokens were never trained, so only check masked ones.
  let err = 0, nSup = 0;
  for (let i = 0; i < T; i++) {
    if (dMasks[i] === 1) {
      err += Math.abs(out[i] - dTargets[i]);
      nSup++;
    }
  }
  const meanErr = nSup > 0 ? err / nSup : 0;
  console.log(`transform mean abs err on supervised tokens: ${meanErr.toFixed(4)} (${nSup} tokens)`);
  assert.ok(meanErr < 0.15, "transform should approximate targets on supervised tokens");

  // ---- 7. patchify roundtrip ----
  const W = 8, H = 8, P = 4, C = 3;
  const img = new Float32Array(W * H * C);
  for (let i = 0; i < img.length; i++) img[i] = rand();
  const toks = patchifyImage(img, W, H, P, C);
  assert.strictEqual(toks.length, (W / P) * (H / P) * C * P * P);
  const back = unpatchifyImage(toks, W, H, P, C);
  for (let i = 0; i < img.length; i++) {
    assert.ok(Math.abs(back[i] - img[i]) < 1e-6, `patchify roundtrip mismatch at ${i}`);
  }

  // ---- 8. error handling ----
  let threw = false;
  try {
    model.score(new Float32Array(5), T); // wrong size
  } catch (e) {
    threw = true;
  }
  assert.ok(threw, "bad input should throw");
  threw = false;
  try {
    PerceptModelWasm.load(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]));
  } catch (e) {
    threw = true;
  }
  assert.ok(threw, "bad model bytes should throw");

  console.log("\nALL E2E TESTS PASSED");
}

test();
