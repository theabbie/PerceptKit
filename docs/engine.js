// PerceptKit demo engine — pure data generation + training orchestration.
// No DOM, no wasm loading: the wasm module namespace is passed in, so this
// file is testable headlessly in Node and drives the browser UI alike.
//
// Three demos, three self-supervision patterns:
//  1. classify: digit classifier (synthetic font digits + noise/shifts)
//  2. score:    text-overlap scorer (render docs, corrupt, pairwise rank)
//  3. generate: colorizer (synthetic scenes, learn gray -> color inverse)

// ---------------------------------------------------------------------------
// RNG (mulberry32, seedable, deterministic)
// ---------------------------------------------------------------------------

export function rng(seed) {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const yieldToUI = () => new Promise((r) => setTimeout(r, 0));

// ---------------------------------------------------------------------------
// Demo 1: digit classifier
// ---------------------------------------------------------------------------

// 5x7 font for digits 0-9. "#" = on.
export const FONT_5X7 = [
  [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
  ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
  [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
  [".###.", "#...#", "....#", "..##.", "....#", "#...#", ".###."],
  ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
  ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
  [".###.", "#....", "####.", "#...#", "#...#", "#...#", ".###."],
  ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
  [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
  [".###.", "#...#", "#...#", ".####", "....#", "....#", ".###."],
];

export const DIGIT_W = 8;
export const DIGIT_H = 10;
export const DIGIT_PATCH = 2; // -> 4x5 grid = 20 tokens x 4 channels

// Render digit `d` with random shift, per-row jitter, and pixel noise
// into an 8x10 image. Wide augmentation is free (synthetic data) and is
// what makes the classifier generalize — small shifts underfit badly.
export function genDigit(d, r) {
  const img = new Float32Array(DIGIT_W * DIGIT_H);
  const dx = Math.floor(r() * 3) - 1; // -1..1
  const dy = Math.floor(r() * 3) - 1;
  const font = FONT_5X7[d];
  for (let fy = 0; fy < 7; fy++) {
    for (let fx = 0; fx < 5; fx++) {
      if (font[fy][fx] !== "#") continue;
      const x = fx + 1 + dx; // font centered in 8-wide image
      const y = fy + 1 + dy; // and 10-tall
      if (x < 0 || x >= DIGIT_W || y < 0 || y >= DIGIT_H) continue;
      img[y * DIGIT_W + x] = 1.0;
    }
  }
  // salt noise, 1-5%
  const noiseP = 0.01 + r() * 0.04;
  for (let i = 0; i < img.length; i++) {
    if (r() < noiseP) img[i] = 1.0 - img[i];
  }
  return img;
}

export function makeDigitDataset(nPerClass, seed) {
  const r = rng(seed);
  const samples = [];
  for (let d = 0; d < 10; d++) {
    for (let n = 0; n < nPerClass; n++) {
      samples.push({ img: genDigit(d, r), label: d });
    }
  }
  return samples;
}

export function digitTokens(wasm, img) {
  return wasm.patchifyImage(img, DIGIT_W, DIGIT_H, DIGIT_PATCH, 1);
}

// Train the classifier with the dense head (10 channels, one-hot targets).
export async function trainClassifier(wasm, opts, onProgress) {
  const { nPerClass = 100, epochs = 150, lr = 1e-2, seed = 1 } = opts;
  const dataset = makeDigitDataset(nPerClass, seed);
  const T = (DIGIT_W / DIGIT_PATCH) * (DIGIT_H / DIGIT_PATCH); // 20
  const IN = DIGIT_PATCH * DIGIT_PATCH; // 4
  const N = dataset.length;

  const model = new wasm.PerceptModelWasm(IN, 16, 2, 2, 32, 10, 32, seed);
  const trainer = new wasm.TrainerWasm(lr, 16, seed);

  const tokensFlat = new Float32Array(N * T * IN);
  const targetsFlat = new Float32Array(N * T * 10);
  const masksFlat = new Float32Array(N * T).fill(1.0);
  dataset.forEach((s, i) => {
    tokensFlat.set(digitTokens(wasm, s.img), i * T * IN);
    for (let t = 0; t < T; t++) {
      targetsFlat[(i * T + t) * 10 + s.label] = 1.0;
    }
  });

  let last = null;
  for (let e = 0; e < epochs; e++) {
    last = trainer.trainDenseEpoch(
      model, tokensFlat, targetsFlat, masksFlat, N, T, "dense_mse");
    if (onProgress) onProgress(e, epochs, last);
    await yieldToUI();
  }
  return { model, dataset, lastLoss: last };
}

// Classify an 8x10 image: mean over per-token one-hot outputs.
export function classify(wasm, model, img) {
  const T = (DIGIT_W / DIGIT_PATCH) * (DIGIT_H / DIGIT_PATCH);
  const tokens = digitTokens(wasm, img);
  const out = model.transform(tokens, T);
  const scores = new Array(10).fill(0);
  for (let t = 0; t < T; t++) {
    for (let c = 0; c < 10; c++) scores[c] += out[t * 10 + c];
  }
  const best = scores.indexOf(Math.max(...scores));
  return { digit: best, scores: scores.map((s) => s / T) };
}

// ---------------------------------------------------------------------------
// Demo 2: text-overlap scorer
// ---------------------------------------------------------------------------

export const DOC_W = 32;
export const DOC_H = 32;

// A "document" is a set of text lines (x, y, w, h) on a 32x32 page.
// Lines are drawn at brightness 0.6; overlapping regions clamp to 1.0, so
// overlap both ADDS a bright blob and REMOVES ink (clamping) — a strong,
// learnable pixel signature.
export function genDoc(r) {
  const nLines = 3 + Math.floor(r() * 3); // 3..5
  const boxes = [];
  let y = 3 + Math.floor(r() * 3);
  for (let i = 0; i < nLines; i++) {
    const w = 12 + Math.floor(r() * 12);
    boxes.push([3 + Math.floor(r() * 4), y, w, 3]);
    y += 5 + Math.floor(r() * 3);
  }
  return boxes;
}

// Render boxes to a DOC_W x DOC_H image. Overlapping regions accumulate
// brightness (clamped), so overlap is encoded in pixels.
export function renderDoc(boxes) {
  const img = new Float32Array(DOC_W * DOC_H);
  for (const [bx, by, bw, bh] of boxes) {
    for (let yy = by; yy < by + bh; yy++) {
      for (let xx = bx; xx < bx + bw; xx++) {
        if (xx < 0 || xx >= DOC_W || yy < 0 || yy >= DOC_H) continue;
        const i = yy * DOC_W + xx;
        img[i] = Math.min(1.0, img[i] + 0.6);
      }
    }
  }
  return img;
}

// Corrupt a doc's boxes: shift one line into another (overlap).
export function corruptDoc(boxes, severity, r) {
  if (boxes.length < 2) return boxes.map((b) => [...b]);
  const out = boxes.map((b) => [...b]);
  const i = Math.floor(r() * out.length);
  let j = Math.floor(r() * out.length);
  while (j === i) j = Math.floor(r() * out.length);
  // move line i toward line j proportionally to severity — at high
  // severity the lines overlap substantially.
  // BUG FIX: keep coordinates INTEGRAL. Fractional y made renderDoc's
  // pixel loop write to fractional Float32Array indices, which JS
  // silently drops — the corrupted line vanished from the render and
  // mild/severe images were identical. (Classic silent-failure foot-gun.)
  const dy = (out[j][1] - out[i][1]) * severity;
  out[i][1] = Math.round(out[i][1] + dy * 0.85 + severity * 1.5);
  return out;
}

export const DOC_PATCH = 4; // -> 8x8 grid = 64 tokens x 16 channels
export const DOC_T = (DOC_W / DOC_PATCH) * (DOC_H / DOC_PATCH); // 64
export const DOC_IN = DOC_PATCH * DOC_PATCH; // 16

// Pairwise training: same base doc, mild vs severe corruption.
// (Ranking, not absolute regression — see README data-design lesson.)
export async function trainOverlapScorer(wasm, opts, onProgress) {
  const { nPairs = 120, epochs = 30, seed = 2 } = opts;
  const r = rng(seed);
  const model = new wasm.PerceptModelWasm(DOC_IN, 16, 2, 2, 32, 1, DOC_T, seed);
  const trainer = new wasm.TrainerWasm(3e-3, 16, seed);

  const betterFlat = new Float32Array(nPairs * DOC_T * DOC_IN);
  const worseFlat = new Float32Array(nPairs * DOC_T * DOC_IN);
  const seqLens = new Array(nPairs).fill(DOC_T);
  const margins = new Float32Array(nPairs).fill(0.5);

  for (let p = 0; p < nPairs; p++) {
    const base = genDoc(r);
    const mild = corruptDoc(base, 0.1 * r(), r);
    const severe = corruptDoc(base, 0.7 + 0.3 * r(), r);
    betterFlat.set(
      wasm.patchifyImage(renderDoc(mild), DOC_W, DOC_H, DOC_PATCH, 1),
      p * DOC_T * DOC_IN);
    worseFlat.set(
      wasm.patchifyImage(renderDoc(severe), DOC_W, DOC_H, DOC_PATCH, 1),
      p * DOC_T * DOC_IN);
  }

  let last = null;
  for (let e = 0; e < epochs; e++) {
    last = trainer.trainPairEpoch(model, betterFlat, worseFlat, seqLens, margins);
    if (onProgress) onProgress(e, epochs, last);
    await yieldToUI();
  }
  const acc = trainer.evalPairAccuracy(model, betterFlat, worseFlat, seqLens);
  return { model, accuracy: acc, lastLoss: last, r };
}

export function docScore(wasm, model, img) {
  return model.score(
    wasm.patchifyImage(img, DOC_W, DOC_H, DOC_PATCH, 1), DOC_T);
}

// ---------------------------------------------------------------------------
// Demo 3: colorizer
// ---------------------------------------------------------------------------

export const SCENE_W = 32;
export const SCENE_H = 32;
export const SCENE_PATCH = 4; // -> 8x8 grid = 64 tokens x 48 channels

// Synthetic color scenes: sky gradient, sun, hills, water.
export function genScene(r) {
  const img = new Float32Array(SCENE_W * SCENE_H * 3);
  const horizon = 12 + Math.floor(r() * 8);
  const sunX = 4 + Math.floor(r() * (SCENE_W - 8));
  const sunY = 3 + Math.floor(r() * Math.max(1, horizon - 8));
  const hillH = 4 + Math.floor(r() * 5);
  const water = r() < 0.5;
  for (let y = 0; y < SCENE_H; y++) {
    for (let x = 0; x < SCENE_W; x++) {
      const i = (y * SCENE_W + x) * 3;
      let cr, cg, cb;
      if (y < horizon) {
        // sky gradient: blue, brighter near horizon
        const t = y / horizon;
        cr = 0.5 + 0.3 * t;
        cg = 0.7 + 0.2 * t;
        cb = 0.9;
      } else if (water && y > SCENE_H - 8) {
        // water: dark blue with horizontal streaks
        cr = 0.15;
        cg = 0.3 + 0.1 * ((x + y) % 4 === 0 ? 1 : 0);
        cb = 0.55;
      } else {
        // hills: green
        const t = (y - horizon) / (SCENE_H - horizon);
        cr = 0.2 * t;
        cg = 0.5 - 0.2 * t;
        cb = 0.15 * t;
      }
      // sun disc
      const dx = x - sunX;
      const dy = y - sunY;
      if (dx * dx + dy * dy < 9) {
        cr = 1.0; cg = 0.85; cb = 0.3;
      }
      // hill silhouette bump
      const hx = (x - SCENE_W / 2) / SCENE_W;
      if (y >= horizon && y < horizon + hillH * (1 - Math.abs(hx) * 2) && !water) {
        cr = 0.1; cg = 0.35; cb = 0.1;
      }
      img[i] = cr;
      img[i + 1] = cg;
      img[i + 2] = cb;
    }
  }
  return img;
}

export function sceneGray(wasm, img) {
  return wasm.toGrayscale(img, SCENE_W, SCENE_H);
}

// Train: input = gray tokens (1 channel), target = RGB (3 channels out).
export async function trainColorizer(wasm, opts, onProgress) {
  const { nScenes = 150, epochs = 40, seed = 3 } = opts;
  const r = rng(seed);
  const T = (SCENE_W / SCENE_PATCH) * (SCENE_H / SCENE_PATCH); // 64
  const IN = SCENE_PATCH * SCENE_PATCH; // 16
  // out_channels = 3 * patch^2: the dense head predicts the FULL RGB
  // patch per gray token (48 channels), which unpatchifyImage reassembles.
  const model = new wasm.PerceptModelWasm(IN, 16, 2, 2, 32, 3 * SCENE_PATCH * SCENE_PATCH, T, seed);
  const trainer = new wasm.TrainerWasm(5e-3, 16, seed);

  const tokensFlat = new Float32Array(nScenes * T * IN);
  const targetsFlat = new Float32Array(nScenes * T * 3 * SCENE_PATCH * SCENE_PATCH);
  const masksFlat = new Float32Array(nScenes * T).fill(1.0);

  for (let s = 0; s < nScenes; s++) {
    const rgb = genScene(r);
    const gray = sceneGray(wasm, rgb);
    tokensFlat.set(
      wasm.patchifyImage(gray, SCENE_W, SCENE_H, SCENE_PATCH, 1),
      s * T * IN);
    targetsFlat.set(
      wasm.patchifyImage(rgb, SCENE_W, SCENE_H, SCENE_PATCH, 3),
      s * T * 3 * SCENE_PATCH * SCENE_PATCH);
  }

  let last = null;
  for (let e = 0; e < epochs; e++) {
    last = trainer.trainDenseEpoch(
      model, tokensFlat, targetsFlat, masksFlat, nScenes, T, "dense_mse");
    if (onProgress) onProgress(e, epochs, last);
    await yieldToUI();
  }
  return { model, lastLoss: last };
}

// Colorize a gray image: tokens in, RGB image out.
export function colorize(wasm, model, gray) {
  const T = (SCENE_W / SCENE_PATCH) * (SCENE_H / SCENE_PATCH);
  const tokens = wasm.patchifyImage(gray, SCENE_W, SCENE_H, SCENE_PATCH, 1);
  const out = model.transform(tokens, T); // (T, 3*16)
  return wasm.unpatchifyImage(out, SCENE_W, SCENE_H, SCENE_PATCH, 3);
}
