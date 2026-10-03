












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
export const DIGIT_PATCH = 2; 




export function genDigit(d, r) {
  const img = new Float32Array(DIGIT_W * DIGIT_H);
  const dx = Math.floor(r() * 3) - 1; 
  const dy = Math.floor(r() * 3) - 1;
  const font = FONT_5X7[d];
  for (let fy = 0; fy < 7; fy++) {
    for (let fx = 0; fx < 5; fx++) {
      if (font[fy][fx] !== "#") continue;
      const x = fx + 1 + dx; 
      const y = fy + 1 + dy; 
      if (x < 0 || x >= DIGIT_W || y < 0 || y >= DIGIT_H) continue;
      img[y * DIGIT_W + x] = 1.0;
    }
  }
  
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


export async function trainClassifier(wasm, opts, onProgress) {
  const { nPerClass = 100, epochs = 150, lr = 1e-2, seed = 1 } = opts;
  const dataset = makeDigitDataset(nPerClass, seed);
  const T = (DIGIT_W / DIGIT_PATCH) * (DIGIT_H / DIGIT_PATCH); 
  const IN = DIGIT_PATCH * DIGIT_PATCH; 
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





export const DOC_W = 32;
export const DOC_H = 32;





export function genDoc(r) {
  const nLines = 3 + Math.floor(r() * 3); 
  const boxes = [];
  let y = 3 + Math.floor(r() * 3);
  for (let i = 0; i < nLines; i++) {
    const w = 12 + Math.floor(r() * 12);
    boxes.push([3 + Math.floor(r() * 4), y, w, 3]);
    y += 5 + Math.floor(r() * 3);
  }
  return boxes;
}



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


export function corruptDoc(boxes, severity, r) {
  if (boxes.length < 2) return boxes.map((b) => [...b]);
  const out = boxes.map((b) => [...b]);
  const i = Math.floor(r() * out.length);
  let j = Math.floor(r() * out.length);
  while (j === i) j = Math.floor(r() * out.length);
  
  
  
  
  
  
  const dy = (out[j][1] - out[i][1]) * severity;
  out[i][1] = Math.round(out[i][1] + dy * 0.85 + severity * 1.5);
  return out;
}

export const DOC_PATCH = 4; 
export const DOC_T = (DOC_W / DOC_PATCH) * (DOC_H / DOC_PATCH); 
export const DOC_IN = DOC_PATCH * DOC_PATCH; 



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





export const SCENE_W = 32;
export const SCENE_H = 32;
export const SCENE_PATCH = 4; 


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
        
        const t = y / horizon;
        cr = 0.5 + 0.3 * t;
        cg = 0.7 + 0.2 * t;
        cb = 0.9;
      } else if (water && y > SCENE_H - 8) {
        
        cr = 0.15;
        cg = 0.3 + 0.1 * ((x + y) % 4 === 0 ? 1 : 0);
        cb = 0.55;
      } else {
        
        const t = (y - horizon) / (SCENE_H - horizon);
        cr = 0.2 * t;
        cg = 0.5 - 0.2 * t;
        cb = 0.15 * t;
      }
      
      const dx = x - sunX;
      const dy = y - sunY;
      if (dx * dx + dy * dy < 9) {
        cr = 1.0; cg = 0.85; cb = 0.3;
      }
      
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


export async function trainColorizer(wasm, opts, onProgress) {
  const { nScenes = 150, epochs = 40, seed = 3 } = opts;
  const r = rng(seed);
  const T = (SCENE_W / SCENE_PATCH) * (SCENE_H / SCENE_PATCH); 
  const IN = SCENE_PATCH * SCENE_PATCH; 
  
  
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


export function colorize(wasm, model, gray) {
  const T = (SCENE_W / SCENE_PATCH) * (SCENE_H / SCENE_PATCH);
  const tokens = wasm.patchifyImage(gray, SCENE_W, SCENE_H, SCENE_PATCH, 1);
  const out = model.transform(tokens, T); 
  return wasm.unpatchifyImage(out, SCENE_W, SCENE_H, SCENE_PATCH, 3);
}
