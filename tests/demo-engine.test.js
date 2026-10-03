

const assert = require("assert");
const wasm = require("../pkg/percept_wasm.js");
const {
  makeDigitDataset, genDigit, trainClassifier, classify,
  genDoc, renderDoc, corruptDoc, trainOverlapScorer, docScore,
  genScene, sceneGray, trainColorizer, colorize,
  DIGIT_W, DIGIT_H, DOC_W, DOC_H, SCENE_W, SCENE_H,
} = require("../docs/engine.js");

async function main() {
  
  console.log("training digit classifier...");
  const t1 = Date.now();
  
  const { model: clf, lastLoss: clfLoss } = await trainClassifier(wasm, {
    nPerClass: 100, epochs: 150, lr: 1e-2, seed: 1,
  });
  console.log(`  done in ${Date.now() - t1}ms, final loss ${clfLoss.toFixed(4)}`);

  
  const testSet = makeDigitDataset(10, 999);
  let correct = 0;
  const perClass = new Array(10).fill(0);
  const perClassTot = new Array(10).fill(0);
  for (const s of testSet) {
    const { digit } = classify(wasm, clf, s.img);
    perClassTot[s.label]++;
    if (digit === s.label) {
      correct++;
      perClass[s.label]++;
    }
  }
  const acc = correct / testSet.length;
  console.log(`  held-out accuracy: ${(acc * 100).toFixed(1)}%`);
  console.log(`  per-class: ${perClass.map((c, i) => `${i}:${c}/${perClassTot[i]}`).join(" ")}`);
  assert.ok(acc > 0.85, `digit accuracy too low: ${acc}`);

  // ---- demo 2: text-overlap scorer ----
  console.log("training overlap scorer...");
  const t2 = Date.now();
  const { model: scorer, accuracy: rankAcc } = await trainOverlapScorer(wasm, {
    nPairs: 120, epochs: 30, seed: 2,
  });
  console.log(`  done in ${Date.now() - t2}ms, train-pair accuracy ${(rankAcc * 100).toFixed(1)}%`);
  assert.ok(rankAcc > 0.9, `ranking accuracy too low: ${rankAcc}`);

  // held-out: fresh docs (new seed) — clean docs must score above corrupted
  const r = (() => {
    let s = 4242 >>> 0;
    return () => {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  })();
  let heldCorrect = 0, heldN = 0;
  for (let i = 0; i < 30; i++) {
    const base = genDoc(r);
    const clean = renderDoc(base);
    const bad = renderDoc(corruptDoc(base, 0.9, r));
    const sClean = docScore(wasm, scorer, clean);
    const sBad = docScore(wasm, scorer, bad);
    heldN++;
    if (sClean > sBad) heldCorrect++;
  }
  const heldAcc = heldCorrect / heldN;
  console.log(`  held-out clean > corrupted: ${(heldAcc * 100).toFixed(1)}%`);
  assert.ok(heldAcc > 0.8, `held-out scoring accuracy too low: ${heldAcc}`);

  // ---- demo 3: colorizer ----
  console.log("training colorizer...");
  const t3 = Date.now();
  const { model: col, lastLoss: colLoss } = await trainColorizer(wasm, {
    nScenes: 150, epochs: 40, seed: 3,
  });
  console.log(`  done in ${Date.now() - t3}ms, final loss ${colLoss.toFixed(4)}`);
  assert.ok(colLoss < 0.02, `colorizer loss too high: ${colLoss}`);

  // held-out colorization: fresh scene, compare model output vs ground truth
  const rgb = genScene(r);
  const gray = sceneGray(wasm, rgb);
  const out = colorize(wasm, col, gray);
  let err = 0;
  for (let i = 0; i < out.length; i++) err += Math.abs(out[i] - rgb[i]);
  const mae = err / out.length;
  console.log(`  held-out colorization MAE: ${mae.toFixed(4)}`);
  assert.ok(mae < 0.15, `colorization MAE too high: ${mae}`);

  // baseline sanity: predicting flat gray would be much worse
  let baseErr = 0;
  for (let i = 0; i < out.length; i++) baseErr += Math.abs(0.5 - rgb[i]);
  console.log(`  (flat-0.5 baseline MAE: ${(baseErr / out.length).toFixed(4)} — model must beat this)`);

  console.log("\nALL DEMO ENGINE TESTS PASSED");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
