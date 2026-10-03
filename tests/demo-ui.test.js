const assert = require("assert");
const wasm = require("../pkg/percept_wasm.js");
const {
  trainClassifier, genDigit,
  genDoc, renderDoc, corruptDoc, trainOverlapScorer,
  DIGIT_W, DIGIT_H, DOC_W, DOC_H,
} = require("../docs/engine.js");

async function main() {
  const { demoInfer } = await import("file://" + require("path").resolve(__dirname, "../docs/demo-ui.js"));

  console.log("case 1: demoInfer classify path (drawn digit pixels)...");
  const { model: clf } = await trainClassifier(wasm, { nPerClass: 60, epochs: 80, seed: 3 });
  const spec = { kind: "classify", W: DIGIT_W, H: DIGIT_H, PATCH: 2, inCh: 1, labels: ["0","1","2","3","4","5","6","7","8","9"] };
  const session = { scoreMin: Infinity, scoreMax: -Infinity };
  let correct = 0, n = 0;
  for (let label = 0; label < 10; label++) {
    for (let k = 0; k < 3; k++) {
      const img = genDigit(label, wasm.rng ? wasm.rng(label * 100 + k) : (() => { let s = (label * 31 + k * 7 + 12345) >>> 0; return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; })());
      const px = new Uint8Array(img.length);
      for (let i = 0; i < img.length; i++) px[i] = Math.max(0, Math.min(255, Math.round(img[i] * 255)));
      const r = demoInfer(px, clf, wasm, spec, session);
      n++;
      if (r.best === label) correct++;
      assert.ok(r.label === spec.labels[r.best], "label mismatch");
      assert.ok(r.norm.length === 10 && r.norm[r.best] > 0.9, "norm bars wrong");
    }
  }
  const acc = correct / n;
  console.log(`  drawn-pixel classify acc: ${(acc * 100).toFixed(1)}% (${correct}/${n})`);
  assert.ok(acc > 0.7, `demo classify accuracy too low: ${acc}`);

  const blank = new Uint8Array(DIGIT_W * DIGIT_H);
  const rEmpty = demoInfer(blank, clf, wasm, spec, session);
  assert.ok(rEmpty.empty === true, "blank canvas must return empty");

  console.log("case 2: demoInfer score path (clean vs corrupted doc)...");
  const { model: scorer } = await trainOverlapScorer(wasm, { nPairs: 120, epochs: 30, seed: 4 });
  const sSpec = { kind: "score", W: DOC_W, H: DOC_H, PATCH: 4, inCh: 1 };
  const sSession = { scoreMin: Infinity, scoreMax: -Infinity };
  const mkRng = (seedv) => { let s = seedv >>> 0; return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
  let cleanWins = 0, pairs = 0;
  for (let k = 0; k < 20; k++) {
    const rng = mkRng(9000 + k);
    const doc = genDoc(rng);
    const cleanPx = new Uint8Array(DOC_W * DOC_H);
    const cImg = renderDoc(doc);
    for (let i = 0; i < cleanPx.length; i++) cleanPx[i] = Math.max(0, Math.min(255, Math.round(cImg[i] * 255)));
    const rc = demoInfer(cleanPx, scorer, wasm, sSpec, sSession);
    assert.ok(rc.kind === "score" && typeof rc.score === "number", "score result missing");
    const bad = corruptDoc(doc, 0.8, rng);
    const bImg = renderDoc(bad);
    const badPx = new Uint8Array(DOC_W * DOC_H);
    for (let i = 0; i < badPx.length; i++) badPx[i] = Math.max(0, Math.min(255, Math.round(bImg[i] * 255)));
    const rb = demoInfer(badPx, scorer, wasm, sSpec, sSession);
    pairs++;
    if (rc.score > rb.score) cleanWins++;
  }
  console.log(`  clean > corrupted on ${(cleanWins / pairs * 100).toFixed(1)}% of ${pairs} pairs`);
  assert.ok(cleanWins / pairs > 0.8, `score ordering too weak: ${cleanWins}/${pairs}`);
  assert.ok(sSession.scoreMax > sSession.scoreMin, "session range not tracked");

  console.log("ALL DEMO UI TESTS PASSED");
}

main().catch((e) => { console.error("FAILED:", e.message); process.exit(1); });
