



const assert = require("assert");
const fs = require("fs");
const path = require("path");
const os = require("os");
const wasm = require("../pkg/percept_wasm.js");



async function loadAgentCore() {
  const core = await import(pathToFileURL("../docs/agent-core.js"));
  return core;
}
function pathToFileURL(rel) {
  const abs = path.resolve(__dirname, rel);
  return "file://" + abs;
}

async function main() {
  const { runPipeline } = await loadAgentCore();
  const engine = await import(pathToFileURL("../docs/engine.js"));

  
  
  
  
  
  console.log("case 1: dense_mse with agent-written corruptor...");
  const corruptorSrc = `
    function (sample, severity, rng, label) {
      const out = new Float32Array(sample);
      const W = 8, H = 10;
      // salt noise proportional to severity
      const n = Math.floor(out.length * severity * 0.3);
      for (let i = 0; i < n; i++) {
        const idx = Math.floor(rng() * out.length);
        out[idx] = 1.0 - out[idx];
      }
      // vertical shift: integral rows only
      const dy = Math.round(severity * 2 * (rng() < 0.5 ? 1 : -1));
      if (dy !== 0) {
        const shifted = new Float32Array(out.length);
        for (let y = 0; y < H; y++) {
          const ny = y + dy;
          if (ny < 0 || ny >= H) continue;
          for (let x = 0; x < W; x++) shifted[ny * W + x] = out[y * W + x];
        }
        return shifted;
      }
      return out;
    }
  `;
  
  const corruptor = new Function("return (" + corruptorSrc.trim() + ")")();

  const logs = [];
  const plan = {
    goal: "classify noisy digits",
    labels: ["0","1","2","3","4","5","6","7","8","9"],
    training_kind: "dense_mse",
    corruptors: ["noise+shift"],
    _corruptor: corruptor,
    _probe: ({ model, wasm, engine, rng }) => {
      
      const img = engine.genDigit(7, rng);
      const tokens = wasm.patchifyImage(img, 8, 10, 2, 1);
      const T = 20;
      const out = model.transform(tokens, T);
      const scores = new Array(10).fill(0);
      for (let t = 0; t < T; t++) for (let c = 0; c < 10; c++) scores[c] += out[t * 10 + c];
      const best = scores.indexOf(Math.max(...scores));
      return { probe: "digit-7", predicted: best };
    },
    seed: 7,
  };
  const t0 = Date.now();
  const res1 = await runPipeline(wasm, engine, plan, {
    log: (t) => logs.push(t),
    progress: () => {},
  });
  console.log("  logs:", logs.join(" | "));
  assert.ok(res1.ok, "dense pipeline failed: " + JSON.stringify(res1).slice(0, 200));
  assert.ok(res1.model, "no model returned");
  const acc1 = parseFloat(res1.metrics.find((m) => m.label === "held-out acc (best round)").value);
  console.log(`  done in ${Date.now() - t0}ms, held-out acc ${acc1}%`);
  assert.ok(acc1 > 60, `dense held-out accuracy too low: ${acc1}`);

  
  const bytes = res1.model.save();
  console.log(`  .pkm export: ${bytes.length} bytes, ${res1.model.nParams()} params`);
  assert.ok(bytes.length > 1000, "suspiciously small model file");
  const reloaded = wasm.PerceptModelWasm.load(bytes);
  const img = engine.genDigit(3, engine.rng(55));
  const tokens = wasm.patchifyImage(img, 8, 10, 2, 1);
  const predA = res1.model.transform(tokens, 20);
  const predB = reloaded.transform(tokens, 20);
  for (let i = 0; i < predA.length; i++) {
    assert.ok(Math.abs(predA[i] - predB[i]) < 1e-4, "reloaded model diverges at " + i);
  }
  console.log("  save/load round-trip: identical predictions");
  assert.ok(res1.probe && res1.probe.predicted !== undefined, "probe did not run");

  
  
  
  console.log("case 2: score_pair with agent-written corruptor...");
  logs.length = 0;
  const docCorruptor = new Function("return (" + `
    function (sample, severity, rng, label) {
      // sample: boxes [[x,y,w,h]...]; shift one line into another
      if (sample.length < 2) return sample.map((b) => [...b]);
      const out = sample.map((b) => [...b]);
      const i = Math.floor(rng() * out.length);
      let j = Math.floor(rng() * out.length);
      while (j === i) j = Math.floor(rng() * out.length);
      const dy = (out[j][1] - out[i][1]) * severity;
      out[i][1] = Math.round(out[i][1] + dy * 0.85 + severity * 1.5);
      return out;
    }
  ` + ")")();

  const plan2 = {
    goal: "score text overlap on document layouts",
    labels: ["quality"],
    training_kind: "score_pair",
    corruptors: ["line-shift"],
    _corruptor: docCorruptor,
    seed: 2,
  };
  const t1 = Date.now();
  const res2 = await runPipeline(wasm, engine, plan2, {
    log: (t) => logs.push(t),
    progress: () => {},
  });
  console.log("  logs:", logs.join(" | "));
  assert.ok(res2.ok, "pair pipeline failed: " + JSON.stringify(res2).slice(0, 200));
  const held = parseFloat(res2.metrics.find((m) => m.label === "held-out clean>corrupt").value);
  console.log(`  done in ${Date.now() - t1}ms, held-out clean>corrupt ${held}%`);
  assert.ok(held > 85, `pair held-out too low: ${held}`);

  console.log("case 3: placement-only corruptor must not become the signal (augment invariance)...");
  logs.length = 0;
  const kitMod = await import(pathToFileURL("../docs/kit/synth.js"));
  const W3 = 32, H3 = 32;
  const gen3 = new Function("kit", `return function(rng, label) {
    const img = new Float32Array(${W3} * ${H3});
    kit.circle(img, ${W3}, ${H3}, 16, 16, 9, { stroke: 1, w: 1.5, soft: 0.4 });
    return img;
  };`)(kitMod);
  const placeOnly = new Function("kit", `return function(sample, severity, rng, label) {
    let out = kit.translate(sample, ${W3}, ${H3}, (rng() - 0.5) * 6 * severity, (rng() - 0.5) * 6 * severity);
    if (severity > 0.3) out = kit.rotate(out, ${W3}, ${H3}, (rng() - 0.5) * 30 * severity);
    return out;
  };`)(kitMod);
  const plan3 = {
    goal: "score circle quality (placement-invariance regression)",
    labels: ["quality"],
    training_kind: "score_pair",
    corruptors: ["place-only"],
    gen: gen3,
    _corruptor: placeOnly,
    n_pairs: 60,
    epochs: 12,
    seed: 9,
  };
  const res3 = await runPipeline(wasm, engine, plan3, { log: (t) => logs.push(t), progress: () => {} });
  const held3 = parseFloat(res3.metrics.find((mm) => mm.label === "held-out clean>corrupt").value);
  console.log("  placement-only corruptor held-out clean>corrupt: " + held3 + "% (without augment this is ~100%)");
  assert.ok(held3 < 85, "placement-only corruptor leaked too strongly: held-out " + held3 + "% (augment should suppress placement signal toward chance)");

  console.log("case 4: quality-degrading corruptor still learns under augmentation...");
  logs.length = 0;
  const wobble = new Function("kit", `return function(sample, severity, rng, label) {
    let out = new Float32Array(sample);
    out = kit.gauss(out, rng, 0.25 * severity);
    out = kit.dropout(out, rng, 0.12 * severity);
    if (rng() < severity) out = kit.occlude(out, ${W3}, ${H3}, 6 + rng() * 16, 6 + rng() * 16, 3 + rng() * 5, 3 + rng() * 5, 0);
    return kit.clamp01(out);
  };`)(kitMod);
  const plan4 = { ...plan3, goal: "score circle quality (wobble learns)", _corruptor: wobble, corruptors: ["wobble"] };
  const res4 = await runPipeline(wasm, engine, plan4, { log: (t) => logs.push(t), progress: () => {} });
  const held4v = parseFloat(res4.metrics.find((mm) => mm.label === "held-out clean>corrupt").value);
  console.log("  wobble corruptor held-out clean>corrupt: " + held4v + "%");
  assert.ok(held4v > 85, "quality corruptor failed to learn under augmentation: held-out " + held4v + "%");

  console.log("\nALL AGENT PIPELINE TESTS PASSED");
}

main().catch((err) => {
  console.error("FAILED:", err);
  process.exit(1);
});
