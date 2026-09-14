// Headless test of the agent pipeline (agent-core.js runPipeline):
// the LLM-planned path must actually train, hit held-out accuracy, and
// produce exportable .pkm bytes — without any LLM call.
// Run: node tests/agent-pipeline.test.js
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const os = require("os");
const wasm = require("../pkg/percept_wasm.js");

// agent-core.js is ESM; runPipeline is exported. Load it via dynamic import.
// (node 22 supports import() of ESM from CJS)
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

  // ---------------------------------------------------------------
  // case 1: dense_mse — LLM-style corruptor (salt noise + row shift)
  // on digit data. This is the exact shape of code write_corruptor
  // accepts: function(sample, severity, rng, label)
  // ---------------------------------------------------------------
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
  // compile the same way write_corruptor does
  const corruptor = new Function("return (" + corruptorSrc.trim() + ")")();

  const logs = [];
  const plan = {
    goal: "classify noisy digits",
    labels: ["0","1","2","3","4","5","6","7","8","9"],
    training_kind: "dense_mse",
    corruptors: ["noise+shift"],
    _corruptor: corruptor,
    _probe: ({ model, wasm, engine, rng }) => {
      // probe contract: returns JSON-serializable result
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

  // export round-trip: save -> bytes -> load -> same prediction
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

  // ---------------------------------------------------------------
  // case 2: score_pair — agent corruptor on docs (layout overlap)
  // ---------------------------------------------------------------
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

  console.log("\nALL AGENT PIPELINE TESTS PASSED");
}

main().catch((err) => {
  console.error("FAILED:", err);
  process.exit(1);
});
