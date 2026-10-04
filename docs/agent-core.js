











import { Agent } from "@earendil-works/pi-agent-core";
import * as kit from "./kit/synth.js?v=1.3";
import { createAssistantMessageEventStream, Type } from "@earendil-works/pi-ai";

const API_ENDPOINT = "https://webiome.abbie.workers.dev/api/stream";
const MODEL = {
  id: "google/gemini-2.5-flash",
  name: "Gemini 2.5 Flash through Exa",
  api: "webiome-exa",
  provider: "webiome",
  baseUrl: "",
  reasoning: false,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 128000,
  maxTokens: 8192,
};





export class VFS {
  constructor() {
    this.map = new Map();
  }
  read(path) { return this.map.get(path) ?? null; }
  write(path, content) { this.map.set(path, String(content)); }
  del(path) { return this.map.delete(path); }
  list() { return [...this.map.keys()].sort(); }
  snapshot() {
    return [...this.map.entries()].map(([k, v]) => ({
      path: k, size: v.length, head: v.slice(0, 60).replace(/\n/g, " "),
    }));
  }
}

const normalizePath = (p) => {
  let s = String(p).replace(/^[a-z]+:\/\/?/, "").replace(/^\/+/, "");
  return s || String(p);
};






export async function runPipeline(wasm, engine, plan, ui) {
  const log = (t) => ui?.log?.(t);
  const onProgress = (e, epochs, loss) =>
    ui?.progress?.(e, epochs, loss);
  const viz = (event) => ui?.viz?.(event); 
  const iterCb = (n, total, text) => ui?.iteration?.(n, total, text);

  const kind = plan.training_kind || "dense_mse";
  const corruptor = plan._corruptor; 
  
  
  const probe = plan._probe;
  if (!corruptor) return { error: "no corruptor — write corruptor.js first" };

  let r = engine.rng(plan.seed ?? 7);

  if (kind === "score_pair") {
    
    
    
    
    const W = plan.img_w ?? engine.DOC_W, H = plan.img_h ?? engine.DOC_H;
    const PATCH = plan.patch ?? engine.DOC_PATCH;
    const T = (W / PATCH) * (H / PATCH);
    const IN = PATCH * PATCH;
    const nPairs = plan.n_pairs ?? 120;
    const epochs = plan.epochs ?? 30;

    const base = plan.gen || engine.genDoc;
    const render = plan.render || ((x) => {
      if (ArrayBuffer.isView(x)) return x;
      return engine.renderDoc(x);
    });
    const toImg = (x) => (ArrayBuffer.isView(x) ? x : render(x, kit));

    const betterFlat = new Float32Array(nPairs * T * IN);
    const worseFlat = new Float32Array(nPairs * T * IN);
    const seqLens = new Array(nPairs).fill(T);
    const margins = new Float32Array(nPairs).fill(0.5);

    for (let p = 0; p < nPairs; p++) {
      const sample = base(r, 0, kit);
      let mild, severe;
      try {
        mild = corruptor(sample, 0.1 * r(), r, 0, kit);
        severe = corruptor(sample, 0.7 + 0.3 * r(), r, 0, kit);
        wasm.patchifyImage(render(mild, kit), W, H, PATCH, 1);
        wasm.patchifyImage(render(severe, kit), W, H, PATCH, 1);
      } catch (e) {
        let renderOut = "(not reached)";
        try {
          renderOut = dump(render(base(r, 0, kit), kit));
        } catch (e2) {
          renderOut = "render itself throws: " + e2.message;
        }
        const dump = (v) => {
          if (ArrayBuffer.isView(v)) return `[${v.constructor.name} ${v.length}]`;
          if (Array.isArray(v)) return `[array ${v.length}: ${JSON.stringify(v[0]).slice(0,40)}...]`;
          return typeof v + " " + JSON.stringify(v)?.slice(0, 60);
        };
        throw new Error(
          `corruptor/render crash at pair ${p}: ${e.message}. ` +
          `sample=${dump(sample)} mild=${dump(mild)} severe=${dump(severe)} render=${renderOut}. ` +
          `Fix the corruptor or render to return a Float32Array of ${W * H} floats.`);
      }
      betterFlat.set(
        wasm.patchifyImage(toImg(mild), W, H, PATCH, 1), p * T * IN);
      worseFlat.set(
        wasm.patchifyImage(toImg(severe), W, H, PATCH, 1), p * T * IN);
    }
    viz({ type: "update", stage: "data", message: `building ${nPairs} pairwise samples` });
    log(`data: ${nPairs} pairs (mild vs severe), ${T} tokens/seq`);

    const model = new wasm.PerceptModelWasm(IN, 16, 2, 2, 32, 1, T, plan.seed ?? 7);
    const trainer = new wasm.TrainerWasm(3e-3, 16, plan.seed ?? 7);

    let last = null;
    for (let e = 0; e < epochs; e++) {
      last = trainer.trainPairEpoch(model, betterFlat, worseFlat, seqLens, margins);
      onProgress?.(e, epochs, last);
      if (e % 4 === 0) {
        viz({ type: "update", stage: "train", losses: [{ epoch: e, value: last }] });
      }
      await new Promise((res) => setTimeout(res, 0));
    }
    const acc = trainer.evalPairAccuracy(model, betterFlat, worseFlat, seqLens);
    log(`train: final pair loss ${last.toFixed(4)}, train acc ${(acc * 100).toFixed(1)}%`);
    if (acc < 0.6) log(`warning: train pair acc ${(acc * 100).toFixed(1)}% is near/below chance — the model likely failed to learn the clean-vs-corrupt distinction; check corruptor severity spread`);

    
    const r2 = engine.rng((plan.seed ?? 7) + 1000);
    let heldOk = 0, heldN = 0;
    for (let p = 0; p < 40; p++) {
      const sample = base(r2, 0, kit);
      const clean = wasm.patchifyImage(toImg(sample), W, H, PATCH, 1);
      const bad = wasm.patchifyImage(toImg(corruptor(sample, 0.8, r2, 0, kit)), W, H, PATCH, 1);
      heldN++;
      if (model.score(clean, T) > model.score(bad, T)) heldOk++;
    }
    const heldAcc = heldOk / heldN;
    log(`held-out: clean > corrupted on ${(heldAcc * 100).toFixed(1)}% of 40 fresh docs`);

    let probeOut = null;
    if (probe) {
      try {
        probeOut = probe({ model, wasm, engine, rng: r2, kit });
        viz({ type: "probe", probe: probeOut, model });
        log("probe: ok");
      } catch (err) { log("probe error: " + String(err).slice(0, 120)); }
    }
    return {
      ok: true, kind, model, metrics: [
        { label: "train pair loss", value: last.toFixed(4) },
        { label: "train pair acc", value: (acc * 100).toFixed(1) + "%" + (acc < 0.6 ? " (near chance — model likely failed)" : "") },
        { label: "held-out clean>corrupt", value: (heldAcc * 100).toFixed(1) + "%" },
      ], probe: probeOut,
      demoSpec: { kind: "score", W, H, PATCH, inCh: 1 },
    };
  }

  
  
  
  
  const W = plan.img_w ?? engine.DIGIT_W, H = plan.img_h ?? engine.DIGIT_H;
  const PATCH = plan.patch ?? engine.DIGIT_PATCH;
  const inCh = plan.in_channels ?? 1;
  const outCh = plan.out_channels ?? (plan.labels ? plan.labels.length : 1);
  const T = (W / PATCH) * (H / PATCH);
  const IN = PATCH * PATCH * inCh;
  const OUT = PATCH * PATCH * outCh;
  const nPer = plan.n_per_class ?? 60;
  const epochs = plan.epochs ?? 80;
  const maxIterations = plan.max_iterations ?? 3;

  const gen = plan.gen || ((r, label) => engine.genDigit(label ?? 0, r));
  const nLabels = plan.labels?.length || 10;

  
  const adv = { bag: [], severity: plan.corrupt_severity ?? 0.15, iteration: 0, losses: [] };

  const buildAndTrainData = () => {
    const N = nPer * nLabels;
    const tokensFlat = new Float32Array(N * T * IN);
    const targetsFlat = new Float32Array(N * T * OUT);
    const masksFlat = new Float32Array(N * T).fill(1.0);
    for (let i = 0; i < N; i++) {
      const label = i % nLabels;
      const raw = gen(r, label, kit);
      const img = corruptor(raw, adv.severity, r, label, kit);
      tokensFlat.set(wasm.patchifyImage(img, W, H, PATCH, inCh), i * T * IN);
      for (let t = 0; t < T; t++) {
        if (outCh === nLabels) targetsFlat[(i * T + t) * OUT + label] = 1.0;
      }
    }
    viz({ type: "update", stage: "data", message: `round ${adv.iteration + 1}: ${N} samples (sev ${adv.severity.toFixed(2)})` });
    return { N, tokensFlat, targetsFlat, masksFlat };
  };

  const evalHeldOut = (model) => {
    const r2 = engine.rng((plan.seed ?? 7) + 1000 + adv.iteration);
    let correct = 0, heldN = 0;
    const perClass = new Array(nLabels).fill(0);
    const perClassTot = new Array(nLabels).fill(0);
    const mismatches = [];
    for (let i = 0; i < 10 * nLabels; i++) {
      const label = i % nLabels;
      const raw = gen(r2, label, kit);
      const img = corruptor(raw, adv.severity, r2, label, kit);
      const tokens = wasm.patchifyImage(img, W, H, PATCH, inCh);
      const out = model.transform(tokens, T);
      const scores = new Array(nLabels).fill(0);
      for (let t = 0; t < T; t++)
        for (let c = 0; c < nLabels; c++) scores[c] += out[t * OUT + c];
      const best = scores.indexOf(Math.max(...scores));
      heldN++; perClassTot[label]++;
      if (best === label) { correct++; perClass[label]++; }
      else if (mismatches.length < 8) mismatches.push({ img, label, predicted: best });
    }
    return { acc: correct / heldN, n: heldN, perClass, perClassTot, mismatches };
  };

  let model = null, roundMetrics = null, probeOut = null, best = { acc: -1, model: null, metrics: null, probe: null };
  for (let it = 0; it < maxIterations; it++) {
    adv.iteration = it;
    iterCb?.(it + 1, maxIterations, `training round ${it + 1}/${maxIterations}`);
    const { N, tokensFlat, targetsFlat, masksFlat } = buildAndTrainData();
    model = new wasm.PerceptModelWasm(IN, 16, 2, 2, 32, OUT, T, (plan.seed ?? 7) + it);
    const trainer = new wasm.TrainerWasm(plan.lr ?? 1e-2, 16, (plan.seed ?? 7) + it);
    let last = null;
    for (let e = 0; e < epochs; e++) {
      last = trainer.trainDenseEpoch(model, tokensFlat, targetsFlat, masksFlat, N, T, "dense_mse");
      adv.losses.push({ round: it, epoch: e, value: last });
      onProgress?.(e, epochs, last);
      if (e % 5 === 0 || e === epochs - 1) viz({ type: "update", stage: "train", losses: adv.losses });
      await new Promise((res) => setTimeout(res, 0));
    }
    const held = evalHeldOut(model);
    roundMetrics = { loss: last, acc: held.acc, n: held.n, perClass: held.perClass, perClassTot: held.perClassTot };
    log(`round ${it + 1}: loss ${last.toFixed(4)}, held-out ${(held.acc * 100).toFixed(1)}% (${held.n})`);

    probeOut = null;
    if (probe) {
      const r2 = engine.rng((plan.seed ?? 7) + 2000 + it);
      try {
        probeOut = probe({ model, wasm, engine, rng: r2, kit, iteration: it });
        viz({ type: "probe", probe: probeOut, model });
        log("probe: ok");
      } catch (err) { log("probe error: " + String(err).slice(0, 120)); }
    }
    
    let added = 0;
    for (const m of ((probeOut?.cases) || []).slice(0, 20)) {
      const sample = m.sample ?? m.input ?? m.img;
      if (sample !== undefined) { adv.bag.push(sample); added++; }
    }
    
    for (const mm of held.mismatches.slice(0, 8)) {
      try { adv.bag.push(mm.img); added++; } catch {  }
    }
    if (added) {
      adv.severity = Math.min(0.55, adv.severity + 0.05);
      log(`adversarial: ${added} hard cases → severity ${adv.severity.toFixed(2)} (bag ${adv.bag.length})`);
      viz({ type: "adversarial", message: `${added} hard cases → severity ${adv.severity.toFixed(2)}` });
    }
    if (held.acc > best.acc) {
      best = { acc: held.acc, model, metrics: roundMetrics, probe: probeOut };
    }
    if (!added && held.acc > 0.9) break; 
  }
  if (best.model) {
    const finalIdx = adv.iteration;
    if (roundMetrics.acc < best.acc) {
      log(`selecting round ${best.metrics ? "best" : ""} model @ ${(best.acc * 100).toFixed(1)}% over last ${(roundMetrics.acc * 100).toFixed(1)}%`);
      model = best.model; roundMetrics = best.metrics; probeOut = best.probe;
    }
  }

  return {
    ok: true, kind: "dense_mse", model, metrics: [
      { label: "final loss (last round)", value: roundMetrics.loss.toFixed(4) },
      { label: "held-out acc (best round)", value: (roundMetrics.acc * 100).toFixed(1) + "%" },
      { label: "iterations", value: String(adv.iteration + 1) },
    ], probe: probeOut, perClass: roundMetrics.perClass, perClassTot: roundMetrics.perClassTot,
    demoSpec: {
      kind: "classify", W, H, PATCH, inCh,
      labels: plan.labels || Array.from({ length: nLabels }, (_, i) => String(i)),
    },
  };
}





export class PerceptKitAgent {
  constructor({ pkg, engine, ui }) {
    this.pkg = pkg;       
    this.engine = engine; 
    this.ui = ui;
    this.vfs = new VFS();
    this.state = { phase: "idle", plan: null, result: null };
    this._tools = this._buildTools();
    this._agent = new Agent(this._buildConfig());
    this._agent.subscribe((event) => this._onEvent(event));
  }

  _buildConfig() {
    return {
      initialState: {
        systemPrompt: this._systemPrompt(),
        model: MODEL,
        tools: this._tools,
      },
      streamFn: (model, context, options) => this._stream(model, context, options),
      toolExecution: "sequential",
      shouldStopAfterTurn: ({ message }) => message.stopReason !== "toolUse",
    };
  }

  _systemPrompt() {
    return [
      "You are the PerceptKit training agent. The user gives a perception goal;",
      "you plan and drive an in-browser training pipeline that ends with a",
      "reusable, downloadable .pkm model. Be efficient: the whole run must",
      "finish in a handful of tool calls.",
      "",
      "Required workflow (exactly this order):",
      "1. plan_label_set — commit to: goal, labels, training_kind",
      "   ('dense_mse' for classification/generation, 'score_pair' for",
      "   scoring/ranking), corruptor names, and pipeline settings.",
      "2. write_corruptor — write the corruption function. CONTRACT:",
      "   function(sample, severity, rng, label) -> corrupted sample of the",
      "   SAME shape/type as the input. rng() returns uniform [0,1). Keep any",
      "   coordinates integral. severity in [0,1].",
      "   Default data generators (pass gen:null to use them):",
      "   - digits: genDigit-like 8x10 Float32Array images, 10 classes",
      "   - docs: genDoc returns boxes [[x,y,w,h]...]; renderDoc renders them",
      "   - scenes: genScene returns 32x32 RGB Float32Array",
      "   If the goal needs different data, pass a `gen` function in the plan",
      "   (same call contract: gen(rng, label)).",
      "KIT (available as the `kit` arg in gen/corruptor/render, and in the",
      "   probe arg object). All operate on Float32Array images (row-major,",
      "   0..1). USE THE KIT instead of writing raw pixel loops:",
      "   kit.blank(W,H) clone(img)",
      "   kit.circle/ellipse/rect(img,W,H,...) opts {fill,stroke,w,soft}",
      "   kit.line(img,W,H,x0,y0,x1,y1,{w,val}) polyline(img,W,H,pts,opts)",
      "   kit.bezier(img,W,H,[p0,p1,p2(,p3)],opts) polygon(img,W,H,pts,opts)",
      "   kit.text(img,W,H,str,x,y,{scale}) — 5x7 caps+digits font",
      "   kit.noise(img,rng,p) gauss(img,rng,sigma) dropout(img,rng,p)",
      "   kit.translate/rotate/shear/scaleImg(img,W,H,...) — return NEW image",
      "   kit.occlude(img,W,H,x,y,w,h,val) invert threshold blur3 clamp01",
      "   kit.coverage(img,t) centroid(img,W,H) bbox(img,W,H,t) — measures",
      "   e.g. corruptor = (s, sev, rng) => kit.clamp01(kit.gauss(",
      "     kit.translate(s, W, H, (rng()-0.5)*8*sev, (rng()-0.5)*8*sev),",
      "     rng, sev*0.3));",
      "",
      "3. write_probe (recommended) — function({model, wasm, engine, rng,",
      "   iteration}) -> JSON result for the score card AND, optionally,",
      "   adversarial cases: return {stats:{...}, cases:[{sample|input|img,",
      "   label, predicted}]} to feed the next iteration's hard-case bag.",
      "   The pipeline runs up to plan.max_iterations rounds; each round's",
      "   probe can mine new hard cases that the corruptor then faces.",
      "   THE LIVE VIEWER consumes the probe return — pick ONE drawable",
      "   field: {losses:[{epoch,value}]} (curve), {values:[num...]} (bars),",
      "   {points:[[x,y]...]} (scatter), {matrix:[[...]...]} (heat map), or",
      "   {scores:[num...]} (prediction bar, best green). Unknown fields",
      "   fall back to a summary card.",
      "4. run_pipeline — trains for real with the wasm trainer, iterating up",
      "   to plan.max_iterations rounds with adversarial mining, and returns",
      "   losses, held-out accuracy (best round), and probe output.",
      "5. export_model — serializes the trained model to .pkm bytes.",
      "",
      "Data-design rules (non-negotiable):",
      "- If corruption is stochastic, never regress absolute severity —",
      "  use score_pair (pairwise ranking) instead.",
      "- Dense per-token targets are for classification/generation; global",
      "  score is for scoring/ranking only.",
      "- Keep corruptor outputs the same shape as inputs; validate by",
      "  construction, not by hoping.",
      "",
      "After export_model succeeds, reply with a 2-3 sentence summary of",
      "what was trained and the held-out numbers. Do not call more tools.",
    ].join("\n");
  }

  

  _stream(model, context, options) {
    const stream = createAssistantMessageEventStream();
    void this._forward(stream, model, context, options);
    return stream;
  }

  async _forward(stream, model, context, options) {
    let output;
    try {
      const res = await fetch(API_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: options?.signal,
        body: JSON.stringify({
          model,
          context: {
            systemPrompt: context.systemPrompt,
            messages: context.messages || [],
            tools: (context.tools || []).map((t) => ({
              name: t.name,
              description: t.description,
              parameters: t.parameters,
            })),
          },
        }),
      });
      if (!res.ok) throw new Error(`API ${res.status}: ${(await res.text()).slice(0, 300)}`);
      if (!res.body) throw new Error("no response body");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let pending = "";
      while (true) {
        const { done, value } = await reader.read();
        pending += decoder.decode(value || new Uint8Array(), { stream: !done });
        const events = pending.split("\n\n");
        pending = done ? "" : (events.pop() || "");
        for (const event of events) {
          const line = event.split("\n").find((l) => l.startsWith("data:"));
          if (!line) continue;
          const data = JSON.parse(line.slice(5).trim());
          output = data.message || data.error || data.partial || output;
          stream.push(data);
        }
        if (done) break;
      }
      stream.end(output);
    } catch (err) {
      const message = {
        role: "assistant", content: [], api: model.api, provider: model.provider,
        model: model.id,
        usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
        stopReason: options?.signal?.aborted ? "aborted" : "error",
        errorMessage: String(err),
        timestamp: Date.now(),
      };
      stream.push({ type: "error", reason: message.stopReason, error: message });
      stream.end(message);
    }
  }

  

  _onEvent(event) {
    const log = (t) => this.ui?.log?.(t);
    switch (event.type) {
      case "agent_start":
        this.state.phase = "plan";
        break;
      case "message_end":
        if (event.message?.role === "assistant") {
          for (const part of event.message.content || []) {
            if (part.type === "text" && part.text) log("agent: " + part.text.slice(0, 300));
          }
        }
        break;
      case "tool_execution_start":
        log(`tool: ${event.toolName}`);
        break;
      case "tool_execution_end":
        if (event.isError) {
          log(`tool ${event.toolName} error: ` + JSON.stringify(event.result).slice(0, 200));
        }
        break;
      case "agent_end":
        this.state.phase = "done";
        break;
    }
  }

  

  _buildTools() {
    const self = this;

    
    
    
    
    
    const compileFn = (src, label) => {
      const tag = label ? `//# sourceURL=vfs/${label}` : "";
      const s0 = String(src)
        .replace(/^```[a-z]*\s*/i, "").replace(/```\s*$/, "")
        .trim().replace(/;+$/, "");
      const s = s0;
      try {
        const inner = new Function(`"use strict";\nreturn (${s});\n${tag}`)();
        if (typeof inner === "function") return inner;
      } catch {  }
      const m = s.match(/function\s+([A-Za-z_$][\w$]*)\s*\(/);
      if (m) {
        const inner = new Function(`"use strict";\n${s}\nreturn ${m[1]};\n${tag}`)();
        if (typeof inner === "function") return inner;
      }
      const am = s.match(/(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function|\()/);
      if (am) {
        const inner = new Function(`"use strict";\n${s}\nreturn ${am[1]};\n${tag}`)();
        if (typeof inner === "function") return inner;
      }
      throw new Error(
        "source did not evaluate to a function — write `function(args){...}` or `(args) => {...}`. " +
        `Got (${s0.length} chars): ` + s0.slice(0, 80).replace(/\n/g, " "));
    };

    const withStack = (err, where) => {
        const stack = String(err?.stack || err?.message || "").split("\n")
        .filter((l) => l.includes("vfs/") || l.trim().startsWith("at ") || l.includes("Error"))
        .slice(0, 4).join(" | ");
      return new Error(
        `${where} failed: ${String(err?.message || err).slice(0, 150)}. ` +
        `Stack: ${stack.slice(0, 300)}. ` +
        `Fix by rewriting that function (write_corruptor / write_probe / plan_label_set gen or render), then retry run_pipeline.`);
    };

    
    
    const ok = (text, details) => ({
      content: [{ type: "text", text }],
      details: details ?? {},
    });

    const wrap = (t) => {
      const exec = t.execute;
      t.execute = async (id, args) => {
        self.ui?.toolStart?.(t.name);
        try {
          const r = await exec(id, args);
          const txt = r?.content?.[0]?.text ?? "";
          self.ui?.toolEnd?.(t.name, true, txt.slice(0, 160));
          return r;
        } catch (e) {
          self.ui?.toolEnd?.(t.name, false, String(e?.message || e).slice(0, 160));
          throw e;
        }
      };
      return t;
    };

    return [
      {
        name: "plan_label_set",
        label: "Plan label set & objective",
        description: "Commit to the training plan: labels, training kind, corruptor names, and pipeline settings. Must be called first.",
        parameters: Type.Object({
          goal: Type.String({ description: "User goal (full text)" }),
          labels: Type.Array(Type.String(), { description: "Class names (dense_mse) or ['quality'] (score_pair)" }),
          training_kind: Type.String({ description: "'dense_mse' or 'score_pair'" }),
          corruptors: Type.Array(Type.String(), { description: "Names of corruptors to write next" }),
          gen: Type.Optional(Type.String({ description: "Optional data generator function source: function(rng, label) -> sample. Omit to use built-in digits/docs/scenes." })),
          render: Type.Optional(Type.String({ description: "Optional render function for non-image samples (score_pair): function(sample) -> Float32Array image" })),
          img_w: Type.Optional(Type.Number()),
          img_h: Type.Optional(Type.Number()),
          patch: Type.Optional(Type.Number()),
          in_channels: Type.Optional(Type.Number()),
          out_channels: Type.Optional(Type.Number()),
          n_per_class: Type.Optional(Type.Number({ description: "Samples per class (default 60)" })),
          n_pairs: Type.Optional(Type.Number({ description: "Pairs for score_pair (default 120)" })),
          epochs: Type.Optional(Type.Number({ description: "Training epochs (default 80 dense / 30 pair)" })),
          max_iterations: Type.Optional(Type.Number({ description: "Adversarial training rounds (default 3). Each round mines hard cases from held-out errors and retrains with stronger corruption." })),
          lr: Type.Optional(Type.Number()),
          seed: Type.Optional(Type.Number()),
          corrupt_severity: Type.Optional(Type.Number({ description: "Training corruption severity 0..1 (default 0.15)" })),
          notes: Type.Optional(Type.String()),
        }),
        execute: async (_toolCallId, args) => {
          const plan = { ...args };
          if (args.gen) {
            try { plan.gen = compileFn(args.gen, "gen.js"); }
            catch (err) { throw new Error("gen compile failed: " + String(err).slice(0, 150)); }
          }
          if (args.render) {
            try { plan.render = compileFn(args.render, "render.js"); }
            catch (err) { throw new Error("render compile failed: " + String(err).slice(0, 150)); }
          }
          self.state.plan = plan;
          self.ui?.log?.(`plan: kind=${plan.training_kind}, labels=${(plan.labels || []).join(",")}, corruptors=${(plan.corruptors || []).join(",")}`);
          return ok(`Plan accepted: kind=${plan.training_kind}, labels=${(plan.labels || []).join(",")}, corruptors=${(plan.corruptors || []).join(",")}. Now write the corruptor.`, { plan: { kind: plan.training_kind, labels: plan.labels, corruptors: plan.corruptors } });
        },
      },
      {
        name: "write_corruptor",
        label: "Write corruptor",
        description: "Write the corruption function source. Contract: function(sample, severity, rng, label, kit) -> corrupted sample, SAME shape as input. rng() is uniform [0,1). kit has image helpers (noise, gauss, translate, rotate, occlude, dropout, clamp01...) — use them instead of raw pixel loops.",
        parameters: Type.Object({
          name: Type.String({ description: "Corruptor name (from the plan)" }),
          code: Type.String({ description: "Function source, e.g. 'function(sample, severity, rng, label) { ... return out; }'" }),
        }),
        execute: async (_toolCallId, args) => {
          let fn;
          try { fn = compileFn(args.code, "corruptor.js"); }
          catch (err) { throw new Error("compile failed: " + String(err).slice(0, 150)); }
          const r = self.engine.rng(42);
          const plan = self.state.plan || {};
          const gen = plan.gen || ((rng2, label) => self.engine.genDigit(label ?? 0, rng2));
          let sample;
          try { sample = gen(r, 0, kit); }
          catch (err) {
            throw new Error("gen crashed in smoke test: " + String(err).slice(0, 150) +
              " — if gen uses kit helpers, declare the kit parameter: function(rng, label, kit)");
          }
          let out;
          try { out = fn(sample, 0.5, r, 0, kit); }
          catch (err) { throw withStack(err, "corruptor smoke test"); }
          try {
            const img = ArrayBuffer.isView(out) ? out : (plan.render ? plan.render(out, kit) : out);
            const W = plan.img_w ?? self.engine.DOC_W, H = plan.img_h ?? self.engine.DOC_H;
            const P = plan.patch ?? self.engine.DOC_PATCH;
            self.pkg.patchifyImage(ArrayBuffer.isView(img) ? img : out, W, H, P, 1);
          } catch (err) {
            throw new Error("corruptor output failed the render/patchify path: " + String(err).slice(0, 120));
          }
          const sameShape = out && typeof out === typeof sample &&
            (ArrayBuffer.isView(sample) ? out.length === sample.length : true);
          if (!sameShape) {
            throw new Error(`corruptor returned wrong shape: in ${sample?.length}, out ${out && out.length}`);
          }
          self.vfs.write("scratch/corruptor-" + normalizePath(args.name) + ".js", args.code);
          if (self.state.plan) self.state.plan._corruptor = fn;
          self.ui?.log?.(`corruptor "${args.name}" written + smoke-tested`);
          return ok(`Corruptor "${args.name}" compiled and smoke-tested (shape preserved). Now write a probe (optional) and run the pipeline.`, { name: args.name });
        },
      },
      {
        name: "write_probe",
        label: "Write probe",
        description: "Optional: write a probe function for the score card. Contract: function({model, wasm, engine, rng, kit}) -> JSON-serializable result. Prefer fields the UI renders natively: losses:[{epoch,value}], values+labels, matrix (+rowLabels/colLabels), points, scores, cases:[{img,w,h,label,predicted,ok}].",
        parameters: Type.Object({
          code: Type.String({ description: "Function source" }),
        }),
        execute: async (_toolCallId, args) => {
          let fn;
          try { fn = compileFn(args.code, "probe.js"); }
          catch (err) { throw new Error("compile failed: " + String(err).slice(0, 150)); }
          self.vfs.write("scratch/probe.js", args.code);
          if (self.state.plan) self.state.plan._probe = fn;
          self.ui?.log?.("probe written");
          return ok("Probe written. Run the pipeline next.", {});
        },
      },
      {
        name: "run_pipeline",
        label: "Run training pipeline",
        description: "Run the real training: corruptor -> dataset -> wasm trainer -> held-out eval -> probe. Requires plan_label_set + write_corruptor first. Returns losses, accuracy, probe output.",
        parameters: Type.Object({}),
        execute: async () => {
          const plan = self.state.plan;
          if (!plan) throw new Error("no plan — call plan_label_set first");
          if (!plan._corruptor) throw new Error("no corruptor — call write_corruptor first");
          let result;
          try {
            result = await runPipeline(self.pkg, self.engine, plan, {
              log: (t) => self.ui?.log?.(t),
              progress: (e, epochs, loss) => self.ui?.progress?.(e, epochs, loss),
              iteration: (n, total, t) => self.ui?.iteration?.(n, total, t),
              viz: (e) => self.ui?.viz?.(e),
            });
          } catch (err) {
            throw withStack(err, "pipeline (likely a gen/render/corruptor bug)");
          }
          self.state.result = result;
          if (result.error) throw new Error(result.error);
          const summary = result.metrics.map((m) => `${m.label}: ${m.value}`).join("; ");
          return ok(
            `Pipeline done (${result.kind}, ${result.model.nParams()} params). ${summary}` +
            (result.probe ? ` Probe: ${JSON.stringify(result.probe).slice(0, 200)}` : "") +
            ". Export the model next.",
            { metrics: result.metrics, probe: result.probe ?? null, kind: result.kind }
          );
        },
      },
      {
        name: "export_model",
        label: "Export model",
        description: "Serialize the trained model to PKM1 bytes. The UI turns this into a download link. Requires run_pipeline to have succeeded.",
        parameters: Type.Object({
          name: Type.String({ description: "Model file name (without .pkm)" }),
        }),
        execute: async (_toolCallId, args) => {
          const result = self.state.result;
          if (!result?.model) throw new Error("no trained model — run run_pipeline first");
          const bytes = result.model.save();
          self.state.export = { name: args.name, bytes };
          self.ui?.log?.(`export: ${args.name}.pkm (${bytes.length} bytes, ${result.model.nParams()} params)`);
          return ok(`Exported ${args.name}.pkm: ${bytes.length} bytes, ${result.model.nParams()} params. The download link is in the metrics panel. Finish with a short summary.`, { name: args.name, size: bytes.length });
        },
      },
      {
        name: "ui_spec",
        label: "UI-spec (optional)",
        description: "Declaring a render kind for the live probe canvas. Accepted: 'loss' (needs probe.losses), 'bars' (values), 'heat' (matrix), 'scatter' (points), 'pred' (scores). Purely declarative; the pipeline falls back to a good default if omitted.",
        parameters: Type.Object({
          kind: Type.String({ description: "'loss'|'bars'|'heat'|'scatter'|'pred'" }),
          field: Type.Optional(Type.String({ description: "Which probe field holds the data (default auto-detect)" })),
        }),
        execute: async (_toolCallId, args) => {
          if (!self.state.plan) throw new Error("no plan — call plan_label_set first");
          self.state.plan._uiSpec = { kind: args.kind, field: args.field };
          return ok(`UI spec set to ${args.kind}; probe-driven render active.`, { kind: args.kind });
        },
      },
      {
        name: "list_vfs",
        label: "List workspace",
        description: "List the virtual filesystem (agent-written code) and current plan state.",
        parameters: Type.Object({}),
        execute: async () => ok(JSON.stringify({
          files: self.vfs.snapshot(),
          plan: self.state.plan
            ? { kind: self.state.plan.training_kind, labels: self.state.plan.labels, corruptors: self.state.plan.corruptors }
            : null,
          trained: !!self.state.result?.model,
        }), {}),
      },
    ].map(wrap);
  }

  

  async prompt(text) {
    await this._agent.prompt(text);
  }
  abort() { this._agent.abort(); }
  snapshot() {
    return {
      phase: this.state.phase,
      plan: this.state.plan && {
        goal: this.state.plan.goal, kind: this.state.plan.training_kind,
        labels: this.state.plan.labels, corruptors: this.state.plan.corruptors,
      },
      files: this.vfs.snapshot(),
      trained: !!this.state.result?.model,
      export: this.state.export && { name: this.state.export.name, size: this.state.export.bytes.length },
    };
  }
}
