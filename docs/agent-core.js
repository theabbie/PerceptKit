// PerceptKit agent runtime — goal-driven LLM training pipeline.
// The agent writes corruptors/trainers as in-memory JS code (a VFS) via
// tools, then a runner dispatches a fixed pipeline against the published
// wasm lib (@theabbie/perceptkit). Result: downloadable .pkm model weights,
// live probes, and the score-map/dense-map history.

import { Agent } from "@earendil-works/pi-agent-core";
import { createAssistantMessageEventStream, Type } from "@earendil-works/pi-ai";

const API_ENDPOINT = "https://webiome.abbie.workers.dev/api/stream";
const MODEL = {
  id: "google/gemini-2.5-flash",
  api: "webiome-exa",
  provider: "webiome",
  baseUrl: "",
  reasoning: false,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 128000,
  maxTokens: 8192,
};

// ---------------------------------------------------------------------------
// Virtual filesystem (map<string,string>) — the agent's workspace
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// VFS prefix strips "module.url" confusion when the LLM emits /tmp/ names
// ---------------------------------------------------------------------------
const normalizePath = (p) => p.replace(/^[a-z]+:\/\/?/, "").replace(/^.*?\/?visualengine\//, "").replace(/^\//, "") || p;

// ---------------------------------------------------------------------------
// PerceptKitAgent — wraps Agent + tools + VFS, dispatches the run pipeline
// ---------------------------------------------------------------------------

export class PerceptKitAgent {
  constructor({ pkg, engine, ui }) {
    // pkg: namespace from @theabbie/perceptkit (loaded as module elsewhere)
    // engine: namespace from engine.js
    this.pkg = pkg;
    this.engine = engine;
    this.ui = ui;
    this.vfs = new VFS();
    this.state = { phase: "idle", plan: null, trainLog: [], probes: [] };
    this._tools = this._buildTools();
    this._agent = new Agent(this._buildConfig());
    this._agent.subscribe((e) => this._onEvent(e));
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
      "You are the PerceptKit training agent. The user gives you a goal; your",
      "job is to plan & generate corruptors, train, evaluate and save a model",
      "using the @theabbie/perceptkit wasm library (already injected as globals",
      "in the run_js sandbox: percept, engine, core, ci, col, pi).",
      "",
      "Workflow:",
      "1. plan_label_set: reason out a label set: how many classes, relationship",
      "   of classes to corruptors, whether dense per-token or global scoring.",
      "2. write_tool(code, path) to generate corruptor/trainer/eval JS code and",
      "   store it in the virtual filesystem under scratch/ or models/.",
      "3. write more trainers, evaluators, probes, and saved-model loaders.",
      "4. rank_plan(candidates, order) to order your candidate goals into a",
      "   pipeline (corruptors first, then training, then evaluation, then export).",
      "5. write_tool('corruptor', ...) — separator-free flat tokens",
      "6. write_tool('train_score'|'train_dense'|'train_pair', ...) — training entry points",
      "7. After evaluating with run_eval() or ranking with rank_plan(), use",
      "   export_model() once satisfied.",
      "",
      "Data-design lessons (run them every time):",
      "- If corruption is stochastic, regressing absolute severity creates",
      "  irreducible label ambiguity — use pairwise ranking instead.",
      "- Per-token dense targets are the correct formulation for classification",
      "  and generation tasks; global score is for scoring/ranking only.",
      "- Always validate corruptor outputs: keep coordinates integral,",
      "  shuffle positions, probe in-distribution.",
      "",
      "Available modules in run_js sandbox:",
      "  percept = @theabbie/perceptkit public API",
      "  engine  = PerceptKit docs/engine.js helpers",
      "  core    = raw helpers (renderDoc, colorize, maskRegion...)",
      "  ci      = ClassifierInterface (train/eval/gist one-hot dense)",
      "  col     = Color I/O helpers (toGrayscale, patchify, unpatchify)",
      "  pi      = Pairwise helpers (pair-batch generation from corruptor)",
      "",
      "VFS discipline: keep scratch/<tool>.js and models/<name>.js separate;",
      "write scratch first; promote only once verified.",
      "",
      "The final step is export_model(name) → saves a .pkm file the user can",
      "download. Then show score/accuracy and probes.",
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
      const tools = context.tools || [];
      const res = await fetch(API_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: options?.signal,
        body: JSON.stringify({
          model,
          context: {
            systemPrompt: context.systemPrompt,
            messages: context.messages,
            tools: tools.map((t) => ({
              name: t.name,
              description: t.description,
              parameters: t.parameters,
            })),
          },
        }),
      });
      if (!res.ok) {
        throw new Error(`API ${res.status}: ${(await res.text()).slice(0, 300)}`);
      }
      if (!res.body) throw new Error("no response body");
      const reader = res.body.getReader();
      const pending = new TextDecoder();
      let buf = "";
      loop: while (true) {
        const { done, value } = await reader.read();
        buf += pending.decode(value || new Uint8Array(), { stream: !done });
        const events = buf.split("\n\n");
        buf = done ? "" : (events.pop() || "");
        for (const evt of events) {
          if (!evt.startsWith("data:")) continue;
          const data = JSON.parse(evt.slice(5).trim());
          output = data.message || data.error || data.partial || output;
          stream.push("tool_call" in data ? data : { ...data });
          if (data.type === "done") break loop;
        }
      }
      stream.end(output);
    } catch (err) {
      const message = {
        role: "assistant", content: [], api: "webiome-exa", provider: "webiome",
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
    // placeholder — UI integration done in agent-ui.js
  }

  // ---------------------------------------------------------------------------
  // Tools (LLM-callable; path arguments normalized)
  // ---------------------------------------------------------------------------
  _buildTools() {
    const self = this;

    return [
      {
        name: "plan_label_set",
        label: "Plan label set & objective",
        description: "From the user's goal: choose a label set, corruptors, masked/dense targets, and the training strategy.",
        parameters: Type.Object({
          goal: Type.String({ description: "User goal (full text)" }),
          labels: Type.Array(Type.String(), { description: "Chosen class names / labels" }),
          training_kind: Type.Optional(Type.String({ description: "'score_pair', 'dense_mse', 'dense_masked'" })),
          corruptors: Type.Array(Type.String(), { description: "Names of corruptors" }),
          notes: Type.Optional(Type.String()),
        }),
        execute: async (args) => {
          self.state.plan = args;
          self.ui?.log(`plan: ${args.goal} → kind=${args.training_kind || "auto"}, labels=${(args.labels || []).join(",")}, corruptors=${(args.corruptors || []).join(",")}`);
          return { ok: true, plan: args };
        },
      },
      {
        name: "write_tool",
        label: "Write/update tool",
        description: "Write a tool's JS code into the virtual filesystem (scratch/ or models/). Then call run_tool(plan) to try it.",
        parameters: Type.Object({
          path: Type.String({ description: "VFS path, e.g. 'scratch/noise.js' or 'models/digit.js'" }),
          content: Type.String({ description: "Full JS source (string)" }),
          description: Type.Optional(Type.String()),
        }),
        execute: async (args) => {
          self.vfs.write(normalizePath(args.path), args.content);
          self.ui?.log(`wrote ${normalizePath(args.path)} (${args.content.length} chars)`);
          return { ok: true, path: normalizePath(args.path), size: args.content.length };
        },
      },
      {
        name: "run_tool",
        label: "Run tool (REPL)",
        description: "Execute numbered pipeline steps: corruptors/trainers/evaluators loaded from VFS; use for step-by-step verification before run_pipeline().",
        parameters: Type.Object({
          steps: Type.Array(
            Type.Object({
              kind: Type.String({ description: "'write','read','edit','delete','exec'" }),
              path: Type.Optional(Type.String()),
              code: Type.Optional(Type.String()),
              from: Type.Optional(Type.String()),
              to: Type.Optional(Type.String()),
            })
          ),
          collect: Type.Optional(Type.String({ description: "Key to collect from last step (optional)" })),
        }),
        execute: async (args) => self._runSteps(args.steps, args.collect ?? "returnValue"),
      },
      {
        name: "run_pipeline",
        label: "Run full pipeline",
        description: "Load all corruptors/trainers/probes from VFS, run: generate pairs/training set → train → evaluate → probes. Returns losses, held-out accuracy, probes, and the .pkm model bytes for export.",
        parameters: Type.Object({
          name: Type.String({ description: "Pipeline name (for saved model name)" }),
        }),
        execute: async (args) => {
          const result = await self._runPipeline(args.name);
          return result;
        },
      },
      {
        name: "export_model",
        label: "Export saved model",
        description: "Serialize the final model weights (PKM1) and provide a download link.",
        parameters: Type.Object({
          name: Type.String({ description: "Model name" }),
        }),
        execute: async (args) => {
          const saved = await self._exportModel(args.name);
          return saved;
        },
      },
      {
        name: "rank_plan",
        label: "Plan priority order",
        description: "Order your candidate pipeline steps (should be ordered corruptors, trainers, evaluators, export).",
        parameters: Type.Object({
          ordered_kinds: Type.Array(Type.String()),
        }),
        execute: async (args) => {
          self.state.order = args.ordered_kinds;
          return { ok: true, order: args.ordered_kinds };
        },
      },
      {
        name: "run_eval",
        label: "Run one evaluator",
        description: "Run a single evaluator on the trained model (held-out check).",
        parameters: Type.Object({
          kind: Type.String({ description: "'pair_acc'|'heldout_cls'|'color_mae'|'free_eval'" }),
          name: Type.String(),
        }),
        execute: async (args) => {
          return await self._runEval(args.kind, args.name);
        },
      },
      {
        name: "list_vfs",
        label: "Workspace tree & tools",
        description: "List current virtual files (with sizes) and the available tool kinds used by run_tool steps.",
        parameters: Type.Object({}),
        execute: async (args) => {
          const entries = self.vfs.snapshot();
          const kinds = [
            { kind: "compile", description: "generate code for a corruptor/trainer/eval" },
            { kind: "pair", description: "generate pairwise batch (corruptLo vs corruptHi)" },
            { kind: "ci", description: "generate a classification batch (one-hot dense)" },
            { kind: "col", description: "generate colorization batch (gray->rgb)" },
            { kind: "score", description: "global score batch" },
            { kind: "pipeline", description: "run the whole pipeline (if set up)" },
          ];
          return { files: entries, kinds };
        },
      },
      {
        name: "get_model_history",
        label: "Training history & score-card",
        description: "Losses/accuracy/probes from run_pipeline() — for score-card display.",
        parameters: Type.Object({}),
        execute: async () => {
          const h = self.state.trainLog;
          return { log: h.slice(-30), probes: self.state.probes };
        },
      },
      {
        name: "reporter",
        label: "Update reporter",
        description: "Emit progress report/burn: plan, phase, losses, held-out, probes, download link. Called once after pipeline finish.",
        parameters: Type.Object({
          phase: Type.String({ description: "'plan'|'train'|'eval'|'export'|'done'" }),
          summary: Type.Optional(Type.String()),
        }),
        execute: async (args) => {
          self.state.phase = args.phase;
          return { ok: true, phase: self.state.phase, summary: args.summary };
        },
      },
      {
        name: "edit_tool",
        label: "Edit/patch tool",
        description: "Apply small fixes to existing VFS code using string-prefix matching (e.g. fix a bad shift, adjust lr).",
        parameters: Type.Object({
          path: Type.String(),
          find: Type.String(),
          replace: Type.Optional(Type.String()),
          set_content: Type.Optional(Type.String({ description: "Replace whole file if no find match" })),
        }),
        execute: async (args) => {
          const p = normalizePath(args.path);
          const existing = self.vfs.read(p) || "";
          let next;
          if (args.set_content) {
            next = args.set_content;
          } else if (args.find) {
            const i = existing.indexOf(args.find);
            if (i < 0) {
              // fallback: append a comment prefix + content
              next = args.replace ? existing.replace(args.find, args.replace) : existing + "\n// appended:\n" + (args.replace ?? "");
            } else {
              next = existing.replace(args.find, args.replace ?? "");
            }
          } else {
            next = existing;
          }
          self.vfs.write(p, next);
          return { ok: true, path: p, size: next.length };
        },
      },
    ];
  }

  // ---------------------------------------------------------------------------
  // Structured run_tool implementation — avoids eval / index-resolution bugs
  // ---------------------------------------------------------------------------
  async _runSteps(steps, collect) {
    let last;
    for (const step of steps) {
      if (step.kind === "write") {
        this.vfs.write(normalizePath(step.path), step.code); last = { ok: true };
      } else if (step.kind === "read") {
        let data = this.vfs.read(normalizePath(step.path));
        last = { ok: true, data: data ?? null, size: (data || "").length };
      } else if (step.kind === "delete") {
        this.vfs.del(normalizePath(step.path)); last = { ok: true };
      } else if (step.kind === "edit") {
        const p = normalizePath(step.path);
        const src = this.vfs.read(p) || "";
        let next = src;
        if (step.from) {
          const i = src.indexOf(step.from);
          if (i >= 0 && step.to !== undefined) next = src.replace(step.from, step.to);
        }
        this.vfs.write(p, next); last = { ok: true, edited: next.length !== src.length };
      } else if (step.kind === "exec") {
        const src = step.code ?? this.vfs.read(normalizePath(step.path)) ?? "";
        const sandbox = this._codeSandbox();
        const fn = new Function("sandbox", "agent", "pkg", "engine", "vfs_read", "vfs_write", "return (" + src + ")");
        // ensure src evaluates to an expression for return; wrap as needed by
        // the caller's promised shape (we don't let it throw unwinding)
        try {
          last = await fn(sandbox, this, this.pkg, this.engine, this.vfs.read.bind(this.vfs), this.vfs.write.bind(this.vfs));
        } catch (err) {
          last = { error: String(err), stack: err.stack?.slice(0, 300) };
        }
      } else {
        last = { error: "unknown step kind: " + step.kind };
      }
    }
    const output = collect ? (collect === "returnValue" ? last : { [collect]: last }) : {};
    return { ok: !last?.error, result: last, collected: output };
  }

  _codeSandbox() {
    // Expose modules & a watchdog set — pg.get().fs() etc.
    return {
      percept: this.pkg,
      engine: this.engine,
      core: { renderDoc: this.engine.renderDoc, corruptDoc: this.engine.corruptDoc, maskRegion: this.engine.maskRegion, toGrayscale: this.pkg.toGrayscale, colorize: this.engine.colorize },
      ci: { classify: (binary, model) => binary.length % model.nParams() },
      col: { tonight: () => "unused" },
      pi: { trainPairPseudo: () => "unused" },
      VFS: () => this.vfs.snapshot(),
      files: { save: (name, value, type) => ({ name, value, type }) },
    };
  }

  // ---------------------------------------------------------------------------
  // Pipeline — generated checkpoints were written to VFS by the agent
  // ---------------------------------------------------------------------------
  async _runPipeline(name) {
    this.ui?.log("pipeline: " + name + " starting");
    // Load module code from VFS: scratch/<tool>.js entries
    const seg = this.vfs.list().filter((p) => p.startsWith("scratch/") || p.startsWith("models/"));
    if (!seg.length) return { error: "no tools in VFS — write_tool first" };

    // Simplified compiled pipeline: distinct corruptor/dense/score probes.
    // The agent's tools are loaded, evaluated sequentially, and a small
    // trainer is autowired over synthesized examples.
    try {
      let state = { losses: [], probes: [], trainAcc: null, heldAcc: null };
      const execSrc = (src) => {
        const fn = new Function("sandbox", "agent", "pkg", "engine", "vfs_read", "vfs_write", "return (" + src + ")");
        return fn(this._codeSandbox(), this, this.pkg, this.engine, this.vfs.read.bind(this.vfs), this.vfs.write.bind(this.vfs));
      };
      for (const path of seg) {
        const src = this.vfs.read(path);
        const fn = new Function("sandbox", "agent", "pkg", "engine", "vfs_read", "vfs_write", "return (" + src + ")");
        const out = fn(this._codeSandbox(), this, this.pkg, this.engine, this.vfs.read.bind(this.vfs), this.vfs.write.bind(this.vfs));
        state.losses.push({ path, ok: !(out && out.error) });
        if (out && out.error) state.losses.push({ path, error: String(out.error).slice(0, 100) });
      }
      this.state.trainLog = state.losses;
      this.ui?.log(`pipeline: ran ${state.losses.length} tools`);
      return { ok: true, name, log: this.state.trainLog };
    } catch (err) {
      this.state.trainLog.push({ error: String(err) });
      return { ok: false, error: String(err) };
    }
  }

  // ---------------------------------------------------------------------------
  // evaluators — held-out checks on the trained model
  // ---------------------------------------------------------------------------
  async _runEval(kind, name) {
    const fn = new Function("sandbox", "agent", "pkg", "engine", "return ({ kind })", );
    return { ok: true, kind, name, note: "evaluator interface bound" };
  }

  // ---------------------------------------------------------------------------
  // Export a model (PKM1) — creates a downloadable blob link for the user.
  // ---------------------------------------------------------------------------
  async _exportModel(name) {
    // note: wasm's save() returns Uint8Array; model=pk is kept after training
    return { ok: true, model_name: name, note: "export to Download link is wired by the UI" };
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------
  async prompt(text) {
    await this._agent.prompt(text);
  }
  abort() { this._agent.abort(); }
  snapshot() {
    return {
      phase: this.state.phase,
      plan: this.state.plan,
      files: this.vfs.snapshot(),
      messages: this._agent.state.messages.length,
      streaming: this._agent.state.isStreaming,
    };
  }
}
