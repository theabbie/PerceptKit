// PerceptKit agent UI — hooks the agent tools into the goal form and
// drives a structured pipeline (corruptor -> trainer -> probes -> export).
import * as pkg from "./pkg-web/percept_wasm.js";
import * as engine from "./engine.js";
import { PerceptKitAgent } from "./agent-core.js";

// log collector (activated by UI to route errors into the page log)
window.__errors = [];

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// state / helpers
// ---------------------------------------------------------------------------
let agent = null;
let currentModel = null;
let currentProbes = [];
let lastProbes = [];
let exportLink = null;
let agentErrorLog = [];
let exportLink = null;

function phase(name) { $("agPhase").textContent = name; }
function logLine(kind, text) {
  const pane = $("agLog");
  const line = document.createElement("div");
  line.className = "log kind-" + kind;
  line.textContent = text;
  pane.appendChild(line);
  pane.scrollTop = pane.scrollHeight;
}

function reviewHtml(blocks, exportLinkOpt) {
  const pane = $("agReview");
  pane.innerHTML = '<label class="ag-report-label">objective & metrics</label>' + blocks.map((b) => `
    <p style="margin:2px 0 0"><span style="color:${b.color || "var(--acc)"}">${b.label}:</span> ${b.value}</p>`).join("")
    + (exportLinkOpt ? ('<p style="margin-left:8px"><a class="ag-download" href="' + exportLinkOpt + '" download><span>download checkpoint</span></a></p>') : "");
}

// collect per-run metrics into a single table
function aggregateLosses(errs, losses, probes) {
  const { training_kind, corruptors } = agent.state.plan || {};
  const fmt = (l) => (l.error ? "error: " + l.error : "loss " + (l.ok ? "ok" : "unk"));
  return {
    roles: losses.map(fmt).join(" | "),
    kind: training_kind || "auto",
    corruptors: (corruptors || []).join(","),
  };
}

// ---------------------------------------------------------------------------
// structured tool kinds — what the agent generates inside the VFS
// ---------------------------------------------------------------------------
const K_KINDS = {
  exemplar: (pkg, engine, args) => {
    return { ok: true, result: { kind: "exemplary" } };
  },
  corruptor: (pkg, engine, args) => {
    return { ok: true, result: { kind: "corruptor", input: args } };
  },
  ci: (pkg, engine, args) => ({ ok: true, result: { kind: "ci" } }),
  col: (pkg, engine, args) => ({ ok: true, result: { kind: "col" } }),
  score: (pkg, engine, args) => ({ ok: true, result: { kind: "score" } }),
};

// ---------------------------------------------------------------------------
// Generated tools — agent.loaded mapped to runPipeline roles
// ---------------------------------------------------------------------------
function sectionRoles(vfs) {
  const seg = {};
  for (const path of vfs.list()) {
    if (path.includes("scratch/") || path.startsWith("models/")) {
      const role = (path.includes("corruptor") || path.includes("gen_") || path.includes("corrupt")) ? "corruptor"
        : (path.includes("ci") || path.includes("cls") || path.includes("dense")) ? "trainer"
        : (path.includes("score") || path.includes("pair")) ? "scorer"
        : (path.includes("probe") || path.includes("eval")) ? "probe"
        : "misc";
      seg[path] = role;
    }
  }
  return seg;
}

async function runStructuredPipeline(agent, gql) {
  // Compile generated snippets into a runner
  const seg = sectionRoles(agent.vfs);
  const roles = Object.entries(seg).filter(([, role]) => ["corruptor", "trainer", "scorer", "probe"].includes(role));
  if (!roles.length) {
    return { error: "no structured roles found in VFS — use write_tool to store corruptor/scorer/probe" };
  }
  logLine("ok", "pipeline roles: " + roles.map(([, r]) => roleName(r)).join(" | "));

  const losses = [];
  let finalModel = null;
  let probes = [];

  for (const [path, role] of Object.entries(seg)) {
    const src = agent.vfs.read(path);
    try {
      const sandbox = {
        percept: agent.pkg,
        engine: agent.engine,
        core: { renderDoc: agent.engine.renderDoc, corruptDoc: agent.engine.corruptDoc, colorize: agent.engine.colorize, maskRegion: agent.engine.maskRegion },
        vfs_read: (p) => agent.vfs.read(p),
        vfs_write: (p, c) => agent.vfs.write(p, c),
        ci: {},
      };
      const fn = new Function("sandbox", "agent", "pkg", "engine", "vfs_read", "vfs_write", "return (" + src + ")");
      const out = fn(sandbox, agent, agent.pkg, agent.engine, sandbox.vfs_read, sandbox.vfs_write);
      losses.push({ path, role, ok: !(out && out.error) });
      if (out && out.error) losses.push({ path, role, error: String(out.error).slice(0, 200) });
      if (out && out.model) {
        finalModel = out.model;
        probes.push({ role, kind: "model", metric: out.probes ?? null });
      }
    } catch (err) {
      losses.push({ path, role, error: String(err).slice(0, 200) });
    }
  }
  return { ok: true, losses, probes, finalModel };
}

function roleName(role) { return role; }

// ---------------------------------------------------------------------------
// model export + download link
// ---------------------------------------------------------------------------
function exportLinkFor(model, name) {
  const bytes = model.save();
  const blob = new Blob([bytes], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  url.name = name;
  return url;
}

// ---------------------------------------------------------------------------
// agent lifecycle
// ---------------------------------------------------------------------------
function build() {
  if (agent) return agent;
  agent = new PerceptKitAgent({ pkg, engine, ui: { log: (t) => logLine("ok", t) } });
  return agent;
}

function hooks() {
  if (typeof document === "undefined") return;
  $("agRun").addEventListener("click", async () => {
    const goal = $("agGoal").value.trim();
    if (!goal) { logLine("bad", "empty goal — describe your perception task"); return; }
    $("agRun").disabled = true;
    phase("plan");
    try {
      const g = build();
      await g.prompt(goal);
      const log = await runStructuredPipeline(g, agent.state);
      lastProbes = log.probes || [];
      currentModel = log.finalModel;
      const blocks = [
        { label: "roles", value: log.ok ? "ok" : "error", color: log.ok ? "var(--good)" : "var(--bad)" },
      ];
      if (log.finalModel) {
        blocks.push({ label: "params", value: "(" + currentModel.nParams() + ")" });
        const url = exportLinkFor(currentModel, "model-" + Date.now() + ".pkm");
        blocks.push({ label: "download", value: "checkpoint", link: url });
      }
      reviewHtml(blocks, exportLink || null);
      phase("done");
    } catch (err) {
      logLine("bad", "error: " + err.message);
      phase("idle");
    } finally {
      $("agRun").disabled = false;
    }
  });

  $("agAB").addEventListener("click", () => {
    if (agent) { agent.abort(); phase("idle"); }
  });
}

// -------------------------------------------------------------------------------
// agent-ui is loaded after index.html is parsed; bind once everything exists.
// Preserve the existing timeline: register run/abort handlers at import end.
// -------------------------------------------------------------------------------
if (typeof document !== "undefined") {
  hooks();
}
