// PerceptKit agent UI — binds the agent panel to PerceptKitAgent.
// The LLM plans via tools; runPipeline (agent-core) does the ML heavy-lifting.
import init, * as pkg from "./pkg-web/percept_wasm.js";
import * as engine from "./engine.js";
import { PerceptKitAgent } from "./agent-core.js";
import { renderProbe } from "./probe-ui.js";

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------
let agent = null;
let currentRun = null; // { controller } for abort

// collect uncaught errors so silent failures surface in the log pane
window.__errors = [];
window.addEventListener("error", (e) => {
  window.__errors.push(String(e.message));
  logLine("bad", "uncaught: " + e.message);
});
window.addEventListener("unhandledrejection", (e) => {
  window.__errors.push(String(e.reason));
  logLine("bad", "unhandled rejection: " + String(e.reason).slice(0, 200));
});

// ---------------------------------------------------------------------------
// log / review panes
// ---------------------------------------------------------------------------
function phase(name) { $("agPhase").textContent = name; }

function logLine(kind, text) {
  const pane = $("agLog");
  if (!pane) return;
  const line = document.createElement("div");
  line.className = "log kind-" + kind;
  line.textContent = text;
  pane.appendChild(line);
  pane.scrollTop = pane.scrollHeight;
}

// live loss curve on probe canvas
function drawLossCurve(losses) {
  const canvas = $("agProbeCanvas");
  renderProbe(canvas, { losses }, null, engine, pkg);
}

function reviewHtml(blocks, download) {
  const pane = $("agReview");
  let html = '<label class="ag-report-label">objective & metrics</label>';
  html += blocks.map((b) =>
    `<p style="margin:2px 0"><span style="color:${b.color || "var(--acc)"}">${b.label}:</span> ${b.value}</p>`
  ).join("");
  if (download) {
    html += `<p style="margin:10px 0 4px"><a class="ag-download" href="${download.url}" download="${download.name}"><span>download ${download.name}</span></a></p>`;
  }
  pane.innerHTML = html;
}

// ---------------------------------------------------------------------------
// agent lifecycle
// ---------------------------------------------------------------------------
async function build() {
  if (agent) return agent;
  await init(); // idempotent; app.js also calls it
  agent = new PerceptKitAgent({
    pkg, engine,
    ui: {
      log: (t) => logLine("ok", t),
      progress: (e, epochs, loss) => {
        phase(`train ${e + 1}/${epochs} (loss ${loss.toFixed(4)})`);
      },
      iteration: (n, total, text) => {
        $("agIter").textContent = `round ${n}/${total}`;
        phase(`round ${n}/${total}`);
        logLine("ok", text);
      },
      viz: (event) => {
        // event: { type: "update"|"probe"|"adversarial", ... }
        try {
          if (event.type === "update" && event.losses) {
            drawLossCurve(event.losses);
            $("agStats").textContent = "training loss curve";
          } else if (event.type === "update" && event.message) {
            $("agStats").textContent = event.message;
          } else if (event.type === "probe") {
            renderProbe($("agProbeCanvas"), event.probe, event.model, engine, pkg);
            $("agStats").textContent = "probe render (" + JSON.stringify(Object.keys(event.probe || {})).slice(0, 200) + ")";
          } else if (event.type === "adversarial") {
            logLine("ok", "adversarial: " + event.message);
            $("agStats").textContent = event.message;
          }
        } catch (err) {
          $("agStats").textContent = "viz error: " + String(err).slice(0, 60);
        }
      },
    },
  });
  window.__agent = agent; // debugging handle
  window.abortLast = () => agent?.abort();
  return agent;
}

// ---------------------------------------------------------------------------
// run handler
// ---------------------------------------------------------------------------
async function onRun() {
  const goal = $("agGoal").value.trim();
  if (!goal) { logLine("bad", "empty goal — describe your perception task"); return; }

  $("agRun").disabled = true;
  phase("starting");
  logLine("ok", "goal: " + goal);
  try {
    const g = await build();
    // fresh agent per run keeps runs independent (memory of past runs would
    // confuse planning; the VFS/plan are per-run state)
    g._agent.reset();
    g.state = { phase: "idle", plan: null, result: null, export: null };
    g.vfs = new (Object.getPrototypeOf(g.vfs).constructor)();

    await g.prompt(goal);

    // post-run summary
    const snap = g.snapshot();
    const blocks = [];
    if (snap.plan) {
      const labels = snap.plan.labels || [];
      blocks.push({ label: "plan", value: `${snap.plan.kind || "?"} · labels [${labels.join(", ")}]` });
    }
    const res = g.state.result;
    if (!res && !snap.plan) {
      logLine("bad", "agent produced no plan — LLM may have failed; check log above");
    }
    if (res?.metrics) {
      for (const m of res.metrics) blocks.push({ label: m.label, value: m.value });
      if (res.probe) blocks.push({ label: "probe", value: JSON.stringify(res.probe).slice(0, 120) });
    }
    if (snap.export) {
      const blob = new Blob([g.state.export.bytes], { type: "application/octet-stream" });
      const url = URL.createObjectURL(blob);
      blocks.push({
        label: "model",
        value: `${snap.export.name}.pkm · ${snap.export.size} bytes`,
      });
      reviewHtml(blocks, { url, name: snap.export.name + ".pkm" });
    } else if (res?.model) {
      // trained but agent forgot export_model — export anyway
      const bytes = res.model.save();
      const blob = new Blob([bytes], { type: "application/octet-stream" });
      const url = URL.createObjectURL(blob);
      g.state.export = { name: "model", bytes };
      blocks.push({ label: "model", value: `model.pkm · ${bytes.length} bytes (auto-exported)` });
      reviewHtml(blocks, { url, name: "model.pkm" });
    } else {
      if (!blocks.length) blocks.push({ label: "run", value: "no plan produced — see log", color: "var(--bad)" });
      reviewHtml(blocks, null);
    }
    phase("done");
  } catch (err) {
    logLine("bad", "error: " + (err?.message || String(err)).slice(0, 300));
    phase("idle");
  } finally {
    $("agRun").disabled = false;
  }
}

// ---------------------------------------------------------------------------
// bind
// ---------------------------------------------------------------------------
if (typeof document !== "undefined") {
  $("agRun").addEventListener("click", () => void onRun());
  $("agAB").addEventListener("click", () => {
    agent?.abort();
    logLine("bad", "aborted");
    phase("idle");
  });
}
