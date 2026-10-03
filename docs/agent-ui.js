import init, * as pkg from "./pkg-web/percept_wasm.js?v=1.0";
import * as engine from "./engine.js?v=1.0";
import { PerceptKitAgent } from "./agent-core.js?v=1.0";
import { renderProbe } from "./probe-ui.js?v=1.0";
import { mountDemo } from "./demo-ui.js?v=1.0";
import * as viz from "./kit/viz.js?v=1.0";

const $ = (id) => document.getElementById(id);

let agent = null;
let runCards = null;

window.__errors = [];
window.addEventListener("error", (e) => {
  window.__errors.push(String(e.message));
  logLine("bad", "uncaught: " + e.message);
});
window.addEventListener("unhandledrejection", (e) => {
  window.__errors.push(String(e.reason));
  logLine("bad", "unhandled rejection: " + String(e.reason).slice(0, 200));
});

const LANE_OF = {
  plan_label_set: "plan",
  write_corruptor: "code",
  write_probe: "code",
  list_vfs: "code",
  run_pipeline: "train",
  ui_spec: "eval",
  export_model: "export",
};

function status(t) { $("agStatus").textContent = t; }

function logLine(kind, text) {
  const pane = $("agLog");
  if (!pane) return;
  const line = document.createElement("div");
  line.className = "log kind-" + kind;
  line.textContent = text;
  pane.appendChild(line);
  pane.scrollTop = pane.scrollHeight;
}

function clearBoard() {
  const placeholders = {
    plan: "awaiting goal", code: "corruptor + probe fns", train: "dataset → wasm trainer",
    eval: "held-out + probe", export: ".pkm artifact", demo: "train a model, then draw here",
  };
  for (const lane of ["plan", "code", "train", "eval", "export", "demo"]) {
    $("body-" + lane).innerHTML = `<div class="lane-empty">${placeholders[lane]}</div>`;
    $("cnt-" + lane).textContent = "";
  }
}

function bumpCount(lane) {
  const n = $("body-" + lane).querySelectorAll(".card").length;
  $("cnt-" + lane).textContent = n ? `${n}` : "";
}

function addCard(lane, name) {
  const body = $("body-" + lane);
  body.querySelector(".lane-empty")?.remove();
  const el = document.createElement("div");
  el.className = "card running";
  const head = document.createElement("div");
  head.className = "card-head";
  const nm = document.createElement("span");
  nm.className = "card-name";
  nm.textContent = name;
  const chip = document.createElement("span");
  chip.className = "chip running";
  chip.textContent = "running";
  head.append(nm, chip);
  const detail = document.createElement("div");
  detail.className = "card-detail";
  el.append(head, detail);
  body.appendChild(el);
  bumpCount(lane);
  return el;
}

function setCard(el, state, detail) {
  el.classList.remove("running", "ok", "error");
  el.classList.add(state);
  const chip = el.querySelector(".chip");
  chip.className = "chip " + state;
  chip.textContent = state === "ok" ? "done" : state;
  if (detail != null) el.querySelector(".card-detail").textContent = detail;
}

function trainCanvas() {
  if (runCards?.trainCanvas) return runCards.trainCanvas;
  const card = runCards?.train;
  if (!card) return null;
  const c = document.createElement("canvas");
  card.appendChild(c);
  runCards.trainCanvas = c;
  return c;
}

async function build() {
  if (agent) return agent;
  await init();
  agent = new PerceptKitAgent({
    pkg, engine,
    ui: {
      log: (t) => logLine("ok", t),
      progress: (e, epochs, loss) => {
        status(`training epoch ${e + 1}/${epochs} · loss ${loss.toFixed(4)}`);
      },
      iteration: (n, total, text) => {
        $("cnt-train").textContent = `round ${n}/${total}`;
        logLine("ok", text);
      },
      toolStart: (name) => {
        const lane = LANE_OF[name] || "code";
        const card = addCard(lane, name);
        if (name === "run_pipeline") runCards.train = card;
        status(`running ${name}…`);
      },
      toolEnd: (name, ok, summary) => {
        const lane = LANE_OF[name] || "code";
        const cards = $("body-" + lane).querySelectorAll(".card");
        let card = null;
        for (const c of cards) {
          if (c.querySelector(".card-name").textContent === name && c.classList.contains("running")) card = c;
        }
        if (card) setCard(card, ok ? "ok" : "error", summary || "");
        status(ok ? `${name} done` : `${name} failed`);
      },
      viz: (event) => {
        try {
          if (event.type === "update" && event.losses) {
            const c = trainCanvas();
            const vals = event.losses.map((l) => l.value).filter(Number.isFinite);
            if (c && vals.length) viz.line(c, [{ pts: vals, color: viz.C.sky, label: "train loss" }]);
          } else if (event.type === "update" && event.message) {
            if (runCards?.train) runCards.train.querySelector(".card-detail").textContent = event.message;
          } else if (event.type === "probe") {
            const card = addCard("eval", "probe render");
            const c = document.createElement("canvas");
            card.appendChild(c);
            renderProbe(c, event.probe, event.model, engine, pkg);
            setCard(card, "ok", JSON.stringify(Object.keys(event.probe || {})).slice(0, 120));
          } else if (event.type === "adversarial") {
            logLine("ok", "adversarial: " + event.message);
            if (runCards?.train) runCards.train.querySelector(".card-detail").textContent = event.message;
          }
        } catch (err) {
          logLine("bad", "viz error: " + String(err).slice(0, 80));
        }
      },
    },
  });
  window.__agent = agent;
  window.abortLast = () => agent?.abort();
  return agent;
}

let demoHandle = null;

function tryMountDemo(g, model, demoSpec) {
  const body = $("body-demo");
  if (!body || !model || !demoSpec) return;
  body.querySelector(".lane-empty")?.remove();
  try {
    demoHandle = mountDemo({
      container: body, model, wasm: pkg, spec: demoSpec, engine,
      corruptor: g.state.plan?._corruptor ?? null,
    });
    window.__demo = demoHandle;
    $("cnt-demo").textContent = "live";
  } catch (err) {
    logLine("bad", "demo mount failed: " + String(err).slice(0, 120));
  }
}

async function onRun() {
  const goal = $("agGoal").value.trim();
  if (!goal) { logLine("bad", "empty goal — describe your perception task"); return; }

  clearBoard();
  runCards = { train: null, trainCanvas: null };
  demoHandle = null;
  $("agRun").disabled = true;
  status("starting…");
  logLine("ok", "goal: " + goal);
  try {
    const g = await build();
    g._agent.reset();
    g.state = { phase: "idle", plan: null, result: null, export: null };
    g.vfs = new (Object.getPrototypeOf(g.vfs).constructor)();

    await g.prompt(goal);

    const res = g.state.result;
    if (res?.metrics) {
      const card = addCard("eval", "held-out metrics");
      setCard(card, "ok", res.metrics.map((m) => `${m.label}: ${m.value}`).join("\n"));
    }
    const snap = g.snapshot();
    if (snap.export) {
      const blob = new Blob([g.state.export.bytes], { type: "application/octet-stream" });
      const url = URL.createObjectURL(blob);
      const card = addCard("export", snap.export.name + ".pkm");
      setCard(card, "ok", `${snap.export.size} bytes`);
      const a = document.createElement("a");
      a.className = "ag-download";
      a.href = url;
      a.download = snap.export.name + ".pkm";
      a.textContent = "download " + snap.export.name + ".pkm";
      card.appendChild(a);
    } else if (res?.model) {
      const bytes = res.model.save();
      const blob = new Blob([bytes], { type: "application/octet-stream" });
      const url = URL.createObjectURL(blob);
      g.state.export = { name: "model", bytes };
      const card = addCard("export", "model.pkm");
      setCard(card, "ok", `${bytes.length} bytes (auto-exported)`);
      const a = document.createElement("a");
      a.className = "ag-download";
      a.href = url;
      a.download = "model.pkm";
      a.textContent = "download model.pkm";
      card.appendChild(a);
    }
    if (res?.model && res.demoSpec) tryMountDemo(g, res.model, res.demoSpec);
    status(res ? "run complete" : "run finished without a trained model — see log");
  } catch (err) {
    logLine("bad", "error: " + (err?.message || String(err)).slice(0, 300));
    status("run failed — see raw log");
  } finally {
    $("agRun").disabled = false;
  }
}

if (typeof document !== "undefined") {
  $("agRun").addEventListener("click", () => void onRun());
  $("agAbort").addEventListener("click", () => {
    agent?.abort();
    logLine("bad", "aborted");
    status("aborted");
  });
  build().then(() => status("ready — describe a goal and Run")).catch((e) => status("wasm load failed: " + e.message));
}
