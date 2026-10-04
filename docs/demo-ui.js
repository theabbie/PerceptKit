import { DrawPad } from "./kit/canvas.js?v=1.8";
import * as viz from "./kit/viz.js?v=1.8";

export function demoInfer(px, model, wasm, spec, session) {
  let min = 255, max = 0;
  for (const v of px) { if (v < min) min = v; if (v > max) max = v; }
  if (max < 24) return { kind: spec.kind, empty: true };
  const img = new Float32Array(px.length);
  for (let i = 0; i < px.length; i++) img[i] = px[i] / 255;
  const tokens = wasm.patchifyImage(img, spec.W, spec.H, spec.PATCH, spec.inCh);
  const T = (spec.W / spec.PATCH) * (spec.H / spec.PATCH);
  if (spec.kind === "classify") {
    const raw = model.transform(tokens, T);
    const nC = spec.labels.length;
    const scores = new Array(nC).fill(0);
    for (let t = 0; t < T; t++)
      for (let c = 0; c < nC; c++) scores[c] += raw[t * nC + c];
    const best = scores.indexOf(Math.max(...scores));
    const mx = Math.max(...scores), mn = Math.min(...scores);
    const span = (mx - mn) || 1;
    return { kind: "classify", best, label: spec.labels[best], norm: scores.map((s) => (s - mn) / span), raw: scores };
  }
  const s = model.score(tokens, T);
  session.scoreMin = Math.min(session.scoreMin, s);
  session.scoreMax = Math.max(session.scoreMax, s);
  const span = (session.scoreMax - session.scoreMin) || 1;
  return { kind: "score", score: s, rel: (s - session.scoreMin) / span };
}

function slider(labelText, min, max, step, val) {
  const wrap = document.createElement("label");
  wrap.style.cssText = "display:flex;align-items:center;gap:8px;font-size:11px;color:var(--dim,#a1a1aa);font-family:monospace";
  const txt = document.createElement("span");
  txt.textContent = labelText;
  const input = document.createElement("input");
  input.type = "range";
  input.min = min; input.max = max; input.step = step; input.value = val;
  input.style.cssText = "width:110px;accent-color:#38bdf8";
  const valEl = document.createElement("span");
  valEl.textContent = val;
  input.addEventListener("input", () => { valEl.textContent = input.value; });
  wrap.append(txt, input, valEl);
  return { wrap, input };
}

export function mountDemo({ container, model, wasm, spec, engine, corruptor }) {
  container.innerHTML = "";
  container.style.display = "";

  const head = document.createElement("div");
  head.className = "probe-header";
  const title = document.createElement("span");
  title.className = "ag-report-label";
  title.textContent = spec.kind === "classify"
    ? "draw a " + (spec.labels.join("/")) + " — the model classifies"
    : "draw — the model scores it";
  const clearBtn = document.createElement("button");
  clearBtn.className = "primary";
  clearBtn.textContent = "clear";
  clearBtn.style.cssText = "padding:4px 10px;font-size:11px";
  head.append(title, clearBtn);

  const controls = document.createElement("div");
  controls.style.cssText = "display:flex;gap:18px;align-items:center;margin:8px 0;flex-wrap:wrap";
  const strokeCtl = slider("stroke", 0.8, 4, 0.2, 1.5);
  controls.append(strokeCtl.wrap);
  let sevCtl = null;
  if (spec.kind === "score" && corruptor) {
    sevCtl = slider("corrupt", 0, 1, 0.05, 0);
    controls.append(sevCtl.wrap);
  }

  const row = document.createElement("div");
  row.style.cssText = "display:grid;grid-template-columns:minmax(220px,340px) 1fr;gap:14px;align-items:start";
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "width:100%;background:#000;border:1px solid var(--edge,#27272a);border-radius:6px;cursor:crosshair";
  const right = document.createElement("div");
  const viewLabel = document.createElement("div");
  viewLabel.className = "ag-report-label";
  viewLabel.textContent = "model\u2019s view";
  const viewBox = document.createElement("canvas");
  viewBox.width = spec.W;
  viewBox.height = spec.H;
  viewBox.style.cssText = "width:96px;image-rendering:pixelated;background:#000;border:1px solid var(--edge,#27272a);border-radius:4px;margin-bottom:8px";
  const out = document.createElement("div");
  out.className = "ag-stats";
  out.textContent = "draw with mouse / finger";
  const barBox = document.createElement("canvas");
  barBox.style.cssText = "width:100%;display:none;margin-top:6px";
  const gaugeBox = document.createElement("canvas");
  gaugeBox.style.cssText = "width:100%;display:none;margin-top:6px";
  const cmpBox = document.createElement("canvas");
  cmpBox.style.cssText = "width:100%;display:none;margin-top:10px";
  const cmpOut = document.createElement("div");
  cmpOut.className = "ag-stats";
  right.append(viewLabel, viewBox, out, barBox, gaugeBox, cmpBox, cmpOut);
  row.append(canvas, right);
  container.append(head, controls, row);

  const pad = new DrawPad(canvas, { W: spec.W, H: spec.H, ss: 8, stroke: 1.5 });
  strokeCtl.input.addEventListener("input", () => pad.setStroke(parseFloat(strokeCtl.input.value)));
  clearBtn.addEventListener("click", () => {
    pad.clear();
    cmpBox.style.display = "none";
    cmpOut.textContent = "";
  });

  const session = { scoreMin: Infinity, scoreMax: -Infinity };
  const cmpSession = { scoreMin: Infinity, scoreMax: -Infinity };

  const run = () => {
    const f32 = pad.pixels();
    viz.imgToCanvas({ img: f32, w: spec.W, h: spec.H }, viewBox);
    const px = new Uint8Array(f32.length);
    for (let i = 0; i < f32.length; i++) px[i] = Math.round(f32[i] * 255);
    const res = demoInfer(px, model, wasm, spec, session);
    if (res.empty) {
      out.textContent = "draw with mouse / finger";
      barBox.style.display = "none";
      gaugeBox.style.display = "none";
      cmpBox.style.display = "none";
      cmpOut.textContent = "";
      return;
    }
    if (res.kind === "classify") {
      out.textContent = `predicted: ${res.label}`;
      barBox.style.display = "";
      viz.bars(barBox, res.norm, { labels: spec.labels });
      gaugeBox.style.display = "none";
    } else {
      out.textContent = `score: ${res.score.toFixed(3)}  (higher = cleaner)`;
      gaugeBox.style.display = "";
      viz.gauge(gaugeBox, res.rel, { raw: res.score });
      barBox.style.display = "none";
      if (sevCtl && corruptor) {
        const sev = parseFloat(sevCtl.input.value);
        if (sev > 0) {
          try {
            const corrupted = corruptor(new Float32Array(f32), sev, engine.rng((Math.random() * 1e9) | 0), 0);
            const cpx = new Uint8Array(corrupted.length);
            for (let i = 0; i < corrupted.length; i++) {
              cpx[i] = Math.max(0, Math.min(255, Math.round(corrupted[i] * 255)));
            }
            const cres = demoInfer(cpx, model, wasm, spec, cmpSession);
            cmpBox.style.display = "";
            viz.compare(cmpBox,
              { img: f32, w: spec.W, h: spec.H },
              { img: corrupted, w: spec.W, h: spec.H },
              { labels: ["yours", `corrupt ${sev.toFixed(2)}`], cell: 110 });
            cmpOut.textContent = cres.empty
              ? "corrupted sample scores below draw threshold"
              : `yours ${res.score.toFixed(3)} vs corrupted ${cres.score.toFixed(3)} — gap ${(res.score - cres.score).toFixed(3)}`;
          } catch (e) {
            cmpOut.textContent = `corrupt preview failed: ${e.message}`;
          }
        } else {
          cmpBox.style.display = "none";
          cmpOut.textContent = "";
        }
      }
    }
  };
  pad.onChange(run);

  window.__demo = { canvas, pad, inferNow: () => pad.inferNow(), container };
  return window.__demo;
}
