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

export function mountDemo({ container, model, wasm, spec, engine }) {
  container.innerHTML = "";
  container.style.display = "";

  const head = document.createElement("div");
  head.className = "probe-header";
  const title = document.createElement("span");
  title.className = "ag-report-label";
  title.textContent = spec.kind === "classify"
    ? "live demo — draw a " + (spec.labels.join("/")) + ", the model classifies"
    : "live demo — draw, the model scores it";
  const clearBtn = document.createElement("button");
  clearBtn.className = "primary";
  clearBtn.textContent = "clear";
  clearBtn.style.cssText = "padding:4px 10px;font-size:11px";
  head.append(title, clearBtn);

  const canvas = document.createElement("canvas");
  canvas.width = spec.W;
  canvas.height = spec.H;
  canvas.style.cssText =
    "width:100%;aspect-ratio:" + spec.W + "/" + spec.H +
    ";image-rendering:pixelated;background:#000;border:1px solid var(--edge);border-radius:4px;cursor:crosshair;touch-action:none";

  const out = document.createElement("div");
  out.className = "ag-stats";
  out.style.marginTop = "8px";
  out.textContent = "draw with the mouse / finger";

  const barWrap = document.createElement("div");
  barWrap.style.cssText = "margin-top:6px;display:none";
  const barBox = document.createElement("canvas");
  barBox.width = 400; barBox.height = 140;
  barBox.style.cssText = "width:100%;image-rendering:pixelated;background:#000;border-radius:4px";
  barWrap.append(barBox);

  container.append(head, canvas, out, barWrap);

  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, spec.W, spec.H);
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = Math.max(1.2, spec.W / 22);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  let drawing = false, dirty = false, raf = 0;

  const pos = (ev) => {
    const r = canvas.getBoundingClientRect();
    return [
      (ev.clientX - r.left) * spec.W / r.width,
      (ev.clientY - r.top) * spec.H / r.height,
    ];
  };

  const luma = () => {
    const d = ctx.getImageData(0, 0, spec.W, spec.H).data;
    const px = new Uint8Array(spec.W * spec.H);
    for (let i = 0; i < px.length; i++) px[i] = d[i * 4];
    return px;
  };

  const session = { scoreMin: Infinity, scoreMax: -Infinity };

  const infer = () => {
    if (!dirty) return;
    dirty = false;
    const r = demoInfer(luma(), model, wasm, spec, session);
    if (r.empty) { out.textContent = "draw with the mouse / finger"; return; }
    if (r.kind === "classify") {
      out.textContent = "prediction: " + r.label;
      barWrap.style.display = "";
      drawClassifyBars(barBox, spec.labels, r.norm, r.best);
    } else {
      out.textContent = "score: " + r.score.toFixed(3) +
        "  (higher = cleaner; relative " + (r.rel * 100).toFixed(0) + "% this session)";
      barWrap.style.display = "";
      drawScoreGauge(barBox, r.rel, r.score);
    }
  };

  const pump = () => { raf = 0; infer(); };
  const mark = () => { dirty = true; if (!raf) raf = requestAnimationFrame(pump); };

  canvas.addEventListener("pointerdown", (ev) => {
    drawing = true;
    canvas.setPointerCapture(ev.pointerId);
    const [x, y] = pos(ev);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 0.01, y + 0.01);
    ctx.stroke();
    mark();
  });
  canvas.addEventListener("pointermove", (ev) => {
    if (!drawing) return;
    const [x, y] = pos(ev);
    ctx.lineTo(x, y);
    ctx.stroke();
    mark();
  });
  const up = () => { drawing = false; };
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
  clearBtn.addEventListener("click", () => {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, spec.W, spec.H);
    session.scoreMin = Infinity; session.scoreMax = -Infinity;
    out.textContent = "draw with the mouse / finger";
    barWrap.style.display = "none";
    dirty = false;
  });

  return { canvas, inferNow: () => { dirty = true; infer(); } };
}

function drawClassifyBars(canvas, labels, norm, best) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0a0a0b"; ctx.fillRect(0, 0, W, H);
  const n = labels.length;
  const bw = (W - 12) / n;
  ctx.font = "11px ui-monospace, monospace";
  ctx.textAlign = "center";
  for (let i = 0; i < n; i++) {
    const h = Math.max(2, norm[i] * (H - 34));
    ctx.fillStyle = i === best ? "#4ade80" : "#0ea5e9";
    ctx.fillRect(6 + i * bw, H - 16 - h, bw - 3, h);
    ctx.fillStyle = "#a1a1aa";
    ctx.fillText(String(labels[i]), 6 + i * bw + bw / 2, H - 3);
  }
  ctx.textAlign = "left";
}

function drawScoreGauge(canvas, rel, raw) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0a0a0b"; ctx.fillRect(0, 0, W, H);
  const gw = W - 24, gh = 26, gx = 12, gy = H / 2 - 26;
  ctx.fillStyle = "#1f2937";
  ctx.fillRect(gx, gy, gw, gh);
  const frac = Math.max(0.02, Math.min(1, rel));
  const grad = ctx.createLinearGradient(gx, 0, gx + gw, 0);
  grad.addColorStop(0, "#f87171");
  grad.addColorStop(0.5, "#fbbf24");
  grad.addColorStop(1, "#4ade80");
  ctx.fillStyle = grad;
  ctx.fillRect(gx, gy, gw * frac, gh);
  ctx.fillStyle = "#a1a1aa";
  ctx.font = "12px ui-monospace, monospace";
  ctx.fillText("raw score " + raw.toFixed(3), gx, gy + gh + 18);
}
