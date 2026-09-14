// Probe viewer — renders the agent's probe output onto the live canvas.
// The probe (LLM code) returns a dict with optional `ui` spec; the default
// renderer shows a loss curve + per-class bars; `ui` can ask for heat maps,
// scatter plots, or confusion tables with the probe's own data.
// Data stays JSON-safe; the renderer operates on numbers only.

export function renderProbe(canvas, probeOut, model, engine, wasm) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0a0a0b"; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "#1f2937"; ctx.strokeRect(0.5, 0.5, W - 1, H - 1);

  const spec = probeOut?.ui || { kind: "auto" };
  const title = document.createElement("div");

  if (spec.kind === "loss" || (!probeOut?.ui && probeOut?.losses)) {
    drawLossCurve(ctx, probeOut.losses, W, H);
    return;
  }
  if (spec.kind === "bars" && probeOut?.values) {
    drawBars(ctx, probeOut.values, W, H);
    return;
  }
  if (spec.kind === "heat" && probeOut?.matrix) {
    drawHeat(ctx, probeOut.matrix, W, H);
    return;
  }
  if (spec.kind === "scatter" && probeOut?.points) {
    drawScatter(ctx, probeOut.points, W, H);
    return;
  }
  if (spec.kind === "pred" && probeOut?.scores) {
    // model prediction bar chart from the probe's own scores
    drawPredBars(ctx, probeOut, W, H);
    return;
  }

  // ---------- auto fallback: pick a sane default based on probe content --
  if (typeof probeOut === "object" && probeOut !== null) {
    if (Array.isArray(probeOut.losses)) { drawLossCurve(ctx, probeOut.losses, W, H); return; }
    if (Array.isArray(probeOut.values)) { drawBars(ctx, probeOut.values, W, H); return; }
    if (Array.isArray(probeOut.points)) { drawScatter(ctx, probeOut.points, W, H); return; }
    if (Array.isArray(probeOut.matrix)) { drawHeat(ctx, probeOut.matrix, W, H); return; }
    if (Array.isArray(probeOut.scores)) { drawPredBars(ctx, probeOut, W, H); return; }
    // summary card fallback: render numbers
    drawKeyValues(ctx, probeOut, W, H);
    return;
  }
  // if the model is given, render a generic "attention/summary" graphic
  if (model) { drawModelCardSafe(ctx, model, W, H); return; }

  ctx.fillStyle = "#a1a1aa";
  ctx.font = "11px ui-monospace, monospace";
  ctx.fillText("(no renderable probe fields — pass ui:{kind:...})", 8, H - 8);
}

function drawModelCardSafe(ctx, model, W, H) {
  try { drawModelCard(ctx, model, W, H); }
  catch (e) {
    ctx.fillStyle = "#a1a1aa";
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText("model summary (unavailable)", 8, 12);
  }
}

function drawLossCurve(ctx, losses, W, H) {
  if (!losses.length) return;
  const maxV = Math.max(...losses.map((l) => l.value), 1e-6);
  ctx.strokeStyle = "#1f2937";
  ctx.beginPath();
  for (let i = 0; i < losses.length; i++) {
    const x = 6 + (i / (losses.length - 1 || 1)) * (W - 12);
    const y = H - 8 - (losses[i].value / maxV) * (H - 20);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = "#0ea5e9";
  ctx.stroke();
  ctx.fillStyle = "#a1a1aa";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(`loss curve (min ${Math.min(...losses.map((l) => l.value)).toFixed(4)})`, 8, 12);
}

function drawBars(ctx, values, W, H) {
  const vs = values.map((v) => typeof v === "number" ? v : (v.value ?? v));
  const maxV = Math.max(...vs, 1e-6);
  const barW = (W - 12) / values.length;
  values.forEach((v, i) => {
    const val = typeof v === "number" ? v : (v.value ?? v);
    const h = (val / maxV) * (H - 24);
    ctx.fillStyle = "#0ea5e9";
    ctx.fillRect(6 + i * barW, H - 8 - h, barW - 2, h);
  });
  ctx.fillStyle = "#a1a1aa";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText("values", 8, 12);
}

function drawHeat(ctx, matrix, W, H) {
  const rows = matrix.length, cols = matrix[0].length;
  const maxV = Math.max(...matrix.flat(), 1e-6);
  const sx = (W - 8) / cols, sy = (H - 8) / rows;
  for (let y = 0; y < rows; y++)
    for (let x = 0; x < cols; x++) {
      const v = matrix[y][x];
      const c = Math.floor(255 * (v / maxV));
      ctx.fillStyle = `rgb(0,${Math.floor(c * 0.7)},${c})`;
      ctx.fillRect(4 + x * sx, 4 + y * sy, sx, sy);
    }
  ctx.fillStyle = "#a1a1aa";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText("heat map", 8, 12);
}

function drawScatter(ctx, points, W, H) {
  let xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  ctx.fillStyle = "#0ea5e9";
  for (const p of points) {
    const x = 6 + ((p[0] - Math.min(...xs)) / ((Math.max(...xs) - Math.min(...xs)) || 1)) * (W - 12);
    const y = H - 8 - ((p[1] - Math.min(...ys)) / ((Math.max(...ys) - Math.min(...ys)) || 1)) * (H - 20);
    ctx.beginPath(); ctx.arc(x, y, 1.5, 0, 2 * Math.PI); ctx.fill();
  }
  ctx.fillStyle = "#a1a1aa";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText("scatter", 8, 12);
}

function drawPredBars(ctx, probeOut, W, H) {
  const scores = probeOut.scores;
  const sum = scores.reduce((a, b) => a + Math.abs(b), 0) || 1;
  const barW = (W - 12) / scores.length;
  ctx.font = "10px ui-monospace, monospace";
  scores.forEach((s, i) => {
    const val = Math.abs(s) / sum;
    const h = Math.max(2, val * (H - 24));
    ctx.fillStyle = s === Math.max(...scores) ? "#4ade80" : "#0ea5e9";
    ctx.fillRect(6 + i * barW, H - 8 - h, barW - 2, h);
  });
  const best = scores.indexOf(Math.max(...scores));
  ctx.fillStyle = "#a1a1aa";
  ctx.fillText(`prediction → class ${best}`, 8, 12);
}

function drawKeyValues(ctx, probeOut, W, H) {
  ctx.fillStyle = "#a1a1aa";
  ctx.font = "10px ui-monospace, monospace";
  let y = 14;
  for (const [k, v] of Object.entries(probeOut)) {
    if (typeof v === "object") continue;
    ctx.fillText(`${k}: ${typeof v === "number" ? v.toPrecision(4) : v}`, 8, y);
    y += 13;
    if (y > H - 8) break;
  }
}

function drawModelCard(ctx, model, W, H) {
  ctx.fillStyle = "#a1a1aa";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText("model summary", 8, 12);
  const lines = [];
  try { lines.push(`params: ${model.nParams()}`); } catch { lines.push("params: ?"); }
  try {
    const cfg = JSON.parse(model.configJson() || "{}");
    if (cfg.in_channels !== undefined)
      lines.push(`in: ${cfg.in_channels}, out: ${cfg.out_channels}`);
    if (cfg.d_model !== undefined)
      lines.push(`d_model: ${cfg.d_model}, layers: ${cfg.n_layers}, heads: ${cfg.n_heads}`);
    if (cfg.max_seq_len !== undefined)
      lines.push(`tokens: ${cfg.max_seq_len}`);
  } catch { }
  let y = 26;
  for (const l of lines) { ctx.fillText(l, 8, y); y += 13; }
}
