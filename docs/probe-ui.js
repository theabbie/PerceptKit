import * as viz from "./kit/viz.js?v=1.1";

export function renderProbe(canvas, probeOut, model, engine, wasm) {
  if (!canvas) return;
  const spec = probeOut?.ui || { kind: "auto" };
  const losses = (probeOut?.losses || []).filter((l) => Number.isFinite(l?.value));

  if (spec.kind === "loss" || (!probeOut?.ui && losses.length)) {
    viz.line(canvas, [{ pts: losses.map((l) => l.value), color: viz.C.sky, label: "loss" }]);
    return;
  }
  if (spec.kind === "bars" && probeOut?.values) {
    viz.bars(canvas, probeOut.values, { labels: probeOut.labels });
    return;
  }
  if (spec.kind === "heat" && probeOut?.matrix) {
    viz.heatmap(canvas, probeOut.matrix, { rowLabels: probeOut.rowLabels, colLabels: probeOut.colLabels });
    return;
  }
  if (spec.kind === "scatter" && probeOut?.points) {
    viz.scatter(canvas, probeOut.points);
    return;
  }
  if (spec.kind === "pred" && probeOut?.scores) {
    viz.bars(canvas, probeOut.scores, { labels: probeOut.labels });
    return;
  }

  if (probeOut && typeof probeOut === "object") {
    if (losses.length) { viz.line(canvas, [{ pts: losses.map((l) => l.value), color: viz.C.sky, label: "loss" }]); return; }
    if (Array.isArray(probeOut.values)) { viz.bars(canvas, probeOut.values, { labels: probeOut.labels }); return; }
    if (Array.isArray(probeOut.points)) { viz.scatter(canvas, probeOut.points); return; }
    if (Array.isArray(probeOut.matrix)) { viz.heatmap(canvas, probeOut.matrix); return; }
    if (Array.isArray(probeOut.scores)) { viz.bars(canvas, probeOut.scores, { labels: probeOut.labels }); return; }
    if (Array.isArray(probeOut.cases) && probeOut.cases.length && probeOut.cases[0]?.img) {
      viz.grid(canvas, probeOut.cases.slice(0, 12).map((c) => ({
        img: c.img, w: c.w, h: c.h,
        label: c.predicted != null ? `${c.label ?? "?"}→${c.predicted}` : c.label,
        ok: c.ok,
      })), { labels: true });
      return;
    }
    drawKeyValues(canvas, probeOut);
    return;
  }

  const ctx = canvas.getContext("2d");
  ctx.fillStyle = viz.C.dim;
  ctx.font = "11px monospace";
  ctx.fillText("(no renderable probe fields — pass ui:{kind:...})", 8, 20);
}

function drawKeyValues(canvas, obj) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = viz.C.dim;
  ctx.font = "10px monospace";
  const entries = Object.entries(obj).filter(([k]) => k !== "ui").slice(0, 12);
  entries.forEach(([k, v], i) => {
    const s = typeof v === "object" ? JSON.stringify(v).slice(0, 60) : String(v).slice(0, 60);
    ctx.fillText(`${k}: ${s}`, 8, 16 + i * 13);
  });
}
