export const C = {
  bg: "#09090b", panel: "#101013", edge: "#27272a", text: "#e4e4e7",
  dim: "#a1a1aa", faint: "#52525b", sky: "#38bdf8", green: "#4ade80",
  red: "#f87171", amber: "#fbbf24", violet: "#a78bfa", zinc: "#3f3f46",
};

function prep(canvas, hCss) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const wCss = canvas.clientWidth || 300;
  const h = hCss ?? canvas.clientHeight ?? 180;
  canvas.width = Math.max(1, Math.round(wCss * dpr));
  canvas.height = Math.max(1, Math.round(h * dpr));
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, wCss, h);
  return { ctx, w: wCss, h };
}

function axes(ctx, w, h, pad) {
  ctx.strokeStyle = C.edge;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, h - pad.b);
  ctx.lineTo(w - pad.r, h - pad.b);
  ctx.stroke();
}

export function line(canvas, series, opts = {}) {
  const { ctx, w, h } = prep(canvas, opts.height);
  const pad = { l: 34, r: 8, t: 8, b: 16 };
  const all = [];
  for (const s of series) {
    const pts = s.pts.map((p, i) => (Array.isArray(p) ? p : [i, p]));
    all.push(...pts.map((p) => p[1]));
    s._n = pts;
  }
  let lo = opts.min ?? Math.min(...all, 0), hi = opts.max ?? Math.max(...all, 1e-6);
  if (hi - lo < 1e-9) hi = lo + 1;
  const X = (i, n) => pad.l + ((w - pad.l - pad.r) * i) / Math.max(1, n - 1);
  const Y = (v) => pad.t + (h - pad.t - pad.b) * (1 - (v - lo) / (hi - lo));
  axes(ctx, w, h, pad);
  ctx.fillStyle = C.faint;
  ctx.font = "9px monospace";
  ctx.textAlign = "right";
  for (let i = 0; i <= 3; i++) {
    const v = lo + ((hi - lo) * i) / 3;
    ctx.fillText(v.toPrecision(2), pad.l - 4, Y(v) + 3);
    ctx.strokeStyle = "rgba(39,39,42,0.5)";
    ctx.beginPath();
    ctx.moveTo(pad.l, Y(v));
    ctx.lineTo(w - pad.r, Y(v));
    ctx.stroke();
  }
  series.forEach((s, si) => {
    ctx.strokeStyle = s.color || [C.sky, C.green, C.amber, C.violet][si % 4];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    s._n.forEach((p, i) => {
      const x = X(i, s._n.length), y = Y(p[1]);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    if (s.label) {
      ctx.fillStyle = ctx.strokeStyle;
      ctx.textAlign = "left";
      ctx.fillText(s.label, pad.l + 6 + si * 90, pad.t + 8);
    }
  });
  for (const s of series) delete s._n;
}

export function bars(canvas, values, opts = {}) {
  const { ctx, w, h } = prep(canvas, opts.height ?? 120);
  const labels = opts.labels || values.map((_, i) => String(i));
  const n = values.length;
  const pad = { l: 4, r: 4, t: 14, b: 16 };
  const bw = (w - pad.l - pad.r) / n;
  const hi = opts.highlight ?? values.indexOf(Math.max(...values));
  values.forEach((v, i) => {
    const bh = (h - pad.t - pad.b) * Math.max(0, Math.min(1, v));
    ctx.fillStyle = i === hi ? C.sky : C.zinc;
    ctx.fillRect(pad.l + i * bw + 2, h - pad.b - bh, bw - 4, bh);
    ctx.fillStyle = i === hi ? C.text : C.dim;
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.fillText(String(labels[i]).slice(0, 8), pad.l + i * bw + bw / 2, h - 4);
    ctx.fillText(v.toFixed(2), pad.l + i * bw + bw / 2, h - pad.b - bh - 3);
  });
}

export function gauge(canvas, frac, opts = {}) {
  const { ctx, w, h } = prep(canvas, opts.height ?? 34);
  const f = Math.max(0, Math.min(1, frac));
  ctx.fillStyle = C.zinc;
  ctx.fillRect(0, 4, w, h - 22);
  ctx.fillStyle = f > 0.6 ? C.green : f > 0.35 ? C.amber : C.red;
  ctx.fillRect(0, 4, w * f, h - 22);
  ctx.fillStyle = C.text;
  ctx.font = "11px monospace";
  ctx.textAlign = "left";
  ctx.fillText(opts.raw != null ? `raw score ${Number(opts.raw).toFixed(3)}` : "", 4, h - 2);
  ctx.textAlign = "right";
  ctx.fillText(`${Math.round(f * 100)}%`, w - 4, h - 2);
}

export function heatmap(canvas, matrix, opts = {}) {
  const rows = matrix.length, cols = matrix[0]?.length ?? 0;
  if (!rows || !cols) return;
  const { ctx, w, h } = prep(canvas, opts.height ?? Math.max(120, rows * 22));
  const padL = opts.rowLabels ? 40 : 4, padB = opts.colLabels ? 14 : 4;
  const cw = (w - padL - 4) / cols, ch = (h - 4 - padB) / rows;
  let lo = Infinity, hi = -Infinity;
  for (const r of matrix) for (const v of r) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (hi - lo < 1e-9) hi = lo + 1;
  const col = (t) => {
    const a = [9, 9, 11], b = [56, 189, 248];
    const m = a.map((v, i) => Math.round(v + (b[i] - v) * t));
    return `rgb(${m[0]},${m[1]},${m[2]})`;
  };
  matrix.forEach((row, r) => {
    row.forEach((v, c) => {
      ctx.fillStyle = col((v - lo) / (hi - lo));
      ctx.fillRect(padL + c * cw, 4 + r * ch, cw - 1, ch - 1);
      if (cw > 24 && ch > 12) {
        ctx.fillStyle = C.text;
        ctx.font = "9px monospace";
        ctx.textAlign = "center";
        ctx.fillText(String(typeof v === "number" ? Math.round(v * 100) / 100 : v), padL + c * cw + cw / 2, 4 + r * ch + ch / 2 + 3);
      }
    });
    if (opts.rowLabels) {
      ctx.fillStyle = C.dim;
      ctx.font = "9px monospace";
      ctx.textAlign = "right";
      ctx.fillText(String(opts.rowLabels[r]).slice(0, 6), padL - 3, 4 + r * ch + ch / 2 + 3);
    }
  });
  if (opts.colLabels) {
    ctx.fillStyle = C.dim;
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    opts.colLabels.forEach((l, c) => ctx.fillText(String(l).slice(0, 6), padL + c * cw + cw / 2, h - 2));
  }
}

export function scatter(canvas, points, opts = {}) {
  const { ctx, w, h } = prep(canvas, opts.height);
  const pad = { l: 34, r: 8, t: 8, b: 16 };
  const pts = points.map((p) => (Array.isArray(p) ? { x: p[0], y: p[1] } : p));
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const X = (v) => pad.l + ((w - pad.l - pad.r) * (v - x0)) / Math.max(1e-9, x1 - x0);
  const Y = (v) => pad.t + (h - pad.t - pad.b) * (1 - (v - y0) / Math.max(1e-9, y1 - y0));
  axes(ctx, w, h, pad);
  for (const p of pts) {
    ctx.fillStyle = p.color || C.sky;
    ctx.beginPath();
    ctx.arc(X(p.x), Y(p.y), 3, 0, 6.284);
    ctx.fill();
  }
}

export function hist(canvas, values, opts = {}) {
  const bins = opts.bins ?? 16;
  const lo = Math.min(...values), hi = Math.max(...values);
  const cnt = new Array(bins).fill(0);
  for (const v of values) {
    const b = Math.min(bins - 1, Math.floor(((v - lo) / Math.max(1e-9, hi - lo)) * bins));
    cnt[b]++;
  }
  const max = Math.max(...cnt, 1);
  bars(canvas, cnt.map((c) => c / max), {
    labels: cnt.map((_, i) => (lo + ((hi - lo) * i) / bins).toPrecision(2)),
    height: opts.height ?? 120, highlight: -1,
  });
}

export function imgToCanvas(src, dst) {
  const w = src.w, h = src.h;
  dst.width = w;
  dst.height = h;
  const ctx = dst.getContext("2d");
  const im = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const v = Math.max(0, Math.min(255, Math.round((src.img[i] ?? 0) * 255)));
    im.data[i * 4] = v;
    im.data[i * 4 + 1] = v;
    im.data[i * 4 + 2] = v;
    im.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
  return dst;
}

export function grid(canvas, images, opts = {}) {
  const cols = opts.cols ?? Math.ceil(Math.sqrt(images.length));
  const cell = opts.cell ?? 48, pad = opts.pad ?? 6, labH = opts.labels ? 12 : 0;
  const rows = Math.ceil(images.length / cols);
  const { ctx } = prep(canvas, rows * (cell + pad + labH) + pad);
  const wCss = canvas.width / Math.min(window.devicePixelRatio || 1, 2);
  const off = document.createElement("canvas");
  images.forEach((it, i) => {
    const r = Math.floor(i / cols), c = i % cols;
    const x = pad + c * ((wCss - 2 * pad) / cols);
    const cw = (wCss - 2 * pad) / cols - pad;
    const y = pad + r * (cell + pad + labH);
    const side = Math.min(cw, cell);
    if (it.canvas) ctx.drawImage(it.canvas, x, y, side, side);
    else if (it.img) {
      imgToCanvas(it, off);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off, x, y, side, side);
      ctx.imageSmoothingEnabled = true;
    }
    ctx.strokeStyle = it.ok === true ? C.green : it.ok === false ? C.red : C.edge;
    ctx.lineWidth = 1;
    ctx.strokeRect(x - 0.5, y - 0.5, side + 1, side + 1);
    if (opts.labels && it.label != null) {
      ctx.fillStyle = it.ok === false ? C.red : C.dim;
      ctx.font = "9px monospace";
      ctx.textAlign = "left";
      ctx.fillText(String(it.label).slice(0, 16), x, y + side + 10);
    }
  });
}

export function compare(canvas, a, b, opts = {}) {
  grid(canvas, [
    { ...a, label: opts.labels?.[0] ?? "a" },
    { ...b, label: opts.labels?.[1] ?? "b" },
  ], { cols: 2, labels: true, cell: opts.cell ?? 96 });
}
