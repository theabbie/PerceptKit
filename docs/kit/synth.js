export const FONT57_COL = {
  "0": [0x3E, 0x51, 0x49, 0x45, 0x3E], "1": [0x00, 0x42, 0x7F, 0x40, 0x00],
  "2": [0x42, 0x61, 0x51, 0x49, 0x46], "3": [0x21, 0x41, 0x45, 0x4B, 0x31],
  "4": [0x18, 0x14, 0x12, 0x7F, 0x10], "5": [0x27, 0x45, 0x45, 0x45, 0x39],
  "6": [0x3C, 0x4A, 0x49, 0x49, 0x30], "7": [0x01, 0x71, 0x09, 0x05, 0x03],
  "8": [0x36, 0x49, 0x49, 0x49, 0x36], "9": [0x06, 0x49, 0x49, 0x29, 0x1E],
  "A": [0x7E, 0x11, 0x11, 0x11, 0x7E], "B": [0x7F, 0x49, 0x49, 0x49, 0x36],
  "C": [0x3E, 0x41, 0x41, 0x41, 0x22], "D": [0x7F, 0x41, 0x41, 0x22, 0x1C],
  "E": [0x7F, 0x49, 0x49, 0x49, 0x41], "F": [0x7F, 0x09, 0x09, 0x09, 0x01],
  "G": [0x3E, 0x41, 0x49, 0x49, 0x7A], "H": [0x7F, 0x08, 0x08, 0x08, 0x7F],
  "I": [0x00, 0x41, 0x7F, 0x41, 0x00], "J": [0x20, 0x40, 0x41, 0x3F, 0x01],
  "K": [0x7F, 0x08, 0x14, 0x22, 0x41], "L": [0x7F, 0x40, 0x40, 0x40, 0x40],
  "M": [0x7F, 0x02, 0x0C, 0x02, 0x7F], "N": [0x7F, 0x04, 0x08, 0x10, 0x7F],
  "O": [0x3E, 0x41, 0x41, 0x41, 0x3E], "P": [0x7F, 0x09, 0x09, 0x09, 0x06],
  "Q": [0x3E, 0x41, 0x51, 0x21, 0x5E], "R": [0x7F, 0x09, 0x19, 0x29, 0x46],
  "S": [0x46, 0x49, 0x49, 0x49, 0x31], "T": [0x01, 0x01, 0x7F, 0x01, 0x01],
  "U": [0x3F, 0x40, 0x40, 0x40, 0x3F], "V": [0x1F, 0x20, 0x40, 0x20, 0x1F],
  "W": [0x3F, 0x40, 0x38, 0x40, 0x3F], "X": [0x63, 0x14, 0x08, 0x14, 0x63],
  "Y": [0x07, 0x08, 0x70, 0x08, 0x07], "Z": [0x61, 0x51, 0x49, 0x45, 0x43],
  "-": [0x08, 0x08, 0x08, 0x08, 0x08], ".": [0x00, 0x60, 0x60, 0x00, 0x00],
  " ": [0x00, 0x00, 0x00, 0x00, 0x00], "+": [0x08, 0x08, 0x3E, 0x08, 0x08],
};

export function blank(W, H) { return new Float32Array(W * H); }
export function clone(img) { return new Float32Array(img); }

function strokeSDF(img, W, H, sdf, { w = 1, val = 1, soft = 0.8 } = {}) {
  const half = w / 2;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const d = sdf(x + 0.5, y + 0.5);
      let a;
      if (d <= half - soft / 2) a = 1;
      else if (d >= half + soft / 2) a = 0;
      else a = (half + soft / 2 - d) / soft;
      if (a > 0) {
        const i = y * W + x;
        const v = a * val;
        if (v > img[i]) img[i] = v;
      }
    }
  }
  return img;
}

function fillSDF(img, W, H, sdf, { val = 1, soft = 0.8 } = {}) {
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const d = sdf(x + 0.5, y + 0.5);
      let a;
      if (d <= -soft / 2) a = 1;
      else if (d >= soft / 2) a = 0;
      else a = 0.5 - d / soft;
      if (a > 0) {
        const i = y * W + x;
        const v = a * val;
        if (v > img[i]) img[i] = v;
      }
    }
  }
  return img;
}

export function circle(img, W, H, cx, cy, r, opts = {}) {
  if (opts.fill != null) {
    fillSDF(img, W, H, (x, y) => Math.hypot(x - cx, y - cy) - r, { val: opts.fill, soft: opts.soft ?? 0.8 });
  }
  if (opts.stroke != null) {
    strokeSDF(img, W, H, (x, y) => Math.abs(Math.hypot(x - cx, y - cy) - r),
      { w: opts.w ?? 1.5, val: opts.stroke, soft: opts.soft ?? 0.8 });
  }
  return img;
}

export function ellipse(img, W, H, cx, cy, rx, ry, opts = {}) {
  const m = Math.min(rx, ry);
  const d = (x, y) => (Math.hypot((x - cx) / rx, (y - cy) / ry) - 1) * m;
  if (opts.fill != null) fillSDF(img, W, H, d, { val: opts.fill, soft: opts.soft ?? 0.8 });
  if (opts.stroke != null) strokeSDF(img, W, H, (x, y) => Math.abs(d(x, y)), { w: opts.w ?? 1.5, val: opts.stroke, soft: opts.soft ?? 0.8 });
  return img;
}

export function rect(img, W, H, rx, ry, rw, rh, opts = {}) {
  const cx = rx + rw / 2, cy = ry + rh / 2;
  const d = (x, y) => {
    const dx = Math.abs(x - cx) - rw / 2, dy = Math.abs(y - cy) - rh / 2;
    return Math.hypot(Math.max(dx, 0), Math.max(dy, 0)) + Math.min(Math.max(dx, dy), 0);
  };
  if (opts.fill != null) fillSDF(img, W, H, d, { val: opts.fill, soft: opts.soft ?? 0.8 });
  if (opts.stroke != null) strokeSDF(img, W, H, (x, y) => Math.abs(d(x, y)), { w: opts.w ?? 1, val: opts.stroke, soft: opts.soft ?? 0.8 });
  return img;
}

function distSeg(px, py, x0, y0, x1, y1) {
  const dx = x1 - x0, dy = y1 - y0;
  const l2 = dx * dx + dy * dy;
  let t = l2 === 0 ? 0 : ((px - x0) * dx + (py - y0) * dy) / l2;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (x0 + t * dx), py - (y0 + t * dy));
}

export function line(img, W, H, x0, y0, x1, y1, opts = {}) {
  return strokeSDF(img, W, H, (x, y) => distSeg(x, y, x0, y0, x1, y1),
    { w: opts.w ?? 1, val: opts.val ?? opts.stroke ?? 1, soft: opts.soft ?? 0.8 });
}

export function polyline(img, W, H, pts, opts = {}) {
  for (let i = 0; i + 1 < pts.length; i++) {
    line(img, W, H, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], opts);
  }
  return img;
}

export function polygon(img, W, H, pts, opts = {}) {
  const val = opts.fill ?? 1;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const px = x + 0.5, py = y + 0.5;
      let inside = false;
      for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
        const [xi, yi] = pts[i], [xj, yj] = pts[j];
        if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside;
      }
      if (inside) {
        const k = y * W + x;
        if (val > img[k]) img[k] = val;
      }
    }
  }
  if (opts.stroke != null) {
    for (let i = 0; i < pts.length; i++) {
      const [x0, y0] = pts[i], [x1, y1] = pts[(i + 1) % pts.length];
      line(img, W, H, x0, y0, x1, y1, { w: opts.w ?? 1, val: opts.stroke, soft: opts.soft ?? 0.8 });
    }
  }
  return img;
}

export function bezier(img, W, H, pts, opts = {}) {
  const n = opts.n ?? 48;
  const path = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n, u = 1 - t;
    let x = 0, y = 0;
    if (pts.length === 3) {
      const [p0, p1, p2] = pts;
      x = u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0];
      y = u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1];
    } else {
      const [p0, p1, p2, p3] = pts;
      x = u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0];
      y = u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1];
    }
    path.push([x, y]);
  }
  return polyline(img, W, H, path, opts);
}

export function glyph(img, W, H, ch, gx, gy, opts = {}) {
  const g = FONT57_COL[String(ch).toUpperCase()] || FONT57_COL[" "];
  const scale = opts.scale ?? 1, val = opts.val ?? 1;
  for (let c = 0; c < 5; c++) {
    for (let r = 0; r < 7; r++) {
      if (!((g[c] >> r) & 1)) continue;
      for (let sy = 0; sy < scale; sy++) {
        for (let sx = 0; sx < scale; sx++) {
          const x = Math.round(gx + c * scale + sx), y = Math.round(gy + r * scale + sy);
          if (x >= 0 && x < W && y >= 0 && y < H && val > img[y * W + x]) img[y * W + x] = val;
        }
      }
    }
  }
  return img;
}

export function text(img, W, H, str, x, y, opts = {}) {
  const scale = opts.scale ?? 1;
  let cx = x;
  for (const ch of String(str)) {
    glyph(img, W, H, ch, cx, y, opts);
    cx += 6 * scale;
  }
  return img;
}

export function normal(rng) {
  const u = Math.max(rng(), 1e-9), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function noise(img, rng, amount) {
  for (let i = 0; i < img.length; i++) if (rng() < amount) img[i] = 1 - img[i];
  return img;
}

export function gauss(img, rng, sigma) {
  for (let i = 0; i < img.length; i++) img[i] += normal(rng) * sigma;
  return img;
}

export function dropout(img, rng, p) {
  for (let i = 0; i < img.length; i++) if (rng() < p) img[i] = 0;
  return img;
}

export function warp(img, W, H, fn) {
  const out = new Float32Array(img.length);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const [sx, sy] = fn(x, y);
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      const fx = sx - x0, fy = sy - y0;
      let v = 0;
      if (x0 >= 0 && y0 >= 0 && x0 < W && y0 < H) v += img[y0 * W + x0] * (1 - fx) * (1 - fy);
      if (x0 + 1 >= 0 && y0 >= 0 && x0 + 1 < W && y0 < H) v += img[y0 * W + x0 + 1] * fx * (1 - fy);
      if (x0 >= 0 && y0 + 1 >= 0 && x0 < W && y0 + 1 < H) v += img[(y0 + 1) * W + x0] * (1 - fx) * fy;
      if (x0 + 1 >= 0 && y0 + 1 >= 0 && x0 + 1 < W && y0 + 1 < H) v += img[(y0 + 1) * W + x0 + 1] * fx * fy;
      out[y * W + x] = v;
    }
  }
  return out;
}

export function translate(img, W, H, dx, dy) {
  return warp(img, W, H, (x, y) => [x - dx, y - dy]);
}

export function rotate(img, W, H, deg, cx = W / 2, cy = H / 2) {
  const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  return warp(img, W, H, (x, y) => {
    const dx = x - cx, dy = y - cy;
    return [cx + dx * c + dy * s, cy - dx * s + dy * c];
  });
}

export function shear(img, W, H, kx, ky) {
  return warp(img, W, H, (x, y) => [x - ky * (y - H / 2), y - kx * (x - W / 2)]);
}

export function scaleImg(img, W, H, sx, sy = sx, cx = W / 2, cy = H / 2) {
  return warp(img, W, H, (x, y) => [cx + (x - cx) / sx, cy + (y - cy) / sy]);
}

export function occlude(img, W, H, x, y, w, h, val = 0) {
  for (let yy = Math.max(0, Math.floor(y)); yy < Math.min(H, y + h); yy++) {
    for (let xx = Math.max(0, Math.floor(x)); xx < Math.min(W, x + w); xx++) {
      img[yy * W + xx] = val;
    }
  }
  return img;
}

export function invert(img) {
  for (let i = 0; i < img.length; i++) img[i] = 1 - img[i];
  return img;
}

export function threshold(img, t = 0.5) {
  for (let i = 0; i < img.length; i++) img[i] = img[i] >= t ? 1 : 0;
  return img;
}

export function blur3(img, W, H) {
  const out = new Float32Array(img.length);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let s = 0, n = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx >= 0 && xx < W && yy >= 0 && yy < H) { s += img[yy * W + xx]; n++; }
        }
      }
      out[y * W + x] = s / n;
    }
  }
  return out;
}

export function clamp01(img) {
  for (let i = 0; i < img.length; i++) img[i] = img[i] < 0 ? 0 : img[i] > 1 ? 1 : img[i];
  return img;
}

export function coverage(img, t = 0.5) {
  let n = 0;
  for (let i = 0; i < img.length; i++) if (img[i] >= t) n++;
  return n / img.length;
}

export function centroid(img, W, H) {
  let sx = 0, sy = 0, sw = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const v = img[y * W + x];
      sx += x * v; sy += y * v; sw += v;
    }
  }
  return sw > 0 ? [sx / sw, sy / sw] : [W / 2, H / 2];
}

export function bbox(img, W, H, t = 0.5) {
  let x0 = W, y0 = H, x1 = -1, y1 = -1;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (img[y * W + x] >= t) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  return x1 < 0 ? null : [x0, y0, x1 - x0 + 1, y1 - y0 + 1];
}

export function mergeMax(dst, src) {
  for (let i = 0; i < dst.length; i++) if (src[i] > dst[i]) dst[i] = src[i];
  return dst;
}

export function over(dst, src, alpha = 1) {
  for (let i = 0; i < dst.length; i++) {
    const v = src[i] * alpha;
    dst[i] = v + dst[i] * (1 - Math.min(1, v));
  }
  return dst;
}

export function grid(img, W, H, opts = {}) {
  const rows = opts.rows ?? 4, cols = opts.cols ?? 4;
  const val = opts.val ?? 1, w = opts.w ?? 1;
  for (let c = 1; c < cols; c++) line(img, W, H, (W * c) / cols, 0, (W * c) / cols, H, { w, val });
  for (let r = 1; r < rows; r++) line(img, W, H, 0, (H * r) / rows, W, (H * r) / rows, { w, val });
  if (opts.border) rect(img, W, H, 0, 0, W, H, { stroke: val, w });
  return img;
}
