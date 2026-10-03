export class DrawPad {
  constructor(canvas, { W = 64, H = 64, ss = 8, stroke = 2, bg = "#000000", fg = "#ffffff" } = {}) {
    this.canvas = canvas;
    this.W = W;
    this.H = H;
    this.ss = ss;
    this.strokeW = stroke;
    this.bg = bg;
    this.fg = fg;
    canvas.width = W * ss;
    canvas.height = H * ss;
    this.ctx = canvas.getContext("2d");
    this.down = document.createElement("canvas");
    this.down.width = W;
    this.down.height = H;
    this.dctx = this.down.getContext("2d", { willReadFrequently: true });
    this.drawing = false;
    this.last = null;
    this.cb = null;
    this.raf = 0;
    this.dirty = false;
    this._clear();
    this._attach();
  }
  _clear() {
    this.ctx.fillStyle = this.bg;
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.lineCap = "round";
    this.ctx.lineJoin = "round";
  }
  clear() {
    this._clear();
    this._fire();
  }
  setStroke(w) {
    this.strokeW = w;
  }
  _pos(e) {
    const r = this.canvas.getBoundingClientRect();
    return [
      ((e.clientX - r.left) / r.width) * this.canvas.width,
      ((e.clientY - r.top) / r.height) * this.canvas.height,
    ];
  }
  _dot([x, y]) {
    this.ctx.fillStyle = this.fg;
    this.ctx.beginPath();
    this.ctx.arc(x, y, (this.strokeW * this.ss) / 2, 0, 6.284);
    this.ctx.fill();
  }
  _seg(from, to) {
    const mx = (from[0] + to[0]) / 2, my = (from[1] + to[1]) / 2;
    this.ctx.strokeStyle = this.fg;
    this.ctx.lineWidth = this.strokeW * this.ss;
    this.ctx.beginPath();
    this.ctx.moveTo(from[0], from[1]);
    this.ctx.quadraticCurveTo(from[0], from[1], mx, my);
    this.ctx.lineTo(mx, my);
    this.ctx.stroke();
  }
  _attach() {
    this.canvas.style.touchAction = "none";
    this.canvas.addEventListener("pointerdown", (e) => {
      this.drawing = true;
      const p = this._pos(e);
      this.last = p;
      this._dot(p);
      try { this.canvas.setPointerCapture(e.pointerId); } catch {}
      e.preventDefault();
      this._fire();
    });
    this.canvas.addEventListener("pointermove", (e) => {
      if (!this.drawing) return;
      const p = this._pos(e);
      this._seg(this.last, p);
      this.last = p;
      this._fire();
    });
    const end = () => {
      if (!this.drawing) return;
      this.drawing = false;
      this._fire();
    };
    this.canvas.addEventListener("pointerup", end);
    this.canvas.addEventListener("pointercancel", end);
  }
  onChange(cb) {
    this.cb = cb;
  }
  _fire() {
    this.dirty = true;
    if (this.raf) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      if (this.dirty) {
        this.dirty = false;
        this.cb?.(this.pixels());
      }
    });
  }
  inferNow() {
    this.dirty = false;
    if (this.cb) this.cb(this.pixels());
  }
  pixels() {
    this.dctx.imageSmoothingEnabled = true;
    this.dctx.imageSmoothingQuality = "high";
    this.dctx.fillStyle = "#000";
    this.dctx.fillRect(0, 0, this.W, this.H);
    this.dctx.drawImage(this.canvas, 0, 0, this.W, this.H);
    const d = this.dctx.getImageData(0, 0, this.W, this.H).data;
    const out = new Float32Array(this.W * this.H);
    for (let i = 0; i < out.length; i++) {
      const o = i * 4;
      out[i] = (d[o] * 0.299 + d[o + 1] * 0.587 + d[o + 2] * 0.114) / 255;
    }
    return out;
  }
}

export function uploadButton(labelText, { W = 64, H = 64 } = {}, cb) {
  const wrap = document.createElement("label");
  wrap.style.cssText = "display:inline-flex;align-items:center;gap:8px;font-size:11px;color:var(--dim,#a1a1aa);font-family:monospace;cursor:pointer;border:1px solid var(--edge,#27272a);border-radius:6px;padding:4px 10px";
  wrap.textContent = labelText || "upload image";
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/*";
  input.style.display = "none";
  wrap.appendChild(input);
  const off = document.createElement("canvas");
  off.width = W;
  off.height = H;
  const octx = off.getContext("2d", { willReadFrequently: true });
  input.addEventListener("change", () => {
    const f = input.files?.[0];
    if (!f) return;
    const url = URL.createObjectURL(f);
    const im = new Image();
    im.onload = () => {
      octx.fillStyle = "#000";
      octx.fillRect(0, 0, W, H);
      const s = Math.min(W / im.width, H / im.height);
      const dw = im.width * s, dh = im.height * s;
      octx.imageSmoothingEnabled = true;
      octx.imageSmoothingQuality = "high";
      octx.drawImage(im, (W - dw) / 2, (H - dh) / 2, dw, dh);
      URL.revokeObjectURL(url);
      const d = octx.getImageData(0, 0, W, H).data;
      const px = new Float32Array(W * H);
      for (let i = 0; i < px.length; i++) {
        const o = i * 4;
        px[i] = (d[o] * 0.299 + d[o + 1] * 0.587 + d[o + 2] * 0.114) / 255;
      }
      cb?.(px, off);
    };
    im.src = url;
  });
  return wrap;
}
