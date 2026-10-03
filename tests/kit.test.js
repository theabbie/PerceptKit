import * as kit from "../docs/kit/synth.js";

let failures = 0;
function check(name, cond, detail = "") {
  if (cond) console.log(`  ok ${name}`);
  else { failures++; console.log(`  FAIL ${name} ${detail}`); }
}

const W = 32, H = 32;

{
  const img = kit.blank(W, H);
  kit.circle(img, W, H, 16, 16, 8, { fill: 1 });
  const cov = kit.coverage(img, 0.5);
  const expect = Math.PI * 64 / (W * H);
  check("circle fill coverage", Math.abs(cov - expect) / expect < 0.15, `cov=${cov.toFixed(3)} expect=${expect.toFixed(3)}`);
  const [cx, cy] = kit.centroid(img, W, H);
  check("circle centroid", Math.abs(cx - 15.5) < 1.2 && Math.abs(cy - 15.5) < 1.2, `c=${cx.toFixed(1)},${cy.toFixed(1)}`);
  const b = kit.bbox(img, W, H);
  check("circle bbox", Math.abs(b[2] - 17) <= 2 && Math.abs(b[3] - 17) <= 2, `bbox=${b}`);
}

{
  const img = kit.blank(W, H);
  kit.line(img, W, H, 2, 16, 29, 16, { w: 3 });
  const cov = kit.coverage(img, 0.5);
  const expect = (28 * 3 + 7) / (W * H);
  check("line coverage ~ w*len+caps", Math.abs(cov - expect) / expect < 0.35, `cov=${cov.toFixed(3)}`);
  const b = kit.bbox(img, W, H);
  check("line thickness bbox", b[3] <= 4 && b[3] >= 2, `h=${b[3]}`);
}

{
  const img = kit.blank(W, H);
  kit.rect(img, W, H, 8, 8, 12, 6, { fill: 1 });
  const cov = kit.coverage(img, 0.5);
  check("rect fill coverage", Math.abs(cov - 72 / (W * H)) < 0.02, `cov=${cov.toFixed(3)}`);
}

{
  const img = kit.blank(W, H);
  kit.text(img, W, H, "8", 12, 12, { scale: 2 });
  const cov = kit.coverage(img, 0.5);
  check("glyph coverage nonzero", cov > 0.01 && cov < 0.3, `cov=${cov.toFixed(3)}`);
  const b = kit.bbox(img, W, H);
  check("glyph bbox within 10x14", b[2] <= 11 && b[3] <= 15, `bbox=${b}`);
}

{
  let s = 42;
  const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const img = kit.blank(W, H).fill(1);
  kit.noise(img, rng, 0.2);
  const zeros = 1 - kit.coverage(img, 0.5);
  check("noise flip fraction", Math.abs(zeros - 0.2) < 0.05, `flipped=${zeros.toFixed(3)}`);
}

{
  const img = kit.blank(W, H);
  kit.circle(img, W, H, 10, 10, 4, { fill: 1 });
  const moved = kit.translate(img, W, H, 8, 5);
  const [cx, cy] = kit.centroid(moved, W, H);
  check("translate shifts centroid", Math.abs(cx - 17.5) < 1.5 && Math.abs(cy - 14.5) < 1.5, `c=${cx.toFixed(1)},${cy.toFixed(1)}`);
  check("translate returns new array", moved !== img);
}

{
  const img = kit.blank(W, H);
  kit.rect(img, W, H, 6, 12, 20, 6, { fill: 1 });
  const rot = kit.rotate(img, W, H, 90);
  const b = kit.bbox(rot, W, H);
  check("rotate 90 swaps aspect", b[3] > b[2], `bbox=${b}`);
}

{
  const img = kit.blank(W, H);
  kit.circle(img, W, H, 16, 16, 8, { fill: 1 });
  kit.occlude(img, W, H, 8, 8, 8, 8, 0);
  check("occlude zeroes region", img[10 * W + 10] === 0 && img[21 * W + 21] > 0.5);
}

{
  const img = kit.blank(W, H);
  kit.circle(img, W, H, 16, 16, 8, { fill: 1 });
  const th = kit.threshold(kit.blur3(img, W, H), 0.5);
  check("blur+threshold binary", [...th].every((v) => v === 0 || v === 1));
}

{
  const img = kit.blank(W, H);
  kit.polygon(img, W, H, [[4, 4], [28, 4], [16, 28]], { fill: 1 });
  const cov = kit.coverage(img, 0.5);
  const expect = (0.5 * 24 * 24) / (W * H);
  check("polygon triangle coverage", Math.abs(cov - expect) / expect < 0.2, `cov=${cov.toFixed(3)} expect=${expect.toFixed(3)}`);
}

{
  const img = kit.blank(W, H);
  kit.bezier(img, W, H, [[2, 28], [16, 2], [30, 28]], { w: 2 });
  check("bezier draws", kit.coverage(img, 0.3) > 0.01);
}



{
  const a = kit.blank(W, H);
  kit.circle(a, W, H, 12, 12, 5, { fill: 0.8 });
  const b = kit.blank(W, H);
  kit.circle(b, W, H, 20, 20, 5, { fill: 0.6 });
  kit.mergeMax(a, b);
  check("mergeMax keeps max", a[12 * W + 12] >= 0.79 && a[20 * W + 20] >= 0.59);
}

{
  const img = kit.blank(W, H);
  kit.grid(img, W, H, { rows: 4, cols: 4 });
  check("grid draws lines", img[4 * W + 8] > 0.3 && img[8 * W + 4] > 0.3);
  check("grid leaves cells empty", img[4 * W + 4] === 0);
}

{
  const base = kit.blank(W, H);
  const lay = kit.blank(W, H);
  kit.circle(lay, W, H, 16, 16, 6, { fill: 1 });
  kit.over(base, lay, 0.5);
  const c = base[16 * W + 16];
  check("over blends", c > 0.4 && c < 0.7, `v=${c.toFixed(2)}`);
}

console.log(failures ? `\n${failures} KIT TESTS FAILED` : "\nALL KIT TESTS PASSED");
process.exit(failures ? 1 : 0);
