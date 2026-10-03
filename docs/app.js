

import init, * as wasm from "./pkg-web/percept_wasm.js";
import {
  trainClassifier, classify, makeDigitDataset,
  genDoc, renderDoc, corruptDoc, trainOverlapScorer, docScore,
  genScene, sceneGray, trainColorizer, colorize,
  DIGIT_W, DIGIT_H, DOC_W, DOC_H, SCENE_W, SCENE_H,
  rng,
} from "./engine.js";

const $ = (id) => document.getElementById(id);




document.querySelectorAll("nav button").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll("nav button").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll("main section").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    $(b.dataset.tab).classList.add("active");
  });
});




function drawGray(canvas, img, w, h) {
  const ctx = canvas.getContext("2d");
  const id = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const v = Math.max(0, Math.min(1, img[i])) * 255;
    id.data[i * 4] = id.data[i * 4 + 1] = id.data[i * 4 + 2] = v;
    id.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}

function drawRGB(canvas, img, w, h) {
  const ctx = canvas.getContext("2d");
  const id = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    for (let c = 0; c < 3; c++) {
      id.data[i * 4 + c] = Math.max(0, Math.min(1, img[i * 3 + c])) * 255;
    }
    id.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}




let clfModel = null;

$("clfTrain").addEventListener("click", async () => {
  const btn = $("clfTrain");
  btn.disabled = true;
  $("clfStatus").textContent = "generating data + training…";
  try {
    const t0 = performance.now();
    const { model, lastLoss } = await trainClassifier(wasm, {}, (e, epochs, loss) => {
      $("clfBar").style.width = `${((e + 1) / epochs) * 100}%`;
      $("clfStatus").textContent = `epoch ${e + 1}/${epochs}, loss ${loss.toFixed(4)}`;
    });
    clfModel = model;
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    const testSet = makeDigitDataset(10, 999);
    let correct = 0;
    for (const s of testSet) {
      if (classify(wasm, clfModel, s.img).digit === s.label) correct++;
    }
    $("clfStatus").textContent =
      `trained in ${secs}s (final loss ${lastLoss.toFixed(4)}), held-out accuracy ${(correct / testSet.length * 100).toFixed(0)}%`;
    $("clfPlay").classList.remove("hidden");
    updateClf();
  } catch (err) {
    $("clfStatus").textContent = "error: " + err.message;
  }
  btn.disabled = false;
});


const gridState = new Uint8Array(DIGIT_W * DIGIT_H);
(function buildGrid() {
  const grid = $("digitGrid");
  for (let i = 0; i < DIGIT_W * DIGIT_H; i++) {
    const b = document.createElement("button");
    b.addEventListener("click", () => {
      gridState[i] ^= 1;
      b.classList.toggle("on", !!gridState[i]);
      updateClf();
    });
    grid.appendChild(b);
  }
})();

function gridImage() {
  const img = new Float32Array(DIGIT_W * DIGIT_H);
  for (let i = 0; i < img.length; i++) img[i] = gridState[i];
  return img;
}

function updateClf() {
  if (!clfModel) return;
  const { digit, scores } = classify(wasm, clfModel, gridImage());
  const tbl = $("clfScores");
  tbl.innerHTML = "";
  const max = Math.max(...scores.map(Math.abs)) || 1;
  for (let d = 0; d < 10; d++) {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td class="digit">${d}</td><td class="track"><div><div style="width:${(Math.max(0, scores[d]) / max) * 100}%"></div></div></td><td class="val">${scores[d].toFixed(2)}</td>`;
    tbl.appendChild(tr);
  }
  $("clfVerdict").textContent = digit;
}

$("clfClear").addEventListener("click", () => {
  gridState.fill(0);
  document.querySelectorAll("#digitGrid button").forEach((b) => b.classList.remove("on"));
  updateClf();
});




let scModel = null;
let scBase = null;
const scR = rng(777);

function newDoc() {
  scBase = genDoc(scR);
  updateSc();
}

function updateSc() {
  if (!scModel || !scBase) return;
  const sev = $("scSev").value / 100;
  $("scSevVal").textContent = sev.toFixed(2);
  const cleanImg = renderDoc(scBase);
  const badImg = renderDoc(corruptDoc(scBase, sev, scR));
  drawGray($("scClean"), cleanImg, DOC_W, DOC_H);
  drawGray($("scBad"), badImg, DOC_W, DOC_H);
  const sClean = docScore(wasm, scModel, cleanImg);
  const sBad = docScore(wasm, scModel, badImg);
  $("scCleanScore").textContent = sClean.toFixed(3);
  $("scBadScore").textContent = sBad.toFixed(3);
  const ok = sClean > sBad;
  $("scVerdict").textContent = ok
    ? "ranks clean above corrupted ✓"
    : "ranks corrupted above clean ✗";
  $("scVerdict").className = "verdict " + (ok ? "verdict-good" : "verdict-bad");
}

$("scTrain").addEventListener("click", async () => {
  const btn = $("scTrain");
  btn.disabled = true;
  $("scStatus").textContent = "training…";
  try {
    const t0 = performance.now();
    const { model, accuracy } = await trainOverlapScorer(wasm, {}, (e, epochs, loss) => {
      $("scBar").style.width = `${((e + 1) / epochs) * 100}%`;
      $("scStatus").textContent = `epoch ${e + 1}/${epochs}, loss ${loss.toFixed(4)}`;
    });
    scModel = model;
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    $("scStatus").textContent =
      `trained in ${secs}s, pair accuracy ${(accuracy * 100).toFixed(0)}%`;
    $("scPlay").classList.remove("hidden");
    newDoc();
  } catch (err) {
    $("scStatus").textContent = "error: " + err.message;
  }
  btn.disabled = false;
});

$("scSev").addEventListener("input", updateSc);
$("scNew").addEventListener("click", newDoc);




let colModel = null;
const colR = rng(888);

function newScene() {
  const rgb = genScene(colR);
  const gray = sceneGray(wasm, rgb);
  drawGray($("colGray"), gray, SCENE_W, SCENE_H);
  drawRGB($("colTrue"), rgb, SCENE_W, SCENE_H);
  if (colModel) {
    const out = colorize(wasm, colModel, gray);
    drawRGB($("colOut"), out, SCENE_W, SCENE_H);
  }
}

$("colTrain").addEventListener("click", async () => {
  const btn = $("colTrain");
  btn.disabled = true;
  $("colStatus").textContent = "training…";
  try {
    const t0 = performance.now();
    const { model, lastLoss } = await trainColorizer(wasm, {}, (e, epochs, loss) => {
      $("colBar").style.width = `${((e + 1) / epochs) * 100}%`;
      $("colStatus").textContent = `epoch ${e + 1}/${epochs}, loss ${loss.toFixed(4)}`;
    });
    colModel = model;
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    $("colStatus").textContent =
      `trained in ${secs}s (final loss ${lastLoss.toFixed(4)})`;
    $("colPlay").classList.remove("hidden");
    newScene();
  } catch (err) {
    $("colStatus").textContent = "error: " + err.message;
  }
  btn.disabled = false;
});

$("colNew").addEventListener("click", newScene);


await init();
$("clfStatus").textContent = "ready — click train.";
