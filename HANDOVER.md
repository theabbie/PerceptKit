# PerceptKit — Handover to the next agent

You are inheriting a working, deployed, honestly-tested system. This document
exists because the most important knowledge here is not in the code — it is the
failure history that explains why every guard exists, the owner's principles
that decide every trade-off, and the process that keeps the project honest.
Read all of it before touching anything.

## 1. What PerceptKit is (the vision)

A browser-native transformer library for self-supervised perception. No server,
no GPU cluster, no training-data bottleneck: the browser generates its own
supervision (corrupt a document and rank the pairs, grayscale an image and
learn colorization back, mask a region and learn to fill it) and a small
transformer learns from it entirely client-side.

The proof surface is the agent demo at https://theabbie.github.io/PerceptKit/:
you type a perception goal in plain English, an LLM (google/gemini-2.5-flash
through the webiome bridge, vendored pi-agent-core) designs the data generator,
corruptor, probe and interactive demo, the wasm transformer trains live in your
browser, and you get an interactive demo plus a downloadable `.pkm` model.

The deeper goal the owner is evaluating: can an agent build ML systems
end-to-end, autonomously, and be TRUSTED — because it reports failures as
loudly as successes. Every design decision below serves that.

## 2. What exists right now (verified as of this handover)

- `crates/percept-core` — pure Rust: tensors with explicit forward/backward,
  pre-LN transformer encoder (fused-QKV MHSA, GELU MLP), global head
  (mean-pool → score), dense head (per-token), Adam, MSE/L1/pairwise-margin/
  dense-masked losses, patchify/unpatchify, PKM1 versioned serialization.
  40 Rust tests pass (unit, convergence, finite-difference gradchecks).
- `crates/percept-wasm` + `pkg/` + `pkg-web/` — wasm-bindgen surface:
  PerceptModelWasm (new/load/save/score/scoreBatch/transform/nParams/
  configJson/attentionMaps), TrainerWasm (trainScoreEpoch/trainPairEpoch/
  trainDenseEpoch/evalPairAccuracy), patchifyImage, corruption helpers.
  Binary ~244KB. Published as @theabbie/perceptkit@0.1.0 on GitHub Packages.
- `docs/` — the deployed site (GitHub Pages, master → docs/):
  - Three static demos: digit classify (94% held-out), text-overlap scorer
    (100% pair acc), grayscale→color colorizer (MAE 0.06 vs 0.28 baseline).
  - The agent UI: kanban pipeline (Plan / Code / Train / Evaluate / Export /
    Demo), one card per tool call, failures visible in place. Tools:
    plan_label_set, write_corruptor, write_probe, run_pipeline, write_demo,
    export_model. `docs/kit/` (synth.js, canvas.js DrawPad, viz.js) is the
    LLM's runtime toolbox. p5.js vendored for custom drag/interaction demos.
  - Gates that protect honesty (see §4): diversity gate, chance gate,
    corruptor contract, demo-must-call-infer check, re-plan guard,
    blocked-vs-error card states.
  - `docs/USECASES.md` — 70 use cases across 10 domains, mapped to
    capability buckets. Not yet surfaced on the site (open item).
- `tests/` — six JS suites, all green: syntax, kit, agent-pipeline,
  demo-engine, demo-ui, e2e.
- Deployed version at handover: `?v=1.11`, HEAD `37ce38c`. Site boots clean
  (status ready, agent loaded, p5 present — re-verified today).

## 3. The owner's principles — non-negotiables

Abhishek (theabbie) is technically sophisticated and deeply skeptical of AI
hype. These are not preferences; violating any of them is a project failure
even if the code works:

1. **Honest engineering above all.** No overclaiming. Limits stated plainly.
   When you claim something works, you must have watched it work on the
   deployed site. Answer the strongest version of his argument, never a
   strawman, and never inflate.
2. **Accountability is not a joke.** Bugs you find get surfaced, not hidden.
   Every bug fix ships with its story (the commit messages are the audit
   trail — read `git log` to see the standard). If you break something,
   the report says so first.
3. **Always test the DEPLOYED site, not localhost.** GitHub Pages serves
   differently (Jekyll, caching, node_modules quirks). Multiple real bugs
   existed ONLY on the deployed site.
4. **Errors must never look normal.** A dependency that isn't ready renders
   as BLOCKED (amber, "waiting on earlier step"), never as ERROR (red).
   A chance-level model must fail loudly, never export, never demo.
   Repeated identical failures must force a re-plan — the system must never
   get stuck, and must never present failure as success.
5. **Tests must verify real behavior.** Assert on held-out metrics, never
   train accuracy (train overfits and lies — it hits 100% even on broken
   data). A smoke test that doesn't exercise the full call path is a lie.
6. **No code comments.** Prose lives in markdown (README, USECASES, this
   file). JS/Rust source stays comment-free — enforced historically by
   scripts/strip-comments.mjs.
7. **Show, don't tell.** Live runs, screenshots, real numbers, orderings
   that pass the eyeball test. A metric nobody can eyeball is suspect.
8. **Briefs are short; execution is yours.** He gives two-line requirements
   and expects you to infer scope, drive autonomously, and own the
   implementation path. He holds accountability on OUTCOMES — honest
   numbers, transparent bug reporting — not on how you got there.
9. **Honest disagreement beats capitulation.** Push back on factual errors
   with evidence. Never hedge, never flatter, never sales-talk. But when
   he's right about a failure, concede plainly and fix it.
10. **An honest failure is a correct outcome.** "No model was produced
    because training stayed at chance" is SUCCESS for the system. A demo
    that shows random verdicts is the worst possible outcome — it lies.

## 4. What the code can't tell you — why every guard exists

Every gate in this repo is scar tissue. Do not remove or weaken one to make
a run pass. Each pairs a real production failure with its lesson:

- **Type-aware `toImg()` + full-path corruptor smoke test.** The pipeline once
  double-rendered already-rendered images: `render()` read `.cx` of a
  Float32Array → undefined → rings at NaN coordinates → all-zero training
  data. Loss 0.5, train acc 0%, while held-out read "100%" (ring-vs-blank
  garbage). A green-looking run with an untrained model. Lesson: a metric can
  be structurally broken while looking perfect — sanity-check end-to-end
  (clean must beat corrupt AND train acc must beat chance).
- **Error messages must carry the DATA, not nicer wording.** When LLM code
  crashed deep in the pair loop, wasm stripped JS stacks and the model looped
  rewriting blind. Dumping the shapes of the actual inputs (`sample=`,
  `mild=`, `severe=` types) fixed it in ONE retry. Lesson: when an agent
  loops on an error, give it the data that caused the error.
- **Corruptor contract: severity degrades the CONCEPT, never placement.**
  A corruptor once treated translate/rotate as defects, so "clean" meant
  "centered" — a single dot scored higher than a hand-drawn circle. Fix:
  neutral augmentation applied to BOTH sides of every pair (translate ±0.4W,
  rotate ±40°, scale 0.8–1.2) plus a contract: severity~1 means the sample
  barely belongs to its class (deform the shape itself: wobble, broken arcs,
  collapse toward a blob), not just pixel noise. Augmentation suppresses
  placement leakage; the contract is the real defense. Honest residual:
  pathological placement-only corruptors still leak ~73%.
- **Diversity gate.** An LLM-written gen drew ONE fixed layout per class
  (overlap always at the same two coordinates). The model memorized two
  images, held-out was 50%, and the demo's verdicts were out-of-distribution
  noise. Fix: plan_label_set draws 5 samples per label and rejects gens
  whose mean pairwise pixel diff < 0.02, with an actionable message.
- **demoInfer indexing (outCh).** Per-token model output width is
  patch² × out_channels, NOT the number of labels. When the LLM set
  out_channels=16 for a 2-class task, the demo summed wrong offsets and
  returned a CONSTANT verdict for every input — the owner's "not remotely
  accurate" report. demoSpec carries outCh now. It worked before only by
  coincidence (out_channels defaulting to nLabels).
- **Chance gate.** 20 corrupted held-out samples let a ~50% model pass as
  "75%" and get exported. Now 30 samples/label, and run_pipeline FAILS
  LOUDLY when held-out is within 15 points of chance (dense) or below 70%
  (score_pair); state.result is cleared so export physically cannot ship a
  failed model. Verified live twice: runs at 63% and 50% were refused,
  export stayed blocked, and the agent said plainly that no model was
  produced.
- **write_demo must call ctx.infer.** An LLM-written drag demo computed the
  overlap verdict geometrically in demo code — the trained model was
  decorative. The tool now rejects demo code that never calls ctx.infer:
  the verdict must come from the model, or the demo is a lie.
- **tests/syntax.test.js.** `node --check file.js` SILENTLY EXITS 0 without
  parsing when a .js file uses ESM syntax outside a module package. It let
  an unparseable agent-core.js (unescaped quotes inside a string) deploy
  and stick the page at "loading wasm". The gate pipes every docs/*.js
  through `node --input-type=module --check`, which catches that class.
- **Cache busting.** Every deploy must bump `?v=N` in ALL docs/*.js and
  index.html. Same-version query strings hit browser HTTP cache and mask
  the fix — this bit the project twice.
- **Pairwise ranking over absolute regression** when corruption has
  stochastic components: seed picks WHICH boxes move, severity picks HOW
  MUCH — same displacement, different severities is irreducible ambiguity
  (Bayes error of the data design). Shared-base pairs cancel the noise:
  100% accuracy where absolute regression plateaued.
- **Below-chance held-out = spurious feature**, not underfitting: the model
  learned something anti-correlated with the target out-of-distribution.
  Check generative-process differences between train/test seeds first.
- **Adversarial iteration rules:** severity escalation can destroy signal
  (cap ~0.55); always track and return the BEST round's model, not the last
  (rounds can regress 65%→52%); early-exit when nothing is mined and
  accuracy > 90%; mine hard cases from probe output (correct labels), not
  held-out mismatches (double corruption corrupts the label).
- **Data diversity beats model capacity** on these tasks (56%→79% from data,
  nothing from a bigger model). Augmentation must preserve task signal —
  row jitter on 8×10 digits destroyed the shape cue and hurt accuracy.

## 5. Known limitations — state these, never hide them

- The tiny wasm transformer (16-dim, 2 heads, 2 layers, patch tokens)
  CANNOT learn overlap-vs-separate of two circles across randomized
  layouts. Three independent LLM attempts with diverse gens landed 48–63%
  held-out. The system now fails honestly instead of shipping a fake.
  Making it learnable likely needs a global-feature head (e.g. row/col
  occupancy pooling before the encoder) or larger dModel — that is the
  top roadmap item, not a claim.
- Colorizer is region-level only (horizon, water); no object semantics.
- Placement leakage is suppressed, not eliminated (see corruptor contract).
- Iteration badge occasionally sticks "–"; canvas can render blank when the
  model is null during probe timing. UX polish, not correctness.
- wasm binary ~244KB, no wasm-opt pass yet.
- USECASES.md (70 cases) is not surfaced on the site — the owner asked to
  see it; linking it from the header is an open small task.

## 6. How to work (process)

- **Deploy** = commit to master, push; GitHub Pages serves docs/. Bump
  `?v=` everywhere. Verify in order: (1) curl the new marker string on the
  deployed JS, (2) browser boot check — `#agStatus` shows
  "ready — describe a goal and Run" and `window.__agent` exists, (3) run a
  real goal end-to-end. A deploy is not done until (3) passes.
- **The loop for any change**: local suites → deploy → deployed-site test →
  report with real numbers. If the deployed test finds a bug, that is the
  most valuable outcome of the iteration — surface it, fix it, add a
  regression test, re-deploy, re-verify.
- **Commits carry the why.** Detailed messages naming the failure, root
  cause, and fix. They are the project's memory.
- **The LLM is stochastic.** One green run is not proof. Run goals 2–3×
  before claiming a behavior works. A run that needed retries is fine —
  visible recovery is a feature; report it honestly.
- **Sandbox fragility**: Chrome and /tmp die on suspend; uncommitted
  MODIFIED tracked files revert to HEAD (untracked new files survive).
  Commit early, commit often. Relaunch Chrome headless on 9222 and create
  tabs via `curl -X PUT .../json/new?url` (the cdp CLI grabs the first page
  target — close stale tabs to retarget).
- **Security note**: an npm PAT was shared in old chat history for the
  package publish. It was never committed, but the owner should revoke and
  rotate it out-of-band.

## 7. How to test (the guide)

Local, in order — `node tests/syntax.test.js` FIRST (the node --check trap),
then kit, agent-pipeline, demo-engine, demo-ui, e2e. All six must pass.
Rust: `cargo test` (source $HOME/.cargo/env; use --manifest-path since
shells reset cwd).

Deployed E2E (headless Chrome + CDP, the cdp CLI):

- Boot check: `cdp eval` on `#agStatus` + `window.__agent` + `typeof p5`.
- **Circle goal** ("score quality of drawn circle"): expect all-green cards,
  high held-out, and the eyeball ordering — draw via pointerdown/move/up on
  `window.__demo.canvas`, read scores from `#body-demo .ag-stats`:
  clean circle > scribble > single dot. If a dot beats a circle, the
  corruptor leaked placement or texture-only severity — diagnose the run's
  corruptor from the transcript before touching code.
- **Drag goal** ("check if two shapes overlap, create demo where I can drag
  shapes"): TWO correct outcomes — (a) a working drag demo whose verdict
  tracks the drag AND comes from ctx.infer, or (b) honest failure: chance
  gate refuses, export stays blocked, agent reports no model produced.
  A demo with random/constant verdicts is a BUG (check demoSpec.outCh
  indexing and whether the demo rasterizer matches the gen distribution).
- **Card states**: dependency failures render BLOCKED (amber) with "waiting
  on earlier step"; real failures render ERROR (red). Never the reverse.
- **Re-plan guard**: a repeated identical tool error must produce the
  "STOP guessing" augmented error and a fresh plan_label_set.
- **Direct model probing** (fastest diagnosis): `cdp eval` with
  `model.transform(tokens, T)`, summing `raw[t*OUT + c]` where
  OUT = patch² × out_channels; `model.score(tokens, T)` for score models.
  Sweep a parameter (e.g. circle separation) and check the verdict tracks
  it. A constant margin for every input = indexing/width bug, not a bad
  model.
- **Transcript extraction**: `window.__agent._agent._state.messages`,
  toolCall parts carry `arguments.code` — read what the LLM ACTUALLY wrote
  before blaming the pipeline. Diagnose from evidence.
- Monitors: watch `#agStatus` for the TRANSITION into "run complete"
  (track previous state — a fresh tab starts at "ready"; matching a stale
  status produces false completion). Treat "run finished without a trained
  model — see log" as a terminal, legitimate outcome and read `#agLog`.

## 8. Your mandate

Self-iterate for as long as you can. Pick from the roadmap (or find better
work from your own deployed-site testing), execute autonomously through the
full loop, and report honestly. Concretely, in rough priority:

1. Make spatial-relation tasks learnable: global-feature head (row/col
   occupancy pooling) or capacity increase — then the drag demo can pass
   for real instead of failing honestly. This is the flagship open item.
2. Surface USECASES.md on the site (header link or a lane) so the 70 cases
   are visible.
3. UX polish: iteration badge flakiness, null-model canvas edge cases.
4. wasm-opt size pass; richer demo scenarios; layout-aware colorizer.

Never compromise: the gates are load-bearing. If the only way to make a run
green is weakening a gate, the run is supposed to fail. Ship the honest
failure, write the limitation down, and move to work that makes it genuinely
passable. Keep testing the deployed site after every change — that is where
every important bug in this project's history was found. Update this file
when you learn something the code can't tell.

The standard, in one line: the system's failures must be as trustworthy as
its successes.
