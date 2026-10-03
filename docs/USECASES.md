# PerceptKit use-case matrix

PerceptKit fits small-image and short-sequence perception tasks with synthetic or procedurally described structure.
Train a compact model to classify, score, rank, or transform examples generated in-browser, then expose it through a live demo.
An LLM can author the generator, corruptor, optional renderer, probe, and interaction; the kit supplies reusable generation and visualization primitives.
The strongest fit is a narrow decision boundary grounded in geometry, layout, signal shape, or controlled visual variation—not open-ended scene understanding.

## Capability buckets

Use these shorthand bucket names in the matrix and frequency table. `gen` produces labeled or preference-paired data; `corrupt` creates controlled variation; `demo` identifies a useful probe or presentation surface.

- **P** — geometric shape primitives: lines, circles, polygons, boxes, paths, symbols.
- **S** — stroke rendering: pen trajectories, pressure/width, joins, handwriting-like traces.
- **N** — noise models: pixel noise, blur, dropout, jitter, quantization, missing samples.
- **G** — geometric transforms: scale, rotation, shear, perspective, displacement, alignment.
- **O** — occlusion and defects: masks, gaps, overlap, scratches, missing regions.
- **T** — text rendering: glyphs, fonts, baseline, spacing, synthetic text blocks.
- **R** — grids and tabular structure: cells, rows/columns, axes, ruled paper, matrices.
- **L** — multi-layer compositing: foreground/background, masks, overlays, z-order.
- **V** — plot/signal visualization: scatter, line, bars, waveform, spectrogram-like image, axes.
- **Q** — gauges and compact indicators: dials, meters, sliders, status marks.
- **C** — confusion matrices and structured label/transition grids.
- **H** — sample galleries: labeled examples and score distributions.
- **A** — A/B compare: side-by-side candidates, preference selection, before/after.
- **D** — smooth draw canvas: pointer capture, stroke preview, interactive sketch input.
- **I** — sliders: parameter control for thresholds, noise, geometry, or signal settings.
- **U** — upload: user-supplied small image or short signal file, with clear format/size limits.
- **W** — wasm training/inference: dense_mse classification-style per-token loss or score_pair pairwise ranking.

## Use cases

### Drawing, handwriting, and gestures

- Circle/square drawing quality score — gen: parametric shape; corrupt: wobble/gap/aspect warp; demo: draw canvas — needs: P, S, N, G, D, H, W.
- Closed-loop completion score — gen: closed paths; corrupt: endpoint gap/overshoot; demo: draw canvas — needs: P, S, O, D, W.
- Straight-line steadiness ranking — gen: line trajectories; corrupt: curvature/jitter; demo: A/B stroke compare — needs: S, N, A, D, W.
- Handwritten digit style classification — gen: rendered digits; corrupt: stroke width/slant/erase; demo: draw canvas — needs: S, T, N, G, D, H, W.
- Handwritten symbol disambiguation — gen: paired glyph-like symbols; corrupt: local stroke edits; demo: draw canvas — needs: S, P, N, D, H, W.
- Signature-like stroke consistency — gen: procedural pen traces; corrupt: timing/shape variation; demo: stroke gallery — needs: S, N, G, H, D, W.
- Gesture direction classification — gen: short trajectory templates; corrupt: jitter/scale/missing points; demo: draw canvas — needs: S, G, N, D, W.
- Sketch-to-template matching — gen: object-outline paths; corrupt: rotation/partial erasure; demo: sample gallery — needs: P, S, G, O, H, D, W.

### Documents and layout

- Form field boundary quality — gen: synthetic forms; corrupt: skew/line breaks/overprint; demo: upload crop — needs: R, T, G, O, L, U, W.
- Checkbox and radio mark classification — gen: boxes and marks; corrupt: tick/cross/partial fill; demo: upload or draw — needs: P, S, N, U, D, W.
- Table cell alignment score — gen: tables with text; corrupt: shifted rules/unequal cells; demo: A/B compare — needs: R, T, G, A, U, W.
- Document column-count classification — gen: multi-column pages; corrupt: gap/width/noise changes; demo: sample gallery — needs: R, T, G, N, H, U, W.
- Text-line baseline quality — gen: synthetic lines; corrupt: waviness/rotation/spacing; demo: upload crop — needs: T, G, N, U, H, W.
- Paragraph spacing consistency — gen: text block layouts; corrupt: irregular leading/indent; demo: A/B compare — needs: T, R, G, A, U, W.
- Page margin and crop anomaly score — gen: page frames and text; corrupt: crop/offset/edge loss; demo: upload crop — needs: T, R, G, O, U, W.
- Flowchart connector validity — gen: nodes and paths; corrupt: broken/overlapping connectors; demo: sample gallery — needs: P, L, O, H, U, W.

### Image restoration and quality

- Synthetic image denoise preference — gen: simple textured patches; corrupt: Gaussian/speckle noise; demo: before/after — needs: P, N, L, A, U, W.
- Blur severity classification — gen: edges and patterns; corrupt: kernel blur levels; demo: score gauge — needs: P, N, Q, H, U, W.
- Compression artifact severity — gen: grids/textures; corrupt: block/quantization artifacts; demo: A/B patches — needs: R, P, N, A, U, W.
- Image sharpness ranking — gen: edge patterns; corrupt: blur/resampling; demo: A/B compare — needs: P, N, A, U, W.
- Small icon edge cleanliness — gen: vector-like icons; corrupt: jagged edges/halo; demo: sample gallery — needs: P, N, G, H, U, W.
- Color-region separation score — gen: layered flat regions; corrupt: palette shift/bleed; demo: upload crop — needs: P, L, N, U, W.
- Mask boundary quality — gen: shapes plus binary masks; corrupt: dilation/erosion/gaps; demo: overlay compare — needs: P, O, L, A, U, W.
- Upscaling candidate preference — gen: pixel patterns; corrupt: nearest/bilinear-like variants; demo: A/B compare — needs: P, N, A, U, W.

### Time-series and signals as images

- Waveform event class — gen: short pulse/ramp templates; corrupt: noise/drift/dropout; demo: waveform plot — needs: V, N, G, H, U, W.
- Sensor trace anomaly score — gen: baseline and injected spikes; corrupt: drift/gaps/outliers; demo: waveform plot — needs: V, N, O, Q, U, W.
- Short ECG-like beat morphology class — gen: abstract beat curves; corrupt: amplitude/time warp/noise; demo: line plot — needs: V, G, N, H, U, W.
- Audio spectrogram pattern class — gen: synthetic bands/chirps; corrupt: masking/frequency shift; demo: heatmap — needs: V, N, O, H, U, W.
- Periodic signal stability ranking — gen: short periodic curves; corrupt: jitter/dropout; demo: A/B compare — needs: V, N, A, U, W.
- Signal threshold-crossing quality — gen: thresholded pulses; corrupt: baseline wander/noise; demo: slider and plot — needs: V, N, I, Q, U, W.
- Short motion trace gesture class — gen: 2D point sequences; corrupt: speed/rotation/jitter; demo: trajectory canvas — needs: P, S, G, N, D, W.

### UI/UX and interaction quality

- Button hit-target geometry score — gen: control rectangles; corrupt: small/overlapping targets; demo: click probe — needs: P, L, O, D, H, W.
- Slider track and thumb alignment — gen: slider variants; corrupt: offset/scale/contrast; demo: interactive slider — needs: P, Q, G, I, D, W.
- Toggle state icon classification — gen: paired states; corrupt: low contrast/crop; demo: sample gallery — needs: P, L, N, H, U, W.
- UI spacing consistency — gen: component grids; corrupt: margin/padding perturbation; demo: A/B compare — needs: R, P, G, A, U, W.
- Text legibility in compact controls — gen: short labels; corrupt: small font/contrast/blur; demo: gallery and score — needs: T, N, H, U, W.
- Focus indicator visibility score — gen: controls with focus rings; corrupt: thickness/color/occlusion; demo: A/B compare — needs: P, L, O, A, U, W.
- Pointer trajectory completion quality — gen: target and path pairs; corrupt: overshoot/jitter; demo: draw/click canvas — needs: P, S, N, D, H, W.

### Games and interaction

- Platformer jump-arc feasibility — gen: short arc trajectories; corrupt: speed/gravity variation; demo: plot and slider — needs: V, G, I, H, W.
- Grid puzzle move validity — gen: board states and moves; corrupt: blocked/adjacent cell edits; demo: clickable grid — needs: R, P, D, H, W.
- Game sprite pose classification — gen: geometric silhouettes; corrupt: shift/limb occlusion; demo: sample gallery — needs: P, L, O, G, H, U, W.
- Aim trajectory ranking — gen: target/path pairs; corrupt: jitter/overshoot; demo: draw canvas — needs: P, S, N, A, D, W.
- Rhythm timing window classification — gen: beat ticks and cursor marks; corrupt: timing jitter/dropout; demo: timeline — needs: R, V, N, D, W.
- Tile-map local pattern classification — gen: symbolic tile grids; corrupt: substitutions/occlusion; demo: interactive grid — needs: R, P, O, H, D, W.

### Science, plots, and technical diagrams

- Scatter-plot trend category — gen: controlled point clouds; corrupt: outliers/noise/scale; demo: scatter plot — needs: V, N, G, H, U, W.
- Line-plot crossing count class — gen: short curves; corrupt: missing segments/axis warp; demo: line plot — needs: V, G, O, H, U, W.
- Bar-chart ordering validity — gen: bars and labels; corrupt: swaps/axis truncation; demo: A/B compare — needs: V, T, G, A, U, W.
- Error-bar visibility score — gen: plotted intervals; corrupt: overlap/low contrast; demo: plot gallery — needs: V, L, O, H, U, W.
- Heatmap block-pattern classification — gen: structured value grids; corrupt: noise/missing cells; demo: heatmap — needs: R, V, N, O, H, U, W.
- Equation layout alignment score — gen: synthetic symbol rows; corrupt: baseline/spacing shifts; demo: upload crop — needs: T, G, N, U, W.
- Circuit diagram local connectivity check — gen: nodes/wires; corrupt: gaps/crossings; demo: layered diagram — needs: P, L, O, H, U, W.

### Manufacturing and inspection

- PCB trace continuity score — gen: traces and pads; corrupt: breaks/bridges; demo: mask overlay — needs: P, O, L, H, U, W.
- Solder joint shape category — gen: abstract pad/joint shapes; corrupt: excess/missing material; demo: sample gallery — needs: P, L, N, H, U, W.
- Part silhouette tolerance check — gen: dimensioned outlines; corrupt: scale/warp/notch; demo: overlay and sliders — needs: P, G, O, A, I, U, W.
- Surface scratch orientation class — gen: textured patches; corrupt: line defects/noise; demo: upload crop — needs: P, N, O, G, U, W.
- Repeated-hole spacing score — gen: circles on fixtures; corrupt: offset/diameter variance; demo: overlay compare — needs: P, G, A, U, W.
- Label/mark presence classification — gen: synthetic marks; corrupt: blur/occlusion/contrast; demo: upload crop — needs: T, L, N, O, U, W.
- Assembly part-count pattern check — gen: layered component silhouettes; corrupt: missing/extra/overlap; demo: sample gallery — needs: P, L, O, H, U, W.

### Maps, paths, and accessibility

- Route line continuity score — gen: polylines on simple map tiles; corrupt: gaps/jumps; demo: draw path — needs: P, S, O, D, U, W.
- Road intersection sketch classification — gen: junction layouts; corrupt: angle/occlusion variation; demo: sample gallery — needs: P, G, O, H, U, W.
- Map symbol legend matching — gen: symbol/legend pairs; corrupt: symbol scale/style variants; demo: probe gallery — needs: P, T, G, H, U, W.
- Path-to-target completion score — gen: short paths and targets; corrupt: deviation/overshoot; demo: draw canvas — needs: P, S, G, D, H, W.
- Tactile-style raised-line pattern classification — gen: high-contrast line motifs; corrupt: gaps/overlap/scale; demo: upload or draw — needs: P, S, O, G, D, U, W.
- High-contrast icon distinguishability — gen: paired icons; corrupt: color/size/occlusion; demo: A/B compare — needs: P, L, O, A, U, W.
- Caption-to-indicator association — gen: compact labels and gauges; corrupt: swaps/overlap; demo: interactive probe — needs: T, Q, L, O, D, W.
- Low-vision chart encoding distinction — gen: simple chart variants; corrupt: palette/line-style confusion; demo: A/B compare — needs: V, L, N, A, U, W.

### Short sequence and token-level structured data

- Delimiter and bracket sequence validity — gen: short symbolic sequences; corrupt: deletion/substitution; demo: token probe — needs: T, N, H, W.
- Short command gesture phase classification — gen: discrete pose/phase tokens; corrupt: missing/reordered tokens; demo: confusion matrix — needs: C, N, H, W.
- Compact event-log pattern ranking — gen: short event sequences; corrupt: reorder/dropout; demo: A/B sequence compare — needs: N, A, H, W.
- Tiny state-transition diagram consistency — gen: small transition grids; corrupt: missing/extra edges; demo: confusion matrix — needs: P, R, O, C, W.

## Out of scope

PerceptKit is not a substitute for large-image vision models, real-photo object recognition, OCR or document understanding at production scale, long-context sequence models, or semantic reasoning about arbitrary scenes. Synthetic generators can teach controlled visual distinctions but do not establish real-world validity; performance on uploaded samples depends on domain match, resolution, and training coverage. Tasks requiring broad labels, extensive datasets, privacy-sensitive learning, or robust physical-world generalization need a different system and validation plan.

## Capability frequency

Counts are computed across the 70 listed use cases; each use case contributes at most once to a bucket. `W` appears throughout because every case assumes wasm training/inference, while the other counts indicate reusable kit coverage priority. Module names below are planned integration targets, not a claim that all listed functionality is already implemented.

| Bucket | Use cases | Existing/planned coverage |
|---|---:|---|
| P — shape primitives | 39 | `kit/synth.js`: shapes |
| S — stroke rendering | 15 | `kit/synth.js`: stroke/path generation; `kit/canvas.js`: draw pad |
| N — noise models | 38 | `kit/synth.js`: noise |
| G — geometric transforms | 30 | `kit/synth.js`: transforms |
| O — occlusion and defects | 27 | `kit/synth.js`: masks, gaps, defects |
| T — text rendering | 14 | `kit/synth.js`: synthetic glyph/text rendering |
| R — grids and tabular structure | 12 | `kit/synth.js`: grid/layout generators |
| L — multi-layer compositing | 18 | `kit/synth.js`: layers and masks |
| V — plots and signals | 14 | `kit/viz.js`: scatter, line, waveform, heatmap plots |
| Q — gauges and indicators | 5 | `kit/viz.js`: gauges and compact indicators |
| C — confusion matrices | 2 | `kit/viz.js`: confusion matrices |
| H — sample galleries | 35 | `kit/viz.js`: labeled sample gallery |
| A — A/B compare | 18 | `kit/viz.js`: paired comparison view |
| D — smooth draw canvas | 21 | `kit/canvas.js`: pointer input and stroke preview |
| I — sliders | 4 | `kit/canvas.js` or `kit/viz.js`: interactive parameter controls |
| U — upload | 47 | `kit/canvas.js`: bounded image/signal upload and preprocessing |
| W — wasm training/inference | 70 | wasm lib: `dense_mse`, `score_pair`, training and inference |
