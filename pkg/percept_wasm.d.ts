/* tslint:disable */
/* eslint-disable */

export class PerceptModelWasm {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Attention maps for interpretability. Returns a flat f32 array:
     * layers x heads x (n_tokens * n_tokens), row-major. Use with
     * `attentionShapes` to slice.
     */
    attentionMaps(tokens_flat: Float32Array, n_tokens: number): Float32Array;
    /**
     * Shapes for slicing `attentionMaps` output:
     * [n_layers, n_heads, n_tokens, n_tokens].
     */
    attentionShapes(n_tokens: number): Uint32Array;
    /**
     * The model's config as a JSON string.
     */
    configJson(): string;
    /**
     * Load a model from bytes produced by `save`.
     */
    static load(bytes: Uint8Array): PerceptModelWasm;
    /**
     * Number of trainable parameters.
     */
    nParams(): number;
    constructor(in_channels: number, d_model: number, n_heads: number, n_layers: number, d_ff: number, out_channels: number, max_seq_len: number, seed: number);
    /**
     * Serialize the model (config + weights) to bytes.
     */
    save(): Uint8Array;
    /**
     * Batched scoring: many sequences in one call.
     * `seq_lens[i]` is the token count of sequence i.
     */
    scoreBatch(tokens_flat: Float32Array, seq_lens: Uint32Array): Float32Array;
    /**
     * Score a token sequence: flat f32 array, row-major (n_tokens x in_channels).
     * Returns the global scalar score.
     */
    score(tokens_flat: Float32Array, n_tokens: number): number;
    /**
     * Dense (per-token) output: returns flat f32 array (n_tokens x out_channels).
     */
    transform(tokens_flat: Float32Array, n_tokens: number): Float32Array;
}

export class TrainerWasm {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Pairwise accuracy on the given pairs (evaluation, no training).
     */
    evalPairAccuracy(model: PerceptModelWasm, better_flat: Float32Array, worse_flat: Float32Array, seq_lens: Uint32Array): number;
    constructor(lr: number, batch_size: number, seed: number);
    /**
     * One epoch of dense transform training. `tokens_flat` and
     * `targets_flat` are (n_tokens x in_channels / out_channels); `masks_flat`
     * is (n_tokens x 1). All samples must have equal seq length.
     */
    trainDenseEpoch(model: PerceptModelWasm, tokens_flat: Float32Array, targets_flat: Float32Array, masks_flat: Float32Array, n_samples: number, seq_len: number, loss_kind: string): number;
    /**
     * One epoch of pairwise ranking. `better_flat`/`worse_flat` are
     * concatenated token sequences; `seq_lens[i]` gives each pair's length
     * (pairs must have equal lengths). Returns mean loss.
     */
    trainPairEpoch(model: PerceptModelWasm, better_flat: Float32Array, worse_flat: Float32Array, seq_lens: Uint32Array, margins: Float32Array): number;
    /**
     * One epoch of score regression. `tokens_flat` is all samples
     * concatenated; `seq_lens[i]` is the token count of sample i;
     * `targets[i]` is the target score of sample i. Returns mean loss.
     */
    trainScoreEpoch(model: PerceptModelWasm, tokens_flat: Float32Array, seq_lens: Uint32Array, targets: Float32Array, loss_kind: string): number;
}

/**
 * Additive uniform noise; severity in [0,1].
 */
export function addNoise(img: Float32Array, severity: number, seed: number): Float32Array;

/**
 * Apply a layout corruption. `boxes_flat` is (n_boxes x 4): x, y, w, h.
 * `kind` is one of: "overlap", "overflow_right", "overflow_bottom",
 * "dead_space", "squeeze". Returns corrupted boxes (n x 4).
 */
export function corruptLayout(boxes_flat: Float32Array, kind: string, severity: number, page_w: number, page_h: number, seed: number): Float32Array;

/**
 * Intersection-over-union of two [x, y, w, h] boxes.
 */
export function iou(a: Float32Array, b: Float32Array): number;

/**
 * Remove a rectangular region for inpainting training.
 * Returns corrupted image (w*h*channels) followed by mask (w*h).
 */
export function maskRegion(img: Float32Array, w: number, h: number, channels: number, x: number, y: number, rw: number, rh: number, fill: number): Float32Array;

/**
 * Patchify an image into token sequences for the model.
 * `img` is a flat f32 array (w*h*channels, channel-interleaved).
 * Returns flat tokens (n_tokens x channels*patch*patch).
 */
export function patchifyImage(img: Float32Array, w: number, h: number, patch: number, channels: number): Float32Array;

/**
 * Rec. 601 grayscale of an RGB image (w*h*3 -> w*h).
 */
export function toGrayscale(img: Float32Array, w: number, h: number): Float32Array;

/**
 * Inverse of patchifyImage.
 */
export function unpatchifyImage(tokens_flat: Float32Array, w: number, h: number, patch: number, channels: number): Float32Array;

/**
 * Library version string.
 */
export function version(): string;
