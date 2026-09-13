/* @ts-self-types="./percept_wasm.d.ts" */

class PerceptModelWasm {
    static __wrap(ptr) {
        const obj = Object.create(PerceptModelWasm.prototype);
        obj.__wbg_ptr = ptr;
        PerceptModelWasmFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        PerceptModelWasmFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_perceptmodelwasm_free(ptr, 0);
    }
    /**
     * The model's config as a JSON string.
     * @returns {string}
     */
    configJson() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.perceptmodelwasm_configJson(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Load a model from bytes produced by `save`.
     * @param {Uint8Array} bytes
     * @returns {PerceptModelWasm}
     */
    static load(bytes) {
        const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.perceptmodelwasm_load(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return PerceptModelWasm.__wrap(ret[0]);
    }
    /**
     * Number of trainable parameters.
     * @returns {number}
     */
    nParams() {
        const ret = wasm.perceptmodelwasm_nParams(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} in_channels
     * @param {number} d_model
     * @param {number} n_heads
     * @param {number} n_layers
     * @param {number} d_ff
     * @param {number} out_channels
     * @param {number} max_seq_len
     * @param {number} seed
     */
    constructor(in_channels, d_model, n_heads, n_layers, d_ff, out_channels, max_seq_len, seed) {
        const ret = wasm.perceptmodelwasm_new(in_channels, d_model, n_heads, n_layers, d_ff, out_channels, max_seq_len, seed);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        PerceptModelWasmFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Serialize the model (config + weights) to bytes.
     * @returns {Uint8Array}
     */
    save() {
        const ret = wasm.perceptmodelwasm_save(this.__wbg_ptr);
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * Score a token sequence: flat f32 array, row-major (n_tokens x in_channels).
     * Returns the global scalar score.
     * @param {Float32Array} tokens_flat
     * @param {number} n_tokens
     * @returns {number}
     */
    score(tokens_flat, n_tokens) {
        const ptr0 = passArrayF32ToWasm0(tokens_flat, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.perceptmodelwasm_score(this.__wbg_ptr, ptr0, len0, n_tokens);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Dense (per-token) output: returns flat f32 array (n_tokens x out_channels).
     * @param {Float32Array} tokens_flat
     * @param {number} n_tokens
     * @returns {Float32Array}
     */
    transform(tokens_flat, n_tokens) {
        const ptr0 = passArrayF32ToWasm0(tokens_flat, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.perceptmodelwasm_transform(this.__wbg_ptr, ptr0, len0, n_tokens);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
}
if (Symbol.dispose) PerceptModelWasm.prototype[Symbol.dispose] = PerceptModelWasm.prototype.free;
exports.PerceptModelWasm = PerceptModelWasm;

class TrainerWasm {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TrainerWasmFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_trainerwasm_free(ptr, 0);
    }
    /**
     * Pairwise accuracy on the given pairs (evaluation, no training).
     * @param {PerceptModelWasm} model
     * @param {Float32Array} better_flat
     * @param {Float32Array} worse_flat
     * @param {Uint32Array} seq_lens
     * @returns {number}
     */
    evalPairAccuracy(model, better_flat, worse_flat, seq_lens) {
        _assertClass(model, PerceptModelWasm);
        const ptr0 = passArrayF32ToWasm0(better_flat, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(worse_flat, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArray32ToWasm0(seq_lens, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.trainerwasm_evalPairAccuracy(this.__wbg_ptr, model.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * @param {number} lr
     * @param {number} batch_size
     * @param {number} seed
     */
    constructor(lr, batch_size, seed) {
        const ret = wasm.trainerwasm_new(lr, batch_size, seed);
        this.__wbg_ptr = ret;
        TrainerWasmFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * One epoch of dense transform training. `tokens_flat` and
     * `targets_flat` are (n_tokens x in_channels / out_channels); `masks_flat`
     * is (n_tokens x 1). All samples must have equal seq length.
     * @param {PerceptModelWasm} model
     * @param {Float32Array} tokens_flat
     * @param {Float32Array} targets_flat
     * @param {Float32Array} masks_flat
     * @param {number} n_samples
     * @param {number} seq_len
     * @param {string} loss_kind
     * @returns {number}
     */
    trainDenseEpoch(model, tokens_flat, targets_flat, masks_flat, n_samples, seq_len, loss_kind) {
        _assertClass(model, PerceptModelWasm);
        const ptr0 = passArrayF32ToWasm0(tokens_flat, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(targets_flat, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF32ToWasm0(masks_flat, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passStringToWasm0(loss_kind, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len3 = WASM_VECTOR_LEN;
        const ret = wasm.trainerwasm_trainDenseEpoch(this.__wbg_ptr, model.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2, n_samples, seq_len, ptr3, len3);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * One epoch of pairwise ranking. `better_flat`/`worse_flat` are
     * concatenated token sequences; `seq_lens[i]` gives each pair's length
     * (pairs must have equal lengths). Returns mean loss.
     * @param {PerceptModelWasm} model
     * @param {Float32Array} better_flat
     * @param {Float32Array} worse_flat
     * @param {Uint32Array} seq_lens
     * @param {Float32Array} margins
     * @returns {number}
     */
    trainPairEpoch(model, better_flat, worse_flat, seq_lens, margins) {
        _assertClass(model, PerceptModelWasm);
        const ptr0 = passArrayF32ToWasm0(better_flat, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(worse_flat, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArray32ToWasm0(seq_lens, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passArrayF32ToWasm0(margins, wasm.__wbindgen_malloc);
        const len3 = WASM_VECTOR_LEN;
        const ret = wasm.trainerwasm_trainPairEpoch(this.__wbg_ptr, model.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * One epoch of score regression. `tokens_flat` is all samples
     * concatenated; `seq_lens[i]` is the token count of sample i;
     * `targets[i]` is the target score of sample i. Returns mean loss.
     * @param {PerceptModelWasm} model
     * @param {Float32Array} tokens_flat
     * @param {Uint32Array} seq_lens
     * @param {Float32Array} targets
     * @param {string} loss_kind
     * @returns {number}
     */
    trainScoreEpoch(model, tokens_flat, seq_lens, targets, loss_kind) {
        _assertClass(model, PerceptModelWasm);
        const ptr0 = passArrayF32ToWasm0(tokens_flat, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(seq_lens, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passStringToWasm0(loss_kind, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len3 = WASM_VECTOR_LEN;
        const ret = wasm.trainerwasm_trainScoreEpoch(this.__wbg_ptr, model.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
}
if (Symbol.dispose) TrainerWasm.prototype[Symbol.dispose] = TrainerWasm.prototype.free;
exports.TrainerWasm = TrainerWasm;

/**
 * Patchify an image into token sequences for the model.
 * `img` is a flat f32 array (w*h*channels, channel-interleaved).
 * Returns flat tokens (n_tokens x channels*patch*patch).
 * @param {Float32Array} img
 * @param {number} w
 * @param {number} h
 * @param {number} patch
 * @param {number} channels
 * @returns {Float32Array}
 */
function patchifyImage(img, w, h, patch, channels) {
    const ptr0 = passArrayF32ToWasm0(img, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.patchifyImage(ptr0, len0, w, h, patch, channels);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}
exports.patchifyImage = patchifyImage;

/**
 * Inverse of patchifyImage.
 * @param {Float32Array} tokens_flat
 * @param {number} w
 * @param {number} h
 * @param {number} patch
 * @param {number} channels
 * @returns {Float32Array}
 */
function unpatchifyImage(tokens_flat, w, h, patch, channels) {
    const ptr0 = passArrayF32ToWasm0(tokens_flat, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.unpatchifyImage(ptr0, len0, w, h, patch, channels);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}
exports.unpatchifyImage = unpatchifyImage;

/**
 * Library version string.
 * @returns {string}
 */
function version() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.version();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}
exports.version = version;
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_67e7344beaa85059: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg___wbindgen_throw_5d9e815e6fdf150f: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./percept_wasm_bg.js": import0,
    };
}

const PerceptModelWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_perceptmodelwasm_free(ptr, 1));
const TrainerWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_trainerwasm_free(ptr, 1));

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
function decodeText(ptr, len) {
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

const wasmPath = `${__dirname}/percept_wasm_bg.wasm`;
const wasmBytes = require('fs').readFileSync(wasmPath);
const wasmModule = new WebAssembly.Module(wasmBytes);
let wasmInstance = new WebAssembly.Instance(wasmModule, __wbg_get_imports());
let wasm = wasmInstance.exports;
wasm.__wbindgen_start();
