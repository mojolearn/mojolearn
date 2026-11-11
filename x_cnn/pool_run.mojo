# SPDX-License-Identifier: Apache-2.0
"""RUN half of the CNN pooling and channel-dropout layers: MaxPool2d,
AvgPool2d, the 1-D pools, AdaptiveAvgPool2d, AdaptiveMaxPool2d and
Dropout2d (`bindings/xcnn_pool_native.mojo` is the UNPACK/PACK boundary;
lane doorsplit-cnn-xseq, 2026-11-11).

Typed values only: the pool settings as words (`PoolSettings`), the launch
words of each operation, the shape checks with the layers' refusal texts,
the dropout mask words, the adaptive pool words, the index cast of a
caller's adaptive indices, and the host or `_m` native entries with the
residency words the door used. The boundary reads the layer's fields,
admits and allocates every buffer (addresses and handles arrive as words),
releases the GIL around the entry calls here, and writes the fields back.
"""
from core.hotpath_impl import _cast_from, HP_I32
from core.entry_table import EntryRef
from core.xcnn_entry_table import (
    XC_MAXPOOL2D_FORWARD, XC_MAXPOOL2D_FORWARD_M, XC_AVGPOOL2D_FORWARD, XC_AVGPOOL2D_FORWARD_M,
    XC_MAXPOOL2D_BACKWARD, XC_MAXPOOL2D_BACKWARD_M, XC_AVGPOOL2D_BACKWARD, XC_AVGPOOL2D_BACKWARD_M, XC_DROPOUT2D,
    XC_DROPOUT2D_M, XC_RES_DOWNLOAD, XC_ADAPTIVE_POOL, XC_ADAPTIVE_POOL_M, XC_MAP2_M, XC_MUL,
)
from x_cnn.door_run import xcnn_run_call, xcnn_run_call_f
from x_cnn.shape_words import pool_params, PP_OH, PP_OW


@fieldwise_init
struct PoolSettings(Copyable, Movable):
    """A 2-D pool's admitted settings (UNPACK): kernel, stride, padding and
    dilation pairs, the two flags and the divisor override (0: none)."""
    var kh: Int
    var kw: Int
    var sh: Int
    var sw: Int
    var ph: Int
    var pw: Int
    var dh: Int
    var dw: Int
    var include_pad: Bool
    var ceil_mode: Bool
    var divisor: Int


# ---------------------------------------------------------------- launch words

def pool_run_words(settings: PoolSettings, shape: List[Int]) -> List[Int]:
    """`_Pool2d._params(shape)` as native words (before `pool_params` fills
    OH, OW)."""
    return [shape[0], shape[1], shape[2], shape[3], settings.kh, settings.kw, settings.sh, settings.sw,
            settings.ph, settings.pw, settings.dh, settings.dw, 0, 0, 1 if settings.include_pad else 0, 0,
            1 if settings.ceil_mode else 0, settings.divisor]


def pool_run_m(dev: Int, words: List[Int]) -> List[Int]:
    """A `_m` operation's integers: the residency word, then the launch words
    (core/xcnn_entry_table.mojo)."""
    var result: List[Int] = [dev]
    for i in range(len(words)):  # small-loop(the launch parameter words)
        result.append(words[i])
    return result^


def pool_run_out_words(words: List[Int], shape: List[Int]) raises -> List[Int]:
    """`_Pool2d._out_shape(shape)` from the launch words the caller already
    built (the same words then serve the launch). N149 (ne3-c-cnn,
    2026-11-07): the binding's `x_cnn_pool_shape` rule (`pool_params`,
    x_cnn/shape_words.mojo, with the same refusals) imported natively: no
    binding crossing per call."""
    var prm = pool_params(words)
    return [shape[0], shape[1], Int(prm[PP_OH]), Int(prm[PP_OW])]


def pool_run_flat_rest(shape: List[Int]) -> Int:
    """The words of one sample: the product of every dimension after the
    first (`out.reshape(out.shape[0], -1)`, `Conv2d.fit`'s record)."""
    var rest = 1
    for i in range(1, len(shape)):  # driver-loop: over shape words
        rest *= shape[i]
    return rest


def pool_run_row_dims(rows: Int, sample: List[Int]) -> List[Int]:
    """A row input's shape: the row count, then the layer's `input_shape`."""
    var dims: List[Int] = [rows]
    for i in range(len(sample)):  # driver-loop: over the admitted sample shape words
        dims.append(sample[i])
    return dims^


# ---------------------------------------------------------------- checks (refusal texts unchanged)

def pool_run_check_nchw(shape: List[Int], name: String) raises:
    if len(shape) != 4:
        raise Error("mojolearn: " + name + ".forward takes (N, C, H, W)")


def pool_run_check_ncl(shape: List[Int], name: String) raises:
    if len(shape) != 3:
        raise Error("mojolearn: " + name + ".forward takes (N, C, L)")


def pool_run_check_backward_ncl(shape: List[Int]) raises:
    if len(shape) != 3:
        raise Error("mojolearn: one-dimensional pool backward takes (N, C, L)")


def pool_run_check_saved(shape: List[Int]) raises:
    if len(shape) != 4:
        raise Error("mojolearn: saved pool input must have four dimensions")


def pool_run_check_adaptive_saved(shape: List[Int]) raises:
    if len(shape) != 4:
        raise Error("mojolearn: saved adaptive pool input must have four dimensions")


def pool_run_check_shape(actual: List[Int], expected: List[Int]) raises:
    if len(actual) != len(expected):
        raise Error("mojolearn: grad_out rank differs from the pool output")
    for i in range(len(expected)):  # driver-loop: compare gradient dimensions before kernel access
        if actual[i] != expected[i]:
            raise Error("mojolearn: grad_out shape differs from the pool output")


def pool_run_check_dim(word: Int) raises:
    if word < 0:
        raise Error("mojolearn: negative pool shape")


def pool_run_check_divisor(divisor: Int) raises:
    if divisor <= 0:
        raise Error("mojolearn: divisor must be not zero")


def pool_run_check_probability(p: Float64) raises:
    if not (p >= 0.0 and p <= 1.0):
        raise Error("mojolearn: dropout probability has to be between 0 and 1")


def pool_run_check_count(count: Int, expected: Int) raises:
    """The dropout backward's gradient against the forward's mask size."""
    if count != expected:
        raise Error("mojolearn: grad_out does not match the forward's input")


def pool_run_check_output(h: Int, w: Int) raises:
    if h <= 0 or w <= 0:
        raise Error("mojolearn: adaptive output_size must be positive")


def pool_run_cache_matches(cached: List[Int], k: List[Int]) raises -> Bool:
    """Whether the adaptive pool's cached inner pool has kernel `k`."""
    if len(cached) != 2:
        raise Error("TypeError: adaptive pool cache must contain two dimensions")
    return cached[0] == k[0] and cached[1] == k[1]


# ---------------------------------------------------------------- Dropout2d words

def pool_run_dropout_shape(shape: List[Int]) raises -> List[Int]:
    """The (N, C, H, W) of a Dropout2d input: a (C, H, W) input is one
    sample."""
    if len(shape) != 3 and len(shape) != 4:
        raise Error("mojolearn: Dropout2d.forward takes (N, C, H, W) or (C, H, W)")
    var s4 = List[Int]()
    if len(shape) == 3:
        s4.append(1)
    for i in range(len(shape)):  # driver-loop: over shape words
        s4.append(shape[i])
    return s4^


def pool_run_dropout_threshold(p: Float64) -> Int:
    """`round(p * 2**32)` with round-to-even."""
    var scaled = p * 4294967296.0
    var threshold = Int(scaled)
    var fraction = scaled - Float64(threshold)
    if fraction > 0.5 or (fraction == 0.5 and (threshold & 1) != 0):
        threshold += 1
    return threshold


def pool_run_dropout_words(p: Float64, seed: UInt64, calls: Int, n: Int, c: Int, hw: Int) -> List[Int]:
    """The mask words of one call: the threshold of `p` and the seed of
    (random_state's low 64 bits, the call count)."""
    var threshold = pool_run_dropout_threshold(p)
    var seed_lo = Int(seed & UInt64(2147483647))
    var seed_hi = Int(((seed >> 31) * UInt64(1000003) + UInt64(calls)) & UInt64(2147483647))
    return [n, c, hw, seed_lo, seed_hi, threshold >> 16, threshold & 0xFFFF]


# ---------------------------------------------------------------- adaptive pool words

def pool_run_adaptive_divides(shape: List[Int], o: List[Int]) -> Bool:
    """The output size divides the input: the adaptive pool runs a plain
    pool of kernel = stride = `pool_run_adaptive_kernel`."""
    return shape[2] % o[0] == 0 and shape[3] % o[1] == 0


def pool_run_adaptive_kernel(shape: List[Int], o: List[Int]) -> List[Int]:
    return [shape[2] // o[0], shape[3] // o[1]]


def pool_run_adaptive_words(shape: List[Int], o: List[Int], maxed: Bool, backward: Bool) -> List[Int]:
    """The adaptive pool's launch words: the mode word is 0 / 2 (avg / max)
    forward and 1 / 3 backward."""
    var mode = (2 if maxed else 0) | (1 if backward else 0)
    return [shape[0], shape[1], shape[2], shape[3], o[0], o[1], mode]


def pool_run_cast_indices(code: Int, item: Int, src: Int, dst: Int, count: Int) raises:
    """A caller's adaptive pool indices (numeric dtype `code`, item size
    `item`, C order) as int32 words, with the layer's refusals."""
    var refused = 0
    if code == 0 or code == 2:
        refused = _cast_from[DType.uint8](src, HP_I32, dst, count)
    elif code == 1:
        refused = _cast_from[DType.int8](src, HP_I32, dst, count)
    elif code == 3:
        refused = _cast_from[DType.int16](src, HP_I32, dst, count)
    elif code == 4:
        refused = _cast_from[DType.uint16](src, HP_I32, dst, count)
    elif code == 5 or (code == 7 and item == 4):
        refused = _cast_from[DType.int32](src, HP_I32, dst, count)
    elif code == 6 or (code == 8 and item == 4):
        refused = _cast_from[DType.uint32](src, HP_I32, dst, count)
    elif code == 7 or code == 9:
        refused = _cast_from[DType.int64](src, HP_I32, dst, count)
    elif code == 8 or code == 10:
        refused = _cast_from[DType.uint64](src, HP_I32, dst, count)
    elif code == 11:
        refused = _cast_from[DType.float32](src, HP_I32, dst, count)
    elif code == 12:
        refused = _cast_from[DType.float64](src, HP_I32, dst, count)
    elif code == 23:
        refused = _cast_from[DType.float16](src, HP_I32, dst, count)
    else:
        raise Error("TypeError: adaptive pool indices require a native numeric dtype")
    if refused != 0:
        raise Error("TypeError: adaptive pool indices do not fit int32")


# ---------------------------------------------------------------- native entries
# The caller keeps every addressed buffer alive and has released the GIL.
# `dev` is the `_m` residency word the door used.

def pool_run_max_forward(entry: EntryRef, x: Int, y: Int, idx: Int, words: List[Int]) raises:
    _ = xcnn_run_call(entry, XC_MAXPOOL2D_FORWARD, [x, y, idx], words)


def pool_run_max_forward_m(entry: EntryRef, dev: Int, x: Int, y: Int, h: Int, words: List[Int]) raises:
    _ = xcnn_run_call(entry, XC_MAXPOOL2D_FORWARD_M, [x, y, h], pool_run_m(dev, words))


def pool_run_avg_forward(entry: EntryRef, x: Int, y: Int, words: List[Int]) raises:
    _ = xcnn_run_call(entry, XC_AVGPOOL2D_FORWARD, [x, y], words)


def pool_run_avg_forward_m(entry: EntryRef, dev: Int, x: Int, y: Int, words: List[Int]) raises:
    _ = xcnn_run_call(entry, XC_AVGPOOL2D_FORWARD_M, [x, y], pool_run_m(dev, words))


def pool_run_max_backward(entry: EntryRef, g: Int, idx: Int, dx: Int, words: List[Int]) raises:
    _ = xcnn_run_call(entry, XC_MAXPOOL2D_BACKWARD, [g, idx, dx], words)


def pool_run_max_backward_m(entry: EntryRef, dev: Int, g: Int, h: Int, dx: Int, words: List[Int]) raises:
    _ = xcnn_run_call(entry, XC_MAXPOOL2D_BACKWARD_M, [g, h, dx], pool_run_m(dev, words))


def pool_run_avg_backward(entry: EntryRef, g: Int, dx: Int, words: List[Int]) raises:
    _ = xcnn_run_call(entry, XC_AVGPOOL2D_BACKWARD, [g, dx], words)


def pool_run_avg_backward_m(entry: EntryRef, dev: Int, g: Int, dx: Int, words: List[Int]) raises:
    _ = xcnn_run_call(entry, XC_AVGPOOL2D_BACKWARD_M, [g, dx], pool_run_m(dev, words))


def pool_run_dropout(entry: EntryRef, x: Int, y: Int, mask: Int, words: List[Int], p: Float64) raises:
    var floats: List[Float64] = [p]
    _ = xcnn_run_call_f(entry, XC_DROPOUT2D, [x, y, mask], words, floats)


def pool_run_dropout_m(entry: EntryRef, dev: Int, x: Int, y: Int, h: Int, words: List[Int], p: Float64) raises:
    var floats: List[Float64] = [p]
    _ = xcnn_run_call_f(entry, XC_DROPOUT2D_M, [x, y, h], pool_run_m(dev, words), floats)


def pool_run_adaptive(entry: EntryRef, x: Int, y: Int, idx: Int, words: List[Int]) raises:
    _ = xcnn_run_call(entry, XC_ADAPTIVE_POOL, [x, y, idx], words)


def pool_run_adaptive_m(entry: EntryRef, dev: Int, x: Int, y: Int, h: Int, words: List[Int]) raises:
    _ = xcnn_run_call(entry, XC_ADAPTIVE_POOL_M, [x, y, h], pool_run_m(dev, words))


def pool_run_download(entry: EntryRef, handle: Int, dst: Int, count: Int) raises:
    """A resident array's `count` words read back into the host buffer."""
    _ = xcnn_run_call(entry, XC_RES_DOWNLOAD, [handle, dst], [count])


def pool_run_mask_m(entry: EntryRef, dev: Int, g: Int, handle: Int, dx: Int, count: Int) raises:
    """The dropout backward against the resident mask (map2 op 3: multiply)."""
    _ = xcnn_run_call(entry, XC_MAP2_M, [g, handle, dx], [dev, 3, count])


def pool_run_mul(entry: EntryRef, g: Int, mask: Int, dx: Int, count: Int) raises:
    """The dropout backward against the host mask."""
    _ = xcnn_run_call(entry, XC_MUL, [g, mask, dx], [count])
