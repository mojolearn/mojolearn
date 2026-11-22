# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""The decode step's cache refusal: the appended-row scan against the whole
scan, and a non-finite value PRODUCED by a step.

Part 1, kernel level (cases 0 to 3). `_kv_rows_scan_enqueue`
(transformer/impl/llama/modeling_llama.mojo, the scan the default build
runs on a granted cache; a FAST build runs the whole scan at every step) reads row
`s - 1` of every (batch, kv head) run of a key or value cache and reports a
FLAT index of the buffer; the default refusal is `DeviceNonfiniteBatch`'s
scan of the whole used prefix. On one device buffer laid out as the cache is,
both scans run and their answers are compared, at both row strides the scan
takes: packed at stride `s` (the layout the linear cache has) and at a fixed
stride `cap` wider than the used slots. The stride is an argument of the
scan, so both are checked in every build.
  0 small cache, packed          1 small cache, fixed stride
  2 large cache, packed          3 large cache, fixed stride
(large: 2,048 runs of 128 floats, more scan elements than one pass of the
scan's 512 blocks of 256 threads, so the strided second pass finds a plant.)
Per case: nothing planted (both -1); a NaN in the last float of the last
run's row s - 1 (both report its flat index); two infinities in row s - 1 of
two runs (both report the smaller flat index); a NaN only in row s - 2 (the
whole scan reports it, the row scan reports -1: the row scan reads the
appended row and nothing else, which is why every other writer must reset the
watermark); fixed stride only: a NaN in an unused slot past the last run's
used slots (neither reports it) and one in an unused slot of the first run
(only the whole scan reports it).

Part 2, a step that produces the non-finite value (cases 4 and 5). One
decoder layer whose key (case 4) or value (case 5) projection has one column
of 1e35 and whose first norm weight is 1e4 on that feature: every weight is
finite, a token whose feature is 0 gives finite rows, a token whose feature
is 1 overflows its whole cache row. Prefill of two tokens, a finite step, the
overflowing step (it is not refused: its inputs are finite), then one more
step, which MUST be refused naming the key (value) cache, the kind (NaN or
infinity) and the flat index of the first non-finite float of the cache as
this build lays it out; the expected index and kind are read from the device
cache itself. The cache is granted (`wm_grant`), so in the default build
the refusing scan is the row scan (the check requires the watermark to sit
at `s - 1` there); under the watermark's _OFF define it is the whole scan.

What a sabotage changes here: a row scan that reads row s - 2 or s, uses the
packed stride on a fixed-stride cache (or the reverse), reports the scan
element instead of the flat index, drops the strided pass or folds the
partials wrongly (part 1); a watermark advanced past a row nobody scanned, a
refusal naming the other cache or another index, or no refusal at all for a
value a step computed (part 2).
"""
from std.testing import TestSuite
from std.memory import bitcast

from max.gpu.host import DeviceBuffer, DeviceContext

from core.device_scan import DeviceNonfiniteBatch
from core.identity_trace import IdentityTrace
from transformer.checks.transformer_fixture import (
    RMS_EPS,
    ROPE_THETA,
    TransformerDims,
    TransformerWeights,
    fixture_case_seed,
    fixture_tensor,
)
from transformer.impl.llama.modeling_llama import (
    LLAMA_FAST_REFUSE_WATERMARK,
    PLANT_AT_NONE,
    LlamaDeviceStages,
    LlamaDeviceWeights,
    LlamaDims,
    LlamaKVCache,
    LlamaRopeTable,
    _kv_rows_scan_enqueue,
    llama_decoder_layer_forward_planted,
)

from checks.native_tests.kit import download, upload
from checks.native_tests.testsuite_cases import require


comptime LRS_CASES = 6

comptime _NAN: UInt32 = 0x7FC00000
comptime _POS_INF: UInt32 = 0x7F800000
comptime _NEG_INF: UInt32 = 0xFF800000

# Part 2's layer: small, one kv head per two heads.
comptime _DM = 32
comptime _HEADS = 2
comptime _KV_HEADS = 1
comptime _HD = 16
comptime _INTER = 64
comptime _B = 2
comptime _CAP = 8
#: The feature whose norm weight and projection column are huge.
comptime _HOT = 5


# ---------------------------------------------------------------- part 1

def _whole_scan(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], cells: Int) raises -> Int:
    """The default refusal's scan of the first `cells` floats: the first
    non-finite flat index, or -1."""
    var lens: List[Int] = [cells]
    var batch = DeviceNonfiniteBatch(ctx, lens)
    batch.enqueue(ctx, 0, buf)
    var hits = batch.finish(ctx)
    return hits[0]


def _row_scan(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], runs: Int, hd: Int, row_stride: Int, first: Int
) raises -> Int:
    """The watermark's scan: `hd` floats at `g * row_stride + first` of every
    run `g`, as `llama_refuse_bad_call` calls it for one cache."""
    var lens: List[Int] = [runs * hd]
    var batch = DeviceNonfiniteBatch(ctx, lens)
    _kv_rows_scan_enqueue(ctx, batch, 0, buf, hd, row_stride, first)
    var hits = batch.finish(ctx)
    return hits[0]


def _both(
    ctx: DeviceContext, values: List[Float32], runs: Int, hd: Int, s: Int, slots: Int,
    want_whole: Int, want_row: Int, what: String,
) raises:
    """Upload `values` (a cache of `runs` runs at `slots` slots of `hd`
    floats, `s` slots used) and require both scans' answers."""
    var buf = upload[DType.float32](ctx, values)
    var cells = ((runs - 1) * slots + s) * hd
    var whole = _whole_scan(ctx, buf, cells)
    var row = _row_scan(ctx, buf, runs, hd, slots * hd, (s - 1) * hd)
    _ = buf^
    require(whole == want_whole, what + ": the whole scan reports " + String(whole) + ", expected " + String(want_whole))
    require(row == want_row, what + ": the row scan reports " + String(row) + ", expected " + String(want_row))


def _layout_case(ctx: DeviceContext, runs: Int, hd: Int, s: Int, slots: Int) raises:
    """Every plant of part 1 on one layout (`slots == s`: packed)."""
    var n = runs * slots * hd
    var seed = fixture_case_seed(200 + runs + slots)
    var clean = fixture_tensor(seed, 41, n, -3.0, 3.0)
    var tag = String("runs ") + String(runs) + ", head_dim " + String(hd) + ", s " + String(s) + ", stride " + String(slots)
    _both(ctx, clean, runs, hd, s, slots, -1, -1, tag + ", nothing planted")

    # the last float the row scan reads (and the last float of the whole scan)
    var last = clean.copy()
    var at_last = ((runs - 1) * slots + s - 1) * hd + hd - 1
    last[at_last] = bitcast[DType.float32](_NAN)
    _both(ctx, last, runs, hd, s, slots, at_last, at_last, tag + ", NaN in the last appended float")

    # two runs, the later one planted at a smaller offset inside its row
    var two = clean.copy()
    var g1 = (2 * runs) // 3
    var g2 = runs - 1
    var first = (g1 * slots + s - 1) * hd + hd // 2
    two[first] = bitcast[DType.float32](_POS_INF)
    two[(g2 * slots + s - 1) * hd] = bitcast[DType.float32](_NEG_INF)
    _both(ctx, two, runs, hd, s, slots, first, first, tag + ", infinities in the appended row of two runs")

    # below the watermark: only the whole scan may see it
    var below = clean.copy()
    var under = ((runs // 2) * slots + s - 2) * hd + 3
    below[under] = bitcast[DType.float32](_NAN)
    _both(ctx, below, runs, hd, s, slots, under, -1, tag + ", NaN in row s - 2 only")

    if slots > s:
        # a fixed-stride cache has unused slots: past the last run's used
        # slots nothing is scanned; inside the scanned prefix only the whole
        # scan reads them
        var past = clean.copy()
        past[((runs - 1) * slots + s) * hd] = bitcast[DType.float32](_NAN)
        _both(ctx, past, runs, hd, s, slots, -1, -1, tag + ", NaN past the last used slot")
        var unused = clean.copy()
        var hole = s * hd + 1
        unused[hole] = bitcast[DType.float32](_NAN)
        _both(ctx, unused, runs, hd, s, slots, hole, -1, tag + ", NaN in an unused slot of the first run")


# ---------------------------------------------------------------- part 2

def _layer_weights(dims: TransformerDims, seed: UInt64, value_cache: Bool) -> TransformerWeights:
    """Finite weights whose key (or value) projection overflows on a token
    with feature `_HOT` set: column `_HOT` is 1e36 and the first norm's
    weight there is 1e4, so the projected row is about 1e40 times the
    normalized feature."""
    var w = TransformerWeights(dims)
    var dm = dims.d_model
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var inter = dims.intermediate
    w.norm1_w = fixture_tensor(seed, 51, dm, 0.5, 1.5)
    w.norm2_w = fixture_tensor(seed, 52, dm, 0.5, 1.5)
    w.w_q = fixture_tensor(seed, 53, qw * dm, -0.25, 0.25)
    w.w_k = fixture_tensor(seed, 54, kw * dm, -0.25, 0.25)
    w.w_v = fixture_tensor(seed, 55, kw * dm, -0.25, 0.25)
    w.w_o = fixture_tensor(seed, 56, dm * qw, -0.25, 0.25)
    w.w_gate = fixture_tensor(seed, 57, inter * dm, -0.25, 0.25)
    w.w_up = fixture_tensor(seed, 58, inter * dm, -0.25, 0.25)
    w.w_down = fixture_tensor(seed, 59, dm * inter, -0.125, 0.125)
    w.norm1_w[_HOT] = Float32(1.0e4)
    for r in range(kw):
        if value_cache:
            w.w_v[r * dm + _HOT] = Float32(1.0e36)
        else:
            w.w_k[r * dm + _HOT] = Float32(1.0e36)
    return w^


def _tokens(seed: UInt64, tid: Int, l: Int, hot: Float32) -> List[Float32]:
    """`[_B, l, _DM]` block input with feature `_HOT` of every token set to
    `hot` (0: finite cache rows; 1: the overflow)."""
    var x = fixture_tensor(seed, tid, _B * l * _DM, -2.0, 2.0)
    for t in range(_B * l):
        x[t * _DM + _HOT] = hot
    return x^


def _first_nonfinite(values: List[Float32]) -> Int:
    for i in range(len(values)):
        if (bitcast[DType.uint32](values[i]) & UInt32(0x7FFFFFFF)) >= UInt32(0x7F800000):
            return i
    return -1


def _forward(
    ctx: DeviceContext, mut stages: LlamaDeviceStages, mut kv: LlamaKVCache, mut rope: LlamaRopeTable,
    mut dw: LlamaDeviceWeights, x: List[Float32], l: Int,
) raises:
    """One block call at `pos0 = kv.s` (a prefill at 0, then decode steps)."""
    var dx = upload[DType.float32](ctx, x)
    var off = IdentityTrace.disabled()
    var pos0 = kv.s
    try:
        llama_decoder_layer_forward_planted(
            ctx, stages, kv, rope, dw, dx, _B, l, pos0, PLANT_AT_NONE, List[Int](), List[UInt32](), off, "lrs",
        )
    except error:
        ctx.synchronize()
        _ = dx^
        raise error
    ctx.synchronize()
    _ = dx^


def _produced_case(ctx: DeviceContext, value_cache: Bool) raises:
    var dims = TransformerDims(_DM, _HEADS, _KV_HEADS, _HD, _INTER, _CAP)
    dims.validate()
    var seed = fixture_case_seed(300 + (1 if value_cache else 0))
    var w = _layer_weights(dims, seed, value_cache)
    var ldims = LlamaDims(_DM, _HEADS, _KV_HEADS, _HD, _INTER)
    var dw = LlamaDeviceWeights(
        ctx, ldims, RMS_EPS, w.norm1_w, w.norm2_w, w.w_q, w.w_k, w.w_v, w.w_o, w.w_gate, w.w_up, w.w_down,
    )
    var kv = LlamaKVCache(ctx, _B, ldims, _CAP, 0)
    kv.wm_grant(ctx, _DM)
    var rope = LlamaRopeTable(ctx, ldims, ROPE_THETA, dims.rope_positions)
    var prefill = LlamaDeviceStages(ctx, _B, 2, _CAP, ldims, 0)
    var step = LlamaDeviceStages(ctx, _B, 1, _CAP, ldims, 0)
    var name = String("past_key_values.value_cache") if value_cache else String("past_key_values.key_cache")

    _forward(ctx, prefill, kv, rope, dw, _tokens(seed, 61, 2, Float32(0.0)), 2)
    _forward(ctx, step, kv, rope, dw, _tokens(seed, 62, 1, Float32(0.0)), 1)
    # the overflowing step: finite inputs, so it is not refused
    _forward(ctx, step, kv, rope, dw, _tokens(seed, 63, 1, Float32(1.0)), 1)
    require(kv.s == 4, "fixture: four positions should be cached, the cache holds " + String(kv.s))

    # What the device cache holds now, as this build lays it out.
    var cells = kv.b * kv.n_kv * kv.s * kv.head_dim
    var keys = download[DType.float32](ctx, kv.k, cells)
    var vals = download[DType.float32](ctx, kv.v, cells)
    var first = _first_nonfinite(vals) if value_cache else _first_nonfinite(keys)
    var other = _first_nonfinite(keys) if value_cache else _first_nonfinite(vals)
    require(first >= 0, "FIXTURE ERROR: the overflowing step left no non-finite float in " + name)
    require(other < 0, "FIXTURE ERROR: the other cache is non-finite too, at " + String(other))
    var slots = kv.s
    require((first // _HD) % slots == kv.s - 1,
            "FIXTURE ERROR: the first non-finite float of " + name + " (flat " + String(first)
            + ") is not in the row the last step appended")
    var word = bitcast[DType.uint32](vals[first] if value_cache else keys[first]) & UInt32(0x7FFFFFFF)
    var kind = String("llama: NaN in ") if word > UInt32(0x7F800000) else String("llama: infinity in ")
    var want = kind + name + " at flat index " + String(first) + " REFUSED"

    comptime if LLAMA_FAST_REFUSE_WATERMARK:
        # the refusing scan below must be the row scan, not the whole scan
        require(kv.wm_on and kv.wm_upto == kv.s - 1,
                "watermark arm: the granted cache is not at s - 1 before the refusing step (wm_upto "
                + String(kv.wm_upto) + ", s " + String(kv.s) + "): the whole scan would run")

    var refused = String("")
    try:
        _forward(ctx, step, kv, rope, dw, _tokens(seed, 64, 1, Float32(0.0)), 1)
    except error:
        refused = String(error)
    require(refused != "", "the step after an overflowing step ran: a non-finite " + name + " row was not refused")
    require(want in refused, "the refusal is `" + refused + "`, expected it to start `" + want + "`")
    _ = prefill^
    _ = step^
    _ = rope^
    _ = kv^
    _ = dw^


def _case(ctx: DeviceContext, which: Int) raises:
    if which == 0:
        _layout_case(ctx, 6, 16, 5, 5)
    elif which == 1:
        _layout_case(ctx, 6, 16, 5, 9)
    elif which == 2:
        _layout_case(ctx, 2048, 128, 3, 3)
    elif which == 3:
        _layout_case(ctx, 2048, 128, 3, 4)
    elif which == 4:
        _produced_case(ctx, False)
    else:
        _produced_case(ctx, True)


def test_cases() raises:
    var ctx = DeviceContext()
    var failed = String("")
    for which in range(LRS_CASES):  # driver-loop: over the check's cases, one device queue
        try:
            _case(ctx, which)
        except problem:
            failed += "\n      llama_refuse_row_scan row " + String(which) + ": " + String(problem)
    if failed.byte_length() > 0:
        raise Error("llama_refuse_row_scan: cases failed" + failed)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
