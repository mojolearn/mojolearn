# SPDX-License-Identifier: Apache-2.0
"""The Embedding and residual-dropout entries through the caller's buffers
(lane ne3-c-blocks-embedding, rows N117, N158 and N160), on a GPU column,
moved from test_ne3_blocks_embedding_entries.py (2026-11-14).

The calls are the typed RUN bodies the Python entries run with the GIL
released: `embedding_forward_released` / `embedding_backward_released`
(embedding/embedding_words_run.mojo, after the words' `_config`,
`emb_refuse_shape` and `emb_resolve_plan`, as `embedding_forward_words` and
`embedding_backward_words` do), the host-array residual-dropout layer
`residual_dropout_device` (training/residual_dropout.mojo, the PACK door's
call) and its device-array form `residual_dropout_dev_run` /
`residual_dropout_backward_dev_run` on train_dev handles
(training/residual_dropout_boundary_run.mojo). No Python participates.

Fixtures (shapes, options and comparisons are the former tests'):
  0     forward: V 37, d 5, T 29; y equals W[ids] word for word.
  1..4  backward: V 13, d 6, T 40, padding_idx 2, accumulate 0 / 1 x plan
        PLAN_SCAN / PLAN_SORT; dW equals the profile's fold (ascending t
        per row, float32 at every step, the padding row stored +0.0) from
        the carried dW (accumulate 1) or from zero (accumulate 0, with the
        buffer prefilled with 7.0 so a missed fresh fill shows), compared by
        value as numpy's assert_array_equal does.
  5     residual dropout, 4 x 33, p 0.25, seed 9, stream 1, offset 3: the
        device-array forward and backward (both outputs) equal the
        host-array layer's words.
Inputs are a fixed 64-bit LCG stream: every oracle is a relation computed
here, so no NumPy stream is needed. The id, table and nonfinite refusals stay
in the Python file (they gate the Python entries' messages), with the
cross-entropy reductions (a Python float and array results).

The fixtures share this process's device context and run in order; each
kernel keeps its own parallelism.
"""
from std.memory import bitcast
from autograd.kernels.embedding_sort import PLAN_SCAN, PLAN_SORT
from core.hostptr import f32_ptr, i32_ptr
from core.neural_context import neural_ctx
from embedding.checks.embedding_identical import emb_resolve_plan
from embedding.checks.embedding_oracle import EMB_NO_PADDING_IDX, emb_refuse_shape
from embedding.embedding_words_run import _config, embedding_backward_released, embedding_forward_released
from training.dev_tensors import train_dev_alloc, train_dev_free, train_dev_get, train_dev_put
from training.residual_dropout import residual_dropout_device
from training.residual_dropout_boundary_run import (
    _RD_DEV_CTX,
    residual_dropout_backward_dev_run,
    residual_dropout_dev_run,
)
from training.residual_dropout_common_run import ResidualDropoutArgs
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


# ---- The device column (was embedding_ne3_entries_gpu_suite.mojo) ----
comptime EMBEDDING_NE3_FIXTURES = 6


def _uniform(mut values: List[Float32], mut seed: UInt64, lo: Float32, hi: Float32):
    # A fixed 64-bit LCG: a fixture, not data work.
    for i in range(len(values)):  # small-loop(fixture: a few hundred test values)
        seed = seed * 6364136223846793005 + 1442695040888963407
        var unit = Float32(Int((seed >> 40) & 0xFFFFFF)) / Float32(16777216.0)
        values[i] = lo + (hi - lo) * unit


def _ids(n: Int, vocab: Int, mut seed: UInt64) -> List[Int32]:
    var out = List[Int32](length=n, fill=0)
    for i in range(n):  # small-loop(fixture ids)
        seed = seed * 6364136223846793005 + 1442695040888963407
        out[i] = Int32(Int((seed >> 33) % UInt64(vocab)))
    return out^


def _same_words(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": extent")
    for i in range(len(a)):  # small-loop(fixture compare)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            raise Error(what + ": words differ at " + String(i) + " (" + String(a[i]) + " vs " + String(b[i]) + ")")


def embedding_forward_gathers() raises:
    var vocab = 37
    var width = 5
    var n = 29
    var seed: UInt64 = 3
    var w = List[Float32](length=vocab * width, fill=0)
    _uniform(w, seed, Float32(-2.0), Float32(2.0))
    var ids = _ids(n, vocab, seed)
    var y = List[Float32](length=n * width, fill=Float32(-1.0))
    var cfg = _config(vocab, width, EMB_NO_PADDING_IDX, False)
    emb_refuse_shape(cfg, n)
    var w_addr = Int(w.unsafe_ptr())
    embedding_forward_released(f32_ptr(w_addr), i32_ptr(Int(ids.unsafe_ptr())), n, cfg,
                               Int(y.unsafe_ptr()), n * width <= 0, 0, w_addr)
    var want = List[Float32](length=n * width, fill=0)
    for t in range(n):
        for j in range(width):
            want[t * width + j] = w[Int(ids[t]) * width + j]
    _same_words(y, want, "embedding forward against W[ids]")
    _ = w
    _ = ids


def embedding_backward_folds(accumulate: Int, plan_code: Int) raises:
    var vocab = 13
    var width = 6
    var n = 40
    var pad = 2
    var seed: UInt64 = 5
    var ids = _ids(n, vocab, seed)
    var dy = List[Float32](length=n * width, fill=0)
    _uniform(dy, seed, Float32(-2.0), Float32(2.0))
    var dw = List[Float32](length=vocab * width, fill=Float32(7.0))
    if accumulate == 1:
        _uniform(dw, seed, Float32(-2.0), Float32(2.0))
    var start = dw.copy()
    var cfg = _config(vocab, width, pad, accumulate == 1)
    emb_refuse_shape(cfg, n)
    var plan = emb_resolve_plan(plan_code, vocab, n)
    embedding_backward_released(f32_ptr(Int(dy.unsafe_ptr())), i32_ptr(Int(ids.unsafe_ptr())), n, cfg,
                                Int(dw.unsafe_ptr()), vocab * width <= 0, plan)
    var want = start.copy() if accumulate == 1 else List[Float32](length=vocab * width, fill=0)
    for pos in range(n):  # the profile's fold: ascending t per row, float32 every step
        var row = Int(ids[pos])
        if row != pad:
            for j in range(width):
                want[row * width + j] = want[row * width + j] + dy[pos * width + j]
    for j in range(width):
        want[pad * width + j] = Float32(0.0)
    var what = "embedding backward (accumulate " + String(accumulate) + ", plan " + String(plan_code) + ")"
    require(len(dw) == len(want), what + ": extent")
    for i in range(len(dw)):  # value comparison, as numpy.testing.assert_array_equal
        require(dw[i] == want[i], what + ": at " + String(i) + " got " + String(dw[i]) + ", expected " + String(want[i]))
    _ = dy
    _ = ids


def residual_dropout_device_arrays_match_host_arrays() raises:
    var n = 4 * 33
    var offset = 3
    var p = Float32(0.25)
    var seed_lo = 9
    var seed_hi = 0
    var stream = 1
    var seed: UInt64 = 11
    var x = List[Float32](length=n, fill=0)
    var r = List[Float32](length=n, fill=0)
    var g = List[Float32](length=n, fill=0)
    _uniform(x, seed, Float32(-2.0), Float32(2.0))
    _uniform(r, seed, Float32(-2.0), Float32(2.0))
    _uniform(g, seed, Float32(-2.0), Float32(2.0))
    var ctx = neural_ctx[_RD_DEV_CTX]()
    # The host-array layer (the PACK door's call).
    var host_y = List[Float32](length=n, fill=0)
    var host_dx = List[Float32](length=n, fill=0)
    var host_dr = List[Float32](length=n, fill=0)
    var yp = f32_ptr(Int(host_y.unsafe_ptr()))
    _ = residual_dropout_device[False](ctx, yp, yp, f32_ptr(Int(x.unsafe_ptr())), f32_ptr(Int(r.unsafe_ptr())),
                                       n, offset, p, seed_lo, seed_hi, stream)
    var gp = f32_ptr(Int(g.unsafe_ptr()))
    _ = residual_dropout_device[True](ctx, f32_ptr(Int(host_dx.unsafe_ptr())), f32_ptr(Int(host_dr.unsafe_ptr())),
                                      gp, gp, n, offset, p, seed_lo, seed_hi, stream)
    # The same layer on training device arrays.
    var args = ResidualDropoutArgs(n, offset, seed_lo, seed_hi, stream, p)
    var handles = List[Int]()
    for _ in range(6):  # small-loop(six device arrays: y, x, r, dx, dr, g)
        handles.append(train_dev_alloc(ctx, n))
    try:
        train_dev_put(ctx, handles[1], f32_ptr(Int(x.unsafe_ptr())), n)
        train_dev_put(ctx, handles[2], f32_ptr(Int(r.unsafe_ptr())), n)
        train_dev_put(ctx, handles[5], gp, n)
        residual_dropout_dev_run([handles[0], handles[1], handles[2]], args)
        residual_dropout_backward_dev_run([handles[3], handles[4], handles[5]], args)
        var dev_y = List[Float32](length=n, fill=0)
        var dev_dx = List[Float32](length=n, fill=0)
        var dev_dr = List[Float32](length=n, fill=0)
        train_dev_get(ctx, handles[0], f32_ptr(Int(dev_y.unsafe_ptr())), n)
        train_dev_get(ctx, handles[3], f32_ptr(Int(dev_dx.unsafe_ptr())), n)
        train_dev_get(ctx, handles[4], f32_ptr(Int(dev_dr.unsafe_ptr())), n)
        ctx.synchronize()
        _same_words(dev_y, host_y, "residual dropout forward: device arrays against host arrays")
        _same_words(dev_dx, host_dx, "residual dropout backward d_values: device arrays against host arrays")
        _same_words(dev_dr, host_dr, "residual dropout backward d_residual: device arrays against host arrays")
    except error:
        for h in handles:
            train_dev_free(ctx, h)
        raise error^
    for h in handles:
        train_dev_free(ctx, h)
    _ = x
    _ = r
    _ = g


def run_embedding_ne3_fixture(fixture_id: Int) raises:
    if fixture_id == 0:
        embedding_forward_gathers()
    elif fixture_id >= 1 and fixture_id <= 4:
        var k = fixture_id - 1
        embedding_backward_folds(k // 2, PLAN_SCAN if k % 2 == 0 else PLAN_SORT)
    elif fixture_id == 5:
        residual_dropout_device_arrays_match_host_arrays()
    else:
        raise Error("embedding_ne3: no fixture " + String(fixture_id))


def _run_gpu(mut report: Report) raises:
    var cases = report.cases(EMBEDDING_NE3_FIXTURES)
    for fixture_id in range(EMBEDDING_NE3_FIXTURES):
        try:
            run_embedding_ne3_fixture(fixture_id)
        except error:
            cases.fail(fixture_id, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("embedding_ne3_entries", cpu=False, gpu=True, serial="", run=run_cases)
