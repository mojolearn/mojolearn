# SPDX-License-Identifier: Apache-2.0
"""The CPU training image's neural RNG: a slice of the stream equals the
same words of the whole stream, and the dropout kinds replay one mask (lane
tmig-sequence-1, 2026-11-14; ported from
`test_cpu_training_samba.py::test_rng_slices_equal_the_whole_stream_when_built`).

`core/mojolearn_training_host_impl.mojo` `neural_rng_host_into` (the host
twin of core/philox_neural.mojo's kernel that `_mojolearn_training_host`'s
`neural_rng` calls) with seed (7 << 32) | 3 and stream 5:

  unit 0  uniform (lo -0.5, span 1.0): 37 words from offset 0 against 11
          words from offset 13, bit for bit; every word finite;
  unit 1  normal (0, 1): the same;
  unit 2  dropout forward (p 0.5, scale 2.0) of x = 1..37 against the 11
          words from offset 13 of x[13:24]; the mask keeps some words and
          drops some; a kept word is x * 2.0 exactly; the backward of a
          constant 3.0 gradient is non-zero exactly where the forward kept.

The units are independent and run across the native host worker pool.
"""
from std.memory import bitcast
from std.math import isfinite
from core.host_parallel import host_parallelize
from core.mojolearn_training_host_impl import neural_rng_host_into
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


# ---- The host column (was training_neural_rng_slices_cpu_suite.mojo) ----
comptime RNG_UNIFORM = 0
comptime RNG_NORMAL = 1
comptime RNG_DROPOUT_FORWARD = 2
comptime RNG_DROPOUT_BACKWARD = 3
comptime SEED_LO = 3
comptime SEED_HI = 7
comptime STREAM = 5
comptime NEURAL_RNG_UNITS = 3


def _draw(kind: Int, n: Int, offset: Int, a: Float32, b: Float32, var inp: List[Float32]) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=0.0)
    _ = neural_rng_host_into(Int(out.unsafe_ptr()), Int(inp.unsafe_ptr()), n, offset, SEED_LO, SEED_HI, STREAM,
                             kind, a, b)
    _ = inp^  # read by address above
    return out^


def _same_slice(whole: List[Float32], part: List[Float32], start: Int, what: String) raises:
    for i in range(len(part)):  # small-loop(eleven slice words)
        require(bitcast[DType.uint32](whole[start + i]) == bitcast[DType.uint32](part[i]),
                what + ": word " + String(start + i) + " of the whole stream differs from the slice")


def _stream(kind: Int, a: Float32, b: Float32, what: String) raises:
    var placeholder: List[Float32] = [0.0]
    var whole = _draw(kind, 37, 0, a, b, placeholder.copy())
    var part = _draw(kind, 11, 13, a, b, placeholder.copy())
    _same_slice(whole, part, 13, what)
    for i in range(len(whole)):  # small-loop(37 stream words)
        require(isfinite(whole[i]), what + ": word " + String(i) + " is not finite")


def _dropout() raises:
    var x = List[Float32](length=37, fill=0.0)
    for i in range(37):  # small-loop(37 fixture words)
        x[i] = Float32(i) + 1.0
    var tail = List[Float32](length=11, fill=0.0)
    for i in range(11):  # small-loop(eleven slice words)
        tail[i] = x[13 + i]
    var fwd = _draw(RNG_DROPOUT_FORWARD, 37, 0, 0.5, 2.0, x.copy())
    var sub = _draw(RNG_DROPOUT_FORWARD, 11, 13, 0.5, 2.0, tail^)
    _same_slice(fwd, sub, 13, "dropout forward")
    var kept = 0
    for i in range(37):  # small-loop(37 stream words)
        if fwd[i] != 0:
            kept += 1
            require(fwd[i] == x[i] * 2.0, "dropout forward: kept word " + String(i) + " is not x * 2.0")
    require(kept > 0 and kept < 37, "a p = 0.5 mask kept everything or nothing")
    var dx = _draw(RNG_DROPOUT_BACKWARD, 37, 0, 0.5, 2.0, List[Float32](length=37, fill=3.0))
    for i in range(37):  # small-loop(37 stream words)
        require((dx[i] != 0) == (fwd[i] != 0), "the backward did not replay the forward's mask at word " + String(i))


def run_case(unit: Int) raises:
    if unit == 0:
        _stream(RNG_UNIFORM, -0.5, 1.0, "uniform")
    elif unit == 1:
        _stream(RNG_NORMAL, 0.0, 1.0, "normal")
    else:
        _dropout()


def _run_cpu(mut report: Report) raises:
    var cases = report.cases(NEURAL_RNG_UNITS)
    def task(unit: Int) {imm cases}:
        try:
            run_case(unit)
        except problem:
            cases.fail(unit, problem)
    report.parallel(task, NEURAL_RNG_UNITS)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("training_neural_rng_slices", cpu=True, gpu=False, serial="", run=run_cases)
