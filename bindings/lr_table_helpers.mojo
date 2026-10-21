# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The learning-rate schedule entries of the core bindings: the Python door
of sequence/lr_table.mojo (the binary64 table, its float32 decision, the
exact route over a run of steps, and `lr_spec_fill`, the one native call of
every public schedule method). A shell (lane neural-optimizers,
2026-10-21): each entry unpacks its argument words, calls one typed kernel
entry and converts the result back; the arithmetic moved word for word to
sequence/lr_table.mojo."""
from std.python import PythonObject

from std.python import Python
from sequence.lr_exact import LrOut, onecycle_exact, pow_exact, schedule_exact
from sequence.lr_table import (
    F32P, F64P, U8P, LrSpec, decide_many, exact_block, lr_spec_fill, onecycle_fill, pow_values,
    schedule_block_values,
)


def _f64p(addr: PythonObject) raises -> F64P:
    var a = Int(py=addr)
    if a == 0:
        raise Error("lr table: null float64 buffer")
    return F64P(unsafe_from_address=a)


def _ints(words: PythonObject, count: Int) raises -> List[Int]:
    var out = List[Int]()
    for i in range(count):  # driver-loop: over argument words (a fixed count per entry point)
        out.append(Int(py=words[i]))
    return out^


def _floats(words: PythonObject, count: Int) raises -> List[Float64]:
    var out = List[Float64]()
    for i in range(count):  # driver-loop: over argument words (a fixed count per entry point)
        out.append(Float64(py=words[i]))
    return out^


def lr_schedule_values_binding(params: PythonObject, vs_addr: PythonObject, es_addr: PythonObject) raises -> PythonObject:
    """`_Schedule`'s binary64 run into vs / es (t1 - t0 float64 each).
    params: [kind (0 constant, 1 linear, 2 cosine), peak, lo, warmup,
    total (-1: none), t0, t1]. Returns 1 when the run has no binary64
    route (every step takes the exact route), else 0."""
    return PythonObject(schedule_block_values(
        Int(py=params[0]), Float64(py=params[1]), Float64(py=params[2]), Int(py=params[3]),
        Int(py=params[4]), Int(py=params[5]), Int(py=params[6]), _f64p(vs_addr), _f64p(es_addr),
    ))


def lr_onecycle_fill_binding(params: PythonObject, vs_addr: PythonObject, es_addr: PythonObject) raises -> PythonObject:
    """One OneCycleLR phase into vs / es from index `at`. params: [linear
    (1) or cos (0), at, st0, count, start_f, end_f, af, bf]. Returns 0."""
    onecycle_fill(
        Int(py=params[0]) != 0, Int(py=params[1]), Int(py=params[2]), Int(py=params[3]),
        Float64(py=params[4]), Float64(py=params[5]), Float64(py=params[6]), Float64(py=params[7]),
        _f64p(vs_addr), _f64p(es_addr),
    )
    return PythonObject(0)


def lr_decide_binding(vs_addr: PythonObject, es_addr: PythonObject, n: PythonObject,
                      out_addr: PythonObject, ok_addr: PythonObject) raises -> PythonObject:
    """Per value the float32 every real in [v - e, v + e] rounds to, kept
    (ok = 1) when both ends round to the same normal positive float32.
    Returns how many steps were NOT decided."""
    var count = Int(py=n)
    var oa = Int(py=out_addr)
    var ka = Int(py=ok_addr)
    if count > 0 and (oa == 0 or ka == 0):
        raise Error("lr table: null output buffer")
    return PythonObject(decide_many(
        _f64p(vs_addr), _f64p(es_addr), count, F32P(unsafe_from_address=oa), U8P(unsafe_from_address=ka),
    ))


# ----------------------------------------------------------- the exact route
# The steps the table cannot decide, and the schedules without a table,
# answered on the big rationals. Each returns [status, value]: status 0 ok,
# 1 overflow (Python's OverflowError), 2 a 64-term straddle
# (ArithmeticError), 3 a bad progress.


def _pair(r: LrOut) raises -> PythonObject:
    # Python.list owns its items once; list.append from Mojo would retain each.
    return Python.list(PythonObject(r.status), PythonObject(Float64(r.value)))


def lr_schedule_exact_binding(ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """The exact route of one `_Schedule` step: ip = [kind (0 constant, 1
    linear, 2 cosine), warmup, total (-1: none), t >= 1]; fp = [peak,
    min_lr]."""
    return _pair(schedule_exact(Int(py=ip[0]), Float64(py=fp[0]), Float64(py=fp[1]), Int(py=ip[1]),
                                Int(py=ip[2]), Int(py=ip[3])))


def lr_onecycle_exact_binding(ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """The exact route of one OneCycleLR step: ip = [linear, three_phase,
    total_steps, torch step]; fp = [max_lr, div_factor, final_div_factor,
    phase-1 end, phase-2 end]."""
    return _pair(onecycle_exact(Int(py=ip[0]) != 0, Int(py=ip[1]) != 0, Float64(py=fp[0]), Float64(py=fp[1]),
                                Float64(py=fp[2]), Float64(py=fp[3]), Float64(py=fp[4]), Int(py=ip[2]),
                                Int(py=ip[3])))


def _status_pair(r: Tuple[Int, Int]) raises -> PythonObject:
    return Python.list(PythonObject(r[0]), PythonObject(r[1]))


def lr_pow_values_binding(fp: PythonObject, ip: PythonObject, out_addr: PythonObject) raises -> PythonObject:
    """StepLR / ExponentialLR: out[i] (float32 words) = the float32 nearest
    base gamma^((e0 + i) // step) for i < n. fp = [base, gamma]; ip = [e0 >=
    0, n, step >= 1]. Returns [0, n], or [1, i] at the first overflowing
    step i."""
    return _status_pair(pow_values(
        Float64(py=fp[0]), Float64(py=fp[1]), Int(py=ip[0]), Int(py=ip[1]), Int(py=ip[2]),
        F32P(unsafe_from_address=Int(py=out_addr)),
    ))


def _exact_block(kind: Int, ip: PythonObject, fp: PythonObject, out_addr: PythonObject,
                 ok_addr: PythonObject) raises -> PythonObject:
    """The exact route over a run: out[i] (float32 words) for every i < n
    whose ok[i] is 0 (ok_addr 0: every i). kind 0: `_Schedule`, ip = [kind,
    warmup, total, t0, n], fp = [peak, min_lr]; kind 1: OneCycleLR, ip =
    [linear, three_phase, total_steps, t0, n], fp as lr_onecycle_exact.
    Returns [status, index]."""
    var words = _ints(ip, 5)
    var oka = Int(py=ok_addr)
    var r = exact_block(
        kind, words, _floats(fp, 2 if kind == 0 else 5), words[3], words[4],
        F32P(unsafe_from_address=Int(py=out_addr)), U8P(unsafe_from_address=oka), oka != 0,
    )
    if r[0] != 0:
        return _status_pair(r)
    return _status_pair((0, 0))


def lr_schedule_exact_block_binding(ip: PythonObject, fp: PythonObject, out_addr: PythonObject,
                                    ok_addr: PythonObject) raises -> PythonObject:
    return _exact_block(0, ip, fp, out_addr, ok_addr)


def lr_onecycle_exact_block_binding(ip: PythonObject, fp: PythonObject, out_addr: PythonObject,
                                    ok_addr: PythonObject) raises -> PythonObject:
    return _exact_block(1, ip, fp, out_addr, ok_addr)


# ------------------------------------------------- the public schedule methods


def _spec(ip: PythonObject, fp: PythonObject) raises -> LrSpec:
    return LrSpec(
        Int(py=ip[0]), Int(py=ip[1]), Int(py=ip[2]), Int(py=ip[3]),
        Float64(py=fp[0]), Float64(py=fp[1]), Float64(py=fp[2]), Float64(py=fp[3]),
    )


def lr_schedule_fill_binding(ip: PythonObject, fp: PythonObject, out_addr: PythonObject,
                             res_addr: PythonObject) raises -> PythonObject:
    """`lr_at` and `lr_values` of every built-in schedule (sequence/
    lr_table.mojo::lr_spec_fill): the float32 rates of ONE-BASED steps t0 ..
    t0 + n - 1 into `out_addr` (n float32). ip = [kind, a, b, c, t0, n] and
    fp = [f0, f1, f2, f3] are `LrSpec`'s words. `res_addr` (one int64)
    receives the index of the first failing step. Returns the status: 0, or
    1 overflow, 2 a float32 straddle, 3 a bad progress, 4 a step below 1, 5
    a OneCycleLR step past total_steps + 1."""
    var res = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=res_addr))
    var r = lr_spec_fill(_spec(ip, fp), Int(py=ip[4]), Int(py=ip[5]), F32P(unsafe_from_address=Int(py=out_addr)))
    res[0] = Int64(r[1])
    return PythonObject(r[0])
