# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Concurrent `linalg.qr` and `linalg.svd` calls in one process (lane linalg-owner, 2026-11-22).

The public QR ('reduced') and SVD (compute_uv, not full, not hermitian) of a tall matrix take the TSQR route: XD_TSQR_R
factors and keeps the factorization, the door signs R (QR) or takes the SVD of R (SVD), then XD_TSQR_Q applies the
kept Q (bindings/linalg_ops_decomp_door.mojo `_lin_qr_q`, `_lin_svd`). The kept factorization was ONE process slot
(x_decomp/tsqr_host.mojo TS_HOST_STATE, its device twin TS_DEV_STATE), so two calls at once could apply each other's
factor or free it mid-apply. It is now a native owner XD_TSQR_R returns and XD_TSQR_Q frees.

This check runs `LINALG_CONCURRENT_CALLS` QR and SVD calls at once across the host cores (the check is run 20
times), through the host x_decomp image's table (`x_decomp_host_v1`) in the door's order (linalg_numpy_cases.mojo
`qr_mode`, `svd_route`), and compares every result word for word with the same call run alone first: a call's
result does not depend on what else the process is doing.

    test_qr_svd_concurrent       QR and SVD dealt over the calls, three shapes (one, two and three TSQR row blocks)
    test_owners_interleaved      two factorizations kept at once, applied in the other order, each equal to alone
"""
from std.testing import TestSuite
from core.entry_table import EntryRef, entry_ref_word, entry_ref_call
from core.host_parallel import host_parallelize
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_decomp.descriptor_schema import XD_OPERATION_COUNT, XD_CAPABILITIES, XD_R_SIGNS
from x_decomp.descriptor_host import x_decomp_host_v1
from bindings.linalg_ops_kit_run import LinKit
from checks.native_tests.linalg_numpy_cases import Mat, mat_f32, qr_mode, svd_route, same_bits, tsqr_r, tsqr_q
from checks.native_tests.testsuite_cases import require

comptime LINALG_CONCURRENT_CALLS = 100
comptime _SHAPES = 3
comptime _KINDS = 6  # (shape, QR or SVD)


def _entry() -> EntryRef:
    return EntryRef(x_decomp_host_v1, entry_ref_word(Int(GLOBAL_NUMERIC_MODE), 0, XD_OPERATION_COUNT))


def _kit() raises -> LinKit:
    var entry = _entry()
    return LinKit(entry, entry_ref_call(entry, XD_CAPABILITIES, List[Int](), List[Int](), List[Float64]()))


def _shape(which: Int) -> Tuple[Int, Int]:
    """(rows, cols): one TSQR row block (TS_ROWS = 2048), two, three; all tall and at most TS_MAX_N wide."""
    if which == 0:
        return (64, 5)
    elif which == 1:
        return (2600, 20)
    return (4500, 7)


def _matrix(rows: Int, cols: Int, salt: Int) -> List[Float32]:
    """A fixed pseudo-random matrix (64-bit LCG, top 24 bits to [-1, 1))."""
    var x = List[Float32](length=rows * cols, fill=0)
    var state = UInt64(0x9E3779B97F4A7C15) ^ UInt64(salt)
    for i in range(rows * cols):  # small-loop(fixture: at most 52,000 words, filled once per call)
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        x[i] = Float32(Int((state >> 40) & 0xFFFFFF)) / Float32(8388608.0) - Float32(1.0)
    return x^


@fieldwise_init
struct CallOut(Movable):
    """QR: (Q, R, -); SVD: (U, S, Vh)."""
    var a: Mat
    var b: Mat
    var c: Mat


def _call(kind: Int) raises -> CallOut:
    """One public call of `kind` (shape kind // 2; QR when kind is even, SVD otherwise) on its own kit."""
    var shape = _shape(kind // 2)
    var rows = shape[0]
    var cols = shape[1]
    var x = _matrix(rows, cols, kind // 2)
    var kit = _kit()
    try:
        # the parts are swapped out (AGENTS.md part 6: no nested field moved with ^)
        var out = CallOut(Mat(1, 1), Mat(1, 1), Mat(1, 1))
        if kind % 2 == 0:
            var qr = qr_mode(kit, x, rows, cols, 0)
            swap(out.a, qr.first)
            swap(out.b, qr.second)
            _ = qr^
        else:
            var usv = svd_route(kit, x, rows, cols, False, False)
            swap(out.a, usv.u)
            swap(out.b, usv.s)
            swap(out.c, usv.vh)
            _ = usv^
        _ = x^
        return out^
    finally:
        kit.close()


def _compare(got: CallOut, want: CallOut, label: String) raises:
    same_bits(label + " first", got.a, want.a)
    same_bits(label + " second", got.b, want.b)
    same_bits(label + " third", got.c, want.c)


def _round(alone: List[CallOut], rnd: Int) raises:
    var failed = List[Int](length=LINALG_CONCURRENT_CALLS, fill=0)
    var messages = List[String](length=LINALG_CONCURRENT_CALLS, fill=String(""))
    var fp = failed.unsafe_ptr()
    var mp = messages.unsafe_ptr()
    var ap = alone.unsafe_ptr()
    def task(unit: Int) {imm fp, imm mp, imm ap, imm rnd}:
        try:
            var kind = (unit + rnd) % _KINDS
            var got = _call(kind)
            _compare(got, ap[unsafe_offset=kind], "round " + String(rnd) + " call " + String(unit)
                     + (" qr" if kind % 2 == 0 else " svd") + " shape " + String(kind // 2))
        except problem:
            mp[unsafe_offset=unit] = String(problem)
            fp.unsafe_store(unit, 1)
    host_parallelize(task, LINALG_CONCURRENT_CALLS)
    var count = 0
    var first = String("")
    for i in range(LINALG_CONCURRENT_CALLS):  # small-loop(the calls' failure words)
        if failed[i] != 0:
            if count == 0:
                first = messages[i]
            count += 1
    _ = alone
    _ = messages^
    _ = failed^
    require(count == 0, "round " + String(rnd) + ": " + String(count) + " of " + String(LINALG_CONCURRENT_CALLS)
        + " concurrent linalg calls differ from the call alone; first: " + first)


def test_qr_svd_concurrent() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var alone = List[CallOut]()
    for kind in range(_KINDS):  # driver-loop: the reference calls, one at a time, before the concurrent ones
        alone.append(_call(kind))
    # one round of LINALG_CONCURRENT_CALLS calls at once per run; the orchestrator runs the check 20 times
    _round(alone, 0)
    _ = alone^


def _q_of(entry: EntryRef, owner: Int, x: List[Float32], r: List[Float32], rows: Int, cols: Int) raises -> Mat:
    """Q (rows x cols) of the factorization `owner` holds, R signed first (the door's QR order); frees the owner."""
    var signed = r.copy()
    var signs = List[Float32](length=cols * cols, fill=0.0)
    var addresses: List[Int] = [Int(signed.unsafe_ptr()), Int(signs.unsafe_ptr())]
    _ = entry_ref_call(entry, XD_R_SIGNS, addresses, [cols], List[Float64]())
    var q = tsqr_q(entry, owner, Int(signs.unsafe_ptr()), rows, cols, cols)
    _ = signed^
    _ = signs^
    _ = x
    return mat_f32(q, rows, cols)


def test_owners_interleaved() raises:
    """Factor A, factor B (both kept), apply B, apply A: each Q equals the same QR alone. Through the one process
    slot, B's factor replaced A's and A's apply refused (or used B's factor)."""
    var entry = _entry()
    var sa = _shape(2)
    var sb = _shape(1)
    var xa = _matrix(sa[0], sa[1], 2)
    var xb = _matrix(sb[0], sb[1], 1)
    var want_a = _call(4)
    var want_b = _call(2)
    var owner_a = 0
    var owner_b = 0
    var ra = tsqr_r(entry, Int(xa.unsafe_ptr()), sa[0], sa[1], owner_a)
    var rb = tsqr_r(entry, Int(xb.unsafe_ptr()), sb[0], sb[1], owner_b)
    require(owner_a != 0 and owner_b != 0 and owner_a != owner_b, "each kept factorization has its own owner")
    var qb = _q_of(entry, owner_b, xb, rb, sb[0], sb[1])
    var qa = _q_of(entry, owner_a, xa, ra, sa[0], sa[1])
    same_bits("interleaved Q of A", qa, want_a.a)
    same_bits("interleaved Q of B", qb, want_b.a)
    _ = xa^
    _ = xb^
    _ = ra^
    _ = rb^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
