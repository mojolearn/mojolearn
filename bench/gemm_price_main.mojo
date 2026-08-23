"""What `mojolearn.identical.gemm.fp32.v1` COSTS: the Phase 4 measurement harness.

Driven by `tools/gemm_price.sh`, which alternates the two build modes round by
round. Emits the same `PRICE <mode> <arm> <ms>` lines
`bench/linalg_price_main.mojo` and `bench/identity_price_main.mojo` emit, so
the median table is SHARED and a v1 number is directly comparable with the
shipped kernels' numbers rather than living in its own units.

THE SPECIFICATION IS CONTRACT SECTION 13.6, WHICH NAMES SIX MEASUREMENTS.
This file implements the ones that can run today and NAMES the ones that
cannot, with the reason, rather than quietly measuring five and reporting six.

    13.6.1  the fold in isolation, both topologies, P sweep     PARTIAL (host)
    13.6.2  fully staged vs single-block-fused                  NOT RUN (device)
    13.6.3  end-to-end over the shape table                     PARTIAL (host)
    13.6.4  the ftz cost SEPARATELY from the fma cost           RUNS (host)
    13.6.5  workspace bytes and the tiling threshold            RUNS (exact)
    13.6.6  which resource limits, per shape                    PARTIAL (inputs)

**PHASE 2b's DEVICE KERNEL DOES NOT EXIST YET, AND THIS FILE IS BUILT SO THAT
LANDING IT IS A ONE-CALL CHANGE.** Everything that is device-shaped -- the
launch-count question 13.6.2 is entirely about, the FAST vendor arm, the
shipped pinned kernel, the achieved GFLOP/s and the limiter classification --
routes through the `device` arm, which is a NAMED RAISING STUB carrying the
exact wiring instruction, exactly as `bench/gemm_card_main.mojo`'s device arm
does. The host arms are not a placeholder for it: `gemm_oracle` is the
NORMATIVE v1 answer and `gemm_oracle_serial` is the diagnostic reference, so
pricing them establishes the reference cost and proves the harness works
before there is a kernel to point it at.

WHAT THE HOST ARMS CAN AND CANNOT SAY, stated once, because a number without
this paragraph beside it will be read as a device number.

  CAN.  The relative cost of the two FOLD TOPOLOGIES over identical partials.
        The relative cost of the leaf loop's three spellings, which is
        13.6.4's ftz-vs-fma question and is the item section 5 calls the
        largest single cost in the profile. Workspace bytes exactly, because
        those are arithmetic on `m`, `n` and `P` and not a measurement at all.

  CANNOT.  Anything about LAUNCHES. 13.4's whole argument is that `D` levels
        fuse into ONE launch under A4, and a host loop has no launches to
        count, so the host CANNOT support or refute 13.4 -- only the device
        arm can, and 13.6.1 says so ("the only measurement that can support or
        refute 13.4"). Nor anything about bandwidth, occupancy, or achieved
        GFLOP/s on a GPU.

THE SHAPES COME FROM `bench/gemm_shapes.mojo` AND NONE ARE INVENTED HERE. That
file is the table with provenance; a benchmark that grows its own shape list
is a second definition of what the profile was tested on.

M AND N ARE CAPPED, K IS EXACT, AND THAT IS THE WHOLE REASON THE HOST ARM IS
WORTH RUNNING. `k` is the axis the leaf rule and the fold topology read;
contract 6.1 forbids them to read `m` or `n`. So a run at reduced `m n` and
exact `k` exercises the EXACT leaf count, the EXACT leaf size, the EXACT
ragged tail and the EXACT tree the full shape would -- and costs
proportionally less, because section 0.3 makes each output cell's arithmetic
independent of every other. The per-cell price is therefore the thing measured
and the full-shape host cost is DERIVED from it by multiplication, printed as
DERIVED and never as measured.

MEASUREMENT DISCIPLINE, and this repository has paid for every line of it.

  - **The two arms alternate INSIDE the timed loop, call by call.** Both host
    arms live in ONE binary, so unlike the FAST/IDENTICAL axis they do not
    need process-level alternation and get the stronger thing instead: `for r
    in REPEATS: time(oracle); time(serial)`. The M4 governor drifts up to 1.7x
    across twenty minutes (`[[mojolearn-box-drifts]]`) and a block of A then a
    block of B measures the drift.
  - **The two MODES are two BINARIES** -- `GLOBAL_NUMERIC_MODE` is comptime --
    and cannot be interleaved inside one process. `tools/gemm_price.sh`
    alternates them round by round and the ratio is a BAND, not a figure.
  - **An untimed warm-up call precedes every timed loop.** A first call pays
    its page faults and its cold caches and that is a price nobody runs twice.
  - **This binary prints the mode it COMPILED in** and the driver reads it
    back. Do not trust the flip; three mislabelled measurements were caught by
    that witness on 2026-08-23 alone.
  - **No number here is trustworthy while another lane runs a GPU leg**
    (contract 13.5 assumption A5). The driver checks for one and says so.

WHAT MAY NOT BE CONCLUDED FROM ANY NUMBER THIS FILE PRINTS -- contract 13.7,
honored rather than cited:

  - **Not that the balanced fold is faster than the serial fold, nor slower.**
    13.6.1 is priced here on the HOST and against the ORACLE's spelling, which
    allocates a fresh List per tree level; a device kernel allocates nothing.
    The host ratio is a fact about two host functions.
  - **Not the old ~15 GFLOP/s hand-written contraction number**, which is not
    this design and is not generalized here.
  - **Not the k-NN lane's 2.85x nor the linalg lane's 4.7x as universal.**
    Those are two specific arms at two specific shapes.
  - **Nothing here licenses changing the tree.** If a number comes out bad the
    answer is an execution-plan change that moves no bits (13.7's last item).
"""

from std.math import fma
from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns

from bench.gemm_shapes import (
    GEMM_SHAPE_COUNT,
    contract_leaf_size,
    gemm_shape_k,
    gemm_shape_m,
    gemm_shape_n,
    gemm_shape_name,
    gemm_shape_op,
    last_leaf_len,
    leaf_count,
)
from bench.gemm_shapes import OP_NN as TBL_OP_NN
from bench.gemm_shapes import OP_NT as TBL_OP_NT
from bench.gemm_shapes import OP_TN as TBL_OP_TN
from gemm.mojo_only.gemm_oracle import (
    OP_NN,
    OP_NT,
    OP_TN,
    fold_balanced_tree,
    fold_level_count,
    fold_node_total,
    gemm_oracle,
    gemm_oracle_serial,
    op_name,
)
from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
)


def _mode() -> String:
    """The mode this binary COMPILED in, read from the comptime constant.

    Not the environment, not the flag that was passed. With the flip replaced
    by a `-D` define this is the only witness between a mis-plumbed build and
    a correctly-labelled measurement of the wrong arm.
    """
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


def _report(name: String, ms: Float64):
    """`PRICE <mode> <arm> <ms>`: four whitespace fields, the format
    `tools/price_linalg_identity.sh` and `tools/gemm_price.sh` both parse."""
    print("PRICE", _mode(), name, ms)


def _padr(s: String, w: Int) -> String:
    """Left-justify into `w` columns. A ragged table is a table people read
    the wrong column of."""
    var out = s
    while out.byte_length() < w:
        out = out + " "
    return out


def _padl(s: String, w: Int) -> String:
    """Right-justify into `w` columns, for numbers."""
    var out = s
    while out.byte_length() < w:
        out = " " + out
    return out


def _ratio(a: Float64, b: Float64) -> Float64:
    """`a / b`, and 0 when `b` is zero. A denominator of zero here means a
    timed region came out below the clock's resolution, which is a fact about
    the clock and not a ratio; printing `inf` or trapping would both be
    worse than printing a zero the caption explains."""
    if b <= 0.0:
        return 0.0
    return a / b


def _fixed(x: Float64, places: Int) -> String:
    """`x` to `places` decimals, WITHOUT a host float chain in the value
    being reported -- this formats a number that was already measured, it
    does not compute one. Mojo's default float printing is variable width,
    which makes a column of them unreadable."""
    var scale = 1
    for _ in range(places):
        scale *= 10
    var neg = x < 0.0
    var v = -x if neg else x
    var whole = Int(v)
    var frac = Int((v - Float64(whole)) * Float64(scale) + 0.5)
    if frac >= scale:
        whole += 1
        frac -= scale
    var fs = String(frac)
    while fs.byte_length() < places:
        fs = "0" + fs
    var out = String(whole) + "." + fs
    if neg:
        out = "-" + out
    return out


# ===========================================================================
# TUNABLES
# ===========================================================================

#: The host budget per shape in multiply-accumulates. The host oracle is a
#: scalar loop with a branch per operand fetch, so this is wall clock.
#:
#: **THE FLOOR IS THE TABLE'S LARGEST `k`, NOT A ROUND NUMBER.** `m` and `n`
#: cap to 1 and no further, so a budget below `max k = 1,000,000` SKIPS
#: `gram.32x32x1M` -- the flagship Gram shape and the table's only real
#: `P = 1024` row, the one row whose fold is at the profile's cap. A budget
#: that silently drops the most interesting shape is exactly the
#: build-to-datasets failure, so the default clears it with room and the
#: harness REFUSES a budget that would not.
comptime DEFAULT_MAC_BUDGET = 2_000_000

#: Timed calls per arm per shape. Small on purpose: the round is the sample
#: unit and `tools/gemm_price.sh` takes the median over rounds, which is what
#: survives a thermal excursion. Averaging harder inside one round just
#: averages one point on the drift curve more precisely.
comptime REPEATS = 2

#: 13.6.1's fixed small output: "a fixed small `m n` (say 64 x 64)".
comptime FOLD_M = 64
comptime FOLD_N = 64
comptime FOLD_CELLS = FOLD_M * FOLD_N

#: 13.6.1's `P` sweep, verbatim: `1, 2, 3, 8, 32, 128, 1018, 1024`.
#: `P = 1018` is the non-power-of-two CARRYING case and is in the sweep
#: because without it the carry path is priced at zero.
comptime FOLD_P_COUNT = 8


def fold_sweep_p(i: Int) -> Int:
    if i == 0:
        return 1
    if i == 1:
        return 2
    if i == 2:
        return 3
    if i == 3:
        return 8
    if i == 4:
        return 32
    if i == 5:
        return 128
    if i == 6:
        return 1018
    return 1024


#: 13.6.4's leaf loop length, and the passes over it. One pass is one output
#: cell's whole-`k` contraction at a `k` in the flat leaf band.
comptime LEAF_K = 262_144
comptime LEAF_PASSES = 8

#: A device budget for 13.6.5's flag, in mebibytes. Not a hardware constant
#: and deliberately not read from the device: this file is host-only and a
#: threshold that changed with the machine would make the workspace table
#: unreproducible. Override with `MOJOLEARN_GEMM_PRICE_WS_BUDGET_MB`.
comptime DEFAULT_WS_BUDGET_MB = 4096


# ===========================================================================
# BIT-ASSEMBLED FIXTURE VALUES -- NO HOST FLOAT ARITHMETIC
# ===========================================================================
# `bench/linalg_trace_main.mojo` learned this the expensive way and
# `bench/gemm_card_main.mojo` states it: a host `v * w` chain is IDENTITY_PATHS
# row 18's contraction decision, so two machines running the same driver would
# feed DIFFERENT inputs to the thing being priced. It matters less for a
# TIMING harness than for a card -- a timing does not diff -- but the same
# generator costs nothing and keeps this file usable as a fixture source if
# anyone ever wants the values as well as the seconds.


def _mix(i: Int, salt: Int) -> UInt64:
    var z = (
        UInt64(i + 1) * 0x9E3779B97F4A7C15
        + UInt64(salt + 1) * 0xBF58476D1CE4E5B9
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _exact(i: Int, salt: Int) -> Float32:
    """A value exact in Float32 on every backend: `<int below 2^20> / 2^20`.

    The numerator is below 2^21 so it is exact in a Float32 mantissa and the
    divisor is a power of two, so the division is exact. Copied in shape from
    `bench/gemm_card_main.mojo::_exact` on purpose -- one generator idiom
    across this lane's drivers.
    """
    var num = Int(_mix(i, salt) % 2097151) - 1048575
    return Float32(num) / Float32(1048576.0)


def _subnormal(i: Int, salt: Int) -> Float32:
    """A SUBNORMAL Float32, assembled from bits and never computed.

    Float32 bit patterns `0x00000001 .. 0x007FFFFF` are exactly the positive
    subnormals. Used only by 13.6.4's fourth arm, which prices the flush
    branch TAKEN rather than merely tested.
    """
    var bits = UInt32(Int(_mix(i, salt) % 8388606) + 1)
    return bitcast[DType.float32](bits)


def _fill(n: Int, salt: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(_exact(i, salt))
    return out^


def _fill_subnormal(n: Int, salt: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(_subnormal(i, salt))
    return out^


def _bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


# ===========================================================================
# THE TWO OP ENCODINGS
# ===========================================================================


def _tbl_op(i: Int) -> Int:
    """Translate `bench/gemm_shapes.mojo`'s orientation code into
    `gemm/mojo_only/gemm_oracle.mojo`'s.

    **THE TWO FILES NUMBER THE SAME THREE WORDS DIFFERENTLY** -- the table is
    `NT=0, TN=1, NN=2` and the oracle is `NN=0, NT=1, TN=2` -- so an untranslated
    hand-off runs every row against the WRONG ADDRESSING and still returns a
    plausible float, because the operand buffers are usually large enough that
    the mis-indexing stays in bounds. `gemm/mojo_only/gemm_device_check.mojo`
    writes the map once for the same reason; this is the same map, and
    `check_op_encodings_are_not_interchangeable` below asserts it rather than
    assuming it.
    """
    var o = gemm_shape_op(i)
    if o == TBL_OP_NT:
        return OP_NT
    if o == TBL_OP_TN:
        return OP_TN
    return OP_NN


def check_op_encodings_are_not_interchangeable() raises:
    """The map is a MAP, and the identity would be wrong.

    Asserted rather than commented, because the failure mode is silent: the
    values come out wrong and the program does not stop. The check has two
    halves and both are load bearing. First, every table code round-trips to
    the oracle code with the SAME NAME, via `op_name`, so the map is verified
    against the oracle's own spelling of the word and not against a constant
    written here. Second, the two encodings genuinely DISAGREE somewhere -- if
    a later edit made them coincide, this check would still pass on the first
    half while the map became dead code, and a reader would be entitled to
    delete it.
    """
    if op_name(_tbl_op(0)) != "TN":
        raise Error(
            "op map: table row 0 is a TN Gram and mapped to '"
            + op_name(_tbl_op(0))
            + "'"
        )
    if op_name(_tbl_op(6)) != "NT":
        raise Error(
            "op map: table row 6 is the NT distance tile and mapped to '"
            + op_name(_tbl_op(6))
            + "'"
        )
    # Compared through Lists rather than as bare aliases on purpose: a direct
    # `TBL_OP_NT == OP_NT` is comptime-foldable and the compiler warns that
    # the branch is unreachable, which is true today and would stop being
    # true exactly when this check matters.
    var tbl = List[Int]()
    tbl.append(TBL_OP_NT)
    tbl.append(TBL_OP_TN)
    tbl.append(TBL_OP_NN)
    var orc = List[Int]()
    orc.append(OP_NT)
    orc.append(OP_TN)
    orc.append(OP_NN)
    var same = 0
    for q in range(3):
        if tbl[q] == orc[q]:
            same += 1
    if same == 3:
        raise Error(
            "op map: the two encodings now COINCIDE, so this map is the"
            " identity and every caller that skipped it is accidentally"
            " correct. Either the encodings were unified on purpose -- in"
            " which case delete the map and this check together -- or one"
            " file's constants moved and the coincidence is temporary."
        )
    print(
        "check_op_encodings_are_not_interchangeable OK: table (NT,TN,NN) =",
        "(" + String(TBL_OP_NT) + "," + String(TBL_OP_TN) + ","
        + String(TBL_OP_NN) + ")",
        "-> oracle",
        "(" + String(OP_NT) + "," + String(OP_TN) + "," + String(OP_NN) + ")",
    )


def _a_elems(op: Int, m: Int, n: Int, k: Int) -> Int:
    """How many floats the LEFT operand buffer holds under the oracle's
    addressing. TN reads `A` as `k x m`; NN and NT read it as `m x k`."""
    if op == OP_TN:
        return k * m
    return m * k


def _b_elems(op: Int, m: Int, n: Int, k: Int) -> Int:
    """How many floats the RIGHT operand buffer holds. NT reads `B` as
    `n x k`; NN and TN read it as `k x n`."""
    if op == OP_NT:
        return n * k
    return k * n


# ===========================================================================
# 13.6.1: THE FOLD IN ISOLATION -- AND THE SUPERSEDED TOPOLOGY IT IS PRICED
#         AGAINST
# ===========================================================================


def _fold_serial_ascending(partials: List[Float32]) -> Float32:
    """The SUPERSEDED serial ascending fold over the SAME leaf partials.

    **THIS IS A COST PROBE AND NOTHING CONSUMES ITS BITS.** It is not a second
    opinion about the contract and it is not an alternative spelling of
    anything shipped: the v1 answer is `fold_balanced_tree` and only
    `fold_balanced_tree`. It exists because 13.6.1 asks for "the fold in
    isolation, BOTH TOPOLOGIES, same leaf stage" and the repository has no
    serial-fold-over-partials function -- `gemm_oracle_serial_cell` is a
    whole-`k` contraction, which is a different thing with no partials in it.

    The seam density MIRRORS `fold_balanced_tree`'s deliberately: one `ftz` on
    each addend, one on each sum, one at the output. A probe with fewer seams
    would price the topology difference plus a flush difference and report the
    sum as the topology.

    `check_serial_fold_probe_is_the_other_topology` proves it is genuinely the
    other topology rather than an accidental copy of the tree.
    """
    if len(partials) == 0:
        return Float32(0.0)
    var acc = ftz(partials[0])
    for t in range(1, len(partials)):
        acc = ftz(ftz(acc) + ftz(partials[t]))
    return ftz(acc)


def check_serial_fold_probe_is_the_other_topology() raises:
    """SABOTAGE-GRADE REACH CHECK on the probe above.

    A probe that quietly computed the same thing as `fold_balanced_tree` would
    report a topology cost of ~1.00x and look like a clean result. So this
    asserts BOTH directions:

      - at `P = 1` and `P = 2` the two topologies MUST agree, because a
        one-node tree has no internal node and a two-node tree is one
        addition either way. If they disagree there, the probe has an extra
        seam and every ratio it produces is contaminated.
      - at fixture F5's construction they MUST DISAGREE, by exactly 6. The
        partials are `[2^24, 1, 1, 1, 1, 1, 1, 1]`: serially every `+1` is
        swallowed by `2^24` and the answer is `2^24`; pairwise the ones sum to
        2, 2, 2, 2 then 4, 4 then 8... and the tree reaches `2^24 + 6` because
        `2^24 + 1` is the halfway case round-half-to-even discards while
        `2^24 + 2` and `2^24 + 6` are exact (ulp at `2^24` is 2). That is
        `gemm/mojo_only/gemm_oracle_check.mojo::check_f5_balanced_vs_serial_fold`'s
        fixture, reused here so the probe is anchored to a construction the
        gate file already justifies rather than to a number chosen here.

    Mode-independent: the separation is about `+`'s rounding, and neither
    topology's addition is an `identical_mul_add`, so it holds in FAST and in
    IDENTICAL alike and this check runs in both.
    """
    var one = List[Float32]()
    one.append(Float32(3.5))
    if _bits(fold_balanced_tree(one)) != _bits(_fold_serial_ascending(one)):
        raise Error("fold probe: P = 1 must agree with the tree and does not")

    var two = List[Float32]()
    two.append(Float32(3.5))
    two.append(Float32(-1.25))
    if _bits(fold_balanced_tree(two)) != _bits(_fold_serial_ascending(two)):
        raise Error("fold probe: P = 2 must agree with the tree and does not")

    var f5 = List[Float32]()
    f5.append(Float32(16777216.0))  # 2^24, assembled as an exact integer
    for _ in range(7):
        f5.append(Float32(1.0))
    var tree = fold_balanced_tree(f5)
    var serial = _fold_serial_ascending(f5)
    if _bits(tree) == _bits(serial):
        raise Error(
            "fold probe: at fixture F5's P = 8 the balanced tree and the"
            " serial fold MUST differ, and both returned "
            + String(tree)
            + ". The probe is not the other topology, so every 13.6.1"
            " ratio it produces is a measurement of one function against"
            " itself."
        )
    if tree != Float32(16777222.0) or serial != Float32(16777216.0):
        raise Error(
            "fold probe: F5 expected tree = 2^24 + 6 and serial = 2^24, got"
            " tree = "
            + String(tree)
            + " serial = "
            + String(serial)
        )
    print(
        "check_serial_fold_probe_is_the_other_topology OK: agree at P=1 and"
        " P=2, separate by 6 at F5's P=8 (tree",
        tree,
        "vs serial",
        serial,
        ")",
    )


def measure_fold_in_isolation() raises:
    """13.6.1, as far as a host can carry it.

    Sweeps `P` over `1, 2, 3, 8, 32, 128, 1018, 1024` at a fixed `64 x 64`
    output and prices the two topologies over BIT-IDENTICAL partials -- the
    leaf stage is not merely "the same", it is absent from the timed region
    entirely, which is a stronger isolation than 13.6.1 asks for.

    **WHAT THIS CANNOT SAY, and it is the important half.** 13.6.1 calls
    itself "the only measurement that can support or refute 13.4", and 13.4 is
    an argument about LAUNCHES: that all `D` levels fuse into one threadgroup
    with a barrier between them under assumption A4. A host loop has no
    launches, no threadgroups and no barriers. **So this measurement does not
    support or refute 13.4 and must not be quoted as doing so.** What it prices
    is the ORACLE's fold spelling, which allocates a fresh `List` per tree
    level; a device kernel allocates nothing, so the host tree arm carries a
    cost the device arm will not have. Read the ratio as an upper bound on the
    tree's disadvantage in the oracle, and nothing else.

    The two arms alternate call by call inside the timed loop.
    """
    print()
    print("-- 13.6.1  the fold in isolation, both topologies, P sweep --")
    print(
        "   fixed output",
        String(FOLD_M) + "x" + String(FOLD_N),
        "=",
        FOLD_CELLS,
        "cells; partials are bit-assembled and IDENTICAL between the arms",
    )
    print(
        "   HOST ONLY. 13.4 is a LAUNCH argument and there are no launches"
        " here; this refutes nothing and supports nothing about it."
    )

    # One flat buffer for the whole sweep, materialized ONCE and outside every
    # timed region. Cell `c`'s partials are `flat[c * PMAX + t]`, so a smaller
    # `P` is a prefix and no arm pays a different generation cost.
    var pmax = fold_sweep_p(FOLD_P_COUNT - 1)
    var flat = _fill(FOLD_CELLS * pmax, 77)

    var scratch = List[Float32]()
    for _ in range(pmax):
        scratch.append(Float32(0.0))

    var sink = Float32(0.0)
    for s in range(FOLD_P_COUNT):
        var p = fold_sweep_p(s)
        var tag = String("m1.P") + String(p)

        # Untimed warm-up, both arms, before either clock starts.
        for t in range(p):
            scratch[t] = flat[t]
        var warm = List[Float32]()
        for t in range(p):
            warm.append(scratch[t])
        sink += fold_balanced_tree(warm) + _fold_serial_ascending(warm)

        var ns_tree = 0
        var ns_serial = 0
        for _ in range(REPEATS):
            var t0 = perf_counter_ns()
            for c in range(FOLD_CELLS):
                var buf = List[Float32]()
                for t in range(p):
                    buf.append(flat[c * pmax + t])
                sink += fold_balanced_tree(buf)
            ns_tree += perf_counter_ns() - t0

            var t1 = perf_counter_ns()
            for c in range(FOLD_CELLS):
                var buf2 = List[Float32]()
                for t in range(p):
                    buf2.append(flat[c * pmax + t])
                sink += _fold_serial_ascending(buf2)
            ns_serial += perf_counter_ns() - t1

        _report(tag + ".tree", Float64(ns_tree) / 1.0e6 / Float64(REPEATS))
        _report(tag + ".serial", Float64(ns_serial) / 1.0e6 / Float64(REPEATS))

        var levels = fold_level_count(p) - 1
        var carries = 0
        var w = p
        while w > 1:
            if w % 2 != 0:
                carries += 1
            w = (w + 1) // 2
        print(
            "   " + _padl("P=" + String(p), 9)
            + _padl("adds/cell=" + String(p - 1), 16)
            + _padl("depth=" + String(levels), 10)
            + _padl("carries=" + String(carries), 12)
            + _padl("nodes/cell=" + String(fold_node_total(p)), 17)
            + _padl("tree " + _fixed(
                Float64(ns_tree) / 1.0e6 / Float64(REPEATS), 3) + " ms", 20)
            + _padl("serial " + _fixed(
                Float64(ns_serial) / 1.0e6 / Float64(REPEATS), 3) + " ms", 22)
            + _padl(_fixed(_ratio(Float64(ns_tree), Float64(ns_serial)), 3)
                    + "x", 9)
        )
    print("   sink", _bits(sink), "(printed so no arm is optimized away)")


# ===========================================================================
# 13.6.2: FULLY STAGED AGAINST SINGLE-BLOCK-FUSED -- NOT RUNNABLE ON A HOST
# ===========================================================================


def report_staging_structure() raises:
    """Print what 13.6.2 WOULD price, because a host cannot price it.

    13.6.2 is a LAUNCH-COUNT measurement and there is nothing to measure until
    the device kernel lands. So this prints the structure and not a time, and
    says which is which.

    What it CAN pin down today is the number 13.6.2 would be pricing: how many
    launches each topology realization costs at each `P` in the sweep, and how
    much scratch a fully staged implementation would need per output cell.
    Those are pure functions of `P` (`fold_level_count`, `fold_node_total`),
    they are the independent variable of the measurement, and having them
    tabulated means the device arm has to produce only the times.

    A driver that printed nothing here would let a reader conclude 13.6.2 was
    covered by 13.6.1. It is not: 13.6.1 holds the topology fixed and varies
    nothing about staging.
    """
    print()
    print("-- 13.6.2  fully staged vs single-block-fused -- NOT MEASURED --")
    print(
        "   Needs the Phase 2b kernel. A host has no launches, so the cost"
        " 13.6.2 exists to price cannot appear in any number this binary"
        " prints. The STRUCTURE it would price:"
    )
    print(
        "     " + _padl("P", 5) + _padl("D", 4) + _padl("staged", 8)
        + _padl("fused", 7) + _padl("nodes/cell", 12) + "  note"
    )
    for s in range(FOLD_P_COUNT):
        var p = fold_sweep_p(s)
        var d = fold_level_count(p) - 1
        # Staged: one leaf kernel plus one kernel per ARITHMETIC level.
        # Fused: one leaf kernel plus ONE kernel that folds every level with a
        # barrier between them (contract 13.4, under assumption A4).
        #
        # AT `P == 1` THERE IS NO FOLD AT ALL -- contract 7.3, and it is not a
        # bypass: a one-node tree HAS no internal node to skip. So both
        # realizations are ONE launch and the earlier draft of this table,
        # which printed a flat 2 in the fused column, overstated the fused
        # cost at exactly the row where the fold does not exist. Four of the
        # twenty real shapes sit on that row.
        var staged = d + 1
        var fused = 1 if p == 1 else 2
        var note = String("")
        if p == 1:
            note = String("P=1: NO fold addition (contract 7.3)")
        elif p % 2 != 0:
            note = String("odd P: the tree CARRIES")
        print(
            "     " + _padl(String(p), 5) + _padl(String(d), 4)
            + _padl(String(staged), 8) + _padl(String(fused), 7)
            + _padl(String(fold_node_total(p)), 12) + "  " + note
        )
    print(
        "   Contract 13.4's expectation, which is an ARGUMENT and not a"
        " measurement: the fused column is achievable at every legal k"
        " because MAX_LEAVES caps P at 1024 and 1024 float32 partials is"
        " 4 KB of threadgroup memory."
    )


# ===========================================================================
# 13.6.3: END TO END OVER THE SHAPE TABLE
# ===========================================================================


def _capped(m: Int, n: Int, k: Int, budget: Int) -> Tuple[Int, Int]:
    """Reduce `m` and `n` to fit the host budget. **`k` IS NEVER TOUCHED.**

    `k` is the axis the leaf rule and the fold topology read; `m` and `n` are
    the axes contract 6.1 forbids them to read. So capping `m` and `n`
    preserves the leaf size, the leaf count, the ragged tail and the tree
    exactly, and capping `k` would destroy all four. Same rule and same
    reasoning as `bench/gemm_card_main.mojo::_capped`; transcribed rather than
    imported because that file is another lane's and carries a `main`.
    """
    var mm = m
    var nn = n
    while mm * nn * k > budget and (mm > 1 or nn > 1):
        if nn >= mm and nn > 1:
            nn = (nn + 1) // 2
        elif mm > 1:
            mm = (mm + 1) // 2
        else:
            break
    return Tuple(mm, nn)


def measure_end_to_end(budget: Int) raises:
    """13.6.3 over `bench/gemm_shapes.mojo`, host arms only.

    13.6.3 asks for three arms -- IDENTICAL, FAST, and the shipped pinned
    kernel -- over the charter's Phase 4 shape list. **Two of the three are
    device arms and are not available.** What runs is:

        oracle   `gemm_oracle`, the NORMATIVE v1 answer: contract leaf size,
                 balanced fold.
        serial   `gemm_oracle_serial`, the DIAGNOSTIC whole-`k` chain, which
                 is also what `core/gemm.mojo`'s two shipped pinned kernels
                 compute today. So `oracle / serial` is the closest thing a
                 host can offer to "what does the v1 partition cost over the
                 arithmetic the repository ships", at the same seams and in
                 the same spelling.

    Every row of the table runs. Nothing is dropped for being slow, being
    unflattering or being uninteresting -- a benchmark that chooses its rows
    has stopped measuring the workload and started measuring the choice. Rows
    that cannot fit the budget even at `m = n = 1` are REPORTED and counted,
    never silently skipped, because a missing row looks exactly like an
    agreeing one.

    The arms ALTERNATE CALL BY CALL, not block by block: both live in this
    binary, so the thermal drift the shell has to work around at the mode
    level is defeated outright at the arm level.
    """
    print()
    print("-- 13.6.3  end to end over bench/gemm_shapes.mojo --")
    print(
        "   HOST ARMS ONLY: 'oracle' is the normative v1 answer, 'serial' is"
        " the diagnostic whole-k chain. The FAST vendor arm and the shipped"
        " pinned kernel are DEVICE arms and are not run here."
    )
    print(
        "   m and n are CAPPED to a",
        budget,
        "MAC budget; k is EXACT, so L, P, the ragged tail and the tree are"
        " the full shape's. Per-cell price is measured; full-shape host cost"
        " is DERIVED by multiplication (contract 0.3: cells are independent)."
    )

    print(
        "   " + _padr("name", 32) + " op" + _padl("m", 9) + _padl("n", 10)
        + _padl("k", 10) + _padl("L", 9) + _padl("P", 8) + _padl("tail", 8)
        + _padl("oracle ms", 11) + _padl("serial ms", 11)
        + _padl("o/s", 9) + _padl("DERIVED ms", 14)
    )
    var ran = 0
    var skipped = 0
    var total_sink = UInt32(0)
    for i in range(GEMM_SHAPE_COUNT):
        var m0 = gemm_shape_m(i)
        var n0 = gemm_shape_n(i)
        var k = gemm_shape_k(i)
        var op = _tbl_op(i)
        var name = gemm_shape_name(i)

        var mn = _capped(m0, n0, k, budget)
        var m = mn[0]
        var n = mn[1]

        if m * n * k > budget:
            print(
                "   SKIP",
                name,
                "-- k =",
                k,
                "alone exceeds the host budget of",
                budget,
                "MACs and k may not be capped; needs the device arm.",
                "Raise MOJOLEARN_GEMM_PRICE_MAC_BUDGET to include it.",
            )
            skipped += 1
            continue

        var a = _fill(_a_elems(op, m, n, k), 11 + i)
        var b = _fill(_b_elems(op, m, n, k), 22 + i)

        # Untimed warm-up on BOTH arms before either clock starts.
        var wa = gemm_oracle(a, b, op, m, n, k)
        var wb = gemm_oracle_serial(a, b, op, m, n, k)
        var sink = _bits(wa[0]) ^ _bits(wb[0])

        var ns_oracle = 0
        var ns_serial = 0
        for _ in range(REPEATS):
            var t0 = perf_counter_ns()
            var c0 = gemm_oracle(a, b, op, m, n, k)
            ns_oracle += perf_counter_ns() - t0
            sink ^= _bits(c0[0])

            var t1 = perf_counter_ns()
            var c1 = gemm_oracle_serial(a, b, op, m, n, k)
            ns_serial += perf_counter_ns() - t1
            sink ^= _bits(c1[0])

        var ms_oracle = Float64(ns_oracle) / 1.0e6 / Float64(REPEATS)
        var ms_serial = Float64(ns_serial) / 1.0e6 / Float64(REPEATS)
        _report(String("m3.") + name + ".oracle", ms_oracle)
        _report(String("m3.") + name + ".serial", ms_serial)
        # The sink leaves the timed region as a printed value so neither arm
        # can be optimized away; it is accumulated across shapes and printed
        # once at the end rather than cluttering every row.
        total_sink ^= sink

        var L = contract_leaf_size(k)
        var p = leaf_count(k, L)
        var scale = Float64(m0) * Float64(n0) / (Float64(m) * Float64(n))
        print(
            "   " + _padr(name, 32) + " " + op_name(op)
            + _padl(String(m) + ("<" + String(m0) if m != m0 else ""), 11)
            + _padl(String(n) + ("<" + String(n0) if n != n0 else ""), 13)
            + _padl(String(k), 10)
            + _padl("L=" + String(L), 9)
            + _padl("P=" + String(p), 8)
            + _padl("t=" + String(last_leaf_len(k, L)), 8)
            + _padl(_fixed(ms_oracle, 3), 11)
            + _padl(_fixed(ms_serial, 3), 11)
            + _padl(_fixed(_ratio(ms_oracle, ms_serial), 3) + "x", 9)
            + _padl(_fixed(ms_oracle * scale, 1), 14)
        )
        ran += 1

    print("   ran", ran, "shapes,", skipped, "over budget; sink",
          total_sink)
    print(
        "   The DERIVED column is arithmetic on the measured per-cell price,"
        " NOT a measurement, and it is a HOST cost. It is printed because a"
        " reader comparing 'oracle 3 ms' across rows whose m n differ by a"
        " factor of 10^5 would otherwise compare nothing."
    )


# ===========================================================================
# 13.6.4: THE FTZ COST, SEPARATELY FROM THE FMA COST, IN THE LEAF LOOP
# ===========================================================================


def measure_leaf_loop_seams() raises:
    """Price the leaf loop's two pins SEPARATELY, which is 13.6.4.

    13.6.4: *"A benchmark that prices the fold and not the flush has priced
    the wrong thing."*

    Section 5 flags 5c -- one compare and one select per `k` step -- as the
    largest single cost item in the profile, and 13.3 puts it at 128 times the
    entire fold. So the leaf loop gets its own measurement, and the two pins
    inside it get separated rather than priced together.

    FOUR ARMS OVER THE SAME `k`-LENGTH OPERANDS, alternating call by call:

        m4.leaf.plain    `acc = a*b + acc`, unpinned, whatever the backend
                         chooses. The baseline.
        m4.leaf.fma      `acc = fma(a, b, acc)`, one rounding, always.
        m4.leaf.contract the REAL leaf loop:
                         `ftz(identical_mul_add(ftz(a), ftz(b), acc))`,
                         character for character `oracle_leaf_partial`'s.
        m4.leaf.denorm   the same real leaf loop over SUBNORMAL operands, so
                         the flush branch is TAKEN rather than merely tested.

    **THE SUBTRACTIONS, and why there is no second spelling of `ftz` anywhere
    in this file.** Under IDENTICAL, `identical_mul_add` IS `fma`, so

        fma cost  =  m4.leaf.fma      -  m4.leaf.plain     (both modes)
        ftz cost  =  m4.leaf.contract -  m4.leaf.fma       (IDENTICAL only)

    and under FAST `identical_mul_add` is `a*b+c` and `ftz` compiles away
    entirely, so `m4.leaf.contract` should collapse onto `m4.leaf.plain`. THAT
    COLLAPSE IS A WITNESS: if the FAST build's contract arm does not sit on its
    plain arm, either the mode did not reach this file or something in the
    chain is not compiling away, and the IDENTICAL numbers are measuring
    something other than the pins. Reading it costs nothing and it is the only
    check available that the FAST arm is genuinely unpinned.

    A HOST CAVEAT THAT MUST TRAVEL WITH THE RATIO. On a CPU the plain and fma
    loops can vectorize and the `ftz` branch may prevent it, so the host ratio
    contains a vectorization difference on top of the compare-and-select
    difference. A GPU pays a predicated select with no such cliff. **The host
    number is therefore an upper bound on the flush's device cost, not an
    estimate of it**, and 13.6.4 is properly settled by the device arm.
    """
    print()
    print("-- 13.6.4  the leaf loop's seams: ftz priced apart from fma --")
    print(
        "   k =",
        LEAF_K,
        "x",
        LEAF_PASSES,
        "passes =",
        LEAF_K * LEAF_PASSES,
        "multiply-accumulates per arm per repeat",
    )

    var a = _fill(LEAF_K, 41)
    var b = _fill(LEAF_K, 42)
    var da = _fill_subnormal(LEAF_K, 43)
    var db = _fill_subnormal(LEAF_K, 44)

    var sink = Float32(0.0)

    # Untimed warm-up on all four arms: page faults on 4 MB of operands are a
    # price nobody runs twice, and whichever arm went first would pay them.
    var w = Float32(0.0)
    for p in range(LEAF_K):
        w = a[p] * b[p] + w
    sink += w
    w = Float32(0.0)
    for p in range(LEAF_K):
        w = fma(a[p], b[p], w)
    sink += w
    w = Float32(0.0)
    for p in range(LEAF_K):
        w = ftz(identical_mul_add(ftz(a[p]), ftz(b[p]), w))
    sink += w
    w = Float32(0.0)
    for p in range(LEAF_K):
        w = ftz(identical_mul_add(ftz(da[p]), ftz(db[p]), w))
    sink += w

    var ns_plain = 0
    var ns_fma = 0
    var ns_contract = 0
    var ns_denorm = 0
    for _ in range(REPEATS):
        var t0 = perf_counter_ns()
        for _ in range(LEAF_PASSES):
            var acc0 = Float32(0.0)
            for p in range(LEAF_K):
                acc0 = a[p] * b[p] + acc0
            sink += acc0
        ns_plain += perf_counter_ns() - t0

        var t1 = perf_counter_ns()
        for _ in range(LEAF_PASSES):
            var acc1 = Float32(0.0)
            for p in range(LEAF_K):
                acc1 = fma(a[p], b[p], acc1)
            sink += acc1
        ns_fma += perf_counter_ns() - t1

        var t2 = perf_counter_ns()
        for _ in range(LEAF_PASSES):
            var acc2 = Float32(0.0)
            for p in range(LEAF_K):
                acc2 = ftz(identical_mul_add(ftz(a[p]), ftz(b[p]), acc2))
            sink += acc2
        ns_contract += perf_counter_ns() - t2

        var t3 = perf_counter_ns()
        for _ in range(LEAF_PASSES):
            var acc3 = Float32(0.0)
            for p in range(LEAF_K):
                acc3 = ftz(identical_mul_add(ftz(da[p]), ftz(db[p]), acc3))
            sink += acc3
        ns_denorm += perf_counter_ns() - t3

    _report(String("m4.leaf.plain"),
            Float64(ns_plain) / 1.0e6 / Float64(REPEATS))
    _report(String("m4.leaf.fma"), Float64(ns_fma) / 1.0e6 / Float64(REPEATS))
    _report(String("m4.leaf.contract"),
            Float64(ns_contract) / 1.0e6 / Float64(REPEATS))
    _report(String("m4.leaf.denorm"),
            Float64(ns_denorm) / 1.0e6 / Float64(REPEATS))
    print("   sink", _bits(sink), "(printed so no arm is optimized away)")
    print(
        "   fma cost = fma - plain. ftz cost = contract - fma, and ONLY in"
        " the IDENTICAL binary; under FAST the contract arm should collapse"
        " onto the plain arm and that collapse is the mode witness."
    )


# ===========================================================================
# 13.6.5: WORKSPACE BYTES AND THE TILING THRESHOLD
# ===========================================================================


def check_workspace_formula_matches_contract_13_5() raises:
    """The workspace arithmetic reproduces contract 13.5's three stated rows.

    13.5 states, in the contract, three `m * n * P` figures:

        m = n = 1024,  k = 4096         P = 32       128 MB
        m = n = 4096,  k = 4096         P = 32         2 GB
        m = n = 4096,  k = 4,000,000    P = 1024      64 GB

    **THIS IS A GATE AND NOT A DEMONSTRATION.** The harness's whole 13.6.5
    output rests on one multiplication and on `contract_leaf_size`, and if
    either drifts the table below would be confidently wrong in a way no
    reader could catch by eye -- a workspace table is exactly the kind of
    output people trust because it is arithmetic. So the formula is checked
    against numbers written down independently, in the contract, before any of
    it is printed. Failing here means either the transcribed leaf rule moved
    or 13.5's figures are wrong; both are findings and neither is ignorable.
    """
    var rows = 3
    for r in range(rows):
        var m = 1024
        var n = 1024
        var k = 4096
        var want_p = 32
        var want_mib = 128
        if r == 1:
            m = 4096
            n = 4096
            want_mib = 2048
        elif r == 2:
            m = 4096
            n = 4096
            k = 4_000_000
            want_p = 1024
            want_mib = 65536
        var p = leaf_count(k, contract_leaf_size(k))
        if p != want_p:
            raise Error(
                "workspace gate: contract 13.5 says P = "
                + String(want_p)
                + " at k = "
                + String(k)
                + " and the leaf rule gives "
                + String(p)
            )
        var mib = (m * n * p * 4) // (1024 * 1024)
        if mib != want_mib:
            raise Error(
                "workspace gate: contract 13.5 says "
                + String(want_mib)
                + " MiB at m = n = "
                + String(m)
                + ", k = "
                + String(k)
                + " and m*n*P*4 gives "
                + String(mib)
                + " MiB"
            )
    print(
        "check_workspace_formula_matches_contract_13_5 OK: all three of"
        " 13.5's stated figures (128 MiB, 2 GiB, 64 GiB) reproduced from"
        " contract_leaf_size and m*n*P*4"
    )


def report_workspace(budget_mib: Int) raises:
    """13.6.5, and it needs no device at all: `m * n * P * 4` is arithmetic.

    **A HARNESS THAT REPORTED ONLY MILLISECONDS WOULD LET THE 2 GB SCRATCH AT
    4096-squared BE DISCOVERED IN PRODUCTION.** That is why this table exists
    beside the times and not in a document: it is the design constraint 13.5
    calls bigger than the fold question, and it applies to either topology
    because it falls out of section 6's partition and not out of section 7.2's
    tree.

    THREE COLUMNS, and they answer three different questions.

      split-K level 0   `m * n * P * 4` bytes: the partials a split-K arm
                        materializes before any fold runs. This is 13.5's
                        number.
      fully staged      `m * n * fold_node_total(P) * 4`: what a naive
                        implementation that writes every tree level to global
                        memory would need -- close to 2x the above, and the
                        reason 13.6.2's staged arm is a memory question as
                        well as a launch question.
      tile edge         the largest square output tile whose split-K partials
                        fit the budget, `floor(sqrt(budget_bytes / (4 P)))`.
                        This is the tiling threshold 13.6.5 asks for, and it
                        is the OUTPUT-TILED escape from 13.5, which moves no
                        bits because cell arithmetic is independent of `m`,
                        `n` and of how many cells shared a launch.

    **THE ONE-BLOCK-OWNS-ITS-TILE ARM NEEDS NO GLOBAL SCRATCH AT ALL** and is
    13.5's second way out. It does not appear as a column because its answer
    is zero at every row.
    """
    print()
    print("-- 13.6.5  workspace bytes per shape, at the FULL m and n --")
    print(
        "   budget",
        budget_mib,
        "MiB (MOJOLEARN_GEMM_PRICE_WS_BUDGET_MB). Not a hardware constant:"
        " a threshold read from the device would make this table"
        " unreproducible across machines."
    )
    print(
        "   " + _padr("name", 32) + _padl("m", 7) + _padl("n", 8)
        + _padl("k", 10) + _padl("P", 6) + _padl("splitK MiB", 13)
        + _padl("staged MiB", 13) + _padl("tile edge", 11) + "  verdict"
    )
    var budget_bytes = budget_mib * 1024 * 1024
    var over = 0
    for i in range(GEMM_SHAPE_COUNT):
        var m = gemm_shape_m(i)
        var n = gemm_shape_n(i)
        var k = gemm_shape_k(i)
        var p = leaf_count(k, contract_leaf_size(k))
        var lvl0 = m * n * p * 4
        var staged = m * n * fold_node_total(p) * 4
        var per_cell = 4 * p
        var cells = budget_bytes // per_cell
        var edge = 1
        while (edge + 1) * (edge + 1) <= cells:
            edge += 1
        var verdict = String("fits")
        if lvl0 > budget_bytes:
            verdict = String("OVER BUDGET -- must tile the output")
            over += 1
        if p == 1:
            verdict = verdict + " (P=1: no fold; partials ARE the output)"
        print(
            "   " + _padr(gemm_shape_name(i), 32) + _padl(String(m), 7)
            + _padl(String(n), 8) + _padl(String(k), 10)
            + _padl(String(p), 6)
            + _padl(_fixed(Float64(lvl0) / 1048576.0, 2), 13)
            + _padl(_fixed(Float64(staged) / 1048576.0, 2), 13)
            + _padl(String(edge), 11) + "  " + verdict
        )
    print("  ", over, "of", GEMM_SHAPE_COUNT, "shapes exceed the budget.")
    print(
        "   Section 6.1 FORBIDS fixing this by making L depend on m or n --"
        " that is the batch-invariance clause. The two legal ways out are"
        " OUTPUT TILING and ONE-BLOCK-OWNS-ITS-K, both pure execution plan,"
        " both moving no bits. Choosing between them on m, n and the device"
        " is legal; choosing the NUMERICAL TREE on them is not."
    )


# ===========================================================================
# 13.6.6: WHICH RESOURCE LIMITS, PER SHAPE
# ===========================================================================


def check_fold_fraction_is_under_the_13_3_bound() raises:
    """13.3's central claim, asserted per shape rather than quoted.

    13.3 derives `fold adds / leaf FMAs = (P - 1)/k < 1/L <= 1/128 = 0.78%`
    and concludes the fold is under one percent of the arithmetic at every
    legal `k`. **A caption saying so is a claim; this is the check.** The
    earlier draft printed "under 13.3's 0.78% bound" beside a row reading
    0.7810%, which is under the real bound of `1/128 = 0.78125%` and over the
    rounded one -- a caption that rounds a bound into being wrong is exactly
    the kind of thing a gate catches and a reader does not.

    The bound compared against is `1/L` per shape, which is tighter than the
    profile-wide `1/128` wherever `L > K_LEAF_MIN` and is the quantity 13.3
    actually derives.
    """
    for i in range(GEMM_SHAPE_COUNT):
        var k = gemm_shape_k(i)
        var L = contract_leaf_size(k)
        var p = leaf_count(k, L)
        var frac = Float64(p - 1) / Float64(k)
        var bound = 1.0 / Float64(L)
        if frac >= bound:
            raise Error(
                "13.3 bound: '"
                + gemm_shape_name(i)
                + "' has (P-1)/k = "
                + String(frac)
                + " which is not below 1/L = "
                + String(bound)
                + ". 13.3's whole argument that the fold is under one percent"
                " of the arithmetic rests on this inequality."
            )
    print(
        "   check_fold_fraction_is_under_the_13_3_bound OK: every row's"
        " (P-1)/k is strictly below its own 1/L, and 1/L <= 1/128 ="
        " 0.78125%, which is 13.3's bound at full precision."
    )


def report_limiter_inputs() raises:
    """Print every INPUT to 13.6.6's classification, and no verdict.

    13.6.6 asks which of compute, bandwidth, staging or the fold limits EACH
    shape. **A host cannot answer that and this does not pretend to.**

    Naming a limiter needs device counters -- achieved bandwidth against peak,
    occupancy, and the kernel timeline that separates the leaf stage from the
    fold. None of those exist without the Phase 2b kernel. What a host can
    produce is every INPUT to the classification, exactly, so that when the
    device arm lands the only new thing needed is the measured time:

      flops          `2 m n k`
      operand bytes  `(|A| + |B| + m n) * 4`, the compulsory traffic
      intensity      flops / operand bytes, the roofline x-axis
      staging bytes  `m n P * 4`, which is 13.5's workspace and is TRAFFIC as
                     well as capacity -- written once and read once by the
                     fold, so it doubles as the staging-limited axis
      fold fraction  `(P - 1) / k`, the share of the arithmetic that is fold.
                     13.3 bounds it under `1/L <= 1/128 = 0.78%` at every
                     legal `k`, and this column is that bound made per-shape:
                     a shape whose fold fraction is small CANNOT be
                     fold-limited by arithmetic, only by dependency latency.

    The limiter column says UNMEASURED for every row, and it will keep saying
    it until a device produces a time.
    """
    print()
    print("-- 13.6.6  limiter inputs per shape (limiter itself UNMEASURED) --")
    print(
        "   " + _padr("name", 32) + _padl("MFLOP", 12)
        + _padl("operand MiB", 13) + _padl("flop/byte", 11)
        + _padl("staging MiB", 13) + _padl("fold %", 9) + "  limiter"
    )
    for i in range(GEMM_SHAPE_COUNT):
        var m = gemm_shape_m(i)
        var n = gemm_shape_n(i)
        var k = gemm_shape_k(i)
        var op = _tbl_op(i)
        var p = leaf_count(k, contract_leaf_size(k))
        var flops = 2.0 * Float64(m) * Float64(n) * Float64(k)
        var obytes = 4.0 * (
            Float64(_a_elems(op, m, n, k))
            + Float64(_b_elems(op, m, n, k))
            + Float64(m) * Float64(n)
        )
        var staging = 4.0 * Float64(m) * Float64(n) * Float64(p)
        print(
            "   " + _padr(gemm_shape_name(i), 32)
            + _padl(_fixed(flops / 1.0e6, 1), 12)
            + _padl(_fixed(obytes / 1048576.0, 2), 13)
            + _padl(_fixed(flops / obytes, 2), 11)
            + _padl(_fixed(staging / 1048576.0, 2), 13)
            + _padl(_fixed(100.0 * Float64(p - 1) / Float64(k), 4), 9)
            + "  UNMEASURED (needs the device arm)"
        )
    check_fold_fraction_is_under_the_13_3_bound()
    print(
        "   That bound is on the fold's ARITHMETIC share and says nothing"
        " about its EXPOSED LATENCY at small m n, which is the shape 13.3"
        " names as the one where the fold can still matter -- and four rows"
        " above have m n small enough to be exactly that shape."
    )


# ===========================================================================
# DRIVER
# ===========================================================================


def _wants(only: String, tag: String) -> Bool:
    if only == "" or only == "all":
        return True
    return only.find(tag) >= 0


def main() raises:
    var arm = String(getenv("MOJOLEARN_GEMM_PRICE_ARM"))
    if arm == "":
        arm = String("host")

    print("== bench/gemm_price_main.mojo [" + _mode() + "] ==")
    print("profile mojolearn.identical.gemm.fp32.v1")
    print("arm", arm)

    if arm == "device":
        raise Error(
            "bench/gemm_price_main: the 'device' arm is Phase 2b's kernel and"
            " it has not landed. THIS DRIVER IS DELIBERATELY WRITTEN TO"
            " ACCEPT IT WITHOUT A REWRITE. What to wire when 2b lands, and"
            " nothing else in this file changes:\n"
            "  (1) import the kernel entry point and open a DeviceContext in"
            " main(), the way bench/linalg_price_main.mojo does;\n"
            "  (2) in measure_end_to_end, add a third timed call beside the"
            " oracle and serial calls, inside the SAME REPEATS loop so it"
            " alternates with them call by call, and emit"
            " 'm3.<shape>.device';\n"
            "  (3) add the FAST vendor arm and the shipped pinned kernel as"
            " two more calls in that same loop -- 13.6.3 names three arms and"
            " only then are all three present;\n"
            "  (4) in measure_fold_in_isolation, run the P sweep as device"
            " launches; ONLY THEN does 13.6.1 bear on contract 13.4, which is"
            " a launch argument;\n"
            "  (5) replace report_staging_structure's table with two timed"
            " arms, staged (D+1 launches) and fused (2 launches), over the"
            " same P sweep -- that is 13.6.2 and it cannot exist on a host;\n"
            "  (6) fill report_limiter_inputs' UNMEASURED column from the"
            " measured time plus the flops and bytes it already computes.\n"
            "Until then use arm 'host', which prices the NORMATIVE oracle"
            " against the diagnostic serial reference and answers 13.6.4 and"
            " 13.6.5 outright."
        )
    if arm != "host":
        raise Error(
            "bench/gemm_price_main: unknown MOJOLEARN_GEMM_PRICE_ARM '"
            + arm
            + "'. Known: host (default), device."
        )

    var budget = DEFAULT_MAC_BUDGET
    var bs = String(getenv("MOJOLEARN_GEMM_PRICE_MAC_BUDGET"))
    if bs != "":
        budget = Int(atol(bs))
    var kmax = 0
    for i in range(GEMM_SHAPE_COUNT):
        if gemm_shape_k(i) > kmax:
            kmax = gemm_shape_k(i)
    if budget < kmax:
        raise Error(
            "bench/gemm_price_main: the MAC budget "
            + String(budget)
            + " is below the shape table's largest k, "
            + String(kmax)
            + ". m and n cap to 1 and k may NEVER be capped, so this budget"
            " would silently drop the table's largest-k row -- which is its"
            " only real P = MAX_LEAVES row and the most interesting fold in"
            " the table. Dropping a benchmark shape because it is expensive"
            " is building to the dataset. Raise the budget or accept the"
            " runtime."
        )

    var ws_mb = DEFAULT_WS_BUDGET_MB
    var ws = String(getenv("MOJOLEARN_GEMM_PRICE_WS_BUDGET_MB"))
    if ws != "":
        ws_mb = Int(atol(ws))

    var only = String(getenv("MOJOLEARN_GEMM_PRICE_ONLY"))

    print(
        "NOT TRUSTWORTHY BESIDE A GPU LEG (contract 13.5, assumption A5).",
        "These arms are host-only, so they contend for CPU rather than for",
        "the GPU -- which makes them LESS fragile than a device timing, not",
        "immune. tools/gemm_price.sh checks and says so.",
    )
    print()

    # Gates first. Every one of them guards a number printed below, and a
    # harness that measured before it checked would publish the wrong number
    # and then check.
    check_op_encodings_are_not_interchangeable()
    check_serial_fold_probe_is_the_other_topology()
    check_workspace_formula_matches_contract_13_5()

    if _wants(only, "m1"):
        measure_fold_in_isolation()
    if _wants(only, "m2"):
        report_staging_structure()
    if _wants(only, "m3"):
        measure_end_to_end(budget)
    if _wants(only, "m4"):
        measure_leaf_loop_seams()
    if _wants(only, "m5"):
        report_workspace(ws_mb)
    if _wants(only, "m6"):
        report_limiter_inputs()

    print()
    print("== done [" + _mode() + "] ==")
