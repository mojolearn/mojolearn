"""The `gemm.fp32.v1` identity card: one stage per shape, for the vendor diff.

Phase 3's exit criterion is *"Apple/NVIDIA/AMD hashes agree"*. That needs a
CARD -- a file of per-stage hashes that `tools/identity_trace_diff.py` can
align and diff -- and it has to exist before there is anything to diff. This
is that emitter.

WHY IT RUNS AGAINST THE ORACLE TODAY AND THE KERNEL TOMORROW
------------------------------------------------------------
The device kernel is Phase 2b and is not landed. The oracle is, it is
NORMATIVE, and it is the value the kernel must reproduce -- so a card emitted
from the oracle is the REFERENCE CARD the kernel's card gets diffed against,
and it is worth having first rather than second. `MOJOLEARN_GEMM_CARD_ARM`
selects:

    oracle   `gemm_oracle`, the normative v1 answer          (available now)
    serial   `gemm_oracle_serial`, the DIAGNOSTIC reference  (available now)
    device   the Phase 2b kernel                             (when it lands)

The `serial` arm is not decoration. Emitting both and diffing them is the
one-command demonstration that the profile's fold topology is load-bearing:
they agree only where `P == 1` and diverge everywhere else. A reader who
doubts that the balanced tree matters can produce the divergence in two
commands rather than reading section 7.

ONE STAGE PER SHAPE, TAGGED BY NAME AND NOT BY INDEX. `bench/gemm_shapes.mojo`
is the table, and its rows are tagged by their NAME (`llama8b.mlp_down.t8`),
never by position, so inserting a shape does not renumber every downstream
card and silently turn a diff into noise. That is the same reason
`core/identity_trace.mojo` tags by algorithm position rather than sequence.

THE FIXTURE IS BIT-ASSEMBLED, NO HOST FLOAT ARITHMETIC ANYWHERE.
`bench/linalg_trace_main.mojo` learned this the expensive way and states it:
a host `target += v * w` chain is IDENTITY_PATHS row 18's contraction
decision, so two machines running the same driver would upload DIFFERENT
inputs and the card would diff their fixtures rather than their GEMMs. Every
value here is assembled from splitmix64 bits and bitcast. The card's FIRST
stage per shape is a hash of the raw input bytes; compare that before
comparing any product.

THE SHAPES ARE CAPPED, AND THE CAP IS PRINTED. A full `llama8b.lm_head.t512`
is 512 x 128256 x 4096, which is 269 GFLOP on a HOST oracle -- days. So the
card runs each shape's ARITHMETIC STRUCTURE at a reduced `m` and `n` while
keeping `k` EXACT, because `k` is the axis the contract's leaf rule and fold
topology depend on and `m`/`n` are the axes it may not read. Reducing `m` and
`n` therefore preserves everything the card is about and costs nothing the
card measures. `MOJOLEARN_GEMM_CARD_FULL=1` removes the cap for the device
arm, where it is affordable.
"""

from std.os import getenv

from bench.gemm_shapes import (
    GEMM_SHAPE_COUNT,
    OP_NN,
    OP_NT,
    OP_TN,
    contract_leaf_size,
    gemm_shape_k,
    gemm_shape_m,
    gemm_shape_n,
    gemm_shape_name,
    gemm_shape_op,
    leaf_count,
)
from core.identity_trace import IdentityTrace
from mojo_only.kernel_matrix import TARGET_COLUMN, column_name
from gemm.mojo_only.gemm_oracle import gemm_oracle, gemm_oracle_serial
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


def _mode_name() -> String:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


#: The host oracle's budget per shape, in multiply-accumulates. The oracle is
#: a scalar host loop, so this is wall-clock, not memory. 16M keeps the whole
#: card under a minute; the DEVICE arm does not need it.
comptime HOST_MAC_BUDGET = 16_000_000


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

    NO DECIMAL CONSTANT AND NO HOST FLOAT CHAIN. The numerator is below 2^21
    so it is exact in a Float32 mantissa and the divisor is a power of two,
    so the division is exact. A generator that multiplied by `1.0e-6` would
    put a rounding of a decimal constant upstream of the thing being carded.
    """
    var num = Int(_mix(i, salt) % 2097151) - 1048575
    return Float32(num) / Float32(1048576.0)


def _capped(m: Int, n: Int, k: Int) -> Tuple[Int, Int]:
    """Reduce `m` and `n` to fit the host budget. `k` is NEVER touched.

    `k` is the axis the leaf rule and the fold topology read; `m` and `n` are
    the axes contract 6.1 forbids them to read. So capping m and n preserves
    every property this card exists to record, and capping k would destroy
    all of them. If that ever stops being true the contract has changed.
    """
    var mm = m
    var nn = n
    while mm * nn * k > HOST_MAC_BUDGET and (mm > 1 or nn > 1):
        if nn >= mm and nn > 1:
            nn = (nn + 1) // 2
        elif mm > 1:
            mm = (mm + 1) // 2
        else:
            break
    return Tuple(mm, nn)


def _fill(n: Int, salt: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(_exact(i, salt))
    return out^


def main() raises:
    var arm = String(getenv("MOJOLEARN_GEMM_CARD_ARM"))
    if arm == "":
        arm = String("oracle")
    var full = String(getenv("MOJOLEARN_GEMM_CARD_FULL")) == "1"

    print("== bench/gemm_card_main.mojo [" + _mode_name() + "] ==")
    print("profile mojolearn.identical.gemm.fp32.v1")
    print("arm", arm)
    # THE COLUMN WITNESS. `tools/gemm_column_invariance.sh` compiles this
    # file against three vendor columns with `-D MOJOLEARN_COLUMN_<COL>=1`
    # and greps this line back out. Without it the sweep can PASS THE DEFINE
    # AND NEVER KNOW WHETHER THE COMPILER SAW IT -- three matching cards
    # would look identical whether the define landed or was silently
    # dropped, which is agreement by coincidence dressed as a gate.
    #
    # Invisible today, because the oracle arm is host-only scalar code that
    # reads no column at all. It becomes load-bearing the moment the device
    # arm lands, which is exactly when the gate starts meaning something --
    # so it goes in now rather than being remembered then.
    print("column", column_name(TARGET_COLUMN))

    if arm == "device":
        raise Error(
            "bench/gemm_card_main: the 'device' arm is Phase 2b's kernel and"
            " it has not landed. This driver is deliberately written to"
            " accept it without further edits -- wire the call in _run_shape"
            " and nothing else here changes. Until then use arm 'oracle'"
            " (the normative v1 answer) or 'serial' (the diagnostic"
            " reference)."
        )
    if arm != "oracle" and arm != "serial":
        raise Error(
            "bench/gemm_card_main: unknown MOJOLEARN_GEMM_CARD_ARM '"
            + arm
            + "'. Known: oracle, serial, device."
        )

    var path = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if path == "":
        raise Error(
            "bench/gemm_card_main: MOJOLEARN_IDENTITY_TRACE is unset, so"
            " this run would emit NO CARD and still exit 0 -- and a driver"
            " that checks nothing would record a missing file as agreement."
        )

    var t = IdentityTrace()
    var ran = 0
    var skipped = 0

    for i in range(GEMM_SHAPE_COUNT):
        var m0 = gemm_shape_m(i)
        var n0 = gemm_shape_n(i)
        var k = gemm_shape_k(i)
        var op = gemm_shape_op(i)
        var name = gemm_shape_name(i)

        var mn = _capped(m0, n0, k)
        var m = mn[0]
        var n = mn[1]
        if full:
            m = m0
            n = n0

        if m * n * k > HOST_MAC_BUDGET and not full:
            # Cannot be reduced far enough without touching k. Reported, not
            # silently dropped: a card that quietly omits a shape looks
            # exactly like a card whose shape agreed.
            print(
                "  SKIP",
                name,
                "-- k =",
                k,
                "alone exceeds the host budget; needs the device arm",
            )
            skipped += 1
            continue

        var a: List[Float32]
        var b: List[Float32]
        if op == OP_TN:
            a = _fill(k * m, 11 + i)
            b = _fill(k * n, 22 + i)
        elif op == OP_NN:
            a = _fill(m * k, 11 + i)
            b = _fill(k * n, 22 + i)
        else:
            a = _fill(m * k, 11 + i)
            b = _fill(n * k, 22 + i)

        # INPUTS FIRST, ALWAYS. A product diff against different inputs
        # measures nothing, and E1_RUNBOOK's ladder reads inputs before
        # products for the same reason.
        t.record_list_f32(name + ".in.a", a)
        t.record_list_f32(name + ".in.b", b)

        var c: List[Float32]
        if arm == "oracle":
            c = gemm_oracle(a, b, op, m, n, k)
        else:
            c = gemm_oracle_serial(a, b, op, m, n, k)

        t.record_list_f32(name + ".out", c)
        ran += 1

        var p = leaf_count(k, contract_leaf_size(k))
        print(
            "  ",
            name,
            "m=" + String(m) + (" (capped from " + String(m0) + ")" if m
                                != m0 else ""),
            "n=" + String(n) + (" (capped from " + String(n0) + ")" if n
                                != n0 else ""),
            "k=" + String(k),
            "P=" + String(p),
        )

    print()
    print("card:", path)
    print("stages:", ran * 3, "over", ran, "shapes;", skipped, "skipped")
    print(
        "Diff two arms to see the fold topology matter:\n"
        "  MOJOLEARN_GEMM_CARD_ARM=oracle MOJOLEARN_IDENTITY_TRACE=/tmp/o.card"
        " ...\n"
        "  MOJOLEARN_GEMM_CARD_ARM=serial MOJOLEARN_IDENTITY_TRACE=/tmp/s.card"
        " ...\n"
        "  python3 tools/identity_trace_diff.py /tmp/o.card /tmp/s.card\n"
        "They agree only where P == 1."
    )
