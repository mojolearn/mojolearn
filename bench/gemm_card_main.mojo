# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The `gemm.fp32.v1` identity card: one stage per shape, for the vendor diff.

Phase 3's exit criterion is *"Apple/NVIDIA/AMD hashes agree"*. That needs a
CARD -- a file of per-stage hashes that `tools/identity_trace_diff.py` can
align and diff -- and it has to exist before there is anything to diff. This
is that emitter.

THE THREE ARMS, AND WHY THE ORACLE ONE CAME FIRST
--------------------------------------------------
The oracle is NORMATIVE, and it is the value the kernel must reproduce -- so
a card emitted from the oracle is the REFERENCE CARD the kernel's card gets
diffed against, and it was worth having before the kernel existed.
`MOJOLEARN_GEMM_CARD_ARM` selects:

    oracle   `gemm_oracle`, the normative v1 answer
    serial   `gemm_oracle_serial`, the DIAGNOSTIC reference
    device   `gemm/original/gemm_identical.mojo`, the Phase 2b kernel

The `serial` arm is not decoration. Emitting both and diffing them is the
one-command demonstration that the profile's fold topology is load-bearing:
they agree only where `P == 1` and diverge everywhere else. A reader who
doubts that the balanced tree matters can produce the divergence in two
commands rather than reading section 7.

THE DEVICE ARM, DEVIATION 534
------------------------------
Phase 2b landed, so the `device` arm is now a real card from the real
kernel: `identical_gemm_into` on the plan `choose_gemm_plan` picks, through a
`DeviceContext`, with the product recorded straight off the DEVICE BUFFER by
`IdentityTrace.record_device`. Recording the device buffer rather than a host
`List` is what makes `MOJOLEARN_IDENTITY_TRACE_DUMP` write a `.bin` sidecar
for the stage, and a sidecar is the difference between the differ saying
"differs" and the differ saying WHICH CELL, by how many ULPs, and whether the
class is denormal-vs-zero. On a cross-vendor divergence that is the whole
investigation.

THIS DRIVER HAS NO FAST ARM, AND ITS `[FAST]` CARD IS NOT A CONTROL
--------------------------------------------------------------------
DEVIATION 1091, 2026-08-25, found on the leg-13 AMD column. The banner prints
`_mode_name()`, which reads the compiled `GLOBAL_NUMERIC_MODE` honestly, so a
run under `MOJOLEARN_NUMERIC_MODE=fast` is labelled `[FAST]`. The KERNEL is
not selected that way. `_device_product` calls `identical_gemm_into`
unconditionally: it imports the identical kernel directly and never reaches
`core/gemm.mojo`'s dispatcher, which is the thing that would pick MAX
`linalg.matmul` under FAST. So the FAST card is the IDENTICAL card with a
different banner.

MEASURED, not deduced: on BOTH Apple and the MI325X, `gemm.fast.card` and
`gemm.identical.card` are byte-identical after stripping comments, while the
other six phase-8 lanes' FAST cards diverge between those same two machines.

Two things follow and neither is optional to state. **The IDENTICAL claim is
untouched** -- those cards are real, they came off the device, and they match
across three vendors. **The gemm FAST card is not evidence of anything** and
must never be read as the control arm for this lane; a leg that shows "gemm
FAST identical across vendors" is showing that this driver ignored the mode.
The gemm row of `tools/lanes_price.sh` inherits the same defect: it times
identical against identical and its ratio is not a price.

The oracle arm and the device arm are the SAME DRIVER, so the fixtures, the
tags, the ordering and the record format are shared by construction rather
than by two authors agreeing. Their cards are byte-identical whenever they
run the same shapes -- see the cap section below, which is the only thing
that can make them not.

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

THE SHAPES ARE CAPPED, THE CAP DEPENDS ON THE ARM, AND THE CAP IS PRINTED
-------------------------------------------------------------------------
A full `llama8b.lm_head.t512` is 512 x 128256 x 4096, which is 269 GFLOP on a
HOST oracle -- days. So the card runs each shape's ARITHMETIC STRUCTURE at a
reduced `m` and `n` while keeping `k` EXACT, because `k` is the axis the
contract's leaf rule and fold topology depend on and `m`/`n` are the axes it
may not read. Reducing `m` and `n` therefore preserves everything the card is
about and costs nothing the card measures.

There are TWO budgets and they are not the same kind of quantity:

    HOST_MAC_BUDGET       multiply-accumulates. The oracle is a scalar host
                          loop, so the host arms' limit is WALL CLOCK.
    DEVICE_FLOAT_BUDGET   float32 elements across A, B, C and the workspace.
                          The device arm's limit is MEMORY; it does not care
                          how many MACs it does.

The device arm runs the FULL shape wherever `DEVICE_FLOAT_BUDGET` allows, and
on this table that includes the largest `k` in it (`gram.32x32x1M`, at its
shipped 32 x 32 x 1,000,000). Both budgets may reduce `m` and `n` and NEITHER
may touch `k`; see `_capped` and `_capped_device`, which say why in the same
words because it is the same reason.

    MOJOLEARN_GEMM_CARD_FULL=1      remove the cap entirely.
    MOJOLEARN_GEMM_CARD_HOST_CAP=1  force the HOST cap onto ANY arm.

    -D MOJOLEARN_GEMM_CARD_TINY_DEVICE_BUDGET=1
                                    the REACH HANDLE for the SKIP branch.
                                    See `SAB_TINY_DEVICE_BUDGET`: neither
                                    budget's skip is reachable against
                                    today's table, and a branch nobody has
                                    made fire is not a branch anybody should
                                    trust.

`HOST_CAP` is how the agreement proof is run. Two cards are only diffable
when their stage lists match, and the stage lists match only when the two
arms ran the same `m` and `n`:

    tools/gemm_card.sh oracle /tmp/o.card
    MOJOLEARN_GEMM_CARD_HOST_CAP=1 tools/gemm_card.sh device /tmp/d.card
    python3 tools/identity_trace_diff.py /tmp/o.card /tmp/d.card

THE EXECUTION PLAN IS PRINTED TO THE LOG AND NEVER RECORDED IN THE CARD.
`choose_gemm_plan` reads `m` and `n` and is allowed to (contract 6.1); the
plan is EXECUTION, not numerics. If a plan name ever reached a hash, the card
would stop testing the property it exists to test -- it would report a
divergence whenever two vendors dispatched differently, which is precisely
the thing the profile permits, and it would report agreement as "the same
plan ran" rather than "the same bits came out". The log carries it because a
reader wants to know which plan produced a diverging stage; the card must
not.

WHAT THE CARD CAN SEE, MEASURED BY SABOTAGE, NOT ARGUED
--------------------------------------------------------
`gemm_identical.mojo` carries five deliberate `-D MOJOLEARN_GEMM_SABOTAGE_*`
defects. The device arm was built against each one and its card diffed
against the oracle card, under IDENTICAL and at the HOST cap so the stage
lists align. FOUR OF FIVE MOVE THE CARD, all at the first product stage
(`gram.32x32x1M.out`, oracle `d8e5aa6938cbf068`):

    FOLD_STRIDE         f424f94c03641502   DIVERGES
    NODE_ORDER          1d186ea64b15d736   DIVERGES
    FOLD_SERIAL         5d394e707906e455   DIVERGES
    LEAF_READS_LAUNCH   9eaa97fdc2ac960c   DIVERGES
    PAD_PLUS_ZERO       d8e5aa6938cbf068   **BYTE-IDENTICAL. NOT SEEN.**

**PAD_PLUS_ZERO IS A BLIND SPOT OF THIS CARD AND IT IS RECORDED RATHER THAN
QUIETLY LIVED WITH.** That sabotage pads an odd level width with `+0.0`
instead of carrying the odd node bit for bit. `x + (+0.0) == x` for every
finite float, every infinity and every NaN -- it differs on exactly ONE
value, `x = -0.0`, where `(-0) + (+0) = +0`. This card's fixture is `_exact`,
a spread of `int / 2^20` values, and no leaf partial it produces is ever
`-0.0`, so the defect is arithmetically invisible here. It is the same
failure IDENTITY_PATHS row 9 records and the same one
`gemm_device_check.mojo::_minus_zero_case` was written to fix after that
build passed 42 shapes with exit code 0.

The card CANNOT fix it the way that check did. `_minus_zero_case` bought its
coverage with a SYNTHETIC fixture whose every leaf partial is `-0.0`, and
`bench/gemm_shapes.mojo` is a record of what this library computes -- adding
a row to it so a gate goes green is `[[never build to datasets]]`, and that
file's own banner says a table containing shapes chosen to satisfy its own
check has stopped being a record. So the coverage lives where it belongs, in
`gemm/original/gemm_device_check.mojo`, and this file states the boundary:
**a green card says two runs AGREE, never that the kernel is RIGHT.** The
correctness gate is `gemm_device_check`; this is the cross-vendor diff
instrument, and it is blind to any defect the shipped shapes do not evaluate
differently.

NO PROVENANCE HEADER IS WRITTEN INTO THE CARD, and that is deliberate rather
than an omission. `IdentityTrace.header` exists and the differ skips comment
lines, but `tools/gemm_column_invariance.sh` compares whole card FILES with
`cmp -s`, so a header naming the arm, the column or the cap policy would make
that gate report DIVERGED for a reason that is not a bit of arithmetic.
Provenance lives in the run's LOG, which `tools/gemm_card.sh` keeps beside
the card.

`[[mojo-string-float-roundtrip]]`: every float printed here carries its hex
bits beside its decimal.
"""

from std.memory import bitcast
from std.os import getenv
from std.sys.compile import is_defined

from max.gpu.host import DeviceBuffer, DeviceContext

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
from original.kernel_matrix import TARGET_COLUMN, column_name
from gemm.original.gemm_identical import (
    choose_gemm_plan,
    gemm_plan_name,
    gemm_sabotage_name,
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.original.gemm_oracle import (
    OP_NN as ORACLE_OP_NN,
    OP_NT as ORACLE_OP_NT,
    OP_TN as ORACLE_OP_TN,
    gemm_oracle,
    gemm_oracle_serial,
)
from original.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29. This used to
    be a local two-way `IDENTICAL`-or-`FAST`, written when there were
    two tiers, and it answered "FAST" for a DETERMINISTIC build -- so
    a driver run under the middle tier printed the wrong arm onto
    every line it produced. A correctly-labelled measurement of the
    wrong arm is the failure this tree has been bitten by repeatedly,
    and forty-four copies of a mode label is how it happens.
    """
    return numeric_mode_name()


def _bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def _show(x: Float32) -> String:
    return String(x) + "/" + hex(_bits(x))


#: The host oracle's budget per shape, in multiply-accumulates. The oracle is
#: a scalar host loop, so this is wall-clock, not memory. 16M keeps the whole
#: card under a minute; the DEVICE arm does not need it.
comptime HOST_MAC_BUDGET = 16_000_000

#: DEVIATION 534. The DEVICE arm's budget per shape, in FLOAT32 ELEMENTS
#: summed over A, B, C and the kernel's workspace. 64Mi elements is 256 MB of
#: device buffers, and about 900 MB once the host staging buffers and the
#: host fixture lists are counted, which is what a run has to fit alongside
#: the other sessions sharing this checkout.
#:
#: **IT IS A FIXED CONSTANT AND IT IS NOT PROBED FROM THE DEVICE.** That is
#: the whole design decision and it is worth stating: a budget read from
#: `DeviceContext` would cap an Apple laptop's card at different `m` and `n`
#: than an MI300X's, the two cards would carry different stage counts, and
#: Phase 3's cross-vendor diff -- the only reason this file exists -- would
#: compare two different sets of shapes and call it a divergence. A card's
#: shapes must be a function of the TABLE and the PROFILE, never of the
#: silicon under it. A machine with more memory gets more shape by raising
#: this constant deliberately, which changes the card for everybody at once.
comptime DEVICE_FLOAT_BUDGET_SHIPPED = 64 * 1024 * 1024

#: THE SKIP BRANCH'S REACH HANDLE, and it is here because that branch is
#: otherwise unreachable and an unreachable branch is not a gate
#: (`[[reached-but-inert]]`).
#:
#: `-D MOJOLEARN_GEMM_CARD_TINY_DEVICE_BUDGET=1` compiles the device arm
#: against 1Mi floats instead of 64Mi. At that budget `gram.32x32x1M` cannot
#: fit even at `m = n = 1` -- its `k` alone needs about 2M floats of operand
#: -- so the shape SKIPS, with its reason and in the skip count, while the
#: other nineteen still run. That is the branch EXECUTING, on a real table
#: row, rather than being argued from a docstring.
#:
#: The HOST arm's skip is dead in the same way and cannot be woken the same
#: way: with `HOST_MAC_BUDGET = 16,000,000` and the table's largest `k` at
#: 1,000,000, `1 * 1 * k` never exceeds the budget, so no row of today's
#: table reaches it. **Both skip branches are unreachable against this
#: table**, and both are kept, because the alternative is a card that
#: silently drops a shape the day a bigger row is added -- and a dropped
#: shape reads exactly like a shape that agreed.
comptime SAB_TINY_DEVICE_BUDGET = is_defined[
    "MOJOLEARN_GEMM_CARD_TINY_DEVICE_BUDGET"
]()

comptime DEVICE_FLOAT_BUDGET = 1024 * 1024 if SAB_TINY_DEVICE_BUDGET else DEVICE_FLOAT_BUDGET_SHIPPED

#: The value written into `C` before every device launch. A cell still
#: holding it afterwards was NEVER WRITTEN, and hashing it into the card
#: would record a hash of poison -- which is a divergence that looks exactly
#: like an arithmetic one and is not. Same constant and same reasoning as
#: `gemm/original/gemm_device_check.mojo::POISON`.
comptime POISON = Float32(-987654.0)


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


def _oracle_op(tbl: Int) -> Int:
    """Translate the TABLE's orientation code into the CONTRACT's.

    **TWO FILES NUMBER THE SAME THREE WORDS DIFFERENTLY**, and this card
    shipped without noticing:

        bench/gemm_shapes.mojo    NT = 0, TN = 1, NN = 2
        gemm/original/gemm_oracle.mojo   NN = 0, NT = 1, TN = 2

    So a table `OP_TN` (1) reached the oracle as `OP_NT`, and a table
    `OP_NT` (0) reached it as `OP_NN`. **Every row of the reference card was
    a product the table does not describe.**

    IT NEVER TRAPPED AND THAT IS WHY IT SURVIVED. The fill branches below
    are written against the TABLE's codes and are correct, so every buffer
    was the right size; the mis-indexing stayed in bounds and returned a
    plausible float. A card of plausible floats is exactly as diffable as a
    card of correct ones, so Phase 3 would have compared vendors against a
    reference that was wrong in the same way on both sides and reported
    agreement.

    Found by the Phase 4 lane, which hit the same mismatch and wrote its own
    map first (`gemm/original/gemm_device_check.mojo::_tbl_op`). Two lanes
    independently needing this map is the argument for one of the two files
    changing its constants; until then the map is written where it is used
    and says so.

    **THE DEVICE KERNEL USES THE ORACLE'S NUMBERING, NOT A THIRD ONE**
    (DEVIATION 534). `gemm/original/gemm_identical.mojo:105-110` does not
    define `OP_NN`/`OP_NT`/`OP_TN` at all -- it IMPORTS them from
    `gemm_oracle.mojo`, and `identical_gemm`'s docstring says so in words:
    *"`op` is `OP_NN`, `OP_NT` or `OP_TN` from `gemm_oracle.mojo`"*. So this
    one map serves both arms and the device call uses it explicitly, at the
    call, rather than passing the table's code through and hoping. Verified
    the same way the bug above should have been: by reading the kernel's
    imports, not by assuming, and by running the two arms against each other
    (a wrong map here transposes an operand and no shape in the table is
    square in `m` and `n`, so the oracle comparison would go red on every
    row -- a mis-map is now a loud failure rather than a plausible float).
    """
    if tbl == OP_NT:
        return ORACLE_OP_NT
    if tbl == OP_TN:
        return ORACLE_OP_TN
    return ORACLE_OP_NN


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


def _device_floats(m: Int, n: Int, k: Int) -> Int:
    """Float32 elements one device launch of this shape needs.

    `A` is `m*k` and `B` is `n*k` in ALL THREE orientations -- `OP_TN` stores
    A as `k x m` and `OP_NT` stores B as `n x k`, which transposes the layout
    and not the element count -- so this is a function of `m`, `n` and `k`
    alone and needs no `op`. `C` is `m*n`. The workspace is whatever the plan
    `choose_gemm_plan` will actually pick costs, asked of the kernel's own
    helper rather than guessed: `gemm_identical.mojo`'s own harness lost a
    run to a workspace sized for the wrong plan, and a budget that guessed
    would mis-size the card's shapes the same way.
    """
    return (
        m * k + n * k + m * n + identical_gemm_workspace_max_floats(m, n, k)
    )


def _capped_device(m: Int, n: Int, k: Int) -> Tuple[Int, Int]:
    """Reduce `m` and `n` to fit `DEVICE_FLOAT_BUDGET`. `k` is NEVER touched.

    THE SAME RULE AS `_capped` AND FOR THE SAME REASON, restated rather than
    cross-referenced because a reader who changes one must change the other.
    `k` is the axis the leaf rule and the fold topology read; `m` and `n` are
    the axes contract 6.1 forbids them to read. Capping `m` and `n` preserves
    every property this card records -- the leaf boundaries, the tree levels,
    the carries -- and capping `k` would destroy all of them and leave a card
    that still looked diffable. If that ever stops being true the contract
    has changed.

    Halving rather than dividing exactly, and halving the LARGER axis first,
    so the reduced shape stays a recognisable relative of the shipped one and
    so the sequence is the same on every machine.
    """
    var mm = m
    var nn = n
    while _device_floats(mm, nn, k) > DEVICE_FLOAT_BUDGET and (
        mm > 1 or nn > 1
    ):
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


def _device_product(
    mut t: IdentityTrace,
    ctx: DeviceContext,
    tag: String,
    ha: List[Float32],
    hb: List[Float32],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
) raises:
    """Run one shape on the device kernel and record `C` OFF THE DEVICE
    BUFFER.

    `op` is already the CONTRACT's code -- `_oracle_op` translated it at the
    call site -- because `gemm_identical.mojo` imports its orientation
    constants from `gemm_oracle.mojo`.

    THE OUTPUT IS POISONED BEFORE THE LAUNCH AND THE POISON IS CHECKED
    BEFORE IT IS HASHED. A cell the kernel never wrote would be hashed as
    whatever the allocator handed back, which is a divergence that looks
    exactly like an arithmetic one and is not; contract section 8 also makes
    the store load-bearing at `k == 0`, where the required `+0.0` must be
    STORED rather than the store skipped. Costs one extra copy of `C` and
    buys a failure mode that names itself.

    `record_device` and not `record_list_f32`, so that
    `MOJOLEARN_IDENTITY_TRACE_DUMP` writes the stage's `.bin` sidecar and
    `tools/identity_trace_diff.py` can go per-cell on a cross-vendor
    divergence instead of reporting "differs". The bytes are the same either
    way, so an oracle card and a device card stay comparable.
    """
    var na = len(ha)
    var nb = len(hb)
    var mn = m * n
    # A zero-length operand is legal (`k == 0`, contract section 8) and a
    # zero-length device buffer is not, so the ALLOCATION is clamped while
    # the copy loops still run over the real element count.
    var na_buf = na
    if na_buf < 1:
        na_buf = 1
    var nb_buf = nb
    if nb_buf < 1:
        nb_buf = 1
    var nws = identical_gemm_workspace_max_floats(m, n, k)
    if nws < 1:
        nws = 1

    var da = ctx.enqueue_create_buffer[DType.float32](na_buf)
    var db = ctx.enqueue_create_buffer[DType.float32](nb_buf)
    var dc = ctx.enqueue_create_buffer[DType.float32](mn)
    var dw = ctx.enqueue_create_buffer[DType.float32](nws)
    var hA = ctx.enqueue_create_host_buffer[DType.float32](na_buf)
    var hB = ctx.enqueue_create_host_buffer[DType.float32](nb_buf)
    var hC = ctx.enqueue_create_host_buffer[DType.float32](mn)
    ctx.synchronize()

    for i in range(na):
        hA.unsafe_ptr().unsafe_store(i, ha[i])
    for i in range(nb):
        hB.unsafe_ptr().unsafe_store(i, hb[i])
    for i in range(mn):
        hC.unsafe_ptr().unsafe_store(i, POISON)
    ctx.enqueue_copy(dst_buf=da, src_ptr=hA.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=db, src_ptr=hB.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dc, src_ptr=hC.unsafe_ptr())
    ctx.synchronize()

    identical_gemm_into(ctx, dc, da, db, dw, m, n, k, op)
    ctx.synchronize()

    ctx.enqueue_copy(dst_ptr=hC.unsafe_ptr(), src_buf=dc)
    ctx.synchronize()
    for i in range(mn):
        var v = hC.unsafe_ptr().unsafe_load(i)
        if _bits(v) == _bits(POISON):
            raise Error(
                "bench/gemm_card_main: POISON SURVIVED at cell ("
                + String(i // n)
                + ", "
                + String(i % n)
                + ") of "
                + tag
                + " -- the kernel never wrote it ("
                + _show(v)
                + "), so hashing this buffer would put uninitialized memory"
                " into the card and Phase 3 would read it as a vendor"
                " divergence. Contract section 8 requires the value to be"
                " STORED, including the +0.0 at k == 0."
            )

    t.record_device(ctx, tag, dc, mn)

    # `[[mojo-buffer-freed-at-last-use]]`: every one of these is dead at its
    # `.unsafe_ptr()` unless something uses it later, and a buffer freed
    # under an in-flight kernel or an in-flight copy is read-after-free that
    # only shows up sometimes. Keep them all alive past the last
    # `synchronize` and past the record.
    _ = da
    _ = db
    _ = dc
    _ = dw
    _ = hA
    _ = hB
    _ = hC


def main() raises:
    var arm = String(getenv("MOJOLEARN_GEMM_CARD_ARM"))
    if arm == "":
        arm = String("oracle")
    var full = String(getenv("MOJOLEARN_GEMM_CARD_FULL")) == "1"
    var host_cap = String(getenv("MOJOLEARN_GEMM_CARD_HOST_CAP")) == "1"

    if arm != "oracle" and arm != "serial" and arm != "device":
        raise Error(
            "bench/gemm_card_main: unknown MOJOLEARN_GEMM_CARD_ARM '"
            + arm
            + "'. Known: oracle, serial, device."
        )
    var use_device = arm == "device"

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
    # LOAD-BEARING AS OF DEVIATION 534. It was invisible while the only arms
    # were host-only scalar code that reads no column; the device arm reads
    # `kernel_matrix.mojo` for its tile shapes and its launch geometry, so
    # from here on the sweep is a real test and this line is what proves the
    # define reached the build that produced the card.
    print("column", column_name(TARGET_COLUMN))
    if use_device:
        # The kernel's own sabotage state, on the device arm's banner, for
        # the same reason `gemm_device_check.mojo` prints it: a card emitted
        # from a deliberately broken build is worth exactly as much as the
        # line that says so.
        print("kernel gemm/original/gemm_identical.mojo  sabotage", gemm_sabotage_name())

    var path = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if path == "":
        raise Error(
            "bench/gemm_card_main: MOJOLEARN_IDENTITY_TRACE is unset, so"
            " this run would emit NO CARD and still exit 0 -- and a driver"
            " that checks nothing would record a missing file as agreement."
        )

    # WHICH BUDGET CAPS THIS RUN, decided once and printed, because two cards
    # made under different policies are two different sets of shapes and the
    # differ can only tell you they have different stage counts.
    var cap = String("host: HOST_MAC_BUDGET = ") + String(HOST_MAC_BUDGET) + " MACs"
    if full:
        cap = String("none: MOJOLEARN_GEMM_CARD_FULL=1")
    elif use_device and not host_cap:
        cap = (
            String("device: DEVICE_FLOAT_BUDGET = ")
            + String(DEVICE_FLOAT_BUDGET)
            + " float32 across A, B, C and the workspace"
        )
        comptime if SAB_TINY_DEVICE_BUDGET:
            cap = (
                cap
                + "  -- **TINY BUDGET BUILD**"
                " (-D MOJOLEARN_GEMM_CARD_TINY_DEVICE_BUDGET=1), a REACH"
                " HANDLE for the skip branch. This card is DELIBERATELY"
                " short of shapes and is not a card anybody should diff"
                " against a shipped one."
            )
    elif host_cap:
        cap = (
            String("host (FORCED by MOJOLEARN_GEMM_CARD_HOST_CAP=1):")
            + " HOST_MAC_BUDGET = "
            + String(HOST_MAC_BUDGET)
            + " MACs -- the arms are comparable stage for stage"
        )
    print("cap", cap)

    var t = IdentityTrace()
    var ran = 0
    var skipped = 0

    # The device context is created ONLY on the device arm. The host arms are
    # scalar host code and must keep running on a box with no GPU at all --
    # `tools/gemm_column_invariance.sh` says so in its banner and a driver
    # that opened a context unconditionally would quietly make that false.
    var ctxs = List[DeviceContext]()
    if use_device:
        ctxs.append(DeviceContext())

    for i in range(GEMM_SHAPE_COUNT):
        var m0 = gemm_shape_m(i)
        var n0 = gemm_shape_n(i)
        var k = gemm_shape_k(i)
        var op = gemm_shape_op(i)
        var name = gemm_shape_name(i)

        var m = m0
        var n = n0
        var over = False
        var why = String("")
        if full:
            pass
        elif use_device and not host_cap:
            var dmn = _capped_device(m0, n0, k)
            m = dmn[0]
            n = dmn[1]
            if _device_floats(m, n, k) > DEVICE_FLOAT_BUDGET:
                over = True
                why = (
                    String("k = ")
                    + String(k)
                    + " alone needs "
                    + String(_device_floats(1, 1, k))
                    + " float32 at m = n = 1, past DEVICE_FLOAT_BUDGET = "
                    + String(DEVICE_FLOAT_BUDGET)
                    + "; m and n cannot be reduced further and k may not be"
                    " touched"
                )
        else:
            var hmn = _capped(m0, n0, k)
            m = hmn[0]
            n = hmn[1]
            if m * n * k > HOST_MAC_BUDGET:
                over = True
                why = (
                    String("k = ")
                    + String(k)
                    + " alone exceeds the host budget of "
                    + String(HOST_MAC_BUDGET)
                    + " MACs; needs the device arm"
                )

        if over:
            # REPORTED, NOT SILENTLY DROPPED: a card that quietly omits a
            # shape looks exactly like a card whose shape agreed, and the
            # differ would align the two cards' tag lists around the hole and
            # report a plausible pairing.
            print("  SKIP", name, "--", why)
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
        # products for the same reason. The device arm records the SAME host
        # lists the oracle arm does, before anything is uploaded, so the two
        # cards' input stages are identical by construction rather than by
        # two upload paths agreeing.
        t.record_list_f32(name + ".in.a", a)
        t.record_list_f32(name + ".in.b", b)

        # THE TABLE'S CODE FILLS THE BUFFERS, THE CONTRACT'S CODE DOES THE
        # PRODUCT. `op` above is the table's and the fill branches want it;
        # the oracle AND the device kernel both want `gemm_oracle.mojo`'s
        # numbering. Translated here, explicitly, once. See `_oracle_op`.
        var oop = _oracle_op(op)
        var plan_note = String("")
        if use_device:
            _device_product(t, ctxs[0], name + ".out", a, b, oop, m, n, k)
            # THE PLAN GOES IN THE LOG AND NOWHERE NEAR THE CARD. It is
            # EXECUTION, not numerics (contract 6.1 lets it read m and n);
            # a plan name inside a hash would turn a permitted dispatch
            # difference into a reported divergence and turn "the same bits
            # came out" into "the same plan ran".
            plan_note = String("plan=") + gemm_plan_name(
                choose_gemm_plan(m, n, k)
            )
        else:
            var c: List[Float32]
            if arm == "oracle":
                c = gemm_oracle(a, b, oop, m, n, k)
            else:
                c = gemm_oracle_serial(a, b, oop, m, n, k)
            t.record_list_f32(name + ".out", c)
        ran += 1

        # `P=` STAYS THE LAST FIELD OF THIS LINE. `tools/gemm_card.sh`'s
        # compare_cards reads the leaf count out of the run's own log with
        # `$NF ~ /^P=/` rather than recomputing `contract_leaf_size` in awk,
        # so anything appended after it silently blinds that gate.
        var p = leaf_count(k, contract_leaf_size(k))
        var line = (
            name
            + " m="
            + String(m)
            + (" (capped from " + String(m0) + ")" if m != m0 else "")
            + " n="
            + String(n)
            + (" (capped from " + String(n0) + ")" if n != n0 else "")
            + " k="
            + String(k)
        )
        if plan_note != "":
            line = line + " " + plan_note
        print("  ", line + " P=" + String(p))

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
        "They agree only where P == 1.\n"
        "Diff the DEVICE arm against the oracle -- these MUST be identical:\n"
        "  MOJOLEARN_GEMM_CARD_HOST_CAP=1 MOJOLEARN_GEMM_CARD_ARM=device"
        " MOJOLEARN_IDENTITY_TRACE=/tmp/d.card ...\n"
        "  python3 tools/identity_trace_diff.py /tmp/o.card /tmp/d.card"
    )

    # `[[mojo-buffer-freed-at-last-use]]`'s sibling for the context itself:
    # keep it alive past the last launch and the last record.
    _ = ctxs^
