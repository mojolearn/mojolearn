"""What `mojolearn.identical.gemm.fp32.v1` COSTS: the Phase 4 measurement harness.

Driven by `tools/gemm_price.sh`, which alternates the two build modes round by
round. Emits the same `PRICE <mode> <arm> <ms>` lines
`bench/linalg_price_main.mojo` and `bench/identity_price_main.mojo` emit, so
the median table is SHARED and a v1 number is directly comparable with the
shipped kernels' numbers rather than living in its own units.

THE SPECIFICATION IS CONTRACT SECTION 13.6, WHICH NAMES SIX MEASUREMENTS.
This file implements the ones the arm it was asked for can run and NAMES the
ones that arm cannot, with the reason, rather than quietly measuring five and
reporting six. **THE STATUS OF A MEASUREMENT IS A PROPERTY OF THE ARM**, so
it is tabulated per arm and not per file:

                                                arm host      arm device
    13.6.1  the fold in isolation, P sweep      PARTIAL       ANSWERED
    13.6.2  fully staged vs single-block-fused  NOT RUN       ANSWERED
    13.6.3  end to end over the shape table     PARTIAL       ANSWERED
    13.6.4  the ftz cost apart from the fma     RUNS          RUNS (the same
                                                              host loop)
    13.6.5  workspace bytes and the tiling edge RUNS (exact)  RUNS (exact)
    13.6.6  which resource limits, per shape    INPUTS ONLY   RATES + a
                                                              RELATIVE verdict

**DEVIATION 535 WIRED THE DEVICE ARMS.** Phase 2b's kernel landed
(`gemm/mojo_only/gemm_identical.mojo`, DEVIATIONS 530-532) and the `device`
arm, which until then was a named raising stub carrying its own wiring
instruction, now opens a `DeviceContext` in `main()` and runs that kernel.
What it changed is the table above. **What it did not change is every refusal
that was about something other than the absence of a kernel**, and those are
still refusals below: no vendor-peak roofline (this lane has no peak figure
and no device counters), no certification of any time (timing belongs to the
identity lane), and no claim that the balanced tree is faster or slower than
the superseded serial fold, which 13.7 forbids and which no arm here measures.

THE HOST ARMS ARE NOT A PLACEHOLDER FOR THE DEVICE ARMS and did not become
one. `gemm_oracle` is the NORMATIVE v1 answer and `gemm_oracle_serial` is the
diagnostic reference; pricing them is what establishes the reference cost, and
they are also the only place the two FOLD TOPOLOGIES can be priced against
each other at all -- the serial topology exists on the device only as a
`-D MOJOLEARN_GEMM_SABOTAGE_FOLD_SERIAL` build, which computes the wrong bits
by construction and is not an arm anyone may time against the shipped one.

**EVERY DEVICE ARM'S NAME CONTAINS THE TOKEN `device` AND NO HOST ARM'S
DOES.** That is structural, not a convention: `_report_device` REFUSES to emit
a name without it, so the arm field of a `PRICE` line says which half of the
machine produced the number, and an achieved GFLOP/s figure -- printed only in
the device tables, never for a host arm -- cannot be quoted as a host number
or a host number as a device one.

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

  - **The arms alternate INSIDE the timed loop, call by call.** Every arm of
    one measurement lives in ONE binary, so unlike the FAST/IDENTICAL axis
    they do not need process-level alternation and get the stronger thing
    instead: `for r in REPEATS: time(oracle); time(serial); time(device);
    time(vendor); time(pinned)`. The M4 governor drifts up to 1.7x across
    twenty minutes (`[[mojolearn-box-drifts]]`) and a block of A then a block
    of B measures the drift. **This is why the device arms went into the
    EXISTING loops rather than into a device sweep of their own**: the
    mechanism behind that 1.7x is thermal (the governor pins at MINIMUM clock
    for 96% of an 11-second trace), so two arms measured in two sweeps are
    two thermal states, not two kernels.
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
    13.6.1's TOPOLOGY comparison is priced here on the HOST and against the
    ORACLE's spelling, which allocates a fresh List per tree level; a device
    kernel allocates nothing. The host ratio is a fact about two host
    functions. The device arm does not measure the two topologies against each
    other at all -- it measures two REALIZATIONS of the one shipped topology.
  - **Not that any device number here is a certified timing.** They come off
    one M4 whose governor drifts 1.7x under heat, in a checkout three other
    sessions were live in on 2026-08-23. They are INDICATIVE, the word is
    printed on every device table, and certification is the identity lane's.
  - **Not a roofline.** 13.6.6's device verdict is relative to the best rate
    OBSERVED IN THE SAME RUN, which is a lower bound on peak and not peak.
    There are no device counters in this file and no vendor peak figure.
  - **Not the old ~15 GFLOP/s hand-written contraction number**, which is not
    this design and is not generalized here.
  - **Not the k-NN lane's 2.85x nor the linalg lane's 4.7x as universal.**
    Those are two specific arms at two specific shapes.
  - **Nothing here licenses changing the tree.** If a number comes out bad the
    answer is an execution-plan change that moves no bits (13.7's last item).
"""

from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from max.gpu.host import DeviceBuffer, DeviceContext
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
from core.gemm import PINNED_GEMM_TPB, pinned_gemm_nt_kernel
from gemm.mojo_only.gemm_identical import (
    PLAN_FLAT,
    PLAN_SPLITK,
    PLAN_SPLITK_STAGED,
    choose_gemm_plan,
    gemm_operand_strides,
    gemm_plan_name,
    identical_gemm_into,
    identical_gemm_with_plan,
    identical_gemm_workspace_floats,
)
from gemm.mojo_only.gemm_oracle import (
    CONTRACT_K_LEAF_MIN,
    OP_NN,
    OP_NT,
    OP_TN,
    contract_leaf_count,
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
    numeric_mode_name,
)


def _mode() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29. A local
    two-way IDENTICAL-or-FAST answers "FAST" for a DETERMINISTIC
    build, mislabelling every line the driver prints.
    """
    return numeric_mode_name()


def _report(name: String, ms: Float64):
    """`PRICE <mode> <arm> <ms>`: four whitespace fields, the format
    `tools/price_linalg_identity.sh` and `tools/gemm_price.sh` both parse."""
    print("PRICE", _mode(), name, ms)


def _report_device(name: String, ms: Float64) raises:
    """The same line for a DEVICE arm, and it REFUSES a name that does not
    say so. DEVIATION 535.

    The whole file is one median table shared with
    `bench/linalg_price_main.mojo` and `bench/identity_price_main.mojo`, and
    the table is keyed on the arm field alone. A device millisecond and a host
    millisecond sitting in that table under names a reader cannot tell apart
    is how a scalar host loop gets quoted as a GPU number -- which is the
    exact failure this file already refuses to enable by not printing achieved
    GFLOP/s for its host arms.

    So the discipline is structural rather than editorial: every device arm
    name carries the token `device`, this function checks it, and the check is
    a `raise` and not a comment because a comment does not survive the next
    arm somebody adds in a hurry.
    """
    if name.find("device") < 0:
        raise Error(
            "_report_device: the arm name '"
            + name
            + "' does not contain 'device'. Every device arm must, because"
            " the shared median table is keyed on the arm field and a device"
            " millisecond that reads like a host one is how a host number"
            " gets quoted as a GPU number."
        )
    print("PRICE", _mode(), name, ms)
    # `[[mojo-string-float-roundtrip]]`: the hex bits go on their OWN line and
    # NOT as a fifth field, because `tools/gemm_price.sh`'s median parser
    # takes a sample only when `len(parts) == 4` -- a fifth field would not
    # break the run, it would silently DROP every device sample, which is the
    # worse failure. Non-PRICE lines are printed by the driver, so this
    # travels with the number.
    print("   device-bits", name, _show(ms))


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

#: The DEVICE budget per shape in multiply-accumulates, and it is a SEPARATE
#: number from the host budget above rather than the same one reused.
#:
#: **THE TWO ARMS DO NOT WANT THE SAME SHAPE AND PRETENDING THEY DO WOULD COST
#: THE MEASUREMENT.** The host arm is a scalar loop with a branch per operand
#: fetch, so its budget is set where wall clock stays bearable; a device at
#: that budget measures its own launch overhead and nothing else. Running both
#: at the host budget would produce a device column that is a constant, and
#: running both at the device budget would take the host arm into minutes per
#: row.
#:
#: So each arm caps `m` and `n` to its OWN budget, `k` is exact for both
#: (contract 6.1: the leaf rule and the tree read `k` alone), and every row
#: PRINTS BOTH SHAPES. The arms still alternate call by call inside one timed
#: loop, which is what defeats the governor drift; they are simply not the
#: same size, and a reader who wants `oracle ms` against `device ms` is
#: reading two different shapes and the table says so in the columns.
#:
#: Override with `MOJOLEARN_GEMM_PRICE_DEV_MAC_BUDGET`. 16M MACs is 32 MFLOP,
#: which at any plausible rate for this box is comfortably above the launch
#: floor and comfortably under a second per row at twenty rows times five
#: arms times `REPEATS`.
comptime DEFAULT_DEV_MAC_BUDGET = 16_000_000

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


def fold_sweep_k(i: Int) -> Int:
    """The `k` that REALIZES `fold_sweep_p(i)` on a device. DEVIATION 535.

    **A DEVICE CANNOT BE ASKED FOR A `P`.** The host sweep hands the fold a
    list of length `P` and that is the end of it; the kernel takes `m`, `n`,
    `k` and an orientation, and `P` falls out of contract section 6 --
    `L = contract_leaf_size(k)`, `P = ceil(k / L)`, a pure function of `k`.
    There is no argument to pass and there must not be one: an entry point
    that took `P` would be a second opinion about the partition, which is the
    thing section 6.1 exists to forbid.

    So the sweep is realized through `k`, and `k = 128 P` does it for every
    `P` in `1 .. 1024`: at `k = 128 P` the leaf rule is still in its flat band
    (`ceil(k/128) = P <= MAX_LEAVES`), so `L = 128` and `P = ceil(k/L)` comes
    back exactly. `P = 1` is `k = 128` for the same reason from the other
    side -- `k <= K_LEAF_MIN` gives `L = k` and one leaf.

    **`P = 1018` IS 13.6.1'S OWN `k` AND IS NOT RE-DERIVED HERE.** The
    contract writes the case out: *"`P = 1018` (`k = 131200`) is the
    non-power-of-two, carrying case and must be in the sweep or the carry path
    is priced at zero."* At `k = 131200` the rule has left its flat band --
    `ceil(131200/128) = 1025 > 1024` -- so `L = 129` and `P = 1018` with a
    ragged tail, which is a strictly more interesting row than the `k = 128 *
    1018` that would also give 1018. Using the contract's number keeps the
    sweep anchored to the document rather than to arithmetic done here.

    `check_fold_sweep_k_realizes_the_p_sweep` asserts every row of this,
    against `contract_leaf_count`, rather than trusting the paragraph above.
    """
    var p = fold_sweep_p(i)
    if p == 1018:
        return 131200
    if p <= 1:
        return CONTRACT_K_LEAF_MIN
    return CONTRACT_K_LEAF_MIN * p


#: 13.6.4's leaf loop length, and the passes over it. One pass is one output
#: cell's whole-`k` contraction at a `k` in the flat leaf band.
comptime LEAF_K = 262_144
comptime LEAF_PASSES = 8

#: 13.6.6's reporting threshold: the share of THIS RUN's best observed rate
#: at which a row is called after that resource. **NOT A BOUND AND NOT A
#: ROOFLINE RIDGE POINT** -- there is no vendor peak figure in this file and
#: no device counter, so the only ceilings available are the best rates the
#: run itself reached, which are lower bounds on peak. The number is printed
#: with the verdict so it travels with it.
comptime LIMITER_SHARE = 0.5

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


def _bits64(x: Float64) -> UInt64:
    return bitcast[DType.uint64](x)


def _show(x: Float64) -> String:
    """`<decimal>/<hex bits>`, `[[mojo-string-float-roundtrip]]`.

    Mojo's `String(Float64)` does not round-trip, so a decimal alone is a
    number nobody can reload. Every float this file states OUTSIDE a
    column-aligned table goes through here. **The fixed-point cells inside
    the tables are DISPLAY** -- three decimals, already lossy by construction
    and never a source anyone reloads -- and the full-width value with its
    bits is always available on the arm's own `PRICE`/`device-bits` pair.
    """
    return String(x) + "/" + hex(_bits64(x))


# ===========================================================================
# THE DEVICE HARNESS (DEVIATION 535)
# ===========================================================================
# Four helpers, and the only subtle thing in them is the `_ = h` at the end of
# each: `[[mojo-buffer-freed-at-last-use]]`, a `DeviceBuffer` is dead at its
# `.unsafe_ptr()`, so a host staging buffer whose last textual use is the
# pointer handed to `enqueue_copy` can be freed before the copy runs. The
# failure is nondeterministic and a small fixture cannot see it, which is
# exactly why the line is there rather than the argument being made once and
# trusted. `gemm/mojo_only/gemm_device_check.mojo::_run_device` carries the
# same tail for the same reason and this is modelled on it.

#: Written into `C` before a device arm runs. A cell still holding it
#: afterwards was NEVER WRITTEN, and a timing of a kernel that did not write
#: its output is a timing of nothing. Same value and same role as
#: `gemm_device_check.mojo::POISON`.
comptime DEVICE_POISON = Float32(-987654.0)


def _dev_fill(
    ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    n: Int,
    salt: Int,
) raises:
    """Upload `n` bit-assembled fixture floats into `buf`.

    The generator is `_exact`, the same one the host arms use, so a device
    arm and a host arm at the same index see the same value. It is filled
    through a host buffer rather than from a `List` so that a `k` in the
    millions does not materialize a second copy on the heap.
    """
    if n < 1:
        return
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n):
        h.unsafe_ptr().unsafe_store(i, _exact(i, salt))
    ctx.enqueue_copy(dst_buf=buf, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h


def _dev_poison(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], mn: Int
) raises:
    """Fill `mn` cells of `buf` with `DEVICE_POISON`."""
    if mn < 1:
        return
    var h = ctx.enqueue_create_host_buffer[DType.float32](mn)
    ctx.synchronize()
    for i in range(mn):
        h.unsafe_ptr().unsafe_store(i, DEVICE_POISON)
    ctx.enqueue_copy(dst_buf=buf, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h


def _dev_digest(
    ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    mn: Int,
    tag: String,
) raises -> UInt64:
    """Read `C` back, REFUSE a surviving poison, and return an FNV-1a digest
    of its bits.

    **THIS IS THE `[[reached-but-inert]]` CHECK FOR A TIMING HARNESS.** A
    kernel that launches, returns instantly and writes nothing produces a
    beautiful millisecond. The poison catches that: every cell must have been
    stored, which contract section 8 requires even at `k == 0` where the
    required answer is a stored `+0.0`.

    The digest is POSITION SENSITIVE (a multiply between cells), so it is not
    an XOR that two different outputs can collide on by permutation, and it is
    what lets two execution PLANS be compared for bit equality at the cost of
    one 64-bit integer instead of an `m n` array.
    """
    var h = ctx.enqueue_create_host_buffer[DType.float32](mn if mn > 0 else 1)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var d = UInt64(0xCBF29CE484222325)
    for i in range(mn):
        var v = h.unsafe_ptr().unsafe_load(i)
        if _bits(v) == _bits(DEVICE_POISON):
            raise Error(
                "POISON SURVIVED at cell "
                + String(i)
                + " of "
                + tag
                + ": the kernel never wrote it, so the milliseconds beside it"
                " are a timing of a launch that did not produce an answer."
            )
        d = (d ^ UInt64(Int(_bits(v)))) * UInt64(0x100000001B3)
    _ = h
    return d


def _plans_agree(tag: String, a_name: String, a: UInt64,
                 b_name: String, b: UInt64) raises:
    """Two execution plans over one shape must produce ONE array of bits.

    Contract 0.3 and section 6.1: the arithmetic for a cell is a function of
    `k` and the operands alone, so a plan is a SCHEDULE and a schedule may not
    reach the answer. Two plans that disagree are not two timings of the same
    thing and their ratio means nothing.

    **ASSERTED UNDER IDENTICAL, REPORTED UNDER FAST**, which is the policy
    `gemm/mojo_only/gemm_device_check.mojo` states at the same seam: under
    FAST both `identical_mul_add` and `ftz` compile away, the backend is free
    to contract one kernel and not another, and whether two unpinned spellings
    agree is a measurement rather than a bug.
    """
    if a == b:
        return
    var msg = String("PLANS DISAGREE at ") + tag + ": "
    msg = msg + a_name + " digest " + hex(a)
    msg = msg + " vs " + b_name + " digest " + hex(b)
    if _mode() == "IDENTICAL":
        raise Error(
            msg
            + ". Under IDENTICAL that is a contract violation, not a"
            " measurement: a plan is a schedule and section 6.1 forbids a"
            " schedule from reaching the arithmetic."
        )
    print("   NOTE (FAST, not a failure):", msg)
    print(
        "        Under FAST neither spelling is pinned and the backend may"
        " contract one kernel and not the other. The IDENTICAL leg is where"
        " this is asserted."
    )


def _gfps(flops: Float64, ms: Float64) -> Float64:
    """Achieved GFLOP/s. **PRINTED FOR DEVICE ARMS ONLY**, and every caller is
    inside a table whose banner says DEVICE and INDICATIVE.

    A host scalar-loop figure printed in these units gets quoted as a device
    number inside a week, which is why the host tables in this file have no
    such column and are not getting one."""
    if ms <= 0.0:
        return 0.0
    return flops / (ms * 1.0e6)


def _launches_fused(p: Int) -> Int:
    """How many kernels `PLAN_SPLITK` actually enqueues at leaf count `p`.

    Read off `identical_gemm_with_plan`'s SPLITK branch and not off contract
    13.6.2's idealized count: one `identical_gemm_leaf_kernel`, then one
    `identical_gemm_fold_kernel` that folds every level of the tree inside one
    block with a barrier between them. That is 13.4's single-block-fused
    realization, and it is 2 launches at every `p >= 1`, including `p == 1`
    where the fold has no addition to perform but still runs (contract 7.3:
    a one-node tree has no internal node to skip, and the output seam is what
    fixes the sign of a zero).

    At `p == 0` -- `k == 0` -- both SPLITK plans fall through to the flat
    kernel, which is ONE launch that stores the required `+0.0`.
    """
    if p <= 0:
        return 1
    return 2


def _launches_staged(p: Int) -> Int:
    """How many kernels `PLAN_SPLITK_STAGED` actually enqueues at leaf count
    `p`, and it is `D + 2`, not the `D + 1` the contract's table names.

    One `identical_gemm_leaf_kernel`, then one
    `identical_gemm_fold_level_kernel` per ARITHMETIC level -- the launcher's
    loop is `for d in range(1, fold_level_count(p))`, so that is
    `fold_level_count(p) - 1 = D` of them -- and then one
    `identical_gemm_emit_kernel` that carries the root through the output
    seam.

    **THE EMIT KERNEL IS THE ONE 13.6.2'S ARITHMETIC OMITS.** 13.6.2 counts
    "`D` fold launches, i.e. `D + 1` kernels counting the leaf stage"; the
    landed staged plan pays one more because the root lives in the workspace
    and the output seam is its own store. Reporting `D + 1` here would
    understate the staged arm by exactly one launch at every `P`, which is
    the whole quantity 13.6.2 exists to price. So the table below prints BOTH
    counts and labels which is the contract's and which is the launcher's.
    """
    if p <= 0:
        return 1
    return fold_level_count(p) + 1


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


def check_device_op_encoding_matches_the_kernel() raises:
    """THE ORIENTATION TRAP, CLOSED AGAINST THE DEVICE KERNEL. DEVIATION 535.

    `check_op_encodings_are_not_interchangeable` above proves the map lands on
    the ORACLE's spelling of the three words. That was enough while the only
    consumer was the oracle. **It is not enough now**: the device arm hands
    `op` to `gemm/mojo_only/gemm_identical.mojo`, and the question that
    matters is which ADDRESSING that file derives from the code it is given,
    not which word a `String` comes back as.

    So this gate goes through the kernel's own
    `gemm_operand_strides(op, m, n, k)` -- the single function every one of its
    launchers calls to turn an orientation into the two index expressions --
    and compares its four numbers, per shape, against contract section 3
    transcribed HERE, keyed off the shape table's OWN word for the row:

        OP_NN, OP_NT :  A is m x k row-major  ->  (a_si, a_sp) = (k, 1)
        OP_TN        :  A is k x m row-major  ->  (a_si, a_sp) = (1, m)
        OP_NN, OP_TN :  B is k x n row-major  ->  (b_sp, b_sj) = (n, 1)
        OP_NT        :  B is n x k row-major  ->  (b_sp, b_sj) = (1, k)

    The two sides come from different files and neither is derived from the
    other: the left from the kernel through `_tbl_op`, the right from the
    contract through `gemm_shape_op`. They can only agree if the map is right.

    **AND THE SABOTAGE DIRECTION IS ASSERTED TOO.** A gate that only checked
    agreement would pass just as happily if the map were the identity and the
    two encodings had quietly coincided. So it also counts the rows whose
    addressing WOULD CHANGE if the raw table code were passed through
    untranslated, and refuses if that count is zero. On the shipped table it
    is every row, because `TBL_OP_NT = 0 = OP_NN` and `TBL_OP_TN = 1 = OP_NT`:
    an untranslated hand-off reads a `k x m` operand as `m x k` and returns
    plausible, in-bounds, wrong products -- which is what a full reference
    card of them cost this lane once already.
    """
    var differs = 0
    for i in range(GEMM_SHAPE_COUNT):
        var m = gemm_shape_m(i)
        var n = gemm_shape_n(i)
        var k = gemm_shape_k(i)
        var st = gemm_operand_strides(_tbl_op(i), m, n, k)

        var want_a_si = k
        var want_a_sp = 1
        if gemm_shape_op(i) == TBL_OP_TN:
            want_a_si = 1
            want_a_sp = m
        var want_b_sp = n
        var want_b_sj = 1
        if gemm_shape_op(i) == TBL_OP_NT:
            want_b_sp = 1
            want_b_sj = k

        if (
            st[0] != want_a_si
            or st[1] != want_a_sp
            or st[2] != want_b_sp
            or st[3] != want_b_sj
        ):
            raise Error(
                "device op map: row '"
                + gemm_shape_name(i)
                + "' is a "
                + op_name(_tbl_op(i))
                + " and the kernel's gemm_operand_strides returned (a_si,"
                " a_sp, b_sp, b_sj) = ("
                + String(st[0]) + "," + String(st[1]) + ","
                + String(st[2]) + "," + String(st[3])
                + ") where contract section 3 requires ("
                + String(want_a_si) + "," + String(want_a_sp) + ","
                + String(want_b_sp) + "," + String(want_b_sj)
                + "). The device arm would address the wrong operand layout"
                " and still return in-bounds floats."
            )

        var raw = gemm_operand_strides(gemm_shape_op(i), m, n, k)
        if (
            raw[0] != st[0]
            or raw[1] != st[1]
            or raw[2] != st[2]
            or raw[3] != st[3]
        ):
            differs += 1

    if differs == 0:
        raise Error(
            "device op map: passing the RAW table code to the kernel would"
            " produce the SAME addressing on every one of the "
            + String(GEMM_SHAPE_COUNT)
            + " shapes, so the translation is inert and this gate proves"
            " nothing. Either the two encodings were unified on purpose --"
            " delete the map and both gates together -- or the table's codes"
            " moved."
        )
    print(
        "check_device_op_encoding_matches_the_kernel OK: all",
        GEMM_SHAPE_COUNT,
        "rows match contract section 3 through the KERNEL's own stride"
        " function;",
        differs,
        "would address DIFFERENTLY if the table code were passed raw",
    )


def _carry_levels(p: Int) -> Int:
    """How many levels of the balanced tree CARRY an unpaired leaf at leaf
    count `p`: the levels whose width is odd. Contract 7.2 copies that leaf
    bit-for-bit rather than padding it with a zero, and it is the sharpest
    clause in the fold."""
    var carries = 0
    var w = p
    while w > 1:
        if w % 2 != 0:
            carries += 1
        w = (w + 1) // 2
    return carries


def check_fold_sweep_k_realizes_the_p_sweep() raises:
    """Every `k` in the device fold sweep produces EXACTLY the `P` it claims.

    The host sweep passes `P` as a parameter; the device sweep cannot and goes
    through `k` (`fold_sweep_k`). **That is a translation, and an untested
    translation in a benchmark is a table of correctly-labelled measurements
    of the wrong sizes** -- the same failure class as the orientation map two
    gates up, and it would be invisible: a run at `P = 512` labelled `P = 1024`
    produces a perfectly smooth curve.

    Three things are asserted and the third is the one worth having:

      1. `contract_leaf_count(fold_sweep_k(i)) == fold_sweep_p(i)` for every
         row of the sweep, through the contract's own function.
      2. 13.6.1's own `k = 131200` is the `P = 1018` row, and at that `k` the
         leaf rule has LEFT its flat band -- `L = 129`, not 128 -- which is
         what makes it the ragged, past-the-crossover case the contract names.
      3. **The carry path is not priced at zero.** 13.6.1 puts `P = 1018` in
         the sweep precisely because it CARRIES; if no swept `P` had an odd
         level width, every number the sweep produced would be about the even
         path and the carry clause would be measured nowhere. So the gate
         counts the carrying rows and refuses a sweep with none, and it
         checks that `P = 1024` -- a power of two -- carries at NO level,
         which is what makes the pair a separation rather than a coincidence.
    """
    for i in range(FOLD_P_COUNT):
        var p = fold_sweep_p(i)
        var k = fold_sweep_k(i)
        var got = contract_leaf_count(k)
        if got != p:
            raise Error(
                "fold sweep: k = "
                + String(k)
                + " was chosen to realize P = "
                + String(p)
                + " and contract_leaf_count gives P = "
                + String(got)
                + ". Every device row of 13.6.1 would be labelled with a leaf"
                " count it does not have."
            )
    if fold_sweep_k(6) != 131200 or contract_leaf_size(131200) != 129:
        raise Error(
            "fold sweep: 13.6.1 names P = 1018 at k = 131200, where the leaf"
            " rule is past its flat band and L = 129. This sweep has k = "
            + String(fold_sweep_k(6))
            + " and L = "
            + String(contract_leaf_size(131200))
        )
    var carrying = 0
    for i in range(FOLD_P_COUNT):
        if _carry_levels(fold_sweep_p(i)) > 0:
            carrying += 1
    if carrying == 0:
        raise Error(
            "fold sweep: NO swept P carries at any level, so the carry path"
            " -- the clause contract 7.2 is sharpest about -- is priced at"
            " zero and the sweep measures only the even path."
        )
    if _carry_levels(1024) != 0 or _carry_levels(1018) == 0:
        raise Error(
            "fold sweep: P = 1024 must carry at NO level and P = 1018 must"
            " carry at at least one; got "
            + String(_carry_levels(1024))
            + " and "
            + String(_carry_levels(1018))
        )
    print(
        "check_fold_sweep_k_realizes_the_p_sweep OK: all",
        FOLD_P_COUNT,
        "swept P realized exactly through k;",
        carrying,
        "of them CARRY (P=1018 at k=131200, L=129, carries",
        _carry_levels(1018),
        "levels; P=1024 carries 0)",
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


def measure_fold_in_isolation(devs: List[DeviceContext]) raises:
    """13.6.1: the fold in isolation, over the `P` sweep. DEVIATION 535.

    Sweeps `P` over `1, 2, 3, 8, 32, 128, 1018, 1024` at a fixed `64 x 64`
    output and prices the two topologies over BIT-IDENTICAL partials -- the
    leaf stage is not merely "the same", it is absent from the timed region
    entirely, which is a stronger isolation than 13.6.1 asks for.

    **WHAT THE HOST ARMS CANNOT SAY, and it is the important half of them.**
    13.6.1 calls itself "the only measurement that can support or refute
    13.4", and 13.4 is an argument about LAUNCHES: that all `D` levels fuse
    into one threadgroup with a barrier between them under assumption A4. A
    host loop has no launches, no threadgroups and no barriers. **So the host
    arms do not support or refute 13.4 and must not be quoted as doing so.**
    What they price is the ORACLE's fold spelling, which allocates a fresh
    `List` per tree level; a device kernel allocates nothing, so the host tree
    arm carries a cost the device arm will not have. Read the ratio as an
    upper bound on the tree's disadvantage in the oracle, and nothing else.

    **THE DEVICE ARMS ARE WHERE 13.4 IS ACTUALLY ADDRESSED, AND THEY ARE NOT
    THE TWO TOPOLOGIES.** DEVIATION 535 adds two more arms at each `P`, both
    computing the SHIPPED balanced tree and differing only in where it is
    realized:

        device.flat     `PLAN_FLAT`: one thread owns a whole output cell, the
                        leaf partials never leave its registers, and the tree
                        is folded in a register stack. ONE launch, no
                        workspace, no global partial ever written.
        device.splitk   `PLAN_SPLITK`: the leaf kernel writes `m n P` partials
                        to global memory and a SECOND kernel folds every level
                        of the tree inside one block with a barrier between
                        them. TWO launches. This is 13.4's single-block-fused
                        realization.

    **THE DIFFERENCE BETWEEN THEM IS NOT ONLY THE EXTRA LAUNCH.** It is the
    extra launch PLUS the staging traffic: `m n P` floats written once and
    read once, which 13.6.6 counts as bandwidth and 13.5 counts as capacity.
    Naming the gap "launch overhead" would be wrong by however much of it is
    the round trip through memory, and at `P = 1018` that round trip is 33 MB
    at a 64x64 output. What the pair does answer, exactly, is 13.4's real
    question: **what does it cost to realize the fold somewhere other than in
    the registers that produced it**, and the answer is a number rather than
    an argument for the first time in this profile.

    The device sweep reaches its `P` through `k`, because a kernel has no `P`
    argument and contract 6.1 forbids one; `fold_sweep_k` does the
    translation and `check_fold_sweep_k_realizes_the_p_sweep` asserts it.

    **TWO PAIRS, TWO LOOPS, AND THE PAIRING IS THE POINT.** The host pair
    alternates call by call with itself and the device pair alternates call by
    call with itself; the two pairs do not interleave with each other. That is
    not a shortcut, it is what the drift argument actually requires: the
    governor drift makes a COMPARISON invalid when its two arms sit in
    different thermal windows, and the two comparisons this measurement makes
    are tree-against-serial and flat-against-splitk. There is no third
    comparison. A host millisecond over `List` allocations and a device
    millisecond over 4096 threads are not two arms of anything, and putting
    them in one loop would suggest they were.
    """
    var on_device = len(devs) == 1
    print()
    print("-- 13.6.1  the fold in isolation, both topologies, P sweep --")
    print(
        "   fixed output",
        String(FOLD_M) + "x" + String(FOLD_N),
        "=",
        FOLD_CELLS,
        "cells; partials are bit-assembled and IDENTICAL between the arms",
    )
    if on_device:
        print(
            "   DEVICE ARMS PRESENT (DEVIATION 535). The host pair is the two"
            " TOPOLOGIES; the device pair is ONE topology in two"
            " REALIZATIONS, at 64x64 OP_NT with k chosen to realize each P."
            " Device times are INDICATIVE: one thermally unstable M4, and"
            " timing is the identity lane's to certify."
        )
    else:
        print(
            "   HOST ONLY. 13.4 is a LAUNCH argument and there are no launches"
            " here; this refutes nothing and supports nothing about it. Run"
            " MOJOLEARN_GEMM_PRICE_ARM=device for the arms that do."
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

        # Untimed warm-up, both host arms, before either clock starts.
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
        var carries = _carry_levels(p)
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
    if on_device:
        _fold_sweep_on_device(devs[0])


def _fold_sweep_on_device(ctx: DeviceContext) raises:
    """13.6.1's `P` sweep AS DEVICE LAUNCHES. DEVIATION 535, and this is the
    half of 13.6.1 that bears on contract 13.4.

    Fixed `64 x 64` output, `OP_NT`, and `k = fold_sweep_k(s)` so that each
    row's leaf count is exactly the swept `P` -- asserted, not assumed, by
    `check_fold_sweep_k_realizes_the_p_sweep`.

    TWO ARMS, ALTERNATING CALL BY CALL, both computing the same bits:

        device.flat     `PLAN_FLAT`, ONE launch. Each thread owns an output
                        cell, accumulates every leaf into its own registers
                        and folds the tree in a register stack. No workspace,
                        no partial ever written to memory.
        device.splitk   `PLAN_SPLITK`, TWO launches. The leaf kernel writes
                        `m n P` partials to global memory; one fold kernel
                        then folds every level inside one block with a barrier
                        between them, which is exactly the realization 13.4
                        argues is available at every legal `k`.

    **BOTH ARMS ARE CHECKED FOR BIT AGREEMENT BEFORE EITHER IS TIMED**
    (`_plans_agree`, asserted under IDENTICAL). Two plans that disagree are
    not two timings of one thing. And every output is read back through
    `_dev_digest`, which refuses a surviving poison: a launch that wrote
    nothing would otherwise turn in the best millisecond in the table.

    WHAT THE GAP IS, STATED BEFORE THE NUMBERS SO IT CANNOT BE RENAMED
    AFTERWARDS. `splitk - flat` is the extra launch PLUS the staging round
    trip -- `m n P` floats written once and read once -- PLUS whatever the two
    kernels differ by in occupancy, because they do not have the same shape.
    It is an upper bound on the launch cost 13.4 is about and it is not a
    measurement of launch overhead alone. The workspace column is printed
    beside the times for exactly that reason.
    """
    print()
    print(
        "-- 13.6.1  the same P sweep AS DEVICE LAUNCHES (INDICATIVE) --"
    )
    print(
        "   " + String(FOLD_M) + "x" + String(FOLD_N) + " OP_NT, k chosen to"
        " realize each P (fold_sweep_k). Both arms are the SHIPPED balanced"
        " tree; they differ in WHERE it is folded, not in what it computes."
    )
    print(
        "     " + _padl("P", 6) + _padl("k", 9) + _padl("L", 6)
        + _padl("flat ms", 11) + _padl("splitk ms", 12)
        + _padl("s/f", 8) + _padl("stage MiB", 11)
        + _padl("GF/s flat", 11) + _padl("GF/s splitk", 13)
    )
    var m = FOLD_M
    var n = FOLD_N
    var mn = m * n
    for s in range(FOLD_P_COUNT):
        var p = fold_sweep_p(s)
        var k = fold_sweep_k(s)
        var na = _a_elems(OP_NT, m, n, k)
        var nb = _b_elems(OP_NT, m, n, k)
        var nws = identical_gemm_workspace_floats(m, n, k, PLAN_SPLITK)
        if nws < 1:
            nws = 1

        var da = ctx.enqueue_create_buffer[DType.float32](na)
        var db = ctx.enqueue_create_buffer[DType.float32](nb)
        var dc = ctx.enqueue_create_buffer[DType.float32](mn)
        var dw = ctx.enqueue_create_buffer[DType.float32](nws)
        ctx.synchronize()
        _dev_fill(ctx, da, na, 101 + s)
        _dev_fill(ctx, db, nb, 202 + s)

        # Untimed warm-up on BOTH arms, and the agreement check on their
        # results, before either clock starts. A first launch pays its
        # pipeline build and its cold caches.
        _dev_poison(ctx, dc, mn)
        identical_gemm_with_plan(ctx, dc, da, db, dw, m, n, k, OP_NT,
                                 PLAN_FLAT)
        ctx.synchronize()
        var d_flat = _dev_digest(ctx, dc, mn, String("m1.P") + String(p)
                                 + ".device.flat")
        _dev_poison(ctx, dc, mn)
        identical_gemm_with_plan(ctx, dc, da, db, dw, m, n, k, OP_NT,
                                 PLAN_SPLITK)
        ctx.synchronize()
        var d_splitk = _dev_digest(ctx, dc, mn, String("m1.P") + String(p)
                                   + ".device.splitk")
        _plans_agree(
            String("m1.P") + String(p),
            gemm_plan_name(PLAN_FLAT),
            d_flat,
            gemm_plan_name(PLAN_SPLITK),
            d_splitk,
        )

        var ns_flat = 0
        var ns_splitk = 0
        for _ in range(REPEATS):
            var t0 = perf_counter_ns()
            identical_gemm_with_plan(ctx, dc, da, db, dw, m, n, k, OP_NT,
                                     PLAN_FLAT)
            ctx.synchronize()
            ns_flat += perf_counter_ns() - t0

            var t1 = perf_counter_ns()
            identical_gemm_with_plan(ctx, dc, da, db, dw, m, n, k, OP_NT,
                                     PLAN_SPLITK)
            ctx.synchronize()
            ns_splitk += perf_counter_ns() - t1

        var ms_flat = Float64(ns_flat) / 1.0e6 / Float64(REPEATS)
        var ms_splitk = Float64(ns_splitk) / 1.0e6 / Float64(REPEATS)
        var tag = String("m1.P") + String(p)
        _report_device(tag + ".device.flat", ms_flat)
        _report_device(tag + ".device.splitk", ms_splitk)

        var flops = 2.0 * Float64(m) * Float64(n) * Float64(k)
        print(
            "     " + _padl(String(p), 6) + _padl(String(k), 9)
            + _padl(String(contract_leaf_size(k)), 6)
            + _padl(_fixed(ms_flat, 3), 11)
            + _padl(_fixed(ms_splitk, 3), 12)
            + _padl(_fixed(_ratio(ms_splitk, ms_flat), 2) + "x", 8)
            + _padl(_fixed(Float64(mn * p * 4) / 1048576.0, 2), 11)
            + _padl(_fixed(_gfps(flops, ms_flat), 2), 11)
            + _padl(_fixed(_gfps(flops, ms_splitk), 2), 13)
        )
        # `[[mojo-buffer-freed-at-last-use]]`: these four keep the owners
        # alive past the last `synchronize()` above. Without them a buffer is
        # dead at its `.unsafe_ptr()` inside the launcher and the free races
        # the kernel -- nondeterministically, and invisibly at a small shape.
        _ = da
        _ = db
        _ = dc
        _ = dw
    print(
        "   GF/s columns are DEVICE arms only and are INDICATIVE. There is no"
        " host GFLOP/s column anywhere in this file and there is not going to"
        " be one: a host scalar-loop figure in these units gets quoted as a"
        " GPU number inside a week."
    )
    print(
        "   splitk - flat is the extra LAUNCH plus the staging round trip"
        " (the stage MiB column, written once and read once) plus an"
        " occupancy difference between two differently shaped kernels. It is"
        " an UPPER BOUND on what 13.4's extra kernel costs, not a measurement"
        " of launch overhead on its own."
    )


# ===========================================================================
# 13.6.2: FULLY STAGED AGAINST SINGLE-BLOCK-FUSED
# ===========================================================================


def report_staging_structure(devs: List[DeviceContext]) raises:
    """13.6.2: fully staged against single-block-fused. DEVIATION 535.

    13.6.2 is a LAUNCH-COUNT measurement, so on the `host` arm it is still
    what it always was -- the STRUCTURE and not a time, printed rather than
    omitted, because a driver that printed nothing here would let a reader
    conclude 13.6.2 was covered by 13.6.1. It is not: 13.6.1 holds the
    staging fixed and 13.6.2 varies it.

    On the `device` arm the two arms 13.6.2 names are TIMED, over the same `P`
    sweep, same tree, same bits:

        device.fused    `PLAN_SPLITK`: leaf kernel, then ONE fold kernel that
                        walks every level inside one block with a barrier
                        between them. **2 launches at every P.**
        device.staged   `PLAN_SPLITK_STAGED`: leaf kernel, one
                        `fold_level_kernel` per ARITHMETIC level, then an
                        emit kernel. **`D + 2` launches**, and the `+2` is
                        explained below because it is not what the contract
                        writes.

    **THE LAUNCH COUNTS THIS TABLE PRINTS ARE THE LAUNCHER'S, NOT 13.6.2'S.**
    13.6.2 says "`D` fold launches, i.e. `D + 1` kernels counting the leaf
    stage". The landed staged plan enqueues one more than that -- the emit
    kernel that carries the tree's root through the output seam, because in
    the staged realization the root is a workspace entry and the store to `C`
    is its own kernel. Printing `D + 1` would understate the staged arm by
    exactly one launch at every `P`, which is precisely the quantity being
    priced. So BOTH counts are printed and each is labelled with whose it is.
    `_launches_fused` and `_launches_staged` carry the derivation.

    **AT `P = 1` THE TWO REALIZATIONS COST THE SAME TWO LAUNCHES**, from
    opposite directions: the fused arm runs its fold kernel over a one-node
    tree (contract 7.3 -- a one-node tree has no internal node to skip, and
    the output seam is what keeps the sign of a zero off the partition count),
    and the staged arm has no arithmetic level to launch and goes straight to
    its emit. Four of the twenty real shapes sit on that row, and an earlier
    draft of this table printed a flat `2` in the fused column and a `1` in
    the staged column, which had the comparison backwards at the one `P` where
    it is exactly even.

    WHAT MAY NOT BE CONCLUDED, 13.7 verbatim in its own words: *"If Phase 2b
    measures the staged tree as slow, the answer is to fuse levels into one
    block, which is legal and produces the same bits. It is never to change
    the pairing."* Nothing in this table licenses a fold change.
    """
    var on_device = len(devs) == 1
    print()
    if on_device:
        print("-- 13.6.2  fully staged vs single-block-fused (INDICATIVE) --")
    else:
        print(
            "-- 13.6.2  fully staged vs single-block-fused -- NOT MEASURED --"
        )
        print(
            "   The host arm has no launches, so the cost 13.6.2 exists to"
            " price cannot appear in any number this binary prints. Run"
            " MOJOLEARN_GEMM_PRICE_ARM=device. The STRUCTURE it prices:"
        )
    print(
        "     " + _padl("P", 5) + _padl("D", 4) + _padl("13.6.2 st", 11)
        + _padl("actual st", 11) + _padl("actual fu", 11)
        + _padl("nodes/cell", 12) + "  note"
    )
    for s in range(FOLD_P_COUNT):
        var p = fold_sweep_p(s)
        var d = fold_level_count(p) - 1
        var note = String("")
        if p == 1:
            note = String("P=1: NO fold addition (contract 7.3)")
        elif p % 2 != 0:
            note = String("odd P: the tree CARRIES")
        print(
            "     " + _padl(String(p), 5) + _padl(String(d), 4)
            + _padl(String(d + 1), 11)
            + _padl(String(_launches_staged(p)), 11)
            + _padl(String(_launches_fused(p)), 11)
            + _padl(String(fold_node_total(p)), 12) + "  " + note
        )
    print(
        "   Column '13.6.2 st' is the contract's count (D + 1) and 'actual"
        " st' is what identical_gemm_with_plan enqueues (D + 2: the emit"
        " kernel). The gap is one launch at every P and it is the staged"
        " arm's, so the contract's arithmetic UNDERSTATES the arm it is"
        " warning about."
    )
    print(
        "   Contract 13.4's expectation, which is an ARGUMENT and not a"
        " measurement: the fused column is achievable at every legal k"
        " because MAX_LEAVES caps P at 1024 and 1024 float32 partials is"
        " 4 KB of threadgroup memory."
    )
    if on_device:
        _staging_arms_on_device(devs[0])


def _staging_arms_on_device(ctx: DeviceContext) raises:
    """13.6.2's two arms, TIMED. DEVIATION 535.

    Fixed `64 x 64` `OP_NT`, `k = fold_sweep_k(s)` so each row's leaf count is
    the swept `P`, the two plans alternating call by call inside one timed
    loop, and their outputs checked for bit agreement before either is timed.

    **ONE WORKSPACE, SIZED FOR THE LARGER PLAN, AND THE SIZE IS GATED.** The
    staged plan strides its scratch by `fold_node_total(P)` and the fused plan
    by `P`, so a workspace sized for the fused arm and handed to the staged
    arm is an out-of-bounds write of about a factor of two.
    `gemm/mojo_only/gemm_identical.mojo` records that exact defect costing a
    run -- a SPLITK dispatch at 64x64x4096 wrote 512 KB of partials into a
    one-float buffer and whole regions of `C` came back `+0.0` -- so the size
    here is the max over BOTH plans and a `raise` guards it rather than a
    comment.
    """
    print()
    print("-- 13.6.2  staged vs fused AS DEVICE LAUNCHES (INDICATIVE) --")
    print(
        "     " + _padl("P", 5) + _padl("k", 9) + _padl("launches", 10)
        + _padl("fused ms", 11) + _padl("staged ms", 12)
        + _padl("st/fu", 8) + _padl("fused MiB", 11)
        + _padl("staged MiB", 12)
    )
    var m = FOLD_M
    var n = FOLD_N
    var mn = m * n
    for s in range(FOLD_P_COUNT):
        var p = fold_sweep_p(s)
        var k = fold_sweep_k(s)
        var na = _a_elems(OP_NT, m, n, k)
        var nb = _b_elems(OP_NT, m, n, k)
        var ws_fused = identical_gemm_workspace_floats(m, n, k, PLAN_SPLITK)
        var ws_staged = identical_gemm_workspace_floats(
            m, n, k, PLAN_SPLITK_STAGED
        )
        var nws = ws_fused
        if ws_staged > nws:
            nws = ws_staged
        if nws < 1:
            nws = 1
        if nws < ws_fused or nws < ws_staged:
            raise Error(
                "13.6.2: the workspace does not cover both plans at P = "
                + String(p)
            )

        var da = ctx.enqueue_create_buffer[DType.float32](na)
        var db = ctx.enqueue_create_buffer[DType.float32](nb)
        var dc = ctx.enqueue_create_buffer[DType.float32](mn)
        var dw = ctx.enqueue_create_buffer[DType.float32](nws)
        ctx.synchronize()
        _dev_fill(ctx, da, na, 301 + s)
        _dev_fill(ctx, db, nb, 402 + s)

        _dev_poison(ctx, dc, mn)
        identical_gemm_with_plan(ctx, dc, da, db, dw, m, n, k, OP_NT,
                                 PLAN_SPLITK)
        ctx.synchronize()
        var d_fused = _dev_digest(ctx, dc, mn, String("m2.P") + String(p)
                                  + ".device.fused")
        _dev_poison(ctx, dc, mn)
        identical_gemm_with_plan(ctx, dc, da, db, dw, m, n, k, OP_NT,
                                 PLAN_SPLITK_STAGED)
        ctx.synchronize()
        var d_staged = _dev_digest(ctx, dc, mn, String("m2.P") + String(p)
                                   + ".device.staged")
        # SAME TREE, SAME BITS is 13.6.2's own precondition ("same tree, same
        # bits, at the same P sweep"). If the two staging realizations
        # disagree there is no ratio to report, only a defect.
        _plans_agree(
            String("m2.P") + String(p),
            gemm_plan_name(PLAN_SPLITK),
            d_fused,
            gemm_plan_name(PLAN_SPLITK_STAGED),
            d_staged,
        )

        var ns_fused = 0
        var ns_staged = 0
        for _ in range(REPEATS):
            var t0 = perf_counter_ns()
            identical_gemm_with_plan(ctx, dc, da, db, dw, m, n, k, OP_NT,
                                     PLAN_SPLITK)
            ctx.synchronize()
            ns_fused += perf_counter_ns() - t0

            var t1 = perf_counter_ns()
            identical_gemm_with_plan(ctx, dc, da, db, dw, m, n, k, OP_NT,
                                     PLAN_SPLITK_STAGED)
            ctx.synchronize()
            ns_staged += perf_counter_ns() - t1

        var ms_fused = Float64(ns_fused) / 1.0e6 / Float64(REPEATS)
        var ms_staged = Float64(ns_staged) / 1.0e6 / Float64(REPEATS)
        var tag = String("m2.P") + String(p)
        _report_device(tag + ".device.fused", ms_fused)
        _report_device(tag + ".device.staged", ms_staged)
        print(
            "     " + _padl(String(p), 5) + _padl(String(k), 9)
            + _padl(String(_launches_fused(p)) + "/"
                    + String(_launches_staged(p)), 10)
            + _padl(_fixed(ms_fused, 3), 11)
            + _padl(_fixed(ms_staged, 3), 12)
            + _padl(_fixed(_ratio(ms_staged, ms_fused), 2) + "x", 8)
            + _padl(_fixed(Float64(ws_fused * 4) / 1048576.0, 2), 11)
            + _padl(_fixed(Float64(ws_staged * 4) / 1048576.0, 2), 12)
        )
        _ = da
        _ = db
        _ = dc
        _ = dw
    print(
        "   launches is fused/staged, from the launcher and not from 13.6.2's"
        " arithmetic. st/fu above 1.00 is the staged arm costing more; the"
        " workspace columns are why it can cost more for a reason that is"
        " NOT the launches, since the staged plan also stages every interior"
        " node of the tree and the fused plan stages only level 0."
    )
    print(
        "   13.7: if the staged arm is slow the answer is to fuse levels into"
        " one block, which is legal and produces the same bits. It is NEVER"
        " to change the pairing."
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


def _host_pair_once(
    a: List[Float32],
    b: List[Float32],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
    mut ns_oracle: Int,
    mut ns_serial: Int,
    mut sink: UInt32,
) raises:
    """ONE timed oracle call and ONE timed serial call, in that order.

    Factored out for exactly one reason: the device arms cannot be declared
    inside an `if` in the same scope as the host arms (a `DeviceBuffer` needs
    a `DeviceContext` that the host arm does not have), so the timed loop
    exists in two shapes -- with device arms and without. **Both shapes must
    run the SAME host pair**, and two transcriptions of a timed region are two
    things that can drift apart while both keep printing. There is one here.
    """
    var t0 = perf_counter_ns()
    var c0 = gemm_oracle(a, b, op, m, n, k)
    ns_oracle += perf_counter_ns() - t0
    sink ^= _bits(c0[0])

    var t1 = perf_counter_ns()
    var c1 = gemm_oracle_serial(a, b, op, m, n, k)
    ns_serial += perf_counter_ns() - t1
    sink ^= _bits(c1[0])


def _timed_shape_with_device(
    ctx: DeviceContext,
    a: List[Float32],
    b: List[Float32],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
    dm: Int,
    dn: Int,
    name: String,
    idx: Int,
    mut ns_oracle: Int,
    mut ns_serial: Int,
    mut ns_device: Int,
    mut ns_vendor: Int,
    mut ns_pinned: Int,
    mut sink: UInt32,
    mut vendor_ok: Bool,
) raises -> Int:
    """One shape, FIVE arms, ONE timed loop, alternating call by call.

    Returns the plan `choose_gemm_plan` picked for the device shape, so the
    caller can print WHICH execution plan the number is a number for.

    THE FIVE ARMS, and 13.6.3 asks for three of them by name:

        oracle   host, `gemm_oracle`: the NORMATIVE v1 answer.
        serial   host, `gemm_oracle_serial`: the diagnostic whole-`k` chain.
        device   `identical_gemm_into`, the v1 kernel on the plan its own
                 dispatcher chose. **This is 13.6.3's IDENTICAL-against-FAST
                 arm**, and the two sides of that comparison are the two
                 BINARIES `tools/gemm_price.sh` alternates round by round --
                 `GLOBAL_NUMERIC_MODE` is comptime, so one process cannot hold
                 both and the median table is where they meet.
        vendor   `linalg.matmul` with `transpose_b=True`, called directly
                 rather than through `core/gemm.mojo::gemm_nt`, because that
                 wrapper is comptime-switched: under IDENTICAL it routes to
                 the pinned kernel, so calling it would give one arm in one
                 binary and a different arm in the other, under one name.
        pinned   `core/gemm.mojo::pinned_gemm_nt_kernel`, the SHIPPED pinned
                 product, launched directly for the same reason. It computes
                 `gemm_oracle_serial`'s chain on the device, so `pinned` is to
                 `device` what `serial` is to `oracle` -- the same pair of
                 questions asked on the other side of the bus.

    **NO ARM HERE ISOLATES THE COST OF IDENTITY, AND THE MISSING ONE IS
    NAMED SO NOBODY INFERS IT FROM `pinned`.** DEVIATION 1092, 2026-08-25.
    `device` against `pinned` is tempting to read as "identity against no
    identity" and it is not that. `pinned_gemm_nt_kernel` is FLAT -- one
    thread per output cell, a serial whole-`k` loop, no tiling and no shared
    memory -- while `device` picks among eight plans including tiled and
    split-K arms. Measured on the M4 at full llama8b shapes on 2026-08-25,
    `device` BEATS `pinned` by 1.6x to 1.8x at every `t512` row. That result
    says tiling beats not-tiling. It says nothing about the fold pin, and
    reading it as "identity is cheap" would be reading an engineering gap as
    a numerical one.

    `device` against `vendor` is not that experiment either: it is our
    kernel engineering plus the pin, against Modular's kernel engineering.
    Both terms move at once.

    The arm that answers it is `gemm/mojo_only/gemm_unpinned.mojo`, written
    2026-08-25 under DEVIATIONS 1130-1136: the identical kernel's OWN plan
    selection and tile constants, imported rather than copied, with the leaf
    loop kept AS A SCHEDULE so the staging window does not move, and only the
    fold-order pin removed. `gemm/UNPINNED_CONTROL.md` carries the
    clause-by-clause table of what was dropped, the confounds that remain,
    and a prediction written down before the run.

    (The sentence that stood here said the arm "does not exist anywhere in
    this repository". It did when it was written and it is false now, so it
    is deleted rather than left with a caveat.)

    IT IS NOT WIRED INTO THIS FILE YET, so no column below reports it, and
    until it is timed the cost of `mojolearn.identical.gemm.fp32.v1` as
    distinct from the cost of writing our own GEMM is still UNMEASURED. Do
    not take the nearest available ratio in the meantime.

    **THE LAST TWO ARE NT-ONLY AND ARE SKIPPED, NOT FAKED, ELSEWHERE.** Both
    read `A` as `m x k` and `B` as `n x k`; the shipped repository has no
    pinned TN product that is not the Gram special case (`gemm_tn`, which
    takes ONE operand and its transposes) and no NN product at all. Feeding a
    TN row's `k x m` operand to an `m x k` kernel would produce a plausible,
    in-bounds, wrong product and a perfectly good millisecond. So the four TN
    rows report `n/a` for these two arms and the count is printed.

    **AND THE VENDOR ARM IS SKIPPED AT `n == 1`, WHICH IS A DEFECT AND NOT A
    SHAPE.** `core/gemm.mojo::gemm_nt` records it, measured through that
    wrapper on 2026-08-19: at `m=64, n=1, k=32` with the output poisoned,
    **63 of the 64 rows still held the poison** -- `transpose_b=True` does not
    write there. The shipped route sends `n == 1` to `gemv_n` for that reason.
    Timing a call that does not write its output would be timing nothing, and
    `_dev_digest`'s poison check would refuse it anyway.
    """
    var dmn = dm * dn
    var plan = choose_gemm_plan(dm, dn, k)
    var nws = identical_gemm_workspace_floats(dm, dn, k, plan)
    if nws < 1:
        nws = 1
    var na = _a_elems(op, dm, dn, k)
    var nb = _b_elems(op, dm, dn, k)
    var do_vendor = op == OP_NT and dn > 1 and vendor_ok
    var do_pinned = op == OP_NT

    var da = ctx.enqueue_create_buffer[DType.float32](na if na > 0 else 1)
    var db = ctx.enqueue_create_buffer[DType.float32](nb if nb > 0 else 1)
    var dc = ctx.enqueue_create_buffer[DType.float32](dmn)
    var dw = ctx.enqueue_create_buffer[DType.float32](nws)
    ctx.synchronize()
    # The SAME salts the host arm uses at this shape index, so the two arms
    # read the same generator stream even though they run at different capped
    # sizes. Nothing depends on it today -- no arm here diffs another's bits --
    # and it costs nothing to keep the file usable as a fixture source.
    _dev_fill(ctx, da, na, 11 + idx)
    _dev_fill(ctx, db, nb, 22 + idx)

    # Untimed warm-up on every device arm, each poisoned first and read back,
    # so a kernel that launches without writing its output cannot turn in the
    # best time in the table. `[[reached-but-inert]]`.
    _dev_poison(ctx, dc, dmn)
    identical_gemm_into(ctx, dc, da, db, dw, dm, dn, k, op)
    ctx.synchronize()
    var dig = _dev_digest(ctx, dc, dmn, String("m3.") + name + ".device")
    sink ^= UInt32(Int(dig & UInt64(0xFFFFFFFF)))

    # DEVIATION 1093: THE VENDOR ARM'S WARM-UP IS ALLOWED TO CRASH, AND A CRASH
    # DISABLES THAT ARM FOR THE REST OF THE RUN RATHER THAN ENDING IT.
    #
    # Measured on an H100 80GB, 2026-08-25, leg 15: at `llama8b.lm_head.t1`
    # (`m = 1`, `n = 128256`, `k = 4096`) MAX's own `linalg.matmul` dispatches
    # into `max/kernels/src/linalg/gemv.mojo:1201` and the launch comes back
    # `CUDA_ERROR_INVALID_VALUE`. `llama8b.qkv.t1` is also `m = 1` and runs
    # fine, so the trigger is the very wide `n` in the gemv path and NOT `m ==
    # 1`; no formula is guessed here because the one measured point does not
    # determine one.
    #
    # Before this, that raise ended the process on shape 17 of 20 and
    # `tools/gemm_price.sh` DISCARDED BOTH LEGS -- nineteen shapes of good
    # measurement on a rented H100 thrown away because the twentieth crashed
    # inside a dependency. The shell was right to discard a failed leg; the
    # bug was that this file let one arm's failure become the run's.
    #
    # The arm is disabled for the REMAINDER OF THE RUN, not just this shape.
    # A CUDA launch failure can leave the context in a state where the next
    # launch fails for a reason that has nothing to do with the next shape,
    # and a vendor column that resumed after a crash would be reporting
    # numbers nobody can defend. A REFUSAL LINE IS PRINTED, so a missing
    # vendor cell is never mistaken for an agreeing one.
    if do_vendor:
        _dev_poison(ctx, dc, dmn)
        var wz = TileTensor(dc, row_major(dm, dn))
        var wx = TileTensor(da, row_major(dm, k))
        var wy = TileTensor(db, row_major(dn, k))
        try:
            matmul[transpose_b=True, target="gpu"](wz, wx, wy, ctx)
            ctx.synchronize()
        except e:
            print(
                "   VENDOR-ARM CRASHED at ",
                name,
                " (m=",
                dm,
                " n=",
                dn,
                " k=",
                k,
                "). The vendor column is REFUSED from here on, in this run",
                " and every later shape. This is MAX's matmul failing, not",
                " ours, and the v1 and pinned arms continue. Error: ",
                # `s[:n]` is refused on a Mojo String (UTF-8, so a byte range and
                # a codepoint range are different questions). Bytes is what a
                # CUDA message is.
                String(e)[byte=0:180],
                sep="",
            )
            do_vendor = False
            vendor_ok = False
        if do_vendor:
            var dv = _dev_digest(
                ctx, dc, dmn, String("m3.") + name + ".device.vendor"
            )
            sink ^= UInt32(Int(dv & UInt64(0xFFFFFFFF)))
    if do_pinned:
        _dev_poison(ctx, dc, dmn)
        ctx.enqueue_function[pinned_gemm_nt_kernel](
            dc.unsafe_ptr(),
            da.unsafe_ptr(),
            db.unsafe_ptr(),
            Int32(dm),
            Int32(dn),
            Int32(k),
            grid_dim=((dmn + PINNED_GEMM_TPB - 1) // PINNED_GEMM_TPB, 1, 1),
            block_dim=(PINNED_GEMM_TPB, 1, 1),
        )
        ctx.synchronize()
        var dp = _dev_digest(
            ctx, dc, dmn, String("m3.") + name + ".device.pinned"
        )
        sink ^= UInt32(Int(dp & UInt64(0xFFFFFFFF)))

    for _ in range(REPEATS):
        _host_pair_once(a, b, op, m, n, k, ns_oracle, ns_serial, sink)

        var t2 = perf_counter_ns()
        identical_gemm_into(ctx, dc, da, db, dw, dm, dn, k, op)
        ctx.synchronize()
        ns_device += perf_counter_ns() - t2

        if do_vendor:
            var t3 = perf_counter_ns()
            var tz = TileTensor(dc, row_major(dm, dn))
            var tx = TileTensor(da, row_major(dm, k))
            var ty = TileTensor(db, row_major(dn, k))
            matmul[transpose_b=True, target="gpu"](tz, tx, ty, ctx)
            ctx.synchronize()
            ns_vendor += perf_counter_ns() - t3

        if do_pinned:
            var t4 = perf_counter_ns()
            ctx.enqueue_function[pinned_gemm_nt_kernel](
                dc.unsafe_ptr(),
                da.unsafe_ptr(),
                db.unsafe_ptr(),
                Int32(dm),
                Int32(dn),
                Int32(k),
                grid_dim=(
                    (dmn + PINNED_GEMM_TPB - 1) // PINNED_GEMM_TPB, 1, 1
                ),
                block_dim=(PINNED_GEMM_TPB, 1, 1),
            )
            ctx.synchronize()
            ns_pinned += perf_counter_ns() - t4

    # `[[mojo-buffer-freed-at-last-use]]`: the four owners live to here.
    _ = da
    _ = db
    _ = dc
    _ = dw
    return plan


def measure_end_to_end(
    budget: Int,
    dev_budget: Int,
    devs: List[DeviceContext],
    mut dev_ms: List[Float64],
    mut dev_m: List[Int],
    mut dev_n: List[Int],
) raises:
    """13.6.3 over `bench/gemm_shapes.mojo`. DEVIATION 535 added the device
    arms; the host pair is unchanged.

    13.6.3 asks for three arms -- IDENTICAL, FAST, and the shipped pinned
    kernel -- over the charter's Phase 4 shape list, and on the `device` arm
    **all three are present**:

        the v1 kernel in THIS binary's mode, as `m3.<shape>.device`, which
        `tools/gemm_price.sh` runs in both modes and pairs in its median
        table: that pair IS "IDENTICAL against FAST";
        `m3.<shape>.device.vendor`, `linalg.matmul`, which is what the
        shipped `gemm_nt` runs under FAST;
        `m3.<shape>.device.pinned`, the shipped pinned product.

    The host pair stays and is not a lesser version of them: `gemm_oracle` is
    the NORMATIVE definition of the answer and `gemm_oracle_serial` is the
    diagnostic reference, and `oracle / serial` is the same question `device /
    pinned` asks, on the other side of the bus.

    **THE HOST ARMS AND THE DEVICE ARMS RUN AT DIFFERENT CAPPED SHAPES AND
    EVERY ROW PRINTS BOTH.** `m` and `n` cap to each arm's own MAC budget and
    `k` IS EXACT FOR BOTH, so `L`, `P`, the ragged tail and the tree are the
    full shape's on both sides (contract 6.1 forbids the leaf rule to read `m`
    or `n`). A single budget would either drown the device arm in its own
    launch overhead or take the host arm into minutes per row. The columns say
    which shape each column was measured at; nothing is derived across them.

    Every row of the table runs. Nothing is dropped for being slow, being
    unflattering or being uninteresting -- a benchmark that chooses its rows
    has stopped measuring the workload and started measuring the choice. Rows
    that cannot fit the budget even at `m = n = 1` are REPORTED and counted,
    never silently skipped, because a missing row looks exactly like an
    agreeing one.

    The arms ALTERNATE CALL BY CALL, not block by block: they all live in this
    binary, so the thermal drift the shell has to work around at the mode
    level is defeated outright at the arm level.

    `dev_ms`, `dev_m` and `dev_n` come back with one entry per shape -- the
    device time and the shape it was measured at, `0` where no device arm ran
    -- and 13.6.6 turns them into its limiter column. That hand-off is why
    they are out-parameters rather than a print: 13.6.6 must not re-derive a
    time, and a second measurement of the same thing is a second number that
    can disagree.
    """
    var on_device = len(devs) == 1
    print()
    print("-- 13.6.3  end to end over bench/gemm_shapes.mojo --")
    if on_device:
        print(
            "   FIVE ARMS. host: oracle (normative v1) and serial"
            " (diagnostic whole-k chain). device: the v1 kernel, the FAST"
            " vendor matmul, and the shipped pinned kernel -- 13.6.3's three."
            " DEVICE TIMES ARE INDICATIVE (one thermally unstable M4; timing"
            " is the identity lane's to certify)."
        )
    else:
        print(
            "   HOST ARMS ONLY: 'oracle' is the normative v1 answer, 'serial'"
            " is the diagnostic whole-k chain. The FAST vendor arm and the"
            " shipped pinned kernel are DEVICE arms; run"
            " MOJOLEARN_GEMM_PRICE_ARM=device for them."
        )
    print(
        "   m and n are CAPPED to a",
        budget,
        "MAC budget on the host arms",
        "and to a " + String(dev_budget) + " MAC budget on the device arms;"
        " k is EXACT on both, so L, P, the ragged tail and the tree are the"
        " full shape's. Per-cell price is measured; full-shape host cost is"
        " DERIVED by multiplication (contract 0.3: cells are independent)."
    )

    print(
        "   " + _padr("name", 32) + " op" + _padl("m", 9) + _padl("n", 10)
        + _padl("k", 10) + _padl("L", 9) + _padl("P", 8) + _padl("tail", 8)
        + _padl("oracle ms", 11) + _padl("serial ms", 11)
        + _padl("o/s", 9) + _padl("DERIVED ms", 14)
    )
    var ran = 0
    var skipped = 0
    var dev_ran = 0
    var dev_skipped = 0
    var nt_only_skipped = 0
    var total_sink = UInt32(0)
    var dev_rows = List[String]()
    # DEVIATION 1093's run-scoped flag. Declared OUTSIDE the shape loop on
    # purpose: a vendor-arm crash disables that arm for every LATER shape
    # too, because a failed CUDA launch can leave the context in a state
    # where the next launch fails for an unrelated reason.
    var vendor_ok = True
    for i in range(GEMM_SHAPE_COUNT):
        # Appended FIRST and for EVERY shape, before any `continue`, so the
        # three out-lists stay index-aligned with the shape table. A list that
        # is a shape short is a list where every row after the gap is a
        # different shape's number under this shape's name.
        dev_ms.append(0.0)
        dev_m.append(0)
        dev_n.append(0)

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

        var dmn = _capped(m0, n0, k, dev_budget)
        var dm = dmn[0]
        var dn = dmn[1]
        var dev_ok = on_device and dm * dn * k <= dev_budget

        var a = _fill(_a_elems(op, m, n, k), 11 + i)
        var b = _fill(_b_elems(op, m, n, k), 22 + i)

        var ns_oracle = 0
        var ns_serial = 0
        var ns_device = 0
        var ns_vendor = 0
        var ns_pinned = 0

        # Untimed warm-up on the host pair before either clock starts, and
        # the sink is SEEDED from it rather than from a zero: a sink assigned
        # a constant and then overwritten is a value the compiler is entitled
        # to drop, which is the one thing a sink exists to prevent.
        var wa = gemm_oracle(a, b, op, m, n, k)
        var wb = gemm_oracle_serial(a, b, op, m, n, k)
        var sink = _bits(wa[0]) ^ _bits(wb[0])

        var plan = -1
        if dev_ok:
            plan = _timed_shape_with_device(
                devs[0], a, b, op, m, n, k, dm, dn, name, i,
                ns_oracle, ns_serial, ns_device, ns_vendor, ns_pinned, sink,
                vendor_ok,
            )
        else:
            for _ in range(REPEATS):
                _host_pair_once(
                    a, b, op, m, n, k, ns_oracle, ns_serial, sink
                )

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

        if not on_device:
            continue
        if not dev_ok:
            print(
                "   DEV SKIP",
                name,
                "-- k =",
                k,
                "alone exceeds the device budget of",
                dev_budget,
                "MACs and k may not be capped. Raise",
                "MOJOLEARN_GEMM_PRICE_DEV_MAC_BUDGET to include it.",
            )
            dev_skipped += 1
            continue

        var ms_device = Float64(ns_device) / 1.0e6 / Float64(REPEATS)
        _report_device(String("m3.") + name + ".device", ms_device)
        dev_ms[i] = ms_device
        dev_m[i] = dm
        dev_n[i] = dn
        dev_ran += 1

        var vcell = String("n/a")
        var pcell = String("n/a")
        if ns_vendor > 0:
            var ms_vendor = Float64(ns_vendor) / 1.0e6 / Float64(REPEATS)
            _report_device(
                String("m3.") + name + ".device.vendor", ms_vendor
            )
            vcell = _fixed(ms_vendor, 3)
        if ns_pinned > 0:
            var ms_pinned = Float64(ns_pinned) / 1.0e6 / Float64(REPEATS)
            _report_device(
                String("m3.") + name + ".device.pinned", ms_pinned
            )
            pcell = _fixed(ms_pinned, 3)
        if op != OP_NT:
            nt_only_skipped += 1

        var flops = 2.0 * Float64(dm) * Float64(dn) * Float64(k)
        var row = String("   DEV ") + _padr(name, 30) + " " + op_name(op)
        row = row + _padl(String(dm) + "x" + String(dn), 12)
        row = row + _padl(String(k), 10)
        row = row + _padl(_fixed(ms_device, 3), 11)
        row = row + _padl(vcell, 11)
        row = row + _padl(pcell, 11)
        row = row + _padl(_fixed(_gfps(flops, ms_device), 2), 11)
        # THE PLAN NAME GOES LAST, and it is the launcher's own sentence
        # rather than an abbreviation invented here. `gemm_plan_name` returns
        # a full description ("SPLITK(leaf kernel -> global workspace ->
        # level-wise threadgroup fold)"), which is far wider than any column,
        # and clipping it would produce a SECOND, shorter name for a plan --
        # a second spelling of an identifier that has to match the kernel's.
        # Trailing and unpadded, every numeric column above stays aligned and
        # the name stays whole.
        row = row + "  " + gemm_plan_name(plan)
        dev_rows.append(row)

    print("   ran", ran, "shapes,", skipped, "over budget; sink",
          total_sink)
    print(
        "   The DERIVED column is arithmetic on the measured per-cell price,"
        " NOT a measurement, and it is a HOST cost. It is printed because a"
        " reader comparing 'oracle 3 ms' across rows whose m n differ by a"
        " factor of 10^5 would otherwise compare nothing."
    )

    if not on_device:
        return
    print()
    print("-- 13.6.3  the DEVICE arms (INDICATIVE, one M4, not certified) --")
    print(
        "   " + _padr("   name", 34) + " op" + _padl("dev m x n", 12)
        + _padl("k", 10) + _padl("v1 ms", 11)
        + _padl("vendor ms", 11) + _padl("pinned ms", 11)
        + _padl("v1 GF/s", 11) + "  plan (full name, trailing: it is wider"
        " than any column and is the launcher's own)"
    )
    for r in range(len(dev_rows)):
        print(dev_rows[r])
    print(
        "   ran", dev_ran, "device shapes,", dev_skipped, "over the device"
        " budget,", nt_only_skipped, "with n/a in the vendor and pinned"
        " columns because those two arms are NT-ONLY (the repository ships no"
        " pinned TN product outside the Gram special case and no NN product"
        " at all). n/a is a REFUSAL, not a zero."
    )
    print(
        "   GF/s is achieved, at the DEVICE shape in the column beside it,"
        " for the v1 kernel only. No host arm in this file prints this"
        " column and none is going to."
    )
    print(
        "   IDENTICAL against FAST -- 13.6.3's first two arms -- is the"
        " m3.<shape>.device row of tools/gemm_price.sh's median table, not a"
        " column here: GLOBAL_NUMERIC_MODE is comptime, so the two modes are"
        " two BINARIES and one process cannot hold both."
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


def check_limiter_rates_use_the_MEASURED_shape(
    dev_ms: List[Float64], dev_m: List[Int], dev_n: List[Int]
) raises:
    """The rate and the time must be about the SAME shape. DEVIATION 535.

    13.6.6's input table is computed at the FULL `m` and `n`, because those
    are the workload's bytes and the workload's flops. The device time is
    measured at the CAPPED `m` and `n` (`MOJOLEARN_GEMM_PRICE_DEV_MAC_BUDGET`).
    **Dividing one by the other is the single easiest way to publish a
    fabricated GFLOP/s figure in this file**, and it would not look wrong: at
    `gram.32x32x1M` the cap is a factor of 64 in `n`, so a full-shape flop
    count over a capped-shape millisecond would report 64x the achieved rate,
    smoothly, on every row, with no assertion anywhere to catch it.

    So the rates below are computed from `dev_m` and `dev_n` -- the shape the
    clock ran at -- and this gate asserts that those are real: present for
    every measured row, positive, and never larger than the table's own `m`
    and `n`. A row with a time and no shape is refused rather than defaulted,
    because the default would be the table's shape and that is exactly the
    fabrication.
    """
    if len(dev_ms) != len(dev_m) or len(dev_ms) != len(dev_n):
        raise Error(
            "13.6.6: the measured-time lists are ragged ("
            + String(len(dev_ms)) + "/" + String(len(dev_m)) + "/"
            + String(len(dev_n))
            + "), so a row's time and a row's shape may not be the same row's."
        )
    if len(dev_ms) != 0 and len(dev_ms) != GEMM_SHAPE_COUNT:
        raise Error(
            "13.6.6: got "
            + String(len(dev_ms))
            + " measured rows for a table of "
            + String(GEMM_SHAPE_COUNT)
            + " shapes. The lists are indexed BY SHAPE INDEX; a short list is"
            " one where every row after the gap carries another shape's"
            " number under this shape's name."
        )
    for i in range(len(dev_ms)):
        if dev_ms[i] <= 0.0:
            continue
        if dev_m[i] <= 0 or dev_n[i] <= 0:
            raise Error(
                "13.6.6: shape '"
                + gemm_shape_name(i)
                + "' has a measured time and no measured shape. Its rate"
                " would be computed against the FULL m and n, which is a"
                " fabricated GFLOP/s."
            )
        if dev_m[i] > gemm_shape_m(i) or dev_n[i] > gemm_shape_n(i):
            raise Error(
                "13.6.6: shape '"
                + gemm_shape_name(i)
                + "' reports a measured shape "
                + String(dev_m[i]) + "x" + String(dev_n[i])
                + " larger than the table's "
                + String(gemm_shape_m(i)) + "x" + String(gemm_shape_n(i))
                + ". m and n may only be CAPPED, never grown."
            )


def report_limiter_inputs(
    dev_ms: List[Float64], dev_m: List[Int], dev_n: List[Int]
) raises:
    """13.6.6: which resource limits each shape.

    13.6.6 asks which of compute, bandwidth, staging or the fold limits EACH
    shape. **A host cannot answer that**, and on the `host` arm this still
    prints every INPUT to the classification and no verdict, which is what it
    always did:

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

    **ON THE `device` ARM THE COLUMN IS FILLED, AND IT IS FILLED WITH A
    RELATIVE VERDICT RATHER THAN A ROOFLINE.** DEVIATION 535 hands 13.6.3's
    measured device time in here (out-parameters, so no time is measured
    twice and no two numbers can disagree) and it becomes an achieved
    GFLOP/s and an achieved compulsory-traffic GB/s. What it does NOT become
    is a peak-relative classification, because:

      - this file has no vendor peak figure for this box and is not going to
        invent one. A limiter named against a made-up peak is a made-up
        limiter;
      - naming a limiter properly needs DEVICE COUNTERS -- achieved bandwidth
        against peak, occupancy, and a kernel timeline that separates the leaf
        stage from the fold. None of those are in this file.

    So the verdict is stated against the best rate THIS RUN OBSERVED, which is
    a LOWER BOUND on peak and is labelled as one on every line that uses it. A
    row at or above `LIMITER_SHARE` of the run's best GFLOP/s is compute-ish;
    otherwise, at or above `LIMITER_SHARE` of the run's best GB/s is
    bandwidth-ish; otherwise the row is neither, which at these sizes means
    latency and launch. The threshold is a REPORTING CHOICE, printed with the
    verdict so it travels, and it is not a bound anyone derived.

    **THE FOLD IS EXCLUDED AS AN ARITHMETIC LIMITER BY 13.3 AND BY A GATE**,
    not by a verdict: `check_fold_fraction_is_under_the_13_3_bound` asserts
    `(P-1)/k < 1/L` on every row, so no row can be fold-limited by arithmetic.
    13.3's other channel -- the fold's EXPOSED LATENCY at small `m n` -- is
    not excluded by anything here, and the rows where it could bite are named
    below rather than classified.
    """
    check_limiter_rates_use_the_MEASURED_shape(dev_ms, dev_m, dev_n)
    # **"MEASURED" MEANS A TIME EXISTS, NOT THAT A LIST HAS THE RIGHT
    # LENGTH.** The host arm fills these lists too -- with zeros, one slot per
    # shape, so the indices stay aligned with the shape table -- and a length
    # test would read that as twenty measurements and print a verdict column
    # of refusals where the honest output is the inputs table.
    var measured = False
    for i in range(len(dev_ms)):
        if dev_ms[i] > 0.0:
            measured = True
    print()
    if measured:
        print("-- 13.6.6  limiter inputs and a RELATIVE verdict per shape --")
    else:
        print(
            "-- 13.6.6  limiter inputs per shape (limiter itself UNMEASURED)"
            " --"
        )
    print(
        "   " + _padr("name", 32) + _padl("MFLOP", 12)
        + _padl("operand MiB", 13) + _padl("flop/byte", 11)
        + _padl("staging MiB", 13) + _padl("fold %", 9) + "  limiter"
    )

    # The two ceilings, computed BEFORE the table so the verdict column can
    # use them. They are the best rates this run reached, over the rows this
    # run measured, at the shapes it measured them at. Lower bounds on peak,
    # never peak.
    var best_gf = 0.0
    var best_gbs = 0.0
    if measured:
        for i in range(GEMM_SHAPE_COUNT):
            if dev_ms[i] <= 0.0:
                continue
            var dk = gemm_shape_k(i)
            var op = _tbl_op(i)
            var f = 2.0 * Float64(dev_m[i]) * Float64(dev_n[i]) * Float64(dk)
            var ob = 4.0 * (
                Float64(_a_elems(op, dev_m[i], dev_n[i], dk))
                + Float64(_b_elems(op, dev_m[i], dev_n[i], dk))
                + Float64(dev_m[i]) * Float64(dev_n[i])
            )
            var gf = _gfps(f, dev_ms[i])
            var gbs = _ratio(ob, dev_ms[i] * 1.0e6)
            if gf > best_gf:
                best_gf = gf
            if gbs > best_gbs:
                best_gbs = gbs

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

        var verdict = String("UNMEASURED (needs the device arm)")
        if measured and dev_ms[i] > 0.0:
            var dmm = dev_m[i]
            var dnn = dev_n[i]
            var dflops = 2.0 * Float64(dmm) * Float64(dnn) * Float64(k)
            var dob = 4.0 * (
                Float64(_a_elems(op, dmm, dnn, k))
                + Float64(_b_elems(op, dmm, dnn, k))
                + Float64(dmm) * Float64(dnn)
            )
            var gf = _gfps(dflops, dev_ms[i])
            var gbs = _ratio(dob, dev_ms[i] * 1.0e6)
            if gf >= LIMITER_SHARE * best_gf and best_gf > 0.0:
                verdict = String("COMPUTE-ish")
            elif gbs >= LIMITER_SHARE * best_gbs and best_gbs > 0.0:
                verdict = String("BANDWIDTH-ish")
            else:
                verdict = String("NEITHER: latency/launch at this size")
            verdict = (
                verdict + " [" + String(dmm) + "x" + String(dnn) + " "
                + _fixed(gf, 1) + " GF/s " + _fixed(gbs, 1) + " GB/s]"
            )
        elif measured:
            verdict = String("NOT RUN on the device (over the device budget)")

        print(
            "   " + _padr(gemm_shape_name(i), 32)
            + _padl(_fixed(flops / 1.0e6, 1), 12)
            + _padl(_fixed(obytes / 1048576.0, 2), 13)
            + _padl(_fixed(flops / obytes, 2), 11)
            + _padl(_fixed(staging / 1048576.0, 2), 13)
            + _padl(_fixed(100.0 * Float64(p - 1) / Float64(k), 4), 9)
            + "  " + verdict
        )
    check_fold_fraction_is_under_the_13_3_bound()
    print(
        "   That bound is on the fold's ARITHMETIC share and says nothing"
        " about its EXPOSED LATENCY at small m n, which is the shape 13.3"
        " names as the one where the fold can still matter -- and four rows"
        " above have m n small enough to be exactly that shape."
    )
    if not measured:
        return
    print(
        "   THE VERDICT IS RELATIVE AND THE CEILINGS ARE THIS RUN'S, NOT THE"
        " DEVICE'S. Best observed:",
        _show(best_gf),
        "GF/s and",
        _show(best_gbs),
        "GB/s (compulsory operand traffic only).",
    )
    print(
        "   Those are LOWER BOUNDS on peak: every row here may be far from"
        " the hardware, and a row called COMPUTE-ish is compute-ish next to"
        " the best this run reached and next to nothing else. A"
        " peak-relative classification needs device counters and a vendor"
        " peak figure, and this file has neither and will not invent them.",
    )
    var share = Float64(LIMITER_SHARE)
    print(
        "   Threshold: a row is called after the first of GF/s or GB/s that"
        " reaches",
        _show(share),
        "of the corresponding best. That is a REPORTING CHOICE printed so it"
        " travels with the column, not a bound anybody derived.",
    )
    print(
        "   The bracket on each row is the shape the clock actually ran at,"
        " which is the CAPPED device shape and not the full m n in the"
        " columns to its left. check_limiter_rates_use_the_MEASURED_shape"
        " refuses a row that has one without the other."
    )


# ===========================================================================
# DRIVER
# ===========================================================================


def _wants(only: String, tag: String) -> Bool:
    if only == "" or only == "all":
        return True
    return only.find(tag) >= 0


def _run_measurements(
    devs: List[DeviceContext],
    budget: Int,
    dev_budget: Int,
    ws_mb: Int,
    only: String,
) raises:
    """The six measurements, in order, for whichever arm `devs` describes.

    ONE body for both arms. `devs` is empty on the `host` arm and holds the
    one open `DeviceContext` on the `device` arm; every measurement that has a
    device half takes it and decides for itself. A `List` rather than an
    optional because a `DeviceBuffer` cannot be declared without a context, so
    "is there a device" has to be answerable before any buffer exists.

    **13.6.6 READS 13.6.3'S TIMES AND DOES NOT MEASURE ITS OWN.** The three
    lists below are how: `measure_end_to_end` fills them, `report_limiter_
    inputs` classifies from them. If `MOJOLEARN_GEMM_PRICE_ONLY` excluded m3
    they stay empty and 13.6.6 prints UNMEASURED on every row -- which is
    correct and is the point of the hand-off. A 13.6.6 that measured its own
    times would be a second number for the same quantity, and two numbers for
    one quantity is one number and one thing to explain away.
    """
    var dev_ms = List[Float64]()
    var dev_m = List[Int]()
    var dev_n = List[Int]()

    if _wants(only, "m1"):
        measure_fold_in_isolation(devs)
    if _wants(only, "m2"):
        report_staging_structure(devs)
    if _wants(only, "m3"):
        measure_end_to_end(budget, dev_budget, devs, dev_ms, dev_m, dev_n)
    if _wants(only, "m4"):
        measure_leaf_loop_seams()
    if _wants(only, "m5"):
        report_workspace(ws_mb)
    if _wants(only, "m6"):
        report_limiter_inputs(dev_ms, dev_m, dev_n)


def main() raises:
    var arm = String(getenv("MOJOLEARN_GEMM_PRICE_ARM"))
    if arm == "":
        arm = String("host")

    print("== bench/gemm_price_main.mojo [" + _mode() + "] ==")
    print("profile mojolearn.identical.gemm.fp32.v1")
    print("arm", arm)

    if arm != "host" and arm != "device":
        raise Error(
            "bench/gemm_price_main: unknown MOJOLEARN_GEMM_PRICE_ARM '"
            + arm
            + "'. Known: host (default), device."
        )

    var budget = DEFAULT_MAC_BUDGET
    var bs = String(getenv("MOJOLEARN_GEMM_PRICE_MAC_BUDGET"))
    if bs != "":
        budget = Int(atol(bs))
    var dev_budget = DEFAULT_DEV_MAC_BUDGET
    var ds = String(getenv("MOJOLEARN_GEMM_PRICE_DEV_MAC_BUDGET"))
    if ds != "":
        dev_budget = Int(atol(ds))
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
    if arm == "device" and dev_budget < kmax:
        # THE SAME REFUSAL FOR THE DEVICE BUDGET, and for the same reason.
        # A device budget below max k drops gram.32x32x1M from the device
        # arm alone, which would leave the flagship Gram shape priced on the
        # host and unpriced on the device -- and it is the row with the
        # profile's only real P = 1024 fold.
        raise Error(
            "bench/gemm_price_main: the DEVICE MAC budget "
            + String(dev_budget)
            + " is below the shape table's largest k, "
            + String(kmax)
            + ". Same rule as the host budget: m and n cap to 1, k never"
            " caps, so this would silently drop the largest-k row from the"
            " device arm. Raise MOJOLEARN_GEMM_PRICE_DEV_MAC_BUDGET."
        )

    var ws_mb = DEFAULT_WS_BUDGET_MB
    var ws = String(getenv("MOJOLEARN_GEMM_PRICE_WS_BUDGET_MB"))
    if ws != "":
        ws_mb = Int(atol(ws))

    var only = String(getenv("MOJOLEARN_GEMM_PRICE_ONLY"))

    if arm == "device":
        print(
            "THIS ARM IS A GPU LEG (contract 13.5, assumption A5), in BOTH",
            "directions: it is contaminated by any other lane's GPU work on",
            "this box AND it contaminates theirs. tools/gemm_price.sh looks",
            "for a concurrent mojo/pixi process at both ends and reports",
            "what it found; it does not refuse, so read that line.",
        )
        print(
            "EVERY DEVICE NUMBER BELOW IS INDICATIVE. The M4's governor pins",
            "at MINIMUM clock under heat -- 96% of an 11-second trace while",
            "93.8% busy -- and drifts up to 1.7x across twenty minutes.",
            "Timing is the identity lane's to certify and nothing here is a",
            "certification.",
        )
    else:
        print(
            "NOT TRUSTWORTHY BESIDE A GPU LEG (contract 13.5, assumption A5).",
            "These arms are host-only, so they contend for CPU rather than",
            "for the GPU -- which makes them LESS fragile than a device",
            "timing, not immune. tools/gemm_price.sh checks and says so.",
        )
    print()

    # Gates first. Every one of them guards a number printed below, and a
    # harness that measured before it checked would publish the wrong number
    # and then check. All five run in BOTH arms: they are arithmetic on the
    # contract, the shape table and the kernel's own host-side functions, and
    # none of them needs a device. A gate that ran only in the arm it was
    # written for is a gate nobody runs.
    check_op_encodings_are_not_interchangeable()
    check_device_op_encoding_matches_the_kernel()
    check_fold_sweep_k_realizes_the_p_sweep()
    check_serial_fold_probe_is_the_other_topology()
    check_workspace_formula_matches_contract_13_5()

    if arm == "device":
        # STEP 1 OF THE WIRING: one `DeviceContext`, opened here and passed
        # down, so every device arm in the run shares one context and one
        # allocator. `bench/linalg_price_main.mojo` opens its own the same
        # way.
        with DeviceContext() as ctx:
            var devs = List[DeviceContext]()
            devs.append(ctx)
            _run_measurements(devs, budget, dev_budget, ws_mb, only)
    else:
        var devs = List[DeviceContext]()
        _run_measurements(devs, budget, dev_budget, ws_mb, only)

    print()
    print("== done [" + _mode() + "] ==")
