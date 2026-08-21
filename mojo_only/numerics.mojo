"""What may vary between backends, and what may not.

NOT a port. CatBoost has no equivalent and does not need one: it ships one
GPU backend and accepts a non-deterministic answer. We ship Metal, CUDA and
HIP from one source, so "the same fit gives the same model" is a property we
have to design for rather than inherit.

THE AXIS, and it is the thing that is easy to get wrong
-------------------------------------------------------
The tempting split is "numeric rows" against "scheduling rows", vary the
scheduling per vendor and pin the numerics. **That split is false**, because
some scheduling decisions ARE numeric decisions. A block count is a summation
order. A partial-sum tree shape is a summation order. How many private copies
a histogram is replicated into decides how many partial sums get combined and
in what sequence.

So a row is classified by ONE question:

    Does it change the SEQUENCE or the PRECISION of the arithmetic?

If yes it is a NUMERIC row and `NumericMode.IDENTICAL` pins it, whatever it
looks like. If no it is a SCHEDULING row and every backend picks freely, in
both modes. Grid shape, threads per block, occupancy targets, launch batching
and prefetch distance are scheduling. Accumulator width, flush order,
replication factor, block count and contraction are numeric.

WHAT A MODE CANNOT DO
---------------------
Two things escape the table and must be handled in source rather than
configured:

- **Floating-point atomics are order-nondeterministic run to run**, not just
  backend to backend. No row pins them. `IDENTICAL` has to REPLACE the
  accumulator, not configure it.
- **FMA contraction is a codegen decision.** The compiler decides whether
  `a*b+c` becomes one rounding or two. A runtime row cannot reach it; it is
  fixed at the source, per kernel.

THE COST IS A MEASUREMENT, NOT AN ARGUMENT
------------------------------------------
Because the modes differ by a named, enumerable set of rows, "what does
determinism cost" has an answer in seconds rather than in opinions. Run the
two modes interleaved and report the ratio.
"""


#: **THE DEFAULT.** Every row is read from the device's own column. Full
#: per-vendor speed, and histograms flush through floating-point atomics.
#:
#: **What that costs, stated because a user must not discover it:** float
#: atomics have no ordering guarantee, so the last bits move between two runs
#: of the SAME fit on the SAME device, not only between vendors. Model files
#: will not be byte-comparable, a fit is not reproducible from its seed
#: alone, and a regression test cannot assert on exact predictions. This is
#: CatBoost's shipped behavior, so it is a defensible default rather than a
#: novel one.
#: **THE SWITCH. One line, the whole build.**
#:
#: It did not exist until 2026-08-21 and the mode was therefore NOT REACHABLE:
#: five kernel files each declared their own `comptime BUILD_MODE =
#: NUMERIC_FAST`, so selecting `IDENTICAL` meant editing five files and
#: knowing which five. A toggle a user cannot flip is a toggle that does not
#: exist, and every claim made about "the IDENTICAL mode" was a claim about a
#: configuration nobody had ever built.
#:
#: Found by Andrew asking whether the toggle actually works. It did not.
#:
#: Every site that used to hardcode `NUMERIC_FAST` now reads this, so the
#: default is unchanged bit for bit and flipping this one line rebuilds the
#: whole tree in the other mode. Comptime, because the two modes are
#: different code (a float atomic and a fixed-point accumulator are not one
#: configured value), which is the same reason `TARGET_COLUMN` is a build
#: rather than a flag.
comptime GLOBAL_NUMERIC_MODE = NUMERIC_FAST


comptime NUMERIC_FAST = 0

#: Opt in when reproducibility is needed: numeric rows are read from
#: `COLUMN_BIT_IDENTICAL` instead of the device's column, so the same fit
#: gives the same model on Apple, NVIDIA and AMD and on repeated runs.
#:
#: The cost is a measurement, not an argument: the two modes differ by a
#: named set of rows, so run them interleaved and report the ratio.
comptime NUMERIC_IDENTICAL = 1


@fieldwise_init
struct NumericMode(Copyable, Movable):
    """The mode, plus the resolved value of every numeric row.

    The rows are resolved ONCE, here, rather than each kernel calling
    `getenv` and reaching its own conclusion. A kernel that decides for itself
    is how two kernels come to disagree about which mode they are in.
    """

    var mode: Int

    @staticmethod
    def default() -> Self:
        """`FAST`. Per-vendor speed everywhere, and results that move in the
        last bits run to run. See `NUMERIC_FAST` for the full consequence."""
        return Self(NUMERIC_FAST)

    def deterministic_flush(self) -> Bool:
        """NUMERIC ROW. Whether the shared histogram flushes to global memory
        through a fixed-point integer accumulator instead of `atomicAdd` on
        `float`.

        Integer addition is associative, so a fixed-point flush is
        order-independent and therefore reproducible; a float atomic is not,
        and no ordering guarantee is available to make it so. This is the row
        that cannot be a toggle over one implementation: the two modes run
        different code.
        """
        return self.mode == NUMERIC_IDENTICAL

    def fixed_replication(self) -> Bool:
        """NUMERIC ROW, and it LOOKS like a scheduling row, which is the whole
        reason this file exists.

        How many private copies the shared histogram is replicated into
        decides how many partial sums are combined and in what order. Under
        `IDENTICAL` the replication factor is pinned to a value every backend
        can meet, so the reduction tree has the same shape everywhere. Under
        `FAST` each backend picks the factor its threadgroup budget allows,
        and Apple's 32 KB cap already forces a different one from CUDA's 48 KB.
        """
        return self.mode == NUMERIC_IDENTICAL

    def free_block_shape(self) -> Bool:
        """SCHEDULING ROW. Threads per block, grid shape, how many blocks are
        launched per partition to fill the machine.

        Free in BOTH modes. It changes which thread does which work and not
        what is added to what, PROVIDED the replication factor and the
        reduction width are pinned separately, which is what
        `fixed_replication` is for. Splitting these two apart is the point of
        the classification: they are usually chosen together and only one of
        them is numeric.
        """
        return True


# ============ THE TWO SOURCE-LEVEL CONSTRUCTIONS ==========================
# The header above names two things no mode row can reach: float atomics
# (REPLACED by the fixed-point accumulator elsewhere) and the pair below,
# which exist so that IDENTICAL's arithmetic is ONE arithmetic on every
# backend. Both are comptime no-ops under FAST.


def ftz(x: Float32) -> Float32:
    """IDENTITY_PATHS row 10's construction: the denormal policy.

    MEASURED, NOT DESIGNED. `check-ieee-arith` (2^20 hashed patterns)
    found Metal through MAX correctly rounded on every normal input with
    FLUSH-TO-ZERO on denormal operands, intermediates and results -- and
    this exact model (flush operands to SIGNED zero, correctly-rounded
    op, flush result) reproduced ALL 53,041 observed divergences bit for
    bit. CUDA's default honors denormals, so without this the same fit
    diverges across vendors on any pathway a denormal can reach.

    Under IDENTICAL, apply at every float SEAM a kernel writes for
    another kernel or the host to read (the row-10 checklist in
    IDENTITY_PATHS.md). On an FTZ backend the flush is bitwise a no-op
    -- flushing what hardware already flushed -- so Apple's IDENTICAL
    bits do not move; on a denormal-honoring backend it aligns them to
    the FTZ ones. Under FAST it compiles away entirely.

    Intermediates INSIDE an expression cannot be reached this way on a
    non-FTZ backend; row 10's checklist therefore also requires that
    pinned expressions be written with their intermediates stored
    through `ftz` (one extra local per step), which is exactly how the
    check's model computes.
    """
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        if abs(x) < Float32(1.1754943508222875e-38) and x != Float32(0.0):
            if x < Float32(0.0):
                return Float32(-0.0)
            return Float32(0.0)
    return x


def identical_mul_add(a: Float32, b: Float32, c: Float32) -> Float32:
    """IDENTITY_PATHS row 9's construction: the contraction pin.

    `a*b + c` is one rounding or two at the CODEGEN'S whim, and the whim
    differs: Metal measured UNFUSED on 2^20 patterns (`check-ieee-arith`,
    fused 0 / unfused 1,046,394), CUDA's compiler contracts by default,
    and Mojo has been seen contracting ACROSS expressions where clang
    would not (the mojotrees plateau-tie incident). No flag pins this
    from a mode table.

    Under IDENTICAL every enumerated multiply-add seam calls this and
    gets `fma` -- ONE rounding, identical on every backend that
    implements IEEE fma, which Metal, PTX and AMDGPU all do. Under FAST
    it is the naive chain and the backend does whatever it measures
    fastest. IDENTICAL-mode bits therefore differ from FAST-mode bits on
    Apple BY DESIGN; the property purchased is that IDENTICAL's bits are
    the same bits everywhere.
    """
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return _fma_f32(a, b, c)
    return a * b + c


def _fma_f32(a: Float32, b: Float32, c: Float32) -> Float32:
    from std.math import fma

    return fma(a, b, c)
