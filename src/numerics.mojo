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


#: Every backend picks every row for speed. Histograms flush through
#: floating-point atomics, so two runs of the SAME fit on the SAME device may
#: differ in the last bits. This is CatBoost's shipped behavior.
comptime NUMERIC_FAST = 0

#: Numeric rows are pinned to the safe column; scheduling rows stay free. The
#: same fit gives the same model, on this device and across backends.
comptime NUMERIC_IDENTICAL = 1


@fieldwise_init
struct NumericMode(Copyable, Movable):
    """The mode, plus the resolved value of every numeric row.

    The rows are resolved ONCE, here, rather than each kernel calling
    `getenv` and reaching its own conclusion. A kernel that decides for itself
    is how two kernels come to disagree about which mode they are in.
    """

    var mode: Int

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
