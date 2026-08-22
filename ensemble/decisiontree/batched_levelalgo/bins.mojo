"""The four histogram bin types, and the atomics that fill them.

MIRRORS `cpp/src/decisiontree/batched-levelalgo/bins.cuh` at
rapidsai/cuml `v26.08.00` (`265b9da6a0e75dbef071a3168398b993a5ff6f0e`),
checked out read-only at `~/CascadeProjects/upstream/cuml-v26.08.00`.

Their file is 189 lines and four structs. Every struct is the same shape:
some accumulators, a static `IncrementHistogram` that computes an offset
and calls a static `AtomicAdd`, an `operator+=` / `operator+` pair for the
warp and block reductions, and a pair of reduction-buffer converters. The
whole learner's numerics live here -- `objectives.cuh` only ever reads
these bins through `Count()`, `Weight()` and `LabelSum()`.

WHY THIS FILE CANNOT BE A TRANSCRIPTION
---------------------------------------
`bins.cuh:14` is

    using BinCountT = unsigned long long int;
    static_assert(sizeof(BinCountT) == 8, "BinCountT must be 64 bits");

and `bins.cuh:29-32` adds to it with a single 64-bit integer `atomicAdd`.
**Apple GPU has no 64-bit integer atomic.** MEASURED, not assumed:
`Atomic.fetch_add` on a `UInt64` is a hard COMPILE error there,
"Atomic operation is not supported for this type on Apple GPU"
(`ensemble/mojo_only/atomic_width_probe.mojo` is the probe that asks the
question; its three outcomes each write a different version of this file,
and outcome 3 is the one that happened).

Their three other bins additionally carry `double` accumulators
(`bins.cuh:56, 101, 142-144`) and this device has no float64 at all.

So there are two forced resolutions in this file, both in DEVIATION 101
below, and they are of OPPOSITE character: the counter width is a width
their own index type never lets them use, so 32 bits is EXACT; the
`double` accumulators are genuine reals, so something has to give and the
question is only what.

================= DEVIATION BLOCK (whole file) =================
DEVIATION 101. `BinCountT` is 64 bits and the accumulators are `double`.
Neither width exists on this device. Resolved per site.

--- 101a. THE COUNTER: 64 -> 32 BITS, AND IT IS EXACT ---------------
THEIRS: `BinCountT = unsigned long long int` (`bins.cuh:14`), incremented
by a 64-bit `atomicAdd` (`bins.cuh:31`).
OURS: `UInt32`, incremented by a 32-bit integer atomic.
REASON, and it is a bound out of their own source rather than a hope: a
bin count is a count of sampled rows, so it is bounded by
`Dataset::n_sampled_rows`, which is declared `IdxT` (`dataset.h:32`), and
`IdxT` is `int` in every instantiation cuML compiles -- `using IdxT = int`
in `decisiontree.cuh:255` and again in each of the eight instantiation
translation units (`kernels/classification-float.cu:14`,
`classification-double.cu`, `regression-float.cu`, `regression-double.cu`
and their weighted twins). `DecisionTree::fit` takes `const int nrows`
(`decisiontree.cuh:240`) and `rmm::device_uvector<int>* row_ids`
(`decisiontree.cuh:242`). A count therefore cannot exceed 2^31-1 and
cannot overflow a `UInt32`. Their 64-bit width buys headroom their own
index type forbids them from using.
PRICE: zero. Not one reachable bin count changes value. What is lost is
the ability to accept a dataset with more than 2^31-1 sampled rows, which
cuML cannot accept either.
THE VENDOR ROW LIVES IN A TABLE, NOT IN THIS FILE. There is no `if apple`
anywhere here; the width is a single constant for every column, which is
what makes one model out of three backends. The per-vendor facts behind
that constant are `ensemble/mojo_only/atomic_matrix.mojo`
(`column_has_64bit_int_atomics`, `bin_counter_bits`,
`bin_counter_is_exact_at_32_bits`), written by the same round. This file
does not IMPORT that table -- `ensemble/mojo_only/` is this lane's CHECKS
directory, and since 2026-08-21 the shipping path imports nothing from it
(the three primitives it once held -- Philox, the segmented sort, the
shuffle iterator -- live in `core/` now). WIRING `UInt32` HERE TO
`bin_counter_bits(TARGET_COLUMN)` THERE IS AN OPEN MERGE-TIME ITEM; until
it is done the two files agree by inspection, which is weaker than
agreeing by construction.
CONSEQUENCE, and it is the reason to be glad rather than sorry: integer
addition is associative, so the unweighted classification histogram is
independent of the order threads arrive in. That, plus the total-order
tie-break in `split.cuh:142-191`, is what makes the unweighted
classification path BIT-IDENTICAL across Metal, CUDA and HIP, with no
fixed-point machinery and no numeric mode. It is the cleanest identity
column in this library.

--- 101b. THE `double` ACCUMULATORS: FIXED-POINT Int32 ---------------
THEIRS: `WeightedClassificationBin::weight`, `RegressionBin::label_sum`,
`WeightedRegressionBin::label_sum` and `::weight` are `double`
(`bins.cuh:57, 101, 142, 144`), summed with `atomicAdd(double*)`
(`bins.cuh:73-74, 114-115, 160-162`).
OURS: each is an `Int32` FIXED-POINT raw slot, summed with a 32-bit
integer atomic, dequantized on read by dividing by a scale the host
chooses once per fit (`mojo_only/fixed_point.mojo::choose_scale`, the
accumulator this repository already built for `gbdt/` and transferred
unchanged to `cluster/`).
REASON: Float32 was the obvious substitute and it is the wrong one. A
float atomic is order-nondeterministic RUN TO RUN, not merely vendor to
vendor (`mojo_only/numerics.mojo`, "what a mode cannot do"), so a Float32
accumulator would put the regression path permanently outside this
library's identity floor -- and that floor is FROZEN, so there would be no
way back later except an API change. Integer addition is associative, so
a fixed-point sum is order-independent by construction. This is not a
choice this lane invented: `ensemble/PLAN.md`'s vendor-call table already
names this exact site -- "double atomicAdd on RegressionBin.label_sum |
histogram | NO float64 on device -- fixed-point accumulate".
PRICE, stated in units rather than adjectives:
  * RESOLUTION. `choose_scale` reserves three headroom bits, so a plane
    holds about 2^28 scaled units against a bound computed from the sum
    of magnitudes over ALL rows. Any node's rows are a subset of all
    rows, so no partial sum at any depth can overflow. Against their
    float64 accumulator (53-bit mantissa, exact for integers to 2^53)
    this is a real precision reduction on regression label sums.
  * QUANTIZATION. `Int32(value * scale)` truncates toward zero, at most
    one unit per row, deterministically. Truncation rather than rounding
    is deliberate: it has no tie-breaking rule to get wrong on another
    vendor.
  * ONE HOST PASS over labels (and over sample weights, when weighted)
    per fit, to compute the sum of magnitudes `choose_scale` needs.
  * AN API CHANGE. Their `Weight()` and `LabelSum()` are nullary const
    methods; ours take a `BinScales`. The scale cannot live in the bin --
    it would be replicated into every bin of every histogram -- and it
    cannot be a compile-time constant, because it is a function of the
    data. The objective function carries it; see DEVIATION 112c
    (`objectives.mojo`; that block records why it is 112 and not the
    105 this lane was assigned).
  * The dequantize `Float32(raw) / scale` is EXACT in its division,
    because `choose_scale` snaps the scale down to a power of two. Only
    the `Int32 -> Float32` conversion rounds, and only above 2^24.
WHAT IS BOUGHT: the regression and weighted paths are order-independent
and therefore bit-identical across Metal, CUDA and HIP -- a property
cuML's own `atomicAdd(double*)` does NOT have.

--- 101c. `ToReductionBuffer` / `FromReductionBuffer`: NOT PORTED -----
`bins.cuh:35-42, 78-86, 120-127, 167-173` exist only to marshal a bin
through the multi-GPU allreduce (`builder.cuh:553`,
`builder_kernels.cuh:167-194`), which is out of scope for a
single-device library; nothing else in their tree calls them.
=================================================================
"""

from std.atomic import Atomic
from max.gpu.memory import AddressSpace


# ------------------------------------------------------------------ scales --
@fieldwise_init
struct BinScales(TrivialRegisterPassable):
    """NO CUML COUNTERPART. The fixed-point scales of DEVIATION 101b.

    Two independent planes, because their magnitudes are independent:
    `label_sum` accumulates labels (or `label * weight`, `bins.cuh:157`)
    and `weight` accumulates sample weights (`dataset.h:22`). Both come
    from `mojo_only/fixed_point.mojo::choose_scale` on the host.

    `ClassificationBin` and `RegressionBin` never read either field --
    their `Weight()` is a count, which is exact -- but they accept the
    argument so that every bin presents ONE interface to the objective
    functions, which is what their `BinT` template parameter assumes.
    """

    # Multiplier applied to a label before it is truncated into an Int32.
    var label_scale: Float32
    # Multiplier applied to a sample weight before it is truncated.
    var weight_scale: Float32

    @staticmethod
    def unit() -> BinScales:
        """The scales of the UNWEIGHTED CLASSIFICATION path, where neither
        plane exists and neither field is ever read."""
        return BinScales(1.0, 1.0)


@always_inline
def _quantize(value: Float32, scale: Float32) -> Int32:
    """The device-side twin of `mojo_only/fixed_point.mojo::quantize`.

    That one takes and returns Float64 because it is host code; this one
    cannot, because there is no float64 on this device. Truncation toward
    zero, identically, and for the same reason: it is a deterministic
    function of the input with no tie-break rule to get wrong elsewhere.
    """
    return Int32(value * scale)


@always_inline
def _dequantize_wide(raw: Int64, scale: Float32) -> Float32:
    """`_dequantize` for a value that has already been SUBTRACTED at
    storage width. `Int32 - Int32` can leave the type (labels may be
    negative, so the two operands can straddle zero), hence Int64 in and
    exactly one rounding out."""
    return Float32(Int(raw)) / scale


@always_inline
def _dequantize(raw: Int32, scale: Float32) -> Float32:
    """Fixed-point back to a real. The division is EXACT: `choose_scale`
    snaps the scale down to a power of two, so only the `Int32 -> Float32`
    conversion rounds, and only above 2^24."""
    return Float32(Int(raw)) / scale


# -------------------------------------------------------------------- trait --
trait Bin(TrivialRegisterPassable):
    """What `objectives.cuh` requires of its `BinT` template parameter.

    NOT A PORT OF A CUML CONSTRUCT -- cuML has no bin concept, it has
    `std::conditional_t<weighted_, WeightedX, X>` (`objectives.cuh:26`,
    `:204`) and duck typing. This trait is that duck typing written down;
    see DEVIATION 112b for why the direction is inverted here.
    """

    # `objectives.cuh:27` / `:205` -- `static constexpr bool weighted`.
    comptime weighted: Bool

    # NOT in their source. Which objective family this bin belongs to, so
    # that pairing a `RegressionBin` with `ClassificationObjectiveFunction`
    # is a compile error rather than a wrong model. Their `conditional_t`
    # made the pairing unstateable; ours has to state it.
    comptime is_classification: Bool

    # NOT in their source. `Weight()` returns `double` for all four of
    # their bins; here two of the four return an EXACT INTEGER instead,
    # because that is what their double is holding. See DEVIATION 114.
    comptime weight_dtype: DType

    def __init__(out self):
        """`bins.cuh:22, 63, 105, 147` -- the zero bin."""
        ...

    def Count(self) -> UInt32:
        """`bins.cuh:33, 76, 118, 165`."""
        ...

    def Weight(self, scales: BinScales) -> Scalar[Self.weight_dtype]:
        """`bins.cuh:34, 77, 119, 166`, plus DEVIATION 101b's `scales`."""
        ...

    def __iadd__(mut self, b: Self):
        """`bins.cuh:43-47, 87-92, 128-133, 175-181` -- `operator+=`."""
        ...

    def __add__(self, b: Self) -> Self:
        """`bins.cuh:48-52, 93-97, 134-138, 182-186` -- `operator+`."""
        ...

    @staticmethod
    def AtomicAdd[
        origin: MutOrigin,
        address_space: AddressSpace, //,
    ](
        address: MutPointer[Self, origin, address_space=address_space],
        val: Self,
    ):
        """`bins.cuh:29-32, 71-75, 112-116, 158-163`.

        The origin and address space are parameters because their kernel
        needs BOTH: the shared-memory histogram is the default arm and it
        falls back to a global one only when the histogram will not fit
        (`kernels/builder_kernels_impl.cuh:322-333`, `builder.cuh:545`).
        A callee annotated `MutAnyOrigin` rejects a `stack_allocation`
        pointer -- see `mojo_only/shared_pointer_probe.mojo`.
        """
        ...

    @staticmethod
    def IncrementHistogram[
        label_dtype: DType,
        origin: MutOrigin,
        address_space: AddressSpace, //,
    ](
        hist: MutPointer[Self, origin, address_space=address_space],
        n_bins: Int32,
        b: Int32,
        label: Scalar[label_dtype],
        weight: Float32,
        scales: BinScales,
    ):
        """`bins.cuh:24-28, 65-70, 108-111, 153-157`."""
        ...


trait RegressionBinLike(Bin):
    """The two bins that carry a label sum. `objectives.cuh:224-225`,
    `:247-249`, `:275-277`, `:302-304` and `:383-384` read it."""

    def LabelSum(self, scales: BinScales) -> Float32:
        """`bins.cuh:117, 164`, plus DEVIATION 101b's `scales`."""
        ...

    def LabelSumMinus(self, other: Self, scales: BinScales) -> Float32:
        """`self.LabelSum() - other.LabelSum()`, taken at STORAGE width.

        NOT a separate function in their source, and it does not need to
        be: `objectives.cuh:249`, `:277` and `:304` write

            DataT right_label_sum =
              DataT(hist[n_bins - 1].LabelSum() - hist[i].LabelSum());

        and their `LabelSum()` returns the `double` that IS the storage,
        so the subtraction already happens at full storage width and is
        narrowed exactly once.

        Ours stores Int32 fixed point, so `LabelSum()` has to round on the
        way out. Calling it twice and subtracting the results puts two
        roundings in front of a CANCELLATION -- the one place a lost low
        bit is amplified rather than absorbed. Subtracting the raw
        integers instead is EXACT and narrows once, which is the same
        shape their line has.

        This exists on the trait rather than at the call site because the
        raw field is private to each bin.
        """
        ...


# ------------------------------------------------------ ClassificationBin --
@fieldwise_init
struct ClassificationBin(Bin):
    """`struct ClassificationBin`, `bins.cuh:17-53`.

    THE SHIP-FIRST PATH, and the only one that needs no deviation beyond
    the counter width. One integer counter, one integer atomic, no scale,
    no float anywhere on the write side.
    """

    # `bins.cuh:18` -- `BinCountT count`. DEVIATION 101a: 64 -> 32 bits.
    var count: UInt32

    comptime weighted = False
    comptime is_classification = True
    # Their `Weight()` is `static_cast<double>(count)` (`bins.cuh:34`) and
    # a count is an exact integer, so an exact integer is what it is
    # carried as here. DEVIATION 114.
    comptime weight_dtype = DType.int64

    @always_inline
    def __init__(out self):
        """`bins.cuh:22` -- `HDI ClassificationBin() : count(0) {}`."""
        self.count = 0

    @staticmethod
    @always_inline
    def IncrementHistogram[
        label_dtype: DType,
        origin: MutOrigin,
        address_space: AddressSpace, //,
    ](
        hist: MutPointer[Self, origin, address_space=address_space],
        n_bins: Int32,
        b: Int32,
        label: Scalar[label_dtype],
        weight: Float32,
        scales: BinScales,
    ):
        """`bins.cuh:24-28`. Their trailing unnamed `double` is the weight
        and they ignore it; so do we, and `scales` with it."""
        var offset = Int(label) * Int(n_bins) + Int(b)
        ClassificationBin.AtomicAdd(
            hist.unsafe_offset(offset), ClassificationBin(1)
        )

    @staticmethod
    @always_inline
    def AtomicAdd[
        origin: MutOrigin,
        address_space: AddressSpace, //,
    ](
        address: MutPointer[Self, origin, address_space=address_space],
        val: Self,
    ):
        """`bins.cuh:29-32` -- `atomicAdd(&address->count, val.count)`.

        `count` is the struct's only field, so slot 0 of the bitcast IS
        `&address->count`. `objectives_check.mojo` writes through this
        path and reads back through the field, per cell, so a wrong slot
        is visible rather than assumed away.
        """
        _ = Atomic.fetch_add(
            address.unsafe_bitcast[UInt32]().unsafe_offset(0), val.count
        )

    @always_inline
    def Count(self) -> UInt32:
        """`bins.cuh:33`."""
        return self.count

    @always_inline
    def Weight(self, scales: BinScales) -> Int64:
        """`bins.cuh:34` -- `static_cast<double>(count)`, held exactly."""
        return Int64(Int(self.count))

    @always_inline
    def __iadd__(mut self, b: Self):
        """`bins.cuh:43-47`."""
        self.count += b.count

    @always_inline
    def __add__(self, b_in: Self) -> Self:
        """`bins.cuh:48-52`. Theirs takes `b` BY VALUE and adds `*this`
        into it, so the surviving order is `b.count += this->count`.
        Integer addition is commutative and associative, so the order is
        immaterial here -- transcribed anyway, because it is not
        immaterial in the three bins below."""
        var b = b_in
        b += self
        return b


# ---------------------------------------------- WeightedClassificationBin --
@fieldwise_init
struct WeightedClassificationBin(Bin):
    """`struct WeightedClassificationBin`, `bins.cuh:55-98`."""

    # `bins.cuh:56` -- `BinCountT count`. DEVIATION 101a.
    var count: UInt32
    # `bins.cuh:57` -- `double weight`. DEVIATION 101b: fixed-point raw,
    # dequantized by `BinScales.weight_scale`.
    var weight: Int32

    comptime weighted = True
    comptime is_classification = True
    comptime weight_dtype = DType.float32

    @always_inline
    def __init__(out self):
        """`bins.cuh:63` -- `count(0), weight(0.0)`."""
        self.count = 0
        self.weight = 0

    @staticmethod
    @always_inline
    def IncrementHistogram[
        label_dtype: DType,
        origin: MutOrigin,
        address_space: AddressSpace, //,
    ](
        hist: MutPointer[Self, origin, address_space=address_space],
        n_bins: Int32,
        b: Int32,
        label: Scalar[label_dtype],
        weight: Float32,
        scales: BinScales,
    ):
        """`bins.cuh:65-70` -- `AtomicAdd(hist + offset, {1, weight})`."""
        var offset = Int(label) * Int(n_bins) + Int(b)
        WeightedClassificationBin.AtomicAdd(
            hist.unsafe_offset(offset),
            WeightedClassificationBin(
                1, _quantize(weight, scales.weight_scale)
            ),
        )

    @staticmethod
    @always_inline
    def AtomicAdd[
        origin: MutOrigin,
        address_space: AddressSpace, //,
    ](
        address: MutPointer[Self, origin, address_space=address_space],
        val: Self,
    ):
        """`bins.cuh:71-75` -- two atomics, count then weight. Both are
        integer atomics here; theirs is integer then float64."""
        _ = Atomic.fetch_add(
            address.unsafe_bitcast[UInt32]().unsafe_offset(0), val.count
        )
        _ = Atomic.fetch_add(
            address.unsafe_bitcast[Int32]().unsafe_offset(1), val.weight
        )

    @always_inline
    def Count(self) -> UInt32:
        """`bins.cuh:76`."""
        return self.count

    @always_inline
    def Weight(self, scales: BinScales) -> Float32:
        """`bins.cuh:77` -- `return weight`, dequantized."""
        return _dequantize(self.weight, scales.weight_scale)

    @always_inline
    def __iadd__(mut self, b: Self):
        """`bins.cuh:87-92`."""
        self.count += b.count
        self.weight += b.weight

    @always_inline
    def __add__(self, b_in: Self) -> Self:
        """`bins.cuh:93-97`."""
        var b = b_in
        b += self
        return b


# ------------------------------------------------------------ RegressionBin --
@fieldwise_init
struct RegressionBin(RegressionBinLike):
    """`struct RegressionBin`, `bins.cuh:100-139`."""

    # `bins.cuh:101` -- `double label_sum`. DEVIATION 101b: fixed-point.
    var label_sum: Int32
    # `bins.cuh:102` -- `BinCountT count`. DEVIATION 101a.
    var count: UInt32

    comptime weighted = False
    comptime is_classification = False
    # `bins.cuh:118` is `static_cast<double>(count)`: an exact integer.
    comptime weight_dtype = DType.int64

    @always_inline
    def __init__(out self):
        """`bins.cuh:105` -- `label_sum(0.0), count(0)`."""
        self.label_sum = 0
        self.count = 0

    @staticmethod
    @always_inline
    def IncrementHistogram[
        label_dtype: DType,
        origin: MutOrigin,
        address_space: AddressSpace, //,
    ](
        hist: MutPointer[Self, origin, address_space=address_space],
        n_bins: Int32,
        b: Int32,
        label: Scalar[label_dtype],
        weight: Float32,
        scales: BinScales,
    ):
        """`bins.cuh:108-111` -- `AtomicAdd(hist + b, {label, 1})`.

        NOTE the offset: `hist + b`, with NO `label * n_bins` term. A
        regression histogram has one plane, and `n_bins` is unused in
        their signature too.
        """
        RegressionBin.AtomicAdd(
            hist.unsafe_offset(Int(b)),
            RegressionBin(
                _quantize(Float32(label), scales.label_scale), 1
            ),
        )

    @staticmethod
    @always_inline
    def AtomicAdd[
        origin: MutOrigin,
        address_space: AddressSpace, //,
    ](
        address: MutPointer[Self, origin, address_space=address_space],
        val: Self,
    ):
        """`bins.cuh:112-116` -- label_sum then count, their order."""
        _ = Atomic.fetch_add(
            address.unsafe_bitcast[Int32]().unsafe_offset(0), val.label_sum
        )
        _ = Atomic.fetch_add(
            address.unsafe_bitcast[UInt32]().unsafe_offset(1), val.count
        )

    @always_inline
    def LabelSum(self, scales: BinScales) -> Float32:
        """`bins.cuh:117`, dequantized."""
        return _dequantize(self.label_sum, scales.label_scale)

    @always_inline
    def LabelSumMinus(self, other: Self, scales: BinScales) -> Float32:
        """`objectives.cuh:249`, `:277`, `:304` -- their subtraction at
        storage width, narrowed once. See the trait for why."""
        return _dequantize_wide(
            Int64(Int(self.label_sum)) - Int64(Int(other.label_sum)),
            scales.label_scale,
        )

    @always_inline
    def Count(self) -> UInt32:
        """`bins.cuh:118`."""
        return self.count

    @always_inline
    def Weight(self, scales: BinScales) -> Int64:
        """`bins.cuh:119` -- `static_cast<double>(count)`, held exactly."""
        return Int64(Int(self.count))

    @always_inline
    def __iadd__(mut self, b: Self):
        """`bins.cuh:128-133`."""
        self.label_sum += b.label_sum
        self.count += b.count

    @always_inline
    def __add__(self, b_in: Self) -> Self:
        """`bins.cuh:134-138`."""
        var b = b_in
        b += self
        return b


# ---------------------------------------------------- WeightedRegressionBin --
@fieldwise_init
struct WeightedRegressionBin(RegressionBinLike):
    """`struct WeightedRegressionBin`, `bins.cuh:141-187`."""

    # `bins.cuh:142` -- `double label_sum`. DEVIATION 101b.
    var label_sum: Int32
    # `bins.cuh:143` -- `BinCountT count`. DEVIATION 101a.
    var count: UInt32
    # `bins.cuh:144` -- `double weight`. DEVIATION 101b.
    var weight: Int32

    comptime weighted = True
    comptime is_classification = False
    comptime weight_dtype = DType.float32

    @always_inline
    def __init__(out self):
        """`bins.cuh:147` -- `label_sum(0.0), count(0), weight(0.0)`."""
        self.label_sum = 0
        self.count = 0
        self.weight = 0

    @staticmethod
    @always_inline
    def IncrementHistogram[
        label_dtype: DType,
        origin: MutOrigin,
        address_space: AddressSpace, //,
    ](
        hist: MutPointer[Self, origin, address_space=address_space],
        n_bins: Int32,
        b: Int32,
        label: Scalar[label_dtype],
        weight: Float32,
        scales: BinScales,
    ):
        """`bins.cuh:153-157` -- `AtomicAdd(hist + b, {label * weight, 1,
        weight})`. The product is theirs; it is what makes `label_sum`
        need its OWN scale rather than the label's.

        THE PRODUCT'S WIDTH IS NOT THEIRS, and it is not fixable here.
        Both of their operands are `double` -- `sample_weight` is declared
        `const double*` (`dataset.h:22`) -- so their multiply is exact and
        lands in a `double` field. Ours multiplies two Float32s and then
        truncates into Int32, which is two roundings where they have none.

        This is the COMPOSITION of two already-priced deviations rather
        than a third one: DEVIATION 100 narrows the weight to Float32
        because this device has no float64, and DEVIATION 101b turns the
        accumulator into fixed point because it has no float64 atomic.
        Neither block says they stack on this one value, so it is said
        here. There is no higher-precision product available to reach for:
        every wider path runs through float64.

        Only the WEIGHTED regression bin is affected. The unweighted one
        (`:504`) quantizes the label alone, and both classification bins
        count integers."""
        WeightedRegressionBin.AtomicAdd(
            hist.unsafe_offset(Int(b)),
            WeightedRegressionBin(
                _quantize(Float32(label) * weight, scales.label_scale),
                1,
                _quantize(weight, scales.weight_scale),
            ),
        )

    @staticmethod
    @always_inline
    def AtomicAdd[
        origin: MutOrigin,
        address_space: AddressSpace, //,
    ](
        address: MutPointer[Self, origin, address_space=address_space],
        val: Self,
    ):
        """`bins.cuh:158-163` -- label_sum, count, weight, their order."""
        _ = Atomic.fetch_add(
            address.unsafe_bitcast[Int32]().unsafe_offset(0), val.label_sum
        )
        _ = Atomic.fetch_add(
            address.unsafe_bitcast[UInt32]().unsafe_offset(1), val.count
        )
        _ = Atomic.fetch_add(
            address.unsafe_bitcast[Int32]().unsafe_offset(2), val.weight
        )

    @always_inline
    def LabelSum(self, scales: BinScales) -> Float32:
        """`bins.cuh:164`, dequantized."""
        return _dequantize(self.label_sum, scales.label_scale)

    @always_inline
    def LabelSumMinus(self, other: Self, scales: BinScales) -> Float32:
        """`objectives.cuh:249`, `:277`, `:304` -- their subtraction at
        storage width, narrowed once. See the trait for why."""
        return _dequantize_wide(
            Int64(Int(self.label_sum)) - Int64(Int(other.label_sum)),
            scales.label_scale,
        )

    @always_inline
    def Count(self) -> UInt32:
        """`bins.cuh:165`."""
        return self.count

    @always_inline
    def Weight(self, scales: BinScales) -> Float32:
        """`bins.cuh:166` -- `return weight`, dequantized."""
        return _dequantize(self.weight, scales.weight_scale)

    @always_inline
    def __iadd__(mut self, b: Self):
        """`bins.cuh:175-181`."""
        self.label_sum += b.label_sum
        self.count += b.count
        self.weight += b.weight

    @always_inline
    def __add__(self, b_in: Self) -> Self:
        """`bins.cuh:182-186`."""
        var b = b_in
        b += self
        return b
