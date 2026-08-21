"""The split criteria: Gini, Entropy, MSE, Poisson, Gamma, InverseGaussian.

MIRRORS `cpp/src/decisiontree/batched-levelalgo/objectives.cuh` at
rapidsai/cuml `v26.08.00` (`265b9da6a0e75dbef071a3168398b993a5ff6f0e`),
checked out read-only at `~/CascadeProjects/upstream/cuml-v26.08.00`.

Two class templates, `ClassificationObjectiveFunction` (`:20-196`) and
`RegressionObjectiveFunction` (`:198-387`), each parameterized on a
`weighted` flag that picks the bin type. Between them they are the entire
decision rule of cuML's random forest: everything else in the directory
exists to build histograms for these six functions to read and to reduce
the `Split` they return.

THE ONE THING TO KNOW BEFORE READING THE ARITHMETIC
---------------------------------------------------
**The histograms these functions read are CUMULATIVE.** The kernel prefix
-sums them before calling in (`kernels/builder_kernels_impl.cuh`), so
`hist[n_bins - 1]` is the whole node and `hist[i]` is the left partition
of the split after bin `i`. That is why `right = total - left` appears in
all six and why nothing here scans.

TRANSCRIPTION RULE, applied literally
--------------------------------------
Expression by expression, in their order, with no algebraic
simplification, no reassociation and no hoisting. `GiniGain` computes

    gain += lval * invLeft * lval * invLen;                (`:64`)

and that grouping is not an accident of typing -- regrouping it to
`lval * lval * invLeft * invLen` changes the last bits, which changes
which of two near-tied splits `split.cuh`'s total order selects.
`EntropyGain` recomputes `raft::log(DataT(2))` on every one of its three
terms in every class (`:99`, `:107`, `:113`); it is NOT hoisted here
either. This repository has been bitten by exactly this class of defect
twice, most recently PORTING.md 54.

================= DEVIATION BLOCK (whole file) =================
NUMBERING NOTE FOR THE MERGE. This lane was assigned 101/105/106/107.
101 is used as assigned, in `bins.mojo`. **105, 106 and 107 were already
consumed by `split.mojo`** (DEVIATION 105 non-associativity, 106 mutex,
107 `printSplits`), which was written concurrently in the same directory,
so the three deviations below took the next free numbers in the
ensemble reserve instead: 112, 113, 114. Renumber at merge if the ledger
prefers the original assignment.

DEVIATION 112. THE EIGHT INSTANTIATION TRANSLATION UNITS COLLAPSE, AND
THE `std::conditional_t` RUNS BACKWARDS.

(a) Their `kernels/*.cu` -- `classification-float.cu`,
`classification-double.cu`, `regression-float.cu`,
`regression-double.cu` and their four weighted twins -- exist only to
force explicit template instantiations into separate objects, "split
across separate .cu files to increase compilation parallelism"
(`kernels/classification-float.cu:20`). They carry no logic. Mojo
comptime specialization produces the same set with no files. This is the
pre-approved deviation of the same shape as DEVIATION 63 and is not
re-argued here.

(b) Their bin selection is `using BinT = std::conditional_t<weighted_,
WeightedClassificationBin, ClassificationBin>` (`:26`, `:204`): a Bool
parameter picks a type. **Mojo 1.0 cannot express that.** MEASURED this
session: `comptime BinT = B if Self.weighted else A` compiles, but the
binding erases to `Copyable & Deinitable`, so `Self.BinT.weighted` is
"value has no attribute". The mapping is therefore inverted -- the
objective is parameterized on the BIN TYPE and reads `weighted` off it
(`BinT.weighted`, which is their `:27` / `:205` `static constexpr bool
weighted`). Same set of instantiations, named from the other end.
PRICE: a caller can now write down a pairing their `conditional_t` made
unstateable (a `RegressionBin` with a classification objective). Bought
back with `comptime is_classification` on every bin and a `comptime
assert` in each objective's constructor, so the bad pairing is a compile
error rather than a wrong model.

(c) Their `Weight()` / `LabelSum()` are nullary; ours take a `BinScales`,
and so every objective carries one. That is DEVIATION 101b's consequence,
priced there.

(d) CLOSED. `detail::CountLeft` (`split.cuh:19-27`) was briefly duplicated
into this file as `_count_left`, because `split.mojo`'s `count_left` had
the wrong signature -- a plane of `UInt32` counts, where theirs takes
`BinT const*` and calls `.Count()`, and where `Gain` (`:168`, `:369`)
hands it the BIN histogram. The duplicate is gone: `split.mojo`'s is now
bin-typed and this file imports it, which is their own arrangement
(`split.cuh` includes `bins.cuh` at `:8`; `objectives.cuh` includes both
and calls `detail::CountLeft`). Nothing is duplicated and nothing is
unreached.

(e) Both objective structs carry `where dtype.is_floating_point()`. Mojo
requires the evidence before `std.math.log` will resolve. It restricts
nothing: their `DataT` is `float` or `double` in all eight
instantiations, so the constraint is exactly their domain written down.

DEVIATION 113. `raft::log` BECOMES `std.math.log`, ON THE DEVICE, WHERE
THE RECORDED FIX IS UNAVAILABLE.
THEIRS: `raft::log` in `EntropyGain` (`:99`, `:107`, `:113`),
`PoissonGain` (`:255-257`) and `GammaGain` (`:283-285`).
OURS: `std.math.log`.
This repository has a recorded defect for exactly this substitution:
`std.math.log` carries ~5e-8 ABSOLUTE error against libm, measured at
w = 840 as 5656.057589200282 against libm's 5656.057589153382, and that
noise silently re-decided dynamic-programming plateau ties in CatBoost's
border selection (PORTING.md 54 and the docstring of
`gbdt/grid_creator/binarization.mojo::_penalty_min_entropy`). **The
recorded fix is `external_call["log", Float64]` -- libm through FFI --
and that fix is HOST-ONLY.** The same docstring says so in as many words:
"This is host code, which is why calling libm is available at all. If any
of this ever moves to a device, `log` has to be revisited from scratch,
because no libm is reachable there." These three gains are `HDI`/`DI`
functions called from inside the split kernel. There is no libm on a GPU
and there is no float64 on this one, so the recorded fix cannot be
applied and no new one is invented here.
PRICE, and it is a real one:
  * GINI and MSE contain no transcendental and are unaffected. They are
    the two defaults (`decisiontree.cuh:253-254` picks GINI for integer
    labels and MSE otherwise), so the shipped path pays nothing.
  * ENTROPY, POISSON and GAMMA are NOT bit-comparable to cuML's, and the
    divergence is not merely a last-bit one on the gain: on a plateau of
    equally-good splits the total order in `split.cuh:142-191` fires on
    the noise instead of on the tie-break, and a DIFFERENT split is
    chosen. Same class of failure as PORTING.md 54, different tree.
  * Across OUR OWN vendors the choice is still deterministic -- one
    source, one `log` -- so the identity column survives; it is
    comparability with cuML that is lost, not reproducibility.
NOT DEVIATED, recorded so nobody "fixes" it: `raft::log(DataT(2))` is
recomputed inside the loop three times per class rather than hoisted,
because that is where they compute it.
ON FMA: `gain += lval * invLeft * lval * invLen` is a multiply-add inside
ONE source expression, which is exactly the case nvcc contracts by
default (`-fmad=true`) and clang contracts at `-ffp-contract=on`. Mojo
contracting it too is agreement, not divergence. The PORTING.md 54 defect
was the other case -- contraction ACROSS an inlined call boundary, which
clang does not do -- and there is no such boundary in this file: every
gain computes its own terms inline. No `@no_inline` is needed here and
none is applied. If a future edit factors a term out into a helper, that
helper needs `@no_inline` and this paragraph is why.

DEVIATION 114. EVERY `double` IN THIS FILE, RESOLVED PER SITE.
Their locals are `double` in four places and the resolutions differ,
which is the point of listing them:
  * `WeightAt`'s accumulator (`:37`), `GiniGain`'s `val_i`/`lval_i`/
    `rval_i` (`:61-72`), `EntropyGain`'s (`:95-110`), and
    `SetLeafVector`'s `total` (`:182`) all hold `BinT::Weight()`.
    - For `ClassificationBin` and `RegressionBin`, `Weight()` is
      `static_cast<double>(count)` (`bins.cuh:34`, `:119`) -- AN EXACT
      INTEGER. Those are carried as **Int64**, which reproduces their
      double BIT FOR BIT for every count their `int` index type admits
      (2^31-1 << 2^53). This is not an approximation and not an
      improvement; it is the same number in the type that holds it.
    - For the two weighted bins `Weight()` is a real, and is **Float32**.
      Precision loss against their float64, on the weighted path only.
  * `RegressionObjectiveFunction::eps_` (`:211`) is
    `10 * numeric_limits<DataT>::epsilon()`, so it follows DataT and
    needs no resolution.
  * THE ONE THAT IS EASY TO MISS: `SetLeafVector` writes
    `out[i] = DataT(shist[i].Weight()) / total` (`:193`) and
    `shist[i].LabelSum() / weight` (`:384`). In C++ those DIVISIONS
    happen in `double` because `total` / `weight` are `double`, and the
    result is narrowed on assignment. Here the classification one divides
    in Int64-to-Float32 space and the regression one in Float32, so a
    leaf value can differ from theirs in the last bit. Priced: a leaf
    value is an output, not a decision -- it feeds no comparison inside
    the fit -- so a last-bit difference cannot change the TREE, only the
    prediction it emits.
  * DECLINED: `classification-double.cu` and `regression-double.cu`
    instantiate DataT = double. There is no float64 on this device, so
    those two of their eight translation units have no counterpart.
    Price: a caller who wants float64 features gets float32. cuML's own
    Python layer converts to float32 by default
    (`randomforestclassifier.pyx`), so this is the arm their users take.
=================================================================
"""

from std.gpu import block_dim, thread_idx
from std.math import log
from std.utils.numerics import nextafter
from max.gpu.memory import AddressSpace

from ensemble.decisiontree.batched_levelalgo.bins import (
    Bin,
    BinScales,
    RegressionBinLike,
)
from ensemble.decisiontree.batched_levelalgo.dataset import DatasetView
from ensemble.decisiontree.batched_levelalgo.split import Split, count_left


# ----------------------------------------------------------------- CRITERION --
# `include/cuml/tree/algo_helper.h:10-19`, `enum CRITERION`. Their values
# are the implicit 0..7 of a plain C enum and `DecisionTreeParams` carries
# it as that enum, so the numbering is theirs and must not be reordered:
# `CRITERION_END` is the "unset" sentinel `decisiontree.cuh:246` tests for.
comptime CRITERION_GINI = Int32(0)
comptime CRITERION_ENTROPY = Int32(1)
comptime CRITERION_MSE = Int32(2)
comptime CRITERION_MAE = Int32(3)
comptime CRITERION_POISSON = Int32(4)
comptime CRITERION_GAMMA = Int32(5)
comptime CRITERION_INVERSE_GAUSSIAN = Int32(6)
comptime CRITERION_END = Int32(7)


@always_inline
def machine_epsilon[dtype: DType]() -> Scalar[dtype]:
    """`std::numeric_limits<T>::epsilon()`: 2^-(mantissa bits), IEEE-754.

    Device-legal, unlike `nextafter` -- see the note on
    `RegressionObjectiveFunction.eps_`. Every literal is an exact power of
    two, so each is exactly representable in its own type and the value is
    identical to what `nextafter(1, 2) - 1` returns on the host.
    """
    comptime if dtype == DType.float32:
        return Scalar[dtype](1.1920928955078125e-07)  # 2^-23
    elif dtype == DType.float64:
        return Scalar[dtype](2.220446049250313e-16)  # 2^-52
    elif dtype == DType.float16:
        return Scalar[dtype](0.0009765625)  # 2^-10
    else:
        comptime assert False, (
            "machine_epsilon: no mantissa width recorded for this dtype;"
            " add it rather than guessing"
        )
        return Scalar[dtype](0)


# ---------------------------------------------------------------- ObjectiveLike --
# MOVED HERE from `kernels/builder_kernels_impl.mojo` (DEVIATION 129a, now
# CLOSED). It had to live in that file while it was written, because
# `objectives.mojo` declares no trait and Mojo traits are NOMINAL -- so the
# kernels could not name what they required of an objective, and the
# launchers were OVERLOADED on the two concrete objective types instead,
# with a forwarding adapter apiece.
#
# The cost of that was not the adapters. It was that `Builder` could not be
# generic over the objective the way `Builder<ObjectiveT>` is, so REGRESSION
# FORESTS COULD NOT TRAIN AT ALL. Declaring the trait here, where both
# objectives can conform to it and the kernels can import it without a
# cycle, deletes the two adapters, the six launcher overloads and that
# restriction together.

trait ObjectiveLike(Copyable & Deinitable):
    """What `buildHistogramsKernel`, `findBestSplitsKernel` and
    `leafKernel` require of `typename ObjectiveT`.

    NOT A PORT OF A CUML CONSTRUCT. Their three kernels take an
    unconstrained `typename ObjectiveT` and use it structurally:
    `::BinT` (`:296`, `:363`, `:219`), `NumClasses()` (`:307`, `:369`),
    `IncrementHistogram(...)` (`:341`, `:234`), `Gain(...)` (`:385`) and
    the static `SetLeafVector(...)` (`:238-239`). This trait is that list
    written down; see DEVIATION 129a for why it lives here.
    """

    comptime DataT: DType
    comptime LabelT: DType
    comptime BinT: Bin

    def __init__(
        out self,
        nclasses: Int32,
        min_samples_leaf: Int32,
        criterion: Int32,
        min_impurity_decrease: Scalar[Self.DataT],
        scales: BinScales,
    ):
        """`objectives.cuh:139-148` / `:341-349`.

        THIS IS WHY THEIR TWO CONSTRUCTORS TAKE THE SAME FOUR ARGUMENTS
        even though the regression one never stores the first: `Builder`
        constructs `ObjectiveT` generically from `params`
        (`builder.cuh:592-596`), so the signature has to be uniform. The
        unnamed `IdxT` on the regression constructor at `:341` exists
        purely to make that one call compile for both families.

        Requiring it here is what lets our `Builder` do the same. Without
        it a `Builder` can only be handed a pre-built objective, and then
        `params.min_samples_leaf`, `params.split_criterion` and
        `params.min_impurity_decrease` have a second source of truth that
        can silently disagree with the one the caller set.

        `scales` is ours (DEVIATION 101b/112c) and has no counterpart.
        """
        ...

    def NumClasses(self) -> Int32:
        """`objectives.cuh:150` / `:361`."""
        ...

    def Scales(self) -> BinScales:
        """NOT theirs. DEVIATION 101b's fixed-point scales, which
        `SetLeafVector` needs and their two-argument static did not."""
        ...

    def IncrementHistogram[
        ho: MutOrigin, aspace: AddressSpace, //
    ](
        self,
        histogram: MutPointer[Self.BinT, ho, address_space=aspace],
        n_bins: Int32,
        bin: Int32,
        label: Scalar[Self.LabelT],
        dataset: DatasetView[Self.DataT, Self.LabelT],
        row: Int32,
    ):
        """`objectives.cuh:152-161` / `:370-379`."""
        ...

    def Gain[
        ho: MutOrigin,
        aspace: AddressSpace,
        qo: MutOrigin,
        qs: AddressSpace, //,
    ](
        self,
        shist: MutPointer[Self.BinT, ho, address_space=aspace],
        squantiles: MutPointer[Scalar[Self.DataT], qo, address_space=qs],
        col: Int32,
        len: Int64,
        n_bins: Int32,
    ) -> Split[Self.DataT]:
        """`objectives.cuh:163-177` / `:381-397`."""
        ...

    @staticmethod
    def SetLeafVector[
        ho: MutOrigin,
        aspace: AddressSpace,
        oo: MutOrigin,
        os: AddressSpace, //,
    ](
        shist: MutPointer[Self.BinT, ho, address_space=aspace],
        nclasses: Int32,
        out_ptr: MutPointer[Scalar[Self.DataT], oo, address_space=os],
        scales: BinScales,
    ):
        """`objectives.cuh:179-195` / `:399-408`."""
        ...


# ------------------------------------- ClassificationObjectiveFunction --
struct ClassificationObjectiveFunction[
    dtype: DType, label_dtype: DType, B: Bin
](ObjectiveLike) where dtype.is_floating_point():
    """`ML::DT::ClassificationObjectiveFunction`, `objectives.cuh:20-196`.

    Their `DataT_`/`LabelT_`/`IdxT_`/`weighted_` become `dtype`,
    `label_dtype`, a fixed Int32 (their IdxT is `int` in every
    instantiation, `decisiontree.cuh:255`) and `BinT` -- see DEVIATION
    112b for why the last one is a type here and a Bool there.
    """

    # `objectives.cuh:23-26` -- their `using DataT = DataT_;` etc.
    #
    # THESE ARE ASSOCIATED MEMBERS, NOT PARAMETERS, AND THE DIFFERENCE IS
    # THE WHOLE REASON THE THIRD PARAMETER IS SPELLED `B`. `ObjectiveLike`
    # requires a member named `BinT`; a struct PARAMETER named `BinT` does
    # not satisfy that, and Mojo says so ("required member 'BinT' is not
    # specified") while printing a candidate signature that looks
    # identical. Renaming the parameter to `B` and aliasing `BinT` to it
    # leaves every `Self.BinT` in this file resolving to the member the
    # trait asked for, with no other edit.
    comptime DataT = Self.dtype
    comptime LabelT = Self.label_dtype
    comptime BinT = Self.B

    # `objectives.cuh:27` -- `static constexpr bool weighted`, now read
    # off the bin instead of selecting it.
    comptime weighted = Self.BinT.weighted

    # `objectives.cuh:30` -- IdxT nclasses
    var nclasses: Int32
    # `objectives.cuh:31` -- IdxT min_samples_leaf
    var min_samples_leaf: Int32
    # `objectives.cuh:32` -- CRITERION criterion
    var criterion: Int32
    # `objectives.cuh:33` -- DataT min_impurity_decrease
    var min_impurity_decrease: Scalar[Self.dtype]
    # NOT in their class. DEVIATION 101b's fixed-point scales.
    var scales: BinScales

    @always_inline
    def __init__(
        out self,
        nclasses: Int32,
        min_samples_leaf: Int32,
        criterion: Int32,
        min_impurity_decrease: Scalar[Self.dtype] = 0,
        scales: BinScales = BinScales(1.0, 1.0),
    ):
        """`objectives.cuh:139-148`. Their `min_impurity_decrease`
        defaults to `DataT{0}` (`:142`); `scales` is ours and defaults to
        the unit scales of the unweighted path, which never reads them."""
        comptime assert Self.BinT.is_classification, (
            "ClassificationObjectiveFunction requires a classification bin"
            " (DEVIATION 112b)"
        )
        self.nclasses = nclasses
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.min_impurity_decrease = min_impurity_decrease
        self.scales = scales

    @always_inline
    def WeightAt[
        ho: MutOrigin, aspace: AddressSpace, //
    ](
        self,
        hist: MutPointer[Self.BinT, ho, address_space=aspace],
        i: Int32,
        n_bins: Int32,
    ) -> Scalar[Self.BinT.weight_dtype]:
        """`objectives.cuh:35-42`. Their accumulator is `double`; see
        DEVIATION 114 for why it is Int64 on the unweighted path."""
        var weight = Scalar[Self.BinT.weight_dtype](0)
        for j in range(Int(self.nclasses)):
            weight += hist[
                unsafe_offset = Int(n_bins) * j + Int(i)
            ].Weight(self.scales)
        return weight

    @always_inline
    def GiniGain[
        ho: MutOrigin, aspace: AddressSpace, //
    ](
        self,
        hist: MutPointer[Self.BinT, ho, address_space=aspace],
        i: Int32,
        n_bins: Int32,
        len: Int64,
        nLeft: Int64,
        nRight: Int64,
    ) -> Scalar[Self.dtype]:
        """`objectives.cuh:44-78`. Their last three parameters are unnamed
        -- Gini does not use them -- and are kept so the four gain
        signatures stay interchangeable, as theirs are."""
        var One = Scalar[Self.dtype](1.0)
        # `:48-50`
        var total_weight = self.WeightAt(hist, n_bins - 1, n_bins)
        var left_weight = self.WeightAt(hist, i, n_bins)
        var right_weight = total_weight - left_weight

        # `:52-53`
        if total_weight <= 0 or left_weight <= 0 or right_weight <= 0:
            return -Scalar[Self.dtype].MAX_FINITE

        # `:55-58`
        var invLen = One / total_weight.cast[Self.dtype]()
        var invLeft = One / left_weight.cast[Self.dtype]()
        var invRight = One / right_weight.cast[Self.dtype]()
        var gain = Scalar[Self.dtype](0.0)

        # `:60-75`
        for j in range(Int(self.nclasses)):
            var val_i = Scalar[Self.BinT.weight_dtype](0)
            var lval_i = hist[
                unsafe_offset = Int(n_bins) * j + Int(i)
            ].Weight(self.scales)
            var lval = lval_i.cast[Self.dtype]()
            gain += lval * invLeft * lval * invLen

            val_i += lval_i
            var total_sum = hist[
                unsafe_offset = Int(n_bins) * j + Int(n_bins) - 1
            ].Weight(self.scales)
            var rval_i = total_sum - lval_i
            var rval = rval_i.cast[Self.dtype]()
            gain += rval * invRight * rval * invLen

            val_i += rval_i
            var val = val_i.cast[Self.dtype]() * invLen
            gain -= val * val

        return gain

    @always_inline
    def EntropyGain[
        ho: MutOrigin, aspace: AddressSpace, //
    ](
        self,
        hist: MutPointer[Self.BinT, ho, address_space=aspace],
        i: Int32,
        n_bins: Int32,
        len: Int64,
        nLeft: Int64,
        nRight: Int64,
    ) -> Scalar[Self.dtype]:
        """`objectives.cuh:80-118`. See DEVIATION 113 for `raft::log`."""
        # `:83-85`
        var total_weight = self.WeightAt(hist, n_bins - 1, n_bins)
        var left_weight = self.WeightAt(hist, i, n_bins)
        var right_weight = total_weight - left_weight

        # `:87-88`
        if total_weight <= 0 or left_weight <= 0 or right_weight <= 0:
            return -Scalar[Self.dtype].MAX_FINITE

        # `:90-93`
        var gain = Scalar[Self.dtype](0.0)
        var invLeft = Scalar[Self.dtype](1.0) / left_weight.cast[
            Self.dtype
        ]()
        var invRight = Scalar[Self.dtype](1.0) / right_weight.cast[
            Self.dtype
        ]()
        var invLen = Scalar[Self.dtype](1.0) / total_weight.cast[
            Self.dtype
        ]()
        # `:94-115`
        for c in range(Int(self.nclasses)):
            var val_i = Scalar[Self.BinT.weight_dtype](0)
            var lval_i = hist[
                unsafe_offset = Int(n_bins) * c + Int(i)
            ].Weight(self.scales)
            if lval_i != 0:
                var lval = lval_i.cast[Self.dtype]()
                gain += (
                    log(lval * invLeft)
                    / log(Scalar[Self.dtype](2))
                    * lval
                    * invLen
                )

            val_i += lval_i
            var total_sum = hist[
                unsafe_offset = Int(n_bins) * c + Int(n_bins) - 1
            ].Weight(self.scales)
            var rval_i = total_sum - lval_i
            if rval_i != 0:
                var rval = rval_i.cast[Self.dtype]()
                gain += (
                    log(rval * invRight)
                    / log(Scalar[Self.dtype](2))
                    * rval
                    * invLen
                )

            val_i += rval_i
            if val_i != 0:
                var val = val_i.cast[Self.dtype]() * invLen
                gain -= val * log(val) / log(Scalar[Self.dtype](2))

        return gain

    @always_inline
    def GainPerSplit[
        ho: MutOrigin, aspace: AddressSpace, //
    ](
        self,
        hist: MutPointer[Self.BinT, ho, address_space=aspace],
        i: Int32,
        n_bins: Int32,
        len: Int64,
        nLeft: Int64,
        nRight: Int64,
    ) -> Scalar[Self.dtype]:
        """`objectives.cuh:121-137`, the criterion dispatch."""
        # `:128-130`
        if nLeft < Int64(Int(self.min_samples_leaf)) or nRight < Int64(
            Int(self.min_samples_leaf)
        ):
            return -Scalar[Self.dtype].MAX_FINITE

        # `:132-136`. Their `default:` arm returns the same sentinel, so a
        # regression criterion handed to a classification objective is a
        # rejected candidate rather than an error -- theirs, kept.
        if self.criterion == CRITERION_GINI:
            return self.GiniGain(hist, i, n_bins, len, nLeft, nRight)
        elif self.criterion == CRITERION_ENTROPY:
            return self.EntropyGain(hist, i, n_bins, len, nLeft, nRight)
        return -Scalar[Self.dtype].MAX_FINITE

    @always_inline
    def NumClasses(self) -> Int32:
        """`objectives.cuh:150`."""
        return self.nclasses

    @always_inline
    def Scales(self) -> BinScales:
        """NOT theirs. DEVIATION 101b's fixed-point scales, which
        `SetLeafVector` needs and their two-argument static did not.
        Required by `ObjectiveLike` so the kernels can reach them."""
        return self.scales

    @always_inline
    def IncrementHistogram[
        ho: MutOrigin, aspace: AddressSpace, //
    ](
        self,
        histogram: MutPointer[Self.BinT, ho, address_space=aspace],
        n_bins: Int32,
        bin: Int32,
        label: Scalar[Self.label_dtype],
        dataset: DatasetView[Self.dtype, Self.label_dtype],
        row: Int32,
    ):
        """`objectives.cuh:152-161`.

        Their `dataset.sample_weight == nullptr` test (`:158`) has no Mojo
        counterpart because `MutPointer` is non-null; `dataset.mojo`
        carries the null-ness as `has_sample_weight` and DEVIATION 100
        prices the Float32 width of the weight itself.
        """
        var weight = Float32(1.0)
        comptime if Self.weighted:
            if dataset.has_sample_weight:
                weight = dataset.sample_weight[unsafe_offset = Int(row)]
            else:
                weight = 1.0
        Self.BinT.IncrementHistogram(
            histogram, n_bins, bin, label, weight, self.scales
        )

    @always_inline
    def Gain[
        ho: MutOrigin, aspace: AddressSpace, qo: MutOrigin, qs: AddressSpace, //
    ](
        self,
        shist: MutPointer[Self.BinT, ho, address_space=aspace],
        squantiles: MutPointer[Scalar[Self.dtype], qo, address_space=qs],
        col: Int32,
        len: Int64,
        n_bins: Int32,
    ) -> Split[Self.dtype]:
        """`objectives.cuh:163-177`. One `Split` per thread, strided over
        the bins; the block reduction is `split.cuh`'s job."""
        var sp = Split[Self.dtype]()
        var i = Int32(Int(thread_idx.x))
        while i < n_bins:
            # `:168-169`
            var nLeft = count_left(shist, i, n_bins, self.nclasses)
            var nRight = len - nLeft
            # `:170-174`
            if nLeft >= Int64(Int(self.min_samples_leaf)) and nRight >= Int64(
                Int(self.min_samples_leaf)
            ):
                var gain = self.GainPerSplit(
                    shist, i, n_bins, len, nLeft, nRight
                )
                if gain > self.min_impurity_decrease:
                    _ = sp.update_bin(
                        squantiles[unsafe_offset = Int(i)],
                        col,
                        gain,
                        nLeft,
                        i,
                    )
            i += Int32(Int(block_dim.x))
        return sp

    @staticmethod
    @always_inline
    def SetLeafVector[
        ho: MutOrigin, aspace: AddressSpace, oo: MutOrigin, os: AddressSpace, //
    ](
        shist: MutPointer[Self.BinT, ho, address_space=aspace],
        nclasses: Int32,
        out_ptr: MutPointer[Scalar[Self.dtype], oo, address_space=os],
        scales: BinScales,
    ):
        """`objectives.cuh:179-195`, the class-probability leaf.

        `scales` is DEVIATION 101b's; theirs is a two-argument static and
        the division's width is DEVIATION 114's last bullet.
        """
        # `:181-185` -- Output probability
        var total = Scalar[Self.BinT.weight_dtype](0)
        for i in range(Int(nclasses)):
            total += shist[unsafe_offset=i].Weight(scales)
        # `:186-191`
        if total <= 0:
            for i in range(Int(nclasses)):
                out_ptr[unsafe_offset=i] = Scalar[Self.dtype](0)
            return
        # `:192-194`
        for i in range(Int(nclasses)):
            out_ptr[unsafe_offset=i] = (
                shist[unsafe_offset=i].Weight(scales).cast[Self.dtype]()
                / total.cast[Self.dtype]()
            )


# ----------------------------------------- RegressionObjectiveFunction --
struct RegressionObjectiveFunction[
    dtype: DType, label_dtype: DType, B: RegressionBinLike
](ObjectiveLike) where dtype.is_floating_point():
    """`ML::DT::RegressionObjectiveFunction`, `objectives.cuh:198-387`."""

    # `objectives.cuh:200-204`; see the note on the classification twin
    # for why the parameter is `B` and `BinT` is an alias to it.
    comptime DataT = Self.dtype
    comptime LabelT = Self.label_dtype
    comptime BinT = Self.B

    # `objectives.cuh:205`
    comptime weighted = Self.BinT.weighted

    # `objectives.cuh:208` -- IdxT min_samples_leaf
    var min_samples_leaf: Int32
    # `objectives.cuh:209` -- CRITERION criterion
    var criterion: Int32
    # `objectives.cuh:210` -- DataT min_impurity_decrease
    var min_impurity_decrease: Scalar[Self.dtype]
    # NOT in their class. DEVIATION 101b's fixed-point scales.
    var scales: BinScales

    # `objectives.cuh:211` -- `10 * numeric_limits<DataT>::epsilon()`.
    #
    # THIS WAS `10 * (nextafter(1, 2) - 1)` AND IT CRASHED THE METAL
    # BACKEND. Not "was slow", not "was imprecise": the first regression
    # kernel ever instantiated failed with "Metal Compiler failed to
    # compile metallib", no line number and no symbol. `nextafter` is a
    # libm-shaped call and does not survive into device code. Isolated to a
    # ten-line kernel that does nothing but call it, so this is measured
    # rather than inferred.
    #
    # HOW IT WAS FOUND, because the shape recurs: `MSEGain` compiled and
    # ran while `PoissonGain`, `GammaGain` and `InverseGaussianGain` all
    # crashed. The only thing those three share and MSE does not is this
    # constant -- and the third of them does not call `log` at all, which
    # is what ruled `log` out. Making it a `comptime` did NOT help; the
    # call still reached the device.
    #
    # So the value is written as the IEEE-754 definition instead:
    # `numeric_limits<T>::epsilon()` is 2^-(mantissa bits), exactly, and
    # each literal below is an exact power of two and therefore exactly
    # representable. `objectives_check` holds them to `nextafter` ON THE
    # HOST, where it works, so this is checked against the definition it
    # replaced rather than trusted.
    comptime eps_ = Scalar[Self.dtype](10) * machine_epsilon[Self.dtype]()

    @always_inline
    def __init__(
        out self,
        nclasses: Int32,
        min_samples_leaf: Int32,
        criterion: Int32,
        min_impurity_decrease: Scalar[Self.dtype] = 0,
        scales: BinScales = BinScales(1.0, 1.0),
    ):
        """`objectives.cuh:341-349`. Their first parameter is UNNAMED --
        the regression objective takes an `IdxT` it never stores, so that
        the two objectives construct identically from `Builder`. Kept, and
        kept unused."""
        comptime assert not Self.BinT.is_classification, (
            "RegressionObjectiveFunction requires a regression bin"
            " (DEVIATION 112b)"
        )
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.min_impurity_decrease = min_impurity_decrease
        self.scales = scales

    @always_inline
    def MSEGain[
        ho: MutOrigin, aspace: AddressSpace, //
    ](
        self,
        hist: MutPointer[Self.BinT, ho, address_space=aspace],
        i: Int32,
        n_bins: Int32,
        len: Int64,
        nLeft: Int64,
        nRight: Int64,
    ) -> Scalar[Self.dtype]:
        """`objectives.cuh:213-234`."""
        # `:216-218`
        var parent_weight = hist[
            unsafe_offset = Int(n_bins) - 1
        ].Weight(self.scales)
        var left_weight = hist[unsafe_offset = Int(i)].Weight(self.scales)
        var right_weight = parent_weight - left_weight

        # `:220-221`
        if parent_weight <= 0 or left_weight <= 0 or right_weight <= 0:
            return -Scalar[Self.dtype].MAX_FINITE

        # `:223-231`
        var invLen = Scalar[Self.dtype](1.0) / parent_weight.cast[
            Self.dtype
        ]()
        var label_sum = hist[unsafe_offset = Int(n_bins) - 1].LabelSum(
            self.scales
        ).cast[Self.dtype]()
        var left_label_sum = hist[unsafe_offset = Int(i)].LabelSum(
            self.scales
        ).cast[Self.dtype]()
        var parent_obj = -label_sum * label_sum * invLen
        var left_obj = -(left_label_sum * left_label_sum) / left_weight.cast[
            Self.dtype
        ]()
        var right_label_sum = label_sum - left_label_sum
        var right_obj = -(
            right_label_sum * right_label_sum
        ) / right_weight.cast[Self.dtype]()
        var gain = parent_obj - (left_obj + right_obj)
        gain *= Scalar[Self.dtype](0.5) * invLen

        return gain

    @always_inline
    def PoissonGain[
        ho: MutOrigin, aspace: AddressSpace, //
    ](
        self,
        hist: MutPointer[Self.BinT, ho, address_space=aspace],
        i: Int32,
        n_bins: Int32,
        len: Int64,
        nLeft: Int64,
        nRight: Int64,
    ) -> Scalar[Self.dtype]:
        """`objectives.cuh:236-262`. See DEVIATION 113 for `raft::log`."""
        # `:239-241`
        var parent_weight = hist[
            unsafe_offset = Int(n_bins) - 1
        ].Weight(self.scales)
        var left_weight = hist[unsafe_offset = Int(i)].Weight(self.scales)
        var right_weight = parent_weight - left_weight

        # `:243-244`
        if parent_weight <= 0 or left_weight <= 0 or right_weight <= 0:
            return -Scalar[Self.dtype].MAX_FINITE

        # `:246-249`
        var invLen = Scalar[Self.dtype](1) / parent_weight.cast[Self.dtype]()
        var label_sum = hist[unsafe_offset = Int(n_bins) - 1].LabelSum(
            self.scales
        ).cast[Self.dtype]()
        var left_label_sum = hist[unsafe_offset = Int(i)].LabelSum(
            self.scales
        ).cast[Self.dtype]()
        var right_label_sum = (
            hist[unsafe_offset = Int(n_bins) - 1].LabelSum(self.scales)
            - hist[unsafe_offset = Int(i)].LabelSum(self.scales)
        ).cast[Self.dtype]()

        # `:251-253` -- label sum cannot be non-positive
        if (
            label_sum <= Self.eps_
            or left_label_sum <= Self.eps_
            or right_label_sum <= Self.eps_
        ):
            return -Scalar[Self.dtype].MAX_FINITE

        # `:255-259`
        var parent_obj = -label_sum * log(label_sum * invLen)
        var left_obj = -left_label_sum * log(
            left_label_sum / left_weight.cast[Self.dtype]()
        )
        var right_obj = -right_label_sum * log(
            right_label_sum / right_weight.cast[Self.dtype]()
        )
        var gain = parent_obj - (left_obj + right_obj)
        gain = gain * invLen

        return gain

    @always_inline
    def GammaGain[
        ho: MutOrigin, aspace: AddressSpace, //
    ](
        self,
        hist: MutPointer[Self.BinT, ho, address_space=aspace],
        i: Int32,
        n_bins: Int32,
        len: Int64,
        nLeft: Int64,
        nRight: Int64,
    ) -> Scalar[Self.dtype]:
        """`objectives.cuh:264-290`. See DEVIATION 113 for `raft::log`."""
        # `:267-269`
        var parent_weight = hist[
            unsafe_offset = Int(n_bins) - 1
        ].Weight(self.scales)
        var left_weight = hist[unsafe_offset = Int(i)].Weight(self.scales)
        var right_weight = parent_weight - left_weight

        # `:271-272`
        if parent_weight <= 0 or left_weight <= 0 or right_weight <= 0:
            return -Scalar[Self.dtype].MAX_FINITE

        # `:274-277`
        var invLen = Scalar[Self.dtype](1) / parent_weight.cast[Self.dtype]()
        var label_sum = hist[unsafe_offset = Int(n_bins) - 1].LabelSum(
            self.scales
        ).cast[Self.dtype]()
        var left_label_sum = hist[unsafe_offset = Int(i)].LabelSum(
            self.scales
        ).cast[Self.dtype]()
        var right_label_sum = (
            hist[unsafe_offset = Int(n_bins) - 1].LabelSum(self.scales)
            - hist[unsafe_offset = Int(i)].LabelSum(self.scales)
        ).cast[Self.dtype]()

        # `:279-281` -- label sum cannot be non-positive
        if (
            label_sum <= Self.eps_
            or left_label_sum <= Self.eps_
            or right_label_sum <= Self.eps_
        ):
            return -Scalar[Self.dtype].MAX_FINITE

        # `:283-287`
        var parent_obj = parent_weight.cast[Self.dtype]() * log(
            label_sum * invLen
        )
        var left_obj = left_weight.cast[Self.dtype]() * log(
            left_label_sum / left_weight.cast[Self.dtype]()
        )
        var right_obj = right_weight.cast[Self.dtype]() * log(
            right_label_sum / right_weight.cast[Self.dtype]()
        )
        var gain = parent_obj - (left_obj + right_obj)
        gain = gain * invLen

        return gain

    @always_inline
    def InverseGaussianGain[
        ho: MutOrigin, aspace: AddressSpace, //
    ](
        self,
        hist: MutPointer[Self.BinT, ho, address_space=aspace],
        i: Int32,
        n_bins: Int32,
        len: Int64,
        nLeft: Int64,
        nRight: Int64,
    ) -> Scalar[Self.dtype]:
        """`objectives.cuh:292-317`. No transcendental -- unaffected by
        DEVIATION 113."""
        # `:295-297`
        var parent_weight = hist[
            unsafe_offset = Int(n_bins) - 1
        ].Weight(self.scales)
        var left_weight = hist[unsafe_offset = Int(i)].Weight(self.scales)
        var right_weight = parent_weight - left_weight

        # `:299-300`
        if parent_weight <= 0 or left_weight <= 0 or right_weight <= 0:
            return -Scalar[Self.dtype].MAX_FINITE

        # `:302-304`
        var label_sum = hist[unsafe_offset = Int(n_bins) - 1].LabelSum(
            self.scales
        ).cast[Self.dtype]()
        var left_label_sum = hist[unsafe_offset = Int(i)].LabelSum(
            self.scales
        ).cast[Self.dtype]()
        var right_label_sum = (
            hist[unsafe_offset = Int(n_bins) - 1].LabelSum(self.scales)
            - hist[unsafe_offset = Int(i)].LabelSum(self.scales)
        ).cast[Self.dtype]()

        # `:306-308` -- label sum cannot be non-positive
        if (
            label_sum <= Self.eps_
            or left_label_sum <= Self.eps_
            or right_label_sum <= Self.eps_
        ):
            return -Scalar[Self.dtype].MAX_FINITE

        # `:310-314`
        var parent_obj = -parent_weight.cast[Self.dtype]() * parent_weight.cast[
            Self.dtype
        ]() / label_sum
        var left_obj = -left_weight.cast[Self.dtype]() * left_weight.cast[
            Self.dtype
        ]() / left_label_sum
        var right_obj = -right_weight.cast[Self.dtype]() * right_weight.cast[
            Self.dtype
        ]() / right_label_sum
        var gain = parent_obj - (left_obj + right_obj)
        gain = gain / (Scalar[Self.dtype](2) * parent_weight.cast[Self.dtype]())

        return gain

    @always_inline
    def GainPerSplit[
        ho: MutOrigin, aspace: AddressSpace, //
    ](
        self,
        hist: MutPointer[Self.BinT, ho, address_space=aspace],
        i: Int32,
        n_bins: Int32,
        len: Int64,
        nLeft: Int64,
        nRight: Int64,
    ) -> Scalar[Self.dtype]:
        """`objectives.cuh:320-339`, the criterion dispatch."""
        # `:327-329`
        if nLeft < Int64(Int(self.min_samples_leaf)) or nRight < Int64(
            Int(self.min_samples_leaf)
        ):
            return -Scalar[Self.dtype].MAX_FINITE

        # `:331-337`. NOTE their `MAE` (`algo_helper.h:14`) has no arm and
        # falls to `default:` -- cuML does not implement MAE on GPU. Kept.
        if self.criterion == CRITERION_MSE:
            return self.MSEGain(hist, i, n_bins, len, nLeft, nRight)
        elif self.criterion == CRITERION_POISSON:
            return self.PoissonGain(hist, i, n_bins, len, nLeft, nRight)
        elif self.criterion == CRITERION_GAMMA:
            return self.GammaGain(hist, i, n_bins, len, nLeft, nRight)
        elif self.criterion == CRITERION_INVERSE_GAUSSIAN:
            return self.InverseGaussianGain(
                hist, i, n_bins, len, nLeft, nRight
            )
        return -Scalar[Self.dtype].MAX_FINITE

    @always_inline
    def NumClasses(self) -> Int32:
        """`objectives.cuh:351` -- `return 1`."""
        return 1

    @always_inline
    def Scales(self) -> BinScales:
        """NOT theirs. DEVIATION 101b's fixed-point scales, which
        `SetLeafVector` needs and their two-argument static did not.
        Required by `ObjectiveLike` so the kernels can reach them."""
        return self.scales

    @always_inline
    def IncrementHistogram[
        ho: MutOrigin, aspace: AddressSpace, //
    ](
        self,
        histogram: MutPointer[Self.BinT, ho, address_space=aspace],
        n_bins: Int32,
        bin: Int32,
        label: Scalar[Self.label_dtype],
        dataset: DatasetView[Self.dtype, Self.label_dtype],
        row: Int32,
    ):
        """`objectives.cuh:353-362`. Identical to the classification one;
        theirs is duplicated in the two classes too."""
        var weight = Float32(1.0)
        comptime if Self.weighted:
            if dataset.has_sample_weight:
                weight = dataset.sample_weight[unsafe_offset = Int(row)]
            else:
                weight = 1.0
        Self.BinT.IncrementHistogram(
            histogram, n_bins, bin, label, weight, self.scales
        )

    @always_inline
    def Gain[
        ho: MutOrigin, aspace: AddressSpace, qo: MutOrigin, qs: AddressSpace, //
    ](
        self,
        shist: MutPointer[Self.BinT, ho, address_space=aspace],
        squantiles: MutPointer[Scalar[Self.dtype], qo, address_space=qs],
        col: Int32,
        len: Int64,
        n_bins: Int32,
    ) -> Split[Self.dtype]:
        """`objectives.cuh:364-378`. Their `CountLeft` here is called with
        `IdxT{1}` (`:369`), not `nclasses`: a regression histogram has one
        plane."""
        var sp = Split[Self.dtype]()
        var i = Int32(Int(thread_idx.x))
        while i < n_bins:
            # `:369-370`
            var nLeft = count_left(shist, i, n_bins, Int32(1))
            var nRight = len - nLeft
            # `:371-375`
            if nLeft >= Int64(Int(self.min_samples_leaf)) and nRight >= Int64(
                Int(self.min_samples_leaf)
            ):
                var gain = self.GainPerSplit(
                    shist, i, n_bins, len, nLeft, nRight
                )
                if gain > self.min_impurity_decrease:
                    _ = sp.update_bin(
                        squantiles[unsafe_offset = Int(i)],
                        col,
                        gain,
                        nLeft,
                        i,
                    )
            i += Int32(Int(block_dim.x))
        return sp

    @staticmethod
    @always_inline
    def SetLeafVector[
        ho: MutOrigin, aspace: AddressSpace, oo: MutOrigin, os: AddressSpace, //
    ](
        shist: MutPointer[Self.BinT, ho, address_space=aspace],
        nclasses: Int32,
        out_ptr: MutPointer[Scalar[Self.dtype], oo, address_space=os],
        scales: BinScales,
    ):
        """`objectives.cuh:380-386`, the mean-label leaf."""
        # `:382-385`
        for i in range(Int(nclasses)):
            var weight = shist[unsafe_offset=i].Weight(scales)
            if weight > 0:
                out_ptr[unsafe_offset=i] = shist[unsafe_offset=i].LabelSum(
                    scales
                ).cast[Self.dtype]() / weight.cast[Self.dtype]()
            else:
                out_ptr[unsafe_offset=i] = Scalar[Self.dtype](0)
