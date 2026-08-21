"""Split-scoring objectives: Gini for classification, MSE for regression.

WHAT THIS FILE IS, AND WHAT IT IS NOT
-------------------------------------
The split RULE this file serves is a PORT FROM PAPER — Geurts, Ernst &
Wehenkel 2006, "Extremely randomized trees" — by way of scikit-learn's
`RandomSplitter`. It is **not** a port of a cuML kernel: no GPU library ships
the histogram-free formulation (`extratrees/PLAN.md` re-verified this on
2026-08-21). Two upstreams are therefore mirrored for two different things,
and every function below says which:

- **cuML** `00094f7` (`~/CascadeProjects/upstream/cuml`) is mirrored for
  SHAPE and NAMES:
  `cpp/src/decisiontree/batched-levelalgo/objectives.cuh` (504 lines) and
  `cpp/src/decisiontree/batched-levelalgo/bins.cuh` (76 lines). Their symbol
  names are kept verbatim (`GiniObjectiveFunction`, `MSEObjectiveFunction`,
  `GainPerSplit`, `SetLeafVector`, `NumClasses`, `CountBin`, `AggregateBin`,
  `BinT`, `DataT`, `LabelT`, `IdxT`) so the two trees diff side by side
  (rule 6).
- **scikit-learn** `1.9.0` (`77def0e`,
  `~/CascadeProjects/upstream/scikit-learn`) is mirrored for ARITHMETIC:
  `sklearn/tree/_criterion.pyx`. Their expressions are transcribed in their
  form and their order of operations. They are NOT algebraically simplified,
  because float arithmetic is not algebra and this repository has already
  been bitten once by a compiler contracting an FMA across a simplification
  (`~/.claude/.../mojo-log-breaks-ties.md`).

PROVENANCE, FUNCTION BY FUNCTION
--------------------------------
| this file                                    | cuML `objectives.cuh` | sklearn `_criterion.pyx` |
|----------------------------------------------|-----------------------|--------------------------|
| `CountBin`                                   | `bins.cuh:22-44`      | --                       |
| `AggregateBin`                               | `bins.cuh:46-74`      | --                       |
| `proxy_impurity_improvement`                 | --                    | `147-163`                |
| `impurity_improvement`                       | --                    | `165-199`                |
| `GiniObjectiveFunction.GainPerSplit`         | `52-83`               | --                       |
| `GiniObjectiveFunction.NodeImpurity`         | --                    | `622-645`                |
| `GiniObjectiveFunction.ChildrenImpurity`     | --                    | `647-687`                |
| `GiniObjectiveFunction.ProxyImpurityImprovement` | --                | `147-163` + `647-687`    |
| `GiniObjectiveFunction.ProxyImpurityExact`   | --                    | `147-163` + `647-687`    |
| `GiniObjectiveFunction.CompareProxyExact`    | --                    | (exact form of the above)|
| `GiniObjectiveFunction.SetLeafVector`        | `97-107`              | --                       |
| `MSEObjectiveFunction.GainPerSplit`          | `225-244`             | --                       |
| `MSEObjectiveFunction.NodeImpurity`          | --                    | `928-942`                |
| `MSEObjectiveFunction.ProxyImpurityImprovement` | --                 | `944-973`                |
| `MSEObjectiveFunction.ChildrenImpurity`      | --                    | `975-1017`               |
| `MSEObjectiveFunction.SetLeafVector`         | `259-264`             | --                       |
| `MSEObjectiveFunction.NumClasses`            | `257`                 | --                       |

THE ONE THING READING BOTH SOURCES CHANGES
------------------------------------------
**cuML's Gini and sklearn's Gini are not the same quantity.** From memory one
would write "the Gini gain" and use it for both. They differ:

- cuML `objectives.cuh:65-80` accumulates
  `sum_j [ l_j*(1/nL)*l_j*(1/n) + r_j*(1/nR)*r_j*(1/n) ] - sum_j (t_j/n)^2`,
  which is the Gini impurity DECREASE, `parent_gini - weighted children gini`.
- sklearn selects on `Criterion.proxy_impurity_improvement`
  (`_criterion.pyx:147-163`), `-wR*imp_R - wL*imp_L`, which for
  `Gini.children_impurity` (`:647-687`) is
  `sq_L/nL + sq_R/nR - n`. It drops the parent term and the `1/n` scale.

Within one node `n` and `parent_gini` are constants, so
`cuML_gain == parent_gini + sklearn_proxy / n` and the two agree on the
ARGMAX. They do not agree on the VALUE, they do not agree on which pairs
round to equality in float, and only one of them is exact in integers. Both
are shipped, and DEVIATION 144 says which one decides.

A second surprise, in cuML's MSE: `objectives.cuh:237` writes
`right_label_sum = hist[i].label_sum - label_sum`, which is the NEGATIVE of
the right child's label sum (`left - total`, not `total - left`). It is
squared on the next line so the result is unaffected. It is transcribed as
written rather than "fixed", because a transcription that silently corrects
its source stops being a diffable transcription.

NOT PORTED HERE, DELIBERATELY
-----------------------------
- `GiniObjectiveFunction::Gain` / `MSEObjectiveFunction::Gain`
  (`objectives.cuh:85-96`, `:246-255`) — the `threadIdx.x`-strided loop over
  BINS that turns per-bin gains into a `Split`. There are no bins here
  (DEVIATION 143), and the reduction over candidates belongs with the
  builder kernels, not with the scoring arithmetic.
- `CountBin::IncrementHistogram` / `AtomicAdd` (`bins.cuh:28-33`) and
  `AggregateBin::IncrementHistogram` / `AtomicAdd` (`bins.cuh:54-62`) — the
  device histogram-scatter ops. Histogram-free: the accumulators are a
  handful of registers per candidate, not a scattered array.
- `EntropyObjectiveFunction` (`:110-193`), `PoissonObjectiveFunction`
  (`:267-346`), `GammaObjectiveFunction` (`:348-424`),
  `InverseGaussianObjectiveFunction` (`:426-502`) — sklearn's ExtraTrees
  defaults are `criterion='gini'` and `criterion='squared_error'`, and rule 3
  says an unported thing must be visible rather than half-present.

HOST/DEVICE DISCIPLINE
----------------------
Every scoring function here is a plain function over scalars and non-owning
pointers. Nothing allocates, nothing raises, nothing touches `List`, `String`
or any other host-only type, so each is callable unchanged from a kernel.
"""


# ==========================================================================
# DEVIATION BLOCK 143 -- histogram-free: the accumulators ARE the arguments
#
# THEIRS (cuML). Every objective takes `(BinT* hist, IdxT i, IdxT n_bins,
#   IdxT len, IdxT nLeft)` (`objectives.cuh:52`, `:132`, `:225`, `:299`,
#   `:379`). `hist` is a PREFIX-SUMMED histogram laid out `n_bins * class + b`;
#   the left child is bin `i` of that CDF and the right child is recovered as
#   `hist[n_bins*c + n_bins - 1] - hist[n_bins*c + i]`, i.e. total minus left
#   (`:72-73`). `n_bins` exists only to index that layout.
# OURS. `(BinT* hist_left, BinT* hist_total, IdxT len, IdxT n_left)`. The two
#   pointers are the left child's accumulators and the node's totals; the
#   right child is still recovered as total minus left, exactly as theirs is.
#   `i` and `n_bins` are gone.
# WHY. This directory exists to delete the histogram (`extratrees/PLAN.md`,
#   and `quantiles.cuh` is permanently absent per `UNPORTED.tsv`). A random
#   threshold is drawn per (node, feature) from the feature's range over that
#   node's rows; there is no bin index to pass and no bin dimension to stride.
#   Keeping their parameters would mean passing `i = 0` and `n_bins = 1`
#   forever, which is a lie in the signature.
# PRICE. The signature does not diff against theirs line for line; the BODY
#   still does, which is where the arithmetic lives. `Gain` (`:85-96`) has no
#   counterpart at all, because it is the loop over the dimension we deleted.
# ==========================================================================


# ==========================================================================
# DEVIATION BLOCK 144 -- classification selects on an EXACT INTEGER proxy
#
# THEIRS. Two different scores, neither of them integer:
#   - cuML `GainPerSplit` (`objectives.cuh:52-83`) computes the Gini impurity
#     DECREASE in `DataT` (float32 or float64) and the builder reduces on it.
#   - sklearn's splitter (`_splitter.pyx:693`, inside `node_split_random`,
#     `:507`) compares
#     `Criterion.proxy_impurity_improvement` (`_criterion.pyx:147-163`),
#     `-wR*imp_R - wL*imp_L`, accumulated in `float64` throughout
#     (`sum_left`/`sum_right` are `float64_t[::1]`).
# OURS. `ProxyImpurityExact` returns sklearn's proxy as an exact rational in
#   INTEGER arithmetic -- numerator `sq_L*nR + sq_R*nL`, denominator `nL*nR`
#   -- and `CompareProxyExact` orders two candidates by cross-multiplication
#   in `Int128`. The float forms (`GainPerSplit`, `ProxyImpurityImprovement`,
#   `ChildrenImpurity`) are kept and are exact transcriptions, for REPORTING.
# WHY. There is no `float64` on device in this project (the root traps
#   register; `mojolearn-hardware-limits`), and DEVIATION 135 leaves device
#   accumulation precision OPEN for regression. For classification the
#   question does not have to be open at all: with sklearn's default
#   `sample_weight=None` the class counts are INTEGERS, so
#   `-wR*imp_R - wL*imp_L` is an exact rational, and the device can be
#   EXACTLY right instead of approximately right. Getting the argmax exactly
#   right also removes a whole class of "the GPU picked a different split"
#   investigation from the identity work.
#   The proxy is dropped by the constant `-len` (`sq_L/nL + sq_R/nR - n`);
#   `len` is the same for every candidate in one node, so the ORDER is
#   sklearn's order. `Proxy.value()` adds it back for reporting.
# ALSO. The exact form guards the EMPTY CHILD (`nL == 0` or `nR == 0`), which
#   cuML's float form does not: with `min_samples_leaf == 0` their
#   `invLeft = One / nLeft` (`objectives.cuh:57`) is `+inf` and their gain is
#   `inf` or `NaN`. Ours marks the candidate invalid. sklearn cannot reach the
#   case: `min_samples_leaf` is `>= 1` by validation
#   (`_classes.py:108-111`, `Interval(Integral, 1, None, closed="left")`).
# PRICE. Weighted samples are out of scope for the exact path -- with real
#   `sample_weight` the counts stop being integers and only the float form
#   applies. `sample_weight` is not ported and nothing reaches that path
#   today. **It is not listed in `extratrees/UNPORTED.tsv` either, which rule
#   3 says it should be; that row is an OPEN item for whoever owns that file,
#   because this lane may not edit it.**
# ==========================================================================


# ==========================================================================
# DEVIATION BLOCK 145 -- the exact comparator is the authority, not `Split`
#
# THEIRS. cuML reduces candidates through `Split::update`
# (`split.cuh:76-90`), whose first test is `other.best_metric_val >
#   this->best_metric_val` on a single `DataT` field. The score IS the
#   reduction key.
# OURS. For CLASSIFICATION the reduction key and the score are different
#   things. `Split.best_metric_val` stays `Float32` and stays cuML's
#   `GainPerSplit` value, for reporting and for `feature_importances_`; the
#   candidate ORDER is `CompareProxyExact`. A builder that reduces
#   classification candidates on the float field alone is using the wrong
#   comparator, and this block is the statement of that contract.
# WHY. Two candidates whose exact proxies differ can round to the same
#   `Float32` (24 bits of mantissa against counts that reach 2^26), and then
#   `Split::update` falls through to its `colid` tie-break and picks by
#   feature index. That is a silent, data-dependent divergence from sklearn's
#   argmax; DEVIATION 133 already accepted a different tie-break for GENUINE
#   ties, which is a much smaller claim than accepting one for near-ties.
# PRICE. The classification reduction carries the four accumulator fields
#   (`sq_L`, `nL`, `sq_R`, `nR`) alongside the `Split`, not just the score.
#   Cost unmeasured, deliberately: this lane is not taking timing numbers yet
#   (`extratrees/PLAN.md`).
# ==========================================================================


comptime MAX_ROWS_EXACT: Int = 1 << 26
"""Row count per node at which `CompareProxyExact` is still exact in `Int128`.

The widths, tightly. With `n` rows in the node and integer class counts,
`sq_L = sum_j l_j^2 <= (sum_j l_j)^2 = nL^2 <= n^2`, so `sq_L` and `sq_R` fit
`Int64` for any `n <= 2^26` (`n^2 <= 2^52`).

The comparison is the one that gets wide. `CompareProxyExact` cross-multiplies
`num_a * den_b` against `num_b * den_a`, and

    num = sq_L*nR + sq_R*nL <= nL^2*nR + nR^2*nL = nL*nR*n <= n^3/4
    den = nL*nR                                            <= n^2/4

so each product is bounded by `n^5 / 16`. **That is `n^5`, not the `n^3` a
count of the numerator alone suggests** -- the numerator is `n^3`, the
cross-multiply squares one more factor of `n^2` onto it. `Int128` holds
`2^127 - 1`, and `n^5/16 <= 2^127` gives `n <= 2^26.2`, so `2^26` is the
largest power of two that is safely inside it: at `n = 2^26` the product is at
most `2^126`, one full bit of headroom.

67,108,864 rows IN ONE NODE. The root node is the largest, so this is a bound
on the training set, and it is above the `Int32` row ids the dataset view uses
anyway (`dataset.mojo`). Past this point the correct answer is a wider
comparison, not a float one.

`ProxyImpurityExact` carries a `debug_assert` for a caller who exceeds it.
That assert is COMPILED OUT by default and only exists under
`mojo run -D ASSERT=all`; it was run that way once and seen to fire with this
message, which is the only reason to believe it is reachable at all
(`mojotrees-verify-reach-not-output`).
"""


# ==========================================================================
# The bin types. `bins.cuh:22-74`, kept as the accumulator element types.
# ==========================================================================


@fieldwise_init
struct CountBin(ImplicitlyCopyable, Movable):
    """One class count. `bins.cuh:22-44`.

    Theirs is `struct CountBin { int x; }`. `x` keeps their name because `x`
    is what every objective in `objectives.cuh` reads (`:67`, `:72`, `:91`,
    `:102`). With `sample_weight=None` this is a true count, which is the
    whole basis of DEVIATION 144.
    """

    var x: Int32
    """Theirs is `int x`."""

    def __init__(out self):
        """`bins.cuh:26`: `HDI CountBin() : x(0) {}`."""
        self.x = 0

    def __iadd__(mut self, b: Self):
        """`bins.cuh:34-38`, their `operator+=`."""
        self.x += b.x

    def __add__(self, b: Self) -> Self:
        """`bins.cuh:39-43`, their `operator+`."""
        return Self(self.x + b.x)


@fieldwise_init
struct AggregateBin[dtype: DType](ImplicitlyCopyable, Movable):
    """A label sum and a count. `bins.cuh:46-74`.

    THE ACCUMULATOR TYPE IS A PARAMETER ON PURPOSE. Theirs is hard-coded
    `double label_sum` (`bins.cuh:47`) and sklearn accumulates in `float64`
    throughout. Neither is available on device here, and choosing between
    `Float32`, a Kahan pair, and the fixed-point label scaling `gbdt/` already
    uses (`mojo_only/fixed_point.mojo`, `choose_scale`) is **DEVIATION 135,
    which is OPEN** -- see `extratrees/DEVIATIONS.md`. This file must not
    settle it, so it does not name a type: instantiate
    `AggregateBin[DType.float64]` on the host oracle and whatever 135 decides
    on the device, and the arithmetic below is identical either way.
    """

    var label_sum: Scalar[Self.dtype]
    """Theirs is `double label_sum`."""

    var count: Int32
    """Theirs is `int count`."""

    def __init__(out self):
        """`bins.cuh:51`: `HDI AggregateBin() : label_sum(0.0), count(0) {}`."""
        self.label_sum = 0
        self.count = 0

    def __iadd__(mut self, b: Self):
        """`bins.cuh:63-68`, their `operator+=`."""
        self.label_sum += b.label_sum
        self.count += b.count

    def __add__(self, b: Self) -> Self:
        """`bins.cuh:69-73`, their `operator+`."""
        return Self(self.label_sum + b.label_sum, self.count + b.count)


# ==========================================================================
# The two criterion-independent expressions from sklearn's BASE class.
# ==========================================================================


def proxy_impurity_improvement[
    dtype: DType
](
    weighted_n_left: Scalar[dtype],
    weighted_n_right: Scalar[dtype],
    impurity_left: Scalar[dtype],
    impurity_right: Scalar[dtype],
) -> Scalar[dtype]:
    """sklearn `Criterion.proxy_impurity_improvement`, `_criterion.pyx:147-163`.

    Their body, verbatim, after `children_impurity` has filled the two
    impurities:

        return (- self.weighted_n_right * impurity_right
                - self.weighted_n_left * impurity_left)

    Their term order is kept: RIGHT first. It is a sum of two negatives, so
    the order is not algebraically load-bearing, but it is the order the
    rounding happens in.
    """
    return -weighted_n_right * impurity_right - weighted_n_left * impurity_left


def impurity_improvement[
    dtype: DType
](
    impurity_parent: Scalar[dtype],
    impurity_left: Scalar[dtype],
    impurity_right: Scalar[dtype],
    weighted_n_node_samples: Scalar[dtype],
    weighted_n_samples: Scalar[dtype],
    weighted_n_left: Scalar[dtype],
    weighted_n_right: Scalar[dtype],
) -> Scalar[dtype]:
    """sklearn `Criterion.impurity_improvement`, `_criterion.pyx:165-199`.

    The FINAL reported improvement for the split that was chosen, which is a
    different quantity from the proxy that chose it. Their body, verbatim:

        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity_parent - (self.weighted_n_right /
                                    self.weighted_n_node_samples * impurity_right)
                                 - (self.weighted_n_left /
                                    self.weighted_n_node_samples * impurity_left)))

    Note `weighted_n_samples` is the WHOLE TREE's weight, not the node's --
    this is the term that makes `feature_importances_` sum to one. Their
    parenthesisation and their division-before-multiplication are preserved:
    `wR / w_node * imp_R` is `(wR / w_node) * imp_R`, not `wR / (w_node *
    imp_R)` and not `wR * imp_R / w_node`.
    """
    return (weighted_n_node_samples / weighted_n_samples) * (
        impurity_parent
        - (weighted_n_right / weighted_n_node_samples * impurity_right)
        - (weighted_n_left / weighted_n_node_samples * impurity_left)
    )


# ==========================================================================
# Classification.
# ==========================================================================


@fieldwise_init
struct GiniProxyExact(ImplicitlyCopyable, Movable):
    """sklearn's Gini proxy held as an exact rational. OURS, DEVIATION 144.

    The proxy is
        `-wR*gini_R - wL*gini_L` = `sq_L/nL + sq_R/nR - n`
    and with integer class counts every one of `sq_L`, `nL`, `sq_R`, `nR`, `n`
    is an integer, so the whole thing is `num/den - n` with

        num = sq_L*nR + sq_R*nL
        den = nL*nR

    `-n` is the same for every candidate in a node and is therefore dropped
    from the comparison and added back only by `value()`.
    """

    var num: Int64
    """`sq_L * nR + sq_R * nL`. Bounded by `n^3/4`; see `MAX_ROWS_EXACT`."""

    var den: Int64
    """`nL * nR`. Bounded by `n^2/4`. Zero only when a child is empty, and
    then `valid` is False and `den` is never used as a denominator."""

    var length: Int64
    """`len`, the node's row count. Only `value()` uses it."""

    var valid: Bool
    """False when the candidate is rejected: `min_samples_leaf` (cuML
    `objectives.cuh:62-63`) or an empty child (ours, DEVIATION 144). A
    rejected candidate orders below every accepted one, which is what cuML's
    `-std::numeric_limits<DataT>::max()` return achieves in float."""

    def value[dtype: DType](self) -> Scalar[dtype]:
        """The proxy as a float, for reporting only. Never for ordering.

        `num/den - len` is NOT the expression sklearn evaluates
        (`_criterion.pyx:147-163` evaluates `-wR*imp_R - wL*imp_L`); it is the
        same number reached by a shorter route, and the two round differently.
        `GiniObjectiveFunction.ProxyImpurityImprovement` is the transcription;
        this is a convenience for printing an exact rational.
        """
        if not self.valid:
            return Scalar[dtype].MIN_FINITE
        return Scalar[dtype](self.num) / Scalar[dtype](self.den) - Scalar[dtype](
            self.length
        )


@fieldwise_init
struct GiniObjectiveFunction[dtype: DType](Copyable, Movable):
    """Gini. cuML `objectives.cuh:29-108` for shape, sklearn `Gini`
    (`_criterion.pyx:605-687`) for arithmetic.

    Theirs is `template <typename DataT_, typename LabelT_, typename IdxT_>
    class GiniObjectiveFunction`. `DataT` is the parameter here; `LabelT` and
    `IdxT` are fixed by `dataset.mojo` (labels are `Float32` carrying a class
    id, indices are `Int32`) rather than being free parameters, because this
    lane has exactly one dataset view.

    Single-output only: sklearn's `n_outputs` loop (`_criterion.pyx:635`,
    `:669`) is present in the transcriptions as a loop of length one that has
    been written out, and the trailing `/ self.n_outputs` (`:645`, `:686-687`)
    is a division by one. Multi-output classification is unported and visible
    as such (rule 3).
    """

    comptime DataT = Scalar[Self.dtype]
    """Theirs is `using DataT = DataT_;` (`objectives.cuh:33`)."""

    comptime BinT = CountBin
    """Theirs is `using BinT = CountBin;` (`objectives.cuh:41`)."""

    var nclasses: Int32
    """Theirs is `IdxT nclasses` (`objectives.cuh:37`)."""

    var min_samples_leaf: Int32
    """Theirs is `IdxT min_samples_leaf` (`objectives.cuh:38`)."""

    def NumClasses(self) -> Int32:
        """`objectives.cuh:47`: `DI IdxT NumClasses() const { return nclasses; }`."""
        return self.nclasses

    # ----------------------------------------------------------------------
    # cuML's score. `objectives.cuh:49-83`.
    # ----------------------------------------------------------------------

    def GainPerSplit[
        ml: Bool,
        mt: Bool, //,
        ol: Origin[mut=ml],
        ot: Origin[mut=mt],
    ](
        self,
        hist_left: Pointer[CountBin, ol],
        hist_total: Pointer[CountBin, ot],
        len: Int32,
        nLeft: Int32,
    ) -> Scalar[Self.dtype]:
        """Compute the gini impurity reduction for this split.

        cuML `objectives.cuh:52-83`, transcribed statement for statement. The
        only edits are the two DEVIATION 143 ones: `hist[n_bins * j + i].x`
        becomes `hist_left[j].x` and `hist[n_bins * j + n_bins - 1].x` becomes
        `hist_total[j].x`. Their local names (`nRight`, `invLen`, `invLeft`,
        `invRight`, `gain`, `val_i`, `lval_i`, `lval`, `total_sum`, `rval_i`,
        `rval`, `val`) are kept.

        This is the impurity DECREASE, not sklearn's proxy -- see the module
        docstring and DEVIATION 144. `-std::numeric_limits<DataT>::max()` is
        `MIN_FINITE` here, the same bit pattern (`split.mojo` records the same
        spelling).

        Their `invLeft` and `invRight` are computed BEFORE the
        `min_samples_leaf` test (`:56-58` precede `:62`), so with
        `min_samples_leaf == 0` and an empty child they are `+inf` on the way
        to a return that never uses them. Transcribed as written; DEVIATION
        144 is where the guard lives.
        """
        var nRight = len - nLeft
        comptime One = Scalar[Self.dtype](1.0)
        var invLen = One / Scalar[Self.dtype](Int(len))
        var invLeft = One / Scalar[Self.dtype](Int(nLeft))
        var invRight = One / Scalar[Self.dtype](Int(nRight))
        var gain = Scalar[Self.dtype](0.0)

        # if there aren't enough samples in this split, don't bother!
        if nLeft < self.min_samples_leaf or nRight < self.min_samples_leaf:
            return Scalar[Self.dtype].MIN_FINITE

        for j in range(Int(self.nclasses)):
            var val_i: Int32 = 0
            var lval_i = hist_left[unsafe_offset=j].x
            var lval = Scalar[Self.dtype](Int(lval_i))
            gain += lval * invLeft * lval * invLen

            val_i += lval_i
            var total_sum = hist_total[unsafe_offset=j].x
            var rval_i = total_sum - lval_i
            var rval = Scalar[Self.dtype](Int(rval_i))
            gain += rval * invRight * rval * invLen

            val_i += rval_i
            var val = Scalar[Self.dtype](Int(val_i)) * invLen
            gain -= val * val

        return gain

    # ----------------------------------------------------------------------
    # sklearn's arithmetic. `_criterion.pyx:605-687`.
    # ----------------------------------------------------------------------

    def NodeImpurity[
        mt: Bool, //, ot: Origin[mut=mt]
    ](
        self,
        hist_total: Pointer[CountBin, ot],
        weighted_n_node_samples: Scalar[Self.dtype],
    ) -> Scalar[Self.dtype]:
        """sklearn `Gini.node_impurity`, `_criterion.pyx:622-645`.

        Their body for `n_outputs == 1`:

            gini = 0.0
            sq_count = 0.0
            for c in range(n_classes): count_k = sum_total[c]; sq_count += count_k*count_k
            gini += 1.0 - sq_count / (weighted_n_node_samples * weighted_n_node_samples)
            return gini / n_outputs

        `sq_count / (w * w)` is NOT `(sq_count / w) / w` and is not
        `sq_count / w**2`; it is theirs.
        """
        var gini = Scalar[Self.dtype](0.0)
        var sq_count: Scalar[Self.dtype]
        var count_k: Scalar[Self.dtype]

        sq_count = 0.0
        for c in range(Int(self.nclasses)):
            count_k = Scalar[Self.dtype](Int(hist_total[unsafe_offset=c].x))
            sq_count += count_k * count_k

        gini += 1.0 - sq_count / (
            weighted_n_node_samples * weighted_n_node_samples
        )
        return gini

    def ChildrenImpurity[
        ml: Bool,
        mt: Bool, //,
        ol: Origin[mut=ml],
        ot: Origin[mut=mt],
    ](
        self,
        hist_left: Pointer[CountBin, ol],
        hist_total: Pointer[CountBin, ot],
        weighted_n_left: Scalar[Self.dtype],
        weighted_n_right: Scalar[Self.dtype],
        mut impurity_left: Scalar[Self.dtype],
        mut impurity_right: Scalar[Self.dtype],
    ):
        """sklearn `Gini.children_impurity`, `_criterion.pyx:647-687`.

        Their loop body, verbatim, with `sum_left[k, c]` read from
        `hist_left[c]` and `sum_right[k, c]` recovered as `total - left` the
        way cuML recovers it (`objectives.cuh:72-73`); sklearn maintains
        `sum_right` incrementally (`_criterion.pyx:870`) but its VALUE is
        `sum_total - sum_left`, exactly, in integers.

        Outputs are `mut` arguments rather than sklearn's `float64_t*`
        out-pointers. Same contract, one less way to alias.
        """
        var gini_left = Scalar[Self.dtype](0.0)
        var gini_right = Scalar[Self.dtype](0.0)
        var sq_count_left: Scalar[Self.dtype]
        var sq_count_right: Scalar[Self.dtype]
        var count_k: Scalar[Self.dtype]

        sq_count_left = 0.0
        sq_count_right = 0.0

        for c in range(Int(self.nclasses)):
            count_k = Scalar[Self.dtype](Int(hist_left[unsafe_offset=c].x))
            sq_count_left += count_k * count_k

            count_k = Scalar[Self.dtype](
                Int(
                    hist_total[unsafe_offset=c].x
                    - hist_left[unsafe_offset=c].x
                )
            )
            sq_count_right += count_k * count_k

        gini_left += 1.0 - sq_count_left / (weighted_n_left * weighted_n_left)

        gini_right += 1.0 - sq_count_right / (
            weighted_n_right * weighted_n_right
        )

        impurity_left = gini_left
        impurity_right = gini_right

    def ProxyImpurityImprovement[
        ml: Bool,
        mt: Bool, //,
        ol: Origin[mut=ml],
        ot: Origin[mut=mt],
    ](
        self,
        hist_left: Pointer[CountBin, ol],
        hist_total: Pointer[CountBin, ot],
        len: Int32,
        nLeft: Int32,
    ) -> Scalar[Self.dtype]:
        """sklearn's SELECTION score, in float. `_criterion.pyx:147-163` over
        `:647-687`.

        This is the quantity `_splitter.pyx:693` compares. It is kept for
        reporting and for the check's cross-examination of the exact form; the
        selection itself uses `CompareProxyExact` (DEVIATION 144).

        `min_samples_leaf` rejection is sklearn's, not this function's --
        their splitter tests it before ever calling the criterion
        (`_splitter.pyx:664-666`). It is applied here so that the float and
        exact paths reject the same candidates and the check can compare them
        cell for cell.
        """
        var nRight = len - nLeft
        if nLeft < self.min_samples_leaf or nRight < self.min_samples_leaf:
            return Scalar[Self.dtype].MIN_FINITE

        var weighted_n_left = Scalar[Self.dtype](Int(nLeft))
        var weighted_n_right = Scalar[Self.dtype](Int(nRight))
        var impurity_left = Scalar[Self.dtype](0.0)
        var impurity_right = Scalar[Self.dtype](0.0)
        self.ChildrenImpurity(
            hist_left,
            hist_total,
            weighted_n_left,
            weighted_n_right,
            impurity_left,
            impurity_right,
        )
        return proxy_impurity_improvement[Self.dtype](
            weighted_n_left, weighted_n_right, impurity_left, impurity_right
        )

    # ----------------------------------------------------------------------
    # OURS. DEVIATION 144: the same proxy, exactly, in integers.
    # ----------------------------------------------------------------------

    def ProxyImpurityExact[
        ml: Bool,
        mt: Bool, //,
        ol: Origin[mut=ml],
        ot: Origin[mut=mt],
    ](
        self,
        hist_left: Pointer[CountBin, ol],
        hist_total: Pointer[CountBin, ot],
        len: Int32,
        nLeft: Int32,
    ) -> GiniProxyExact:
        """sklearn's selection score as an exact rational. DEVIATION 144.

        Derivation, once, so the code below is checkable against it:

            gini_L      = 1 - sq_L/nL^2                (`_criterion.pyx:680`)
            wL * gini_L = nL - sq_L/nL
            proxy       = -(nR - sq_R/nR) - (nL - sq_L/nL)   (`:162-163`)
                        = sq_L/nL + sq_R/nR - n
                        = (sq_L*nR + sq_R*nL) / (nL*nR) - n

        Every symbol on the last line is an integer when `sample_weight` is
        None. No rounding happens anywhere in this function.
        """
        debug_assert(
            Int(len) <= MAX_ROWS_EXACT,
            (
                "GiniObjectiveFunction.ProxyImpurityExact: node has more rows"
                " than the Int128 cross-multiply in CompareProxyExact is"
                " exact to; see MAX_ROWS_EXACT"
            ),
        )

        var nRight = len - nLeft

        # cuML's rejection (`objectives.cuh:62-63`), plus the empty-child
        # guard their float form does not have (DEVIATION 144).
        if nLeft < self.min_samples_leaf or nRight < self.min_samples_leaf:
            return GiniProxyExact(0, 0, Int64(Int(len)), False)
        if nLeft == 0 or nRight == 0:
            return GiniProxyExact(0, 0, Int64(Int(len)), False)

        var sq_left: Int64 = 0
        var sq_right: Int64 = 0
        for c in range(Int(self.nclasses)):
            var lval_i = Int64(Int(hist_left[unsafe_offset=c].x))
            sq_left += lval_i * lval_i
            var rval_i = Int64(
                Int(
                    hist_total[unsafe_offset=c].x
                    - hist_left[unsafe_offset=c].x
                )
            )
            sq_right += rval_i * rval_i

        var nl = Int64(Int(nLeft))
        var nr = Int64(Int(nRight))
        return GiniProxyExact(
            sq_left * nr + sq_right * nl, nl * nr, Int64(Int(len)), True
        )

    @staticmethod
    def CompareProxyExact(a: GiniProxyExact, b: GiniProxyExact) -> Int:
        """Order two candidates by sklearn's proxy, exactly. DEVIATION 144.

        Returns `-1` if `a` scores below `b`, `+1` if above, `0` on an EXACT
        tie -- and an exact tie here really is a tie, not two floats that
        happened to round together, which is the whole point. The caller
        breaks a `0` with `Split.update`'s total order (`split.mojo`,
        DEVIATION 133).

        `a.num/a.den` against `b.num/b.den` with both denominators strictly
        positive is `a.num*b.den` against `b.num*a.den`, in `Int128`. See
        `MAX_ROWS_EXACT` for why `Int128` and how far it reaches.

        An invalid candidate is below every valid one, matching what cuML's
        `-max<DataT>()` return does in their float reduction
        (`objectives.cuh:63`). Two invalid candidates tie.
        """
        if not a.valid and not b.valid:
            return 0
        if not a.valid:
            return -1
        if not b.valid:
            return 1

        var lhs = Int128(a.num) * Int128(b.den)
        var rhs = Int128(b.num) * Int128(a.den)
        if lhs < rhs:
            return -1
        if lhs > rhs:
            return 1
        return 0

    # ----------------------------------------------------------------------
    # Leaf prediction. `objectives.cuh:97-107`.
    # ----------------------------------------------------------------------

    @staticmethod
    def SetLeafVector[
        ms: Bool, //, os: Origin[mut=ms], oo: MutOrigin
    ](
        shist: Pointer[CountBin, os],
        nclasses: Int32,
        out_ptr: Pointer[Scalar[Self.dtype], oo],
    ):
        """`objectives.cuh:97-107`, transcribed.

            // Output probability
            int total = 0;
            for (int i = 0; i < nclasses; i++) total += shist[i].x;
            for (int i = 0; i < nclasses; i++) out[i] = DataT(shist[i].x) / total;

        Note their `total` is an `int` and the division promotes it, so the
        numerator is converted and the denominator is not -- the same shape is
        kept here. sklearn's equivalent normalises `value` the same way
        (`_classes.py`, `tree_.value` is divided by the node weight), so the
        two agree on the quantity.

        `out` is `out_ptr` here only because `out` is the Mojo keyword for an
        uninitialised output argument and cannot be an argument NAME.
        """
        var total: Int32 = 0
        for i in range(Int(nclasses)):
            total += shist[unsafe_offset=i].x
        for i in range(Int(nclasses)):
            out_ptr[unsafe_offset=i] = Scalar[Self.dtype](
                Int(shist[unsafe_offset=i].x)
            ) / Scalar[Self.dtype](Int(total))


# ==========================================================================
# Regression.
# ==========================================================================


@fieldwise_init
struct MSEObjectiveFunction[dtype: DType](Copyable, Movable):
    """MSE. cuML `objectives.cuh:195-265` for shape, sklearn `MSE`
    (`_criterion.pyx:922-1017`) for arithmetic.

    THE SEARCH AND THE REPORT NEED DIFFERENT ACCUMULATORS, and that is
    sklearn's observation, not ours. `MSE.proxy_impurity_improvement`
    (`_criterion.pyx:944-973`) reads ONLY `sum_left`, `sum_right`,
    `weighted_n_left`, `weighted_n_right` -- no sum of squares anywhere,
    because `sq_sum_total` is constant across the candidates of one node and
    a proxy is allowed to drop constants. The sum of squares appears only in
    `MSE.children_impurity` (`:975-1017`) and `MSE.node_impurity`
    (`:928-942`), which run once, for the split that already won. So the
    score pass carries `(count, sum)` per candidate and only the winner needs
    `(count, sum, sum_of_squares)`.

    PRECISION IS NOT SETTLED HERE. `dtype` is a parameter, and DEVIATION 135
    in `extratrees/DEVIATIONS.md` is OPEN: sklearn accumulates in `float64`,
    there is no `float64` on device, and the candidates (`Float32`, a
    compensated sum, `gbdt/`'s fixed-point label scaling) are a decision
    recorded in `PLAN.md`. Hard-coding `Float32` here would settle by
    accident a question that is deliberately open, so this file never names a
    concrete type.
    """

    comptime DataT = Scalar[Self.dtype]
    """Theirs is `using DataT = DataT_;` (`objectives.cuh:198`)."""

    comptime BinT = AggregateBin[Self.dtype]
    """Theirs is `using BinT = AggregateBin;` (`objectives.cuh:201`)."""

    var min_samples_leaf: Int32
    """Theirs is `IdxT min_samples_leaf` (`objectives.cuh:204`). Their
    constructor takes `(IdxT nclasses, IdxT min_samples_leaf)` and DISCARDS
    `nclasses` (`:207-210`); ours does not take it."""

    def NumClasses(self) -> Int32:
        """`objectives.cuh:257`: `DI IdxT NumClasses() const { return 1; }`."""
        return 1

    # ----------------------------------------------------------------------
    # cuML's score. `objectives.cuh:212-244`.
    # ----------------------------------------------------------------------

    def GainPerSplit(
        self,
        hist_left: AggregateBin[Self.dtype],
        hist_total: AggregateBin[Self.dtype],
        len: Int32,
        nLeft: Int32,
    ) -> Scalar[Self.dtype]:
        """Compute the MSE impurity reduction for this split.

        cuML `objectives.cuh:225-244`, transcribed statement for statement,
        with DEVIATION 143's substitution `hist[n_bins - 1]` -> `hist_total`
        and `hist[i]` -> `hist_left`.

            auto label_sum        = hist[n_bins - 1].label_sum;
            DataT parent_obj      = -label_sum * label_sum * invLen;
            DataT left_obj        = -(hist[i].label_sum * hist[i].label_sum) / nLeft;
            DataT right_label_sum = hist[i].label_sum - label_sum;
            DataT right_obj       = -(right_label_sum * right_label_sum) / nRight;
            gain                  = parent_obj - (left_obj + right_obj);
            gain *= DataT(0.5) * invLen;

        `right_label_sum = left - total` is `-(total - left)`, i.e. the
        NEGATIVE of the right child's label sum. It is squared immediately, so
        the result is right; it is transcribed as written rather than
        corrected, because a transcription that silently corrects its source
        stops being one. If this ever feeds something that is not squared,
        that is the line to look at.
        """
        # Theirs opens `auto gain{DataT(0)}` (`objectives.cuh:227`). That
        # initialiser is dead on BOTH paths -- the early return never reads it
        # and the else-branch assigns over it -- and Mojo says so, so the
        # declaration is kept and the dead store is not.
        var gain: Scalar[Self.dtype]
        var nRight = len - nLeft
        var invLen = Scalar[Self.dtype](1.0) / Scalar[Self.dtype](Int(len))
        # if there aren't enough samples in this split, don't bother!
        if nLeft < self.min_samples_leaf or nRight < self.min_samples_leaf:
            return Scalar[Self.dtype].MIN_FINITE
        else:
            var label_sum = hist_total.label_sum
            var parent_obj = -label_sum * label_sum * invLen
            var left_obj = -(
                hist_left.label_sum * hist_left.label_sum
            ) / Scalar[Self.dtype](Int(nLeft))
            var right_label_sum = hist_left.label_sum - label_sum
            var right_obj = -(right_label_sum * right_label_sum) / Scalar[
                Self.dtype
            ](Int(nRight))
            gain = parent_obj - (left_obj + right_obj)
            gain *= Scalar[Self.dtype](0.5) * invLen

            return gain

    # ----------------------------------------------------------------------
    # sklearn's arithmetic. `_criterion.pyx:922-1017`.
    # ----------------------------------------------------------------------

    def NodeImpurity(
        self,
        sq_sum_total: Scalar[Self.dtype],
        hist_total: AggregateBin[Self.dtype],
        weighted_n_node_samples: Scalar[Self.dtype],
    ) -> Scalar[Self.dtype]:
        """sklearn `MSE.node_impurity`, `_criterion.pyx:928-942`.

            impurity = self.sq_sum_total / self.weighted_n_node_samples
            for k: impurity -= (self.sum_total[k] / self.weighted_n_node_samples)**2.0
            return impurity / self.n_outputs

        `**2.0` is kept as `**2.0` rather than rewritten to `x * x`. They are
        the same number for a correctly-rounded `pow`, but rule 0c is that our
        recollection of "the same number" is what keeps being wrong, and this
        repository has a live memory of exactly this class of edit changing a
        tie (`mojo-log-breaks-ties`).
        """
        var impurity: Scalar[Self.dtype]

        impurity = sq_sum_total / weighted_n_node_samples
        impurity -= (hist_total.label_sum / weighted_n_node_samples) ** 2.0

        return impurity

    def ProxyImpurityImprovement(
        self,
        hist_left: AggregateBin[Self.dtype],
        hist_total: AggregateBin[Self.dtype],
        len: Int32,
        nLeft: Int32,
    ) -> Scalar[Self.dtype]:
        """sklearn `MSE.proxy_impurity_improvement`, `_criterion.pyx:944-973`.

        THE SELECTION SCORE for regression. Their body, verbatim, for
        `n_outputs == 1`:

            proxy_impurity_left  = 0.0
            proxy_impurity_right = 0.0
            for k: proxy_impurity_left  += sum_left[k]  * sum_left[k]
                   proxy_impurity_right += sum_right[k] * sum_right[k]
            return (proxy_impurity_left / weighted_n_left +
                    proxy_impurity_right / weighted_n_right)

        No sum of squares of `y` appears, and that is not an omission -- see
        their derivation at `:955-962`. `sum_right` is recovered as
        `total - left`, which is what their `update` maintains (`:870`).

        `min_samples_leaf` is sklearn's splitter's test
        (`_splitter.pyx:664-666`), applied here for the same reason as on the
        Gini side: so that this and `GainPerSplit` reject the same candidates
        and the check can compare them cell for cell.
        """
        var nRight = len - nLeft
        if nLeft < self.min_samples_leaf or nRight < self.min_samples_leaf:
            return Scalar[Self.dtype].MIN_FINITE

        var weighted_n_left = Scalar[Self.dtype](Int(nLeft))
        var weighted_n_right = Scalar[Self.dtype](Int(nRight))
        var sum_left = hist_left.label_sum
        var sum_right = hist_total.label_sum - hist_left.label_sum

        var proxy_impurity_left = Scalar[Self.dtype](0.0)
        var proxy_impurity_right = Scalar[Self.dtype](0.0)

        proxy_impurity_left += sum_left * sum_left
        proxy_impurity_right += sum_right * sum_right

        return (
            proxy_impurity_left / weighted_n_left
            + proxy_impurity_right / weighted_n_right
        )

    def ChildrenImpurity(
        self,
        sq_sum_left: Scalar[Self.dtype],
        sq_sum_total: Scalar[Self.dtype],
        hist_left: AggregateBin[Self.dtype],
        hist_total: AggregateBin[Self.dtype],
        weighted_n_left: Scalar[Self.dtype],
        weighted_n_right: Scalar[Self.dtype],
        mut impurity_left: Scalar[Self.dtype],
        mut impurity_right: Scalar[Self.dtype],
    ):
        """sklearn `MSE.children_impurity`, `_criterion.pyx:975-1017`.

        The FINAL impurity of the chosen split, which is where the sum of
        squares finally earns its keep. Their body, verbatim:

            sq_sum_right = self.sq_sum_total - sq_sum_left
            impurity_left[0]  = sq_sum_left  / self.weighted_n_left
            impurity_right[0] = sq_sum_right / self.weighted_n_right
            for k: impurity_left[0]  -= (self.sum_left[k]  / self.weighted_n_left)  ** 2.0
                   impurity_right[0] -= (self.sum_right[k] / self.weighted_n_right) ** 2.0
            impurity_left[0]  /= self.n_outputs
            impurity_right[0] /= self.n_outputs

        Their `sq_sum_left` comes from re-walking the left rows
        (`:997-1006`); ours is passed in, because the caller has just walked
        those rows and there is no `sample_indices` array to re-walk on a
        device. `sq_sum_right` is recovered from the total exactly as theirs
        is (`:1007`).
        """
        var sq_sum_right = sq_sum_total - sq_sum_left

        impurity_left = sq_sum_left / weighted_n_left
        impurity_right = sq_sum_right / weighted_n_right

        impurity_left -= (hist_left.label_sum / weighted_n_left) ** 2.0
        impurity_right -= (
            (hist_total.label_sum - hist_left.label_sum) / weighted_n_right
        ) ** 2.0

    # ----------------------------------------------------------------------
    # Leaf prediction. `objectives.cuh:259-264`.
    # ----------------------------------------------------------------------

    @staticmethod
    def SetLeafVector[
        ms: Bool, //, os: Origin[mut=ms], oo: MutOrigin
    ](
        shist: Pointer[AggregateBin[Self.dtype], os],
        nclasses: Int32,
        out_ptr: Pointer[Scalar[Self.dtype], oo],
    ):
        """`objectives.cuh:259-264`, transcribed.

            for (int i = 0; i < nclasses; i++)
              out[i] = shist[i].label_sum / shist[i].count;

        The mean of the node's labels. sklearn reaches the same value by
        `RegressionCriterion.node_value` (`_criterion.pyx:882-887`),
        `dest[k] = sum_total[k] / weighted_n_node_samples`.

        Their parameter is still called `nclasses` in a regression objective
        (it is the output count, which `NumClasses` returns as 1); the name is
        kept because rule 6 says the symbol is the diff surface, and their
        dispatch passes the same variable to both.
        """
        for i in range(Int(nclasses)):
            out_ptr[unsafe_offset=i] = shist[
                unsafe_offset=i
            ].label_sum / Scalar[Self.dtype](
                Int(shist[unsafe_offset=i].count)
            )
