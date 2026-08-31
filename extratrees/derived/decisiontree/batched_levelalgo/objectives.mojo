# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""Split-scoring objectives: Gini and Entropy for classification, MSE for regression.

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
| `EntropyObjectiveFunction.GainPerSplit`      | `110-168`             | --                       |
| `EntropyObjectiveFunction.NodeImpurity`      | --                    | `548-567`                |
| `EntropyObjectiveFunction.ChildrenImpurity`  | --                    | `569-602`                |
| `EntropyObjectiveFunction.ProxyImpurityImprovement` | --             | `147-163` + `569-602`    |
| `EntropyObjectiveFunction.GainKeyExact`      | --                    | (DEVIATION 459)          |
| `EntropyObjectiveFunction.SetLeafVector`     | `181-191`             | --                       |
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
- `PoissonObjectiveFunction` (`:267-346`), `GammaObjectiveFunction`
  (`:348-424`), `InverseGaussianObjectiveFunction` (`:426-502`) — sklearn's
  ExtraTrees regression default is `criterion='squared_error'`, and rule 3
  says an unported thing must be visible rather than half-present.
  `EntropyObjectiveFunction` (`:110-193`) WAS in this list until DEVIATION
  459 (2026-08-23) ported it; it is below, beside Gini.

HOST/DEVICE DISCIPLINE
----------------------
Every scoring function here is a plain function over scalars and non-owning
pointers. Nothing allocates, nothing raises, nothing touches `List`, `String`
or any other host-only type, so each is callable unchanged from a kernel.
"""

from std.math import fma, log

from original.numerics import ftz, identical_log

from extratrees.derived.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import (
    classification_key_shift,
    float_gain_key,
)



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
#   and `quantiles.cuh` is permanently absent per `NOT_IMPLEMENTED.tsv`). A random
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
#   today. **It is not listed in `extratrees/NOT_IMPLEMENTED.tsv` either, which rule
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


comptime MAX_ROWS_EXACT: Int = 1 << 21
"""Row count per node at which `GiniProxyExact` is exact.

**THIS BOUND WAS 2^26 AND THAT WAS WRONG BY THIRTEEN BITS.** The old
derivation bounded the `Int128` CROSS-MULTIPLY in `CompareProxyExact` and
forgot that `num` is STORED in an `Int64` field first. Two different widths,
and the binding one is the storage.

Measured, because a bound is not an opinion:

    n = 2^20   num = 288230376151711744    Int64 agrees
    n = 2^21   num = 2305843009213693952   Int64 agrees
    n = 2^22   num = 18446744073709551616  Int64 reads 0   *** WRAPPED ***
    n = 2^25   num = 9444732965739290427392    Int64 reads 0
    n = 2^26   num = 75557863725914323419136   Int64 reads 0

**It wraps to exactly ZERO**, which is the worst failure available: every
candidate in the node ties at the same numerator and the winner is decided by
`Split.update`'s `colid` arm — silently, by feature index, with no symptom.

The widths, tightly. With `n` rows and integer class counts,
`sq_L = sum_j l_j^2 <= nL^2`, so

    num = sq_L*nR + sq_R*nL <= nL^2*nR + nR^2*nL = nL*nR*n <= n^3/4
    den = nL*nR                                            <= n^2/4

`num` lives in an `Int64`, which holds `2^63 - 1`, and `n^3/4 <= 2^63` gives
`n <= 2^21.67`. **`2^21` is the largest safe power of two**: `num <= 2^61`,
two bits of headroom.

The cross-multiply is then far inside `Int128`: `num_a * den_b <= 2^61 * 2^40
= 2^101`. The old 2^26 was the correct answer to that question and the wrong
answer to this one.

2,097,152 rows IN ONE NODE, and the root is the largest node, so it is a bound
on the dataset. Above it, `ProxyImpurityExact` raises rather than returning a
wrapped rational; `objectives_check` measures the wrap at the boundary.

The DEVICE has its own, separate bound for the same field
(`SCORE_MAX_ROWS_EXACT` in `builder_kernels_impl.mojo`) because it publishes
the numerator without ever forming the `Int128` product. The two agree by
construction and both are stated where they are enforced.
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
    uses (`original/fixed_point.mojo`, `choose_scale`) is **DEVIATION 135,
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

        THE THREE ACCUMULATIONS ARE EXPLICIT `fma`s, FOR DEVIATION 142's
        REASON. `gain += lval * invLeft * lval * invLen` is a multiply chain
        followed by an add, and a GPU backend contracts the LAST multiply with
        the add whatever the source says -- six source-level barriers were
        measured doing exactly that on the threshold draw. Written as
        `fma(prod, invLen, gain)` the rounding is fixed by the SOURCE on every
        backend, and it is also what nvcc's default `--fmad=true` makes of
        their expression, so it is what cuML computes on the hardware they ship
        for. MEASURED while this was mismatched: the host and device forms
        disagreed on 1655 of 4000 hashed candidates, and 11 of 747 tree nodes
        chose a different split as a result.

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
            gain = fma(lval * invLeft * lval, invLen, gain)

            val_i += lval_i
            var total_sum = hist_total[unsafe_offset=j].x
            var rval_i = total_sum - lval_i
            var rval = Scalar[Self.dtype](Int(rval_i))
            gain = fma(rval * invRight * rval, invLen, gain)

            val_i += rval_i
            var val = Scalar[Self.dtype](Int(val_i)) * invLen
            gain = fma(-val, val, gain)

        # DEVIATION 217: the true Gini decrease is non-negative; a negative
        # value is float cancellation, and it fed `split_not_valid`. See
        # `builder.mojo::gain_per_split` for the measurement; the three gain
        # forms clamp identically or the arms would grow different trees.
        if gain < Scalar[Self.dtype](0.0):
            gain = Scalar[Self.dtype](0.0)
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
        # DEVIATION 218: nodes past MAX_ROWS_EXACT no longer refuse -- the
        # node-uniform shift below keeps the published pair inside Int64 and
        # the comparator's Int128 cross-multiply, at 2^-40 relative
        # granularity. Below 2^21 rows the shift is zero and this function
        # is bit-for-bit its pre-218 self.

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
        # DEVIATION 218: mirrored from the finalize kernel and the host
        # oracle -- the three publish sites shift identically or the arms
        # would rank differently.
        var cs = Int64(classification_key_shift(Int(len)))
        return GiniProxyExact(
            (sq_left >> cs) * nr + (sq_right >> cs) * nl,
            nl * nr,
            Int64(Int(len)),
            True,
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
# DEVIATION BLOCK 459 -- Entropy: cuML's float gain IS the key, as a
#                        float seam beside the integer Gini core
#
# THEIRS. `EntropyObjectiveFunction` (`objectives.cuh:110-193`):
#   `GainPerSplit` is the information gain in `DataT`, built from
#   `raft::log(p) / raft::log(DataT(2))` per class and per child, and the
#   builder reduces candidates on that float through `Split::update`
#   (`split.cuh:76-90`): greater gain wins, equal gain falls to `colid`,
#   then `quesval`.
# OURS. The Gini core selects on DEVIATION 144's EXACT INTEGER rational
#   (`num/den`), published per cell as two `Int64`s and compared by
#   cross-multiplication in the reduction. Entropy CANNOT live in that core:
#   `p*log(p)` is not a rational in the counts, there is no integer form of
#   it, and a fixed-point `log` table would be an invention neither upstream
#   has. So the entropy criterion is a FLOAT SEAM published THROUGH the same
#   integer pair: the cell's key is `num = float_gain_key(gain)`, `den = 1`,
#   where `float_gain_key` (`builder_kernels_impl.mojo`) is the
#   order-preserving sign-magnitude map of the Float32 gain onto `Int64`.
#   With `den == 1` the reduction's cross-multiply IS an integer compare of
#   the keys, so the candidate ORDER is exactly cuML's float `>`, and an
#   EQUAL key -- the same gain bits, which the sign-magnitude map also
#   gives to `-0.0` and `+0.0`, as float `==` does -- falls through to
#   DEVIATION 194's tie arms (`colid`, then `quesval`), which are cuML's
#   remaining two arms. Nothing in the reduction, the readback or the
#   host-side `Split` construction changes; only the finalize site writes
#   a different pair. The host oracle builds the SAME `GiniProxyExact(key,
#   1)` and runs the SAME comparator, so host and device order identically
#   by construction -- a device/host difference can only come from the
#   gain BITS.
# THE LOG, AND WHERE ITS BITS COME FROM. Every `raft::log` routes through
#   `identical_log` (`original/numerics.mojo`, IDENTITY_PATHS row 12),
#   copying the shape `ensemble/decisiontree/batched_levelalgo/
#   objectives.mojo`'s DEVIATION 406 gave RF's entropy: under NUMERIC_FAST
#   the wrapper IS `std.math.log` verbatim (each backend's own lowering --
#   the host's and Metal's CAN differ in the last bit, so FAST does not
#   promise device==host for entropy and `device_forest_check` pins that
#   equality only under IDENTICAL); under NUMERIC_IDENTICAL it is
#   `portable_logf`, one Cephes polynomial through fma on every backend,
#   and the arithmetic AROUND it is pinned with row 9/10's `ftz` on every
#   stored intermediate, decomposed in cuML's association order so the
#   FAST arm is the transcription's arithmetic unchanged.
#   `raft::log(DataT(2))` is recomputed per term, where they compute it.
# THE CLAMP. DEVIATION 217's argument holds verbatim: the information gain
#   is provably non-negative and a negative float value is cancellation,
#   which `split_not_valid` would leaf. Clamped at zero on both arms.
# PRICE. Entropy candidates are ordered by a Float32 computed with a
#   vendor `log` under FAST -- two candidates whose true gains differ by
#   less than the rounding can order either way, which is cuML's own
#   behaviour and sklearn's (float64) is not bit-comparable to either. The
#   exact-argmax guarantee DEVIATION 144 gives Gini is NOT extended to
#   entropy and is not claimed for it.
# ==========================================================================


@always_inline
def _log_seam[
    dt: DType, //
](x: Scalar[dt]) -> Scalar[dt] where dt.is_floating_point():
    """`numerics.identical_log` for a generic-dtype seam -- the shape of RF's
    DEVIATION 406 `_log_seam`. Under FAST the wrapper IS `std.math.log`;
    under IDENTICAL it is `portable_logf`. Any non-float32 width keeps the
    stdlib call (no float64 on the device, no portable polynomial for it)."""
    comptime if dt == DType.float32:
        return identical_log(x.cast[DType.float32]()).cast[dt]()
    return log(x)


@always_inline
def _ftz_seam[dt: DType, //](x: Scalar[dt]) -> Scalar[dt]:
    """`numerics.ftz` for a generic-dtype seam (RF's DEVIATION 405 shape):
    a comptime no-op under FAST, the row-10 flush under IDENTICAL."""
    comptime if dt == DType.float32:
        return ftz(x.cast[DType.float32]()).cast[dt]()
    return x


@fieldwise_init
struct EntropyObjectiveFunction[dtype: DType](
    Copyable, Movable
) where dtype.is_floating_point():
    """Entropy. cuML `objectives.cuh:110-193` for shape and for the SELECTION
    score, sklearn `Entropy` (`_criterion.pyx:532-602`) for the reported
    impurities. DEVIATION BLOCK 459 above.

    `nclasses` / `min_samples_leaf` are theirs (`objectives.cuh:117-118`);
    `NumClasses`, `GainPerSplit`, `SetLeafVector` are theirs by name; the
    `BinT` is `CountBin` as theirs is (`:122`).
    """

    comptime DataT = Scalar[Self.dtype]
    comptime BinT = CountBin

    var nclasses: Int32
    var min_samples_leaf: Int32

    def NumClasses(self) -> Int32:
        """`objectives.cuh:127`."""
        return self.nclasses

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
        """compute the Entropy (or information gain) for each split.

        cuML `objectives.cuh:132-168`, transcribed statement for statement
        with DEVIATION 143's two index edits (`hist[n_bins*c + i]` ->
        `hist_left[c]`, `hist[n_bins*c + n_bins-1]` -> `hist_total[c]`).
        Their locals keep their names. Every `raft::log` is `_log_seam`,
        every stored intermediate is `_ftz_seam`d, and the three
        accumulations are decomposed in THEIR association order:
        `log(a)/log(2) * lval * invLen` is `((log(a)/log(2)) * lval) *
        invLen`, left to right as C++ parses it. Under FAST the seams are
        the identity, so this IS their arithmetic; no `fma` is introduced
        where theirs has none, because the gain is the key here and an fma
        would be a different key.

        THIS FUNCTION IS CALLED FROM THE DEVICE (`builder.mojo::
        entropy_gain_per_split` bitcasts the kernel's Int32 accumulators to
        `CountBin`), so there is ONE transcription, not a host copy and a
        device copy that can drift.
        """
        var nRight = len - nLeft
        var gain = Scalar[Self.dtype](0.0)
        # if there aren't enough samples in this split, don't bother!
        if nLeft < self.min_samples_leaf or nRight < self.min_samples_leaf:
            return Scalar[Self.dtype].MIN_FINITE
        comptime One = Scalar[Self.dtype](1.0)
        comptime Two = Scalar[Self.dtype](2.0)
        var invLeft = _ftz_seam(One / Scalar[Self.dtype](Int(nLeft)))
        var invRight = _ftz_seam(One / Scalar[Self.dtype](Int(nRight)))
        var invLen = _ftz_seam(One / Scalar[Self.dtype](Int(len)))
        for c in range(Int(self.nclasses)):
            var val_i: Int32 = 0
            var lval_i = hist_left[unsafe_offset=c].x
            if lval_i != 0:
                var lval = Scalar[Self.dtype](Int(lval_i))
                # gain += raft::log(lval * invLeft) / raft::log(DataT(2)) * lval * invLen;
                var larg = _ftz_seam(lval * invLeft)
                var l1 = _ftz_seam(_log_seam(larg) / _log_seam(Two))
                var l2 = _ftz_seam(l1 * lval)
                var l3 = _ftz_seam(l2 * invLen)
                gain = _ftz_seam(gain + l3)

            val_i += lval_i
            var total_sum = hist_total[unsafe_offset=c].x
            var rval_i = total_sum - lval_i
            if rval_i != 0:
                var rval = Scalar[Self.dtype](Int(rval_i))
                # gain += raft::log(rval * invRight) / raft::log(DataT(2)) * rval * invLen;
                var rarg = _ftz_seam(rval * invRight)
                var r1 = _ftz_seam(_log_seam(rarg) / _log_seam(Two))
                var r2 = _ftz_seam(r1 * rval)
                var r3 = _ftz_seam(r2 * invLen)
                gain = _ftz_seam(gain + r3)

            val_i += rval_i
            if val_i != 0:
                var val = _ftz_seam(Scalar[Self.dtype](Int(val_i)) * invLen)
                # gain -= val * raft::log(val) / raft::log(DataT(2));
                var v1 = _ftz_seam(val * _log_seam(val))
                var v2 = _ftz_seam(v1 / _log_seam(Two))
                gain = _ftz_seam(gain - v2)

        # DEVIATION 217, applied to entropy: the true information gain is
        # non-negative; a negative float is cancellation and would feed
        # `split_not_valid`. Both arms clamp, or they would grow different
        # trees.
        if gain < Scalar[Self.dtype](0.0):
            gain = Scalar[Self.dtype](0.0)
        return gain

    def GainKeyExact(self, gain: Scalar[Self.dtype], len: Int32) -> GiniProxyExact:
        """The float gain as the reduction's exact pair. DEVIATION 459.

        `num = float_gain_key(gain)`, `den = 1`, valid. `MIN_FINITE` -- the
        rejected-candidate sentinel -- is INVALID rather than keyed, so a
        rejected entropy candidate ranks exactly where a rejected Gini one
        does. `GiniProxyExact` is reused rather than given a sibling because
        the reduction's `ExactKey` already carries exactly these three
        fields and nothing downstream reads the name."""
        if gain == Scalar[Self.dtype].MIN_FINITE:
            return GiniProxyExact(0, 0, Int64(Int(len)), False)
        return GiniProxyExact(
            float_gain_key(gain.cast[DType.float32]()),
            1,
            Int64(Int(len)),
            True,
        )

    # ----------------------------------------------------------------------
    # sklearn's arithmetic, for REPORTING. `_criterion.pyx:532-602`.
    # ----------------------------------------------------------------------

    def NodeImpurity[
        mt: Bool, //, ot: Origin[mut=mt]
    ](
        self,
        hist_total: Pointer[CountBin, ot],
        weighted_n_node_samples: Scalar[Self.dtype],
    ) -> Scalar[Self.dtype]:
        """sklearn `Entropy.node_impurity`, `_criterion.pyx:548-567`, the
        `n_outputs == 1` body:

            for c: count_k = sum_total[c]
                   if count_k > 0.0: count_k /= w; entropy -= count_k * log(count_k)
            return entropy / n_outputs

        Their `log` is libm's double; here it is `_log_seam` on `dtype`
        (DEVIATION 459), because this value is reporting and the report
        must be one arithmetic everywhere under IDENTICAL."""
        var entropy = Scalar[Self.dtype](0.0)
        var count_k: Scalar[Self.dtype]
        for c in range(Int(self.nclasses)):
            count_k = Scalar[Self.dtype](Int(hist_total[unsafe_offset=c].x))
            if count_k > 0.0:
                count_k /= weighted_n_node_samples
                entropy -= count_k * _log_seam(count_k)
        return entropy

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
        """sklearn `Entropy.children_impurity`, `_criterion.pyx:569-602`,
        with `sum_right` recovered as `total - left` the way
        `GiniObjectiveFunction.ChildrenImpurity` recovers it."""
        var entropy_left = Scalar[Self.dtype](0.0)
        var entropy_right = Scalar[Self.dtype](0.0)
        var count_k: Scalar[Self.dtype]
        for c in range(Int(self.nclasses)):
            count_k = Scalar[Self.dtype](Int(hist_left[unsafe_offset=c].x))
            if count_k > 0.0:
                count_k /= weighted_n_left
                entropy_left -= count_k * _log_seam(count_k)

            count_k = Scalar[Self.dtype](
                Int(
                    hist_total[unsafe_offset=c].x
                    - hist_left[unsafe_offset=c].x
                )
            )
            if count_k > 0.0:
                count_k /= weighted_n_right
                entropy_right -= count_k * _log_seam(count_k)

        impurity_left = entropy_left
        impurity_right = entropy_right

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
        """sklearn's proxy (`_criterion.pyx:147-163` over `:569-602`), for
        REPORTING ONLY: entropy SELECTS on cuML's `GainPerSplit` (DEVIATION
        459), not on this. `min_samples_leaf` is applied as Gini's twin
        applies it, so the two report the same rejections."""
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

    @staticmethod
    def SetLeafVector[
        ms: Bool, //, os: Origin[mut=ms], oo: MutOrigin
    ](
        shist: Pointer[CountBin, os],
        nclasses: Int32,
        out_ptr: Pointer[Scalar[Self.dtype], oo],
    ):
        """`objectives.cuh:181-191` -- byte for byte Gini's `:97-107`, so it
        CALLS Gini's rather than transcribing the same ten lines twice."""
        GiniObjectiveFunction[Self.dtype].SetLeafVector(shist, nclasses, out_ptr)



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

    PRECISION IS RULED AND `dtype` STAYS A PARAMETER FOR A DIFFERENT REASON.
    DEVIATION 135 is closed: the device accumulates in FIXED POINT
    (`original/fixed_point.mojo`), because integer addition is associative
    and exact and a parallel float reduction has no fixed order. `dtype`
    remains a parameter because the two sides now legitimately differ -- the
    host oracle runs at `Float64`, matching sklearn, while the device runs
    over quantized integers -- and this file must serve both without naming
    either.
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

            # DEVIATION 217: the true variance reduction is non-negative; a
            # negative value is float cancellation (measured on year: a
            # valid 148k-row winner at -0.027 leafed half a tree). See
            # `builder.mojo::gain_per_split`; all three gain forms clamp
            # identically.
            if gain < Scalar[Self.dtype](0.0):
                gain = Scalar[Self.dtype](0.0)
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
