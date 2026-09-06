# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ExtraTrees split scoring: Gini/entropy classification and MSE regression."""

from std.math import fma, log

from checks.numerics import ftz, identical_log

from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import (
    classification_key_shift,
    float_gain_key,
)









comptime MAX_ROWS_EXACT: Int = 1 << 21
"""Row count per node at which `GiniProxyExact` is exact."""




@fieldwise_init
struct CountBin(ImplicitlyCopyable, Movable):
    """One class count."""

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
    """A label sum and a count. This file must not settle it, so it does not name a type: instantiate `AggregateBin[DType.float64]` on the host oracle and whatever 135 decides on the device, and the arithmetic below is identical either way."""

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




def proxy_impurity_improvement[
    dtype: DType
](
    weighted_n_left: Scalar[dtype],
    weighted_n_right: Scalar[dtype],
    impurity_left: Scalar[dtype],
    impurity_right: Scalar[dtype],
) -> Scalar[dtype]:
    """sklearn `Criterion.proxy_impurity_improvement`, `_criterion.pyx:147-163`."""
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
    """sklearn `Criterion.impurity_improvement`, `_criterion.pyx:165-199`."""
    return (weighted_n_node_samples / weighted_n_samples) * (
        impurity_parent
        - (weighted_n_right / weighted_n_node_samples * impurity_right)
        - (weighted_n_left / weighted_n_node_samples * impurity_left)
    )




@fieldwise_init
struct GiniProxyExact(ImplicitlyCopyable, Movable):
    """sklearn's Gini proxy held as an exact rational."""

    var num: Int64
    """`sq_L * nR + sq_R * nL`. Bounded by `n^3/4`; see `MAX_ROWS_EXACT`."""

    var den: Int64
    """`nL * nR`."""

    var length: Int64
    """`len`, the node's row count. Only `value()` uses it."""

    var valid: Bool
    """False when the candidate is rejected: `min_samples_leaf` (cuML `objectives.cuh:62-63`) or an empty child (ours, DEVIATION 144)."""

    def value[dtype: DType](self) -> Scalar[dtype]:
        """The proxy as a float, for reporting only."""
        if not self.valid:
            return Scalar[dtype].MIN_FINITE
        return Scalar[dtype](self.num) / Scalar[dtype](self.den) - Scalar[dtype](
            self.length
        )


@fieldwise_init
struct GiniObjectiveFunction[dtype: DType](Copyable, Movable):
    """Gini."""

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
        """Compute the gini impurity reduction for this split."""
        var nRight = len - nLeft
        comptime One = Scalar[Self.dtype](1.0)
        var invLen = One / Scalar[Self.dtype](Int(len))
        var invLeft = One / Scalar[Self.dtype](Int(nLeft))
        var invRight = One / Scalar[Self.dtype](Int(nRight))
        var gain = Scalar[Self.dtype](0.0)

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

        if gain < Scalar[Self.dtype](0.0):
            gain = Scalar[Self.dtype](0.0)
        return gain


    def NodeImpurity[
        mt: Bool, //, ot: Origin[mut=mt]
    ](
        self,
        hist_total: Pointer[CountBin, ot],
        weighted_n_node_samples: Scalar[Self.dtype],
    ) -> Scalar[Self.dtype]:
        """sklearn `Gini.node_impurity`, `_criterion.pyx:622-645`."""
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
        """sklearn `Gini.children_impurity`, `_criterion.pyx:647-687`."""
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
        """sklearn's SELECTION score, in float."""
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
        """sklearn's selection score as an exact rational."""

        var nRight = len - nLeft

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
        var cs = Int64(classification_key_shift(Int(len)))
        return GiniProxyExact(
            (sq_left >> cs) * nr + (sq_right >> cs) * nl,
            nl * nr,
            Int64(Int(len)),
            True,
        )

    @staticmethod
    def CompareProxyExact(a: GiniProxyExact, b: GiniProxyExact) -> Int:
        """Order two candidates by sklearn's proxy, exactly."""
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


    @staticmethod
    def SetLeafVector[
        ms: Bool, //, os: Origin[mut=ms], oo: MutOrigin
    ](
        shist: Pointer[CountBin, os],
        nclasses: Int32,
        out_ptr: Pointer[Scalar[Self.dtype], oo],
    ):
        """`objectives.cuh:97-107`, transcribed."""
        var total: Int32 = 0
        for i in range(Int(nclasses)):
            total += shist[unsafe_offset=i].x
        for i in range(Int(nclasses)):
            out_ptr[unsafe_offset=i] = Scalar[Self.dtype](
                Int(shist[unsafe_offset=i].x)
            ) / Scalar[Self.dtype](Int(total))




@always_inline
def _log_seam[
    dt: DType, //
](x: Scalar[dt]) -> Scalar[dt] where dt.is_floating_point():
    """`numerics.identical_log` for a generic-dtype seam -- the shape of RF's DEVIATION 406 `_log_seam`."""
    comptime if dt == DType.float32:
        return identical_log(x.cast[DType.float32]()).cast[dt]()
    return log(x)


@always_inline
def _ftz_seam[dt: DType, //](x: Scalar[dt]) -> Scalar[dt]:
    """`numerics.ftz` for a generic-dtype seam (RF's DEVIATION 405 shape): a comptime no-op under FAST, the row-10 flush under IDENTICAL."""
    comptime if dt == DType.float32:
        return ftz(x.cast[DType.float32]()).cast[dt]()
    return x


@fieldwise_init
struct EntropyObjectiveFunction[dtype: DType](
    Copyable, Movable
) where dtype.is_floating_point():
    """Entropy."""

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
        """compute the Entropy (or information gain) for each split."""
        var nRight = len - nLeft
        var gain = Scalar[Self.dtype](0.0)
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
                var rarg = _ftz_seam(rval * invRight)
                var r1 = _ftz_seam(_log_seam(rarg) / _log_seam(Two))
                var r2 = _ftz_seam(r1 * rval)
                var r3 = _ftz_seam(r2 * invLen)
                gain = _ftz_seam(gain + r3)

            val_i += rval_i
            if val_i != 0:
                var val = _ftz_seam(Scalar[Self.dtype](Int(val_i)) * invLen)
                var v1 = _ftz_seam(val * _log_seam(val))
                var v2 = _ftz_seam(v1 / _log_seam(Two))
                gain = _ftz_seam(gain - v2)

        if gain < Scalar[Self.dtype](0.0):
            gain = Scalar[Self.dtype](0.0)
        return gain

    def GainKeyExact(self, gain: Scalar[Self.dtype], len: Int32) -> GiniProxyExact:
        """The float gain as the reduction's exact pair."""
        if gain == Scalar[Self.dtype].MIN_FINITE:
            return GiniProxyExact(0, 0, Int64(Int(len)), False)
        return GiniProxyExact(
            float_gain_key(gain.cast[DType.float32]()),
            1,
            Int64(Int(len)),
            True,
        )


    def NodeImpurity[
        mt: Bool, //, ot: Origin[mut=mt]
    ](
        self,
        hist_total: Pointer[CountBin, ot],
        weighted_n_node_samples: Scalar[Self.dtype],
    ) -> Scalar[Self.dtype]:
        """sklearn `Entropy.node_impurity`, `_criterion.pyx:548-567`, the `n_outputs == 1` body: for c: count_k = sum_total[c] if count_k > 0.0: count_k /= w; entropy -= count_k * log(count_k) return entropy / n_outputs THEIR `log` IS BASE 2, NOT libm's NATURAL LOG. `_log_seam` is `identical_log` on `dtype` (DEVIATION 459), because this value is reporting and the report must be one arithmetic everywhere under IDENTICAL."""
        comptime Two = Scalar[Self.dtype](2.0)
        var entropy = Scalar[Self.dtype](0.0)
        var count_k: Scalar[Self.dtype]
        for c in range(Int(self.nclasses)):
            count_k = Scalar[Self.dtype](Int(hist_total[unsafe_offset=c].x))
            if count_k > 0.0:
                count_k /= weighted_n_node_samples
                entropy -= count_k * (_log_seam(count_k) / _log_seam(Two))
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
        """sklearn `Entropy.children_impurity`, `_criterion.pyx:569-602`, with `sum_right` recovered as `total - left` the way `GiniObjectiveFunction.ChildrenImpurity` recovers it."""
        comptime Two = Scalar[Self.dtype](2.0)
        var entropy_left = Scalar[Self.dtype](0.0)
        var entropy_right = Scalar[Self.dtype](0.0)
        var count_k: Scalar[Self.dtype]
        for c in range(Int(self.nclasses)):
            count_k = Scalar[Self.dtype](Int(hist_left[unsafe_offset=c].x))
            if count_k > 0.0:
                count_k /= weighted_n_left
                entropy_left -= count_k * (_log_seam(count_k) / _log_seam(Two))

            count_k = Scalar[Self.dtype](
                Int(
                    hist_total[unsafe_offset=c].x
                    - hist_left[unsafe_offset=c].x
                )
            )
            if count_k > 0.0:
                count_k /= weighted_n_right
                entropy_right -= count_k * (
                    _log_seam(count_k) / _log_seam(Two)
                )

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
        """sklearn's proxy (`_criterion.pyx:147-163` over `:569-602`), for REPORTING ONLY: entropy SELECTS on cuML's `GainPerSplit` (DEVIATION 459), not on this."""
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
        """`objectives.cuh:181-191` -- byte for byte Gini's `:97-107`, so it CALLS Gini's rather than transcribing the same ten lines twice."""
        GiniObjectiveFunction[Self.dtype].SetLeafVector(shist, nclasses, out_ptr)





@fieldwise_init
struct MSEObjectiveFunction[dtype: DType](Copyable, Movable):
    """MSE. `dtype` remains a parameter because the two sides now legitimately differ -- the host oracle runs at `Float64`, matching sklearn, while the device runs over quantized integers -- and this file must serve both without naming either."""

    comptime DataT = Scalar[Self.dtype]
    """Theirs is `using DataT = DataT_;` (`objectives.cuh:198`)."""

    comptime BinT = AggregateBin[Self.dtype]
    """Theirs is `using BinT = AggregateBin;` (`objectives.cuh:201`)."""

    var min_samples_leaf: Int32
    """Theirs is `IdxT min_samples_leaf` (`objectives.cuh:204`)."""

    def NumClasses(self) -> Int32:
        """`objectives.cuh:257`: `DI IdxT NumClasses() const { return 1; }`."""
        return 1


    def GainPerSplit(
        self,
        hist_left: AggregateBin[Self.dtype],
        hist_total: AggregateBin[Self.dtype],
        len: Int32,
        nLeft: Int32,
    ) -> Scalar[Self.dtype]:
        """Compute the MSE impurity reduction for this split."""
        var gain: Scalar[Self.dtype]
        var nRight = len - nLeft
        var invLen = Scalar[Self.dtype](1.0) / Scalar[Self.dtype](Int(len))
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

            if gain < Scalar[Self.dtype](0.0):
                gain = Scalar[Self.dtype](0.0)
            return gain


    def NodeImpurity(
        self,
        sq_sum_total: Scalar[Self.dtype],
        hist_total: AggregateBin[Self.dtype],
        weighted_n_node_samples: Scalar[Self.dtype],
    ) -> Scalar[Self.dtype]:
        """sklearn `MSE.node_impurity`, `_criterion.pyx:928-942`."""
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
        """sklearn `MSE.proxy_impurity_improvement`, `_criterion.pyx:944-973`."""
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
        """sklearn `MSE.children_impurity`, `_criterion.pyx:975-1017`."""
        var sq_sum_right = sq_sum_total - sq_sum_left

        impurity_left = sq_sum_left / weighted_n_left
        impurity_right = sq_sum_right / weighted_n_right

        impurity_left -= (hist_left.label_sum / weighted_n_left) ** 2.0
        impurity_right -= (
            (hist_total.label_sum - hist_left.label_sum) / weighted_n_right
        ) ** 2.0


    @staticmethod
    def SetLeafVector[
        ms: Bool, //, os: Origin[mut=ms], oo: MutOrigin
    ](
        shist: Pointer[AggregateBin[Self.dtype], os],
        nclasses: Int32,
        out_ptr: Pointer[Scalar[Self.dtype], oo],
    ):
        """`objectives.cuh:259-264`, transcribed."""
        for i in range(Int(nclasses)):
            out_ptr[unsafe_offset=i] = shist[
                unsafe_offset=i
            ].label_sum / Scalar[Self.dtype](
                Int(shist[unsafe_offset=i].count)
            )
