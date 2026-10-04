# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""AutoARIMA: the order search of cuML `python/cuml/cuml/tsa/auto_arima.pyx`
(`AutoARIMA.search`, `fit`, `predict`, `forecast`) over mojolearn's batched
`ARIMA` (`arima/`, the `_mojolearn_arima` binding) and the KPSS choice of `d`
(`tsa/`, `select_d`).

The search, as the reference's:
  1. `d` per series: the KPSS test at d = 0, 1, ... (`select_d`), unless one
     `d` is given; series are grouped by their `d`.
  2. For each group, every (p, q, P, Q, k) of the product of the options, in
     `itertools.product` order, skipping p + q + P + Q + k == 0, is fitted on
     the whole group as one batch; each series keeps the order of least
     information criterion, the FIRST such order on a tie (the reference's
     `_divide_by_min` is an argmin).
  3. `fit` refits every chosen order on its sub-batch.
The information criterion is batched_arima.cu `information_criterion`'s:
-2 loglike + 2N (aic), + 2N(N+1)/(n - N - 1) (aicc), log(n) N (bic), with
n = n_obs - d - s D and N the model's parameter count, computed in float64 in
Mojo (`ic_running_min_f64`) from the fit's float32 log-likelihood. Builds whose
ARIMA binding reports `arima_order_caps() & 4` (IDENTICAL since lane
fam2-timeseries) take the criterion in float32, one rounding of the same
value, on the device with the order choice (`arima_order_search_device`) or,
on the host column and for grids the grouped search declines, in Mojo
(`ic_running_min_f32`).

Layout: mojolearn's ARIMA layout, `(batch_size, n_obs)`, one series per row
(cuML's AutoARIMA takes series in columns; transpose to call it the same way).
`s` of 0, None or 1 is non-seasonal (the reference's `if s is 1: s = None`).
Refused by name: `seasonal_test="seas"` over a `D` list of two options
(statsmodels' STL, not a GPU path in the reference either; pass one `D`), `method` css / css-ml (this
ARIMA fits by `ml`), a `d` option list that is not 0..d_max, `truncate`,
`h` other than the default, prediction intervals (`level`)."""
import itertools

from ._optional_numpy import require_numpy
np = require_numpy('_x_sequence_autoarima')

from . import _portable_math as _pm
from ._arima_impl import ARIMA
from ._tsa_impl import select_d
from ._buffer import _native

_IC = ("aic", "aicc", "bic")


def _rows_where(words, want, count):
    """The ascending positions whose int32 word is `want` (int64, `count`
    of them), by the base binding's select_fold_i64."""
    n = len(words)
    out = np.empty(max(count, 1), dtype=np.int64)
    scratch = np.empty(max(n, 1), dtype=np.int64)
    got = int(_native("select_fold_i64")(words.ctypes.data, n, int(want),
                                         out.ctypes.data if count else scratch.ctypes.data, scratch.ctypes.data))
    if got != count:
        raise RuntimeError("AutoARIMA: select_fold_i64 disagrees with the counts")
    return out[:count]


def _counts(words, k):
    """Rows per int32 word value in [0, k) (bincount)."""
    out = np.zeros(k, dtype=np.int64)
    if len(words):
        _native("bincount_i64")(words.ctypes.data, 2, len(words), k, out.ctypes.data, 0)
    return out


def _take(block, ids):
    """block[ids] for a C-contiguous 2-D (or 1-D) block, a byte gather."""
    out = np.empty((len(ids),) + block.shape[1:], dtype=block.dtype)
    if len(ids):
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        _native("gather_rows_bytes")(block.ctypes.data, out.ctypes.data, ids.ctypes.data, block.shape[0],
                                     len(ids), block.itemsize * (block.size // max(block.shape[0], 1)))
    return out


def _put(block, ids, rows):
    """block[ids] = rows, a byte scatter."""
    rows = np.ascontiguousarray(rows, dtype=block.dtype)
    if len(ids):
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        _native("scatter_rows_bytes")(rows.ctypes.data, ids.ctypes.data, len(ids),
                                      block.itemsize * (block.size // max(block.shape[0], 1)),
                                      block.ctypes.data, block.shape[0])


def _options(name, value, lo, hi):
    """The reference's `_parse_sequence`: an int or an iterable, clipped to
    [lo, hi]; empty is refused."""
    vals = [int(value)] if isinstance(value, (int, np.integer)) else [int(v) for v in value]  # glue: user order options, a few ints
    vals = [v for v in vals if lo <= v <= hi]  # glue: user order options, a few ints
    if not vals:
        raise ValueError(f"AutoARIMA: no valid option for {name} in [{lo}, {hi}]")
    return vals


class AutoARIMA:
    """Batched ARIMA order search. `endog` is `(batch_size, n_obs)` float32
    (or one 1-D series)."""

    #: The bindings the search runs, through `ARIMA` and `select_d` (named
    #: here so the lane map derives them and their CPU host bindings).
    _ARIMA_BINDING = "_mojolearn_arima"
    _TSA_BINDING = "_mojolearn_tsa"

    def __init__(self, endog):
        y = np.asarray(endog)
        if y.dtype == np.float64:
            raise TypeError("AutoARIMA: float64 is refused; pass float32 (no float64 on the device)")
        y = np.ascontiguousarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        if y.ndim != 2 or y.size == 0:
            raise ValueError("AutoARIMA: endog must be (batch_size, n_obs)")
        self.endog = y
        self.batch_size, self.n_obs = int(y.shape[0]), int(y.shape[1])
        self.models = []

    def search(self, s=None, d=range(3), D=range(2), p=range(1, 4), q=range(1, 4), P=range(3), Q=range(3),
               fit_intercept="auto", ic="aicc", test="kpss", seasonal_test="seas", h=1e-8, maxiter=20,
               method="auto", truncate=0):
        if ic not in _IC:
            raise ValueError(f"AutoARIMA: ic must be one of {_IC}")
        if test != "kpss":
            raise ValueError("AutoARIMA: test must be 'kpss', the reference's only stationarity test")
        if method not in ("auto", "ml"):
            raise NotImplementedError("AutoARIMA: this ARIMA fits by 'ml' only; css and css-ml are refused")
        if h != 1e-8 or truncate:
            raise NotImplementedError("AutoARIMA: h and truncate are not carried by mojolearn.ARIMA")
        # the reference's `if s is 1: s = None` (R users pass s=1 for a
        # non-seasonal series): s <= 1 is non-seasonal, so D = P = Q = 0 and
        # p, q range over 0..4, not 0..s-1
        s = int(s) if s and int(s) > 1 else 0
        if s:
            # the reference's D choice: one option is taken as given; only a
            # list of several runs the seasonal test
            D_opts = _options("D", D, 0, 1)
            if len(D_opts) > 1:
                raise NotImplementedError(
                    f"AutoARIMA: seasonal_test={seasonal_test!r} (statsmodels' STL in the reference) is "
                    "not implemented; pass D as one value")
            D_ = D_opts[0]
        else:
            D_ = 0
        d_opts = _options("d", d, 0, 2 - D_)
        p_opts = _options("p", p, 0, s - 1 if s else 4)
        q_opts = _options("q", q, 0, s - 1 if s else 4)
        P_opts = _options("P", P, 0, 4 if s else 0)
        Q_opts = _options("Q", Q, 0, 4 if s else 0)
        y = self.endog
        if len(d_opts) == 1:
            dser = np.full(self.batch_size, d_opts[0], dtype=np.int64)
        else:
            if d_opts != list(range(len(d_opts))):
                raise NotImplementedError("AutoARIMA: a d option list must be 0, 1, ..., d_max")
            # the whole batch in one call, (n_obs, n_series) (apple2: one call
            # per series was 2000 launches and waits, 2.4 s of the search on
            # an M4 Pro). select_d is a pure function of each series' bits
            # (tsa/checks/stationarity_check.mojo gates the batch composition
            # invariant), so every series' d is the one-series call's.
            dser = np.asarray(select_d(np.ascontiguousarray(y.T), D=D_, s=s, d_max=d_opts[-1]),
                              dtype=np.int64).reshape(-1)
        self.d_ = dser
        self.models, self._ids = [], []
        self.order_ = np.zeros((self.batch_size, 8), dtype=np.int64)
        self.ic_ = np.zeros(self.batch_size, dtype=np.float64)
        # the series of each d by the base binding (bincount, select, gather;
        # lane cgr4-py-compute), the groups in ascending d
        dw = np.ascontiguousarray(dser, dtype=np.int32)
        dcount = _counts(dw, 3)
        for d_ in range(3):  # glue: the three differencing orders d
            if not dcount[d_]:
                continue
            ids = _rows_where(dw, d_, int(dcount[d_]))
            sub = _take(y, ids)
            k_opts = ([1 if d_ + D_ <= 1 else 0] if fit_intercept == "auto"
                      else _options("k", fit_intercept, 0, 1))
            orders, nb = [], len(ids)
            binding = ARIMA()._extension()
            caps_fn = getattr(binding, "arima_order_caps", None)
            caps = int(caps_fn()) if caps_fn is not None else 0
            ic_dtype = np.float32 if caps & 4 else np.float64
            ic_fold = _native("ic_running_min_f32" if caps & 4 else "ic_running_min_f64")
            best_ic = np.empty(nb, dtype=ic_dtype)
            best = np.empty(nb, dtype=np.int64)
            ic_k = np.empty(nb, dtype=ic_dtype)
            # Metadata only: retain the exact itertools.product order. The
            # native experiment groups GPU work by state dimension but writes
            # likelihood rows back in this order, preserving first-min ties.
            grid = [o for o in itertools.product(p_opts, q_opts, P_opts, Q_opts, k_opts)  # glue: user order metadata
                    if o[0] + o[1] + o[2] + o[3] + o[4]]  # glue: user order options, no series data
            grouped_ll = None
            chosen_on_device = False
            plain = not s and D_ == 0 and 4 not in p_opts and 4 not in q_opts
            enabled = getattr(binding, "arima_order_batch_enabled", None)
            if grid and self.n_obs > 2 and enabled is not None and enabled():
                if caps & 1 and (plain or caps & 2):
                    # lane fam2-timeseries: the grouped search on the device
                    # end to end (seasonal grids with caps & 2); with
                    # caps & 4 the criterion and the first-minimum choice
                    # are device kernels and only the choice crosses back.
                    # 0 written = a grid the build declines (its workspace
                    # bound, or the period rule): the per-order fits below.
                    want_ic = 1 if caps & 4 else 0
                    packed_grid = [v for o in grid for v in o]  # glue: native order metadata arguments
                    pens = [self._penalty_of(p_ + q_ + P_ + Q_ + k_ + 1, ic, d_, D_,  # glue: one scalar per order
                                             s if (P_ + D_ + Q_) else 0)
                            for p_, q_, P_, Q_, k_ in grid]
                    best32 = np.empty(nb, dtype=np.int32)
                    out = np.empty(nb if want_ic else len(grid) * nb, dtype=np.float32)
                    written = int(binding.arima_order_search_device(
                        sub.ctypes.data, out.ctypes.data, best32.ctypes.data, packed_grid, pens,
                        [nb, self.n_obs, d_, D_, s, int(maxiter), want_ic]))
                    if written and written != out.size:
                        raise RuntimeError("AutoARIMA: incomplete grouped search output")
                    if written and want_ic:
                        chosen_on_device = True
                        best = np.asarray(best32, dtype=np.int64)  # glue: widen the device's choice
                        best_ic = out
                    elif written:
                        grouped_ll = out.reshape(len(grid), nb)
                elif plain:
                    grouped_ll = np.empty((len(grid), nb), dtype=np.float32)
                    packed_grid = [v for p_, q_, _, _, k_ in grid for v in (p_, q_, k_)]  # glue: native order metadata arguments
                    written = int(binding.arima_order_search(sub.ctypes.data, grouped_ll.ctypes.data,
                                                             packed_grid, [nb, self.n_obs, d_, int(maxiter)]))
                    if written == 0:
                        grouped_ll = None  # the build's workspace bound: per-order fits
                    elif written != grouped_ll.size:
                        raise RuntimeError("AutoARIMA: incomplete grouped likelihood output")
            for trial, (p_, q_, P_, Q_, k_) in enumerate(grid):  # glue: user order grid, not series or observations
                s_ = s if (P_ + D_ + Q_) else 0
                orders.append((p_, q_, P_, Q_, s_, k_))
                if chosen_on_device:
                    continue
                m = ARIMA(order=(p_, d_, q_), seasonal_order=(P_, D_, Q_, s_),
                          trend="c" if k_ else "n", maxiter=maxiter)
                if grouped_ll is None:
                    m.fit(sub)
                    llf = np.ascontiguousarray(m.llf_, dtype=np.float32).reshape(-1)
                else:
                    llf = np.ascontiguousarray(grouped_ll[trial])
                # every series' criterion and the running first-minimum
                # choice, in Mojo (lane cgr4-py-compute)
                ic_fold(llf.ctypes.data, nb, self._penalty(m, ic, d_, D_, s_),
                        trial, ic_k.ctypes.data, best_ic.ctypes.data, best.ctypes.data)
            if not orders:
                raise ValueError("AutoARIMA: no (p, q, P, Q, k) order to try")
            table = np.asarray([[p_, d_, q_, P_, D_, Q_, s_, k_] for (p_, q_, P_, Q_, s_, k_) in orders],  # glue: one table row per tried order
                               dtype=np.int64)
            _put(self.order_, ids, _take(table, best))
            _put(self.ic_, ids, best_ic)
            bw = np.ascontiguousarray(best, dtype=np.int32)
            bcount = _counts(bw, len(orders))
            for i, (p_, q_, P_, Q_, s_, k_) in enumerate(orders):  # glue: tried orders, not series or observations
                if not bcount[i]:
                    continue
                chosen = _take(ids, _rows_where(bw, i, int(bcount[i])))
                self.models.append(((p_, d_, q_), (P_, D_, Q_, s_), k_))
                self._ids.append(chosen)
        self._fitted = [None] * len(self.models)
        return self

    def _penalty(self, m, ic, d_, D_, s_):
        """The criterion's parameter penalty (a scalar per order): 2N (aic),
        + 2N(N+1)/(n - N - 1) (aicc), log(n) N (bic); the criterion is
        -2 loglike + penalty, per series in `ic_running_min_f64`."""
        return self._penalty_of(m.complexity_, ic, d_, D_, s_)

    def _penalty_of(self, complexity, ic, d_, D_, s_):
        """`_penalty` from the parameter count alone (no model object)."""
        N = float(complexity)
        n = float(self.n_obs - d_ - s_ * D_)
        if ic == "aic":
            return 2.0 * N
        if ic == "aicc":
            return 2.0 * N + 2.0 * N * (N + 1.0) / (n - N - 1.0)
        return _pm.log(n) * N  # the pinned log, as ARIMA.bic_ (DEVIATION 6900)

    def fit(self, h=1e-8, maxiter=1000, method="ml", truncate=0):
        if not self.models:
            raise RuntimeError("AutoARIMA: call search() before fit()")
        if h != 1e-8 or truncate or method != "ml":
            raise NotImplementedError("AutoARIMA.fit: method 'ml' with the default h only")
        for i, (order, sorder, k) in enumerate(self.models):  # glue: chosen order groups, at most the grid
            self._fitted[i] = ARIMA(order=order, seasonal_order=sorder, trend="c" if k else "n",
                                    maxiter=maxiter).fit(_take(self.endog, self._ids[i]))
        return self

    def _gather(self, fn, width):
        if any(m is None for m in self._fitted):  # glue: chosen order groups, at most the grid
            raise RuntimeError("AutoARIMA: call fit() first")
        out = np.zeros((self.batch_size, width), dtype=np.float32)
        for m, ids in zip(self._fitted, self._ids):  # glue: chosen order groups, at most the grid
            _put(out, ids, np.asarray(fn(m), dtype=np.float32).reshape(len(ids), width))
        return out

    def predict(self, start=0, end=None, level=None):
        if level is not None:
            raise NotImplementedError("AutoARIMA.predict: prediction intervals (level) are not implemented")
        end = self.n_obs if end is None else int(end)
        return self._gather(lambda m: m.predict(start, end), end - int(start))

    def forecast(self, nsteps, level=None):
        if level is not None:
            raise NotImplementedError("AutoARIMA.forecast: prediction intervals (level) are not implemented")
        return self._gather(lambda m: m.forecast(int(nsteps)), int(nsteps))
