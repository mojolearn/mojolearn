"""Time-series estimators on the GPU, mirroring cuML's `cuml.tsa`.

PRIVATE ON PURPOSE. Nothing here is re-exported from `mojolearn/__init__.py`
by this file; that is the package owner's call. Import it as
`from mojolearn import _tsa_impl` until it is.

WHAT IS HERE

    ExponentialSmoothing   cuml.tsa.ExponentialSmoothing, backed by
                           `holtwinters/` (DEVIATIONS 660-665, 697-699)
    kpss_test              cuml.tsa.stationarity.kpss_test, backed by
                           `tsa/` (DEVIATIONS 671-672)
    select_d               auto_arima's "Choose the hyper-parameter d"
                           block, backed by the same lane

WHAT IS NOT HERE, AND WHY IT IS NOT

    ARIMA / AutoARIMA      `arima/` ports cuML's batched Kalman filter and
                           does it well: given ARIMA coefficients it
                           computes the log-likelihood, the finite-
                           difference gradient, the in-sample predictions
                           and the forecast. It does NOT port
                           `estimate_x0` / `_start_params` /
                           `_arma_least_squares`, nor `arima.pyx`'s batched
                           L-BFGS driver; both are listed NOT PORTED in
                           `arima/UNPORTED.tsv`. There is no `fit`, and the
                           coefficients its entry points require are the
                           output of the optimizer that is missing. A class
                           named `ARIMA` whose `fit` did not exist, or
                           whose `fit` demanded the answer as an argument,
                           would be a published lie about a lane that is
                           honestly unfinished at the surface. It stays
                           unreachable until the optimizer lands.

THESE ESTIMATORS ARE NOT sklearn-SHAPED, AND THAT IS DELIBERATE. The rest
of `mojolearn` promises `fit(X, y)` over a design matrix and `predict(X)`.
Time series do not have that shape and neither does cuML's surface for
them, so these follow cuML: the data goes into `ExponentialSmoothing`'s
CONSTRUCTOR as `endog`, `fit()` takes nothing, and the successor method is
`forecast(h)` rather than `predict`. Every default below is cuML's, not
statsmodels', and the two differ (statsmodels' `ExponentialSmoothing` has
no `ts_num`, defaults `seasonal=None`, and optimizes with L-BFGS-B rather
than the BFGS in `hw_optim.cuh`).

CROSS-VENDOR STATUS, STATED PLAINLY BECAUSE IT IS THE LIBRARY'S HEADLINE
CLAIM AND IT DOES NOT APPLY HERE. mojolearn's distinguishing property is
byte-identical float32 results across Apple, NVIDIA and AMD. Neither lane
behind this module has that property established. Both were gated on ONE
Apple M4, in both numeric modes, against a host oracle; neither has ever
been carried on an E1 leg (`tools/e1_bootstrap.sh` phase 8 names gemm, cd,
kde, linkage, svm, metrics and mamba, and none of these), so neither
appears in any `bench/results/e1/*/lanes/` artifact and neither is judged
by `tools/e3_round_judge.sh` section 7. `holtwinters/` holds row 57 of
`IDENTITY_PATHS.md` and that row says in its own words that no second
vendor has run it; `arima/` holds row 58 on the same terms; `tsa/` has no
row in the ledger at all. Under `MOJOLEARN_NUMERIC_MODE=identical` these
run the pinned spelling that is DESIGNED to be vendor-independent, which
is a claim about the source and not a measurement of three GPUs.

UPSTREAM IS RETIRING WHAT ExponentialSmoothing MIRRORS. The pinned tree's
`holtwinters.pyx` carries a `.. deprecated:: 26.08` and says
`cuml.tsa.ExponentialSmoothing` will be removed in cuML 26.12. The port is
faithful to v26.08.00 and stays valid; what expires is the ability to
check our numbers against a real cuML run.
"""

import importlib.machinery
import importlib.util
import os
import sys

import numpy as np

from ._arrays import _addr, _addr_ro

# The numeric-mode selector (`_backend.select()`, called from
# `mojolearn/__init__.py`) installs the identical build of each extension
# under its canonical module name -- but only for the modules named in its
# `_MODULES` tuple, and `_mojolearn_tsa` is not one of them yet. So: use
# what the selector installed if it installed anything, and otherwise load
# the binary the requested mode asks for, BY PATH. Never fall back to the
# fast binary when identical was asked for; a wrong-mode module that
# imports is a mislabelled measurement, which is `_backend.py`'s own rule
# and the reason it plants raising stubs instead of falling back.
def _load_binding():
    pkg_name = __name__.rsplit(".", 1)[0]
    pkg = sys.modules[pkg_name]
    installed = getattr(pkg, "_mojolearn_tsa", None)
    if installed is not None:
        return installed
    from . import _backend

    mode = _backend.requested_mode()
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if mode == "identical":
        path = os.path.join(pkg_dir, "identical", "_mojolearn_tsa.so")
    else:
        path = os.path.join(pkg_dir, "_mojolearn_tsa.so")
    if not os.path.exists(path):
        raise ImportError(
            f"mojolearn: {path} is not built; build it with\n    "
            + ("MOJOLEARN_NUMERIC_MODE=identical " if mode == "identical" else "")
            + "bash bindings/build_tsa.sh"
        )
    full = f"{pkg_name}._mojolearn_tsa"
    loader = importlib.machinery.ExtensionFileLoader(full, path)
    spec = importlib.util.spec_from_loader(full, loader, origin=path)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    sys.modules[full] = module
    setattr(pkg, "_mojolearn_tsa", module)
    return module


_mojolearn_tsa = _load_binding()


def _series_major(y, name):
    """A C-contiguous float32 buffer whose flat order is SERIES-MAJOR.

    `y` arrives as cuML's `(n_obs, batch_size)` -- each time series in a
    COLUMN, which is what `stationarity.pyx`'s `check_array(y, order="F")`
    produces. The kernels index series `b` at `[b * n_obs, (b+1) * n_obs)`,
    so the transpose happens here. It COPIES, always, and that is named
    rather than hidden: on a long batch it is the dominant cost of the
    call, and a caller who already holds `(batch_size, n_obs)` can avoid
    it by handing that in transposed.

    Returns `(array, n_obs, batch_size)`. The caller MUST keep `array`
    alive across the Mojo call; the Mojo side borrows the address and
    holds nothing after it returns (`_arrays.py`).
    """
    a = np.asarray(y)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(
            f"mojolearn {name}: y must be 1-D or 2-D (n_obs, n_series), got "
            f"{a.ndim}-D shape {a.shape}"
        )
    if a.size == 0:
        raise ValueError(f"mojolearn {name}: y is empty, shape {a.shape}")
    n_obs, batch_size = a.shape
    flat = np.ascontiguousarray(a.T, dtype=np.float32)
    return flat, int(n_obs), int(batch_size)


def kpss_test(y, d=0, D=0, s=0, pval_threshold=0.05, return_statistic=False):
    """The KPSS stationarity test, mirroring
    `cuml.tsa.stationarity.kpss_test` (`tsa/`, DEVIATIONS 671-672).

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY -- one line per parameter,
    because a parameter that is accepted and ignored is a wrong answer
    waiting for a caller:

        y               honored   `(n_obs, n_series)`, each series in a
                                  COLUMN, which is cuML's layout. A 1-D
                                  array is one series. NON-FINITE VALUES
                                  ARE REFUSED BY NAME with the offending
                                  index, where cuML passes
                                  `ensure_all_finite=False` and lets a NaN
                                  flow into the statistic
                                  (`tsa/UNPORTED.tsv`).
        d               honored   order of simple differencing
        D               honored   order of seasonal differencing
        s               honored   seasonal period; `D > 0` needs `s >= 2`,
                                  refused by name in `prepare_data`
        pval_threshold  honored   default 0.05, cuML's
        return_statistic honored  OURS, not cuML's: cuML returns the flags
                                  only. False (the default) returns exactly
                                  what cuML returns.
        float64 input   CONVERTED to float32, and the conversion COPIES.
                                  Named rather than hidden. cuML runs this
                                  test on `double` too and picks the arm
                                  from the input dtype; Metal has no
                                  float64, so there is no double arm here
                                  to pick (DEVIATION 670). Results are
                                  float32 answers to a float32 problem, not
                                  cuML's float64 answers.

    `d + D > 2` is refused by name (`prepare_data`; cuML enforces the same
    bound one layer up at `arima.pyx:313`).

    Returns a bool array of length `n_series`, True where the series is
    judged stationary after differencing. With `return_statistic=True`,
    returns `(stationary, statistic)` where `statistic` is float32. A
    constant series has statistic `0.0` rather than a computed NaN
    (DEVIATION 672) and is judged stationary, which is the decision cuML's
    NaN also falls through to.

    Cross-vendor status: see this module's docstring. One Apple M4.
    """
    flat, n_obs, batch_size = _series_major(y, "kpss_test")
    flags = np.empty(batch_size, dtype=np.int32)
    stat = np.empty(batch_size, dtype=np.float32)
    _mojolearn_tsa.kpss_test(
        _addr_ro(flat),
        _addr(flags),
        _addr(stat),
        # ORDER MATCHES bindings/_mojolearn_tsa.mojo::kpss_test_binding.
        #   0 batch_size, 1 n_obs, 2 d, 3 D, 4 s, 5 pval_threshold
        [
            int(batch_size),
            int(n_obs),
            int(d),
            int(D),
            int(s),
            float(pval_threshold),
        ],
    )
    stationary = flags.astype(bool)
    if return_statistic:
        return stationary, stat
    return stationary


def select_d(y, D=0, s=0, d_max=None, pval_threshold=0.05):
    """Choose the simple differencing order `d` per series.

    This is the "Choose the hyper-parameter d" block of
    `AutoARIMA.search` (`auto_arima.pyx:318-343`) and nothing else of
    auto_arima. It runs `kpss_test` for `d = 0, 1, ... d_max - 1` and
    takes the first order at which a series tests stationary, falling back
    to `d_max`.

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY:

        y               honored   the same layout and the same non-finite
                                  refusal as `kpss_test`
        D               honored   and REQUIRED. cuML CHOOSES `D` with
                                  `seasonal_test="seas"`, which is
                                  statsmodels' STL on the host; that is not
                                  a GPU path in cuML either and it is not
                                  ported (`tsa/UNPORTED.tsv`). Pass the `D`
                                  you want.
        s               honored   seasonal period
        d_max           honored   None (the default) means `2 - D`, which
                                  is cuML's `d_options = range(0, 2 - D + 1)`
        pval_threshold  honored   default 0.05, cuML's
        the p/q/P/Q/k   REFUSED   auto_arima's information-criterion grid
        search                    over ARIMA fits is not ported, because
                                  the ARIMA fit it searches over is not
                                  ported. There is no `AutoARIMA` class.

    Returns an int32 array of length `n_series`.

    A DIFFERENCE FROM cuML THAT CHANGES NO ANSWER. cuML physically splits
    the batch after each round (`_divide_by_mask`) so the next test runs on
    the undecided sub-batch; this runs every round on the full batch and
    masks on the host. The test is per series and its result is a pure
    function of that series' bits, which is the property
    `check_kpss_batch_composition_invariant` gates in
    `tsa/mojo_only/stationarity_check.mojo`.

    Cross-vendor status: see this module's docstring. One Apple M4.
    """
    if d_max is None:
        d_max = 2 - int(D)
    flat, n_obs, batch_size = _series_major(y, "select_d")
    out = np.empty(batch_size, dtype=np.int32)
    _mojolearn_tsa.select_d(
        _addr_ro(flat),
        _addr(out),
        # ORDER MATCHES bindings/_mojolearn_tsa.mojo::select_d_binding.
        #   0 batch_size, 1 n_obs, 2 D, 3 s, 4 d_max, 5 pval_threshold
        [
            int(batch_size),
            int(n_obs),
            int(D),
            int(s),
            int(d_max),
            float(pval_threshold),
        ],
    )
    return out


class ExponentialSmoothing:
    """Holt-Winters exponential smoothing, mirroring
    `cuml.tsa.ExponentialSmoothing` (`holtwinters/`, DEVIATIONS 660-665 and
    697-699).

    THE SHAPE IS cuML's, NOT sklearn's. `endog` goes in the CONSTRUCTOR,
    `fit()` takes no arguments, and the successor method is `forecast(h)`.
    There is no `predict(X)` here and there is none in cuML.

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY -- one line per parameter:

        endog            honored   `(ts_num, n)` with each series in a ROW,
                                   or 1-D for a single series. That is
                                   `holtwinters.pyx::_check_dims` on a
                                   numpy input. NON-FINITE VALUES ARE
                                   REFUSED BY NAME with the series and the
                                   position (DEVIATION 664), where cuML
                                   passes `ensure_all_finite=False`; under
                                   `seasonal='multiplicative'` a value that
                                   is not strictly positive is refused the
                                   same way, because their multiplicative
                                   arm divides by it.
        seasonal         honored   'additive'/'add' (default) or
                                   'multiplicative'/'mul'; anything else is
                                   refused by name in cuML's own words.
        seasonal_periods honored   cuML's frequency; must be >= 2.
        start_periods    honored   must be >= 2 and <= seasonal_periods.
        ts_num           honored   the number of series; must match
                                   `endog`'s first dimension, and cuML's
                                   mismatch message is mirrored.
        eps              honored   default 2.24e-3, cuML's. Must be > 0.
        verbose          REFUSED   cuML's logging plumbing; there is no
                                   logger here.
        output_type      REFUSED   cuML's cudf/cupy output selector. This
                                   returns numpy float32 and nothing else.
        float64 input    CONVERTED to float32, and the conversion
                                   COPIES. Named rather than hidden. cuML
                                   fits on `double` too and picks the arm
                                   from the input dtype; Metal has no
                                   float64, so there is no double arm here
                                   to pick (DEVIATION 670).
        OptimParams      ABSENT    cuML's `runner.cuh:236-253` override
                                   block is dead code in their own fit
                                   (`HoltWintersFitHelper` passes a null
                                   pointer), so the defaults at
                                   `runner.cuh:226-234` are always what
                                   runs. Exposing knobs their fit cannot
                                   reach would be inventing a surface.
        single-parameter REFUSED   `hw_optim.cuh`'s golden-section arm,
        optimization               taken only when exactly one of alpha /
                                   beta / gamma is optimized. `ML::
                                   HoltWinters::fit` optimizes all three,
                                   so the arm is unreachable from cuML's
                                   own surface; the port raises naming it
                                   rather than quietly running BFGS
                                   instead.

    ATTRIBUTES AFTER `fit()`

        level_, trend_, season_   `(ts_num, n - seasonal_periods)` float32,
                                  the fitted components. cuML's shape.
        sse_                      `(ts_num,)` float32. cuML's `SSE`.
        alpha_, beta_, gamma_     `(ts_num,)` float32, the fitted smoothing
                                  parameters. OURS: cuML leaves these in
                                  device scratch its Python surface never
                                  reads back.
        n_iter_                   `(ts_num,)` int32, BFGS iterations.
                                  OURS, DEVIATION 665.
        criterion_                `(ts_num,)` int32, why the optimizer
                                  stopped: 0 BFGS_ITER_LIMIT,
                                  1 MIN_PARAM_DIFF, 2 MIN_ERROR_DIFF,
                                  3 MIN_GRAD_NORM. OURS, DEVIATION 665;
                                  cuML writes this only in the arm its fit
                                  does not take.

    A cuML DEFECT THAT IS PORTED FAITHFULLY AND NOT FIXED. When the line
    search hits its iteration limit, `hw_optim.cuh:485-508` stores the LAST
    trial point rather than the one that minimized the loss. That is
    rapidsai/cuml#888 and it is flagged in their own comment. It is
    deterministic and vendor-independent, so fixing it would move the
    fitted parameters away from cuML's for no identity gain
    (`holtwinters/UNPORTED.tsv`).

    A DIVERGENCE FROM cuML's PYTHON THAT IS NOT A NUMERIC ONE. cuML caches
    `forecasted_points` and recomputes only when `h` grows, so a second
    `forecast(h)` with a smaller `h` returns a slice of the larger array.
    This recomputes every call. Same numbers, no cache to reason about.

    Cross-vendor status: see this module's docstring. One Apple M4, both
    numeric modes, against a host oracle; no second vendor has run it.
    """

    def __init__(
        self,
        endog,
        *,
        seasonal="additive",
        seasonal_periods=2,
        start_periods=2,
        ts_num=1,
        eps=2.24e-3,
    ):
        if seasonal not in ("additive", "add", "multiplicative", "mul"):
            raise ValueError(
                f"mojolearn ExponentialSmoothing: seasonal={seasonal!r} is "
                "refused; it must be 'additive'/'add' or "
                "'multiplicative'/'mul' (holtwinters.pyx:197)"
            )
        if not isinstance(ts_num, int) or isinstance(ts_num, bool):
            raise TypeError(
                "mojolearn ExponentialSmoothing: type of ts_num must be int. "
                f"Given: {type(ts_num)}"
            )
        if not isinstance(seasonal_periods, int) or isinstance(seasonal_periods, bool):
            raise TypeError(
                "mojolearn ExponentialSmoothing: type of seasonal_periods "
                f"must be int. Given: {type(seasonal_periods)}"
            )
        if not isinstance(start_periods, int) or isinstance(start_periods, bool):
            raise TypeError(
                "mojolearn ExponentialSmoothing: type of start_periods must "
                f"be int. Given: {type(start_periods)}"
            )
        self.endog = endog
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.start_periods = start_periods
        self.ts_num = ts_num
        self.eps = eps
        self.fit_executed_flag = False

    def _check_dims(self, ts_input):
        """`holtwinters.pyx:230-262` for a numpy input, by name.

        A 2-D array is `(ts_num, n)`: their `d1 = shape[1]` is `n` and
        their `d2 = shape[0]` is the series count, and they `ravel()` in C
        order, so each series is contiguous. A 1-D array is one series and
        `ts_num` must be 1.
        """
        a = np.asarray(ts_input)
        err = (
            "mojolearn ExponentialSmoothing: initialized with "
            f"{self.ts_num} time series, but data has dimension "
        )
        if a.ndim == 1:
            n = int(a.shape[0])
            if self.ts_num != 1:
                raise ValueError(err + "1.")
            d2 = 1
        elif a.ndim == 2:
            n = int(a.shape[1])
            d2 = int(a.shape[0])
            if self.ts_num != d2:
                raise ValueError(err + str(d2) + ".")
        else:
            raise ValueError(
                "mojolearn ExponentialSmoothing: data input must have 1 or 2 "
                f"dimensions, got {a.ndim}"
            )
        flat = np.ascontiguousarray(a.reshape(d2, n), dtype=np.float32)
        return flat, n

    def fit(self):
        """Fit level, trend, season and SSE. Returns self.

        Every validation cuML does in `holtwinters.pyx` runs on the Mojo
        side, by name, in their words: `ts_num >= 1`, `seasonal_periods
        >= 2`, `start_periods >= 2`, `seasonal_periods >= start_periods`,
        `eps > 0`, `n >= 1`, `n >= start_periods * seasonal_periods`. They
        are not restated here, so there is one place they can drift from.
        """
        data, n = self._check_dims(self.endog)
        self._data = data  # kept alive across the call (_arrays.py)
        components_len = (n - self.seasonal_periods) * self.ts_num
        if components_len <= 0:
            # This layer has to size the output buffers BEFORE the Mojo
            # side validates, so it needs its own guard against a negative
            # allocation. It is not a second copy of a rule:
            # `holtwinters_validate_params` refuses `n < start_periods *
            # frequency`, which is strictly stronger, and that refusal is
            # the one a caller normally sees.
            raise ValueError(
                "mojolearn ExponentialSmoothing: n "
                f"({n}) must exceed seasonal_periods ({self.seasonal_periods})"
            )
        comps = np.empty(3 * components_len, dtype=np.float32)
        stats = np.empty(4 * self.ts_num, dtype=np.float32)
        flags = np.empty(2 * self.ts_num, dtype=np.int32)
        _mojolearn_tsa.holtwinters_fit(
            _addr_ro(data),
            _addr(comps),
            _addr(stats),
            _addr(flags),
            # ORDER MATCHES bindings/_mojolearn_tsa.mojo::holtwinters_fit_binding.
            #   0 n, 1 batch_size, 2 frequency, 3 start_periods, 4 eps
            [
                int(n),
                int(self.ts_num),
                int(self.seasonal_periods),
                int(self.start_periods),
                float(self.eps),
            ],
            self.seasonal,
        )
        self.n = n
        self._components_len = components_len
        self._comps = comps  # kept for forecast(), which re-uploads them
        # comps LAYOUT -- the same words as in bindings/_mojolearn_tsa.mojo
        # and holtwinters/estimator.mojo:
        #   [0 * components_len, 1 * components_len)   level
        #   [1 * components_len, 2 * components_len)   trend
        #   [2 * components_len, 3 * components_len)   season
        # Each block is TIME-MAJOR: series s at step i is [s + i * ts_num].
        # `.reshape((ts_num, num_rows), order="F")` is cuML's own line
        # (holtwinters.pyx:341-344) and undoes exactly that.
        num_rows = components_len // self.ts_num
        cl = components_len
        self.level_ = comps[0:cl].reshape((self.ts_num, num_rows), order="F")
        self.trend_ = comps[cl : 2 * cl].reshape((self.ts_num, num_rows), order="F")
        self.season_ = comps[2 * cl : 3 * cl].reshape(
            (self.ts_num, num_rows), order="F"
        )
        # stats LAYOUT -- the same words as in the other two files:
        #   [0 * ts_num, 1 * ts_num)   sse
        #   [1 * ts_num, 2 * ts_num)   alpha
        #   [2 * ts_num, 3 * ts_num)   beta
        #   [3 * ts_num, 4 * ts_num)   gamma
        b = self.ts_num
        self.sse_ = stats[0:b]
        self.alpha_ = stats[b : 2 * b]
        self.beta_ = stats[2 * b : 3 * b]
        self.gamma_ = stats[3 * b : 4 * b]
        # flags LAYOUT -- the same words as in the other two files:
        #   [0 * ts_num, 1 * ts_num)   niter
        #   [1 * ts_num, 2 * ts_num)   criterion
        self.n_iter_ = flags[0:b]
        self.criterion_ = flags[b : 2 * b]
        self.fit_executed_flag = True
        return self

    def forecast(self, h=1, index=None):
        """Forecast `h` points per series.

        Return shapes are cuML's, including its single-series special case
        (`holtwinters.pyx:417-426`): with `index=None` and `ts_num > 1` a
        `(h, ts_num)` array; with `index=None` and `ts_num == 1` a 1-D
        array of length `h`; with an integer `index` a 1-D array of length
        `h` for that series.
        """
        if not self.fit_executed_flag:
            raise ValueError(
                "mojolearn ExponentialSmoothing: fit() the model before "
                "forecast()"
            )
        if not isinstance(h, int) or isinstance(h, bool):
            raise TypeError(
                f"mojolearn ExponentialSmoothing: h must be int, got {type(h)}"
            )
        if index is not None and (not isinstance(index, int) or isinstance(index, bool)):
            raise TypeError(
                "mojolearn ExponentialSmoothing: index must be int or None, "
                f"got {type(index)}"
            )
        if h <= 0:
            raise ValueError(
                f"mojolearn ExponentialSmoothing: h must be > 0. Currently: {h}"
            )
        if index is not None and (index < 0 or index >= self.ts_num):
            raise IndexError(
                f"mojolearn ExponentialSmoothing: index input: {index} outside "
                f"of range [0, {self.ts_num})"
            )
        out = np.empty(h * self.ts_num, dtype=np.float32)
        _mojolearn_tsa.holtwinters_forecast(
            _addr_ro(self._comps),
            _addr(out),
            # ORDER MATCHES bindings/_mojolearn_tsa.mojo::holtwinters_forecast_binding.
            #   0 n, 1 batch_size, 2 frequency, 3 h
            [
                int(self.n),
                int(self.ts_num),
                int(self.seasonal_periods),
                int(h),
            ],
            self.seasonal,
        )
        # out is TIME-MAJOR: series s at step i is [s + i * ts_num], which
        # is cuML's `(ts_num, h)` array with order="F".
        points = out.reshape((self.ts_num, h), order="F")
        if index is not None:
            return points[index]
        if self.ts_num == 1:
            return points.ravel(order="F")
        return points.T

    def score(self, index=None):
        """The SSE of the fitted model, which is what cuML's `score`
        returns (they note in `holtwinters.pyx` that it is the SSE rather
        than the log-likelihood gradient, rapidsai/cuml#876).

        `index=None` gives all `ts_num` of them; an integer `index` gives
        that series' SSE as a scalar. Those are cuML's two returns.
        """
        if not self.fit_executed_flag:
            raise ValueError(
                "mojolearn ExponentialSmoothing: fit() the model before score()"
            )
        if index is None:
            return self.sse_
        if index < 0 or index >= self.ts_num:
            raise IndexError(
                f"mojolearn ExponentialSmoothing: index input: {index} outside "
                f"of range [0, {self.ts_num})"
            )
        return self.sse_[index]

    def _component(self, comp, index, who):
        """cuML's return shapes for `get_level` / `get_trend` /
        `get_season` (`holtwinters.pyx:479-490`): `index=None` gives the
        transposed `(num_rows, ts_num)` block, or a 1-D array when
        `ts_num == 1`; an integer `index` gives that series."""
        if not self.fit_executed_flag:
            raise ValueError(
                f"mojolearn ExponentialSmoothing: fit() the model to get "
                f"{who} values"
            )
        if index is None:
            if self.ts_num == 1:
                return comp.ravel(order="F")
            return comp.T
        if index < 0 or index >= self.ts_num:
            raise IndexError(
                f"mojolearn ExponentialSmoothing: index input: {index} outside "
                f"of range [0, {self.ts_num})"
            )
        return comp[index]

    def get_level(self, index=None):
        return self._component(self.level_, index, "level")

    def get_trend(self, index=None):
        return self._component(self.trend_, index, "trend")

    def get_season(self, index=None):
        return self._component(self.season_, index, "season")
