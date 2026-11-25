# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE PUBLIC FUNCTIONS' LANES: shared helpers, and the feature-selection, graph,
# time-series, ANN, manifold, resample and host-only functions.
#
# Owned by sub-lane vdv-cov-func (verify through device=, items B and C,
# 2026-11-24). tools/identity_break.py executes this file in ITS OWN namespace
# after every helper and registry exists (`_load_lane_fragments`). Fragments load
# in id order, so this file ("functions") loads before functions_linalg,
# functions_metrics, functions_msel and functions_training, which call the
# `_functions_*` helpers defined here at run time.
#
# THE CONTRACT (docs/plans/VERIFY_THROUGH_DEVICE.md, vdv-_common.md):
# - A lane is device-agnostic: it calls the public function through `ml` with no
#   `device=`; the harness's device-bound view passes device="cpu" or
#   device="gpu". `_lane_side()` names the running side; a lane reads it only to
#   decide whether the device-resident run is owed (the GPU side), never to choose
#   what it computes.
# - Every function lane goes through `_functions_resident`: the host-input call
#   is the digest. On the GPU side the same call runs again with its inputs made
#   device-resident by the product's convention for the function's family, and
#   every part must be byte-for-byte the host-input digest of the same side (a
#   difference raises NumericalMismatch: the cell fails). The arena families
#   (metrics, feature selection, resample, model selection, time series, graph)
#   must also show they READ the resident copy: a `resident()` scope with no
#   device-store hit after the second call is a cell failure, in words.
# - Each lane is an independent unit: its own bounded fixture slice, no shared
#   mutable state; the only shared pieces are these pure helpers.

#: The batch-invariance property does not apply to a function lane: every one
#: of them reduces, splits or factors the whole input it is given.
_FUNCTIONS_BATCH_NA = ("n/a:function (a public function over the whole input it is given; a row subset "
                       "is another call, so there is no row-wise batch proposition)")


def _functions_host_value(v):
    """`v` as host data: a TrainDeviceArray through `to_host()`, a
    SequenceDeviceTensor through `numpy()` (its native download), else as is."""
    name = type(v).__name__
    if name == "TrainDeviceArray":
        return v.to_host()
    if name == "SequenceDeviceTensor":
        return v.numpy()
    return v


def _functions_arrays(v, out):
    """Flatten a function's answer into numeric arrays, in a fixed order:
    dicts by sorted key (the key's UTF-8 bytes first), sequences in order,
    Python scalars widened (bool -> uint8, int -> int64, float -> float64),
    device results downloaded once."""
    v = _functions_host_value(v)
    if v is None:
        out.append(np.zeros(0, dtype=np.uint8))
    elif isinstance(v, dict):
        for k in sorted(v, key=str):  # glue: a result dict's keys, in sorted order
            out.append(np.frombuffer(str(k).encode("utf-8"), dtype=np.uint8))
            _functions_arrays(v[k], out)
    elif isinstance(v, (tuple, list)):
        for item in v:  # glue: a result tuple's members, in order
            _functions_arrays(item, out)
    elif isinstance(v, (bool, np.bool_)):
        out.append(np.asarray([int(v)], dtype=np.uint8))
    elif isinstance(v, (int, np.integer)):
        out.append(np.asarray([int(v)], dtype=np.int64))
    elif isinstance(v, (float, np.floating)):
        out.append(np.asarray([v], dtype=np.float64))
    else:
        a = np.asarray(v)
        if a.dtype == object:
            _functions_arrays(list(a), out)
        else:
            out.append(np.ascontiguousarray(a))


def _functions_d(*values):
    """The digest of a function's answer (any of: scalar, array, device array,
    tuple, dict), through the harness's `_h`."""
    out = []
    _functions_arrays(values, out)
    return _h(*out)


def _functions_as_given(a):
    return a


def _functions_train_put(a):
    """The training family's device form: a float32 host array as a
    TrainDeviceArray, uploaded once (`train_to_device`); other inputs (ids,
    targets, scalars, lists) as given. `train_to_device` uploads float32
    only, so an integer operand a function needs on the device (lu_solve's
    int32 pivots) comes from the function that makes it there (lu_factor of
    the uploaded matrix): the lane builds it (functions_linalg.py fn-lu-solve)."""
    from mojolearn._training_impl import train_to_device
    if isinstance(a, np.ndarray) and a.dtype == np.float32:
        return train_to_device(np.ascontiguousarray(a), numeric_mode="identical")
    return a


def _functions_seq_put(a):
    """The sequence family's device form: a float32 host array as a
    SequenceDeviceTensor (`to_device`); other inputs as given."""
    from mojolearn._x_sequence_device import to_device
    if isinstance(a, np.ndarray) and a.dtype == np.float32:
        return to_device(np.ascontiguousarray(a), numeric_mode="identical")
    return a


#: form -> how inputs become device-resident. "arena": the `resident()` scope of
#: python/mojolearn/_arena_io.py (identity-keyed store copies; the run is made
#: twice on the same buffers and the second must read them in place); "train":
#: TrainDeviceArray; "seq": SequenceDeviceTensor; "scalar": the function takes no
#: array input (a count, a box, a name), so there is nothing to make resident.
_FUNCTIONS_FORMS = ("arena", "train", "seq", "scalar")


def _functions_compare(name, form, host, got, which):
    moved = sorted(k for k in set(host) | set(got) if host.get(k) != got.get(k))
    if moved:
        raise NumericalMismatch(
            f"{name}: the GPU side with device-resident inputs ({form} form, {which}) differs from the "
            f"host-input call of the same side in parts {moved}", host)


def _functions_resident(ml, name, run, form="arena"):
    """THE FUNCTION LANES' ONE HELPER. `run(put)` calls the function(s) with
    every input passed through `put` and returns {part: digest}. The host-input
    call is the lane's answer on both sides. On the GPU side (`_lane_side()`)
    the same call is made with device-resident inputs (`form`, see
    _FUNCTIONS_FORMS) and must give the same digests part for part; an arena
    form must also show a device-store hit. A failure raises, naming the
    function and the parts that moved: the cell fails, it is never "n/a"."""
    if form not in _FUNCTIONS_FORMS:
        raise ValueError(f"_functions_resident: unknown form {form!r}")
    host = run(_functions_as_given)
    if form == "scalar" or _lane_side() != "gpu":
        return _fit(host)
    if form == "arena":
        from mojolearn._arena_io import resident
        with resident(min_words=0):
            first = run(_functions_as_given)      # uploads each input once into the store
            second = run(_functions_as_given)     # the same buffers: read from the store in place
            stats = resident.stats() or {}
        _functions_compare(name, form, host, first, "first call in the scope")
        _functions_compare(name, form, host, second, "second call in the scope")
        hits = sum(int(h) for _, h in stats.values())  # glue: the scope's per-binding counters
        if hits == 0:
            raise RuntimeError(f"{name}: the GPU side did not read a device-resident input in place "
                               f"(a resident() scope over two calls on the same buffers made no store hit; "
                               f"stats {stats})")
    else:
        put = _functions_train_put if form == "train" else _functions_seq_put
        _functions_compare(name, form, host, run(put), "inputs uploaded once")
    return _fit(host)


def _functions_pos(X, n, col, seed):
    """n strictly positive float32 values in (0.5, 2.5): the fixture column's
    tanh (bounded on every fixture, the wide and denormal ones included) plus a
    hashed-stream term, so no fixture makes them constant."""
    t = np.tanh(np.asarray(X[:n, col], dtype=np.float64)).astype(np.float32)
    return np.ascontiguousarray(np.float32(1.0) + np.float32(0.5) * t + _hw((n,), seed, 0.0, 1.0))


def _functions_cont(X, n, col, seed):
    """n float32 values in (-1.5, 1.5) from the fixture column (tanh) and the
    hashed stream."""
    t = np.tanh(np.asarray(X[:n, col], dtype=np.float64)).astype(np.float32)
    return np.ascontiguousarray(t + _hw((n,), seed, -0.5, 0.5))


def _functions_mat(X, n, d, seed, col0=0):
    """An (n, d) float32 matrix: tanh of fixture columns col0.. (cycled) plus
    a hashed-stream term."""
    cols = [(col0 + j) % X.shape[1] for j in range(d)]  # glue: column indices
    t = np.tanh(np.asarray(X[:n][:, cols], dtype=np.float64)).astype(np.float32)
    return np.ascontiguousarray(t + _hw((n, d), seed, -0.25, 0.25))


def _functions_classes(X, n, k, seed):
    """n int32 labels in 0..k-1, every class present on every fixture: a
    fixture-column bin cycled with the row index."""
    t = np.tanh(np.asarray(X[:n, 3], dtype=np.float64))
    b = np.floor((t + 1.0) * 0.5 * k).astype(np.int64)
    return np.ascontiguousarray(((b + np.arange(n) + (_hw((n,), seed, 0.0, 2.0) > 1.0)) % k).astype(np.int32))


# ---------------------------------------------------------------- feature selection (x_prep)

@lane("fn-chi2")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.chi2 on 1500 non-negative rows of 6 features, 3 classes."""
    Xp = np.ascontiguousarray(np.abs(_functions_mat(X, 1500, 6, "fn-chi2:x")))
    y = _functions_classes(X, 1500, 3, "fn-chi2:y")
    return _functions_resident(ml, "chi2", lambda p: dict(chi2=_functions_d(ml.chi2(p(Xp), p(y)))))


@lane("fn-f-classif")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.f_classif on 1500 rows of 6 features, 3 classes."""
    Xf = _functions_mat(X, 1500, 6, "fn-f-classif:x")
    y = _functions_classes(X, 1500, 3, "fn-f-classif:y")
    return _functions_resident(ml, "f_classif", lambda p: dict(f=_functions_d(ml.f_classif(p(Xf), p(y)))))


@lane("fn-f-regression")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.f_regression, centered and not, force_finite both ways."""
    Xf = _functions_mat(X, 1500, 6, "fn-f-regression:x")
    y = _functions_cont(X, 1500, 4, "fn-f-regression:y")
    return _functions_resident(ml, "f_regression", lambda p: dict(
        centered=_functions_d(ml.f_regression(p(Xf), p(y))),
        raw=_functions_d(ml.f_regression(p(Xf), p(y), center=False, force_finite=False))))


@lane("fn-r-regression")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.r_regression, centered and not."""
    Xf = _functions_mat(X, 1500, 6, "fn-r-regression:x")
    y = _functions_cont(X, 1500, 4, "fn-r-regression:y")
    return _functions_resident(ml, "r_regression", lambda p: dict(
        centered=_functions_d(ml.r_regression(p(Xf), p(y))),
        raw=_functions_d(ml.r_regression(p(Xf), p(y), center=False))))


@lane("fn-mutual-info-classif")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.mutual_info_classif, 800 rows of 4 continuous features, seeded."""
    Xf = _functions_mat(X, 800, 4, "fn-mutual-info-classif:x")
    y = _functions_classes(X, 800, 3, "fn-mutual-info-classif:y")
    return _functions_resident(ml, "mutual_info_classif", lambda p: dict(
        mi=_functions_d(ml.mutual_info_classif(p(Xf), p(y), random_state=4))))


@lane("fn-mutual-info-regression")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.mutual_info_regression, 800 rows of 4 features, 5 neighbors, seeded."""
    Xf = _functions_mat(X, 800, 4, "fn-mutual-info-regression:x")
    y = _functions_cont(X, 800, 5, "fn-mutual-info-regression:y")
    return _functions_resident(ml, "mutual_info_regression", lambda p: dict(
        mi=_functions_d(ml.mutual_info_regression(p(Xf), p(y), n_neighbors=5, random_state=4))))


# ---------------------------------------------------------------- graph, time series, ANN, manifold

def _functions_graph(X, n, directed):
    """A float32 adjacency from exact comparisons of fixture values: rows whose
    column-3 tanh falls in the same eighth are joined, weight 1 + (i*j) % 3;
    a directed graph keeps the upper triangle."""
    q = np.floor((np.tanh(np.asarray(X[:n, 3], dtype=np.float64)) + 1.0) * 4.0).astype(np.int64)
    i = np.arange(n)
    A = ((q[:, None] == q[None, :]) & (i[:, None] != i[None, :])).astype(np.float32)
    A *= (1 + (i[:, None] * i[None, :]) % 3).astype(np.float32)
    return np.ascontiguousarray(np.triu(A) if directed else A, dtype=np.float32)


@lane("fn-connected-components")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.connected_components on a 128-node undirected graph and a
    directed one (weak), and connection='strong' on the undirected graph (the
    documented 'weak' equivalence). Strong components of a directed graph are
    refused by the product (NotImplementedError, documented), so the lane does
    not ask for them (lane w3-fix-9-s4: the clean run was REFUSED)."""
    U = _functions_graph(X, 128, False)
    D = _functions_graph(X, 128, True)
    return _functions_resident(ml, "connected_components", lambda p: dict(
        undirected=_functions_d(ml.connected_components(p(U), directed=False)),
        weak=_functions_d(ml.connected_components(p(D), directed=True, connection="weak")),
        strong=_functions_d(ml.connected_components(p(U), directed=False, connection="strong"))))


@lane("fn-kpss-test")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.kpss_test on four 256-observation series, plain and seasonally
    differenced with the statistic."""
    S = _functions_mat(X, 256, 4, "fn-kpss-test:s")
    return _functions_resident(ml, "kpss_test", lambda p: dict(
        flat=_functions_d(ml.kpss_test(p(S), d=0)),
        diffed=_functions_d(ml.kpss_test(p(S), d=1, D=1, s=12, return_statistic=True))))


@lane("fn-select-d")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.select_d on a stationary, an integrated and a twice-integrated
    series (256 observations), with and without seasonal D."""
    S = np.array(_functions_mat(X, 256, 3, "fn-select-d:s"), dtype=np.float32, order="C", copy=True)
    S[:, 1] = np.cumsum(S[:, 1], dtype=np.float32)
    S[:, 2] = np.cumsum(np.cumsum(S[:, 2], dtype=np.float32), dtype=np.float32)
    S = np.ascontiguousarray(S)
    return _functions_resident(ml, "select_d", lambda p: dict(
        d0=_functions_d(ml.select_d(p(S), D=0, s=0, d_max=2)),
        d1=_functions_d(ml.select_d(p(S), D=1, s=12, d_max=1))))


@lane("fn-refine")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.refine: 32 queries re-ranked exactly over 16 candidates each
    from a 1024-row dataset, k = 4, padding and repeated ids included."""
    data = _functions_mat(X, 1024, 8, "fn-refine:data")
    queries = _functions_mat(X[1024:], 32, 8, "fn-refine:q")
    c = ((np.arange(32 * 16, dtype=np.int64).reshape(32, 16) * 7919 + 13) % 1024).astype(np.int32)
    c[:, 5] = -1
    c[:, 7] = c[:, 6]
    c = np.ascontiguousarray(c)
    return _functions_resident(ml, "refine", lambda p: dict(
        sq=_functions_d(ml.refine(p(data), p(queries), p(c), 4))))


@lane("fn-spectral-embedding")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.manifold.spectral_embedding on 256 rows of 4 features, the
    normalized and the unnormalized Laplacian."""
    P = _functions_mat(X, 256, 4, "fn-spectral-embedding:p")
    return _functions_resident(ml, "manifold.spectral_embedding", lambda p: dict(
        norm=_functions_d(ml.manifold.spectral_embedding(p(P), n_components=3, n_neighbors=10, random_state=3)),
        raw=_functions_d(ml.manifold.spectral_embedding(p(P), n_components=3, n_neighbors=10, random_state=3,
                                                        norm_laplacian=False, drop_first=False))))


# ---------------------------------------------------------------- resample

@lane("fn-resample-bootstrap")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.resample.bootstrap of 256 values, 999 resamples: the mean and
    the median (statistic "quantile", q_or_prop 0.5) with the percentile
    interval, and the standard deviation with BCa (a statistic with a
    jackknife arm)."""
    # HARNESS FIX (w3-fix-9-s3, 2026-11-25): the median part asked for
    # statistic="median", which is not a name in resample.STATISTICS; the
    # boundary refuses it ("mojolearn bootstrap: statistic must be one of
    # ..."), so the clean lane was REFUSED in every run. The median is the 0.5
    # quantile; BCa moves to "std" (quantile has no jackknife arm).
    v = _functions_cont(X, 256, 3, "fn-resample-bootstrap:v")

    def answer(r):
        return (r.point_estimate, r.distribution, r.standard_error,
                r.confidence_interval.low, r.confidence_interval.high)

    return _functions_resident(ml, "resample.bootstrap", lambda p: dict(
        mean=_functions_d(answer(ml.resample.bootstrap(p(v), statistic="mean", n_resamples=999, random_state=5))),
        median=_functions_d(answer(ml.resample.bootstrap(p(v), statistic="quantile", q_or_prop=0.5,
                                                         n_resamples=999, random_state=5))),
        std_bca=_functions_d(answer(ml.resample.bootstrap(p(v), statistic="std", n_resamples=999,
                                                          method="BCa", random_state=5)))))


@lane("fn-resample-permutation-test")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.resample.permutation_test: independent difference of means,
    and the paired sign flip, 999 resamples."""
    a = _functions_cont(X, 128, 3, "fn-resample-permutation-test:a")
    b = _functions_cont(X, 128, 4, "fn-resample-permutation-test:b")
    diff = np.ascontiguousarray(a - b)

    def answer(r):
        return (r.statistic, r.null_distribution, r.pvalue)

    return _functions_resident(ml, "resample.permutation_test", lambda p: dict(
        independent=_functions_d(answer(ml.resample.permutation_test(p(a), p(b), statistic="diff_means",
                                                                     n_resamples=999, random_state=6))),
        samples=_functions_d(answer(ml.resample.permutation_test(p(diff), statistic="mean",
                                                                 permutation_type="samples",
                                                                 n_resamples=999, random_state=6)))))


@lane("fn-resample-monte-carlo")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.resample.monte_carlo_integrate, the three compiled integrands
    over a box from the fixture's tanh range, 16384 draws. No array input, so
    no device-resident run (form "scalar")."""
    t = np.tanh(np.asarray(X[:256, :2], dtype=np.float64))
    lo = [float(t[:, 0].min()) - 0.5, float(t[:, 1].min()) - 0.5]
    hi = [float(t[:, 0].max()) + 0.5, float(t[:, 1].max()) + 0.5]

    def answer(r):
        return (r.integral, r.mean, r.volume, r.closed_form)

    return _functions_resident(ml, "resample.monte_carlo_integrate", lambda p: {
        f: _functions_d(answer(ml.resample.monte_carlo_integrate(f, lo, hi, 16384, random_state=1)))
        for f in ("const", "sum", "product")}, form="scalar")


@lane("fn-resample-resample")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.resample.resample of two aligned arrays, with and without
    replacement, and a full permutation of one array (the default n_samples)."""
    # HARNESS FIX (w3-fix-9-s3, 2026-11-25): the third part passed stratify=,
    # which the product refuses by name (resample/NOT_IMPLEMENTED.tsv;
    # providers/python/resample_rows_operation_pyapi_pack.mojo:28), so the
    # clean lane was REFUSED in every run. It is now the default-size
    # permutation of one array.
    v = _functions_cont(X, 256, 3, "fn-resample-resample:v")
    M = _functions_mat(X, 256, 3, "fn-resample-resample:m")
    return _functions_resident(ml, "resample.resample", lambda p: dict(
        replace=_functions_d(ml.resample.resample(p(v), p(M), n_samples=200, random_state=8)),
        noreplace=_functions_d(ml.resample.resample(p(v), p(M), replace=False, n_samples=128, random_state=8)),
        permutation=_functions_d(ml.resample.resample(p(v), replace=False, random_state=8))))


@lane("fn-resample-indices")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.resample.resample_indices, with and without replacement. No
    array input (a count): form "scalar"."""
    return _functions_resident(ml, "resample.resample_indices", lambda p: dict(
        replace=_functions_d(ml.resample.resample_indices(512, 300, random_state=8)),
        noreplace=_functions_d(ml.resample.resample_indices(512, 300, replace=False, random_state=8))),
        form="scalar")


_batch_decl(_FUNCTIONS_BATCH_NA, "fn-chi2", "fn-f-classif", "fn-f-regression", "fn-r-regression",
            "fn-mutual-info-classif", "fn-mutual-info-regression", "fn-connected-components", "fn-kpss-test",
            "fn-select-d", "fn-refine", "fn-spectral-embedding", "fn-resample-bootstrap",
            "fn-resample-permutation-test", "fn-resample-monte-carlo", "fn-resample-resample",
            "fn-resample-indices")
