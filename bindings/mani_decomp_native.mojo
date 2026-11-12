# SPDX-License-Identifier: Apache-2.0
# The decomp lane's manifold estimators, one native call per public method
# (python/mojolearn/_expansion_decomp.py): Isomap (`fit`, `fit_transform`,
# `transform`, `reconstruction_error`), ClassicalMDS (`fit`, `fit_transform`),
# MDS (`fit`, `fit_transform`) and LocallyLinearEmbedding (`fit`,
# `fit_transform`, `transform`).
#
# Door shape (lane entangled-mkit, 2026-11-12): UNPACK reads the settings into
# typed plans (the Python methods' refusals, same wording), admits the decomp
# binding (and the process-default helper binding) into the typed kit
# (bindings/mani_kit.mojo `mk_image`) and the fitted `_ManifoldMatrix` owners
# (`mk_fitted`); RUN (bindings/mani_decomp_run.mojo on bindings/mani_kit_run.mojo)
# does the fit or transform with the GIL released, holding no Python object;
# PACK builds the public Arrays and publishes every fitted field once at the
# end, then returns the operation's pool ids. X is admitted as the first RUN
# step, so the refusals keep their order: the settings checks the Python
# methods made before reading X, X's own refusals, then the rest.
# Runtime/vendor evidence is pending.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_setitem, py_contains, py_type
from bindings.classical_attributes import classical_connect_numpy
from bindings.admitted_family import admit_family_meta
from providers.python.estimator_binding_pyapi import estimator_binding_mode
from bindings.manifold_settings_native import ManifoldSettings
from bindings.manifold_scalar_native import manifold_number
from providers.python.array_metadata_pyapi import ArrayMetadata
from bindings.metrics_native_support import metric_warn
from bindings.family_common import family_repr, family_str, family_is_str, family_is_int, family_type_name, family_field
from bindings.mani_kit import mk_image, mk_fitted, mk_keep, mk_owner, mk_out, mk_out2, mk_finish, mk_seed_word
from bindings.mani_kit_run import MRun
from bindings.mani_decomp_run import (
    MANI_METRIC_PRECOMPUTED, MANI_METRIC_EUCLIDEAN, LLE_STANDARD, LLE_LTSA, LLE_HESSIAN, LLE_MODIFIED,
    MDS_INIT_ARRAY, MDS_INIT_CLASSICAL, MDS_INIT_RANDOM, MdsPlan, IsoPlan, IsoFit, IsoState, LlePlan, LleFit,
    cmds_core, mds_dissimilarity_run, mds_fit_run, iso_fit_run, iso_transform_run, iso_reconstruction_error_run,
    lle_fit_run, lle_transform_run,
)

comptime _LLE_ITER_MIN_N = 200
comptime _LLE_ITER_MAX_K = 10


def _mani_eq(value: PythonObject, text: String, context: PythonObject) raises -> Bool:
    """`value == text`, as a truth value."""
    # Remaining migration: retain the original rich-comparison behavior until
    # native record/temporal and version-specific array errors are covered.
    var b = native_loaded_module("builtins")
    # An exact str compares natively (the settings are exact strings; two
    # module lookups and two Python calls per comparison before; p3 rank 702).
    if py_type(value) is b.str:
        return String(py=value) == text
    return Bool(py=py_call(b.bool, py_call(native_loaded_module("operator").eq, value, PythonObject(text))))


def _mani_in(value: PythonObject, choices: PythonObject, context: PythonObject) raises -> Bool:
    """`value in choices`."""
    var b = native_loaded_module("builtins")
    if py_type(value) is b.str and py_type(choices) is b.tuple:
        # An exact str among a tuple of exact strs: native comparisons; any
        # other member type takes Python's containment test.
        var text = String(py=value)
        var plain = True
        for i in range(len(choices)):  # small-loop(the fixed choices of one setting)
            if py_type(choices[i]) is not b.str:
                plain = False
                break
            if String(py=choices[i]) == text:
                return True
        if plain:
            return False
    return Bool(py=py_call(native_loaded_module("operator").contains, choices, value))


def _mani_truth(value: PythonObject, context: PythonObject) raises -> Bool:
    # Structured subarray nonzero behavior remains with the original slot;
    # this numeric-conversion increment does not claim truth migration.
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is b.bool or cls is b.int:
        return Bool(py=value)
    if value is None:
        return False
    return Bool(py=py_call(b.bool, value))


def _mani_int(value: PythonObject, context: PythonObject) raises -> Int:
    """`int(value)`."""
    return Int(py=manifold_number(value, True, context))


def _mani_float(value: PythonObject, context: PythonObject) raises -> Float64:
    """`float(value)`."""
    return Float64(py=manifold_number(value, False, context))


def _mani_f32(v: Float64) -> Float64:
    """`_f32(v)`: rounded once to float32."""
    return Float64(Float32(v))


def _mani_kit(mut run: MRun, mode: PythonObject, binding: PythonObject, context: PythonObject) raises -> Int:
    """UNPACK `_Kit(mode)`: `binding` is the decomp binding the shell resolved
    for `mode`."""
    return mk_image(run, String(py=py_call(native_loaded_module("builtins").str, mode)), binding, context)


def _mani_x(mut run: MRun, k: Int, x: ArrayMetadata) raises -> Int:
    """RUN step 1: X admitted into the kit (`k.from_snapshot(x, "X")`), the
    GIL released; the caller keeps `x` alive."""
    var m = -1
    var address = x.source.addr
    var rows = x.source.rows
    var cols = x.source.cols
    var ndim = len(x.shape)
    with GILReleased(Python()):
        m = run.from_snapshot(k, address, rows, cols, ndim, "X")
    return m


def _mani_check(model: PythonObject, fields: PythonObject, attr: String) raises:
    """`_check(attr)`."""
    if not py_contains(fields, PythonObject(attr)):
        raise Error("RuntimeError: " + family_type_name(model) + " is not fitted; call fit first")


def _mani_seed(random_state: PythonObject) raises -> PythonObject:
    """`_seed_of(random_state)`."""
    if random_state is None:
        return PythonObject(0)
    if not family_is_int(random_state):
        raise Error("TypeError: random_state must be None or an int (the counter-based RNG takes an integer seed)")
    return random_state


def _mani_eigen_solver(name: PythonObject, who: String, context: PythonObject) raises:
    """`_exact_eigen_solver(name, who)`."""
    if _mani_eq(name, "arpack", context):
        raise Error("NotImplementedError: " + who + ": eigen_solver='arpack' is not implemented; the exact solve "
                    + "runs for 'auto' and 'dense', and running it under ARPACK's name "
                    + "would be a silent substitution (x_decomp/NOT_IMPLEMENTED.tsv)")
    if not _mani_in(name, Python.tuple(PythonObject("auto"), PythonObject("dense")), context):
        raise Error(who + ": eigen_solver must be 'auto', 'dense' or 'arpack', got " + family_repr(name))


def _mani_metric_spec(metric: PythonObject, p_in: PythonObject, metric_params: PythonObject,
                      who: String, context: PythonObject) raises -> Tuple[Int, Float64]:
    """`_metric_spec(metric, p, metric_params, who)`: (kind, p)."""
    var b = native_loaded_module("builtins")
    var p = p_in
    if _mani_truth(metric_params, context):
        var extra = py_call(native_loaded_module("operator").sub, py_call(b.set, metric_params),
                            py_call(b.set, Python.list(PythonObject("p"))))
        if _mani_truth(extra, context):
            raise Error("NotImplementedError: " + who + ": metric_params " + family_repr(py_call(b.sorted, extra))
                        + " are not carried")
        p = py_call(metric_params.get, PythonObject("p"), p)
    if _mani_in(metric, Python.tuple(PythonObject("euclidean"), PythonObject("l2")), context):
        return (Int(0), Float64(2.0))
    if _mani_eq(metric, "minkowski", context):
        var pw = _mani_float(p, context)
        if not pw > 0:
            raise Error(who + ": minkowski p must be > 0")
        if pw == 2.0:
            return (Int(0), Float64(2.0))
        if pw == 1.0:
            return (Int(1), Float64(1.0))
        if pw > 0 and not (pw - pw == 0.0):
            return (Int(2), Float64(0.0))
        return (Int(3), pw)
    var kinds = context["family_mani_pd_kind"]
    if family_is_str(metric) and py_contains(kinds, metric):
        return (Int(py=kinds[metric]), Float64(0.0))
    raise Error("NotImplementedError: " + who + ": metric=" + family_repr(metric) + " is not carried (euclidean, "
                + "minkowski p, manhattan, chebyshev, cosine)")


def _mani_metric(metric: PythonObject, metric_params: PythonObject, who: String,
                 context: PythonObject) raises -> Tuple[Int, Float64]:
    """ClassicalMDS / MDS: 'precomputed', 'euclidean' (the two named routes),
    else `_metric_spec(metric, 2, metric_params, who)`."""
    if _mani_eq(metric, "precomputed", context):
        return (MANI_METRIC_PRECOMPUTED, Float64(0.0))
    if _mani_eq(metric, "euclidean", context):
        return (MANI_METRIC_EUCLIDEAN, Float64(2.0))
    return _mani_metric_spec(metric, PythonObject(2), metric_params, who, context)


# ---------------------------------------------------------------- ClassicalMDS
def _cmds_fit(model: PythonObject, settings: ManifoldSettings, args: PythonObject, x: ArrayMetadata, context: PythonObject) raises -> PythonObject:
    """`ClassicalMDS.fit(X)`; returns embedding_."""
    classical_connect_numpy(context)
    var fields = settings.values
    var mode = settings.mode
    var run = MRun()
    try:
        var k = _mani_kit(run, mode, args[0], context)
        var m = _mani_x(run, k, x)
        var spec = _mani_metric(family_field(fields, "metric"), family_field(fields, "metric_params"), "ClassicalMDS",
                                context)
        var nc = _mani_int(family_field(fields, "n_components"), context)
        var got = Tuple[Int, Int, Int](-1, -1, -1)
        with GILReleased(Python()):
            got = cmds_core(run, k, m, spec[0], spec[1], nc)
        _ = x  # the admitted X stays alive through the RUN above
        # PACK
        var dissimilarity = mk_out2(run, got[0], context)
        var values = mk_out(run, got[1], Python.tuple(run.c(got[1])), context)
        var embedding = mk_out2(run, got[2], context)
        var owner = mk_owner(run, got[2])
        py_setitem(settings.fields, "numeric_mode_", mode)
        py_setitem(settings.fields, "dissimilarity_matrix_", dissimilarity)
        py_setitem(settings.fields, "eigenvalues_", values)
        py_setitem(settings.fields, "embedding_m_", owner)
        py_setitem(settings.fields, "embedding_", embedding)
        py_setitem(settings.fields, "n_features_in_", PythonObject(run.c(m)))
        return embedding
    finally:
        mk_finish(run, context)


# ---------------------------------------------------------------- MDS
def _mds_fit(model: PythonObject, settings: ManifoldSettings, args: PythonObject, x: ArrayMetadata, context: PythonObject) raises -> PythonObject:
    """`MDS.fit_transform(X, init=init)`; returns embedding_. args = (decomp
    binding, default-mode binding, X, init)."""
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    var fields = settings.values
    var mode = settings.mode
    var run = MRun()
    try:
        var k = _mani_kit(run, mode, args[0], context)
        var m = _mani_x(run, k, x)
        var plan = MdsPlan()
        var spec = _mani_metric(family_field(fields, "metric"), family_field(fields, "metric_params"), "MDS", context)
        plan.metric = spec[0]
        plan.pw = spec[1]
        var dis = -1
        with GILReleased(Python()):
            dis = mds_dissimilarity_run(run, k, m, plan)
        plan.metric_mds = _mani_truth(family_field(fields, "metric_mds"), context)
        var normalized = family_field(fields, "normalized_stress")
        plan.norm = (normalized is PythonObject(True)) or (_mani_eq(normalized, "auto", context) and not plan.metric_mds)
        plan.nc = _mani_int(family_field(fields, "n_components"), context)
        var seed = _mani_seed(family_field(fields, "random_state"))
        var init_arg = args[3]
        var init = family_field(fields, "init")
        # the admitted `init` array stays alive (`init_meta`) through the RUN
        var init_meta = Optional[ArrayMetadata]()
        if not (init_arg is None):
            init_meta = Optional(admit_family_meta(init_arg, 2, "init", context, True))
            plan.init = MDS_INIT_ARRAY
            plan.init_address = init_meta.value().source.addr
            plan.init_rows = init_meta.value().source.rows
            plan.init_cols = init_meta.value().source.cols
            plan.init_ndim = len(init_meta.value().shape)
        elif _mani_eq(init, "classical_mds", context):
            plan.init = MDS_INIT_CLASSICAL
        elif _mani_eq(init, "random", context):
            plan.init = MDS_INIT_RANDOM
            plan.restarts = _mani_int(family_field(fields, "n_init"), context)
            plan.seed = mk_seed_word(seed)
        else:
            raise Error("init must be 'random', 'classical_mds' or an array")
        plan.max_iter = _mani_int(family_field(fields, "max_iter"), context)
        plan.eps = _mani_float(family_field(fields, "eps"), context)
        var got = Tuple[Int, Float64, Int](-1, 0.0, 0)
        with GILReleased(Python()):
            got = mds_fit_run(run, k, dis, plan)
        _ = init_meta^  # the admitted init stays alive through the RUN above
        _ = x  # the admitted X stays alive through the RUNs above
        # PACK
        var dissimilarity = mk_out2(run, dis, context)
        var embedding = mk_out2(run, got[0], context)
        py_setitem(settings.fields, "numeric_mode_", mode)
        py_setitem(settings.fields, "dissimilarity_matrix_", dissimilarity)
        py_setitem(settings.fields, "_norm", PythonObject(plan.norm))
        py_setitem(settings.fields, "stress_", py_call(b.float, PythonObject(got[1])))
        py_setitem(settings.fields, "n_iter_", PythonObject(got[2]))
        py_setitem(settings.fields, "embedding_", embedding)
        py_setitem(settings.fields, "n_features_in_", PythonObject(run.c(m)))
        return embedding
    finally:
        mk_finish(run, context)


# ---------------------------------------------------------------- Isomap
def _iso_fit(model: PythonObject, settings: ManifoldSettings, args: PythonObject, x: ArrayMetadata, context: PythonObject) raises -> PythonObject:
    """`Isomap.fit(X)`; returns embedding_."""
    classical_connect_numpy(context)
    var fields = settings.values
    var mode = settings.mode
    var neighbors = family_field(fields, "n_neighbors")
    var radius = family_field(fields, "radius")
    if not (neighbors is None) and not (radius is None):
        raise Error("Both n_neighbors and radius are provided. Use Isomap(radius=" + family_str(radius)
                    + ", n_neighbors=None) if intended to use radius-based neighbors")
    if neighbors is None and radius is None:
        raise Error("Isomap: one of n_neighbors and radius must be set")
    if not (radius is None) and not (_mani_float(radius, context) >= 0):
        raise Error("Isomap: radius must be >= 0")
    if not _mani_in(family_field(fields, "path_method"), Python.tuple(PythonObject("auto"), PythonObject("FW"),
                                                                       PythonObject("D")), context):
        raise Error("path_method must be 'auto', 'FW' or 'D'")
    var solver = family_field(fields, "eigen_solver")
    _mani_eigen_solver(solver, "Isomap", context)
    var spec = _mani_metric_spec(family_field(fields, "metric"), family_field(fields, "p"),
                                 family_field(fields, "metric_params"), "Isomap", context)
    var plan = IsoPlan()
    plan.kind = spec[0]
    plan.pw = spec[1]
    var run = MRun()
    try:
        var k = _mani_kit(run, mode, args[0], context)
        var m = _mani_x(run, k, x)
        var nn_out = PythonObject(None)
        if not (radius is None):
            plan.use_radius = True
            plan.radius = _mani_f32(_mani_float(radius, context))
        else:
            plan.nn = _mani_int(family_field(fields, "n_neighbors"), context)
            nn_out = PythonObject(plan.nn)
        plan.topk = _mani_in(solver, Python.tuple(PythonObject("auto"), PythonObject("arpack")), context)
        plan.nc = _mani_int(family_field(fields, "n_components"), context)
        var fit = IsoFit()
        with GILReleased(Python()):
            fit = iso_fit_run(run, k, m, plan)
        _ = x  # the admitted X stays alive through the RUNs above
        # PACK
        if fit.components > 1:
            metric_warn("The number of connected components of the neighbors graph is "
                        + String(fit.components) + " > 1. Completing the graph to fit Isomap might be slow.",
                        native_loaded_module("builtins").UserWarning, context)
        var dist = mk_out2(run, fit.d, context)
        var embedding = mk_out2(run, fit.em, context)
        # Fitted state is read-only from here (rows N291, N301): reads never
        # move it, each side's copy is made once (bindings/mani_kit_run.mojo).
        var dist_m = mk_owner(run, fit.d)
        var k_col = mk_owner(run, fit.col)
        var k_all = mk_owner(run, fit.allm)
        var values = mk_owner(run, fit.w)
        var vectors = mk_owner(run, fit.v)
        var em = mk_owner(run, fit.em)
        var fit_x = mk_owner(run, m)
        var kc = mk_owner(run, fit.kc)
        # transform's eigenvectors / sqrt(eigenvalues) (row N295)
        var iv = mk_owner(run, fit.iv)
        py_setitem(settings.fields, "numeric_mode_", mode)
        py_setitem(settings.fields, "_kind", PythonObject(plan.kind))
        py_setitem(settings.fields, "_pw", PythonObject(plan.pw))
        py_setitem(settings.fields, "n_connected_components_", PythonObject(fit.components))
        py_setitem(settings.fields, "dist_matrix_m_", dist_m)
        py_setitem(settings.fields, "dist_matrix_", dist)
        py_setitem(settings.fields, "_k_col", k_col)
        py_setitem(settings.fields, "_k_all", k_all)
        py_setitem(settings.fields, "eigenvalues_m_", values)
        py_setitem(settings.fields, "eigenvectors_m_", vectors)
        py_setitem(settings.fields, "embedding_m_", em)
        py_setitem(settings.fields, "embedding_", embedding)
        py_setitem(settings.fields, "_fit_X", fit_x)
        py_setitem(settings.fields, "_knn", nn_out)
        py_setitem(settings.fields, "n_features_in_", PythonObject(run.c(m)))
        py_setitem(settings.fields, "_Kc", kc)
        py_setitem(settings.fields, "_iso_v_m_", iv)
        return embedding
    finally:
        mk_finish(run, context)


def _iso_transform(model: PythonObject, settings: ManifoldSettings, args: PythonObject, x: ArrayMetadata, context: PythonObject) raises -> PythonObject:
    """`Isomap.transform(X)`."""
    _mani_check(model, settings.values, "embedding_m_")
    classical_connect_numpy(context)
    var fields = settings.values
    var run = MRun()
    try:
        var k = _mani_kit(run, settings.mode, args[0], context)
        var q = _mani_x(run, k, x)
        var st = IsoState()
        var fit_x = family_field(fields, "_fit_X")
        var d = family_field(fields, "dist_matrix_m_")
        var col = family_field(fields, "_k_col")
        var allm = family_field(fields, "_k_all")
        st.fit_x = mk_fitted(run, fit_x, "_fit_X")
        st.d = mk_fitted(run, d, "dist_matrix_m_")
        st.col = mk_fitted(run, col, "_k_col")
        st.allm = mk_fitted(run, allm, "_k_all")
        var knn = family_field(fields, "_knn")
        st.kind = _mani_int(family_field(fields, "_kind"), context)
        st.pw = _mani_float(family_field(fields, "_pw"), context)
        if knn is None:
            var radius = family_field(fields, "radius")
            st.radius = _mani_f32(_mani_float(radius, context))
            st.radius_text = family_str(radius)
        else:
            st.knn = _mani_int(knn, context)
        # eigenvectors / sqrt(eigenvalues), kept by fit; a model fitted before
        # it was kept makes it from eigenvectors_m_ / eigenvalues_m_
        var iv = py_get(fields, PythonObject("_iso_v_m_"))
        var w = PythonObject(None)
        var v = PythonObject(None)
        if iv is None:
            w = family_field(fields, "eigenvalues_m_")
            v = family_field(fields, "eigenvectors_m_")
            st.w = mk_fitted(run, w, "eigenvalues_m_")
            st.v = mk_fitted(run, v, "eigenvectors_m_")
        else:
            st.iv = mk_fitted(run, iv, "_iso_v_m_")
        var out = -1
        with GILReleased(Python()):
            out = iso_transform_run(run, k, q, st)
        _ = x  # the admitted X stays alive through the RUNs above
        # PACK: the result, and the copies the RUN made of the fitted matrices
        var result = mk_out2(run, out, context)
        mk_keep(run, st.fit_x, fit_x)
        mk_keep(run, st.d, d)
        mk_keep(run, st.col, col)
        mk_keep(run, st.allm, allm)
        if iv is None:
            mk_keep(run, st.w, w)
            mk_keep(run, st.v, v)
        else:
            mk_keep(run, st.iv, iv)
        return result
    finally:
        mk_finish(run, context)


def _iso_reconstruction_error(model: PythonObject, settings: ManifoldSettings, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Isomap.reconstruction_error()`."""
    _mani_check(model, settings.values, "embedding_m_")
    var fields = settings.values
    var run = MRun()
    try:
        var k = _mani_kit(run, settings.mode, args[0], context)
        var kc_owner = family_field(fields, "_Kc")
        var w_owner = family_field(fields, "eigenvalues_m_")
        var kc = mk_fitted(run, kc_owner, "_Kc")
        var w = mk_fitted(run, w_owner, "eigenvalues_m_")
        var error = 0.0
        with GILReleased(Python()):
            error = iso_reconstruction_error_run(run, k, kc, w)
        mk_keep(run, kc, kc_owner)
        mk_keep(run, w, w_owner)
        return PythonObject(error)
    finally:
        mk_finish(run, context)


# ---------------------------------------------------------------- LocallyLinearEmbedding
def _lle_fit(model: PythonObject, settings: ManifoldSettings, args: PythonObject, x: ArrayMetadata, context: PythonObject) raises -> PythonObject:
    """`LocallyLinearEmbedding.fit(X)`; returns embedding_."""
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    var fields = settings.values
    var mode = settings.mode
    var method = family_field(fields, "method")
    if not _mani_in(method, Python.tuple(PythonObject("standard"), PythonObject("hessian"), PythonObject("modified"),
                                         PythonObject("ltsa")), context):
        raise Error("LocallyLinearEmbedding: unrecognized method " + family_repr(method))
    var solver = family_field(fields, "eigen_solver")
    _mani_eigen_solver(solver, "LocallyLinearEmbedding", context)
    var plan = LlePlan()
    if _mani_eq(method, "standard", context):
        plan.method = LLE_STANDARD
    elif _mani_eq(method, "ltsa", context):
        plan.method = LLE_LTSA
    elif _mani_eq(method, "hessian", context):
        plan.method = LLE_HESSIAN
    else:
        plan.method = LLE_MODIFIED
    var run = MRun()
    try:
        var k = _mani_kit(run, mode, args[0], context)
        var hk = mk_image(run, String(settings.default_mode), args[1], context)
        var m = _mani_x(run, k, x)
        var n = run.r(m)
        plan.nn = _mani_int(family_field(fields, "n_neighbors"), context)
        plan.nc = _mani_int(family_field(fields, "n_components"), context)
        var nn = plan.nn
        var nc = plan.nc
        if nn >= n:
            raise Error("Expected n_neighbors < n_samples, but n_samples = " + String(n) + ", n_neighbors = "
                        + String(nn))
        if plan.method == LLE_HESSIAN and nn <= nc + nc * (nc + 1) // 2:
            raise Error("for method='hessian', n_neighbors must be greater than "
                        + "[n_components * (n_components + 3) / 2]")
        if plan.method == LLE_MODIFIED and nn < nc:
            raise Error("modified LLE requires n_neighbors >= n_components")
        if plan.method == LLE_STANDARD:
            plan.reg = Float64(py=py_call(b.float, family_field(fields, "reg")))
        elif plan.method == LLE_HESSIAN:
            plan.tol = _mani_float(family_field(fields, "hessian_tol"), context)
        elif plan.method == LLE_MODIFIED:
            plan.tol = _mani_float(family_field(fields, "modified_tol"), context)
        plan.auto_solver = _mani_eq(solver, "auto", context)
        if plan.method == LLE_STANDARD and plan.auto_solver and n > _LLE_ITER_MIN_N and nc + 1 < _LLE_ITER_MAX_K:
            plan.max_iter = _mani_int(family_field(fields, "max_iter"), context)
            plan.seed = mk_seed_word(_mani_seed(family_field(fields, "random_state")))
        var fit = LleFit()
        with GILReleased(Python()):
            fit = lle_fit_run(run, k, hk, m, plan)
        _ = x  # the admitted X stays alive through the RUNs above
        if fit.refusal == 1:
            raise Error("RuntimeError: LocallyLinearEmbedding: the shift-invert subspace iteration did not settle in "
                        + String(plan.max_iter) + " iterations (last subspace change "
                        + String(py=py_call(b.format, PythonObject(fit.change), PythonObject(".3g")))
                        + "); pass eigen_solver='dense' for the full SVD. "
                        + "An unconverged embedding is not returned as if it were one.")
        if fit.refusal == 2:
            var norms = Python.list()
            for i in range(len(fit.norms)):  # driver-loop: over (components: the refusal lists each column's squared norm): bounded by n_components
                _ = py_call(norms.append, PythonObject(fit.norms[i]))
            raise Error("RuntimeError: LocallyLinearEmbedding: the shift-invert subspace iteration returned columns of "
                        + "squared norm " + family_repr(norms) + " (not unit); pass eigen_solver='dense' for the full SVD.")
        # PACK
        var embedding = mk_out2(run, fit.em, context)
        var em = mk_owner(run, fit.em)
        var fit_x = mk_owner(run, m)
        py_setitem(settings.fields, "numeric_mode_", mode)
        py_setitem(settings.fields, "embedding_m_", em)
        py_setitem(settings.fields, "embedding_", embedding)
        py_setitem(settings.fields, "reconstruction_error_", PythonObject(fit.reconstruction_error))
        py_setitem(settings.fields, "_fit_X", fit_x)
        py_setitem(settings.fields, "_knn", PythonObject(nn))
        py_setitem(settings.fields, "n_features_in_", PythonObject(run.c(m)))
        return embedding
    finally:
        mk_finish(run, context)


def _lle_transform(model: PythonObject, settings: ManifoldSettings, args: PythonObject, x: ArrayMetadata, context: PythonObject) raises -> PythonObject:
    """`LocallyLinearEmbedding.transform(X)`."""
    _mani_check(model, settings.values, "embedding_m_")
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    var fields = settings.values
    var run = MRun()
    try:
        var k = _mani_kit(run, settings.mode, args[0], context)
        var q = _mani_x(run, k, x)
        var fit_x_owner = family_field(fields, "_fit_X")
        var em_owner = family_field(fields, "embedding_m_")
        var fit_x = mk_fitted(run, fit_x_owner, "_fit_X")
        var nn = _mani_int(family_field(fields, "_knn"), context)
        var reg = Float64(py=py_call(b.float, family_field(fields, "reg")))
        var em = mk_fitted(run, em_owner, "embedding_m_")
        var out = -1
        with GILReleased(Python()):
            out = lle_transform_run(run, k, q, fit_x, em, nn, reg)
        _ = x  # the admitted X stays alive through the RUNs above
        var result = mk_out2(run, out, context)
        mk_keep(run, fit_x, fit_x_owner)
        mk_keep(run, em, em_owner)
        return result
    finally:
        mk_finish(run, context)


def _mani_arguments(settings: ManifoldSettings, operation: String, args: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    # Public shells hand over settings and buffers only. Resolve both legacy
    # binding slots after crossing the native boundary. Explicit modules remain
    # supported for internal/legacy callers and are admitted by mk_image.
    if len(args) < 2:
        raise Error("TypeError: manifold operation requires two binding slots")
    var requested = settings.mode
    var binding = args[0]
    if binding is None:
        binding = estimator_binding_mode(requested, "_mojolearn_x_decomp", context)
    var default_binding = args[1]
    if default_binding is None:
        default_binding = estimator_binding_mode(settings.default_mode, "_mojolearn_x_decomp", context)
    var resolved = Python.list(binding, default_binding)
    for i in range(2, len(args)):  # driver-loop: over (args: fixed public operation argument envelope): bounded by the public signature
        _ = py_call(resolved.append, args[i])
    return py_call(native_loaded_module("builtins").tuple, resolved)


def mani_decomp_operation(model: PythonObject, settings: ManifoldSettings, operation: String, var args: PythonObject,
                          x: Optional[ArrayMetadata], context: PythonObject) raises -> PythonObject:
    """args = (optional decomp binding, optional default binding, X[, init])."""
    args = _mani_arguments(settings, operation, args, context)
    if operation == "family_mani_cmds_fit":
        _ = _cmds_fit(model, settings, args, x.value(), context)
        return model
    if operation == "family_mani_cmds_fit_transform":
        return _cmds_fit(model, settings, args, x.value(), context)
    if operation == "family_mani_mds_fit":
        _ = _mds_fit(model, settings, args, x.value(), context)
        return model
    if operation == "family_mani_mds_fit_transform":
        return _mds_fit(model, settings, args, x.value(), context)
    if operation == "family_mani_iso_fit":
        _ = _iso_fit(model, settings, args, x.value(), context)
        return model
    if operation == "family_mani_iso_fit_transform":
        return _iso_fit(model, settings, args, x.value(), context)
    if operation == "family_mani_iso_transform":
        return _iso_transform(model, settings, args, x.value(), context)
    if operation == "family_mani_iso_reconstruction_error":
        return _iso_reconstruction_error(model, settings, args, context)
    if operation == "family_mani_lle_fit":
        _ = _lle_fit(model, settings, args, x.value(), context)
        return model
    if operation == "family_mani_lle_fit_transform":
        return _lle_fit(model, settings, args, x.value(), context)
    if operation == "family_mani_lle_transform":
        return _lle_transform(model, settings, args, x.value(), context)
    raise Error("RuntimeError: unknown manifold operation " + operation)
