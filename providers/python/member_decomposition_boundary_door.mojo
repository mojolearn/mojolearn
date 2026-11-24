# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Decomposition family (PCA, TruncatedSVD): its
place in `bindings/member_boundary_door.mojo` (recipe: MEMBER-ABI-V2.md;
RUN: core/decomposition_member.mojo). Kind 0 is PCA, kind 1 TruncatedSVD
(`_mojolearn_member = (31, kind)` on each class).

  * `decomposition_member_admit` (UNPACK, once per member): the estimator's
    own route through the decomposition resolver (`pca_route`, the
    TruncatedSVD door's `_route`: a hit is one record read), turned into the
    member's `DeviceRoute` (the native entries of its connections and their
    capability bits), and the nine parameter words of the member file. The
    parameter refusals are the estimators' own, in the order their fits
    raise the data-independent ones; the data-dependent ones (n_components
    against the shape, a wide 'full' fit) the member raises at its fit.
  * `decomposition_member_pack` (PACK): the fitted attributes of the
    estimator's own fit, from the member state's blocks.
  * `decomposition_member_state` (UNPACK): a member state from a fitted
    estimator.

SERIAL BY NATURE: one member's admission or publication (a few record reads
and five or six block copies); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_input_unpack import admit_x
from bindings.classical_arrays_door import classical_address
from bindings.member_owner_unpack import MemberOwner
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_item, py_type, py_setitem
from bindings.sequential_arrays_door import sequential_array, sequential_address
from bindings.pca_keys_unpack import pca_keys
from bindings.pca_descriptor_boundary_door import pca_route, PCA_PROVIDER_PINS
from bindings.decomp_device_route_door import DecompRoute, decomp_route_expected_vendor
from bindings.decomp_scalar_admission_unpack import decomp_scalar_admit, decomp_scalar_repr
from bindings.tsvd_descriptor_boundary_door import tsvd_connect, tsvd_provider
from bindings.tsvd_native_door import _route as tsvd_route, _decomp as tsvd_decomp, _algorithm as tsvd_algorithm
from bindings.tsvd_native_door import _int as tsvd_int, _f32 as tsvd_f32, _address as tsvd_address
from bindings.tsvd_native_door import _host_only as tsvd_host_only
from bindings.tsvd_owner_boundary_unpack import tsvd_owner_drop
from providers.python.pca_fit_scalar_pyapi_unpack import PCAFitScalar
from providers.python.pca_fit_model_pyapi_unpack import pca_fit_host_only, pca_fit_preflight
from x_decomp.pca_descriptor_boundary_run import pca_driver, pca_capabilities, pca_vendor, pca_whiten_capability
from x_decomp.tsvd_descriptor_boundary_run import tsvd_driver, tsvd_capabilities, tsvd_qfix, tsvd_vendor
from core.device_route import DeviceRoute
from core.estimators_entry_table import (
    EstimatorsEntryKey, decomp_fold_entry, DECOMP_FOLD_PCA, DECOMP_FOLD_TSVD, DECOMP_FOLD_STORE,
)
from x_prep.transport_abi import XP_OP_COUNT
from core.entry_table import EntryRef, entry_ref_word
from core.hostptr import f32_ptr
from core.pca_native_api import (
    PCA_NATIVE_COUNT, PCA_CAP_EXACT, PCA_CAP_FULL, PCA_CAP_RANDOMIZED, pca_vendor_code,
)
from core.tsvd_native_api import TSVD_NATIVE_COUNT, TSVD_CAP_FUSED, TSVD_CAP_RANDOMIZED, TSVD_CAP_TSQR, TSVD_CAP_RESIDENT
from core.decomposition_member import (
    DECOMP_KIND_PCA, DECOMP_BLOCK_COMPONENTS, DECOMP_BLOCK_MEAN, DECOMP_BLOCK_VARIANCE, DECOMP_BLOCK_RATIO,
    DECOMP_BLOCK_SINGULAR, DECOMP_ROUTE_FUSED, DECOMP_ROUTE_TSQR, DECOMP_ROUTE_TSQR_MISSING, DECOMP_ROUTE_RANDOMIZED, DECOMP_ROUTE_RESIDENT,
    DECOMP_ROUTE_FOLD,
)
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_IMPORT_DONE, MEMBER_F64,
)

#: x_decomp/qfix.mojo TSVD_QFIX's bit in `x_decomp_qfix_flags` (the
#: TruncatedSVD door's `_QFIX_TSVD`).
comptime _QFIX_TSVD = 2


def _route_of(route: DecompRoute, vendor_code: Int) -> DeviceRoute:
    # The decomposition resolver's words: `vendor_code` is the connection's
    # PCA-style vendor code (1 cpu, 2 metal, 3 cuda, 4 hip).
    var built = DeviceRoute(route.device)
    built.mode = route.mode
    built.vendor = vendor_code - 1
    return built


def _fold_route(mut route: DeviceRoute, which: Int):
    """Lane vdf-scope-neural2: THE RESIDENT ROUTE of the PCA / TruncatedSVD
    members. On an identical GPU route, the base image's resident-fold entry
    of the kind (`which`, core/estimators_entry_table.mojo DECOMP_FOLD_PCA /
    DECOMP_FOLD_TSVD) in `route.fourth` and the fold store's transport entry
    in `route.fifth` with DECOMP_ROUTE_FOLD, resolved once here; any other
    route (fast, host, a build without the base binding's table) is kept as
    it was, so its kind answers no row index."""
    if route.mode != 1 or route.vendor <= 0:
        return
    try:
        var key = EstimatorsEntryKey(1, route.vendor)
        var fold = decomp_fold_entry(key, which)
        var store = decomp_fold_entry(key, DECOMP_FOLD_STORE)
        var count = PCA_NATIVE_COUNT if which == DECOMP_FOLD_PCA else TSVD_NATIVE_COUNT
        route.fourth = EntryRef(fold, entry_ref_word(1, route.vendor, count))
        route.has_fourth = True
        route.fifth = EntryRef(store, entry_ref_word(1, route.vendor, XP_OP_COUNT))
        route.has_fifth = True
        route.flags |= DECOMP_ROUTE_FOLD
    except:
        pass


def _randomized_words(fields: PythonObject, power_name: String, context: PythonObject,
    mut words: List[Float64]) raises:
    """n_oversamples, the power iterations ('auto': -1), the seed word."""
    var b = native_loaded_module("builtins")
    var normalizer = fields["power_iteration_normalizer"]
    var name = String(normalizer) if py_type(normalizer) is b.str else String("")
    if name != "auto" and name != "QR" and name != "LU" and name != "none":
        raise Error("power_iteration_normalizer must be 'auto', 'QR', 'LU' or 'none'")
    var seed = fields["random_state"]
    if seed is not None and py_type(seed) is not b.int:
        raise Error("TypeError: random_state must be None or an int (the counter-based RNG takes an integer seed)")
    words[6] = Float64(tsvd_int(fields["n_oversamples"], context))
    if words[6] < 0.0:
        raise Error("n_oversamples must be nonnegative")
    var power = fields[power_name]
    words[7] = -1.0 if (py_type(power) is b.str and String(power) == "auto") else Float64(tsvd_int(power, context))
    if seed is not None:
        # The Philox seed is taken mod 2^32 with Python integer semantics.
        words[8] = Float64(Int(py=py_call(py_item(b.int.__dict__, "__and__"), seed, PythonObject(0xFFFFFFFF))))


def _pca_admit(model: PythonObject, fields: PythonObject, context: PythonObject, mut owner: MemberOwner) raises:
    var nk = pca_keys()
    var b = native_loaded_module("builtins")
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    if py_type(mode) is not b.str:
        raise Error("TypeError: numeric_mode must be identical, fast or None")
    var held = pca_route(model, fields, mode, context)
    var connection = held.connection
    if pca_fit_host_only(model):
        raise Error("ImportError: mojolearn: no CPU implementation of the PCA fit on the host class; "
                    + "the host PCA transforms a saved model only")
    var capabilities = pca_capabilities(connection)
    var preflight = pca_fit_preflight(fields, "fit", context)
    var solver = preflight[0]
    var whiten = preflight[1]
    if whiten:
        pca_whiten_capability(connection)
    var randomized = solver == "randomized"
    var words = List[Float64](length=9, fill=0.0)
    words[0] = 0.0
    words[1] = -1.0
    words[4] = 2.0 if randomized else (1.0 if solver == "full" else 0.0)
    words[5] = 1.0 if whiten else 0.0
    var requested = py_item(fields, nk[].k_n_components)
    if not randomized and py_type(requested) is b.str and String(requested) == "mle":
        words[2] = 2.0
    elif not randomized and (py_type(requested) is b.float or py_type(requested) is py_item(context, nk[].k_float64).type):
        var fraction = Float64(py=PCAFitScalar(requested, context).convert("fit", context, False))
        if not (fraction > 0 and fraction < 1):
            raise Error("mojolearn PCA: a float n_components must be in (0, 1)")
        words[2] = 1.0
        words[3] = fraction
    elif requested is not None:
        words[1] = Float64(Int(py=PCAFitScalar(requested, context).convert("fit", context, True)))
    if randomized:
        _randomized_words(fields, "iterated_power", context, words)
        if (capabilities & PCA_CAP_RANDOMIZED) == 0:
            raise Error("NotImplementedError: mojolearn PCA: this build of _mojolearn_estimators does not export "
                        + "pca_fit_randomized; rebuild it with bindings/build_estimators.sh")
    elif (capabilities & PCA_CAP_EXACT) == 0:
        raise Error("NotImplementedError: mojolearn PCA: this build of _mojolearn_estimators does not export "
                    + "pca_fit_owned; rebuild it with bindings/build_estimators.sh")
    if solver == "full" and (capabilities & PCA_CAP_FULL) == 0:
        raise Error("NotImplementedError: mojolearn PCA: svd_solver='full' needs the dense arm and this build of "
                    + "_mojolearn_estimators does not export it; rebuild it with bindings/build_estimators.sh")
    var vendor = pca_vendor_code(pca_vendor(connection))
    var route = _route_of(held, vendor)
    route.entry = EntryRef(pca_driver(connection).entry, entry_ref_word(route.mode, route.vendor, PCA_NATIVE_COUNT))
    route.flags = capabilities
    _fold_route(route, DECOMP_FOLD_PCA)
    owner.plan.route = route
    owner.plan.params = words^
    owner.pins = Python.tuple(PCA_PROVIDER_PINS.get_or_create_ptr()[].providers[connection], mode)


def _tsvd_admit(model: PythonObject, fields: PythonObject, context: PythonObject, mut owner: MemberOwner) raises:
    var b = native_loaded_module("builtins")
    if tsvd_host_only(model):
        raise Error("TypeError: mojolearn: the host TruncatedSVD class does not run as a member")
    var tolerance = fields["tol"]
    decomp_scalar_admit(tolerance, context)
    if Float64(py=py_call(b.float, tolerance)) != 0.0:
        raise Error("NotImplementedError: mojolearn TruncatedSVD: tol is ARPACK's convergence tolerance "
                    + "and algorithm='arpack' is not implemented; leave tol=0.0")
    var algorithm = tsvd_algorithm(fields)
    if algorithm != "randomized" and algorithm != "covariance_eigh" and algorithm != "jacobi":
        raise Error("NotImplementedError: mojolearn TruncatedSVD: algorithm=" + decomp_scalar_repr(fields["algorithm"], context)
                    + " is not implemented; this class runs ('covariance_eigh', 'jacobi', 'randomized'). "
                    + "'arpack' is scipy's svds (implicitly restarted Lanczos), a different algorithm; "
                    + "accepting the name and running the Gram arm would be a silent substitution, which is "
                    + "why this raises (decomposition/NOT_IMPLEMENTED.tsv)")
    var held = tsvd_route(model, fields, context)
    var words = List[Float64](length=9, fill=0.0)
    words[0] = 1.0
    var requested = fields["n_components"]
    words[1] = -1.0 if requested is None else Float64(tsvd_int(requested, context))
    var randomized = algorithm == "randomized"
    words[4] = 2.0 if randomized else 0.0
    var extra = held.extra
    if randomized:
        if extra < 0:
            var decomp = tsvd_decomp(held, context)
            if py_type(decomp) is b.str:
                raise Error(String(decomp))
            extra = tsvd_connect(decomp, held.mode, decomp_route_expected_vendor(held.device, context))
        _randomized_words(fields, "n_iter", context, words)
    var route = _route_of(held, tsvd_vendor(held.connection))
    route.entry = EntryRef(tsvd_driver(held.connection).entry,
        entry_ref_word(route.mode, route.vendor, TSVD_NATIVE_COUNT))
    var held_caps = tsvd_capabilities(held.connection)
    var flags = DECOMP_ROUTE_FUSED if (held_caps & TSVD_CAP_FUSED) != 0 else 0
    if (held_caps & TSVD_CAP_RESIDENT) != 0:
        flags |= DECOMP_ROUTE_RESIDENT
    if extra >= 0:
        route.second = EntryRef(tsvd_driver(extra).entry, entry_ref_word(route.mode, route.vendor, TSVD_NATIVE_COUNT))
        route.has_second = True
        var qfix = 0
        try:
            qfix = tsvd_qfix(extra)
        except ignored:
            # The door treats an unavailable flag entry as zero (the Gram fit).
            qfix = 0
        var second = tsvd_capabilities(extra)
        if (qfix & _QFIX_TSVD) != 0:
            flags |= DECOMP_ROUTE_TSQR if (second & TSVD_CAP_TSQR) != 0 else DECOMP_ROUTE_TSQR_MISSING
        if (second & TSVD_CAP_RANDOMIZED) != 0:
            flags |= DECOMP_ROUTE_RANDOMIZED
    route.flags = flags
    _fold_route(route, DECOMP_FOLD_TSVD)
    owner.plan.route = route
    owner.plan.params = words^
    # The connections' images, as the door's provider pins hold them.
    owner.pins = Python.tuple(tsvd_provider(held.connection), tsvd_provider(extra) if extra >= 0 else PythonObject(None))


def decomposition_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator."""
    if kind == DECOMP_KIND_PCA:
        _pca_admit(model, fields, context, owner)
    elif kind == 1:
        _tsvd_admit(model, fields, context, owner)
    else:
        raise Error("ValueError: mojolearn: unknown decomposition member kind " + String(kind))
    return True


def _export(plan: MemberPlan, mut call: MemberCall, block: Int) raises -> MemberMatrix:
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    return call.dst


def _pca_array(view: MemberMatrix, matrix: Bool, context: PythonObject) raises -> PythonObject:
    var nk = pca_keys()
    var shape = Python.tuple(view.rows, view.cols) if matrix else Python.tuple(view.cols)
    var array = sequential_array(shape, py_item(context, nk[].k_float32), context)
    var count = view.rows * view.cols
    if count > 0:
        unsafe_memcpy(dest=f32_ptr(Int(py=sequential_address(array))), src=f32_ptr(view.data), count=count)
    return array


def _tsvd_array(view: MemberMatrix, matrix: Bool, context: PythonObject) raises -> PythonObject:
    var shape = Python.tuple(view.rows, view.cols) if matrix else Python.tuple(view.cols)
    var array = tsvd_f32(shape, context)
    var count = view.rows * view.cols
    if count > 0:
        unsafe_memcpy(dest=f32_ptr(tsvd_address(array, context)), src=f32_ptr(view.data), count=count)
    return array


def decomposition_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does (the same names and array types its own
    transform, inverse_transform and save read)."""
    var call = MemberCall(plan.route, state)
    var scalars_view = _export(plan, call, MEMBER_BLOCK_SCALARS)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars_view.data)
    var pca = plan.kind == DECOMP_KIND_PCA
    if pca:
        var noise = scalars[0]
        var keep = Int(scalars[1])
        var cols = Int(scalars[2])
        var rows = Int(scalars[3])
        var components = _pca_array(_export(plan, call, DECOMP_BLOCK_COMPONENTS), True, context)
        var mean = _pca_array(_export(plan, call, DECOMP_BLOCK_MEAN), False, context)
        var variance = _pca_array(_export(plan, call, DECOMP_BLOCK_VARIANCE), False, context)
        var ratio = _pca_array(_export(plan, call, DECOMP_BLOCK_RATIO), False, context)
        var singular = _pca_array(_export(plan, call, DECOMP_BLOCK_SINGULAR), False, context)
        # The member was fitted on the caller's own rows, never on the user's array.
        py_setitem(fields, "input_copied_", True)
        py_setitem(fields, "components_", components)
        py_setitem(fields, "mean_", mean)
        py_setitem(fields, "explained_variance_", variance)
        py_setitem(fields, "explained_variance_ratio_", ratio)
        py_setitem(fields, "singular_values_", singular)
        py_setitem(fields, "noise_variance_", PythonObject(noise))
        py_setitem(fields, "n_components_", keep)
        py_setitem(fields, "n_features_in_", cols)
        py_setitem(fields, "n_samples_", rows)
        py_setitem(fields, "_mojolearn_fitted", True)
        return
    var keep = Int(scalars[0])
    var cols = Int(scalars[1])
    # New components replace the old: a resident device copy of the old goes now.
    tsvd_owner_drop(model)
    var components = _tsvd_array(_export(plan, call, DECOMP_BLOCK_COMPONENTS), True, context)
    var variance = _tsvd_array(_export(plan, call, DECOMP_BLOCK_VARIANCE), False, context)
    var ratio = _tsvd_array(_export(plan, call, DECOMP_BLOCK_RATIO), False, context)
    var singular = _tsvd_array(_export(plan, call, DECOMP_BLOCK_SINGULAR), False, context)
    py_setitem(fields, "input_copied_", True)
    py_setitem(fields, "components_", components)
    py_setitem(fields, "singular_values_", singular)
    py_setitem(fields, "explained_variance_", variance)
    py_setitem(fields, "explained_variance_ratio_", ratio)
    py_setitem(fields, "n_components_", PythonObject(keep))
    py_setitem(fields, "n_features_in_", PythonObject(cols))
    py_setitem(fields, "_mojolearn_fitted", True)


def _import(plan: MemberPlan, mut call: MemberCall, fields: PythonObject, name: String, ndim: Int, block: Int,
    context: PythonObject) raises:
    var source = py_get(fields, name)
    if source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var words = admit_x(source, ndim, name, context)
    var size = Int(py=words.size)
    call.block = block
    call.x = MemberMatrix(Int(py=classical_address(words)), 1, size)
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    _ = words


def decomposition_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`.
    The caller frees it (`member_free`)."""
    var pca = plan.kind == DECOMP_KIND_PCA
    var keep_source = py_get(fields, "n_components_")
    var cols_source = py_get(fields, "n_features_in_")
    if keep_source is None or cols_source is None or py_get(fields, "components_") is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var scalars = List[Float64]()
    if pca:
        var rows_source = py_get(fields, "n_samples_")
        var noise_source = py_get(fields, "noise_variance_")
        scalars = [Float64(py=noise_source) if noise_source is not None else 0.0, Float64(Int(py=keep_source)),
            Float64(Int(py=cols_source)), Float64(Int(py=rows_source)) if rows_source is not None else 0.0]
    else:
        scalars = [Float64(Int(py=keep_source)), Float64(Int(py=cols_source))]
    var call = member_new(plan)
    try:
        _import(plan, call, fields, "components_", 2, DECOMP_BLOCK_COMPONENTS, context)
        if pca:
            _import(plan, call, fields, "mean_", 1, DECOMP_BLOCK_MEAN, context)
        _import(plan, call, fields, "explained_variance_", 1, DECOMP_BLOCK_VARIANCE, context)
        _import(plan, call, fields, "explained_variance_ratio_", 1, DECOMP_BLOCK_RATIO, context)
        _import(plan, call, fields, "singular_values_", 1, DECOMP_BLOCK_SINGULAR, context)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(scalars)
    return call.state
