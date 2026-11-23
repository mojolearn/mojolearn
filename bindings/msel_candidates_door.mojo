# SPDX-License-Identifier: Apache-2.0
"""The CANDIDATES of a search, read at UNPACK (lane mf-search4, 2026-11-23;
moved word for word from bindings/msel_search_door.mojo so that the search
META kind's admission, providers/python/member_ensemble_boundary_door.mojo,
reads them without importing the search door, whose member units import the
member boundary): `ParameterGrid` / `ParameterSampler` iteration and the
scalar admissions they use. Python objects at the boundary only: these are
the UNPACK of a search fit.
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.py_slice_native import py_slice
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.native_numpy_metadata_unpack import native_numpy_dtype
from bindings.native_scalar_decode_unpack import native_scalar_buffer
from bindings.arena_materialize_native_door import _cast
from bindings.msel_parameter_grid_door import parameter_grid_dicts, PreparedParameterGrid
from bindings.family_common_unpack import family_str
from bindings.family_common_pack import family_call
from bindings.msel_common_door import (
    msel_b, msel_warn, msel_is_bool, msel_is_integral, msel_cmp, msel_materialize, MselRng,
)
from bindings.msel_split_door import msel_cls
from bindings.msel_search_state_door import msel_search_fields


# Scalar options and native numeric container leaves are admitted before
# int/comparison/slicing. Data admission alone accepts Array/ndarray owners,
# whose ordinary number slots are not a safe scalar conversion boundary.
def _search_scalar(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var b = msel_b()
    var cls = py_type(value)
    if cls is b.int or cls is b.float or cls is b.bool:
        return value
    classical_connect_numpy(context)
    var ndarray = py_get(context, "ndarray")
    if ndarray is not None:
        var records = py_get(context, "family_msel_search_scalar_records")
        if records is None:
            records = Python.list()
            var factory = native_numpy_dtype(context)
            for code in ["?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "q", "Q", "f2", "f4", "f8", "g"]:  # driver-loop: exact native scalar/dtype classes, including aliases and long double
                var dtype = py_call(factory, code)
                var integer = code == "?" or code.startswith("i") or code.startswith("u") or code == "q" or code == "Q"
                _ = py_call(records.append, Python.tuple(dtype, PythonObject(integer)))
            py_setitem(context, "family_msel_search_scalar_records", records)
        var dtype = value.dtype if cls is ndarray else PythonObject(None)
        for i in range(len(records)):  # driver-loop: identity-only admission before reading numeric buffer bytes
            var canonical = records[i][0]
            if cls is canonical.type or (cls is ndarray and py_type(dtype) is py_type(canonical)):
                return native_scalar_buffer(value).to_python_number(Bool(py=records[i][1]))
    if cls is context["array_class"] or (ndarray is not None and cls is ndarray):
        # Numeric object ndarray leaves are checked by the existing native
        # materializer before conversion. The Array path captures native
        # shape/storage and never invokes its Python number convenience hooks.
        var packed = msel_materialize(value, name, context)
        if Int(py=packed.size) != 1:
            raise Error("TypeError: " + name + " requires a single native numeric value")
        if String(packed.dtype) == "<f2":
            packed = _cast(packed, "<f8", context["family_msel_resident_context"])
        return packed._mv[0]
    raise Error("TypeError: " + name + " requires native numeric scalar or single-element buffer storage")


def _search_index(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var scalar = _search_scalar(value, name, context)
    var cls = py_type(scalar)
    if cls is not msel_b().int and cls is not msel_b().bool:
        raise Error("TypeError: " + name + " must be an integer")
    return scalar


def _search_seed(value: PythonObject, context: PythonObject) raises -> PythonObject:
    if value is None:
        return value
    if msel_is_bool(value, context) or not msel_is_integral(value, context):
        raise Error("TypeError: random_state must be None or a non-negative native integer")
    return _search_index(value, "random_state", context)


# ---------------------------------------------------------------- parameter grids


def msel_grid_dicts(param_grid_in: PythonObject, context: PythonObject) raises -> PythonObject:
    # Public construction keeps the original mutable value containers.
    return parameter_grid_dicts(param_grid_in, context)


def msel_grid_list(param_grid: PythonObject, context: PythonObject) raises -> PythonObject:
    """Admit/capture each value sequence once and emit the final candidates."""
    var prepared = PreparedParameterGrid(param_grid, context)
    return prepared.expand()


def msel_grid_len(param_grid: PythonObject, context: PythonObject) raises -> Int:
    var prepared = PreparedParameterGrid(param_grid, context, False, False)
    return prepared.count()


def msel_sampler_check(param_distributions_in: PythonObject, context: PythonObject,
                       capture_values: Bool = False) raises -> PythonObject:
    # The public contract already excludes scipy.stats distributions. Refuse
    # custom iterators/distributions without reading an rvs/__iter__ property.
    var b = msel_b()
    var rows = Python.tuple(param_distributions_in) if py_type(param_distributions_in) is b.dict else param_distributions_in
    if py_type(rows) is not b.list and py_type(rows) is not b.tuple:
        raise Error("TypeError: parameter distributions require native dictionaries")
    for k in range(len(rows)):  # driver-loop: distribution refusal reads class dictionaries, never rvs properties
        if py_type(rows[k]) is not b.dict:
            raise Error("TypeError: parameter distributions require native dictionaries")
        var items = py_call(b.list, rows[k].items())
        for j in range(len(items)):  # driver-loop: distribution value identities before protocol inspection
            var cls = py_type(items[j][1])
            if py_type(cls) is not b.type:
                raise Error("TypeError: parameter distributions cannot invoke Python metaclass hooks")
            var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
            for a in range(len(mro)):  # driver-loop: native class dictionary metadata, no descriptor binding
                var fields = py_call(b.type.__dict__["__dict__"].__get__, mro[a])
                if py_contains(fields, "rvs"):
                    raise Error("NotImplementedError: mojolearn ParameterSampler samples native parameter lists; scipy.stats and Python distribution callbacks are not supported")
    return parameter_grid_dicts(param_distributions_in, context, capture_values)


def msel_sampler_list(dists: PythonObject, n_iter_in: PythonObject, random_state: PythonObject,
                      context: PythonObject, captured: Bool = False) raises -> PythonObject:
    """`list(ParameterSampler)`: a random permutation of the full grid,
    its first n_iter combinations."""
    var n_iter = _search_index(n_iter_in, "n_iter", context)
    var seed = _search_seed(random_state, context)
    var prepared = PreparedParameterGrid(dists, context, captured)
    var grid = prepared.expand()
    if msel_cmp("lt", PythonObject(len(grid)), n_iter):
        msel_warn("The total space of parameters " + String(len(grid)) + " is smaller than n_iter="
                  + family_str(n_iter) + ". Running " + String(len(grid)) + " iterations. For exhaustive "
                  + "searches, use GridSearchCV.", msel_b().UserWarning, context)
        n_iter = PythonObject(len(grid))
    var rng = MselRng(seed, context)
    var perm = rng.permutation(len(grid), context)
    var head = py_slice(perm, PythonObject(None), n_iter, PythonObject(None))
    var out = Python.list()
    for k in range(len(head)):  # driver-loop: native permutation indices choose parameter dictionaries
        _ = py_call(out.append, grid[Int(py=head[k])])
    _ = perm  # the native RNG returns a plain list of integer indices
    return out


def msel_sampler_len(dists: PythonObject, n_iter: PythonObject, context: PythonObject) raises -> PythonObject:
    """`len(ParameterSampler)`."""
    var count = _search_index(n_iter, "n_iter", context)
    return family_call(msel_b().min, count, PythonObject(msel_grid_len(dists, context)))


# ---------------------------------------------------------------- score tables


# RUN: bindings/msel_search_stats_run.mojo (row means and deviations, the
# candidates' ranks, the best candidate, the p-value) on native tables; this
# door packs its answers into `cv_results_`.


# ---------------------------------------------------------------- search


def msel_candidates(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = msel_search_fields(model, context)
    var kind = py_type(model)
    if kind is msel_cls("GridSearchCV", context):
        return msel_grid_list(fields["param_grid"], context)
    if kind is msel_cls("RandomizedSearchCV", context):
        var dists = msel_sampler_check(fields["param_distributions"], context, True)
        return msel_sampler_list(dists, fields["n_iter"], fields["random_state"], context, True)
    raise Error("TypeError: Python candidate callbacks are not supported; use GridSearchCV or RandomizedSearchCV")
