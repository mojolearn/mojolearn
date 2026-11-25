# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The native owner of the Python-sharded multi-GPU drivers
(lane/native-parallel-driver-20261024,
plan NATIVE_PARALLEL_DRIVER_20261024 (git history), "kind B"):
`parallel_classical.transform_rbf_sampler`, `parallel_preprocessing.fit_scaler`
and `transform_scaler`, `parallel_gaussian_process.fit_gaussian_process_classifier`
and `predict_gaussian_process_classifier`, and
`parallel_ensemble.fit_forest`, one call each into `parallel_shard_call`.
The forest workers receive typed request words and arrays and call the one
forest fit core (lane forest-typed-fit); they build no estimator.

Each driver's partition, request order, driver negative control and merge
are its old Python driver's (bindings/parallel_shard_common_door.mojo for the
shared parts): the shards run their plain native owners in worker processes
on the native local process pool, one per device, and every shard's block is
placed at its own offset in shard order. The same compiled entries on the
same words: no bit moves.

The worker (`mojolearn._native_parallel_shard_worker`) imports the package,
whose driver modules register their classes, entries and contexts here at
import, and enters `shard_worker`. A worker rebuilds each shard's model from
its state record without a Python constructor. Bindings are connected before
any request and retained as namespaces of compiled entries; workers never
call the Python binding loader. Neighbor and IVF requests dispatch to the
native handlers in parallel_neighbor_worker; no estimator callback is stored.
"""
from std.ffi import external_call
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from core.native_label_codes import NativeLabelCodes
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.admitted_family_door import admit_family, admit_family_meta, admit_family_input, admit_family_cast
from providers.python.array_metadata_pyapi_door import ArrayPin
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_fields, py_contains
from bindings.py_attr import py_attr
from bindings.classical_arrays_door import classical_address
from bindings.classical_attributes_door import classical_connect_numpy, classical_check_layout
from bindings.classifier_public_lifecycle_door import classifier_after_fit
from bindings.scaler_lifecycle_native_door import _predict_result, _entries, scaler_standard_rows, _fitted as scaler_fitted
from bindings.scaler_native_door import scaler_door_calls, scaler_standard_publish, _bool as _scaler_bool
from bindings.minmax_native_door import minmax_rows, minmax_fit_publish, _configuration as minmax_configuration
from bindings.minmax_metadata_unpack import _flag as _minmax_flag
from bindings.minmax_metadata_pack import minmax_range_pair
from core.scaler_core import ScalerTransformSpec, scaler_standard_fit_core, scaler_minmax_fit_core, scaler_transform_core
from bindings.parallel_public_result_door import parallel_public_result
from bindings.msel_member_device_unpack import member_says_cpu
from bindings.parallel_neighbor_worker_door import parallel_neighbor_worker, parallel_ivf_admit, neighbor_radius_rows
from bindings.native_process_pool_door import (
    process_worker_connect, process_worker_next, process_worker_value, process_worker_reply, process_worker_refuse,
    process_pool_create, process_pool_close, process_pool_closed, process_pool_pids, process_pool_devices,
    process_pool_host,
)
from bindings.native_transport_door import transport_close
from core.entry_table import EntryRef
from core.parallel_entry_table import (
    ParallelEntryKey, parallel_ref, PARALLEL_TABLE_BASE, PARALLEL_TABLE_IVF, parallel_ivf_finalize, parallel_ivf_merge_shards, parallel_ivf_shard_plan,
    parallel_rebase_offsets_i32, parallel_topk_merge_f32,
)
from std.python._cpython import GILReleased
from bindings.native_codec_door import codec_initialize
from bindings.native_loaded_module_door import NativeModules
from bindings.native_prepared_owner_unpack import prepared_key_fields, prepared_key_same, prepared_get, prepared_put, prepared_drop
from bindings.parallel_shard_common_door import (
    shard_b,
    shard_ranges,
    shard_devices,
    shard_run,
    shard_rows,
    shard_concat,
    shard_state,
    shard_model,
    shard_stored_model,
    shard_array,
    shard_join_rows,
    shard_row_vector,
    shard_open,
    shard_identity,
    shard_trace_take,
    shard_map,
    shard_dtype,
    shard_run_stored,
    shard_stored_ref,
    shard_store_put,
    shard_store_drop,
    shard_resolve_stored,
    ShardWave,
    shard_wave,
)
from bindings.parallel_shard_common_run import (
    shard_request_op,
    PARALLEL_SHARD_WORKER_OPN_FOREST_FIT,
    PARALLEL_SHARD_WORKER_OPN_GPC_CLASS_FIT,
    PARALLEL_SHARD_WORKER_OPN_GPC_CLASS_PREDICT,
    PARALLEL_SHARD_WORKER_OPN_IDENTITY,
    PARALLEL_SHARD_WORKER_OPN_IVF_SEARCH,
    PARALLEL_SHARD_WORKER_OPN_IVF_STORE,
    PARALLEL_SHARD_WORKER_OPN_QUERY_ROWS,
    PARALLEL_SHARD_WORKER_OPN_QUERY_STATE,
    PARALLEL_SHARD_WORKER_OPN_RBF_ROWS,
    PARALLEL_SHARD_WORKER_OPN_REFERENCE_QUERY,
    PARALLEL_SHARD_WORKER_OPN_REFERENCE_SHARD,
    PARALLEL_SHARD_WORKER_OPN_REFERENCE_STORE,
    PARALLEL_SHARD_WORKER_OPN_REFERENCE_VOTE,
    PARALLEL_SHARD_WORKER_OPN_RELEASE,
    PARALLEL_SHARD_WORKER_OPN_SCALER_FIT,
    PARALLEL_SHARD_WORKER_OPN_SCALER_TRANSFORM,
    PARALLEL_SHARD_WORKER_OPN_SHARD_RELEASE,
    PARALLEL_SHARD_WORKER_OPN_SHARD_STORE,
)
from bindings.parallel_shard_common_run import shard_read_shift, shard_faults_take, shard_store_token, shard_faults_arm, shard_sabotage_arm, shard_width_admit, shard_batch_admit
from bindings.parallel_shard_native_run import (
    shard_mode_identical, shard_identical_admit, shard_fitted_admit, shard_positive_admit, shard_features_admit,
    shard_envelope_admit, rbf_features_admit, rbf_part_admit, scaler_input_admit, scaler_weight_admit,
    scaler_fit_mode_admit, scaler_block_admit, gpc_mode_kind_admit, gpc_mode_defaults, gpc_mode_admit,
    gpc_labels_admit, gpc_method_admit, gpc_features_admit, gpc_rows_admit, forest_count_admit,
    forest_offsets_admit, reference_algorithm_admit, reference_knn_admit, reference_k_admit, reference_cell_limit,
    reference_width, reference_ids_admit, reference_merge_admit, ivf_open_admit, ivf_offsets_admit,
    ivf_receipt_admit, ivf_queries_admit, ivf_features_admit, ivf_parts_admit, ivf_merge_admit,
)
from bindings.kernel_methods_native_door import _rbf_transform
from bindings.gp_classifier_native_door import (
    gpc_validate, gpc_worker_fit, gpc_worker_latent, _gpc_binary_result, GPCFitPublication, _gpc_set_fitted, _gpc_ovr_combine,
)
from bindings.gp_common_native_unpack import gp_kernel_arrays_transport
from bindings.gp_labels_native_door import gp_encode_labels_transport, gp_decode_labels
from bindings.trees_public_forest_door import forest_shard_prepare, forest_shard_fit, forest_shard_publish
from core.parallel_forecast_impl import forecast_gather_cols, forecast_scatter_cols
from core.hotpath_impl import cast_elements, HP_I32, HP_I64, HP_U32
from core.shared_slot import SharedSlot
from bindings.parallel_keys_unpack import (
    parallel_keys,
    PARALLEL_SHARD_OPN_FAULTS_ARM,
    PARALLEL_SHARD_OPN_FAULTS_TAKE,
    PARALLEL_SHARD_OPN_FIT_FOREST,
    PARALLEL_SHARD_OPN_FIT_GAUSSIAN_PROCESS_CLASSIFIER,
    PARALLEL_SHARD_OPN_FIT_SCALER,
    PARALLEL_SHARD_OPN_HOLDER_CLOSE,
    PARALLEL_SHARD_OPN_HOLDER_ENTER,
    PARALLEL_SHARD_OPN_IVF_CLOSE,
    PARALLEL_SHARD_OPN_IVF_ENTER,
    PARALLEL_SHARD_OPN_IVF_FROM_INDEX,
    PARALLEL_SHARD_OPN_IVF_INIT,
    PARALLEL_SHARD_OPN_IVF_SEARCH,
    PARALLEL_SHARD_OPN_PARALLEL_QUERIES,
    PARALLEL_SHARD_OPN_PREDICT_GAUSSIAN_PROCESS_CLASSIFIER,
    PARALLEL_SHARD_OPN_QUERIES_INIT,
    PARALLEL_SHARD_OPN_REFERENCE_INIT,
    PARALLEL_SHARD_OPN_REFERENCE_QUERY,
    PARALLEL_SHARD_OPN_REGISTER,
    PARALLEL_SHARD_OPN_SABOTAGE_ARM,
    PARALLEL_SHARD_OPN_TRACE_TAKE,
    PARALLEL_SHARD_OPN_TRANSFORM_RBF_SAMPLER,
    PARALLEL_SHARD_OPN_TRANSFORM_SCALER,
    PARALLEL_SHARD_OPN_WORKER,
)
from bindings.py_keys import py_op_code
from bindings.py_balanced import py_item

comptime _CHILD = "mojolearn._native_parallel_shard_worker"


struct ParallelShardBoundary(Defaultable, Movable):
    var tables: PythonObject

    def __init__(out self):
        self.tables = PythonObject(None)


comptime PARALLEL_SHARD_BOUNDARY = SharedSlot[ParallelShardBoundary, "MojolearnParallelShardOwnerV1"]


def _t(name: String) raises -> PythonObject:
    """A registered table entry (a class, an entry, a context or a getter)."""
    var s = PARALLEL_SHARD_BOUNDARY.get_or_create_ptr()
    if s[].tables is None:
        raise Error("RuntimeError: the parallel shard owner is not connected")
    var value = py_get(s[].tables, PythonObject(name), PythonObject(None))
    if value is None:
        raise Error("RuntimeError: the parallel shard owner has no " + name + "; import its driver module")
    return value


def _ctx() raises -> PythonObject:
    """The family owners' shared context (array types, codes, classes)."""
    return _t("context")


def _table_key(family: String) raises -> EntryRef:
    return _table_key(family, False)


def _table_key(family: String, host: Bool) raises -> EntryRef:
    """UNPACK: the entry of the image whose native table the owner calls for
    `family` ("base" or "ivf"), keyed by the (tier, vendor) pair recorded at
    import by python/mojolearn/_native_parallel_shard.py
    (core/parallel_entry_table.mojo), resolved once per operation (lane
    entry-table-zero-alloc); the table refusals raise here. `host` reads the
    CPU image's key (`host_table_keys`): the merges of a device='cpu'
    member's host pool run on the CPU column's code, never the GPU image."""
    var key = py_get(_t("host_table_keys" if host else "table_keys"), PythonObject(family))
    if key is None:
        raise Error("ImportError: mojolearn: the " + family + " binding publishes no native entry table")
    return parallel_ref(PARALLEL_TABLE_BASE if family == "base" else PARALLEL_TABLE_IVF,
                        ParallelEntryKey(Int(py=key[0]), Int(py=key[1])))


def _worker_binding(name: String) raises -> PythonObject:
    return _worker_binding(name, False)


def _worker_binding(name: String, host: Bool) raises -> PythonObject:
    """The import-connected identical binding, without a loader callback.
    `host` takes its CPU twin (`host_worker_bindings`, connected at import
    beside the device one): the parent-side label codes and combinations of
    a device='cpu' member's host pool."""
    var binding = py_get(_t("host_worker_bindings" if host else "worker_bindings"), PythonObject(name))
    if binding is None:
        raise Error("ImportError: parallel native binding is not connected: " + name)
    if py_type(binding) is shard_b().str:
        raise Error("ImportError: " + String(binding))
    return binding


def _member_host(fields: PythonObject) raises -> Bool:
    """UNPACK, once per driver call: True when the member says device='cpu'
    in a process with a GPU set. The driver then runs its shards on a host
    pool (`shard_open(..., host=True)`: workers on the CPU-only set, the
    CPU implementation of every shard) and its parent-side merges on the CPU
    images (`_table_key(..., host)`, `_worker_binding(..., host)`); `devices`
    are logical worker indices there (vdv-gaps, 2026-11-24: the member was
    refused, `member_multi_gpu_refusal`, so the CPU side could not run in a
    GPU process). The shared read is bindings/msel_member_device_unpack.mojo;
    the process's vendor word is the base image's recorded table key (0 CPU
    only, where 'cpu' is the installation's device and its own pool is
    already the host one), read only for a member that says 'cpu'."""
    if not member_says_cpu(fields, shard_b()):
        return False
    var key = py_get(_t("table_keys"), PythonObject("base"))
    if key is not None:
        if Int(py=key[1]) == 0:
            return False
    return True


def _pool_host(pool: PythonObject) raises -> Bool:
    """A held pool's route (a holder's or a distributed index's), read once
    per query call: True for a host pool, one moved off the process's
    installed route (`process_pool_host`); never on a CPU-only install, whose
    own pools merge on the recorded `table_keys` (w3f7-a, 2026-11-24)."""
    return process_pool_host(pool)


def _identical(fields: PythonObject) raises -> Bool:
    var mode = py_get(fields, PythonObject("numeric_mode"), PythonObject(None))
    var exact = mode is not None and py_type(mode) is shard_b().str
    return shard_mode_identical(mode is not None, exact, String(mode) if exact else String(""))


# ---------------------------------------------------------------- RBFSampler

def _transform_rbf_sampler(args: PythonObject) raises -> PythonObject:
    """args (estimator, X, devices, rows_per_shard, binding): whole query rows
    in `rows_per_shard` blocks; each output row is a pure function of its
    input row, so blocks are joined by copying bytes in input order."""
    var b = shard_b()
    var context = _ctx()
    var estimator = args[0]
    if py_type(estimator) is not _t("rbf_class"):
        raise Error("TypeError: requires mojolearn.RBFSampler")
    var fields = py_fields(estimator)
    var host = _member_host(fields)
    shard_identical_admit(_identical(fields), "parallel RBFSampler requires IDENTICAL numeric mode")
    shard_fitted_admit(py_contains(fields, PythonObject("random_weights_")), "RBFSampler is not fitted")
    var rps = args[3]
    var rps_exact = py_type(rps) is b.int
    shard_positive_admit(rps_exact, Int(py=rps) if rps_exact else 0, "rows_per_shard must be a positive int")
    var x_pin = ArrayPin()
    var x_meta = admit_family_meta(args[1], 2, "X", context, x_pin, True)
    var x = x_pin.keep
    var d = Int(py=x.shape[1])
    var nk = parallel_keys()
    rbf_features_admit(d, Int(py=py_item(fields, nk[].k_n_features_in_)))
    var rows = Int(py=x.shape[0])
    if rows == 0:
        return _rbf_transform(estimator, Python.tuple(args[4], x), context, x_meta, x_pin)
    var n_devices = shard_devices(args[2])
    var ranges = shard_ranges(rows, rps, "rows_per_shard", "")
    var state = shard_state(fields, b.frozenset(Python.list("_resident_holder")), context)
    # With more row blocks than workers the fitted state (the random weights
    # and offsets) goes to each worker once (`shard_run_stored`) and each
    # worker builds its model once for the call (`shard_stored_model`); it
    # rode with every row block, and every block rebuilt the model, before
    # 2026-11-21. With at most one block per worker the state rides with it.
    var stored = len(ranges) // 2 > n_devices
    var token = shard_store_token() if stored else String("")
    var carried = Python.tuple(shard_stored_ref(token, 0)) if stored else state
    var requests = shard_wave()
    for r in range(len(ranges) // 2):  # driver-loop: one worker request per row block
        var lo = ranges[2 * r]
        var hi = ranges[2 * r + 1]
        var shift = shard_read_shift(r, lo, n_devices)
        requests.add(PARALLEL_SHARD_WORKER_OPN_RBF_ROWS, carried, Python.tuple(shard_rows(x, lo - shift, hi - shift, context)))
    var parts = (shard_run_stored(Python.tuple(state), token, requests, args[2], _CHILD, host) if stored
                 else shard_run(requests, args[2], _CHILD, host))
    var q = Int(py=py_item(fields, nk[].k_random_weights_).shape[1])
    for r in range(len(parts)):  # driver-loop: checks one result block per row shard
        var part = parts[r]
        var part_rank = Int(py=part.ndim)
        rbf_part_admit(part_rank, Int(py=part.shape[1]) if part_rank == 2 else -1, q)
    return shard_join_rows(parts, rows, context)


# ---------------------------------------------------------------- scalers

def _scaler_kind(estimator: PythonObject) raises -> String:
    var cls = py_type(estimator)
    if cls is _t("standard_class"):
        return "standard"
    if cls is _t("minmax_class"):
        return "minmax"
    raise Error("TypeError: parallel preprocessing requires MinMaxScaler or StandardScaler")


def _scaler_stats(kind: String) raises -> PythonObject:
    if kind == "minmax":
        return Python.list("data_min_", "data_max_", "data_range_", "scale_", "min_")
    return Python.list("mean_", "var_", "scale_")


def _scaler_input(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_ScalerProtocol._input(X)`'s refusals (float32, nonempty 2-D, the
    Int32 bound), then the C-contiguous float32 Array. The finiteness scan
    is the plain owners' (each shard's fit or transform)."""
    # The one admission path in its two steps: the metadata (read once, with
    # the rank and emptiness refusals), the Scaler's own refusals on it, then
    # the borrow or the cast of the same capture (family_view and the second
    # admission of `source` deleted). A rank above 2 is now refused by the
    # metadata step ("must be at most 2-D") before the dtype test.
    var pin = ArrayPin()
    var snapshot = admit_family_input(source, -1, "input", context, pin)
    var b = shard_b()
    var listed = py_type(source) is b.list or py_type(source) is b.tuple
    var input_rank = len(snapshot.shape)
    scaler_input_admit(listed, snapshot.dtype_name, input_rank, snapshot.shape[0] if input_rank == 2 else 0,
                       snapshot.shape[1] if input_rank == 2 else 0)
    admit_family_cast(snapshot, pin, "input", context)
    return pin.keep


def _columns(data: PythonObject, lo: Int, hi: Int, context: PythonObject) raises -> PythonObject:
    """`data[:, lo:hi]` of a C-ordered (n, d) float32 Array, as a C-ordered copy."""
    var n = Int(py=data.shape[0])
    var d = Int(py=data.shape[1])
    var out = shard_array("<f4", Python.tuple(n, hi - lo), n * (hi - lo), context)
    if n > 0 and hi > lo:
        forecast_gather_cols(Int(py=classical_address(out)), Int(py=classical_address(data)), n, d, lo, hi - lo)
        _ = out  # keep the Python object alive through the native call above
        _ = data  # keep the Python object alive through the native call above
    return out


def _fit_scaler(args: PythonObject) raises -> PythonObject:
    """args (estimator, X, devices, columns_per_shard, sample_weight): whole
    columns in `columns_per_shard` blocks; each statistic is per column, so
    every shard's statistics are placed at its column offset."""
    var b = shard_b()
    var context = _ctx()
    # As _fit_forest: connect NumPy's native types in the shared context on
    # first use, or an ordinary ndarray is refused ("Sequential requires
    # native input types").
    classical_connect_numpy(context)
    var estimator = args[0]
    var kind = _scaler_kind(estimator)
    var names = _scaler_stats(kind)
    var fields = py_fields(estimator)
    var host = _member_host(fields)
    shard_identical_admit(_identical(fields), "parallel scalers require IDENTICAL numeric mode")
    scaler_weight_admit(args[4] is not None)
    var data = _scaler_input(args[1], context)
    var n = Int(py=data.shape[0])
    var d = Int(py=data.shape[1])
    var ranges = shard_ranges(d, args[3], "columns_per_shard", "")
    # `estimator.get_params()` with numeric_mode='identical'
    var init = b.list()
    var nk = parallel_keys()
    if kind == "minmax":
        init = Python.list(py_item(fields, nk[].k_feature_range), py_item(fields, nk[].k_copy), py_item(fields, nk[].k_clip), "identical")
    else:
        init = Python.list(py_item(fields, nk[].k_copy), py_item(fields, nk[].k_with_mean), py_item(fields, nk[].k_with_std), "identical")
    var n_devices = shard_devices(args[2])
    var requests = shard_wave()
    for r in range(len(ranges) // 2):  # driver-loop: one worker request per column block
        var lo = ranges[2 * r]
        var hi = ranges[2 * r + 1]
        var shift = shard_read_shift(r, lo, n_devices)
        requests.add(PARALLEL_SHARD_WORKER_OPN_SCALER_FIT, Python.tuple(kind, py_call(b.tuple, init)),
                                                  Python.tuple(_columns(data, lo - shift, hi - shift, context)))
    var parts = shard_run(requests, args[2], _CHILD, host)
    var merged = b.dict()
    for k in range(len(names)):  # small-loop(the scaler's per-column statistics)
        var name = names[k]
        if parts[0][name] is None:
            py_setitem(merged, name, PythonObject(None))
            continue
        var out = shard_array("<f4", Python.tuple(d), d, context)
        for r in range(len(ranges) // 2):  # driver-loop: places one shard's statistic block
            var lo = ranges[2 * r]
            var w = ranges[2 * r + 1] - lo
            var value = parts[r][name]
            unsafe_memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out))) + lo,
                   src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(value))),
                   count=w)
            _ = out  # keep the Python object alive through the native call above
            _ = value  # keep the Python object alive through the native call above
        py_setitem(merged, name, out)
    # `result = type(estimator)(**params); result.__dict__.update(parts[0].__dict__)`
    _ = py_call(fields.clear)
    _ = py_call(fields.update, parts[0])
    _ = py_call(fields.update, merged)
    py_setitem(fields, nk[].k_n_features_in_, d)
    py_setitem(fields, nk[].k_n_samples_seen_, n)
    return estimator


def _transform_scaler(args: PythonObject) raises -> PythonObject:
    """args (estimator, X, devices, columns_per_shard, inverse): every column
    block transformed with its columns' statistics, the blocks joined at
    their column offsets."""
    var b = shard_b()
    var context = _ctx()
    classical_connect_numpy(context)  # as _fit_scaler
    var estimator = args[0]
    var kind = _scaler_kind(estimator)
    var names = _scaler_stats(kind)
    # The fitted test is the scalers' own (`_fitted`: `scale_` and
    # `numeric_mode_` in the instance fields), read here on the fields this
    # door reads next. REMOVED: the `family_call(entry, estimator,
    # "is_fitted", ...)` crossing through the scaler binding's Python entry
    # (an argument tuple, the entry's dispatch and a `bool` call per
    # transform) and the registered `standard_call` / `minmax_call` entries.
    # NOT a member ABI call: a held member state (`member_held`) would IMPORT
    # the fitted arrays to answer one flag, and the column shards below run
    # under the driver's device tuple, which a member's one route cannot
    # carry (MEMBER-ABI-V2 "Not covered" terms).
    var fields = py_fields(estimator)
    shard_fitted_admit(scaler_fitted(fields), "scaler is not fitted")
    var host = _member_host(fields)
    var fitted_mode = py_get(fields, PythonObject("numeric_mode_"), PythonObject(None))
    var fitted_exact = fitted_mode is not None and py_type(fitted_mode) is b.str
    scaler_fit_mode_admit(fitted_exact, String(fitted_mode) if fitted_exact else String(""))
    var inverse = args[4]
    if py_type(inverse) is not b.bool:
        raise Error("ValueError: inverse must be a bool")
    var data = _scaler_input(args[1], context)
    var n = Int(py=data.shape[0])
    var d = Int(py=data.shape[1])
    var nk = parallel_keys()
    shard_features_admit(d, Int(py=py_item(fields, nk[].k_n_features_in_)))
    var ranges = shard_ranges(d, args[3], "columns_per_shard", "")
    var base = shard_state(fields, b.frozenset(), context)
    var n_devices = shard_devices(args[2])
    var requests = shard_wave()
    for r in range(len(ranges) // 2):  # driver-loop: one worker request per column block
        var lo = ranges[2 * r]
        var hi = ranges[2 * r + 1]
        var part = py_call(b.dict, base)
        py_setitem(part, nk[].k_n_features_in_, hi - lo)
        for k in range(len(names)):  # small-loop(the scaler's per-column statistics)
            var value = fields[names[k]]
            py_setitem(part, names[k], PythonObject(None) if value is None else shard_rows(value, lo, hi, context))
        var shift = shard_read_shift(r, lo, n_devices)
        requests.add(PARALLEL_SHARD_WORKER_OPN_SCALER_TRANSFORM, Python.tuple(kind, part),
                                                  Python.tuple(_columns(data, lo - shift, hi - shift, context), inverse))
    var parts = shard_run(requests, args[2], _CHILD, host)
    var out = shard_array("<f4", Python.tuple(n, d), n * d, context)
    for r in range(len(ranges) // 2):  # driver-loop: places one column block
        var lo = ranges[2 * r]
        var w = ranges[2 * r + 1] - lo
        var part = parts[r]
        var part_rows = Int(py=part.shape[0])
        scaler_block_admit(part_rows, Int(py=part.shape[1]) if part_rows == n else -1, n, w)
        forecast_scatter_cols(Int(py=classical_address(out)), Int(py=classical_address(part)), n, d, lo, w)
        _ = out  # keep the Python object alive through the native call above
        _ = part  # keep the Python object alive through the native call above
    return out


# ---------------------------------------------------------------- GaussianProcessClassifier

def _gpc_admit(estimator: PythonObject, context: PythonObject, mut host: Bool) raises -> PythonObject:
    """`parallel_gaussian_process._validate`: the class, IDENTICAL, then the
    native parameter validation on a copy (`copy.copy(estimator)._validate_params()`),
    which is returned: it carries the validated `_kernel` and
    `_max_iter_predict` while the estimator itself is untouched until a fit
    publishes."""
    if py_type(estimator) is not _t("gpc_class"):
        raise Error("TypeError: requires mojolearn.GaussianProcessClassifier")
    classical_check_layout(estimator, context)
    var fields = py_fields(estimator)
    host = _member_host(fields)
    var mode = py_get(fields, PythonObject("numeric_mode"), PythonObject(None))
    var b = shard_b()
    var own_exact = mode is not None and py_type(mode) is b.str
    gpc_mode_kind_admit(mode is None, own_exact)
    if gpc_mode_defaults(mode is None, String(mode) if own_exact else String("")):
        var nk = parallel_keys()
        mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    var mode_exact = py_type(mode) is b.str
    gpc_mode_admit(mode_exact, String(mode) if mode_exact else String(""))
    var work = shard_model(py_type(estimator), py_call(b.dict, fields))
    gpc_validate(work, context)
    return work


def _fit_gpc(args: PythonObject) raises -> PythonObject:
    """args (estimator, X, y, devices, gp binding, base binding): one binary
    Laplace fit per one-vs-rest class column (class 1 alone for two
    classes), each on a worker, stored as the plain fit stores them."""
    var b = shard_b()
    var context = _ctx()
    var estimator = args[0]
    var host = False
    var work = py_fields(_gpc_admit(estimator, context, host))
    var fields = py_fields(estimator)
    var source = admit_family(args[1], 2, "X", context).array
    var copied = not (source is args[1])
    var x = source
    var encoded = gp_encode_labels_transport(args[2], _worker_binding("_mojolearn", host), context)
    var classes = encoded.classes
    var codes = encoded.codes
    gpc_labels_admit(Int(py=codes.size), Int(py=x.shape[0]), len(classes))
    var columns = b.list()
    if len(classes) == 2:
        _ = py_call(columns.append, PythonObject(1))
    else:
        for k in range(len(classes)):  # driver-loop: over the one-vs-rest classes
            _ = py_call(columns.append, PythonObject(k))
    var nk = parallel_keys()
    var arrays = gp_kernel_arrays_transport(py_item(work, nk[].k__kernel), context)
    var carried = b.list()
    for i in range(len(arrays)):  # small-loop(the kernel's five packed arrays and words)
        _ = py_call(carried.append, arrays[i])
    var max_iter = py_item(work, nk[].k__max_iter_predict)
    var n_devices = shard_devices(args[3])
    # N452 (lane ne2-i-j): with more class columns than workers, X and the
    # codes go to each worker once (`shard_run_stored`), not once per column
    var stored = len(columns) > n_devices
    var token = shard_store_token() if stored else String("")
    var requests = shard_wave()
    for i in range(len(columns)):  # driver-loop: one worker request per class column
        if stored:
            requests.add(PARALLEL_SHARD_WORKER_OPN_GPC_CLASS_FIT, Python.tuple(carried, max_iter),
                                                      Python.tuple(shard_stored_ref(token, 0), shard_stored_ref(token, 1),
                                                                   columns[i]))
        else:
            requests.add(PARALLEL_SHARD_WORKER_OPN_GPC_CLASS_FIT, Python.tuple(carried, max_iter),
                                                      Python.tuple(x, codes, columns[i]))
    var parts = (shard_run_stored(Python.tuple(x, codes), token, requests, args[3], _CHILD, host) if stored
                 else shard_run(requests, args[3], _CHILD, host))
    var fits = GPCFitPublication(len(parts))
    for i in range(len(parts)):  # driver-loop: UNPACK one transported binary fit per class column
        var f = parts[i]
        var member = _gpc_binary_result(context, py_item(f, nk[].k_y_train_), py_item(f, nk[].k_L_), py_item(f, nk[].k_pi_), py_item(f, nk[].k_W_sr_),
            Float64(py=py_item(f, nk[].k_log_marginal_likelihood_value_)), Int(py=py_item(f, nk[].k_n_iter_)), Int(py=py_item(f, nk[].k_nb_)))
        fits.append(member)
    # published only now that every class fitted: `_fresh(estimator)` carried
    # the validated kernel, max_iter_predict and IDENTICAL
    py_setitem(work, nk[].k_numeric_mode, "identical")
    py_setitem(work, nk[].k_input_copied_, copied)
    _gpc_set_fitted(work, x, classes, fits, context)
    _ = py_call(fields.update, work)
    return estimator


def _predict_gpc(args: PythonObject) raises -> PythonObject:
    """args (estimator, X, devices, method, gp binding, base binding): one
    latent pass per fitted class on a worker; two classes take the binding's
    decoded codes or [1 - p, p] rows from the one fit, more take the
    one-vs-rest combination of the class probability columns."""
    var b = shard_b()
    var context = _ctx()
    var estimator = args[0]
    var host = False
    var work = py_fields(_gpc_admit(estimator, context, host))
    var method = args[3]
    var method_exact = py_type(method) is b.str
    gpc_method_admit(method_exact, String(method) if method_exact else String(""))
    var fields = py_fields(estimator)
    shard_fitted_admit(py_contains(fields, PythonObject("estimators_")), "GaussianProcessClassifier: call fit() or load() first")
    var q = admit_family(args[1], 2, "X", context).array
    var nk = parallel_keys()
    gpc_features_admit(Int(py=q.shape[1]), Int(py=py_item(fields, nk[].k_n_features_in_)))
    var proba_method = String(method) == "predict_proba"
    var n_classes = Int(py=py_item(fields, nk[].k_n_classes_))
    var want_proba = proba_method or n_classes > 2
    var out_kind = 0
    if n_classes == 2:
        out_kind = 2 if proba_method else 1
    var arrays = gp_kernel_arrays_transport(py_item(work, nk[].k__kernel), context)
    var carried = b.list()
    for i in range(len(arrays)):  # small-loop(the kernel's five packed arrays and words)
        _ = py_call(carried.append, arrays[i])
    var fits = py_item(fields, nk[].k_estimators_)
    var n_devices = shard_devices(args[2])
    # N452 (lane ne2-i-j): with more fitted classes than workers, X_train_
    # and the query rows go to each worker once (`shard_run_stored`)
    var stored = len(fits) > n_devices
    var token = shard_store_token() if stored else String("")
    var requests = shard_wave()
    for i in range(len(fits)):  # driver-loop: one worker request per fitted class
        var fit_state = shard_state(py_fields(fits[i]), b.frozenset(), context)
        if stored:
            requests.add(PARALLEL_SHARD_WORKER_OPN_GPC_CLASS_PREDICT, Python.tuple(carried, shard_stored_ref(token, 0), fit_state),
                                                      Python.tuple(shard_stored_ref(token, 1), want_proba, out_kind))
        else:
            requests.add(PARALLEL_SHARD_WORKER_OPN_GPC_CLASS_PREDICT, Python.tuple(carried, py_item(fields, nk[].k_X_train_), fit_state),
                                                      Python.tuple(q, want_proba, out_kind))
    var columns = (shard_run_stored(Python.tuple(py_item(fields, nk[].k_X_train_), q), token, requests, args[2], _CHILD, host) if stored
                   else shard_run(requests, args[2], _CHILD, host))
    var n_star = Int(py=q.shape[0])
    for i in range(len(columns)):  # driver-loop: checks one result column per class
        gpc_rows_admit(Int(py=columns[i].shape[0]), n_star)
    if n_classes == 2:
        if proba_method:
            return columns[0]
        return gp_decode_labels(py_item(fields, nk[].k_classes_), columns[0], _worker_binding("_mojolearn", host), context)
    var retained_columns = List[PythonObject]()
    for i in range(len(columns)):  # driver-loop: unpack one returned worker column owner per class
        retained_columns.append(columns[i])
    var combined = _gpc_ovr_combine(_worker_binding("_mojolearn_gp", host), retained_columns, n_star, context, proba_method)
    if proba_method:
        return combined.probabilities
    # `codes32.astype("<i8")`: the exact widening
    var codes32 = NativeLabelCodes(0)
    swap(codes32, combined.codes)
    var codes = List[Int64](length=n_star, fill=0)
    _ = cast_elements(codes32.address(), HP_I32, Int(codes.unsafe_ptr()), HP_I64, n_star)
    _ = codes32  # keep the Python object alive through the native call above
    _ = codes  # keep the Python object alive through the native call above
    return gp_decode_labels(py_item(fields, nk[].k_classes_), codes, _worker_binding("_mojolearn", host), context)


# ---------------------------------------------------------------- forests

def _fit_forest(args: PythonObject) raises -> PythonObject:
    """args (estimator, X, y, devices, trees_per_shard, the native
    `msel_rebase_offsets_i32` entry the shell resolved): whole global tree-ID
    ranges, each fitted by the one typed forest fit core from its first
    global tree ID in a worker; the trees joined in ID order, their node
    offsets rebased natively.

    Lane forest-typed-fit (2026-11-12): the driver runs the forest door's
    UNPACK once (`forest_shard_prepare`: refresh, labels or targets, class
    weights, the typed request words); a worker receives the words and the
    arrays and calls `forest_shard_fit` (the typed core `forest_fit_core`,
    xtrees/trees_forest_run.mojo), never building an estimator; the driver
    stores the joined model with the door's PACK (`forest_shard_publish`).
    A tree is a pure function of its global ID and the request words are the
    plain fit's with the range's tree count, so the joined forest is the
    one-device forest."""
    var b = shard_b()
    var context = _ctx()
    # A fresh caller may reach this driver before another operation has
    # connected NumPy's native array/scalar types in the shared context.
    # Admit ordinary ndarray inputs on first use through the same native
    # metadata connection as the other public estimator owners.
    classical_connect_numpy(context)
    var estimator = args[0]
    var cls = py_type(estimator)
    var classes = _t("forest_classes")
    var admitted = False
    for i in range(len(classes)):  # driver-loop: exact connected forest classes
        admitted = admitted or cls is classes[i]
    if not admitted:
        raise Error("TypeError: fit_forest requires a mojolearn RandomForest or ExtraTrees estimator")
    classical_check_layout(estimator, context)
    var host = _member_host(py_fields(estimator))
    var tps = args[4]
    var tps_exact = py_type(tps) is b.int
    shard_positive_admit(tps_exact, Int(py=tps) if tps_exact else 0, "trees_per_shard must be a positive integer")
    var trees_context = _t("trees_context")
    var x_admitted = admit_family(args[1], 2, "X", context)
    var x = x_admitted.array
    var plan = forest_shard_prepare(estimator, x_admitted.rows, x_admitted.cols, args[2], trees_context)
    var nk = parallel_keys()
    var count = Int(py=py_item(plan.fields, nk[].k_n_estimators))
    forest_count_admit(count)
    var ranges = shard_ranges(count, tps, "trees_per_shard", "")
    var n_devices = shard_devices(args[3])
    # N452 (lane ne2-i-j): with more tree ranges than workers, X, y and the
    # weights go to each worker once (`shard_run_stored`), not once per range
    var stored = len(ranges) // 2 > n_devices
    var token = shard_store_token() if stored else String("")
    var weighted = plan.weights is not None
    var payload = Python.tuple(x, plan.y, plan.weights) if weighted else Python.tuple(x, plan.y)
    var requests = shard_wave()
    for r in range(len(ranges) // 2):  # driver-loop: one worker request per tree range
        var lo = ranges[2 * r]
        var hi = ranges[2 * r + 1]
        var shift = shard_read_shift(r, lo, n_devices)
        if stored:
            var w_ref = shard_stored_ref(token, 2) if weighted else PythonObject(None)
            requests.add(PARALLEL_SHARD_WORKER_OPN_FOREST_FIT, plan.state,
                                                      Python.tuple(shard_stored_ref(token, 0), shard_stored_ref(token, 1),
                                                                   w_ref, lo - shift, hi - lo))
        else:
            requests.add(PARALLEL_SHARD_WORKER_OPN_FOREST_FIT, plan.state,
                                                      Python.tuple(x, plan.y, plan.weights, lo - shift, hi - lo))
    var parts = (shard_run_stored(payload, token, requests, args[3], _CHILD, host) if stored
                 else shard_run(requests, args[3], _CHILD, host))
    var merged = b.dict()
    # node offsets: one rebased table over every shard's trees
    var offsets = b.list()
    for r in range(len(parts)):  # driver-loop: one offset table per tree range
        _ = py_call(offsets.append, py_item(parts[r], nk[].k__offsets))
    py_setitem(merged, nk[].k__offsets, _rebase_offsets(offsets, _table_key("base", host), context))
    var node_names = Python.list("_colid", "_quesval", "_left_child", "_leaves")
    for k in range(len(node_names)):  # small-loop(the four flat node arrays)
        var name = String(node_names[k])
        var total = 0
        for r in range(len(parts)):  # driver-loop: one node array per tree range
            total += Int(py=parts[r][name].shape[0])
        py_setitem(merged, name, shard_concat(parts, name, total, context))
    forest_shard_publish(plan, merged, parts, count, trees_context)
    var fields = plan.fields
    if plan.classifier:
        # The plain fit's completed-classifier metadata: public `classes_`
        # labels, and `feature_names_in_` from this caller's X.
        classifier_after_fit(fields, args[1], trees_context)
    else:
        py_setitem(fields, nk[].k__mojolearn_fitted, True)
    _ = x_admitted^
    return estimator


def _rebase_offsets(arrays: PythonObject, key: EntryRef, context: PythonObject) raises -> PythonObject:
    """`parallel_ensemble._merge_offsets`: the shards' int32 node-offset
    tables joined, then rebased by the base image's `msel_rebase_offsets_i32`
    (its parallel table entry, core/parallel_entry_table.mojo) into one table
    of 1 + sum(sizes) - len(shards) entries."""
    var b = shard_b()
    var parts = b.list()
    var n_raw = 0
    for p in range(len(arrays)):  # driver-loop: one offset table per tree range
        var a = arrays[p]
        forest_offsets_admit(shard_dtype(a))
        _ = py_call(parts.append, a)
        n_raw += Int(py=a.shape[0])
    var raw = shard_join_rows(parts, n_raw, context)
    var lens = shard_array("<i8", Python.tuple(len(parts)), len(parts), context)
    var lp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(lens)))
    for p in range(len(parts)):  # driver-loop: one size word per tree range
        lp[p] = Int64(Int(py=parts[p].shape[0]))
    _ = lens  # keep the Python object alive through the native call above
    var n_out = 1 + n_raw - len(parts)
    var out = shard_array("<i4", Python.tuple(n_out), n_out, context)
    var raw_addr = Int(py=classical_address(raw))
    var lens_addr = Int(py=classical_address(lens))
    var out_addr = Int(py=classical_address(out))
    var n_parts = len(parts)
    with GILReleased(Python()):
        _ = parallel_rebase_offsets_i32(key, raw_addr, n_raw, lens_addr, n_parts, n_out, out_addr)
    _ = raw  # keep the Python object alive through the native call above
    _ = lens  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    return out


# ---------------------------------------------------------------- persistent pools
# ReferenceShardedNeighbors, ParallelQueries and DistributedIVFIndex keep one
# native local pool for their lifetime (the old `DevicePool` they held): the
# workers start at the first request and keep their snapshots until close.
# Their per-shard work calls the same native search/vote kernels directly;
# partition, scheduling, resident snapshots, merge and join are native.

struct ParallelIvfOwner(Movable, Writable):
    """One distributed IVF index's native owner, stored once in the instance
    dictionary as `_native`. The words a search decides on are typed: closed,
    the feature and neighbor counts, the metric code, and per stored shard
    the row count and the address of its global-id map. The Python objects
    are pins the door built itself at construction: the process pool, the
    list that keeps the map arrays alive, and the store token as the one
    str every search request carries. `_closed`, `devices`, `n_neighbors`,
    `n_features_in_`, `metric_code_` and `n_candidates_` on the instance are
    mirrors for callers; this door writes them and never reads them."""
    var owner_id: Int
    var closed: Bool
    var n_features: Int
    var n_neighbors: Int
    var metric: Int
    var map_rows: List[Int]
    var map_addrs: List[Int]
    var token: PythonObject
    var pool: PythonObject
    var maps: PythonObject

    def __init__(out self, owner_id: Int, n_features: Int, n_neighbors: Int, metric: Int,
                 token: PythonObject, pool: PythonObject, maps: PythonObject):
        self.owner_id = owner_id
        self.closed = False
        self.n_features = n_features
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.map_rows = List[Int]()
        self.map_addrs = List[Int]()
        self.token = token
        self.pool = pool
        self.maps = maps

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ParallelIvfOwner(closed=", self.closed, ", shards=", len(self.map_rows), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


struct ParallelNeighborOwner(Movable, Writable):
    """The native owner of one `ReferenceShardedNeighbors` or
    `ParallelQueries` holder, stored once in the instance dictionary as
    `_native`: the process pool it keeps for its lifetime (a pin) and the
    counter of the query tokens it has issued. `_pool` on the instance is a
    mirror; this door writes it and never reads it. `estimator` and the
    rows-per-shard attributes stay public settings, admitted at each call."""
    var owner_id: Int
    var tokens: Int
    var pool: PythonObject

    def __init__(out self, owner_id: Int, pool: PythonObject):
        self.owner_id = owner_id
        self.tokens = 0
        self.pool = pool

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ParallelNeighborOwner(tokens=", self.tokens, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def _ivf_record_or_none(obj: PythonObject) raises -> PythonObject:
    """The index's native owner record (None before `from_index` stored
    it): one exact-type check and one instance-dictionary read per call."""
    var record = py_get(_driver_fields(obj, "distributed_ivf_class"), parallel_keys()[].k__native)
    if record is None:
        return record
    var owner_id = Int(py=py_call(shard_b().id, obj))
    try:
        if record.downcast_value_ptr[ParallelIvfOwner]()[].owner_id != owner_id:
            raise Error("foreign")
    except:
        raise Error("TypeError: distributed IVF storage does not belong to this index")
    return record


def _ivf_record(obj: PythonObject) raises -> PythonObject:
    var record = _ivf_record_or_none(obj)
    if record is None:
        raise Error("RuntimeError: DistributedIVFIndex is not initialized; use DistributedIVFIndex.from_index(index, devices=...)")
    return record


def _holder_record_or_none(obj: PythonObject) raises -> PythonObject:
    """A neighbor holder's native owner record (None when its construction
    was refused before the pool was opened). The caller has checked the
    holder's exact class."""
    var record = py_get(py_fields(obj), parallel_keys()[].k__native)
    if record is None:
        return record
    var owner_id = Int(py=py_call(shard_b().id, obj))
    try:
        if record.downcast_value_ptr[ParallelNeighborOwner]()[].owner_id != owner_id:
            raise Error("foreign")
    except:
        raise Error("TypeError: parallel neighbor storage does not belong to this holder")
    return record


def _holder_record(obj: PythonObject) raises -> PythonObject:
    var record = _holder_record_or_none(obj)
    if record is None:
        raise Error("RuntimeError: the parallel neighbor holder is not initialized")
    return record


def _pool_open(devices: PythonObject, host: Bool) raises -> PythonObject:
    _ = shard_devices(devices)
    return shard_open(devices, _CHILD, host)


def _map(pool: PythonObject, requests: ShardWave) raises -> PythonObject:
    var result = shard_map(pool, requests)
    shard_batch_admit(len(result), requests.count())
    return result


def _driver_fields(obj: PythonObject, class_key: String) raises -> PythonObject:
    if py_type(obj) is not _t(class_key):
        raise Error("TypeError: parallel driver requires its exact connected class")
    return py_fields(obj)


def _holder_init(args: PythonObject, reference: Bool) raises -> PythonObject:
    var b = shard_b()
    var own = _driver_fields(args[0], "reference_driver_class" if reference else "query_driver_class")
    var estimator = args[1]
    var cls = py_type(estimator)
    if reference:
        if cls is not _t("nearest_class") and cls is not _t("knc_class") and cls is not _t("knr_class"):
            raise Error("TypeError: reference shards require a brute-force KNN estimator")
    elif len(py_get(_t("query_methods"), cls, Python.tuple())) == 0:
        raise Error("TypeError: parallel queries require an exact neighbors or KernelDensity estimator")
    var fields = py_fields(estimator)
    var host = _member_host(fields)
    var nk = parallel_keys()
    if reference:
        var algorithm = py_get(fields, nk[].k_algorithm)
        var algorithm_exact = py_type(algorithm) is b.str
        reference_algorithm_admit(algorithm_exact, String(algorithm) if algorithm_exact else String(""))
    shard_identical_admit(_identical(fields), "parallel neighbors require IDENTICAL numeric mode")
    if reference and args[3] is not None:
        _ = shard_ranges(0, args[3], "reference_rows_per_shard", "")
    _ = shard_ranges(0, args[4] if reference else args[3], "query_rows_per_shard" if reference else "rows_per_shard", "")
    py_setitem(own, nk[].k_estimator, estimator)
    if reference:
        py_setitem(own, nk[].k_reference_rows_per_shard, args[3])
        py_setitem(own, nk[].k_query_rows_per_shard, args[4])
    else:
        py_setitem(own, nk[].k_rows_per_shard, args[3])
    var pool = _pool_open(args[2], host)
    # PACK, once per holder: the native owner; `_pool` is a mirror for callers.
    py_setitem(own, nk[].k__native, PythonObject(alloc=ParallelNeighborOwner(Int(py=py_call(b.id, args[0])), pool)))
    py_setitem(own, nk[].k__pool, pool)
    py_setitem(own, nk[].k_last_shards_, Python.list())
    return PythonObject(None)


def _join(parts: PythonObject, ragged: Bool, context: PythonObject) raises -> PythonObject:
    """`parallel_neighbors._join`: tuples joined component by component,
    lists either chained (ragged) or joined position by position, Arrays
    joined along their first axis in shard order."""
    var b = shard_b()
    var first = parts[0]
    if py_type(first) is b.tuple:
        var out = b.list()
        for i in range(len(first)):  # small-loop(the result tuple's components)
            var column = b.list()
            for p in range(len(parts)):  # driver-loop: one component per query shard
                _ = py_call(column.append, parts[p][i])
            _ = py_call(out.append, _join(column, ragged, context))
        return py_call(b.tuple, out)
    if py_type(first) is b.list:
        var out = b.list()
        if ragged or (len(first) > 0 and (py_type(first[0]) is b.str or py_type(first[0]) is b.bool)):
            for p in range(len(parts)):  # driver-loop: one ragged row list per query shard
                _ = py_call(out.extend, parts[p])
            return out
        for i in range(len(first)):  # driver-loop: over the per-output result columns
            var column = b.list()
            for p in range(len(parts)):  # driver-loop: one column per query shard
                _ = py_call(column.append, parts[p][i])
            _ = py_call(out.append, _join(column, False, context))
        return out
    var total = 0
    for p in range(len(parts)):  # driver-loop: one block per query shard
        total += Int(py=parts[p].shape[0])
    return shard_join_rows(parts, total, context)


def _reference_query(args: PythonObject) raises -> PythonObject:
    """args (sharded, X, method, n_neighbors, return_distance):
    `ReferenceShardedNeighbors._query`. Reference rows in blocks of
    `reference_rows_per_shard` (or an even split over the devices), queried
    in waves of len(devices) per block of `query_rows_per_shard` query rows;
    every block's candidates merged per query row by the native
    `shard_topk_merge_f32`; predictions voted over the merged neighbors on
    the first worker."""
    var b = shard_b()
    var context = _ctx()
    var holder = _driver_fields(args[0], "reference_driver_class")
    var nk = parallel_keys()
    var model = py_item(holder, nk[].k_estimator)
    var owner_record = _holder_record(args[0])
    var pool = owner_record.downcast_value_ptr[ParallelNeighborOwner]()[].pool
    var method = String(args[2])
    var cls = py_type(model)
    if cls is not _t("nearest_class") and cls is not _t("knc_class") and cls is not _t("knr_class"):
        raise Error("TypeError: reference shards require an exact KNN estimator")
    var fields = py_fields(model)
    var mode = py_get(fields, PythonObject("numeric_mode"), PythonObject(None))
    var algorithm = py_get(fields, PythonObject("algorithm"), PythonObject(None))
    var algorithm_exact = py_type(algorithm) is b.str
    reference_knn_admit(_identical(fields), algorithm_exact, String(algorithm) if algorithm_exact else String(""))
    _ = mode
    var index = py_get(fields, PythonObject("_index"), PythonObject(None))
    if index is None:
        raise Error("ValueError: fit the estimator before querying")
    if method != "kneighbors":
        if cls is not _t("knc_class") and cls is not _t("knr_class"):
            raise Error("TypeError: prediction requires a classifier or regressor")
        if method == "predict_proba" and cls is not _t("knc_class"):
            raise Error("TypeError: predict_proba requires a classifier")
        if py_get(fields, PythonObject("_y_cols"), PythonObject(None)) is None:
            raise Error("ValueError: fit the estimator before prediction")
    var data = admit_family(args[1], 2, "X", context).array
    var d = Int(py=py_item(fields, nk[].k_n_features_in_))
    shard_features_admit(Int(py=data.shape[1]), d)
    var n = Int(py=py_item(fields, nk[].k_n_samples_fit_))
    var k_obj = py_item(fields, nk[].k_n_neighbors) if args[3] is None else args[3]
    var k_exact = py_type(k_obj) is b.int
    reference_k_admit(k_exact, Int(py=k_obj) if k_exact else 0, n)
    var k = Int(py=k_obj)
    if py_type(args[4]) is not b.bool:
        raise Error("TypeError: return_distance must be a bool")
    var cell_limit = reference_cell_limit(n, d)
    var devices = process_pool_devices(pool)
    var n_devices = len(devices)
    var width_obj = py_item(holder, nk[].k_reference_rows_per_shard)
    var width_given = width_obj is not None
    var width_exact = width_given and py_type(width_obj) is b.int
    if width_given:
        shard_width_admit(width_exact, Int(py=width_obj) if width_exact else 0, "reference_rows_per_shard")
    var width = reference_width(cell_limit, n, n_devices, width_given, Int(py=width_obj) if width_given else 0)
    var ranges = shard_ranges(n, PythonObject(width), "reference_rows_per_shard", "")
    var params = b.dict()
    py_setitem(params, nk[].k_n_neighbors, k)
    py_setitem(params, nk[].k_query_tile, py_item(fields, nk[].k_query_tile))
    py_setitem(params, nk[].k_metric, py_item(fields, nk[].k_metric))
    py_setitem(params, nk[].k_algorithm, "brute")
    py_setitem(params, nk[].k_p, py_item(fields, nk[].k_p))
    py_setitem(params, nk[].k_numeric_mode, "identical")
    var vote_state = PythonObject(None)
    if method != "kneighbors":
        vote_state = shard_state(fields, b.frozenset(Python.list("_index", "_resident", "_resident_holder")), context)
        py_setitem(vote_state, nk[].k__index, PythonObject(None))
        py_setitem(vote_state, nk[].k_numeric_mode, "identical")
    var rows = Int(py=data.shape[0])
    _ = shard_ranges(0, py_item(holder, nk[].k_query_rows_per_shard), "query_rows_per_shard", "")
    var step = Int(py=py_item(holder, nk[].k_query_rows_per_shard))
    var n_ranges = len(ranges) // 2
    # Each reference block is cut from the fitted rows once for the whole
    # query, not once per query row block (the same bytes every time).
    var blocks = b.list()
    for r in range(n_ranges):  # driver-loop: one copy per reference block
        var start = ranges[2 * r]
        var shift = shard_read_shift(r, start, n_devices)
        _ = py_call(blocks.append, shard_rows(index, start - shift, ranges[2 * r + 1] - shift, context))
    # pass 3 (rank 108): each reference block is sent to and fitted on its
    # worker (block r on worker r % len(devices), the map's placement) ONCE
    # per call under a token, then every query row block sends only its
    # query rows. The old request carried the block and the worker fitted
    # (uploaded) and released a fresh model for every query row block. The
    # tokens are released when the call ends, failed or not.
    var tokens = b.list()
    var token_base = (String("rs-") + String(external_call["getpid", Int32]()) + "-"
                      + String(Int(py=py_call(b.id, blocks))) + "-")
    for r in range(n_ranges):  # driver-loop: one token per reference block
        _ = py_call(tokens.append, PythonObject(token_base + String(r)))
    var outputs = b.list()
    var diagnostics = b.list()
    var first = 0
    # The base entry, resolved at the first query row block and reused by
    # every later block (perf-audit-2: was one key read and table resolution
    # per query row block). Still resolved only when a block runs.
    var base_ref = Optional[EntryRef]()
    try:
        var store_wave = 0
        while store_wave < n_ranges:  # driver-loop: over waves of len(devices) reference blocks
            var stores = shard_wave()
            for r in range(store_wave, min(store_wave + n_devices, n_ranges)):  # driver-loop: one store per reference block
                stores.add(PARALLEL_SHARD_WORKER_OPN_REFERENCE_STORE, PythonObject(None), Python.tuple(tokens[r], params, blocks[r]))
            _ = _map(pool, stores)
            store_wave += n_devices
        while first < rows:  # driver-loop: over the query row blocks
            var end = min(rows, first + step)
            var query = shard_rows(data, first, end, context)
            var parts = b.list()
            var wave = 0
            while wave < n_ranges:  # driver-loop: over waves of len(devices) reference blocks
                var requests = shard_wave()
                for r in range(wave, min(wave + n_devices, n_ranges)):  # driver-loop: one request per reference block
                    var start = ranges[2 * r]
                    var stop = ranges[2 * r + 1]
                    requests.add(PARALLEL_SHARD_WORKER_OPN_REFERENCE_QUERY, PythonObject(None), Python.tuple(
                        tokens[r], query, min(k, stop - start)))
                _ = py_call(parts.extend, _map(pool, requests))
                wave += n_devices

            if not base_ref:
                base_ref = _table_key("base", _pool_host(pool))
            var merged = _merge_candidates(parts, ranges, end - first, k, base_ref.value(), context)
            var output = PythonObject(None)
            if method == "kneighbors":
                var wide = shard_array("<i8", Python.tuple(end - first, k), (end - first) * k, context)
                _ = cast_elements(Int(py=classical_address(merged[1])), HP_U32, Int(py=classical_address(wide)), HP_I64,
                                  (end - first) * k)
                _ = wide  # keep the Python object alive through the native call above
                output = Python.tuple(merged[0], wide) if Bool(py=args[4]) else wide
            else:
                var vote = shard_wave()
                vote.add(PARALLEL_SHARD_WORKER_OPN_REFERENCE_VOTE, PythonObject(None),
                         Python.tuple(cls.__name__, vote_state, merged[0], merged[1], method))
                output = _map(pool, vote)[0]
            _ = py_call(outputs.append, output)
            for r in range(n_ranges):  # driver-loop: one diagnostic record per reference block
                var record = b.dict()
                py_setitem(record, nk[].k_query_start, first)
                py_setitem(record, nk[].k_query_end, end)
                py_setitem(record, nk[].k_reference_start, ranges[2 * r])
                py_setitem(record, nk[].k_reference_end, ranges[2 * r + 1])
                py_setitem(record, nk[].k_reference_bytes, (ranges[2 * r + 1] - ranges[2 * r]) * d * 4)
                py_setitem(record, nk[].k_device, PythonObject(devices[r % n_devices]))
                _ = py_call(diagnostics.append, record)
            first = end
    finally:
        _release_reference(pool, tokens, n_devices)
    var result = _join(outputs, False, context)
    py_setitem(holder, nk[].k_last_shards_, diagnostics)
    var kind = PythonObject("regressor") if cls is _t("knr_class") else PythonObject(None)
    return parallel_public_result(result, method, kind, context)


def _merge_candidates(parts: PythonObject, ranges: List[Int], n_queries: Int, k: Int, key: EntryRef,
                      context: PythonObject) raises -> PythonObject:
    """`parallel_neighbors_reference._merge`: (float32 distances, uint32
    global indices), each (n_queries, k), from the reference blocks'
    candidates by the base image's `shard_topk_merge_f32` (its parallel table
    entry, core/parallel_entry_table.mojo)."""
    var b = shard_b()
    var distances = shard_array("<f4", Python.tuple(n_queries, k), n_queries * k, context)
    var indices = shard_array("<u4", Python.tuple(n_queries, k), n_queries * k, context)
    if n_queries == 0:
        return Python.tuple(distances, indices)
    var count = len(parts)
    var table = shard_array("<i8", Python.tuple(5 * count), 5 * count, context)
    var tp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(table)))
    var keep = b.list()
    for p in range(count):  # driver-loop: one candidate table row per reference block
        var dist = admit_family(parts[p][0], 2, "distance shard", context).array
        var ids = parts[p][1]
        var ids_dtype = shard_dtype(ids)
        if ids_dtype != "<i8":
            reference_ids_admit(ids_dtype)
            var wide = shard_array("<i8", py_call(b.tuple, ids.shape), Int(py=ids.size), context)
            var code = HP_I32 if ids_dtype == "<i4" else HP_U32
            _ = cast_elements(Int(py=classical_address(ids)), code, Int(py=classical_address(wide)), HP_I64,
                              Int(py=ids.size))
            _ = wide  # keep the Python object alive through the native call above
            ids = wide
        _ = py_call(keep.append, Python.tuple(dist, ids))
        tp[5 * p] = Int64(Int(py=classical_address(dist)))
        tp[5 * p + 1] = Int64(Int(py=classical_address(ids)))
        tp[5 * p + 2] = Int64(ranges[2 * p])
        tp[5 * p + 3] = Int64(ranges[2 * p + 1])
        tp[5 * p + 4] = Int64(Int(py=dist.shape[1]))
        _ = table  # keep the Python object alive through the native call above
    var table_addr = Int(py=classical_address(table))
    var dist_addr = Int(py=classical_address(distances))
    var idx_addr = Int(py=classical_address(indices))
    var rc = 0
    with GILReleased(Python()):
        rc = parallel_topk_merge_f32(key, table_addr, count, n_queries, k, dist_addr, idx_addr)
    _ = distances  # keep the Python object alive through the native call above
    _ = indices  # keep the Python object alive through the native call above
    _ = keep  # keep the Python objects alive through the native call above
    _ = table  # keep the Python object alive through the native call above
    reference_merge_admit(rc)
    return Python.tuple(distances, indices)


def _release_reference(pool: PythonObject, tokens: PythonObject, workers: Int) raises:
    """Best-effort release of a call's stored reference blocks (token r on
    worker r % workers, where it was stored); a failed release closes the
    pool, as `_release_query_state` does."""
    var nk = parallel_keys()
    if process_pool_closed(pool):
        return
    var b = shard_b()
    try:
        var wave = 0
        while wave < len(tokens):  # driver-loop: over waves of len(devices) stored blocks
            var releases = shard_wave()
            for r in range(wave, min(wave + workers, len(tokens))):  # driver-loop: one release per stored block
                releases.add(PARALLEL_SHARD_WORKER_OPN_RELEASE, PythonObject(None), Python.tuple(tokens[r]))
            _ = _map(pool, releases)
            wave += workers
    except:
        process_pool_close(pool)


def _pool_pids(pool: PythonObject) raises -> PythonObject:
    # PACK of the record's process ids (the prepared-registry slot keeps them
    # as a Python list beside the pool).
    var b = shard_b()
    var pids = b.list()
    var words = process_pool_pids(pool)
    for i in range(len(words)):  # driver-loop: over the pool's worker processes
        _ = py_call(pids.append, PythonObject(words[i]))
    return pids


def _pool_pids_same(recorded: PythonObject, pool: PythonObject) raises -> Bool:
    var workers = process_pool_pids(pool)
    if len(workers) != len(recorded) or len(workers) == 0:
        return False
    for i in range(len(workers)):  # driver-loop: over the pool's worker processes
        if workers[i] != Int(py=recorded[i]):
            return False
    return True


def _release_query_state(pool: PythonObject, token: String, name: PythonObject, workers: Int) raises:
    # Best-effort release of a stored query snapshot on every worker; a
    # failed release closes the pool, exactly as the per-query release did.
    var nk = parallel_keys()
    if process_pool_closed(pool):
        return
    var b = shard_b()
    var releases = shard_wave()
    for i in range(workers):  # driver-loop: one release request per worker
        releases.add(PARALLEL_SHARD_WORKER_OPN_QUERY_STATE, PythonObject(None), Python.tuple(token, name, PythonObject(None)))
    try:
        _ = _map(pool, releases)
    except:
        process_pool_close(pool)


# Prepared-registry slots of a ParallelQueries driver: its resident worker
# snapshot claim (token, pool, worker ids, state key, name) and its token counter.
comptime _QUERY_RESIDENT_SLOT = "parallel_query_resident_v1"


def _freeze_carried(value: PythonObject, context: PythonObject, depth: Int) raises:
    """Mark every native Array inside a carried worker state read-only
    (`_readonly`, the flag the Array's write paths and its NumPy interface
    honour). The arrays are the estimator's own fitted snapshots."""
    var b = shard_b()
    if depth > 8:
        return
    var cls = py_type(value)
    var nk = parallel_keys()
    if cls is py_item(context, nk[].k_array_class):
        py_setattr(value, "_readonly", PythonObject(True))
    elif cls is b.tuple or cls is b.list:
        for i in range(len(value)):  # driver-loop: over a carried field's entries
            _freeze_carried(value[i], context, depth + 1)
    elif cls is b.dict:
        var values = py_call(b.list, py_call(value.values))
        for i in range(len(values)):  # driver-loop: over a carried field's entries
            _freeze_carried(values[i], context, depth + 1)


def _parallel_queries(args: PythonObject) raises -> PythonObject:
    """args (queries object, X, method, kwargs): `ParallelQueries.query`.
    Every worker stores one snapshot of the estimator under the holder's
    token, kept resident until the fitted state or the pool changes (the
    pool's close releases it); whole query rows in `rows_per_shard` blocks
    are answered by the plain query method (block i on worker
    i % len(devices)); the blocks joined in input order."""
    var b = shard_b()
    var context = _ctx()
    var holder = _driver_fields(args[0], "query_driver_class")
    var nk = parallel_keys()
    var model = py_item(holder, nk[].k_estimator)
    var owner_record = _holder_record(args[0])
    var pool = owner_record.downcast_value_ptr[ParallelNeighborOwner]()[].pool
    var methods = py_get(_t("query_methods"), py_type(model), Python.tuple())
    if len(methods) == 0:
        raise Error("TypeError: parallel queries require an exact neighbors or KernelDensity estimator")
    var fields = py_fields(model)
    var method = methods[0] if args[2] is None else args[2]
    if py_type(method) is not b.str:
        raise Error("TypeError: query method must be an ordinary string")
    if not Bool(py=py_call(py_attr(methods, "__contains__"), method)):
        raise Error("ValueError: unsupported parallel query method: " + String(py=py_call(b.str, method)))
    shard_identical_admit(_identical(fields), "parallel queries require IDENTICAL numeric mode")
    # `getattr(estimator, '_index', getattr(estimator, '_x', None))`
    var index = PythonObject(None)
    if py_contains(fields, PythonObject("_index")):
        index = py_item(fields, nk[].k__index)
    else:
        index = py_get(fields, PythonObject("_x"), PythonObject(None))
    if index is None:
        raise Error("ValueError: fit the estimator before querying")
    var source = args[1]
    if source is None and String(method) == "radius_neighbors":
        source = index
    var data = admit_family(source, 2, "X", context).array
    shard_features_admit(Int(py=data.shape[1]), Int(py=py_item(fields, nk[].k_n_features_in_)))
    var n = Int(py=data.shape[0])
    var ranges = shard_ranges(n, py_item(holder, nk[].k_rows_per_shard), "rows_per_shard", "")
    if len(ranges) == 0:
        ranges.append(0)
        ranges.append(0)
    var state = shard_state(fields, b.frozenset(Python.list("_resident", "_resident_holder")), context)
    # Lanes a7b / a8b keep a borrowed float32 C-order X as the fitted rows
    # (`_index` is then the caller's NumPy array), which the worker codec does
    # not carry: the workers got no `_index` and refused "call fit before
    # kneighbors". Send a native copy of the rows; the key below holds the
    # estimator's own object, so the copy is made only when the state is sent.
    var index_object = PythonObject(None)
    if index is not None and py_contains(fields, PythonObject("_index")) and not py_contains(state, PythonObject("_index")):
        index_object = index
    var devices = process_pool_devices(pool)
    var workers = len(devices)
    var name = py_type(model).__name__
    # Resident fitted state: the workers keep the snapshot stored under the
    # holder's token across queries. It is sent again only when the
    # estimator, its fitted state (identity of every carried value, nested
    # containers included) or the pool's worker processes changed. The old
    # code stored and released the full fitted index on every worker for
    # every query call.
    # The key holds the carried field NAMES and VALUES (the estimator's own
    # objects), never the state dict itself nor the "identical" str set
    # below: both are new objects on every call, so a key holding them could
    # never match and every query re-sent the whole index (pass 3, rank 317).
    var mods = NativeModules()
    var key = b.list()
    _ = py_call(key.append, model)
    _ = py_call(key.append, name)
    prepared_key_fields(mods, key, state, List[String]())
    if index_object is not None:
        _ = py_call(key.append, "_index")
        _ = py_call(key.append, index_object)
    py_setitem(state, nk[].k_numeric_mode, "identical")
    # The resident claim lives in the native prepared registry, weakly keyed
    # by the driver object (outside its `__dict__`; never pickled or copied).
    # The token counter is a word of the holder's native owner.
    var driver = args[0]
    var resident = prepared_get(mods, _QUERY_RESIDENT_SLOT, driver)
    var token = String()
    if (resident is not None and resident[1] is pool and not process_pool_closed(pool)
            and _pool_pids_same(resident[2], pool) and prepared_key_same(resident[3], key)):
        token = String(resident[0])
    else:
        if resident is not None:
            prepared_drop(mods, _QUERY_RESIDENT_SLOT, driver)
            _release_query_state(pool, String(resident[0]), resident[4], workers)
        var held = owner_record.downcast_value_ptr[ParallelNeighborOwner]()
        held[].tokens += 1
        var counter = held[].tokens
        token = String("nq-") + String(external_call["getpid", Int32]()) + "-" + String(Int(py=py_call(b.id, args[0]))) + "-" + String(counter)
        # The key notices replaced attributes by identity only. A carried Array
        # written in place would leave the workers' copy stale, so every
        # carried Array is made read-only now (the forests' `trees_freeze`
        # scheme): an in-place write raises; replacing the attribute or a
        # refit is noticed and re-sent. No per-call content hash.
        _freeze_carried(state, context, 0)
        if index_object is not None:
            py_setitem(state, nk[].k__index, shard_rows(index_object, 0, Int(py=index_object.shape[0]), context))
        var stores = shard_wave()
        for i in range(workers):  # driver-loop: one snapshot request per worker
            stores.add(PARALLEL_SHARD_WORKER_OPN_QUERY_STATE, PythonObject(None), Python.tuple(token, name, state))
        _ = _map(pool, stores)
        # The pool starts its workers on the first map: record their IDs now.
        prepared_put(mods, _QUERY_RESIDENT_SLOT, driver, Python.tuple(token, pool, _pool_pids(pool), key, name))
    var results = PythonObject(None)
    var n_devices = workers
    try:
        var requests = shard_wave()
        for r in range(len(ranges) // 2):  # driver-loop: one worker request per query row block
            var lo = ranges[2 * r]
            var hi = ranges[2 * r + 1]
            var shift = shard_read_shift(r, lo, n_devices)
            requests.add(PARALLEL_SHARD_WORKER_OPN_QUERY_ROWS, PythonObject(None), Python.tuple(
                token, shard_rows(data, lo - shift, hi - shift, context), method, args[3]))
        results = _map(pool, requests)
    except error:
        # A failed query leaves no resident claim: release the snapshot (as
        # before) and send it again on the next call.
        prepared_drop(mods, _QUERY_RESIDENT_SLOT, driver)
        _release_query_state(pool, token, name, workers)
        raise error^
    var outputs = b.list()
    var diagnostics = b.list()
    var radius = String(method) == "radius_neighbors"
    for r in range(len(results)):  # driver-loop: one result and diagnostic per query row block
        if radius:
            # N451: the worker sent its CSR triple; the ragged rows are built here once
            _ = py_call(outputs.append, neighbor_radius_rows(results[r][0], context))
        else:
            _ = py_call(outputs.append, results[r][0])
        var record = b.dict()
        py_setitem(record, nk[].k_start, ranges[2 * r])
        py_setitem(record, nk[].k_end, ranges[2 * r + 1])
        py_setitem(record, nk[].k_device, PythonObject(devices[r % workers]))
        _ = py_call(record.update, results[r][1])
        _ = py_call(diagnostics.append, record)
    var output = _join(outputs, String(method) == "radius_neighbors", context)
    py_setitem(holder, nk[].k_last_shards_, diagnostics)
    if String(method) == "radius_neighbors":
        return output  # The plain ragged API is not an _mode public-output method.
    var kind = PythonObject("regressor") if py_type(model) is _t("knr_class") else PythonObject(None)
    return parallel_public_result(output, String(method), kind, context)


def _ivf_from_index(args: PythonObject) raises -> PythonObject:
    """args (class, index, devices): `DistributedIVFIndex.from_index`.
    Native admission and the shard plan (`ivf_shard_plan`) give
    every device its contiguous stored rows with list offsets and local IDs;
    each worker stores its shard under the object's token."""
    var b = shard_b()
    var context = _ctx()
    var index = args[1]
    if args[0] is not _t("distributed_ivf_class"):
        raise Error("TypeError: distributed IVF requires its exact connected class")
    var admitted = parallel_ivf_admit(index, PARALLEL_SHARD_BOUNDARY.get_or_create_ptr()[].tables, context)
    var fields = py_fields(index)
    var host = _member_host(fields)
    var nk = parallel_keys()
    var n = Int(py=py_item(fields, nk[].k_n_rows_))
    var n_lists = Int(py=py_item(fields, nk[].k_n_lists_))
    var ind = admitted[1]
    var offs = admitted[2]
    _ = shard_devices(args[2])
    var all_devices = py_call(b.tuple, args[2])
    _ = shard_devices(all_devices)
    var parts_count = min(len(all_devices), n)
    var soff = shard_array("<i4", Python.tuple(parts_count, n_lists + 1), parts_count * (n_lists + 1), context)
    var local_all = shard_array("<i4", Python.tuple(n), n, context)
    var map_all = shard_array("<i4", Python.tuple(n), n, context)
    var ivf_key = _table_key("ivf", host)
    var ind_addr = Int(py=classical_address(ind))
    var offs_addr = Int(py=classical_address(offs))
    var soff_addr = Int(py=classical_address(soff))
    var local_addr = Int(py=classical_address(local_all))
    var map_addr = Int(py=classical_address(map_all))
    var status = 0
    with GILReleased(Python()):
        status = parallel_ivf_shard_plan(ivf_key, ind_addr, offs_addr, soff_addr, local_addr, map_addr,
                                         n, n_lists, parts_count)
    _ = soff  # keep the Python object alive through the native call above
    _ = local_all  # keep the Python object alive through the native call above
    _ = map_all  # keep the Python object alive through the native call above
    # `ind` and `offs` are fresh int32 copies owned only by these locals (and
    # `admitted`): without these uses Mojo released them at their last named
    # use, before the plan read their addresses, and the plan saw freed
    # memory ("invalid global IVF list offsets" on a valid index).
    _ = ind  # keep the Python object alive through the native call above
    _ = offs  # keep the Python object alive through the native call above
    _ = admitted  # keep the Python object alive through the native call above
    ivf_offsets_admit(status)
    var devices = all_devices[py_call(b.slice, PythonObject(None), PythonObject(n))]
    var pool = shard_open(devices, _CHILD, host)
    var obj = py_call(b.object.__new__, args[0])
    var own = py_fields(obj)
    py_setitem(own, nk[].k__closed, False)
    py_setitem(own, nk[].k_n_features_in_, py_item(fields, nk[].k_n_features_in_))
    py_setitem(own, nk[].k_n_neighbors, py_item(fields, nk[].k_n_neighbors))
    py_setitem(own, nk[].k_metric_code_, py_item(fields, nk[].k_metric_code_))
    py_setitem(own, nk[].k_devices, devices)
    py_setitem(own, nk[].k_n_candidates_, PythonObject(None))
    var owner_id = Int(py=py_call(b.id, obj))
    # The store token as one str, built once: every store and search request carries this object.
    var token = PythonObject(String("ivf-") + String(external_call["getpid", Int32]()) + "-" + String(owner_id))
    var maps = b.list()
    var held = ParallelIvfOwner(owner_id, Int(py=py_item(fields, nk[].k_n_features_in_)),
                                Int(py=py_item(fields, nk[].k_n_neighbors)), Int(py=py_item(fields, nk[].k_metric_code_)),
                                token, pool, maps)
    var requests = shard_wave()
    var base = shard_state(fields, b.frozenset(Python.list("_resident", "_resident_holder", "list_data_",
                                                           "list_indices_", "list_offsets_", "n_rows_")), context)
    py_setitem(base, nk[].k_numeric_mode, "identical")
    for part in range(parts_count):  # driver-loop: one stored shard per device
        var lo = part * n // parts_count
        var hi = (part + 1) * n // parts_count
        var state = py_call(b.dict, base)
        py_setitem(state, nk[].k_n_rows_, hi - lo)
        py_setitem(state, nk[].k_list_offsets_, shard_row_vector(soff, part, context))
        py_setitem(state, nk[].k_list_indices_, shard_rows(local_all, lo, hi, context))
        py_setitem(state, nk[].k_list_data_, shard_rows(py_item(fields, nk[].k_list_data_), lo, hi, context))
        var mapping = shard_rows(map_all, lo, hi, context)
        _ = py_call(maps.append, mapping)
        # The map array is pinned by `maps` for the owner's lifetime: its row
        # count and address are read here once, never again per search.
        held.map_rows.append(hi - lo)
        held.map_addrs.append(Int(py=classical_address(mapping)) if hi > lo else 0)
        requests.add(PARALLEL_SHARD_WORKER_OPN_IVF_STORE, PythonObject(None), Python.tuple(token, state))
    # PACK, once per index: the native owner.
    var record = PythonObject(alloc=held^)
    py_setitem(own, nk[].k__native, record)
    try:
        var receipts = _map(pool, requests)
        var rec = record.downcast_value_ptr[ParallelIvfOwner]()
        for part in range(parts_count):  # driver-loop: one storage receipt per shard
            ivf_receipt_admit(Int(py=receipts[part]), rec[].map_rows[part])
    except error:
        process_pool_close(pool)
        record.downcast_value_ptr[ParallelIvfOwner]()[].closed = True
        py_setitem(own, nk[].k__closed, True)
        raise error^
    _ = record  # the owner record is read through its pointer above
    return obj


def _ivf_search(args: PythonObject) raises -> PythonObject:
    """args (distributed index, queries): every worker's partial search of
    its stored shard, merged by the native `ivf_merge_shards` (cosine
    distances finalized natively)."""
    var b = shard_b()
    var context = _ctx()
    var record = _ivf_record(args[0])
    var rec = record.downcast_value_ptr[ParallelIvfOwner]()
    var nk = parallel_keys()
    ivf_open_admit(rec[].closed)
    var q = admit_family(args[1], 2, "queries", context).array
    ivf_queries_admit(Int(py=q.shape[0]))
    ivf_features_admit(Int(py=q.shape[1]), rec[].n_features)
    var shards = len(rec[].map_rows)
    try:
        var requests = shard_wave()
        # One operand tuple for the wave: every shard's request carries the same token and queries.
        var operands = Python.tuple(rec[].token, q)
        for i in range(shards):  # driver-loop: one search request per stored shard
            requests.add(PARALLEL_SHARD_WORKER_OPN_IVF_SEARCH, PythonObject(None), operands)
        var parts = _map(rec[].pool, requests)
        ivf_parts_admit(len(parts), shards)
        var m = Int(py=q.shape[0])
        var k = rec[].n_neighbors
        for i in range(len(parts)):  # driver-loop: checks one result shard
            var p = parts[i]
            if (Int(py=p[0].shape[0]) != m or Int(py=p[0].shape[1]) != k or Int(py=p[1].shape[0]) != m
                    or Int(py=p[1].shape[1]) != k or Int(py=p[2].shape[0]) != m):
                raise Error("ValueError: invalid IVF shard output shapes")
        var result = shard_array("<f4", Python.tuple(m, k), m * k, context)
        var ids = shard_array("<i4", Python.tuple(m, k), m * k, context)
        var counts = shard_array("<i4", Python.tuple(m), m, context)
        var ivf_key = _table_key("ivf", _pool_host(rec[].pool))
        var addrs: List[Int] = [Int(py=classical_address(result)), Int(py=classical_address(ids)),
                                Int(py=classical_address(counts))]
        var sizes: List[Int] = [len(parts), m, k]
        for i in range(len(parts)):  # driver-loop: four buffer addresses per result shard
            var p = parts[i]
            var counts_addr = Int(py=classical_address(p[2]))
            addrs.append(Int(py=classical_address(p[0])))
            addrs.append(Int(py=classical_address(p[1])))
            addrs.append(counts_addr)
            # The shard's global-id map: its address and rows are the owner's words (an empty map
            # passes the counts address, as before, so the merge never sees a null pointer).
            addrs.append(rec[].map_addrs[i] if rec[].map_rows[i] > 0 else counts_addr)
            sizes.append(rec[].map_rows[i])
        var status = 0
        with GILReleased(Python()):
            status = parallel_ivf_merge_shards(ivf_key, addrs, sizes)
        _ = parts  # keep the Python objects alive through the native call above
        _ = result  # keep the Python object alive through the native call above
        _ = ids  # keep the Python object alive through the native call above
        _ = counts  # keep the Python object alive through the native call above
        ivf_merge_admit(status)
        if rec[].metric == 1:
            var result_addr = Int(py=classical_address(result))
            var metric = rec[].metric
            with GILReleased(Python()):
                _ = parallel_ivf_finalize(ivf_key, result_addr, m * k, metric)
            _ = result  # keep the Python object alive through the native call above
        # PACK: the candidate counts mirror on the instance.
        py_setitem(py_fields(args[0]), nk[].k_n_candidates_, counts)
        _ = record  # the owner (and through it the pinned map arrays) outlives the merge above
        return Python.tuple(result, ids)
    except error:
        _ivf_close(args[0])
        raise error^


def _ivf_close(obj: PythonObject) raises:
    """Close is safe and idempotent: an index whose construction never
    stored its owner has no pool."""
    var record = _ivf_record_or_none(obj)
    py_setitem(py_fields(obj), parallel_keys()[].k__closed, True)
    if record is None:
        return
    var rec = record.downcast_value_ptr[ParallelIvfOwner]()
    rec[].closed = True
    if not process_pool_closed(rec[].pool):
        process_pool_close(rec[].pool)
    _ = record  # the owner record is read through its pointer above


# ---------------------------------------------------------------- the scaler worker

def _scaler_worker(operation: Int, state: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """One column block's scaler fit or transform through the typed cores
    (core/scaler_core.mojo) and the identical images' scaler and x_prep table
    entries (lane scaler-typed-core): UNPACK the block and the state record
    into typed values once, RUN the core (the public doors' one core, with
    its NaN handling, statistic checks, overflow refusals and restore step),
    PACK the fitted state record or the transformed block once. No model
    object, no Python entry call. The block is the driver's own C-ordered
    Float32 copy (`_columns`); a fit's state record carries the same fields
    as the public fit's (`_fit_scaler` merges them)."""
    var kind = String(state[0])
    var owner_context = _t(kind + "_context")
    var calls = scaler_door_calls(_entries(PythonObject("identical"), owner_context))
    var x = args[0]
    var n = Int(py=x.shape[0])
    var w = Int(py=x.shape[1])
    var xp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(x)))
    if operation == PARALLEL_SHARD_WORKER_OPN_SCALER_FIT:
        var init = state[1]
        var fields = shard_b().dict()
        if kind == "standard":
            # init: (copy, with_mean, with_std, "identical")
            var names = Python.tuple("copy", "with_mean", "with_std", "numeric_mode")
            for k in range(4):  # small-loop(params: the four stored constructor arguments)
                py_setitem(fields, names[k], init[k])
            _ = _scaler_bool(init[0], owner_context)
            var with_mean = _scaler_bool(init[1], owner_context)
            var with_std = _scaler_bool(init[2], owner_context)
            var fit = scaler_standard_fit_core(calls, xp, n, w, with_mean, with_std, 0)
            _ = x  # keep the block alive through the native calls above
            scaler_standard_publish(fields, fit, w, with_mean, with_std, PythonObject("identical"), owner_context)
        else:
            # init: (feature_range, copy, clip, "identical")
            var names = Python.tuple("feature_range", "copy", "clip", "numeric_mode")
            for k in range(4):  # small-loop(params: constructor metadata)
                py_setitem(fields, names[k], init[k])
            var configuration = minmax_configuration(fields, owner_context)
            var rows = scaler_minmax_fit_core(calls, xp, n, w, Float32(Float64(py=configuration[1])),
                                              Float32(Float64(py=configuration[2])))
            _ = x  # keep the block alive through the native calls above
            minmax_fit_publish(fields, rows, n, w, configuration, owner_context)
        return shard_state(fields, shard_b().frozenset(), context)
    # scaler_transform: state[1] is the block's fitted fields
    var fitted = state[1]
    var inverse = Bool(py=args[1])
    var rows = List[Float32]()
    var spec = ScalerTransformSpec(False, inverse, True, True, False, 0, 0)
    var nk = parallel_keys()
    if kind == "standard":
        var with_mean = _scaler_bool(py_item(fitted, nk[].k_with_mean_), owner_context)
        var with_std = _scaler_bool(py_item(fitted, nk[].k_with_std_), owner_context)
        rows = scaler_standard_rows(calls, py_item(fitted, nk[].k_mean_), py_item(fitted, nk[].k_scale_), w, with_mean, with_std, owner_context)
        spec = ScalerTransformSpec(False, inverse, with_mean, with_std, False, 0, 0)
    else:
        var bounds = minmax_range_pair(py_item(fitted, nk[].k_feature_range_), owner_context)
        var clip = _minmax_flag(py_item(fitted, nk[].k_clip_), "clip_", owner_context)
        rows = minmax_rows(calls, py_item(fitted, nk[].k_min_), py_item(fitted, nk[].k_scale_), w, owner_context)
        spec = ScalerTransformSpec(True, inverse, True, True, clip, Float32(Float64(py=bounds[0])), Float32(Float64(py=bounds[1])))
    var out = shard_array("<f4", Python.tuple(n, w), n * w, context)
    scaler_transform_core(calls, spec, MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(rows.unsafe_ptr())),
                          xp, n, w, MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out))))
    _ = rows^
    _ = x  # keep the block and the output alive through the native calls above
    _ = out
    return out


# ---------------------------------------------------------------- the worker

def shard_worker_dispatch(operation: Int, request: PythonObject) raises -> PythonObject:
    """One shard: its model rebuilt natively from the state record, then the
    plain native owner. `operation` is the decoded request word (a
    `PARALLEL_SHARD_WORKER_OPN_*` code); `request` is the payload record
    `(state, args)` (None for the identity request)."""
    var b = shard_b()
    if operation == PARALLEL_SHARD_WORKER_OPN_IDENTITY:
        return shard_identity()
    var context = _ctx()
    shard_envelope_admit(py_type(request) is b.tuple, len(request) if py_type(request) is b.tuple else 0)
    # N452 (lane ne2-i-j): a call's shared operands, stored once per worker
    if operation == PARALLEL_SHARD_WORKER_OPN_SHARD_STORE:
        return shard_store_put(request[0], request[1])
    if operation == PARALLEL_SHARD_WORKER_OPN_SHARD_RELEASE:
        return shard_store_drop(request[0])
    if operation == PARALLEL_SHARD_WORKER_OPN_RBF_ROWS:
        # The model of the call, built once per worker when its state is
        # stored (`shard_stored_model`); only the row block rides here.
        var model = shard_stored_model(_t("rbf_class"), request[0])
        var args = request[1]
        var rows_pin = ArrayPin()
        var rows_meta = admit_family_meta(args[0], 2, "X", context, rows_pin, True)
        var rows_out = _rbf_transform(model, Python.tuple(_worker_binding("_mojolearn_kernel_methods"), rows_pin.keep),
                                      context, rows_meta, rows_pin)
        _ = rows_pin^
        return rows_out
    var state = shard_resolve_stored(request[0])
    var args = shard_resolve_stored(request[1])
    if operation == PARALLEL_SHARD_WORKER_OPN_SCALER_FIT or operation == PARALLEL_SHARD_WORKER_OPN_SCALER_TRANSFORM:
        return _scaler_worker(operation, state, args, context)
    # The GPC sites call the GP table entry (core/gp_entry_table.mojo) of the
    # connected identical binding with plain arrays and words
    # (bindings/gp_classifier_native_door.mojo gpc_worker_fit/gpc_worker_latent).
    if operation == PARALLEL_SHARD_WORKER_OPN_GPC_CLASS_FIT:
        var fields = gpc_worker_fit(_worker_binding("_mojolearn_gp"), args[0], args[1], state[0], Int(py=state[1]),
                                    Int(py=args[2]), context)
        return shard_state(fields, b.frozenset(), context)
    if operation == PARALLEL_SHARD_WORKER_OPN_GPC_CLASS_PREDICT:
        return gpc_worker_latent(_worker_binding("_mojolearn_gp"), state[1], state[2], args[0], Bool(py=args[1]),
                                 Int(py=args[2]), state[0], context)
    if operation == PARALLEL_SHARD_WORKER_OPN_FOREST_FIT:
        # Lane forest-typed-fit: the request words and the arrays, fitted by
        # the one typed forest fit core; no estimator is built here.
        return forest_shard_fit(state, args, _t("trees_context"))
    # The neighbor shard operations are the last codes of the worker table
    # (reference_shard .. query_rows): one range test.
    if operation >= PARALLEL_SHARD_WORKER_OPN_REFERENCE_SHARD and operation <= PARALLEL_SHARD_WORKER_OPN_QUERY_ROWS:
        return parallel_neighbor_worker(operation, args, PARALLEL_SHARD_BOUNDARY.get_or_create_ptr()[].tables, context)
    raise Error("parallel shards: unsupported worker operation " + String(operation))


def shard_worker(argv: PythonObject) raises:
    var socket = process_worker_connect(argv)
    try:
        while True:  # driver-loop: over native local shard requests
            var turn = process_worker_next(socket)
            if not turn.open:
                break
            try:
                # The typed request: one decode of the frame's words (a frame
                # without request words is refused), then the code's branch.
                var operation = shard_request_op(Int(turn.request.unsafe_ptr()), len(turn.request) * 8)
                process_worker_reply(socket, turn.sequence, shard_worker_dispatch(operation, process_worker_value(turn)))
            except error:
                process_worker_refuse(socket, turn.sequence, String(error))
    except error:
        transport_close(socket)
        raise error^
    transport_close(socket)


def parallel_shard_call(operation: Int, name: PythonObject, args: PythonObject) raises -> PythonObject:
    """`operation` is a `PARALLEL_SHARD_OPN_*` code (-1 for a name the table
    does not hold); `name` is read only for the unknown operation's text."""
    var s = PARALLEL_SHARD_BOUNDARY.get_or_create_ptr()
    var nk = parallel_keys()
    if operation == PARALLEL_SHARD_OPN_REGISTER:
        # A driver module's fixed classes, entries and contexts, at its import.
        if s[].tables is None:
            s[].tables = shard_b().dict()
        _ = py_call(s[].tables.update, args)
        var context = py_get(s[].tables, PythonObject("context"), PythonObject(None))
        if context is not None:
            codec_initialize(py_item(context, nk[].k_array_class), PythonObject(None), PythonObject(None))
        return PythonObject(None)
    if operation == PARALLEL_SHARD_OPN_TRANSFORM_RBF_SAMPLER:
        return _transform_rbf_sampler(args)
    if operation == PARALLEL_SHARD_OPN_FIT_SCALER:
        return _fit_scaler(args)
    if operation == PARALLEL_SHARD_OPN_TRANSFORM_SCALER:
        var result = _transform_scaler(args)
        return parallel_public_result(result, "inverse_transform" if Bool(py=args[4]) else "transform",
                                      PythonObject(None), _ctx())
    if operation == PARALLEL_SHARD_OPN_FIT_GAUSSIAN_PROCESS_CLASSIFIER:
        return _fit_gpc(args)
    if operation == PARALLEL_SHARD_OPN_PREDICT_GAUSSIAN_PROCESS_CLASSIFIER:
        var context = _ctx()
        classical_connect_numpy(context)
        # predict: one result for every caller (lane predict-dtype, 2026-11-22).
        return _predict_result(_predict_gpc(args), context)
    if operation == PARALLEL_SHARD_OPN_FIT_FOREST:
        return _fit_forest(args)
    if operation == PARALLEL_SHARD_OPN_TRACE_TAKE:
        return shard_trace_take()
    if operation == PARALLEL_SHARD_OPN_FAULTS_TAKE:
        return PythonObject(shard_faults_take())
    if operation == PARALLEL_SHARD_OPN_FAULTS_ARM:
        # The verifier's transport negative control (lane
        # envgate-classical-deviceplan: replaces MOJOLEARN_PAR_TRANSPORT_FAULT,
        # read per numerical batch): "drop_result", "reverse_results" or "".
        shard_faults_arm(String(args[0]))
        return PythonObject(None)
    if operation == PARALLEL_SHARD_OPN_SABOTAGE_ARM:
        # The driver read-shift negative control inside a running process.
        shard_sabotage_arm(Bool(py=args[0]))
        return PythonObject(None)
    if operation == PARALLEL_SHARD_OPN_REFERENCE_INIT or operation == PARALLEL_SHARD_OPN_QUERIES_INIT:
        return _holder_init(args, operation == PARALLEL_SHARD_OPN_REFERENCE_INIT)
    if operation == PARALLEL_SHARD_OPN_HOLDER_ENTER:
        if py_type(args[0]) is not _t("reference_driver_class") and py_type(args[0]) is not _t("query_driver_class"):
            raise Error("TypeError: parallel driver requires its exact connected class")
        return args[0]
    if operation == PARALLEL_SHARD_OPN_HOLDER_CLOSE:
        if py_type(args[0]) is not _t("reference_driver_class") and py_type(args[0]) is not _t("query_driver_class"):
            raise Error("TypeError: parallel driver requires its exact connected class")
        # A construction that failed (every refusal of _holder_init runs
        # before the owner is stored, and a failed _pool_open stores none) has
        # no pool: close is safe and idempotent there, as _ivf_close is
        # (cpu-surface-defects, 2026-11-16).
        var held = _holder_record_or_none(args[0])
        if held is not None:
            var held_pool = held.downcast_value_ptr[ParallelNeighborOwner]()[].pool
            if not process_pool_closed(held_pool):
                process_pool_close(held_pool)
        return PythonObject(None)
    if operation == PARALLEL_SHARD_OPN_IVF_INIT:
        raise Error("TypeError: use DistributedIVFIndex.from_index(index, devices=...)")
    if operation == PARALLEL_SHARD_OPN_IVF_ENTER:
        var entered = _ivf_record(args[0])
        ivf_open_admit(entered.downcast_value_ptr[ParallelIvfOwner]()[].closed)
        return args[0]
    if operation == PARALLEL_SHARD_OPN_REFERENCE_QUERY:
        return _reference_query(args)
    if operation == PARALLEL_SHARD_OPN_PARALLEL_QUERIES:
        return _parallel_queries(args)
    if operation == PARALLEL_SHARD_OPN_IVF_FROM_INDEX:
        return _ivf_from_index(args)
    if operation == PARALLEL_SHARD_OPN_IVF_SEARCH:
        return _ivf_search(args)
    if operation == PARALLEL_SHARD_OPN_IVF_CLOSE:
        _ivf_close(args[0])
        return PythonObject(None)
    if operation == PARALLEL_SHARD_OPN_WORKER:
        shard_worker(args)
        return PythonObject(None)
    raise Error("RuntimeError: parallel shards: unknown operation " + String(name))


def parallel_shard_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    try:
        var nk = parallel_keys()
        var result = parallel_shard_call(py_op_code(nk[].ops_shard, args[0]), args[0], args[1])
        return result.steal_data()
    except error:
        var message = String(error)
        if message.startswith("InterruptedError:"):
            return raise_python_exception(error_without_class(error, "InterruptedError"), ExceptionType("PyExc_InterruptedError"))
        if message.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if message.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if message.startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if message.startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
