# SPDX-License-Identifier: Apache-2.0
# UMAP's public methods, one native call each (python/mojolearn/_umap_impl.py:
# `fit`, `fit_transform`, `transform`). This owner selects exact retained
# compiled entries in the requested or fitted mode; it validates parameters with
# `_parameters` / `_extras` / `_target`'s refusals (same wording, same order),
# builds a 'random' or 'pca' start with the decomp kit's cells exactly as
# `_initial` did (the typed kit, bindings/mani_umap_init_run.mojo: the same
# x_decomp operations, operands and order; the identical-mode image,
# `_svd_flip_v` on the default-mode image),
# encodes a categorical target, runs `umap_fit_transform` /
# `umap_fit_transform_ex` / `umap_transform` with the argument lists they
# always took and stores the fitted attributes the saved-model schema reads.
# The same entries on the same operands: the bits do not move.
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family_meta
from bindings.native_loaded_module_door import native_loaded_module
from std.python._cpython import GILReleased
from bindings.py_balanced import py_call, py_get, py_setitem, py_contains, py_type, py_fields
from bindings.py_attr import py_attr
from bindings.classical_arrays_door import classical_address
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.pca_exact_unpack import pca_copied
from bindings.manifold_settings_native_door import ManifoldSettings
from providers.python.array_metadata_pyapi_door import ArrayMetadata, ArrayPin, array_shape_once, array_metadata_once, array_python_shape
from bindings.classical_validate_unpack import classical_view
from bindings.umap_parameters_native_door import umap_parameters_fields, umap_scalar as _umap_scalar, umap_integer as _umap_integer
from x_ann.mani_umap_run import (
    umap_finite, umap_mode_code, umap_check_input, umap_check_neighbors, umap_check_spectral_samples,
    umap_init_shape_matches,
    umap_init_shape_refusal, umap_check_init_finite, umap_check_target_weight, umap_target_neighbors,
    umap_label_dtype_codable, umap_continuous_target_dims, umap_legacy_abi, umap_check_embedding,
    umap_check_transform_state, umap_check_transform_input, umap_transform_reuses_fit, umap_check_transform_output,
)
from bindings.family_common_unpack import (
    FAMILY_F32P,
    family_repr,
    family_is_str,
)
from bindings.family_common_pack import (
    family_array,
    family_field,
)
from bindings.spectral_umap_routes_pack import spectral_umap_route
from core.neighbors_entry_table import NeighborsEntryKey
from core.neighbors_entry_table_umap import (
    UmapConfigWords, UmapTargetWords, umap_ops_entry, umap_fit_call, umap_fit_ex_call, umap_transform_call,
    UMAP_TAIL_LEGACY, UMAP_TAIL_RATES, UMAP_TAIL_EXTRAS,
)
from core.cast_in import CAST_KIND_SIGNED, CAST_KIND_FLOAT
from bindings.label_input_sources_unpack import labels_list_scalar_kind, LABELS_LIST_INT, LABELS_LIST_FLOAT, LABELS_LIST_BOOL
from bindings.p3_labels_metrics_native_door import p3_text_codes, P3_TEXT_OK
from bindings.label_list_words_unpack import label_store_address
from bindings.label_list_words_pack import label_list_store
from core.label_words_impl import label_words_f64, label_words_i64, label_dense_codes, label_codes_i32_to_f32
from bindings.mani_kit_door import mk_image, mk_out2, mk_finish, mk_seed_word
from bindings.mani_kit_run import MRun
from bindings.mani_umap_init_run import umap_random_start_run, umap_pca_start_run

comptime _F32_MAX = 3.4028234663852886e+38
comptime _I64_MAX = 9223372036854775807


def _umap_mode(settings: ManifoldSettings, context: PythonObject) raises -> String:
    """`(self.numeric_mode or _backend.default_mode()).strip().lower()`."""
    var b = native_loaded_module("builtins")
    var want = settings.requested
    if want is not None and py_type(want) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    if not Bool(py=py_call(b.bool, want)):
        want = settings.default_mode
    return String(py=py_call(py_call(want.strip).lower))


def _umap_check_mode(binding: PythonObject, mode: String, message: String) raises -> NeighborsEntryKey:
    """`binding.umap_numeric_mode() != {"fast": 0, "identical": 1}[mode]`, as
    a key comparison: the UMAP operations table key
    (core/neighbors_entry_table_umap.mojo) the image behind `binding` recorded
    as its module attribute `umap_native_key` when it published, one exact
    dictionary read at UNPACK. Returns the key; the UMAP entries are then
    called natively."""
    var want = umap_mode_code(mode)
    if want < 0:
        raise Error("KeyError: " + family_repr(PythonObject(mode)))
    var b = native_loaded_module("builtins")
    if py_type(binding) is not py_type(b):
        raise Error("TypeError: spectral/UMAP binding must be an exact module")
    var key = py_get(py_fields(binding), PythonObject("umap_native_key"))
    if key is None or py_type(key) is not b.tuple or len(key) != 2:
        raise Error("ImportError: mojolearn: the UMAP binding publishes no native UMAP table; rebuild it")
    var native_key = NeighborsEntryKey(Int(py=key[0]), Int(py=key[1]))
    if native_key.mode != want:
        raise Error("RuntimeError: " + message)
    return native_key^


def _umap_config_words(config: PythonObject, extras: PythonObject) raises -> UmapConfigWords:
    """UNPACK of the validated `config` (11 values) and `extras` (4 values)
    of `umap_parameters_fields`, read once as plain words."""
    return UmapConfigWords(
        Int(py=config[0]), Int(py=config[1]), Int(py=config[2]), Float64(py=config[3]), Float64(py=config[4]),
        Float64(py=config[5]), Float64(py=config[6]), Int(py=config[7]), Float64(py=config[8]),
        Float64(py=config[9]), Int(py=config[10]), Int(py=extras[0]), Float64(py=extras[1]),
        Float64(py=extras[2]), Float64(py=extras[3]),
    )


def _umap_copy(shape: PythonObject, address: Int, count: Int, context: PythonObject) raises -> PythonObject:
    """An owned float32 copy using already-captured shape/address/count."""
    var output = family_array(shape, "<f4", context)
    if count > 0:
        unsafe_memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(output))),
               src=FAMILY_F32P(unsafe_from_address=address), count=count)
    return output


def _umap_state(value: PythonObject, name: String, context: PythonObject, mut pin: ArrayPin) raises -> ArrayMetadata:
    var view = classical_view(value, context, name)
    var shape = array_shape_once(view, context)
    var snapshot = array_metadata_once(view, shape^, name, context, pin)
    if len(snapshot.shape) != 2 or not snapshot.f32_c():
        raise Error("UMAP fitted " + name + " must be a C-order float32 matrix")
    return snapshot^


def _umap_initial(settings: ManifoldSettings, x: ArrayMetadata, n: Int, nc: Int, seed: PythonObject,
                  context: PythonObject, mut pin: ArrayPin) raises -> Optional[ArrayMetadata]:
    """`_initial(x, n, nc, seed)`: None for 'spectral'; 'random' uniform(-10,
    10) on the Philox stream; 'pca' umap-learn's PCA of the data scaled to
    max |coordinate| 10 plus N(0, 1e-4) noise; an array as given."""
    var init = family_field(settings.values, "init")
    if family_is_str(init) and String(py=init) == "spectral":
        return Optional[ArrayMetadata]()
    if family_is_str(init):
        # UNPACK `_Kit("identical")` and `_host_kit()` (the default mode, for
        # the host moves and `_svd_flip_v`) as images of one typed kit; RUN
        # the start with the GIL released (bindings/mani_umap_init_run.mojo);
        # PACK it into the `init` Array (lane entangled-mkit, 2026-11-12).
        var run = MRun()
        try:
            var k = mk_image(run, "identical", spectral_umap_route(context, "decomp", PythonObject("identical")),
                             context)
            var hk = mk_image(run, String(settings.default_mode),
                              spectral_umap_route(context, "decomp", settings.default_mode), context)
            var seed_word = mk_seed_word(seed)
            var start = -1
            if String(py=init) == "random":
                with GILReleased(Python()):
                    start = umap_random_start_run(run, k, n, nc, seed_word)
            else:
                # 'pca'
                var address = x.source.addr
                var rows = x.source.rows
                var cols = x.source.cols
                var ndim = len(x.shape)
                with GILReleased(Python()):
                    start = umap_pca_start_run(run, k, hk, address, rows, cols, ndim, n, nc, seed_word)
                _ = x  # the admitted X stays alive through the RUN above (the door's pin holds its storage)
            return Optional(admit_family_meta(mk_out2(run, start, context), 2, "init", context, pin))
        finally:
            mk_finish(run, context)
    var a = admit_family_meta(init, 2, "init", context, pin)
    if not umap_init_shape_matches(a.source.rows, a.source.cols, n, nc):
        var refusal = umap_init_shape_refusal(family_repr(array_python_shape(a)), n, nc)
        raise refusal^
    umap_check_init_finite(a.source.addr, a.count)
    return Optional(a^)


# Lane i3-move-data-loops (2026-11-09): the label-word reads, the dense
# coding and the code widening run in core/label_words_impl.mojo (the loops
# P3 wrote here moved word for word, split across the host's cores); an exact
# list or tuple is unpacked by one array.array construction
# (bindings/label_list_words.mojo). This file keeps the unpack and the pack.


def _umap_native_codes(y: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """(codes Array, codes address) of a 1-D numeric categorical target:
    labels other than -1 numbered 0.. in ascending value order, -1 kept as
    -1, exactly the codes the object path gives one-dtype labels. None when
    the source is not a 1-D array of n words of a supported dtype or holds
    NaN (the object path then decides, with its refusals)."""
    var shape = array_shape_once(y, context)
    if len(shape) != 1 or shape[0] != n:
        return PythonObject(None)
    var pin = ArrayPin()
    var meta = array_metadata_once(y, shape^, "y", context, pin, allow_non_native=True, allow_unsupported=True)
    if not meta.castable or not meta.native_byte_order:
        return PythonObject(None)
    var kind = meta.source.kind
    var item = meta.source.itemsize
    var stride = meta.source.s0
    var base = meta.source.addr
    if not umap_label_dtype_codable(kind, item):
        return PythonObject(None)
    if kind == CAST_KIND_FLOAT:
        var words = List[Float64]()
        var ok = label_words_f64(base, stride, item, n, words)
        _ = pin^  # the door's pin held the source through the reads above
        if not ok:
            return PythonObject(None)
        return _umap_codes(words, context)
    var ints = label_words_i64(base, stride, kind, item, n)
    _ = pin^  # the door's pin held the source through the reads above
    return _umap_codes(ints, context)


def _umap_codes[dt: DType, //](words: List[Scalar[dt]], context: PythonObject) raises -> PythonObject:
    """(codes Array, codes address): labels other than -1 numbered 0.. in
    ascending value order, -1 kept as -1 (`label_dense_codes`)."""
    var n = len(words)
    var t = family_array(Python.tuple(n), "<f4", context)
    var tp = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(t)))
    label_dense_codes(words, tp)
    _ = t  # keep the Python object alive through the native writes above
    return Python.tuple(t, PythonObject(Int(tp)))


def _umap_list_codes(y: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """(codes Array, codes address) of an exact list or tuple of n labels
    that are all exact ints, all exact bools, all exact floats (no NaN) or
    all exact strs: the codes the (type, value) object order gives one-type
    labels (one type, so the value order; -1 is unknown for numbers and never
    equals a str). Numbers are unpacked by one array.array construction
    (`label_list_store`), strs coded from their UTF-8 bytes. None for
    anything else (the object path then decides, with its refusals)."""
    var b = native_loaded_module("builtins")
    if (py_type(y) is not b.list and py_type(y) is not b.tuple) or len(y) != n or n == 0:
        return PythonObject(None)
    var kind = labels_list_scalar_kind(y)
    var is_tuple = py_type(y) is b.tuple
    if kind == LABELS_LIST_FLOAT:
        var store = label_list_store(y, "d")
        if store is None:
            return PythonObject(None)
        var words = List[Float64]()
        var ok = label_words_f64(label_store_address(store), 8, 8, n, words)
        _ = store  # keep the unpacked words alive through the reads above
        if not ok:
            return PythonObject(None)
        return _umap_codes(words, context)
    if kind == LABELS_LIST_INT or kind == LABELS_LIST_BOOL:
        var store = label_list_store(y, "q")
        if store is None:
            return PythonObject(None)  # an int outside int64: the object path keeps it
        var ints = label_words_i64(label_store_address(store), 8, CAST_KIND_SIGNED, 8, n)
        _ = store  # keep the unpacked words alive through the reads above
        return _umap_codes(ints, context)
    var listed = y if not is_tuple else py_call(b.list, y)
    var text = p3_text_codes(listed, False)
    if text.status != P3_TEXT_OK:
        return PythonObject(None)
    var out = family_array(Python.tuple(n), "<f4", context)
    var op = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(out)))
    label_codes_i32_to_f32(text.codes, op)
    _ = out  # keep the Python object alive through the native writes above
    _ = listed
    return Python.tuple(out, PythonObject(Int(op)))


def _umap_target(settings: ManifoldSettings, y: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_target(y, n)`: (kind, dims, codes, target_n_neighbors, weight)."""
    var b = native_loaded_module("builtins")
    var op = native_loaded_module("operator")
    var fields = settings.values
    var tm = family_field(fields, "target_metric")
    var w = _umap_scalar(family_field(fields, "target_weight"), "target_weight", context)
    umap_check_target_weight(w)
    var tk = _umap_integer(family_field(fields, "target_n_neighbors"), "target_n_neighbors", -1, _I64_MAX, context)
    var tk_kept = PythonObject(umap_target_neighbors(Int(py=tk)))
    if Bool(py=py_call(b.bool, py_call(op.eq, tm, PythonObject("categorical")))):
        # A 1-D NumPy array or mojolearn Array of one numeric dtype: the codes
        # come from one native pass over its words (p3 rank 11). One dtype
        # means one Python type per label, so the (type, value) order is the
        # value order; NaN, uint64, float16 and every other source keep the
        # object path below.
        var y_type = py_type(y)
        if y_type is py_get(context, "ndarray") or y_type is context["array_class"]:
            var fast = _umap_native_codes(y, n, context)
            if fast is not None:
                return Python.tuple(PythonObject(1), PythonObject(1), fast[0], tk_kept, PythonObject(w), fast[0], fast[1])
        # Lane p2-labels-lists-arrays: an exact list / tuple of one exact
        # scalar type or of exact strs is coded from its C storage
        # (_umap_list_codes); the object path below runs only for anything else.
        var listed = _umap_list_codes(y, n, context)
        if listed is not None:
            return Python.tuple(PythonObject(1), PythonObject(1), listed[0], tk_kept, PythonObject(w), listed[0], listed[1])
        # P1: only an exact list, tuple or NumPy array is read (ndarray.tolist
        # and list() are C code), and every label must be an exact int,
        # float, str or bool, refused before any comparison or hash: the
        # set, sort and equality below then run no Python method of a
        # caller's object (the former hasattr/tolist on any object and the
        # (type name, value) sort over arbitrary labels are removed).
        var y_cls = py_type(y)
        var vals: PythonObject
        if y_cls is py_get(context, "ndarray"):
            vals = py_call(py_attr(py_get(context, "ndarray"), "tolist"), y)
        elif y_cls is b.list or y_cls is b.tuple:
            vals = py_call(b.list, y)
        else:
            raise Error("TypeError: UMAP categorical y must be a list, tuple or NumPy array of labels")
        var count = Int(py=py_call(b.len, vals))
        var nested = False
        for i in range(count):  # cpu-route: categorical y label objects ordered by (type, value)
            var label_cls = py_type(vals[i])
            if label_cls is b.list or label_cls is b.tuple:
                nested = True
                break
            if label_cls is not b.int and label_cls is not b.float and label_cls is not b.str and label_cls is not b.bool:
                raise Error("TypeError: UMAP categorical y labels must be int, float, str or bool values")
        if count != n or nested:
            raise Error("UMAP categorical y must be one label per row")
        var unknown = PythonObject(-1)
        var labels = py_call(b.set)
        for i in range(count):  # cpu-route: categorical y label objects ordered by (type, value)
            if Bool(py=py_call(b.bool, py_call(op.ne, vals[i], unknown))):
                _ = py_call(labels.add, vals[i])
        var keyed = Python.list()
        for v in labels:  # cpu-route: categorical y label objects ordered by (type, value)
            _ = py_call(keyed.append, Python.tuple(py_call(b.str, py_type(v)), v))
        var order = py_call(b.sorted, keyed)
        var code = Python.dict()
        for i in range(Int(py=py_call(b.len, order))):  # cpu-route: categorical y label objects ordered by (type, value)
            py_setitem(code, order[i][1], PythonObject(Float64(i)))
        var t = family_array(Python.tuple(n), "<f4", context)
        var tp = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(t)))
        _ = t  # keep the Python object alive through the native call above
        for i in range(count):  # cpu-route: categorical y label objects ordered by (type, value)
            if Bool(py=py_call(b.bool, py_call(op.eq, vals[i], unknown))):
                tp[i] = Float32(-1.0)
                _ = t  # keep the Python object alive through the native call above
            else:
                tp[i] = Float32(Float64(py=code[vals[i]]))
                _ = t  # keep the Python object alive through the native call above
        return Python.tuple(PythonObject(1), PythonObject(1), t, tk_kept, PythonObject(w), t, PythonObject(Int(tp)))
    if Bool(py=py_call(op.contains, Python.tuple(PythonObject("l2"), PythonObject("euclidean")), tm)):
        var target_pin = ArrayPin()
        var target = admit_family_meta(y, -1, "y", context, target_pin)
        var dims = umap_continuous_target_dims(len(target.shape), target.source.rows, target.source.cols, n,
                                               target.source.addr, target.count)
        return Python.tuple(PythonObject(2), PythonObject(dims), target_pin.keep, tk_kept, PythonObject(w),
                            target_pin.owner, PythonObject(target.source.addr))
    raise Error("NotImplementedError: UMAP target_metric=" + family_repr(tm) + " is not implemented; "
                + "'categorical' and 'l2' / 'euclidean' are")


def _umap_fit(model: PythonObject, settings: ManifoldSettings, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`UMAP.fit(X, y)`; returns the embedding."""
    classical_connect_numpy(context)
    var binding = spectral_umap_route(context, "metrics", settings.mode)
    var source = args[0]
    var y = args[1]
    var validated = umap_parameters_fields(settings.values, context)
    var config = validated[0]
    var extras = validated[1]
    var x_pin = ArrayPin()
    var x = admit_family_meta(source, 2, "X", context, x_pin)
    var b = native_loaded_module("builtins")
    var source_type = py_type(source)
    var copied = True
    if source_type is py_get(context, "ndarray") or source_type is context["array_class"]:
        copied = x.copied
    elif source_type is b.memoryview or source_type is context["array_type"] or source_type is b.bytearray:
        # Remaining shared-provider debt: native buffer materialization does
        # not yet carry the original input_copied provenance in ArrayMetadata.
        # Preserve this public flag until the provider can capture it once.
        copied = pca_copied(source, context)
    umap_check_input(x.source.addr, x.count)
    var n = x.source.rows
    var d = x.source.cols
    var nc = Int(py=config[1])
    umap_check_neighbors(Int(py=config[0]), n)
    var initial_pin = ArrayPin()
    var initial = _umap_initial(settings, x, n, nc, config[7], context, initial_pin)
    umap_check_spectral_samples(True if initial else False, n, nc)
    var target = PythonObject(None)
    if not (y is None):
        target = _umap_target(settings, y, n, context)
    var embedding = family_array(Python.tuple(n, nc), "<f4", context)
    var embedding_address = Int(py=classical_address(embedding))
    var mode = _umap_mode(settings, context)
    var key = _umap_check_mode(binding, mode, "UMAP binary numeric mode disagrees with requested mode")
    var cfg = _umap_config_words(config, extras)
    var legacy = umap_legacy_abi(True if initial else False, not (target is None), cfg.default_extras(), nc)
    var entry = umap_ops_entry(key)
    var x_address = x.source.addr
    var columns = 0
    if legacy:
        # Preserve the legacy layout for default controls: the rates go only
        # when they differ from `[1.0, 1.0, 5]`.
        var rates = not cfg.legacy_rates()
        with GILReleased(Python()):
            columns = umap_fit_call(entry, x_address, embedding_address, n, d, cfg, rates)
        _ = x  # keep the Python object alive through the native call above
        _ = embedding  # keep the Python object alive through the native call above
    else:
        var target_words = UmapTargetWords.none()
        if not (target is None):
            target_words = UmapTargetWords(Int(py=target[0]), Int(py=target[1]), Int(py=target[3]),
                                           Float64(py=target[4]), Int(py=target[6]))
        var initial_address = 0 if not initial else initial.value().source.addr
        with GILReleased(Python()):
            columns = umap_fit_ex_call(entry, x_address, embedding_address, initial_address, n, d, cfg,
                                       target_words)
        _ = target  # keep the Python object alive through the native call above
        _ = embedding  # keep the Python object alive through the native call above
        _ = initial  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
    _ = initial^
    _ = initial_pin^  # the door's pin held the start embedding through RUN
    _ = target^
    umap_check_embedding(columns == nc, embedding_address, n * nc)
    # Prepare all retained state before publishing a successful fit. Copies
    # prevent caller edits of X or embedding_ from changing transform's model.
    var training = _umap_copy(array_python_shape(x), x.source.addr, x.count, context)
    var frozen = _umap_copy(Python.tuple(n, nc), embedding_address, n * nc, context)
    _ = x^
    _ = x_pin^  # the door's pin held X through every RUN read and the copy
    py_setitem(settings.fields, "embedding_", embedding)
    py_setitem(settings.fields, "n_features_in_", PythonObject(d))
    py_setitem(settings.fields, "input_copied_", PythonObject(copied))
    py_setitem(settings.fields, "_transform_training", training)
    py_setitem(settings.fields, "_transform_embedding", frozen)
    py_setitem(settings.fields, "_transform_config", _umap_concat_tuple(config, extras))
    py_setitem(settings.fields, "_transform_mode", PythonObject(mode))
    return embedding


def _umap_concat_tuple(config: PythonObject, extras: PythonObject) raises -> PythonObject:
    """`tuple(config) + tuple(extras)`."""
    var b = native_loaded_module("builtins")
    return py_call(native_loaded_module("operator").add, py_call(b.tuple, config), py_call(b.tuple, extras))


def _umap_transform(model: PythonObject, settings: ManifoldSettings, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`UMAP.transform(X)`."""
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    var op = native_loaded_module("operator")
    if not py_contains(settings.values, "_transform_training"):
        raise Error("UMAP transform requires a successful fit")
    var validated = umap_parameters_fields(settings.values, context)
    var config = validated[0]
    var extras = validated[1]
    var mode = _umap_mode(settings, context)
    var fitted_config = family_field(settings.values, "_transform_config")
    var fitted_mode = family_field(settings.values, "_transform_mode")
    if (Bool(py=py_call(b.bool, py_call(op.ne, _umap_concat_tuple(config, extras), fitted_config)))
            or Bool(py=py_call(b.bool, py_call(op.ne, PythonObject(mode), fitted_mode)))):
        raise Error("UMAP parameters or numeric mode changed after fit; refit before transform")
    # The fitted route is selected only after the live-mode/refit refusal,
    # so changing a fitted identical model to unavailable fast still says refit.
    var binding = spectral_umap_route(context, "metrics", fitted_mode)
    var x_pin = ArrayPin()
    var training_pin = ArrayPin()
    var fitted_pin = ArrayPin()
    var x = admit_family_meta(args[0], 2, "X", context, x_pin)
    var training = _umap_state(family_field(settings.values, "_transform_training"), "training", context, training_pin)
    var fitted_embedding = _umap_state(family_field(settings.values, "_transform_embedding"), "embedding", context, fitted_pin)
    var nq = x.source.rows
    var d = x.source.cols
    var nt = training.source.rows
    var nc = Int(py=config[1])
    umap_check_transform_state(fitted_embedding.source.rows, fitted_embedding.source.cols, nt, nc, d,
                               training.source.cols)
    umap_check_transform_input(x.source.addr, x.count)
    # the C-order bytes compared (`x.tobytes() == training.tobytes()`)
    if umap_transform_reuses_fit(x.source.addr, training.source.addr, nq, nt, d):
        var reused = _umap_copy(array_python_shape(fitted_embedding), fitted_embedding.source.addr,
                                fitted_embedding.count, context)
        _ = fitted_pin^
        _ = training_pin^
        _ = x_pin^
        return reused
    var key = _umap_check_mode(binding, mode, "UMAP binary numeric mode disagrees with fitted mode")
    var output = family_array(Python.tuple(nq, nc), "<f4", context)
    var output_address = Int(py=classical_address(output))
    var cfg = _umap_config_words(config, extras)
    var tail = UMAP_TAIL_LEGACY if cfg.legacy_rates() else UMAP_TAIL_RATES
    if not cfg.default_extras():
        tail = UMAP_TAIL_EXTRAS
    var entry = umap_ops_entry(key)
    var training_address = training.source.addr
    var fitted_address = fitted_embedding.source.addr
    var query_address = x.source.addr
    var columns = 0
    with GILReleased(Python()):
        columns = umap_transform_call(entry, training_address, fitted_address, query_address, output_address,
                                      nt, nq, d, cfg, tail)
    _ = fitted_embedding  # keep the Python object alive through the native call above
    _ = training  # keep the Python object alive through the native call above
    _ = output  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = x^
    _ = training^
    _ = fitted_embedding^
    _ = x_pin^  # the door's pins held every input through RUN
    _ = training_pin^
    _ = fitted_pin^
    umap_check_transform_output(columns == nc, output_address, nq * nc)
    return output


def umap_operation(model: PythonObject, settings: ManifoldSettings, operation: String, args: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """UMAP: fit / fit_transform args = (X, y); transform args = (X,)."""
    if operation == "family_mani_umap_fit":
        _ = _umap_fit(model, settings, args, context)
        return model
    if operation == "family_mani_umap_fit_transform":
        return _umap_fit(model, settings, args, context)
    if operation == "family_mani_umap_transform":
        return _umap_transform(model, settings, args, context)
    raise Error("RuntimeError: unknown UMAP operation " + operation)
