# SPDX-License-Identifier: Apache-2.0
# Whole sequence-optimizer operations. All data preparation, owner routing,
# schedule evaluation and state/output handling execute natively here.
# TOMBSTONE: MOJOLEARN_OPTIMIZER_RESIDENT_OFF (LOSER: per-call optimizer step, arm/base geomean 1.129 over 5 cells, samba-train-step 1.97, NV L40S ab3 2026-11-18) deleted 2026-11-18 by lane/fx-verdicts2; code recoverable at 85c36e324f.
# Restore: git apply experiments/removed/MOJOLEARN_OPTIMIZER_RESIDENT_OFF.patch; record in docs/TOMBSTONES.md.
from bindings.mode_answer_door import answer_mode_object
from core.mode_answer import MODE_FAMILY_NEURAL
from std.memory import bitcast
from std.python import Python, PythonObject
from bindings.optimizer_container_boundary_unpack import OptimizerInputOwners
from bindings.py_balanced import py_call, py_type, py_fields, py_get, py_setitem
from bindings.optimizer_real_unpack import optimizer_real
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_class_admission_door import dict_keys_admitted, dict_keys_admit
from bindings.optimizer_admission_door import opt_exact, opt_scalar, opt_value, opt_schedule
from bindings.optimizer_public_door import tensor_view
from bindings.tensor_view import TensorView, base_format
from training.optimizer_input_plan import same_shape
from providers.python.array_layout_boundary_pack import native_shape_pack
from bindings.optimizer_device_native_door import device_shape, shape_count
from bindings.optimizer_schedule_native_door import schedule_run
from bindings.classical_arrays_door import classical_input, classical_typed, classical_address, classical_is_array, classical_cast_kind, classical_numpy
from bindings.classical_attributes_door import classical_connect_numpy
from core.sequence_optimizer_inputs import sequence_optimizer_cast
from core.scaler_fit_impl import scaler_fill
from core.byte_lm_entry_table import ByteLmEntryKey
from core.entry_table import EntryRef
from sequence.opt_entry_abi import (
    seq_opt_ref, seq_opt_call, SO_INIT, SO_STEP, SO_LAMB_STEP, SO_ADAFACTOR_STEP, SO_STATE, SO_EXPORT, SO_IMPORT, SO_CLOSE,
)
from bindings.sequence_keys_unpack import sequence_keys
from bindings.optimizer_keys_unpack import (
    OPTIMIZER_OPN_SEQ_BINDING,
    OPTIMIZER_OPN_SEQ_CONSTRUCT,
    OPTIMIZER_OPN_SEQ_EXPORT,
    OPTIMIZER_OPN_SEQ_FINALIZE,
    OPTIMIZER_OPN_SEQ_HANDLE,
    OPTIMIZER_OPN_SEQ_IMPORT,
    OPTIMIZER_OPN_SEQ_RESIDENT,
    OPTIMIZER_OPN_SEQ_SLOT,
    OPTIMIZER_OPN_SEQ_STATE,
    OPTIMIZER_OPN_SEQ_STATE_SET,
    OPTIMIZER_OPN_SEQ_STEP,
    OPTIMIZER_OPN_SEQ_WORD,
    OPTIMIZER_OPN_SEQ_WORD_SET,
    OPTIMIZER_OPN_SCHEDULE_AT,
)
from bindings.py_balanced import py_item


def seq_key(entries: PythonObject) raises -> EntryRef:
    """The optimizer entry, resolved once per operation (lane
    entry-table-zero-alloc), of the table key (core/byte_lm_entry_table.mojo) of the
    x_sequence image these connected entries name: the bridge records it at
    import as `native_key` (python/mojolearn/_native_optimizer_bridge.py).
    The optimizer operations run through that image's published native
    entry (sequence/opt_entry_abi.mojo), not through its Python entries
    (lane calltable-byte-lm)."""
    var nk = sequence_keys()
    var key = py_get(entries, nk[].k_native_key)
    if key is None:
        raise Error("ImportError: the sequence optimizer binding publishes no native entry table")
    return seq_opt_ref(ByteLmEntryKey(Int(py=key[0]), Int(py=key[1])))


def seq_class(owner: PythonObject, context: PythonObject) raises -> Int:
    var nk = sequence_keys()
    var classes = py_item(context, nk[].k_sequence_owners)
    for i in range(len(classes)):  # driver-loop: over exact public sequence optimizer classes
        if py_type(owner) is classes[i]:
            return i
    raise Error("TypeError: sequence optimizer requires an exact native optimizer owner")


def seq_binding(mode: PythonObject, context: PythonObject, binding: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var nk = sequence_keys()
    var bindings = py_item(context, nk[].k_sequence_bindings)
    if binding is not None:
        for name in ["identical", "fast"]:  # driver-loop: over two compiled numeric-mode owners
            var entry = bindings[PythonObject(name)]
            if py_type(entry) is b.dict and binding is py_item(entry, nk[].k_module):
                return entry
        var extra = py_get(context, nk[].k_sequence_extra_bindings, Python.tuple())
        for i in range(len(extra)):  # driver-loop: fixed connected native buffer owners
            if py_type(extra[i]) is b.dict and binding is py_item(extra[i], nk[].k_module):
                return extra[i]
        raise Error("TypeError: sequence device/state requires its connected native binding")
    var selected = py_item(py_item(context, nk[].k_backend), nk[].k__DEFAULT_MODE) if mode is None else mode
    opt_scalar(selected, context)
    if py_type(selected) is not b.str:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    selected = py_call(py_call(selected.strip).lower)
    var selected_word = String(selected)
    if selected_word != "fast" and selected_word != "identical":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    answer_mode_object(selected, selected_word, MODE_FAMILY_NEURAL)
    var entries = bindings[selected]
    if py_type(entries) is b.str:
        raise Error("ImportError: " + String(entries))
    return entries


def seq_device_get(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var nk = sequence_keys()
    if py_type(value) is not py_item(context, nk[].k_sequence_device_class):
        raise Error("TypeError: sequence device tensor requires its exact native owner")
    return py_call(py_item(context, nk[].k_sequence_device_getters)[PythonObject(name)], value)


def seq_device(value: PythonObject, context: PythonObject) raises -> Bool:
    var nk = sequence_keys()
    if py_type(value) is not py_item(context, nk[].k_sequence_device_class):
        return False
    opt_scalar(seq_device_get(value, "h", context), context)
    _ = shape_count(device_shape(seq_device_get(value, "shape", context), context))
    _ = seq_binding(PythonObject(None), context, seq_device_get(value, "b", context))
    return True


def seq_list(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(value) is b.list or py_type(value) is b.tuple:
        return py_call(b.list, value)
    if not seq_device(value, context):
        opt_value(value, context)
    return Python.list(value)


def seq_shape(value: PythonObject, context: PythonObject) raises -> List[Int]:
    return seq_shape_known(value, seq_device(value, context), context)


def seq_shape_known(value: PythonObject, device: Bool, context: PythonObject) raises -> List[Int]:
    var buffer_pins = OptimizerInputOwners()
    try:
        # `seq_shape` for a value whose `seq_device` admission already ran in
        # this operation: the step reuses that one admission instead of
        # re-running the device owner/binding checks per parameter and gradient.
        if device:
            return device_shape(seq_device_get(value, "shape", context), context)
        var view = TensorView()
        if not tensor_view(value, view, buffer_pins):
            raise Error("TypeError: sequence parameter requires a native float32 buffer")
        return view.shape.copy()
    finally:
        _ = buffer_pins^



def seq_addr(value: PythonObject, count: Int, context: PythonObject, mut buffer_pins: OptimizerInputOwners, wide: Bool = False) raises -> Int:
    var view = TensorView()
    if not tensor_view(value, view, buffer_pins) or not view.c_contig or view.readonly or view.count() != count:
        raise Error("TypeError: sequence optimizer state requires writable contiguous buffers of the declared size")
    var format = base_format(view.fmt)
    if (wide and (view.itemsize != 8 or (format != "q" and format != "l"))) or (not wide and (format != "f" or view.itemsize != 4)):
        raise Error("TypeError: sequence optimizer state has the wrong native dtype")
    return view.addr


#: seq_array's host buffers of at least this many bytes are private anonymous
#: mappings (zero on first touch) instead of memset storage (see there).
comptime _SEQ_LAZY_ZERO_BYTES = 1 << 20


def seq_array(n: Int, context: PythonObject, fill: Float32 = 0.0, wide: Bool = False) raises -> PythonObject:
    if n < 0 or n > 9223372036854775807 // 8:
        raise Error("OverflowError: sequence optimizer buffer size is invalid")
    var nk = sequence_keys()
    var input = py_item(context, nk[].k_input_context)
    classical_connect_numpy(input)
    var result: PythonObject
    var nbytes = n * (8 if wide else 4)
    if nbytes >= _SEQ_LAZY_ZERO_BYTES and classical_numpy(input):
        # lane fb-optim (2026-11-17): the optimizer's host state copies (one
        # to three slots of n_total floats) are zeros that a resident or
        # device step never reads or writes (W_PRISTINE + the zeroed open
        # skip their upload). `classical_typed` builds them over a
        # bytearray, whose constructor memsets every byte: each page is
        # faulted in and zeroed at construction (~35 ms per 64 MB slot on
        # the L40S host; 0.8.55's np.zeros left the pages untouched). A
        # private anonymous mapping is zero filled by the kernel on first
        # touch only, as calloc, so untouched slots cost nothing at
        # construction or release. Below 1 MiB the memset is microseconds
        # and a mapping per buffer would cost more (one syscall, page
        # granularity), so small buffers keep `classical_typed`. Size
        # reasoning only: neighbors (any parameter count above 256 Ki
        # floats, any optimizer kind) take the same route.
        var storage = py_call(py_item(input, nk[].k_native_mmap_type), PythonObject(-1), PythonObject(nbytes),
                              py_item(input, nk[].k_native_mmap_private))
        result = py_call(py_item(input, nk[].k_ndarray), Python.tuple(n), dtype=input["int64" if wide else "float32"],
                         buffer=storage)
    else:
        result = classical_typed(Python.tuple(n), "<i8" if wide else "<f4", input)
    if not wide and bitcast[DType.uint32](fill) != 0:
        scaler_fill(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(result))), n, fill)
    return result


def seq_cast(value: PythonObject, context: PythonObject, target: Int = -1) raises -> PythonObject:
    opt_value(value, context)
    var nk = sequence_keys()
    var input = py_item(context, nk[].k_input_context)
    classical_connect_numpy(input)
    var source = classical_input(value, input)
    var n = Int(py=source.size)
    if target >= 0 and ((Int(py=source.ndim) > 1) or (n != 1 and n != target)):
        raise Error("ValueError: optimizer checkpoint cannot broadcast to its state slot")
    var result = classical_typed(source.shape if target < 0 else Python.tuple(target), "<f4", input)
    var shape = List[Int]()
    var strides = List[Int]()
    for i in range(len(source.shape)):  # driver-loop: over native buffer shape and stride words
        shape.append(Int(py=source.shape[i]))
        strides.append(Int(py=source.strides[i]))
    var item = Int(py=source.itemsize)
    var kind = classical_cast_kind(source, item, "sequence optimizer input", input)
    var native = classical_is_array(source, input) or Bool(py=source.dtype.isnative)
    sequence_optimizer_cast(Int(py=classical_address(source)), shape, strides, item, kind, native,
                            Int(py=classical_address(result)), n if target < 0 else target, target >= 0 and n == 1)
    _ = source^
    return result


def seq_pair(value: PythonObject, context: PythonObject) raises -> PythonObject:
    opt_value(value, context)
    var b = native_loaded_module("builtins")
    if py_type(value) is b.list or py_type(value) is b.tuple:
        opt_scalar(value, context)
        if len(value) != 2:
            raise Error("ValueError: optimizer configuration requires two values")
        return Python.tuple(value[0], value[1])
    var nk = sequence_keys()
    var input = py_item(context, nk[].k_input_context)
    classical_connect_numpy(input)
    var source = classical_input(value, input)
    if Int(py=source.ndim) != 1 or Int(py=source.size) != 2:
        raise Error("ValueError: optimizer configuration requires two values")
    # Native ndarray/Array scalar reads cannot call user conversion hooks.
    if classical_is_array(source, input):
        var packed = py_call(native_loaded_module("builtins").memoryview, source._store)
        return Python.tuple(packed[0], packed[1])
    return Python.tuple(source[0], source[1])


def seq_fields(owner: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = seq_class(owner, context)
    var fields = py_fields(owner)
    var b = native_loaded_module("builtins")
    # Key admission kept per dictionary under the dict watcher
    # (bindings/native_class_admission.mojo); Python 3.10/3.11 scan per call.
    if py_type(fields) is not b.dict or not dict_keys_admitted(fields):
        var keys = py_call(b.list, fields)
        for i in range(len(keys)):  # driver-loop: over stored owner field names before native dictionary lookups
            if py_type(keys[i]) is not b.str:
                raise Error("TypeError: sequence optimizer metadata requires native string keys")
        if py_type(fields) is b.dict:
            dict_keys_admit(fields)
    for name in ["numeric_mode", "lr", "n_total", "_kind", "_flags", "_fp", "beta2_decay", "d", "weight_decay", "eps", "betas", "max_grad_norm"]:  # driver-loop: over fixed native optimizer configuration fields
        opt_scalar(py_get(fields, PythonObject(name)), context)
    var nk = sequence_keys()
    opt_schedule(py_get(fields, nk[].k_lr_schedule), context)
    for name in ["_words", "_handles", "_state", "_af_state", "_sc", "options"]:  # driver-loop: over stored native optimizer buffer containers
        opt_value(py_get(fields, PythonObject(name)), context)
    return fields


def seq_slots(fields: PythonObject, kind: Int, context: PythonObject, mut buffer_pins: OptimizerInputOwners, padded: Bool = False) raises -> PythonObject:
    var result = Python.list()
    var b = native_loaded_module("builtins")
    var nk = sequence_keys()
    if kind == 5:
        var states = py_item(fields, nk[].k__af_state)
        var params = py_item(fields, nk[].k_params)
        if (py_type(states) is not b.list and py_type(states) is not b.tuple) or len(states) != len(params):
            raise Error("ValueError: Adafactor state must have one native dictionary per parameter")
        for i in range(len(states)):  # driver-loop: over factored state slots per parameter tensor
            var state = states[i]
            if not opt_exact(py_type(state), py_item(context, nk[].k_mapping_types)):
                raise Error("TypeError: Adafactor state requires native dictionaries")
            var shape = seq_shape(params[i], context)
            var first = seq_addr(state["row_var" if len(shape) == 2 else "variance"], shape[0], context, buffer_pins)
            var second = seq_addr(py_item(state, nk[].k_col_var), shape[1], context, buffer_pins) if len(shape) == 2 else 0
            _ = py_call(result.append, first)
            _ = py_call(result.append, second)
            if padded:
                _ = py_call(result.append, 0)
    else:
        var states = py_item(fields, nk[].k__state)
        var expected = 2 if kind == 6 else 3
        if (py_type(states) is not b.list and py_type(states) is not b.tuple) or len(states) != expected:
            raise Error("ValueError: sequence optimizer state slot count changed")
        for i in range(expected):  # driver-loop: over two or three optimizer state slots
            _ = py_call(result.append, seq_addr(states[i], Int(py=py_item(fields, nk[].k_n_total)), context, buffer_pins))
        if padded and expected == 2:
            _ = py_call(result.append, 0)
    return result


def _seq_real(value: PythonObject, label: String) raises -> PythonObject:
    # A real number for an optimizer option as a Python float; a str, None or
    # an object without __float__ is refused as TypeError naming the option
    # (the ops.Session `_float` pattern), never parsed.
    var real: Float64
    try:
        real = Float64(py=value)
    except:
        raise Error("TypeError: " + label + " must be a real number")
    # an integer above 2^53 rounds once to float32 (bindings/optimizer_real_unpack.mojo)
    return PythonObject(optimizer_real(value, real))


def seq_construct(owner: PythonObject, values: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var buffer_pins = OptimizerInputOwners()
    try:
        var kind = seq_class(owner, context)
        var b = native_loaded_module("builtins")
        var fields = py_fields(owner)
        var params = seq_list(values[0], context)
        var options = values[2]
        opt_value(options, context)
        if py_type(options) is b.tuple:
            var names = Python.tuple(
                Python.tuple("alpha", "eps", "weight_decay", "momentum", "centered"),
                Python.tuple("lr_decay", "weight_decay", "initial_accumulator_value", "eps"),
                Python.tuple("betas", "weight_decay"), Python.tuple("betas", "eps", "weight_decay"),
                Python.tuple("betas", "eps", "weight_decay", "momentum_decay", "decoupled_weight_decay"),
                Python.tuple("beta2_decay", "eps", "d", "weight_decay", "maximize"),
                Python.tuple("bias_correction", "betas", "eps", "weight_decay", "grad_averaging", "max_grad_norm", "trust_clip", "always_adapt"))
            if len(options) != len(names[kind]):
                raise Error("TypeError: invalid sequence optimizer constructor envelope")
            var config = Python.dict()
            for i in range(len(options)):  # driver-loop: over fixed native constructor argument names
                py_setitem(config, names[kind][i], options[i])
            options = config
        if py_type(options) is not b.dict:
            raise Error("TypeError: optimizer options require a native dictionary")
        opt_scalar(values[1], context)
        opt_scalar(mode, context)
        var selected = seq_binding(mode, context, PythonObject(None))
        var device = False
        var sizes = List[Int]()
        var shapes = Python.list()
        var nk = sequence_keys()
        for i in range(len(params)):  # driver-loop: over public parameter buffer owners
            var dev = seq_device(params[i], context)
            if i == 0:
                device = dev
                if dev:
                    selected = seq_binding(mode, context, seq_device_get(params[i], "b", context))
            if dev != device or (dev and seq_device_get(params[i], "b", context) is not py_item(selected, nk[].k_module)):
                raise Error("TypeError: sequence params must all use host buffers or one native device binding")
            var shape = seq_shape(params[i], context)
            var n = shape_count(shape)
            if not dev:
                var view = TensorView()
                _ = tensor_view(params[i], view, buffer_pins)
                if base_format(view.fmt) != "f" or view.itemsize != 4 or not view.c_contig:
                    raise Error("TypeError: sequence params must be C-contiguous float32 buffers")
            if kind == 6 and n < 1:
                raise Error("TypeError: LAMB params must be nonempty")
            if kind == 5 and len(shape) != 1 and len(shape) != 2:
                raise Error("NotImplementedError: Adafactor parameters must have one or two dimensions")
            sizes.append(n)
            _ = py_call(shapes.append, native_shape_pack(shape))
        if device:
            var prefix = String("adafactor") if kind == 5 else (String("lamb") if kind == 6 else String("optimizer"))
            for suffix in ["_resident_open", "_resident_step", "_resident_step_dev"]:  # driver-loop: over required resident optimizer entries
                if py_get(selected, PythonObject(prefix + suffix)) is None:
                    raise Error("RuntimeError: this binding does not support resident optimizer parameters")
        py_setitem(fields, nk[].k_params, params)
        py_setitem(fields, nk[].k_numeric_mode, mode)
        py_setitem(fields, nk[].k__dev_b, py_item(selected, nk[].k_module) if device else PythonObject(None))
        py_setitem(fields, nk[].k__res_b, PythonObject(None))
        py_setitem(fields, nk[].k_lr, _seq_real(values[1], "lr"))
        var extra = List[Int]()
        var fp = Python.list(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        var flags = 0
        if kind < 5:
            py_setitem(fields, nk[].k_options, py_call(b.dict, options))
            opt_scalar(py_item(options, nk[].k_weight_decay), context)
            py_setitem(fp, PythonObject(3), _seq_real(py_item(options, nk[].k_weight_decay), "weight_decay"))
            if kind == 0 or kind == 1:
                for name in ["eps", "alpha" if kind == 0 else "lr_decay", "momentum" if kind == 0 else "initial_accumulator_value"]:  # driver-loop: over RMSprop or Adagrad scalar options
                    opt_scalar(options[PythonObject(name)], context)
                py_setitem(fp, PythonObject(2), _seq_real(py_item(options, nk[].k_eps), "eps"))
                py_setitem(fp, PythonObject(1 if kind == 0 else 0), _seq_real(options["alpha" if kind == 0 else "lr_decay"], String("alpha" if kind == 0 else "lr_decay")))
                py_setitem(fp, PythonObject(4 if kind == 0 else 5), _seq_real(options["momentum" if kind == 0 else "initial_accumulator_value"], String("momentum" if kind == 0 else "initial_accumulator_value")))
                if kind == 0:
                    opt_scalar(py_item(options, nk[].k_centered), context)
                    flags = Int(Bool(py=py_item(options, nk[].k_centered)))
            else:
                var betas = seq_pair(py_item(options, nk[].k_betas), context)
                py_setitem(fp, PythonObject(0), _seq_real(betas[0], "betas"))
                py_setitem(fp, PythonObject(1), _seq_real(betas[1], "betas"))
                if kind == 3 and not (0.0 <= Float64(py=fp[0]) < 1.0 and 0.0 <= Float64(py=fp[1]) < 1.0):
                    raise Error("ValueError: Adamax betas must lie in [0, 1)")
                if kind != 2:
                    opt_scalar(py_item(options, nk[].k_eps), context)
                    py_setitem(fp, PythonObject(2), _seq_real(py_item(options, nk[].k_eps), "eps"))
                if kind == 4:
                    opt_scalar(py_item(options, nk[].k_momentum_decay), context)
                    opt_scalar(py_item(options, nk[].k_decoupled_weight_decay), context)
                    py_setitem(fp, PythonObject(4), _seq_real(py_item(options, nk[].k_momentum_decay), "momentum_decay"))
                    if Float64(py=fp[4]) < 0:
                        raise Error("ValueError: NAdam momentum_decay must be >= 0")
                    flags = Int(Bool(py=py_item(options, nk[].k_decoupled_weight_decay)))
            py_setitem(fields, nk[].k__kind, PythonObject(3 if kind == 0 else (4 if kind == 1 else kind + 5)))
            py_setitem(fields, nk[].k__flags, PythonObject(flags))
            py_setitem(fields, nk[].k__fp, fp)
        elif kind == 5:
            for name in ["beta2_decay", "d", "weight_decay", "maximize"]:  # driver-loop: over Adafactor scalar options
                opt_scalar(options[PythonObject(name)], context)
            if Bool(py=py_item(options, nk[].k_maximize)):
                raise Error("NotImplementedError: Adafactor maximize is not implemented")
            var beta2_decay = Float64(py=_seq_real(py_item(options, nk[].k_beta2_decay), "beta2_decay"))
            var d = Float64(py=_seq_real(py_item(options, nk[].k_d), "d"))
            var decay = Float64(py=_seq_real(py_item(options, nk[].k_weight_decay), "weight_decay"))
            if beta2_decay > 0 or d < 1 or Float64(py=py_item(fields, nk[].k_lr)) < 0 or decay < 0:
                raise Error("ValueError: Adafactor requires beta2_decay <= 0, d >= 1 and nonnegative lr/weight_decay")
            for name in ["beta2_decay", "d", "weight_decay"]:  # driver-loop: over stored Adafactor scalar options
                py_setitem(fields, PythonObject(name), _seq_real(options[PythonObject(name)], String(name)))
            var eps = seq_pair(py_item(options, nk[].k_eps), context)
            py_setitem(fields, nk[].k_eps, Python.tuple(1.1920928955078125e-7 if eps[0] is None else Float64(py=_seq_real(eps[0], "eps")), _seq_real(eps[1], "eps")))
        else:
            for name in ["eps", "weight_decay", "max_grad_norm", "trust_clip", "always_adapt", "grad_averaging", "bias_correction"]:  # driver-loop: over LAMB scalar options
                opt_scalar(options[PythonObject(name)], context)
            var betas = seq_pair(py_item(options, nk[].k_betas), context)
            var first = Float64(py=_seq_real(betas[0], "betas"))
            var second = Float64(py=_seq_real(betas[1], "betas"))
            if not (0.0 <= first < 1.0 and 0.0 <= second < 1.0):
                raise Error("ValueError: LAMB betas must lie in [0, 1)")
            py_setitem(fields, nk[].k_betas, Python.tuple(first, second))
            for name in ["eps", "weight_decay"]:  # driver-loop: over LAMB floating-point configuration
                py_setitem(fields, PythonObject(name), _seq_real(options[PythonObject(name)], String(name)))
            py_setitem(fields, nk[].k_max_grad_norm, PythonObject(None) if py_item(options, nk[].k_max_grad_norm) is None else _seq_real(py_item(options, nk[].k_max_grad_norm), "max_grad_norm"))
            for name in ["trust_clip", "always_adapt", "grad_averaging", "bias_correction"]:  # driver-loop: over LAMB native flag words
                extra.append(Int(Bool(py=options[PythonObject(name)])))
            extra.append(Int(py_item(options, nk[].k_max_grad_norm) is not None))
        var words = seq_array(9, context, wide=True)
        var nh = len(params) if kind == 5 else 1
        var handles = seq_array(nh, context, wide=True)
        var hp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(handles)))
        for i in range(nh):  # driver-loop: over per-parameter resident handle metadata
            hp[i] = -1
        py_setitem(fields, nk[].k__words, words)
        py_setitem(fields, nk[].k__handles, handles)
        # p = [words, ip...] (seq_optimizer_init's words and ip)
        var init: List[Int] = [Int(py=classical_address(words)), 2 if kind == 5 else (1 if kind == 6 else 0), len(params)]
        init.extend(sizes^)
        init.extend(extra^)
        var n = seq_opt_call(seq_key(selected), SO_INIT, List[Int](), init, List[Float64]())
        if kind < 5 and n < 1:
            raise Error("ValueError: sequence optimizer has no parameters")
        py_setitem(fields, nk[].k_n_total, PythonObject(n))
        var states = Python.list()
        if kind == 5:
            for i in range(len(params)):  # driver-loop: over Adafactor moment buffer owners per parameter
                var shape = shapes[i]
                var state = Python.dict(row_var=seq_array(Int(py=shape[0]), context), col_var=seq_array(Int(py=shape[1]), context)) if len(shape) == 2 else Python.dict(variance=seq_array(Int(py=shape[0]), context))
                _ = py_call(states.append, state)
            py_setitem(fields, nk[].k__af_state, states)
        else:
            for i in range(2 if kind == 6 else 3):  # driver-loop: over optimizer moment buffer owners
                # lane fb-optim (2026-11-17): Adagrad's accumulator is its
                # admitted float initial_accumulator_value (fp[5], already
                # `_seq_real`) rounded once to float32 and filled natively;
                # it was `seq_cast` of the scalar broadcast to n_total, a
                # per-element index/stride/byte-assembly loop (~290 ms for
                # 16 Mi floats; board adagrad 9.4 -> 417 ms) over a freshly
                # memset array. The value 0 (the default) keeps the untouched
                # zero slot, as 0.8.55 (`if init: fill`).
                _ = py_call(states.append, seq_array(n, context, Float32(Float64(py=fp[5]))) if kind == 1 and i == 1 else seq_array(n, context))
            py_setitem(fields, nk[].k__state, states)
            py_setitem(fields, nk[].k__sc, seq_array(2 if kind == 6 else 3, context, 1.0))
        return PythonObject(None)
    finally:
        _ = buffer_pins^



def seq_step(owner: PythonObject, fields: PythonObject, params: PythonObject, gradients: PythonObject, kind: Int,
             entries: PythonObject, words: Int, handles: Int, context: PythonObject) raises -> PythonObject:
    var buffer_pins = OptimizerInputOwners()
    try:
        # `params` is the operation's one `seq_list(fields["params"])`, read by
        # sequence_optimizer_operation before the step (not read twice).
        var grads = seq_list(gradients, context)
        if len(grads) != len(params):
            raise Error("ValueError: sequence optimizer needs one gradient per parameter")
        var nk = sequence_keys()
        var dev = py_get(fields, nk[].k__dev_b) is not None
        # Every request List is sized once for the step from the parameter
        # count J (no growth reallocation per appended word): addrs holds the
        # J parameters, J gradients and up to 2J + 1 slot words; ip holds the
        # four head words, J kinds, J sizes and up to 2J shape words.
        var count = len(params)
        var addrs = List[Int](capacity=4 * count + 1)
        var ga = List[Int](capacity=count)
        var kinds = List[Int](capacity=count)
        var sizes = List[Int](capacity=count)
        var rows = List[Int](capacity=count if kind == 5 else 0)
        var cols = List[Int](capacity=count if kind == 5 else 0)
        var keep = Python.list()
        var total = 0
        var input = py_item(context, nk[].k_input_context)
        classical_connect_numpy(input)
        for i in range(len(params)):  # driver-loop: over parameter/gradient buffer owners
            var parameter = params[i]
            var parameter_device = seq_device(parameter, context)
            var shape = seq_shape_known(parameter, parameter_device, context)
            var n = shape_count(shape)
            if n > 9223372036854775807 - total:
                raise Error("OverflowError: sequence parameter count exceeds native indexing")
            total += n
            sizes.append(n)
            if kind == 5:
                if len(shape) != 1 and len(shape) != 2:
                    raise Error("ValueError: Adafactor parameter rank changed")
                rows.append(shape[0])
                cols.append(shape[1] if len(shape) == 2 else 0)
            if parameter_device != dev:
                raise Error("TypeError: sequence parameter host/device ownership changed")
            if dev:
                if seq_device_get(parameter, "b", context) is not py_item(entries, nk[].k_module):
                    raise Error("TypeError: sequence parameter binding ownership changed")
                addrs.append(Int(py=seq_device_get(parameter, "h", context)))
            else:
                addrs.append(seq_addr(parameter, n, context, buffer_pins))
            var grad = grads[i]
            if seq_device(grad, context):
                if not dev or seq_device_get(grad, "b", context) is not py_item(entries, nk[].k_module) or not same_shape(seq_shape_known(grad, True, context), shape):
                    raise Error("ValueError: device gradient must match its parameter shape and binding")
                ga.append(Int(py=seq_device_get(grad, "h", context)))
                kinds.append(0)
                _ = py_call(keep.append, grad)
            else:
                opt_value(grad, context)
                var source = classical_input(grad, input)
                var view = TensorView()
                _ = tensor_view(source, view, buffer_pins)
                if not same_shape(view.shape, shape):
                    raise Error("ValueError: gradient shape must match its parameter")
                if base_format(view.fmt) == "d" or (dev and base_format(view.fmt) != "f"):
                    raise Error("ValueError: Adafactor gradient must be float32 of its parameter shape" if kind == 5 and not dev else "TypeError: sequence gradients must be float32; float64 gradients are refused")
                var prepared = source if base_format(view.fmt) == "f" and view.c_contig else seq_cast(source, context)
                ga.append(Int(py=classical_address(prepared)))
                kinds.append(1 if dev else 0)
                _ = py_call(keep.append, prepared)
        if total != Int(py=py_item(fields, nk[].k_n_total)):
            raise Error("ValueError: sequence parameter size changed after construction")
        addrs.extend(ga^)
        # p = [words, handles, ip...] with ip = [J, dev, kinds.., sizes.., (rows.., cols.. | kind, flags)]
        var ip = List[Int](capacity=4 + 4 * count)
        ip.append(words)
        ip.append(handles)
        ip.append(count)
        ip.append(Int(dev))
        ip.extend(kinds^)
        ip.extend(sizes^)
        var slots = seq_slots(fields, kind, context, buffer_pins)
        if kind == 5:
            for i in range(len(params)):  # driver-loop: over Adafactor first moment slot addresses
                addrs.append(Int(py=slots[i * 2]))
            for i in range(len(params)):  # driver-loop: over Adafactor second moment slot addresses
                addrs.append(Int(py=slots[i * 2 + 1]) if Int(py=slots[i * 2 + 1]) != 0 else Int(py=slots[i * 2]))
            ip.extend(rows^)
            ip.extend(cols^)
        else:
            for i in range(len(slots)):  # driver-loop: over two or three optimizer state slot addresses
                addrs.append(Int(py=slots[i]))
            addrs.append(seq_addr(py_item(fields, nk[].k__sc), 2 if kind == 6 else 3, context, buffer_pins))
            if kind < 5:
                ip.append(Int(py=py_item(fields, nk[].k__kind)))
                ip.append(Int(py=py_item(fields, nk[].k__flags)))
        var schedule = py_get(fields, nk[].k_lr_schedule)
        if schedule is not None:
            var t = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=words)[0]
            py_setitem(fields, nk[].k_lr, schedule_run(OPTIMIZER_OPN_SCHEDULE_AT, schedule, Python.tuple(Int(t) + 1), context))
        # fp: the same Float64 conversion of each value the Python entry made
        var fp = List[Float64](capacity=7)  # lr and at most six optimizer scalars
        fp.append(Float64(py=py_item(fields, nk[].k_lr)))
        if kind < 5:
            var stored = py_item(fields, nk[].k__fp)
            for i in range(len(stored)):  # small-loop(the six element-wise optimizer scalars)
                fp.append(Float64(py=stored[i]))
        elif kind == 5:
            fp.append(Float64(py=py_item(fields, nk[].k_beta2_decay)))
            fp.append(Float64(py=py_item(fields, nk[].k_eps)[0]))
            fp.append(Float64(py=py_item(fields, nk[].k_eps)[1]))
            fp.append(Float64(py=py_item(fields, nk[].k_d)))
            fp.append(Float64(py=py_item(fields, nk[].k_weight_decay)))
        else:
            fp.append(Float64(py=py_item(fields, nk[].k_betas)[0]))
            fp.append(Float64(py=py_item(fields, nk[].k_betas)[1]))
            fp.append(Float64(py=py_item(fields, nk[].k_eps)))
            fp.append(Float64(py=py_item(fields, nk[].k_weight_decay)))
            fp.append(Float64(1.0) if py_item(fields, nk[].k_max_grad_norm) is None else Float64(py=py_item(fields, nk[].k_max_grad_norm)))
        py_setitem(fields, nk[].k__res_b, py_item(entries, nk[].k_module))
        _ = seq_opt_call(seq_key(entries), SO_ADAFACTOR_STEP if kind == 5 else (SO_LAMB_STEP if kind == 6 else SO_STEP),
                         addrs, ip, fp)
        _ = keep^
        return owner
    finally:
        _ = buffer_pins^



def sequence_optimizer_operation(operation: Int, owner: PythonObject, values: PythonObject,
                                 mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var buffer_pins = OptimizerInputOwners()
    try:
        if operation == OPTIMIZER_OPN_SEQ_CONSTRUCT:
            return seq_construct(owner, values, mode, context)
        var kind = seq_class(owner, context)
        var b = native_loaded_module("builtins")
        var nk = sequence_keys()
        if operation == OPTIMIZER_OPN_SEQ_WORD or operation == OPTIMIZER_OPN_SEQ_WORD_SET:
            # One state word reads or writes only the `_words` buffer: admit the
            # owner class, its string keys and that buffer (seq_addr runs the same
            # tensor admission), not the whole optimizer, its binding, the params
            # list and the handle buffer.
            var word_fields = py_fields(owner)
            if py_type(word_fields) is not b.dict or not dict_keys_admitted(word_fields):
                var keys = py_call(b.list, word_fields)
                for i in range(len(keys)):  # driver-loop: over stored owner field names before native dictionary lookups
                    if py_type(keys[i]) is not b.str:
                        raise Error("TypeError: sequence optimizer metadata requires native string keys")
                if py_type(word_fields) is b.dict:
                    dict_keys_admit(word_fields)
            var word_ptr = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=seq_addr(py_item(word_fields, nk[].k__words), 9, context, buffer_pins, True))
            opt_scalar(values, context)
            var word_index = Int(py=values[0])
            if word_index < 0 or word_index >= 9:
                raise Error("ValueError: optimizer word index out of range")
            if operation == OPTIMIZER_OPN_SEQ_WORD_SET:
                word_ptr[word_index] = Int64(Int(py=py_call(b.int, values[1])))
                return PythonObject(None)
            return PythonObject(word_ptr[word_index] != 0) if Bool(py=values[1]) else PythonObject(Int(word_ptr[word_index]))
        var fields = py_fields(owner) if operation == OPTIMIZER_OPN_SEQ_FINALIZE else seq_fields(owner, context)
        if operation == OPTIMIZER_OPN_SEQ_FINALIZE and py_get(fields, nk[].k__res_b) is None:
            return PythonObject(None)
        var entries = seq_binding(py_item(fields, nk[].k_numeric_mode), context, py_get(fields, nk[].k__res_b) if py_get(fields, nk[].k__res_b) is not None else py_get(fields, nk[].k__dev_b))
        if operation == OPTIMIZER_OPN_SEQ_BINDING:
            return py_item(seq_binding(py_item(fields, nk[].k_numeric_mode), context, py_get(fields, nk[].k__dev_b)), nk[].k_module)
        var params = seq_list(py_item(fields, nk[].k_params), context)
        var nh = len(params) if kind == 5 else 1
        var words = seq_addr(py_item(fields, nk[].k__words), 9, context, buffer_pins, True)
        var handles = seq_addr(py_item(fields, nk[].k__handles), nh, context, buffer_pins, True)
        var w = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=words)
        if operation == OPTIMIZER_OPN_SEQ_WORD or operation == OPTIMIZER_OPN_SEQ_WORD_SET:
            opt_scalar(values, context)
            var index = Int(py=values[0])
            if index < 0 or index >= 9:
                raise Error("ValueError: optimizer word index out of range")
            if operation == OPTIMIZER_OPN_SEQ_WORD_SET:
                w[index] = Int64(Int(py=py_call(b.int, values[1])))
                return PythonObject(None)
            return PythonObject(w[index] != 0) if Bool(py=values[1]) else PythonObject(Int(w[index]))
        if operation == OPTIMIZER_OPN_SEQ_FINALIZE:
            try:
                var close: List[Int] = [words, handles, nh]
                _ = seq_opt_call(seq_key(entries), SO_CLOSE, List[Int](), close, List[Float64]())
            except:
                return PythonObject(None)
            py_setitem(fields, nk[].k__res_b, PythonObject(None))
            return PythonObject(None)
        if operation == OPTIMIZER_OPN_SEQ_RESIDENT:
            return PythonObject(w[2] >= 0)
        if operation == OPTIMIZER_OPN_SEQ_HANDLE:
            if w[2] < 0:
                return PythonObject(None)
            var hp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=handles)
            return PythonObject(Int(hp[0]))
        if operation == OPTIMIZER_OPN_SEQ_STATE_SET:
            opt_value(values[0], context)
            var proposed = py_call(b.dict, fields)
            py_setitem(proposed, "_af_state" if kind == 5 else "_state", values[0])
            _ = seq_slots(proposed, kind, context, buffer_pins)
            py_setitem(fields, "_af_state" if kind == 5 else "_state", values[0])
            w[4] = 1
            w[5] = 1
            w[6] = 0
            return PythonObject(None)
        if operation == OPTIMIZER_OPN_SEQ_STEP:
            var selected = seq_binding(py_item(fields, nk[].k_numeric_mode), context, py_get(fields, nk[].k__dev_b))
            if w[2] >= 0 and py_item(selected, nk[].k_module) is not py_item(entries, nk[].k_module):
                raise Error("ValueError: resident optimizer state must keep its owning numeric-mode binding")
            return seq_step(owner, fields, params, values[0], kind, selected, words, handles, context)
        if operation == OPTIMIZER_OPN_SEQ_STATE or operation == OPTIMIZER_OPN_SEQ_SLOT:
            var slots = seq_slots(fields, kind, context, buffer_pins, True)
            var slot_words = List[Int]()
            for i in range(len(slots)):  # driver-loop: over padded optimizer state slot addresses
                slot_words.append(Int(py=slots[i]))
            var state_words: List[Int] = [words, handles]
            _ = seq_opt_call(seq_key(entries), SO_STATE, slot_words, state_words, List[Float64]())
            var state = fields["_af_state" if kind == 5 else "_state"]
            if operation == OPTIMIZER_OPN_SEQ_SLOT:
                opt_scalar(values[0], context)
                return state[values[0]]
            return state
        if operation == OPTIMIZER_OPN_SEQ_EXPORT or operation == OPTIMIZER_OPN_SEQ_IMPORT:
            if kind == 5:
                raise Error("TypeError: Adafactor exposes state, without an optimizer checkpoint method")
            var slots = seq_slots(fields, kind, context, buffer_pins)
            var k = len(slots)
            var n = Int(py=py_item(fields, nk[].k_n_total))
            var sc_len = 2 if kind == 6 else 3
            var sc_addr = seq_addr(py_item(fields, nk[].k__sc), sc_len, context, buffer_pins)
            var out = Python.list()
            var sc = PythonObject(None)
            var t = Int(w[0])
            var sc_t = 0
            var sc_given = False
            if operation == OPTIMIZER_OPN_SEQ_EXPORT:
                for i in range(k):  # driver-loop: over independent checkpoint output buffers
                    _ = py_call(out.append, seq_array(n, context))
                sc = seq_array(sc_len, context)
            else:
                var state = values[0]
                opt_value(state, context)
                if not opt_exact(py_type(state), py_item(context, nk[].k_mapping_types)):
                    raise Error("TypeError: optimizer checkpoint requires a native dictionary")
                opt_scalar(py_item(state, nk[].k_t), context)
                opt_scalar(py_item(state, nk[].k_lr), context)
                var saved = py_item(state, nk[].k_state)
                if py_type(saved) is not b.list and py_type(saved) is not b.tuple:
                    raise Error("TypeError: checkpoint state requires a native list or tuple")
                for i in range(min(k, len(saved))):  # driver-loop: over checkpoint source buffer owners
                    _ = py_call(out.append, seq_cast(saved[i], context, n))
                t = Int(py=py_call(b.int, py_item(state, nk[].k_t)))
                var scalar_state = py_get(state, nk[].k_scalars)
                sc_given = scalar_state is not None
                if sc_given:
                    if (py_type(scalar_state) is not b.list and py_type(scalar_state) is not b.tuple) or len(scalar_state) != 2:
                        raise Error("TypeError: checkpoint scalars require a native pair")
                    opt_scalar(scalar_state[0], context)
                    sc_t = Int(py=py_call(b.int, scalar_state[0]))
                    sc = seq_cast(scalar_state[1], context, sc_len)
                else:
                    sc = py_item(fields, nk[].k__sc)
                py_setitem(fields, nk[].k_lr, py_call(b.float, py_item(state, nk[].k_lr)))
            var addrs = List[Int]()
            for i in range(len(slots)):  # driver-loop: over optimizer state slot addresses
                addrs.append(Int(py=slots[i]))
            for i in range(len(out)):  # driver-loop: over state copy buffer addresses
                addrs.append(Int(py=classical_address(out[i])))
            addrs.append(sc_addr)
            addrs.append(Int(py=classical_address(sc)))
            var copy_words: List[Int] = [words, handles, k, n, sc_len, len(out), t, Int(sc_given), sc_t]
            _ = seq_opt_call(seq_key(entries), SO_EXPORT if operation == OPTIMIZER_OPN_SEQ_EXPORT else SO_IMPORT, addrs, copy_words,
                             List[Float64]())
            _ = out  # keep the copy buffers alive through the native call above
            _ = sc
            if operation == OPTIMIZER_OPN_SEQ_IMPORT:
                return owner
            return Python.dict(t=Int(w[0]), lr=py_item(fields, nk[].k_lr), state=out, options=py_call(b.dict, py_get(fields, nk[].k_options, Python.dict())), scalars=Python.tuple(Int(w[1]), sc))
        raise Error("TypeError: unknown sequence optimizer native operation")
    finally:
        _ = buffer_pins^
