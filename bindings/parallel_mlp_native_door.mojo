# SPDX-License-Identifier: Apache-2.0
"""Native small-MLP state and gradient operations for parallel owners.

The fixed architecture and arithmetic entries are the existing supported MLP.
This module owns admission, snapshots and kernel sequences; it introduces no
new numerical kernels or topology-dependent reduction order.
"""
from std.python import Python, PythonObject
from core.parallel_mlp_native_impl import mlp_check_finite, mlp_check_optimizer_state, mlp_targets_into
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from std.math import isfinite
from std.memory import unsafe_memcpy
from training.optimizer_device_plan import optimizer_device_setting
from bindings.generation_buffers_door import GENERATION_BOUNDARY, generation_array, generation_copy, py_ints
from bindings.py_balanced import py_call, py_get, py_setitem, py_type
from bindings.py_cpython_table import cpython_table
from bindings.parallel_byte_native_door import _parallel_array, _parallel_view, _parallel_int, _parallel_float, _parallel_schedule_check
from bindings.native_transport_door import _transport_json
from bindings.native_json_door import native_json_parse
from bindings.small_mlp_native_door import (
    small_mlp_call, small_mlp_from_state, small_mlp_grads_open, small_mlp_grads_close, small_mlp_grads_rows,
)
from bindings.small_mlp_keys_unpack import SMALL_MLP_OPN_LOAD_STATE_DICT, SMALL_MLP_OPN_STATE_DICT
from core.shared_slot import SharedSlot
from std.python._cpython import GILReleased
from core.entry_table import EntryRef, entry_ref_call, entry_ref_unresolved
from core.samba_entry_table import (
    SambaEntryKey, samba_ref, samba_vendor_word, SAMBA_TABLE_TRAINING, SAMBA_PRIM_ACCUMULATE,
    SAMBA_PRIM_OPTIMIZER_STEP,
)
from bindings.parallel_mlp_native_run import _mlp_names, _mlp_dims
from bindings.parallel_keys_unpack import (
    parallel_keys,
    PARALLEL_MLP_OPN_APPLY,
    PARALLEL_MLP_OPN_FROM_STATE,
    PARALLEL_MLP_OPN_GRADIENT,
    PARALLEL_MLP_OPN_INITIALIZE,
    PARALLEL_MLP_OPN_ORDERED_SUM,
    PARALLEL_MLP_OPN_PUBLISH,
    PARALLEL_MLP_OPN_SNAPSHOT,
    PARALLEL_MLP_OPN_VALIDATE,
)
from bindings.py_keys import py_op_code
from bindings.py_balanced import py_item


struct ParallelMlpBoundary(Defaultable, Movable):
    var bi: PythonObject
    var model_cls: PythonObject
    var opt_cls: PythonObject
    var lock: PythonObject
    var entries: PythonObject
    var environ: PythonObject
    var mode: String
    # The training image's neural primitive entry (core/samba_entry_table.mojo,
    # SAMBA_TABLE_TRAINING), resolved ONCE at initialize for the admitted
    # (mode, vendor) key. `accumulate` (op 9) and `optimizer_step` (op 25)
    # are called through it with plain words; they were Python attribute
    # fetches of the binding called through CPython (lane
    # entangled-native-modules, 2026-11-12). A resolution refusal is kept and
    # raised at the first call, so initialize refuses exactly what it did.
    var primitives: EntryRef
    var primitives_error: String

    def __init__(out self):
        self.bi = PythonObject(None)
        self.model_cls = PythonObject(None)
        self.opt_cls = PythonObject(None)
        self.lock = PythonObject(None)
        self.entries = PythonObject(None)
        self.environ = PythonObject(None)
        self.mode = String()
        self.primitives = entry_ref_unresolved()
        self.primitives_error = String("parallel MLP: training primitives are not connected")


# V2: the layout gained the resolved primitive entry (V1 held Python entries).
comptime PARALLEL_MLP_BOUNDARY = SharedSlot[ParallelMlpBoundary, "MojolearnParallelMlpBoundaryV2"]


def _mlp_primitives() raises -> EntryRef:
    var b = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()
    if b[].primitives.word == 0:
        raise Error(b[].primitives_error)
    return b[].primitives


def mlp_initialize(model_cls: PythonObject, opt_cls: PythonObject, binding: PythonObject,
                   environ: PythonObject, mode: String, vendor: String) raises:
    var b = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()
    b[].bi = Python.import_module("builtins")
    b[].model_cls = model_cls
    b[].opt_cls = opt_cls
    b[].lock = Python.import_module("_thread").RLock
    b[].environ = environ
    b[].mode = mode
    b[].entries = b[].bi.dict()
    for name in [String("training_numeric_mode"), String("training_vendor"),  # small-loop(the fixed training entry names)
                 String("optimizer_parallel_available"), String("accumulate_parallel_available"),
                 String("clip_parallel_available")]:
        var entry = PythonObject(None)
        try:
            entry = py_call(b[].bi.getattr, binding, name)
        except:
            pass
        if entry is not None and py_type(entry) is not py_type(b[].bi.len):
            raise Error("parallel MLP: training entries must be compiled native functions")
        py_setitem(b[].entries, name, entry)
    var nk = parallel_keys()
    if Int(py=py_item(b[].entries, nk[].k_training_numeric_mode)()) != (1 if mode == "identical" else 0) or String(py_item(b[].entries, nk[].k_training_vendor)()) != vendor:
        raise Error("parallel MLP: compiled mode/vendor admission mismatch")
    b[].primitives = entry_ref_unresolved()
    var word = samba_vendor_word(vendor)
    if word < 0:
        b[].primitives_error = String("parallel MLP: unsupported native vendor ") + vendor
    else:
        try:
            b[].primitives = samba_ref[SAMBA_TABLE_TRAINING](SambaEntryKey(1 if mode == "identical" else 0, word))
        except problem:
            b[].primitives_error = String(problem)

def _mlp_mode() raises:
    if PARALLEL_MLP_BOUNDARY.get_or_create_ptr()[].mode != "identical":
        raise Error("parallel training requires an identical model")


def _mlp_fields(value: PythonObject, fields: String) raises:
    var bi = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    if py_type(value) is not bi.dict or Bool(py=py_call(bi.set, value) != py_call(bi.set, py_call(PythonObject(fields).split, "|"))):
        raise Error("parallel MLP: state has missing or unknown fields")


def _mlp_schedule(value: PythonObject) raises -> PythonObject:
    var bi = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    if py_type(value) is not bi.dict or len(value) == 0:
        raise Error("SmallMLPTrainer data_schedule must be a nonempty JSON object")
    var budget = 128
    _parallel_schedule_check(value, 0, budget)
    var encoded = _transport_json(value)
    if encoded.byte_length() > 2048:
        raise Error("SmallMLPTrainer data_schedule exceeds 2048 bytes")
    return native_json_parse(PythonObject(encoded))


def _mlp_finite(value: PythonObject, label: String) raises:
    var data = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=value._addr))
    mlp_check_finite(data, Int(py=value.size), label)


def mlp_validate_state(state: PythonObject) raises -> PythonObject:
    _mlp_mode()
    var bi = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    _mlp_fields(state, "schema|architecture|numeric_mode|parameter_order|weights|optimizer|config|data_schedule")
    var names = _mlp_names()
    var name_list = bi.list()
    for name in names:  # small-loop(the fixed MLP field-name words)
        _ = py_call(name_list.append, name)
    var nk = parallel_keys()
    if String(py_item(state, nk[].k_schema)) != "mojolearn.small-mlp-trainer.v1" or String(py_item(state, nk[].k_numeric_mode)) != "identical" or Bool(py=py_item(state, nk[].k_architecture) != py_ints([8, 16, 3])) or Bool(py=py_item(state, nk[].k_parameter_order) != name_list):
        raise Error("SmallMLPTrainer state schema/architecture/mode/order mismatch")
    _mlp_fields(py_item(state, nk[].k_weights), "weight1|bias1|weight2|bias2")
    _mlp_fields(py_item(state, nk[].k_optimizer), "kind|step|m|v|flags")
    _mlp_fields(py_item(state, nk[].k_config), "lr|beta1|beta2|eps|weight_decay")
    var config = bi.dict()
    for name in [String("lr"), String("beta1"), String("beta2"), String("eps"), String("weight_decay")]:  # small-loop(the five optimizer config words)
        py_setitem(config, name, Float64(_parallel_float(py_item(state, nk[].k_config)[name], name)))
    if Float64(py=py_item(config, nk[].k_lr)) <= 0 or Float64(py=py_item(config, nk[].k_eps)) <= 0 or Float64(py=py_item(config, nk[].k_weight_decay)) < 0 or Float64(py=py_item(config, nk[].k_beta1)) < 0 or Float64(py=py_item(config, nk[].k_beta1)) >= 1 or Float64(py=py_item(config, nk[].k_beta2)) < 0 or Float64(py=py_item(config, nk[].k_beta2)) >= 1:
        raise Error("SmallMLPTrainer optimizer configuration outside admitted range")
    var old = py_item(state, nk[].k_optimizer)
    if String(py_item(old, nk[].k_kind)) != "AdamW":
        raise Error("SmallMLPTrainer state optimizer mismatch")
    var optimizer = bi.dict()
    py_setitem(optimizer, nk[].k_kind, "AdamW")
    py_setitem(optimizer, nk[].k_step, _parallel_int(py_item(old, nk[].k_step), 0, 2147483647, "SmallMLPTrainer step", False))
    py_setitem(optimizer, nk[].k_m, _parallel_array(py_item(old, nk[].k_m), [195]))
    py_setitem(optimizer, nk[].k_v, _parallel_array(py_item(old, nk[].k_v), [195]))
    py_setitem(optimizer, nk[].k_flags, _parallel_array(py_item(old, nk[].k_flags), [4], True))
    var v = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=py_item(optimizer, nk[].k_v)._addr))
    var flags = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=Int(py=py_item(optimizer, nk[].k_flags)._addr))
    mlp_check_optimizer_state(v, flags)
    var weights = bi.dict()
    for i in range(4):  # driver-loop: over the existing fixed MLP parameter registry
        py_setitem(weights, names[i], _parallel_array(py_item(state, nk[].k_weights)[names[i]], _mlp_dims(i)))
    var result = py_call(bi.dict, state)
    py_setitem(result, nk[].k_architecture, py_ints([8, 16, 3]))
    py_setitem(result, nk[].k_parameter_order, name_list)
    py_setitem(result, nk[].k_weights, weights)
    py_setitem(result, nk[].k_optimizer, optimizer)
    py_setitem(result, nk[].k_config, config)
    py_setitem(result, nk[].k_data_schedule, _mlp_schedule(py_item(state, nk[].k_data_schedule)))
    return result


def mlp_snapshot(model: PythonObject) raises -> PythonObject:
    """The model's complete state, read from its native state handle (the
    public `state_dict` call, under the model's lock), admitted as identical."""
    _mlp_mode()
    var nk = parallel_keys()
    return mlp_validate_state(small_mlp_call(SMALL_MLP_OPN_STATE_DICT, PythonObject(None), Python.tuple(py_get(model.__dict__, nk[].k__handle))))


def mlp_publish(model: PythonObject, state: PythonObject) raises:
    """Replace the model's native state with a validated copy of `state`."""
    _mlp_mode()
    var checked = mlp_validate_state(state)
    var nk = parallel_keys()
    _ = small_mlp_call(SMALL_MLP_OPN_LOAD_STATE_DICT, PythonObject(None), Python.tuple(py_get(model.__dict__, nk[].k__handle), checked))


def mlp_from_state(state: PythonObject) raises -> PythonObject:
    _mlp_mode()
    var b = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()
    return small_mlp_from_state(b[].model_cls, mlp_validate_state(state), "identical")


def mlp_batch(value: PythonObject) raises -> PythonObject:
    var bi = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    var view = _parallel_view(value)
    var shape = value.shape if py_type(value) is GENERATION_BOUNDARY.get_or_create_ptr()[].array_cls else view.shape
    if len(shape) != 2 or Int(py=shape[1]) != 8 or Int(py=shape[0]) < 1 or Int(py=shape[0]) > 256:
        raise Error("SmallMLPTrainer X must have shape (batch, 8), batch in [1,256]")
    return _parallel_array(value, [Int(py=shape[0]), 8])


def mlp_targets(value: PythonObject, rows: Int) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var view = _parallel_view(value)
    var shape = value.shape if py_type(value) is g[].array_cls else view.shape
    var fmt = String(py_call(view.format.lstrip, "=@<"))
    if len(shape) != 1 or Int(py=shape[0]) != rows:
        raise Error("SmallMLPTrainer targets must match batch rows")
    if fmt != "b" and fmt != "B" and fmt != "h" and fmt != "H" and fmt != "i" and fmt != "I" and fmt != "l" and fmt != "L" and fmt != "q" and fmt != "Q":
        raise Error("TypeError: SmallMLPTrainer targets must be a native integer array")
    var packed = py_call(g[].arrays, fmt)
    var item = Int(py=packed.itemsize)
    var output = generation_array([rows], "i")
    var ids = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=output._addr))
    var signed = fmt == "b" or fmt == "h" or fmt == "i" or fmt == "l" or fmt == "q"
    # A native-order C-contiguous export whose item size matches the packing
    # array's is already the packed buffer tobytes + frombytes would build:
    # widen from it in place while the export is held. Other layouts keep the
    # packing copy.
    var done = False
    if Int(py=view.itemsize) == item and Bool(py=view.c_contiguous):
        var buffer = List[Int](length=10, fill=0)
        var bp = buffer.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
        # PyBUF_C_CONTIGUOUS (0x38): only a C-contiguous export succeeds
        if cpython_table()[].PyObject_GetBuffer(view._obj_ptr, bp, Int32(0x38)) == 0:
            if buffer[2] == rows * item:
                try:
                    mlp_targets_into(buffer[0], item, signed, rows, ids)
                except error:
                    # release the export before the refusal propagates
                    cpython_table()[].PyBuffer_Release(bp)
                    raise error^
                done = True
            cpython_table()[].PyBuffer_Release(bp)
            _ = buffer^  # the Py_buffer words stay alive through the release above
        else:
            ref cpy = Python().cpython()
            cpython_table()[].PyErr_Clear()
    if not done:
        _ = py_call(packed.frombytes, py_call(view.tobytes, "C"))
        mlp_targets_into(Int(py=packed.buffer_info()[0]), item, signed, rows, ids)
    _ = view  # the exporter stays alive through the reads above
    _ = output  # keep the Python object alive through the native call above
    _ = packed^
    return output


# TOMBSTONE: the shard gradient's per-operation route (_mlp_gemm / _mlp_bias / _mlp_sum over the training binding's neural_gemm, mlp_bias_activation, ce_loss, mlp_relu_backward and mlp_sum_rows entries, one upload, launch and wait each) deleted 2026-11-09 by lane/m2-neural-leftovers; code recoverable at 0ce88411d3. A shard's gradient is the SmallMLPTrainer program's loss-and-gradients run (autograd/small_mlp_program.mojo, mode 1), so the parallel owner's gradients are the single trainer's.
# Restore: git apply experiments/removed/M2_SMALL_MLP_PRIMITIVES.patch


def mlp_gradients_open(checked: PythonObject) raises -> Int:
    """A gradient slot over a state `mlp_validate_state` returned: the
    trainer's program opens on the slot's first shard and stays resident."""
    _mlp_mode()
    return small_mlp_grads_open(checked, "identical")


def mlp_gradients_close(handle: Int) raises:
    small_mlp_grads_close(handle)


def mlp_gradients_on(handle: Int, inputs: PythonObject, targets: PythonObject) raises -> PythonObject:
    """(loss, [dweight1, dbias1, dweight2, dbias2]) of one shard on an open
    gradient slot: ONE run of the program (forward, mean cross-entropy and
    the tape backward; the weights are read only)."""
    var x = mlp_batch(inputs)
    var rows = Int(py=x.shape[0])
    var y = mlp_targets(targets, rows)
    var result = small_mlp_grads_rows(handle, Int(py=x._addr), Int(py=y._addr), rows)
    # Integer addresses do not retain their owners across sibling native
    # entries; keep the admitted copies alive through the consuming call.
    _ = x^
    _ = y^
    return result


def mlp_gradients_from_state(state: PythonObject, inputs: PythonObject, targets: PythonObject) raises -> PythonObject:
    return mlp_gradients_from_checked(mlp_validate_state(state), inputs, targets)


def mlp_gradients_from_checked(checked: PythonObject, inputs: PythonObject, targets: PythonObject) raises -> PythonObject:
    """`mlp_gradients_from_state` on a state `mlp_validate_state` already
    returned (read only here), on a slot opened for this one call. A worker
    keeps its slot across shards instead (`mlp_gradients_on`)."""
    var handle = mlp_gradients_open(checked)
    var result = PythonObject(None)
    try:
        result = mlp_gradients_on(handle, inputs, targets)
    except error:
        mlp_gradients_close(handle)
        raise error^
    mlp_gradients_close(handle)
    return result


def neural_ordered_sum(parts: PythonObject) raises -> PythonObject:
    _mlp_mode()
    var b = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()
    var shards = py_call(b[].bi.list, parts)
    if len(shards) == 0:
        raise Error("ordered reduction needs at least one shard")
    var first = py_call(b[].bi.list, shards[0])
    var total = b[].bi.list()
    var dims = List[List[Int]]()
    var counts = List[Int]()
    var count = 0
    for j in range(len(first)):  # driver-loop: over the caller's gradient tensor registry
        var view = _parallel_view(first[j])
        var shape = first[j].shape if py_type(first[j]) is GENERATION_BOUNDARY.get_or_create_ptr()[].array_cls else view.shape
        var axes = List[Int]()
        var n = 1
        for axis in range(len(shape)):  # small-loop(gradient shape dimension words)
            var dimension = Int(py=shape[axis])
            if dimension < 0 or (dimension != 0 and n > 2147483647 // dimension):
                raise Error("gradient registry exceeds signed 32-bit kernel indexing")
            axes.append(dimension)
            n *= dimension
        if count > 2147483647 - n:
            raise Error("gradient registry exceeds signed 32-bit kernel indexing")
        count += n
        counts.append(n)
        # Ordered addition historically admits nonfinite float32 inputs; the
        # existing accumulation kernel owns canonicalization. Do not turn a
        # preparation helper's finite scan into a new reduction restriction.
        _ = py_call(total.append, _parallel_array(first[j], axes, False, False))
        dims.append(axes^)
    if len(shards) == 1:
        return total
    if len(first) == 0 or count == 0:
        raise Error("ordered reduction needs nonempty gradient tensors")
    # Row N093: every shard is copied ONCE into one [S, count] buffer (slot
    # k = shard k), then the same pair fold as before runs in the same
    # ascending order IN PLACE: step k reads slots k-1 (the running total)
    # and k (shard k) as one [2, count] block and writes the pair's sum over
    # slot k. The `accumulate` entry uploads its parts before it writes its
    # output, so the overlap is safe, and the operands of every pair add are
    # the same values as the former copy-in / copy-out walk: same bits. Host
    # copies per step drop from 3 * (S - 1) * count words to (S + 1) * count.
    var n_shards = len(shards)
    var stacked = generation_array([n_shards, count])
    var stack = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(py=stacked._addr))
    var offset = 0
    for j in range(len(total)):  # driver-loop: over the first shard's gradient tensors
        unsafe_memcpy(dest=stack + offset, src=Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(py=total[j]._addr)), count=counts[j])
        offset += counts[j]
    for shard in range(1, n_shards):  # driver-loop: over logical gradient shards (one copy each)
        var current = py_call(b[].bi.list, shards[shard])
        if len(current) != len(total):
            raise Error("ordered reduction gradient tensor count mismatch")
        offset = 0
        for j in range(len(total)):  # driver-loop: over this shard's gradient tensors
            var value = _parallel_array(current[j], dims[j], False, False)
            unsafe_memcpy(dest=stack + shard * count + offset, src=Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(py=value._addr)), count=counts[j])
            offset += counts[j]
            _ = value^
    var base = Int(py=stacked._addr)
    # Primitive op 9 (the former `accumulate` entry: addresses [out, parts],
    # ints [n, a, t_tokens]) through the entry resolved at initialize. One GIL
    # release covers the whole ordered fold (the Python entry released it per
    # shard); the fold order and every pair add are unchanged.
    var primitives = _mlp_primitives()
    with GILReleased(Python()):
        for shard in range(1, n_shards):  # driver-loop: over fixed ascending logical gradient shard order
            var addresses: Array[Int, 2] = [base + 4 * shard * count, base + 4 * (shard - 1) * count]
            var ints: Array[Int, 3] = [count, 2, -1]
            var floats: Array[Float64, 0] = []
            _ = entry_ref_call(primitives, SAMBA_PRIM_ACCUMULATE, addresses, ints, floats)
    _ = stacked  # keep the Python owner alive through the native calls above
    var source = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=base) + (n_shards - 1) * count
    offset = 0
    for j in range(len(total)):  # driver-loop: over owned folded gradient output tensors
        var destination = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(py=total[j]._addr))
        unsafe_memcpy(dest=destination, src=source + offset, count=counts[j])
        offset += counts[j]
    _ = stacked^
    return total


def neural_cooperative_admit() raises:
    var b = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()
    # MOJOLEARN_OPTIMIZER_DEVICE_COUNT, read once per process
    # (training/optimizer_device_plan.mojo); was an environment read per call.
    var words = optimizer_device_setting()
    if words == "1":
        return
    # Any other value keeps the int() parse (and its refusal) of before.
    var count = Int(py=py_call(b[].bi.int, py_call(PythonObject(words).encode, "ascii")))
    if count > 1:
        for name in [String("optimizer_parallel_available"), String("accumulate_parallel_available"), String("clip_parallel_available")]:  # small-loop(the three cooperative entry names)
            if b[].entries[name] is None:
                raise Error("ImportError: training binding lacks cooperative neural update entry " + name)
            if Int(py=b[].entries[name]()) != 1:
                raise Error("RuntimeError: training binding refused " + name)


def mlp_apply_from_state(state: PythonObject, gradients: PythonObject) raises -> PythonObject:
    neural_cooperative_admit()
    var next = mlp_validate_state(state)
    var nk = parallel_keys()
    var step = Int(py=py_item(py_item(next, nk[].k_optimizer), nk[].k_step))
    if step == 2147483647:
        raise Error("SmallMLPTrainer step exceeds its supported range")
    if len(gradients) != 4:
        raise Error("SmallMLPTrainer requires exactly four parameter gradients")
    var params = generation_array([195])
    var grads = generation_array([195])
    var offsets = generation_array([5], "i")
    var info = generation_array([3])
    var pp = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(py=params._addr))
    var gp = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(py=grads._addr))
    var op = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=offsets._addr))
    var names = _mlp_names()
    var offset = 0
    for j in range(4):  # driver-loop: over transactional MLP parameter and gradient buffers
        var value = _parallel_array(gradients[j], _mlp_dims(j))
        var n = Int(py=value.size)
        op[j] = Int32(offset)
        var src = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(py=py_item(next, nk[].k_weights)[names[j]]._addr))
        unsafe_memcpy(dest=pp + offset, src=src, count=n)
        src = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(py=value._addr))
        unsafe_memcpy(dest=gp + offset, src=src, count=n)
        offset += n
        _ = value^
    op[4] = Int32(offset)
    var cfg = py_item(next, nk[].k_config)
    # Primitive op 25 (the former `optimizer_step` entry's words): ints
    # [J, kind, t, nesterov, maximize], floats [lr, beta1, beta2, eps,
    # weight_decay, momentum, dampening, max_norm]; the same AdamW plan as
    # the former 13-word Python list (momentum, dampening, max_norm and
    # maximize 0), the same Float64 -> Float32 narrowing in the callee.
    var ints: Array[Int, 5] = [4, 2, step + 1, 0, 0]
    var floats: Array[Float64, 8] = [Float64(py=py_item(cfg, nk[].k_lr)), Float64(py=py_item(cfg, nk[].k_beta1)), Float64(py=py_item(cfg, nk[].k_beta2)),
        Float64(py=py_item(cfg, nk[].k_eps)), Float64(py=py_item(cfg, nk[].k_weight_decay)), 0.0, 0.0, 0.0]
    var saved = py_item(next, nk[].k_optimizer)
    var addresses: Array[Int, 7] = [Int(py=params._addr), Int(py=grads._addr), Int(py=py_item(saved, nk[].k_m)._addr),
        Int(py=py_item(saved, nk[].k_v)._addr), Int(py=offsets._addr), Int(py=py_item(saved, nk[].k_flags)._addr), Int(py=info._addr)]
    var primitives = _mlp_primitives()
    var written = 0
    with GILReleased(Python()):
        written = entry_ref_call(primitives, SAMBA_PRIM_OPTIMIZER_STEP, addresses, ints, floats)
    _ = saved  # the moment and flag owners stay alive through the native call above
    if written != 195:
        raise Error("RuntimeError: SmallMLPTrainer update returned incomplete state")
    _mlp_finite(params, "updated parameter")
    _mlp_finite(py_item(saved, nk[].k_m), "first moment")
    _mlp_finite(py_item(saved, nk[].k_v), "second moment")
    offset = 0
    for j in range(4):  # driver-loop: over completed transactional parameter publication
        var value = py_item(next, nk[].k_weights)[names[j]]
        var n = Int(py=value.size)
        var dst = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(py=value._addr))
        unsafe_memcpy(dest=dst, src=pp + offset, count=n)
        offset += n
    py_setitem(saved, nk[].k_step, step + 1)
    _ = params^
    _ = grads^
    _ = offsets^
    _ = info^
    return next


def mlp_call(operation: Int, name: PythonObject, args: PythonObject) raises -> PythonObject:
    """`operation` is a `PARALLEL_MLP_OPN_*` code (-1 for a name the table
    does not hold); `name` is read only for the unknown operation's text."""
    if operation == PARALLEL_MLP_OPN_INITIALIZE:
        mlp_initialize(args[0], args[1], args[2], args[3], String(args[4]), String(args[5]))
    elif operation == PARALLEL_MLP_OPN_SNAPSHOT:
        return mlp_snapshot(args[0])
    elif operation == PARALLEL_MLP_OPN_VALIDATE:
        return mlp_validate_state(args[0])
    elif operation == PARALLEL_MLP_OPN_FROM_STATE:
        return mlp_from_state(args[0])
    elif operation == PARALLEL_MLP_OPN_PUBLISH:
        mlp_publish(args[0], args[1])
    elif operation == PARALLEL_MLP_OPN_GRADIENT:
        return mlp_gradients_from_state(args[0], args[1], args[2])
    elif operation == PARALLEL_MLP_OPN_ORDERED_SUM:
        return neural_ordered_sum(args[0])
    elif operation == PARALLEL_MLP_OPN_APPLY:
        return mlp_apply_from_state(args[0], args[1])
    else:
        raise Error("parallel MLP: unknown operation " + String(name))
    return PythonObject(None)


def parallel_mlp_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    try:
        var nk = parallel_keys()
        var result = mlp_call(py_op_code(nk[].ops_mlp, args[0]), args[0], args[1])
        return result.steal_data()
    except error:
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
