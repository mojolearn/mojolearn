# SPDX-License-Identifier: Apache-2.0
# Native Torch whole-program execution of an already imported artifact. Graph
# capture/export is the caller's explicit preparation, never product execution.
# Tensor admission, routing, contiguous copies, output allocation, stream order,
# replay and result packing are owned here. MPS retains its supported host-copy
# route; no integer is reinterpreted as a Metal buffer. Host inputs use their
# admitted native Torch addresses directly, without NumPy conversion.
from std.memory import memcpy
from std.time import perf_counter_ns
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_fields, py_type, py_setitem
from bindings.native_loaded_module import native_loaded_module
from bindings.native_torch_admission import (
    native_torch_admit, native_torch_modes_admit, native_torch_device, native_torch_storage_metadata,
    native_torch_contiguous, native_torch_to,
)
from sequence.exec_trait import Exec
from sequence.ops import FP
from core.cast_in import CastIn, CAST_KIND_FLOAT, CAST_KIND_SIGNED, CAST_KIND_UNSIGNED, CAST_KIND_BOOL
from autograd.rankn import ShapeN
from autograd.program import PROGRAM_IN_IDS, same_shape
from autograd.program_io import ProgramInput
from autograd.program_host_request import NativeProgramHostRun
from providers.python.autograd_host_provider_pyapi import imported_host_provider_request
from providers.python.autograd_program_pyapi import _imported_session_dims
from providers.python.autograd_device_provider_pyapi import imported_device_provider_run
from providers.python.autograd_import_metadata_pyapi import imported_program_snapshot


def _torch_entry(context: PythonObject, name: String) raises -> PythonObject:
    var entry = py_get(context, name)
    if entry is None or py_type(entry) is not py_type(native_loaded_module("builtins").len):
        raise Error("TypeError: Torch execution requires compiled entry " + name)
    return entry


def _torch_kind(dtype: String) raises -> Tuple[Int, Int]:
    if dtype == "float32":
        return (CAST_KIND_FLOAT, 4)
    if dtype == "float64":
        return (CAST_KIND_FLOAT, 8)
    if dtype == "float16":
        return (CAST_KIND_FLOAT, 2)
    if dtype == "int64":
        return (CAST_KIND_SIGNED, 8)
    if dtype == "int32":
        return (CAST_KIND_SIGNED, 4)
    if dtype == "int16":
        return (CAST_KIND_SIGNED, 2)
    if dtype == "int8":
        return (CAST_KIND_SIGNED, 1)
    if dtype == "uint8":
        return (CAST_KIND_UNSIGNED, 1)
    if dtype == "bool":
        return (CAST_KIND_BOOL, 1)
    raise Error("TypeError: unsupported Torch program input dtype " + dtype)


def _torch_describe(value: PythonObject) raises -> PythonObject:
    var info = native_torch_storage_metadata(value)
    var dims = py_call(native_loaded_module("builtins").tuple, info[1])
    return Python.tuple(info[3], dims, String(info[0]).removeprefix("torch."))


struct TorchDeviceSnapshot(Movable):
    var owner: PythonObject
    var kind: String
    var index: Int

    def __init__(out self, owner: PythonObject) raises:
        self.owner = owner
        self.kind = String(owner.type)
        var index = owner.index
        self.index = -1 if index is None else Int(py=index)


def _torch_devices(inputs: PythonObject) raises -> List[TorchDeviceSnapshot]:
    var b = native_loaded_module("builtins")
    if py_type(inputs) is not b.list and py_type(inputs) is not b.tuple:
        raise Error("TypeError: Torch program inputs require a native list or tuple")
    var devices = List[TorchDeviceSnapshot]()
    for i in range(len(inputs)):  # driver-loop: over input device metadata, once per public operation
        devices.append(TorchDeviceSnapshot(native_torch_device(inputs[i])))
    return devices^


def _torch_route_with_devices(snapshot: PythonObject, devices: List[TorchDeviceSnapshot]) raises -> PythonObject:
    if len(devices) == 0:
        return Python.tuple("host", "the graph has no tensor inputs")
    ref device = devices[0]
    if device.kind == "mps":
        return Python.tuple("host", "PyTorch exposes no supported raw pointer for an MPS tensor; Apple inputs are copied through the host")
    if device.kind != "cuda":
        return Python.tuple("host", "the inputs are on the " + device.kind)
    if device.index != 0:
        return Python.tuple("host", "mojolearn runs on GPU 0 and the input is on another visible GPU")
    for i in range(1, len(devices)):  # driver-loop: over remaining admitted native device metadata
        ref other = devices[i]
        if other.kind != "cuda" or other.index != 0:
            return Python.tuple("host", "the inputs use different devices")
    if py_get(snapshot[0], "_run_device") is None:
        return Python.tuple("host", "this mojolearn binding takes no device buffers")
    return Python.tuple("device", "")


def _torch_route(snapshot: PythonObject, inputs: PythonObject) raises -> PythonObject:
    # Untimed route diagnostic uses the same operation-local device snapshot.
    return _torch_route_with_devices(snapshot, _torch_devices(inputs))


def _torch_run_host(
    snapshot: PythonObject, inputs: PythonObject, outputs: PythonObject,
) raises:
    var fields = snapshot[0]
    var session = snapshot[1]
    if session["_id"] is None:
        raise Error("ValueError: this session is closed")
    var sid = Int(py=session["_id"])
    var request = NativeProgramHostRun()
    imported_host_provider_request(snapshot, request, 0)
    ref plan = request.plan
    var declared = snapshot[2]
    if len(inputs) != len(declared) or len(inputs) != plan.program.n_in:
        raise Error("ValueError: Torch program input count differs from the captured graph")
    var owners = List[ProgramInput]()
    for k in range(len(inputs)):  # driver-loop: admitted host Tensor descriptors
        var info = _torch_describe(inputs[k])
        var dims = info[1]
        var expected = declared[k][1]
        if len(dims) != len(expected):
            raise Error("ValueError: Torch input must keep its captured shape")
        var raw = List[Int]()
        for j in range(len(dims)):  # driver-loop: tensor rank metadata
            if Int(py=dims[j]) != Int(py=expected[j]):
                raise Error("ValueError: Torch input must keep its captured shape")
            raw.append(Int(py=dims[j]))
        var shape = ShapeN.of(_imported_session_dims(raw))
        if shape.count() == 0 or not same_shape(shape, plan.program.in_shapes[k]):
            raise Error("ValueError: Torch input must keep its nonempty captured shape")
        var kind = _torch_kind(String(info[2]))
        if plan.program.in_kinds[k] == PROGRAM_IN_IDS and kind[0] != CAST_KIND_SIGNED and kind[0] != CAST_KIND_UNSIGNED:
            raise Error("TypeError: program input holds ids and must be integers")
        var src = CastIn(Int(py=info[0]), kind[0], kind[1], shape.rows(), shape.cols(), shape.cols() * kind[1], kind[1])
        src.check("Torch program input")
        owners.append(ProgramInput(src, shape, List[Float64](), List[Int64]()))
    request.inputs = owners^
    imported_host_provider_request(snapshot, request, 1)
    ref result = request.result
    if len(result.outputs) != len(outputs):
        raise Error("ValueError: Torch program output count differs from the captured graph")
    for k in range(len(outputs)):  # driver-loop: output buffer copies
        var info = _torch_describe(outputs[k])
        var count = 1
        for j in range(len(info[1])):  # driver-loop: output shape metadata
            count *= Int(py=info[1][j])
        if count != result.outputs[k].shape.count():
            raise Error("ValueError: Torch program output shape differs from the captured graph")
        memcpy(dest=FP(unsafe_from_address=Int(py=info[0])), src=result.outputs[k].data.unsafe_ptr(), count=count)
    _ = inputs  # retain every admitted host Tensor through replay
    _ = outputs


def _torch_artifact_context(snapshot: PythonObject) raises -> PythonObject:
    # Resolve only loaded framework native entries. No capture, model callbacks,
    # weight-version polling or export runs when executing an artifact snapshot.
    var torch = native_loaded_module("torch")
    var native = native_loaded_module("torch._C")
    var b = native_loaded_module("builtins")
    var fields = py_fields(native)
    var variables = fields["_VariableFunctions"]
    var flags = b.type.__dict__["__flags__"].__get__
    if (Int(py=py_call(flags, py_type(variables))) & 512) != 0:
        raise Error("TypeError: Torch allocation requires its native function container")
    var context = Python.dict()
    py_setitem(context, "empty", variables.empty)
    _ = _torch_entry(context, "empty")
    for pair in [Python.tuple("get_device", "_cuda_getDevice"),  # small-loop(fixed native framework entries)
                 Python.tuple("set_device", "_cuda_setDevice"),
                 Python.tuple("synchronize", "_cuda_synchronize"),
                 Python.tuple("synchronize_mps", "_mps_deviceSynchronize")]:
        py_setitem(context, pair[0], py_get(fields, pair[1]))
    var public = py_fields(torch)
    var device_type = public["device"]
    if py_type(device_type) is not b.type or (Int(py=py_call(flags, device_type)) & 512) != 0:
        raise Error("TypeError: Torch device requires its native type")
    py_setitem(context, "cpu", py_call(device_type, "cpu"))
    py_setitem(context, "f32", public["float32"])
    py_setitem(context, "i32", public["int32"])
    var count = len(snapshot[2])
    var free = Python.list()
    for index in range(count):  # driver-loop: declared graph input positions
        py_call(free.append, index)
    py_setitem(context, "free", free)
    py_setitem(context, "n_args", count)
    py_setitem(context, "sequence", len(snapshot[3]) != 1)
    return context


def ops_torch_program_run_py(
    program: PythonObject, args: PythonObject, var context: PythonObject, route_only: Bool = False,
) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var snapshot = imported_program_snapshot(program)
    if context is None:
        context = _torch_artifact_context(snapshot)
    var profile = Bool(py=py_get(context, "profile", False))
    var started = perf_counter_ns() if profile else 0
    native_torch_modes_admit()
    if route_only:
        return _torch_route(snapshot, args)
    if len(args) != Int(py=context["n_args"]):
        raise Error("ValueError: compiled graph input count differs from its captured signature")
    var inputs = Python.list()
    var free = context["free"]
    for i in range(len(free)):  # driver-loop: graph input selection
        py_call(inputs.append, args[Int(py=free[i])])
    var devices = _torch_devices(inputs)
    var route = _torch_route_with_devices(snapshot, devices)
    if Bool(py=py_get(context, "force_host", False)):
        route = Python.tuple("host", "host route selected by the transfer test controller")
    var on_device = String(route[0]) == "device"
    var target = context["cpu"]
    if len(inputs) > 0:
        target = devices[0].owner
    var prepared = Python.list()
    for k in range(len(inputs)):  # driver-loop: native framework buffer preparation
        py_call(prepared.append, native_torch_contiguous(inputs[k], not on_device))
    var fields = snapshot[0]
    var specs = snapshot[3]
    var outputs = Python.list()
    var destination = target if on_device else context["cpu"]
    for k in range(len(specs)):  # driver-loop: native framework output allocation
        var dtype = context["i32"] if String(specs[k][2]) == "int32" else context["f32"]
        var output = py_call(_torch_entry(context, "empty"), specs[k][1], dtype=dtype, device=destination)
        native_torch_admit(output)
        py_call(outputs.append, output)
    var ready = 0
    var ran = 0
    if on_device:
        # The supported C CUDA entries also serve ROCm. Select GPU zero while
        # synchronizing, then restore the caller's current device on every exit.
        # Device-wide synchronization is conservative; transfer measurements
        # must retain its cost and must not claim a stream-only cost.
        var previous = py_call(_torch_entry(context, "get_device"))
        py_call(_torch_entry(context, "set_device"), PythonObject(0))
        try:
            py_call(_torch_entry(context, "synchronize"))
            var descriptors = Python.list()
            var addresses = Python.list()
            for k in range(len(prepared)):  # driver-loop: admitted device buffer descriptors
                py_call(descriptors.append, _torch_describe(prepared[k]))
            for k in range(len(outputs)):  # driver-loop: caller-owned output addresses
                py_call(addresses.append, _torch_describe(outputs[k])[0])
            var run = fields["_run_device"]
            if py_type(run) is not py_type(b.len):
                raise Error("TypeError: Torch execution requires the compiled device program entry")
            ready = perf_counter_ns() if profile else 0
            imported_device_provider_run(snapshot, descriptors, addresses)
            ran = perf_counter_ns() if profile else 0
            py_call(_torch_entry(context, "synchronize"))
        finally:
            py_call(_torch_entry(context, "set_device"), previous)
    else:
        ready = perf_counter_ns() if profile else 0
        _torch_run_host(snapshot, prepared, outputs)
        ran = perf_counter_ns() if profile else 0
        var copied = Python.list()
        for k in range(len(outputs)):  # driver-loop: native framework device transfer
            py_call(copied.append, native_torch_to(outputs[k], target))
        if len(devices) > 0 and devices[0].kind == "mps":
            py_call(_torch_entry(context, "synchronize_mps"))
        outputs = copied
    _ = prepared  # contiguous copies stay alive until native replay completes
    _ = snapshot  # graph/session owners and admitted descriptors survive replay
    if profile:
        py_setitem(context, "last_profile", Python.tuple(route[0], ready - started, ran - ready, perf_counter_ns() - ran))
    if Bool(py=context["sequence"]):
        return py_call(b.tuple, outputs)
    return outputs[0]
