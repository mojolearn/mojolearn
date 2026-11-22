# SPDX-License-Identifier: Apache-2.0
"""Native Samba registry, explicit-weight and seeded constructor ownership.

The seeded initialization is split: UNPACK and PACK here
(`_samba_initialize_parameters`), RUN in training/samba_construct_run.mojo
(typed words, no Python). The rest of the constructor publishes Python
fields and owners (the stack's state is its public field set) and holds no
computation of its own."""
from core.weight_epoch import registry_epoch_bump
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from std.python._cpython import GILReleased
# RUN: the typed initializer, no Python import.
from training.samba_construct_run import samba_initialize_run
from std.memory import unsafe_memcpy
from std.sys import is_defined
# The same build define as training/optimizer_public_common.mojo
# OPT_LAZY_MOMENTS (kept local: this unit does not import the training
# package); B arm -D MOJOLEARN_OPT_LAZY_MOMENTS_OFF.
comptime OPT_LAZY_MOMENTS = not is_defined["MOJOLEARN_OPT_LAZY_MOMENTS_OFF"]()
from bindings.generation_buffers_door import generation_initialize_binding, generation_array
from bindings.generator_state_native_unpack import generator_fields
from bindings.generator_state_native_pack import generator_init, generator_validate
from bindings.samba_state_native_door import samba_fields, samba_view, _samba_registry, samba_build_blocks, samba_schedule_snapshot
from bindings.samba_route_unpack import samba_route_process, samba_device_admit
from bindings.causal_lm_construct_door import _model_f32, _model_shape
from bindings.py_balanced import py_call, py_setitem, py_setattr, py_get
from bindings.py_model_records import bump_model_generation
from bindings.optimizer_owner_boundary_unpack import OptimizerOwnerLease
from bindings.optimizer_owner_boundary_pack import optimizer_owner_new


def samba_load_weights(model: PythonObject, weights: PythonObject, context: PythonObject) raises -> PythonObject:
    # Generation word (bindings/py_model_records.mojo, lane ne1-e-records):
    # every stamped record of this object misses from here on.
    _ = bump_model_generation(model)
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var bi = native_loaded_module("builtins")
    var fields = samba_fields(model)
    var names = fields["names"]
    if Bool(py=py_call(bi.set, names) != py_call(bi.set, weights)):
        raise Error("SambaStack: weight dict mismatch; missing or unknown registry names")
    for i in range(len(names)):  # driver-loop: over ordered named parameter buffers
        var name = names[i]
        var value = _model_f32(weights[name], String(name), context["neural"])
        var dims = List[Int]()
        var shape = fields["shapes"][name]
        for axis in range(len(shape)):  # driver-loop: over tensor shape metadata
            dims.append(Int(py=shape[axis]))
        _model_shape(value, dims, String(name))
        if Int(py=py_call(context["all_finite_f32"], value._addr, value.size)) != 1:
            raise Error("SambaStack: " + String(name) + " is not finite")
        var target = fields["arrays"][name]
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=target._addr)),
               src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=Int(py=value._addr)), count=Int(py=value.nbytes))
        _ = value^
    # The written registry's own epoch (core/weight_epoch.mojo; hot-path
    # audit 2026-11-13: stamped by the write, not by the public door)
    var flat = py_get(fields, "flat", PythonObject(None))
    if flat is not None:
        registry_epoch_bump(Int(py=flat._addr))
    return PythonObject(None)


def samba_parameters(model: PythonObject) raises -> PythonObject:
    var fields = samba_fields(model)
    var result = Python.dict()
    for i in range(len(fields["names"])):  # driver-loop: over the public named parameter registry
        var name = fields["names"][i]
        py_setitem(result, name, fields["arrays"][name])
    return result


def _samba_initialize_parameters(model: PythonObject, context: PythonObject) raises:
    # UNPACK: the generator's seed and stream counter, the primitive entry
    # and each registry tensor's address, size, rank and second dimension.
    var fields = samba_fields(model)
    var generator = fields["generator"]
    var rng = generator_fields(generator)
    _ = generator_validate(rng["seed"], rng["counter"])
    # The RNG runs on the training image's primitive entry (op 6, the former
    # `neural_rng` Python entry), admitted for the GENERATOR's mode word as
    # before. The stack's device was admitted by `samba_construct` before any
    # field: on every installation a constructed stack runs this
    # installation's device, so the process route is the stack's route.
    var primitives = samba_route_process(rng["numeric_mode"], context).primitives()
    var seed = rng["seed"]
    var low = Int(py=seed & PythonObject(0xffffffff))
    var high = Int(py=(seed >> PythonObject(32)) & PythonObject(0xffffffff))
    var counter = Int(py=rng["counter"])
    var unused = generation_array([1])
    var names = List[String]()
    var addresses = List[Int]()
    var sizes = List[Int]()
    var ranks = List[Int]()
    var columns = List[Int]()
    var arrays = fields["arrays"]
    for i in range(len(fields["names"])):  # driver-loop: over the saved registry order (model structure)
        var output = arrays[fields["names"][i]]
        var rank = len(output.shape)
        names.append(String(fields["names"][i]))
        addresses.append(Int(py=output._addr))
        sizes.append(Int(py=output.size))
        ranks.append(rank)
        columns.append(Int(py=output.shape[1]) if rank == 2 else 0)
    # RUN (training/samba_construct_run.mojo): typed words only, one GIL
    # release for the whole registry (it was one release per tensor).
    # The scratch address is read here, with the GIL held: an attribute read
    # inside the released block ran PyObject_GetAttrString with no thread
    # state and segfaulted every seeded SambaStack (smoke 2026-11-22).
    var unused_addr = Int(py=unused._addr)
    var drawn = 0
    try:
        with GILReleased(Python()):
            samba_initialize_run(primitives, names, addresses, sizes, ranks, columns, low, high, counter,
                unused_addr, drawn)
    finally:
        # PACK: the streams consumed, also when a launch was refused (each
        # draw advanced the counter before its launch, as before).
        if drawn > 0:
            py_setitem(rng, "counter", PythonObject(counter + drawn))
        # The parameter owners and the scratch stay alive through the native
        # call above, named here (GIL held) so a raising RUN does not drop
        # their references inside the released block.
        _ = len(arrays)
        _ = len(unused)
    _ = unused^


def samba_construct(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var bi = native_loaded_module("builtins")
    var config = args[0]
    if not Bool(py=py_call(bi.isinstance, config, context["config_class"])):
        raise Error("TypeError: SambaStack: config must be a SambaConfig")
    var weights = args[1]
    var generator = args[2]
    if (weights is None) == (generator is None):
        raise Error("SambaStack: pass exactly one of weights= or generator=")
    var mode = args[10]
    # The device, checked before any field is published (shared words); the
    # mode and vendor admission of the connected binding follows it.
    samba_device_admit(args[11], context)
    _ = samba_route_process(mode, context)
    var accumulation = Int(py=py_call(bi.int, args[9]))
    if accumulation < 1 or (accumulation & (accumulation - 1)) != 0:
        raise Error("SambaStack: accumulation_steps must be a power of two >= 1")
    _ = samba_schedule_snapshot(args[7], context)
    var registry = _samba_registry(config)
    var last = registry[len(registry) - 1]
    var n = Int(py=last["offset"]) + Int(py=last["size"])
    var fields = samba_fields(model)
    py_setitem(fields, "config", config)
    py_setitem(fields, "numeric_mode", mode)
    py_setitem(fields, "device", args[11])
    py_setitem(fields, "n_total", n)
    var flat = generation_array([n])
    # lane x56d: the registry a kernel reads is read-only; the native
    # initialization and loads below write it by address
    py_setattr(flat, "_readonly", PythonObject(True))
    py_setitem(fields, "flat", flat)
    py_setitem(fields, "names", Python.list())
    py_setitem(fields, "shapes", Python.dict())
    py_setitem(fields, "offsets", Python.list())
    py_setitem(fields, "arrays", Python.dict())
    var params = Python.list()
    for i in range(len(registry)):  # driver-loop: over flat storage views in registry order
        var item = registry[i]
        var dims = List[Int]()
        for axis in range(len(item["shape"])):  # driver-loop: over tensor shape metadata
            dims.append(Int(py=item["shape"][axis]))
        var value = samba_view(fields["flat"], Int(py=item["offset"]), dims, context)
        _ = py_call(fields["names"].append, item["name"])
        py_setitem(fields["shapes"], item["name"], py_call(bi.tuple, item["shape"]))
        _ = py_call(fields["offsets"].append, item["offset"])
        py_setitem(fields["arrays"], item["name"], value)
        _ = py_call(params.append, value)
    _ = py_call(fields["offsets"].append, n)
    if generator is None:
        generator = py_call(bi.object.__new__, context["generator_class"])
        _ = generator_init(generator, PythonObject(0), mode)
    py_setitem(fields, "generator", generator)
    if weights is None:
        _samba_initialize_parameters(model, context)
    else:
        _ = samba_load_weights(model, weights, context)
    py_setitem(fields, "max_norm", PythonObject(None) if args[8] is None else py_call(bi.float, args[8]))
    var optimizer = py_call(bi.object.__new__, context["optimizer_class"])
    var opt = samba_fields(optimizer)
    py_setitem(opt, "_resident_wanted", PythonObject(None))
    var lazy = OPT_LAZY_MOMENTS  # build constant (was an environment read per construction)
    var owner = optimizer_owner_new(optimizer, 2, lazy)
    var publication = OptimizerOwnerLease(optimizer, owner)
    py_setitem(opt, "_native_optimizer_owner", owner)
    py_setitem(opt, "_m", PythonObject(None) if lazy else generation_array([n]))
    py_setitem(opt, "_v", PythonObject(None) if lazy else generation_array([n]))
    py_setitem(opt, "params", params)
    py_setitem(opt, "n_total", n)
    py_setitem(opt, "accumulation_steps", accumulation)
    py_setitem(opt, "numeric_mode", mode)
    py_setitem(opt, "lr", py_call(bi.float, args[3]))
    py_setitem(opt, "betas", Python.tuple(py_call(bi.float, args[4][0]), py_call(bi.float, args[4][1])))
    py_setitem(opt, "eps", py_call(bi.float, args[5]))
    py_setitem(opt, "weight_decay", py_call(bi.float, args[6]))
    py_setitem(opt, "lr_schedule", args[7])
    py_setitem(opt, "maximize", False)
    py_setitem(opt, "t", 0)
    py_setitem(opt, "packed_", PythonObject(None))
    py_setitem(opt, "total_norm_", PythonObject(None))
    py_setitem(opt, "clip_coef_", PythonObject(None))
    py_setitem(opt, "buf_initialized", generation_array([len(registry)], "i"))
    var offsets = generation_array([len(registry) + 1], "i")
    var ptr = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=offsets._addr))
    for i in range(len(registry) + 1):  # driver-loop: over registry offset metadata
        ptr[i] = Int32(Int(py=fields["offsets"][i]))
    py_setitem(opt, "offsets", offsets)
    py_setitem(fields, "optimizer", optimizer)
    samba_build_blocks(model, context)
    py_setitem(fields, "last_", PythonObject(None))
    _ = publication
    return PythonObject(None)
