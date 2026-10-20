# SPDX-License-Identifier: Apache-2.0
# One-call native ABI for model/tensor/pipeline partitioned captured graphs.
from std.ffi import _Global
from bindings.native_json import native_path
from std.memory import alloc, memcpy
from std.python import Python, PythonObject
from sequence.exec_trait import Exec
from sequence.ops import FP
from autograd.ops_api import ops_session, _OpsStore
from autograd.partition_plan import GraphBytes, partition_plan, column_bytes
from autograd.partition_runtime import PartitionedProgram
from autograd.partition_checkpoint import partition_checkpoint, partition_from_checkpoint, restore_partition_state
from autograd.distributed_pyapi import distributed_optimizer_config
from autograd.program_pyapi import array_input, output_array
from autograd.program_arrays import initialize_program_arrays
from autograd.rankn import ShapeN
from autograd.cast_ops import upload_cast, upload_cast_i32
from autograd.optim import optimizer_slots
from core.cast_in import CastIn
from training.checkpoint import read_checkpoint_unchecked, save_checkpoint
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime _PARTITION_NAME = "MojoPartitionedGraphsIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoPartitionedGraphsFast"
comptime _PARTITIONS = _Global[StorageType=_OpsStore, name=_PARTITION_NAME, init_fn=_OpsStore.__init__]


def register_partition[E: Exec & Movable & Deinitable, KIND: Int](var d: PartitionedProgram[E]) raises -> PythonObject:
    var p = alloc[PartitionedProgram[E]](1)
    p.init_pointee_move(d^)
    var s = _PARTITIONS.get_or_create_ptr()
    var id = s[].next_id
    s[].next_id += 1
    s[].ids.append(id)
    s[].addr.append(Int(p))
    s[].kind.append(KIND)
    return PythonObject(id)


def partition_slot(id: Int, kind: Int) raises -> Int:
    var s = _PARTITIONS.get_or_create_ptr()
    for k in range(len(s[].ids)):  # driver-loop: live native partition handles
        if s[].ids[k] == id and s[].kind[k] == kind:
            return k
    raise Error("partitioned program is closed or belongs to another binding")


def partition[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject) raises -> UnsafePointer[PartitionedProgram[E], MutUntrackedOrigin]:
    var slot = partition_slot(Int(py=id), KIND)
    return UnsafePointer[PartitionedProgram[E], MutUntrackedOrigin](unsafe_from_address=_PARTITIONS.get_or_create_ptr()[].addr[slot])


def partition_open[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, pid: PythonObject, devices: List[Int], stages: PythonObject,
    tensor_shards: PythonObject, microbatches: PythonObject, optimizer: PythonObject,
    parameters: PythonObject, seed: PythonObject, var executors: List[E],
) raises -> PythonObject:
    if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("partitioned programs require identical mode")
    var s = ops_session[E, KIND](Int(py=sid))
    var p = s[].program(Int(py=pid))
    var oi = s[]._opt_index(Int(py=optimizer))
    var o = s[].opts[oi]
    var ip: List[Int] = [o.cfg.kind, 1 if o.cfg.nesterov else 0, o.sched, o.sched_step]
    var fp: List[Float32] = [o.cfg.lr, o.cfg.momentum, o.cfg.beta1, o.cfg.beta2, o.cfg.eps, o.cfg.weight_decay, o.gamma]
    var values = List[GraphBytes]()
    var trainable = List[Bool](length=len(p.params), fill=False)
    for k in range(len(parameters)):  # driver-loop: selected trainable source handles
        var found = False
        var h = Int(py=parameters[k])
        for j in range(len(p.params)):  # driver-loop: captured parameter handles
            if p.params[j] == h:
                if trainable[j]:
                    raise Error("partitioned program: duplicate trainable parameter")
                trainable[j] = True
                found = True
        if not found:
            raise Error("partitioned program: trainable handle is not in capture")
    for j in range(len(p.params)):  # driver-loop: canonical source parameter bytes
        var integer = s[].int_words[s[]._slot(p.params[j])]
        if len(parameters) == 0:
            trainable[j] = not integer
        if trainable[j] and integer:
            raise Error("partitioned program: integer parameters cannot be trainable")
        var words = List[Float32](length=p.param_shapes[j].count(), fill=Float32(0))
        s[].download(p.params[j], FP(unsafe_from_address=Int(words.unsafe_ptr())))
        values.append(GraphBytes(p.param_shapes[j], words^, integer))
    var seed_int = Int(py=seed)
    if seed_int < 0:
        raise Error("partitioned program: seed must be a nonnegative signed 64-bit integer")
    var plan = partition_plan(p, values, trainable, Int(py=stages), Int(py=tensor_shards))
    var d = PartitionedProgram[E](plan^, devices, executors^, o.cfg, ip, fp, Int(py=microbatches), UInt64(seed_int))
    # Existing optimizer moments are column-packed exactly as their parameters.
    var ck = partition_checkpoint(d)
    var clocks = -1
    var slots = optimizer_slots(o.cfg)
    var scalar = 7 + Int(ck.ext_ints[10])
    for j in range(len(d.plan.parameters)):  # driver-loop: complete source optimizer migration
        if not d.plan.trainable[j]:
            continue
        var original = d.plan.origins[j]
        var si = -1
        for k in range(len(s[].opt_states)):  # driver-loop: optimizer state lookup
            if s[].opt_states[k].opt == oi and s[].opt_states[k].param == p.params[original]:
                si = k
        var t = s[].opt_states[si].t if si >= 0 else 0
        if clocks >= 0 and t != clocks:
            raise Error("partitioned program: selected parameters have different optimizer clocks")
        clocks = t
        if si < 0:
            continue
        for which in range(2):  # small-loop(2: first and second optimizer moment): byte migration
            if (which == 0 and not slots[0]) or (which == 1 and not slots[1]):
                continue
            var sh = p.param_shapes[original]
            var words = List[Float32](length=sh.count(), fill=Float32(0))
            var ptr = s[].opt_states[si].s1 if which == 0 else s[].opt_states[si].s2
            s[].ex.download(FP(unsafe_from_address=Int(words.unsafe_ptr())), ptr, sh.count())
            var state = GraphBytes(sh, words^)
            if d.plan.column_width[j] > 0:
                state = column_bytes(state, d.plan.column_first[j], d.plan.column_width[j])
            var dst = FP(unsafe_from_address=Int(ck.m_state.unsafe_ptr())) if which == 0 else FP(unsafe_from_address=Int(ck.v_state.unsafe_ptr()))
            memcpy(dest=dst + ck.offsets[j], src=state.words.unsafe_ptr(), count=state.shape.count())
        ck.ext_floats[scalar + 3 * j] = s[].opt_states[si].pw1
        ck.ext_floats[scalar + 3 * j + 1] = s[].opt_states[si].pw2
        ck.ext_floats[scalar + 3 * j + 2] = s[].opt_states[si].mu_prod
    ck.t = max(clocks, 0)
    ck.ext_ints[4] = Int64(o.epoch)
    ck.ext_ints[5] = Int64(ck.t)
    restore_partition_state(d, ck)
    return register_partition[E, KIND](d^)


def partition_load[E: Exec & Movable & Deinitable, KIND: Int](
    path: PythonObject, devices: List[Int], var executors: List[E],
) raises -> PythonObject:
    initialize_program_arrays()
    var ck = read_checkpoint_unchecked(String(py=native_path(path)))
    var plan = partition_from_checkpoint(ck)
    var ip = List[Int]()
    var fp = List[Float32]()
    for k in range(4):  # small-loop(4: optimizer option integers): metadata
        ip.append(Int(ck.ext_ints[12 + k]))
    for k in range(7):  # small-loop(7: optimizer option floats): metadata
        fp.append(ck.ext_floats[k])
    var cfg = distributed_optimizer_config(ip, fp)
    var d = PartitionedProgram[E](plan^, devices, executors^, cfg, ip, fp, Int(ck.ext_ints[2]), ck.seed)
    restore_partition_state(d, ck)
    return register_partition[E, KIND](d^)


def partition_fit_py[E: Exec & Movable & Deinitable, KIND: Int](
    id: PythonObject, arrays: PythonObject, targets: PythonObject, loss: PythonObject,
    epochs: PythonObject, batch: PythonObject, shuffle: PythonObject, delta: PythonObject,
) raises -> PythonObject:
    var name = String(py=loss)
    var kind = -1
    if name == "output":
        kind = 0
    elif name == "mse":
        kind = 1
    elif name == "mae":
        kind = 2
    elif name == "huber":
        kind = 3
    elif name == "bce_with_logits":
        kind = 4
    elif name == "cross_entropy":
        kind = 5
    else:
        raise Error("partitioned fit: unknown loss")
    var srcs = List[CastIn]()
    var shapes = List[ShapeN]()
    for k in range(len(arrays)):  # driver-loop: native input descriptors
        var parsed = array_input(arrays[k], "partitioned input")
        srcs.append(parsed[0])
        shapes.append(parsed[1])
    var target = CastIn.f32_c(0, 1, 1)
    if kind != 0:
        target = array_input(targets, "partitioned target")[0]
    var ne = Int(py=epochs)
    if ne < 1:
        raise Error("partitioned fit: epochs must be positive")
    var curve = output_array(ShapeN.of([ne]))
    var d = partition[E, KIND](id)
    d[].fit(srcs, shapes, target, kind, Float32(Float64(py=delta)), ne, Int(py=batch), Bool(py=shuffle),
            FP(unsafe_from_address=Int(py=curve.__array_interface__["data"][0])))
    return curve


def graph_array_output(a: GraphBytes) raises -> PythonObject:
    var out = output_array(a.shape)
    memcpy(dest=FP(unsafe_from_address=Int(py=out.__array_interface__["data"][0])),
           src=a.words.unsafe_ptr(), count=a.shape.count())
    return out


def partition_run_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject, arrays: PythonObject) raises -> PythonObject:
    var d = partition[E, KIND](id)
    d[].ready()
    if len(arrays) != d[].plan.program.n_in:
        raise Error("partitioned run: input count mismatch")
    var inputs = List[GraphBytes]()
    for k in range(len(arrays)):  # driver-loop: native dtype conversion and byte staging
        var parsed = array_input(arrays[k], "partitioned input")
        var words = List[Float32](length=parsed[1].count(), fill=Float32(0))
        var buf = d[].workers[0].session.ex.alloc(parsed[1].count())
        var integer = d[].plan.program.in_kinds[k] != 0
        if integer:
            upload_cast_i32(d[].workers[0].session.ex, buf, parsed[0], "partitioned ids")
        else:
            upload_cast(d[].workers[0].session.ex, buf, parsed[0])
        d[].workers[0].session.ex.download(FP(unsafe_from_address=Int(words.unsafe_ptr())), buf, parsed[1].count())
        d[].workers[0].session.ex.free(buf)
        inputs.append(GraphBytes(parsed[1], words^, integer))
    return graph_array_output(d[].run(inputs))


def partition_save_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject, path: PythonObject) raises -> PythonObject:
    var d = partition[E, KIND](id)
    var ck = partition_checkpoint(d[])
    return PythonObject(save_checkpoint(String(py=native_path(path)), ck).bytes)


def partition_state_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject) raises -> PythonObject:
    var d = partition[E, KIND](id)
    var ck = partition_checkpoint(d[])
    var result = Python.dict()
    var names: List[String] = ["parameters", "first_moments", "second_moments"]
    for which in range(3):  # small-loop(3: parameter and two optimizer arrays): native ABI outputs
        var arrays = Python.list()
        for j in range(len(d[].plan.parameters)):  # driver-loop: transformed parameter tensors
            var sh = d[].plan.parameters[j].shape
            var out = output_array(sh)
            var ptr = FP(unsafe_from_address=Int(ck.param.unsafe_ptr()))
            if which == 1:
                ptr = FP(unsafe_from_address=Int(ck.m_state.unsafe_ptr()))
            elif which == 2:
                ptr = FP(unsafe_from_address=Int(ck.v_state.unsafe_ptr()))
            memcpy(dest=FP(unsafe_from_address=Int(py=out.__array_interface__["data"][0])), src=ptr + ck.offsets[j], count=sh.count())
            arrays.append(out)
        result[PythonObject(names[which])] = arrays
    result["epoch"] = PythonObject(d[].epoch)
    result["step"] = PythonObject(d[].step)
    result["seed"] = PythonObject(d[].seed)
    result["logical_stages"] = PythonObject(d[].plan.stages)
    result["tensor_shards"] = PythonObject(d[].plan.tensor_shards)
    result["microbatches"] = PythonObject(d[].microbatches)
    return result


def partition_residency_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject) raises -> PythonObject:
    var d = partition[E, KIND](id)
    d[].ready()
    var result = Python.dict()
    var devices = Python.list()
    var counts = Python.list()
    var owners = Python.list()
    var placements = Python.list()
    for rank in range(len(d[].workers)):  # driver-loop: actual device-resident parameter words
        devices.append(PythonObject(d[].devices[rank]))
        var count = 0
        for j in range(len(d[].plan.parameters)):  # driver-loop: authoritative owned parameters
            if d[].workers[rank].owns[j]:
                count += d[].plan.parameters[j].shape.count()
        counts.append(PythonObject(count))
    for j in range(len(d[].plan.parameters)):  # driver-loop: unique parameter owners
        owners.append(PythonObject(d[].devices[d[].plan.owners[j] % len(d[].workers)]))
    for j in range(d[].plan.program.n_nodes()):  # driver-loop: execution plan witness
        placements.append(PythonObject(d[].devices[d[].plan.placement[j] % len(d[].workers)]))
    result["devices"] = devices
    result["parameter_elements_per_rank"] = counts
    result["parameter_owner_devices"] = owners
    result["node_devices"] = placements
    result["parameter_residency"] = PythonObject("one authoritative device owner per tensor shard")
    result["optimizer_residency"] = PythonObject("same unique owner as its parameter shard")
    result["activation_residency"] = PythonObject("native host-staged bytes with local recomputation")
    result["logical_stages"] = PythonObject(d[].plan.stages)
    result["tensor_shards"] = PythonObject(d[].plan.tensor_shards)
    result["microbatches"] = PythonObject(d[].microbatches)
    result["partitioned_products"] = PythonObject(d[].plan.partitioned_products)
    return result


def partition_profile_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject) raises -> PythonObject:
    var d = partition[E, KIND](id)
    d[].ready()
    var result = Python.list()
    for k in range(len(d[].events)):  # driver-loop: actual native operation intervals
        var e = d[].events[k]
        var row = Python.dict()
        row["node"] = PythonObject(e.node)
        row["microbatch"] = PythonObject(e.microbatch)
        row["stage"] = PythonObject(e.stage)
        row["device"] = PythonObject(e.device)
        row["backward"] = PythonObject(e.backward)
        row["start_ns"] = PythonObject(e.start_ns)
        row["end_ns"] = PythonObject(e.end_ns)
        result.append(row)
    return result


def partition_enter_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject, owner: PythonObject) raises -> PythonObject:
    var d = partition[E, KIND](id)
    d[].ready()
    return owner


def partition_close_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject) raises -> PythonObject:
    var table = _PARTITIONS.get_or_create_ptr()
    var slot = partition_slot(Int(py=id), KIND)
    var p = partition[E, KIND](id)
    _ = table[].ids.pop(slot)
    _ = table[].addr.pop(slot)
    _ = table[].kind.pop(slot)
    try:
        p[].close()
    except err:
        p.destroy_pointee()
        p.free()
        raise err
    p.destroy_pointee()
    p.free()
    return PythonObject(0)
