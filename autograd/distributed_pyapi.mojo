# SPDX-License-Identifier: Apache-2.0
# Native ABI and lifetime table. Python forwards arrays/options in one call;
# parsing, topology, ownership, graph walks and checkpoint bytes stay here.
from std.ffi import _Global
from std.memory import alloc, memcpy
from std.python import Python, PythonObject
from sequence.exec_trait import Exec
from sequence.ops import FP, OPT_SGD, OPT_ADAM, OPT_ADAMW
from autograd.ops_api import OpsSession, ops_session, _OpsStore
from autograd.distributed_program import ProgramReplica, DistributedProgram
from autograd.distributed_checkpoint import (
    distributed_checkpoint, distributed_graph, restore_distributed_state,
    save_distributed, DISTRIBUTED_HEADER_WORDS,
)
from autograd.program import OpsProgram, PROGRAM_IN_IDS
from autograd.program_runtime import replay_values
from autograd.program_pyapi import array_input, output_array
from autograd.program_arrays import initialize_program_arrays
from autograd.rankn import ShapeN
from autograd.cast_ops import upload_cast, upload_cast_i32
from autograd.optim import OptimizerConfig, optimizer_slots, optimizer_config_packed
from core.cast_in import CastIn
from training.checkpoint import read_checkpoint_unchecked
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime _DISTRIBUTED_NAME = "MojoCapturedDistributedIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoCapturedDistributedFast"
comptime _DISTRIBUTED = _Global[StorageType=_OpsStore, name=_DISTRIBUTED_NAME, init_fn=_OpsStore.__init__]


def distributed_devices(devices: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for k in range(len(devices)):  # small-loop(devices: physical device IDs): metadata
        var i = Int(py=devices[k])
        if i < 0:
            raise Error("distributed program: negative physical device ID")
        for j in range(len(out)):  # small-loop(out: preceding physical devices): reject aliasing
            if out[j] == i:
                raise Error("distributed program: duplicate physical device ID")
        out.append(i)
    if len(out) < 1:
        raise Error("distributed program: at least one physical device is required")
    return out^


def register_distributed[E: Exec & Movable & Deinitable, KIND: Int](var d: DistributedProgram[E]) raises -> PythonObject:
    var p = alloc[DistributedProgram[E]](1)
    p.init_pointee_move(d^)
    var s = _DISTRIBUTED.get_or_create_ptr()
    var id = s[].next_id
    s[].next_id += 1
    s[].ids.append(id)
    s[].addr.append(Int(p))
    s[].kind.append(KIND)
    return PythonObject(id)


def distributed_slot(id: Int, kind: Int) raises -> Int:
    var s = _DISTRIBUTED.get_or_create_ptr()
    for k in range(len(s[].ids)):  # driver-loop: live native distributed handles
        if s[].ids[k] == id and s[].kind[k] == kind:
            return k
    raise Error("distributed program handle is closed or belongs to another binding")


def distributed[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject) raises -> UnsafePointer[DistributedProgram[E], MutUntrackedOrigin]:
    var slot = distributed_slot(Int(py=id), KIND)
    return UnsafePointer[DistributedProgram[E], MutUntrackedOrigin](unsafe_from_address=_DISTRIBUTED.get_or_create_ptr()[].addr[slot])


def distributed_open[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, pid: PythonObject, device_ids: List[Int], logical_shards: PythonObject,
    optimizer: PythonObject, parameters: PythonObject, seed: PythonObject, var executors: List[E],
) raises -> PythonObject:
    if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("distributed programs require identical mode")
    var s = ops_session[E, KIND](Int(py=sid))
    var p = s[].program(Int(py=pid))
    var opt = Int(py=optimizer)
    var oi = s[]._opt_index(opt)
    var o = s[].opts[oi]
    var ip: List[Int] = [o.cfg.kind, 1 if o.cfg.nesterov else 0, o.sched, o.sched_step]
    var fp: List[Float32] = [o.cfg.lr, o.cfg.momentum, o.cfg.beta1, o.cfg.beta2, o.cfg.eps, o.cfg.weight_decay, o.gamma]
    var values = List[List[Float32]]()
    var trainable = List[Bool](length=len(p.params), fill=False)
    for k in range(len(parameters)):  # driver-loop: selected trainable handles
        var found = False
        var h = Int(py=parameters[k])
        for j in range(len(p.params)):  # driver-loop: captured parameter handles
            if p.params[j] == h:
                if trainable[j]:
                    raise Error("distributed program: duplicate trainable parameter")
                trainable[j] = True
                found = True
        if not found:
            raise Error("distributed program: trainable handle is not a captured parameter")
    for j in range(len(p.params)):  # driver-loop: copy canonical source parameter bytes
        if len(parameters) == 0:
            trainable[j] = not s[].int_words[s[]._slot(p.params[j])]
        if trainable[j] and s[].int_words[s[]._slot(p.params[j])]:
            raise Error("distributed program: integer parameters cannot be trainable")
        var data = List[Float32](length=p.param_shapes[j].count(), fill=Float32(0))
        s[].download(p.params[j], FP(unsafe_from_address=Int(data.unsafe_ptr())))
        values.append(data^)
    var seed_int = Int(py=seed)
    if seed_int < 0:
        raise Error("distributed program: seed must be a nonnegative signed 64-bit integer")
    var d = DistributedProgram[E](p, trainable, device_ids, Int(py=logical_shards), o.cfg, ip, fp, UInt64(seed_int))
    for rank in range(len(device_ids)):  # driver-loop: native context/replica construction
        var ex = executors.pop(0)
        d.workers.append(ProgramReplica[E](ex^, p, values, trainable, o.cfg, rank, len(device_ids)))
    # Migrate existing moments and powers too. The source remains independent
    # after construction. A global logical step requires equal selected clocks.
    var ck = distributed_checkpoint(d)
    var clock = -1
    var slots = optimizer_slots(o.cfg)
    var scalar_at = 7 + Int(ck.ext_ints[8])
    for j in range(len(p.params)):  # driver-loop: source optimizer migration
        if not trainable[j]:
            continue
        var si = -1
        for k in range(len(s[].opt_states)):  # driver-loop: native optimizer state table
            if s[].opt_states[k].opt == oi and s[].opt_states[k].param == p.params[j]:
                si = k
        var t = s[].opt_states[si].t if si >= 0 else 0
        if clock >= 0 and clock != t:
            raise Error("distributed program: selected parameters have different optimizer step counts")
        clock = t
        if si < 0:
            continue
        var n = p.param_shapes[j].count()
        var off = ck.offsets[j]
        if slots[0]:
            s[].ex.download(FP(unsafe_from_address=Int(ck.m_state.unsafe_ptr())) + off, s[].opt_states[si].s1, n)
        if slots[1]:
            s[].ex.download(FP(unsafe_from_address=Int(ck.v_state.unsafe_ptr())) + off, s[].opt_states[si].s2, n)
        ck.ext_floats[scalar_at + 3 * j] = s[].opt_states[si].pw1
        ck.ext_floats[scalar_at + 3 * j + 1] = s[].opt_states[si].pw2
        ck.ext_floats[scalar_at + 3 * j + 2] = s[].opt_states[si].mu_prod
    ck.t = max(0, clock)
    ck.ext_ints[2] = Int64(o.epoch)
    ck.ext_ints[3] = Int64(ck.t)
    restore_distributed_state(d, ck)
    return register_distributed[E, KIND](d^)


def distributed_load[E: Exec & Movable & Deinitable, KIND: Int](
    path: PythonObject, device_ids: List[Int], var executors: List[E],
) raises -> PythonObject:
    initialize_program_arrays()
    var ck = read_checkpoint_unchecked(String(py=path))
    var p = distributed_graph(ck)
    var ip = List[Int]()
    var fp = List[Float32]()
    var trainable = List[Bool]()
    var values = List[List[Float32]]()
    for k in range(4):  # small-loop(4: optimizer integer options): metadata
        ip.append(Int(ck.ext_ints[9 + k]))
    for k in range(7):  # small-loop(7: optimizer float options): metadata
        fp.append(ck.ext_floats[k])
    for j in range(len(p.params)):  # driver-loop: checkpoint parameter bytes
        trainable.append(ck.ext_ints[DISTRIBUTED_HEADER_WORDS + j] != 0)
        var data = List[Float32](length=p.param_shapes[j].count(), fill=Float32(0))
        memcpy(dest=data.unsafe_ptr(), src=ck.param.unsafe_ptr() + ck.offsets[j], count=len(data))
        values.append(data^)
    var cfg = distributed_optimizer_config(ip, fp)
    var d = DistributedProgram[E](p, trainable, device_ids, Int(ck.ext_ints[1]), cfg, ip, fp, ck.seed)
    var ex0 = executors.pop(0)
    d.workers.append(ProgramReplica[E](ex0^, p, values, trainable, cfg, 0, len(device_ids)))
    for rank in range(1, len(device_ids)):  # driver-loop: changed physical topology
        var ex = executors.pop(0)
        d.workers.append(ProgramReplica[E](ex^, p, values, trainable, cfg, rank, len(device_ids)))
    restore_distributed_state(d, ck)
    return register_distributed[E, KIND](d^)


def distributed_fit_py[E: Exec & Movable & Deinitable, KIND: Int](
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
        raise Error("distributed fit: unknown loss")
    var srcs = List[CastIn]()
    var shapes = List[ShapeN]()
    for k in range(len(arrays)):  # driver-loop: ABI input descriptors
        var parsed = array_input(arrays[k], "distributed input")
        srcs.append(parsed[0])
        shapes.append(parsed[1])
    var target = CastIn.f32_c(0, 1, 1)
    if kind != 0:
        target = array_input(targets, "distributed target")[0]
    var ne = Int(py=epochs)
    if ne < 1:
        raise Error("distributed fit: epochs must be positive")
    var curve = output_array(ShapeN.of([ne]))
    var p = distributed[E, KIND](id)
    p[].fit(srcs, shapes, target, kind, Float32(Float64(py=delta)), ne, Int(py=batch), Bool(py=shuffle),
            FP(unsafe_from_address=Int(py=curve.__array_interface__["data"][0])))
    return curve


def distributed_run_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject, arrays: PythonObject) raises -> PythonObject:
    var d = distributed[E, KIND](id)
    d[].ready()
    var p = d[].workers[0].program.copy()
    if len(arrays) != p.n_in:
        raise Error("distributed run: input count differs from captured program")
    ref w = d[].workers[0]
    w.session._drop_temps(w.temps)
    w.temps.clear()
    w.session.tape.reset()
    w.session.tape.no_grad()
    var buffers = List[FP]()
    var shapes = List[ShapeN]()
    try:
        for k in range(len(arrays)):  # driver-loop: inference input buffers
            var parsed = array_input(arrays[k], "distributed inference input")
            if parsed[1].ndim != p.in_shapes[k].ndim:
                raise Error("distributed inference input rank differs from capture")
            var buf = w.session.ex.alloc(parsed[1].count())
            buffers.append(buf)
            shapes.append(parsed[1])
            if p.in_kinds[k] == PROGRAM_IN_IDS:
                upload_cast_i32(w.session.ex, buf, parsed[0], "distributed ids")
            else:
                upload_cast(w.session.ex, buf, parsed[0])
        var phs = List[Int]()
        var trainable = List[Bool](length=len(w.params), fill=False)
        var outs = replay_values(w.session, p, buffers, shapes, w.params, trainable, w.temps, phs)
        var sh = w.session.shapes[w.session._slot(outs[0])]
        var result = output_array(sh)
        w.session.download(outs[0], FP(unsafe_from_address=Int(py=result.__array_interface__["data"][0])))
        w.session._drop_temps(w.temps)
        w.temps.clear()
        for k in range(len(buffers)):  # driver-loop: temporary inputs
            w.session.ex.free(buffers[k])
        return result
    except err:
        w.session._drop_temps(w.temps)
        w.temps.clear()
        for k in range(len(buffers)):  # driver-loop: failed inference cleanup
            w.session.ex.free(buffers[k])
        raise err


def distributed_save_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject, path: PythonObject) raises -> PythonObject:
    var d = distributed[E, KIND](id)
    return PythonObject(save_distributed(d[], String(py=path)))


def distributed_state_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject) raises -> PythonObject:
    var d = distributed[E, KIND](id)
    var ck = distributed_checkpoint(d[])
    var result = Python.dict()
    var params = Python.list()
    var first = Python.list()
    var second = Python.list()
    for j in range(len(d[].program.params)):  # driver-loop: returned canonical state tensors
        var sh = d[].program.param_shapes[j]
        var p = output_array(sh)
        var m = output_array(sh)
        var v = output_array(sh)
        memcpy(dest=FP(unsafe_from_address=Int(py=p.__array_interface__["data"][0])), src=ck.param.unsafe_ptr() + ck.offsets[j], count=sh.count())
        memcpy(dest=FP(unsafe_from_address=Int(py=m.__array_interface__["data"][0])), src=ck.m_state.unsafe_ptr() + ck.offsets[j], count=sh.count())
        memcpy(dest=FP(unsafe_from_address=Int(py=v.__array_interface__["data"][0])), src=ck.v_state.unsafe_ptr() + ck.offsets[j], count=sh.count())
        params.append(p)
        first.append(m)
        second.append(v)
    result["parameters"] = params
    result["first_moments"] = first
    result["second_moments"] = second
    result["epoch"] = PythonObject(d[].epoch)
    result["step"] = PythonObject(d[].step)
    result["seed"] = PythonObject(d[].seed)
    result["logical_shards"] = PythonObject(d[].logical_shards)
    return result


def distributed_residency_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject) raises -> PythonObject:
    var d = distributed[E, KIND](id)
    d[].ready()
    var result = Python.dict()
    var devices = Python.list()
    var owned = Python.list()
    var ranges = Python.list()
    for rank in range(len(d[].workers)):  # driver-loop: physical allocation witness
        devices.append(PythonObject(d[].devices[rank]))
        var count = 0
        var row = Python.list()
        for j in range(len(d[].program.params)):  # driver-loop: owned optimizer intervals
            var pair = Python.list()
            pair.append(PythonObject(d[].workers[rank].first[j]))
            pair.append(PythonObject(d[].workers[rank].width[j]))
            row.append(pair)
            if d[].trainable[j]:
                count += d[].workers[rank].width[j]
        owned.append(PythonObject(count))
        ranges.append(row)
    result["devices"] = devices
    result["optimizer_elements_per_rank"] = owned
    result["optimizer_ranges"] = ranges
    result["parameter_residency"] = PythonObject("replicated")
    result["optimizer_residency"] = PythonObject("disjoint contiguous ranges")
    result["logical_shards"] = PythonObject(d[].logical_shards)
    result["transport"] = PythonObject("native host staging")
    return result


def distributed_close_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject) raises -> PythonObject:
    var table = _DISTRIBUTED.get_or_create_ptr()
    var slot = distributed_slot(Int(py=id), KIND)
    var p = distributed[E, KIND](id)
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


def distributed_optimizer_config(ip: List[Int], fp: List[Float32]) raises -> OptimizerConfig:
    # The file carries the same three optimizer kinds as OpsSession.
    # Validate before allocating state or accepting corrupted configuration.
    if ip[0] != OPT_SGD and ip[0] != OPT_ADAM and ip[0] != OPT_ADAMW:
        raise Error("distributed checkpoint: unknown optimizer")
    if ip[1] < 0 or ip[1] > 1 or ip[2] < 0 or ip[2] > 2 or ip[3] < 1:
        raise Error("distributed checkpoint: invalid optimizer or schedule flags")
    if not (fp[0] > 0 and fp[1] >= 0 and fp[1] < 1 and fp[2] >= 0 and fp[2] < 1
            and fp[3] >= 0 and fp[3] < 1 and fp[4] > 0 and fp[5] >= 0):
        raise Error("distributed checkpoint: invalid optimizer hyperparameters")
    if ip[2] != 0 and not fp[6] > 0:
        raise Error("distributed checkpoint: invalid schedule gamma")
    if ip[1] != 0 and (ip[0] != OPT_SGD or not fp[1] > 0):
        raise Error("distributed checkpoint: invalid Nesterov configuration")
    return optimizer_config_packed(ip[0], ip[1], fp[1] if ip[0] == OPT_SGD else fp[2],
                                     Float32(0) if ip[0] == OPT_SGD else fp[3], fp[4], fp[5], Float32(0), fp[0])
