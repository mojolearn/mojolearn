# SPDX-License-Identifier: Apache-2.0
# Portable complete state of a partitioned captured DAG. Logical placement is
# retained; physical device IDs are intentionally absent from the file.
from std.memory import memcpy
from sequence.exec_trait import Exec
from sequence.ops import FP
from autograd.partition_plan import PartitionPlan, GraphBytes
from autograd.partition_runtime import PartitionedProgram
from autograd.program import program_words, program_from_words
from autograd.optim import optimizer_slots
from training.checkpoint import Checkpoint, save_checkpoint, CKPT_SECTION_PARTITIONED_PROGRAM

comptime PARTITION_RECORD_VERSION = 1
comptime PARTITION_HEADER = 17


def partition_checkpoint[E: Exec & Movable & Deinitable](mut d: PartitionedProgram[E]) raises -> Checkpoint:
    d.ready()
    var ck = Checkpoint()
    ck.section_kind = CKPT_SECTION_PARTITIONED_PROGRAM
    ck.seed = d.seed
    ck.t = d.step
    ck.opt_kind = d.cfg.kind
    ck.lr = d.cfg.lr
    ck.beta1 = d.cfg.beta1
    ck.beta2 = d.cfg.beta2
    ck.eps = d.cfg.eps
    ck.weight_decay = d.cfg.weight_decay
    ck.momentum = d.cfg.momentum
    ck.nesterov = d.cfg.nesterov
    var p = d.plan.program.copy()
    var words = program_words(p)
    ck.ext_ints = [Int64(PARTITION_RECORD_VERSION), Int64(d.plan.stages), Int64(d.microbatches),
        Int64(d.plan.tensor_shards), Int64(d.epoch), Int64(d.step), Int64(d.seed & 0xffffffff),
        Int64(d.seed >> 32), Int64(len(p.params)), Int64(len(words[0])), Int64(len(words[1])), Int64(p.n_nodes())]
    for k in range(4):  # small-loop(4: optimizer integer options): metadata
        ck.ext_ints.append(Int64(d.optimizer_ints[k]))
    ck.ext_ints.append(Int64(d.plan.partitioned_products))
    for j in range(len(p.params)):  # driver-loop: parameter semantics
        ck.ext_ints.append(Int64(1) if d.plan.trainable[j] else Int64(0))
        ck.ext_ints.append(Int64(1) if d.plan.parameters[j].integer else Int64(0))
    for k in range(p.n_nodes()):  # driver-loop: logical operation placement
        ck.ext_ints.append(Int64(d.plan.placement[k]))
    for k in range(len(words[0])):  # driver-loop: captured graph metadata
        ck.ext_ints.append(words[0][k])
    ck.ext_floats = d.optimizer_floats.copy()
    for k in range(len(words[1])):  # driver-loop: captured graph metadata
        ck.ext_floats.append(words[1][k])
    var total = 0
    ck.offsets.append(0)
    for j in range(len(p.params)):  # driver-loop: canonical transformed parameter layout
        total += p.param_shapes[j].count()
        ck.offsets.append(total)
        ck.names.append(String("p") + String(j))
        ck.buf_initialized.append(d.plan.trainable[j] and d.step > 0)
    ck.param = List[Float32](length=total, fill=Float32(0))
    ck.m_state = List[Float32](length=total, fill=Float32(0))
    ck.v_state = List[Float32](length=total, fill=Float32(0))
    var slots = optimizer_slots(d.cfg)
    for j in range(len(p.params)):  # driver-loop: each authoritative owner once
        var owner = d.plan.owners[j] % len(d.workers)
        var n = p.param_shapes[j].count()
        var off = ck.offsets[j]
        d.workers[owner].session.ex.download(FP(unsafe_from_address=Int(ck.param.unsafe_ptr())) + off,
                                             d.workers[owner].params[j], n)
        if d.plan.trainable[j]:
            if slots[0]:
                d.workers[owner].session.ex.download(FP(unsafe_from_address=Int(ck.m_state.unsafe_ptr())) + off,
                                                     d.workers[owner].states[j].s1, n)
            if slots[1]:
                d.workers[owner].session.ex.download(FP(unsafe_from_address=Int(ck.v_state.unsafe_ptr())) + off,
                                                     d.workers[owner].states[j].s2, n)
            ck.ext_floats.append(d.workers[owner].states[j].pw1)
            ck.ext_floats.append(d.workers[owner].states[j].pw2)
            ck.ext_floats.append(d.workers[owner].states[j].mu_prod)
        else:
            ck.ext_floats.append(Float32(1))
            ck.ext_floats.append(Float32(1))
            ck.ext_floats.append(Float32(1))
    return ck^


def partition_from_checkpoint(ck: Checkpoint) raises -> PartitionPlan:
    var x = ck.ext_ints.copy()
    if ck.section_kind != CKPT_SECTION_PARTITIONED_PROGRAM or len(x) < PARTITION_HEADER or x[0] != PARTITION_RECORD_VERSION:
        raise Error("not a supported partitioned-program checkpoint")
    var K = Int(x[8])
    var ni = Int(x[9])
    var nf = Int(x[10])
    var nodes = Int(x[11])
    if K < 1 or ni < 1 or nf < 0 or nodes < 1 or K != ck.j_count():
        raise Error("damaged partitioned checkpoint dimensions")
    if len(x) != PARTITION_HEADER + 2 * K + nodes + ni or len(ck.ext_floats) != 7 + nf + 3 * K:
        raise Error("damaged partitioned checkpoint record")
    if x[1] < 1 or x[2] < 1 or x[3] < 1 or x[3] > x[1] or x[4] < 0 or x[5] < 0 or ck.t != Int(x[5]):
        raise Error("damaged partitioned checkpoint counters")
    if x[6] < 0 or x[6] > 0xffffffff or x[7] < 0 or x[7] > 0xffffffff or ck.seed != ((UInt64(x[7]) << 32) | UInt64(x[6])):
        raise Error("damaged partitioned checkpoint seed")
    var gi = List[Int64]()
    var gf = List[Float32]()
    for k in range(ni):  # driver-loop: graph metadata
        gi.append(x[PARTITION_HEADER + 2 * K + nodes + k])
    for k in range(nf):  # driver-loop: graph metadata
        gf.append(ck.ext_floats[7 + k])
    var plan = PartitionPlan()
    plan.program = program_from_words(gi, gf, "partitioned checkpoint")
    plan.stages = Int(x[1])
    plan.tensor_shards = Int(x[3])
    plan.partitioned_products = Int(x[16])
    if plan.partitioned_products < 0 or plan.partitioned_products > nodes:
        raise Error("damaged tensor-partition count")
    if plan.program.n_nodes() != nodes or len(plan.program.params) != K:
        raise Error("partitioned graph differs from checkpoint layout")
    for k in range(nodes):  # driver-loop: logical stage admission
        var stage = Int(x[PARTITION_HEADER + 2 * K + k])
        if stage < 0 or stage >= plan.stages:
            raise Error("partitioned checkpoint has an invalid logical stage")
        plan.placement.append(stage)
    for j in range(K):  # driver-loop: exact parameter bytes
        var sh = plan.program.param_shapes[j]
        if ck.offsets[j + 1] - ck.offsets[j] != sh.count() or ck.names[j] != String("p") + String(j):
            raise Error("partitioned parameter shape/name differs from checkpoint layout")
        var train = Int(x[PARTITION_HEADER + 2 * j])
        var integer = Int(x[PARTITION_HEADER + 2 * j + 1])
        if (train != 0 and train != 1) or (integer != 0 and integer != 1) or (train != 0 and integer != 0):
            raise Error("partitioned checkpoint has invalid parameter flags")
        var data = List[Float32](length=sh.count(), fill=Float32(0))
        memcpy(dest=data.unsafe_ptr(), src=ck.param.unsafe_ptr() + ck.offsets[j], count=sh.count())
        plan.parameters.append(GraphBytes(sh, data^, integer != 0))
        plan.trainable.append(train != 0)
    plan.schedule()
    return plan^


def restore_partition_state[E: Exec & Movable & Deinitable](mut d: PartitionedProgram[E], ck: Checkpoint) raises:
    d.epoch = Int(ck.ext_ints[4])
    d.step = ck.t
    var slots = optimizer_slots(d.cfg)
    var scalar = 7 + Int(ck.ext_ints[10])
    for j in range(len(d.plan.parameters)):  # driver-loop: owner-specific optimizer restore
        if not d.plan.trainable[j]:
            continue
        var owner = d.plan.owners[j] % len(d.workers)
        var n = d.plan.parameters[j].shape.count()
        var off = ck.offsets[j]
        if slots[0]:
            d.workers[owner].session.ex.upload(d.workers[owner].states[j].s1,
                FP(unsafe_from_address=Int(ck.m_state.unsafe_ptr())) + off, n)
        if slots[1]:
            d.workers[owner].session.ex.upload(d.workers[owner].states[j].s2,
                FP(unsafe_from_address=Int(ck.v_state.unsafe_ptr())) + off, n)
        d.workers[owner].states[j].pw1 = ck.ext_floats[scalar + 3 * j]
        d.workers[owner].states[j].pw2 = ck.ext_floats[scalar + 3 * j + 1]
        d.workers[owner].states[j].mu_prod = ck.ext_floats[scalar + 3 * j + 2]
    for rank in range(len(d.workers)):  # driver-loop: restore completion
        d.workers[rank].session.ex.sync()
