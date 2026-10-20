# SPDX-License-Identifier: Apache-2.0
# Complete portable state for native captured-program data parallelism.
# Files contain the canonical unpartitioned bytes, never device IDs or just
# hashes. Restoring repartitions those bytes onto the requested topology.
from std.memory import memcpy
from sequence.exec_trait import Exec
from sequence.ops import FP
from autograd.distributed_program import DistributedProgram
from autograd.program import OpsProgram, program_words, program_from_words
from autograd.optim import optimizer_slots
from training.checkpoint import Checkpoint, save_checkpoint, read_checkpoint_unchecked, CKPT_SECTION_DISTRIBUTED_PROGRAM

comptime DISTRIBUTED_RECORD_VERSION = 1
comptime DISTRIBUTED_HEADER_WORDS = 13


def distributed_checkpoint[E: Exec & Movable & Deinitable](mut d: DistributedProgram[E]) raises -> Checkpoint:
    d.ready()
    var ck = Checkpoint()
    ck.section_kind = CKPT_SECTION_DISTRIBUTED_PROGRAM
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
    var p = d.program.copy()
    var words = program_words(p)
    # Header: version, logical shards, epoch, step, seed low/high, K,
    # graph integer/float lengths, optimizer's four option integers.
    ck.ext_ints = [Int64(DISTRIBUTED_RECORD_VERSION), Int64(d.logical_shards), Int64(d.epoch),
                   Int64(d.step), Int64(d.seed & 0xffffffff), Int64(d.seed >> 32), Int64(len(p.params)),
                   Int64(len(words[0])), Int64(len(words[1]))]
    for k in range(4):  # small-loop(4: optimizer option words): metadata
        ck.ext_ints.append(Int64(d.optimizer_ints[k]))
    for k in range(len(d.trainable)):  # driver-loop: parameter flags
        ck.ext_ints.append(Int64(1) if d.trainable[k] else Int64(0))
    for k in range(len(words[0])):  # driver-loop: serialized graph metadata
        ck.ext_ints.append(words[0][k])
    ck.ext_floats = d.optimizer_floats.copy()
    for k in range(len(words[1])):  # driver-loop: graph float arguments
        ck.ext_floats.append(words[1][k])
    var total = 0
    ck.offsets.append(0)
    for j in range(len(p.params)):  # driver-loop: canonical tensor layout
        total += p.param_shapes[j].count()
        ck.offsets.append(total)
        ck.names.append(String("p") + String(j))
        ck.buf_initialized.append(d.trainable[j] and d.step > 0)
    ck.param = List[Float32](length=total, fill=Float32(0))
    ck.m_state = List[Float32](length=total, fill=Float32(0))
    ck.v_state = List[Float32](length=total, fill=Float32(0))
    var slots = optimizer_slots(d.cfg)
    for j in range(len(p.params)):  # driver-loop: physical owners to canonical bytes
        var off = ck.offsets[j]
        d.workers[0].session.ex.download(FP(unsafe_from_address=Int(ck.param.unsafe_ptr())) + off,
                                         d.workers[0].params[j], p.param_shapes[j].count())
        var scalar_owner = -1
        for rank in range(len(d.workers)):  # driver-loop: each owned slice once
            var n = d.workers[rank].width[j]
            if n == 0 or not d.trainable[j]:
                continue
            scalar_owner = rank
            var at = off + d.workers[rank].first[j]
            if slots[0]:
                d.workers[rank].session.ex.download(FP(unsafe_from_address=Int(ck.m_state.unsafe_ptr())) + at,
                                                    d.workers[rank].states[j].s1, n)
            if slots[1]:
                d.workers[rank].session.ex.download(FP(unsafe_from_address=Int(ck.v_state.unsafe_ptr())) + at,
                                                    d.workers[rank].states[j].s2, n)
        if scalar_owner >= 0:
            ck.ext_floats.append(d.workers[scalar_owner].states[j].pw1)
            ck.ext_floats.append(d.workers[scalar_owner].states[j].pw2)
            ck.ext_floats.append(d.workers[scalar_owner].states[j].mu_prod)
        else:
            ck.ext_floats.append(Float32(1))
            ck.ext_floats.append(Float32(1))
            ck.ext_floats.append(Float32(1))
    return ck^


def save_distributed[E: Exec & Movable & Deinitable](mut d: DistributedProgram[E], path: String) raises -> Int:
    var ck = distributed_checkpoint(d)
    return save_checkpoint(path, ck).bytes


def distributed_graph(ck: Checkpoint) raises -> OpsProgram:
    var x = ck.ext_ints.copy()
    if ck.section_kind != CKPT_SECTION_DISTRIBUTED_PROGRAM or len(x) < DISTRIBUTED_HEADER_WORDS:
        raise Error("not a distributed captured-program checkpoint")
    if x[0] != DISTRIBUTED_RECORD_VERSION:
        raise Error("unknown distributed checkpoint record version")
    var K = Int(x[6])
    var ni = Int(x[7])
    var nf = Int(x[8])
    if K < 1 or ni < 1 or nf < 0 or len(x) != DISTRIBUTED_HEADER_WORDS + K + ni:
        raise Error("damaged distributed graph integer record")
    if len(ck.ext_floats) != 7 + nf + 3 * K or K != ck.j_count():
        raise Error("damaged distributed graph float record")
    if x[1] < 1 or x[2] < 0 or x[3] < 0 or ck.t != x[3]:
        raise Error("damaged distributed training counters")
    if x[4] < 0 or x[4] > 0xffffffff or x[5] < 0 or x[5] > 0xffffffff:
        raise Error("damaged distributed random seed")
    if ck.seed != ((UInt64(x[5]) << 32) | UInt64(x[4])):
        raise Error("distributed checkpoint seed disagrees with descriptor")
    var gi = List[Int64]()
    var gf = List[Float32]()
    for k in range(ni):  # driver-loop: graph metadata
        gi.append(x[DISTRIBUTED_HEADER_WORDS + K + k])
    for k in range(nf):  # driver-loop: graph metadata
        gf.append(ck.ext_floats[7 + k])
    var p = program_from_words(gi, gf, "distributed checkpoint")
    if len(p.params) != K or len(p.outs) != 1 or p.n_in < 1:
        raise Error("distributed checkpoint graph does not match its parameter table")
    for j in range(K):  # driver-loop: tensor and trainability contracts
        if x[DISTRIBUTED_HEADER_WORDS + j] != 0 and x[DISTRIBUTED_HEADER_WORDS + j] != 1:
            raise Error("damaged distributed trainable flag")
        if ck.offsets[j + 1] - ck.offsets[j] != p.param_shapes[j].count() or ck.names[j] != String("p") + String(j):
            raise Error("distributed checkpoint tensor shape or name mismatch")
    return p^


def restore_distributed_state[E: Exec & Movable & Deinitable](mut d: DistributedProgram[E], ck: Checkpoint) raises:
    d.epoch = Int(ck.ext_ints[2])
    d.step = ck.t
    var scalar_at = 7 + Int(ck.ext_ints[8])
    var slots = optimizer_slots(d.cfg)
    for rank in range(len(d.workers)):  # driver-loop: new physical ownership
        for j in range(len(d.program.params)):  # driver-loop: state shards
            if not d.trainable[j]:
                continue
            var n = d.workers[rank].width[j]
            var at = ck.offsets[j] + d.workers[rank].first[j]
            if n > 0 and slots[0]:
                d.workers[rank].session.ex.upload(d.workers[rank].states[j].s1,
                    FP(unsafe_from_address=Int(ck.m_state.unsafe_ptr())) + at, n)
            if n > 0 and slots[1]:
                d.workers[rank].session.ex.upload(d.workers[rank].states[j].s2,
                    FP(unsafe_from_address=Int(ck.v_state.unsafe_ptr())) + at, n)
            d.workers[rank].states[j].pw1 = ck.ext_floats[scalar_at + 3 * j]
            d.workers[rank].states[j].pw2 = ck.ext_floats[scalar_at + 3 * j + 1]
            d.workers[rank].states[j].mu_prod = ck.ext_floats[scalar_at + 3 * j + 2]
        d.workers[rank].session.ex.sync()
