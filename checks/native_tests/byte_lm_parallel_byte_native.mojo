# SPDX-License-Identifier: Apache-2.0
"""The cooperative byte LM trainer's decay-only step and its single-device
replay, ported from `test_parallel_byte_native.py::
test_gpu_parallel_step_independent_decay_loss_and_complete_replay` (lane
tmig-lm-1, 2026-11-14).

The fixture is the Python `state_fixture`: a [1, 4] shape with d_model 8,
vocabulary 16 and one layer, every parameter zero except 0.25 in the unused
last embedding row, AdamW lr 0.125, betas (0.5, 0.75), eps 1e-8, weight decay
0.25, three logical shards of ids 0..4. Every consumed embedding and hidden
state is zero, so each shard's loss is log(16) and the gradient is zero:
the step is weight decay alone, `p * (1 - 0.125 * 0.25)`, with both moments
and the flags still zero. A replay opened on the first device from the
exported state and stepped beside the cooperative trainer must end on the
same bytes.

It runs `training/byte_lm_parallel.mojo::ByteParallelTrainer` and the
replica export the door calls (`training/byte_lm_door_run.mojo::
byte_parallel_export_run`). Devices come from MOJOLEARN_PARALLEL_BYTE_DEVICES
(default "0"), as in the Python test. One fixture, one trainer pair: the
shards already run on every listed device at once.
"""
from std.os import getenv
from std.math import log
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from training.byte_lm_config import ByteConfig
from training.byte_lm_parallel import ByteParallelTrainer
from training.byte_lm_door_run import byte_parallel_export_run
from training.checks.optimizer_contract import OptimizerConfig, OPT_ADAMW
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


# ---- The device column (was byte_lm_parallel_byte_native_gpu_suite.mojo) ----
def devices() raises -> List[Int]:
    var text = String(getenv("MOJOLEARN_PARALLEL_BYTE_DEVICES"))
    var out = List[Int]()
    if text == "":
        out.append(0)
        return out^
    for part in text.split(","):  # driver-loop: over the listed device ids
        out.append(Int(atol(String(part))))
    return out^


@fieldwise_init
struct Exported(Movable):
    var p: List[Float32]
    var m: List[Float32]
    var v: List[Float32]
    var flags: List[Int32]


def export_replica(mut owner: ByteParallelTrainer, cfg: ByteConfig) raises -> Exported:
    var n = cfg.n_total()
    var state = Exported(List[Float32](length=n, fill=Float32(-1)), List[Float32](length=n, fill=Float32(-1)),
                         List[Float32](length=n, fill=Float32(-1)), List[Int32](length=cfg.n_tensors(), fill=Int32(-1)))
    var a: List[Int] = [Int(state.p.unsafe_ptr()), Int(state.m.unsafe_ptr()), Int(state.v.unsafe_ptr()),
                        Int(state.flags.unsafe_ptr())]
    _ = byte_parallel_export_run(owner, a, 0, False)
    return state^


def gradient(mut owner: ByteParallelTrainer, cfg: ByteConfig) raises -> List[Float32]:
    var g = List[Float32](length=cfg.n_total(), fill=Float32(-1))
    var a: List[Int] = [Int(g.unsafe_ptr())]
    _ = byte_parallel_export_run(owner, a, 0, True)
    return g^


def same_f32(a: List[Float32], b: List[Float32], label: String) raises:
    require(len(a) == len(b), label + ": lengths differ")
    for i in range(len(a)):  # driver-loop: over the fixture's compared cells
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            raise Error(label + ": bits differ at " + String(i))


def same_i32(a: List[Int32], b: List[Int32], label: String) raises:
    require(len(a) == len(b), label + ": lengths differ")
    for i in range(len(a)):  # driver-loop: over the fixture's compared cells
        if a[i] != b[i]:
            raise Error(label + ": words differ at " + String(i))


def shards() -> List[List[Int32]]:
    var out = List[List[Int32]]()
    for _ in range(3):  # driver-loop: over the three logical shards
        var ids: List[Int32] = [0, 1, 2, 3, 4]
        out.append(ids^)
    return out^


def _run_gpu(mut report: Report) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var cfg = ByteConfig(batch=1, length=4, d_model=8, n_heads=2, n_kv=1, head_dim=4, intermediate=16,
                         n_layers=1, vocab_size=16, chunked_lm_head_v2=False)
    var n = cfg.n_total()
    var nt = cfg.n_tensors()
    var decay_index = cfg.offsets()[0] + (cfg.vocab_size - 1) * cfg.d_model
    var p = List[Float32](length=n, fill=Float32(0))
    p[decay_index] = Float32(0.25)
    var zeros = List[Float32](length=n, fill=Float32(0))
    var flags = List[Bool](length=nt, fill=False)
    var opt = OptimizerConfig(OPT_ADAMW, 0.125, 0.5, 0.75, 1e-8, 0.25, 0, 0, False, 0)
    var ids = devices()
    var owner = ByteParallelTrainer()
    owner.open(ids, 3, p, zeros, zeros, flags, 0, opt, cfg, False)
    var replay = ByteParallelTrainer()
    try:
        var losses = owner.step(shards())
        var expected_loss = log(Float64(cfg.vocab_size))
        for k in range(3):  # driver-loop: over the three logical shards
            require(abs(Float64(losses[k]) - expected_loss) <= 2e-6,
                    "shard " + String(k) + " loss " + String(losses[k]) + " is not log(16)")
        require(owner.trainers[0].completed_steps == 1, "the cooperative step did not complete one step")
        var exported = export_replica(owner, cfg)
        var decayed = List[Float32](capacity=n)
        for i in range(n):  # driver-loop: over the fixture's parameter cells
            decayed.append(p[i] * Float32(1.0 - 0.125 * 0.25))
        same_f32(exported.p, decayed, "decay-only parameters")
        same_f32(exported.m, zeros, "first moment")
        same_f32(exported.v, zeros, "second moment")
        same_i32(exported.flags, List[Int32](length=nt, fill=Int32(0)), "flags")
        same_f32(gradient(owner, cfg), zeros, "summed gradient")
        var replay_flags = List[Bool](length=nt, fill=False)
        for i in range(nt):  # driver-loop: over the parameter tensors' flags
            replay_flags[i] = exported.flags[i] != 0
        var first: List[Int] = [ids[0]]
        replay.open(first, 3, exported.p, exported.m, exported.v, replay_flags, 1, opt, cfg, False)
        _ = owner.step(shards())
        _ = replay.step(shards())
        var a = export_replica(owner, cfg)
        var b = export_replica(replay, cfg)
        same_f32(a.p, b.p, "replay parameters")
        same_f32(a.m, b.m, "replay first moment")
        same_f32(a.v, b.v, "replay second moment")
        same_i32(a.flags, b.flags, "replay flags")
        require(owner.trainers[0].completed_steps == 2 and replay.trainers[0].completed_steps == 2,
                "the replay and the cooperative trainer disagree on the step count")
    finally:
        owner.close()
        replay.close()
    print("PASS byte_lm_parallel_byte_native_gpu: decay-only step on", len(ids), "devices and its replay")


def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("byte_lm_parallel_byte_native", cpu=False, gpu=True, serial="", run=run_cases)
