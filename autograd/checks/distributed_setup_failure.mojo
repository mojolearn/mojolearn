# SPDX-License-Identifier: Apache-2.0
# Testing-only fault boundaries, backed by the real HostExec allocation table.
# No numerical/performance claim or Python product execution is involved.
from autograd.checks.program_setup_failure import FaultExec, require
from autograd.distributed_program import ProgramReplica, DistributedProgram
from autograd.partition_runtime import PartitionedProgram
from autograd.partition_plan import GraphBytes, partition_plan
from autograd.partition_stage import stage_input_bytes
from autograd.program import OpsProgram
from autograd.ops_api import OPS_MATMUL
from autograd.optim import optimizer_config
from autograd.rankn import ShapeN
from core.cast_in import CastIn, CAST_KIND_FLOAT, CAST_KIND_SIGNED
from sequence.ops import FP, OPT_SGD


def fixture() -> OpsProgram:
    var p = OpsProgram()
    p.n_in = 1
    p.in_shapes.append(ShapeN.matrix(2, 1))
    p.in_kinds.append(0)
    p.params.append(1)
    p.param_shapes.append(ShapeN.matrix(1, 1))
    p.code.append(OPS_MATMUL)
    p.args.append(0)
    p.args.append(-1)
    p.end_node()
    p.outs.append(1)
    p.out_shapes.append(ShapeN.matrix(2, 1))
    return p^


def weights() -> List[List[Float32]]:
    var values = List[List[Float32]]()
    values.append(List[Float32](length=1, fill=2.0))
    return values^


def check_replica(kind: Int, at: Int) raises:
    var p = fixture()
    var selected = List[Bool](length=1, fill=True)
    var cfg = optimizer_config(OPT_SGD, 0.01)
    var replica = ProgramReplica[FaultExec](FaultExec(), p, weights(), selected, cfg, 0, 1)
    var data = List[Float32](length=2, fill=1.0)
    var target = List[Float32](length=2, fill=0.0)
    var sources = List[CastIn]()
    sources.append(CastIn.f32_c(Int(data.unsafe_ptr()), 2, 1))
    var shapes = p.in_shapes.copy()
    var y = CastIn.f32_c(Int(target.unsafe_ptr()), 2, 1)
    var before = len(replica.session.ex.host.bufs)
    replica.session.ex.arm(kind, at)
    var failed = False
    try:
        replica.prepare(sources, shapes, y, 1, 1)
    except error:
        require(String(error) == "injected setup failure", "replica masked original fault: " + String(error))
        failed = True
    require(failed and replica.session.ex.hit, "replica fault not reached")
    require(len(replica.session.ex.host.bufs) == before, "replica retained partial dataset until close")
    require(not replica.dataset_ready and len(replica.full) == 0 and len(replica.batches) == 0,
            "replica partial setup remains published")
    replica.session.ex.arm(0, 0)
    replica.prepare(sources, shapes, y, 1, 1)
    require(replica.dataset_ready and len(replica.session.ex.host.bufs) == before + 5,
            "replica positive setup lost its owners")
    replica.release_dataset()
    replica.release_dataset()
    require(len(replica.session.ex.host.bufs) == before, "replica release is not idempotent")
    replica.close()
    require(len(replica.session.ex.host.bufs) == 0, "replica close retained registered model storage")
    _ = data^
    _ = target^


def check_partition(kind: Int, at: Int) raises:
    var p = fixture()
    var selected = List[Bool](length=1, fill=True)
    var values = List[GraphBytes]()
    values.append(GraphBytes(ShapeN.matrix(1, 1), List[Float32](length=1, fill=2.0)))
    var plan = partition_plan(p, values, selected, 2, 1)
    var devices = List[Int](length=1, fill=0)
    var executors = List[FaultExec]()
    executors.append(FaultExec())
    var integers: List[Int] = [OPT_SGD, 0, 0, 1]
    var scalars = List[Float32](length=7, fill=0.0)
    scalars[6] = 1.0
    var model = PartitionedProgram[FaultExec](plan^, devices, executors^,
        optimizer_config(OPT_SGD, 0.01), integers, scalars, 2, UInt64(7))
    var data = List[Float32](length=2, fill=1.0)
    var target = List[Float32](length=2, fill=0.0)
    var curve = List[Float32](length=1, fill=0.0)
    var sources = List[CastIn]()
    sources.append(CastIn.f32_c(Int(data.unsafe_ptr()), 2, 1))
    var y = CastIn.f32_c(Int(target.unsafe_ptr()), 2, 1)
    var before = len(model.workers[0].session.ex.host.bufs)
    model.workers[0].session.ex.arm(kind, at)
    var failed = False
    try:
        model.fit(sources, p.in_shapes, y, 1, Float32(1.0), 1, 1, False,
                  FP(unsafe_from_address=Int(curve.unsafe_ptr())))
    except error:
        require(String(error) == "injected setup failure", "partition masked original fault: " + String(error))
        failed = True
    require(failed and model.workers[0].session.ex.hit, "partition fault not reached")
    require(len(model.workers[0].session.ex.host.bufs) == before, "partition retained partial dataset until close")
    require(not model.busy and not model.usable, "failed partition has inconsistent admission state")
    require(model.epoch == 0 and model.step == 0, "partition setup fault advanced optimizer clock")
    model.workers[0].session.ex.arm(0, 0)
    model.close()
    require(len(model.workers) == 0, "partition close retained workers")
    _ = data^
    _ = target^
    _ = curve^


def check_staging(integer: Bool, kind: Int) raises:
    var ex = FaultExec()
    var floats = List[Float64](length=2, fill=1.0)
    var integers = List[Int64](length=2, fill=1)
    var source = CastIn(Int(integers.unsafe_ptr()) if integer else Int(floats.unsafe_ptr()),
        CAST_KIND_SIGNED if integer else CAST_KIND_FLOAT, 8, 2, 1, 8, 8)
    ex.arm(kind, 1)
    var failed = False
    try:
        _ = stage_input_bytes(ex, source, ShapeN.matrix(2, 1), integer)
    except error:
        require(String(error) == "injected setup failure", "staging masked original fault: " + String(error))
        failed = True
    require(failed and ex.hit, "staging fault not reached")
    require(len(ex.host.bufs) == 0, "failed staging retained conversion storage")
    ex.arm(0, 0)
    var result = stage_input_bytes(ex, source, ShapeN.matrix(2, 1), integer)
    require(result.shape.count() == 2 and result.integer == integer, "staging positive metadata changed")
    require(len(ex.host.bufs) == 0, "successful staging retained conversion storage")
    _ = floats^
    _ = integers^


def check_prepared_peer() raises:
    var p = fixture()
    var selected = List[Bool](length=1, fill=True)
    var devices: List[Int] = [0, 1]
    var integers: List[Int] = [OPT_SGD, 0, 0, 1]
    var scalars = List[Float32](length=7, fill=0.0)
    scalars[6] = 1.0
    var cfg = optimizer_config(OPT_SGD, 0.01)
    var model = DistributedProgram[FaultExec](p, selected, devices, 2, cfg, integers, scalars, UInt64(7))
    var before = List[Int]()
    for rank in range(2):
        model.workers.append(ProgramReplica[FaultExec](FaultExec(), p, weights(), selected, cfg, rank, 2))
        before.append(len(model.workers[rank].session.ex.host.bufs))
    model.workers[1].session.ex.arm(1, 1)
    var data = List[Float32](length=2, fill=1.0)
    var target = List[Float32](length=2, fill=0.0)
    var curve = List[Float32](length=1, fill=0.0)
    var sources = List[CastIn]()
    sources.append(CastIn.f32_c(Int(data.unsafe_ptr()), 2, 1))
    var failed = False
    try:
        model.fit(sources, p.in_shapes, CastIn.f32_c(Int(target.unsafe_ptr()), 2, 1), 1,
            Float32(1.0), 1, 1, False, FP(unsafe_from_address=Int(curve.unsafe_ptr())))
    except error:
        require(String(error) == "injected setup failure", "peer setup masked original error")
        failed = True
    require(failed and model.workers[1].session.ex.hit, "peer setup fault not reached")
    for rank in range(2):
        require(len(model.workers[rank].session.ex.host.bufs) == before[rank], "prepared peer retained dataset")
        model.workers[rank].session.ex.arm(0, 0)
    require(not model.busy and not model.usable and model.step == 0, "failed coordinator changed state")
    model.close()
    _ = data^
    _ = target^
    _ = curve^


def main() raises:
    var cases = 0
    for at in range(1, 6):
        check_replica(1, at)
        check_partition(1, at)
        cases += 2
    for at in range(1, 3):
        check_replica(2, at)
        check_partition(2, at)
        cases += 2
    for integer in range(2):
        for kind in range(1, 4):
            check_staging(integer != 0, kind)
            cases += 1
    check_prepared_peer()
    cases += 1
    print("PASS distributed setup exception ownership", cases, "fault cases")
