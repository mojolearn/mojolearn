# Compile-only CPU distributed fit scheduling, worker replay and lifecycle.
from sequence.exec import HostExec
from sequence.ops import FP
from core.cast_in import CastIn
from autograd.rankn import ShapeN
from autograd.program import OpsProgram
from autograd.optim import OptimizerConfig
from autograd.distributed_program import ProgramReplica, DistributedProgram

@export
def compile_fit(mut d: DistributedProgram[HostExec], srcs: List[CastIn], shapes: List[ShapeN],
                target: CastIn, kind: Int, delta: Float32, epochs: Int, batch: Int,
                shuffle: Bool, curve: FP) raises:
    d.fit(srcs, shapes, target, kind, delta, epochs, batch, shuffle, curve)

@export
def compile_worker(p: OpsProgram, values: List[List[Float32]], trainable: List[Bool],
                   cfg: OptimizerConfig, rank: Int, world: Int) raises:
    var w = ProgramReplica(HostExec(), p, values, trainable, cfg, rank, world)
    w.close()
