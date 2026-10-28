# Focused compile-only HostExec training/control and replay paths; never run.
from sequence.exec import HostExec
from sequence.ops import FP
from autograd.ops_api import OpsSession
from autograd.program_runtime import train_program_control
from autograd.rankn import ShapeN
from core.cast_in import CastIn

@export
def compile_training(mut s: OpsSession[HostExec], pid: Int, srcs: List[CastIn], shapes: List[ShapeN],
    target: CastIn, loss: Int, delta: Float32, optimizer: Int, parameters: List[Int],
    epochs: Int, batch: Int, shuffle: Bool, seed: UInt64, curve: FP, condition: Int, otherwise: Int) raises -> Tuple[Int, Int]:
    return train_program_control(s, pid, srcs, shapes, target, loss, delta, optimizer, parameters,
                                epochs, batch, shuffle, seed, curve, condition, otherwise)
