# Focused compile-only inference replay and registry lifecycle. Never run.
from sequence.exec import HostExec
from sequence.ops import FP
from autograd.ops_api import OpsSession
from autograd.program import OpsProgram
from core.cast_in import CastIn

@export
def compile_inference(mut s: OpsSession[HostExec], pid: Int, srcs: List[CastIn], dsts: List[FP]) raises:
    s.run_program(pid, srcs, dsts)

@export
def compile_program_lifecycle(mut s: OpsSession[HostExec], var p: OpsProgram) raises:
    var pid = s.add_program(p^)
    s.program_free(pid)
    s.close()

@export
def compile_session() raises:
    var s = OpsSession(HostExec(), False)
    s.close()
