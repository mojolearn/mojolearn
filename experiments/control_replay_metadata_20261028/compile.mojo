# Focused CPU compilation of ordinary and VJP control operations; never run.
from sequence.exec import HostExec
from autograd.ops_api import OPS_HOST
from autograd.program_io import ProgramInput
from autograd.program_control import ControlPlan, ControlResult, control_prepare, control_run
from autograd.program_control_grad import (
    ControlVjpPlan, ControlVjpState, ControlVjpResult, control_vjp_prepare,
    control_vjp_forward, control_vjp_finish, control_vjp_abort,
)

@export
def compile_prepare(sid: Int, body: Int, condition: Int, other: Int, maximum: Int, optimize: Bool) raises -> ControlPlan:
    return control_prepare[HostExec, OPS_HOST](sid, body, condition, other, sid, sid, maximum, optimize)

@export
def compile_run(sid: Int, plan: ControlPlan, var inputs: List[ProgramInput]) raises -> ControlResult:
    return control_run[HostExec, OPS_HOST](sid, plan, inputs^)

@export
def compile_vjp_prepare(sid: Int, body: Int, programs: List[Int], maximum: Int) raises -> ControlVjpPlan:
    return control_vjp_prepare[HostExec, OPS_HOST](sid, body, programs, maximum)

@export
def compile_vjp_forward(sid: Int, plan: ControlVjpPlan, var inputs: List[ProgramInput],
                        parameters: List[Int], explicit_parameters: Bool) raises -> ControlVjpState:
    return control_vjp_forward[HostExec, OPS_HOST](sid, plan, inputs^, parameters, explicit_parameters)

@export
def compile_vjp_finish(sid: Int, var state: ControlVjpState, var seeds: List[ProgramInput]) raises -> ControlVjpResult:
    return control_vjp_finish[HostExec, OPS_HOST](sid, state^, seeds^)

@export
def compile_vjp_abort(sid: Int, var state: ControlVjpState) raises:
    control_vjp_abort[HostExec, OPS_HOST](sid, state^)
