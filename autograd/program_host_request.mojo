# SPDX-License-Identifier: Apache-2.0
# Same-build native request for imported programs with host input/output buffers.
# The provider's image owns the registry and executor. Neither phase carries
# Python objects or calls a CPython computational entry point.
from sequence.exec_trait import Exec
from sequence.execution_profile import ExecutionProfile
from autograd.ops_api import ops_session_registry_token
from autograd.program import OpsProgram
from autograd.program_optimize import ProgramOptimization
from autograd.program_entries import ProgramExecPlan, ProgramExecResult, program_execute_prepare, program_execute_run
from autograd.program_io import ProgramInput


struct NativeProgramHostRun(Movable):
    var plan: ProgramExecPlan
    var inputs: List[ProgramInput]
    var result: ProgramExecResult
    var prepared: Bool
    var finished: Bool

    def __init__(out self):
        self.plan = ProgramExecPlan(OpsProgram(), ProgramOptimization(OpsProgram()), 0)
        self.inputs = List[ProgramInput]()
        self.result = ProgramExecResult(ExecutionProfile())
        self.prepared = False
        self.finished = False


def program_host_request[E: Exec & Movable & Deinitable, KIND: Int](
    addresses: List[Int], words: List[Int],
) raises -> Int:
    if len(addresses) != 1 or addresses[0] == 0 or len(words) != 5:
        raise Error("native program: invalid host request descriptor")
    if words[0] != ops_session_registry_token() or words[1] != KIND:
        raise Error("native program: Session belongs to another registry or executor")
    var request = MutPointer[NativeProgramHostRun, MutUntrackedOrigin](unsafe_from_address=addresses[0])
    var sid = words[2]
    if words[4] == 0 and not request[].prepared:
        request[].plan = program_execute_prepare[E, KIND](sid, words[3], sid, False, False)
        request[].prepared = True
    elif words[4] == 1 and request[].prepared and not request[].finished:
        var inputs = List[ProgramInput]()
        swap(inputs, request[].inputs)
        request[].result = program_execute_run[E, KIND](sid, request[].plan, inputs^, False)
        request[].finished = True
    else:
        raise Error("native program: invalid host request phase")
    return 0
