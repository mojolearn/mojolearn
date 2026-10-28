# Compile only. No runtime, identity or performance execution.
from autograd.ops_api import OpsSession
from sequence.exec import HostExec

@export
def compile_state_index(mut session: OpsSession[HostExec], optimizer: Int, handle: Int, count: Int, handles: List[Int]) raises:
    _ = session._state_of(optimizer, handle, count)
    session.step(optimizer, handles)
    session.free(handle)
    session.reset()
    session.close()

@export
def compile_session_init() raises:
    var session = OpsSession(HostExec(), True)
    session.close()

@export
def compile_optimizer_create(mut session: OpsSession[HostExec], ints: List[Int], floats: List[Float32]) raises -> Int:
    return session.optimizer(ints, floats)
