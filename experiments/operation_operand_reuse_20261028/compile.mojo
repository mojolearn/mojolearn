# Compile-only eager native operation and recording/inference session lifecycle.
from sequence.exec import HostExec
from autograd.ops_api import OpsSession

@export
def compile_operation(mut s: OpsSession[HostExec], code: Int, hs: List[Int], ints: List[Int], fs: List[Float32]) raises -> Int:
    return s.op(code, hs, ints, fs)

@export
def compile_lifecycle(record: Bool) raises:
    var s = OpsSession(HostExec(), record)
    s.close()
