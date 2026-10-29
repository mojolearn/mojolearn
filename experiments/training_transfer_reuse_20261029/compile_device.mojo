# Compile-only changed DeviceExec publication through the same generic call
# shape as program_execute_run. This does not compile a full GPU graph replay.
from sequence.exec_trait import Exec
from sequence.exec_device import DeviceExec
from sequence.ops import FP
from autograd.program_output_copy import program_output_copy


def _program_equivalent_copy[E: Exec, DEVICE: Bool](mut ex: E, dst: FP, src: FP, n: Int) raises:
    program_output_copy[E, DEVICE](ex, dst, src, n)


@export
def compile_program_device_publication(mut ex: DeviceExec, dst: FP, src: FP, n: Int) raises:
    _program_equivalent_copy[DeviceExec, True](ex, dst, src, n)
