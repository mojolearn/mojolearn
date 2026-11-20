# SPDX-License-Identifier: Apache-2.0
"""RUN half of `providers/python/autograd_program_control_pyapi.mojo` (wave 8
run split): the bounded loop's carried-state view of its body (the iteration
register, the body's last input, lives on the device, so the caller passes one
input fewer). Typed Mojo only: no Python import."""
from autograd.program import OpsProgram


def control_state_view(body: OpsProgram) raises -> OpsProgram:
    """`body` without its last input (the device-resident iteration register)."""
    var state_view = body.copy()
    state_view.n_in -= 1
    _ = state_view.in_kinds.pop()
    _ = state_view.in_shapes.pop()
    return state_view^
