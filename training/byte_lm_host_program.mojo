# SPDX-License-Identifier: Apache-2.0
"""The byte LM trainer's captured program on the CPU column (lane
x7b-step-ops-host-arm, 2026-11-09): `ByteLmProgram` (training/byte_lm_program.mojo)
on a host session, so the CPU column trains and evaluates with the GPU
columns' algorithm. The attention steps run the step ops' host arm
(autograd/block_step_host.mojo, through the host executor); the embedding,
GEMM, cross-entropy and OP_OPT AdamW are the executor's host bodies.

The session owner (training/byte_lm_host_session.mojo `ByteHostSession`) keeps
the committed flat lists (param, m, v, grad in `ByteConfig.offsets` order);
`byte_host_program_load` / `byte_host_program_store` copy them into and out
of the session, one block copy per tensor piece, no arithmetic. A host
module: no GPU binding imports it.
"""
from std.memory import unsafe_memcpy
from sequence.ops import FP
from sequence.exec import HostExec
from autograd.ops_api import OpsSession, ops_session, OPS_HOST
from training.byte_lm_config import ByteConfig
from training.checks.optimizer_contract import OptimizerConfig
from training.byte_lm_program import (
    ByteLmProgram, BYTE_PART_PARAM, BYTE_PART_M, BYTE_PART_V, BYTE_PART_GRAD, byte_program_open_on,
    byte_program_piece, byte_program_loaded,
)

comptime ByteHostProgram = ByteLmProgram[HostExec, OPS_HOST]


def byte_host_program_open(config: ByteConfig, cfg: OptimizerConfig) raises -> ByteHostProgram:
    """The trainer's program on a host session: the same capture as the
    GPU columns'."""
    return byte_program_open_on[HostExec, OPS_HOST](HostExec(), config, cfg)


def byte_host_program_workers(mut e: ByteHostProgram, workers: Int) raises:
    """Cap the host session's task count at `workers` (0 keeps one per
    physical core; 1 runs every launch on the calling thread). The count
    decides how many threads run a launch, never what a task computes
    (sequence/exec.mojo host_launch_tasks), so the bits are the same."""
    if workers < 0 or workers > 1024:
        raise Error("byte LM host: threads must be in [0, 1024] (0: one per physical core)")
    if workers > 0:
        var s = ops_session[HostExec, OPS_HOST](e.sid)
        s[].ex.workers = workers


def _move_host(mut s: OpsSession[HostExec], e: ByteHostProgram, flat: Int, part: Int, into_program: Bool) raises:
    """Copy `part` between the owner's flat list at host address `flat`
    and the session's tensors, one block copy per tensor piece."""
    var c = e.config.copy()
    var o = c.offsets()
    for j in range(c.n_tensors()):  # driver-loop: over model parameter tensors
        var n = o[j + 1] - o[j]
        var p = byte_program_piece(s, e, j, part)
        var f = FP(unsafe_from_address=flat) + o[j]
        if n > 0:
            if into_program:
                unsafe_memcpy(dest=p, src=f, count=n)
            else:
                unsafe_memcpy(dest=f, src=p, count=n)


def byte_host_program_load(mut e: ByteHostProgram, param: Int, m: Int, v: Int, completed: Int) raises:
    """The owner's committed parameters and AdamW moments (host addresses
    of n_total floats each) into the session, every trained tensor at step
    `completed`."""
    var s = ops_session[HostExec, OPS_HOST](e.sid)
    _move_host(s[], e, param, BYTE_PART_PARAM, True)
    _move_host(s[], e, m, BYTE_PART_M, True)
    _move_host(s[], e, v, BYTE_PART_V, True)
    byte_program_loaded(e, completed)


def byte_host_program_load_params(mut e: ByteHostProgram, param: Int) raises:
    """The owner's parameters alone into the session (a loss-only call)."""
    var s = ops_session[HostExec, OPS_HOST](e.sid)
    _move_host(s[], e, param, BYTE_PART_PARAM, True)
    for l in range(len(e.blocks)):  # driver-loop: over decoder layers
        s[].steps_weights_changed(Int(s[].buffer(e.blocks[l])))


def byte_host_program_store(mut e: ByteHostProgram, param: Int, m: Int, v: Int, grad: Int) raises:
    """Parameters, moments and the last step's pre-update gradient into the
    owner's flat lists (host addresses of n_total floats each)."""
    var s = ops_session[HostExec, OPS_HOST](e.sid)
    _move_host(s[], e, param, BYTE_PART_PARAM, False)
    _move_host(s[], e, m, BYTE_PART_M, False)
    _move_host(s[], e, v, BYTE_PART_V, False)
    _move_host(s[], e, grad, BYTE_PART_GRAD, False)
