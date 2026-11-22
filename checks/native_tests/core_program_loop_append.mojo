# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The bounded program loop over an append-only KV cache: the state a loop
leaves does not depend on its chunk (lane fx-kv-inplace, 2026-11-18).

A decode body holds one attention step op (autograd/block_step_ops.mojo
OPS_ATTN_STEP, full causal, b = 2, n_kv = 2, one token per row) whose cache
is declared append-only behind its position word
(`OpsSession.declare_append`, as transformer/causal_lm_program.mojo declares
every full-causal layer). The loop's gate restores the word after a body
enqueued past the stop and does not shadow the cache; the device's linear
cache is repacked at the new stride by every append, so before this lane a
stop inside a chunk left the committed slots at a stride the word did not
describe (`CausalLM.stream(..., eos_token_id=E).read(16)` against `read(1)`
repeated: same ids, different exported K/V bytes).

The loop carries x (2, 8) and a counter (1, 1); the condition is
counter < 2, so it stops after two committed iterations, the third position
of the cache (the capture run appended the first). Each fixture runs the same
model on three independent (weights, cache, word) triples with check_every 1
(no body runs past the stop: the reference, `read(1)` repeated), 8 (six
bodies run past the stop) and 0 (the loop's chunk rule, what the decode
program passes) and requires the same words in the whole cache tensor, the
position word and the carried x.

  0  s_max 16: every body past the stop appends.
  1  s_max 6: the cache fills while bodies run past the stop, so one of them
     is refused (the loop's own error path for a body past the stop).
No Python participates.
"""
from std.memory import bitcast
from autograd.ops_api import OpsSession, ops_session, ops_session_open, ops_session_close, OPS_HOST, OPS_DEVICE
from autograd.block_step_ops import (
    OPS_ATTN_STEP, BLOCK_PART_WEIGHTS, BLOCK_PART_STATE, BLOCK_PART_WORDS, block_weight_sizes, block_state_sizes,
    block_part_floats,
)
from autograd.program_control import control_prepare, control_run, ControlResult
from autograd.program_io import ProgramInput
from autograd.rankn import ShapeN
from core.cast_in import CastIn
from sequence.exec_trait import Exec
from checks.native_tests.ops_higher_common import tensor, op2, values, same_words
from checks.native_tests.core_ops_common import dims2, opi
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD, check_cases_serial, require


comptime CORE_PROGRAM_LOOP_APPEND_FIXTURES = 2

comptime _ADD = 2
comptime _FULL = 30
comptime _COMPARE = 69
comptime _LT = 2

comptime _B = 2
comptime _DM = 8
comptime _NH = 2
comptime _NKV = 2
comptime _HD = 4
comptime _IT = 16
comptime _STOP = 2
comptime _LIMIT = 16


def _full[E: Exec & Movable & Deinitable](mut s: OpsSession[E], value: Float32) raises -> Int:
    var fs: List[Float32] = [value]
    return opi(s, _FULL, List[Int](), dims2(1, 1), fs)


def _less[E: Exec & Movable & Deinitable](mut s: OpsSession[E], a: Int, b: Int) raises -> Int:
    var hs: List[Int] = [a, b]
    var kind: List[Int] = [_LT]
    return opi(s, _COMPARE, hs, kind, List[Float32]())


def _attn_ints(smax: Int) -> List[Int]:
    """OPS_ATTN_STEP's eight integers (b, d_model, n_heads, n_kv, head_dim,
    intermediate, s_max, window 0 = full causal), the default block options."""
    var ints: List[Int] = [_B, _DM, _NH, _NKV, _HD, _IT, smax, 0]
    return ints^


def _decode_once[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, smax: Int, every: Int,
) raises -> List[List[Float32]]:
    """One model on its own (weights, cache, word) triple: the capture run
    (one token per row), then the bounded loop with `every`. Returns the
    whole cache tensor, the position word (as its float32 bit pattern) and
    the carried x, after checking the loop's count and stop flag."""
    var s = ops_session[E, KIND](sid)
    var ints = _attn_ints(smax)
    # The weights: both norm weights 1, every projection a small fixed
    # pattern (finite, not symmetric across rows or heads).
    var sizes = block_weight_sizes(OPS_ATTN_STEP, ints)
    var at = List[Int](capacity=len(sizes))
    var total = 0
    for k in range(len(sizes)):
        at.append(total)
        total += sizes[k]
    var flat = List[Float32](length=total + 1, fill=Float32(0))
    for k in range(len(sizes)):
        for i in range(sizes[k]):
            if k < 2:
                flat[at[k] + i] = Float32(1.0)
            else:
                flat[at[k] + i] = Float32((i * 7 + k * 5 + 3) % 11 - 5) * Float32(0.03)
    var wsrc = List[CastIn](capacity=len(sizes))
    for k in range(len(sizes)):
        if sizes[k] > 0:
            wsrc.append(CastIn.f32_c(Int(flat.unsafe_ptr()) + 4 * at[k], sizes[k], 1))
        else:
            wsrc.append(CastIn.f32_c(0, 0, 1))
    var wh = s[].upload_block(OPS_ATTN_STEP, ints, BLOCK_PART_WEIGHTS, wsrc, List[Int]())
    _ = len(flat)  # keep-alive past the staging copies
    # The caches: zeros (a fresh state), the word 0.
    var ssizes = block_state_sizes(OPS_ATTN_STEP, ints)
    var zero = List[Float32](length=ssizes[0] + 1, fill=Float32(0))
    var zsrc = List[CastIn](capacity=len(ssizes))
    for k in range(len(ssizes)):
        zsrc.append(CastIn.f32_c(Int(zero.unsafe_ptr()), ssizes[k], 1))
    var kh = s[].upload_block(OPS_ATTN_STEP, ints, BLOCK_PART_STATE, zsrc, List[Int]())
    _ = len(zero)  # keep-alive past the staging copies
    var word0: List[Int] = [0]
    var qh = s[].upload_block(OPS_ATTN_STEP, ints, BLOCK_PART_WORDS, List[CastIn](), word0)
    s[].declare_append(kh, qh)
    # The body and the condition, captured from one eager step (it appends
    # position 0, as the decode program's capture step does).
    var x0 = List[Float32](length=_B * _DM, fill=Float32(0))
    for i in range(_B * _DM):
        x0[i] = Float32((i * 5 + 2) % 13 - 6) * Float32(0.11)
    var c0: List[Float32] = [0.0]
    var x = tensor(s[], x0, dims2(_B, _DM))
    var c = tensor(s[], c0, dims2(1, 1))
    var hs: List[Int] = [x, wh, kh, qh]
    var y = opi(s[], OPS_ATTN_STEP, hs, ints.copy(), List[Float32]())
    var ins: List[Int] = [x, c]
    var body_out: List[Int] = [y, op2(s[], _ADD, c, _full(s[], 1.0))]
    var body = s[].capture(ins, body_out)
    var cond_out: List[Int] = [_less(s[], c, _full(s[], Float32(_STOP)))]
    var condition = s[].capture(ins, cond_out)
    var plan = control_prepare[E, KIND](sid, body, condition, 0, sid, sid, _LIMIT, False, every, False)
    var inputs = List[ProgramInput]()
    inputs.append(ProgramInput(CastIn.f32_c(Int(x0.unsafe_ptr()), _B, _DM), ShapeN.of(dims2(_B, _DM)),
                               List[Float64](), List[Int64]()))
    inputs.append(ProgramInput(CastIn.f32_c(Int(c0.unsafe_ptr()), 1, 1), ShapeN.of(dims2(1, 1)),
                               List[Float64](), List[Int64]()))
    var r = control_run[E, KIND](sid, plan, inputs^)
    _ = len(x0)
    _ = len(c0)
    var tag = "check_every " + String(every) + " s_max " + String(smax)
    require(r.iterations == _STOP and r.stopped, tag + ": the loop's count or stop flag")
    var out = List[List[Float32]]()
    out.append(values(s[], kh, dims2(1, block_part_floats(OPS_ATTN_STEP, ints, BLOCK_PART_STATE))))
    out.append(values(s[], qh, dims2(1, 1)))
    out.append(r.outputs[0].data.copy())
    return out^


def run_core_program_loop_append_fixture[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, fixture_id: Int,
) raises:
    require(fixture_id < CORE_PROGRAM_LOOP_APPEND_FIXTURES, "unknown append-only loop fixture")
    var smax = 16 if fixture_id == 0 else 6
    var want = _decode_once[E, KIND](sid, smax, 1)
    var word: List[Float32] = [bitcast[DType.float32](Int32(1 + _STOP))]
    same_words(want[1], word, "check_every 1: the position word after the capture step and two iterations")
    var everies: List[Int] = [8, 0]
    for k in range(len(everies)):
        var got = _decode_once[E, KIND](sid, smax, everies[k])
        var tag = "check_every " + String(everies[k]) + " s_max " + String(smax)
        same_words(got[1], want[1], tag + ": the position word differs from check_every 1's")
        same_words(got[2], want[2], tag + ": the carried x differs from check_every 1's")
        same_words(got[0], want[0], tag + ": the exported key/value cache differs from check_every 1's")


# ---- The host and device columns ----
# Serial by nature: control_prepare/control_run reach a session through the
# process-wide session table (autograd/ops_api.mojo _OPS_SESSIONS), whose
# open/close append to and pop from unlocked lists, so concurrent sessions
# from host tasks would race on that table (core_program_loop says the same).
# Two fixtures of nine block steps each over a (2, 8) input, one after
# another, each on its own session; on the device column they share the device context.
def test_program_loop_append() raises:
    comptime if CPU_COLUMN_BUILD:
        def host_fixture(fixture_id: Int) raises:
            var sid = ops_session_open[HostExec, OPS_HOST](HostExec(), False)
            try:
                run_core_program_loop_append_fixture[HostExec, OPS_HOST](sid, fixture_id)
            except error:
                ops_session_close[HostExec, OPS_HOST](sid)
                raise error
            ops_session_close[HostExec, OPS_HOST](sid)
        check_cases_serial("program_loop_append", CORE_PROGRAM_LOOP_APPEND_FIXTURES, host_fixture)
    else:
        def device_fixture(fixture_id: Int) raises:
            var sid = ops_session_open[DeviceExec, OPS_DEVICE](DeviceExec(), False)
            try:
                run_core_program_loop_append_fixture[DeviceExec, OPS_DEVICE](sid, fixture_id)
            except error:
                ops_session_close[DeviceExec, OPS_DEVICE](sid)
                raise error
            ops_session_close[DeviceExec, OPS_DEVICE](sid)
        check_cases_serial("program_loop_append", CORE_PROGRAM_LOOP_APPEND_FIXTURES, device_fixture)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
