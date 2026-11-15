# SPDX-License-Identifier: Apache-2.0
"""Apple FAST neural switches for the byte LM (lane afn-lm, 2026-10-03).

`BYTE_LM_FAST_APPLE` is true only on a FAST build (no
`-D MOJOLEARN_NUMERIC_IDENTICAL`) whose host is Apple silicon, so an
IDENTICAL build, and a FAST build on NVIDIA or AMD, compiles main's code
unchanged. docs/apple-fast/notes/neural-lm.md lists what each one changes.
"""
from std.sys.compile import is_defined
from core.accelerator_target import TARGET_HAS_APPLE_GPU
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from max.gpu.host import DeviceBuffer, DeviceContext
from core.step_phase import step_count_h2d

comptime BYTE_LM_FAST_APPLE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and TARGET_HAS_APPLE_GPU and not is_defined["MOJOLEARN_COLUMN_CPU"]()
)


# TOMBSTONE: MOJOLEARN_AFN_LM_ALL, MOJOLEARN_AFN_LM_NOSYNC, MOJOLEARN_AFN26_LM_NOSYNC, MOJOLEARN_AFN_LM_BWD_FUSE,
# MOJOLEARN_AFN26_LM_BWD_FUSE, MOJOLEARN_AFN_LM_PARAM_VIEWS, MOJOLEARN_AFN26_LM_PARAM_VIEWS, MOJOLEARN_AFN_LM_HEAD_FUSE,
# MOJOLEARN_AFN26_LM_HEAD_FUSE, MOJOLEARN_AFN26_LM_CE_BLOCK128, MOJOLEARN_AFN26_LM_CE_BLOCK512 (Apple FAST opt-in arms,
# no A/B recorded or never promoted) and MOJOLEARN_AFN_LM_BWD_NOSYNC_OFF (old path beside the promoted F08 default)
# deleted 2026-11-15 by lane/expdel2-31-neural; with them the NOSYNC status kernel, afn_step_finish, afn_reset and the
# HEAD_FUSE row kernel afn_ce_fused (nothing launched them any more); code recoverable at f6652fc1ac.
# Restore: git apply experiments/removed/MOJOLEARN_AFN_LM_ALL.patch (one patch per define, same diff)
#: -D MOJOLEARN_AFN_LM_BWD_NOSYNC: the block backward's RMSNorm stage drops
#: the three host waits the non-IDENTICAL tiers kept there ("old fences",
#: transformer_backward.mojo `_bwd_rms_norm_kernels`): 6 per layer.
# F08/default M3 Ultra 2026-10-06: promote this FAST Apple switch only.
# Whole train-checkpoint-refusal task [2,7,24,3,1,8,40,2,513], 12 steps:
# candidate/baseline trajectory 0.8766, first step 0.9357; heldout loss
# 6.231029987335205 equal, resume_error=0 on both, refusal/recovery captured.
# One excluded warmup and one scored trajectory; one shape, no universal
# hardware claim. Caller67d0efb29; reused artifact hashes/builds in
# ~/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F08/default.
# Identity/compile receipts reused, not rerun. Active frozen binaries unchanged.
# The _OFF rollback is deleted (expdel2-31-neural, 2026-11-15): the promoted arm is the only one.
comptime AFN_LM_BWD_NOSYNC = BYTE_LM_FAST_APPLE


def afn_upload_ids(ctx: DeviceContext, mut ids_dev: DeviceBuffer[DType.int32],
                   mut targets_dev: DeviceBuffer[DType.int32],
                   ids: List[Int32], batch: Int, length: Int) raises:
    """NOSYNC: inputs `ids[b, 0:L]` and targets `ids[b, 1:L+1]` copied from
    the caller's List straight into the device buffers, two copies per row,
    no staging buffer, no wait. The caller keeps `ids` alive past the step's
    one wait (the List is borrowed for the whole resident step, and every
    failure path waits inside `_byte_recover` before it returns)."""
    if len(ids) != batch * (length + 1):
        raise Error("byte LM: token count differs from row-major [batch,length+1]")
    var src = ids.unsafe_ptr()
    for b in range(batch):
        var vi = ids_dev.create_sub_buffer[DType.int32](b * length, length)
        var vt = targets_dev.create_sub_buffer[DType.int32](b * length, length)
        step_count_h2d()
        ctx.enqueue_copy(dst_buf=vi, src_ptr=src + b * (length + 1))
        step_count_h2d()
        ctx.enqueue_copy(dst_buf=vt, src_ptr=src + b * (length + 1) + 1)
