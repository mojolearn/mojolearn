# SPDX-License-Identifier: Apache-2.0
# Private Program.execute result publication. Destinations are freshly allocated
# HostArray storage, pinned by the caller until this synchronous copy returns.
from sequence.exec_trait import Exec
from sequence.exec_device import DeviceExec, SEQ_PIPE_DOWN, SEQ_PIPE_CH
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from sequence.ops import FP


def program_output_copy[E: Exec, DEVICE: Bool](mut ex: E, dst: FP, src: FP, n: Int) raises:
    # KIND is already fixed by the owning registry. rebind is a supported
    # compile-time exact-type assertion, not an unchecked layout conversion.
    # A generic overload otherwise resolves to Exec.download before E becomes
    # concrete, silently retaining the bounce copy (retained compile evidence).
    comptime if DEVICE:
        _program_device_output_policy(rebind[DeviceExec](ex), dst, src, n)
    else:
        ex.download(dst, src, n)


def _program_device_output_policy(mut ex: DeviceExec, dst: FP, src: FP, n: Int) raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        ex.download(dst, src, n)
        return
    comptime if SEQ_PIPE_DOWN:
        # Preserve the executor's already-selected large pipelined transfer
        # policy and its chunk completion/error order. This is its existing
        # byte-capacity rule, not a new model/benchmark-size route.
        if ex.pooled and n >= 2 * SEQ_PIPE_CH:
            ex.download(dst, src, n)
            return
    _program_device_output_copy(ex, dst, src, n)


def _program_device_output_copy(mut ex: DeviceExec, dst: FP, src: FP, n: Int) raises:
    # This boundary's destination is private and cannot be observed until the
    # operation returns. Copy directly into it, removing the temporary HostBuffer
    # and its CPU memcpy. Keep each output's original completion boundary, so a
    # device failure precedes validation/allocation of the following output.
    var started = ex.execution_profile.clock()
    if n > 0:
        var found = ex._find(src, n)
        var view = ex._sub(found[0], found[1], n)
        try:
            ex.ctx.enqueue_copy(dst_ptr=dst, src_buf=view)
            ex.execution_profile.copied(1, n * 4)
            ex.sync()
        except problem:
            # The destination owner and source view stay pinned through a
            # partial enqueue/drain failure; preserve the original copy error.
            try:
                ex.sync()
            except:
                pass
            raise problem^
        finally:
            _ = view
    ex.execution_profile.copy_timed(1, started)
