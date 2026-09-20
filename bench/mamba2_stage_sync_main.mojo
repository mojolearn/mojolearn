"""Long-sequence Mamba2 production-forward stage synchronization price."""

from std.memory import bitcast
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from mamba.checks.mamba2_fixture import (
    m2_case_weights,
    m2_case_x,
    m2_pos_inf,
)
from mamba.impl.modeling.modeling_mamba import mamba_download, mamba_upload
from mamba.impl.modules.mamba2 import (
    Mamba2DeviceStages,
    Mamba2DeviceState,
    Mamba2DeviceWeights,
    mamba2_block_forward,
)

comptime B = 2
comptime L = 770
comptime LAYERS = 12
comptime REPS = 7


def digest(xs: List[Float32]) -> UInt64:
    var h = UInt64(0xCBF29CE484222325)
    for x in xs:
        h = (h ^ UInt64(bitcast[DType.uint32](x))) * UInt64(0x100000001B3)
    return h


def run_layers(
    ctx: DeviceContext,
    mut stages: Mamba2DeviceStages,
    mut state: Mamba2DeviceState,
    mut weights: Mamba2DeviceWeights,
    mut x: DeviceBuffer[DType.float32],
    mut trace: IdentityTrace,
) raises:
    for _ in range(LAYERS):
        # The price card reuses one allocation. Resetting only the logical
        # open-chunk length keeps the public call shape stable; device state
        # remains carried and therefore live across the weighted calls.
        state.buf_len = 0
        mamba2_block_forward(
            ctx, stages, state, weights, x, B, L, 0.0, m2_pos_inf(), trace,
            String("price"),
        )


def main() raises:
    var ctx = DeviceContext()
    var weights_h = m2_case_weights(9)
    var dims = weights_h.dims.copy()
    var weights = Mamba2DeviceWeights(ctx, weights_h)
    var state = Mamba2DeviceState(ctx, B, dims)
    var stages = Mamba2DeviceStages(ctx, B, L, 0, dims)
    var x = mamba_upload(ctx, m2_case_x(9))
    var trace = IdentityTrace.disabled()
    ctx.synchronize()

    run_layers(ctx, stages, state, weights, x, trace)
    ctx.synchronize()
    for rep in range(REPS):
        var t0 = perf_counter_ns()
        run_layers(ctx, stages, state, weights, x, trace)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        print("MAMBA2_SYNC rep=", rep, " layers=", LAYERS, " ms=", Float64(t1 - t0) / 1e6)

    var out = mamba_download(ctx, stages.residual_out, B * L * dims.d_model)
    var hstate = mamba_download(
        ctx, state.h, B * dims.nheads * 64 * 128
    )
    print("MAMBA2_SYNC out_hash=", digest(out), " state_hash=", digest(hstate))
