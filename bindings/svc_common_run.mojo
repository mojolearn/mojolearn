# SPDX-License-Identifier: Apache-2.0
# RUN part of bindings/svc_common.mojo: typed Mojo only (no Python imports).
from core.host_finite_scan import host_all_finite_f32


def svc_round_f32(value: Float64) -> Float64:
    """`linear_model._round_f32`: one round-to-nearest-even to binary32."""
    return Float64(Float32(value))


def svc_isfinite(value: Float64) -> Bool:
    return value == value and value - value == 0.0


def svc_r2_check(y: Int, n: Int, pred: Int, m: Int) raises:
    """RUN of `svc_r2_sums`' input checks (lane entangled-programs): both
    float32 vectors finite, then equal lengths, the same refusals in the same
    order. The door keeps both buffers alive."""
    if not host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=y), n):
        raise Error("mojolearn score: y_true contains NaN or infinity")
    if not host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=pred), m):
        raise Error("mojolearn score: y_pred contains NaN or infinity")
    if n != m:
        raise Error("mojolearn score: y_true and y_pred have different shapes (" + String(n) + ", 1) and ("
                    + String(m) + ", 1)")
