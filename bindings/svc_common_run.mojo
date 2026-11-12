# SPDX-License-Identifier: Apache-2.0
# RUN part of bindings/svc_common.mojo: typed Mojo only (no Python imports).


def svc_round_f32(value: Float64) -> Float64:
    """`linear_model._round_f32`: one round-to-nearest-even to binary32."""
    return Float64(Float32(value))


def svc_isfinite(value: Float64) -> Bool:
    return value == value and value - value == 0.0
