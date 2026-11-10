# SPDX-License-Identifier: Apache-2.0
"""Compiled linear fit primitives; the selected solver kernels are unchanged."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.gram_multi_gpu import gram_plan_current
from core.hostptr import f32_ptr, f64_ptr
from core.linear_fit_types import (
    LINEAR_FIT_ABI, LF_FLAGS, LF_GRAM, LF_RIDGE_RESIDENT, LF_PREP,
    LF_RIDGE, LF_OLS, LF_OLS_RESIDENT, LF_HAS_GRAM, LF_HAS_RIDGE_RESIDENT,
    LF_PAIRED_MEANS, LF_HAS_OLS_RESIDENT, linear_fit_check_spans,
)
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.neural_context import process_ctx
from core.shared_slot import SharedSlot
from experiments.classical_identical_ideas.linear_controls import LINEAR_GRAM_SOLVE
from experiments.classical_identical_ideas.shared_controls import C02_LINEAR_PAIR
from glm.estimator import ridge_fit_host, ridge_fit_resident_host, ols_fit_host, ols_fit_resident_host
from glm.impl.gram_solve import linear_gram_fit_host
from glm.impl.ols_prep_device import ols_prep_device
from glm.linear_route_controls import RIDGE_FAST_RESIDENT, OLS_FAST_NORMAL_EQ

comptime _CONTEXT = "MojolearnEstimatorsContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnEstimatorsContextFastV3"


def linear_fit_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        linear_fit_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var value = 0
        if operation == LF_FLAGS:
            value = Int(LINEAR_GRAM_SOLVE) * LF_HAS_GRAM + Int(RIDGE_FAST_RESIDENT) * LF_HAS_RIDGE_RESIDENT + Int(C02_LINEAR_PAIR) * LF_PAIRED_MEANS + Int(OLS_FAST_NORMAL_EQ) * LF_HAS_OLS_RESIDENT
        else:
            var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
            var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
            var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
            var ctx = process_ctx[_CONTEXT]()
            if operation == LF_PREP:
                ols_prep_device(ctx, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                    p[0], p[1], p[2] != 0, p[3])
            elif operation == LF_RIDGE:
                ridge_fit_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]), p[0], p[1], Float32(f[0]))
                ctx.synchronize()
            elif operation == LF_OLS:
                ols_fit_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]), p[0], p[1])
                ctx.synchronize()
            elif operation == LF_GRAM:
                comptime if LINEAR_GRAM_SOLVE:
                    if p[0] <= 0 or p[1] <= 0:
                        raise Error("linear_gram_fit: n_rows and n_features must be positive")
                    if Float32(f[0]) > Float32(0.0) and p[1] == 1:
                        value = 1
                    else:
                        var plan = gram_plan_current()
                        value = linear_gram_fit_host(plan, ctx, f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]),
                            f32_ptr(a[3]), f64_ptr(a[4]), p[0], p[1], Float32(f[0]), p[2] != 0)
                        ctx.synchronize()
                else:
                    raise Error("linear_gram_fit: this build lacks the route (FAST, or -D MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE_OFF)")
            elif operation == LF_RIDGE_RESIDENT:
                comptime if RIDGE_FAST_RESIDENT:
                    if p[0] <= 0 or p[1] <= 0:
                        raise Error("ridge_fit_resident: n_rows and n_features must be positive")
                    ridge_fit_resident_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]),
                        f32_ptr(a[3]), f64_ptr(a[4]), p[0], p[1], Float32(f[0]), p[2] != 0)
                    ctx.synchronize()
                else:
                    raise Error("ridge_fit_resident: only the FAST Apple build without -D MOJOLEARN_RIDGE_RESIDENT_OFF has this route")
            elif operation == LF_OLS_RESIDENT:
                comptime if OLS_FAST_NORMAL_EQ:
                    if p[0] <= 0 or p[1] <= 0:
                        raise Error("ols_fit_resident: n_rows and n_features must be positive")
                    ols_fit_resident_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]),
                        f32_ptr(a[3]), f64_ptr(a[4]), p[0], p[1], p[2] != 0)
                    ctx.synchronize()
                else:
                    raise Error("ols_fit_resident: only the FAST Apple build (OLS_FAST_NORMAL_EQ) has this route")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct LinearFitDeviceAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(LINEAR_FIT_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(linear_fit_device_v1)


comptime _API_SLOT = "MojolearnLinearFitDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnLinearFitDeviceFastV1"
comptime LINEAR_FIT_DEVICE_API = SharedSlot[LinearFitDeviceAPI, _API_SLOT]
