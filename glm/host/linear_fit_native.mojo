# SPDX-License-Identifier: Apache-2.0
"""The existing host linear fit kernels behind native primitive requests."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.estimators_host_impl import _positive, ridge_fit_host_run, ols_fit_host_run
from core.hostptr import f32_ptr, f64_ptr
from core.linear_fit_types import (
    LINEAR_FIT_ABI, LF_FLAGS, LF_GRAM, LF_RIDGE_RESIDENT, LF_PREP,
    LF_RIDGE, LF_OLS, LF_OLS_RESIDENT, LF_HAS_GRAM, LF_PAIRED_MEANS, linear_fit_check_spans,
)
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from experiments.classical_identical_ideas.linear_controls import LINEAR_GRAM_SOLVE
from experiments.classical_identical_ideas.shared_controls import C02_LINEAR_PAIR
from glm.host.gram_solve_host import host_linear_gram_fit
from glm.host.ols_prep_host import ols_prep_host


def linear_fit_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        linear_fit_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var value = 0
        if operation == LF_FLAGS:
            value = Int(LINEAR_GRAM_SOLVE) * LF_HAS_GRAM + Int(C02_LINEAR_PAIR) * LF_PAIRED_MEANS
        else:
            var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
            var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
            var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
            if operation == LF_PREP:
                ols_prep_host(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                    p[0], p[1], p[2] != 0, p[3])
            elif operation == LF_RIDGE:
                ridge_fit_host_run(a[0], a[1], f32_ptr(a[2]), p[0], p[1], Float32(f[0]))
            elif operation == LF_OLS:
                ols_fit_host_run(a[0], a[1], f32_ptr(a[2]), p[0], p[1])
            elif operation == LF_GRAM:
                comptime if LINEAR_GRAM_SOLVE:
                    _positive(p[0], "n_rows")
                    _positive(p[1], "n_features")
                    if Float32(f[0]) > Float32(0.0) and p[1] == 1:
                        value = 1
                    else:
                        value = host_linear_gram_fit(a[0], a[1], p[0], p[1], Float32(f[0]), p[2] != 0,
                            f32_ptr(a[2]), f32_ptr(a[3]), f64_ptr(a[4]))
                else:
                    raise Error("linear_gram_fit: this build lacks the route (FAST, or -D MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE_OFF)")
            elif operation == LF_RIDGE_RESIDENT or operation == LF_OLS_RESIDENT:
                raise Error("linear fit: the host column has no FAST resident route")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct LinearFitHostAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(LINEAR_FIT_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(linear_fit_host_v1)


comptime _API_SLOT = "MojolearnLinearFitHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnLinearFitHostFastV1"
comptime LINEAR_FIT_HOST_API = SharedSlot[LinearFitHostAPI, _API_SLOT]
