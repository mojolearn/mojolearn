# SPDX-License-Identifier: Apache-2.0
"""Existing resident OneClassSVM solver, with unchanged compiled route flags."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.ocsvm_resident_api import OcResidentKernels, OcResidentRequest
from x_neighbors.ocsvm_dev import op_ocsvm_x, OCSVM_CLS2_RES, OCSVM_IDN_RES


struct OcDeviceResident(OcResidentKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def flags() -> Int:
        return Int(OCSVM_CLS2_RES) | (Int(OCSVM_IDN_RES) << 1)

    @staticmethod
    def run(r: OcResidentRequest) raises:
        if r.fast:
            comptime if not OCSVM_CLS2_RES:
                raise Error("AttributeError: x_neighbors_ocsvm_resident is not exported by this binding")
        else:
            comptime if not OCSVM_IDN_RES:
                raise Error("AttributeError: x_neighbors_ocsvm_resident_idn is not exported by this binding")
        op_ocsvm_x(r.x, r.bounds, r.alpha, r.info, r.iterations,
            r.rows, r.features, r.kernel, Float32(r.gamma), Float32(r.coef0),
            r.degree, Float32(r.tolerance), r.max_iter)
