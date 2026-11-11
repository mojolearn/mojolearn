# SPDX-License-Identifier: Apache-2.0
"""The CPU provider retains its existing nonresident OneClassSVM route."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.ocsvm_resident_api import OcResidentKernels, OcResidentRequest


struct OcHostResident(OcResidentKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def flags() -> Int:
        return 0

    @staticmethod
    def run(r: OcResidentRequest) raises:
        raise Error("AttributeError: OneClassSVM resident solve is not exported by the host binding")
