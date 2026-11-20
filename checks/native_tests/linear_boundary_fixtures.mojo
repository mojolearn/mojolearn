# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of Mojolearn.
"""Compiled descriptor fixtures for native boundary and ownership contracts.

Every mock device handle owns a four-word native record: release count,
capacity, injected release failure, upload count. Records stay allocated
until the test inspects and frees them; this observes cleanup after errors
without touching freed storage. There is no shared mutable state or GPU.
"""
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.linear_fit_types import LF_FLAGS, LF_MODE, LF_HAS_OLS_RESIDENT, linear_fit_check_spans
from x_decomp.descriptor_schema import (
    XD_CAPABILITIES, XD_CAP_RESIDENT, XD_DEV_ALLOC, XD_DEV_FREE,
    XD_DEV_UPLOAD, XD_DEV_DOWNLOAD,
)


def plan_descriptor[RESIDENT: Bool](operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        linear_fit_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var value = 0
        if operation == LF_FLAGS:
            value = LF_HAS_OLS_RESIDENT if RESIDENT else 0
        elif operation == LF_MODE:
            comptime if RESIDENT:
                raise Error("resident plan must not query the unused TSQR mode")
            else:
                value = 17
        else:
            raise Error("planning must not execute a numerical operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


def storage_descriptor[RESIDENT: Bool, FAIL_UPLOAD: Bool = False](
    operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        var value = 0
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        if operation == XD_CAPABILITIES:
            value = XD_CAP_RESIDENT if RESIDENT else 0
        elif operation == XD_DEV_ALLOC:
            if not RESIDENT or np != 1 or p[0] < 1:
                raise Error("invalid native allocation request")
            var record = alloc[Int](4)
            record[0] = 0
            record[1] = p[0]
            record[2] = 0
            record[3] = 0
            value = Int(record)
        elif operation == XD_DEV_FREE:
            var record = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=p[0])
            record[0] += 1
            if record[0] != 1:
                raise Error("native handle released twice")
            if record[2]:
                raise Error("injected native release failure")
        elif operation == XD_DEV_UPLOAD or operation == XD_DEV_DOWNLOAD:
            var record = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=p[0])
            if na != 1 or np != 2 or p[1] > record[1] or record[0]:
                raise Error("invalid native transfer request")
            if operation == XD_DEV_UPLOAD:
                record[3] += 1
                comptime if FAIL_UPLOAD:
                    raise Error("injected native upload failure")
            else:
                var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=a[0])
                for i in range(p[1]):
                    dst[i] = Float32(7 + i)
        else:
            raise Error("ownership fixture received an unexpected numerical operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1
