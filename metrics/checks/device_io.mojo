# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host <-> device copies for the metric checks and drivers. Plumbing, not
an implementation: RAFT receives device pointers and never uploads anything."""

from max.gpu.host import DeviceBuffer, DeviceContext
from core.transfer import (
    TRANSFER_TIER_IDENTICAL,
    transfer_download_f32_list,
    transfer_download_list,
    transfer_upload_list,
)

#: The pinned download stages of this module and tier (core/transfer.mojo).
comptime _XFER_POOL = "MojolearnMetricsIoTransferStagesIdenticalV1" if TRANSFER_TIER_IDENTICAL else "MojolearnMetricsIoTransferStagesFastV1"


def upload_f32(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    # core/transfer.mojo (lane fx-transfer-helpers, 2026-11-18): read in place
    # from the caller's list; the private copy of the list per call is gone.
    var n = len(values)
    return transfer_upload_list[DType.float32](ctx, values, n, n if n > 0 else 1)


def upload_i32(
    ctx: DeviceContext, values: List[Int32]
) raises -> DeviceBuffer[DType.int32]:
    # core/transfer.mojo (lane fx-transfer-helpers, 2026-11-18): read in place
    # from the caller's list; the private copy of the list per call is gone.
    var n = len(values)
    return transfer_upload_list[DType.int32](ctx, values, n, n if n > 0 else 1)


def download_f32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    # core/transfer.mojo (lane fx-transfer-helpers, 2026-11-18): the copy
    # engine writes the list's own words. Gone: a pinned buffer per call and
    # one append per word.
    return transfer_download_f32_list[_XFER_POOL](ctx, buf, n)


def download_i32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.int32], n: Int
) raises -> List[Int32]:
    # core/transfer.mojo: one copy into the list's own words (no pinned
    # buffer per call, no append per word).
    return transfer_download_list[DType.int32](ctx, buf, n)


def download_u32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.uint32], n: Int
) raises -> List[UInt32]:
    # core/transfer.mojo: one copy into the list's own words (no pinned
    # buffer per call, no append per word).
    return transfer_download_list[DType.uint32](ctx, buf, n)
