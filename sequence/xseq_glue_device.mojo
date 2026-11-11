# SPDX-License-Identifier: Apache-2.0
"""The device image's x_sequence glue entry and its publication (lane
calltable-xseq, 2026-11-11). `_mojolearn_x_sequence` publishes
`x_sequence_glue_device_v1` and its whole-model entry in the sequence table
(`core/xseq_entry_table.mojo`) at module init; the sequence owner calls it
with plain spans instead of the former Python entries.

The device choices are those the Python entries made: ETS(A, A|Ad, N) on the
team (`sequence/ets_team_impl.mojo`) when `ETS_TEAM or IDN_ETS_TEAM`, GARCH
and the Prophet fit on their one-block-per-series teams
(`sequence/fit_team_impl.mojo`); everything else `sequence/xseq_glue.mojo`.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.xseq_entry_table import (
    XseqEntryKey, xseq_table_publish, xseq_vendor_word, xseq_glue_check_spans, xseq_glue_ints, xseq_glue_floats,
    XS_ETS, XS_GARCH, XS_PROPHET_FIT,
)
from autograd.model_descriptor import MODEL_ABI_NAME
from sequence.neural_model_dispatch import MODEL_OPERATION_COUNT
from sequence.model_device_dispatch import NATIVE_MODEL_API
from sequence.exec_device import DeviceExec
from sequence.ets_team import ETS_TEAM, IDN_ETS_TEAM
from sequence.ets_team_impl import ets_team_impl
from sequence.fit_team_impl import garch_team_impl, prophet_fit_team_impl
from sequence.xseq_glue import xseq_glue_run


def xseq_ets_team_applies(ip: List[Int]) -> Bool:
    """Whether ets's integer parameters name the model the team fits:
    additive errors (ip[3] 0), an additive trend (ip[4] 1), no season
    (ip[7] 0)."""
    if len(ip) != 9 and len(ip) != 10:
        return False
    return ip[3] == 0 and ip[4] == 1 and ip[7] == 0


def xseq_glue_device_run(op: Int, addrs: List[Int], ip: List[Int], fp: List[Float64]) raises -> Int:
    var ex = DeviceExec()
    if op == XS_ETS:
        comptime if ETS_TEAM or IDN_ETS_TEAM:
            # Apple FAST default (off: -D MOJOLEARN_ETS_TEAM_OFF), and the
            # IDENTICAL default (off: -D MOJOLEARN_IDN_ETS_TEAM_OFF; the host
            # binding replays it in sequence/exec.mojo): ETS(A, A|Ad, N) one
            # series per block; every other model keeps ets_impl.
            if xseq_ets_team_applies(ip):
                return ets_team_impl(ex, addrs, ip, fp)
    if op == XS_GARCH:
        return garch_team_impl(ex, addrs, ip)
    if op == XS_PROPHET_FIT:
        return prophet_fit_team_impl(ex, addrs, ip, fp)
    return xseq_glue_run(ex, op, addrs, ip, fp)


def x_sequence_glue_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The device image's glue entry (`XSEQ_GLUE_ABI`). Writes the body's
    result to `result`; a refusal goes to `sink` in the body's own words."""
    try:
        xseq_glue_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var value = xseq_glue_device_run(operation, xseq_glue_ints(aa, na), xseq_glue_ints(pp, np),
                                         xseq_glue_floats(ff, nf))
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct XSequenceDeviceOwner(Defaultable, Movable):
    """The image-lifetime publisher word of the device x_sequence table."""

    var word: Int

    def __init__(out self):
        self.word = 1


comptime XSEQ_DEVICE_OWNER = SharedSlot[XSequenceDeviceOwner, "MojolearnXSequenceDeviceGlueOwnerV1"]


def publish_x_sequence_device() raises:
    """Publish this image's glue and whole-model entries at module init. A
    build with no GPU vendor publishes nothing; its callers refuse with the
    table's "provider is not published"."""
    var word = xseq_vendor_word(COMPILED_VENDOR)
    if word < 0:
        return
    var owner = XSEQ_DEVICE_OWNER.get_or_create_ptr()
    var api = NATIVE_MODEL_API.get_or_create_ptr()
    xseq_table_publish(XseqEntryKey(Int(GLOBAL_NUMERIC_MODE), word), Int(owner), x_sequence_glue_device_v1,
                       api[].entries[0], String(MODEL_ABI_NAME), MODEL_OPERATION_COUNT)
