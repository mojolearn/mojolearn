# SPDX-License-Identifier: Apache-2.0
"""The host image's x_sequence glue entry and its publication (lane
calltable-xseq, 2026-11-11). `_mojolearn_x_sequence_host` publishes
`x_sequence_glue_host_v1` and its whole-model entry in the sequence table
(`core/xseq_entry_table.mojo`, vendor word 0) at module init; every
operation runs `sequence/xseq_glue.mojo` on `HostExec`, as the host image's
Python entries did.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.xseq_entry_table import (
    XseqEntryKey, xseq_table_publish, xseq_glue_check_spans, xseq_glue_ints, xseq_glue_floats,
)
from autograd.model_descriptor import MODEL_ABI_NAME
from sequence.neural_model_dispatch import MODEL_OPERATION_COUNT
from sequence.model_host_dispatch import NATIVE_MODEL_API
from sequence.exec import HostExec
from sequence.xseq_glue import xseq_glue_run


def x_sequence_glue_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The host image's glue entry (`XSEQ_GLUE_ABI`)."""
    try:
        xseq_glue_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var ex = HostExec()
        var value = xseq_glue_run(ex, operation, xseq_glue_ints(aa, na), xseq_glue_ints(pp, np),
                                  xseq_glue_floats(ff, nf))
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct XSequenceHostOwner(Defaultable, Movable):
    """The image-lifetime publisher word of the host x_sequence table."""

    var word: Int

    def __init__(out self):
        self.word = 1


comptime XSEQ_HOST_OWNER = SharedSlot[XSequenceHostOwner, "MojolearnXSequenceHostGlueOwnerV1"]


def publish_x_sequence_host() raises:
    """Publish this image's glue and whole-model entries at module init
    (the host image is CPU, vendor word 0)."""
    var owner = XSEQ_HOST_OWNER.get_or_create_ptr()
    var api = NATIVE_MODEL_API.get_or_create_ptr()
    xseq_table_publish(XseqEntryKey(Int(GLOBAL_NUMERIC_MODE), 0), Int(owner), x_sequence_glue_host_v1,
                       api[].entries[0], String(MODEL_ABI_NAME), MODEL_OPERATION_COUNT)
