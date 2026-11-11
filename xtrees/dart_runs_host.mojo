# SPDX-License-Identifier: Apache-2.0
"""The host x_trees image's DART entry of the trees DART table (lane calltable-gbdt-forest-dart,
2026-11-11): one `abi("C")` dispatch over the operations of `xtrees/dart_abi.mojo`,
running the bodies of `xtrees/dart_host.mojo` that the `x_trees_dart_*` Python
entries of `bindings/_mojolearn_x_trees_host.mojo` ran. Published at that binding's module
init under its tier/vendor key. Plumbing only: the bodies, their kernels and
their fold orders are unchanged. Publication is a small metadata operation
with no independent numerical work to split.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.trees_entry_table import TreesEntryKey
from core.trees_entry_table_dart import trees_dart_table_publish
from xtrees.glue_abi import XtCall
from xtrees.dart_abi import (
    XD_DART_ABI, XD_OP_COUNT, XD_CAPS, XD_OPEN, XD_STEP, XD_ADD, XD_CLOSE, XD_PREDICT, XD_FOREST_PREPARE,
    XD_FOREST_PREDICT, XD_FOREST_RELEASE, XD_CAP_ROUND, XD_CAP_RESIDENT, xd_need, xd_addr, xd_int, xd_float,
    xd_words, xd_f64,
)
from xtrees.dart_host import DART_HOST, dart_open, dart_step, dart_add, dart_close, dart_predict


def _dart_host_dispatch(op: Int, c: XtCall) raises -> Int:
    if op == XD_CAPS:
        var caps = 0
        comptime if DART_HOST:
            caps |= XD_CAP_ROUND
        return caps
    if op == XD_PREDICT:
        xd_need(c, 9, 4, "x_trees_dart_predict")
        var nt = xd_int(c, 3)
        var k = xd_int(c, 2)
        dart_predict(xd_addr(c, 0), xd_words(xd_addr(c, 2), nt, "x_trees_dart_predict"),
                     xd_words(xd_addr(c, 3), nt, "x_trees_dart_predict"),
                     xd_words(xd_addr(c, 4), nt, "x_trees_dart_predict"),
                     xd_words(xd_addr(c, 5), nt, "x_trees_dart_predict"),
                     xd_words(xd_addr(c, 6), nt, "x_trees_dart_predict"),
                     xd_f64(xd_addr(c, 7), nt, "x_trees_dart_predict"),
                     xd_f64(xd_addr(c, 8), k, "x_trees_dart_predict"), xd_addr(c, 1), xd_int(c, 0), xd_int(c, 1), k)
        return xd_int(c, 0)
    if op == XD_FOREST_PREPARE or op == XD_FOREST_PREDICT or op == XD_FOREST_RELEASE:
        # The host column keeps no resident forest (XD_CAPS bit 1 is 0);
        # its callers predict with XD_PREDICT.
        raise Error("x_trees dart: the host binding keeps no resident forest")
    if op == XD_OPEN or op == XD_STEP or op == XD_ADD or op == XD_CLOSE:
        comptime if DART_HOST:
            if op == XD_OPEN:
                xd_need(c, 3, 6, "x_trees_dart_open")
                return dart_open(xd_addr(c, 0), xd_addr(c, 1), xd_addr(c, 2), xd_int(c, 0), xd_int(c, 1), xd_int(c, 2),
                                 xd_int(c, 3), xd_int(c, 4), xd_int(c, 5))
            if op == XD_STEP:
                xd_need(c, 4, 5, "x_trees_dart_step")
                var targets = List[Int]()
                for i in range(4, c.n_addrs):  # driver-loop: one output address per class (model structure)
                    targets.append(xd_addr(c, i))
                dart_step(xd_int(c, 0), xd_addr(c, 0), xd_addr(c, 1), xd_addr(c, 2), xd_addr(c, 3), targets,
                          xd_int(c, 1), xd_int(c, 2), xd_int(c, 3), xd_int(c, 4))
                return xd_int(c, 1)
            if op == XD_ADD:
                xd_need(c, 5, 11, "x_trees_dart_add")
                dart_add(xd_int(c, 0), xd_addr(c, 0), xd_addr(c, 1), xd_addr(c, 2), xd_addr(c, 3), xd_int(c, 1),
                         xd_int(c, 2), xd_int(c, 3), xd_int(c, 4), xd_float(c, 5), xd_float(c, 6), xd_float(c, 7),
                         xd_float(c, 8), xd_float(c, 9), xd_addr(c, 4), xd_int(c, 10))
                return xd_int(c, 4)
            xd_need(c, 1, 1, "x_trees_dart_close")
            dart_close(xd_int(c, 0), xd_addr(c, 0))
            return 0
        raise Error("NotImplementedError: mojolearn DART: this binary has no x_trees_dart_* round")
    raise Error("x_trees dart: unknown operation " + String(op))


def x_trees_dart_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The host x_trees image's DART entry (`XD_DART_ABI`): addresses aa[0:na], the
    parameter views pp[0:np] (int) and ff[0:nf] (float, nf == np). Writes
    the operation's count to result; a refusal goes to sink."""
    try:
        if operation < 0 or operation >= XD_OP_COUNT:
            raise Error("x_trees dart: unknown operation " + String(operation))
        if na < 0 or np < 0 or np != nf or not result:
            raise Error("x_trees dart: invalid request spans")
        if (na > 0 and not aa) or (np > 0 and (not pp or not ff)):
            raise Error("x_trees dart: null request span")
        var value = _dart_host_dispatch(operation, XtCall(aa, na, pp, ff, np, 0))
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct XTreesDartHostOwner(Defaultable, Movable):
    """The image-lifetime publisher word of this image's DART table."""

    var word: Int

    def __init__(out self):
        self.word = 1


comptime _OWNER = SharedSlot[XTreesDartHostOwner, "MojolearnXTreesDartHostOwnerV1"]


def publish_x_trees_dart_host(vendor_word: Int) raises:
    """Publish this image's DART entry at module init under the key of its
    glue table (`x_trees_glue_vendor` of the vendor string it exports). A
    build with no vendor word (-1) publishes nothing; its callers refuse
    with the table's "provider is not published"."""
    if vendor_word < 0:
        return
    var owner = _OWNER.get_or_create_ptr()
    var key = TreesEntryKey(Int(GLOBAL_NUMERIC_MODE), vendor_word)
    trees_dart_table_publish(key, Int(owner), [String(XD_DART_ABI)], [XD_OP_COUNT], [x_trees_dart_host_v1])
