# SPDX-License-Identifier: Apache-2.0
"""Typed checks moved word for word from bindings/msel_host.mojo."""


def msel_rebase_check_parts(la: Int, p: Int, nr: Int, no: Int) raises:
    """Refuses part lengths that do not add up to the raw and merged sizes."""
    var lens = MutPointer[Int64, MutAnyOrigin](unsafe_from_address=la)
    var total = 0
    var outs = 1
    for s in range(p):
        var l = Int(lens.unsafe_load(s))
        if l < 1:
            raise Error("msel_rebase_offsets_i32: every part holds its leading offset")
        total += l
        outs += l - 1
    if total != nr or outs != no:
        raise Error("msel_rebase_offsets_i32: part lengths disagree with the sizes")
