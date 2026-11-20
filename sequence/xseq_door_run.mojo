# SPDX-License-Identifier: Apache-2.0
"""RUN half shared by the sequence-expansion doors (lane doorsplit-cnn-xseq,
2026-11-11; docs/plans/PYTHON_AT_THE_BOUNDARY.md section 2.2).

The doors in `bindings/xseq_*_native.mojo` are the UNPACK/PACK boundary:
they read the model's Python fields, admit the caller's arrays and publish
the results. What they hand to `sequence/` is typed: the glue entry of the
model's connected image, resolved once per operation (`EntryRef`), address
words, integer and Float64 scalars. No
module under `sequence/` imports Python (`tools/python_free_dirs.sh`).

GIL: RUN never touches the GIL; the boundary releases it around a RUN call
(`bindings/xseq_native_common.mojo` `xseq_glue`), where the door released it
before the split.
"""
from core.entry_table import EntryRef, entry_ref_call


def xseq_run_glue(entry: EntryRef, operation: Int, addresses: List[Int], ints: List[Int],
                  floats: List[Float64]) raises -> Int:
    """One synchronous call of the image's glue entry
    (core/xseq_entry_table.mojo layouts) through the resolved ref: one
    indirect call. The caller keeps every addressed buffer alive and has
    released the GIL."""
    return entry_ref_call(entry, operation, addresses, ints, floats)
