# SPDX-License-Identifier: Apache-2.0
"""Typed index stores moved word for word from bindings/_mojolearn_resample.mojo and
bindings/_mojolearn_resample_host.mojo."""
from std.memory import memcpy


def resample_store_indices(op: MutPointer[Int32, MutUntrackedOrigin], idx: List[Int32], count: Int):
    # p3-cnn-sequence-tsa: one memcpy of the drawn indices into the caller's
    # int32 buffer (it was a serial one-element-per-iteration host loop)
    if count > 0:
        memcpy(dest=op, src=idx.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), count=count)
