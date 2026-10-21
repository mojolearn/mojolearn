# SPDX-License-Identifier: Apache-2.0
"""Typed index stores moved word for word from bindings/_mojolearn_resample.mojo and
bindings/_mojolearn_resample_host.mojo."""


def resample_store_indices(op: MutPointer[Int32, MutUntrackedOrigin], idx: List[Int32], count: Int):
    for i in range(count):
        op.unsafe_store(i, idx[i])
