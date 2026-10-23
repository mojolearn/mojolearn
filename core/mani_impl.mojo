# SPDX-License-Identifier: Apache-2.0
"""Typed host helpers of the manifold owners (bindings/mani_*_native.mojo)."""


def mani_same_words(a: MutPointer[UInt32, MutUntrackedOrigin], b: MutPointer[UInt32, MutUntrackedOrigin],
                    count: Int) -> Bool:
    """Whether two float32 buffers hold the same bytes (UMAP.transform's
    `x.tobytes() == training.tobytes()`)."""
    for i in range(count):  # native input marshaling: one byte compare of the query against the fitted rows
        if a[i] != b[i]:
            return False
    return True
