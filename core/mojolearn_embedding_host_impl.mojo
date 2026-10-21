# SPDX-License-Identifier: Apache-2.0
"""Typed output store moved word for word from bindings/_mojolearn_embedding_host.mojo."""


def store_each_f32(dst: MutPointer[Float32, MutUntrackedOrigin], values: List[Float32], n: Int):
    for i in range(n):
        dst.unsafe_store(i, values[i])
