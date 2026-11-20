# SPDX-License-Identifier: Apache-2.0
"""RUN of `bindings/xseq_native_common.mojo` (wave 8 run split): the
splitmix64 child seed of `XseqStream.child_seed`, typed Mojo only. Imports no
Python (tools/python_free_dirs.sh). The arithmetic is the former method body
unchanged; the method keeps the counter advance (lane rs-sequence-tokenizer,
2026-11-12)."""


def xseq_stream_child_seed(seed: UInt64, offset: Int) -> UInt64:
    """A derived 63-bit seed: splitmix64 of `seed` and counter `offset`."""
    var z = seed + UInt64(offset + 1) * UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return (z ^ (z >> 31)) >> 1
