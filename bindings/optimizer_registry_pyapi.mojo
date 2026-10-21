# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The public optimizers' constructor entry on the core bindings
(`_mojolearn`, `_mojolearn_core_host`; lane neural-optimizers, 2026-10-21):
the parameter registry's offsets, computed once when `SGD` / `Adam` /
`AdamW` are built, in the tier-free core binding (an optimizer is built
before its numeric mode is known). Unpacks the words and calls
training/optimizer_public_common.mojo `registry_offsets`."""
from std.python import PythonObject

from training.optimizer_public_common import I32P, eager_moments, registry_offsets


def optimizer_registry_binding(nbytes: PythonObject, offsets_addr: PythonObject, flags_addr: PythonObject) raises -> PythonObject:
    """optimizer_registry(nbytes, offsets_addr, flags_addr): `offsets[0 ..
    J]` (J + 1 int32) of a registry of J float32 tensors of `nbytes` bytes
    each. flags_addr (one int64) gets 1 when the optimizer state is
    allocated with the optimizer (MOJOLEARN_OPT_LAZY_MOMENTS=0, the lazy
    allocation's A/B) and 0 when it is allocated on first use. Returns N."""
    var words = List[Int]()
    for j in range(len(nbytes)):  # small-loop(argument words: one byte count per parameter tensor)
        words.append(Int(py=nbytes[j]))
    var a = Int(py=offsets_addr)
    if a == 0:
        raise Error("optimizer_registry: null offsets buffer")
    var flags = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=flags_addr))
    var n = registry_offsets(words, I32P(unsafe_from_address=a))
    flags[0] = Int64(1) if eager_moments() else Int64(0)
    return PythonObject(n)
