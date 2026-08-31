# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in gbdt/DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""`CalcHash`: the model-side combination key fold.

PORT OF `catboost/libs/model/hash.h:11-14` at `54a8143a` -- the
"specially designed hash function for low collision rate" their apply path
folds a feature combination with -- plus the exact widening rule its one
caller applies to a category hash on the way in.

Where it is used in their system (`libs/model/ctr_provider.h:94-122`,
`CalcHashes`): a tree-CTR's apply-time table is keyed by starting at 0 and
folding, in the combination's stored order, `CalcHash(acc, element)` where
an element is either a category's `CalcCatFeatureHash` value or a binary
split's 0/1 arm. A SIMPLE ctr's key is the bare category hash and never
takes this fold.

THE SIGN-EXTENSION QUIRK, ported on purpose: `ctr_provider.h:107` feeds a
category hash through `(ui64)(int)`, so a ui32 hash at or above 2^31
enters the fold as `0xffffffff________`. A port that widens with a zero
extension agrees on exactly the half of all hashes below 2^31 and silently
disagrees on the rest -- `cat_hash_chain_element` is that cast, kept as its
own named function so it cannot be inlined away as a "cleanup". The binary
split arm (`:113-120`) is a bare 0/1 and takes no extension.

Not reached by `train()`: this port's tree CTRs do not exist yet, and its
own model format keys tables by dense code. The fold becomes live when
tree-CTR tables land in the model file (RECON_CTRS.md step 6). Gated by
`pixi run check-cityhash` chain rows against their compiled source.
"""

# hash.h:12.
comptime MAGIC_MULT = UInt64(0x4906BA494954CB65)


def calc_hash(a: UInt64, b: UInt64) -> UInt64:
    # hash.h:11-14.
    return MAGIC_MULT * (a + MAGIC_MULT * b)


def cat_hash_chain_element(h: UInt32) -> UInt64:
    # The (ui64)(int) of ctr_provider.h:107: bit-preserving to i32, then
    # SIGN extension to 64 bits, written out as an explicit branch. The
    # first draft chained SIMD casts (uint32 -> int32 -> int64 -> uint64)
    # and check-cityhash caught it ZERO-extending -- every chain row with a
    # hash at or above 2^31 mismatched in exactly the high 32 bits. Assume
    # stdlib numerics are approximate until measured; that rule now covers
    # cast chains too.
    var v = UInt64(h)
    if v >= UInt64(0x80000000):
        return v | UInt64(0xFFFFFFFF00000000)
    return v
