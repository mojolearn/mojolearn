# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Apple FAST bundle for Ordered boosting (lane/apple-fast-sym-ordered,
2026-10-03; default since lane/apple-fast-ordall). One module so the fit
(`ordered_boosting.mojo`) and the structure searcher
(`oblivious_tree_doc_parallel_structure_searcher.mojo`) read the same gate.

`ORD_ALL` is compiled only under the FAST tier on an Apple GPU; IDENTICAL
and non-Apple FAST compile main's statements unchanged.
Its rollback MOJOLEARN_ORD_ALL_OFF was deleted 2026-11-15 (docs/TOMBSTONES.md). Since 2026-10-04 it runs at
every width; the old width gate (more than 32 features) was removed as
benchmark-tuned (its switch MOJOLEARN_LEGACY_NARROW_ORD_ALL was deleted
2026-11-15, docs/TOMBSTONES.md).

WHY THE OLD WIDTH GATE EXISTED (M3 A/Bs, one run per arm, 2026-10-03, gbdt-ordered):
    istella (220 features): 75,758 -> 63,554 ms (-16.1%), auc .979518 ->
        .979529, logloss .190603 -> .190385 (better on both).
    taxi (11 features): 56,473 -> 48,163 ms (-14.7%) but auc .628875 ->
        .628572, logloss .52909 -> .529215 (worse). FAST never degrades
        quality, so narrow data keeps the current path. Of the pieces
        alone, only the fold-order index moved taxi (-5.9%) and it is
        what drops taxi's auc; the others alone were noise (STD_PARALLEL
        istella -1.8%, TREE_LEAN istella ~-0.8%, FOLD_BINS_ONE taxi
        -1.0%), so they no longer have their own defines.

The bundle's four pieces, all on together when `ord_all_on`:
    fold bins in one launch: the fold layout's initial bins from a
        device-resident partition-start table (uploaded once a fit)
        instead of `write_fold_based_initial_bins`' staging buffer, fill
        and copy PER PARTITION and its drain, every tree.
    wide score-noise reduce: the noise sum and the scale's two magnitudes
        over REDUCE_LANES_BLOCK blocks of ORDERED_BLOCK threads (the
        library block sum) instead of the eight 32-lane blocks of
        `_ord_std_lanes_kernel`; no per-tree `MOJOLEARN_ORD_STD_SPLIT` read.
    fold-order index: the compressed index at the fold POSITION (gathered
        once per learn permutation, cached in the pool) as the compiled
        default; no per-level doc-id gather, no `d_observations` buffer.
    lean tree buffers: both planes, the score-noise partials, the tree's
        bins and the searcher's observation scratch allocated once a fit
        and rewritten each tree; the partition sort's three dead host
        readbacks per permutation skipped under the batched estimation.
"""

from std.sys.compile import is_defined
from gbdt.trees_identical_switches import T22
from core.accelerator_target import TARGET_HAS_APPLE_GPU

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

#: the bundle is compiled in (FAST + Apple, not opted out)
# F12/ordered-storage M3 2026-10-06 confirms this already-ON Apple FAST
# bundle vs ORD_ALL_OFF: 24-tree fit+first-predict B/A0.7802 (511x11,
# depth4),0.8846 (1023x17,depth5); repeated predict0.9203,0.9955.
# Seed7, lr0.08, four permutations, bootstrapNo; task errors exactly equal
# (0.7239750566,0.6157665953). One warmup/one score; caller67d0efb29,
# reused per-arm hashes/builds: ~/mojolearn-evidence/ab-overnight-20261006/
# m3/artifacts/results/F12/ordered-storage. Keep existing default ON;
# these public caller cases do not claim full-dataset/opponent admission.
# TOMBSTONE: MOJOLEARN_ORD_ALL_OFF (DELETE-OLD-PATH, EXP:180: the default is the kept path, rule e) deleted 2026-11-15 by lane/expdel2-44-trees; code recoverable at 611ff32f4b.
# Restore: git apply experiments/removed/MOJOLEARN_ORD_ALL_OFF.patch; record in docs/TOMBSTONES.md.
comptime ORD_ALL = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and TARGET_HAS_APPLE_GPU
# TOMBSTONE: MOJOLEARN_LEGACY_NARROW_ORD_ALL (DELETE-OLD-PATH, the benchmark-tuned width gate, no A/B recorded for the old path, rule b) deleted 2026-11-15 by lane/expdel2-44-trees; code recoverable at f8ce124897.
# Restore: git apply experiments/removed/MOJOLEARN_LEGACY_NARROW_ORD_ALL.patch; record in docs/TOMBSTONES.md.


def ord_all_on(n_features: Int) -> Bool:
    """True when the bundle is compiled in (any width since 2026-10-04)."""
    comptime if ORD_ALL:
        return n_features >= 1
    else:
        return False


# F12 pending device quality/performance. Restore stable document-key dither
# while retaining coalesced fold-position compressed storage. Existing ORD_ALL
# promotion is unchanged; IDENTICAL/non-Apple builds cannot admit this candidate.
# TOMBSTONE: MOJOLEARN_ORD_DOC_ID_STORAGE (DELETE-ARM, opt-in arm with no deciding A/B, rule a) deleted 2026-11-15 by lane/expdel2-44-trees; code recoverable at 48af6b7d48.
# Restore: git apply experiments/removed/MOJOLEARN_ORD_DOC_ID_STORAGE.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/restore-apple-fast-trees: opt-in arm back as it was, default off, Apple fast only; default path unchanged.
comptime ORD_DOC_ID_STORAGE = T22 or (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and TARGET_HAS_APPLE_GPU
    # MEASURED M3 FAST; broader workload evidence remains separate.
# F12/ordered-docids M3 2026-10-06: 4 scored caller times; B/A
# 0.8957..1.0334 (mixed/regressing); FAST candidate remains OFF.
# Scored FAST quality 2/2 within existing bands; PASS.
# One warmup/one score; caller67d0efb29; exact cases/builds/hashes:
# ~/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/ordered-docids.
# Compilation/identity reused. No combined-toggle/full-board claim.
    and is_defined["MOJOLEARN_ORD_DOC_ID_STORAGE"]()
)


from std.atomic import Atomic, Ordering
from core.shared_slot import SharedSlot

struct _OrderedDocAudit(Defaultable,Movable):
    var submissions: Int64
    def __init__(out self):self.submissions=Int64(0)

comptime _ORD_DOC_AUDIT=SharedSlot[_OrderedDocAudit, "MojolearnOrderedDocAuditV2"]

def ordered_doc_count() raises -> Int:
    comptime if ORD_DOC_ID_STORAGE:
        ref audit=_ORD_DOC_AUDIT.get_or_create_ptr()[]
        return Int(Atomic.load[ordering=Ordering.RELAXED](MutPointer(to=audit.submissions)))
    return 0

def ordered_doc_hit() raises:
    comptime if ORD_DOC_ID_STORAGE:
        ref audit=_ORD_DOC_AUDIT.get_or_create_ptr()[]
        _=Atomic.fetch_add[ordering=Ordering.RELAXED](MutPointer(to=audit.submissions),Int64(1))
