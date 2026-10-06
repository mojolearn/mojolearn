# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Apple FAST bundle for Ordered boosting (lane/apple-fast-sym-ordered,
2026-10-03; default since lane/apple-fast-ordall). One module so the fit
(`ordered_boosting.mojo`) and the structure searcher
(`oblivious_tree_doc_parallel_structure_searcher.mojo`) read the same gate.

`ORD_ALL` is compiled only under the FAST tier on an Apple GPU; IDENTICAL,
DETERMINISTIC and non-Apple FAST compile main's statements unchanged.
`-D MOJOLEARN_ORD_ALL_OFF` compiles it out. Since 2026-10-04 it runs at
every width; the old width gate (more than 32 features) was removed as
benchmark-tuned and survives only behind MOJOLEARN_LEGACY_NARROW_ORD_ALL.

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
from std.sys.info import has_apple_gpu_accelerator

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

#: the bundle is compiled in (FAST + Apple, not opted out)
comptime ORD_ALL = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_ORD_ALL_OFF"]()
)
#: LEGACY, default OFF: the old width gate admitted only data with more than
#: 32 compressed-index features, chosen between taxi (11, auc -0.0003 in one
#: run) and istella (220). Removed as benchmark-tuned on 2026-10-04; the
#: bundle now runs at every width, and that is UNMEASURED (taxi quality owed).
comptime ORD_ALL_LEGACY_WIDTH = is_defined["MOJOLEARN_LEGACY_NARROW_ORD_ALL"]()
comptime ORD_ALL_MIN_FEATURES = 32


def ord_all_on(n_features: Int) -> Bool:
    """True when the bundle is compiled in (any width since 2026-10-04)."""
    comptime if ORD_ALL:
        comptime if ORD_ALL_LEGACY_WIDTH:
            return n_features > ORD_ALL_MIN_FEATURES
        else:
            return n_features >= 1
    else:
        return False


# F12 pending device quality/performance. Restore stable document-key dither
# while retaining coalesced fold-position compressed storage. Existing ORD_ALL
# promotion is unchanged; IDENTICAL/non-Apple builds cannot admit this candidate.
comptime ORD_DOC_ID_STORAGE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_ORD_DOC_ID_STORAGE"]()
)


from std.atomic import Atomic, Ordering
from std.ffi import _Global

struct _OrderedDocAudit(Defaultable,Movable):
    var submissions: Int64
    def __init__(out self):self.submissions=Int64(0)

comptime _ORD_DOC_AUDIT=_Global[StorageType=_OrderedDocAudit,name="MojoOrderedDocAuditV1",init_fn=_OrderedDocAudit.__init__]

def ordered_doc_count() raises -> Int:
    comptime if ORD_DOC_ID_STORAGE:
        ref audit=_ORD_DOC_AUDIT.get_or_create_ptr()[]
        return Int(Atomic.load[ordering=Ordering.RELAXED](MutPointer(to=audit.submissions)))
    return 0

def ordered_doc_hit() raises:
    comptime if ORD_DOC_ID_STORAGE:
        ref audit=_ORD_DOC_AUDIT.get_or_create_ptr()[]
        _=Atomic.fetch_add[ordering=Ordering.RELAXED](MutPointer(to=audit.submissions),Int64(1))
