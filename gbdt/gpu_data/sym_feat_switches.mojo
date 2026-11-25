"""lane/apple-fast-sym-feat (2026-10-03): the A/B switches of the symmetric
feature set outside histograms and leaf estimation (quantization, index
build, predict). Every switch but `GBDT_QUANT_DEVICE` is compiled
ONLY under `GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()`
and is OFF by default; `GBDT_QUANT_DEVICE` is OFF by default and opts in on
any GPU image (its note below). One module so `train.mojo`, `gls_borders_device.mojo`,
`doc_parallel_boosting.mojo` and `resident_model.mojo`
read the same flags without importing each other."""
from std.sys.compile import is_defined
from core.accelerator_target import TARGET_HAS_APPLE_GPU
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

comptime SYM_FEAT_FAST_APPLE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and TARGET_HAS_APPLE_GPU
)
# TOMBSTONE: MOJOLEARN_SYM_FEAT_ALL (NEUTRAL: rab4-symfeat istella -1.3%, taxi -0.1%) deleted 2026-11-15 by
# lane/expdel-05-trees; code recoverable at 54b6a264fc.
# Restore: git apply experiments/removed/MOJOLEARN_SYM_FEAT_ALL.patch

#: `-D MOJOLEARN_GBDT_INDEX_PACK_DEVICE`: the compressed index is written by
#: ONE `pack_cindex_words_kernel` launch over every float feature (grid: row
#: blocks x index words, the word's borders staged in threadgroup memory,
#: the word assembled in a register and stored once) instead of one
#: `binarize_float_feature_kernel` launch per feature. Needs the float
#: columns resident on the device, so it implies `GBDT_QUANT_DEVICE`.
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-feat@bca0e3a48; never built (compile owed at the
#: source), never timed. Fixed: the pack kernel compares by `exact_f32_gt`,
#: as main's binarize kernel (subnormal borders on Metal).
# TOMBSTONE: MOJOLEARN_GBDT_INDEX_PACK_DEVICE (DELETE-EXPERIMENT, EXP:170 NEUTRAL; prerequisite of a never-run Apple FAST card, rule g) deleted 2026-11-15 by lane/expdel2-42-trees; code recoverable at b763f5cf63.
# Restore: git apply experiments/removed/MOJOLEARN_GBDT_INDEX_PACK_DEVICE.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/restore-apple-fast-trees: opt-in arm back as it was, default off, Apple fast only; default path unchanged.
comptime GBDT_INDEX_PACK_DEVICE = SYM_FEAT_FAST_APPLE and (
    is_defined["MOJOLEARN_GBDT_INDEX_PACK_DEVICE"]()
)
#: `-D MOJOLEARN_GBDT_QUANT_DEVICE`: the float columns go host to device
#: ONCE, into one resident column-major matrix, before the border build;
#: `device_float_borders` reads its chunks from it instead of uploading the
#: columns itself, and `_build_cindex_from_columns` binarizes from it with
#: one border-slab upload instead of a pinned memcpy, two uploads and a
#: ring drain per feature. A row-major caller (`gbdt_fit_rowmajor`, the
#: C-order array `GradientBoosting.fit` would otherwise transpose on the
#: host) uploads its rows as handed over and the matrix is transposed on the
#: device (`transpose_rows_to_columns_kernel`). Same border kernels, same
#: binarize arithmetic, same words.
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-feat@bca0e3a48; never built (compile owed at the
#: source), never timed. Fixed: the pack kernel compares by `exact_f32_gt`
#: (as INDEX_PACK_DEVICE).
# TOMBSTONE: MOJOLEARN_GBDT_QUANT_DEVICE (DELETE-EXPERIMENT, EXP:1303 NEUTRAL; prerequisite of a never-run Apple FAST card, rule g) deleted 2026-11-15 by lane/expdel2-42-trees; code recoverable at 408e1ba726.
# Restore: git apply experiments/removed/MOJOLEARN_GBDT_QUANT_DEVICE.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/restore-apple-fast-trees: opt-in arm back as it was, default off, Apple fast only; default path unchanged.
# OPT-IN AGAIN 2026-11-24 (lane w3-integrate-9, AGENTS.md "A measured winner is
# never deleted on reasoning": a default changes only on an A/B). Lane
# vdf-scope-trees4 had promoted the device quantizer to the default on every
# GPU image with no timing (EXP:1303 was NEUTRAL on Apple fast only). The
# default is back to round 7's: off on every image, the staged per-chunk path.
# `-D MOJOLEARN_GBDT_QUANT_DEVICE` opts in on any GPU image (no longer Apple
# fast only, so the owed NVIDIA L40S A/B can time it): the float columns
# resident once, and the image answers XG_CAP_FIT_SLOT_X so a model-selection
# fold's X reaches the fit as THE FOLD X WORD (core/fold_x_device.mojo); off,
# the member fit takes the host fold gather (core/gbdt_member.mojo). Same
# border kernels and binarize arithmetic either way. A/B OWED on the L40S:
# docs/apple-fast/EXPERIMENTS.md row MOJOLEARN_GBDT_QUANT_DEVICE (pending).
comptime GBDT_QUANT_DEVICE = (
    is_defined["MOJOLEARN_GBDT_QUANT_DEVICE"]()
) or GBDT_INDEX_PACK_DEVICE
# TOMBSTONE: MOJOLEARN_GBDT_EVAL_SKIP_EMPTY (NEUTRAL) deleted 2026-11-15 by lane/expdel-05-trees; code recoverable at
# 8815b00d75.
# Restore: git apply experiments/removed/MOJOLEARN_GBDT_EVAL_SKIP_EMPTY.patch
# TOMBSTONE: MOJOLEARN_GBDT_EVAL_FUSED (NEUTRAL: rab7-gbdtevalfused istella -0.3%) deleted 2026-11-15 by
# lane/expdel-05-trees; code recoverable at 70801eb24c.
# Restore: git apply experiments/removed/MOJOLEARN_GBDT_EVAL_FUSED.patch
#: `-D MOJOLEARN_GBDT_PREDICT_PACKED`: the resident predict quantizes with
#: ONE `pack_cindex_words_kernel` launch (not one per feature) and applies
#: the whole ensemble with ONE `compute_bins_and_add_all_kernel` launch (not
#: one per tree), the trees in order, the same float32 adds.
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-feat@bca0e3a48; never built (compile owed at the
#: source), never timed.
# TOMBSTONE: MOJOLEARN_GBDT_PREDICT_PACKED (DELETE-EXPERIMENT, EXP:171 NEUTRAL; prerequisite of a never-run Apple FAST card, rule g) deleted 2026-11-15 by lane/expdel2-42-trees; code recoverable at da9c45fa84.
# Restore: git apply experiments/removed/MOJOLEARN_GBDT_PREDICT_PACKED.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/restore-apple-fast-trees: opt-in arm back as it was, default off, Apple fast only; default path unchanged.
comptime GBDT_PREDICT_PACKED = SYM_FEAT_FAST_APPLE and (
    is_defined["MOJOLEARN_GBDT_PREDICT_PACKED"]()
)
# TOMBSTONE: MOJOLEARN_GBDT_BOOT_DEVICE (NEUTRAL: rab7-gbdtbootdevice istella -0.2%; its device seed fill is every
# tier's default since cpu2-l6-bindings, so the switch read nothing) deleted 2026-11-15 by lane/expdel-05-trees; code
# recoverable at 2f51f8b97d.
# Restore: git apply experiments/removed/MOJOLEARN_GBDT_BOOT_DEVICE.patch
