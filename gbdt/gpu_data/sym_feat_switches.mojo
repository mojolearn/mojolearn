"""lane/apple-fast-sym-feat (2026-10-03): the A/B switches of the symmetric
feature set outside histograms and leaf estimation (quantization, index
build, held-out arm, predict, bootstrap seeds). Every switch is compiled
ONLY under `GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()`
and is OFF by default, so IDENTICAL (and FAST on any other vendor) compiles
main's code unchanged. `-D MOJOLEARN_SYM_FEAT_ALL` turns on every switch
that composes. One module so `train.mojo`, `gls_borders_device.mojo`,
`doc_parallel_boosting.mojo`, `resident_model.mojo` and `bootstrap.mojo`
read the same flags without importing each other."""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

comptime SYM_FEAT_FAST_APPLE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
)
comptime SYM_FEAT_ALL = (
    SYM_FEAT_FAST_APPLE and is_defined["MOJOLEARN_SYM_FEAT_ALL"]()
)

#: `-D MOJOLEARN_GBDT_INDEX_PACK_DEVICE`: the compressed index is written by
#: ONE `pack_cindex_words_kernel` launch over every float feature (grid: row
#: blocks x index words, the word's borders staged in threadgroup memory,
#: the word assembled in a register and stored once) instead of one
#: `binarize_float_feature_kernel` launch per feature. Needs the float
#: columns resident on the device, so it implies `GBDT_QUANT_DEVICE`.
comptime GBDT_INDEX_PACK_DEVICE = SYM_FEAT_FAST_APPLE and (
    is_defined["MOJOLEARN_GBDT_INDEX_PACK_DEVICE"]() or SYM_FEAT_ALL
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
comptime GBDT_QUANT_DEVICE = SYM_FEAT_FAST_APPLE and (
    is_defined["MOJOLEARN_GBDT_QUANT_DEVICE"]() or GBDT_INDEX_PACK_DEVICE
)
#: `-D MOJOLEARN_GBDT_EVAL_SKIP_EMPTY`: with NO eval set `train` still ran
#: `_build_cindex_from_floats` over a one-row dummy (two uploads and a
#: launch per bordered feature, a drain per eight) for a test arm nothing
#: reads; the switch allocates the one-word index and skips the launches.
comptime GBDT_EVAL_SKIP_EMPTY = SYM_FEAT_FAST_APPLE and (
    is_defined["MOJOLEARN_GBDT_EVAL_SKIP_EMPTY"]() or SYM_FEAT_ALL
)
#: `-D MOJOLEARN_GBDT_EVAL_FUSED`: the held-out arm's per-tree apply takes
#: ONE packed upload (split records and leaf values in one pinned buffer)
#: instead of six, and when no overfitting detector can stop the fit the
#: per-tree loss is left on the device (`fv_all[iteration]`) and read back
#: once after the last tree instead of a readback and a drain per tree.
comptime GBDT_EVAL_FUSED = SYM_FEAT_FAST_APPLE and (
    is_defined["MOJOLEARN_GBDT_EVAL_FUSED"]() or SYM_FEAT_ALL
)
#: `-D MOJOLEARN_GBDT_PREDICT_PACKED`: the resident predict quantizes with
#: ONE `pack_cindex_words_kernel` launch (not one per feature) and applies
#: the whole ensemble with ONE `compute_bins_and_add_all_kernel` launch (not
#: one per tree), the trees in order, the same float32 adds.
comptime GBDT_PREDICT_PACKED = SYM_FEAT_FAST_APPLE and (
    is_defined["MOJOLEARN_GBDT_PREDICT_PACKED"]() or SYM_FEAT_ALL
)
#: `-D MOJOLEARN_GBDT_BOOT_DEVICE`: the 65,536 bootstrap seeds are filled
#: by one launch (`bootstrap_seed_fill_kernel`, splitmix64 of
#: `base + (i + 1) * golden`, the host loop's exact values) instead of a
#: host loop, a pinned upload and a drain.
comptime GBDT_BOOT_DEVICE = SYM_FEAT_FAST_APPLE and (
    is_defined["MOJOLEARN_GBDT_BOOT_DEVICE"]() or SYM_FEAT_ALL
)
