# lane/apple-fast-sym-feat: symmetric-tree work outside histograms and leaf estimation

Profile: `docs/apple-fast/notes/sym-feat.md`. Switches: `gbdt/gpu_data/sym_feat_switches.mojo`. Every switch is
compiled only under FAST + Apple (`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()`) and is
OFF by default; IDENTICAL compiles main's code. No switch changes a bit: same kernels or the same arithmetic in
the same order, so the FAST words, model and predictions match the switch-off FAST build. Requests:
`docs/apple-fast/ab/sym-feat.txt` (arm A = FAST off, arm B = the switch; istella first, then taxi; ALL also on
symmetric-1000 taxi).

| define | site | what it changes |
|---|---|---|
| `-D MOJOLEARN_GBDT_QUANT_DEVICE` | `gbdt/train.mojo` (`_qd_upload_float_columns`, `_cindex_from_device_columns`, the `train` upload), `gbdt/grid_creator/gls_borders_device.mojo`, `gbdt/estimator.mojo` + `bindings/_mojolearn_gbdt.mojo` (`gbdt_fit_rowmajor`), `python/mojolearn/ensemble.py` | float columns go to the device once; borders and binarization read that copy; a C-order X is uploaded as is and transposed on the device |
| `-D MOJOLEARN_GBDT_INDEX_PACK_DEVICE` | `gbdt/train.mojo` `_cindex_from_device_columns`, `gbdt/gpu_data/kernel/binarize.mojo` `pack_cindex_words_kernel` | the compressed index in one launch over every word (implies QUANT_DEVICE) |
| `-D MOJOLEARN_GBDT_EVAL_SKIP_EMPTY` | `gbdt/train.mojo` `_sym_feat_test_cindex` | no eval set: no one-row dummy held-out index build |
| `-D MOJOLEARN_GBDT_PREDICT_PACKED` | `gbdt/resident_model.mojo` `_build_packed_predict`, `_apply_all`, `_predict_device`; `gbdt/models/kernel/add_bin_values.mojo` `compute_bins_and_add_all_kernel` | resident predict: one pack launch to quantize, one launch for the whole ensemble |
| `-D MOJOLEARN_GBDT_EVAL_FUSED` | `gbdt/methods/doc_parallel_boosting.mojo` (`_apply_last_tree_to_test`, `_test_loss_enqueue`, `_test_loss_flush`, the fit loop) | eval set: one upload per tree for the held-out apply; with an inactive detector the held-out loss is read back once per 1024 trees |
| `-D MOJOLEARN_GBDT_BOOT_DEVICE` | `gbdt/gpu_util/kernel/bootstrap.mojo` `bootstrap_seed_fill_kernel` | the 65,536 bootstrap seeds in one launch |
| `-D MOJOLEARN_SYM_FEAT_ALL` | all of the above | every switch (they compose) |

**QUANT_DEVICE.** Mechanism: before, the matrix crossed to the device twice (per chunk for the borders, then per
feature for the index through a pinned memcpy, two uploads and a ring drain per eight features; istella: 220
memcpys of 4 MB, 440 uploads, 28 drains), and on C-order input `GradientBoosting.fit` first transposed the
whole matrix on the host (istella 880 MB). Now the float columns are uploaded once into one resident
column-major matrix; `device_float_borders` reads its chunks in place, and the index build uploads one border-slab
table and binarizes from the device copy. A C-order float32 X without categorical/one-hot features goes through
`gbdt_fit_rowmajor`: rows go up as handed over and `transpose_rows_to_columns_kernel` lays them out. Expected:
the fit's pre-tree phase drops by the host transpose plus the second upload and the drains; larger on istella
than taxi. Risk: one extra resident buffer the size of X during quantization (freed before the first tree);
a fit with categorical/one-hot columns keeps the column-major staging for those columns.

**INDEX_PACK_DEVICE.** Mechanism: `pack_cindex_words_kernel` takes one block row per index word; the word's
features' borders are staged in threadgroup memory (fits gate: at most 32 features and 1,024 borders a word,
else the per-feature launches), each thread builds 8 rows' words in registers and stores each word once (no
OR into a zeroed buffer). 220 launches on istella become 1. Expected: a smaller win on top of QUANT_DEVICE,
mostly on istella (taxi has 11 features). Risk: one thread per block fills the per-word metadata (at most 32
entries), negligible.

**EVAL_SKIP_EMPTY.** Mechanism: with no eval set, `train` still built the held-out index over a one-row dummy
(two uploads and a launch per bordered feature, a drain per eight: 220 launches, 440 uploads, ~29 drains on
istella). Nothing reads it; the switch allocates one word instead. Expected: a few ms to tens of ms per fit on
istella, less on taxi. Risk: none (the arm with `n_rows == 0` reads no buffer).

**PREDICT_PACKED.** Mechanism: the board scores the held-out rows through the resident predict, which launched
one binarize per bordered feature (220 on istella) and one `compute_bins_and_add_kernel` per tree (500 or
1,000). Now one `pack_cindex_words_kernel` launch and one `compute_bins_and_add_all_kernel` launch (trees walked
in chunks of up to 1,024 levels staged in threadgroup memory, the cursor held in a register across a chunk, the
same float32 adds in tree order). The tables ride one upload; one extra drain per predict call. Expected: the
predict phase (if the board times it) drops by the launch overhead, most on symmetric-1000. Risk: one thread
now walks every tree for its rows, so a launch can run longer; very large ensembles x rows could approach the
macOS ~4 s command-buffer limit (1M rows x 1,000 depth-6 trees is well under it on an M3 Ultra).

**EVAL_FUSED.** Mechanism: with an eval set, each tree's held-out apply packed its records and leaves into six
pinned buffers and six uploads; now one region of `h_vals` and one upload. When no overfitting detector can stop
the fit (`od_type` None), the held-out loss stays on the device in one of 1,024 slots and is read back once per
1,024 trees and after the loop, instead of a readback and a drain per tree; the detector and `test_losses` see
the same values in the same order. Expected: no change on the board (it fits without `eval_set`); helps fits
with an eval set. Risk: the packed host region is rewritten each tree without its own drain, relying on the
per-tree leaf-value readback that precedes the append (documented at the call site).

**BOOT_DEVICE.** Mechanism: the 65,536 bootstrap seeds came from a host splitmix64 loop, a pinned upload and a
drain; now `bootstrap_seed_fill_kernel` computes seed `i` as splitmix64 of `base + (i + 1) * golden`, the same
values, in one launch. Expected: no change on the board (`bootstrap_type='No'`); a fraction of a ms per fit
with a bootstrap. Risk: none.

**SYM_FEAT_ALL.** All six together; they compose (PACK implies QUANT). The deciding line is istella.

## Compile status (2026-10-03)

No build of this branch completed on the laptop: the one queued build (FAST + `-D MOJOLEARN_SYM_FEAT_ALL`) never
got a compile slot and was stopped on Andrew's instruction. Every build is **compile owed: peer**:
FAST per define (QUANT_DEVICE, INDEX_PACK_DEVICE, EVAL_SKIP_EMPTY, PREDICT_PACKED, EVAL_FUSED, BOOT_DEVICE),
FAST + SYM_FEAT_ALL, FAST with every define off, and IDENTICAL once. Python: `python/mojolearn/ensemble.py`
passes `py_compile`. Build the ALL arm first: it instantiates every new kernel and every guarded branch.
