# SPDX-License-Identifier: Apache-2.0
"""Opt-in strategy admission for the base bindings (base, metrics, preprocessing, x_prep, x_metrics, the scoped GEMM probe).

Each binding of this family imports BASE_GUARDS_OK and asserts it in its build function, so every compile of one of
those bindings (per-binding or merged) evaluates these guards. A guard whose define is read in code by bindings
of more than one family belongs in core/six_lane_experiment_guards.mojo, which every binding reaches through
checks/numerics.mojo.
"""
from std.sys.compile import is_defined, get_defined_int
# The entries below moved verbatim, in their original order, from core/six_lane_experiment_guards.mojo on
# 2026-11-04 (lane split-hot-groups; the placement of each entry is docs/build/guard_split.json). A header comment
# that headed lines of several families is carried with each of them.


def _check_configuration() -> Bool:
    # Lane grid-act-3 (2026-10-08): IDENTICAL grid ge123e6f9 promotions and losers (docs/apple-fast/EXPERIMENTS.md). Deleted code recoverable at main bc10b8b56.
    comptime assert not (is_defined["MOJOLEARN_CLASSICAL_C61_NB_CLASS_STATS"]() or is_defined["MOJOLEARN_CLASSICAL_C61_DA_CLASS_STATS"]()), "removed: MOJOLEARN_CLASSICAL_C61_NB_CLASS_STATS (=1|2) and MOJOLEARN_CLASSICAL_C61_DA_CLASS_STATS retired 2026-10-08: slower, gaussian-nb NV 1.37-1.58x / AMD 1.02-1.08x, lda-clf NV 1.38x/1.54x / AMD 1.04x/1.11x (istella/taxi), quality SAME (grid ge123e6f9); see EXPERIMENTS.md"
    # Lane postmerge-act-1 (2026-10-09): post-merge A/B races on main, one run per arm (nv L40S, amd MI325X; docs/apple-fast/EXPERIMENTS.md). Old code recoverable at main 432d6e8ff.
    comptime assert not (is_defined["MOJOLEARN_XPREP_PINNED_UPLOAD_OFF"]() or is_defined["MOJOLEARN_XPREP_PINNED_UPLOAD"]()), "removed 2026-10-09 (post-merge A/B nv n0607->n0628, amd a1066->a1088, lane/postmerge-act-1): the G1 DeviceStore pinned upload stage (2 x 32 MB) was SLOWER than the pageable copy on both vendors: gaussian-nb istella NV 129.8 -> 88.7 ms with _OFF (0.68x), AMD 46.4 -> 27.9 (0.60x); taxi NV 20.9 -> 19.3 (0.92x), AMD 14.1 -> 12.9 (0.91x); same bits; the direct copy is the only path; code at main 432d6e8ff; see docs/apple-fast/EXPERIMENTS.md"
    # Lane postmerge-act-2 (2026-10-09): post-merge A/B races on main, one run per arm (nv L40S, amd MI325X; docs/apple-fast/EXPERIMENTS.md). Deleted code recoverable at main 5c137b55e.
    comptime assert not is_defined["MOJOLEARN_XPREP_DEVICE_CODES"](), "removed 2026-10-09 (post-merge A/B nv n0668->n0631, amd a1066->a1091, lane/postmerge-act-2): fg-knn-nb G5 (GaussianNB.fit label codes from the device unique_inverse) was SLOWER: gaussian-nb istella NV 1.29x / AMD 1.12x, taxi NV 1.70x / AMD 1.35x, accuracy SAME, same digests; probe and Python route deleted; code at main 5c137b55e; see EXPERIMENTS.md"
    comptime assert not is_defined["MOJOLEARN_XPREP_NO_SLOT_HOP"](), "removed 2026-10-09 (post-merge A/B nv n0668->n0630, amd a1066->a1090, lane/postmerge-act-2): fg-knn-nb G3 (large direct inputs as host spans straight into the x_prep arena) was SLOWER on the average: gaussian-nb istella NV 1.25x / AMD 1.00x, taxi NV 1.02x / AMD 1.03x, accuracy SAME, same digests; x_prep_run_ranges_host, upload_ranges_host and the _Prog.run glue deleted; code at main 5c137b55e; see EXPERIMENTS.md"
    # Lane owed-deletions-D2 (2026-10-09): owed deletions of DROPPED Apple FAST / neural defines (docs/TOMBSTONES.md, docs/apple-fast/EXPERIMENTS.md). Deleted code recoverable at main b639a2bd2.
    comptime assert not (is_defined["MOJOLEARN_MI_ALL"]()), "removed 2026-10-09 (lane/owed-deletions-D2): MOJOLEARN_MI_ALL, the mutual-information FAST bundle, was DROP (quality): M3 miv-reg-all istella 46,430 -> 1,151 ms but it includes MI_FAST_FOLDS (selected set changes); name the members instead; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not (is_defined["MOJOLEARN_MI_FAST_FOLDS"]()), "removed 2026-10-09 (lane/owed-deletions-D2): MOJOLEARN_MI_FAST_FOLDS, the mutual-information column threadgroup folds, was DROP (speed + quality): istella -0.2%, scores move up to 3.2% of scale and the selected set changes; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not (is_defined["MOJOLEARN_PTIMPUTE_ALL"]()), "removed 2026-10-09 (lane/owed-deletions-D2): MOJOLEARN_PTIMPUTE_ALL, the PowerTransformer FAST bundle, was DROP (quality): M3 istella 2,258 -> 512 ms but lambda relative shift 9.5e-3 fails the 1e-4 gate; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not (is_defined["MOJOLEARN_PT_SPEC"]()), "removed 2026-10-09 (lane/owed-deletions-D2): MOJOLEARN_PT_SPEC, the FAST speculated PowerTransformer search on the tiled kernel, was DROP: M3 istella +10% vs COLBATCH (910 -> 1,012 ms); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not (is_defined["MOJOLEARN_PT_FUSED_TRANSFORM"]()), "removed 2026-10-09 (lane/owed-deletions-D2): MOJOLEARN_PT_FUSED_TRANSFORM, the fused PowerTransformer standardize tail, was DROP (quality, with COLBATCH): batchv-pt-nospec lambda shift 9.5e-3; off under the default PT_SCORE_STABLE; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not (is_defined["MOJOLEARN_PT_FOLD_NOX"]()), "removed 2026-10-09 (lane/owed-deletions-D2): MOJOLEARN_PT_FOLD_NOX, the PowerTransformer fold that skips X after K = 0, was DROP: no recorded gain (ptimpute-pt-nox-*), moot under COLBATCH; code at main b639a2bd2; see docs/TOMBSTONES.md"
    # Lane consol-k0-dead-code (2026-11-08): the opt-in name of the PT_SCORE default kernels (docs/TOMBSTONES.md batch 3).
    # Restore: git apply experiments/removed/MOJOLEARN_PT_COLBATCH.patch
    comptime assert not is_defined["MOJOLEARN_PT_COLBATCH"](), "removed 2026-11-08 (lane consol-k0-dead-code): MOJOLEARN_PT_COLBATCH, the opt-in name of the tiled PowerTransformer evaluation, was DROP (quality): M3 batchv-pt-nospec istella 2,262 -> 425 ms but lambdas shift up to 9.5e-3 relative vs define-off (docs/apple-fast/EXPERIMENTS.md); its kernels stay as the FAST + Apple PT_SCORE default, so drop the define; code at 41ffcec665; see docs/TOMBSTONES.md"
    # Lane owed-deletions-D3 (2026-10-09): owed DROP deletions (docs/TOMBSTONES.md 'Owed deletions'; docs/apple-fast/EXPERIMENTS.md). Deleted code recoverable at main b639a2bd2.
    comptime assert not is_defined["MOJOLEARN_KMEANS_FAST_LAZY_SHIFT"](), "removed 2026-10-09 (lane/owed-deletions-D3): KMEANS_FAST_LAZY_SHIFT (lazy convergence read on the KMeans estimator's fits) was SLOWER: kmeans istella 1451.5 -> 1527.7 ms, taxi 977.6 -> 974.8 ms, inertia equal; the IVF callers' lazy_shift stays; code at main b639a2bd2; see docs/TOMBSTONES.md"
    # Lane i17-decide-nbr-lin-prep (2026-10-11), removed: A/B on main b2657cc89, one run per arm, NV L40S nv2 / AMD MI325X (docs/TOMBSTONES.md,
    # docs/apple-fast/EXPERIMENTS.md, experiments/removed/*.patch).
    comptime assert not is_defined["MOJOLEARN_IDN_LAB_UNIQUE_HIST"](), "removed 2026-10-11 (lane/i17-decide-nbr-lin-prep, de9ce7b39): IDN_LAB_UNIQUE_HIST (G1 label classes by presence flags) was SLOWER: label-encoder 2.1-2.6x NV + AMD, label-binarizer istella 1.15-1.39x (v0041-v0044, a1624-a1627); code at de9ce7b39^, patch experiments/removed/MOJOLEARN_IDN_LAB_UNIQUE_HIST_CODES_DIRECT_ONE_ENTRY.patch"
    comptime assert not is_defined["MOJOLEARN_IDN_LAB_CODES_DIRECT"](), "removed 2026-10-11 (lane/i17-decide-nbr-lin-prep, de9ce7b39): IDN_LAB_CODES_DIRECT (G2 flag-rank codes) was SLOWER: label-encoder 2.1-2.6x NV + AMD (v0041-v0044, a1624-a1627); code at de9ce7b39^, patch experiments/removed/MOJOLEARN_IDN_LAB_UNIQUE_HIST_CODES_DIRECT_ONE_ENTRY.patch"
    comptime assert not is_defined["MOJOLEARN_IDN_LAB_ONE_ENTRY"](), "removed 2026-10-11 (lane/i17-decide-nbr-lin-prep, de9ce7b39): IDN_LAB_ONE_ENTRY (G3 one-entry LabelBinarizer.fit_transform) was SLOWER: label-binarizer istella 1.15-1.39x, taxi ~1.0 (v0041-v0044, a1624-a1627); code at de9ce7b39^, patch experiments/removed/MOJOLEARN_IDN_LAB_UNIQUE_HIST_CODES_DIRECT_ONE_ENTRY.patch"
    # wave-2 w2-decide-algos (2026-10-11): promoted (absent = new path; _OFF the old path) and removed switches.
    comptime assert not is_defined["MOJOLEARN_IDN_DRADIX_RANGE_PASSES"](), "promoted 2026-10-11 (lane/w2-decide-algos, same-box A/B on main c8ef6c66d): range-pass radix sort is the IDENTICAL default (prep mean 0.945, ordinal/onehot AMD 0.59-0.69); drop -D MOJOLEARN_IDN_DRADIX_RANGE_PASSES, use -D MOJOLEARN_IDN_DRADIX_RANGE_PASSES_OFF for the old path"
    # Lane expdel-01-cluster (2026-11-15): owed deletions of docs/plans/EXPERIMENT_DELETIONS_2026-11-15.md (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_XN_FAST_SCHI2_LAZYW_OFF.patch
    comptime assert not is_defined["MOJOLEARN_XN_FAST_SCHI2_LAZYW_OFF"](), "removed 2026-11-15 (lane/expdel-01-cluster): XN_FAST_SCHI2_LAZYW_OFF (FAST+Apple rollback to SkewedChi2Sampler's eager weights) is gone; the lazy weights are the FAST+Apple path (taxi 1.8 -> 0.4 ms, istella 2.1 -> 0.9 ms, same words); restore: experiments/removed/MOJOLEARN_XN_FAST_SCHI2_LAZYW_OFF.patch"
    # Lane expdel-06-base-cluster-kernel (2026-11-15): owed deletions of docs/plans/EXPERIMENT_DELETIONS_2026-11-15.md (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_AF_FAST_RESIDENT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AF_FAST_RESIDENT_OFF"](), "removed 2026-11-15 (lane/expdel-06-base-cluster-kernel): AF_FAST_RESIDENT_OFF (FAST+Apple rollback to Adafactor's per-call entry) is gone; the resident second moment is the FAST+Apple path (adafactor 392.8 -> 205.2 ms, digest identical, rab10-afresident); IDENTICAL keeps MOJOLEARN_IDN_AF_RESIDENT_OFF; restore: experiments/removed/MOJOLEARN_AF_FAST_RESIDENT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_APPLE_FAST_GEMM_TN_V1_OFF.patch
    comptime assert not is_defined["MOJOLEARN_APPLE_FAST_GEMM_TN_V1_OFF"](), "removed 2026-11-15 (lane/expdel-06-base-cluster-kernel): APPLE_FAST_GEMM_TN_V1_OFF (FAST+Apple rollback of square gemm_tn past split-K to the vendor matmul) is gone; IDENTICAL's gemm_tn_identical_v1 arm is the FAST+Apple path (ols istella 2,219 -> 935 ms, tier-ols-tnv1b); restore: experiments/removed/MOJOLEARN_APPLE_FAST_GEMM_TN_V1_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KM_FAST_RBF_RESIDENT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KM_FAST_RBF_RESIDENT_OFF"](), "removed 2026-11-15 (lane/expdel-06-base-cluster-kernel): KM_FAST_RBF_RESIDENT_OFF (FAST+Apple rollback of RBFSampler fit_transform to main's two binding calls) is gone; the one-call resident fit_transform is the FAST+Apple path (rbf-sampler istella 93.4 -> 76.3 ms, w2-kfeat-rbf-q-r1 PASS); restore: experiments/removed/MOJOLEARN_KM_FAST_RBF_RESIDENT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_X_PREP_FAST_UNIQUE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_X_PREP_FAST_UNIQUE_OFF"](), "removed 2026-11-15 (lane/expdel-06-base-cluster-kernel): X_PREP_FAST_UNIQUE_OFF (FAST+Apple rollback of the encoders' category fit to one thread per column) is gone; the chunked run scan is the FAST+Apple path (onehot taxi 69.7 -> 30.4 ms, ordinal 67.2 -> 25.3 ms, prep-onehot-uniq-taxi); restore: experiments/removed/MOJOLEARN_X_PREP_FAST_UNIQUE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SPLINE_FAST_FUSED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SPLINE_FAST_FUSED_OFF"](), "removed 2026-11-15 (lane/expdel-06-base-cluster-kernel): SPLINE_FAST_FUSED_OFF (FAST+Apple rollback of SplineTransformer to main's unfused fit and transform) is gone; the fused program is the FAST+Apple path (spline istella 48.1 -> 11.6 ms, taxi 34.2 -> 10.8 ms, kap2-spl-fused); restore: experiments/removed/MOJOLEARN_SPLINE_FAST_FUSED_OFF.patch"
    # Lane expdel2-02-base (2026-11-15): unmeasured arms deleted, docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_BYTE_LM_LOGITS_ZERO_FILL.patch
    comptime assert not is_defined["MOJOLEARN_BYTE_LM_LOGITS_ZERO_FILL"](), "removed 2026-11-15 (lane/expdel2-02-base): BYTE_LM_LOGITS_ZERO_FILL (opt-in arm, never measured) is gone; the default is kept; code at 337c096a5b; restore: experiments/removed/MOJOLEARN_BYTE_LM_LOGITS_ZERO_FILL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_DART_ROUND_REF_OFF.patch
    comptime assert not is_defined["MOJOLEARN_DART_ROUND_REF_OFF"](), "removed 2026-11-15 (lane/expdel2-02-base): DART_ROUND_REF_OFF (old path, never measured) is gone; the default is kept; code at 650388311e; restore: experiments/removed/MOJOLEARN_DART_ROUND_REF_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GBDT_STEP_OLD.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GBDT_STEP_OLD"](), "removed 2026-11-15 (lane/expdel2-02-base): IDN_GBDT_STEP_OLD (old path, never measured) is gone; the default is kept; code at f644604c7a; restore: experiments/removed/MOJOLEARN_IDN_GBDT_STEP_OLD.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_HPDEV_INIT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_HPDEV_INIT_OFF"](), "removed 2026-11-15 (lane/expdel2-02-base): IDN_HPDEV_INIT_OFF (old path, never measured) is gone; the default is kept; code at 2136ab3bb0; restore: experiments/removed/MOJOLEARN_IDN_HPDEV_INIT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_HPDEV_NORMAL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_HPDEV_NORMAL_OFF"](), "removed 2026-11-15 (lane/expdel2-02-base): IDN_HPDEV_NORMAL_OFF (old path, never measured) is gone; the default is kept; code at 31aef6de0f; restore: experiments/removed/MOJOLEARN_IDN_HPDEV_NORMAL_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_IF_FIT_OWN_X.patch
    comptime assert not is_defined["MOJOLEARN_IDN_IF_FIT_OWN_X"](), "removed 2026-11-15 (lane/expdel2-02-base): IDN_IF_FIT_OWN_X (opt-in arm, never measured) is gone; the default is kept; code at 0a98ac1ae9; restore: experiments/removed/MOJOLEARN_IDN_IF_FIT_OWN_X.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_IF_FIT_EXPORT.patch
    comptime assert not is_defined["MOJOLEARN_IDN_IF_FIT_EXPORT"](), "removed 2026-11-15 (lane/expdel2-02-base): IDN_IF_FIT_EXPORT (opt-in arm, never measured) is gone; the default is kept; code at cfce0d30ce; restore: experiments/removed/MOJOLEARN_IDN_IF_FIT_EXPORT.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_LM_RESIDENT_GENERATE.patch
    comptime assert not is_defined["MOJOLEARN_IDN_LM_RESIDENT_GENERATE"](), "removed 2026-11-15 (lane/expdel2-02-base): IDN_LM_RESIDENT_GENERATE (opt-in arm, never measured) is gone; the default is kept; code at 29fe2d5748; restore: experiments/removed/MOJOLEARN_IDN_LM_RESIDENT_GENERATE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_FORWARD_ZERO_FILL.patch
    comptime assert not is_defined["MOJOLEARN_MAMBA3_FORWARD_ZERO_FILL"](), "removed 2026-11-15 (lane/expdel2-02-base): MAMBA3_FORWARD_ZERO_FILL (opt-in arm, never measured) is gone; the default is kept; code at 54f0ca8d71; restore: experiments/removed/MOJOLEARN_MAMBA3_FORWARD_ZERO_FILL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MLP_SESSIONS_PACKED.patch
    comptime assert not is_defined["MOJOLEARN_MLP_SESSIONS_PACKED"](), "removed 2026-11-15 (lane/expdel2-02-base): MLP_SESSIONS_PACKED (opt-in arm, never measured) is gone; the default is kept; code at 1c0608ab17; restore: experiments/removed/MOJOLEARN_MLP_SESSIONS_PACKED.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MSEL_UNITS_GPU_PARALLEL.patch
    comptime assert not is_defined["MOJOLEARN_MSEL_UNITS_GPU_PARALLEL"](), "removed 2026-11-15 (lane/expdel2-02-base): MSEL_UNITS_GPU_PARALLEL (opt-in arm, never measured) is gone; the default is kept; code at 853bb6d0ee; restore: experiments/removed/MOJOLEARN_MSEL_UNITS_GPU_PARALLEL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NEURAL_TABLE_ROUTE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_NEURAL_TABLE_ROUTE_OFF"](), "removed 2026-11-15 (lane/expdel2-02-base): NEURAL_TABLE_ROUTE_OFF (old path, never measured) is gone; the default is kept; code at 0ace134aa9; restore: experiments/removed/MOJOLEARN_NEURAL_TABLE_ROUTE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_MOE_DEVICE_IO_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SEQ_MOE_DEVICE_IO_OFF"](), "removed 2026-11-15 (lane/expdel2-02-base): SEQ_MOE_DEVICE_IO_OFF (old path, never measured) is gone; the default is kept; code at 193542c352; restore: experiments/removed/MOJOLEARN_SEQ_MOE_DEVICE_IO_OFF.patch"
    # Lane expdel2-09-base (2026-11-15): unmeasured arms deleted, docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_AFCL_P04.patch
    comptime assert not is_defined["MOJOLEARN_AFCL_P04"](), "removed 2026-11-15 (lane/expdel2-09-base): AFCL_P04 (never-run Apple FAST programme card) is gone; the default is kept; code at 75ea427c34; restore: experiments/removed/MOJOLEARN_AFCL_P04.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AFCL_P09.patch
    comptime assert not is_defined["MOJOLEARN_AFCL_P09"](), "removed 2026-11-15 (lane/expdel2-09-base): AFCL_P09 (never-run Apple FAST programme card) is gone; the default is kept; code at 74cbc6951a; restore: experiments/removed/MOJOLEARN_AFCL_P09.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NB_TEXT_CSR_OFF.patch
    comptime assert not is_defined["MOJOLEARN_NB_TEXT_CSR_OFF"](), "removed 2026-11-15 (lane/expdel2-09-base): NB_TEXT_CSR_OFF (old path (rule d)) is gone; the default is kept; code at a6a4828242; restore: experiments/removed/MOJOLEARN_NB_TEXT_CSR_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NB_CSR_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NB_CSR_OFF"](), "removed 2026-11-15 (lane/expdel2-09-base): IDN_NB_CSR_OFF (old path, never measured) is gone; the default is kept; code at 2aca6c096a; restore: experiments/removed/MOJOLEARN_IDN_NB_CSR_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NB_ONEPASS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NB_ONEPASS_OFF"](), "removed 2026-11-15 (lane/expdel2-09-base): IDN_NB_ONEPASS_OFF (old path, never measured) is gone; the default is kept; code at 2909d00ab4; restore: experiments/removed/MOJOLEARN_IDN_NB_ONEPASS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_CLASS_ONEPASS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_CLASS_ONEPASS_OFF"](), "removed 2026-11-15 (lane/expdel2-09-base): IDN_CLASS_ONEPASS_OFF (old path, never measured) is gone; the default is kept; code at a7d5d989b5; restore: experiments/removed/MOJOLEARN_IDN_CLASS_ONEPASS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NB_CSR_DENSE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NB_CSR_DENSE_OFF"](), "removed 2026-11-15 (lane/expdel2-09-base): IDN_NB_CSR_DENSE_OFF (old path, never measured) is gone; the default is kept; code at 93d56e9d87; restore: experiments/removed/MOJOLEARN_IDN_NB_CSR_DENSE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GRAM_BLOCKED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GRAM_BLOCKED_OFF"](), "removed 2026-11-15 (lane/expdel2-09-base): IDN_GRAM_BLOCKED_OFF (old path, never measured) is gone; the default is kept; code at 2b7e58bca5; restore: experiments/removed/MOJOLEARN_IDN_GRAM_BLOCKED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GRAM_ROWTILE.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GRAM_ROWTILE"](), "removed 2026-11-15 (lane/expdel2-09-base): IDN_GRAM_ROWTILE (opt-in arm, never measured) is gone; the default is kept; code at eb6ccbf3bc; restore: experiments/removed/MOJOLEARN_IDN_GRAM_ROWTILE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GRAM_ROWS_512.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GRAM_ROWS_512"](), "removed 2026-11-15 (lane/expdel2-09-base): IDN_GRAM_ROWS_512 (opt-in arm, never measured) is gone; the default is kept; code at 00a8ccee0a; restore: experiments/removed/MOJOLEARN_IDN_GRAM_ROWS_512.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GRAM_ROWS_8192.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GRAM_ROWS_8192"](), "removed 2026-11-15 (lane/expdel2-09-base): IDN_GRAM_ROWS_8192 (opt-in arm, never measured) is gone; the default is kept; code at 46e0b201cf; restore: experiments/removed/MOJOLEARN_IDN_GRAM_ROWS_8192.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_PT_BLOCKED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_PT_BLOCKED_OFF"](), "removed 2026-11-15 (lane/expdel2-09-base): IDN_PT_BLOCKED_OFF (old path, never measured) is gone; the default is kept; code at b358d19039; restore: experiments/removed/MOJOLEARN_IDN_PT_BLOCKED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SELECT_BLOCKED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SELECT_BLOCKED_OFF"](), "removed 2026-11-15 (lane/expdel2-09-base): IDN_SELECT_BLOCKED_OFF (old path, never measured) is gone; the default is kept; code at 27b4b5a66a; restore: experiments/removed/MOJOLEARN_IDN_SELECT_BLOCKED_OFF.patch"
    return True


comptime BASE_GUARDS_OK = _check_configuration()
