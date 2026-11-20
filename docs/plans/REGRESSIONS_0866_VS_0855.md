# Regressions of 0.8.66 against 0.8.55 on the paper's recipes (AMD MI300X): diagnosis by code reading

Scope. Our arm only, same explicit parameters, same GPU (AMD Instinct MI300X VF, `hw_diff false`), 0.8.55 numbers from the
mlsys boards, 0.8.66 numbers from `bench/results/paper_compare/REGRESSIONS-amd.txt` and `compare-amd.json` (main 22c3d09fef).
42 of 82 timed recipes regressed by more than 1.10x. This note is a read-only diagnosis made from code and git history at
0.8.55 = `4e927c92495` (tag `alpha-api-0.8.55-amd-20261020`) and 0.8.66 = `5504d877ac` (tag `alpha-api-0.8.66-amd-20261108`).
Nothing was run, compiled or measured. File:line references are on 0.8.66 unless they say otherwise.

The bench drivers are not the cause. `tools/classical_two_datasets.py` and `tools/bench_board_more.py` changed only in prep
options and probe metadata. `bench/speed/forest_speed_arm.py` did not change. The timed `call()` bodies in
`tools/bench_board_neural.py` are unchanged (its diff is in the opponent arms). Each round builds a fresh estimator and
fits or predicts on the same NumPy float32 C-order arrays. The boxes match too: same image class, ROCm 6.4, Python 3.12.3,
and the same Mojo/MAX pins.

## 1. The shared cause: a serial byte-wise host pass over every element of X at the public door (C1)

`classical_check_x` (`bindings/classical_validate.mojo:75` `_cast_x`) calls `classical_host_f32`
(`bindings/classical_arrays.mojo:288-324`), which calls `classical_cast_rows` (`core/classical_arrays_impl.mojo:7-21`).
That function loops over rows and then columns on one thread:
- it rebuilds every element from its bytes (`for k in range(item)`);
- it calls `classical_f32_bits` on it;
- it tests the exponent for NaN/inf.

The loop runs even when X is already float32 C-order (`borrowed`); it only skips the store. 0.8.55's Python
`_buffer.as_f32_c` (`python/mojolearn/_buffer.py:775`) borrowed such an array with no per-element work. Where 0.8.55
checked finiteness at all, it did so on the device (`_buffer.all_finite`, `core/input_device.mojo`) or with a parallel
native scan.

Where it came from. The loop came in with `9b4fdce42d` (2026-10-20, "Add native Ridge owner"), part of the native-call
migration. It moved to `core/` in `544341e94d` and `f7f1aaa605`. Then each family's public door was moved onto it:
- OLS in `ed0fb49814` (10-20);
- TruncatedSVD in `f2179aac95`;
- KMeans in `b1ef5df836` (10-22, via `family_f32` and `family_cast_f32`, `bindings/family_common.mojo:201`);
- forests in `6e1da0b373` (10-23, `bindings/trees_public_lifecycle_native.mojo:86`);
- IsolationForest in `115b5eb8bb` (10-28, `bindings/iforest_public_lifecycle.mojo:125`);
- HDBSCAN in `7891a4845a` (`providers/python/hdbscan_public_lifecycle_pyapi.mojo:374`).

The slope proves it. Dividing each measured delta by the number of X elements in that call gives one constant, about
13 ns per element (about 75M elements/s), with no fixed part, across five families, two datasets and both fit and predict.

Shapes (driver constants):
- istella train is 2,043,304 x 220 = 449.5M elements;
- the istella query or test split is 500,000 x 220 = 110M;
- taxi fit is 4,000,000 x 11 = 44M;
- the TSVD lane is 1M x 220 on istella and 1M x 11 on taxi;
- HDBSCAN is 100,000 x 220.

| cell | delta ms | elements | ns/element |
|---|---|---|---|
| ols istella infer | 1402 | 110M | 12.7 |
| ols istella fit | 5997 | 449.5M | 13.3 |
| ols taxi fit | 624 | 44M | 14.2 |
| ols taxi infer | 70.5 | 5.5M | 12.8 |
| kmeans istella fit | 5892 | 449.5M | 13.1 |
| tsvd istella fit | 2845 | 220M | 12.9 |
| tsvd taxi fit | 143 | 11M | 13.0 |
| iforest istella infer (test) | 1468 | 110M | 13.4 |
| iforest istella infer (large, 1M rows) | 2894 | 220M | 13.2 |
| iforest taxi infer | 121 | ~8M | ~15 |
| rf istella infer | 1545 | 110M | 14 |
| rf istella fit (45.4 s -> 51.2 s) | 5786 | 449.5M | 12.9 |
| hdbscan istella fit | 281 | 22M | 12.8 |

No GPU route changed for these lanes:
- the OLS fused TSQR entry `x_decomp_ols_tsqr_r` is still selected (`x_decomp_idn_flags` bit 0 in identical builds,
  `x_decomp/api.mojo:264`), the same gate 0.8.55 used at `linear_model.py:545`;
- `predict_public` is one device call (`ols_predict_wide_binding`, `bindings/_mojolearn_estimators.mojo:968`);
- the forest and iforest predicts stay device-resident (`trees_resident_find` and
  `forest_predict_resident_reuse_gpu`, `bindings/trees_public_forest.mojo:1320`; `bindings/trees_public_iforest.mojo:218-228`);
- TSVD still runs `tsvd_fit_explained`.

Plan rows N007/N036/N038/N044 (host rf rebuilds trees) and N385/N390/N392/N411 (iforest refit per `fast` score) are on
the CPU column and the `fast` route. They do not apply to the identical GPU cells measured here.

The fix (F1, effort S, one lane, one file pair, no bit change). In `classical_host_f32` / `classical_cast_rows`:
- When `borrowed` (float32 C-order), do no cast. Run a finiteness test on the exponent bits (`0x7F800000`) that is
  vectorized with SIMD words and split into row tasks over every host core. Use `host_parallelize` with the grain floor
  of `core/host_predict_threads.mojo`, as `family_cast_rows_finite` already does.
- Better still, where the binding already scans on the device after the upload it makes anyway (`iforest_device_scan`,
  0.8.55's `all_finite`), skip the host scan and read back one status word.
- When a cast is needed, use the existing parallel `family_cast_rows_finite` (`core/family_cast_impl.mojo:27`) and a typed
  load per `kind` instead of the byte assembly.
- Keep the refusal wording.

Every caller of `classical_check_x`, `classical_host_f32`, `family_f32` and `family_cast_f32` gains at once. The call list
has about 30 bindings (`git grep -l classical_host_f32`): Ridge, logistic, QN, PLS, LDA, NMF, ICA, FA, KPCA, IPCA, random
projection, the prep and select natives, the metrics, and more. This one fix should return 20 of the 42 regressed cells to
within a few tens of ms of 0.8.55.

## 2. Secondary shared causes

C2. A single-thread host finiteness scan where 0.8.55 had none or ran it on the device. It costs about 0.54 ns per element.
- PCA uses `family_all_finite_f32` (`core/family_cast_impl.mojo:109-116`, scalar and serial), reached through
  `pca_input_snapshot` (`providers/python/pca_input_pyapi.mojo:86`) from fit (`bindings/pca_fit_native.mojo:67`) and
  transform (`bindings/pca_query_native.mojo:29`). It came in with `5a66fb200a` (10-27).
- KMeans predict uses `p3nc_all_finite_f32` (`core/p3nc_scan.mojo:20`, 8-wide, one thread) at `bindings/kmeans_native.mojo:411`.

Measured: PCA istella fit +244 ms over 449.5M elements, PCA taxi fit +24 ms over 44M, PCA istella infer +58.7 ms over
110M, KMeans istella infer +25.7 ms over 110M, KMeans taxi infer +1.4 ms over 5.5M.

Fix F2 (S): make both scans SIMD and split them across all cores, or do the scan on the device after the upload. This can
be folded into F1 as one helper (lesson 8: one helper per concern).

C3. A zero-filled allocation followed by a full copy where 0.8.55 borrowed or used uninitialized storage.
- kNN fit: `bindings/neighbors_knn_native.mojo:633-634` (`_nn_fit`) takes `family_f32_snapshot` and then
  `family_owned_snapshot`. The caller's float32 block is not "owned writable", so
  `bindings/family_common.mojo:346` allocates through `family_array` and `native_array_zeros`
  (`bindings/native_array_storage.mojo:42`, `PySequence_Repeat` of a one-item `array.array`) and copies X (176 MB on
  istella). 0.8.55 borrowed it (`neighbors.py:850-855`). It came in with `d4b347c981` (10-27). The lane's "fit" clock
  covers fit plus a 4,000-query predict; `separate_inference` is off. The resident search is unchanged: `classical/knn`,
  which times the same search, is flat at 1.02x.
- IsolationForest fit: `bindings/trees_public_iforest.mojo:328-340` zero-fills a `trees_empty` (`array('f',[0]) * count`)
  and copies the 1.8 GB X into it. That is about 0.4-0.8 s on top of C1. The copy has to stay: it protects against a
  caller editing X after fit. The zero fill does not.
- GBDT ranker fit: on AMD identical, `gbdt_fit_row_major_available` is 0, so X is transposed to F-order through `trees_f32`
  (`bindings/trees_public_inputs.mojo:451`, `bindings/trees_public_gbdt_fit.mojo:808-811`). The destination comes from
  `trees_empty` (`bindings/trees_public_inputs.mojo:139-151`), which zero-fills 1.8 GB before `transpose_f32` overwrites it.
  0.8.55 used an uninitialized `PyMem_RawMalloc` store (`_buffer.py:738`). It came in with `480176d769`. Measured: +405 ms,
  about 0.9 ns per element, which is one extra memory pass.

Fix F3 (S):
- add one uninitialized native output store for destinations that are fully written before they escape (conversion,
  transpose, owned training copy);
- make `native_array_zeros` one bulk allocation and memset instead of an item repeat;
- let kNN hold the caller's float32 C array as 0.8.55 did, since the resident key is already address plus shape. Where a
  snapshot is required, copy it in parallel into uninitialized storage.

C4. The per-call Mamba3/Samba native door (neural). The cost is about +1.6 ms per Mamba3 layer call, fixed and independent
of sequence length. It shows as:
- mamba3-decode +1.64 ms per token over 2,048 tokens;
- samba-decode +4.2 ms per token (two Mamba3 layers plus attention);
- samba-forward +3.3 ms;
- mamba3-forward +1.2 ms;
- the warm-up as well (3456 -> 6684 ms), so it is not noise.

The path goes `bindings/mamba_public_native.mojo:1122` `mamba_native_call("step_public"|"forward_public")`, then
`mamba3_forward` (:333), then `neural_block_snapshot_prepared` (`bindings/neural_block_public.mojo:1031`) and
`_neural_input` (:467), then `_m3_state_call` (`bindings/samba_inference_native.mojo:105`). It came in with `c5e8e0ad83`
(Mamba3Block as one native call) and `3d465a620e` (Samba from native layer descriptors). 0.8.55's `_mamba_impl.py:1189`
handed the arrays straight to `ext.mamba3_decode_step`.

The device code is identical. `mamba3.mojo`, `mamba1.mojo`, `_m3_load_weights` and `_mamba3_run` are unchanged: 11 uploads,
15 downloads and 1 sync per call in both versions. Weights were uploaded per call in 0.8.55 too. What the reading
accounts for comes to about 0.1-0.3 ms:
- metadata snapshots over 9 weights and 10 state buffers;
- five zero-filled `generation_array` outputs per call;
- two dict copies.

So the exact line is not settled. Something in the door is rebuilt per call. Confidence is medium that the door is the
cause and low on the line.

Fix F4 (M):
- cache the per-block plan, as `NativeBlockPlan` does for `neural_block_forward`;
- then route Mamba3 step through the resident decode session (`_decode_session`), as transformer-decode does, which also
  removes the per-call weight upload.

RUN OWED: one mamba3-decode with the existing `-D MOJOLEARN_MAMBA3_PHASE_TIMERS` plus a timer around `mamba3_forward`, to
split host door time from device time before the fix.

C5. A per-call copy of NumPy `x` in neural block forwards. `bindings/neural_block_admission.mojo:404-412` `neural_block_f32`
puts `x` (1x2048x384 float32, 3.1 MB) into a fresh zero-filled `generation_array` and copies it. 0.8.55's
`_f32_strict` / `as_f32_c` borrowed it. The cost is about 0.5-1.0 ms per call. It explains part of mamba1-forward +1.5 ms
and mamba2-forward +1.4 ms (via `causal_layer_run_fields`, `bindings/causal_lm_public.mojo:764`). It is hidden in
transformer-forward by that lane's resident session gain.

Fix F5 (S): borrow a C-contiguous float32 exporter, pinned and read-only for the call, as `lin_admitted_c` /
`lin_borrow_array` already do. It shares the uninitialized-store helper of F3.

C6. The identical GLM gradient moved to tile order (`4b8a8e8f2b`, 10-21, "GLM gradient: X^T dZ in the tile order",
retiring `QN_TILED_MULTI_OFF`). The `C == 1` `linear_bwd` changed from one block per gradient cell to `qntm_tiled_into`,
chunked by `QNTM_MAX_WS`. Partial and fold launches per iteration grow with width:
- slower when wide: linearsvr istella 2x (221 cells);
- faster when narrow: linearsvc taxi 0.29x (11 features, where the old form ran about 12 blocks).

LinearSVR's X goes through `family_f32_snapshot` with no element loop (`bindings/svc_linear_native.mojo:278`), so it is not
C1. The bits changed, so `n_iter_` may differ as well. Quality is equal (r2 0.8998, accuracy 0.76333). Also in range:
`b9b5180816` (shards the tile-order gradient) and `584cd1b8c0`. Confidence is low to medium.

Fix F6 (M): an NVIDIA+AMD A/B with `experiments/removed/MOJOLEARN_QN_TILED_MULTI_OFF.patch` as the B arm, recording
`n_iter_`. If the cost is the chunking, size the workspace bound from device memory so a wide `C == 1` gradient runs as one
tiled pass per iteration. Do not restore the serial form, since narrow shapes gain 3.4x.

Unexplained, small. gemm-bf16 1.18x, gemm-int8 1.17x, lm-forward 1.14x and transformer-decode 1.10x have no code cause
found:
- the bf16 and int8 paths are code-equivalent end to end;
- f32 `gemm` on the same door is SAME (1.03);
- lm-train-step is SAME.

Every cell is a single timed run (one run per arm), so these are within reach of noise. The candidate is the descriptor
provider route in `lin_call` (`bindings/linalg_ops_lowbit.mojo:47`, `82395c6572`). RUN OWED: one repeat of each lane on
the old and new wheels before code work.

## 3. The faster lanes: the same mechanism in reverse

The faster lanes are the ones where a device-resident state or a single native session replaced per-call host work.
- svc infer 0.18x: 0.8.55's `_svm_impl.py:558-563` brought the fitted model back to the host and uploaded it again on
  every predict. 0.8.66 keeps it on the device (`_SVC_RESIDENT_SLOT`, `fitted_resident_lease`, `svc_predict_resident`,
  `bindings/svc_kernel_native.mojo:127-130` and `:554-594`; `b54d465e63`, lane ne1-a-gp-svm). The svc query X (2.2M
  elements) does not pass the C1 loop.
- mlp-train-step 0.29x: `SmallMLPTrainer` became one native call per method with resident device state (`26dcfdd6ad`,
  `87806f4cba`, `b2eac07b39`, `a0316efb9c`). The per-step Python, uploads and readback are gone.
- transformer-forward 0.72x: `_neural_plain` (`bindings/neural_block_public.mojo:647-720`) keeps a resident
  `forward_session`, so about 7 MB of weights are no longer uploaded per call. That more than covers C5.
- program-run 0.66x and sequential-predict 0.64x: one native call each, the same pattern. Not traced line by line.
- linearsvc taxi 0.29x: C6, narrow shape.

The lesson: the native-call migration (0.8.57-0.8.62) moved input admission from NumPy (vectorized C, zero-copy borrows)
into hand-written serial Mojo loops and zero-filled copies. Where a lane also gained a resident state, the gain
outweighed the admission cost. Where it did not, the admission cost dominates, and on the large paper shapes it dominates
by 10-80x. The "Python out of the hot paths" plan removed Python but left these loops on the host and serial. The CLAUDE.md
rules already forbid that ("write code to run in parallel"; "a new serial loop over independent units is a defect"). The
`# native input marshaling` tags on these loops (about 60 in `core/`, none at 0.8.55) are the list to audit:
- `core/prep_boundary_impl.mojo` (14)
- `core/xseq_marshal.mojo` (11)
- `core/family_cast_impl.mojo` (9)
- `core/family_weights_impl.mojo` (6)
- `core/neighbors_ann_impl.mojo` (5)
- `core/mcls_impl.mojo` (5)
- `core/classical_arrays_impl.mojo` (3)
- `core/pca_exact_impl.mojo` (2)
- and others

## 4. Table

| lane (cells) | ratio | cause | shared? | fix | effort |
|---|---|---|---|---|---|
| trees/iforest infer (istella test, large; taxi x2) | 82.8x, 82.4x, 42.7x, 29.9x | C1: serial byte-wise cast/finite pass at the door, `iforest_public_lifecycle.mojo:125` -> `core/classical_arrays_impl.mojo:7` | yes, C1 | F1 | S |
| classical/ols infer (istella, taxi) | 69.4x, 20.9x | C1 via `classical_ridge.mojo:224` | yes, C1 | F1 | S |
| classical2/tsvd fit (istella, taxi) | 60.8x, 31.5x | C1 via `tsvd_native.mojo:313` -> `decomp_inputs.mojo:41` | yes, C1 | F1 | S |
| classical/ols fit (taxi, istella) | 48.4x, 14.4x | C1 via `classical_ols.mojo:69` (TSQR route unchanged) | yes, C1 | F1 | S |
| trees/iforest fit (istella, taxi) | 39.6x, 6.5x | C1, plus C3 zero-fill and copy `trees_public_iforest.mojo:328-340` | yes, C1 and C3 | F1 + F3 | S |
| trees/rf infer (istella x2, taxi x2) | 29.2x, 26.4x, 11.2x, 11.2x | C1 via `trees_public_lifecycle_native.mojo:86`; resident predict unchanged | yes, C1 | F1 | S |
| classical/kmeans fit (istella, taxi) | 15.5x, 4.6x | C1 via `kmeans_native.mojo:279` -> `family_common.mojo:201` (taxi has ~126 ms more, not traced) | yes, C1 | F1 | S |
| classical2/knn-reg, knn-clf fit (istella; taxi) | 8.5x, 8.2x; 2.2x, 2.0x | C3: zero-filled owned copy of X `neighbors_knn_native.mojo:633-634`, item-repeat alloc `native_array_storage.mojo:42` | partly (C3 alloc) | F3 | S |
| classical/pca fit, infer (istella, taxi; istella) | 3.8x, 3.3x; 2.9x | C2: serial `family_all_finite_f32` via `pca_input_pyapi.mojo:86` | yes, C2 | F2 | S |
| classical/kmeans infer (istella, taxi) | 2.4x, 1.7x | C2: one-thread `p3nc_all_finite_f32` `kmeans_native.mojo:411` | yes, C2 | F2 | S |
| neural/mamba3-decode, samba-decode | 2.17x, 1.57x | C4: per-call Mamba3 door ~1.6 ms per layer call `mamba_public_native.mojo:1122` (line not settled) | yes, C4 (all Mamba3/Samba) | F4 (plan cache, then resident session); phase-timer run owed | M |
| classical2/linearsvr fit (istella, taxi) | 1.98x, 1.16x | C6: identical GLM gradient tile order `4b8a8e8f2b` (low-med confidence) | yes, C6 (wide GLMs) | F6 A/B, size the workspace | M |
| neural/samba-, mamba1-, mamba3-, mamba2-forward | 1.55x, 1.51x, 1.35x, 1.27x | C4 (Mamba3/Samba) + C5 per-call copy of x `neural_block_admission.mojo:404-412` | yes, C4 and C5 | F4 + F5 | S-M |
| classical/hdbscan fit | 1.27x | C1 via `hdbscan_public_lifecycle_pyapi.mojo:374` | yes, C1 | F1 | S |
| trees/gbdt-rank-pairlogit fit | 1.24x | C3: zero-filled transpose destination `trees_public_inputs.mojo:139-151` | yes, C3 (all `trees_f32` copies) | F3 | S |
| neural/gemm-bf16, gemm-int8, lm-forward | 1.18x, 1.17x, 1.14x | not found; code-equivalent; n=1 noise suspected | ? | repeat run first | S |
| trees/rf fit istella | 1.13x | C1 (5.8 s of 51 s) | yes, C1 | F1 | S |
| neural/transformer-decode | 1.10x | session-step admission, small; possibly noise | ? | repeat run first | S |
| FASTER classical/svc infer | 0.18x | resident fitted state `svc_kernel_native.mojo:554-594` | reverse of C1-C4 | keep | - |
| FASTER classical2/linearsvc taxi fit | 0.29x | C6, narrow shape fills the GPU | C6 | keep, check in F6 A/B | - |
| FASTER neural/mlp-train-step; transformer-forward; program-run; sequential-predict | 0.29x; 0.72x; 0.66x; 0.64x | resident sessions and one native call | reverse | keep | - |

## 5. Order

1. F1 + F2 as one lane on `classical_arrays_impl.mojo`, `family_cast_impl.mojo` and `p3nc_scan.mojo`. It is one parallel
   SIMD finite scan and one parallel typed cast. It covers 25 of the 42 cells (20 by F1 and 5 by F2), including the twelve worst.
2. F3 in the same round, in `native_array_storage.mojo`, `trees_public_inputs.mojo`, `family_common.mojo` and
   `neighbors_knn_native.mojo`: an uninitialized store, bulk zeros, and the kNN borrow.
3. F5 with F3, since they share the store helper.
4. F4 after its phase-timer run.
5. F6 as an A/B.
6. The repeat runs for the unexplained small cells.

None of the fixes changes the bytes the kernels read, so identity is unchanged. Bindings to rebuild:
- F1/F2: every binding that imports `core/classical_arrays_impl.mojo`, `core/family_cast_impl.mojo` or
  `core/p3nc_scan.mojo` (a wide rebuild);
- F3: `_mojolearn_trees` and the forest and iforest groups, `neighbors_knn_native`, and the `native_array_storage` users;
- F4/F5: the mamba, samba and neural block bindings.
