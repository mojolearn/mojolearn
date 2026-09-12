# Brief: AMD gfx942 confirmations for the 2026-09-11/12 trees, classical and Jacobi flips

Ten speed flips merged on 2026-09-11/12, plus the GBDT pair 2634/2635 and
DEVIATION 2680, were measured and bit-verified on an NVIDIA H100 ONLY. The
library's headline claim is bitwise identity across Apple Metal, NVIDIA CUDA
and AMD HIP, so until a gfx942 build had COMPILED and RUN those branches the
claim was unclosed for two days of merges. This leg closes it for identity and
for the shipped build. It measures NO speed, by design.

## The box, and the provenance

Hot Aisle AMD Instinct MI300X, one 13core VM, `rocm/dev-ubuntu-22.04:6.4.1-complete`,
`box_gfx=gfx942` read back from the box. `tools/pick_box.sh --need amd` printed
`hotaisle`, which is first in the ENGINEERING_RULES 10 box order.

- commit `5696ff68` (`leg.txt`), `source_sha256_match=yes`
- body `extra_sha256=5ae140435564bfbbaa06f10b679360919254a04fe68e53b3365961174aa82254`
- VM `b9edb560-eb3b-4633-abd3-b54dcd429f42` (`enc1-gpuvm016`), lease used 1,649 s
- **VERIFIED GONE**: DELETE HTTP 204, then `get=404 list=200 listed=no`,
  `destroy_confirmed=1`, slot released 20:14:02Z
- evidence `bench/results/e1g/2026-09-12_194618-amd-mi300x-hotaisle-amd-confirmations`

## 1. Identity, the first question

### Cross-vendor, at the shipped default: THREE VENDORS AGREE

`tools/identity_break.py` on the five forest lanes and nine hostile fixtures,
then diffed on the Mac against the Apple M4 baseline
(`bench/results/forest_finish_2026-09-11/logs/ib_2663_apple_m4.json`) and the
H100 `dflt` set from `forest-finish`:

    summary: IDENTICAL=45

45 of 45 cells equal across Apple M4, NVIDIA H100 and AMD gfx942 — every
rf-clf, rf-reg, et-clf, et-reg and iforest fixture, the `denormal` and
`denormal_ftz` pairs included. On the box itself: `cells=45 stable=45 moved=0
refused=0`. The wider run (kmeans, pca, ols, knn, knn-clf, knn-reg, kde, svc
and four gbdt lanes) read `cells=108 stable=108 moved=0 refused=0`; those are
WITNESSES, not a diff, because no other vendor has a baseline at this default.

### Before and after on this box, where a switch still exists

Each arm is gated on its `.so` sha256 DIFFERING from the shipped one. An equal
hash means the define selected nothing, and that is reported as an inert
comparison rather than as "bits unchanged".

| deviation | arm | shipped .so | arm .so | reach | verdict |
|---|---|---|---|---|---|
| 2663 ET batch width 16384 -> 4096 | `-D MOJOLEARN_ET_DEVICE_BATCH_4096=1` | `a763f4a3ba15557e` | `c9c91af89157fe79` | proven | `IDENTICAL=18`, bits unchanged |
| 2623/2666 SVM schedule -> halving trees | `-D MOJOLEARN_SVM_TREE_FOLDS=1` | `604ea5fe7862e81d` | `0e8082ae8f386d12` | proven | `IDENTICAL=18` (svc + iforest), bits unchanged |
| 2634 + 2635 GBDT CTR prep and bound walks | `-D MOJOLEARN_2634_CTR_PREP_OFF=1 -D MOJOLEARN_2635_LINEAR_BOUNDS=1` | `6d897a417669b4f0` | `67a253bbddee5da2` | proven | `IDENTICAL=36`, bits unchanged |
| 2631 kNN query tile | `-D MOJOLEARN_KNN_LEGACY_QUERY_TILE=1 -D MOJOLEARN_KNN_IDENTICAL_FULL_RADIX_SCRATCH=1` | `f04216f187895782` | `f04216f187895782` | **FAILED** | INERT, see finding 2 |

## 2. The shipped-build gate

Every build is a SHIPPED build: `MOJOLEARN_NUMERIC_MODE=identical`, no trial
define, no EVERY_COLUMN knob, so the AMD column's own routing rows had to
resolve the flipped defaults unaided. Six bindings carry the flipped lanes and
all six built clean on gfx942 (`bindings.sha256` in `gate.txt`); eleven more
built afterwards for completeness. Then the named gates, all under IDENTICAL:

| gate | result |
|---|---|
| `svm/svc_main.mojo` | **44/44 gates passed** |
| `extratrees/checks/device_batched_check.mojo` | **PASS, 45 cells**; sabotages moved 2688/2052 and 2716/2676 nodes |
| `isolation_forest/checks/if_check.mojo` | OK; 6 fixtures, 37,496 cells bit-equal to the oracle; 123 card stages |
| `tools/check_forest_resident_layouts.sh` | PASS, the 3 modes x 2 layouts matrix |
| `gbdt/.../sub_byte_layout_gate.mojo` | OK, four arms live, three sabotage arms failed as required |
| `kde/checks/kde_check.mojo` | OK (launch invariance, signed zero, cosine, Minkowski) |
| `kde/checks/kde_stage_profile.mojo` | one hash `17888536843391681998` for staged replay, device staged and device dispatch |
| `neighbors/` knn_identity, knn_main, query_batch | OK / OK / `QUERY BATCH PASS enabled False default_tile 256 cases 3` |
| SVC cross-vendor fit hashes | `457e29b82bca9df9`, `733a383c5699f427`, `2b66bc991a9c9ed0` — all three **MATCH** |
| `decomposition/checks/pca_check.mojo` | **exit 1, PRE-EXISTING, see finding 3** |

## 3. DEVIATION 2680 on AMD: the launch width is not the fold width

`check_jacobi_is_launch_invariant` on gfx942, fold width 32, the ladder running
64/128/256/512 threads over seven sizes:

    0 cells differing at EVERY rung and EVERY size,
    including 0 of 96,803 at n = 220

which is exactly the NVIDIA result. `check_jacobi_merged_phases_equal_four_phase`
(DEVIATION 2671's own gate) is also 0 differing at all seven sizes. The shipped
launch width reads 256, as it does on every column — `JACOBI_ROT_TPB` is 256
under IDENTICAL and is not vendor-dependent.

**Which widths gfx942 accepts.** All four rungs launched. NVIDIA's 1024 refusal
(`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES` on sm_90a) does not arise here, and it
never entered the gate on either vendor: the ladder tops out at 16x the fold
width, which is 512. So AMD accepts every width the gate asks for; no rung was
skipped or refused on this box.

One AMD-specific note from `check_jacobi_fold_shape`: the library fold was NOT
INSTANTIABLE at width 32 on this device (block size must exceed the lane width,
DEVIATION 528, AMD's lane width being 64), so that one comparison was not taken.
The device-vs-host halving-tree comparison it gates still passed.

## Findings

**1. Five flips can never be A/B'd, on any vendor.** DEVIATIONS 2620, 2621,
2622 (OLS), 2671 (Jacobi), 2672 (k-means) and KDE's 2625, 2626, 2660 have NO
compile-time switch, no env var and no alias: `grep -rn is_defined` returns 0
in `glm/`, 0 in `cluster/` and 0 in `kde/`. The pre-flip code was replaced, not
gated. Two kept their old code but unreachably: 2671's
`jacobi_eigh_kernel_four_phase` survives at
`decomposition/checks/jacobi_eigh_device.mojo:398` but is called only from
`jacobi_check.mojo:1854`, and 2625's staged path survives behind an internal
`staged_only` parameter (`kernel_density.mojo:2516`) that no caller in
`bindings/`, `python/` or `kde/estimator.mojo` ever sets. So a future
regression in any of the eight is diagnosable only by editing source or by git
archaeology: `df77d6c1` (2620, 2621, `lstsq.mojo` +267), `460f0045` / `26683dba`
(2622, `lstsq_min_norm.mojo` +118), `0c6c1249` (2671, 2672,
`cluster/estimator.mojo` +36 and `jacobi_eigh_device.mojo` +166). For those
eight, cross-vendor identity at the shipped default is the only question that
can be asked, and section 1 answers it.

**2. DEVIATION 2631 does not reach the AMD column at all.** The inert A/B is
not a harness error and not a broken define. `knn_query_tile_for` returns
`KNN_IDENTICAL_WIDE_QUERY_TILE if column == COLUMN_NVIDIA else 0` and
`knn_radix_scratch_shrink_for` returns `column == COLUMN_NVIDIA`
(`checks/kernel_matrix.mojo:1281,1297`), and `QUERY_TILE_512_CANDIDATE`
requires `TARGET_COLUMN == COLUMN_NVIDIA` (`neighbors/estimator.mojo:233`). So
on gfx942 the tile is 256 with or without the define, and the two binaries are
byte-identical because they ARE the same program. The running binary says so
independently: `QUERY BATCH PASS enabled False default_tile 256`. The kNN
0.926 flip is an NVIDIA-column row; there is no AMD behavior to confirm and
none is owed. A real AMD tile A/B would use
`-D MOJOLEARN_KNN_QUERY_TILE_ARM_4096`, which forces a tile on every column;
`MOJOLEARN_KNN_LEGACY_QUERY_TILE` only suppresses a 512 candidate AMD never had.

**3. The one red check is pre-existing and is not this lane's.**
`pca_check` exits 1 on `the 129-column FALLBACK arm COMPLETED under IDENTICAL.
It cannot have run on the split-K kernel at that width, so it ran on the vendor
matmul and returned a model this mode promises is vendor-independent and is
not.` That is verbatim controls C/D of `bench/results/jacobi_speed_2026-09-12/README.md`,
which records the same text failing on the BEFORE and AFTER trees on NVIDIA and
says "this lane changes no Gram code at all". It is a Gram split-K dispatch
that falls through to `linalg.matmul` instead of REFUSING past capacity —
IDENTITY_PATHS row 27's business, now confirmed to fail on AMD as well as
NVIDIA. It is not a 2671/2680 regression: every eigensolver gate above is green
on this box.

**4. Two benign non-zero exits.** `bindings/build_byte_lm_host.sh` exits 2 with
`byte LM host: a CPU build takes no MOJOLEARN_GPU_ARCHS` — a refusal by design,
since the runner exports the arch for the GPU builds. And `gate.txt` says
`commit=unknown` because the body reads `/root/mojolearn/COMMIT`, which the Hot
Aisle runner does not write; this is the same defect recorded in the 2026-09-12
ship gate and provenance comes from `leg.txt` instead. A future body should read
the commit the way the other legs do.

## What is NOT closed

- **Speed on AMD: NOT MEASURED.** This leg is correctness only. Every
  after/before ratio for these flips remains NVIDIA-only.
- **2637/2638 have no before/after on any vendor here.** `identity_break`'s
  fixtures are `np.ascontiguousarray`, so they exercise the shipped rowmajor
  path; the pre-flip path is selected by Fortran-order input
  (`MOJOLEARN_SPEED_FORTRAN=1`), which this leg never produced. Their
  cross-vendor verdict is in section 1; their before/after is UNKNOWN.
- **2636 and 2661 were not exercised.** Both are opt-in, so the shipped default
  IS the old arm and there was nothing to confirm.
- The Gram split-K refusal of finding 3 stays owed to whoever owns
  IDENTITY_PATHS row 27.
