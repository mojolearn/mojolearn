# IDENTICAL CUDA PTX: rounding pinned, approximate instructions audited (2026-09-25)

## The change

`packaging/linux/build_sets.sh` runs `packaging/linux/ptx_contract.py patch` over every
IDENTICAL-tier CUDA binary of a set before the manifest hashes it. The pass gives every plain
`mul/add/sub .f32/.f64` its `.rn` spelling in place (length-preserving), which ptxas and the
driver JIT never contract into FFMA. `packaging/portable_math/wheel.py` (the wheel audit,
`--audit-only` in CI) refuses any IDENTICAL CUDA PTX still carrying a float
mul/add/sub/fma/mad without a rounding modifier. On 0.8.19 that audit fails (46 binaries,
71,250 plain ops) and passes after the pass.

## Proof on 0.8.19 (the published wheel, patched on the Mac)

Wheel: 0.8.19 manylinux wheel with the pass applied, sha256
`2296577101b19ae26dfb88a37580a12468e427c31c139aa42243791616a83dea` (46 IDENTICAL
binaries, 71,250 rewrites; RECORD and LINUX_PAYLOAD.json updated).

Release column from the installed wheel (`tools/release_wheel_smoke.sh --column`, the 0.8.19
selection, 209 lanes, fixtures base/denormal/odd, one fit), diffed against the recorded
0.8.19 Apple, AMD and unpatched RTX 4090 columns:

| box | expanded smoke | cells vs Apple + AMD + 0.8.19 NVIDIA |
|---|---|---|
| RTX 4090 (sm_89), driver 580.159.04 | PASSED, 13 jobs | IDENTICAL=627, infer/model IDENTICAL=984 |
| H100 80GB HBM3 (sm_90a), driver 580.126.09 | PASSED, 13 jobs | IDENTICAL=627, infer/model IDENTICAL=984 |

RTX 4090 against H100 (both patched): IDENTICAL=627, none divergent
(`diff-4090-vs-h100-summary.txt`).

Per module (`boxcheck.py`, all 2,216 distinct IDENTICAL modules, ptxas 13.0 and the
driver JIT; `boxcheck-*.log`): ptxas `--fmad=true` on the patched PTX equals `--fmad=false` on
the original in 2,216 of 2,216. The driver JIT of the patched PTX equals that of the original
in every module but 92 (46 per arch, in `_mojolearn_gbdt`, `_estimators`, `_kernel_methods`,
`_linalg`); the change in those is 920 FFMA becoming FMUL+FADD on the H100 (460 per arch), the
`hist2_dither` product by 2^-24, which is exact, so the bits are unchanged, as the columns
show.

## Approximate instructions in IDENTICAL PTX (0.8.19)

| instruction | where | IDENTICAL result path? |
|---|---|---|
| `lg2.approx.ftz.f32` | `PointwisePartOffsetsHelper.data_partition_offset` (`gbdt/methods/kernel/split_properties_helpers.mojo`), 28 modules | yes: fold stripe `1 << ceil(log2(fold_count))` |
| `sqrt.approx.ftz.f32` | `sqrt_postprocess_kernel` (`neighbors/impl/detail/fused_l2_knn.mojo`) | no: AUTO never takes the fused arm under IDENTICAL (row 23); routed through `identical_sqrt` now anyway |
| `sqrt/ex2/cos/tanh.approx` | `*_sabotage*` check kernels (cholesky, gp, kernel_methods, hierarchy) | no: sabotage fixtures |

`lg2test.py` runs the fold-stripe expression exactly as the PTX spells it for n = 1..2^24 on
both GPUs (`lg2test-*.json`): exact for every n <= 2^20, and the same 18 misses on sm_89 and
sm_90 above it (n = 2^k + small, k >= 20, one low). Fold counts are grid_dim.z (tens at most),
so no shipped fit reaches the range; the same `sqrt.approx` is off by one ulp for 359,384,981
of 2,130,706,432 positive normals on both, which is why the fused arm now uses
`identical_sqrt`.
