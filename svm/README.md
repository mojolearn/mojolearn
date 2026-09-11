# SVM

Dense FP32 binary C-SVC and epsilon-SVR derived from cuML's SMO solver and cuVS kernel matrices. `NOT_IMPLEMENTED.tsv` defines unsupported
multiclass, sparse, kernel, and parameter combinations. Unsupported behavior must fail clearly
rather than silently selecting a different algorithm.

## Verify

```bash
pixi run check-svm
pixi run check-svm-oracle
```

Identity depends on working-set selection, tie handling, kernel evaluation, and reduction order.
Performance changes must preserve those choices in IDENTICAL mode.

## The SMO's schedule (DEVIATIONS 2491 and 2492, 2026-09-10)

Measured on the M4 at HIGGS 50,000 x 28 (156 outer iterations, 71,153
inner) with `MOJOLEARN_STAGE_TIMES=1`, which `svm/impl/smosolver.mojo` now
honors: the block solve was 2.73 s and the full kernel tile 3.58 s of a
7.53 s fit. Two changes, one per phase:

- **2491** `svm/checks/pinned_argreduce.mojo::block_argext`: the three
  arg-reductions of the block solve fold with `shuffle_xor` butterflies
  inside a warp and one threadgroup exchange across warps (two barriers
  instead of twelve), and return the winning thread beside the pair so the
  ballot that recovered it is gone. Same total order on (value, key) with
  unique keys, so the same element wins under any topology or lane width;
  IDENTICAL takes it too. Block solve 2.73 s to 0.55 s.
- **2492** `svm/impl/distance/kernel_matrices.mojo::rbf_fused_tile`: FAST
  only. The RBF tile is one kernel (a row of `b` per thread in registers,
  rows of `a` through shared memory, `exp(-gamma(na + nb - 2 dot))`
  written once) instead of a k = n_features GEMM plus an epilogue. Register
  rows to 64 features; wider inputs keep the GEMM path. IDENTICAL keeps
  `identical_gemm_into` and the pinned epilogue. Full tile 3.58 s to 0.39 s.

Fit 7.5 s to 2.2 s under the stage clock; `svc_main.mojo` 44/44 in FAST
and, after the trace fix below, 44/44 in IDENTICAL.

**2623** (2026-09-11) `checks/kernel_matrix.mojo::svm_block_solve_warp_folds_for`:
2491 made `smo_block_solve_kernel[1024]` unlaunchable on CUDA. An NVIDIA
H100 80GB HBM3 refuses it with CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, so every
SVC fit with more than 512 training rows (`n_ws = min(1024, n_train)`)
failed there from 2491 through 0.8.2, and `svc_main.mojo` failed seven
IDENTICAL gates there, while [512], the MI300X and the M4 launch it. The
pre-2491 kernel fits at 600 and 2,000 rows on the same GPU. The compiled
warp kernel reports 61 registers and 5,256 threadgroup bytes, so the refused
resource is not one the function reports, and neither folding `f_u` and
`f_max` into one exchange nor reading the diagonal from the kernel buffer
instead of a WSIZE threadgroup array made it launch. The row keeps the warp
folds everywhere except NVIDIA above width 512, which takes the halving trees
with a one-slot thread ballot again. Both schedules select the same element,
so the bits are the same on every column.

**2627 and 2628** (2026-09-11, lane/svm-speed, measured, NOT flipped)
`checks/kernel_matrix.mojo::svm_block_solve_schedule_for` names four
schedules for the three reductions, selectable by `-D`, all selecting the
same element: TREE (0), WARP (1, 2491), WARP_LANE0 (2, 2627:
`block_argext_lane0`, the butterflies with the cross-warp fold on lane 0 in
a runtime loop and a warp broadcast) and FUSED_TREE (3, 2628:
`pinned_block_argmin_argmax_tid` carries the argmin, its thread and the
argmax in one halving tree, and `pinned_block_argext_tid` carries `l`'s
thread, so both ballots are gone). The default is unchanged (2623's row).
On the H100 (RunPod pod xss2n2qzc1ed8q, 36ca51fd plus this change), every
schedule gives the same SVC fits at n=400/600/2000 (457e29b82bca9df9,
733a383c5699f427, 2b66bc991a9c9ed0) and on the two benchmark blocks
(taxi b0f91a7958162936, Istella-S 5c19df95159208ef). One staged fit each,
block solve (taxi 109 outer and 172,540 inner iterations; Istella-S 13 and
4,664): TREE 783 ms and 21.5 ms, FUSED_TREE 690 ms and 18.9 ms, WARP_LANE0
1,700 ms and 46 ms (1,675 ms without its trailing barrier). So the
comptime-unrolled cross-warp fold is what CUDA refused at width 1024 (the
runtime loop launches), but warp shuffles cost more than barriers on this
GPU, and barriers are only about a fifth of the tree's 4.5 us per inner
iteration. cuML on the same pod: taxi 110 outer, 166,517 inner, 420 ms for
the whole fit (393 ms at cache_size 0, so its kernel cache is not its
advantage here); Istella-S 12 and 4,503, 20.4 ms. Istella-S also spends
about 36 ms outside the solver's stage clock. Evidence
`~/mojolearn-evidence/svm-speed-2026-09-11/`.
On the Apple M4 (orchestrator gate, 2026-09-11) the default schedule passes
`svm/svc_main.mojo` 44/44 IDENTICAL, but `-D MOJOLEARN_SVM_SCHED_FUSED_TREE`
fails seven gates because Metal refuses the width-1024 pipeline ("Threadgroup
memory size (36872) exceeds the maximum threadgroup memory allowed (32768)").
FUSED_TREE is therefore an NVIDIA-only candidate; a flip must route it through
the kernel-matrix row for NVIDIA alone, not the global define. The R-ary
thread-carrying tree (lane commit 48f92b19) was never built and is not merged.

`svr_device_matches_oracle` had failed under IDENTICAL since the SVR path
landed ("ws sequence differs at outer iteration 0", every SVR fixture; FAST
only reported it). The solver's trace recorded the working set PROJECTED
into `[0, n_rows)` (`ws_idx_mod`, the buffer that addresses rows of X)
while the oracle keeps the raw 2n-space indices; b, the dual coefficients
and every other stage were already bit-equal. The trace now records the
raw indices (`ws_idx_mod_svr`; equal to the projected ones for C_SVC), and
the six SVR fixtures compare IDENTICAL. Numbers against scikit-learn:
`bench/results/svm_fast_2026-09-10/`. The next phase by size is
`select_ws` (a 32-pass one-bit radix sort, about 130 launches per outer
iteration, 0.58 s of the 2.2 s).

**2493** (same day) `svm/impl/svc_impl.mojo::svc_fused_decision`, FAST
only: predict folds `sum_j dual_j K(x_i, sv_j)` inside the kernel
evaluation, one thread per query row, support vectors streamed through
shared memory; no `[batch x n_support]` tile, no batch loop, no
stride-`n_support` reads. HIGGS 50k model (39,349 SVs), 10,000 predictions
121 ms to 40 ms, 100,000 predictions 783 ms to 312 ms, decision values
bit-equal to the tiled FAST path. SVR predict shares it. Taken only when no
identity card is recording; IDENTICAL keeps the tiled path.
