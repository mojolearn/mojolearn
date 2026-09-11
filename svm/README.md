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
the kernel-matrix row for NVIDIA alone, not the global define. DEVIATION 2666
below is that flip.

SVM_SCHED_RARY_TREE (4, 2628's second shape) IS BUILT NOW AND IT LOSES.
`pinned_block_argmin_argmax_tid_rary` and `pinned_block_argext_tid_rary` fold
the same selections on a threadgroup tree of arity R
(`svm_block_solve_tree_arity_for`, default 32), two levels at width 1024, so
about 10 barriers per inner iteration where FUSED_TREE has 26 and with no warp
shuffles. On the H200 (taxi, five fits each, median): arity 32 1,945.6 ms,
arity 32 without its trailing and second update barriers 1,937.3 ms, arity 16
1,324.4 ms, against FUSED_TREE 771.1 ms and TREE 866.9 ms. It trades barriers
for serial work: at arity 32 one thread in 32 folds 31 threadgroup slots in a
runtime loop while the other 31 idle, and two such levels cost more than the
binary tree's twelve cheap ones. Dropping the two barriers is worth 0.4
percent of that arm, which says again that barriers are not where this kernel
spends. The shape is kept behind its define and is not a candidate.

48f92b19 SHIPPED IT UNREACHABLE: it added the constant and
`svm_block_solve_tree_arity_for` but no branch for
`-D MOJOLEARN_SVM_SCHED_RARY_TREE` in `svm_block_solve_schedule_for`, so the
define selected nothing and the first three "R-ary" builds were the default
schedule. The row now has the branch, and each arm above is a different
`_mojolearn_svm.so` (sha256 596dafdcc12acbb4, 75e7c15e05b52863,
a73de7fbbaf6ce3a, 3ded2bfae86f14ab, 5c6ea682b62986f6), so each schedule was
really reached. All five give the same fits: n=400/600/2000
457e29b82bca9df9, 733a383c5699f427, 2b66bc991a9c9ed0, taxi b0f91a7958162936,
Istella-S 5c19df95159208ef.

**2666** (2026-09-11, lane/svm-finish, MEASURED AND FLIPPED)
`checks/kernel_matrix.mojo::svm_block_solve_schedule_for`: the column DEVIATION
2623 sends to the halving trees, NVIDIA above width 512, takes FUSED_TREE
instead. Nothing else moves: every other column keeps 2491's warp folds, and
`-D MOJOLEARN_SVM_TREE_FOLDS` still takes the pre-2491 trees everywhere for an
A/B. It stays a row rather than a global default because Metal refuses the
fused kernel's width-1024 pipeline (threadgroup memory 36872 > 32768).

**2665** (same lane, MEASURED AND FLIPPED) the fixed cost outside the solver,
which was about a third of an Istella-S fit. Three changes, no bits moved:
`bindings/_mojolearn_svm.mojo` hands `svc_fit_host_borrowed` the caller's NumPy
address instead of copying X into a host `List` (8.8 MB per Istella-S fit) and
then into a pinned buffer element by element; `svm_parameter.mojo::check_finite_ptr`
walks the borrowed cells once on the host pool, sixteen at a time by exponent
bits, and reports the same first flat index and the same message as
`check_finite_list`; and `KernelCache.__init__` runs its three scratch fills
only for the launch-invariance gate's padded or poisoned arms, which skips a
41 MB fill of the kernel tile at 10,000 rows and a 1,024 working set. The gate
is what licenses the last one: `check_device_is_launch_invariant` fills with
-7.25e20 and 3.0e-39 under paddings 37 and 1029 and requires the base arm,
which now runs unfilled, to match byte for byte.

Measured on an NVIDIA H200 (RunPod pod 4oih8bhjepzlmm, driver 570.211.01,
below Mojo's CUDA floor so `MODULAR_NVPTX_COMPILER_PATH` points at the pod's
ptxas 12.9.86, Modular's documented older-driver path). The race, 1 warm-up
plus 5 interleaved rounds against cuML 26.8.0 in one conductor, before being
origin/main 2c64a778 built on the same pod:

| dataset | cuML ms | before ms | after ms | after/before | ours/cuML after | accuracy |
|---|---|---|---|---|---|---|
| taxi 10,000 x 11 | 418.4 | 865.0 | 768.7 | 0.8886 | 1.84x | 0.7675 both |
| Istella-S 10,000 x 220 | 20.29 | 72.2 | 61.3 | 0.8498 | 3.02x | 0.9222 both |

Geometric mean 0.869, quality not worse on either dataset and the same number
of support vectors (5,527 and 2,400), so both flip (ENGINEERING_RULES.md
section 9). Which change buys what, five fits per cell on the same pod: taxi
866.3 before, 865.7 with 2665 alone, 772.2 with both; Istella-S 67.2, 60.9,
57.8. So 2666 is the taxi win and 2665 is most of the Istella-S win, which is
what the shapes predict, the taxi block being 11 columns wide and Istella-S
220. `svm/svc_main.mojo` passes 44/44 under IDENTICAL on the H200 with both,
launch invariance and the six SVR fixtures included, and the fit hashes above
are unchanged. Evidence `bench/results/svm_finish_2026-09-11/` and
`~/mojolearn-evidence/svm-finish-2026-09-11/`.

RUN OWED, both for 2665 (2666 is inert off NVIDIA, but these gates cover it):
on the Apple M4 and on an AMD MI300X, `sh bindings/build_svm.sh` then
`tools/with_identical_mode.sh pixi run mojo run -I . svm/svc_main.mojo`
(expect 44/44) and
`MOJOLEARN_NUMERIC_MODE=identical python3 bench/results/svm_speed_2026-09-11/svm_probe.py hash . /tmp`
(expect 457e29b82bca9df9, 733a383c5699f427, 2b66bc991a9c9ed0).

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
