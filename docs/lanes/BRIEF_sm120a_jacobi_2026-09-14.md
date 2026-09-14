# BRIEF: the sm_120a Jacobi refusals (DEVIATION 2711), 2026-09-14

Lane `lane/sm120a-jacobi`. Status: the 5090 leg of the probe RAN (section 9): the Jacobi is not the
site, the 17-wide split-K Gram is, and only when the source is AOT-compiled for `sm_120a`; the same
source run JIT on the same GPU is bit-identical to the Mac. A second arm of the Gram's strided loop is
on the branch behind `-D MOJOLEARN_2711_GRAM_STRIDED_SCALAR=1` (section 10), held equal to the shipped
arm on the M4, and the 5090 leg that confirms it is OWED (section 10). Sections 1 to 8 are the state
before that leg and are kept as written; where the leg superseded a sentence, section 9 says so.

## 1. The finding

`bench/results/identity_break/2026-09-14_rtx5090-sm_120a/` (RunPod NVIDIA GeForce RTX 5090, driver
580.126.16, Mojo 1.0.0 ed45d567, commit 6796ceff9, `sm_120a` from compute capability 12.0, 25 bindings
built, gemm device check 8 gates green): 410 of 414 training cells and every inference and model cell
carry the Apple M4, H100 and MI300X bits; four cells refuse, all on `odd` (12345 x 17 standard
normals), all through `jacobi_eigh_kernel` on a 17 x 17 matrix:

| cell | matrix handed to Jacobi | reported ratio after 15 sweeps |
|---|---|---|
| pca | `compute_covariance` (fused centered split-K Gram, scaled by 1/(n-1)) | 0.002727252 |
| tsvd | `gemm_tn` Gram of the raw design | 0.0011619153 |
| ols (`lstsq_eig`) | `gemm_tn` Gram, equilibrated by power-of-two scales | 0.5050234 |
| ridge (`svdEig`) | `gemm_tn` Gram of the host-centered design | 0.0027272287 |

Every one of those matrices starts at an off-diagonal ratio of 0.0343 (numpy Float64 and the Mac probe
agree). On three vendors each reaches about 1e-9 in five sweeps. 0.0027 after fifteen is a 13x
reduction; 0.505 is 15x worse than the input. This is gross, not last-bit.

## 2. The reading, questions (1) to (3)

### (1) Kernel-matrix rows that answer differently for sm_120a than for sm_90a

None. The build's column is `TARGET_COLUMN` (`checks/kernel_matrix.mojo:413-428`): the `-D` column
define, else `has_nvidia_gpu_accelerator()` (`:425`), which is True for any NVIDIA target. No row takes
a compute capability except `vendor_fp32_matmul_is_lossy` and `vendor_fp32_matmul_precision_name`
(`:261-278`), which answer for the NVIDIA column before reading the number and are called only from
checks (`checks/gram_splitk_check.mojo`, `cluster/checks/kmeans_check.mojo`, `glm/checks/ols_check.mojo`,
`neighbors/checks/estimator_check.mojo`). So the H100 and the 5090 compile the same rows:

| row | NVIDIA value | reaches the refusing path? |
|---|---|---|
| `column_shared_limit` `:199-217` | 48 KB | no (nothing on the path sizes by it) |
| `column_lane_width` `:303-323` | 32 | only through `lib_block_size_for` below |
| `lib_block_size_for[K_LIB_JACOBI_EIGH]` `:843-864` | 32 (a float-fold row, `:831-841`, resolved to the BIT_IDENTICAL column) | yes: `JACOBI_TPB`, the fold width |
| `lib_gemm_block_parallelism_for` `:913-929` | 132 (H100 SM count; the 5090 has 170) | NO: the 17 x 17 Gram takes the split-K kernel, not the tuned GEMM (`core/gemm.mojo:224-242`) |
| `lib_gemm_kernel_body_for` `:930-950` | 1 (kpack_hg) | NO, same reason |

The two NVIDIA-specific scheduling rows both live in the tuned 128x128 GEMM dispatch, which the Gram
shape never enters under IDENTICAL: `gemm_tn` asks `gram_splitk_applies(17, 17, 12345)`
(`core/gram_splitk.mojo:355-442`), which under IDENTICAL resolves the column to BIT_IDENTICAL and
answers True on capacity alone (17 <= 128, 289 <= 256 * 64).

### (2) What the Jacobi asks for at n = 17, and what the Gram asks for

`jacobi_eigh_kernel` (`decomposition/checks/jacobi_eigh_device.mojo:243-396`), launched by all four
fits as ONE block of `JACOBI_ROT_TPB = 256` threads (`:68-70`; `pca.mojo:265-274`,
`lstsq.mojo:421-430`, `lstsq_min_norm.mojo:273-282`, `svd.mojo:129-138`): static shared memory is
`rot` (2 floats, `:292`) plus the fold slab `red` (32 floats, `:136`), about 140 bytes, no dynamic
shared memory, no attribute set. The fold strides by `JACOBI_TPB = 32` over lanes 0..31; the rotation
loop runs `k = tid; while k < n; k += 256`, so at n = 17 lanes 0..16 carry a row and at n = 16 lanes
0..15 do, both inside warp 0. Nothing else changes with n. A Blackwell consumer SM offers 1024 threads
per block and 48 KB per block without opt-in (99 KB with), so the launch is two orders of magnitude
under every limit and the kernel-matrix rows that model limits are not consulted on this path.

The Gram is where 17 is special. `odd` is the only fixture whose width is not a multiple of 4 and not
even, so it is the only one that takes the split-K partial kernel's SCALAR staging arm
(`gram_splitk_stage_vectorized(17)` False, `core/gram_splitk.mojo:296-306`) and its STRIDED-SINGLES
ownership arm (`gram_splitk_reg_tiled[4](17)` False, `:343-352`; `CELLS = 4`, `:311-325`). Its
geometry: 128 pinned chunks (`:229`) of 97 rows, a 16,384-byte static staging tile (`:179-187`),
256 threads per block, one barrier after staging (`:619`) and one after accumulation (`:674`) inside
the tile loop (`:548-675`), the writeback (`:683-697`), then the serial per-cell reduce (`:731-765`).
Both arms are read as race-free and the M4 measures them
bitwise symmetric and repeatable; no other cell on the 5090 exercised them.

### (3) Predicates that could put sm_120a on an untested branch

`git grep` over `*.mojo`: `has_nvidia_gpu_accelerator()` is read at `checks/kernel_matrix.mojo:425,434`
and `checks/vendor.mojo:60` (column selection only); no `is_nvidia_gpu[...]`, `_is_sm_9x`,
`_is_sm_100x` or any capability number appears on a library path; `ctx.compute_capability()` is read
only by checks. The arch reaches the build as `--target-accelerator sm_120a`
(`tools/identity_three_columns_leg.sh:45`, `bindings/build.sh:240-246,316-317`) and nothing of ours
branches on it. Whatever sm_120a does differently is therefore in the toolchain's lowering for that
target (`barrier`, shared allocation, `fma`, the fp32 divide the rotation uses, warp scheduling) or in
the part itself, not in a row we wrote.

## 3. What the Mac measurement rules out: a deterministic violation

`ols.equilibrated` is `tsvd.gram` times one power of two on every cell (all 17 scales are
`0x3c000000` = 2^-7; the Gram's diagonal sits in one binade). Every operation in the Jacobi is exactly
invariant to a uniform power-of-two scale (the rotation angle is a ratio, the updates scale, the fold
and the stop test scale by 2^-28 together), and the Mac reference shows it: the two cases share
`v_hash` and the host-recomputed ratio at every sweep, and their `fold_off_bits` differ only in the
exponent (`49baa492` against `3bbaa492`). A contract violation that is a pure function of the input
bits, a contracted FMA, a different fold order, a different rounding, a wrong row, would therefore
have reported the SAME ratio for ols and tsvd on the 5090. It reported 0.505 and 0.0012. So the
5090's failure is not reproducible as a deterministic function of its input: a data race, an
uninitialized or stale read, or a launch that silently does less work in a timing-dependent way,
in the Gram or in the Jacobi. The probe's `repeat_identical` and `shipped_repeat_identical` lines
are the direct test of that; its `host64.cov.f32` control (Jacobi with no device Gram in the chain)
and `pca.cov.16` control (the width the 5090 converged at) name the stage.

## 4. The probe

`decomposition/checks/jacobi_sm120a_probe.mojo`, one argument, the `odd` fixture bytes
(`tools/identity_break.py` `fixture("odd")[0].tofile(...)`, 12345 x 17 float32, 839,460 bytes,
sha256 `595dda3a45cf8a3ece941d8c3565249e752294a2cf790862f7d1c4673b13eec4`). It

1. builds pca's covariance with `compute_covariance` (twice), tsvd's Gram with `gemm_tn` (twice) and
   ols's equilibrated Gram with `ols_equilibration_scale` and the two scaling kernels, exactly as the
   fits do, and prints each matrix's FNV hash, its BITWISE asymmetry count, its error against a
   Float64 host product, its off-diagonal ratio, and whether the repeat is identical;
2. runs the shipped `jacobi_eigh_kernel[JACOBI_ROT_TPB]` on each (twice, compared bitwise), then
   `jacobi_probe_kernel`, the same kernel statement for statement with a dump of `a`, `v` and the
   fold's `off` at the top of every sweep, and REFUSES if the probe's final bits differ from the
   shipped kernel's, so the per-sweep lines are the shipped trajectory;
3. prints per sweep: the device fold's `off` (value and bits), the host-recomputed ratio, and hashes
   of `a` and `v`; plus `CELLS` lines with the full bits of the input and of the final state;
4. runs two controls: the Float64 host covariance rounded to fp32, and the 16 x 16 leading block of
   the device covariance.

Run on this Mac (one compile-and-run, about a minute, load 4.9):

    OMP_NUM_THREADS=2 MOJOLEARN_CPU_THREADS=2 MOJOLEARN_BUILD_JOBS=1 nice -n 19 pixi run mojo run -j 1 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . decomposition/checks/jacobi_sm120a_probe.mojo <odd_x.f32>

## 5. The Mac reference (committed, `bench/results/identity_break/2026-09-14_rtx5090-sm_120a/jacobi_probe.apple-m4.txt`)

Apple M4, IDENTICAL, Mojo 1.0.0 (ed45d567), `x_hash=552ad56e8cb9dc63`. Every matrix: 0 asymmetric
cells, repeat identical. Every case: shipped and probe kernels bit-equal, shipped repeat identical,
converged in 5 sweeps.

| case | input hash | ratio at sweeps 0..5 | final `a` hash |
|---|---|---|---|
| pca.cov | 2345be8d29a6e663 | 0.0343, 0.0158, 0.00530, 0.000605, 1.07e-5, 2.68e-9 | 8ae8ea7142bcef95 |
| tsvd.gram | 643c4667f795927f | 0.0343, 0.0158, 0.00533, 0.000571, 7.00e-6, 1.14e-9 | ec761ef062512c36 |
| ols.equilibrated | b2cb137a38ea7700 | same six as tsvd.gram, same `v_hash` per sweep | 69f5311387f2d1fc |
| host64.cov.f32 | 40c3ce6a00608857 | 0.0343, 0.0158, 0.00530, 0.000605, 1.07e-5, 2.67e-9 | dcdc77a61a836d8e |
| pca.cov.16 | 17698e310014dd51 | 0.0339, 0.0156, 0.00483, 0.00125, 1.90e-5, 2.22e-9 | 15f9f468c2d65beb |

The Gram's error against the Float64 host product is 4.5e-7 on the covariance (max cell 1.02) and
5.3e-3 on the raw Gram (max cell 12,633), fp32 accumulation over 12,345 rows; the input `X` is
restored to the byte after `compute_covariance`.

## 6. The owed 5090 leg (one leg answers it)

From a checkout at this branch's head (the leg ships the pinned commit's tree, so the wrapper body and
the probe must be committed and pushed first; `tools/gemm_remote_leg.sh` refuses a dirty payload path):

    MOJOLEARN_GEMM_LEG_EXTRA=tools/jacobi_sm120a_probe_leg.sh \
    tools/gemm_remote_leg.sh nvidia --payload gemm --gpu "NVIDIA GeForce RTX 5090" --rent --minutes 40

(Dry run first: the same line with `--dry-run` in place of `--rent`.) The body resolves
`MOJOLEARN_GPU_ARCHS` from `compute_cap` (12.0 to `sm_120a`, the identity leg's own table),
regenerates the fixture bytes with `tools/identity_break.py`'s generator on the box, AOT-builds the
probe with the binding build's flags (`--target-accelerator sm_120a -D MOJOLEARN_NUMERIC_IDENTICAL=1`)
and runs it, then runs it JIT with `mojo run`, and leaves `gate.txt`, `probe_aot.txt`, `probe_jit.txt`,
`build_aot.log`, `gpu.txt`, `mojo_version.txt` under `/root/gemm_leg_out/jacobi_probe/`, which the
leg fetches home. The gemm payload's own device check and card run first on the same lease (they
were green on the 5090 on 2026-09-14).

How to read it, in order:

1. `MATRIX pca.cov ... asymmetric_cells=` and `repeat_identical=`: non-zero or NO means the Gram
   kernel's scalar/strided arms race on this part; the Jacobi was handed a broken matrix.
2. `PROBE case=host64.cov.f32 ... shipped_converged=`: 0 with a clean Gram means the Jacobi kernel
   itself is the site; `shipped_repeat_identical=NO` says it is a race there.
3. `PROBE case=pca.cov.16`: converging while the 17-wide cases do not reproduces the leg's pattern
   inside one binary.
4. The per-sweep lines against section 5: the first sweep whose `a_hash` departs from the Mac's is
   where the trajectory leaves the contract; `CELLS` lines give the cell.
5. `probe_aot.txt` against `probe_jit.txt`: a difference between the two compilations of the same
   source is a toolchain reading.

## 7. Proposed fix

None proposed on this branch. Nothing read names a row or a line, and section 3 says the defect is
not the kind a define can select between; a fix that is not measured against the leg's lines would be
a guess. If the leg shows the Gram racing, the site to read next is the scalar staging loop and the
strided writeback in `core/gram_splitk.mojo:548-675,683-697`; if it shows the Jacobi racing with a
clean Gram, the shared-slab reuse across back-to-back folds
(`decomposition/checks/jacobi_eigh_device.mojo:94-165`) and the `rot` broadcast (`:354-368`).

## 8. Doc corrections

Statements read for this brief and found false or superseded:

- `docs/lanes/HANDOFF_next_session_2026-09-13.md:107` "Untested architectures worth a cheap leg
  each: RTX 5090 (sm_120a), ...": superseded, the 5090 leg ran on 2026-09-14 and is recorded at
  `bench/results/identity_break/2026-09-14_rtx5090-sm_120a/`. Not this lane's file; left as written
  with this note.
- `bench/results/identity_break/2026-09-14_rtx5090-sm_120a/README.md:17-19` "Under IDENTICAL the
  sweep is pinned arithmetic, so a different convergence on one architecture is a contract violation
  on that architecture": true as far as it goes, but it reads as a deterministic violation and
  section 3 shows the reported numbers cannot come from one. A paragraph naming the probe and the
  deduction is appended to that README on this branch.

No statement in `checks/kernel_matrix.mojo`, `checks/hardware_matrix.mojo`,
`decomposition/checks/jacobi_eigh_device.mojo`, `core/gram_splitk.mojo` or `SUPPORT_MATRIX.md:99-104`
was found false by this reading.

## 9. The 5090 leg of the probe (measured 2026-09-14 10:59Z)

`bench/results/e1g/2026-09-14_105031-nvidia-rtx5090-jacobi-probe/remote/jacobi_probe/` (RunPod
RTX 5090, driver 580.126.16, compute capability 12.0 resolved to `sm_120a`, Mojo 1.0.0 ed45d567, no
`MODULAR_NVPTX_COMPILER_PATH`, commit d0b132d0a); its `gate.txt` is kept beside the Mac reference as
`jacobi_probe.rtx5090-sm_120a.gate.txt`. The wrapper built the probe twice from one source: AOT
(`mojo build --target-accelerator sm_120a`, the binding build's flags) and JIT (`mojo run`).

**JIT: bit-identical to the Mac on every line.** `pca.cov` 2345be8d29a6e663, `tsvd.gram`
643c4667f795927f, `ols.equilibrated` b2cb137a38ea7700, all 17 ols scales `0x3c000000`, 0 asymmetric
cells everywhere, every case converged in 5 sweeps to the Mac's final hashes.

**AOT: the Gram is wrong, deterministically; the Jacobi is right.**

| line | AOT sm_120a | Mac |
|---|---|---|
| `pca.cov` | hash 34a8dcbdc84f3782, 60 asymmetric cells, max error 1.0086 vs host64, repeat identical | 0 asymmetric, error 4.5e-7 |
| `tsvd.gram` | 60 asymmetric, max asymmetry 350.4, error 12,450 vs host64, repeat identical | 0, 5.3e-3 |
| `ols.scales` | `3e000000,3f800000,3c000000,...` (2^-3, 1, then 2^-7 x 15) | 2^-7 x 17 |
| `ols.equilibrated` | 15 sweeps, not converged, ratio 0.0138, fold oscillating 23.3, 0.028, 0.46, 13.5, ... | converged in 5 |
| `host64.cov.f32` (Jacobi with no device Gram) | IDENTICAL to the Mac, every sweep, final dcdc77a61a836d8e | same |
| `pca.cov`, `tsvd.gram` (Jacobi on the wrong Gram) | converged in 5 sweeps, `shipped_repeat_identical=yes` | |

Cell by cell (`diff_cells.py` over the CELLS lines, AOT against the Mac; JIT differs in 0 cells):
`pca.cov` and `tsvd.gram` differ in exactly cells 0..32, row 0 whole and row 1 through column 15;
`pca.cov.16` in cells 0..31. Those are precisely the `c = 0` cells of the 33 threads whose `c = 1`
cell (256..288, rows 15 and 16) is ALSO live under `CELLS = 4`, and those threads' `c = 1` cells are
right, as is every cell of a thread with one live lane. The wrong values are not any recognizable
chain: for each wrong cell, no product `sum x_a x_b` over all rows, over the first, second, third or
one-row tail tiles of every chunk, for any pair among the cell's and its partner's indices, matches
(closest candidates miss by 10 to 50 percent). The lane is corrupted, not redirected. `ols` is the
same 33 cells plus every (k,0) scaled by exactly 16 and every (k,1) by exactly 128: the wrong diagonal
cells (0,0) = 48.2 and (1,1) = -113.4 (true 12,193 and 12,337) set two wrong power-of-two scales.
The 60 asymmetric cells are (0,k) against (k,0) for k = 1..16 and (1,k) against (k,1) for k = 2..15.

**What this settles.** The Jacobi kernel is not the site: handed a correct matrix it is bit-identical
to the Mac in the same AOT binary. The site is the split-K Gram's strided-singles arm
(`core/gram_splitk.mojo`, the `else` branch of the accumulation), and only as AOT-compiled for
`sm_120a`. It is not a race and not a missing barrier: the wrong cells repeat bit for bit within the
binary (`repeat_identical=yes` on both products and `shipped_repeat_identical=yes` on every Jacobi
case), and a race would show in the JIT compilation of the same source on the same part, which is
clean. The reading offered on 2026-09-14 that Blackwell consumer warp scheduling exposes a missing
barrier that Hopper and the MI300X hide is therefore contradicted by the measurement; so is section
3's inference of a race (the ols against tsvd ratio disagreement it rested on is explained by the two
wrong scales, not by non-determinism). The identity leg's refusal of `pca` and `tsvd` after 15 sweeps
against this leg's convergence in 5 on the same GPU type is two AOT builds (commits 6796ceff9 and
d0b132d0a) producing two different wrong Grams, each repeatable; the Jacobi's fate on a corrupted
matrix is whatever that matrix makes it.

What is NOT settled: which pass of the AOT pipeline for `sm_120a` (the Mojo lowering, the embedded
PTX-to-SASS compiler of driver 580, or the `a`-suffixed target itself) miscompiles the loop. The
modular checkout at `/Users/andrewhendel/CascadeProjects/upstream/modular` carries `max/` only, no
stdlib, so how the stdlib classifies `sm_120a` was not read. No SASS was captured. Every other
kernel on the 5090 (410 cells, the 8 GEMM gates) is right, so the trigger is specific to this loop's
shape: a `SIMD[float32, CELLS]` accumulator whose lanes are updated one at a time inside a
`comptime for` under per-lane runtime guards, with lane indices read from `SIMD[int32, CELLS]`
vectors.

## 10. The fix arm, and the leg that confirms it (OWED)

`core/gram_splitk.mojo`: `GRAM_STRIDED_ARM` (0 shipped, 1 with `-D MOJOLEARN_2711_GRAM_STRIDED_SCALAR=1`)
selects the strided-singles arm of `_gram_splitk_partial_body`. Arm 1 gives each live cell its own
scalar accumulator (`InlineArray[Float32, CELLS]`) and its own pass over the staged rows, with the
cell's row and column as plain integers; no two lanes are updated in one loop body and no int32
vector is indexed. Every cell's chain is the same products in the same k-ascending order through the
same `identical_mul_add`, so where arm 0 computes what it says the two arms are the same bits, and
the M4 holds that: `STAGE arm=0` and `STAGE arm=1` in `jacobi_probe.apple-m4.txt` share
`partials_hash=5043c2a4194223e8` and `reduce_hash=643c4667f795927f`, equal to `gemm_tn`'s output, 0
bad chunks against the Float64 host partial per chunk (worst 5.6e-5). The tiled arm and the
uniform-jj arm are untouched. `gram_splitk_partial_kernel_arm[CELLS, ARM]` launches either arm by
name; the shipped entries take the define. The default stays 0 until the leg below holds arm 1 right
on `sm_120a`; flipping it is one line.

The probe now prints, per arm, `STAGE` lines: chunk count, partials hash, how many of the 128 chunk
partials differ from a Float64 host partial over the same rows, the cells of the first wrong chunk,
the reduce hash and whether the reduce equals `gemm_tn`. The wrapper builds the probe three ways: AOT
at the shipped arm, AOT with the define (so `compute_covariance` and `gemm_tn` themselves take arm 1
in the pca, tsvd and ols cases), and JIT.

    MOJOLEARN_GEMM_LEG_EXTRA=tools/jacobi_sm120a_probe_leg.sh \
    tools/gemm_remote_leg.sh nvidia --payload gemm --gpu "NVIDIA GeForce RTX 5090" --rent --minutes 40

Read `gate.txt` in this order: (1) `== probe_aot`, `STAGE arm=0`: expected `bad_chunks` > 0 with
`bad_cells_of_first_bad_chunk` naming cells 0..32, the defect reproduced at the stage; (2) the same
binary's `STAGE arm=1`: `bad_chunks=0` and a `reduce_hash` of 643c4667f795927f is the fix holding in
the AOT compilation; (3) `== probe_aot_arm1`: `MATRIX pca.cov` hash 2345be8d29a6e663 with 0
asymmetric cells, `tsvd.gram` 643c4667f795927f, `ols.scales` all `3c000000`, every `PROBE case`
converged in 5 with the Mac's final hashes, which is the shipped paths through arm 1; (4) `== probe_jit`
unchanged. Anything else in (2) or (3) means the miscompile is not confined to the lane update and
the arm is not the fix.

After a green leg: flip `GRAM_STRIDED_ARM`'s default to 1, rebuild the 25 bindings for `sm_120a`
and rerun `tools/identity_three_columns_leg.sh` on the 5090 (the four refused cells must read
IDENTICAL x4 against the 2026-09-14 record), then correct `SUPPORT_MATRIX.md:99-104`, which still
describes the symptom as the Jacobi not converging, to name the AOT Gram compilation.
