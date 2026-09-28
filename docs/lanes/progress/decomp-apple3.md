# decomp-apple3: progress (Apple FAST speed round 3)

Brief: ~/mojolearn-evidence/apple3_speed_brief.md. Branch lane/decomp-apple3
off lane/apple3-merged 6856b5f8f. Rounds 1 and 2:
docs/lanes/progress/decomp-apple.md, decomp-apple2.md. Evidence:
~/mojolearn-evidence/decomp-apple3/ (cmdN.txt = the steward command of job N).

## Item 1: the state of eigh on Metal (read from the tree, 2026-09-28 20:20Z)

Three jacobi2 eigh variants have existed. They differ only in the rotation
barrier (`dev_barrier`, x_decomp/jacobi2.mojo):

| variant | commits | barrier on Apple | Metal result |
|---|---|---|---|
| fenced | 5c144678d, 5a2f15039 | `std.atomic.fence()` then `barrier()` | pipeline creation FAILS. m4pro-b 1790606245923 ("Failed to create compute pipeline state"), and the consolidated check of main 308878e80 on m4-a ("cannot select: 113 7, 1" in agc.main). 308878e80 holds 5c144678d and 5a2f15039 and NOT eba5d65de, so main's failure is this variant: the atomic fence has no Metal instruction selection. |
| plain | eba5d65de, 4752a6a3c | `barrier()` = `llvm.air.wg.barrier(2, 1)` | builds and runs (m4pro-b 1790619265077, the round-2 2.1x), digests equal, but the barrier orders threadgroup memory only while lanes hand DEVICE words to each other: a hazard, not a proof. |
| device barrier | 342469dae (did not build: `external_call` declaration conflict), bd6af0c4b | `llvm_intrinsic["llvm.air.wg.barrier"](3, 1)`, the team_barrier flags | IN THE MERGED TREE NOW. The shipped `jacobi_eigh_kernel` got the same barrier in bd6af0c4b. |

The merged tree (6856b5f8f) therefore holds the device-barrier variant, with
main's switch 8b219bfd4 on top: on Metal `jacobi2_eigh_on()` reads
MOJOLEARN_XD_JACOBI with default "1", so the Metal default is `device_eigh`
and jacobi2 eigh is an opt-in (MOJOLEARN_XD_JACOBI=2). The one-sided SVD
default (jacobi2 from n = 256) is unchanged. The reason main gave for the
switch (the fence) no longer exists in the merged tree.

Not known from the tree: whether the device-barrier variant creates its
pipeline on M4, and what `air.wg.barrier(3, 1)` costs against
`air.wg.barrier(2, 1)`. bd6af0c4b changed the barrier of BOTH kernels, so
the round-2 seconds (98.6 s old, 46.3 s new at n = 1500) describe neither
kernel of the merged tree. Job 1 measures both.

## Jobs

| # | steward id | Mac | commit | what |
|---|---|---|---|---|
| 1 | 1790626766529 | m4-a | 0097b3d0c | builds IDENTICAL and FAST x_decomp; eigh A/B in both modes, arms d (tree default: device_eigh), 2 (jacobi2 unroll 4), 3 (jacobi2 unroll 1), with float64 quality columns; MinCovDet, Isomap, ClassicalMDS at 1000 rows; FAST Lanczos quality (N = 600, 2 seeds); out_digest default against MOJOLEARN_XD_JACOBI=2 |

| 2 | 1790628127426 | m4-a | 8c02dd401 | first build of the round-robin solvers (x_decomp/jacobi_par.mojo); FAST eigh A/B (device_eigh, jacobi2, round robin) to n = 800, FAST svd A/B to 800 x 800, LLE / Isomap / ClassicalMDS quality at 500 rows, IDENTICAL out_digest |

## Results

### Job 1 (1790626766529, m4-a = M4, commit 0097b3d0c). Output: ~/mojolearn-evidence/decomp-apple3/job1_m4-a.txt

The device-barrier jacobi2 eigh BUILDS AND RUNS on the M4 that produced
"cannot select", at unroll 4 and at unroll 1, in the IDENTICAL and in the
FAST binding. Arm d = the tree default on Metal (`device_eigh`), arm 2 =
jacobi2 unroll 4, arm 3 = jacobi2 unroll 1. Every row's three hashes are
EQUAL, in both modes.

| call | mode | d (device_eigh) s | 2 (unroll 4) s | 3 (unroll 1) s | d / 3 | hashes |
|---|---|---|---|---|---|---|
| eigh 64 | IDENTICAL | 0.020 | 0.022 | 0.016 | 1.25x | equal |
| eigh 256 | IDENTICAL | 0.375 | 0.459 | 0.326 | 1.15x | equal |
| eigh 800 | IDENTICAL | 9.922 | 10.768 | 6.586 | 1.51x | equal |
| Isomap(10nn) 1000 | IDENTICAL | 21.657 | 23.764 | 12.398 | 1.75x | equal |
| ClassicalMDS 1000 | IDENTICAL | 11.944 | 13.124 | 6.819 | 1.75x | equal |
| MinCovDet 20000 x 8 | IDENTICAL | 1.191 | 1.192 | 1.195 | 1.00x | equal |
| eigh 64 | FAST | 0.015 | 0.019 | 0.012 | 1.25x | equal |
| eigh 256 | FAST | 0.291 | 0.348 | 0.226 | 1.29x | equal |
| eigh 800 | FAST | 8.876 | 7.332 | 5.359 | 1.66x | equal |
| Isomap(10nn) 1000 | FAST | 20.183 | 15.173 | 10.465 | 1.93x | equal |
| ClassicalMDS 1000 | FAST | 11.111 | 8.381 | 5.659 | 1.96x | equal |
| MinCovDet 20000 x 8 | FAST | 0.840 | 0.801 | 0.806 | 1.04x | equal |

(eigh 8 is left out: each arm's first call pays its pipeline creation.
MinCovDet is the native x_decomp_mcd of lane/py-decomp-nbrs now: 1.2 s
where round 2 measured 22 s, and its small solves run on the host executor,
so the eigh kernel no longer shows in it.)

Quality of the eigh arms against numpy's float64 eigh of the same matrix
(largest eigenvalue error over the largest |eigenvalue| / residual /
orthogonality), the same for the three arms because the bytes are the same:
n = 64: 8.8e-06 / 8.1e-06 / 8.3e-06; n = 256: 4.2e-05 / 4.2e-05 / 4.4e-05;
n = 800: 1.4e-04 / 1.5e-04 / 1.6e-04.

IDENTICAL digests (bench/decomp_out_digest.py, 27 algorithms): tree default
== MOJOLEARN_XD_JACOBI=2 == the round-2 record (m4pro-b 1790619265077) on
every row.

DECISION: FAST on Metal takes jacobi2 eigh at unroll 1 by default
(8c02dd401). A/B gain on every row from n = 64, outputs byte-equal to the
before arm, so the quality is the before arm's. IDENTICAL on Metal keeps
main's `device_eigh` default: the evidence above says jacobi2 is safe there
too (1.5x to 1.75x, equal digests), and that flip is left to the
consolidation (MOJOLEARN_XD_JACOBI=2 is the opt-in).

FAST Lanczos (MOJOLEARN_XD_LANCZOS=1), first GPU quality check
(bench/decomp_fast_quality.py, N = 600, 2 datasets x 2 seeds, FAST binding
built in the job): 8/8 PASS. Errors against the float64 eigendecomposition
(eigenvalue / eigenvector), IDENTICAL = exact dense Jacobi:

| fit | data, seed | IDENTICAL s | IDENTICAL w / v | FAST s | FAST w / v |
|---|---|---|---|---|---|
| Isomap | swissroll 0 | 3.692 | 3.6e-05 / 2.0e-05 | 0.167 | 7.6e-07 / 1.1e-06 |
| ClassicalMDS | swissroll 0 | 1.805 | 5.3e-06 / 7.4e-06 | 0.035 | 2.2e-07 / 4.9e-07 |
| Isomap | swissroll 1 | 4.109 | 2.6e-05 / 1.5e-05 | 0.114 | 5.7e-07 / 9.4e-07 |
| ClassicalMDS | swissroll 1 | 1.809 | 2.3e-06 / 4.3e-06 | 0.035 | 1.3e-07 / 2.1e-06 |
| Isomap | gauss 0 | 4.120 | 5.4e-05 / 2.9e-05 | 0.125 | 2.8e-06 / 1.6e-06 |
| ClassicalMDS | gauss 0 | 2.260 | 7.7e-06 / 6.0e-06 | 0.036 | 7.2e-07 / 6.8e-07 |
| Isomap | gauss 1 | 4.133 | 6.4e-05 / 3.3e-05 | 0.128 | 2.7e-06 / 1.6e-06 |
| ClassicalMDS | gauss 1 | 2.251 | 7.7e-06 / 6.6e-06 | 0.033 | 1.3e-06 / 7.0e-07 |

FAST's errors are under IDENTICAL's on every row. The check at the timing
size (1000 and 1500 rows) and on the M3 Ultra is in job 3.
