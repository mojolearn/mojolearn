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

## Results

(none yet)
