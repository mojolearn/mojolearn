# The MI300X device-to-device divergence: a kernel reads memory the copy has not written yet

Follow-up to `../../2026-09-14/cholesky-mi300x-diag/`, where the Cholesky
column solve read wrong columns on two MI300X for every factor above 1 MiB
and `peer_copy_check.mojo` (a bare copy) passed. Every leg here is a RunPod
pod with two GPUs running `training/checks/peer_copy_check.mojo`
(`body.sh`, `commit.txt`, `pod_id.txt`, `gate.txt`, and the full
`peer.log` and `peer-repeat.log`). The MI300X pods report card SKU
`MI3SRIOV` (SR-IOV virtual functions, `gpu.txt`).

## What the check does

- **PEERCOPY**: the bare copy of the 2026-09-14 check.
- **PEERSOLVE**: the pre-staging Cholesky column-solve transport rebuilt
  outside the estimator (gather kernel on the root, `peer_clone` of the
  gathered columns and of the whole factor, drains of both contexts,
  `trsm_lower_kernel` on each owner, staged gather and scatter kernel) at
  n 512, 513, 1024 (and 2048 in the repeats), two right-hand sides, in
  17 variants, each compared bit for bit with the one-device forward
  substitution.
- **PEERRACE**: a bare copy of 65536 to 4194304 cells into a target (pre-filled
  with 7.25, or never written, or recycled, or with a second copy or a new
  context), then a device-1 copy kernel at once; eight trials per case.
- **PEERALIAS**: each owner's factor and column device addresses.

## Results

| leg | GPUs | source | PEERSOLVE | PEERRACE | repeats |
| --- | --- | --- | --- | --- | --- |
| `1-peersolve-mi300x` | 2x MI300X | `f67246bfa` | 7 cases differ, then 4 | not built | not built |
| `1-peersolve-h100` | 2x H100 | `f67246bfa` | 0, then 0 | not built | not built |
| `2-peerrace-mi300x` | 2x MI300X | `e520c8f49` | 4 cases differ | 0 of 5 modes | 18 and 18 cases differ |
| `2-peerrace-h100` | 2x H100 | `e520c8f49` | 0 | 0 of 5 modes | 0 and 0 |
| `3-peerrace2-mi300x` | 2x MI300X | `8977e1816` | not run | 0 of 9 modes | 16 and 16 cases differ |
| `4-peeralias-mi300x` | 2x MI300X | `17732509e` | 2 cases differ; no buffer overlaps | 0 of 9 modes | 8 and 14 cases differ |
| `4-peeralias-h100` | 2x H100 | `17732509e` | 0; no buffer overlaps | 0 of 9 modes | 0 and 0 |
| `5-peeraccess-mi300x` | 2x MI300X | `3e07d6f2c` | 4 cases differ, with `max.driver.enable_all_peer_access()` called first | 0 of 9 modes | 4 with it, 4 without it |

(`3e07d6f2c` is the merge of origin/main into `32a39bd61`, the commit the
body names; `peer_copy_check.mojo` is the same file in both.)

On two MI300X:

- **The factor copied before the columns (`l_first`) at n=513 failed in 30
  of 31 trials across the five MI300X legs**, every failure the same wrong
  last row (`got 29.288471 want -0.6508549`). At n=1024, `l_first` failed
  in 27 of 31 trials, `serial` in 23 of 31 and `prealloc` in 21 of 31; the
  old transport failed in 6 of 31 trials at n=513 and 5 of 26 at n=2048; a
  plain copy kernel in place of the solve read wrong factor cells in 1 of 23
  trials at n=1024 (`copyk`), and in 3 of 5 after a two-device factorization
  (`history_copyk`). Totals over all sizes and legs are in the tally below.
- **A two-second host wait (`sleep`), a readback of the owner's copies
  (`readback`), every host-staged variant, and the factor copied as
  sub-buffer chunks of at most 1 MiB or 4 MiB (`chunked`, `chunked4MiB`)
  never failed** (`chunked` 0 of 87 trials, `chunked4MiB` 0 of 60, `sleep`
  0 of 39, `readback`, `l_host`, `b_host` and `back_host` 0 of 15 each).
  At n=512, a factor of exactly 1 MiB, no variant failed.
- **No bare copy failed**: PEERRACE read no difference in any case (five
  modes in leg 2, nine in legs 3 to 5, 65536 to 4194304 cells), including
  targets never written before the copy.
- The owners' factor and column buffers never overlap, and the addresses are
  the same in passing and failing trials (`4-peeralias-mi300x`).
- The wrong values are the PREVIOUS contents of that memory. The variants run
  in sequence and reuse the same addresses. In `serial` at n=1024 the
  solution's first row is `205.97257`, which is `b[0] / l[1][0]`
  (`-1.858 / -0.00902`): the factor's cell 0 held cell (1, 0), which is what
  the previous variant, with its factor starting one row (4096 bytes)
  earlier, left at that address. In `l_first` at n=1024 it is exactly `1.0`:
  the factor's cell 0 held the previous variant's solution `x[0]`, which
  equals `b[0]`.
- Enabling peer access through MAX's Python driver in the same process did
  not change it (`5-peeraccess-mi300x`).

Two H100s ran the same source and read no difference in any variant, run or
leg.

Tally over the ten MI300X logs (failing trials of all trials, by n):

    l_first      59/93  512 0/5  513 30/31  1024 27/31  2048 2/26
    old          11/93  512 0/5  513 6/31   1024 0/31   2048 5/26
    serial       23/93  512 0/5  513 0/31   1024 23/31  2048 0/26
    prealloc     21/93  512 0/5  513 0/31   1024 21/31  2048 0/26
    copyk         1/69  1024 1/23, 0 elsewhere
    history_copyk 3/15  1024 3/5, 0 elsewhere
    chunked       0/87   chunked4MiB 0/60   sleep 0/39
    readback 0/15   l_host 0/15   b_host 0/15   back_host 0/15

## Verdict

Our code enqueues the copy and then calls `synchronize()` on both the source
and the target context, which is the only completion the `DeviceContext` API
offers; after that, a kernel on the MI300X target can still read cells the
copy has not written. The readback, the wait and host staging make the target
context write the bytes itself, or give the copy time to finish. We found no
indexing or lifetime error on our side (the buffers are alive, the addresses
do not overlap, and the H100 run of the same source is exact), so this is
recorded as a **platform behavior**: the MAX HIP runtime's cross-device copy,
or ROCm on these SR-IOV MI300X, completes after both drains return. Its
internals are not visible here, and the 1 MiB threshold is observed, not
explained: no variant with a factor at or below 1 MiB, and no chunked copy,
has failed.

What follows from it:

- The Cholesky column solve keeps its host staging
  (`cholesky/checks/trsm.mojo::_cho_solve_columns`).
- `core/multi_gpu.mojo::transfer_bytes` stages cross-device copies through
  host memory on AMD builds, and the byte-LM replica pools and the Cholesky
  trailing-update rows use it, because the transport audit found both reading
  wrong above 1 MiB (`../transport-audit/README.md`).
- The failing variant is the repro to rerun on a new ROCm or MAX release:
  `training/checks/peer_copy_check.mojo`, PEERSOLVE `l_first` at n=513.
