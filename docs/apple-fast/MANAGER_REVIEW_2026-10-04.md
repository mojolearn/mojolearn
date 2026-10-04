# Codex manager review, 2026-10-04

The user confirmed handover from the prior 30-minute management check. The
existing M3 runner (92814) and watchdog remain in place. M2 compiles; M3 is
the sole measurement authority. No completed scored arms are replayed.

At 16:09 UTC, serial cleanup `manager-serial-reclaim-1605` had completed,
raising M3 free space from about 7.2 GiB to 16.09 GiB. M2 had 17 GiB free.
The queue includes the remaining wave-4 jobs and a pre-existing IDENTICAL
sweep; that sweep is not new FAST candidate evidence. Candidate jobs take
priority through the existing insertion helper, without a second runner.

## Harvest and decisions

| Candidate / measured source | M3 evidence | Current decision |
| --- | --- | --- |
| `PREP3_MAXABS_POOL`, `74d233862` | `w2-w4s-maxabs-istella`: 104.6 -> 17.0 ms; quality PASS, 12 output arrays byte-identical | PROMOTED: `033fbbe10`; M2 default + `_OFF` both rc=0. Same parallel kernels and synchronized caller-owned output; no first-read deferral. |
| `KPCA_RESIDENT`, `34b4f6c72` | `w2-w4d-kpca-taxi`: 850.7 -> 127.5 ms; istella 919.5 -> 201.0 ms; quality PASS | Preparing RBF-scoped promotion review. Quality fixture first reads are 0.1 ms for both arms; eigenvalues/subspaces meet unchanged tolerances. |
| `XN_FAST_NAN_FIT_LEAN`, `74d233862` | `w2-w4s-knn-taxi`: 2.2 -> 1.3 ms; quality PASS | HOLD: newly copied host sum over downloaded counts in `nan_cells_device.mojo`; repair GPU computation before fresh candidate evidence. |
| `EIGH_FAST_TRIDIAG`, `97304d34c` | `w2-w4eigh-t-synthetic`: 43729.2 -> 701.9 ms | HOLD: single-block `td_tfac_kernel` needs parallel panel launch. Preserve the failed absolute quality gate and opponent-quality hold. |
| `CHOL_FAST_NB512`, `19bb1c7a1` | `w2-cholnb512-synthetic`: 260.4 -> 265.7 ms; quality PASS | HOLD-speed: no demonstrated improvement from one sample per arm. Do not promote. |
| `RESAMPLE_FAST_ROW_GATHER`, `74d233862` | taxi 59.9 -> 53.3 ms; istella 386.5 -> 395.4 ms | HOLD: host-side gather violates GPU computation rules; istella shows no gain. |
| `LU_FAST_TSLU`, `5867b9fbe` | factor 848.7 -> 589.2 ms; solve 841.9 -> 585.1 ms | Remains DROP-quality; timings do not override worse hard-matrix residuals. |

The eigh strict gate failed in two cases, not just the board case mentioned
in the handoff. Board-4096 eigenvalue error improves from 6.08669274745795e-5
to 4.2695061514265396e-7 but misses the fixed 3.5e-7 target. Gram-1024 falls
back to main, so its failing orthogonality error is unchanged at
0.00026960937863630826. Board residual improves 5.374987821245467e-5 ->
6.134092151174076e-7 and orthogonality 0.0011713128602654362 ->
1.688676693674385e-6. Keep these absolute gate failures visible. The handoff
separately authorizes review against main's quality; do not relabel the
strict gate PASS or count an opponent-quality-held row as qualified.

## Infrastructure failures eligible for repair

`w2-w4d-rsvd-istella`, `w2-w4d-rsvd-taxi`, and `w2-w4d-lle-taxi` fail in
`verified_arms.py` before timing because the bundled define argument differs
from the command's define argument. `w2-w4d-pca-istella` produces no timing:
the wrapper forces `AFC_FAMILY=algos`, whose harness has no `pca` lane.
Repair source/receipt validation without weakening it, preserve the failed
logs, and use new `-r1` tags only for these unmeasured jobs.

Promotion remains conditional on exact source review, acceptable quality,
GPU-only parallel computation, call plus first read where applicable, and
successful M2 default and rollback builds. Main merges, board updates,
mirror synchronization and page rebuilding belong to the manager.
