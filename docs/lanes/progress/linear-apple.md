# linear-apple: Apple (Metal) speed for the linear family

Brief: ~/mojolearn-evidence/apple_speed_brief.md (2026-09-28). Branch
`lane/linear-apple` off origin/main 5b622763d; not merged by this lane (the
merge-gate runners merge it after an NVIDIA + CPU gate). Home Mac for speed
jobs: m4pro-a (Apple M4 Pro). Every job below went through
`tools/apple_steward.py submit --kind speed --target m4pro-a`; before and
after are on the same Mac.

## Step 1: profile on Metal

### x_linear (the 21 expansion estimators), 100k rows, IDENTICAL, main 5b622763d

`bench/x_linear_speed.py --rows 100000 --column both` (steward
1790579978918-speed-linear-5b622763d3). gpu = the Metal binding, host = the
CPU host binding, one thread. Digests equal gpu vs host on every line.

| case | Metal fit s | host fit s | Metal / host |
|---|---|---|---|
| sgd-clf | 6.468 | 0.214 | 30x |
| sgd-reg | 3.893 | 0.123 | 32x |
| perceptron | 3.097 | 0.130 | 24x |
| pa-clf | 4.406 | 0.196 | 22x |
| pa-reg | 3.122 | 0.115 | 27x |
| sgd-ocsvm | 3.524 | 0.118 | 30x |
| poisson | 176.102 | 5.460 | 32x |
| gamma | 199.551 | 6.813 | 29x |
| tweedie | 6.611 | 0.123 | 54x |
| huber | 22.400 | 0.727 | 31x |
| bayes-ridge | 5.629 | 0.143 | 39x |
| ard | 5.848 | 0.159 | 37x |
| lars | 4.972 | 0.120 | 41x |
| lasso-lars | 4.970 | 0.119 | 42x |

NVIDIA reference (RTX 4090, the linear lane's pass-2 board,
~/mojolearn-evidence/linear/before.log, same 100k rows): the same one-thread
fits take 0.55x to 0.7x of the Apple time (sgd-clf 3.526 s, poisson
124.525 s, huber 14.087 s, lars 2.235 s) and print the SAME digests as the
Metal column (80a3e27c5f39e888, cf55b32452bb72e8, ...): one thread is one
thread, the M4 Pro's is slower.

Where Apple is slow: on main every x_linear fit is ONE device thread
(x_linear/device.mojo `fit_kernel`, pass 1's design), so the Metal column is
a single GPU thread against one CPU core: 22x to 54x slower. The same is true
on NVIDIA and AMD (it is the design, not Apple); the cure is the team-fit
schedule on lane/algos-linear (x_linear/team.mojo: one block of 256 threads,
independent outputs dealt across threads, every fold still one thread's
ascending loop, so bits do not move), which is owed its gates there (see
docs/lanes/progress/linear.md, "Speed phase"). This lane does not duplicate
it: it times that branch on Metal (below) so the linear lane has the Apple
column.

Andrew 2026-09-28: lane/merged (which now holds lane/algos-linear's team
fits) merged into this branch at 8a38e574d; this lane runs no verification
(no identity requests, no sabotage runs), only speed measurements with
digest comparison. The x_linear board is re-timed at lane/merged (the team
fits) against the table above.

## Step 2: IDENTICAL speed, the QN solver (LogisticRegression, LinearSVC, LinearSVR, QNRegressor)

Where Apple is slow: every L-BFGS iteration brought device scalars home one
at a time. Per iteration on main: the line search's dg_init (1), per line
search step the regularizer and the loss (2), the gradient norm (1), and
lbfgs_search_dir's ys, yy and one dot per history pair in each of the two
loops (2 + 2 * min(m, n_vec), m = 5: up to 12). About 17 synchronizes per
iteration; a Metal synchronize with pending work costs ~4 ms on the M4.

Changes (glm/impl/qn only, no moves):
- a45361dbc: `GLMWithData.evaluate` puts the loss, the regularizer, the
  gradient norm (speculatively, for the `g` it leaves) and the OWL-QN l1
  term in four slot words read behind ONE synchronize; `grad_norm` returns
  the stored value for that `g`. `update_pseudo` / `project_direction` lose
  a synchronize that read nothing. The two-loop recursion is one launch.
- 10eab0721: the whole search direction (ys, yy, the skip test, ys / yy,
  both loops) is one launch with no synchronize; its verdict rides home with
  the line search's dg_init read.
- Bits: every stored word is the word the host-driven sequence stored. The
  device dots are `dot_kernel`'s code; the host scalar arithmetic now done
  on the device (`/`, `-`, the `ys <= 2^-23 * yy` test) goes through
  `ieee_div_f32`, `ieee_sub_f32`, `host_le_eps_times` (glm/impl/qn), which
  return the host's IEEE words including subnormal results (integer paths
  keyed on bits). glm/checks/qn_scalar_ieee_check.mojo compares them with
  the host over 220000 pairs; a host-only build of it (laptop CPU, one core)
  PASSES with 46058 subnormal results; the Metal run is in the step-2 speed
  job below.
- Syncs per iteration now: 1 (dg_init, carrying the direction's verdict) +
  1 per line-search step.

Measured, IDENTICAL, 1M rows, gpu column (the Metal binding), fit seconds;
base = lane/merged 21be273e9 (glm identical to 003ea19ba), after =
10eab0721; digests equal base vs after on every line:

| Mac | case | base | after | speedup | digest |
|---|---|---|---|---|---|
| m3ultra | logistic | 0.611 | 0.404 | 1.51x | 270ffb405d8c6a64 |
| m3ultra | linear-svc | 0.404 | 0.257 | 1.57x | bf851fa5a9479b6c |
| m3ultra | linear-svr | 0.391 | 0.213 | 1.84x | 29bbbb73d7b93520 |
| m3ultra | ols / ridge / lasso / elasticnet (untouched) | 0.054 / 0.032 / 0.208 / 0.200 | 0.049 / 0.035 / 0.201 / 0.199 | noise | equal |
| m4pro-a | logistic / linear-svc / linear-svr, step 1 only (8a38e574d) | (base below) | 0.415 / 0.238 / 0.204 | | equal to m3ultra's |

(steward 1790586887244 / 1790586889113 on m3ultra; 1790584997484 on
m4pro-a.) The Metal run of qn_scalar_ieee_check PASSES on the M3 Ultra, and
the raw hardware words differed from the host's on 9521 of the divisions and
43059 of the subtractions: this GPU flushes subnormal results, so the
integer paths are what keep those words equal.

Queued: one m4pro-a job and one m3ultra job at 10eab0721 that time after
and base (glm/ of 003ea19ba checked out and rebuilt in the same job),
IDENTICAL and FAST, core twice each, plus the x_linear board at lane/merged.

Earlier queue (superseded):
base lane/merged 003ea19ba (also the x_linear board, 100k, both columns),
step 1 8a38e574d, step 2 10eab0721 (with the Metal IEEE check); FAST base
and step 2 as well.


### Within-job before/after (M3 Ultra, steward 1790588163184 and 1790588226739)

One job at 1a5f45e6a timed each arm with glm/ of that arm checked out and
rebuilt in the same tree, IDENTICAL and FAST, 1M rows, gpu column, two runs:

| mode | case | base 003ea19ba | 10eab0721 | 1a5f45e6a | digest (all arms) |
|---|---|---|---|---|---|
| IDENTICAL | logistic | 0.632 / 0.626 | 0.330 / 0.321 | 0.313 / 0.310 | 270ffb405d8c6a64 |
| IDENTICAL | linear-svc | 0.391 / 0.390 | 0.250 / 0.246 | 0.239 / 0.240 | bf851fa5a9479b6c |
| IDENTICAL | linear-svr | 0.380 / 0.382 | 0.209 / 0.207 | 0.201 / 0.194 | 29bbbb73d7b93520 |
| FAST | logistic | 1.202 / 1.237 | 0.891 / 0.895 | 0.882 / 0.886 | c13471c2a06967db |
| FAST | linear-svc | 0.719 / 0.721 | 0.552 / 0.554 | 0.555 / 0.549 | 5255746bf9706738 |
| FAST | linear-svr | 1.015 / 1.024 | 0.742 / 0.735 | 0.753 / 0.726 | 740abbad894c93a9 |

FAST bits are unchanged by the sync work (same digests), so no quality run is
owed for it. Per-iteration cost (bench/linear_apple_profile.py, iteration caps
1..32 at tol 1e-12, slope): logistic 6.82 -> 3.30 ms, linear-svr 7.84 ->
3.86 ms; Lasso 9.06 ms per epoch (16 coordinates, ~0.57 ms each: launch
bound), untouched by the QN work.

Found on the way: on Apple `barrier()` orders threadgroup memory only; the
x_linear team fits broke on Metal (the orchestrator fixed it on lane/merged,
2979a9de0). lbfgs_dir_kernel shares `drt` through device memory too: it now
fences around its barriers on Apple (33cd0d549; the team's air.wg.barrier
spelling conflicts with the stdlib barrier the same kernel reaches).

Next commits, measured by the queued jobs: 90129fdd4 (the direction's launch
also does S/Y, the xp/gradp saves and dg_init; the gemv reads w in place: five
fewer launches per iteration) and 06ef7f558 (the CD coordinate in two
launches instead of six).

## x_linear on Metal: the team fits were wrong (ROOT CAUSE FIXED, 64c57a692)

lane/merged's team fits (x_linear/team.mojo) gave wrong Metal results: coef
all zero (sgd-*, poisson), NaN (bayes-ridge), off by 1e-6 (lars); only
ridge-cv matched. The Apple device-memory barrier (2979a9de0) was needed but
did not cure it. Cause: `Team` stored its scratch as Int addresses and
rebuilt pointers with `FP(unsafe_from_address=...)`; on Metal that pointer
has no device address space behind it. Fix: the fields are pointers
(MutUntrackedOrigin), derived from the kernel argument by arithmetic.

Proof, M3 Ultra, steward 1790594257141 (team.mojo swapped in one job):
- 2000 x 8, Metal coef vs host coef: BEFORE sgd-reg/sgd-clf/poisson all
  zero, bayes-ridge NaN, lars 1.2e-6 off; AFTER all six bit-identical.
- Board 100k, AFTER: Metal digest == host digest on all 14 cases, and equal
  to main's one-thread digests and the RTX 4090's.

| case (100k rows) | main, one thread, m4pro-a | lane/merged + fix, m3ultra | host |
|---|---|---|---|
| sgd-clf | 6.468 | 8.192 | 0.241 |
| sgd-reg | 3.893 | 5.746 | 0.138 |
| perceptron | 3.097 | 4.745 | 0.149 |
| pa-clf | 4.406 | 6.714 | 0.219 |
| pa-reg | 3.122 | 4.659 | 0.131 |
| sgd-ocsvm | 3.524 | 5.090 | 0.137 |
| tweedie | 6.611 | 0.118 | 0.122 |
| bayes-ridge | 5.629 | 0.055 | 0.053 |
| ard | 5.848 | 0.059 | 0.071 |
| lars | 4.972 | 0.021 | 0.022 |
| lasso-lars | 4.970 | 0.021 | 0.022 |
| ridge-clf | | 0.052 | 0.053 |
| ridge-cv | | 0.087 | 0.634 |
| isotonic | | 0.083 | 0.053 |

(Different Macs across the first two columns; the team fits are 50x to 240x
faster than the one-thread fits, the SGD family is not: its pass is
row-serial.) 75b48fb07 runs an SGD problem on a warp (below).

Also reverted: 06ef7f558 (a two-launch CD coordinate) gave no speedup on
Metal, see 84cff1857.

## FINAL (wind-down, 2026-09-28)

Branch tip before this note: f441fefb0. Worktree clean, no refs/wip snapshot
on origin, nothing half done. `git merge-tree` against origin/main 7d7f6b079
is clean (main already holds 64c57a692 and 2979a9de0 through lane/merged).

### What changed (all on by default; no opt-in defines added)

- QN solver (LogisticRegression, LinearSVC, LinearSVR, QNRegressor), glm/impl/qn:
  fewer synchronizes (a45361dbc, 10eab0721, 1a5f45e6a) and fewer launches
  (90129fdd4) per L-BFGS / OWL-QN iteration; lbfgs_dir_kernel keeps no
  cross-thread device words (96a7fe158, which supersedes the barrier / fence
  attempts d62e6757b and 33cd0d549); FAST fast_xtdz partials reuse the
  GLMWithData workspace (b05f9a501). Every change claims the same words.
- x_linear: Team stores pointers (64c57a692, already on main); the SGD family
  runs one problem per warp on the GPU (75b48fb07; sgd_one stays for the host
  and for d > 8 * WARP_SIZE).
- Reverted: the two-launch CD coordinate (06ef7f558, reverted by 84cff1857).

### Last measurements on record (Metal, gpu column, same job, digests compared)

QN, m4pro-a, steward 1790594941298 at 64c57a692 (HEAD includes 90129fdd4,
96a7fe158, b05f9a501); base = glm/ and solver/ of 003ea19ba rebuilt in the same
job; 1M rows, fit s, second run; digests equal base vs HEAD on every line,
IDENTICAL and FAST:

| mode | case | base 003ea19ba | 64c57a692 | per-iteration (profile slope) |
|---|---|---|---|---|
| IDENTICAL | logistic | 0.465 | 0.290 | 4.70 -> 2.86 ms |
| IDENTICAL | linear-svc | 0.304 | 0.221 | |
| IDENTICAL | linear-svr | 0.273 | 0.172 | 5.30 -> 3.21 ms |
| FAST | logistic | 1.017 | 0.816 | 10.15 -> 7.97 ms |
| FAST | linear-svc | 0.608 | 0.513 | |
| FAST | linear-svr | 0.852 | 0.685 | 13.13 -> 10.72 ms |

ols / ridge / lasso / elasticnet: untouched, noise, digests equal. Lasso
(CD) is still ~10 ms per epoch, launch bound (the only attempt, 06ef7f558,
was reverted).

x_linear SGD, m4pro-a, steward 1790595110716 at 75b48fb07 (sgd.mojo of
64c57a692 rebuilt in the same job as base): SGDDIAG, 36 cases (d = 8, 40,
300; every loss, penalty, class weight, sample weight, adaptive schedule),
Metal vs host same_bits=True maxdiff=0 at both arms; 100k board digests equal
to the host's and main's. Fit s, 100k rows:

| case | base (64c57a692) | 75b48fb07 | host |
|---|---|---|---|
| sgd-clf | 7.412 | 4.429 | 0.194 |
| sgd-reg | 4.961 | 3.295 | 0.100 |
| perceptron | 4.252 | 2.756 | 0.125 |
| pa-clf | 5.784 | 4.267 | 0.185 |
| pa-reg | 4.179 | 2.867 | 0.111 |
| sgd-ocsvm | 4.377 | 3.205 | 0.101 |

The rest of the x_linear board at 64c57a692 (team fits), m4pro-a: tweedie,
bayes-ridge, ard, lars, lasso-lars at host speed (0.017 to 0.100 s), Metal
digest == host digest.

### Unproven: owed the integration check (identity gates on CPU, NVIDIA, AMD, Metal)

This lane ran speed jobs with digest comparison only, never the verifier.
Every commit below changes a default path and is compile-checked plus
digest-equal on Metal, but has no identity/gate run on any column, and none
has run on NVIDIA or AMD:

- a45361dbc, 10eab0721, 1a5f45e6a (QN syncs; qn_scalar_ieee_check PASSES on
  the M3 Ultra Metal and host)
- 90129fdd4 (QN launches fused into lbfgs_dir_kernel)
- d62e6757b, 33cd0d549 (superseded in effect by 96a7fe158; still in history)
- 96a7fe158 (lbfgs_dir_kernel without cross-thread device words)
- b05f9a501 (FAST QN workspace)
- 84cff1857 (revert of 06ef7f558)
- 75b48fb07 (SGD one problem per warp; WARP_SIZE 32 and 64 paths never run)

64c57a692 is already on main and proven there.

### Known issues

- 96a7fe158 caps lbfgs_memory at LBFGS_FUSED_MAX_M = 256 (alpha in
  threadgroup memory); a larger lbfgs_memory now raises instead of running.
- The Metal compiler service crash (XPC_ERROR_CONNECTION_INTERRUPTED) was seen
  on the M3 Ultra at 06ef7f558 (fenced lbfgs_dir_kernel); 96a7fe158 builds and
  runs on the M4 Pro, but has not been rebuilt on an M3 Ultra.
- The SGD family on Metal is still 20x to 30x the one-core host (row-serial
  pass); the warp schedule is 1.4x to 1.7x, not a cure.
- ridge-cv / isotonic / ridge-clf were not in the last board's case list.
