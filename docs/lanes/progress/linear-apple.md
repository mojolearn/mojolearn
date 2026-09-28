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
