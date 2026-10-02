# lane/apple-fast-cluster2: affinity-prop, bayesian-gmm, bisecting-kmeans, optics (x_cluster, FAST + Apple)

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check.
Every switch is a build define (`-D MOJOLEARN_<NAME>=1`, `is_defined` at module scope), compiled under
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and TARGET_COLUMN == COLUMN_APPLE` (`XC2_FAST` in each file) and taken
only when `ops.fast_device()`; all default OFF; no build reads the environment for them; IDENTICAL, the host
column and a FAST build without the define compile main's code. The earlier env switches of the same names
are gone; `AFFINITY_FAST_SPLIT` / `_EXACT` and `BGMM_FAST_ENT` / `_ESTEP1` were env twins of main's own
opt-ins and are requested as those defines (`MOJOLEARN_AP_SPLIT`, `MOJOLEARN_AP_EXACT`, `MOJOLEARN_BGMM_ENT`,
`MOJOLEARN_BGMM_ESTEP1`). No birch lane or class exists.

Risky compile sites (M3 build-errors first): `DeviceOps.ap_loop` (`_apl_*` kernels, integer atomics on an
int slot pointer), `DeviceOps.optics_order` (one 1024-thread group, threadgroup memory), `DeviceOps.gauss_q_gemm`
and `_moments_gemm` (the `identical_gemm_into` workspace pointers through `unsafe_origin_cast[MutAnyOrigin]`),
`DeviceOps.sqdist_rows`.

| switch | lane | site | what it changes | bits |
|---|---|---|---|---|
| `-D MOJOLEARN_AFFINITY_FAST_LOOP=1` (`AP_FAST_LOOP`) | affinity-prop | `x_cluster/affinity.mojo` (the `dev_loop` branch), `device_ops.mojo` `_apl_*` kernels, `DeviceOps.ap_loop` | the iteration loop on the device, 16 iterations per host wait: `_apl_e_kernel` keeps the convergence window (n x convergence_iter ints) and per-iteration counts (settled rows, exemplars; integer atomics) on the device, the next iteration's `_apl_r_kernel` reads the counts and raises a `done` flag, later kernels of the batch return at once; one read of 2 max_iter + 2 ints per batch | same |
| `-D MOJOLEARN_AP_SPLIT=1` (main's `AP_SPLIT`) | affinity-prop | `affinity.mojo` (`split`) | `ops.ap_a_split` (availability column sums over row slices on every block); also inside the device loop | move |
| `-D MOJOLEARN_AP_EXACT=1` (main's `AP_EXACT`) | affinity-prop | `affinity.mojo` (`fast_exact`), `device_ops.mojo` `AP_R_TOP2` | the median by the grid-wide `kth_flat` straight from the device's distances (no second n^2 host copy, no n^2 upload of them), the final A/R diagonals gathered on the device (not two n^2 readbacks), the equal-similarities scan stopped at its first difference | same |
| `-D MOJOLEARN_BGMM_FAST_MOMENTS_GEMM=1` (`BGMM_FAST_MOMENTS_GEMM`) | bayesian-gmm | `device_ops.mojo` `DeviceOps.moments` | past MOM_MAX_D = 64 features (Istella-S: 200 after the constant columns) the M-step moments take `_moments_gemm` (the IDENTICAL column's route: `identical_gemm_into` at OP_TN, whose FAST arm is the vendor GEMM; `mixture/checks/mstep.mojo` kernels around it, wrapped, not changed) | move |
| `-D MOJOLEARN_BGMM_FAST_MAHAL_GEMM=1` (`BGMM_FAST_MAHAL_GEMM`) | bayesian-gmm | `bgmm.mojo` (`mahal_gemm`), `DeviceOps.gauss_q_gemm` | the E-step's Mahalanobis squares as the plain mixture forms them: per component y = X . P_k and mu_k . P_k through `identical_gemm_into` (OP_NN, FAST vendor arm) into a scratch grown once per fit, then `mixture/checks/estep.mojo::mahal_kernel` (wrapped) | move |
| `-D MOJOLEARN_BGMM_ENT=1` (main's `BGMM_ENT`) | bayesian-gmm | `bgmm.mojo` (`ent_dev`) | the lower bound's entropy products on the device (`ops.dot_groups`), the host adds ceil(n K / 4) partials | move |
| `-D MOJOLEARN_BGMM_ESTEP1=1` (main's `BGMM_ESTEP1`) | bayesian-gmm | `bgmm.mojo` (`estep1`) | the E-step's three kernels as one launch, a row per thread | same |
| `-D MOJOLEARN_BISECT_FAST_RESIDENT=1` (`BISECT_FAST_RESIDENT`) | bisecting-kmeans | `bisect.mojo` (`resident`), `DeviceOps.sqdist_rows` | the centered and the raw data uploaded once; a split's child scores read the resident rows by an index slot (m ints up, not the m x d subset again), the final inertia reuses the resident x | same |
| `-D MOJOLEARN_OPTICS_FAST_DEVICE_ORDER=1` (`OPTICS_FAST_DEVICE_ORDER`) | optics | `optics.mojo` (`optics_graph`), `device_ops.mojo` `_optics_order_kernel`, `DeviceOps.optics_order` | the ordering loop on the device: the n serial steps as n + 1 grid launches of ceil(n / 512) blocks (`_optics_step_kernel`), enqueued without a host wait; block b owns 512 rows: it folds the previous launch's partial keys to the pick (an integer min of (reachability bits, index): lowest index on a tie, as the host's strict `<`), marks and relaxes its rows from the point's distance row, then scans them to the next partial (double-buffered partials); nothing n^2 crosses to the host | same |

## Causes (the FAST fit paths, read whole)

affinity-prop (`x_cluster/affinity.mojo::affinity_fit`, n = 5,000, S is n^2 = 25M floats):
- `while it < max_iter` (line ~200): `ops.get_i(e_s, n)` EVERY iteration, a stream drain and a
  host round trip (up to 200), plus the convergence window on the host -> LOOP.
- `ops.ap_a` one thread per column, n threads walking n rows twice -> SPLIT (`ap_a_split` exists).
- `ap_r` block per row with TWO passes over the row (the `-D MOJOLEARN_AP_EXACT` kernel does one):
  still a define, not exposed.
- Host n^2 passes that stay: `s_m = ops.get(dm)` and its negation, `affinity = s_m.copy()`
  (returned to Python), `ops.put(s_m)` after the preference (100 MB upload, synced),
  `s_m = ops.get(ss)` after the noise (100 MB readback, kept for the host exemplar refinement),
  `_argmax_cols` x 2 and the per-cluster column sums (n x K each). EXACT removes the median's
  second copy/upload and the two n^2 final readbacks; the refinement's n^2 copy is listed below.

bayesian-gmm (`x_cluster/bgmm.mojo::bgmm_fit`, n = 100,000, d = 200 Istella / small taxi, K = 8, 100 iterations):
- `DeviceOps.moments`: for d > 64 FAST fell through every grid path (`_momf_*` and the shared-tile
  pass are d <= MOM_MAX_D) to `_nk_kernel` / `_xk_kernel` / `_cov_kernel`, ONE THREAD PER OUTPUT CELL
  walking all n rows: K d^2 = 320,000 threads x 100,000 dependent loads an iteration -> MOMENTS_GEMM.
- `_gauss_q_kernel`: one thread per (row, component) folding the d x d upper triangle itself,
  n K d^2 / 2 = 1.6e9 multiply-adds an E-step with every P from global memory -> MAHAL_GEMM.
- the lower bound read resp and log-resp back (2 n K floats, 6.4 MB) and formed n K Float64 products
  on the host every iteration -> ENT. Three kernels per E-step -> ESTEP1.
- Host per iteration that stays (listed below): `_m_step_host` (the Wishart covariances K d^2 host
  Float64, and `_precision_cholesky`: K x (d^3 / 6 Cholesky + d^3 / 6 triangular inverse) `identical_mul64`
  on one host thread, 32M products an iteration at d = 200); three `ops.set` uploads (K d^2 floats);
  the digamma / log-det constants (K d, negligible). The kmeans start is `ops.kmeans` (cluster/).

bisecting-kmeans (`x_cluster/bisect.mojo::bisect_fit`, board rows-full, k = 8: 7 splits):
- per split: `gather_rows` (host, m x d), `ops.kmeans` -> `kmeans_fit` (`x.copy()` on the host, its
  own upload of m x d synced, its k-means|| start and Lloyd loop with their host checks: cluster/
  code, wrapped), then `ops.put(sub)` AGAIN (m x d, synced past 1M floats) for the child scores and a
  2m readback with a host loop -> RESIDENT removes the second upload; and `ops.put(x)` (n x d) for
  the final inertia -> RESIDENT reuses the resident x.
- stays: the host column means (one Float64 chain over n x d) and centering; the gather, copy and
  upload inside `ops.kmeans` (4 host copies of m x d a split); the final n x k readback and host chain.

optics (`x_cluster/optics.mojo::optics_graph`, n = 10,000):
- `dist = ops.get(dm, n * n)`: a 400 MB readback, then n serial steps on one host thread
  (or 8 SIMD lanes under `-D MOJOLEARN_OPTICS_SIMD`), each a scan of n and a relaxation of n
  -> DEVICE_ORDER. The step sequence stays serial by construction (sklearn's `compute_optics_graph`).
- `ops.kth` (block per row, four radix passes) and `ops.sqdist` + `ops.sqrt` (one thread per cell)
  are grid-wide already; the xi extraction is O(n) host.

## Listed, not done (shape, cost estimate)

| lane | item | shape | estimate |
|---|---|---|---|
| affinity-prop | exemplar refinement on the device: `_argmax_cols` over the K exemplar columns per row (n x K, one thread per row), the per-cluster column sums (K blocks over their members), labels; drops `s_m = ops.get(ss)` (100 MB) and the host n^2 passes | 3 small kernels + n K readback | ~2 x 100 MB transfers + 4-6 host passes over 25M, est. 100-250 ms of a fit |
| affinity-prop | the `_ap_r_top2_kernel` (one pass per row) inside the device loop too (`-D MOJOLEARN_AP_EXACT` picks it in `ap_r` only) | comptime kernel pick in `ap_loop` | one n^2 read of A + S less per iteration (~0.1-0.3 ms at n = 5,000, x n_iter) |
| affinity-prop | `ops.put(s_m)` after the preference (100 MB synced upload): write the preference diagonal on the device instead | one n-thread kernel on the resident `dm` negated in place | ~20-40 ms |
| bayesian-gmm | `_precision_cholesky` on the device (`mixture/checks/mstep.mojo::gmm_precision_cholesky`, float32, wrapped) | K blocks, d = 200 | host 32M `identical_mul64` an iteration (est. 50-150 ms x 100 iterations); float32 factors change the bits and the precision: the paired quality check decides |
| bayesian-gmm | the Wishart covariance assembly on the device (K d^2 cells) and the three `ops.set` uploads folded into one | one K d^2 kernel | K d^2 = 320,000 host Float64 ops an iteration, est. 2-5 ms |
| bayesian-gmm | the moments' GEMM route for d <= 64 too (taxi): A/B `_momf_*` (row slices) against `_moments_gemm` | a second define for `moments` | unknown; the slices path is already grid-wide |
| bisecting-kmeans | a device-resident 2-means over a row mask of the resident xc (Lloyd on the grid, deterministic chunked sums, a device convergence flag), replacing `ops.kmeans` per split | n_split launches per Lloyd iteration | removes 4 host copies of m x d per split (est. 50-150 ms a split at 1M x 20) and `kmeans_fit`'s own host checks; the k-means / k-means++ start must match cuVS's to keep quality: a bigger item |
| bisecting-kmeans | the column means and the centering on the device (chunked deterministic column sums, a center kernel) | 2 kernels + n d readback | the host Float64 chain over n x d twice, est. 50-100 ms at 1M x 20; the gather still needs xc on the host while `kmeans_fit` takes host rows |
| bisecting-kmeans | the per-child score and the final inertia as device partial sums (fixed-order chunk partials, a final fold) instead of 2m / n k readbacks and host loops | 2 kernels each | est. 10-30 ms a fit at 1M rows |
| optics | `sqdist` of n x n in the GEMM form (row norms + X X^T through `core/gemm.mojo gemm_nt`, FAST vendor arm) | one GEMM + one kernel | n^2 d = 2.2e10 multiply-adds one-thread-per-cell at d = 220, est. 30-80 ms -> ~5-10 ms; bits move |
| optics | DONE as the only form (the pre-push hook refuses a one-block launch over a runtime size): the per-step scan and relaxation on the grid, one launch a step, the combine folded into the next launch, 10,001 launches enqueued without a wait. Open: OPT_ORD_PER (rows per block) against the launch count if the M3 shows the step latency-bound | 2 kernels, OPT_ORD_TPB tunable | unknown on Metal (launch cost per enqueue vs 10 serial loads per thread per step) |

## Keep rule
A switch becomes the FAST default when its arm is faster on the M3 and held-out quality stays within
FAST's run-to-run spread (the "same" rows must also keep the digest); then the define goes and the
arm is the code. The queue (docs/apple-fast/ab/cluster2.txt, 9 lines) is the light form: `tools/afc_ab_def.sh`
(two FAST builds of `x_cluster`, "" vs the define), `1 2`, one tag per lane x dataset, no -ident lines, ONE
dataset per define: Istella where the gap table lists the lane on Istella (affinity-prop, bisecting-kmeans,
optics), taxi for bayesian-gmm (listed on taxi only), except `BGMM_FAST_MOMENTS_GEMM`, which only runs past
MOM_MAX_D = 64 features and is a no-op on taxi, so its one line is Istella. After a win, the other dataset of
the same define, the same form: `cluster2-ap-loop-taxi`, `cluster2-bgmm-mahal-istella`, `cluster2-bgmm-ent-istella`,
`cluster2-bisect-resident-taxi`, `cluster2-optics-devorder-taxi`; and the combined arms (`cluster2-ap-all-*`,
`cluster2-bgmm-all-istella`) once each define has won alone. Dropped: the single-arm `XCPHASE` diagnostic
(`cluster2-bgmm-phases-istella`, `MOJOLEARN_XC_PHASES=1`, main's read): ask for it separately if the host
M-step's share is needed before the Cholesky item above is taken.

## Compile risks to watch (first M3 build)
`_apl_*` kernels call the plain kernels (`_ap_r_kernel`, `_ap_a_kernel`, `_apf_*`) as device functions;
`Atomic.fetch_add[ordering = Ordering.RELAXED]` on a global `IPtr` (precedent core/block_reduce.mojo:241);
`block_dim=1024` with `grid_dim=1` on Metal (precedent svm/impl/smosolver.mojo:413);
`mixture.checks.estep` (`mahal_kernel`) newly imported by the x_cluster binding;
`identical_gemm_into` on sub-buffers of one scratch buffer (`y`, `murow`, `ws` at distinct offsets).
