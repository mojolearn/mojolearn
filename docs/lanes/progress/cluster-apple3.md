# cluster-apple3: progress

Apple FAST speed round 3 for the cluster family (brief: ~/mojolearn-evidence/apple3_speed_brief.md,
2026-09-28). Branch `lane/cluster-apple3` off lane/apple3-merged 6856b5f8f. Speed machine
m3ultra-b. No identity, sabotage or lane-check runs in this lane.

## Targets, from the measurements already on file

FAST seconds on the M3 Ultra (cluster-apple.md, 1790591694704 after dc5a20a43; round 2 made no
FAST change, 1790613748972 confirms the times), largest first:

| case | rows | taxi s | higgs s | where the time is expected |
|---|---|---|---|---|
| affinity-prop | 5k | 1.3834 | 0.9313 | per iteration device kernels (a column per thread), n^2 host loops |
| agglomerative-ward | 10k | 1.1856 | 1.2353 | the sequential host merge loop over a 400 MB matrix |
| KMeans coarse 1M x 28, k 1024 (probe, IVF shape) | 1M | | 0.70 | k-means-parallel rounds, Lloyd (shared with ann) |
| bayesian-gmm | 100k | 0.5515 | 0.6459 | 75 iterations, 2 x 3.2 MB read per iteration, host bound |
| hdbscan | 40k | 0.4805 | 0.4456 | kNN core distances (neighbors), dense pairwise, MST |
| optics | 10k | 0.4050 | 0.4035 | 400 MB read, the sequential host ordering loop |
| gmm | 1M | 0.3417 | 0.4022 | E-step, M-step |
| bisecting-kmeans | 1M | 0.2829 | 0.2953 | |
| meanshift | 10k | 0.2562 | 0.2388 | |
| minibatch-kmeans | 1M | 0.2303 | 0.2961 | |
| spectral | 10k | 0.1825 | 0.0693 | |
| dbscan | 100k | 0.0649 | 0.0333 | |
| agglomerative (single) | 10k | 0.0584 | 0.0456 | |
| kmeans | 1M | 0.0559 | 0.0906 | |

Round 2 left nothing opt-in in the tree. Its unproven commits are the consolidation's.

## Tooling (this lane)

- `MOJOLEARN_XC_PHASES=1`: `x_cluster/device_ops.mojo` drains after every primitive and prints
  the wall time per primitive and the driver's own (`host`); `x_cluster/agglo.mojo` prints the
  merge loop's four parts. Diagnostic, off by default.
- bench/cluster_apple3_prof.py: cProfile of one fit per board case (Python against binding time).
- bench/cluster_apple3_quality.py: the paired quality check, FAST against the IDENTICAL
  reference in one process (board quality number of each, labels equal, adjusted Rand index;
  for the agglomerative cases tree equal and the largest relative merge value difference).
- bench/cluster_apple3_ab.sh: round 2's A/B driver plus `phases:`, `prof:`, `quality:` cases and
  `identical/<binding>` builds.
- bench/cluster_apple3_ward_model.py: NumPy model of the ward rounds against the greedy
  Lance-Williams loop (n = 300, run on one laptop core: 6 of 6 fixtures give EQUAL children,
  planted exact duplicates included; 18 to 27 rounds).

## Jobs

| steward id | Mac | mode | commit | what | state |
|---|---|---|---|---|---|
| 1790626681580 | m4pro-b | fast | 8241f61dc | base FAST board, probes, cProfile, the existing stage timers | PASS |
| 1790627893200 | m4pro-a | fast | c55df5b59 | before (base x_cluster) / phase tables / ward rounds + quality | PASS |
| 1790628511800 | m3ultra-b | fast + identical boards | 057e55f6e | before (base sources) / after (ward rounds, GMM init fill, Python door), phases, quality | claimed about 21:19Z; m3ultra-b stopped answering ssh at about 21:26Z (reported to the orchestrator, nothing touched) |

Outputs are kept in ~/mojolearn-evidence/cluster-apple3/.

## Where the FAST time goes (M4 Pro, m4pro-b 1790626681580 and m4pro-a 1790627893200)

Base lane/apple3-merged 6856b5f8f, FAST. Phase tables are drained per primitive
(`MOJOLEARN_XC_PHASES=1`), so their sums run a little above the undrained fit.

| case | fit s (taxi) | phases, ms (taxi) |
|---|---|---|
| affinity-prop 5k | 1.34 | 127 iterations: ap_a 530, ap_r 316, ap_e 31, get_i 26; kth 255 (2 calls, ONE block each over 25M values); get 163 (4 x 100 MB, two of them for 2n diagonal values); host 108; zeros 21 |
| agglomerative-ward 10k | 1.00 | merge loop 809 (Lance-Williams 458, rescans 186, argmin 149, init 15); get 140 (400 MB); sqdist 24; zeros 20 |
| bayesian-gmm 100k | 0.74 | 73 iterations: moments 295, gets 174 (2 x 3.2 MB each iteration for the bound), host 177 (the bound's 800k products), gauss_q 60, kmeans 33, resp 15, exp 15, set 13 |
| optics 10k | 0.38 | host ordering loop 180; get 140 (400 MB); sqdist 24; zeros 19; sqrt 10; kth 9 |
| meanshift 10k | 0.25 | meanshift kernel 194 (one THREAD per seed); sqdist 24; zeros 22; kth 9 |
| bisecting-kmeans 1M | 0.33 | Python isfinite walk 69; kmeans 192 (7 fits); host 54; get 16; put 9 |
| minibatch-kmeans 1M | 0.22 | Python isfinite walk 69; 301 steps: host 66, nearest 41, get_if 21, set 20 |
| gmm 1M | 0.32 | EM (21 iterations): E 54, M 81, Cholesky 4; about 180 outside the loop (k-means start, n x K host lists, three 32 MB uploads) |
| KMeans coarse 1M x 28 k 1024 (probe) | 1.18 | k-means-parallel rounds 456; Lloyd 10 x 37; k-means++ over 16,573 candidates 150 (12 launches a pick); step 7 67; init Lloyd 25 |
| hdbscan 40k | 0.43 | Boruvka rounds 45; the rest not split (kNN core distances, neighbors) |

cProfile (1790626681580): every fit is inside its binding call except the `_f32` finite walk
(`all(map(math.isfinite, ...))`, 0.069 s at 1M x 8, in every x_cluster fit, predict and score)
and DBSCAN's `_store_core` loops (0.011 s of 0.136 s).

## Results

### FAST ward by rounds of reciprocal nearest neighbours (4d1740ff8, opt-in `-D MOJOLEARN_WARD_ROUNDS=1`)

m4pro-a, FAST, 1790627893200, same job and Mac, min of 2:

| case | before s | after s | labels digest before / after | rounds |
|---|---|---|---|---|
| agglomerative-ward taxi 10k | 1.0001 | 0.0440 | 4c92e70e869dbfe0 / 4c92e70e869dbfe0 | 42 |
| agglomerative-ward higgs 10k | 1.0258 | 0.0540 | e19cb487909db5f7 / e19cb487909db5f7 | 45 |

Phases after (taxi): ward_nn 36 ms over 42 rounds, reads 3, uploads 3, host 2.

Paired quality check against the IDENTICAL reference (bench/cluster_apple3_quality.py, rows
standardized over their own 10k in this run): labels EQUAL on both datasets (adjusted Rand
index 1.0, silhouette 0.22797949 = 0.22797949 taxi, 0.057547617 = 0.057547617 HIGGS). HIGGS:
the merge tree is equal, merge values within 2.2e-07 relative. Taxi: the tree is not equal
(cause not examined; exactly tied merges may order differently), merge values within 3.7e-06
relative. M3 Ultra A/B pending (1790628511800); the switch stays opt-in until then.

FAST only, GPU binding only (`ops.fast_device()`), ward + euclidean + no connectivity. Each round
is one device kernel (`_ward_nn_kernel`, one block of 256 threads per live cluster, the ward
dissimilarity from centroids and sizes, an integer min of (float bits, index) keys), one read of
the nearest indices and values, and the Float64 centroid updates on the host. Ward is reducible,
so every reciprocal pair of a round is a merge of the greedy tree. The merges are sorted by
(value, sequence) and numbered by rank. A budget of 256 rounds or a round without a pair returns
the fit to the matrix loop.

New `ClusterOps` methods `fast_device` and `ward_nn` (both columns implement them; the host
column answers False and keeps the matrix loop).

### Pending their A/B (1790628511800)

- 330aabc56 Python door: `_expansion_cluster._f32` takes the base binding's native finite scan;
  DBSCAN `_store_core` walks through C iterators. Same outputs; `MOJOLEARN_HOTPATH=python` is
  the reference arm. Every mode and column.
- a8cf6f407 GaussianMixture start (opt-in `-D MOJOLEARN_GMM_INIT_FILL=1`): the one-hot rows and
  their logs by fills, `_safe_log(0)` and `_safe_log(1)` taken once. Same value in every cell.
  Host code of every mode and column.

### Written, opt-in, not built or measured yet (next job)

Every switch below is `GLOBAL_NUMERIC_MODE == NUMERIC_FAST and is_defined[...]`, so an IDENTICAL
build cannot take any of them; the device paths also need `ops.fast_device()` (the GPU binding).
bench/cluster_apple3_arms/ holds one patch per change against the tree before them, so the job
builds each alone.

| switch | change | bits |
|---|---|---|
| MOJOLEARN_AP_EXACT | AffinityPropagation: the median preference by a radix select over every block of the grid (`kth_flat`; `kth` ran the 25M values on ONE block), read straight from the device's distance matrix; the two final diagonals gathered on the device (`get_diag`, was two 100 MB reads); the equal-similarities scan stops at its first difference; the responsibilities' max and second max in one walk of the row | same values |
| MOJOLEARN_AP_SPLIT | AffinityPropagation: availability column sums over row slices on every block (`ap_a_split`; `ap_a` ran one thread per column) | FAST bits move |
| MOJOLEARN_BGMM_ENT | BayesianGaussianMixture: the bound's entropy from device sums over runs of 4 products (`dot_groups`), read as n K / 4 floats; was 2 n K floats read and n K Float64 products on the host every iteration | FAST bits move |
| MOJOLEARN_MOMENTS_ROWS | mixture moments, d <= 8: every row read once per component, all of its chains folded by the thread that owns the row group | FAST bits move |
| MOJOLEARN_MEANSHIFT_BLOCK | MeanShift, d <= 16: one block of 256 threads per seed (was one thread per seed) | FAST bits move |
| MOJOLEARN_OPTICS_SIMD | OPTICS: the ordering loop's two row walks by vectors, the same decisions | same values |
| MOJOLEARN_OPTICS_HOSTROWS | OPTICS (with OPTICS_SIMD, euclidean): the loop forms the step's distance row from X; the 400 MB matrix is not read to the host | FAST bits move |
| MOJOLEARN_XC_ALLOC | a slot a distance kernel fills completely is not zeroed first (`alloc`) | same values |
| MOJOLEARN_MINIBATCH_ONE_PASS | MiniBatchKMeans: the center update in one walk of the batch; every center's chain of operations unchanged | same values |

Diagnostics added: `HDB_STAGE` times in hdbscan (`MOJOLEARN_STAGE_TIMES=1`).

New `ClusterOps` methods (both columns implement each): `fast_device`, `ward_nn`, `kth_flat`,
`get_diag`, `ap_a_split`, `dot_groups`, `alloc`.
