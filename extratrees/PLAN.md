# extratrees/ — histogram-free ExtraTrees, the parallel lane

Written 2026-08-21, alongside `ensemble/PLAN.md` (the RF lane). The two are
designed to run as PARALLEL LANES and this file records why that is safe:
they share NO compute-path machinery, and the substrate they would share is
deliberately duplicated instead (rule 12: parallelize across directories,
serialize on files).

## What this is, and why it gets its own directory

The histogram-free formulation of Extremely Randomized Trees (Geurts, Ernst
& Wehenkel 2006; sklearn's `RandomSplitter` in `sklearn/tree/_splitter.pyx`
is the reference implementation): per node, draw ONE random real-valued
threshold per candidate feature inside that feature's range over the node's
rows, score the induced partition, keep the best feature. No global
quantiles, no bin index, no histogram — per node the state is a few
accumulators per candidate feature instead of bins x stats, so the
3.26M-cell footprint the GBDT port measured as ~90% of per-tree fixed cost
does not exist here.

**Nobody has this on a GPU** (re-verified 2026-08-21: the only ET-on-GPU
implementations in existence are LightGBM's histogram-space `USE_RAND` CUDA
kernel — random BIN, a different algorithm — and cuML's unlanded design in
FEA #8133). So this directory mirrors NO incumbent GPU source; it is a
port-from-paper, which the house rule allows only with the deviation block
saying so explicitly. That is also why it is a top-level sibling directory
(like `cluster/`, `dbscan/`, `glm/`) and not part of `ensemble/`, which
mirrors cuML file-for-file.

This is the library's one genuinely novel candidate and the paper hook.

## The gate, shared with the RF lane

The gather-bandwidth probe (`ensemble/PLAN.md` Step 0) gates this lane
identically: the min/max pass and the accumulate pass are still scattered
gathers. The probe runs ONCE, before either lane starts kernels. This lane
wins on FOOTPRINT, not arithmetic intensity — if the probe says gathers
lose, both lanes stop together.

## Design sketch (to be firmed against the reference before kernels)

Per node batch (breadth-first frontier, same shape as any level builder):

1. **Range pass**: per (node, sampled feature), min/max of the feature over
   the node's rows. One gather, block reduce.
2. **Draw**: threshold ~ Uniform(min, max) per (node, feature). Constant
   features (min == max) are excluded, per sklearn.
3. **Score pass**: per (node, feature), accumulate left/right stats against
   the drawn threshold — class counts for Gini, (count, sum, sum-of-squares)
   for MSE. A handful of registers per candidate, no shared-memory
   histogram.
4. **Select**: best score per node, tie-break on a total order (metric,
   feature id, threshold) — the cuML `split.cuh` discipline, cited not
   invented.
5. **Partition**: stable partition of the node's rows by the winning
   predicate; children join the frontier.

Defaults are sklearn's: `bootstrap=False` (the whole learn set, every tree —
so this lane does not even need a bootstrap sampler), `max_features='sqrt'`
classification / `1.0` regression.

## Determinism and the oracle problem, stated up front

Exact parity with sklearn is NOT achievable on a GPU: sklearn draws
thresholds from one sequential 32-bit xorshift stream (`our_rand_r`,
`sklearn/utils/_random.pxd:20-34` -- NOT MT19937, which only seeds the
per-tree state) in traversal order, and that order depends on the
Fisher-Yates feature walk and on which features were found constant, which
is the wrong shape for a parallel builder. The recorded deviation: threshold
draws are keyed counter-based on `(seed, tree_id, node_id, feature_id)` —
order-independent, bit-reproducible across Metal/CUDA/HIP by construction
(the same property the RF lane gets from integer bins).

Oracles, in the house pattern:

1. **Exact**: a host-side Mojo transcription of the splitter using the SAME
   keyed draws — per-node comparison of ranges, thresholds, scores and the
   chosen split, cell for cell on 4096-row hashed fixtures.
2. **Analytic**: fixtures where the correct split is hand-computable for any
   threshold in a known interval; adversarial constant features; sabotage
   per mechanism (rule 8).
3. **Quality band**: sklearn `ExtraTreesClassifier/Regressor` at fixed seed,
   holdout accuracy/MSE bands — never bitwise, and never a gate on a real
   dataset (rule 4).

## Lane ownership (binding while both lanes run)

- This lane owns `extratrees/` and NOTHING else. It may not touch
  `ensemble/`, `core/`, `gbdt/`, or any shared harness file.
- Substrate the RF lane also needs (flat tree node struct, predict
  traversal, row partition) is DUPLICATED here in minimal form, not shared.
  Reconciliation into `core/` is the orchestrator's merge-time decision,
  not either lane's.
- Deviation numbers **130–159** are reserved for this lane, assigned up
  front (rule 3; the ledger stood at 90 when reserved). The RF lane holds
  100–129.
- Own checks only; the full suite runs once at merge (rule 12). Benchmark
  arms: interleaved, our GPU vs sklearn CPU `n_jobs=-1` on the same
  machine, sklearn defaults, one process (rules 5 and 7).

## What this lane is NOT

- Not the sklearn-API-parity ET. That product — cuML's #8133 design, a
  random-splitter flag on the ported cuML builder — is a ~150-line serial
  follow-on inside `ensemble/` AFTER RF lands, because it touches the RF
  lane's files by definition. It cannot be parallelized and should not be.
- Not a LightGBM port. Their `USE_RAND` kernel is a cited precedent for
  randomized split gating, not a source being mirrored.


---

# Session 2 log — decisions taken, and what is open

Appended 2026-08-21 by the ExtraTrees lane. Everything above this line is the
brief; everything below is what happened to it on contact with the upstreams.

## Decision: MIRROR cuML for everything except the split rule

Andrew, mid-session: *"you should be mirroring what cuml does."* Taken
literally and applied structurally. The sketch above described a builder in the
abstract ("breadth-first node frontier, same shape as any level builder"); that
abstraction is now replaced by cuML's actual files, pinned at `00094f7`:

| ours | cuML |
|---|---|
| `ported/decisiontree/batched_levelalgo/split.mojo` | `split.cuh` |
| `ported/decisiontree/batched_levelalgo/dataset.mojo` | `dataset.h` |
| `ported/decisiontree/batched_levelalgo/objectives.mojo` | `objectives.cuh` |
| `ported/decisiontree/batched_levelalgo/builder.mojo` | `builder.cuh` (`NodeQueue` + `Builder`) |
| `ported/decisiontree/batched_levelalgo/kernels/builder_kernels.mojo` | `kernels/builder_kernels.cuh` |
| `ported/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo` | `kernels/builder_kernels_impl.cuh` |
| `ported/decisiontree/flatnode.mojo` | `cpp/include/cuml/tree/flatnode.h` |
| `mojo_only/pcg_rng.mojo` | RAFT `random/detail/rng_device.cuh` + cuML's fnv1a32 chain |

What that buys, concretely: the node work queue, the frontier batching, the
ragged-node `WorkloadInfo` scheme, the split record, the tie-break, the
validity test, the in-place swap partition, the leaf-value pass and the RNG
keying all stop being things this lane designs. **The invented surface shrinks
to exactly one kernel**: cuML's `computeSplitKernel`
(`builder_kernels_impl.cuh:216-340`) builds a histogram over quantile bins,
scans it to a CDF and calls `objective.Gain` over all bins. Ours replaces that
middle with range -> keyed draw -> single-candidate score. It lives in the
mirrored file under a `DEVIATION BLOCK`, per `PORTING_RULES.md` rule 4 ("a
replacement for a step of their file lives IN THAT FILE").

Reading their partition also killed a deviation outright — see `DEVIATIONS.md`
134, which is now a NOT-a-deviation entry.

## Decision: the gather-probe gate is superseded, and here is the reasoning

The brief says: write no kernel until `bench/results/GATHER_PROBE_*.md` exists,
and stop if it reports the GPU below ~2x CPU gather bandwidth. **That file does
not exist** (checked; the RF lane owns the probe and this lane must not run
it). Andrew then said, mid-session: *"DO NOT DO NOT check for time in
subagents or in your flow... I will optimize time later. But you should be
mirroring what cuml does."*

The probe is a pure perf triage — its only output is a go/no-go on bandwidth —
so a standing instruction to defer perf and get the basics right is the same
instruction that empties the gate of content. Recorded rather than assumed:
**this lane proceeds on correctness work, takes no timing numbers at all, and
quotes no perf claim.** If the probe lands and reports gathers losing, that is
a finding about the shape of the eventual kernels, not about the host oracle,
the objectives, the RNG or the checks — none of which would be wasted.

## OPEN — decisions only Andrew can make

1. **DEVIATION 135, score accumulation precision for regression.** No `float64`
   on device. Classification is exact in integer arithmetic and needs no
   ruling. Regression label sums do: either fixed-point scaling (the precedent
   already in this repo, `gbdt/mojo_only/fixed_point.mojo::choose_scale`) or
   `Float32` with a fixed reduction-tree shape. Fixed point makes the device
   answer exactly equal to the host oracle's; `Float32` makes it equal only to
   a tolerance, which weakens every downstream check from "per cell exact" to
   "per cell within eps". Recommendation: fixed point, on the precedent. Not
   taken unilaterally because it changes what a user's regression model is.
2. **Where the deviation ledger lives.** This lane writes `DEVIATIONS.md` in
   its own directory instead of appending 130-159 to the root `PORTING.md`,
   because the RF lane is appending to `PORTING.md` concurrently and rule 12
   says file convergence is the thing that predicts integration pain. The text
   is in `PORTING.md`'s format so the merge is an append. Whoever merges
   decides whether to do it.
3. **`n_estimators` / forest-level defaults.** Not touched. This lane is
   building the LEARNER; sklearn's `ExtraTreesClassifier` defaults
   (`n_estimators=100`, `bootstrap=False`, `max_features='sqrt'` for
   classification and `1.0` for regression) are recorded as the target, and
   the forest wrapper is downstream of the tree working.

## Status

Wave 1 (host-side, no GPU, no timing) — the substrate every later check stands
on. `split.mojo` / `dataset.mojo` / `builder_kernels.mojo` landed with
`split_check.mojo` green and four sabotages proven red. In flight, one file per
sub-lane so nothing converges: the RAFT `PCGenerator` port with a C++ oracle
built from their own header; `objectives.mojo`; `flatnode.mojo` with the
predict traversal; and the fixture generators.

Wave 2, in order: the exact host transcription of `node_split_random` using our
keyed draws (`mojo_only/host_splitter.mojo`), then `builder.mojo`
(`NodeQueue` + `Builder`), then the device passes inside the mirrored
`builder_kernels_impl.mojo`.
