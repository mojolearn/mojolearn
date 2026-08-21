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

Everything below is host-side. No kernel has been enqueued, no timing number
has been taken, and none will be until the perf round.

**Landed and checked** (every check has a sabotage per mechanism that was seen
to turn it red, then restored):

| file | upstream | check | cells |
|---|---|---|---|
| `batched_levelalgo/split.mojo` | `split.cuh:32-90` | `split_check` | 54k pairs |
| `batched_levelalgo/dataset.mojo` | `dataset.h:22-38` | (used by all) | — |
| `batched_levelalgo/kernels/builder_kernels.mojo` | `builder_kernels.cuh:34-67` | `split_check` | — |
| `batched_levelalgo/kernels/builder_kernels_impl.mojo` | `builder_kernels_impl.cuh:43-88` + DEVIATION 137 steps 1-3 | `partition_check`, `range_draw_check` | 983k + 18.7k |
| `batched_levelalgo/builder.mojo` | `builder.cuh:44-135`, `:556-599` | `builder_check`, `leaf_check` | 1081 + 370 |
| `batched_levelalgo/objectives.mojo` | `objectives.cuh` + `_criterion.pyx` | `objectives_check` | 1577 |
| `decisiontree/decisiontree.mojo` | `decisiontree.hpp` + `decisiontree.cu:27-45` | `params_check` | 35 |
| `decisiontree/flatnode.mojo` | `flatnode.h:33-77`, `decisiontree.cuh:394-413` | `flatnode_check` | 5706 |
| `mojo_only/pcg_rng.mojo` | RAFT `rng_device.cuh:546-683` | `pcg_rng_check` vs a C++ oracle built from their own header | 2658 |
| `mojo_only/fixtures.mojo` | — | `fixtures_check` | 205 |

**In flight**, one file pair each so nothing converges: the exact host
transcription of `node_split_random` (`mojo_only/host_splitter.mojo`), and
cuML's two feature samplers with their dispatch rule
(`builder_kernels.cuh:152`, `:246`, dispatched at `builder.cuh:400-470`).

**Next, in order.** The device passes inside `builder_kernels_impl.mojo`:
range, draw, score, `evalBestSplit`, partition, leaf — each checked per cell
against the host form that already exists for it. Then `Builder::train`'s loop
(`builder.cuh:344-359`), which is the only thing standing between these pieces
and a tree. Then the forest wrapper.

## What reading the upstreams changed, and it was not marginal

Four times so far a plan sentence has been deleted because a file said
otherwise (rule 10), and each is recorded where it happened:

1. The partition was going to be stable and ours. cuML's is a two-pointer
   misfit swap and is deterministic anyway, so the deviation died and became a
   port (`DEVIATIONS.md` 134).
2. `PLAN.md` said sklearn draws from MT19937. It is a 32-bit xorshift
   (`_random.pxd:20-34`); MT19937 only seeds it.
3. The classification score was assumed to be `sum_c^2/n_L + sum_c^2/n_R`.
   `Gini` does not override `proxy_impurity_improvement`, so sklearn actually
   maximizes `-n_R*gini_R - n_L*gini_L` (`DEVIATIONS.md` 144).
4. A `break` in `NodeQueue.push` was described as semantically different from
   a `continue`. A sabotage showed it is not, because `leaf_counter` is
   monotone inside the loop.

And two facts about sklearn that no amount of reasoning would have produced:
their constant-feature band is computed in float32, so it WIDENS near zero and
VANISHES above magnitude ~2; and their `threshold == max` guard is reachable
here too, 191 times in 13,120 draws, purely from float32 rounding.
