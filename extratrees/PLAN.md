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
thresholds from one sequential MT19937 stream in traversal order, which is
the wrong shape for a parallel builder. The recorded deviation: threshold
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
