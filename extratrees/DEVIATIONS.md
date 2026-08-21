# Deviations in the `extratrees/` section — reserved range 130-159

Written 2026-08-21. This lane holds deviation numbers **130-159**, assigned up
front per rule 3 (the root ledger stood at 90 when the range was reserved; the
RF lane in `ensemble/` holds 100-129).

**Why this text lives here and not in the root `PORTING.md`.** `PORTING.md` is
a file the RF lane is also appending to, concurrently, and rule 12 is explicit
that the predictor of integration pain is file convergence. Every entry below
is written in `PORTING.md`'s own format so that merging it is an append, not an
edit. Moving it is a merge-time action for whoever runs the merge. **Nothing
in this file is a deviation "recorded elsewhere later" — this IS the record,
and each one is also stated as a `DEVIATION BLOCK` banner in the file it
governs, per rule 5.**

Two upstreams are cited throughout, and every entry says which one it departs
from:

- **paper/sklearn** — Geurts, Ernst & Wehenkel 2006, with scikit-learn `1.9.0`
  (`77def0e`, `~/CascadeProjects/upstream/scikit-learn`) as the reference
  implementation of the split rule.
- **cuML** — `00094f7` (`~/CascadeProjects/upstream/cuml`), the upstream for
  the builder, the split record, the tie-break, the node layout and the RNG
  keying; RAFT `661a3b8` for the generator itself.

---

## 130. sklearn's draw ORDER cannot be reproduced by a parallel builder

**Theirs.** Every random number in `node_split_random` comes from one
sequential 32-bit xorshift stream: `our_rand_r`
(`sklearn/utils/_random.pxd:20-34`) advancing a single `rand_r_state` word.
Draw order is the order of the Fisher-Yates feature walk
(`_splitter.pyx:592`), and that walk's path depends on which features were
found constant, which depends on data (`_splitter.pyx:611-621`). One threshold
draw at `_splitter.pyx:633` per surviving feature, in that order.

**Ours.** Counter-based keyed draws. The key is
`(seed, tree_id, node_id, feature_id)`, hashed with the fnv1a32 chain cuML uses
for exactly this purpose (`kernels/builder_kernels.cuh:167-170` chains
`threadIdx`, `treeid`, `nodeid`) into the `subsequence` argument of RAFT's
`PCGenerator` (`raft/random/detail/rng_device.cuh:546`). Our chain adds
`feature_id`, because ET draws a threshold per (node, feature) where cuML's
sampler draws per (node, thread).

**Reason, and it is not a preference.** A sequential stream makes draw *k*
depend on draws *0..k-1*, so a builder that evaluates features in parallel
cannot produce it without serializing the thing being parallelized. The
counter-based key is order-independent by construction, which buys a property
sklearn does not have: the same tree on Metal, CUDA and HIP, and the same tree
regardless of how the frontier was batched.

**Price.** Bitwise parity with sklearn is off the table permanently. sklearn is
therefore a QUALITY-BAND oracle only (holdout accuracy/MSE), never a gate. The
exact oracle is a host-side transcription of `node_split_random` using OUR
keyed draws — see `mojo_only/host_splitter.mojo`.

---

## 131. The feature sampler is cuML's, not sklearn's

**Theirs (sklearn).** Fisher-Yates without replacement over a persistent
`features` array, interleaved with constant-feature bookkeeping, drawing
`max_features` survivors (`_splitter.pyx:566-624`).

**Theirs (cuML).** `excess_sample_with_replacement_kernel` /
`algo_L_sample_kernel` (`kernels/builder_kernels.cuh:152`, `:246`), selected by
a problem-size test in `builder.cuh:427`, producing `n_sampled_cols` unique
column ids per node into `colids`.

**Ours.** cuML's. It is a real GPU implementation of the same intent by people
who measured, and rule 0b-i says port the path their dispatch takes.

**Price.** Our sampled feature SET for a node differs from sklearn's at the
same seed even before 130 is considered. Both are uniform without replacement
over the same space, so the quality band is unaffected; the exact oracle
transcribes OUR sampler, not sklearn's.

---

## 132. Constant features are re-discovered per node, not inherited

**Theirs.** sklearn threads `n_constant_features` down the tree through
`ParentInfo`, so a feature found constant at an ancestor is never re-examined
and is excluded from the draw
(`_splitter.pyx:557-559`, `:585-588`, `:721-730`).

**Ours.** Each node computes its own range pass and excludes constants found
there (`max <= min + FEATURE_THRESHOLD`, `_splitter.pyx:616-617`, with
`FEATURE_THRESHOLD = 1e-7` from `_partitioner.pxd:13`). Nothing is inherited.

**Reason.** The inherited counter is an artifact of the sequential loop: it is
carried in the same `features` array the Fisher-Yates walk permutes, and the
permutation IS the state. Reproducing it would serialize the frontier.

**Price, and it is paid twice.** (a) Work: a constant feature is re-scanned at
every descendant that samples it. The range pass is one gather-min/max, so the
cost is bounded by the range pass itself; it is NOT re-scored, because a
constant feature is excluded before the draw. (b) Behaviour: because sklearn
excludes known constants from the DRAW, its effective `max_features` over
non-constant features is larger than ours deep in the tree. This is a real
quality difference on datasets with many constant columns and is what the
adversarial constant-feature fixture exists to expose.

---

## 133. Tie-break is cuML's total order, not sklearn's first-wins

**Theirs (sklearn).** `if current_proxy_improvement > best_proxy_improvement`
(`_splitter.pyx:690`) — strictly greater, so on a tie the FIRST candidate in
sequential draw order survives. That is not a total order; it is a statement
about loop order.

**Ours.** `Split::update` from `split.cuh:78-90`, transcribed branch for
branch: greater metric wins; on equal metric the greater `colid` wins; on equal
colid the greater `quesval` wins.

**Reason.** Loop order does not exist here (130). A total order over the
candidate's own fields is the only tie-break that is reproducible when the
candidates are reduced in an unspecified order, and cuML already wrote it.

---

## 134. NOT a deviation: the partition is cuML's swap partition, ported

This entry started life as a deviation ("ours will be a stable partition") and
is kept, rewritten, because reading the upstream killed it — which is the
outcome rule 1 predicts and rule 10 says to write down rather than quietly drop.

**What reading `builder_kernels_impl.cuh:43-88` (`partitionSamples`) found.**
cuML does the split partition IN PLACE on `dataset.row_ids`, with a two-pointer
block algorithm: each side scans for "misfits" (`col[row_ids[loff]] >
split.quesval` on the left, `col[row_ids[roff]] <= split.quesval` on the right),
compacts them through two `cub::BlockScan` exclusive sums, and SWAPS the paired
misfits. It is not stable, and it does not need to be.

**Two things that fall out of their code and are worth stating.**

1. **Left is `<= quesval`.** `builder_kernels_impl.cuh:65-66` tests
   `col[...] > quesval` for a left-side misfit, so a row equal to the threshold
   stays left. That is the same convention as sklearn's `partition_samples`
   (`_partitioner.pyx:233-236`, `feature_values[...] <= threshold` goes left).
   **The two upstreams agree**, so the boundary row has one answer here and no
   deviation is needed for it.
2. **Their partition is deterministic.** Given the block size, the sequence of
   scans, compactions and swaps is fixed; nothing about it depends on warp
   scheduling. So the row order inside a child is reproducible, and the host
   oracle can transcribe it exactly rather than being forced to compare
   order-insensitively. A stable partition of our own would have bought the
   same property at the cost of departing from the upstream for no measured
   reason, which is the trade rule 1 forbids.

**Ours.** Theirs, transcribed, with the `cub::BlockScan` exclusive sums written
against `max.gpu.primitives.block` — a language-level counterpart, explicitly
allowed by `PORTING_RULES.md` 0b-i and not a vendor-library substitution.

## 135. OPEN — score accumulation precision on device

**Theirs.** sklearn accumulates class counts and label sums in `float64`
(`_criterion.pyx`, `sum_left`/`sum_right` are `float64_t[::1]`), and the MSE
proxy is `sum_left^2/n_left + sum_right^2/n_right` (`_criterion.pyx:944-975`).

**Ours.** There is no `float64` on device (root traps register). For
CLASSIFICATION this is not a problem and is not a deviation: with the default
`sample_weight=None` the class counts are integers, so the Gini proxy is an
exact rational in integer arithmetic. For REGRESSION the label sums are floats
and a parallel reduction has no fixed order.

**Status: OPEN, and it is the one design fork this lane cannot settle from an
upstream** — `gbdt/` solved the same problem with fixed-point label scaling
(`mojo_only/fixed_point.mojo`, `choose_scale`), which is a precedent inside
this repository rather than an invention. Resolving it is a decision recorded
in `PLAN.md`; nothing downstream is blocked on it because the classification
path is exact either way.

---

## 136. Missing values (NaN) are refused, not randomized

**Theirs.** sklearn sends missing values left or right by a COIN FLIP per
candidate (`_splitter.pyx:649`, `missing_go_to_left = rand_int(0, 2, ...)`),
and `find_min_max` counts NaNs into `n_missing` (`_partitioner.pyx:152-155`).

**Ours.** Refused by name at the API boundary, the way `gbdt/`'s `check()`
refuses every unported CatBoost option. A NaN in the input is an error, not a
silently different tree.

**Reason.** It is one more keyed draw per candidate and it is not hard; it is
simply not ported yet, and rule 3 says an unported thing must be visible. A
refusal is visible; a wrong answer is not.

**Price.** Datasets with missing values are forfeit until this is ported.
Tracked in `UNPORTED.tsv`.

---

## 137. `computeSplitKernel` has no histogram — THE deviation this lane exists for

**Theirs** (`kernels/builder_kernels_impl.cuh:216-340`), per (node, feature)
block: load the feature's quantile borders into shared memory (`:265-266`); one
pass over the node's rows, `lower_bound` each value into a bin and increment a
shared histogram (`:281-291`); large nodes atomically merge partial histograms
in global memory and the last block continues (`:295-313`); PDF to CDF in
place, one scan per class (`:316-322`); `objective.Gain` scores EVERY bin
border as a candidate (`:328`); `evalBestSplit` reduces (`:340`).

**Ours**, per (node, feature) block: range pass (min/max over the node's rows,
sklearn `_partitioner.pyx:129-165`); constant test (`max <= min + 1e-7`,
`_splitter.pyx:616-617`); ONE keyed threshold draw (`_splitter.pyx:633`);
score pass against that single threshold; `evalBestSplit` **unchanged**.

**Reason.** There is no upstream GPU implementation of this formulation to be
faithful to. The upstream is Geurts, Ernst & Wehenkel 2006 with sklearn's
`RandomSplitter` as its reference implementation, and every step above cites
it by line. Recorded as a `DEVIATION BLOCK` inside the mirrored file rather
than in a new file, per `PORTING_RULES.md` rule 4.

**Price, stated as arithmetic and not as a measurement** (this lane takes no
timing numbers): theirs reads the feature column ONCE per node, ours reads it
TWICE, because the threshold cannot be drawn until the range is known. Drawing
from the PARENT's range would fuse the two passes; the paper's
`Pick_a_random_split` draws inside the range of the node's OWN subset and
sklearn does the same, so it is not taken.

**What it buys.** Per-node state becomes a handful of accumulators per
candidate feature instead of bins x classes, and steps 1, 3 and 4 of theirs
disappear entirely rather than being made cheaper: no global quantile array, no
`lower_bound` per row, no CDF scan, no cross-block histogram merge.

---

## 138. `max_n_bins` is refused, not defaulted

**Theirs.** `DecisionTreeParams::max_n_bins` (`decisiontree.hpp:41`), default
128, validated into `(0, 1024]` (`decisiontree.cu:36-38`). It sizes the
quantile set their split search scans.

**Ours.** The field does not exist and the name is refused.

**Reason.** Deviation 137 deletes the histogram, so there is nothing for the
parameter to size. Refusing rather than ignoring follows `gbdt/`'s `check()`,
which refuses every unported CatBoost option by name: a caller who passes
`max_n_bins=1024` expecting a finer search would otherwise get a tree that
ignored the request in silence.

**Price.** A cuML configuration ported across needs that line deleted. That is
the intended cost — it is the one line that says these are different
algorithms.

**Same treatment, same reason, for four criteria.** `ENTROPY`, `POISSON`,
`GAMMA` and `INVERSE_GAUSSIAN` exist in `algo_helper.h:20-29` with real kernels
(`kernels/poisson-*.cu` and siblings) and are refused here rather than
downgraded to MSE. `MAE` is refused by cuML itself (`decisiontree.cu:38`).
