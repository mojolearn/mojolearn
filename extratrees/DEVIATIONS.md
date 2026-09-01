# Deviations in the `extratrees/` section

Written 2026-08-21. This lane's FIRST reserved block was **130-159**, assigned
up front per rule 3 (the root ledger stood at 90 when the range was reserved;
the RF lane in `ensemble/` holds 100-129). That block filled, and the lane
continued at 160; the blocks this file now holds are **130-218, 450-465, 1943
and 1945**. See the INDEX below for every one of them.

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

## INDEX

Every deviation in this file, by number. **STATUS** is CLOSED (decided, with
the reason and any measurement recorded in the entry), OPEN (the entry names
work that must land before the deviation is settled), or SUPERSEDED (a later
entry replaced it; the text stays for the argument it makes).

**OPEN today: 1945 only.** **SUPERSEDED: 151 by 205, 161 by 204.** Every other
deviation here is decided. That is not the same as nothing being owed; several
CLOSED entries carry a correction or a still-wanted measurement and say so in
place.

Line numbers are as of this revision and drift with any edit; the deviation
number in the heading is the durable anchor, and every heading now carries its
status in square brackets.

```
NUM   STATUS      LINE  SUMMARY
130   CLOSED        164  sklearn's draw ORDER cannot be reproduced by a parallel builder
131   CLOSED        196  The feature sampler is cuML's, not sklearn's
132   CLOSED        217  Constant features are re-discovered per node, not inherited
133   CLOSED        252  Tie-break is cuML's total order, not sklearn's first-wins
134   CLOSED        277  NOT a deviation: the partition is cuML's swap partition, ported
135   CLOSED        310  RULED: regression accumulates in FIXED POINT
136   CLOSED        383  Missing values (NaN) are refused, not randomized
137   CLOSED        402  computeSplitKernel has no histogram, the deviation this lane exists for
138   CLOSED        437  max_n_bins is refused, not defaulted
140   CLOSED        462  RAFT's wide 64-bit multiply has no PTX fast path here
141   CLOSED        485  Only PCGenerator's three-argument constructor is ported; no half
142   CLOSED        508  Thresholds are Float32, and the rescale is not allowed to fuse
143   CLOSED        794  Histogram-free: the accumulators ARE the arguments
144   CLOSED        820  Classification selects on an EXACT INTEGER proxy
145   CLOSED        867  The exact comparator is the authority, not Split.best_metric_val
146   CLOSED        566  TreeMetaDataNode is reduced: train_time is not ported
147   CLOSED        589  predict_one accumulates and never zeroes; a zeroing entry point beside it
148   CLOSED        617  Node template narrowed: LabelT dropped, IdxT fixed to Int32
149   CLOSED        656  Fixtures are counter-based, not stream-based
150   CLOSED        678  No fixture contains a missing value
151   SUPERSEDED    894  The supplied column list IS the search (CLOSED by 205)
152   CLOSED        931  The candidate scan COUNTS; only the winner is partitioned
153   CLOSED        959  Regression selects on sklearn's MSE proxy, reports cuML's gain
154   CLOSED        984  Two of sklearn's four rejection branches are absent; their inputs do not exist
156   CLOSED       1016  The k == n guard moved into the sampler, and materializes an identity
157   CLOSED       1045  Block collectives are explicit loops; the block width is a parameter
158   CLOSED       1082  log, exp and ceil come from libm through FFI, not std.math
159   CLOSED       1106  Index widths, and a bound on their unbounded loop
160   CLOSED       1185  NOT a deviation: the range pass is bit-checkable, the histogram pass is not
161   SUPERSEDED   1210  The cross-block combine is a MUTEX MERGE (CLOSED by 204)
162   CLOSED       1248  The output is a STRUCT OF ARRAYS; Dataset passes as its components
163   CLOSED       1275  The empty range is carried as min > max; NaN never reaches the reduction
164   CLOSED       1320  FIXED cuML BUG: no head flag on the block minimum, so column 0 never sampled
165   CLOSED       1358  FIXED cuML BUG: a slot that never drew voted anyway, as column n-1
166   CLOSED       1387  The reduction carries the EXACT RATIONAL KEY beside the Split
167   CLOSED       1429  The Int128 cross-multiply is HAND-WIDENED from 64-bit limbs
168   CLOSED       1467  warpReduce is a shuffle_xor butterfly with NO ASSUMED WARP WIDTH
169   CLOSED       1493  evalBestSplit's publish: portable mutex, SoA cell, device path report
170   CLOSED       1533  The score pass is TWO launches, because no block can be elected last
171   CLOSED       1554  NOT a deviation: this merge IS their atomicAdd
172   CLOSED       1568  No dynamic shared memory: private accumulators plus a block reduction
173   CLOSED       1587  The draw is an explicit fma, and it is why 142 was amended
174   CLOSED       1603  The device sees only INTEGERS, and a refusal is a STATUS
175   CLOSED       1623  The published width is Int64, TIGHTER than the host's Int128 bound
176   CLOSED       1758  partitionSamples on device, with a BOUND on their unbounded loop
177   CLOSED       1791  NOT a deviation: one block per node is kept
178   CLOSED       1804  Leaf histogram is a comptime-bounded private array plus one block reduction
179   CLOSED       1817  RULED: the device regression leaf is FIXED POINT; Float64 host form reports
180   CLOSED       1844  One leaf launch over all nodes, and two device-written path reports
181   CLOSED       1862  volatile IdxT* row_ids has no Mojo spelling; barrier() carries it
182   CLOSED       1658  score_to_candidate_kernel exists because 170 split their kernel in two
183   CLOSED       1683  The device applies cuML's zero-gain gate, from exact integers on the host
184   CLOSED       1877  The dataset is uploaded once for the FOREST, not once per tree
185   CLOSED       1910  The per-tree row_ids is a contract, not a necessity, and that is MEASURED
186   CLOSED       1935  The class-id cast is hoisted, and carries a range guard the host lacks
187   CLOSED       1951  The device is an ENTRY POINT, not a device=True field on the config
188   CLOSED       1978  The estimator's device regressor: refused BY NAME, then closed
189   CLOSED       2016  The regression key is cuML's OWN MSE gain, not sklearn's proxy
190   CLOSED       2043  MEASURED: the alternatives lose orderings and this one does not
191   CLOSED       2067  The shift is node-uniform and derived, so there is no row cap
192   CLOSED       2079  What the key is NOT: order preservation, measured
193   CLOSED       2095  The precondition is a refusal STATUS, and the gate is host-side
194   CLOSED       2106  On an exact tie between two VALID rationals, the metric arm is skipped
195   CLOSED       2191  cub::BlockRadixSort has no counterpart; a hand-written bitonic network
196   CLOSED       2207  MEASURED: their sort's storage does not fit, so items lives in global scratch
197   CLOSED       2218  The block scan is per-thread totals then one collective
198   CLOSED       2225  The DISPATCH stays on the host, because it is on the host in cuML
199   CLOSED       2235  MEASURED: Metal has no double, so cuML's algorithm L cannot be a kernel
200   CLOSED       2150  row_ids is filled ON THE DEVICE, as thrust::sequence does
201   CLOSED       2256  The algo-L arm runs on the HOST, a placement difference, not a refusal
202   CLOSED       2276  The workspace is allocated once per tree, not once per level
203   CLOSED       2317  The frontier partition is multi-block
204   CLOSED       2360  The cross-block range merge is lock-free; CLOSES 161
205   CLOSED       2409  sklearn's constant-draw rescue clause; CLOSES 151; regression via 206
206   CLOSED       2507  Regression takes 184, 202, 203 and 205, which it never had
208   CLOSED       2572  The label gather hoist, MEASURED AND DECLINED; holds the only gather probe
211   CLOSED       2678  The batch spans TREES: cuML's stream pool as a wider grid
212   CLOSED       2758  The materialized score pass, MEASURED AND DECLINED, at forest scale
213   CLOSED       2820  The merged batch bound, MEASURED, NO EFFECT, with the Instruments profile
214   CLOSED       2902  The phase clock, and the leaf tail batched
215   CLOSED       2953  cuML's feature sampler is BIASED; k survivors now a uniform keyed-hash subset
216   CLOSED       3019  The zero-gain gate: cuML's <= becomes sklearn's boundary
217   CLOSED       3088  cuML's float gain goes NEGATIVE on non-negative splits; exact-sign clamp
218   CLOSED       3142  The 2^21 classification row cap LIFTED, via 191's shift on the Gini pair
450   CLOSED       3183  One drain per batch: five of the level cycle's six synchronizes removed
452   CLOSED       3285  Under NUMERIC_IDENTICAL the range pass's block fold runs in KEY space
453   CLOSED       3329  IDENTITY_PATHS row 10 applied to the ET device path's float seams
454   CLOSED       3382  Identity-trace checkpoints at the audit's hazard stages
455   CLOSED       3404  The rescue re-stage broke 450's invariant: a MEASURED race, one-drain fix
456   CLOSED       3453  The algo-L device kernel's FLOAT seams route through row 12's portable pair
457   CLOSED       3489  n_parallel_samples_for is mode-gated; cross-vendor is cross-HOST here
458   CLOSED       3534  The regressor binding overwrote max_features after reading it
459   CLOSED       3602  Entropy: cuML's float gain is the key, published as a FLOAT SEAM
460   CLOSED       3707  bootstrap=True through the RF lane's Philox port; sklearn's max_samples
461   CLOSED       3789  The Python boundary: 22 params slots, slot 18 a count, meta[5] the size
463   CLOSED       3809  Exact ties resolve by a keyed pseudorandom rank, not the highest column id
464   CLOSED       3924  excess_selection_hash runs through a full-avalanche finalizer
465   CLOSED       3969  key_for gets its own salt link; no collision with the sampler's stream
1943  CLOSED       4018  The frontier block is 512 threads on a 64-lane wavefront, 128 on 32
1945  OPEN         4114  The MI325X phase timer bills 88% of a higgs ET fit to the host queue push;
                       the same host loop on Apple is 0.53%, so the attribution is what is open
```

Two sections carry no deviation number:

```
              LINE  SUMMARY
              2643  What was left after 208, and what each item turned into (211, 212, 450)
              3272  AMENDMENT to commit d85c6ce's staleness note: the two compile
                    graphs are DISJOINT, an extratrees edit stales _mojolearn_trees.so only
```

Numbers 139 and 155 have no entry here; 462 is reserved and unused (see the
note inside DEVIATION 464). The blocks this file holds are 130-218, 450-465,
1943 and 1945, and they are not contiguous because other lanes own the gaps.

---

## 130. [CLOSED] sklearn's draw ORDER cannot be reproduced by a parallel builder

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
keyed draws — see `checks/host_splitter.mojo`.

---

## 131. [CLOSED] The feature sampler is cuML's, not sklearn's

**Theirs (sklearn).** Fisher-Yates without replacement over a persistent
`features` array, interleaved with constant-feature bookkeeping, drawing
`max_features` survivors (`_splitter.pyx:573-626`, the draw itself at `:592-593`).

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

## 132. [CLOSED] Constant features are re-discovered per node, not inherited

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

**Price.** (a) Work: a constant feature is re-scanned at every descendant that
samples it. The range pass is one gather-min/max, so the cost is bounded by the
range pass itself; it is NOT re-scored, because a constant feature is excluded
before the draw.

**Price (b), RETRACTED 2026-08-26 by the covtype audit.** This entry used to
claim that because sklearn excludes known constants from the DRAW, its
effective `max_features` over non-constant features is larger than ours deep in
the tree. Reading `node_split_random`'s loop guard against its draw shows
otherwise: a KNOWN constant drawn again still consumes a visit
(`n_drawn_constants` counts toward `n_visited_features`, `_splitter.pyx:
573-577`, `:585-603`), so sklearn spends its per-node draw budget on known
constants exactly like us. The only behavioural residue of inheritance is
DEVIATION 151's fact (b) — the guard's second disjunct keeps drawing past
`max_features` only while EVERY feature drawn so far was constant — which that
entry already carries. The covtype gap this sentence was once suspected of
explaining is DEVIATION 463's tie-break bias, not this.

---

## 133. [CLOSED] Tie-break is cuML's total order, not sklearn's first-wins

**Theirs (sklearn).** `if current_proxy_improvement > best_proxy_improvement`
(`_splitter.pyx:693`) — strictly greater, so on a tie the FIRST candidate in
sequential draw order survives. That is not a total order; it is a statement
about loop order.

**Ours.** `Split::update` from `split.cuh:78-90`, transcribed branch for
branch: greater metric wins; on equal metric the greater `colid` wins; on equal
colid the greater `quesval` wins.

**Reason.** Loop order does not exist here (130). A total order over the
candidate's own fields is the only tie-break that is reproducible when the
candidates are reduced in an unspecified order, and cuML already wrote it.

**AMENDED by DEVIATION 463 (2026-08-26).** The total-order requirement stands;
the ORDER changed. "Greater colid wins" is a systematic bias, not a neutral
choice, and it was measured costing accuracy on covtype. The shipping tie
order is now a keyed pseudorandom rank over `(tree, node, colid)`; the
max-colid rule survives as `MOJOLEARN_ET_TIE_MAX_COLID` /
`SPLIT_SAB_MAX_COLID_TIE`, so this entry's reproducibility gate keeps its
subject.

---

## 134. [CLOSED] NOT a deviation: the partition is cuML's swap partition, ported

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

## 135. [CLOSED] RULED — regression accumulates in FIXED POINT

**Theirs.** sklearn accumulates class counts and label sums in `float64`
(`_criterion.pyx`; `sum_left`/`sum_right` are `float64_t[::1]`), and the MSE
proxy is `sum_left^2/n_left + sum_right^2/n_right` (`:944-973`).

**Ours.** There is no `float64` on device.

- **Classification was never open.** With the default `sample_weight=None` the
  class counts are integers, so the Gini proxy is an exact rational in integer
  arithmetic — deviation 144.
- **Regression is now ruled: FIXED POINT.** Andrew, 2026-08-21. Labels are
  quantized once, on the host, by a power-of-two scale derived from the whole
  dataset's sum of magnitudes, and every accumulation on device is integer.

**Reason.** Integer addition is associative and exact, so the accumulation is
order-independent by construction: however partial sums combine across lanes,
blocks and vendors, the total is the same bits. A `Float32` accumulator would
make the answer a function of the reduction's shape, which would weaken every
downstream check from "exact, per cell" to "within eps" and make the same fit
on Metal, CUDA and HIP three different models.

**This is a port of a precedent in this repository, not a new idea.** Root's
`checks/fixed_point.mojo` already solves it for the GBDT learner, where
CatBoost flushes histograms with a float `atomicAdd` Metal does not have. Three
of its ideas are taken unchanged with their reasoning: the bound comes from the
WHOLE dataset once (any node's rows are a subset, so a scale that keeps the
global magnitude sum inside the slot keeps every partial sum inside it, at
every node, at every depth); the scale is SNAPPED DOWN to a power of two so it
is a step function of the magnitude rather than a lever arm for its last bits;
and quantization truncates toward zero, with no tie-break to get wrong on
another vendor.

**What this lane had to derive for itself, and it is not a detail.** Root's
bound is about the ACCUMULATOR. This lane has a second, tighter constraint
root does not: deviation 144's exact comparison cross-multiplies
`num = sum_L^2 n_R + sum_R^2 n_L` against `den = n_L n_R`, and it is THAT
product that must fit `Int128`. With `M = 2^b` the largest scaled sum and `n`
the node's rows:

    num <= 2 M^2 n      den <= n^2/4      num*den <= M^2 n^3 / 2

so the requirement is `2b + 3 log2(n) <= 128`. **A scale chosen only for the
accumulator overflows the comparator**: at root's `2^30` slot and this lane's
`2^26` row cap the product reaches `2^137`, ten bits past `Int128`. So
`accumulator_bits_for` spends resolution against the row count —

    n <= 2^22 (4.2M)  ->  b = 30, nothing lost
    n  = 2^24 (16.8M) ->  b = 27
    n  = 2^26 (67M)   ->  b = 24

— and refuses below a 16-bit floor rather than returning a useless scale.
`comparator_product_fits` computes the worst case in `Int128` at every row
count so the algebra above is a CHECKED claim, not a comment.

**Price, measured rather than asserted.** Quantization truncates, so the
recovered sum is within one unit per row of the float64 sum: on a 4096-row
hashed fixture the round-trip error was 0.0081 against a budget of 4.0.
Resolution costs nothing below ~4M rows in a single node and degrades
gracefully above it. Against that, the float32 control in the same check
disagrees with itself across summation orders (586.2111 forward, 586.2113
reverse, 586.20996 pairwise) — which is the cost fixed point removes.

**Implementation:** `checks/fixed_point.mojo`, checked by
`checks/fixed_point_check.mojo` (8280 cells, seven sabotages).

**Consequence for `AggregateBin`.** `objectives.mojo` deliberately left its
accumulator type a parameter pending this ruling. The device instantiates it
over an integer type; the host oracle may keep `Float64`, and where the two are
compared the comparison is through the quantized values, not the floats.

---

## 136. [CLOSED] Missing values (NaN) are refused, not randomized

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
Tracked in `NOT_IMPLEMENTED.tsv`.

---

## 137. [CLOSED] `computeSplitKernel` has no histogram — THE deviation this lane exists for

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

## 138. [CLOSED] `max_n_bins` is refused, not defaulted

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

---

## 140. [CLOSED] RAFT's wide 64-bit multiply has no PTX fast path here

**Theirs.** `raft/util/integer_utils.hpp:207-237` (`wmul_64bit`) picks between
two implementations on `__CUDA_ARCH__`: two inline-PTX instructions
(`mul.hi.u64`, `mul.lo.u64`) on device, and a four-partial-product schoolbook
expansion with an explicit carry on host.

**Ours.** Only the schoolbook expansion, transcribed line for line. Mojo has no
inline PTX, and under the ALWAYS-GPU-AGNOSTIC rule an `if nvidia:` arm would be
forbidden even if it did.

**Why it is safe.** The two branches compute the same 128-bit product — a
code-generation difference, not an arithmetic one. The oracle exercises the
result through `uniform_int_u64` over seven ranges (including `diff = 2^63 + 1`,
which puts a bit in every position of the high word) across nine streams, plus
four hand-written cells pinning the low word.

**Price.** On device this is several instructions instead of two. Perf is
deferred; when it stops being deferred this is a KERNEL MATRIX row, not a file
edit.

---

## 141. [CLOSED] Only PCGenerator's three-argument constructor is ported, and there is no `half`

**Theirs.** `PCGenerator` has a second constructor taking
`DeviceState<PCGenerator>` (`rng_device.cuh:557-560`) which forms
`_init_pcg(state.seed, state.base_subsequence + subsequence, subsequence)` —
note it passes the subsequence AGAIN as the offset. It also has `next_half`
(`:653-657`).

**Ours.** Only `PCGenerator(seed, subsequence, offset)`, which is the one cuML's
decision-tree code calls (`builder_kernels.cuh:172`), and no `half`.

**Why.** `DeviceState` and `Rng` are RAFT's whole-array RNG driver, one
generator per thread of a fill kernel. Nothing in this lane fills an array with
noise; every draw is keyed. `half` has no use either — thresholds are compared
against `Float32` feature values.

**Price.** Anyone later porting RAFT's `Rng` array fills must port the
`DeviceState` constructor AND notice that its offset argument is the
subsequence, not zero. An omission, not a behaviour change: no ported call site
takes the missing path.

---

## 142. [CLOSED] Thresholds are Float32, and the rescale is not allowed to fuse

**Theirs, twice over.** sklearn's `rand_uniform` (`_utils.pyx:57-61`) is
`float64_t` throughout. RAFT's `custom_next` for `UniformDistParams<OutType>`
(`rng_device.cuh:173-183`) writes the rescale as ONE C++ expression,
`(res * (params.end - params.start)) + params.start`, which both nvcc
(`--fmad=true`) and clang (`-ffp-contract=on`) may contract into a single FMA.

**Ours.** `Float32`, because there is no `float64` on device. And the multiply
is isolated in a `@no_inline` helper so the product is rounded before the add —
Mojo contracts multiply-then-add ACROSS STATEMENTS, so putting the two halves
on separate lines is not enough. This is the third sibling in this repository's
FMA-contraction family.

**Measured, because it is easy to overclaim.** Unfused Mojo vs unfused
reference: 0 of 2658 cells differ. Fused Mojo vs unfused reference: 99 cells
differ, each by exactly one ULP — so the barrier is load-bearing. Fused Mojo vs
a reference built WITHOUT `-ffp-contract=off`: also 0 of 2658, because both
toolchains lower the fused form to the same arm64 `fmadd`.

**AMENDED 2026-08-21, WHEN THE DEVICE ARRIVED: the rescale is now an EXPLICIT
`fma`, on both sides, and the un-fused form is gone.**

The entry above chose unfused on the argument that the fused form's agreement
was an accident of two compilers agreeing on one target while the unfused form
is determined by the source. That argument was sound on the HOST and did not
survive contact with a GPU backend.

**What was measured.** The score kernel's draw and the host's disagreed in
**9 of 105 scored cells**, worst 14 ulps (8 ulps over a 2,048-draw sweep), and
the device's value was ALWAYS exactly `fma(res, span, start)` and never a third
value. Six source-level barriers were tried and every one fused anyway on
device: plain, `@no_inline`, an integer bitcast round trip, both together, and
a private `stack_allocation` round trip — which fused on device AND made the
host fuse. One shared-memory round trip looked like a fix in a probe and was an
artefact of the probe using the product twice.

**So the choice was never fused against unfused.** It was "fused on device and
unfused on host" against "fused on both". An explicit `fma` is ONE IEEE
operation, fixed by the source on every backend — strictly more determined than
either — and it is also what RAFT's own expression becomes under nvcc's default
`--fmad=true`, which is to say **the fused form is what the upstream actually
computes on the hardware they ship for.** The unfused form was, in hindsight,
the one that diverged from them.

**What moved with it.** `tools/rng_oracle/main.cpp` now writes `std::fma` and
its `-ffp-contract=off` flag is no longer load-bearing;
`extratrees/tools/rng_oracle/pcg_reference.txt` was regenerated and
`pcg_rng_check` is green on all 2,658 cells against it. Thresholds shift by up
to one rounding, so trees drawn before and after this change differ — which is
the cost, paid once, of host and device agreeing at all.

**Price now.** One rounding against a `float64` sklearn, which deviation 130
already puts out of reach, and nothing else. The `@no_inline` helper and the
build flag are both deleted.

---

## 146. [CLOSED] `TreeMetaDataNode` is reduced: `train_time` is not ported

**Theirs.** `decisiontree.hpp:101-109` has seven members, including
`double train_time`, a wall-clock figure the builder stamps and the text/JSON
dumps print.

**Ours.** Six members; `train_time` is absent.

**Why.** Timing is deferred in this lane by explicit instruction, so a seconds
field would be written by nothing and read by nothing, and rule 3 says an
unported thing must be VISIBLE rather than present-but-dead. It is also the one
member that could not survive a move to device memory (no `float64` on device).

**Price.** `get_tree_summary_text` / `get_tree_json` (`decisiontree.hpp:119`,
`:139`) would print a train time we cannot print. No such dump exists here yet,
so the price is currently zero and becomes one line when there is one.

**NOT a deviation:** `std::vector` -> `List` (same contiguous owning array),
and the member ORDER differing, because nothing indexes this struct by offset;
the node array's own layout is untouched.

---

## 147. [CLOSED] `predict_one` accumulates and never zeroes; we mirror that and add a zeroing entry point beside it

**Theirs.** `decisiontree.cuh:410-412` is
`preds_out[i] += tree.vector_leaf[idx * num_outputs + i];` — `+=`, not `=` — and
`predict_all` (`:382-392`) does not zero `preds` either. The reason is one level
up: `RandomForest::predict` declares `std::vector<T> row_prediction(num_outputs)`
INSIDE the row loop (`randomforest.cuh:229`), which value-initialises to zero,
calls the tree predictor once per tree into that same buffer (`:231-237`), then
divides by `n_trees` (`:240-242`). **The accumulation across trees IS the forest
sum, and the only thing that ever zeroes the buffer is `std::vector`'s
constructor.** Hand `predict_all` a dirty buffer and it silently adds to garbage.

**Ours.** `predict_one_accumulate` / `predict_all_accumulate` are theirs, `+=`
and no zeroing. `predict_all` — the name a caller reaches for first — zeroes
`preds` and then calls the accumulating form.

**Why.** The "output must already be zero" contract is invisible in the
signature and documented nowhere in their header, and it is exactly the kind of
thing a check cannot see: a check that allocates a fresh zero buffer passes
under both behaviours.

**Price.** Two names where cuML has one. A lane porting `RandomForest::predict`
later MUST reach for the accumulating one or the forest sum is lost. Both
behaviours are checked: the check dirties the buffer, confirms `predict_all`
overwrites, then confirms a second accumulating pass exactly doubles.

---

## 148. [CLOSED] The node's template shape is narrowed: `LabelT` dropped, `IdxT` fixed to Int32

**Theirs.**
`template <typename DataT, typename LabelT, typename IdxT = int> struct SparseTreeNode`
(`flatnode.h:33`). `LabelT` appears in none of the five fields (`:36-40`) and
none of the accessors (`:52-57`); it exists only so `TreeMetaDataNode<T,L>` can
spell a matching node type. Both factories spell their return type
`SparseTreeNode<DataT, LabelT>` (`:62`, `:67`), **dropping `IdxT`** — so a
caller who instantiated with a non-default `IdxT` gets a different type back
from `CreateSplitNode`. Nobody upstream does, so it never fires.

**Ours.** One parameter carrying `DataT`. `LabelT` is not modelled. `IdxT` is
`Int32` everywhere, spelled out.

**Why.** `LabelT` is a phantom parameter Mojo would force onto every use site of
a struct that never reads it. Fixing `IdxT` removes their latent factory bug by
construction. `int` is 32-bit on every platform cuML builds for, so `Int32` is
what `IdxT = int` means, not a narrowing.

**Price.** No 64-bit node indices — but cuML cannot have them either, since
`left_child_id` is `IdxT` (`flatnode.h:39`), so their tree is already capped at
2^31 nodes whatever you pass. The cap is theirs; we only stopped pretending it
is configurable.

**Explicitly NOT a deviation, mirrored exactly:** their `LeftChildId()` /
`RightChildId()` return `int64_t` (`:55-56`) out of an `IdxT` field, and their
constructor takes `int64_t left_child_id` (`:42`) and narrows it into the
`IdxT` field (`:46`) unchecked. Ours does the same. Rule: transcribe, do not
tidy.

**Also recorded, and it is a behaviour we inherited rather than chose:** cuML's
argmax at a leaf (`randomforest.cuh:243-253`) initialises `best_prob` to `0.0`,
not `-inf`, and tests strictly greater while scanning `k` ascending. So an exact
tie keeps the LOWEST class index, and a leaf whose scores are all `<= 0` returns
class 0 by default without the comparison ever firing. Ported as written, and
both cases are in the check.

---

## 149. [CLOSED] Fixtures are counter-based, not stream-based

**Theirs.** sklearn's own tests build data with `numpy.random.RandomState`, a
sequential MT19937 stream: value *k* depends on every value drawn before it.

**Ours.** Every value is `splitmix64` of `(seed, row, col, salt)` — a
counter-based hash. Cell `(r, c)` depends on nothing but its own coordinates.

**Reason.** A sequential stream cannot be reproduced by a parallel generator
without materializing the whole stream in traversal order, so a device-side
fixture and a host-side fixture would have to agree on an ordering the device
does not have. This is deviation 130's discipline applied to the data instead
of the draws.

**Price.** These fixtures are NOT byte-comparable with anything a
`RandomState` seed produces, so no check here can be cross-checked against a
stored numpy array. Nothing needs that: the analytic fixtures are checked
against a closed form and the hashed ones against their own stated statistical
properties.

---

## 150. [CLOSED] No fixture contains a missing value

**Theirs.** `find_min_max` counts NaNs into `n_missing`
(`_partitioner.pyx:152-155`); the constant test is `... and n_missing == 0`
(`_splitter.pyx:617`); missing values go left or right by a per-candidate coin
flip (`_splitter.pyx:649`).

**Ours.** No fixture contains a NaN, because deviation 136 refuses missing
values by name at the API boundary.

**Reason.** A fixture for a refused path is a fixture for code that does not
exist, and manufacturing one now would freeze a guess at how the coin flip gets
keyed.

**Price, stated so it cannot be forgotten.** The `n_missing == 0` conjunct of
the constant test HAS NO FIXTURE COVERAGE, and neither does
`missing_go_to_left`. If 136 is ever reversed, this file grows a NaN shape and
this entry is REWRITTEN, not annotated.

---

# Findings from sklearn's source that change how this lane must be written

Not deviations — corrections to what the plan assumed, found by reading
`~/CascadeProjects/upstream/scikit-learn` at `1.9.0`. Recorded here because
each one silently changes an answer.

1. **The constant test has three arms, not one.** `_splitter.pyx:614-618`:
   `if end - start == n_missing or (max <= min + FEATURE_THRESHOLD and
   n_missing == 0)`. All-missing is ALSO constant, and a column that would
   otherwise be constant is NOT constant when it contains NaNs.
2. **That test's addition is FLOAT32, so the band widens and then VANISHES.**
   Both operands are `float32_t`. At `min == 0.5`, `0.5f + 1e-7f` rounds up to
   `0.5 + 1.192e-7`, so a column whose spread is `1.19e-7` — comfortably above
   `1e-7` — is still reported CONSTANT. And further out it does not widen at
   all: at `min == 3.25` one ulp is `2.384e-7`, half an ulp exceeds
   `FEATURE_THRESHOLD`, and `3.25f + 1e-7f` rounds straight back to `3.25`, so
   **the test degenerates to `max <= min` for every feature whose magnitude is
   above about 2** — a column at that scale is "constant" only if it is exactly
   constant. Both facts are asserted in `range_draw_check`, and the second was
   found by a SABOTAGE: tightening the test to `<` made the all-3.25 constant
   column drawable, which is only possible if `min + FEATURE_THRESHOLD == min`.
   **A port that computed this in float64 would both miss constants near zero
   and call constant what sklearn splits further out.**
3. **`rand_uniform` can return `high`.** `_utils.pyx:57-61` is
   `(high-low) * our_rand_r() / RAND_R_MAX + low` and `our_rand_r` can return
   exactly `RAND_R_MAX`. sklearn guards it at `_splitter.pyx:653-654`:
   `if threshold == max_feature_value: threshold = min_feature_value` — the
   draw is replaced by **min**, not by max, turning a degenerate
   everything-left into only-the-min-valued-rows-left. **Any splitter port
   must carry that guard.**
4. **`Gini` does NOT override `proxy_impurity_improvement`.** Only the base
   (`:147`), `MSE` (`:944`) and `Poisson` (`:1556`) define it. So the
   classification quantity sklearn actually maximizes is
   `-n_R*gini_R - n_L*gini_L`, NOT the `sum_c^2/n_L + sum_c^2/n_R` form ports
   usually assume. The two differ by the constant `-n_total`: harmless for an
   argmax within one node, **not** harmless for any check that compares a
   number, and not harmless for a `min_impurity_decrease` threshold.
5. **`min_samples_leaf` rejection is a `continue`, not a redraw**
   (`:659-670`). A rejected candidate feature is discarded for that node; the
   threshold is not re-sampled.
6. **`max_features` is not a hard cap.** The loop condition (`:571-576`) is
   `n_visited_features < max_features OR n_visited_features <=
   n_found_constants + n_drawn_constants`, so it keeps drawing past
   `max_features` until at least one non-constant feature has been drawn.
7. **`find_min_max` initialises on the first non-missing value and then uses
   `if v < min: elif v > max:`** (`_partitioner.pyx:150-163`) — an `elif`, not
   two `if`s. Equivalent for non-NaN input; a device reduction must not assume
   otherwise for NaN-bearing input.

---

# Toolchain traps found in this lane

**`&+` SILENTLY PARSES AS BITWISE AND WITH A UNARY PLUS.** `x &+ k` compiles
and computes `x & k`. Measured: with `x = 0xF0F0F0F0F0F0F0F0` and `k = 0xFF`,
`x &+ k` and `x & k` both print 240 while `x + k` prints
17361641481138401775. This was a REAL DEFECT in a hash written in this lane —
rows collapsed onto each other wherever the constant had a clear bit — and it
produces a wrong answer rather than a compile error. `&*` is a hard parse
error. Plain `+` and `*` on `UInt64` already wrap (`UInt64.MAX + 1 == 0`,
measured), so plain operators are the only correct spelling.

**`Float32.from_bits` and `SIMD.bitcast` do not exist.** The reverse of
`to_bits` is `from std.memory import bitcast` then
`bitcast[DType.float32](u32)`.

**`Int128` MULTIPLIED INSIDE A DEVICE KERNEL DOES NOT COMPILE.** A kernel
containing `Int128(a) * Int128(b)` fails at Metal pipeline-state creation with
`Compilation failed due to an interrupted connection:
XPC_ERROR_CONNECTION_INTERRUPTED. This error occurred after multiple retries.`
The identical kernel with `Int64` products builds and runs, so the 128-bit
multiply is the failure and not the harness. Reproduced twice on 2026-08-21 in
a 35-line standalone probe, Mojo 1.0.0 (ed45d567). **The error message names a
connection, not a type**, which is what makes this expensive to diagnose — it
reads as a flaky toolchain rather than as an unsupported operation. Deviation
167 hand-widens from 32-bit limbs instead. Same shape of backend refusal as the
whole-struct kernel argument in deviation 162.

**`DeviceAttribute.WARP_SIZE` is not queryable on Apple** (`Attribute
"WARP_SIZE" not supported for Apple GPU`). `MAX_THREADS_PER_BLOCK` (1024) and
`MAX_SHARED_MEMORY_PER_BLOCK` (32768) do work. A check that wants the width
must catch and REPORT it as unchecked rather than skip silently;
`std.gpu.WARP_SIZE` is the compile-time value to build against.

**A struct field cannot expose `MutAnyOrigin`** ("struct fields cannot expose
AnyOrigin in their type"). Use `MutUntrackedOrigin` for non-owning views whose
lifetime is managed explicitly. `UnsafePointer` is deprecated in favour of
`Pointer`; the repo's `MutPointer[T, origin]` spelling is the one that compiles
without a warning.

**A field cannot be moved out of a live struct** ("destroyed out of the middle
of a value"), so an accessor that hands back an owned member returns `.copy()`.

---

## 143. [CLOSED] Histogram-free: the accumulators ARE the arguments

**Theirs (cuML).** Every objective takes
`(BinT* hist, IdxT i, IdxT n_bins, IdxT len, IdxT nLeft)`
(`objectives.cuh:52`, `:132`, `:225`, `:299`, `:379`). `hist` is a
prefix-summed histogram laid out `n_bins*class + b`; the left child is bin `i`
of that CDF and the right child is recovered as
`hist[n_bins*c + n_bins-1] - hist[n_bins*c + i]` (`:72-73`). `n_bins` exists
only to index that layout.

**Ours.** `(BinT* hist_left, BinT* hist_total, IdxT len, IdxT n_left)`. The two
pointers are the left child's accumulators and the node's totals; the right
child is still recovered as total minus left, exactly as theirs is. `i` and
`n_bins` are gone.

**Why.** Deviation 137 deletes the histogram. A random threshold is drawn per
(node, feature) inside that node's own range; there is no bin index to pass and
no bin dimension to stride. Keeping their parameters would mean passing `i=0,
n_bins=1` forever, which is a lie in the signature.

**Price.** The signature no longer diffs against theirs line for line; the body
still does, which is where the arithmetic lives. `Gain` (`:85-96`) has no
counterpart at all, because it is the loop over the dimension we deleted.

---

## 144. [CLOSED] Classification selects on an EXACT INTEGER proxy

**Theirs — and they are two different quantities, which is the part a
from-memory port gets wrong.**

- cuML's `GainPerSplit` (`objectives.cuh:52-83`) computes the Gini impurity
  DECREASE in `DataT`: `sum_j[l_j/nL * l_j/n + r_j/nR * r_j/n] - sum_j(t_j/n)^2`.
- sklearn's random splitter compares `proxy_impurity_improvement`
  (`_splitter.pyx:693` over `_criterion.pyx:147-163`), which over
  `Gini.children_impurity` (`:647-687`) is `sq_L/nL + sq_R/nR - n`: the parent
  term dropped and the `1/n` scale dropped.
- The relation is `cuML_gain == parent_gini + sklearn_proxy / n`. Same argmax
  within a node, different VALUE, different float tie sets, and only sklearn's
  is an exact rational in integers.

**Ours.** `ProxyImpurityExact` returns sklearn's proxy as an exact rational in
integer arithmetic — numerator `sq_L*nR + sq_R*nL`, denominator `nL*nR` — and
`CompareProxyExact` orders two candidates by `Int128` cross-multiplication. The
float forms of both are kept, as exact transcriptions, for reporting.

**Why.** There is no `float64` on device, and deviation 135 leaves regression
accumulation open. For classification the question need not be open at all:
with sklearn's default `sample_weight=None` the class counts are INTEGERS, so
the proxy is an exact rational and the device can be exactly right rather than
approximately right. That also removes a whole class of "the GPU picked a
different split" investigation from the identity work.

**The width, derived rather than guessed.** `sq_L <= nL^2`, so
`num = sq_L*nR + sq_R*nL <= nL*nR*n <= n^3/4` and `den = nL*nR <= n^2/4`; the
cross-multiply `num_a * den_b` is bounded by `n^5/16`, NOT `n^3`. `Int128`
gives `n <= 2^26.2`, so the guard is `MAX_ROWS_EXACT = 2^26 = 67,108,864` rows
in one node, with one bit of headroom. Accumulators stay `Int64` (`<= 2^52`).
The `debug_assert` on that bound was PROVED LIVE by running the check under
`mojo run -D ASSERT=all` at `2^27` and watching it fire — a `debug_assert`
nobody has seen fail is not a guard.

**Also.** The exact form guards the empty child, which cuML's float form does
not: with `min_samples_leaf == 0` their `invLeft = One/nLeft`
(`objectives.cuh:57`) is `+inf` and the gain is NaN. That is transcribed as
written rather than fixed, and the check asserts the NaN, because it is theirs.
sklearn cannot reach the case — `min_samples_leaf >= 1` by validation.

**Price.** Weighted samples are out of scope for the exact path.
`sample_weight` is now listed in `NOT_IMPLEMENTED.tsv`.

---

## 145. [CLOSED] The exact comparator is the authority, not `Split.best_metric_val`

**Theirs.** cuML reduces candidates through `Split::update`
(`split.cuh:76-90`), whose first test is
`other.best_metric_val > this->best_metric_val` on a single `DataT` field. The
score IS the reduction key.

**Ours.** For classification the reduction key and the score are different
things. `Split.best_metric_val` stays `Float32` and stays cuML's `GainPerSplit`
value, for reporting and `feature_importances_`; the candidate ORDER is
`CompareProxyExact`.

**Why.** Two candidates whose exact proxies differ can round to the same
`Float32` — 24 mantissa bits against counts reaching 2^26 — and `Split::update`
then falls through to its `colid` tie-break and picks by feature index. That is
a silent, data-dependent divergence from sklearn's argmax. Deviation 133
accepted a different tie-break for GENUINE ties, which is a much smaller claim
than accepting one for near-ties.

**Price, and it is a contract on code not yet written.** The classification
reduction must carry the four accumulator fields (`sq_L`, `nL`, `sq_R`, `nR`)
alongside the `Split`, not just the score. **A score pass that reduces
classification candidates on the float field alone is using the wrong
comparator.** Cost unmeasured, deliberately.

---

## 151. [SUPERSEDED] The supplied column list IS the search; sklearn's "keep drawing until one is non-constant" is gone -- **CLOSED by DEVIATION 205**

**Theirs.** `_splitter.pyx:573-577`, the loop guard, holds two separate facts:
(a) it stops early when every remaining feature is known constant; (b) the
second disjunct keeps drawing PAST `max_features` for as long as every feature
drawn so far was constant, up to all `n_features`. The guarantee that buys: if
any non-constant feature exists anywhere, sklearn finds one and the node
splits.

**Ours.** Evaluate every supplied `colid` exactly once, in the supplied order
(which must not matter, and a check proves it does not). No re-draw. If every
supplied column is constant, the node is a leaf.

**The analogue, stated precisely.** `len(colids)` plays `max_features`. Fact
(a) is subsumed — a constant column costs one range pass and no score pass.
**Fact (b) has no analogue and is genuinely gone.**

**Why it cannot be emulated.** The extension is a property of a sequential
sampler that still holds un-drawn features. cuML's samplers commit to
`n_sampled_cols` per node on the device before any range is known
(`kernels/builder_kernels.cuh:152`, `:246`, launched at `builder.cuh:427`);
nothing downstream can ask for more. Re-invoking the sampler under a different
key appears in neither upstream, so it would be invention. **Declined**, and
the decline is priced below rather than left open.

**Which upstream this sides with.** cuML's: their kernel scores the sampled
`colids` and a node with no valid split becomes a leaf via `split_not_valid`.
Not a third, invented behaviour.

**Price.** A node all of whose SAMPLED columns are constant becomes a leaf
where sklearn would keep drawing. This compounds with deviation 132. On data
with many constant columns our trees are SHALLOWER than sklearn's at the same
`max_features` — a quality difference, not a rounding one. `n_constant` is
reported per node so a builder can count how often it bites.

---

## 152. [CLOSED] The candidate scan COUNTS; only the winner is partitioned

**Theirs.** Every candidate calls `partition_samples` (`_splitter.pyx:656-659`),
which reorders `samples[start:end]` and returns `pos`; `n_left = pos - start`;
the criterion then accumulates over the freshly partitioned array. After the
loop they re-partition if the last candidate examined is not the winner
(`:705-710`).

**Ours.** One pass per candidate over the node's rows, counting and
accumulating, moving nothing. The winner is handed to cuML's
`partition_samples` (`builder_kernels_impl.cuh:43-88`) by the caller, once.

**Why, and it is forced rather than chosen.** cuML's `partitionSamples` takes
`split.nLeft` as an INPUT (`:52`), so it cannot discover the count. Something
has to count first, and that is the score pass, which walks the rows anyway.
Their `:705-710` re-partition then has nothing to undo.

**Price, two parts.** (1) Row order inside a child is cuML's, not sklearn's —
already deviation 134. (2) **The accumulation ORDER differs**: sklearn sums the
left child in post-partition order, we sum in `row_ids` order skipping
right-going rows, and the device will use a third order (a tree reduction).
That is nil for classification, where the accumulators are integers
(deviation 144), and it is exactly the ground deviation 135 rules on for
regression — which is why the accumulator type is a parameter and the host
oracle drives it at `Float64`.

---

## 153. [CLOSED] Regression selects on sklearn's MSE proxy and REPORTS cuML's gain

**Theirs, and they are two different quantities.** sklearn selects on
`sum_L^2/n_L + sum_R^2/n_R` (`_splitter.pyx:691` over `_criterion.pyx:944-973`).
cuML selects on `0.5/n * (that - S^2/n)` (`objectives.cuh:225-244`). The two are
affine with positive slope and a within-node constant offset: same argmax,
different value, different float tie sets.

**Ours.** The reduction key is sklearn's proxy. `Split.best_metric_val` carries
cuML's `GainPerSplit` for reporting, for `min_impurity_decrease` (whose
threshold in `split_not_valid` is scaled to THEIR gain) and for
`feature_importances_`.

**Why this is not just deviation 145.** 145 is argued from an exact integer
comparator existing for Gini. None exists for MSE. This is a choice between two
FLOAT quantities with the same argmax, settled by which upstream is the spec
for the split rule: this file transcribes `node_split_random`, so it compares
what `:691` compares.

**Price.** On a near-tie the two forms can disagree and we take sklearn's
feature rather than cuML's. Both numbers are carried per candidate, so the rate
is countable whenever someone wants it. Unmeasured, deliberately.

---

## 154. [CLOSED] Two of sklearn's four rejection branches are absent because their inputs do not exist

**Theirs.** Four rejections, in order: `min_samples_leaf` (`:664-666`),
`min_weight_leaf` (`:674-677`), monotonicity (`:679-689`), and the `>` at
`:693`.

**Ours.** The first and the last only.

- `min_weight_leaf` is unreachable in the only configuration this port
  supports: `sample_weight=None` makes `weighted_n_left == n_left`, and
  sklearn's own `min_weight_fraction_leaf=0.0` makes the threshold zero.
  `sample_weight` is already in `NOT_IMPLEMENTED.tsv`.
- `monotonic_cst` has no field in `DecisionTreeParams` and no counterpart in
  cuML at all.

**Why not refuse by name, the way deviation 138 refuses `max_n_bins`.** There
is no name to refuse: neither parameter exists in this port's parameter struct.

**Price, and it is a real gap with an owner.** A user arriving from sklearn who
sets `min_weight_fraction_leaf` or `monotonic_cst` gets no error from this
layer. **Whoever writes the Python binding owes both names an explicit
refusal**; this entry is the record of that debt.

**Also recorded here because it is the same shape.** `Split.update`'s third
arm, the `quesval` tie-break (`split.cuh:86-88`), can only fire on equal
`colid`s, which a sampler never produces. The only reachable case is a caller
supplying a duplicated column, and then the two candidates tie in every field
and the incumbent survives. The check pins that behaviour as one cell rather
than pretending the arm is exercised.

---

## 156. [CLOSED] The `k == n` guard moved into the sampler, and materializes an identity

**Theirs.** The guard is in the CALLER: `builder.cuh:399`,
`if (dataset.n_sampled_cols != dataset.N) { ...sample... }`. When they are
equal neither kernel launches and `colids` is left untouched — holding whatever
the previous batch wrote. The consumer then tests the same condition itself
(`builder_kernels_impl.cuh:250-254`) and uses `colStart + blockIdx.y` directly
instead of reading `colids`.

**Ours.** `plan_feature_sampling` reports a third arm, `SAMPLE_ALL_FEATURES`,
and `sample_features` fills `colids` with `0..k-1` for every work item.

**Why.** The consumer branch does not exist in this lane yet — the score pass
of deviation 137 is unported — so there is nothing to carry their two-place
test. One place keeps `colids` meaning the same thing on every arm, which is
what lets one check assert the same properties across all three.

**Why it is safe.** The candidate column set is identical either way: theirs is
`0..N-1` implicitly, ours is `0..N-1` written down.

**Price, and it is a trap for whoever writes the device version.** One write of
`len(work_items) * n` int32s that theirs does not do, on the one configuration
where sampling is off. When the device version lands, whoever writes it must
decide whether to keep the fill or restore their two-place branch — **and if
they restore it they must restore BOTH halves**: the guard alone, without the
consumer's `if`, reads an uninitialized `colids`.

---

## 157. [CLOSED] The block collectives are explicit loops, and the block width is a parameter

**Theirs.** `excess_sample_with_replacement_kernel` is one CUDA block per node
over a `BLOCK_THREADS x MAX_SAMPLES_PER_THREAD` register array, using three CUB
block-wide collectives — `BlockRadixSort::Sort` (`:224`),
`BlockAdjacentDifference::SubtractLeft` (`:231-232`) and
`BlockScan::ExclusiveSum` (`:237`) — sharing one `union` of temp storage
(`:216-221`) with a `__syncthreads()` between each.

**Ours.** A host function over one flat `List` of
`block_threads * max_samples_per_thread` slots in THEIR blocked arrangement,
with the sort done by Mojo's `sort`, the adjacent difference by a descending
loop mirroring CUB's own write order, and the exclusive sum by the running
prefix it is defined to be. `block_threads` and `max_samples_per_thread` are
ARGUMENTS rather than template parameters, so the dispatch's choice between
instantiation 1 and instantiation 72 is a value a check can enumerate.

**Why.** The same reason `partition_samples` is a host function: this is the
ORACLE the device kernel is checked against, and `PORTING_RULES.md` 0b-ii says
an oracle is not a CPU path and stays. Both samplers are deterministic given
the block width — nothing depends on warp scheduling — so the host form
reproduces the device form's exact output, not merely an equivalent sample.

**Why the sort substitution cannot change an answer.** `BlockRadixSort::Sort`
sorts KEYS ONLY here (`:224` passes one array) and the keys are the column ids
themselves. Two equal keys are indistinguishable, so every ascending sort
produces the identical array; there is no payload whose permutation could
differ. This is the one collective not transcribed instruction for instruction,
and it is the one where that is provably free.

**Price.** `n_slots` int32s of host memory per block instead of registers, and
an O(n log n) sort where theirs is a radix pass. No timing claim attached. The
device kernel will need `max.gpu.primitives.block` and a real block radix sort,
which is a KERNEL MATRIX row.

---

## 158. [CLOSED] `log`, `exp` and `ceil` come from libm through FFI, not `std.math`

**Theirs.** `raft::log` / `raft::exp` (`raft/core/math.hpp:324-331`) are
`std::log` / `std::exp` on host and `::log` / `::exp` on device, plus
`std::ceil` at `builder.cuh:420`.

**Ours.** `external_call["log"|"logf"|"expf"|"ceil"]`, the same libm the C++
host build links against, keeping the float/double split of their expressions
exactly.

**Why, and it is this repository's standing finding rather than caution.**
`std.math.log` carries ~5e-8 ABSOLUTE error against libm, enough to re-decide a
tie. **Both call sites turn a log into an INTEGER**: `n_parallel_samples_for`
takes `ceil` of a ratio of logs, so an error near an integer flips the sample
count and — at the measured boundaries 9216 and 128 — flips WHICH KERNEL RUNS;
`algo_l_sample` truncates `log(u)/log(1-W)` to an int, so an error near an
integer picks a different column. Neither is a rounding-quality question.

**Price.** Four un-inlinable calls per use and a dependency on the host libm.
The device version cannot use FFI and will have to re-establish this; the jump
line is the one to check.

---

## 159. [CLOSED] Index widths, and a bound on their unbounded loop

**Theirs.** `IdxT` is `int` throughout the sampler. The retry loop
`do { } while (n_uniques < k)` (`:188-241`) has NO iteration bound, and the
dispatch's inputs are trusted — `k > n` would make `log(1 - k/n)` a NaN and
`k < 1` is meaningless.

**Ours.** Column ids stay `Int32` (their `IdxT`, and the width `colids` is);
counts and loop indices are Mojo's 64-bit `Int`. `EXCESS_MAX_ITERATIONS = 1024`
bounds the do-while and RAISES instead of hanging, and `plan_feature_sampling`
REFUSES `k < 1` or `k > n` by name rather than returning a NaN-derived arm.

**Why the width choice is free.** `n_parallel_samples` is about
`n*ln(n/(n-k))`, so overflowing int32 needs `n` around 1e8 columns — a dataset
that cannot be held. Every value the two agree on is identical.

**Why the bound and the refusal exist.** This is a host oracle a check runs
unattended: **a hang is a failure mode a check cannot report.** cuML reaches
neither state because `max_features` is clamped into `[1, N]` before `doSplit`
sees it.

**Price, and it is a warning to the device port.** Three departures from their
control flow that must NOT be carried over blindly: an `Error` cannot be raised
from a kernel, so the device version needs the bound expressed some other way,
or dropped with the argument that the dispatch guarantees termination written
down.

---

# What porting the samplers found in cuML itself

Not deviations — findings about the upstream, transcribed rather than fixed
(rule 1: theirs is right and ours is wrong, and "right" means "what their file
does"). Each is ASSERTED in `feature_sampler_check.mojo`, so a later
"improvement" turns the check red.

1. **`SubtractLeft`'s tile predecessor is `mask[0]`, not an item**
   (`builder_kernels.cuh:231-232`). CUB computes
   `output[0] = difference_op(input[0], tile_predecessor_item)` on thread 0
   (`cub/block/block_adjacent_difference.cuh:393-419`), so the block MINIMUM is
   compared against the previous iteration's FLAG — a 0 or a 1 — and is marked
   a duplicate when it equals it. Measured consequence: at `n=2, k=1` their
   kernel returns column **1 for all 64 nodes and never column 0**; at
   `n=64, k=8` over 4096 nodes, column 0 is chosen **2 times against 512
   expected**. This is a real statistical defect in cuML.
2. **Slots past `n_parallel_samples` are filled with `n-1`, not left idle**
   (`:201-203`), so column `n-1` is in the block's sample on every iteration
   and wins whenever the loop stops at exactly `k` uniques — measured 662
   against 512 expected.
3. **The output is the `k` SMALLEST uniques**, not a random `k` of them
   (`:243-246`): the gather index is the prefix sum and everything `>= k` is
   simply not written.
4. **The items migrate between threads.** `BlockRadixSort::Sort` re-blocks the
   whole array, so after the first sort the slot a thread redraws is not the
   slot it drew. A port that kept each thread's samples thread-local is a
   different algorithm.
5. **algo-L's fill loop leaves `col` at `k-1`, not `k`** (`:293-301`).
   Textbook Algorithm L sets `i = k`; theirs relies on the `+1` in the jump.
   Writing `k` skips a column.
6. **algo-L's mixed float/double types decide the answer.** `logf(fp)/k`
   divides in FLOAT while `log(1-W)` is DOUBLE; the numerator is promoted and
   the quotient is `static_cast<int>`-truncated. Computing that line entirely
   in float, or entirely in double, moves the truncation boundary onto a
   different column.
7. **The two samplers key DIFFERENTLY.** Excess uses the fnv1a32
   three-component chain; algo-L uses a plain `(treeid << 32) | nodeid` bit
   pack. Using one chain for both passes every property test.
8. **There are THREE dispatch arms, not two.** The excess kernel is
   instantiated at two different `MAX_SAMPLES_PER_THREAD` (`builder.cuh:434`),
   and the two consume the RNG at different rates, so they return DIFFERENT
   column sets for the same key — measured, 158 of 160 slots differ at
   `n=1000, k=20`.

---

# The device range pass — deviations 160-163

The lane's reserved range 130-159 is full; the device work continues at 160.

## 160. [CLOSED] NOT a deviation: the range pass is bit-checkable and the histogram pass is not

**Theirs.** `computeSplitKernel` (`builder_kernels_impl.cuh:216-340`) makes one
pass over a node's rows per (node, feature) block and accumulates a HISTOGRAM.
That accumulator is a SUM — an exact integer count for classification, but for
regression a running sum of float LABELS, which rounds, so which bits a bin
ends up holding depends on the order the rows and the partial histograms
arrive in.

**Ours.** `node_feature_range_kernel` makes the same pass over the same rows
with the same `WorkloadInfo` flattening and the same strided loop, and
accumulates MIN, MAX and a NaN COUNT.

**Why this kernel can be checked bit for bit and theirs cannot.** `min` and
`max` on IEEE-754 floats are associative AND commutative EXACTLY: they select
one of their inputs and return it unchanged, so nothing rounds at any step and
no regrouping can change which input survives. A count is an exact integer sum
with the same property. So the device's block-strided, cross-block-merged
reduction and the host's sequential loop are OBLIGED to produce identical bits,
and a tolerance in the check would hide a defect rather than absorb float
noise. That is why `range_kernel_check` asserts on `.to_bits()` and never on a
difference.

---

## 161. [SUPERSEDED] The cross-block combine is a MUTEX MERGE, not their atomicAdd + signalDone

**Theirs** (`builder_kernels_impl.cuh:295-317`): a large node's blocks each
`BinT::AtomicAdd` their partial histogram into a global slot keyed by
`large_nodeid`, `__threadfence()`, then call `MLCommon::signalDone`
(`src_prims/common/grid_sync.cuh:238-247`), which atomically increments a
per-(node, feature) counter; the block that drives it to zero is "last" and
alone continues.

**Ours.** Each block publishes its partial range into the SAME output cell
under a per-(node, feature) spin mutex, merging with `min`/`max`/`+`. No block
needs to know it is last.

**Why — a platform wall, not a preference.** Their merge step is an ATOMIC ADD,
and a histogram bin has one; a RANGE does not. There is no portable device
`atomicMin`/`atomicMax` on `float32` — the CUDA idiom is a signed-magnitude bit
twiddle into an integer atomic, which is a different sequence on every vendor
and is exactly the inline `if apple` this tree forbids. And `__threadfence()` is
not expressible: Mojo 1.0 comptime-asserts that `threadfence` is
NVIDIA-only.

**What replaces it is not invented here either.** The spin is the translation
this repository has already established and enqueued twice: spin on an ACQUIRE
load until the mutex reads free, claim it with a WEAK RELAXED compare-exchange,
release with a store. Only thread 0 of a block ever takes the lock, so no two
threads of one warp contend and the spin cannot livelock a warp.

**Why it cannot change an answer.** The merge is `min`, `max` and integer `+`,
all exactly associative and commutative (160), so the order the mutex grants
the lock in is not observable. The multi-block sabotage is what proves the
merge runs at all.

**Price.** One `Int32` mutex per (node, feature) that the caller must zero, and
a serialized publish per block instead of a wait-free atomic. No timing number
is attached and none will be until the perf round.

---

## 162. [CLOSED] The output is a STRUCT OF ARRAYS, and `Dataset` is passed as its components

**Theirs.** `computeSplitKernel` takes `const Dataset<...> dataset` BY VALUE
(`:221`) and reads `work_items[nid]` and `workload_info[blockIdx.x]` as
whole-struct loads (`:238-241`).

**Ours.** The kernel takes `data`, `row_ids`, `m` and `n` as separate
arguments, and its output is four parallel arrays rather than an array of
`FeatureRange`.

**Why.** `PORTING_RULES.md` rule 4: a whole-struct load in a kernel kills the
Metal compiler, reproduced in a 25-line probe on 2026-08-19, with per-field
access through the pointer compiling and running. `WorkloadInfo` and
`NodeWorkItem` ARE still passed as their ported structs and read field by field
through the pointer — this round established that a TWO-LEVEL field read
(`work_items[i].instances.begin`) also compiles and runs on Metal, so the ban
is on the whole-struct load, not on nested access.

**The fourth array is ours and has no cuML counterpart.** `out_n_merges` is the
device's own report of how many blocks published into a cell, so a check can
ASSERT that a large node really was served by more than one block instead of
inferring it from host arithmetic. It is written unconditionally by the
shipping kernel, not behind a flag, **because a check that runs a different
binary from the one that ships proves nothing about the one that ships.**

---

## 163. [CLOSED] The empty range is carried IN the output as `min > max`, and NaN never reaches the reduction

**sklearn's.** `find_min_max` (`_partitioner.pyx:143-163`) initializes
`min = INFINITY, max = -INFINITY`, seeds BOTH from the first non-missing value,
counts NaNs into `n_missing`, and leaves the initial state untouched when every
value is missing.

**Ours, on device.** A thread that has seen no non-missing value holds
`(+inf, -inf)` — sklearn's own initial state, and the exact identity of
min/max. NaNs are tested with `v != v` and diverted to the counter BEFORE the
comparison, so no NaN is ever an operand of the reduction and `fmin`/`fmax`'s
NaN rule — the one way min/max could fail to be commutative, and therefore the
one thing that could break 160 — is unreachable. After the block reduction,
`blk_min > blk_max` is true exactly when the block contributed nothing, and the
merge writes the host's `(1.0, -1.0)` form so the OUTPUT CELL is a
correctly-formed `FeatureRange` after every merge, not only after the last one.

**Why that matters.** It is what makes the mutex merge closed over its own
output: no block needs to be identified as last in order to convert an internal
`(+inf, -inf)` accumulator into the published sentinel.

**The one thing a caller must not do.** Read `out_min`/`out_max` as numbers
without testing `min > max` first. `node_feature_is_constant` already reports
such a range constant, which is what `_splitter.pyx:611-618` gives an
all-missing column, so the normal path is safe; a caller that draws a threshold
from the raw pair is not.

---

# DO NOT PORT BUGS — Andrew, 2026-08-21

The rule this lane was operating under was `PORTING_RULES.md` 0b, COPY DO NOT
IMPROVE, plus 0c, "theirs is right and ours is wrong until their file says
otherwise". Read strictly, that required reproducing a defect in cuML's
sampler that made the learner unable to see a feature. Andrew's ruling:
**don't port bugs, fix them.**

The refinement, stated so it does not swing too far: **the upstream is the
authority on DESIGN, not on defects.** A behaviour of theirs is ported when it
is a choice — even an odd one — and fixed when it is demonstrably a mistake
that changes an answer. The bar for calling something a mistake is a
MEASUREMENT, not an opinion: both entries below carry the number that made the
call, and both have a check asserting the fixed behaviour so the fix cannot
silently rot.

## 164. [CLOSED] FIXED cuML BUG: the block minimum had no head flag, so column 0 was never sampled

**Theirs.** `builder_kernels.cuh:231-232` calls
`SubtractLeft(items, mask, CustomDifference<IdxT>(), mask[0])`. CUB does
`output[0] = difference_op(input[0], tile_predecessor_item)` on thread 0
(`cub/block/block_adjacent_difference.cuh:393-419`), so the block MINIMUM is
compared against the previous iteration's FLAG — a 0 or a 1 — and is marked a
duplicate when they are equal.

**Measured before the fix, which is what makes this a bug and not a taste:**

    n=2 k=1  ->  column 0 drawn 0 of 64 nodes
    n=3 k=1  ->  column 0 drawn 0 of 64
    n=4 k=1  ->  column 0 drawn 0 of 64
    n=8 k=1  ->  column 0 drawn 0 of 64

Never, not rarely. And the consequence at the learner level, on a two-column
fixture whose separating feature is column 0: accuracy **0.523** against
sklearn's exact **1.0**, mean depth 10.8 against their 3.4, mean leaves 70.5
against their 7.9 — the tree splitting on noise forever because it could not
see the signal.

**Ours.** `mask[0] = 1`, unconditionally. A minimum has no predecessor, so its
head flag is 1 by definition; this is what CUB's no-predecessor overload gives
and what the head-flag idiom means.

**After the fix, same measurement:** `separable_gap` accuracy **1.0**, mean
depth **2.99** against sklearn's [2.83, 4.07], mean leaves **6.52** against
their [6.15, 9.73] — inside their band on all three. `tie_pair` moved from
depth 5.6 to 2.59, also into band.

**Guarded by** `feature_sampler_check`, which now asserts the ratio for column
0 is above 0.75 of expectation and that both columns appear at `n=2, k=1`. The
section previously asserted the opposite; the old numbers are kept in it so
nobody re-derives them.

---

## 165. [CLOSED] FIXED cuML BUG: a slot that never drew voted anyway, as column n-1

**Theirs.** `builder_kernels.cuh:201-203` fills every slot past
`n_parallel_samples` with `n - 1` — a REAL column id — commented "indices that
exceed `n_parallel_samples` will not generate". They do not generate, but they
do get sorted, deduped and gathered, so column `n - 1` is present in the
block's sample on every iteration and is selected whenever the loop stops at
exactly `k` uniques.

**Measured before the fix:** column `n - 1` chosen **662** times against
**512** expected, a 1.29x over-representation of one specific column.

**Ours.** A non-drawing slot is filled with `NON_DRAWING_SENTINEL(n) == n`,
which is above every valid column id, and its head flag is cleared before the
prefix sum. It therefore contributes nothing to `n_uniques` and can never be
gathered: **a slot that did not draw casts no vote.**

**Why the sentinel rather than leaving the slot alone.** The array is sorted
as a whole and the slots migrate between threads (point 2 of the docstring), so
"which slots drew" cannot survive the sort as a position. A value above every
column id survives it, sorts to the tail, and is identified by one comparison.

**Guarded by** `feature_sampler_check`, which asserts column `n - 1` is under
1.25x expectation.

---

# The device split reduction — deviations 166-169

## 166. [CLOSED] The reduction carries the EXACT RATIONAL KEY beside the `Split`, and `Split.update` becomes its tie-break

**Theirs.** `warpReduce` (`split.cuh:92-105`) shuffles four fields and `update`
decides on the first it can. The score IS the key: one `DataT`.

**Ours.** The reduced payload is `SplitExact` — their four fields plus
`ExactKey(num, den, valid)` — compared by (1) `compare_exact_key`, deviation
145's cross-multiply, and then, ONLY on an exact tie, (2) `Split.update`'s
chain, unchanged and **called rather than re-transcribed**: the tie branch
copies its own `Split`, calls the checked `Split.update`, uses only the boolean
it returns as a predicate, then assigns the whole payload so the key travels
with its split. The shipping path therefore executes the one transcription
`split_check` already validates; there is no second copy to drift.

**Why it had to be carried at all.** Two candidates whose exact proxies differ
can round to the same `Float32` — 24 mantissa bits against counts reaching
2^26 — and a float-keyed reduction then falls through to `colid` and picks by
feature index. Measured in this fixture: **946 candidate pairs are bit-equal in
`Float32` and 80 of them have different exact proxies.** No single scalar can
stand in for the rational: `num/den` in `Float32` is the very rounding being
avoided, `Float64` does not exist on device, and normalising to a common
denominator hits the same overflow the cross-multiply was sized against.

**Regression is the same code with the key switched off.** A caller with no
exact key passes `ExactKey()` (valid = 0); every comparison then ties and the
order degenerates to `Split.update` — cuML's own reduction, exactly. One
kernel, no branch on task, and the check asserts the degenerate case equals a
plain `Split.update` fold.

**The order is total, with an exact condition.** Lexicographic on (rational,
metric, colid, quesval); its only ties are payloads equal in all four, which
could still differ in `n_left`. Unreachable in the pipeline because (colid,
quesval) plus the node's rows determine `n_left` — and the check ASSERTS the
fixture contains no such pair rather than assuming it.

**Price.** 36-byte payload instead of 16, so seven shuffles per butterfly step
instead of four and 36 bytes of shared per warp (288 B at TPB 256, against a
queried 32768 B budget); two 64x64->128 multiplies per keyed comparison;
`Int64` lane shuffles. No timing number, and none until the perf round.

---

## 167. [CLOSED] The Int128 cross-multiply is HAND-WIDENED from 64-bit limbs, because Int128 in a device kernel does not compile

**Theirs.** None — they compare one float.

**Ours on host.** `Int128(a.num) * Int128(b.den)` against
`Int128(b.num) * Int128(a.den)`, exact to `MAX_ROWS_EXACT = 2^26` rows
(deviation 135's derivation).

**Ours on device.** The same comparison built from 32-bit limbs into a
(hi, lo) pair compared lexicographically: four 32x32->64 multiplies, two adds,
two shifts, nothing vendor-specific.

**Why — a MEASURED toolchain wall, not a preference.** A kernel containing an
`Int128` multiply fails to BUILD: Metal pipeline-state creation dies with
`Compilation failed due to an interrupted connection:
XPC_ERROR_CONNECTION_INTERRUPTED. This error occurred after multiple retries.`
The identical kernel with `Int64` products builds and runs, so the 128-bit
multiply is the failure and not the harness. Reproduced twice on 2026-08-21,
Mojo 1.0.0 (ed45d567), in a 35-line standalone probe. Same shape of Metal
backend refusal deviation 162 records for a whole-struct kernel argument.

**Not an invention under the vendor rule.** There is no MAX entry point and no
vendor intrinsic for a 128-bit integer product on any of the three targets —
the TYPE is what the backend refuses, so there is nothing to call.

**Why it is safe to believe.** The check runs it against the host's `Int128`
form pairwise, per cell: **490,000 ordered pairs, 0 disagreements**, over
products deliberately built to exceed 2^64 — and it counts how many were
decided by the high limb alone (487,790), so the wide path is demonstrably
exercised rather than assumed.

**Also ours:** the sign split covers a negative numerator, which Gini cannot
produce. It is there because deviation 153 leaves the regression key open and
an MSE numerator is not sign-constrained, and the fixture carries a mixed-sign
node so the branch is not dead code.

---

## 168. [CLOSED] `warpReduce` is a `shuffle_xor` butterfly with NO ASSUMED WARP WIDTH

**Theirs.** `for (int i = raft::WarpSize/2; i >= 1; i /= 2) { auto id = lane + i; ... }`
with `WarpSize` a compile-time 32.

**Ours.** The same loop with `warp.shuffle_xor` over `std.gpu.WARP_SIZE` — the
target's width, 32 on NVIDIA and Apple, 64 on AMD CDNA, never a literal.

**Why XOR rather than their `lane + i`.** For lane 0 the two are the same
permutation at every step (`0 + i == 0 XOR i`), so lane 0's result is
bit-identical to theirs and their "best split will be with 0th lane" note still
holds. They differ for the other lanes, where theirs leaves partials and XOR
leaves the full answer everywhere — a strengthening, and it matters because
this reduction is called twice in a row.

**Why not a MAX collective.** `max.gpu.primitives` offers `sum`/`max`/`min`/
`broadcast`/`prefix_sum` over a SCALAR; this reduces a 36-byte record under a
four-level lexicographic order and no MAX collective takes a user comparator.
The shuffle primitives ARE the vendor library here and they are what is called;
only the combine is ours, and the combine is `Split.update`.

**Price.** `log2(WARP_SIZE)` steps of seven shuffles — one more step on a
64-wide wavefront, which is correct behaviour rather than a cost.

---

## 169. [CLOSED] `evalBestSplit`'s publish: a portable mutex, a struct-of-arrays cell, and the device's own report of which path it ran

Same six steps in the same order as `split.cuh:107-152`. Three things are
spelled differently and each is forced:

**(a) The lock.** `threadfence` is comptime-asserted NVIDIA-only and Metal
rejects a strong compare-exchange by name, so the established translation is
used — already enqueued twice in this repo (cuVS's cross-block mutex, and
`node_feature_range_kernel` under deviation 161): spin on an ACQUIRE load,
claim with a WEAK RELAXED compare-exchange, hand back with a RELEASE store,
which is where their `__threadfence(); atomicExch()` goes. Only thread 0 of a
block ever takes it, exactly as theirs does, so no two threads of one warp
contend and the spin cannot livelock a warp.

**(b) The cell is seven arrays**, not one `volatile Split*` — deviation 162's
measured Metal failure on whole-struct pointer access. Their code is already
field-by-field because of `volatile`, so it is the same shape for a different
reason.

**(c) Two counters with no cuML counterpart**, written unconditionally by the
shipping kernel: `out_n_merges` (blocks that published) and `out_n_warps`
(warps that carried a valid candidate into the cross-warp combine). **A
reduction check that cannot NAME the path it ran can pass about a path it never
took** — with one warp per block the cross-warp step is a no-op that copies
lane 0's own value. The device reports the path; the host does not infer it.

**Why the publish order cannot change the answer.** The merge is `update` under
166's total order, and a maximum under a total order is independent of arrival
order — the same argument 161 makes for min/max, made for a lexicographic one.
The check proves it by PERMUTING (eight permutations x two block shapes, 1120
field comparisons, not one bit moved) rather than by asserting it.

**Price.** One `Int32` mutex per node the caller must zero, one init launch,
and a serialized publish per block instead of a wait-free atomic — which no
float rational could have used anyway.

---

# The device draw and score pass — deviations 170-175

## 170. [CLOSED] The score pass is TWO launches, because no block can be elected last

**Theirs.** `computeSplitKernel` elects a last block with `signalDone` plus
`__threadfence()` (`builder_kernels_impl.cuh:295-317`) and scores inside it.

**Ours.** A kernel boundary IS that fence: an accumulate launch, then a
finalize launch of one thread per cell.

**Why.** `threadfence` is comptime-asserted NVIDIA-only in Mojo 1.0, so no
block can be elected. The same wall deviation 161 hit.

**Why it costs no extra row work.** The second launch reads no dataset row —
three range loads, the constant test, one draw, and an `n_acc` loop — so the
pass over the node's rows still happens exactly once, which is the property
that mattered.

**Price.** One launch of `ceil(n_cells / 64)` blocks per level per objective.
No timing number attached.

---

## 171. [CLOSED] NOT a deviation: this merge IS their `atomicAdd`

Deviation 161 replaced their cross-block combine with a mutex because a RANGE
has no portable float atomic. This merge is an INTEGER ADD, which does have
one, so `:301` is transcribed directly: one `Atomic.fetch_add` per accumulator
per block, from thread 0. Order-independence is exact, which is what licenses
the bit-exact check — the same argument as 160, for a different operation.

Worth stating because the two kernels sitting next to each other with different
combine strategies looks like an inconsistency and is not: it is the difference
between an operation that has a portable atomic and one that does not.

---

## 172. [CLOSED] No dynamic shared memory: private accumulators plus a block reduction

**Theirs.** `extern __shared__` sized from `max_n_bins * num_outputs`
(`builder.cuh:497-505`), incremented with shared-memory atomics.

**Ours.** Two `stack_allocation[MAX_ACC, Int32]` PRIVATE arrays per thread and
one `block.sum` per accumulator. Mojo's `stack_allocation[..., SHARED]` is
comptime-sized, so their dynamic sizing has no counterpart.

**Why it is not a loss.** Nothing scatters here: deviation 137 deleted the bin
dimension, so the bound is over CLASSES alone. Their shared atomics existed to
serialise scatter into bins that no longer exist.

**Price.** `MAX_ACC * 2 * 4` bytes of private memory per thread and `n_acc`
block reductions per block. A launch with `n_acc > MAX_ACC` publishes nothing
and the cell stays `UNVISITED` — a refusal, not a truncation.

---

## 173. [CLOSED] The draw is an explicit `fma`, and it is why deviation 142 was amended

Every thread draws redundantly, because the key has nothing thread-local in it.

The substance of this entry is in the amendment to deviation 142 above: the
device fuses the rescale no matter how the source is written, six barriers were
measured, and the resolution is an explicit `fma` on both sides. **The
divergence was real before that: 9 of 105 scored cells differed between host and
device**, which is to say a device tree and a host tree were different models.

`node_feature_score_host` takes the draw as an explicit argument so a check can
run the oracle against EITHER form and count the divergence, rather than
assuming the question is settled.

---

## 174. [CLOSED] The device sees only INTEGERS, and a refusal is a STATUS

**Labels arrive pre-quantized.** Deviation 135 rules that regression
accumulates in fixed point, so the host quantizes and the device receives
`Int32`. The device therefore performs no float-to-int conversion at all and no
rounding mode enters the accumulation. Class ids are cast on the host for the
same reason.

**A kernel cannot raise**, so `_refuse_missing`'s error becomes
`SCORE_STATUS_MISSING_REFUSED`, in the same position in the branch order the
host raise occupies. The caller converts it back into an error (deviation 136).

**This discharges deviation 163's caller obligation**, and the discharge is a
proof rather than a promise: the two ways to hold `min > max` are an all-missing
column, which the refusal catches, and a zero-row node, which the constant
test's `0 == 0` arm catches. **No threshold is ever drawn from a `(1.0, -1.0)`
pair.**

---

## 175. [CLOSED] The published width is Int64, and it is TIGHTER than the host's Int128 bound

**Classification publishes `GiniProxyExact`'s `Int64` pair** and does not
compute cuML's float `GainPerSplit` on device: deviation 145 makes that a
reporting quantity, and computing it would drag deviation 142's FMA question
into scoring.

`num <= n^3/4` and `Int64` holds `2^63`, so the device bound is
`SCORE_MAX_ROWS_EXACT = 2^21`.

**AND THAT EXPOSED A BUG IN `objectives.mojo`.** Its `MAX_ROWS_EXACT` was
`2^26`, which is the correct bound for the `Int128` CROSS-MULTIPLY and the
wrong one for the `Int64` field the numerator is STORED in first. Measured:

    n = 2^21   num = 2305843009213693952   Int64 agrees
    n = 2^22   num = 18446744073709551616  Int64 reads 0   *** WRAPPED ***

It wraps to exactly ZERO, so every candidate in the node would tie at the same
numerator and the winner would fall to `Split.update`'s `colid` arm — decided
by feature index, silently. `MAX_ROWS_EXACT` is now `2^21` with the measurement
in its docstring. **A bound that permits garbage is worse than no bound**, and
this one was written by the same lane three commits earlier.

**MSE publishes no rational** — TRUE WHEN WRITTEN, FALSE NOW, and replaced
rather than annotated. Deviation 189 gives regression a publishable `Int64`
key, so `out_gini_num`/`out_gini_den` carry a real regression score and a
caller no longer forms one with `mse_proxy_exact`. The two field names are
misnamed on that path and are deliberately NOT renamed: the rename would cross
into `builder.mojo` and `score_kernel_check.mojo`, and a documented misnomer is
cheaper than a three-file rename mid-round.

---

# The device path, wired — deviations 182-183

## 182. [CLOSED] `score_to_candidate_kernel` exists because 170 split their kernel in two

**Theirs.** `computeSplitKernel`'s elected last block scores the bins and hands
the result straight to `sp.evalBestSplit(...)` in the same function
(`builder_kernels_impl.cuh:328-340`). The candidate never exists as memory.

**Ours.** Deviation 170 could not elect a last block, so the score pass ends at
a kernel boundary and its output is a struct-of-arrays in global memory.
Something must turn that into the reduction's input layout.

**Why elementwise and not fused into either neighbour.** Fusing it into the
finalize kernel would make that kernel write two layouts of the same fact;
fusing it into the reduction would make the reduction read a layout it does not
own. Both couple two ported files through a shape neither upstream has.

**The one piece of policy in it.** A cell whose status is not `SCORED` becomes
the DEFAULT `Split` — `colid = -1`, `best_metric_val = MIN_FINITE` — with an
invalid exact key. That is `initSplit`'s value (`split.cuh:54-59`), so a
non-scored cell loses to every scored one and ties with other non-scored cells.
A node all of whose candidates were constant reduces to `colid == -1`, which
`split_not_valid` rejects and `NodeQueue.push` turns into a leaf — the same
outcome the host reaches by never producing a candidate.

---

## 183. [CLOSED] The device applies cuML's zero-gain gate, from exact integers on the host

**Theirs.** `split_not_valid` (`kernels/builder_kernels.cuh:59-67`) rejects a
split whose `best_metric_val` is `<= min_impurity_decrease`. cuML's Gini gain
is `>= 0` always, so at the default `0.0` a ZERO-GAIN split — one that does not
reduce impurity at all — is rejected and the node becomes a leaf.

**Ours.** The device does not compute the float gain (deviation 175: it would
drag deviation 142's FMA question into scoring, and deviation 145 makes the
gain a reporting quantity for classification), so it publishes a constant and
the gate cannot fire.

**MEASURED, both ways, and this is what makes it a bounded gap rather than an
unknown.** `device_tree_check` runs the same fixture through both paths twice:

    gate DISABLED on the host:  9 configurations, 747 nodes,
                                0 differing in (colid, quesval, left_child,
                                instance_count), 0 differing row predictions
                                -- BIT-IDENTICAL
    gate at its DEFAULT:        9 configurations, 4 identical in node count,
                                5 where the device has MORE nodes, 0 where it
                                has fewer; 689 host nodes against 747 device

So the device split search reproduces the host's **exactly**, and this gate is
the whole of the remaining difference. The direction is asserted, not assumed:
the gate can only ever REJECT, so the device can only ever have more nodes, and
a single configuration where it had fewer would mean something else diverging.

**THE FIX, AND IT NEEDED NO FLOAT ON THE DEVICE AND NO 128-BIT COMPARE.**
Writing `sq_total` for the node's `sum_j t_j^2`:

    cuML gain = parent_gini + sklearn_proxy / n
              = (1 - sq_total/n^2) + (num/den - n)/n
              = num/(den*n) - sq_total/n^2

Every quantity on the right is an exact integer the score pass ALREADY
computed: `num` and `den` come back with the winning split, and `sq_total` and
`n` are the NODE's totals, which `out_acc_total` and `out_n_total` hold for
every scored cell of that node — the same values in each, because they are the
node's own class counts.

So the gain is formed on the HOST, in `Float64`, from integers that carry no
rounding, and `split_not_valid` applies unchanged.

**The first sketch of this fix was worse and it is worth recording why.** It
put the comparison on the device as `num * n > sq_total * den`, which is the
same inequality — but `num * n` reaches `2^82` at `SCORE_MAX_ROWS_EXACT`, so it
needed deviation 167's hand-widened 128-bit multiply inside a kernel. Bringing
three small arrays back per level instead makes that unnecessary: `Int128`
works fine on the host. **The cheaper fix was the one that moved less code, not
the one that stayed on the device.**

**Measured, before and after, on the same fixture:**

    before:  689 host nodes against 747 device nodes,
             4 of 9 configurations identical in node count
    after:   689 against 689, 9 of 9 identical,
             0 nodes differing in any compared field,
             0 of 2241 leaf values differing

**What is still not bit-identical, and it is smaller than what it replaced.**
The host's `best_metric_val` is cuML's `GainPerSplit` accumulated in `Float32`
from float reciprocals (`objectives.cuh:52-83`); the device path's is the same
quantity formed in `Float64` from exact integers. They agree far inside the
gate's resolution but not in the last bits, so `device_tree_check` still
excludes the FIELD while now requiring the DECISION it drives to agree.

**`min_impurity_decrease` is therefore honoured on the device path**, and the
one place in this lane where a parameter was accepted and not honoured is
gone.

---

# The device partition and leaf pass — deviations 176-181

## 176. [CLOSED] `partitionSamples` on device: their collectives, and a BOUND on their unbounded loop

**Theirs.** `builder_kernels_impl.cuh:43-88`, two `cub::BlockScan::ExclusiveSum`
calls per iteration (each returning prefix AND aggregate), a `smem` carve of
`2 * TPB` indices, and `do { } while` with no iteration bound.

**Ours.** `block.prefix_sum[exclusive=True]` plus `block.sum` — the aggregate
must be block-uniform because `llen`/`rlen` drive the loop condition, and
`sum`'s default `broadcast=True` gives that. Their `smem` becomes ONE
`stack_allocation[2*TPB]` split into `lcomp`/`rcomp`, one allocation ON PURPOSE
because two with identical comptime parameters may be aliased.

**The bound is DERIVED, not chosen.** `ceildiv(n_left, TPB) + ceildiv(n_right,
TPB) + 2`, proved from `minlen = min(llen, rlen)`, and an overrun is REPORTED
as `PARTITION_OVERRUN` rather than hanging. This discharges deviation 159's
explicit hand-off: a host oracle could raise, a kernel cannot.

**AND IT FALSIFIED A SENTENCE THIS LANE HAD ALREADY WRITTEN.** Deviation 134
and `partition_samples`' docstring say their algorithm is deterministic GIVEN
the block width. It is stronger than that: measured, **0 of 1024 slots differ
between TPB 32 and TPB 128** — the output order is independent of the width
entirely. The reason is structural: compaction packs flagged misfits in
ascending slot order, the first `minlen` are consumed, and a side not fully
consumed keeps its flags, so globally the k-th left misfit always swaps with
the k-th right misfit and `TPB` only decides how many pairs happen per
iteration. Reported, not gated on.

**Measured against the host transcription: 6144 slot comparisons, 0
differences** — the whole permutation, not an equivalent partition, with
328-358 of 1024 slots actually moving so the match is not vacuous.

---

## 177. [CLOSED] NOT a deviation: one block per node is kept

Their header says so (`:39-41`) and `nodeSplitKernel` launches
`<<<work_items_size, TPB>>>`. A multi-block partition needs a grid-wide barrier
per iteration — `__threadfence` plus a done counter — which deviations 161 and
170 both measured as inexpressible.

The cost is recorded as an ITERATION COUNT, published by the kernel every
launch, not as a time: 6 / 4 / 2 items ran more than one iteration at TPB
32 / 64 / 128.

---

## 178. [CLOSED] The leaf histogram is a comptime-bounded private array plus one block reduction

**Theirs.** `extern __shared__` sized `BinT[num_outputs]`, incremented with
shared atomics (`:398-412`).

**Ours.** A `stack_allocation[MAX_OUT]` PRIVATE array per thread and one block
reduction per output. Same comptime-sizing wall as deviation 172; private
rather than shared because with one bin per class nothing scatters.

The answer is unchanged because every accumulator is an integer.

---

## 179. [CLOSED] RULED: the device regression leaf is FIXED POINT, and the Float64 host form is the reporting one

**The situation.** `builder.mojo::set_leaf_predictions_regression` accumulates
in `Float64` to match sklearn. There is no `float64` on device, and a `Float32`
accumulator through a block reduction would not be bit-checkable at all.

**The ruling, and it is deviation 135 applied rather than a new decision.** 135
already says the device accumulates in fixed point and the host oracle runs at
`Float64` "matching sklearn, while the device runs over quantized integers".
The leaf value is an accumulation like any other, so the device leaf is fixed
point and `leaf_values_host` — the fixed-point host form — is its oracle,
bit-identical. `set_leaf_predictions_regression` stays as the sklearn-matching
REPORTING form.

**The gap is measured every run rather than assumed small**: 0 of 7 leaves
bit-identical between the two forms, largest disagreement 7.629e-06, **exactly
half the quantization step**. That is the resolution the ruling buys
order-independence with.

**A fixture defect found on the way, and it is the instructive part.**
Regression labels of the form `hash/64.0 - 64.0` are all exact multiples of the
`1/65536` scale, so this measurement first came back "0.0 disagreement, 7 of 7
bit-identical" — **a fixture that could not round, reporting that nothing
rounds.** The divisor was changed to 97 and the real number appeared.

---

## 180. [CLOSED] One leaf launch over all nodes, and two device-written path reports

**Theirs.** `SetLeafPredictions` batches launches at 100,000 nodes
(`builder.cuh:559`), which is a HOST MEMORY-BUDGET policy, not an algorithm
step — their comment says "to reduce peak memory usage in extreme cases".

**Ours.** One launch over all nodes, plus `out_visit`, a device-written report
of whether each node PUBLISHED or returned early at `IsLeaf`. Measured: 8
published, 7 internal, 0 unvisited — **both branches named by the device**, not
inferred from host arithmetic.

**The caller obligations this creates are contracts, not conventions**: the
leaf array must be ZEROED before the launch (`builder.cuh:582`, and an internal
node's zero IS its value because the kernel never writes those slots), and
`out_visit` must be zeroed so `LEAF_VISIT_NONE` means what it says.

---

## 181. [CLOSED] `volatile IdxT* row_ids` has no Mojo spelling; `barrier()` carries it

**Theirs.** `partitionSamples` casts `row_ids` to `volatile IdxT*` (`:51`).

**Ours.** `barrier()`, which is STRONGER: `volatile` orders one thread's own
accesses, while the cross-thread ordering their algorithm actually depends on
comes from CUB's internal `__syncthreads()`, which `barrier()` is.

The two places it is load-bearing are named in the file so nobody tidies a
barrier away as redundant.

---

# The forest and the estimator on device — deviations 184-188

## 184. [CLOSED] The dataset is uploaded once for the FOREST, not once per tree

**Theirs.** cuML's `Dataset` holds device pointers for the whole fit
(`dataset.h:22-38`); every tree reads one resident copy.

**What ours did for one round.** `train_classification_device` was written as a
whole-tree entry point with no forest above it, so it allocated and filled
`d_data` and `d_labels` on entry. An `n_trees`-tree forest uploaded the same
IMMUTABLE matrix `n_trees` times — `n_trees - 1` redundant copies of
`4*n_rows*n_cols + 4*n_rows` bytes, plus that many redundant host staging fills
and `synchronize()` points.

**It could never have been a wrong answer, only redundant traffic**, because
the matrix is immutable and every tree uploaded identical bytes. That is why it
was allowed to stand for a round rather than being rushed — and why closing it
required no result to be re-checked.

**Closed by splitting the function**, exactly as the forest lane specified:
`upload_dataset` is the old prologue, `train_classification_device_resident` is
the old body, and `train_classification_device` survives as a two-line wrapper
so `device_tree_check` — which fits ONE tree — goes on exercising the same body
rather than a copy of it. `fit_classification_device` hoists the upload above
its tree loop. Identity re-verified after the move: `device_forest_check` still
reports 0 differing nodes and 0 differing leaf values.

**Why it was not done in the round that created it.** `builder.mojo` was owned
by another session, and changing a converged file is the failure mode rule 12
names. The alternative — a forest-private copy of a 500-line function with a
different prologue — would have guaranteed drift. Waiting one round cost
nothing because the defect was traffic, not correctness.

---

## 185. [CLOSED] The per-tree `row_ids` is a contract, not a necessity, and that is MEASURED

`train_classification_device` declares `mut row_ids` and reads it once into a
host staging buffer. `node_split_kernel` then mutates `d_row_ids` ON THE DEVICE
across every level, and **nothing copies that permutation back**. So the `mut`
is currently vacuous, and a single shared buffer across all trees produces a
bit-identical forest.

**`device_forest_check` MEASURES that rather than arguing it**: a whole 12-tree
forest fitted from one shared buffer differs from the per-tree forest in **0 of
1034 nodes**.

**It is kept per-tree regardless, for two reasons that are not style.** The two
arms of the forest must be the same loop and the host arm NEEDS it; and the
moment the device partition is copied back — which a device-resident frontier
would do — a shared buffer becomes silent cross-tree corruption, and the tree
that noticed would be tree 1.

So the check PINS the fact that makes it safe: `row_ids` comes back from the
device trainer unchanged. **When that assertion goes red, this entry is stale
and the shared buffer has become dangerous.** A sabotage simulating the
write-back proves the pin fires.

---

## 186. [CLOSED] The class-id cast is hoisted, and it carries a range guard the host path does not have

The device sees only integers (deviation 174), so the forest casts labels to
`Int32` class ids ONCE rather than per tree — the cast depends on nothing that
varies across trees, which is the only reason it may be hoisted while the
dataset upload could not be.

**The range check is new and is NOT on the host path.** A class id outside
`[0, n_classes)` indexes the score kernel's `MAX_ACC`-wide accumulator, so it is
an out-of-bounds write rather than a wrong answer. This is a guard on a cast
this file owns, not a change to a ported file — and it means the two arms can
refuse different inputs: a label of `7.0` with `n_classes == 3` reaches the host
trainer and is refused here.

---

## 187. [CLOSED] The device is an ENTRY POINT, not a `device=True` field on the config

A boolean field would be wrong twice. `ExtraTreesConfig` is documented as
sklearn's constructor with sklearn's names and sklearn has no such parameter,
so the struct would become a mix of sklearn's surface and ours that the next
reader cannot separate; and a flag cannot carry a `DeviceContext`, which is a
signature difference no boolean hides.

**Refusals cannot drift between the arms STRUCTURALLY rather than by
discipline**: both call `classifier_plan`, the only place the criterion check
and `resolve`/`refuse_unported` are invoked, and both call `depth_cap_bound`.
Neither arm holds a line of policy.

Two refusals fire on the device arm ONLY — the device trainer's class-count
bound (deviation 172) and row-count bound (deviation 175) — raised out of the
first tree before any kernel is enqueued, and neither is restated in
`estimator.mojo` because restating a bound is how a copy drifts from its
constant.

**That asymmetry is also the reach proof.** The two arms' forests are
bit-identical, so no output can tell them apart; `device_forest_check` uses the
33-class refusal instead, and a sabotage pointing the device arm at
`fit_classification` left every identity assertion green and reddened only that
one cell.

---

## 188. [CLOSED] The estimator's device regressor: refused BY NAME, then CLOSED

**The refusal, as it stood.** When this entry was written there was no
`train_regression_device`: every device kernel supported regression — range,
draw, fixed-point score accumulation, partition, leaf — but the finalize
kernel published no exact rational for MSE, so `split_reduce_kernel` could not
rank regression candidates. `fit_extra_trees_regressor_device` existed only to
refuse: offering no symbol leaves a caller who found
`fit_extra_trees_classifier_device` to guess, and quietly forwarding to the
host regressor hands them a host fit under a device name — the same class of
defect as a parameter accepted and ignored.

**CLOSED 2026-08-21.** Deviation 189 gave the reduction an exact `Int64` MSE
key, deviation 206 brought the regression device path level with
classification and added `fit_regression_device`, and at that point the
refusal was a switch that had outlived its reason — the defect class the
Borders default flip already named. `fit_extra_trees_regressor_device` now
follows 187's shape exactly: both regressor arms call `regressor_plan` (the
new `classifier_plan` twin) and nothing else before their trainer, and the
device arm additionally derives deviation 135's quantization
(`quantize_labels`: whole-vector magnitude sum through `choose_scale`, the
same derivation `device_regression_check` uses) so callers need not know
fixed point exists.

**Measured, in `device_regression_check` section 4:** 3 trees through the
sklearn surface, 0 nodes differing in structure between the arms, 62 leaf
values moved by quantization and none past one quantization step. REACH is
proved by that movement rather than by a device-exclusive refusal (regression
has none): a device arm silently serving the host fit returns bit-equal
leaves, and the sabotage that did exactly that reddened only the reach
assertion, while a doubled trainer scale reddened only the one-step bound.
The shared-plan refusals (`bootstrap=True`, a classification criterion) are
asserted through the device arm as well.

---

# The regression key, and the tie-break — deviations 189-194

## 189. [CLOSED] The regression key is cuML's OWN MSE gain, not sklearn's proxy

**The problem.** sklearn's proxy numerator `sum_L^2*n_R + sum_R^2*n_L` over
sums bounded by the `2^30` slot needs `2b + log2(n) <= 63`, i.e. `log2(n) <= 3`
at `b = 30`. **Measured, in `Int128` and read back through `Int64`:**

    n =  9   true 9223372019674906632     Int64 agrees
    n = 10   true 10376293522134269961
             Int64 reads -8070450551575281655   *** WRAPPED, AND NEGATIVE ***

**Unpublishable above NINE ROWS**, and the wrap is negative, so
`compare_exact_key`'s sign split would rank the wrapped candidate BELOW every
other one — an inversion, not a rounding.

**The fix is to publish THEIR quantity instead of sklearn's.** cuML's
`MSEObjectiveFunction::GainPerSplit` (`objectives.cuh:225-244`) is sklearn's
proxy minus `sum_T^2/n`, times `0.5/n` — both node constants — which is exactly
the relation deviation 144 already records for Gini. Dropping node constants
cannot reorder anything within a node, and the reduction only ever compares
within a node. The parent term is the BIGGEST thing in sklearn's numerator;
subtracting it leaves a perfect square of a quantity bounded by `n * 2^30`,
which can be shrunk by truncating ONE number ONCE instead of every label.

    A = sum_L*n_R - sum_R*n_L     num = (|A| >> j)^2     den = n_L*n_R

---

## 190. [CLOSED] MEASURED: the alternatives lose orderings and this one does not

Against a `Float64` ground truth, 190 ordered pairs of one node at
n = 1,048,576:

    exact Int128 proxy at b=30 (does NOT fit Int64)   0 pairs backwards
    THIS key (Int64)                                   0 pairs backwards
    route 1, narrowed accumulator b=21 (Int64)        60 pairs backwards
    route 2b, sklearn's proxy on sums >> 9 (Int64)    42 pairs backwards

**The published key is as good as the `Int128` form it cannot afford to
publish**, and both narrowing routes lose a third and a fifth of the orderings
at a million rows. Narrowing the accumulator does worst because its truncation
costs up to one unit PER ROW.

**Route 3 is also true and is stated in the code**: exact order preservation in
two `Int64` fields is impossible above ten rows, and not for want of
cleverness — an exact rational equal to the proxy has reduced denominator
`n_L n_R / gcd`, unbounded, and a common-denominator rescale needs MORE bits,
since two distinct proxies of one node differ by at least `16/n^4`. The question
was never "exact or not", it was "which approximation", and it was measured.

---

## 191. [CLOSED] The shift is node-uniform and derived, so there is no row cap

`j = ceil_log2(n) + REGRESSION_SUM_BITS - 31` comes from the node's row count
and the slot width alone — no caller argument, no extra pass, and **no row cap
at all**, because `j` grows with `n`. The single precondition is
`|sum_L|, |sum_R| <= 2^30`, which `choose_scale` already guarantees.

The truncation is on `|A|`, so the key is invariant under negating every label.
An arithmetic shift floors and would not be.

---

## 192. [CLOSED] What the key is NOT: order preservation, measured

Hashed candidates of one node: **0 of 1,128 inverted** at n = 1024 / 65,536 /
1,048,576. Adversarial — adjacent `n_left`, with the second `A` solved into the
first's truncation bucket — **0 / 19,955, 3 / 18,520, 18 / 20,000 (0.09%)**, and
**0 ties anywhere**.

**A cross-node control is part of the fixture and must FAIL**: comparing
candidates of different nodes through a form that drops a node constant inverts
175 of 1,128. Without it the fixture could not tell a node-local key from a
global one — and the check's first draft made exactly that mistake, drawing an
independent `sum_total` per candidate and reporting 66 "inversions" that were
the fixture's fault, not the key's.

---

## 193. [CLOSED] The precondition is a refusal STATUS, and the gate is host-side

`SCORE_STATUS_REGRESSION_REFUSED = 5`. A kernel cannot raise, so a violated
slot precondition is a status in the same position the host raise occupies —
deviation 174's shape. Any existing `!= SCORED` test is already correct.

The `min_impurity_decrease` gate for regression is `mse_gain_from_exact_totals`
on the host, which is deviation 183's shape applied to MSE.

---

## 194. [CLOSED] On an exact tie between two VALID rationals, the metric arm is skipped

**Theirs.** `Split::update` (`split.cuh:78-90`) tests `best_metric_val`, then
`colid`, then `quesval`. For cuML that is right: the metric IS their key.

**Ours.** The exact rational is the key (deviation 145), and
`host_splitter.mojo::_wins_on_total_order` already ordered an exact tie by
`(colid, quesval)` with the metric arm dropped. The DEVICE reduction still
called `Split.update`, which had it — harmless while the device published a
constant metric, and not harmless once deviation 183's second form made it
real.

**MEASURED, which is how it was found.** The device and host metrics then
differed on 4 of 747 nodes by at most 4 ulp — a float DIVISION rounds
differently on Metal than on the host — and those 4 flipped exact-rational
ties, cascading into **11 of 747 nodes with a different split and 18 differing
leaf values**.

**Why dropping it is more correct, not a concession.** Within a node,
`gain = num/(den*n) - sq_total/n^2` and `sq_total/n` is a node constant, so the
gain is a strictly increasing function of the exact rational. Two candidates
whose rationals are EQUAL therefore have EQUAL true gains, and any difference
between their floats is pure rounding. Ordering by it is ordering by noise,
which is what deviation 145 exists to forbid — 145 argued it from a `Float32`
collision, and this is the same argument with a second measurement.

**THE CONDITION IS "BOTH KEYS VALID", NOT "EXACT TIE", AND THE FIRST VERSION
GOT THAT WRONG.** Written as an unconditional skip it also stripped the metric
from the NO-KEY path — where `compare_exact_key` ties every pair and the metric
is the only ranking that exists — so a caller with no exact key would have
ranked by feature index alone. `split_reduce_check`'s arm A' caught it: "no-key
node did not degenerate to Split.update".

**And it moved a sabotage's meaning, which had to be restored.** The
`SPLIT_SAB_FLOAT_KEY` arm means "reduce on `best_metric_val` alone", and forcing
the rational to tie is only half of that now; the arm also takes the no-key path
so that it really is a float-keyed fold.

**What it bought.** `best_metric_val` is now bit-identical between the two
paths — 0 of 747, 0 ulp — so `device_tree_check` ASSERTS it instead of
excluding it.

---

## 200. [CLOSED] `row_ids` is filled ON THE DEVICE, as `thrust::sequence` does

**Theirs.** `get_row_sample` (`randomforest.cuh:50-72`) writes into an
`rmm::device_uvector`; the `bootstrap == false` arm is
`thrust::sequence(..., selected_rows->begin(), selected_rows->end())` (`:69`),
and `fit` hands that device vector straight to the builder (`:169`, `:186`).
**The host never materialises the permutation.**

**What this lane did.** Built the identity permutation as a host `List` and
uploaded it, once per tree.

**Why it was a drift and not just a cost.** It could never have been a wrong
answer — with `bootstrap=False` the value is the identity and nothing is being
decided — but rule 2 is about the SHAPE, not only about decisions, and it is one
`n_rows` H2D copy per tree that their design does not have.

**Fixed** with `row_ids_sequence_kernel`, a grid-stride write-only map, which is
what `thrust::sequence` is.

**A LARGER VERSION OF THIS CHANGE WAS TRIED FIRST AND BACKED OUT, and the
reason is worth keeping.** The most faithful shape puts the device buffer in
the CALLER, as theirs does — `fit` owns `selected_rows` and passes it in — so
`train_classification_device_resident` would take a `DeviceBuffer` instead of a
host `List`. That version produced correct trees (identity held: 0 of 1552
nodes, 0 of 4656 leaf values) and then **crashed at the end of
`device_forest_check`**, after every assertion had passed. The narrower change —
the buffer stays inside the trainer, the FILL becomes a kernel — gets the
property that mattered (no row list is uploaded, ever) without a signature
change across two files while a third was being edited by another session.

**What is still not mirrored, stated so it is not lost:** their buffer is owned
one level up, ours one level down. That is a lifetime difference, not a
host/device one, and it costs one allocation per tree instead of one per
forest. It is the same class of thing deviation 184 closed for the dataset and
should be closed the same way — in a round where `randomforest.mojo` and
`builder.mojo` are not both moving.

---

# The feature sampler on device — deviations 195-201

## 195. [CLOSED] `cub::BlockRadixSort` has no counterpart, so the sort is a hand-written bitonic network

`max.gpu.primitives.block` offers `sum`, `min`, `max`, `broadcast` and
`prefix_sum` and nothing else — checked this session. `PORTING_RULES.md` 0b-i's
terms are met literally: zero warp intrinsics, no assumed wavefront width,
`barrier()` per stage, no vendor branch anywhere in the kernel.

Deviation 157 already established why the substitution is FREE and it is not
re-argued: their `Sort` here sorts KEYS ONLY and the keys are the column ids,
so two equal keys are indistinguishable and every ascending sort produces the
identical array.

The network covers `next_pow2(n_parallel_samples)` rather than the full slot
count, because slots at or above `n_parallel_samples` hold a constant run of
the array's maximum and are already in sorted place.

## 196. [CLOSED] MEASURED: their sort's storage does not fit, so `items` lives in global scratch

cuML's own `BLOCK_THREADS * MAX_SAMPLES_PER_THREAD * sizeof(IdxT)` is 36,864
bytes and this box's threadgroup limit is 32,768: *"Threadgroup memory size
(36864) exceeds the maximum threadgroup memory allowed (32768)"*. 8,192 `Int32`
is the ceiling.

`mask` stays private per thread — slots do not migrate, items do — and
`col_indices` is not stored at all. A comptime capacity ladder was considered
and REJECTED, because its refusal would fire inside cuML's own supported range.

## 197. [CLOSED] The block scan is per-thread totals then one collective

Mojo's `prefix_sum` has no `ITEMS_PER_THREAD`, and nothing returns prefix AND
aggregate together, so it is two calls — the same finding deviation 176 records
for the partition. The aggregate must be block-uniform because it drives the
loop exit.

## 198. [CLOSED] The DISPATCH stays on the host, because it is on the host in cuML

`builder.cuh:419-421` computes `n_parallel_samples` on the host and a C++ `if`
picks between three launches. Rule 2 does not bite: nothing is being decided on
the host that they decide on the device.

Deviation 159's raise becomes `SAMPLER_OVERRUN`, and `report` is added with no
cuML counterpart — the device's own statement of which kernel ran and how many
iterations it took, so a check does not infer the path from host arithmetic.

## 199. [CLOSED] MEASURED: Metal has no `double`, so cuML's algorithm L cannot be a kernel here

cuML's algorithm L is a `double` algorithm in four places
(`builder_kernels.cuh:291`, `:306` twice, `:313`). Metal rejects `double` at
COMPILE time, not at enqueue: *"function's return type 'double' is not
supported"*, *"llvm.fma.f64 has Metal-unsupported instructions"*, *"LLVM ERROR:
Failed to verify LLVM IR for Metal"*.

**A `Float32` substitute was written and REJECTED.** At the `k/n` their dispatch
actually routes to this arm, `W` is about `1 - 1e-4`, so forming `1 - W` in
`Float32` is catastrophic cancellation — roughly 13 bits survive. That is a
DIFFERENT ALGORITHM wearing this one's name, on the one arm nobody would look
at. Tracking `V = 1 - W` through `expm1f` was rejected for the opposite reason:
it is numerically BETTER than cuML, and this is a port.

**A second reason it will not be bit-identical even where `double` exists:**
deviation 158 records that there is no libm through FFI on device, so
`std.math.log`'s ~5e-8 absolute error applies to the jump computation. The
kernel below `algo_l_sample_kernel` is a portable DRAFT for a CUDA/ROCm box,
not a verified kernel, and it says so.

## 201. [CLOSED] The algo-L arm runs on the HOST, and that is a placement difference, not a refusal

**The choice.** Refusing the arm would make the device path unusable whenever
`k/n` is near 1 at large `n`. Running cuML's algorithm on the host is a
PLACEMENT difference; refusing to fit is a CAPABILITY loss.

**Why the host form is trustworthy for it.** It is not a guess — it is the
checked oracle the device kernels are verified against, cell for cell, over
1,063,780 cells.

**The branch is a host-side capability query** (`device_has_float64()`), never
an `if apple` inside a kernel. On CUDA and ROCm the same call takes the device
kernel and the copy disappears; the source is one source.

**The price, stated:** on a target without `double`, that arm costs one
`work_items_size * k` H2D copy per level their design does not have. The
returned plan reports which arm ran, so it is visible rather than silent.

---

## DEVIATION 202 [CLOSED] -- the workspace is allocated once per tree, not once per level

(Since DEVIATION 211: once per GROUP of in-flight trees, one step further in
the same direction; the entry below records the per-level -> per-tree move as
it was made.)

**THEIRS.** cuML computes `workspaceSize()` once (`builder.cuh:272-296`) and
hands out pointers into one allocation with `assignWorkspace()`
(`builder.cuh:302-341`). Nothing inside their level loop allocates. Every size
is a CAPACITY -- `max_batch * n_sampled_cols` for `colids`, `max_batch` for
`splits` and `d_work_items`, `max_blocks_dimx` for `workload_info` -- and
`max_blocks_dimx` is `1 + params.max_batch_size + dataset.n_sampled_rows /
TPB_DEFAULT` (`builder.cuh:230`).

**WHAT OURS DID.** Allocated all 51 per-level buffers INSIDE
`while queue.has_work()`, sized to the current level. A depth-12 tree runs 13
levels, so a ten-tree forest performed about 5,500 buffer creations where cuML
performs ten.

**IT WAS NEVER A WRONG ANSWER**, which is why no check saw it: every buffer is
explicitly initialised before use, by `node_feature_range_init_kernel`,
`node_feature_score_init_kernel`, `split_reduce_init_kernel` or an
`enqueue_memset`, so a reused buffer and a fresh one are indistinguishable to
every kernel that reads one.

**REUSING THE HOST STAGING BUFFERS NEEDED AN ARGUMENT, NOT A HOPE.** The copies
out of them are asynchronous, so a staging buffer rewritten under an in-flight
copy would corrupt it. Every level ends with a `synchronize()` before the
splits are read back, so by the time the loop returns to the top, every copy
the previous level issued has completed.

**AND IT BOUGHT NOTHING, MEASURED.** Bit-identical -- the three scale digests
(`0xf4d3f26ad592d04d` at `max_features=14`, `0xfef885360b06f02c` at `log2`,
`0x37f7156a89f8e725` at `k27`, covtype 581,012 rows, 10 trees, depth 12) are
unchanged -- and the time did not move: 3206 ms before and 3206 ms after at
`k14`. **Allocation was not the cost.** It is kept because it is cuML's shape
and because the previous shape was an unrecorded departure from it, NOT because
it is faster, and this paragraph exists so nobody re-derives a speedup from it.

---

## DEVIATION 203 [CLOSED] -- the frontier partition is multi-block

**THEIRS.** `launchNodeSplitKernel` is `<<<work_items_size, TPB>>>`
(`builder_kernels_impl.cuh:109-134`): one block per node, walking two cursors
down the node's range and swapping misfits in pairs (`:43-88`).

**WHY IT CHANGED, MEASURED.** That grid has `n_nodes` blocks and the root level
has ONE node, so on covtype at 581,012 rows the root's partition is a single
threadgroup walking 581,012 rows. The signature is visible from outside the
kernel: at 145,253 rows a whole fit is FASTER at `max_features=54`
(10.7 ms/level, 11,080 nodes) than at `max_features=5` (14.4 ms/level, 1,244
nodes), despite ten times the split-search work. The only quantity moving the
right way is the NODE COUNT, and this was the only per-node-serial pass. A100s
hide it under 108 SMs; ten Apple GPU cores do not.

**WHAT REPLACED IT.** A counting partition in three passes over the same
`WorkloadInfo` flattening the split search already uses, so the grid is
`(n_blocks_dimx, 1, 1)` -- blocks over ROWS: count left-going rows per block,
exclusive-scan those counts within each node (still one block per node, but
over `ceil(count/TPB)` values instead of `count` rows -- 4,540 instead of
581,012 at the root), then scatter and write back. `cub::BlockScan::ExclusiveSum`
becomes MAX's `prefix_sum` plus a broadcast `sum` for the aggregate, per rule
0b-i: port the CALL.

**THE ANSWER IS UNCHANGED AND IT IS CHECKED, NOT ASSUMED.** The order WITHIN
each side differs from their pairwise-swap order. Nothing downstream reads it:
the split search takes min, max and INTEGER class counts over a node's rows,
the leaf pass sums the same integers, and child ranges come from `split.n_left`.
`device_tree_check` still reports 0 of 747 nodes and 0 of 2,241 leaf values
differing against the HOST trainer, which still uses the swap partition, and
the three scale digests are unchanged.

**MEASURED, on covtype 581,012 x 54, 10 trees, depth 12:** 3554 -> 2952 ms at
`log2`, 3206 -> 2523 at `k14`, 4180 -> 3500 at `k27`. 1.19-1.20x. Real, and
much smaller than the model predicted, which is what sent the next measurement
into the range pass.

**`node_split_kernel` STAYS**, checked, as the oracle
`partition_multiblock_check` compares against. Deleting the thing you verify
against leaves nothing to verify against tomorrow.

---

## DEVIATION 204 [CLOSED] -- the cross-block range merge is lock-free. THIS CLOSES 161

**WHAT 161 SAID, AND WHAT WAS WRONG WITH IT.** That Mojo has no portable float
`atomicMin`/`atomicMax`, which is true, and that the cross-block min/max merge
therefore had to take a lock, which does not follow. The standard
order-preserving map turns the float compare into an INTEGER one, and integer
`atomicMin`/`Max` exist on every backend this lane targets.

**THE MAP.** Non-negative float: set the sign bit. Negative float: invert every
bit. Monotone on all of IEEE-754 except NaN, and exactly invertible -- NaN
never reaches it because the row loop counts NaNs separately and never lets one
become an operand (DEVIATION 163).

**WHAT THE LOCK COST, MEASURED.** Instrumented per phase at 581,012 rows,
`max_features=5`: the range pass was 199-260 ms of a 250-330 ms tree -- 80% --
while the SCORE pass, reading the same rows and doing strictly more arithmetic,
took 27-36 ms. Six to seven times slower for less work is not the work; it is
the lock. At the root, 4,540 blocks took the same spin lock on
`n_sampled_cols` cells.

**MEASURED AFTER, same fixture:** 2952 -> 1215 ms at `log2` (2.4x), 2523 ->
1959 at `k14`, 3500 -> 3000 at `k27`. Against the ORIGINAL shipping code the
whole round is 3554 -> 1215 ms at `log2`, **2.9x**. Digests unchanged
throughout.

**THE SENTINEL GATE IN THE MERGE GOES; THE SENTINEL DOES NOT.** 163 gated the
merge on `not (blk_min > blk_max)` so a block that saw no value could not push
`+inf`/`-inf` into the cell. Under an atomic min/max that gate is a NO-OP:
`range_key(+inf)` is the largest key and cannot win a minimum,
`range_key(-inf)` is the smallest and cannot win a maximum. They are the
identities. The gate is deleted rather than left as a line that can never fire.
The EMPTY-CELL rule survives one level up, in
`node_feature_range_decode_kernel`, and so does `RANGE_SAB_NO_SENTINEL`, which
now selects that kernel's arm and predicts exactly what it predicted before.

**TWO NEW SABOTAGES, AND THE FIRST TRY AT ONE WAS REFUSED BY THE CHECK.**
`RANGE_SAB_SIGN_UNFLIPPED` drops the key's negative branch -- the same
expression on non-negative data, the wrong order on negative data -- and moves
exactly the 28 of 84 cells that carry a negative value.
`RANGE_SAB_EMPTY_NOT_IDENTITY` makes an empty block publish `range_key(0.0)`
instead of its identities, and moves exactly the 18 cells with an empty
contribution, which is what makes deleting the gate safe rather than merely
plausible. The first attempt merged the raw bit pattern, which the decode then
misread as a key: all 84 cells moved where 28 were predicted, and the shape
check rejected it. **A sabotage that moves everything proves nothing**, and the
check said so before a human did.

---

## DEVIATION 205 [CLOSED] -- sklearn's "keep drawing while every draw was constant". THIS CLOSES 151

**WHAT 151 SAID.** That we stop splitting a node when every one of its
`max_features` sampled columns is constant while sklearn keeps drawing, and
that the difference was priced but not fixed.

**WHAT IT COST, MEASURED on covtype 581,012 x 54, 10 trees, depth 12.** Not
small, and it got worse the narrower the sample:

| `max_features` | our accuracy | sklearn | our nodes | sklearn's nodes |
|---|---|---|---|---|
| 5 | 0.645-0.677 | 0.660-0.716 | 3,798-5,818 | 20,558-24,052 |
| 7 | 0.674-0.691 | 0.683-0.736 | 7,620-10,044 | 22,428-24,434 |
| 14 | 0.707-0.745 | 0.737-0.749 | 22,884-26,238 | 27,962-29,446 |

We were building **four to six times fewer nodes** than sklearn at a narrow
sample. covtype is 44 binary one-hot columns out of 54, so a narrow draw is
constant often, and every time it was, we stopped and they did not.

**THE RULE, `_splitter.pyx:573-577`::**

    while (f_i > n_total_constants and
            (n_visited_features < max_features or
             n_visited_features <= n_found_constants + n_drawn_constants)):

`max_features` is a budget on TOTAL DRAWS and a constant feature SPENDS one --
it is not "keep drawing until you have `max_features` usable columns". But the
second clause overrides the exhausted budget for exactly as long as every draw
has been constant. So in the regime this deviation is about, sklearn draws on
one at a time, without replacement, and stops at the FIRST non-constant
feature, having evaluated exactly one.

**AND THAT MAKES IT CHEAP TO PORT EXACTLY.** The remaining features are drawn
in a uniformly random order, so the first non-constant one in that order is
UNIFORMLY DISTRIBUTED over the node's non-constant columns. `rescue_pick`
(`checks/rescue.mojo`) draws that choice directly, with RAFT's own
`uniform_int_u32` on a key whose `feature_id` slot is `0xFFFFFFFF` -- a slot no
column can occupy, so the choice of column is not correlated with that column's
own threshold draw. Same distribution, one RNG draw instead of a sequential
loop no GPU wants to run.

**THE HOST AND THE DEVICE MUST LAND ON THE SAME COLUMN**, and they do it by
calling the same function. The host trainer scans the node's columns with
`node_feature_min_max`; the device runs the SAME range pass over every column
for a SUB-BATCH of exactly the nodes that need it, and the host picks from the
cells it returns. Both build the non-constant list in ASCENDING COLUMN ORDER,
which is part of the contract because `rescue_pick` returns an index into it.

**AFTER, same fixture:** accuracy 0.687-0.692 at `max_features=5` (sklearn
0.660-0.716), 0.686-0.707 at 7 (sklearn 0.683-0.691), 0.740-0.757 at 14
(sklearn 0.737-0.749), and the node counts now MATCH -- 22,566-24,302 against
their 20,558-24,052 at 5, where we used to build 3,798.

**IT COSTS SPEED AND THAT IS NOT HIDDEN.** At `max_features=5` a fit went from
96-140 ms/tree to 354-455, and at 7 from 121-125 to 435-453. We are building
four to six times more nodes, which is the point; per node we are no slower.
The comparison against scikit-learn at a narrow sample is now apples to apples
and at the time of this entry we LOST it against their ten cores (0.28-0.50x at
5, 0.35-0.45x at 7) where before we "won" it by building a smaller and worse
tree. At `max_features=14`, where the node counts always matched, we were
1.02-1.75x faster than their ten cores. **THAT RATIO HAS SINCE INVERTED at
`sqrt`**, after DEVIATIONS 211 and 450: covtype 522,911 x 54 reads 9,826.8 ms
against `sklearn-et-cpu`'s 11,006.8, 1.12x faster
(`bench/results/fast_speed/2026-08-28-APPLE-forest.md`; 1.06x on
`2026-08-27-APPLE-covtype-et-flip.md`, and 2.46x at 581k in
`bench/results/THREE_SUITE_2026-08-22.md`). The numbers above are this entry's
own and stay as its cost accounting; they are not the lane's current position.

**THE SURVEY IS THE COST AND IT IS THE OBVIOUS NEXT LEVER.** A rescued node
pays a full `n_cols` range pass. The `k` cells it already computed are thrown
away and recomputed inside it, and the survey could stop at the first
non-constant column rather than ranging all of them. Neither is done; both are
measurable.

**WHAT IS NOT PORTED.** sklearn also carries a node's discovered-constant set
DOWN to its children (`_splitter.pyx:723-734`) so a child never re-tests what
an ancestor proved constant. That is a cost optimisation on their sequential
loop: the skipped features are constant at the child too and could never have
been selected, so it changes the work and not the answer. Our scan tests every
column afresh.

**REGRESSION HAS THIS, CLOSED BY DEVIATION 206.** `train_regression` and
`train_regression_device` take the same clause and now take the same
`rescue_columns`, `rescue_pick` and key. Measured on `shaped_constant_heavy`
at `max_features=0.15`, MSE criterion: 11 to 805 nodes at seed 0, 1 to 681,
1 to 671, 3 to 751, with 0 of 805 nodes differing host against device on every
seed and no movement on `all_constant`.

**THE REACH IS PROVED, NOT ASSUMED.** `rescue_check` fits the same fixture with
the rescue on and off: `shaped_constant_heavy` goes from 1-11 nodes to 459-597
on every seed, `all_constant` does not move at all (every column constant, so
sklearn's first clause fails too), and a fixture with no constant column is
bit-identical. It then fits host against device WITH the rescue firing and
asserts 0 of 549 nodes differ -- which `device_tree_check` could not have
caught, because its fixtures never trigger the clause.

---

## DEVIATION 206 [CLOSED] -- regression takes 184, 202, 203 and 205, which it never had

The previous rounds landed on the classification device path only. This one
brings regression level, and the reason it is one entry rather than four is
that nothing about any of them is regression-specific -- each is the same
change against the same loop.

**205, THE CLAUSE.** sklearn's constant-feature loop lives in
`node_split_random` (`_splitter.pyx:507-736`), which `RandomSplitter` reaches
for BOTH criteria. It is not a classification rule. A regression tree stopped
early for exactly the same reason and is fixed with the same
`rescue_columns`, the same `rescue_pick` and the same key. Measured on
`shaped_constant_heavy` at `max_features=0.15`, MSE criterion: 11 -> 805 nodes
at seed 0, 1 -> 681, 1 -> 671, 3 -> 751. Host against device with the rescue
firing: **0 of 805 nodes differ**, every seed. Inert on `all_constant`.

**202, THE WORKSPACE.** It allocated ~50 buffers per LEVEL. It now shares the
same `LevelWorkspace`, allocated once per tree, with `n_classes = 1` because
the regression score pass accumulates ONE fixed-point sum per cell
(DEVIATION 135) where the classification pass accumulates a class count.

**203, THE PARTITION.** It still used the one-block-per-node kernel. Same
multi-block count/scan/scatter/writeback now.

**184, THE DATASET, AND THIS ONE HAD A MEASURED PRICE.** There was no
forest-level regression entry point at all, so a caller fitting `n_trees`
trees uploaded the same immutable matrix `n_trees` times and rebuilt the
workspace `n_trees` times. At 100,000 rows that floor was about **100 ms per
tree** -- most of what a depth-8 regression tree costs. `train_regression_device`
is now a wrapper over `train_regression_device_resident`, exactly as
DEVIATION 184 split the classification pair, and
`fit_regression_device` uploads once for the whole forest. Depth 8 at 100,000
rows went from 182 ms/tree to 76-128. The answers are unchanged.

**MEASURED against scikit-learn's `ExtraTreesRegressor`, covtype 581,012 rows,
column 0 (Elevation) as the target and the other 53 as features, 10 trees,
depth 12, interleaved in one process.** The target is column 0 because that is
where it is, not because of anything it showed.

| `max_features` | ours ms/tree | vs their 1 core | **vs their 10 cores** | our MSE | their MSE |
|---|---|---|---|---|---|
| 7 (sqrt) | 165-421 | 1.8-3.3x | 0.69-0.84x | 22,051-22,722 | 21,645-22,348 |
| 27 | 155-186 | 7.7-9.8x | **2.60-2.87x** | 19,043-19,100 | 18,674-18,922 |
| 53 (all) | 232-250 | 8.5-15.3x | **3.42-3.48x** | 17,715-17,735 | 17,985-18,295 |

At every setting the node counts match theirs within a few percent (60,518
against 60,066 at `all`), and at `all` our MSE is LOWER than theirs. This is
the widest margin against scikit-learn recorded in this lane.

**AND THE SCALE STORY WAS THE SAME ONE, UNTIL DEVIATION 211.** At 100,000 rows
this code read 0.06-0.29x against their ten cores, and classification at
100,000 rows read about 0.25x. That was a starved grid, not a
regression-specific weakness, and merging the frontier batch across trees
(DEVIATION 211) moved the classification cell to 0.72-0.86x of ten cores and
2.33-2.99x of one core (`bench/results/WINDOW_2026-08-22_extratrees-batched.md`).
The 0.06-0.29x figure is this entry's own measurement and is kept as such; it
is not the lane's current small-n position.

**AND THE WINNING NUMBERS ARE NOT ALL AT 581,012.** DEVIATION 218's scoring
fits the full 8.8M-row higgs train split at 36.94 s against `skl-et-cpu`'s
170.73 s, 4.6x, with AUC 0.70922 against 0.69861
(`bench/results/gbm_bench_higgs_2026-08-22_124301.json`).

---

## DEVIATION 208 [CLOSED] -- the label gather hoist, MEASURED AND DECLINED

(Renumbered from 207 on 2026-08-21 by the perf lane: two lanes claimed
207 within the hour, and the gbdt blind level loop landed first
(59496dc, ~21:15) before this entry's commit (3b0f333). Commit message
3b0f333 says 207 and cannot change; this ledger is the authority. 208
was verified unclaimed repo-wide at rename time.)

**WHAT WAS TRIED.** `node_feature_score_kernel` is launched with
`gridDim.y = n_sampled_cols`, and its row loop does TWO scattered reads per
visit: `data[col * m + row]`, per (row, column) and irreducible, and
`labels_q[row]`, per ROW and repeated for EVERY column -- seven times per row
at `max_features='sqrt'`, fifty-four at `max_features=None`. A
`materialize_labels_kernel` gathered the level's labels once into slot order so
the score pass could read them sequentially. It was written, wired into both
trainers, both direct-call checks were updated, and it was **bit-identical**
(digest `0x50addb0c6f00b47c`, 23,726 nodes, unchanged).

**IT DID NOT PAY, AND IT IS REVERTED.** Alternating A/B in one window, covtype
581,012 x 54, 10 trees, depth 12, `max_features='sqrt'`, three rounds of
off-then-on:

    off 1958 ms / on 2006 ms      off 2075 / on 2060      off 2435 / on 1851

Fully overlapping. No effect.

**THE FIRST MEASUREMENT SAID 1.75x AND IT WAS WRONG, WHICH IS THE LESSON.**
A stash/pop before-and-after -- not interleaved -- read 2037-2421 ms before and
1153-1237 ms after and was recorded here as a 1.75x win. It was window drift:
the "before" ran on a box still loaded from a concurrent build and the "after"
did not. The interleaved benchmark then showed `ours-gpu` at 115-118 ms/tree
against 113-118 before the change -- identical -- which is what forced the
alternating A/B. **The harness exists for exactly this and it was not used for
a build-level change, because a build-level change cannot alternate inside one
process.** Alternate the BUILDS instead; three rounds is enough.

**AND THE MICRO-BENCHMARK OVERSTATED THE COST IT WAS PREDICTING FROM.** A
standalone probe measured a second scattered gather from a 2.3 MB array at
44-47% on top of one from a 16 MB array, which is what made this look worth
doing. In the real kernel that cost is not additive: the two loads are
independent and the memory system overlaps the small one behind the large one.
**A two-line probe cannot see the latency the surrounding kernel already
hides.**

**WHAT THE PROBES ARE STILL GOOD FOR**, since they were paid for:

* per-phase instrumentation inside a level (581,012 rows, sqrt, 83 batches):
  score 57.7% (3.41 ms/batch), range 13.3%, and the seven remaining phases
  0.08-0.32 ms/batch each. After DEVIATION 204 the range pass is no longer the
  problem; the SCORE pass is, and it is nearly all `data` gather.
* the gather probe, Apple M4, 4M reads, real shuffled permutation, steady
  state: sequential 46-62 GB/s, permuted gather ~1.74 G gathers/s and ~7 GB/s
  effective -- **scattered is 6.7-9.0x slower**. The score pass runs at ~1.19 G
  cell-visits/s, so it is already **within ~1.5x of this machine's pure-gather
  roofline**. It is memory-latency bound, not inefficient.

  **READ THE AXIS OF THAT PROBE BEFORE QUOTING IT.** It compares GPU
  SEQUENTIAL against GPU PERMUTED, both on the device, on one vendor. It is
  NOT a GPU-versus-CPU achieved-bandwidth comparison, which is what the gate
  that commissioned it asked for. **No GPU-versus-CPU gather number exists
  anywhere in this repository**, and four documents have disagreed about
  whether this probe ran because the two questions were not kept apart. This
  is the only gather-probe measurement in the tree; anything stated about CPU
  gather bandwidth is unmeasured.

**WHICH MAKES THE CEILING A FORMULATION QUESTION, NOT A KERNEL ONE.** An exact
ExtraTrees rereads real `Float32` feature values at every level; a histogram
method reads one-byte bins. DEVIATION 137 deleted the histogram deliberately --
that is what makes this ExtraTrees rather than a binned learner -- and the
traffic that follows is the price of that decision, not a defect to tune away.

## What was left after DEVIATION 208, and what each item turned into

The three candidates named here on 2026-08-21 have all since been taken. This
section is kept for the estimates it made and how they landed, because two of
the three were overestimates.

* **Tree-level batching.** Estimated here at ~11% and not the 2x it looked
  like, on the reasoning that cuML parallelises trees across CUDA streams
  (`randomforest.cuh:164`), `ctx.create_stream()` on this device answers
  **"createStream is not supported on this device"**, the batched-frontier
  equivalent was available because nodes from different trees are just more
  nodes in the same `WorkloadInfo` flattening, the per-batch fixed cost it
  would amortise was only ~1.7 ms of an 8.7 ms batch, and **the grid was never
  starved** (`n_blocks_dimx` is driven by ROW count, so the root level already
  launches ~4,540 x k blocks). **BUILT as DEVIATION 211, AND BOTH OF THOSE
  READINGS WERE WRONG.** Measured merged against serial, forest digests
  identical: **1.72-1.85x at 581,012 rows** and **4.06-5.37x at 100,000**
  (`bench/results/WINDOW_2026-08-22_extratrees-batched.md`). The win is largest
  exactly where the grid IS starved, so the ~11% understated it and "the grid
  was never starved" was the wrong diagnosis.
* **Materialising the sampled columns**, estimated here at ~20% on the argument
  that the gather has to happen once anyway so it converts two scattered passes
  into one gather plus sequential reads. **BUILT, MEASURED AND DECLINED, as
  DEVIATION 212. The ~20% is FALSIFIED**, priced at 0.6-1.0x measured across
  three regimes, wrong for the same reason 208's own 1.75x was wrong.
* **The per-level non-constant flag readback** (DEVIATION 205), which ran on
  every level including those with nothing to rescue, and which at
  `max_features=None` can never fire. **HALF TAKEN, by DEVIATION 450**:
  `h_nonconst` was the one host read before the reduce sync, its consumer runs
  after `search_batch` returns, so the read moved past the reduce drain and the
  range pass keeps only its enqueued copy. The DRAIN this bullet costed out is
  gone; the copy itself still runs on every level.

---

## DEVIATION 211 [CLOSED] -- the batch spans TREES: cuML's stream pool, expressed as a wider grid

(209 is the ensemble lane's handle-launches negative and 210 is the gbdt
scheduling fold; both were verified claimed repo-wide before this number was
taken.)

**Theirs.** cuML's cross-tree parallelism is `#pragma omp parallel for
num_threads(n_streams)` over the tree loop, one CUDA stream per OpenMP thread
(`randomforest.cuh:336-341`), Python default `n_streams=4`
(`randomforestclassifier.py:94`). The shipped cuML overlaps four trees.

**Why it cannot be transcribed.** Metal has no streams -- `ctx.create_stream()`
is unsupported on this backend (the traps register). The RF lane met the same
wall and ported the OVERLAP as K-way pipelining over one queue (its DEVIATION
117). This lane's formulation admits something strictly stronger, so that is
what was built.

**Ours.** The frontier batch itself spans trees.
`train_forest_classification_device` / `train_forest_regression_device` pop
work items from EVERY tree's `NodeQueue` into one merged batch per cycle; the
tree id, which crossed the kernel boundary as a per-launch scalar, now rides
PER ITEM (`item_trees`, staged into `LevelWorkspace.d_tree`; read by the score,
finalize and sampler kernels as `tree_ids[nid]`). Each in-flight tree owns slot
`s` -- rows `[s * n_rows, (s + 1) * n_rows)` -- of ONE `d_row_ids` buffer
(`row_ids_tiled_sequence_kernel`; `NodeQueue` gained `row_base` so every range
a tree ever holds is carved from its slot). The per-tree trainers survive as
one-tree calls of the forest trainers, so there is exactly ONE copy of the
loop; `fit_classification_device` / `fit_regression_device` call the forest
trainers directly. `FOREST_ROW_SLOT_CAP` (2^26 row slots) bounds the in-flight
group; trees beyond it run as further sequential groups.

**Why the trees cannot move, mechanism by mechanism.** Every draw was ALREADY a
pure function of `(seed, tree_id, node_id, feature_id)` (deviation 130) -- the
same property that makes cuML's own stream overlap "free on output" -- and
`bootstrap=False` means every tree reads the same resident dataset (deviation
184). The score accumulators are per-(node, feature) integer atomics (171); the
reduction is per node; the partition is range-addressed and the slots are
disjoint; each queue's push order is its own FIFO order, and batch width was
already a scheduling parameter contracted not to change the tree
(`NodeQueue.pop`).

**What it buys.** Launches, split readbacks and `synchronize()` points per
level divide by the number of in-flight trees, and the grid per launch
multiplies by it -- which is aimed at the small-n regime, where the ledger's
standing diagnosis is a STARVED grid, not slow kernels. PRICE: `2 * 4 *
min(n_trees, cap) * n_rows` bytes of row-id buffers, and the workspace's
row-scaled pieces grow the same way (the workspace itself is now one per GROUP,
deviation 202 taken further; the regression trainer's last per-level
allocations -- its split staging -- moved into the workspace on the way).

**Numbers are in `bench/results/`,** not here: this entry records the
mechanism and the gate; the lane's no-timing rule was lifted by the repo owner
for this round and the measurement discipline is the benchmark file's.

**GATED** by `device_batched_check.mojo`: the merged forest against one-tree
device builds, node for node and leaf BIT for leaf bit, on the public fit
entry points; a `max_batch_size=3` config that splits trees across cycles
mid-level; a `max_leaves` config (the budget's break order must survive
merging); a `row_slot_cap` of two trees that forces the multi-group path; and
BOTH isolation mechanisms sabotaged on BOTH objectives, asserted RED in the
check itself rather than applied by hand: `FOREST_SAB_SCALAR_TREE` (every item
staged with the batch-first tree id -- what keeping the scalar would silently
do) moved 2,148 / 2,862 nodes (clf/reg), and `FOREST_SAB_SHARED_ROW_BASE`
(every tree rooted at slot 0 -- what losing the offsets would do) moved 2,882 /
2,912. `device_forest_check` and `device_tree_check` still hold the device
forest to the HOST forest, so the merge is also transitively pinned to the
host trees.

**Doc corrections carried with this change (rule 17).**
`fit_classification_device`'s docstring described a per-tree loop with "a
FRESH `row_ids` per tree" and still carried 184's completed "what the next
round should do" plan; both replaced. `device_forest_check`'s module
docstring said the tree id crosses as "a scalar argument"; now per-node, and
the sentence points at the check that sabotages that staging. Deviation 185's
warning -- that a device-resident frontier would need per-tree row isolation
-- is superseded by exactly that isolation arriving as the slot offsets, and
`fit_classification_device` says so where the old block stood.

---

## DEVIATION 212 [CLOSED] -- the materialized score pass: MEASURED AND DECLINED, and it is 208's lesson at forest scale

**WHAT WAS TRIED.** cuML reads the data matrix ONCE per level because their
quantization turned it into bins up front; the histogram-free formulation
(137) reads the raw floats TWICE -- the range pass gathers
`data[col * m + row_ids[i]]` to bound each (node, feature) cell, and the
score pass gathers the same cells again. Since the range pass pays the
gather anyway, it was extended to STASH each value it read in slot order
(`d_mat[fslot * batch_rows + row_base[nid] + (i - range_start)]`, coalesced
on both sides -- writer and reader walk the identical strided loop), and the
score pass read the copy sequentially instead of gathering. A second arm
also stashed the LABELS in slot order -- 208's hoist re-measured, because
removing the big gather voids the condition under which 208 declined it. The
whole thing was built, threaded through both objectives, held bit-identical
by every device check, and its reach was proved by a poison sabotage
(perturb only the COPY, leave the reduction's values true -- ranges and
thresholds stay right, so only a score pass actually READING the copy can
move; it moved 2,388 / 2,610 nodes clf/reg in `device_batched_check`).

**IT DID NOT PAY, ANYWHERE, AND IT IS REVERTED.** Three regimes, three modes
alternated inside ONE process per rep, forest digests asserted identical on
every rep and every mode (mode 0 = gather, mode 1 = values materialized,
mode 2 = values + labels), covtype, depth 12, ratios are mode0/modeN so
above 1 would mean the copy won:

    clf 581,012 x sqrt, 10 trees:   mode1 0.90 / 0.95 / 1.01   mode2 0.86 / 0.97 / 1.11
    reg 581,012 x all,  10 trees:   mode1 0.74 / 0.89 / 0.92   mode2 0.89 / 0.93 / 0.94
    clf 100,000 x sqrt, 100 trees:  mode1 0.72 / 0.95 / 0.99   mode2 0.62 / 0.96 / 0.98

A wash at best, a loss at worst, and never a sustained win. The
classification arms ran the SAME group sizes in both modes (10 and 100 trees
both fit one materialization-capped group), so the number isolates the copy
itself, not the group shrink that the `all` arm additionally pays.

**WHY, and what this pins for the next reader.** This is DEVIATION 208's
finding at full scale: a standalone probe measured scattered gathers at
6.7-9.0x sequential cost, and the ledger priced "column materialization
~20%" from it -- but in the REAL kernel the score pass's gather is not the
serialized cost the probe modelled. The memory system overlaps it with the
label read, the accumulator work and the other blocks in flight, and a
16 MB-per-tree-level sequential copy (write in the range pass + read in the
score pass) costs as much as the gather it replaces. **The standing
diagnosis "the score pass is within ~1.5x of the gather roofline" therefore
needs its reading corrected: the pass is near a roofline, but removing the
gather does not move it, so the binding resource is the whole memory
system's throughput over the level's row traffic, not the gather's latency
per se.** Two consequences, stated so they are not re-derived: (a) the
~20% materialization estimate in the old parallelization ledger is
FALSIFIED, priced at 0.6-1.0x measured; (b) any future lever that only
REARRANGES the same row traffic (prefetch, layout, hoists) should be
presumed a wash until an alternating in-process A/B says otherwise -- the
levers that have actually paid in this lane (211, 202, 184) all REMOVED
work or launches outright rather than rearranging reads.

**THE CODE IS REVERTED** to DEVIATION 211's state -- kernels, workspace,
checks and the `mat_ab.mojo` harness -- because a dead-by-default mode kept
"for later" is the unwired-path defect rule 3 names. This entry is the
result; the harness is three hundred lines anyone can rewrite from the
paragraph above in an hour, and the numbers are what they would get.

---

## DEVIATION 213 [CLOSED] -- the merged batch bound: MEASURED, NO EFFECT, and the Instruments profile that closes the schedule question

**WHAT WAS TRIED.** cuML pops frontier batches at `max_batch_size = 4096`, a
bound tuned for ONE tree; DEVIATION 211's merged frontier holds tens of
thousands of nodes at sklearn's default 100 trees, so the bound splits deep
levels into dozens of batches, each with its own staging, launches, readback
and synchronize. Batch width is contracted not to change the tree, so
`bench/batchwidth_ab.mojo` swept 4096 / 16384 / 32768 alternated in one
process (digests identical across every arm and rep -- the width contract,
live at full covtype scale): covtype 581,012 x sqrt, 100 trees, depth 12 --
**0.96x / 1.07x / 1.01x for 16384 and 0.96-1.04x for 32768. No effect;**
4096 stays, being cuML's value and the smallest workspace.

**WHY, measured rather than argued -- the Apple Instruments profile**
(`bench/fit_once.mojo` traced under Metal System Trace, read by
`bench/profile_et_metal.py`; the parsing technique is the RF lane's, reused
in this lane's own file). One merged 100-tree covtype fit, 3,645 compute
dispatches:

    GPU busy fraction           93.8%  (busy 10,715 ms of 11,422 ms span)
    the 10 longest dispatches   50.8% of busy time (max single: 799 ms)
    the 50 longest              83.2%
    the 200 longest             96.7%
    GPU performance state       "Minimum M4" for 11,015 of 11,422 ms
    thermal state               "Fair" throughout

Three findings, each of which retires a hypothesis:

1. **The schedule is DONE, and that reading was too strong; see the note
   below.** 93.8% busy means the host is not the cost on THIS trace, which
   is what this entry's wash and 212's wash showed from the outside.
   DEVIATION 211 is what bought it; the pre-211 sibling measurement on this
   repo was launch-bound.
2. **The cost is a handful of GIANT dispatches: the shallow levels' range
   and score passes, where every row of every tree is active.** Ten
   dispatches are half the fit. That is the formulation's own row traffic
   (two passes over `rows x k` per level, deviation 137's deliberate
   trade), not overhead -- and 212 measured that rearranging those reads
   does not help. Within this formulation, at this size, the kernels are
   the floor. The formulation-level alternative (bin-space ET on the
   histogram builder, LightGBM's `USE_RAND`) is the RF lane's Step 2, a
   different product by design.
3. **THE GPU RAN AT ITS MINIMUM CLOCK FOR 96% OF THE TRACE, thermal state
   "Fair".** This is the MECHANISM behind the repo's standing rule that
   this box drifts 1.7x in twenty minutes: a heat-soaked chassis pins the
   GPU governor at minimum, and every absolute number in these windows --
   including the sklearn ratios, whose CPU arms throttle on their own
   schedule -- is a number AT WHATEVER CLOCK THE WINDOW GOT. Alternation
   inside one process protects the ratios; nothing protects the absolutes.
   A cold-chassis window would be faster everywhere and possibly by a lot;
   taking one is a bench-hygiene action, not a code change.

**CORRECTED BY THREE LATER ENTRIES: SCHEDULING WAS NOT DONE AND THE
FORMULATION WAS NOT THE FLOOR.** Three levers of exactly the kinds ruled out
above were found afterward, all inside this formulation.

* DEVIATION 214, a "fewer syncs" lever: the leaf tail batched, clocked leaf
  phase 735 ms to 130 ms (8.3% to 3.5%), removing 99 allocation sets, launches
  and synchronizes per 100-tree group. It was found by the very instrument
  this entry motivated.
* DEVIATION 450, another one: five of the level cycle's six synchronizes
  removed, motivated by the small-n rows this profile could not see.
* DEVIATION 1943, a pure block-width change with no algorithm change: on an
  MI325X the range pass went 8107 ms to 2819 and the score pass 8799 to 3288,
  the two hot kernels 17002 to 6207.

Two of the three do not move on the box this profile was taken on, which is
the honest reading of the discrepancy: **the finding held for this trace, on
this box, and it was written as a formulation-level statement it could not
support.** A one-box profile does not close a schedule.

**WHAT THIS CLOSES AND WHAT IT LEAVES.** Closed on this box: read
rearrangement (212), batch width (this entry). Left, priced by
the profile: nothing else inside the current formulation at the shipped size
on Apple --
the next real large-data moves are a different formulation (bin-space ET,
the other lane's product) or better silicon conditions. The small-n regime
is a different story and its lever (the sampled-column sweep, `gridDim.y`)
is unchanged by any of this.

---

## DEVIATION 214 [CLOSED] -- the phase clock, and the first thing it paid for: the leaf tail batched

**THE INSTRUMENT.** DEVIATIONS 212 and 213 were both chosen from whole-fit
inference and both measured out as washes; the lane had NO micro-step
timing, and Instruments cannot name kernels from the stock template. So
`PhaseClock` (builder.mojo): threaded through `search_batch`, both twins,
and the `train_forest_*_device_timed` variants; DISABLED it is inert -- the
shipping entry points construct a disabled clock and add no synchronize --
and ENABLED it syncs at every phase boundary and accumulates TEN phases
(setup, stage+sampler, range, score, candidate+reduce+readback, host splits,
partition, host pop + batch assembly, leaf, host queue push). The tenth,
`PHASE_HOST_PUSH`, was split off the host-queue phase at commit 9c8ffc23 and
is what DEVIATION 1945 rests on; `builder.mojo` carries `PHASE_HOST_PUSH = 9`
and `N_PHASES = 10`. An enabled clock measures a SERIALIZED
program, so `bench/fit_once.mojo phases` prints the clocked total NEXT TO an
unclocked warm run of the same config: the gap is the measurement's own
distortion, stated beside the numbers it distorts (the RF lane's profiler
carries the same caution; a warmup fit precedes both, because the process's
first fit pays several hundred ms of pipeline creation).

**FIRST TABLES** (covtype, depth 12, sqrt; shares are the signal -- the box
was mid-drift and absolutes swung 2.7-10.2 s on one config within the hour,
see the entry-213 clock finding):

    581k x 10 trees:  score 41%  range 28%  stage+sampler 11%  reduce 10%  partition 8%  leaf 1%
    100k x 100 trees: score 30%  range 25%  stage+sampler 18%  reduce 9%   partition 9%  leaf 8%

The two row passes dominate everywhere (the entry-213 reading, now
attributed); at small n the per-batch fixed phases (stage+sampler, reduce)
grow to a quarter of the fit -- the next levers, now NAMED instead of
guessed.

**THE FIRST FIX IT PRICED: the leaf tail.** The per-tree leaf loop allocated
SEVEN buffers and synchronized once per tree -- deviation 202's
per-level-allocation disease, still alive in the tail -- and the clock
priced it at 8.3% of a 100-tree fit. The kernel was batch-ready all along
(deviation 180's own words: slice the pointers, shrink the grid), and every
range is already global in the shared row buffer (211), so the group's trees
are now CONCATENATED: one allocation set, one `leaf_kernel` launch over
every node of every tree, one readback, one synchronize per GROUP where
there were a hundred of each. Clocked leaf phase 735 ms -> 130 ms (8.3% ->
3.5%, residual is the host-side node staging). Bit-identity gated by the
same checks as always -- `device_batched_check`, `device_forest_check`,
`device_regression_check` all hold node for node and leaf BIT for bit.
Whole-fit confirmation rides the next clean window per the drift rule; the
mechanism removes 99 allocations-sets, launches and synchronizes per
100-tree group outright, which is the lever class with a 3-for-3 record here
(184, 202, 211) against 0-for-2 for traffic rearrangement (212, 213).

---

## DEVIATION 215 [CLOSED] -- cuML's feature sampler is BIASED, and higgs paid for it: the k survivors are now a uniform keyed-hash subset

**Theirs, and it is a bug.** `excess_sample_with_replacement_kernel` draws
`n_parallel_samples` columns with replacement, dedupes, and -- when more
than `k` uniques came out -- keeps the `k` SMALLEST unique column ids
(`builder_kernels.cuh:243-246`: the gather index is the prefix sum of the
head flags over the SORTED array, and everything past `k` is not written).
That is a selection bias, not a tie-break. At `(n=28, k=5)` the dispatch
draws only 6 samples (`ceil(log(1-5/28)/log(1-1/28))`), and simulating
their rule over 200,000 nodes puts column 27 in the sample at **0.38x
column 0's rate** -- the last column of a 28-feature dataset is sampled at
barely a third the frequency of the first.

**How it was FOUND, because the trail is the lesson.** The higgs board
(addendum 4) showed our train accuracy 1.0-1.5 points UNDER sklearn's,
consistently, while covtype showed ours equal or better -- and the year
discriminator showed a residual at `max_features=all`, where the sampler
never runs, so the sampler could only explain the `sqrt` gap. The sign
pattern is what convicted it: higgs's seven most informative features (the
derived masses) are its HIGHEST-indexed columns, 21-27 -- starved by the
bias -- while covtype's informative continuous columns are its LOWEST, 0-9
-- boosted by it. One mechanism, both observations, opposite signs.

**Ours.** When the loop lands on exactly `k` uniques there is nothing to
select and cuML's gather runs verbatim. When it overshoots, the `k` kept
are the smallest by `excess_selection_hash(tree, node, col)` -- a keyed
fnv1a32 chain, salted so the stream is disjoint from the subsequence and
threshold keys. By symmetry of the i.i.d. draws, a hash-ranked `k`-subset
of the uniques is EXACTLY uniform over columns; it is deterministic from
`(tree, node, col)` alone, identical on host and device (one shared
function), independent of thread scheduling, and it changes neither the
sort network nor the dedupe. Same fix-their-bug footing as DEVIATIONS 164
and 165, which repaired two other defects in this same kernel.

**GATED, and the gate can see what no prior check could.** The sets were
always valid, host and device always matched slot for slot, and the
distribution was never asserted -- which is exactly how the bias shipped
through 23,462 asserted cells. `sampler_kernel_check` section 8 now
aggregates 4,000 nodes at `(28, 5)`: the fixed rule's per-column counts
land 670-768 around the uniform 714 (a +/-15% band, over four sigma), and
cuML's original rule -- kept alive as `SAMP_SAB_SMALLEST_K`, the required-
RED sabotage arm -- lands 273-836, the starved column at the simulated
0.38x. Every existing sampler assertion still passes: the fix is invisible
to set-validity and slot-parity checks, visible only to the distribution
gate, which is the whole finding.

**UPSTREAM CONFIRMATION (reported by the RF lane, 2026-08-22, from its own
v26.08.00 pin 265b9da6; not re-verified at this lane's pin).** cuML's current
release has NO `excess_sample_with_replacement` anywhere in `decisiontree/`
-- the only feature sampler is a `shuffle_iterator`/minstd path, uniform by
construction. Upstream evidently moved off this sampler after the pin this
lane transcribed; a future rebase turns this fix into a straight port of
their shuffle path, and their removing commit may carry their own reasoning,
worth citing if found.

**PRICE.** The overshoot path pays one extra pass over the block's scratch
per unique (rank by hash; the scratch was already global per deviation
195). The exact-`k` path -- the common case, since `n_parallel_samples`
targets `E[uniques] = k` -- is bit-for-bit cuML's. Forests built at
`max_features < n` CHANGE with this fix, everywhere, by design: the
candidate sets are now actually uniform. Accuracy movements are recorded
in `bench/results/` -- including covtype, where the bias had been HELPING
and honesty requires re-measuring, not just the dataset that benefits.

---

## DEVIATION 216 [CLOSED] -- the zero-gain gate: cuML's `<=` becomes sklearn's boundary, because year's TEST set was paying for it

**The trail, in order, because each step eliminated a suspect.** After 215,
our-vs-sklearn accuracy was clean on higgs at `sqrt` AND at `all`, and on
covtype both objectives -- but the year pair still read test MSE 98.04 vs
sklearn ET's 96.25 at the depth-8 parity config. The train-side
discriminator (interleaved, same seeds, `max_features=all`, depth 8) showed
ours OVERLAPPING sklearn on TRAIN MSE (96.6-97.8 vs 97.1-98.0, ours better
on 2 of 3 reps) while building ~13% FEWER nodes (4,082-4,242 vs
4,656-4,892). Equal train fit from fewer nodes, worse test fit: the missing
nodes were splits that do not move TRAIN MSE at all -- ZERO-GAIN splits --
whose absence still changes how test rows are partitioned.

**The mechanism.** `split_not_valid`'s first clause was cuML's
`best_metric_val <= min_impurity_decrease`: at the default 0, a zero-gain
winner becomes a leaf. sklearn accepts equality, so it SPLITS zero-gain
nodes and keeps refining. On continuous targets exact zeros are rare and
the two gates agree; on INTEGER targets (year is years; covtype's elevation
column is integers too) exact zero gain is common, and cuML's gate silently
prunes. The split rule this lane implements is sklearn's -- the reference
named in the lane header -- so this is the same footing as 215: their
boundary was a defect against the reference, not a design to preserve.

**The change.** One comparison in ONE shared function (`split_not_valid`,
used by `NodeQueue.push`, the host trainers, and the partition kernels via
`_skip_node`): `<=` to `<`. Invalid candidates still carry `MIN_FINITE` and
still fail. Host and device change together, so every identity gate holds
by construction. Nonzero `min_impurity_decrease` scaling parity between the
two libraries is a separate, unchanged question.

**THE COMPANION THE AUDIT FOUND, before any measurement ran.** sklearn
leafs a PURE node before splitting (`_tree.pyx:240`, `impurity <= EPSILON`)
-- under cuML's `<=` gate purity fell out of zero-gain rejection for free,
so accepting zero gain WITHOUT that pre-leaf would cascade pure regions
down to the depth cap (identical child predictions, real node/time bloat).
For CLASSIFICATION the test is exact from counts already in hand -- Gini is
0 iff one class holds every row -- and it now lives in all three scoring
paths (the finalize kernel as `SCORE_STATUS_PURE_NODE`, the shared host
oracle, and `host_splitter`'s Gini splitter), so host and device leaf the
same nodes. For REGRESSION zero variance is not detectable from the sums
the device accumulates; the exposure is bounded by `min_samples_split` and
the depth cap, and the year node-count prediction below is the measurement
that says whether it bites.

**MEASURED, AND THE PRE-COMMITTED PREDICTIONS FAILED -- which is what led
to the real defect.** Predictions (a) and (b), committed above before the
run: both FAILED. Node counts did not move (4,088-4,258, same as before)
and test MSE did not move (98.05). The revert clause's own premise --
that the gate was firing and shrinking trees -- was falsified along with
them: zero-gain WINNERS essentially never occur (a regression winner is
zero-gain only when all ninety candidates are). The gate change alone is a
measured NO-OP.

**Why it is KEPT anyway, and what made it load-bearing after all.** The
failed predictions forced the depth probe that found the true mechanism --
DEVIATION 217: valid winners whose true gain is positive but whose FLOAT
evaluation is NEGATIVE by cancellation, which the gate then leafed. 217's
clamp maps those to 0.0 -- and a 0.0 gain passes only under THIS entry's
sklearn boundary; under cuML's `<=` it would still leaf. So the pair is
the fix and neither half works alone: with both, year's seed-2 tree went
217 -> 477 nodes (sklearn 497), every seed landed inside sklearn's range,
and test MSE closed 98.04 -> 96.57 against sklearn ET's own 96.25-96.61.
The purity companion above is what keeps the pair from bloating pure
regions. Classification did not regress (covtype average up ~1 point,
higgs at parity, measured interleaved).


---

## DEVIATION 217 [CLOSED] -- cuML's float gain evaluates NEGATIVE on provably non-negative splits, and the gate was leafing them: the exact-sign clamp

**The find, exactly as it happened, because the trail is the method.** After
216 measured as a no-op, the depth probe showed our year trees collapsing
SEED-DEPENDENTLY (seed 2: 217 nodes vs sklearn's 497) with an entire
half-tree leafed at depth 1 -- 148,196 rows, ninety non-constant columns,
label sums comfortably inside the fixed-point slot. `bench/node2_probe.mojo`
reran `search_batch_regression` on that exact node with the fitted keys and
printed the winner: `colid 77, metric -0.026946679` -- a VALID candidate
carrying a NEGATIVE gain.

**The defect.** The true impurity decrease is provably non-negative for
both objectives -- the within-group sum of squares never exceeds the total;
that is the Gini and variance decompositions -- so a negative value can only
be arithmetic. `GainPerSplit` (`objectives.cuh:52-83`, `:225-244`,
transcribed statement for statement, fma placement and all, per DEVIATION
142/183) forms three terms near 1e5-1e12 magnitude from label sums near
3e8 in FLOAT32 and cancels them; at year's scale the rounding of the terms
is the SIZE of the true gain, so tiny-but-real gains come out negative,
`split_not_valid` leafs the node, and which nodes die depends on the drawn
thresholds -- hence the seed dependence. **cuML ships this defect** (their
own gate consumes the same float), on the same footing as 215's sampler
bias and 164/165: transcribed faithfully, then fixed rather than ported.
sklearn evaluates its criterion in float64 and does not exhibit it.

**The fix.** One clamp -- `gain < 0 -> 0` -- at all THREE gain forms (the
device `gain_per_split`, the host Gini `GainPerSplit`, the host MSE
`GainPerSplit`), identically, or the arms would grow different trees. The
clamp restores the sign the mathematics guarantees; ranking is untouched
(the reduction orders by the exact integer keys, DEVIATIONS 144/189). It is
load-bearing ONLY with 216's boundary: a clamped 0.0 passes sklearn's `<`
and would still leaf under cuML's `<=` -- the two entries are one fix.

**MEASURED.** Year, depth 8, `max_features=all`: per-seed node counts moved
from 217-481 (collapsing) to 475-507, statistically indistinguishable from
sklearn's 475-505; gbm-bench TEST MSE 98.04 -> 96.57 (sklearn ET 96.25 and
96.61 in this window's two runs; the remaining distance to LightGBM's 92.85
is the depth-8-parity model-family gap sklearn shares, per this window's
earlier decomposition). Classification: covtype interleaved average up
about a point, higgs at the post-215 parity, both at 2.3-3.8x sklearn's
ten cores in this window's clock state. All 29 checks pass; the probe that
caught it stays in `bench/node2_probe.mojo` as the exact repro.

**What this closes.** The year accuracy row's our-vs-sklearn residual. The
scoreboard's remaining year gap is model-family, shared with sklearn's own ET
at this config.

**IT WAS NOT THE LAST ET-BEHIND ACCURACY ITEM, and this entry used to say it
was.** On 2026-08-26 covtype (522,911 x 54, depth 16, `max_features=sqrt`) read
our ET 0.6701 against sklearn's 0.6768, the largest accuracy gap in any tree
lane on any box (`bench/results/fast_speed/2026-08-26-APPLE-trees.md:27`,
`:73`). DEVIATION 463 is what took it.
---

## DEVIATION 218 [CLOSED] -- the 2^21 classification row cap LIFTED: deviation 191's shift, applied to the Gini pair

**What stood.** DEVIATION 175 published sklearn's proxy as an exact Int64
rational and priced the exactness at a cap: the worst numerator is `n^3/4`,
so past 2,097,152 rows the pair can wrap and the classification device path
REFUSED -- a theoretical bound until full higgs (8.8M train rows) became the
first real request it denied, as a named refusal in the RF lane's pairs
table.

**The lift.** `classification_key_shift(row_count) = max(0,
3*ceil_log2(n) - 64)` -- a NODE-UNIFORM right shift on the squared sums
before the pair is formed, exactly deviation 191's scheme for the
regression key. Three publish sites shift identically (the finalize
kernel, the host oracle, the host trainer's `ProxyImpurityExact`) or the
arms would rank differently. The `-64` is `-62` of budget plus `-2` for
the `/4` in the worst case; the first draft wrote `-62` and would have
shifted AT 2^21, breaking the next sentence -- arm E's new assertions are
what caught it, before any run.

**What is preserved and what is surrendered.** `s == 0` at and below 2^21
rows: the entire formerly-legal regime is BIT-FOR-BIT unchanged and every
existing identity gate still pins it. Above, candidates within one granule
(`2^s` on squared sums of relative size `2^-40` at any `n`) tie into the
total order -- the same surrender 191 made for regression, four orders
finer at its worst. `score_row_bound_ok` survives as the statement of
where exactness ends; the refusal it fed is gone.

**GATED before any fit ran**: arm E of `score_kernel_check` asserts the
shift is 0/2/8 at 2^21/2^22/2^24, that the 2^26 worst case fits Int64
shifted (in Int128), that order survives past one granule, and that half
a granule TIES -- the granularity claim held to its own number.

**PREDICTIONS, committed before the measurement**: (a) full-higgs ET
(8.8M-row train split) FITS through the pairs harness and fills the
refused cell; (b) its accuracy metrics land within the band of lgbm-et's
on the same split (no bit-identity claim across libraries); (c) no check
regresses; (d) the higgs2m board (all under 2^21) is bit-for-bit
untouched. Numbers land beside this entry when the bench lock clears.

---

## DEVIATION 450 [CLOSED] -- one drain per batch: five of the level cycle's six synchronizes removed

**What stood.** One level cycle of the merged forest trainer drained the
queue SIX times: the workspace-staging sync at `stage_batch`'s tail, the
`search_batch` entry sync, the rescue's colids-copy sync, the range pass's
readback sync, the reduce readback sync, and a pre- plus post-partition
pair in the forest loop. cuML's `doSplit` enqueues everything and drains
ONCE, at `raft::update_host(h_splits...)` + `handle.sync_stream`
(`builder.cuh:492-494`) -- our extra five were ported caution, not ported
design, and on Metal a drain is a fixed per-cycle stall that the
scoreboard's small-n rows (100k at 0.7-0.9x sklearn) pay in full.

**The lift.** The cycle now drains exactly where the host READS device
results: the reduce readback (always) and the survey's readback (rescue
paths only). The arguments, per removed site:

  * a single in-order queue orders every enqueued op after everything
    enqueued before it, so device-side hazards (kernels reading `d_items`,
    `d_colids`, `d_splits` behind their copies) never needed a drain;
  * every `h_*` staging buffer is REWRITTEN only after a required drain
    has retired the last copy that read it -- every `search_batch` path
    ends in one (the reduce sync, or the survey's) before returning;
  * `h_nonconst` was the one host READ before the reduce sync, and it was
    read EARLY: its consumer (the rescue's retry scan) runs after
    `search_batch` returns, so the read moved past the reduce drain and
    the range pass keeps only its enqueued copy;
  * first-ever host writes are covered by a NEW one-time drain at the tail
    of `make_level_workspace` (host buffers are created by ENQUEUED ops;
    the leaf pass already modeled create-sync-write).

`PhaseClock`'s contract is unchanged: an enabled clock's `tick` drains at
every boundary as before, so the TIMED program is the same serialized
program it always was; only the untimed program lost its stalls.

**GATED, bit-for-bit.** device_batched_check, device_forest_check,
device_regression_check, device_tree_check all PASS. Stronger: full
identity traces of device_batched_check (every fit, every stage) before
and after the change are HASH-IDENTICAL on every real fit -- 4198 of 4358
record lines, every non-sabotage fit -- and the two diverging fits are
exactly the two `FOREST_SAB_SHARED_ROW_BASE` subjects, which overwrite one
row-slot from every tree BY DESIGN and which two runs of IDENTICAL
post-change code also disagree on (moved-node counts 2812/3074 vs
2780/3094). The sabotage arm was nondeterministic before this change; the
gate it feeds asserts only that the sabotage MOVES the forest, which held
in every run.

**What this buys, to be measured by the orchestrator.** Five fewer full
queue drains per level cycle, every cycle of every group. The stage-timer
flag (`MOJOLEARN_STAGE_TIMES=1`) attributes exactly the phases these
stalls sat in. PREDICTION, committed before the measurement: the 100k-row
clf/reg rows move toward parity and no large-n row regresses; the win
scales with cycle count over device seconds, so small-n gains most.

### DEVIATION 218 -- the four predictions, SCORED (2026-08-22, against `bench/results/gbm_bench_higgs_2026-08-22_124301.json`)

The predictions were committed in this entry before any run; the scoring
is therefore explicit, hit or miss, one line each.

* **(a) full-higgs ET fits through the pairs harness and fills the
  refused cell -- HIT.** 8.8M-row train split, `mojolearn-et-gpu`
  train_time 36.94 s, on the rebuilt `_mojolearn_trees.so` (the first
  binary carrying the lift; the previous run's refusal came from a STALE
  binary whose source-level fix this ledger had already recorded -- the
  shipped artifact is a call site too). The comparator arm ran skl-et-cpu
  at 170.73 s: 4.6x, with our AUC 0.70922 / LogLoss 0.66334 against their
  0.69861 / 0.66648 -- accuracy ABOVE the comparator at the formerly
  refused scale.
* **(b) accuracy within the band of lgbm-et's on the same split -- NOT
  SCOREABLE AS WRITTEN.** The JSON carries no lgbm-et arm, and the only
  lgbm-et higgs row on record (`RF_ET_2026-08-22_lightgbm-pairs.md`)
  logged time only, under a contamination flag. Against the comparator
  that WAS run, skl-et-cpu, every accuracy metric landed above. The
  prediction stays open until an lgbm-et higgs arm runs clean; it is not
  claimed as a hit.
* **(c) no check regresses -- HIT.** score_kernel_check (210 cells, arm
  E's 218 assertions included), objectives_check, device_batched_check,
  device_forest_check, device_regression_check, device_tree_check,
  estimator_check: all PASS on this source.
* **(d) the higgs2m board (all under 2^21) is bit-for-bit untouched --
  HELD BY GATE, not re-measured.** `classification_key_shift` is 0 at and
  below 2^21 (arm E asserts it), so the formerly-legal regime publishes
  the identical pair; corroborated by identity traces of the fixture
  suite hashing identical pre/post on every non-raced fit. The board
  itself was not re-run for this scoring; a direct re-run remains the
  orchestrator's option and this line is not upgraded to HIT until it
  happens.

---

## AMENDMENT to commit d85c6ce's staleness note (2026-08-22)

That commit's message claimed "extratrees builder source feeds TWO shipped
binaries, {_mojolearn_rf.so, _mojolearn_trees.so}". The ruling of record is
that the claim was WRONG: the two compile graphs are DISJOINT --
`_mojolearn_trees.so` is extratrees/ + core/ + its binding, and
`_mojolearn_rf.so` does not consume extratrees source. An extratrees source
edit stales `_mojolearn_trees.so` only. The commit message cannot be
rewritten (shared checkout, no history rewrites); this note is the
correction, placed where the next reader of the ledger will find it.

---

## DEVIATION 452 [CLOSED] -- under NUMERIC_IDENTICAL the range pass's block fold runs in KEY space

**The pathway** (cross-vendor identity audit, 2026-08-22, IDENTITY_PATHS
row 8's residue class surfacing in this lane). `node_feature_range_kernel`
reduced each block's per-thread float min/max partials with the library
block collectives (`max.gpu.primitives.block.min/max`), whose internal
cross-lane fold follows the HARDWARE warp width -- 32 on Apple and NVIDIA,
64 on AMD wavefronts. Float `min`/`max` select an input and return it
unchanged, so the fold shape is invisible in VALUE and in BITS on every
input except one: `-0.0` and `+0.0` compare EQUAL, so which zero survives
`min(-0.0, +0.0)` is decided by operand ORDER, i.e. by the fold's grouping.
A (node, column) whose rows carry both zeros could publish a range whose
sign bit differs on the AMD column, and that bit reaches the MODEL: the
`threshold == max -> min` guard returns the published min, sign bit and
all, into `Split.quesval`.

**Theirs.** cuML has no counterpart pass (DEVIATION 137 replaced the
histogram with the range pass), and CatBoost/cuML make no cross-vendor
bit-identity claim at all; there is nothing to transcribe.

**Ours.** Under `NUMERIC_IDENTICAL` (comptime `BUILD_MODE`, the
`checks/numerics.GLOBAL_NUMERIC_MODE` gate), the block fold runs over
`range_key(partial)` -- the SAME order-preserving UInt32 map the
cross-block `Atomic.min/max` merge has used since DEVIATION 204 -- so the
fold is an INTEGER min/max under a TOTAL order: any width, any grouping,
same bits, and the two zeros are ordered (`-0.0` below `+0.0`) instead of
tied. The per-thread row loop is untouched (its order is the strided
assignment, a pure function of `(count, TPB, num_blocks)`, all
data-derived). The empty-block test becomes `kmin > kmax`, which is
exactly `blk_min > blk_max` through the monotone map.

**FAST -- the default -- is bit-for-bit the old code**: the comptime
branch keeps the float fold, `kmin/kmax` are computed from its result
exactly as before, and the `RANGE_SAB_SIGN_UNFLIPPED` arm is unchanged
(the decode-then-rebit round trip is exact because `range_unkey` inverts
`range_key` bit for bit).

**Gate.** `range_kernel_check` green on this source (FAST arm; the
IDENTICAL arm compiles the same kernel with the other branch and is
exercised when the orchestrator builds the IDENTICAL column). Apple
default bits cannot move: the shipping branch is textually the old code.

---

## DEVIATION 453 [CLOSED] -- IDENTITY_PATHS row 10 applied to the ET device path's float seams

**The pathway.** Row 10's measured model: Metal's arithmetic is IEEE
correct-rounding on normals with FLUSH-TO-SIGNED-ZERO on denormal
operands, intermediates and results; CUDA's default honors denormals. Any
float seam the ET device path writes for another kernel, the reduction or
the host to read can therefore differ across vendors when a denormal is
reachable. The audit found three reachable seams and one non-seam:

  1. `draw_threshold_device` -- the range floats are SELECTED raw data
     bits (the range pass never does arithmetic on them), so a
     denormal-scale column puts denormals into `max - min` and into the
     fma; the `== max` guard then also compares flushed against unflushed.
     FIXED: `min`, `max`, the span and the fma result all pass through
     `numerics.ftz`, and the guard compares the flushed pair. The host
     oracle shares the function, so `node_feature_score_host` follows.
  2. `gain_per_split` (builder.mojo, the device gain of DEVIATION 183
     second form) -- every operand and intermediate term is bounded below
     by `1/n^2` (normal at any legal n), but the accumulated gain can
     CANCEL into the denormal band, where Metal publishes a signed zero
     and a denormal-honoring backend publishes the denormal; DEVIATION
     217's clamp then reads different signs (`-0.0 < 0.0` is false,
     `-1e-39 < 0.0` is true) and the published `best_metric_val` differs.
     FIXED: the final gain passes through `ftz` before the clamp. An
     intermediate denormal is below half an ulp of every later normal
     term and cannot survive into the result, so the final flush is the
     whole model here.
  3. `leaf_kernel`'s regression publish and its oracle `leaf_values_host`
     -- `sum/count` is safe (integer-valued operands, quotient magnitude
     `>= 2^-31`), but the `* inv_scale` dequantization can land denormal
     for a tiny-magnitude label vector. FIXED: the product passes through
     `ftz` on both sides.
  4. NON-seam, recorded so nobody re-audits it: the classification leaf
     (`count/total`, quotient in {0} U [2^-31, 1]), the reciprocals in
     `gain_per_split` (`>= 2^-31`), `node_feature_is_constant` (a
     denormal min is below half an ulp of `FEATURE_THRESHOLD` on both
     sides of the add, and the compare decides identically), and
     `mse_gain_from_exact_totals` (host Float64 from exact integers --
     IEEE-correct division and multiplication, identical on every host).

**FAST is untouched**: `ftz` is a comptime no-op under the default mode,
so every shipping value is computed by the same operations as before.
Under IDENTICAL on Metal the flushes are bitwise inert (flushing what the
hardware already flushed); on a denormal-honoring backend they align its
bits to Metal's. Each converted site cites row 10 as its authority.

**Gate.** score_kernel_check / partition_leaf_kernel_check /
device_tree_check green on this source (FAST). The IDENTICAL column's
flush behavior is certified by the cross-vendor run (E1), which these
seams exist to make bisectable.

---

## DEVIATION 454 [CLOSED] -- identity-trace checkpoints at the audit's hazard stages

The cross-vendor audit (this session) named the stages a bit can move at:
the sampler's column draw (and its host-libm dispatch), the range fold
(DEVIATION 452), the threshold draw (DEVIATION 453 site 1), the score
accumulation and the reduction. The trace recorded only the REDUCED
winners and the selected splits, so a divergence at any earlier stage
surfaced two stages late and read as "the reduction moved".

Both forest trainers now record, per merged batch, in PIPELINE order and
before the reduce records: `gN.cM.colids` (the sampler's output),
`gN.cM.range.min` / `gN.cM.range.max` (the decoded range cells),
`gN.cM.draw.thresh` (the per-cell drawn thresholds), each over the
batch's LOGICAL first `n_nodes * k` slots of the capacity-sized
workspace buffer (trace rule 3). Tags name algorithm positions only
(trace rule 2); the differ bisects by the FIRST differing tag: colids =
sampler / dispatch, range = the fold, thresh = the draw, reduce = score
or reduction. Shipping state (`MOJOLEARN_IDENTITY_TRACE` unset) is
unchanged: every record returns on one boolean test.

---

## DEVIATION 455 [CLOSED] -- the rescue path's re-stage broke 450's invariant: a MEASURED device-fit race, and its one-drain fix

**Found by the cross-vendor identity audit's Phase B run, 2026-08-22.**
`rescue_check` was RED at clean HEAD, and worse than red: two runs of
IDENTICAL source gave DIFFERENT device trees on the rescue-heavy fixture
(shaped_constant_heavy device node counts 587/459/495/581 vs
605/461/465/597 across two runs; the host counts were stable at
651/595/543/607). A device fit that varies run to run on ONE machine is an
identity failure before any second vendor enters.

**The mechanism, from DEVIATION 450's own invariant.** 450 removed five of
six per-cycle drains on the argument that "every `h_*` staging buffer is
REWRITTEN only after a required drain has retired the last copy that read
it -- every `search_batch` path ends in one". TRUE for every `stage_batch`
call INSIDE `search_batch`. FALSE for the one OUTSIDE it: when a cycle
takes the rescue (`len(retry) > 0`), the batch's work items are re-staged
before the partition -- `stage_batch` enqueues `h_items`/`h_wl`/`h_nb`/
`h_nc`/`h_blk_base`/`h_tree` copies AFTER the cycle's reduce drain, and
the next thing that touches those staging buffers is the NEXT cycle's
`stage_batch`, whose HOST writes race the still-possibly-in-flight DMA
reads. A corrupted `d_items`/`d_wl` then partitions the wrong ranges,
which is a silent wrong tree, not a crash -- exactly the failure the
`stage_batch` docstring warned about from the other side.

Why the gates missed it: the race needs a cycle that TAKES the rescue AND
a following cycle, and it is a race -- `device_batched_check`'s fixtures
either did not hit that shape or got lucky; `rescue_check` is the one
check built to hammer it, and it is the one that went red. (It was also
absent from d85c6ce's "no check regresses" scoring list.)

**The fix.** One `ctx.synchronize()` immediately after the retry-path
`stage_batch`, in both forest trainers. The rescue-free cycle -- the
common path, and the whole of 450's measured win -- keeps its single-drain
shape; only a cycle that actually rescued pays one extra drain.

**MEASURED after the fix.** `rescue_check` PASSES, twice, with
run-to-run-identical device counts -- and the device trees now agree with
the host rescue EXACTLY (651/595/543/607, 0 nodes differ on every seed),
so the entire previous red was this race. device_tree_check,
device_batched_check, device_forest_check, device_regression_check all
PASS on the fixed source.

**Standing instruction this leaves behind:** any future `stage_batch` (or
other staging write + enqueue) added OUTSIDE `search_batch` must either be
followed by a drain or must prove the next rewrite of its staging sits
behind one. 450's invariant is per-call-site, not ambient.

---

## DEVIATION 456 [CLOSED] -- the algo-L device kernel's FLOAT seams route through row 12's portable pair

**The pathway** (IDENTITY_PATHS row 12, ET consumer, 2026-08-23).
`algo_l_sample_kernel`'s float32 transcendentals (`_dev_logf32`,
`_dev_expf32` in `builder_kernels.mojo`) compiled to `std.math.log` /
`std.math.exp`, which lower to whatever each device target ships -- PTX
fast paths, OCML, Metal's own -- so `exp(x)`'s last bit was a vendor
choice, the SECOND reason DEVIATION 199 recorded for that arm never being
bit-identical to the host oracle even where `double` exists.

**Ours.** Both helpers now return `identical_log(x)` / `identical_exp(x)`
(`checks/numerics.mojo`, commit `ed0fe5d`). Under FAST each wrapper IS
the stdlib call verbatim, so FAST bits cannot move; under IDENTICAL each
is `portable_logf` / `portable_expf`, one Cephes polynomial through fma,
one arithmetic on every backend. The wrappers were chosen over calling
`portable_*` directly because the kernel's enqueue is gated on
`device_has_float64()`, not on the numeric mode -- both modes can reach it
on a double-capable target, so the seam needs both arms.

**What this does NOT close, stated so nobody reads more into it.**
DEVIATION 199's core refusal is the DOUBLE type -- `W`, `log(1 - W)` and
the truncated jump -- and the portable pair is float32. `_dev_log64` stays
the stdlib `log` under both modes, because a float32 stand-in for `1 - W`
is the catastrophic-cancellation fork 199 rejects by name. Consequences
that stand unchanged -- the Metal refusal, the kernel's never-enqueued
status on Apple, and (on CUDA or ROCm) a device arm that is still not
bit-pinned across vendors or against the libm host oracle until a portable
DOUBLE log exists. 199 carries a dated addendum saying exactly this.

**Gate.** FAST -- the full ET lane suite (`extratrees/tools/check.sh`)
green with unchanged printed numbers; `sampler_kernel_check` still asserts
the algo-L refusal by name on this box. IDENTICAL is exercised by the
orchestrator's consolidated pass.

---

## DEVIATION 457 [CLOSED] -- `n_parallel_samples_for` is mode-gated; cross-vendor is cross-HOST here

**The pathway** (IDENTITY_PATHS rows 12 and 18, ET half). The sampler
dispatch count `ceil(log(1 - k/n) / log(1 - 1/n))` ran through HOST libm
(DEVIATION 158's FFI arm) on every build. libm is host-vendor arithmetic
-- macOS and glibc differ in `log`'s last bits -- and this value is turned
into an INTEGER whose 128/9216 comparisons pick WHICH KERNEL runs, so two
hosts could train different forests from identical inputs. The device is
not involved at all; this seam's cross-vendor axis is cross-HOST.

**Ours.** The function splits into two named arms plus a comptime gate --
`n_parallel_samples_libm` (the exact former body, libm in double, what
FAST compiles to and nothing else) and `n_parallel_samples_portable`
(both logs through `portable_logf`, float32, what IDENTICAL compiles to).
In the portable arm every non-log seam is already one arithmetic
everywhere and is left as a single operation each -- `Float32(k)` and
`Float32(n)` are exact below 2^24, the subtract and both divides are
single correctly rounded IEEE operations on every host, and `ceil` of a
float is exact because its result is integral.

**MEASURED, the knife edge made visible.** Swept on this host (M4, macOS
libm) over n in {64, 500, 1000, 2000, 20000}, all k: the two arms differ
on most counts at large n (17278 of 19999 k at n=20000, every difference
in [-36, -1]; float32 cannot form 1 - 1/n to double precision) and cross
a dispatch boundary at EXACTLY ONE swept pair -- (n=20000, k=7385), libm
9217 = algo-L against portable 9216 = excess. That pair is row 18's knife
edge, now printed and asserted by `feature_sampler_check`'s DEVIATION 457
sweep case rather than latent. A different count is a different (still
correct) draw budget, so the only behavior that moves is kernel choice,
and it moves only under IDENTICAL, whose bits differ from FAST BY DESIGN.

**Check plumbing this forced.** The n=20000 boundary pins in
`feature_sampler_check.mojo` and `sampler_kernel_check.mojo` were
FAST-libm facts (smallest algo-L k = 7385, exact-9216 k = 7384). Both
files now compute the pinned k from the build mode (`ALGO_L_FRONTIER_K` =
7385 FAST / 7386 IDENTICAL, `EXCESS_EDGE_K` = 7384 FAST / 7385 IDENTICAL),
so under FAST they compile to the exact values and printed numbers they
have always had, and the orchestrator's IDENTICAL pass pins the portable
arm's boundary instead of going red on the libm arm's.

**Gate.** FAST -- ET lane suite green, `feature_sampler_check` prints the
sweep tallies and the single ARM FLIP line above.

---

## DEVIATION 458 [CLOSED] -- the regressor binding overwrote `max_features` after reading it

**What the port did.** `bindings/_mojolearn_trees.mojo` reads the 21-slot
params list into an `ExtraTreesConfig` (`_config_from`, slots 3-19, with
slots 8-9 = `max_features_spec` / `max_features_fraction`) and then, on the
regressor entry only, called `.for_regression()` on the RESULT:

    var config = _config_from(params).for_regression()

`for_regression()` is `estimator.mojo`'s DEFAULTS switch -- it sets
`criterion = MSE` AND `max_features_spec = MAX_FEATURES_ALL` (sklearn's
regressor default of 1.0). Applied after the read, the second assignment
overwrote whatever the caller sent in slots 8-9. Every
`ExtraTreesRegressor` fit, host arm and device arm, sampled ALL columns at
every node; `max_features_` reported `n_features` for every form. The
classifier entry builds from `ExtraTreesConfig()` and never re-applied a
default, so it honoured every form -- which is why the lane's Python
surface looked right whenever the classifier was the one under test.

**Why no check saw it.** Every check in `extratrees/tools/check.sh` is a
Mojo program that constructs an `ExtraTreesConfig` directly; none crossed
the Python boundary. `bindings/build_trees.sh`'s smoke fits the regressor
at its default (1.0), the one value the overwrite preserves. The module
docstring's "every parameter is honoured or refused by name" was false for
this one parameter on this one estimator, and nothing was positioned to
say so. A path that runs is not a path that is gated.

**MEASURED** (E2 fixture, 20000x24 uniform float32, integer-matmul target,
`n_estimators=10, max_depth=8, random_state=7`, prediction sha256 prefix):

    max_features      before (all five = one forest)   after
    1.0               49243f47fb559159                 49243f47fb559159
    0.5               49243f47fb559159                 11d4dad380b22bd7
    0.1               49243f47fb559159                 f0be1f4b94dc52c8
    3                 49243f47fb559159                 4efabfa1836532ab
    "sqrt"            49243f47fb559159                 717ceb5e2c6a01c6

`max_features_` after: 24 / 12 / 2 / 3 / 4, sklearn's resolution of each
form on 24 columns. The 1.0 hash is UNCHANGED -- 1.0 was the one form the
overwrite happened to agree with, so the E2 `et_reg` cell stays comparable
and nothing about the 1.0 result was wrong. The classifier's hashes are
bit-unchanged on the same fixture (1.0 `05b257ee…`, 0.5 `0832c8a9…`, 3
`7da20d1d…`, "sqrt" `f6ecbc1a…`): its entry did not move.

**The fix.** `_config_from(params, base)` now takes the defaults struct as
an argument and writes slots 3-19 OVER it; the classifier passes
`ExtraTreesConfig()`, the regressor `ExtraTreesConfig().for_regression()`.
The criterion still does not ride the params (the `N_FIT_PARAMS` contract
is unchanged), it arrives in `base`, and no default can shadow a slot
because every slot is written after the base is taken. One line moved in
each entry; `estimator.mojo`, the Python wrapper's slot encoding and the
sentinels are untouched. The wrapper's regressor still folds a float `1.0`
into `None` before the slots; both resolve to `n_features`, so that fold
is a no-op and is left as is.

**Gate.** `extratrees/tools/wrapper_reach_check.py`, run by `check.sh` after
the Mojo checks against the SHIPPED `_mojolearn_trees.so`: for the
regressor (gpu AND cpu arms) and the classifier, `max_features` 1.0 / 0.5 /
"sqrt" / 3 must build pairwise DIFFERENT forests (sha256 of the five forest
arrays) and `max_features_` must equal sklearn's resolved count. Sabotaged
against HEAD's pre-fix extension: the regressor's four forms collapse to
ONE digest per arm (three FAIL pairs on gpu, three on cpu) and
`max_features_` is 16 for every form (three FAILs per arm), classifier ok
-- the check sees exactly the defect and nothing else. FAST rebuild shipped; the
full lane suite result is in the commit message.

---

## DEVIATION 459 [CLOSED] -- Entropy: cuML's float gain is the key, published as a FLOAT SEAM beside the integer Gini core (NOT_IMPLEMENTED.tsv row 11 retired)

**Theirs.** `EntropyObjectiveFunction` (`objectives.cuh:110-193` at
`00094f7`): `GainPerSplit` (`:132-168`) is the information gain in `DataT`,
`raft::log(p) / raft::log(DataT(2))` per class per child with the
`min_samples_leaf` early return and the three `!= 0` guards; the builder
reduces candidates on that float through `Split::update` (`split.cuh:76-
90`): greater gain wins, equal gain falls to `colid`, then `quesval`.
`SetLeafVector` (`:181-191`) is byte for byte Gini's.

**Ours.** `objectives.mojo::EntropyObjectiveFunction` transcribes
`GainPerSplit` statement for statement (DEVIATION 143's two index edits,
their locals' names, THEIR association order -- no `fma` introduced where
theirs has none, because the gain IS the key here and an fma would be a
different key), plus sklearn `Entropy.node_impurity` /
`children_impurity` (`_criterion.pyx:548-602`) for the reported
impurities, and `GainKeyExact`, which is the whole deviation:

The Gini core selects on DEVIATION 144's EXACT INTEGER rational, published
per cell as `(num, den)` and compared by cross-multiplication in the
reduction (`split.mojo::compare_exact_key`). Entropy CANNOT live in that
core -- `p log p` has no rational form in the counts, and a fixed-point log
table would be an invention neither upstream has. So entropy is a FLOAT
SEAM published THROUGH the same integer pair: `num =
float_gain_key(gain)`, `den = 1`, `valid = 1`, where `float_gain_key`
(`builder_kernels_impl.mojo`, shared by the kernel and the host oracle) is
the order-preserving sign-magnitude map of the Float32 gain onto `Int64`.
With `den == 1` the reduction's cross-multiply IS an integer compare of the
keys, so the candidate order is exactly cuML's float `>`; an EQUAL key
(equal gain bits -- and the map gives `-0.0` and `+0.0` the same key, as
float `==` does) falls through to DEVIATION 194's tie arms, which are
cuML's `colid`, then `quesval`. `range_key` was NOT reused: it orders the
two zeros, right for a range fold and wrong for a gain tie. Nothing in the
reduction, the readback, `split_not_valid` or the host-side `Split`
construction changes; `score_to_candidate_kernel` gains a `criterion`
argument and, for ENTROPY, computes the gain through the ONE transcription
(`builder.mojo::entropy_gain_per_split` bitcasts the kernel's Int32
accumulators to `CountBin` -- no second copy to drift, unlike Gini's
historical `gain_per_split`) and publishes it as metric AND key. The host
`node_split_random_gini` takes the same `criterion`, builds the same
`GiniProxyExact(key, 1)` and runs the same `CompareProxyExact`, so host
and device ORDER identically by construction; a device/host difference
can only come from the gain BITS.

**The log, and where its bits come from.** Every `raft::log` is
`identical_log` (`checks/numerics.mojo`, IDENTITY_PATHS row 12) --
the shape `ensemble/decisiontree/batched_levelalgo/objectives.mojo`'s
DEVIATION 406 gave RF's entropy, with its `_log_seam` / `_ftz_seam`
helpers copied. Under NUMERIC_FAST the wrapper IS `std.math.log`
verbatim, so the default build is cuML's expression through each
backend's own lowering; under NUMERIC_IDENTICAL it is `portable_logf`,
one Cephes polynomial through fma on every backend, and every stored
intermediate is `ftz`'d (row 10) in cuML's association order so the FAST
arm is the transcription's arithmetic unchanged. The 217 clamp applies
(the information gain is provably non-negative; a negative float is
cancellation and would leaf the node through `split_not_valid`).

**What is claimed about device == host, and what is not.** Under
IDENTICAL the host arm and the device arm compute the same `log` and the
same pinned arithmetic, so the entropy forest is bit-identical across the
arms, and `device_forest_check` ASSERTS it under that build. Under FAST
the host's `std.math.log` and Metal's `log` are two lowerings; a last-bit
difference in a gain can re-decide a near-tie. `device_forest_check`
REPORTS the node difference under FAST instead of asserting zero (0 of
424 nodes on the 4-class hashed fixture on this M4, 2026-08-23, and 0 of
the 20000-row E2 fixture's forest: `et_clf_entropy` 10 trees hash equal
gpu vs cpu through the wrapper -- measured, not promised). The wrapper's
docstring states this. The exact-argmax guarantee DEVIATION 144 gives
Gini is NOT extended to entropy.

**Price.** Entropy candidates are ordered by a Float32 built with a
vendor `log` under FAST -- two candidates whose true gains differ by less
than the rounding can order either way, which is cuML's own behaviour;
sklearn (float64, libm) is not bit-comparable to either.

**MEASURED** (wrapper_reach_check fixture, 4096x16 hashed, 4-class target,
`n_estimators=4, max_depth=6, random_state=7`, forest sha256 prefix):
gini `c24fb8381387eb18`, entropy `1ba1ae84157063fa`, log_loss
`1ba1ae84157063fa` (the alias), same on gpu and cpu; SABOTAGE -- the
wrapper says entropy, params slot 21 forced to gini's code -- collapses
to `c24fb8381387eb18`. On the separable BINARY fixture the build smoke
used (512x6), gini and entropy built the SAME forest (`bb123f221810c43b`)
-- a binary separable target cannot see the criterion, so every reach
check uses a 4-class one. E2 (20000x24, 10 trees, depth 8, seed 7):
`et_clf_entropy` pred `b8924c3a87f9aba5` FAST, `et_clf_logloss_alias`
equal. `objectives_check` holds the f32 `GainPerSplit` within 1e-5
relative and the f64 one within 1e-12 of an independent class-major
float64 information gain on 64 hashed candidates, the sklearn impurities
and proxy within 1e-12 of their definition, the key's order equal to the
float's on every consecutive pair, the two zeros tied, and 5 consecutive
candidate pairs ranked oppositely by gini and entropy (the fixture CAN see
the criterion).

**Gate.** objectives_check (entropy section), host_splitter_check,
estimator_check (entropy accepted and reaches `DecisionTreeParams`;
poisson still refused), device_forest_check section 11, params_check
(ENTROPY moved to the accepted list), wrapper_reach_check (the table above
with the slot-21 sabotage), and the build smoke's 4-class reach line.
Existing cells' hashes unchanged (E2 FAST: et_clf `f6ecbc1ac97d3cf0`,
et_reg `49243f47fb559159`, et_clf_allfeat `05b257ee8fe8c61a`,
et_clf_3class `2ed393e4c6085e0e`, et_clf_deep `3d78c1ea6e3ba776`, et_reg_cpu
`f1e6628a87edf97e`, all equal to the pre-change baseline run).

---

## DEVIATION 460 [CLOSED] -- bootstrap=True: cuML's `get_row_sample` bootstrap arm, through the RF lane's Philox port; `max_samples` as sklearn resolves it (NOT_IMPLEMENTED.tsv row 15 retired)

**Theirs.** `randomforest.cuh:50-72`: `rs = fnv1a32(fnv1a32(basis,
seed), tree_id)` (`:59-62`), `raft::random::Rng rng(rs, GenPhilox)`, and
`rng.uniformInt<int>(selected_rows->data(), selected_rows->size(), 0,
n_rows, stream)` when `bootstrap` (`:64-67`) else `thrust::sequence`
(`:68-70`). `selected_rows.size()` is `n_sampled_rows`, cuML's
`max_samples * n_rows`.

**Ours.** The `bootstrap == true` arm is ported on BOTH arms of the
forest, and nothing about it is invented here:

* the seed chain is `checks/pcg_rng.mojo::row_sample_seed` -- their
  two lines on this lane's own `fnv1a32`, WITH the RF lane's DEVIATION
  400 (`ensemble/.../random_utils.mojo::fnv1a32_hash_seed_tree`): the
  high half of a 64-bit seed gets its round exactly when nonzero, so every
  seed below 2^32 keeps the transcription's bits and a bootstrap forest
  here and in `ensemble/` draw the same rows from the same `(seed,
  tree)`. `forest_check` pins six values computed by hand in Python.
* the draw is `core/philox.mojo`, the RF lane's port of RAFT's Philox
  `uniformInt` with its oracle and its DEVIATION 184 geometry (the
  stride pinned to 4 x 108 x 256, because RAFT's mapping is a function of
  the GPU model and ONE constant had to be chosen): the host arm calls
  `uniform_int_host` (`randomforest.mojo::row_sample_for`), the device
  arm calls `launch_uniform_int` ONCE PER TREE SLOT on a
  `create_sub_buffer` view of that slot (`builder.mojo::fill_row_slots`,
  replacing `row_ids_tiled_sequence_kernel` when bootstrapping). No
  synchronize after the draw, as theirs has none: every reader is
  queue-ordered behind it. The two arms draw the SAME rows, row for row
  (`device_forest_check` section 11(a), 1152 slot cells, 0 mismatches).
* the per-tree row SLOT is `n_sampled_rows` wide (`selected_rows.size()`),
  not `n_rows`: `NodeQueue`'s root count, the slot base, the group cap
  and the workspace's row-scaled pieces all take `slot_rows`; the
  resident dataset stays `n_rows` (M) wide and a slot's entries INDEX it.
  With-replacement duplicates are fine everywhere a row id is READ
  (score, range, leaf, partition -- the partition permutes positions, not
  ids), which is how cuML's builder consumes its own bootstrap.
* `max_samples` crosses the Python boundary ALREADY RESOLVED to a count by
  `_n_samples_bootstrap` -- sklearn's `_get_n_samples_bootstrap`
  (`ensemble/_bootstrap.py:10-62`, unweighted branch): None -> `n_rows`,
  int -> itself, float f -> `max(int(f * n_rows), 1)`; the value
  constraints are `_forest.py:198-202`'s. The count rides as params slot
  18 (the old 0/1 `max_samples_set` flag, which only existed to refuse),
  `ExtraTreesConfig.max_samples: Int` (0 = None), `FitPlan.n_sampled_rows`,
  and is reported back as `meta[5]` / `_n_samples_bootstrap` so a check can
  see the knob reach the far side.
* what stays REFUSED, by name: `max_samples` without `bootstrap` (sklearn
  raises, `_forest.py:411-416`; `resolve_n_sampled_rows` refuses on both
  arms), and `oob_score` -- the out-of-bag set needs the per-tree
  bootstrap MASK (`RowSampler::store_bootstrap_mask`, `:170-183`), which
  neither this lane nor `ensemble/` carries. The refusal text names that,
  not "bootstrap is not ported".
* the identity trace (DEVIATION 454) records `gN.bootstrap.rowids` --
  the drawn slots before any level touches them -- when bootstrapping, so
  a cross-vendor Philox difference has an address ahead of the first
  `partition.rowids`. Algorithm position only.

**MEASURED** (wrapper_reach_check fixture as in 459, forest sha256 prefix):
classifier no-bootstrap `c24fb8381387eb18`, bootstrap `9f7ccc54f231982f`
twice at random_state 7, max_samples=0.5 `8c3bb214800eddb5` reporting
2048, SAME on gpu and cpu; SABOTAGES -- slot 11 forced to 0 under
bootstrap=True collapses to `c24fb8381387eb18`, slot 18 forced to 0 under
max_samples=0.5 collapses to `9f7ccc54f231982f`. Regressor gpu plain
`47a66dec39eb5828` / bootstrap `a9e235c0a9722740` (twice), cpu
`b6c8e13769e61582` / `e8076e105848a6a4` (twice). E2 (20000x24, 10 trees,
depth 8, seed 7, FAST): `et_clf_bootstrap` `5b62e7ed79521ca1`,
`et_reg_bootstrap` `ba50139fd1769415`, `et_clf_bootstrap_maxsamples`
`6da435a95ef785b8` -- the two former REFUSED= cells run. `forest_check`:
1024 draws from 1024 repeat (with replacement), every id in range, tree 0
and tree 1 differ, a 300-draw sample is the 1024-draw sample's prefix
(thread t writes t, t+stride, ...; 300 < stride), bootstrap forest !=
plain, == itself, and every root of a `n_sampled_rows=512` forest holds
exactly 512 rows. `device_forest_check`: device bootstrap forest ==
host bootstrap forest bit for bit (also at max_samples=700), != plain,
repeats. `device_regression_check`: device bootstrap regressor structure
== host over 396 nodes, != plain.

**Gate.** forest_check, estimator_check, device_forest_check sections 10
and 11, device_regression_check, wrapper_reach_check, the build smoke.

---

## DEVIATION 461 [CLOSED] -- the Python boundary: 22 params slots (criterion rides as slot 21), slot 18 is a count, `meta[5]` reports the sample size

`N_FIT_PARAMS` 21 -> 22. While each entry point had exactly ONE criterion
the criterion did not ride the params list -- it arrived in the `base`
config DEVIATION 458 introduced. With ENTROPY admitted (459) the classifier
entry has two, so the criterion is slot 21 (`decisiontree.mojo`'s
`CRITERION_*` code), written over `base` like every other slot, and each
entry point checks the code it admits by name (`et_classifier_fit`: GINI or
ENTROPY; `et_regressor_fit`: MSE). Slot 18 carries `max_samples` RESOLVED
TO A COUNT (460) instead of a 0/1 flag. `meta` grows to six:
`n_sampled_rows` last. The wrapper's `_fit_params` names the 22 slots in
the same words; `wrapper_reach_check` sabotages slots 11, 18 and 21 from
the Python side and watches each forest collapse to the unsabotaged one.
The 21-slot contract never shipped in a release, so no compatibility arm
is kept.

DEVIATION 462 is reserved and unused by this round.

---

## DEVIATION 463 [CLOSED] -- exact ties resolve by a keyed pseudorandom rank, not by the highest column id

**MOTIVATION, measured before any fix.** covtype (522,911 x 54, depth 16,
`max_features=sqrt` -> 7, seed pinned both arms): our ET 0.6701 accuracy
against sklearn's 0.6768. An audit of the three tie-adjacent seams found this
one first and rated it the likely cause: DEVIATION 145's exact integer
rational makes ties REAL, frequent events (integer class counts over
duplicate-heavy columns collide exactly, not within-an-ulp), and every one of
them resolved through cuML's `other.colid > colid` arm -- toward the HIGHEST
column id. covtype's columns are 10 informative continuous ones at ids 0-9
and 44 one-hot indicators at ids 10-53, so the max-colid rule systematically
funneled tied nodes into the one-hot block, tree after tree, at every depth.
sklearn's `>` at `_splitter.pyx:693` gives a tie to the FIRST candidate in a
uniformly-random visit order -- uniform among the tied.

**Ours now.** On `compare_exact_key(...) == 0` between two VALID keys,
challengers are ranked by `split_tie_rank(tie_salt, colid) =
fmix32(fnv1a32(tie_salt, colid))` with `tie_salt =
split_tie_salt_for(tree_id, node_id)` (a fresh salt constant
`SPLIT_TIE_SALT`, then tree, then node -- fnv1a32 throughout), and the
GREATEST rank wins; a rank collision falls back to colid, then quesval, so
the order stays TOTAL and the reduction stays blind to grouping and arrival
order (DEVIATION BLOCK 166's argument is untouched). Uniform-among-ties in
sklearn's sense, deterministic from `(tree, node, colid)` alone, pure integer
math, identical on host and device -- the IDENTICAL cross-vendor mode is
unaffected. One copy: `split.mojo::keyed_tie_wins`, CALLED by
`SplitExact.update` and by `host_splitter._wins_on_total_order`. The device
reduction gets the salt as a per-node kernel input (`node_tie_salt`, staged
by `stage_batch` from the same `(item_trees[i], work_items[i].idx)` the host
oracle keys on). The NO-KEY degeneration (both keys invalid, DEVIATION 166)
is untouched: still `Split.update` exactly, still arm A's subject; no
shipping caller reaches it (device regression carries valid MSE keys,
DEVIATION 189).

**Arms.** `-D MOJOLEARN_ET_TIE_MAX_COLID=1` rebuilds the OLD max-colid rule
everywhere (host and device together, so oracle parity holds in both
builds); `SPLIT_SAB_MAX_COLID_TIE` (= 8) restores it at runtime in the
shipping reduce kernel for the checks. DEVIATION 133 is amended, not
retired.

**The counter, so the rate is a number and not an argument.** Host:
`HostSplitResult.n_tied_best` -- scored candidates whose key exactly ties the
winner's, winner included; `>= 2` means the tie-break arm decided the node.
Device: `split_tie_count_kernel` (launched, with its readback and a
per-batch `ET_TIE_STATS batch decided=<d> tied=<t>` print, only under
`-D MOJOLEARN_ET_TIE_STATS=1`) computes the same per-node quantity from the
reduced winner and the candidate arrays; the exact-tie RATE is
`sum(tied)/sum(decided)` over the printed lines.

**Pre-registered predictions (written before the A/B ran).** (1) The
exact-tie rate on covtype is HIGH at depth 16 -- deep nodes are small and
integer-count collisions dominate -- and the keyed arm's accuracy moves
toward 0.6768 while `MOJOLEARN_ET_TIE_MAX_COLID` reproduces ~0.6701. (2) The
per-tree structure changes on a large fraction of trees (any tied node
cascades). If the rate comes back LOW, this entry's "likely cause" rating is
wrong and must be re-argued against 464/465.

**Price.** Two hash evaluations per exact-tie comparison (a handful of
integer ops; ties only), 4 bytes per work item staged per batch, and one
more kernel argument. Bit-compatibility with every pre-463 model is
deliberately broken on any fit that ever hit an exact tie.

**Gate.** `split_reduce_check` (oracle re-expressed independently as
`oracle_rank`, fixture ties planted at the top, permutation arm B),
`regression_score_check` arm E, `host_splitter_check` via the shared
`_wins_on_total_order`, and the A/B above.

**CHECK-ROUND ADDENDUM (2026-08-26, first Apple run of the merged branch).**
Three checks went red and each taught something. (1) `split_reduce_check`,
two arms: the planted tie group's colid classes were {singleton, pair, pair}
and the per-node rank crowned the SINGLETON in three of five nodes
(recomputed exactly from the rank arithmetic, not observed), so E1's
quesval-decisiveness floor and the NO_QUESVAL sabotage's must-not-move cells
both failed -- the fixture assumed max-colid would always crown a
multi-member class. Fixed by planting TWO classes of THREE members each,
decisive under any rank outcome and under the max-colid build alike. (2)
`host_splitter_check`: its independent argmax broke exact-rational ties by
FIRST-WINS (strict `>` only) -- dead code by luck until 465's re-rolled
draws produced a real tie (features 0 and 4, hashed/gini fixture) that the
keyed rule resolves to feature 4. The check now re-derives the rank in its
own transcription (`_tie_rank_ref`). (3) `regression_score_check`, one arm:
an exact host-side recomputation of every draw-dependent gate under the new
chain (thresholds, statuses, keys, winners, pair orders -- validated against
the run's own "20 scored cells" line) shows every floor passing and the
reduce's host/device folds equal in exact arithmetic, so the remaining
suspect is the new salt staging buffers' LIFETIME (freed at their last
enqueue, the trap this repo documents and this very check already defends
`d_tree_ids` against); both checks now hold their salt buffers past the
drain, and the reduce arm's mismatch print now reports the tie multiplicity
so a recurrence names its mechanism.

**CHECK-ROUND 2 (2026-08-26, after the fixture fixes).** `split_reduce_check`
went green; the regression red arm was then LOCATED: not the reduce at all,
but the slot guard's WRAP arm. The old poison (2140000000 / -1070000000 /
-1070000000) could only refuse-and-wrap when a draw ISOLATED the one
positive row, and 465's re-rolled draws never do at this seed -- so all five
refused cells were node 0's OVER_SLOT cells, whose numerators cannot reach
the wrap bound (max `(|A| >> 5)^2 ~ 4.5e18 < 2^63`). The poison is now
derived, not tuned: labels (+2066666667, -2066666667, 0) sum to zero, so a
3-row split isolating row `x` has `|A| = |3x|`; either big row gives
`|A| = 6200000001 >= 6074001000` (the first `(|A| >> 1)^2` past
`Int64.MAX`) AND puts `> 2^30` on both sides -- refuse-and-wrap on 2 of 3
rows, the provable maximum (the three `A`s sum to zero, so all three
exceeding 6.074e9 in magnitude needs a spread Int32 labels cannot span).
The zero label parks on the one position no current draw isolates (verified
in exact arithmetic; placement, not substance). `host_splitter_check` had a
SECOND verbatim old-rule assertion (`check_tie_break`: "greater colid wins
the tie", colid 1 pinned); it now derives the expected winner per tree from
the rank -- and reports the winner flipping with the tree id, which is the
uniform-among-ties property made visible. A file-wide sweep found no other
pinned tie winners (the analytic gap/step/145 sections all pin STRICT
orders).

---

## DEVIATION 464 [CLOSED] -- `excess_selection_hash` runs through a full-avalanche finalizer

**The defect.** DEVIATION 215's fix ranked an overshoot's unique columns by
`fnv1a32(fnv1a32(fnv1a32(fnv1a32(BASIS, SALT), tree), node), col)` and kept
the `k` smallest by hash. For `col < 256` only the first byte round of the
final step does anything; the three trailing zero-byte rounds collapse to an
affine map `K + perm(col) * C mod 2^32` with `C = FNV1A32_PRIME^3`, and
`C / 2^32` is within 2^-9 of 5/16 -- so ranking columns by this hash walks a
~16-cluster Weyl rotation, not a random permutation. Every MARGINAL stays
uniform (each column's inclusion rate is fine), but the JOINT k-of-uniques
selection -- which columns co-occur in a node's candidate set -- is
structured, on every node that takes the overshoot branch (~58% of covtype
nodes at `(n=54, k=7)`).

**The gate that missed it, named.** The sampler distribution gate
(`sampler_kernel_check` section 8) only ever asserted `(n=28, k=5)`
MARGINALS -- per-column counts in a +/-15% band. A marginal gate cannot see
a joint distortion, which is why `SAMP_SAB_RAW_SELECTION_HASH` is documented
as an arm that gate does NOT catch; a co-occurrence assertion is the missing
gate and is left to the check's owner, on the record here.

**Ours now.** The combined word runs through murmur3's `fmix32`
(`pcg_rng.mojo`) after the FNV combine -- `h ^= h>>16; h *= 0x85ebca6b;
h ^= h>>13; h *= 0xc2b2ae35; h ^= h>>16` -- a bijection, so marginals stay
uniform and no collisions are introduced; the joint selection becomes what
DEVIATION 215 claimed. Host and device share the one function, so the
slot-for-slot host-vs-device control stays bit-identical in both builds.

**Arms.** `-D MOJOLEARN_ET_RAW_SELECTION_HASH=1` rebuilds the un-finalized
chain everywhere; `SAMP_SAB_RAW_SELECTION_HASH` (= 10) selects it at runtime
in the device kernel (differs from the host oracle exactly on overshoot
nodes).

**Pre-registered prediction.** Column MARGINALS were fine before this fix
and stay fine after it (the existing section-8 band moves by noise only);
what moves is co-occurrence, and with it which candidate sets deep covtype
nodes see. Expected accuracy contribution: smaller than 463's, possibly nil;
it is a correctness fix for 215's stated claim either way.

**Price.** Five integer ops per hash evaluation, only on the overshoot
gather. Column sets change on every overshoot node, so pre-464 fits are not
bit-reproducible under the default build.

---

## DEVIATION 465 [CLOSED] -- `key_for` gets its own salt link; the threshold stream no longer collides with the sampler's

**The invariant violation, one line.** `key_for` (pcg_rng.mojo) chained
`fnv1a32(fnv1a32(fnv1a32(BASIS, feature_id), tree_id), node_id)`;
`excess_subsequence` (builder_kernels.mojo) chains
`fnv1a32(fnv1a32(fnv1a32(BASIS, thread_id), tree_id), node_id)`. For
`feature_id == thread_id` -- covtype's informative columns 0-7 against the
sampler's threads 0-7, every node -- the two produced the SAME PCG
subsequence: the threshold draw and the sampler draw were correlated streams
by construction. The surrounding comments already asserted disjointness
(`excess_selection_hash`'s "the salt keeps this stream disjoint from
`excess_subsequence`'s and `key_for`'s", `rescue.mojo`'s reserved-slot
argument), and `key_for`'s own docstring said "We chain FOUR" over a body
that chained three. The comments were right and the code was wrong; fixed in
the code's direction that the comments describe.

**Ours now.** `THRESHOLD_KEY_SALT` (fresh constant `0xA24BAED4`) is the
FIRST link, making the chain actually four: salt, feature, tree, node. The
stale docstring is rewritten. Every threshold in every fit changes;
host and device move together because both call the one `key_for`.

**The oracle pins moved with it.** `tools/rng_oracle/main.cpp::key_chain`
carries the same salt, and `pcg_reference.txt`'s seven `chain` lines were
regenerated by an independent Python transcription of fnv1a32 that FIRST
reproduced all seven OLD committed values bit-for-bit (so the transcription
is checked against the upstream-built reference before being trusted for the
new values). The stream/uint/ufloat sections are self-describing
(seed, subsequence, offset recorded per case) and remain valid pins
unchanged; a full re-run of `build.sh` regenerates their subsequences under
the new chain without changing what they assert. Under
`-D MOJOLEARN_ET_KEY_UNSALTED=1` (the A/B arm) the chain pins go RED by
design -- that define is for measurement, not for gates.

**Pre-registered prediction.** This fix changes every draw, so its covtype
delta is NOISE-LIKE (a re-seed), plus whatever systematic effect the 0-7
stream correlation had; it is not expected to carry the 0.0067 on its own.
The invariant, not the accuracy, is the reason it ships.

**Price.** Four more FNV rounds per key (once per (node, feature) draw), and
every pre-465 model, fixture hash and identity trace that embedded a
threshold is invalidated.

**Gate.** `pcg_rng_check`'s chain pins (regenerated), `range_draw_check` /
`score_kernel_check` / `host_splitter_check` / `device_*_check`
host-vs-device parity (both sides move together), and the build smoke.


---

## DEVIATION 1943 [CLOSED] -- the frontier block is 512 threads on a 64-lane wavefront, 128 on a 32-lane warp

**Where.** `impl/decisiontree/batched_levelalgo/builder.mojo`, `_device_tpb()`
(the one definition `DEVICE_TPB` reads); the measurement harness is
`tools/et_profile_leg.sh`, the `et-profile` case of `tools/e2_remote_leg.sh`.

**Theirs.** `TPB_DEFAULT = 128` (`builder_kernels_impl.cuh:33`) and
`n_blocks_per_node = ceildiv(count, TPB_DEFAULT)` (`builder.cuh:365-385`),
ONE ROW PER THREAD, on CUDA only.

**The number this answers.** `bench/results/BOARD_2026-08-28_three-vendor.md`
2.3: ExtraTrees on higgs 1M / 2M, same source, FAST, 4160 / 7661 ms on an
H100 and 18294 / 35521 ms on an MI325X, while rf and iforest were FASTER on
the MI325X. Nothing had been profiled.

**Measured, MI325X, FAST, `extratrees/bench/fit_once.mojo` with its
`PhaseClock`, higgs 1M x 28, 100 trees, depth 16, sqrt features (leg
`bench/results/e1/2026-08-29_202227-mojolearn-e2-amd/lanes/et_profile/`).**

| pass | TPB 128 (shipped then) | 256 | 512 |
|---|---|---|---|
| range (init+range+decode+nonconst) | 8107 ms | 4614 ms | 2819 ms |
| score (init+score+finalize) | 8799 ms | 5186 ms | 3288 ms |
| partition (4 kernels) | 221 ms | 192 ms | 185 ms |
| candidate+reduce+readback | 43 ms | 44 ms | 47 ms |
| leaf pass | 34 ms | 34 ms | 35 ms |
| stage + sampler | 259 ms | 212 ms | 194 ms |

`rocprofv3 --kernel-trace --stats` on the same fit: 19 kernels, two of them
(474 calls = `node_feature_range_kernel`, 286 calls =
`node_feature_score_kernel`) hold 99.2% of the device time, 17002 ms at TPB
128 and 6207 ms at TPB 512. The 2M rung: 108578 / 94261 / 86452 ms whole
fit at 128 / 256 / 512.

**The cause.** Both hot kernels are launched at
`grid = (n_blocks_dimx, k)`, one block per 128 rows per node
(`build_workload_info`, `builder_kernels_impl.mojo`), and each thread
handles exactly one row (`stride = TPB * num_blocks`, `range_len <= TPB *
num_blocks`), then the block runs six collectives and thread 0 publishes
three to seven atomics. On CDNA a 128-thread workgroup is two wavefronts,
so at higgs 1M the two passes launch about 2.6M two-wave workgroups per
level cycle that each do one compare per lane. The MI325X is bound by the
workgroup dispatch rate, not by the work: halving the workgroup count
halves the pass, twice. The H100 runs the same 128 and its whole fit is
4160 ms, so the width is keyed on the compile-time fact that separates the
two, `WARP_SIZE`, and the row is a no-op by construction where it is 32
(NVIDIA, Apple).

**What was NOT it.** The `atomicCAS` spin lock the memory notes remember as
80% of ET runtime survives only in `split_reduce_kernel`
(`split.mojo:1036-1052`), which launches `ceildiv(k, TPB)` = one block per
node at every `k <= TPB`; the reduce phase is 43 ms. The range pass's merge
is four lock-free integer atomics (DEVIATION 204). The score pass's
per-thread private arrays are `MAX_ACC = 32` ints; not scratch-bound (the
pass scales with the block count, not with the class count).

**The bits.** The tree cannot depend on TPB: the range fold is in key
space under IDENTICAL (DEVIATION 452) and an integer min/max across
blocks (DEVIATION 204); the score pass sums integers through atomics
(DEVIATIONS 135, 171); the partition is stable by row index at any tiling
(DEVIATION 203); the reduce is one block per node. Verified on the second
leg, `bench/results/e1/2026-08-29_204736-mojolearn-e2-amd/`, at commit
9c8ffc23 with the row live: phase 9 `stability/identical.txt` et-clf
`707d2c92e92f6a40`, et-reg `a1d2eb9ddb149617`, 6/6 STABLE, the same two
hashes as the Apple M4 and the RTX 4090 columns
(`bench/results/stability/2026-08-29-apple-29lanes/identical.txt`,
`bench/results/e1/2026-08-29_130931-runpod-nvidia/stability/`); FAST the
same two hashes. `tools/identity_break.py --lanes et-clf,et-reg` under
IDENTICAL: 18 of 18 cells equal the Apple column in
`bench/results/identity_break/apple-m4.identical.json` (et-clf/base
`22334a0351e6ce65`, predict `9ccaf28b51d37da6`, proba `d9d163336091bd8b`;
et-reg/base `754d8c127ecfc04d`), file
`lanes/et_profile/identity_break.et.identical.txt`. Both arms of
`fit_once` build the same 1,829,804 nodes at 1M and 2,173,962 at 2M.

**After, same leg, same box.** Device kernel time at 1M 16983 ms with
`-D MOJOLEARN_ET_TPB_128` versus 6200 ms shipped; range 8080 -> 2815 ms,
score 8803 -> 3289 ms. The whole-fit number is NOT the 18294 ms row's
opposite, because of DEVIATION 1945 below.

**Price.** None on a 32-lane vendor (the value is unchanged). On a 64-lane
vendor, `split_reduce_shared_bytes(512, 64) = 288` bytes of shared memory
and a 2*512-int `smem` in `node_split_kernel` (the oracle kernel, not the
shipping path); `partition_iteration_bound` and the workspace's
`1 + max_batch + n_rows // tpb` block bound take the same TPB.

**Owed.** The NVIDIA and Apple ET timings are unchanged by construction
(`WARP_SIZE == 32` selects the old 128). Apple has since been re-measured at
HEAD on the shipped arm, 11.808168 s for the 1M x 28 higgs fit
(`bench/results/et_profile/APPLE_M4_2026-09-01.md`, and see DEVIATION 1945);
NVIDIA has not. The `-D MOJOLEARN_ET_TPB_256` / `_512` arms on a 32-lane
device have not been timed on either. AMD's `DEVICE_TPB` has not been swept
past 512 (1024 is the CDNA maximum).

---

## DEVIATION 1945 [OPEN] -- the MI325X phase timer bills 88% of a higgs ET fit to the host `NodeQueue.push`, and the same host loop on Apple is 0.53%

**READ THE TWO NUMBERS TOGETHER, BECAUSE NEITHER ONE STANDS ALONE.**
`NodeQueue.push` is HOST code. Nothing in it is vendor-specific and it does
not touch a device, so it cannot be 870x slower on one host than another. Both
measurements below were taken and both are recorded. What does not survive
them is the generalization that a host loop is 88% of an ExtraTrees fit.

**Where.** `impl/decisiontree/batched_levelalgo/builder.mojo`,
`NodeQueue.push` (`builder.cuh:91-134` transcribed) and the push loop in
`train_forest_*_device_timed` (`queues[seg_queue[t]].push(items_s,
splits_s)`), clocked on its own since commit 9c8ffc23 as `PHASE_HOST_PUSH`
("host: queue push (children of the batch)").

**MEASUREMENT 1, MI325X, 2026-08-29, leg `2026-08-29_204736-mojolearn-e2-amd`,
FAST.** The shipping Python surface, not a micro-benchmark:
`mojolearn.ExtraTreesClassifier(n_estimators=100, max_depth=16,
max_features="sqrt", device="gpu")` on the first 1M rows of HIGGS under
`MOJOLEARN_STAGE_TIMES=1` (`lanes/et_profile/binding_higgs_1000000.txt`).
Total 60.5 s, of which `host: queue push` **53.87 s**, range 2.83 s, score
3.31 s, everything else under 0.3 s; the untimed fit 61.8 s. At 2M
(`binding_higgs_2000000.txt`): 87.0 s, push 74.8 s. `fit_once`, the same call
without Python, shows the same 52.2-52.4 s push at 1M at BOTH TPB arms, so it
is neither the kernel change nor the driver. The cost grows faster than the
node count (1.83M nodes to 52 s, 2.17M nodes to 75 s; 29 to 34 us per pushed
node), on an EPYC 9575F on both legs. Nothing in the push body is written as
O(tree) per item, and where that time goes has not been found; it is not
guessed here.

**MEASUREMENT 2, APPLE M4, 2026-09-01, at HEAD**
(`bench/results/et_profile/APPLE_M4_2026-09-01.md`). Same fixture, same shape,
same instrument, same shipped binding arm.

| phase | Apple M4 | share |
|---|---|---|
| score pass (init+score+finalize) | 5.549429 s | 47.0% |
| range pass (init+range+decode+nonconst) | 3.313286 s | 28.1% |
| partition (4 kernels) | 1.776916 s | 15.0% |
| stage + feature sampler | 0.587036 s | 5.0% |
| candidate+reduce+splits readback | 0.275109 s | 2.3% |
| leaf pass | 0.091464 s | 0.8% |
| host: split records | 0.084643 s | 0.7% |
| setup | 0.068277 s | 0.6% |
| **host: batch assembly + queue push** | **0.062008 s** | **0.53%** |
| total | 11.808168 s | |

**WHAT THE PAIR ESTABLISHES.** 0.062008 s against 53.87 s is a factor of about
870 in the same host code path, with the whole fit 11.81 s against 60.5 s.
Since the code cannot carry that factor, the MI325X figure is measuring
something the Apple figure is not.

**THE LEADING READING.** The phase timers are host wall clocks, so on a
discrete-memory backend a region that ends in an implicit synchronization
bills the device wait to whichever phase happens to contain it.
`host: queue push` would then be a mislabeled device drain, and the fit would
have no host bottleneck at all. The competing reading is that the MI325X host
path really is pathological for this loop, for a reason nobody has found.

**THE DISCRIMINATING LEG, and it is cheaper than the bisect.** Place an
explicit synchronize BEFORE the push timer starts on the MI325X. Reading A
predicts the push figure collapses, the same milliseconds move into the
partition or score phase, and the total does not change.

**WHAT IS RETRACTED AND WHAT IS NOT.** The MI325X measurement stands. The
sentence "the host `NodeQueue.push` is 88% of a higgs ET fit" does not: it is
a property of one box's phase attribution, not of the host code, and it may
not be quoted as a whole-fit statement about ExtraTrees on any vendor.

**Leg 3, `2026-08-29_211714-mojolearn-e2-amd`, MI325X, UNRESOLVED.** The push
cost is the same again at both TPB arms (`fit_once` 1M: 52.23 s at the shipped
512, 52.35 s at 128; whole fit 58.9 s against 69.9 s, the 11 s difference
being the two GPU passes), so it is not DEVIATION 1943. The OLD-commit arm
(4f6a17a) DID NOT RUN: `tools/e2_remote_leg.sh` forwarded only
`ET_PROFILE_ARMS` over ssh, so `ET_PROFILE_OLD_COMMIT` never reached the box.
Fixed in the script. The host `perf record -g` of the shipped arm attributes
76% of samples to `core::identity_trace::IdentityTrace::_emit` with string
bytes ("partition") where return addresses should be; the trace was NOT
enabled on that arm (`MOJOLEARN_IDENTITY_TRACE` unset), so that is perf
resolving a stripped Mojo binary to its nearest exported symbol, not evidence.

**THE OPEN QUESTION, in order.** First, what is the MI325X timer attributing.
Second, and only after that, does the 18294 ms row of
`bench/results/fast_speed/2026-08-28-AMD-forest-higgs.md` (commit 4f6a17a, 1M,
five rounds within 0.4%) reproduce at 4f6a17a on the SAME box as HEAD. 18.3 s
is what the two GPU passes cost at TPB 128 plus a few hundred ms, so a 52 s
push cannot have been inside it, and
`git diff 4f6a17a..9c8ffc23 -- extratrees/impl/decisiontree/` touches only
`_device_tpb` and the clock, so the push code is identical in both runs. Until
the first question is answered, the AMD ET column of the 2026-08-28 board
carries BOTH numbers, and a whole-fit AMD ET speed claim in either direction
is not supportable from this repository's evidence.

**RUN OWED, one MI325X leg.** The env pass-through is fixed:

    E2_LANE_WAVES=et-profile \
    ET_PROFILE_ARMS=shipped \
    ET_PROFILE_OLD_COMMIT=4f6a17a \
    nohup bash tools/e2_remote_leg.sh amd <token_file> &

---

## DEVIATION 470 [OPEN] -- the search cycle's six setup enqueues fused into TWO seeder launches (Metal's 31-binding cap forbids one)

**Where.** `impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo`,
`phase_setup_a_kernel` + `phase_setup_b_kernel` (+ `PHASE_SETUP_TPB`),
enqueued A then B from BOTH twins -- `search_batch` and
`search_batch_regression` in
`impl/decisiontree/batched_levelalgo/builder.mojo` -- immediately after
`stage_batch` and before the feature sampler. The six they replace, in
their old queue order: `enqueue_memset(d_samp_report, SAMPLER_UNVISITED)`,
`node_feature_range_init_kernel`, `enqueue_memset(d_nonconst, 0)` (half A);
`node_feature_score_init_kernel`, `enqueue_memset(r_mx, 0)`,
`split_reduce_init_kernel` (half B). Every enqueue is host-priced here (the
covtype 94-launch/tree tax), and this takes six per search cycle to two.

**Why TWO kernels and not one -- a MEASURED Metal ABI wall, do not re-fuse.**
The first cut WAS one kernel: 27 pointer arguments plus 7 scalars, 34
bindings. It failed the very first gate attempt at compile time -- "Metal
Compiler failed to compile metallib / failed to run the pass manager", no
source diagnostic -- because Metal caps a kernel's buffer-binding table at
31 entries and MAX's ABI binds scalar arguments too. The split is by phase
half: A carries the report memset + range init + nonconst memset (8
pointers + 3 scalars = 11 bindings), B the score init + mutex memset +
reduce init (19 pointers + 4 scalars = 23 bindings), both comfortably under
the cap. The split point changes NOTHING about the bit-inertness argument
below: A and B are enqueued back to back at the same hoist point, on the
same in-order queue, writing the same disjoint buffers with the same
imported bodies.

**Why the hoist is bit-inert, and it is an ORDER argument, not a values one.**
All six are pure write-only seeders over DISJOINT buffers; every cell they
write is a constant or a pure function of its flat index, so fusing them
changes no value and no accumulation order (there is none). Hoisting them to
one point is then a queue-order question: on the in-order device queue, a
write may move earlier past other enqueues iff nothing between its old
position and its new one writes the same buffer, and nothing between its new
position and its FIRST READER writes it either. Checked seeder by seeder:
the sampler writes `d_colids`/`d_samp_scratch`/`d_samp_report` (its report
seed is the one being hoisted TO it, still ahead of it); the range kernels
write min/max/key/missing/merge cells (their seeds were already ahead of
them and did not move relative to them); the flag kernel writes `d_nonconst`
(its seed stays ahead); the score kernels write the score cells (seed stays
ahead); `score_to_candidate_kernel` writes `c_*` only; the reduce kernel is
the reduce seeds' first reader. No interleaved kernel writes any seeded
buffer, so every reader sees exactly the bytes it saw before.

**The two hard constraints, kept.** (1) CAPACITY extents where the memsets
had them: `d_nonconst` and `r_mx` are seeded over `cap_nodes` (the
workspace's `max_batch` capacity) and `d_samp_report` over `cap_report`
(`sampler_report_len(cap_nodes)`) -- exactly the full buffers the three
`enqueue_memset`s covered. Seeding only the live `n_nodes` leaves stale
cells a later LARGER batch reads; that is ensemble's DEVIATION 1916 lesson
and it is restated in the kernels' docstrings. (2) The fused kernels IMPORT
the seeder bodies -- `range_init_seed_at`, `score_init_seed_cell_at`,
`score_init_seed_acc_at` (extracted in `builder_kernels_impl.mojo`) and
`split_reduce_seed_at` (extracted in `split.mojo`) -- rather than
transcribing them; a memset has no body to import, so its FILL CONSTANT is
the imported thing (`SAMPLER_UNVISITED` by name). The standalone init
kernels STAY, delegating to the same helpers, because
`range_kernel_check`, `score_kernel_check`, `regression_score_check` and
`split_reduce_check` seed through them in isolation.

**Zero extents (and one skipped launch) reproduce the special cycles
exactly.** The survey (`range_only`) skips half B outright -- every B
extent would be zero, and it never ran those seeders; the rescue
(`use_sampler == False`) passes 0 for half A's report extent -- it never
memset the report. No cycle seeds more or less than it did before.

**THE REFUSAL LIST, from the scoping pass.** No seeder is fused into its
CONSUMER, and the reason is stated where the temptation will be felt:
`node_nonconstant_flag_kernel` does cross-block `Atomic.fetch_add` into
`d_nonconst`, and the range/score/reduce kernels accumulate GRID-wide into
their seeded cells. A seed folded into any consumer would need every block
to observe the seed before any block accumulates, and Metal has no
grid-wide sync. The seeders fuse with each other and with nothing else.

**Phase-clock attribution moves; the bits do not.** The six seeders now
bill to `PHASE_STAGE` instead of their old phases when
`MOJOLEARN_STAGE_TIMES` is on. The timed program was always the serialized
one; per-phase comparisons across this commit must not read the stage/range
/score/reduce split as a kernel change.

**Sabotage, wired under `-D MOJOLEARN_ET_SAB_PHASE_SETUP=1` (a measurement
arm, never a gate).** Half B's class-accumulator seed is poisoned
to the cell's BATCH-SLOT parity (`acc_left[idx] = idx & 1` instead of 0). A
node's flat cell index depends on its slot in the batch, and the merged and
serial arms batch differently, so corresponding cells receive different
poison and the two arms must DIVERGE: `device_batched_check`'s
merged-vs-serial sections (classification defaults, narrow batch,
max_leaves, groups, regression) must go RED with node diffs. The
per-kernel checks must stay GREEN under the same define because they seed
through the STANDALONE kernels -- which localizes the poison to the fused
path and proves the fused acc region is the one the fit reaches. THE
FIXTURE MUST BE SEEN TO WITNESS IT (three fixture-blind sabotages was the
lesson of Sep 1): the sabotage run is only evidence if the RED is observed
with nonzero diff counts; a green sabotage run means the poison is
fixture-blind and must be strengthened, never waved through. A
deliberately constant poison (`= 1` everywhere) would be exactly such a
blind arm -- both arms would move identically -- which is why the poison is
slot-keyed.

**Price.** Two wider launches instead of six narrow ones; the survey cycle
runs one fused launch (half A) where it ran two seeders plus a memset (its
zero report extent is dead threads, not writes). 11 + 23 kernel arguments,
each half under Metal's 31-binding table.

**Gate results, 2026-09-01 (orchestrator's box, at the A/B-split commit).**
`device_batched_check` GREEN: 45 cells, 0 node diffs, 0 leaf-bit diffs,
both live FOREST_SAB sabotages held. The FULL lane suite GREEN (all
extratrees lane checks pass). The `MOJOLEARN_ET_SAB_PHASE_SETUP` arm went
RED correctly (nodes differ, assertion at device_batched_check:269) --
the fixture witnessed it. Still OWED: the IdentityTrace pre/post fit diff
and the timing A/B on a cold box.

**Gate (the owed commands, in order -- this Mac runs nothing).**

    pixi run mojo run -I . extratrees/checks/device_batched_check.mojo
    pixi run mojo run -I . extratrees/checks/range_kernel_check.mojo
    pixi run mojo run -I . extratrees/checks/score_kernel_check.mojo
    pixi run mojo run -I . extratrees/checks/regression_score_check.mojo
    pixi run mojo run -I . extratrees/checks/split_reduce_check.mojo
    pixi run mojo run -I . extratrees/checks/sampler_kernel_check.mojo
    pixi run mojo run -I . extratrees/checks/rescue_check.mojo
    sh extratrees/tools/check.sh

the sabotage A/B (must be RED, then rebuilt clean and GREEN):

    pixi run mojo run -I . -D MOJOLEARN_ET_SAB_PHASE_SETUP=1 extratrees/checks/device_batched_check.mojo

the IdentityTrace pre/post diff (fit the same fixture at the parent commit
and at this one under `MOJOLEARN_IDENTITY_TRACE=<dir>`, then
`python3 tools/identity_trace_diff.py <pre> <post>` -- zero moved cells is
the claim), and the timing A/B at the scoreboard shapes on a cold box, owed
to the orchestrator's window.

---

## DEVIATION 471 [OPEN] -- the leaf tail's two memsets folded into `leaf_kernel[zero_fill=True]`

**Where.** `leaf_kernel` in
`impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo`
gains a comptime `zero_fill: Bool = False`; the two shipped tails in
`builder.mojo` (`train_forest_classification_device_timed`'s leaf tail and
the regression twin's) pass `zero_fill=True` and drop their
`enqueue_memset(d_leaves, 0.0)` + `enqueue_memset(d_visit, 0)` pair. Two
enqueues per GROUP become zero.

**Why it is bit-inert: block-exclusive slot ownership.** The launch is one
block per node over the WHOLE concatenated group buffer
(`grid_dim = total_nodes`, and `d_leaves`/`d_visit` are sized
`total_nodes * num_outputs` / `total_nodes` exactly), so block `node_id`
owns `out_visit[node_id]` and `out_leaves[node_id * k ..+ k]` outright --
the caller-obligation note beside the kernel documents that ownership.
Under `zero_fill` each block zeroes ITS OWN cells BEFORE the `MAX_OUT`
refusal and the `IsLeaf` early return, because the memsets it replaces
zeroed every slot regardless of either: an internal node's slots keep the
zero that IS their value. ALL the zeroing runs on `tid == 0` -- the same
thread that later publishes into the same cells -- so every cell has ONE
writer in program order and nothing rests on a barrier's device-memory
fencing; `k` is bounded by the class count, so the serial zero loop is
noise. Every cell the memsets wrote is written once, with the same value,
before any read.

**The default keeps the pre-change contract.** `zero_fill` DEFAULTS False,
so `leaf_check`, `partition_leaf_kernel_check` and every `LEAF_SAB_*` arm
run the kernel exactly as before, with their own memsets discharging the
caller obligation. Only the two shipped tails pass True.

**Sabotage, wired under `-D MOJOLEARN_ET_SAB_LEAF_ZERO_FILL=1` (a
measurement arm, never a gate).** The zero-fill's leaf fill is poisoned to
the node's CONCATENATED-slot parity (`node_id & 1`) instead of 0.0. The
merged group concatenates trees at `leaf_base[s]` offsets while the serial
arm's one-tree tails all start at 0, so internal-node slots take different
poison in the two arms and `device_batched_check`'s LEAF-BIT assertions
must go RED. WITNESS CAVEAT (Sep 1's lesson): the divergence needs at
least one tree whose `leaf_base` is odd in the merged group; the run is
only evidence if nonzero leaf-bit diffs are OBSERVED, and a green run
means fixture-blind -- strengthen (e.g. parity over `node_id % 3`), never
wave through. `partition_leaf_kernel_check` must stay GREEN under the same
define (it never passes `zero_fill=True`), which localizes the poison.

**Price.** None measurable expected: the zeroing work moves from two
driver memsets into threads the launch already runs.

**Gate results, 2026-09-01 (orchestrator's box, at the A/B-split commit).**
`device_batched_check` GREEN (45 cells, 0/0 diffs) and the full lane suite
GREEN. The `MOJOLEARN_ET_SAB_LEAF_ZERO_FILL` arm went RED correctly (leaf
bits differ, assertion at device_batched_check:270) -- the concatenated-
slot parity was witnessed. Still OWED: the IdentityTrace pre/post diff and
the timing A/B.

**Gate (the owed commands).**

    pixi run mojo run -I . extratrees/checks/device_batched_check.mojo
    pixi run mojo run -I . extratrees/checks/leaf_check.mojo
    pixi run mojo run -I . extratrees/checks/partition_leaf_kernel_check.mojo
    sh extratrees/tools/check.sh

the sabotage A/B (RED observed, then clean GREEN):

    pixi run mojo run -I . -D MOJOLEARN_ET_SAB_LEAF_ZERO_FILL=1 extratrees/checks/device_batched_check.mojo

plus the shared IdentityTrace pre/post diff and timing A/B named in 470.

---

## DEVIATION 472 [OPEN] -- `stage_batch`'s seven H2D uploads skip on byte equality

**Where.** `_stage_upload_if_changed` + the copy tail of `stage_batch` in
`impl/decisiontree/batched_levelalgo/builder.mojo`; `LevelWorkspace` gains
one shadow host buffer per staged slot (`s_items`, `s_tree`, `s_tsalt`,
`s_wl`, `s_nb`, `s_nc`, `s_blk_base`), the capacity fields
(`cap_nodes`, `cap_blocks`, `cap_report` -- the last for 470) and
`stage_valid`. `stage_batch` stages seven slots and used to copy all seven
every cycle; now each slot's bytes are compared against the bytes LAST
ENQUEUED for it and the `enqueue_copy` is skipped on equality.

**Why a skip is sound.** The seven device buffers are READ-ONLY on the
device (audited: no kernel writes `d_items`, `d_tree`, `d_tsalt`, `d_wl`,
`d_nb`, `d_nc` or `d_blk_base`; the partition's `blk_base` is a read), the
queue is in-order, and every kernel reading a slot is enqueued after the
copy that staged it -- so on byte equality the device already holds the
value and the copy is pure transfer waste. The shadow snapshot taken at
enqueue time is faithful to what the in-flight copy sends because
DEVIATION 450's invariant keeps the `h_*` staging unrewritten until after
the next required drain retires the copy (455's rescue drain and 469's
best-first drain are the two extra drains that keep this true off the
plain path).

**EXACT bytes, no bookkeeping, and the fail-safe direction (DEV-1917's
shape).** The compare is bytewise over the FULL capacity extent the copy
sends -- never a semantic summary -- so the rescue's `k` changes (survey
`k = n_cols`, rescue `k = 1`, back to the batch's `k`) restage `d_nb`/
`d_nc` automatically with no flags to forget. A spurious mismatch (a
first-cycle pinned-tail byte, struct padding) costs ONE extra copy; a skip
happens only on bytewise equality, so a changed value is NEVER skipped.
`d_nb`/`d_nc` are provably byte-constant across a group at fixed `k`
(staged as `i * k` and `k`, never reading the work items) and `d_tree` is
constant while the frontier composition is stable, so those collapse to
one copy per group BY the compare, not by special-casing. The no-retry
restage skip at the level loop's retry check (DEVIATION 455's
`elif len(retry) > 0` -- 1919's shape) is a different mechanism, already
present, and NOT duplicated here.

**Sabotage, wired under `-D MOJOLEARN_ET_SAB_STAGE_SKIP_ALWAYS=1` (a
measurement arm, never a gate) -- REDESIGNED 2026-09-01 after the first
form HUNG, and the finding is a design rule.** A REQUIRED-RED ARM MUST
PROVABLY TERMINATE: an arm that hangs is not a gate, it is an outage. The
first form skipped every re-upload after the first for ALL SEVEN slots,
and two of those slots are the level loop's CONTROL state: with `d_items`/
`d_wl` frozen, a later cycle's plan can have MORE workload blocks than the
frozen prefix, at which point the kernels read garbage `WorkloadInfo`
entries (the first upload sends the pinned buffer's uninitialized tail)
whose `node_id`s index `d_items` OUT OF BOUNDS -- undefined device
behavior. The 2026-09-01 gate run compiled, entered section 1, and ground
past 10 minutes with no output (a clean run takes ~3); two attempts, same
hang. Frozen `d_blk_base` carries the same addressing hazard (stale bases
plus live block counts can write `blk_off` out of bounds), so it is
excluded too.

The redesigned arm freezes ONLY the pure per-node PAYLOAD slots --
`d_tree`, `d_tsalt`, `d_nb`, `d_nc` (a `payload_slot` flag on
`_stage_upload_if_changed`, read only under the define) -- while
`d_items`, `d_wl` and `d_blk_base` stay LIVE. Termination: every device
loop bound and every address derives from the live control slots, so the
arm terminates by the same argument as the clean run; frozen `d_nb`/
`d_nc` reads are in-bounds by construction (`i * k_old + k_old <= nodes *
k_cap`). Witness: the frozen tree ids and tie salts still give every tree
after the batch-first one the WRONG draw keys in the merged arm while the
serial arm re-stages per tree, so `device_batched_check`'s
merged-vs-serial sections must go RED -- the check's
`trees_mutually_differ >= 2` fixture guard exists precisely so a frozen
tree id cannot hide. The RED must still be OBSERVED with nonzero diffs
before the clean GREEN is trusted (Sep 1); this redesigned arm has NOT
yet been run.

**Price.** A host-side bytewise compare over the seven capacity extents
per `stage_batch` call, traded against seven `enqueue_copy` calls plus
their transfers. Whether the trade wins is the owed timing A/B's verdict,
not this entry's; if it loses at the scoreboard shapes, 472 reverts.

**Gate results, 2026-09-01 (orchestrator's box, at the A/B-split commit).**
`device_batched_check` GREEN (45 cells, 0/0 diffs) and the full lane suite
GREEN -- the shipped compare-and-skip is exercised by both. The
skip-always sabotage arm HUNG in its first form (the finding above) and
its redesigned form is UNRUN. Still OWED: the redesigned sabotage A/B, the
IdentityTrace pre/post diff, and the timing A/B.

**Gate (the owed commands).**

    pixi run mojo run -I . extratrees/checks/device_batched_check.mojo
    pixi run mojo run -I . extratrees/checks/rescue_check.mojo
    pixi run mojo run -I . extratrees/checks/bestfirst_check.mojo
    sh extratrees/tools/check.sh

the sabotage A/B (RED observed, then clean GREEN):

    pixi run mojo run -I . -D MOJOLEARN_ET_SAB_STAGE_SKIP_ALWAYS=1 extratrees/checks/device_batched_check.mojo

plus the shared IdentityTrace pre/post diff and timing A/B named in 470.
