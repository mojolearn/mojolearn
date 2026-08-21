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

---

## 140. RAFT's wide 64-bit multiply has no PTX fast path here

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

## 141. Only PCGenerator's three-argument constructor is ported, and there is no `half`

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

## 142. Thresholds are Float32, and the rescale is not allowed to fuse

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

**So unfused is a CHOICE, not the only thing that works.** It is chosen because
the fused form's agreement is an accident of two compilers making the same
decision on one target, and would have to be re-established on every backend;
the unfused form is determined by the source. FMA versus mul-then-add differ by
one rounding of the product, and on a threshold compared with `<=` against
feature values, one rounding decides which side a row falls on when the
threshold lands exactly on a value — which, for a threshold drawn between the
observed min and max, is not rare.

**Price.** One extra rounding against a `float64` sklearn, one un-inlinable
call in the threshold path, and a build flag in `tools/rng_oracle/build.sh`
that must not be dropped.

---

## 146. `TreeMetaDataNode` is reduced: `train_time` is not ported

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

## 147. `predict_one` accumulates and never zeroes; we mirror that and add a zeroing entry point beside it

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

## 148. The node's template shape is narrowed: `LabelT` dropped, `IdxT` fixed to Int32

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

## 149. Fixtures are counter-based, not stream-based

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

## 150. No fixture contains a missing value

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
2. **That test's addition is FLOAT32, and it widens the band.** Both operands
   are `float32_t`. At `min == 0.5`, `0.5f + 1e-7f` rounds up to
   `0.5 + 1.192e-7`, so a column whose spread is `1.19e-7` — comfortably above
   `1e-7` — is still reported CONSTANT. **A port that does this arithmetic in
   float64 will disagree with sklearn on real data.** Every near-constant
   fixture in this lane is anchored at `min == 0.0`, where the addition is
   exact, so that the fixture tests the rule and not the rounding.
3. **`rand_uniform` can return `high`.** `_utils.pyx:57-61` is
   `(high-low) * our_rand_r() / RAND_R_MAX + low` and `our_rand_r` can return
   exactly `RAND_R_MAX`. sklearn guards it at `_splitter.pyx:651-652`:
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

**A struct field cannot expose `MutAnyOrigin`** ("struct fields cannot expose
AnyOrigin in their type"). Use `MutUntrackedOrigin` for non-owning views whose
lifetime is managed explicitly. `UnsafePointer` is deprecated in favour of
`Pointer`; the repo's `MutPointer[T, origin]` spelling is the one that compiles
without a warning.

**A field cannot be moved out of a live struct** ("destroyed out of the middle
of a value"), so an accessor that hands back an owned member returns `.copy()`.
