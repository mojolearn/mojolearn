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
(`_splitter.pyx:693`) — strictly greater, so on a tie the FIRST candidate in
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

## 135. RULED — regression accumulates in FIXED POINT

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
`mojo_only/fixed_point.mojo` already solves it for the GBDT learner, where
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

**Implementation:** `mojo_only/fixed_point.mojo`, checked by
`mojo_only/fixed_point_check.mojo` (8280 cells, seven sabotages).

**Consequence for `AggregateBin`.** `objectives.mojo` deliberately left its
accumulator type a parameter pending this ruling. The device instantiates it
over an integer type; the host oracle may keep `Float64`, and where the two are
compared the comparison is through the quantized values, not the floats.

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

## 143. Histogram-free: the accumulators ARE the arguments

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

## 144. Classification selects on an EXACT INTEGER proxy

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
`sample_weight` is now listed in `UNPORTED.tsv`.

---

## 145. The exact comparator is the authority, not `Split.best_metric_val`

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

## 151. The supplied column list IS the search; sklearn's "keep drawing until one is non-constant" is gone

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

## 152. The candidate scan COUNTS; only the winner is partitioned

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

## 153. Regression selects on sklearn's MSE proxy and REPORTS cuML's gain

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

## 154. Two of sklearn's four rejection branches are absent because their inputs do not exist

**Theirs.** Four rejections, in order: `min_samples_leaf` (`:664-666`),
`min_weight_leaf` (`:674-677`), monotonicity (`:679-689`), and the `>` at
`:693`.

**Ours.** The first and the last only.

- `min_weight_leaf` is unreachable in the only configuration this port
  supports: `sample_weight=None` makes `weighted_n_left == n_left`, and
  sklearn's own `min_weight_fraction_leaf=0.0` makes the threshold zero.
  `sample_weight` is already in `UNPORTED.tsv`.
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

## 156. The `k == n` guard moved into the sampler, and materializes an identity

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

## 157. The block collectives are explicit loops, and the block width is a parameter

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

## 158. `log`, `exp` and `ceil` come from libm through FFI, not `std.math`

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

## 159. Index widths, and a bound on their unbounded loop

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

## 160. NOT a deviation: the range pass is bit-checkable and the histogram pass is not

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

## 161. The cross-block combine is a MUTEX MERGE, not their atomicAdd + signalDone

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

## 162. The output is a STRUCT OF ARRAYS, and `Dataset` is passed as its components

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

## 163. The empty range is carried IN the output as `min > max`, and NaN never reaches the reduction

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

## 164. FIXED cuML BUG: the block minimum had no head flag, so column 0 was never sampled

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

## 165. FIXED cuML BUG: a slot that never drew voted anyway, as column n-1

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

## 166. The reduction carries the EXACT RATIONAL KEY beside the `Split`, and `Split.update` becomes its tie-break

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

## 167. The Int128 cross-multiply is HAND-WIDENED from 64-bit limbs, because Int128 in a device kernel does not compile

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

## 168. `warpReduce` is a `shuffle_xor` butterfly with NO ASSUMED WARP WIDTH

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

## 169. `evalBestSplit`'s publish: a portable mutex, a struct-of-arrays cell, and the device's own report of which path it ran

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

## 170. The score pass is TWO launches, because no block can be elected last

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

## 171. NOT a deviation: this merge IS their `atomicAdd`

Deviation 161 replaced their cross-block combine with a mutex because a RANGE
has no portable float atomic. This merge is an INTEGER ADD, which does have
one, so `:301` is transcribed directly: one `Atomic.fetch_add` per accumulator
per block, from thread 0. Order-independence is exact, which is what licenses
the bit-exact check — the same argument as 160, for a different operation.

Worth stating because the two kernels sitting next to each other with different
combine strategies looks like an inconsistency and is not: it is the difference
between an operation that has a portable atomic and one that does not.

---

## 172. No dynamic shared memory: private accumulators plus a block reduction

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

## 173. The draw is an explicit `fma`, and it is why deviation 142 was amended

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

## 174. The device sees only INTEGERS, and a refusal is a STATUS

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

## 175. The published width is Int64, and it is TIGHTER than the host's Int128 bound

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

**MSE publishes no rational**: its numerator needs `Int128` by deviation 135's
derivation, so the four accumulators ARE the score and `mse_proxy_exact` forms
it on the host in one multiply each.
