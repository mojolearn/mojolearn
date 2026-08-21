# mojolearn

**GPU machine-learning algorithms in Mojo, running on hardware the originals
cannot reach.**

Every line here is written in this repository. What it mirrors is the DESIGN
of the implementations that already got these algorithms right, and the rule
for doing that is in `PORTING_RULES.md`: follow their algorithm, keep their
control plane and their data plane, keep what they do on the GPU on the GPU
and what they do on the host on the host, and adapt to Mojo only where the
toolchain forces it. The mirroring is deliberate. A port that drifts cannot be
checked against the original, and this one is checked against it constantly.

The point is the target. Every design below was written for CUDA, and none of
those implementations runs on Apple silicon at all. This one does, from a
single source that also targets CUDA and ROCm.

Designs mirrored, table corrected 2026-08-20. It said "two upstreams so far"
and listed two. There are five, and the omission was not cosmetic: cuML had no
attribution section in `NOTICE` at all, which is an Apache-2.0 section 4
obligation, and the MIT-licensed FAISS code RAFT vendors had no notice
anywhere. Both are fixed; see `NOTICE`.

| directory | upstream | what |
|---|---|---|
| `gbdt/`, `mojo_only/` | CatBoost | the GPU oblivious (symmetric) tree learner, control plane included |
| `cluster/` | cuVS | k-means |
| `neighbors/` | cuVS, RAFT, FAISS (MIT) | brute-force k-NN, the fused L2 kernel, ball cover, top-k selection |
| `dbscan/` | cuML, RAFT | DBSCAN, epsilon-neighborhood, label merging |
| `decomposition/` | cuML, RAFT | PCA and truncated SVD |
| `glm/` | RAFT | ordinary least squares (`lstsqEig`, cuML's `olsFit` algo=1) |

Every derivation is recorded per file in the `PORTED_MAP.tsv` beside each
section, with a status of transliterated, partial, replaced, or substitute,
and an `UNPORTED.tsv` naming what was deliberately left out.

Renamed from `catboost-symmetric-trees` on 2026-08-19, because it stopped
being one port.

## The CatBoost port

**GPU path only. No CPU path. Nothing from mojotrees.**

## Why this exists

mojotrees is 2.5x behind LightGBM and 10x behind CatBoost per symmetric tree,
and a full day of measurement established that the gap is not the inner loop:
our histogram kernel is about 2x FASTER per update than CatBoost's.

That paragraph used to end with a claim about launch count, that this port
issues "73 command buffers and 1 host sync per tree". **It is false and it is
deleted rather than annotated**, per the standing rule that a document a
result falsifies is part of the result.

This section then said, for a day, that the deficit was the CONTROL PLANE.
**That is falsified too, by measurement, and is deleted under the same
rule.** On 2026-08-19 the counters read 77 launches and 12 drains per
depth-6 tree at 800k x 100 (12 is exactly CatBoost's own two-per-level
discipline), and pricing them on this Metal device (one launch+drain
191 us, one undrained launch 23 us) puts ALL dispatch at 3.8 ms of the
measured 41.7 ms fixed cost per tree. **Nine percent.** The other ~38 ms is
KERNEL TIME on work whose size does not depend on the rows: the depth-6
histogram footprint is 64 leaves x 100 features x 255 bins x 2 stats =
3.26M cells, zeroed, flushed, scanned, subtracted and scored every level.
The deficit is row-independent KERNEL WORK, and the per-row work is also
2.1x CatBoost CPU's (103 vs 48.6 us per 1000 rows), so no dataset size
closes the gap by itself. A per-kernel itemization taken the same day put
the histogram ACCUMULATE at ~68 ms of the 124 at 800k, found the score
kernel launched at grid (1,1,1) where CatBoost launches
`min(ceil(binFeatureCount/256), 64)` blocks (`greedy_search_helper.cpp:439`),
and fixing that one dispatch took the curve to {100k: 42.4, 200k: 48.2,
400k: 63.7, 800k: 107.1} ms/tree, splits still 48/48 against the oracle. A
90 GB/s streaming probe on the same buffers cleared the substrate: MAX's
Metal path delivers bandwidth, and what remains is kernel geometry. `HOST_AND_DEVICE.md` still carries the rule for
what may be done about the control plane; the control plane is just no
longer where the time is.

What that day also established is that mojotrees' own code and its own
instruments cannot be trusted to say why. Four features were found built,
tested, documented and unreachable. A static attribution predicted a 41
percent win and measured minus 53. A traffic model predicted 20x and measured
2.9x. The phase profiler answered `no span is device time`.

So the experiment is to stop reasoning about our implementation and
**transliterate theirs**, in a tree where our architecture cannot leak in.

## The one rule

**COPY. DO NOT IMPROVE.**

Port what CatBoost does, in the order and shape it does it, including the
parts that look wrong. Every deviation is a confound: if this ends up slow we
must be able to say it is their design that is slow here and not our
interpretation of it. Where Mojo cannot express something (see PORTING.md),
write the closest thing and MARK IT, rather than substituting a better idea.

Specifically forbidden in this tree:
- reading mojotrees for guidance on what a stage should do
- "while I am here" restructuring
- our per-leaf histogram model, our row-list model, our three-plane cell
- optimizing before the whole thing runs

## Scope

CatBoost commit `54a8143a`, `catboost/cuda/`. Symmetric (oblivious) growth,
GPU, pointwise objectives.

**The tree mirrors CatBoost's tree, file for file, with their constant `catboost/cuda/` prefix dropped**, so a reviewer can put
`gbdt/methods/.../hist_binary.mojo` beside their `hist_binary.cu` and diff
them, and so "did we port this file?" is answered by `ls` rather than by
reading. Anything with no CatBoost counterpart lives under `mojo_only/` and
has to justify its existence there.

| stage | CatBoost source | port |
|---|---|---|
| feature grouping and bit packing | `gpu_data/grid_policy.h` | `gbdt/gpu_data/grid_policy.mojo` |
| leaf as a contiguous range | `cuda_util/gpu_data/partitions.h` | `gbdt/cuda_util/gpu_data/partitions.mojo` |
| histogram, binary (32 features/ui32) | `methods/greedy_subsets_searcher/kernel/hist_binary.cu` | `gbdt/methods/greedy_subsets_searcher/kernel/hist_binary.mojo` |
| histogram, half-byte (8/ui32) | `.../hist_half_byte.cu`, `point_hist_half_byte_template.cuh` | `.../kernel/hist_half_byte.mojo` |
| histogram, one-byte (4/ui32) | `.../hist_one_byte.cu`, `compute_hist_loop_two_stats.cuh` | `.../kernel/hist_one_byte.mojo` |
| bin prefix scan, sibling subtraction | `.../histogram_utils.cu` | `.../kernel/histogram_utils.mojo` |
| score and argmax | `.../compute_scores.cu` | `.../kernel/compute_scores.mojo` |
| in-leaf reorder after a split | `.../split_points.cu` | `.../kernel/split_points.mojo` |
| the level loop | `methods/greedy_subsets_searcher/structure_searcher_template.h`, `split_properties_helper.cpp` | `.../greedy_subsets_searcher/structure_searcher_template.mojo` |

## The one thing here that is NOT a port: `mojo_only/numerics.mojo`

CatBoost ships one GPU backend and accepts a non-deterministic answer: its
histogram flushes through `atomicAdd` on `float`, so two runs of the same fit
on the same device can differ in the last bits. We ship Metal, CUDA and HIP
from one source, so we need "the same fit gives the same model" to be
available.

The design is a table: **bit-identical**, apple, nvidia, amd (CDNA),
amd-rdna, and -- declared 2026-08-21, with no target to build for yet --
qualcomm and intel. The columns are **the GPUs people train on**, which is
also why AMD is two of them: CDNA runs wave64 and RDNA wave32, and one column
resolved a 512-thread block for parts whose lane group is 32.
The bit-identical column is not a mode flag, it is a real column holding the
value every ADMITTED vendor can meet, and it can be printed and diffed
against a device column to show exactly what identity costs there.

**It is a FROZEN, VERSIONED FLOOR and no longer the intersection of whoever
is in the table.** As an intersection it was a tripwire: adding one vendor
with a smaller shared-memory budget would have shrunk the safe column, which
changes the block size, the replication factor, and therefore which partial
sums combine -- so every model ever produced under `IDENTICAL` would have
stopped matching the ones produced afterwards, with no error and no version
to notice it by. A new vendor now either meets the floor and joins with no
bit moving, or is refused for `IDENTICAL` by name and runs `FAST`. The floor
never drops to fit one; widening it is a profile bump, which is a different
guarantee about a different set of models.

Today every declared vendor meets it, including the three nothing can build
for: Adreno advertises the same 32 KB per workgroup that Metal does, and
Intel more. **The lowest common denominator has not moved: Apple was the
binding constraint and still is.** An eighth column, `spec-baseline`, holds
what the Vulkan and WebGPU specifications GUARANTEE rather than what a vendor
ships (16 KB, 128 invocations) -- it is refused, permanently and on purpose,
because it is half our floor's memory. It is there to give the admission gate
a member it must reject, since a guard whose refusal branch has never
executed is untested rather than working. `mojo build -I . matrix_main.mojo` prints the whole table
with each vendor's minimums and its admission verdict, and touches no device.

Resolution substitutes per ROW, not per spec: a scheduling row always comes
from the device's column, and a numeric row comes from the device's column
under the default `FAST` mode or from bit-identical under `IDENTICAL`.

**`FAST` is the default.** Full per-vendor speed, and because histograms
flush through float atomics the last bits move between two runs of the same
fit on the same device, not only across vendors. That is CatBoost's shipped
behavior. `IDENTICAL` is the opt-in for reproducibility.

Each row is classified by ONE question. **Does it change the sequence or the precision of the arithmetic?**
Rows that do are NUMERIC and `IDENTICAL` pins them to a safe column; rows
that do not are SCHEDULING and every backend picks freely in both modes. So
geometry runs at full per-backend speed always, and identity costs only the
named subset.

**The classification is not "numerics versus scheduling", and that
distinction is the whole point.** A block count is a summation order. A
replication factor is a summation order. Both look like scheduling and both
are numeric. Getting this axis wrong is not hypothetical: mojotrees shipped a
wrong bit-identity claim on exactly that mistake, because
`AccumulationPlan.row_blocks` reads as a schedule and moves bits.

Two things escape the table and are handled in source rather than
configured: floating-point atomics are order-nondeterministic run to run, so
`IDENTICAL` REPLACES the accumulator rather than configuring it; and FMA
contraction is a codegen decision a runtime row cannot reach.

## A second section, a second upstream: `cluster/`

`cluster/` is a port of **cuVS** k-means at commit `94c2819` (branch-25.08, cloned 2026-08-19). The commit `2140532c` that this
file and every `cluster/` header used to name is NOT A VALID OBJECT in the
cuVS repository; it was never verifiable and is corrected rather than kept. Built the same
way and under the same rule. It is the first algorithm here with no histogram
in it, and per `PLAN.md` that is why it was built first: it is the smallest
thing that can answer whether the shared substrate is actually shared or is
quietly tree-shaped.

**The first answer is in and it is a good one.** The fixed-point accumulator
in `mojo_only/fixed_point.mojo`, written for the histogram flush, serves the
k-means centroid update unchanged.
Its overflow argument transferred with one noun changed: "any leaf's rows are
a subset of all rows" became "any cluster's rows are a subset of all rows",
and nothing else moved. That also gives the file its first reader; it had
none.

k-means moved out of RAFT and into cuVS, so the mirror is two-layer:
algorithms from cuVS into `cluster/gbdt/`, and the RAFT and cuBLAS
primitives they call into `cluster/mojo_only/`. See `cluster/README.md`.

`cluster/` is LAUNCHED and passing: 4 of 4 centroids recovered as a
permutation, 0 of 512 rows misassigned, inertia within 0.3% of the
analytically known value, and reach proved by two sabotages that predict
different movements. The k-means++ init path is still unreached. See
`cluster/README.md`.

## What is deliberately NOT here

No CPU fallback. No binning: this consumes an already-quantized matrix. No
categorical features, no CTRs, no ordered boosting, no ranking. No tests
until the thing runs end to end.

## Testing the columns we do not have

`tools/remote_gpu.sh <user@host> [nvidia|amd]` syncs this tree to a rented
GPU, rewrites `TARGET_COLUMN`, builds, and runs the correctness checks.

`TARGET_COLUMN` is a **comptime** constant, so a column is a BUILD and not a
flag. That is the honest shape of the constraint rather than a limitation of
the script: a threadgroup allocation size is fixed at compile time and cannot
follow a runtime device query.

It runs correctness only, never timing. A rented box is shared, throttled and
unknown, and this repository's rule is that only interleaved arms inside one
process compare. Correctness checks are verdicts and do not care about the
machine's mood.

**One of this port's deviations is currently justified by reasoning that has
never executed on another vendor**, which is what the script exists to fix:
`replication_lanes` is pinned at 32 so AMD's 64-wide wavefront cannot change
the reduction geometry.

This paragraph used to name a second one, that "the float-atomic flush
branch is unreachable on Apple". **That is false and it is deleted rather
than annotated**: the histogram kernels build at `BUILD_MODE = NUMERIC_FAST`
and `deterministic_flush_for` returns `identical`, so the FAST build runs
CatBoost's verbatim `atomicAdd` on `float` on Apple too (float atomics work
on Metal; the old "absent" reading was a wrong import path). The
fixed-point branch is what `IDENTICAL` builds take, on every vendor.

## License and attribution

Created and maintained by **Andrew Hendel**
([@ajhendel](https://github.com/ajhendel)). [AUTHORS.md](AUTHORS.md) records
who holds what, and is worth reading here specifically because most of the
*algorithm* is CatBoost's and most of the *translation* is not.

If this is useful in work you publish, we would appreciate a citation.
[CITATION.cff](CITATION.cff) is what GitHub's **Cite this repository** button
reads.

Apache-2.0, in [LICENSE](LICENSE). [NOTICE](NOTICE) records what this derives
from and must travel with any redistribution.

**Everything under `gbdt/` is a derivative work of CatBoost**, Copyright
2017-2026 YANDEX LLC, Apache-2.0, translated from CUDA C++ into Mojo at commit
`54a8143a`. `PORTED_MAP.tsv` maps each file to its origin.

**This is not a clean-room reimplementation and must not be described as one.**
Clean-room means reproducing behavior from a specification without reading the
source. This project does the opposite deliberately: `README.md`'s own rule is
COPY, DO NOT IMPROVE, and the tree mirrors CatBoost's paths file for file so a
reviewer can diff `hist_binary.mojo` against `hist_binary.cu`. That is a port,
which Apache-2.0 permits, and it carries attribution obligations that a
clean-room reimplementation would not.

Not affiliated with, endorsed by, or sponsored by YANDEX LLC or Modular, Inc.

MAX (R) and Mojo (R) are trademarks of Modular, Inc. used under license.
