# catboost-symmetric-trees

**A port of CatBoost's GPU oblivious (symmetric) tree learner into
Mojo. GPU path only. No CPU path. Nothing from mojotrees.**

## Why this exists

mojotrees is 2.5x behind LightGBM and 10x behind CatBoost per symmetric tree,
and a full day of measurement established that the gap is not the inner loop:
our histogram kernel is about 2x FASTER per update than CatBoost's.

That paragraph used to end with a claim about launch count, that this port
issues "73 command buffers and 1 host sync per tree". **It is false and it is
deleted rather than annotated**, per the standing rule that a document a
result falsifies is part of the result. `RESUME.md` measured
`run_tree_layout` at 9 `ctx.synchronize()` and about 16 launches PER LEVEL,
which is 54 host round trips and about 96 launches per depth-6 tree, and
about 34 of our 50 ms per tree is fixed cost independent of the data. The
deficit is the CONTROL PLANE, and `HOST_AND_DEVICE.md` now carries the rule
that decides what may be done about it.

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
`ported/methods/.../hist_binary.mojo` beside their `hist_binary.cu` and diff
them, and so "did we port this file?" is answered by `ls` rather than by
reading. Anything with no CatBoost counterpart lives under `mojo_only/` and
has to justify its existence there.

| stage | CatBoost source | port |
|---|---|---|
| feature grouping and bit packing | `gpu_data/grid_policy.h` | `ported/gpu_data/grid_policy.mojo` |
| leaf as a contiguous range | `cuda_util/gpu_data/partitions.h` | `ported/cuda_util/gpu_data/partitions.mojo` |
| histogram, binary (32 features/ui32) | `methods/greedy_subsets_searcher/kernel/hist_binary.cu` | `ported/methods/greedy_subsets_searcher/kernel/hist_binary.mojo` |
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

The design is a four-column table: **bit-identical**, apple, nvidia, amd.
The bit-identical column is not a mode flag, it is a real column holding the
value every vendor can meet, so it is the INTERSECTION of the other three and
can be printed and diffed against a device column to show exactly what
identity costs there.

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

`cluster/` is a port of **cuVS** k-means at commit `2140532c`, built the same
way and under the same rule. It is the first algorithm here with no histogram
in it, and per `PLAN.md` that is why it was built first: it is the smallest
thing that can answer whether the shared substrate is actually shared or is
quietly tree-shaped.

**The first answer is in and it is a good one.** The fixed-point accumulator
in `mojo_only/fixed_point.mojo`, written for the histogram flush because
Metal has no float atomic add, serves the k-means centroid update unchanged.
Its overflow argument transferred with one noun changed: "any leaf's rows are
a subset of all rows" became "any cluster's rows are a subset of all rows",
and nothing else moved. That also gives the file its first reader; it had
none.

k-means moved out of RAFT and into cuVS, so the mirror is two-layer:
algorithms from cuVS into `cluster/ported/`, and the RAFT and cuBLAS
primitives they call into `cluster/mojo_only/`. See `cluster/README.md`.

**Nothing in `cluster/` has been launched yet.** See `UNWIRED.md`.

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

**Two of this port's deviations are currently justified by reasoning that has
never executed on another vendor**, which is what the script exists to fix:
`replication_lanes` is pinned at 32 so AMD's 64-wide wavefront cannot change
the reduction geometry, and the float-atomic flush branch is unreachable on
Apple and is exactly what NVIDIA and AMD would take under `determinism=off`.

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

**Everything under `ported/` is a derivative work of CatBoost**, Copyright
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
