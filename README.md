# catboost-symmetric-trees

**A clean-room port of CatBoost's GPU oblivious (symmetric) tree learner into
Mojo. GPU path only. No CPU path. Nothing from mojotrees.**

## Why this exists

mojotrees is 2.5x behind LightGBM and 10x behind CatBoost per symmetric tree,
and a full day of measurement established that the gap is not the inner loop
(our histogram kernel is about 2x FASTER per update than CatBoost's) and not
launch count (our oblivious path issues 73 command buffers and 1 host sync
per tree against CatBoost's ~400-900 launches and ~24 syncs).

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

| stage | CatBoost source | port |
|---|---|---|
| feature grouping and bit packing | `gpu_data/grid_policy.h` | `src/grid_policy.mojo` |
| leaf as a contiguous range | `cuda_util/gpu_data/partitions.h` | `src/partitions.mojo` |
| histogram, binary (32 features/ui32) | `methods/greedy_subsets_searcher/kernel/hist_binary.cu` | `src/hist_binary.mojo` |
| histogram, half-byte (8/ui32) | `.../hist_half_byte.cu`, `point_hist_half_byte_template.cuh` | `src/hist_half_byte.mojo` |
| histogram, one-byte (4/ui32) | `.../hist_one_byte.cu`, `compute_hist_loop_two_stats.cuh` | `src/hist_one_byte.mojo` |
| bin prefix scan, sibling subtraction | `.../histogram_utils.cu` | `src/histogram_utils.mojo` |
| score and argmax | `.../compute_scores.cu` | `src/compute_scores.mojo` |
| in-leaf reorder after a split | `.../split_points.cu` | `src/split_points.mojo` |
| the level loop | `methods/greedy_subsets_searcher/structure_searcher_template.h`, `split_properties_helper.cpp` | `src/searcher.mojo` |

## What is deliberately NOT here

No CPU fallback. No binning: this consumes an already-quantized matrix. No
categorical features, no CTRs, no ordered boosting, no ranking. No tests
until the thing runs end to end.
