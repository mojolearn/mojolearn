# PairLogit, the pairwise ranking loss (lane/gbdt-learning-to-rank, stage 3)

`GradientBoosting(loss="PairLogit").fit(X, y, group_id=..., pairs=None, pairs_weight=None)`
on the GPU through `gbdt/targets/kernel/pair_logit.mojo` (the CatBoost reference's
`PairLogitPointwiseTargetImpl` and `MakePairWeightsImpl`, `pair_logit.cu`, through the
querywise target's `InitPairLogit` arm), the pairs from `gbdt/data/pairs.mojo` (the reference's
default `GenerateBruteForce` path, or explicit `[winner, loser]` pairs, in the order its device
grouping flattens them), and the CPU reference column restating the same arithmetic in
`gbdt/host/gbdt_oracle_pair.mojo`.

Two named DEVIATIONs, both in `pair_logit.mojo`'s docstring:

1. Each row's pair derivatives, and the per-row pair weights, are summed sequentially in
   increasing pair index. The reference sums them with `atomicAdd` over the pair threads, in
   arrival order.
2. The search weight plane follows `secondDerAsWeights` as the pointwise target does (Cosine
   and L2 read the per-row pair weights; NewtonL2 and NewtonCosine read the der2 sums). The
   reference's querywise `StochasticDer` has the two arms reversed
   (`querywise_targets_impl.h:161-181` against `pointwise_target_impl.h:173-216`, the flag
   forwarded unchanged through `weak_objective_impl.h:21-45` and `target_func.h:346-356`).

## Machine and builds

Apple M4, one core, shared machine, every local run through a shared Mac slot, identical tier,
`mojo build -j 1`.

| binding | sha256 |
|---|---|
| `_mojolearn_gbdt.so` (Metal) | 4ac24c5987b5d667 |
| `_mojolearn_gbdt_host.so` (CPU column) | 0dac7114a59fd553 |
| `_mojolearn_gbdt_host.so`, `-D MOJOLEARN_HOST_SABOTAGE=1` | b5b6feb255c9d3bf |

## The lane

`gbdt-pair-logit`: the gbdt-query-rmse queries (sizes 1, 2, 7, 3, 16, 1, 5, 40, 2, 9 cycled) and
grades (0 to 4, ties). A 20-tree depth-6 fit on the generated pairs, and an 8-tree depth-4 fit
on explicit pairs from the first 40 queries with hashed `pairs_weight`, both predictions and both
loss curves hashed; the held-out probe and the batch part are the first fit's.

## Verdicts, fixtures base, ties and odd

| check | result | file |
|---|---|---|
| Metal vs CPU, two repeats each (`--require-columns 2`) | IDENTICAL=3; (infer/model): IDENTICAL=6; (batch): IDENTICAL=3 | `diff-metal-cpu.txt` |
| host sabotage build vs Metal | DIVERGENT=3; (infer/model): DIVERGENT=6; (batch): DIVERGENT=3 | `diff-metal-cpu-sabotage.txt` |
| `MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1`, Metal, base | batch_moved=1, exit 1 | `apple-m4.batch-sabotage.json` |
| the 17 earlier gbdt lanes, Metal, against the committed stage 2 confirmation columns | IDENTICAL=51; (infer/model): IDENTICAL=96, N/A=6; (batch): IDENTICAL=51 | `diff-earlier-17-metal.txt` |
| the 17 earlier gbdt lanes, CPU vs Metal | IDENTICAL=51; (infer/model): IDENTICAL=96, N/A=6; (batch): IDENTICAL=51 | `diff-earlier-17-metal-cpu.txt` |

## Tests

`test_gbdt_pair_logit`, `test_gbdt_query_rmse`, `test_gbdt_group_id` and `test_host_surface` on
the Metal route: 148 passed, 1 warning in 1.51s. The first three on the CPU route:
25 passed in 1.46s.

## The CatBoost CPU reference (correctness, not bits)

CatBoost 1.2.10 on its CPU, one thread, every option equal to our 20-tree generated-pairs fit
(`catboost_pair_logit_reference.py`), against our Metal fit (`ours_pair_logit_reference.py`);
`compare_pair_logit_reference.py` prints the differences in `compare-catboost.txt`, and nothing
was tuned to match. Our PairLogit learn loss is their PairLogit metric.

| fixture | learn loss, largest relative diff over 20 iterations | final NDCG diff | final DCG diff | first 8 raw predictions, largest diff |
|---|---|---|---|---|
| base | 6.99e-05 | -1.0e-15 | +7.1e-15 | 4.64e-02 |
| odd | 6.06e-05 | +7.8e-16 | -1.3e-14 | 4.63e-02 |
| ties | 3.21e-06 | -2.2e-16 | -5.3e-14 | 5.63e-01 |

The learn loss agrees to about 1e-4 relative and the rankings the NDCG and DCG read agree; the
raw predictions differ more than QueryRMSE's did. Not measured here: which of float32 against
float64 over ten Newton steps, the reference CPU's leaf estimation, or the deviations above
produces that difference.

Not run here: NVIDIA and AMD columns (owed to the next release record), and the fixtures other
than base, ties and odd.
