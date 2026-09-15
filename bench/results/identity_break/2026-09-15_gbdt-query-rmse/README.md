# QueryRMSE, the first ranking loss (lane/gbdt-learning-to-rank, stage 2)

`GradientBoosting(loss="QueryRMSE").fit(X, y, group_id=...)` on the GPU through
`gbdt/targets/kernel/query_rmse.mojo` and `gbdt/gpu_data/kernel/query_helper.mojo`
(the CatBoost reference's `query_rmse.cu`, `TQueryRmseKernel::Run`, and the group means
and ids of `query_helper.cu`), with leaf estimation reading the point through the inverse
bin order (`permutation_der_calcer.h:171-247`). The CPU reference column restates the same
arithmetic in `gbdt/host/gbdt_oracle_query.mojo`.

## Machine and builds

Apple M4, one core, shared machine, `MOJOLEARN_NUMERIC_MODE=identical`, `mojo build -j 1`.

| binding | sha256 |
|---|---|
| `_mojolearn_gbdt.so` (Metal) | 93b9a2463e11219d |
| `_mojolearn_gbdt_host.so` (CPU column) | c70c7878cebcca3d |
| `_mojolearn_gbdt_host.so`, `-D MOJOLEARN_HOST_SABOTAGE=1` | 78a7b56b301e6a39 |

## The lane

`gbdt-query-rmse` in tools/identity_break.py: 20 depth-6 symmetric trees. Query sizes cycle
through 1, 2, 7, 3, 16, 1, 5, 40, 2 and 9 rows (the last query is truncated at the fixture's
row count). Relevance is graded 0 to 4 from the fixture's regression target, cut by
comparisons only, so every box reads the same grades. On `base` that is 20,000 rows in 2,328
queries, with grade counts 6765, 3242, 3244, 2657 and 4092. The lane hashes the training
predictions and the learn loss curve. The held-out probe and the batch part (predict is
row-wise) apply.

## Verdicts, fixtures base, ties and odd

| check | result | file |
|---|---|---|
| Metal, two repeats | STABLE on all train, infer, model and batch cells | `apple-m4.json` |
| CPU column, two repeats | STABLE on all cells | `cpu-apple-m4.json` |
| Metal vs CPU (`--require-columns 2`) | train IDENTICAL=3, infer/model IDENTICAL=6, batch IDENTICAL=3 | `diff-metal-cpu.txt` |
| host sabotage build vs Metal | train DIVERGENT=3, infer/model DIVERGENT=6, batch DIVERGENT=3 | `diff-metal-cpu-sabotage.txt` |
| `MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1`, Metal, base | batch_moved=1, exit 1 | `apple-m4.batch-sabotage.json` |
| the 16 existing gbdt lanes, Metal, vs the stage 1 merge column | IDENTICAL=48, infer/model IDENTICAL=90 N/A=6, batch IDENTICAL=48 | `diff-existing-metal.txt` |
| the 16 existing gbdt lanes, CPU, vs the stage 1 merge column | IDENTICAL=36 with the same 12 REFUSED, infer/model IDENTICAL=66 N/A=6, batch IDENTICAL=36 | `diff-existing-cpu.txt` |

## The CatBoost CPU reference (correctness, not bits)

CatBoost 1.2.10 on its CPU, one thread, every option equal to our fit's
(`catboost_query_rmse_reference.py`), against our Metal fit (`ours_query_rmse_reference.py`).
The differences come from `compare_query_rmse_reference.py`, and nothing was tuned to match.
Our learn loss is the square of their QueryRMSE metric at unit weights.

| fixture | learn loss, largest relative diff over 20 iterations | final QueryRMSE diff | final NDCG diff | final DCG diff | first 8 raw predictions, largest diff |
|---|---|---|---|---|---|
| base | 7.06e-08 | +2.38e-09 | +7.8e-16 | +8.0e-15 | 5.94e-08 |
| ties | 7.13e-08 | +8.84e-10 | +2.2e-16 | -5.5e-14 | 2.63e-08 |
| odd | 8.56e-08 | -6.43e-10 | 0 | +2.7e-15 | 4.00e-08 |

The residual differences are float32 against float64. NDCG and DCG here come from the
reference's CPU definitions, applied in float64 to our predictions. The GPU metric kernels
of `dcg.cu` are not implemented yet (gbdt/NOT_IMPLEMENTED.tsv).

## Tests

`test_gbdt_query_rmse` with `test_gbdt_group_id`: 20 passed on the Metal route and 20 passed
on the CPU route. `test_host_surface` and `test_gbdt_search_option_guards` pass on the Metal
route. `docs_facts --check` and `wheel_ci pins` pass.

Not run here: NVIDIA and AMD columns (owed to the next release record), and the fixtures
other than base, ties and odd.
