# YetiRank, the sampled-permutation ranking loss (lane/gbdt-learning-to-rank, stage 4)

`GradientBoosting(loss="YetiRank").fit(X, y, group_id=...)` on the GPU through
`gbdt/targets/kernel/yeti_rank.mojo` (the CatBoost reference's `yeti_rank_pointwise.cu` with the
two `radix_sort_block.cuh` passes, through the querywise target's `InitYetiRank` arm), over the task
table of `gbdt/data/yeti_rank_tasks.mojo`, and the CPU reference column restating the same order in
`gbdt/host/gbdt_oracle_yeti.mojo`. Defaults are the reference's: 10 permutations, decay 0.85, Newton
at one iteration, L2 0 (with its 1e-20 substitution), leaves shifted to average zero.

Two named DEVIATIONs:

1. One device thread per task of at most 1024 rows, sequential in the reference's per-document
   order (draws by round, thread and lane; the stable order by query then key; each lane's phase 1
   then phase 2), where the reference runs 256 threads with shared memory. The float sums per
   document have one order in both.
2. The derivative seeds come from `TRandom(random_state ^ "YETIRANK")`, one draw for each tree's
   search pass and one seeding its estimation stream, where the reference draws from the one random
   stream its whole fit shares. The reference GPU's bits cannot be reproduced; both columns here agree.

`GradientBoosting(l2_leaf_reg=None)` is the new default and resolves to 3.0 for every other loss,
the value it had, so no existing lane's hash moves (the 18 earlier gbdt lanes below).

## Machine and builds

Apple M4, one core, identical tier, `mojo build -j 1`; every GPU step alone under the exclusive
Metal slot.

| binding | sha256 |
|---|---|
| `_mojolearn_gbdt.so` (Metal) | 5cbca71bbfd72ebd |
| `_mojolearn_gbdt_host.so` (CPU column) | 616b6ecf7f32f5e3 |
| `_mojolearn_gbdt_host.so`, `-D MOJOLEARN_HOST_SABOTAGE=1` | 6ffb6710004fbca2 |

## The lane

`gbdt-yeti-rank`: the gbdt-query-rmse queries and grades. A 20-tree depth-6 fit and an 8-tree
depth-4 fit at `random_state=7` (unweighted: the host column refuses `sample_weight`), both
predictions and the zero loss curve hashed; the held-out probe and the batch part are the first fit's.

## Verdicts, fixtures base, ties and odd

| check | result | file |
|---|---|---|
| Metal vs CPU, two repeats each, both stable | IDENTICAL=3; (infer/model): IDENTICAL=6; (batch): IDENTICAL=3 | `diff-metal-cpu.txt` |
| host sabotage build vs Metal | DIVERGENT=3; (infer/model): DIVERGENT=6; (batch): DIVERGENT=3 | `diff-metal-cpu-sabotage.txt` |
| `MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1`, Metal, base | batch_moved=1, exit 1 | `apple-m4.batch-sabotage.json` |
| the 18 earlier gbdt lanes, Metal, against the committed columns | IDENTICAL=54; (infer/model): IDENTICAL=102, N/A=6; (batch): IDENTICAL=54 | `diff-earlier-18-metal.txt` |
| the 18 earlier gbdt lanes, CPU vs Metal | IDENTICAL=54; (infer/model): IDENTICAL=102, N/A=6; (batch): IDENTICAL=54 | `diff-earlier-18-metal-cpu.txt` |

Found while proving it: the first Metal vs CPU probe diverged on every multi-query fixture and
matched on a single query. The device read `qids`, scratch only the QueryRMSE launcher fills;
`launch_yeti_rank_with` now runs `ComputeGroupIds` on every call, as the reference's `Run` does
(`kernel.h:444`). The binaries above are the fixed ones.

## x86 CPU, base fixture, one RunPod pod

Pod mew1yzpi0kwjze (195 s billed, $0.0130; DELETE 204, then GET 404 and absent from the pod
listing), host bindings `core` and `gbdt` built from this tree (`x86-64/`):

| check | result | file |
|---|---|---|
| `gbdt-yeti-rank`, x86 CPU against the M4 Metal and CPU columns | IDENTICAL on train, infer/model and batch | `x86-64/diff-yeti-rank-base.txt` |
| the 18 earlier gbdt lanes, x86 CPU against the committed M4 columns | IDENTICAL=18; (infer/model): IDENTICAL=34, N/A=6; (batch): IDENTICAL=18 | `x86-64/diff-earlier-18-base.txt` |
| `test_gbdt_yeti_rank`, `test_cpu_training_gbdt_losses`, `test_host_surface` | 153 passed, 2 failed | `x86-64/pytest-x86-64.log` |

The two failures are `test_host_surface`'s checks that the recordings and GPU columns the manifest
names exist in the tree: the pod leg ships no `bench/results`, so they cannot pass there. The same
module passes on the M4 with the tree present (144 passed).

## Not run here

The run was stopped, on the coordinator's instruction to reduce load on the shared Mac, after the
verdicts above and before these steps, which are owed: the Metal and CPU test routes for
`test_gbdt_yeti_rank`, and the CatBoost CPU quality comparison (`yeti_rank_reference.py`: final NDCG
and DCG only, because the reference's CPU YetiRank samples with its own generator and neither side
has a loss value). NVIDIA and AMD columns are owed to the next release record, as are PFound and the
fixtures other than base, ties and odd.
