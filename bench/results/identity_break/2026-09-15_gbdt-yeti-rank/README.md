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

## The CatBoost CPU reference (quality only)

CatBoost 1.2.10 CPU YetiRank, one thread, every option equal to our 20-tree fit (permutations 10,
decay 0.85, l2 0, Newton at one iteration, Cosine, 128 borders, seed 0, no bootstrap), against our
fit through the CPU binding, both on one RunPod pod (ww4v9j24fh94i2, 75 s billed, $0.0050; DELETE
204, then GET 404). The reference's CPU YetiRank samples its permutations with its own generator
(`private/libs/algo/yetirank_helpers.cpp`) and neither side has a loss value, so only final NDCG and
DCG (type Base, all positions) are compared; nothing was tuned (`yeti_rank_reference.py`,
`compare-catboost.txt`).

| fixture | our NDCG | reference NDCG | diff | our DCG | reference DCG | diff |
|---|---|---|---|---|---|---|
| base | 0.956819404 | 0.967912373 | -1.109e-02 | 7.22739082 | 7.33475 | -1.074e-01 |
| odd | 0.957565603 | 0.968994671 | -1.143e-02 | 7.23216854 | 7.33731752 | -1.051e-01 |
| ties | 0.99526113 | 0.996828765 | -1.568e-03 | 12.6471162 | 12.6644142 | -1.730e-02 |

Our fit ranks about 1.1 NDCG points below the reference's CPU learner on base and odd, and 0.16 on
ties. Not measured here: how much of that is the reference CPU's different sampler and pair
construction, and how much is anything in this implementation; the GPU learner this implementation
follows cannot run on this Mac.

Our fit on the M4's Metal (binding 5cbca71bbfd72ebd, alone under the exclusive Metal slot) gives the
same final NDCG and DCG as the x86 CPU values above, equal as float64 on base, ties and odd
(`ours-reference-metal.json` against `ours-reference-x86-64.json`). `test_gbdt_yeti_rank` on the Metal
route: 4 passed (`pytest-metal.log`); on the CPU route it passed on the x86 pod.

## Not run here

NVIDIA and AMD columns are owed to the next release record, as are PFound (the reference's score
metric for YetiRank) and the fixtures other than base, ties and odd.
