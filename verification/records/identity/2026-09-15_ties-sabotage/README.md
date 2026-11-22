# lane/ties-sabotage: value arms for the neighbor and IVF host oracles

One RunPod CPU pod (x86-64, 16 vCPU, pod z8rbbimhd1br3l, DELETE verified: GET
404 and absent from the listing; `x86-runpod/teardown.txt`), commit 1475db274,
every host family built production and sabotage through the binding cache. The
OLD arms are origin/main's `core/knn_host_predict.mojo` and
`ivf/host/ivf_host.mojo`, rebuilt on the pod into the core, ivf and ivf_search
sabotage bindings (`old_sources_flip_count.txt` reads 0 flips in them and 5 and
3 in the restored branch files). Command: `leg_cmd.tmpl.sh`, arguments
`leg_args.sh`, results `x86-runpod/`.

## Identity columns, six lanes, all nine fixtures (`moved_counts.txt`)

| sabotage set | cell parts moved from production | not moved |
|---|---|---|
| old arms | 166 of 216 | 36 model parts, plus `ties`: knn-cosine infer and batch; knn-rbc, radius, radius-manhattan and ivf train, infer and batch |
| new arms | 180 of 216 | the same 36 model parts only |

The 36 model parts are the saved-file hash (`identity_break._probe_fit`,
`_hfile(path)`) of knn-cosine, knn-rbc, radius and radius-manhattan on nine
fixtures. That file holds the fitted index rows and parameters, which no host
oracle computes, so no host sabotage arm can move it (old or new). The ivf and
ivf-euclidean model parts hold the quantizer's centroids and move on every
fixture under both arms.

Against the committed records (`diff.*.txt`):

| diff | old arms | new arms |
|---|---|---|
| neighbor lanes vs the 166-lane record | DIVERGENT=33, IDENTICAL=3; infer/model DIVERGENT=32, IDENTICAL=22; batch DIVERGENT=32, IDENTICAL=4 | DIVERGENT=36; infer/model DIVERGENT=36, IDENTICAL=18 (the unmovable model file hashes); batch DIVERGENT=36 |
| ivf lanes vs the ivf-euclidean record | DIVERGENT=17, IDENTICAL=1; infer/model DIVERGENT=17, IDENTICAL=1; batch DIVERGENT=17, IDENTICAL=1 | DIVERGENT=18 on train, infer/model and batch |

## Production unchanged

The production column against the committed records on all nine fixtures
(base and ties included): neighbor lanes `summary: IDENTICAL=36`, infer/model
`IDENTICAL=54, OWED=18`, batch `IDENTICAL=36`, require-columns 4 OK; ivf lanes
`IDENTICAL=18`, infer/model `IDENTICAL=18, OWED=18`, batch `IDENTICAL=18`,
require-columns 4 OK. The classical host gate reads `IDENTICAL (54 fixtures, 3
GPU columns)` on the neighbors and density recording and `IDENTICAL (115
fixtures, 3 GPU columns)` on `saved_model_recorded()`.

## Owed check

`owed_iv_sabotage_{old,new}.txt`: OK, 18 of 18 owed parts moved.
`owed_nb_sabotage_{old,new}.txt`: FAIL, 0 of 18, on both arms: the owed parts are
the radius and radius-manhattan model file hashes above, which no GPU record
carries yet and no host sabotage arm reaches. This predates the lane (the old
arms read the same) and is not changed here.

## The classical host gate on the new rule

| check | old arms | new arms |
|---|---|---|
| the six lanes' recordings (30 fixtures), `--every-fixture` | SABOTAGE NOT CAUGHT ON FIXTURES ivf-euclidean/ties, ivf/ties, knn-cosine/ties, knn-rbc/ties, radius-manhattan/ties, radius/ties | EXPECTED MISMATCH SEEN |
| `saved_model_recorded()` (115 fixtures), `--every-fixture` | SABOTAGE NOT CAUGHT ON FIXTURES ivf-euclidean/ties, ivf/ties | EXPECTED MISMATCH SEEN |
| `saved_model_recorded()`, `--every-lane` (the old rule) | EXPECTED MISMATCH SEEN (ivf/ties, ivf-euclidean/ties unmoved) | |
| neighbors and density recording (54 fixtures), `--every-fixture` | | EXPECTED MISMATCH SEEN |

No saved-model lane fails `--every-fixture` under the new arms, so the workflow
step uses it with no `--lane-rule-only` exemption. Only the neighbors and
density recording of CLASSICAL_RECORDED shipped (the 40 MB uplink limit), so
the classical sabotage step keeps its plain rule. `tools/test_cpu_identity_gate.py`:
27 tests OK on the pod.
