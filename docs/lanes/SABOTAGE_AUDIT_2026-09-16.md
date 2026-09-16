# Sabotage audit, one question per lane (2026-09-16)

Branch `lane/sabotage-audit`, from main at bfb8f725a.

THE QUESTION, asked of all 176 `tools/identity_break.py` lanes. Has this
lane's sabotage been SEEN to make the lane divergent, or do we only believe it
would? A sabotage nobody has watched fail is indistinguishable from no
sabotage.

## What counts as seen

A lane is category (a) only when a COMMITTED column pair shows a cell part
change. The pair is found by reading each column's own metadata rather than
its filename. A column records `host.families[<binding>].sabotage`, the
binding's own `<prefix>_sabotage()` read back, and `batch_sabotage`, the
harness switch. A sabotage column is one whose binding reports the arm
compiled in; it is diffed part by part (train, infer, model, batch) against a
production column of the same vendor in the same directory, falling back to
the 166-lane record.

TWO RULES THIS AUDIT APPLIES that earlier readings did not.

1. A move under `MOJOLEARN_IDENTITY_BATCH_SABOTAGE`,
   `MOJOLEARN_IDENTITY_BATCHGRAD_SABOTAGE` or
   `MOJOLEARN_IDENTITY_RLPAIR_SABOTAGE` is NOT evidence for a lane's host arm.
   Those are harness switches that perturb the harness's own whole-batch
   evaluation, so they move a batch cell whatever the binding does. Six lanes
   had nothing else behind them (`byte-lm-host-infer`, `pca`, `par-dbscan`,
   `par-graph-agglomerative`, `par-kmeans`, `par-queries-knn`).
2. A pair must share a vendor. Pairing by filename alone matched an Apple
   sabotage column against an NVIDIA production column in the rlpair
   directory and would have counted vendor difference as a sabotage move.

## Counts

| category | lanes | meaning |
|---|---|---|
| (a) seen to move | 100 | a committed column shows this lane's host arm change a cell part |
| (b) arm exists, never seen to move | 48 | a define reaches the lane's family, no committed column shows it move |
| (c) no sabotage covers the lane | 28 | no host family and no host sabotage define reaches the lane at all |

Every one of the 32 host families' sabotage defines does reach at least one
`comptime if` arm, so no family define is dead. That was checked statically
against each family's declared `host_modules` in
`python/mojolearn/host_surface.py`.

## Cheap gaps closed here

Eleven lanes moved from (b) to (a) on this Mac's CPU route, one core, no box
rented and no Metal job. Evidence
`bench/results/identity_break/2026-09-16_sabotage-audit/`. The `estimators`
and `core` families were built production and `-D MOJOLEARN_HOST_SABOTAGE=1`
into two directories, four binaries with four sha256 values, and the columns
diffed cell by cell.

`pca`, `pca-whiten`, `tsvd`, `ols`, `ridge`, `logistic`, `logistic-multiclass`,
`kde`, `knn`, `knn-clf`, `knn-reg`. 79 cell parts moved, 9 did not.

## Sabotage that cannot fail

### 1. Saved-model cells that hold only the caller's input

The `model` part is `identity_break._hfile(path)`, the sha256 of the saved
file. For the neighbor and density estimators that file holds the fitted
index, which is the caller's own rows as fitted, plus scalars. Every host
sabotage arm in `core/knn_host_predict.mojo` perturbs a distance computed at
query time, so none of them reaches the stored index. These cells cannot be
moved by any arm that exists, and an owed cell that no sabotage can move rests
on nothing.

MEASURED HERE for `kde`, `knn`, `knn-clf` and `knn-reg` (8 unmoved model parts
on base and ties). ALREADY KNOWN for `radius` and `radius-manhattan`, where
`tools/cpu_identity_gate_check.py owed` reads `FAIL (0 of 18 owed cell parts
moved)` under BOTH the old and the new arms
(`bench/results/identity_break/2026-09-15_ties-sabotage/x86-runpod/owed_nb_sabotage_new.txt`).

This is Andrew's open item and it is wider than radius. The mechanism for
naming it already exists and is already used. `identity_break.Fit.model_na`
carries an `n/a:<reason>` for a saved file that holds only what the lane
passed in, and the Embedding lane uses `model_na="n/a:input-table"`
(`tools/identity_break.py` lines 3116 and 3147) for exactly this reason. The
straight fix is `model_na="n/a:input-index"` on the k-NN, radius and KDE
lanes, which removes the cell at the source so it never enters the owed list,
rather than an exemption inside the owed checker.

### 2. A CTR sabotage column is indistinguishable from production

`bindings/_mojolearn_forest_host.mojo` exports `forest_host_sabotage()`
returning `FOREST_HOST_SABOTAGE` only. The CTR arm
(`MOJOLEARN_GBDT_CTR_HOST_SABOTAGE`, `core/gbdt_host_ctr.mojo`) is exported
separately as `forest_host_gbdt_ctr_sabotage()`. A column built with the CTR
define therefore records `sabotage: false` in `host.families`, so the column
cannot witness its own arm, and `_backend`'s `MOJOLEARN_HOST_ALLOW_SABOTAGE`
guard does not fire for it either. This audit's scanner classified the
committed `2026-09-15_gbdt-ctr-tables/cpu-x86-ctr-sabotage.json` as a
production column for that reason. The arm itself is real and was watched to
fail in that lane; what is missing is the read-back, so the metadata lies
about which binary ran.

The related trap of a sabotage build that silently loads the production
binding IS closed in code. `_probe_fit_host` and `_probe_saved_host` compare
the bound binding's path against `binary_path()` and raise when they differ
(`tools/identity_break.py` lines 6208 to 6217), which is what caught the
forest sabotage column reading IDENTICAL on a RunPod x86 pod.

### 3. The tokenizer refuses instead of diverging

In the committed column
`bench/results/identity_break/2026-09-15_tokenizer-batch/cpu.host-sabotage.json`
the tokenizer lane reads REFUSED on all nine fixtures, not DIVERGENT. A
refusal is a pass that proves nothing. The manifest records the cause and the
intended fix, that `MOJOLEARN_HOST_SABOTAGE` reaches nothing in a binding of
integers and tables, so the gate now builds the tokenizer with its own define
through `GATE_SABOTAGE_OWN_DEFINES`. What is still true is that the only
tokenizer part any committed column has SEEN move is `batch`, under
`MOJOLEARN_TOKENIZER_BATCH_SABOTAGE`. Its `train` and `infer` parts read
UNMOVED in that same column, which is correct for that arm by design, and no
committed column shows them move under
`MOJOLEARN_TOKENIZER_HOST_SABOTAGE`. `metrics` and `metrics-fowlkes-mallows`
also rest on a single part, but those lanes have only a train cell, so that is
complete rather than thin.

### 4. Arms that are inert on particular fixtures

| lane | what stays unmoved | where |
|---|---|---|
| `holtwinters`, `holtwinters-multiplicative` | 6 of 36 cells, the `denormal`, `denormal_ftz` and `wide` fixtures | `2026-09-15_holtwinters-linesearch-fix/diff.166-vs-cpu-sabotage.txt`, DIVERGENT=30 IDENTICAL=6 |
| `tsvd` | `model` on `ties`, while `base` moves | measured here |
| `knn-cosine`, `knn-rbc`, `radius`, `radius-manhattan`, `ivf` | the whole `ties` fixture under the OLD order-only arm | `2026-09-15_ties-sabotage/x86-runpod/moved_counts.txt` |

The `ties` family of failures is the integer fixture problem named in the
brief, and it is FIXED. `lane/ties-sabotage` replaced the order perturbation
with a value flip (`host_sabotage_value_flip`, `ivf_sabotage_value_flip`), and
the same file records the new arms moving 180 of 216 parts against 166 of 216,
with every remaining unmoved part being the input-copy model cells of finding
1. The Holt-Winters and tsvd rows above are NOT fixed.

`metrics` is the other fixed case. The old arm moved 5 of 9 metrics cells and
0 of 9 metrics-classification cells; the new arm moves 9 of 9 for both
(`2026-09-15_metrics-sabotage-coverage/sabotage_moves.txt`).

## Category (c), the 28 par-* lanes, and why

These lanes name no host family in `python/mojolearn/host_surface.py`, so no
host sabotage define reaches their cells and none can. They are multi-device
driver lanes. Their negative controls are device side and defend the multi-GPU
CHECKS rather than the lane cells, and four of those have been watched to fail
in committed records.

| define | check | committed evidence of failing |
|---|---|---|
| `MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE` | `training/checks/cholesky_parallel_check.mojo` | `bench/results/multi_gpu/2026-09-14/cholesky-rows-h100/README.md` |
| `MOJOLEARN_GMM_PARALLEL_SABOTAGE` | `training/checks/gmm_parallel_check.mojo` | `bench/results/multi_gpu/2026-09-14/gmm-rows-h100/README.md`, `gmm-rows-mi300x` |
| `MOJOLEARN_HIERARCHY_PARALLEL_SABOTAGE` | HDBSCAN multi-GPU | `bench/results/multi_gpu/2026-09-14/hdbscan-h100/README.md`, `hdbscan-mi300x` |
| `MOJOLEARN_RESAMPLE_PARALLEL_SABOTAGE` | `training/checks/resample_parallel_check.mojo` | `bench/results/multi_gpu/2026-09-14/resample-h100/README.md`, `resample-mi300x` |

The other 24 par-* lanes have no arm of any kind that reaches their cells. A
par-* lane also REFUSES on a CPU column, because it asks for devices, so the
CPU route cannot supply a negative control for any of them. `par-kmeans` reads
REFUSED on train, infer and batch in four committed sabotage columns, which is
a refusal and not a catch. Moving these out of (c) needs a two-device box, so
it is owed rather than cheap.

## One lane no sabotage can ever move, correctly

`kmeans-cosine` is category (b) by the table but belongs with the
unmovable. Its cell is the sha256 of a REFUSAL SENTENCE, because
`metric='cosine'` is refused by name in
`cluster/impl/kmeans_params.mojo::validate`. There is no arithmetic to
perturb. A column that hashes anything else there means the refusal was
lifted, which is what the lane is for. This wants a named exemption for the
same reason the input-copy cells do.

## Lanes whose arm reaches only the shared GEMM leaf

`linalg`, `mamba` and `transformer` each reach their sabotage arm through one
site only, `gemm/host/gemm_oracle.mojo`'s descending leaf. `training` and
`neural` reach two, that leaf and `training/host/mlp_oracle.mojo`. Any cell in
those families that does not route through the host GEMM cannot be moved, and
that is visible in the committed record. Under the `neural` family's arm the
`mamba1`, `mamba2`, `mamba2-dtlimit`, `mamba3`, `transformer` and
`transformer-window` train parts read UNMOVED
(`2026-09-15_neural-forward-inference/cpu-amd-epyc-4564p.host-sabotage.json`).
Those lanes are in (a) on the strength of their OWN families' arms
(`mamba`, `transformer`), which is the right reading, but the neural family's
coverage of them is thinner than one arm per lane.

## Owed

- A two-device box for any negative control over the 28 par-* lanes.
- The 48 category (b) lanes not closed here, by host family: gbdt 14, rf 8,
  training 5, core 5, byte_lm 3, trees 3, resample 3, linalg 2, forecast with
  arima 2, and one each in preprocessing, arima and estimators. The gbdt, rf
  and trees arms are claimed DIVERGENT in `python/mojolearn/host_surface.py`
  comments citing CI gate runs (34884487749, 34900811380 and others), but no
  committed column in this tree carries them. Those 25 lanes are believed on a
  CI log, not on a record here.
- Ten of the (b) lanes are par-* lanes that DO name a host family, so they are
  not in (c): `par-forest`, `par-forest-et`, `par-scaler`, `par-arima`,
  `par-mlp`, `par-queries-knn`, `par-queries-radius`, `par-queries-kde`,
  `par-reference-knn` and `par-reference-knn-reg`. They are CPU covered and
  reachable by a host arm, so they are closable without a box.
- `byte-lm-host-train`, `byte-lm-host-infer-threaded`, `arima-exog`,
  `arima-exog-seasonal`, `gemm-pinned`, `gemm-transposed`, `cross-val`,
  `bootstrap`, `permutation-test`, `monte-carlo`, `training-primitives`,
  `optim-sgd`, `optim-adam-clip` and `cross-entropy-arms` are each one family
  build away from being closed the same way this lane closed eleven, on CPU,
  with no box.

## The per-lane table

Category (a) means at least one cell part has been seen to move. Where an arm
ran and left a part unmoved, the part is named.

| lane | cat | host family | CPU-covered | evidence / reason |
|---|---|---|---|---|
| `rf-clf` | b | rf | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `rf-reg` | b | rf | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `et-clf` | a | trees | yes | moved: infer,model,train; UNMOVED: infer,model,train |
| `et-reg` | a | trees | yes | moved: infer,model,train; UNMOVED: infer,model,train |
| `gbdt-symmetric` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-depthwise` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-lossguide` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-rmse` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `kmeans` | a | core | yes | moved: batch,infer,train |
| `knn` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `knn-clf` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `knn-reg` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `dbscan` | a | estimators | yes | moved: batch,infer; UNMOVED: model,train |
| `pca` | a | estimators | yes | moved: batch,infer,model,train |
| `pca-whiten` | a | estimators | yes | moved: batch,infer,model,train |
| `tsvd` | a | estimators | yes | moved: batch,infer,model,train; UNMOVED: model |
| `ols` | a | estimators | yes | moved: batch,infer,model,train |
| `ridge` | a | estimators | yes | moved: batch,infer,model,train |
| `logistic` | a | estimators | yes | moved: batch,infer,model,train |
| `lasso` | a | estimators,solver | yes | moved: batch,infer,model,train |
| `elasticnet` | a | estimators,solver | yes | moved: batch,infer,model,train |
| `svc` | a | svm | yes | moved: batch,infer,model,train |
| `kde` | a | estimators | yes | moved: batch,infer,train; UNMOVED: model |
| `agglomerative` | a | solver | yes | moved: batch,infer,train; UNMOVED: model,train |
| `spectral` | a | metrics | yes | moved: batch,infer; UNMOVED: model,train |
| `holtwinters` | a | forecast,tsa | yes | moved: batch,infer,train; UNMOVED: batch,infer,train |
| `gemm-pinned` | b | linalg | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `metrics` | a | metrics | yes | moved: train; UNMOVED: train |
| `svr` | a | svm | yes | moved: batch,infer,model,train |
| `arima` | a | arima,forecast | yes | moved: batch,infer,model,train |
| `gp` | a | gp,gp_infer | yes | moved: batch,infer,train |
| `gpc` | a | gp,gp_infer | yes | moved: batch,infer,model,train |
| `gpc-multiclass` | a | gp,gp_infer | yes | moved: batch,infer,model,train |
| `umap` | a | metrics | yes | moved: infer,model,train |
| `radius` | a | core | yes | moved: batch,infer,train; UNMOVED: batch,infer,model,train |
| `standard-scaler` | a | estimators,preprocessing | yes | moved: batch,infer,model,train |
| `minmax-scaler` | a | estimators,preprocessing | yes | moved: batch,infer,model,train |
| `gbdt-ordered-rmse` | a | gbdt | yes | moved: infer,model; UNMOVED: batch,model,train |
| `gbdt-feature-freq` | a | gbdt | yes | moved: infer,model; UNMOVED: batch,model,train |
| `mlp` | a | training | yes | moved: batch,infer; UNMOVED: model,train |
| `byte-lm` | a | byte_lm | yes | moved: batch,infer,model,train |
| `byte-lm-host-infer` | b | byte_lm | yes | arm exists (MOJOLEARN_BYTE_LM_HOST_SABOTAGE); no committed column shows it move; harness batch-env only |
| `byte-lm-host-train` | b | byte_lm | yes | arm exists (MOJOLEARN_BYTE_LM_HOST_SABOTAGE); no committed column shows it move |
| `mamba1` | a | mamba | yes | moved: batch,infer,train; UNMOVED: infer,train |
| `mamba2` | a | mamba | yes | moved: batch,infer,train; UNMOVED: train |
| `mamba3` | a | mamba | yes | moved: batch,infer,train; UNMOVED: train |
| `transformer` | a | transformer | yes | moved: batch,infer,train; UNMOVED: train |
| `samba` | a | training | yes | moved: batch,infer,model,train; UNMOVED: model,train |
| `rf-clf-entropy-log2-noboot` | b | rf | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `rf-clf-balanced-parallel` | b | rf | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `rf-reg-poisson` | b | rf | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `rf-reg-gamma-ig` | b | rf | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `et-clf-entropy-bestfirst` | b | trees | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `et-reg-bootstrap-parallel` | b | trees | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-multiclass` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-onevsall` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-parametric-losses` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-lossguide-newtoncosine` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-pointwise-l2-bayesian-eval` | a | gbdt | yes | moved: infer,model; UNMOVED: batch,model,train |
| `gbdt-exact-mae` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-categorical-ctr` | a | gbdt | yes | moved: infer,model; UNMOVED: batch,model,train |
| `gbdt-categorical-ctr-tables` | a | forest | yes | moved: batch,infer,train; UNMOVED: model |
| `gbdt-tensor-ctr-tables` | a | forest | yes | moved: batch,infer,train; UNMOVED: model |
| `gbdt-nan-modes` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-adapter-clf` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-adapter-reg` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-query-rmse` | a | gbdt | yes | moved: batch,infer,model,train |
| `gbdt-pair-logit` | a | gbdt | yes | moved: batch,infer,model,train |
| `gbdt-yeti-rank` | a | gbdt | yes | moved: batch,infer,model,train |
| `gbdt-adapter-score-weighted` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `rf-score-weighted` | b | rf | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `mamba2-dtlimit` | a | mamba | yes | moved: batch,infer,train; UNMOVED: train |
| `transformer-window` | a | transformer | yes | moved: batch,infer,train; UNMOVED: train |
| `byte-lm-resident` | a | byte_lm | yes | moved: batch,infer,model,train |
| `byte-lm-host-infer-threaded` | b | byte_lm | yes | arm exists (MOJOLEARN_BYTE_LM_HOST_SABOTAGE); no committed column shows it move |
| `samba-untied-dropout-accum` | a | training | yes | moved: batch,infer,model,train; UNMOVED: model,train |
| `optim-sgd` | b | training | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `optim-adam-clip` | b | training | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `cross-entropy-arms` | b | training | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `kmeans-random` | a | core | yes | moved: batch,infer,train |
| `kmeans-array` | a | core | yes | moved: batch,infer,train |
| `kmeans-weighted` | a | core | yes | moved: batch,infer,train |
| `dbscan-brute-l1` | a | estimators | yes | moved: batch,infer; UNMOVED: model,train |
| `dbscan-weighted` | a | estimators | yes | moved: batch,infer; UNMOVED: model,train |
| `kde-weighted` | a | estimators | yes | moved: batch,infer,train; UNMOVED: model |
| `pca-full-whiten` | a | estimators | yes | moved: batch,infer,model,train |
| `ols-no-intercept` | a | estimators | yes | moved: batch,infer,model,train |
| `ols-weighted` | a | estimators | yes | moved: batch,infer,model,train |
| `ridge-no-intercept` | a | estimators | yes | moved: batch,infer,model,train |
| `logistic-l1` | a | estimators | yes | moved: batch,infer,model,train |
| `logistic-multiclass` | a | estimators | yes | moved: batch,infer,model,train |
| `logistic-elasticnet` | a | estimators | yes | moved: batch,infer,model,train |
| `logistic-unpenalized-no-intercept` | a | estimators | yes | moved: batch,infer,model,train |
| `elasticnet-l2end-no-intercept` | a | estimators,solver | yes | moved: batch,infer,model,train |
| `svc-linear` | a | svm | yes | moved: batch,infer,model,train |
| `svc-poly` | a | svm | yes | moved: batch,infer,model,train |
| `svr-linear` | a | svm | yes | moved: batch,infer,model,train |
| `knn-clf-distance` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `knn-reg-distance` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `standard-scaler-no-mean` | a | estimators,preprocessing | yes | moved: batch,infer,model,train |
| `standard-scaler-no-std` | a | estimators,preprocessing | yes | moved: batch,infer,model,train |
| `minmax-scaler-clip` | a | estimators,preprocessing | yes | moved: batch,infer,model,train |
| `spectral-precomputed` | a | metrics | yes | moved: batch,infer; UNMOVED: model,train |
| `holtwinters-multiplicative` | a | forecast,tsa | yes | moved: batch,infer,train; UNMOVED: batch,infer,train |
| `kpss` | a | tsa | yes | moved: batch,train |
| `arima-011` | a | arima,forecast | yes | moved: batch,infer,model,train |
| `arima-seasonal-c` | a | arima,forecast | yes | moved: batch,infer,model,train |
| `arima-exog` | b | arima,forecast | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `arima-exog-seasonal` | b | arima,forecast | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gp-normalize-y` | a | gp,gp_infer | yes | moved: batch,infer,train |
| `gemm-transposed` | b | linalg | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `metrics-classification` | a | metrics | yes | moved: batch,train; UNMOVED: batch |
| `metrics-fowlkes-mallows` | a | metrics | yes | moved: train |
| `tokenizer` | a | tokenizer | yes | moved: batch; UNMOVED: infer,train |
| `cross-val` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `cholesky` | a | linalg | yes | moved: batch,infer,model,train; UNMOVED: batch,infer |
| `kernel-ridge` | a | estimators,kernel_methods | yes | moved: batch,infer,model,train |
| `nystroem` | a | estimators,kernel_methods | yes | moved: batch,infer,model,train |
| `rbf-sampler` | a | estimators,kernel_methods | yes | moved: batch,infer,train; UNMOVED: model |
| `gmm` | a | mixture,mixture_infer | yes | moved: batch,infer,model,train |
| `gmm-random-init` | a | mixture,mixture_infer | yes | moved: batch,infer,model,train; UNMOVED: batch,infer |
| `hdbscan` | a | hdbscan,hdbscan_infer | yes | moved: batch,infer,model,train |
| `hdbscan-leaf` | a | hdbscan,hdbscan_infer | yes | moved: batch,infer,model,train |
| `bootstrap` | b | resample | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `permutation-test` | b | resample | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `monte-carlo` | b | resample | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `training-primitives` | b | training | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `ivf` | a | ivf,ivf_search | yes | moved: batch,infer,model,train; UNMOVED: batch,infer,train |
| `ivf-euclidean` | a | ivf,ivf_search | yes | moved: batch,infer,model,train |
| `ivf-extend` | a | ivf,ivf_search | yes | moved: batch,infer,model,train |
| `embedding` | a | embedding,embedding_infer | yes | moved: batch,infer,train |
| `embedding-sort` | a | embedding | yes | moved: batch,infer,train |
| `kmeans-sqrt` | a | core | yes | moved: batch,infer,train |
| `kmeans-classic-pp` | a | core | yes | moved: batch,infer,train |
| `kmeans-cosine` | b | core | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move; arm RAN, did NOT move: train |
| `par-forest` | b | rf | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `par-forest-et` | b | trees | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `par-boosting` | c | - | no | no host family and no host sabotage define |
| `par-kmeans` | c | - | no | no host family and no host sabotage define; harness batch-env move only |
| `par-gram` | c | - | no | no host family and no host sabotage define |
| `par-logistic` | c | - | no | no host family and no host sabotage define |
| `par-cd` | c | - | no | no host family and no host sabotage define |
| `par-svm` | c | - | no | no host family and no host sabotage define |
| `par-gp` | c | - | no | no host family and no host sabotage define |
| `par-dbscan` | c | - | no | no host family and no host sabotage define; harness batch-env move only |
| `par-scaler` | b | preprocessing | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `par-arima` | b | arima | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `par-mlp` | b | training | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `par-samba` | c | - | no | no host family and no host sabotage define |
| `par-byte-lm` | c | - | no | no host family and no host sabotage define |
| `par-queries-knn` | b | core | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move; harness batch-env only |
| `par-queries-radius` | b | core | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `par-queries-kde` | b | estimators | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `par-reference-knn` | b | core | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `par-reference-knn-reg` | b | core | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `par-graph-agglomerative` | c | - | no | no host family and no host sabotage define; harness batch-env move only |
| `par-graph-spectral` | c | - | no | no host family and no host sabotage define |
| `par-graph-umap` | c | - | no | no host family and no host sabotage define |
| `par-ordered-rmse` | c | - | no | no host family and no host sabotage define |
| `par-feature-freq` | c | - | no | no host family and no host sabotage define |
| `par-boosting-pointwise` | c | - | no | no host family and no host sabotage define |
| `par-holtwinters` | a | tsa | yes | moved: batch,infer,train |
| `par-byte-lm-model-pool` | c | - | no | no host family and no host sabotage define |
| `par-byte-lm-offload` | c | - | no | no host family and no host sabotage define |
| `par-samba-clip` | c | - | no | no host family and no host sabotage define |
| `iforest` | a | svm | yes | moved: batch,infer,train; UNMOVED: model |
| `iforest-tuned` | a | svm | yes | moved: batch,infer,model,train |
| `par-iforest` | c | - | no | no host family and no host sabotage define |
| `par-forest-pool` | c | - | no | no host family and no host sabotage define |
| `par-gmm` | c | - | no | no host family and no host sabotage define |
| `par-resample` | c | - | no | no host family and no host sabotage define |
| `par-hdbscan` | c | - | no | no host family and no host sabotage define |
| `par-cholesky` | c | - | no | no host family and no host sabotage define |
| `par-kernel-ridge` | c | - | no | no host family and no host sabotage define |
| `par-nystroem` | c | - | no | no host family and no host sabotage define |
| `par-rbf-sampler` | c | - | no | no host family and no host sabotage define |
