# Sabotage audit, one question per lane (2026-09-16)

Branch `lane/sabotage-audit`, from main at bfb8f725a.

THE QUESTION, asked of all 199 `tools/identity_break.py` lanes. Has this
lane's sabotage been SEEN to make the lane divergent, or do we only believe it
would? A sabotage nobody has watched fail is indistinguishable from no
sabotage.

THE DENOMINATOR IS 199, NOT 176. The first version of this document said 176,
which is the number of literal `@lane("...")` decorators a grep can see. The
registry holds 23 more that are registered in loops and are invisible to a
decorator count: five kde kernel and metric variants, six knn metric variants,
three radius metric variants, three gp kernels, `gp-sample-y` with its
normalized twin, `gp-optimize` with its restarts twin, and two `gmm-sample`
lanes. The count here is `len(LANES)` with `tools/identity_break.py` loaded as
a module, which is the only way to get it right.

Twenty-one of those 23 were already category (a), so the correction moved the
base without moving the evidence. TWO WERE NOT. `gp-optimize` and
`gp-optimize-restarts` appear in no host family in
`python/mojolearn/host_surface.py` and are not in `covered_lanes()`, so no
host sabotage define reaches them and they are category (c).

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
| (a) seen to move | 126 | a committed column shows this lane's host arm change a cell part |
| (b) arm exists, never seen to move | 43 | a define reaches the lane's family, no committed column shows it move |
| (c) no sabotage covers the lane | 30 | no host family and no host sabotage define reaches the lane at all |

(c) is the 28 par-* driver lanes plus `gp-optimize` and `gp-optimize-restarts`.

Every one of the 32 host families' sabotage defines does reach at least one
`comptime if` arm, so no family define is dead. That was checked statically
against each family's declared `host_modules` in
`python/mojolearn/host_surface.py`.

## Cheap gaps closed here

Sixteen lanes moved from (b) to (a) on this Mac's CPU route, one core, no box
rented and no Metal job. Evidence
`bench/results/identity_break/2026-09-16_sabotage-audit/`. Four families were
built production and `-D MOJOLEARN_HOST_SABOTAGE=1` into two directories,
eight binaries with eight sha256 values, and the columns diffed cell by cell.

| round | families | lanes | result |
|---|---|---|---|
| 1 | estimators, core | `pca`, `pca-whiten`, `tsvd`, `ols`, `ridge`, `logistic`, `logistic-multiclass`, `kde`, `knn`, `knn-clf`, `knn-reg` | 79 parts moved, 9 did not |
| 2 | linalg, resample | `gemm-pinned`, `gemm-transposed`, `bootstrap`, `permutation-test`, `monte-carlo` | 14 parts moved, 4 did not |

Round 2 deliberately took the two families whose sabotage define reaches
EXACTLY ONE `comptime if` site, because a single-site arm is the one most
likely to be inert. One of them was.

## Sabotage that cannot fail

### 1. The shared GEMM leaf arm is inert on the integer fixture

`gemm/host/gemm_oracle.mojo`'s arm walks each accumulation leaf DESCENDING
instead of ascending. An order perturbation cannot change a sum whose values
add exactly, and the `ties` fixture is integer valued, so on `ties` the arm
moves nothing. MEASURED HERE: `gemm-pinned` and `gemm-transposed` move train
and batch on `base` and read UNMOVED on `ties`, both parts
(`sabotage_moves_round2.txt`).

This reaches further than those two lanes. That one site is the ONLY arm that
`linalg`, `mamba` and `transformer` reach, and one of only two for `training`
and `neural`. On the `ties` fixture those five families have no working
negative control at all.

FIXED, on branch `lane/sabotage-evidence` at cc5234c00, not yet on main
(`origin/main` is 8cc2002b0 and still carries the order arm, so the finding
above stands against main). `gemm/host/gemm_oracle.mojo` now perturbs a VALUE
through `gemm_oracle_sabotage_value_flip`, the same remedy `lane/ties-sabotage`
applied to the neighbor and IVF families, and the old order arm is kept behind
`MOJOLEARN_GEMM_ORACLE_SABOTAGE_LEGACY_ORDER` so the defect can still be
watched failing rather than believed. No build script and no gate sets that
legacy define.

### 2. Saved-model cells that hold only the caller's input

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

### 3. A CTR sabotage column is indistinguishable from production

`bindings/_mojolearn_forest_host.mojo` exports `forest_host_sabotage()`
returning `FOREST_HOST_SABOTAGE` only. The CTR arm
(`MOJOLEARN_GBDT_CTR_HOST_SABOTAGE`, `core/gbdt_host_ctr.mojo`) is exported
separately as `forest_host_gbdt_ctr_sabotage()`. A column built with the CTR
define therefore records `sabotage: false` in `host.families`, so the column
cannot witness its own arm, and `_backend`'s `MOJOLEARN_HOST_ALLOW_SABOTAGE`
guard does not fire for it either. This audit's scanner classified the
committed `2026-09-15_gbdt-ctr-tables/cpu-x86-ctr-sabotage.json` as a
production column for that reason. The arm itself is real and was watched to
fail in that lane; what was missing is the read-back, so the metadata was
wrong about which binary ran.

FIXED, on branch `lane/sabotage-evidence` at cc5234c00, not yet on main.
`forest_host_sabotage_binding` now reports BOTH arms, so a column built with
the CTR define records `sabotage: true` and can witness its own binary.

The related trap of a sabotage build that silently loads the production
binding IS closed in code. `_probe_fit_host` and `_probe_saved_host` compare
the bound binding's path against `binary_path()` and raise when they differ
(`tools/identity_break.py` lines 6208 to 6217), which is what caught the
forest sabotage column reading IDENTICAL on a RunPod x86 pod.

### 4. The tokenizer refuses instead of diverging

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

### 5. Arms that are inert on particular fixtures

| lane | what stays unmoved | where |
|---|---|---|
| `gemm-pinned`, `gemm-transposed` | train and batch on `ties` | measured here, finding 1; fixed on `lane/sabotage-evidence`, not yet on main |
| `holtwinters`, `holtwinters-multiplicative` | 6 of 36 cells, `denormal`, `denormal_ftz` and `wide` | `2026-09-15_holtwinters-linesearch-fix/diff.166-vs-cpu-sabotage.txt`, DIVERGENT=30 IDENTICAL=6 |
| `tsvd` | `model` on `ties`, while `base` moves | measured here |
| `knn-cosine`, `knn-rbc`, `radius`, `radius-manhattan`, `ivf` | the whole `ties` fixture under the OLD order-only arm | `2026-09-15_ties-sabotage/x86-runpod/moved_counts.txt` |

The `ties` family of failures is the integer fixture problem named in the
brief. It is FIXED for the neighbor and IVF families, where
`lane/ties-sabotage` replaced the order perturbation with a value flip and
recorded the new arms moving 180 of 216 parts against 166 of 216, every
remaining unmoved part being the input-copy model cells of finding 2. The GEMM
leaf is fixed the same way on `lane/sabotage-evidence` at cc5234c00, not yet on
main. It is NOT fixed for Holt-Winters or tsvd.

`metrics` is the other fixed case. The old arm moved 5 of 9 metrics cells and
0 of 9 metrics-classification cells; the new arm moves 9 of 9 for both
(`2026-09-15_metrics-sabotage-coverage/sabotage_moves.txt`).

### 6. One arm states its own inert condition, correctly

`resample/host/resample_host.mojo` shifts every per-replicate and per-chunk
tree's boundaries by one value, and its docstring says plainly that "the
`const` integrand folds exact integers and does not move". That is what an
honest arm looks like. The `monte-carlo` lane still moved on both fixtures
here, because it hashes more than the const integrand.

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

## Owed

- A two-device box for any negative control over the 28 par-* lanes.
- The 43 category (b) lanes not closed here, by host family: gbdt 14, rf 8,
  training 5, core 5, byte_lm 3, trees 3, forecast with arima 2, and one each
  in preprocessing, arima and estimators. The gbdt, rf and trees arms are
  claimed DIVERGENT in `python/mojolearn/host_surface.py` comments citing CI
  gate runs (34884487749, 34900811380 and others), but no committed column in
  this tree carries them. Those 25 lanes are believed on a CI log, not on a
  record here.
- Ten of the (b) lanes are par-* lanes that DO name a host family, so they are
  not in (c): `par-forest`, `par-forest-et`, `par-scaler`, `par-arima`,
  `par-mlp`, `par-queries-knn`, `par-queries-radius`, `par-queries-kde`,
  `par-reference-knn` and `par-reference-knn-reg`. All ten are in
  `covered_lanes()` and a host arm reaches their families. WHETHER THEY CAN BE
  CLOSED WITHOUT A BOX IS UNSETTLED, and an earlier version of this document
  contradicted itself by calling them closable while also saying par-* lanes
  refuse on a CPU column. Nothing in the tree settles it: no committed CPU
  column carries a `par-forest` or `par-forest-et` cell at all, production or
  sabotage, so there is no evidence either that the CPU route runs them or
  that it refuses them. They are OWED, not cheap, until a column exists.
- Each remaining family is one build pair away from being closed the way these
  four were, at about 15 to 90 seconds a build on one core.

## SETTLED 2026-09-20 (lane/broken-par-sabotage-arms): the CPU route DOES run 23 par-* lanes

The question the Owed item above left open -- whether a `par-*` lane can take
a CPU column at all -- has been answered by running all 59 of them on this
Mac's CPU host route, one core, base fixture, `--repeats 2`, with the full
32-family host set built from this tree. Evidence
`bench/results/identity_break/2026-09-20_broken-par-sabotage-arms/`.

| result on the CPU column | lanes |
|---|---|
| ran, STABLE, full parts | 23 |
| refused: no host binding restates the cooperative driver | 36 |

The 36 refuse with a message that names the reason in the driver's own words
("its shards are device row tiles"), or, for three byte-LM lanes, a missing
GPU binding, or, for `par-causal-lm` and `par-cross-val`, a CUDA/HIP-only
guard. Those 36 are structural and a CPU box cannot supply a control for
them; the statement that this is true of EVERY par-* lane was too wide, and
it is the 23 above that the earlier text was wrong about.

A CPU-side negative control for the 23 exists and works. Building ONLY the
`core` family with `-D MOJOLEARN_HOST_SABOTAGE=1` and leaving the other 31
families production:

| under the core arm alone | before | after |
|---|---|---|
| CAUGHT (a part moved and `negative_control_moves` credits it) | 13 | 17 |
| INERT (ran, every part at its clean hash) | 6 | 6 |
| BROKEN (ran clean, REFUSED sabotaged) | 4 | 0 |

The BROKEN four were `par-arima`, `par-holtwinters`, `par-queries-knn` and
`par-queries-radius`, fixed on that branch; so are the four the lane was
opened for (`par-ivf`, `par-queries-nn`, `par-rbf-sampler`,
`par-forecast-arima`). The reach is wider than any family table predicts,
because `-D MOJOLEARN_HOST_SABOTAGE=1` on `core` also arms
`bindings/hotpath_helpers.mojo::HOTPATH_SABOTAGE`, which perturbs the generic
array helpers (`cast_elements`, `gather_*`, `reduce_stat`, `equal_elements`)
that EVERY driver's shard staging runs through. A lane whose host family is
`arima` is therefore reachable by the `core` define, which is why
`par-holtwinters` and `par-arima` move under a build that touches neither
family's own arm.

## CLOSED 2026-09-20 (lane/par-sabotage-defines): the 28 category (c) lanes now have arms

The Owed item above -- "a two-device box for any negative control over the 28
par-* lanes" -- was owed because no arm reached those lanes at all. The four
defines in the table above defended the native multi-GPU CHECKS under
`training/checks/`; none of them had ever been pointed at an `identity_break`
cell, and the other thirteen partitions had no define of any kind.

THIRTEEN NEW DEFINES, one per cooperative partition no host binding restates,
each a `comptime if` that shifts a READ offset for owners above rank 0 only.
`docs/multi_gpu/PAR_SABOTAGE_ARMS.md` is the manifest: the define, the module,
the build script that compiles it (computed with `tools/bincache.py`'s own
import-closure resolver, not guessed), and the exact two-device command.

| define | lanes it is for |
|---|---|
| `MOJOLEARN_GBDT_PARALLEL_SABOTAGE` | par-boosting, -clf, -reg, -pointwise, par-border-types, par-feature-freq, par-ordered, par-ordered-rmse |
| `MOJOLEARN_GRAM_PARALLEL_SABOTAGE`, `MOJOLEARN_QR_PARALLEL_SABOTAGE` | par-gram, par-gram-ols, par-gram-pca, par-gram-tsvd |
| `MOJOLEARN_SVM_PARALLEL_SABOTAGE` | par-svm, par-svm-svr, par-kernel-ridge, par-nystroem |
| `MOJOLEARN_SOLVER_PARALLEL_SABOTAGE` | par-cd, par-cd-elasticnet |
| `MOJOLEARN_NEIGHBORS_PARALLEL_SABOTAGE` | par-graph-agglomerative, par-graph-spectral, par-graph-umap |
| `MOJOLEARN_KMEANS_PARALLEL_SABOTAGE` | par-kmeans |
| `MOJOLEARN_GLM_PARALLEL_SABOTAGE` | par-logistic |
| `MOJOLEARN_DBSCAN_PARALLEL_SABOTAGE` | par-dbscan |
| `MOJOLEARN_GP_PARALLEL_SABOTAGE` | par-gp |
| `MOJOLEARN_ISOLATION_FOREST_PARALLEL_SABOTAGE` | par-iforest |
| `MOJOLEARN_FOREST_POOL_PARALLEL_SABOTAGE` | par-forest-pool |
| `MOJOLEARN_BYTE_LM_PARALLEL_SABOTAGE` | par-byte-lm, par-byte-lm-model-pool, par-byte-lm-offload |

WRITTEN IS NOT WATCHED, and this document exists to keep the two apart. None
of the thirteen has been seen to move a cell: every one of their lanes refuses
on a CPU column for the structural reason the driver states itself ("its
shards are device row tiles, chunks or ranges inside the GPU binding, which no
host binding restates"), or, for the three byte-LM lanes, because the GPU
binding is absent. They stay category (c) in the table below until a
two-device CUDA or HIP column carries them, and the command that would do it
is written down so the next box can run it without re-deriving anything.

WHAT WAS WATCHED. Two things, on this Mac's CPU route, one core,
`--repeats 2`, base fixture
(`bench/results/identity_break/2026-09-20_par-sabotage-defines/`):

* the 44 `par-*` lanes whose in-cell oracle was a bare `_same_bytes` now carry
  `_mismatch_bytes` / `NumericalMismatch(msg, parts)`, so an arm that fires
  produces a DIVERGENT cell instead of a REFUSED one. `audit_bare_oracles.py`
  reads 0 `par-*` lanes with a bare body oracle, against 45 at the branch
  point. Without this, every one of the thirteen defines would have credited
  nothing even when working perfectly, which is `par-scaler`'s defect and the
  eight lanes of `lane/broken-par-sabotage-arms`.
* `MOJOLEARN_PAR_DRIVER_SABOTAGE`, the Python-side arm for the
  NON-cooperative drivers, whose partition is the driver's own Python and
  which therefore has no binding to rebuild. See
  `docs/multi_gpu/PAR_SABOTAGE_ARMS.md` for what a column carrying it may and
  may not be counted as.

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
| `gemm-pinned` | a | linalg | yes | moved: batch,train; UNMOVED: batch,train |
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
| `kde-tophat-sqeuclidean` | a | estimators | yes | moved: batch,infer,train; UNMOVED: model |
| `kde-epanechnikov-l1` | a | estimators | yes | moved: batch,infer,train; UNMOVED: model |
| `kde-exponential-chebyshev` | a | estimators | yes | moved: batch,infer,train; UNMOVED: model |
| `kde-linear-cosine` | a | estimators | yes | moved: batch,infer,train; UNMOVED: model |
| `kde-cosine-minkowski` | a | estimators | yes | moved: batch,infer,train; UNMOVED: model |
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
| `knn-sqeuclidean` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `knn-manhattan` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `knn-chebyshev` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `knn-cosine` | a | core | yes | moved: batch,infer,train; UNMOVED: batch,infer,model |
| `knn-minkowski-p3` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `knn-rbc` | a | core | yes | moved: batch,infer,train; UNMOVED: batch,infer,model,train |
| `knn-clf-distance` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `knn-reg-distance` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `radius-manhattan` | a | core | yes | moved: batch,infer,train; UNMOVED: batch,infer,model,train |
| `radius-chebyshev` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `radius-minkowski-p3` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
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
| `gp-sample-y` | a | gp | yes | moved: infer,train |
| `gp-sample-y-normalize` | a | gp | yes | moved: infer,train |
| `gp-optimize` | c | - | no | no host family and no host sabotage define |
| `gp-optimize-restarts` | c | - | no | no host family and no host sabotage define |
| `gp-matern12` | a | gp,gp_infer | yes | moved: batch,infer,train |
| `gp-matern32` | a | gp,gp_infer | yes | moved: batch,infer,train |
| `gp-matern52-ard` | a | gp,gp_infer | yes | moved: batch,infer,train |
| `gemm-transposed` | a | linalg | yes | moved: batch,train; UNMOVED: batch,train |
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
| `gmm-sample` | a | mixture,mixture_infer | yes | moved: infer,train |
| `gmm-random-init-sample` | a | mixture,mixture_infer | yes | moved: infer,train |
| `hdbscan` | a | hdbscan,hdbscan_infer | yes | moved: batch,infer,model,train |
| `hdbscan-leaf` | a | hdbscan,hdbscan_infer | yes | moved: batch,infer,model,train |
| `bootstrap` | a | resample | yes | moved: batch,train |
| `permutation-test` | a | resample | yes | moved: batch,train |
| `monte-carlo` | a | resample | yes | moved: train |
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
