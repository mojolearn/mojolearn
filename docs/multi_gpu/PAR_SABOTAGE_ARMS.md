# The `par-*` negative controls: every arm, where it lives, how to watch it

`lane/par-sabotage-defines`, 2026-09-20. THE MANIFEST a two-device leg reads.

A sabotage nobody has watched fail is indistinguishable from no sabotage. On
2026-09-16 the [sabotage audit](../lanes/SABOTAGE_AUDIT_2026-09-16.md) found
that **no `par-*` lane had an arm that reached its CELLS**: the four defines
that existed (Cholesky, GMM, hierarchy, resample) defended the native
multi-GPU CHECKS under `training/checks/`, never an `identity_break` column.
On 2026-09-20 `lane/broken-par-sabotage-arms` closed eight of them on a CPU
column. This file is the rest: **thirteen new defines**, one per cooperative
partition no host binding restates, plus the Python-side switch for the
drivers that have no binding to rebuild.

## What every arm here does, and what makes it evidence

Each arm shifts a **READ offset** for owners above rank 0 and leaves the
length, the allocation, the validation and the write-back destination on the
true offset. The shard therefore computes a complete, well-formed, WRONG
answer, which only the hash can catch. Three properties are not negotiable:

* `comptime if is_defined[...]`, so no production bit can move.
* guarded by `rank > 0`, so the arm is **INERT AT ONE DEVICE**. That guard is
  what makes a moved cell attributable to the define rather than to the second
  device, and it is the control that makes the whole thing evidence.
* never a shape, a buffer size or anything a binding validates first, so the
  lane COMPLETES and its hash differs, instead of refusing.

And on the lane side: `tools/identity_break.py`'s 44 `par-*` lanes no longer
hold their sharded driver to the plain call with a bare `_same_bytes`, which
RAISES. Under any of these arms the two sides legitimately differ, so the
equality would fire and the cell would read REFUSED, and
`verification_matrix.negative_control_moves` does not count a REFUSED cell --
a correctly-working negative control would credit nothing. Every one now uses
`_mismatch_bytes` / `_oracle_mismatch` and `raise NumericalMismatch(msg,
parts)`, which `_run_reference` catches and hashes.

## The seventeen build defines

| define | module(s) | lanes it is for |
|---|---|---|
| `MOJOLEARN_GBDT_PARALLEL_SABOTAGE` | `gbdt/methods/pointwise_multi_gpu.mojo`, `gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo` | `par-boosting`, `par-boosting-clf`, `par-boosting-reg`, `par-boosting-pointwise`, `par-border-types`, `par-feature-freq`, `par-ordered`, `par-ordered-rmse` |
| `MOJOLEARN_GRAM_PARALLEL_SABOTAGE` | `core/gram_multi_gpu.mojo`, `core/gram_splitk.mojo` | `par-gram`, `par-gram-ols`, `par-gram-pca`, `par-gram-tsvd` |
| `MOJOLEARN_QR_PARALLEL_SABOTAGE` | `core/householder_qr.mojo` | `par-gram-pca`, `par-gram-tsvd` (the tall-QR arm) |
| `MOJOLEARN_SVM_PARALLEL_SABOTAGE` | `svm/impl/distance/kernel_matrices.mojo` | `par-svm`, `par-svm-svr`, `par-kernel-ridge`, `par-nystroem` |
| `MOJOLEARN_SOLVER_PARALLEL_SABOTAGE` | `solver/multi_gpu.mojo` | `par-cd`, `par-cd-elasticnet` |
| `MOJOLEARN_NEIGHBORS_PARALLEL_SABOTAGE` | `neighbors/impl/multi_gpu.mojo` | `par-graph-agglomerative`, `par-graph-spectral`, `par-graph-umap` |
| `MOJOLEARN_KMEANS_PARALLEL_SABOTAGE` | `cluster/multi_gpu.mojo` | `par-kmeans` |
| `MOJOLEARN_GLM_PARALLEL_SABOTAGE` | `glm/impl/qn/multi_gpu.mojo` | `par-logistic` |
| `MOJOLEARN_DBSCAN_PARALLEL_SABOTAGE` | `dbscan/impl/multi_gpu.mojo` | `par-dbscan` |
| `MOJOLEARN_GP_PARALLEL_SABOTAGE` | `gaussian_process/checks/kernels.mojo` | `par-gp` |
| `MOJOLEARN_ISOLATION_FOREST_PARALLEL_SABOTAGE` | `isolation_forest/impl/isolation_forest.mojo` | `par-iforest` |
| `MOJOLEARN_FOREST_POOL_PARALLEL_SABOTAGE` | `core/forest_inference_pool.mojo` | `par-forest-pool` |
| `MOJOLEARN_BYTE_LM_PARALLEL_SABOTAGE` | `training/byte_lm_parallel.mojo` | `par-byte-lm`, `par-byte-lm-model-pool`, `par-byte-lm-offload` |
| `MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE` (existed) | `cholesky/multi_gpu.mojo`, `cholesky/checks/trsm.mojo` | `par-cholesky`, `par-kernel-ridge` |
| `MOJOLEARN_GMM_PARALLEL_SABOTAGE` (existed) | `mixture/multi_gpu.mojo` | `par-gmm` |
| `MOJOLEARN_HIERARCHY_PARALLEL_SABOTAGE` (existed) | `hierarchy/impl/cluster/detail/multi_gpu.mojo` | `par-hdbscan` |
| `MOJOLEARN_RESAMPLE_PARALLEL_SABOTAGE` (existed) | `resample/estimator.mojo` | `par-resample` |

`MOJOLEARN_ISOLATION_FOREST_PARALLEL_SABOTAGE` is the one that is not a row
offset. `isolation_forest`'s `shard.first` is a curand SUBSEQUENCE base and
nothing else -- every buffer is indexed by the LOCAL tree id -- so the arm
shifts the RNG stream. The shard's trees are then entirely different rather
than slightly shifted; expect a large delta, not a small one.

## Which binding carries which define

Computed with `tools/bincache.py`'s own import-closure resolver, every script
at `scope="closure"` with no whole-tree widening, so these lists are exact and
not a guess. A define is passed through `MOJOLEARN_BUILD_EXTRA_DEFINES`, which
every one of these scripts appends verbatim to its `mojo build` line.

| build script | relevant `MOJOLEARN_*_PARALLEL_SABOTAGE` defines |
|---|---|
| `bindings/build.sh` | KMEANS, GRAM, NEIGHBORS |
| `bindings/build_estimators.sh` | GRAM, QR, GLM, DBSCAN |
| `bindings/build_gbdt.sh` | GBDT |
| `bindings/build_gp.sh` | GRAM, GP, CHOLESKY |
| `bindings/build_hdbscan.sh` | GRAM, NEIGHBORS, HIERARCHY |
| `bindings/build_ivf.sh` | KMEANS, GRAM |
| `bindings/build_kernel_methods.sh` | GRAM, SVM, CHOLESKY |
| `bindings/build_linalg.sh` | GRAM, QR |
| `bindings/build_metrics.sh` | KMEANS, GRAM, NEIGHBORS |
| `bindings/build_mixture.sh` | KMEANS, GRAM, CHOLESKY, GMM |
| `bindings/build_solver.sh` | GRAM, SOLVER, HIERARCHY |
| `bindings/build_svm.sh` | GRAM, SVM, ISOLATION_FOREST |
| `bindings/build_rf.sh`, `bindings/build_trees.sh` | FOREST_POOL |
| `bindings/build_byte_lm.sh` | BYTE_LM |
| `bindings/build_resample.sh` | RESAMPLE |
| `build_arima.sh`, `build_embedding.sh`, `build_mamba.sh`, `build_preprocessing.sh`, `build_training.sh`, `build_transformer.sh`, `build_tsa.sh` | none |

`MOJOLEARN_GRAM_PARALLEL_SABOTAGE` has the widest blast radius: eleven of the
twenty-three scripts reach `core/gram_multi_gpu.mojo` and `core/gram_splitk.mojo`
together. A leg that wants one lane's arm in isolation should build ONE script
with ONE define; a leg that wants the whole set in one pass should build all
sixteen scripts with every define at once and accept that a moved cell names
the SET rather than the arm. Both are useful and they answer different
questions; `lane/broken-par-sabotage-arms` showed why, when a five-family
build and a one-family build landed on different sabotaged hashes for the same
lane doing the same thing.

`tools/bincache.py::is_sabotage` matches `SABOTAGE` in the build args and
environment and gives the key `variant="sabotage"`, so none of these builds can
read or poison the production cache.

## THE COMMAND a two-device column should run

One clean column and one sabotage column, on the same box, the same commit and
the same fixture. `--repeats 2` is not optional: `stable_digest()` refuses a
part with fewer than two repeats, so a sabotage column at one repeat has its
move silently discarded.

```sh
# 0. the box: two CUDA or HIP devices. Apple is excluded structurally
#    (`_parallel_pool.DevicePool._start` admits metal only at group == (0,)).
export ARCH=...            # sm_90a, gfx942, ...
export VENDOR=nvidia       # or amd
LANES=$(python3 - <<'PY'
import importlib.util, sys
s = importlib.util.spec_from_file_location('ib', 'tools/identity_break.py')
m = importlib.util.module_from_spec(s); sys.modules['ib'] = m; s.loader.exec_module(m)
print(','.join(sorted(n for n in m.LANES if n.startswith('par-'))))
PY
)

# 1. CLEAN bindings, and the clean two-device column.
for s in build build_estimators build_gbdt build_gp build_hdbscan build_ivf \
         build_kernel_methods build_linalg build_metrics build_mixture \
         build_solver build_svm build_rf build_trees build_byte_lm build_resample \
         build_arima build_embedding build_mamba build_preprocessing \
         build_training build_transformer build_tsa; do
    MOJOLEARN_GPU_ARCHS=$ARCH MOJOLEARN_TARGET_COLUMN=$VENDOR \
        sh bindings/$s.sh
done
MOJOLEARN_PAR_DEVICES=0,1 MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py \
    --lanes "$LANES" --fixtures base --repeats 2 \
    --json par-2dev.clean.json

# 2. SABOTAGE bindings, every arm at once, into a separate directory, and the
#    sabotage two-device column against the same lanes.
DEFINES="-D MOJOLEARN_GBDT_PARALLEL_SABOTAGE=1 \
 -D MOJOLEARN_GRAM_PARALLEL_SABOTAGE=1 \
 -D MOJOLEARN_QR_PARALLEL_SABOTAGE=1 \
 -D MOJOLEARN_SVM_PARALLEL_SABOTAGE=1 \
 -D MOJOLEARN_SOLVER_PARALLEL_SABOTAGE=1 \
 -D MOJOLEARN_NEIGHBORS_PARALLEL_SABOTAGE=1 \
 -D MOJOLEARN_KMEANS_PARALLEL_SABOTAGE=1 \
 -D MOJOLEARN_GLM_PARALLEL_SABOTAGE=1 \
 -D MOJOLEARN_DBSCAN_PARALLEL_SABOTAGE=1 \
 -D MOJOLEARN_GP_PARALLEL_SABOTAGE=1 \
 -D MOJOLEARN_ISOLATION_FOREST_PARALLEL_SABOTAGE=1 \
 -D MOJOLEARN_FOREST_POOL_PARALLEL_SABOTAGE=1 \
 -D MOJOLEARN_BYTE_LM_PARALLEL_SABOTAGE=1 \
 -D MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE=1 \
 -D MOJOLEARN_GMM_PARALLEL_SABOTAGE=1 \
 -D MOJOLEARN_HIERARCHY_PARALLEL_SABOTAGE=1 \
 -D MOJOLEARN_RESAMPLE_PARALLEL_SABOTAGE=1"
for s in build build_estimators build_gbdt build_gp build_hdbscan build_ivf \
         build_kernel_methods build_linalg build_metrics build_mixture \
         build_solver build_svm build_rf build_trees build_byte_lm build_resample; do
    MOJOLEARN_GPU_ARCHS=$ARCH MOJOLEARN_TARGET_COLUMN=$VENDOR \
        MOJOLEARN_BUILD_EXTRA_DEFINES="$DEFINES" sh bindings/$s.sh
done
MOJOLEARN_PAR_DEVICES=0,1 MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py \
    --lanes "$LANES" --fixtures base --repeats 2 \
    --json par-2dev.sabotage-all-arms.json

# 3. THE ANSWER. Every arm's lanes must read DIVERGENT; the parts are compared
#    as LISTS, never as strings, and the reader self-checks first.
python3 bench/results/identity_break/2026-09-20_par-sabotage-defines/compare_columns.py \
    par-2dev.clean.json par-2dev.sabotage-all-arms.json

# 4. THE CONTROL THAT MAKES IT EVIDENCE. The same sabotage bindings at ONE
#    device must reproduce the clean one-device column exactly: every arm is
#    guarded by `rank > 0`, so a cell that moves here means an arm is NOT
#    inert at one device and the two-device move above is not attributable.
MOJOLEARN_PAR_DEVICES=0 MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py \
    --lanes "$LANES" --fixtures base --repeats 2 \
    --json par-1dev.sabotage-all-arms.json
python3 bench/results/identity_break/2026-09-20_par-sabotage-defines/compare_columns.py \
    par-1dev.clean.json par-1dev.sabotage-all-arms.json --changed-only   # must be 0
```

Step 4 is the step this whole file exists for. Without it a moved two-device
cell says only "these bindings differ"; with it, it says "the arm fired, and
only above rank 0".

## The Python-side switch, for the drivers with no binding to rebuild

A COOPERATIVE driver hands the whole fit to one worker and splits inside the
GPU binding, which is what the seventeen defines above reach. A
NON-cooperative driver's partition and merge are the driver's OWN PYTHON --
the column ranges of `parallel_preprocessing`, the global tree-ID ranges of
`parallel_ensemble.fit_forest`, the series ranges of `parallel_classical`, the
query rows of `parallel_neighbors`. No define reaches that code, so until now
the only negative control those lanes had was a HOST ARM, which perturbs the
shard's arithmetic and the plain call's arithmetic EQUALLY and therefore says
nothing about the partition.

`MOJOLEARN_PAR_DRIVER_SABOTAGE=1` with `MOJOLEARN_HOST_ALLOW_SABOTAGE=1`
(`python/mojolearn/_parallel_pool.py::driver_read_shift`) makes every shard
after the first read one position early while the merge still writes at the
true offset. Two variables, for `MOJOLEARN_FOLD_ORDER_SABOTAGE`'s reason: a
switch that quietly returns wrong answers on one env var is a footgun, and no
build script, workflow or gate sets either one. It is inert at one shard, so
it is the same `rank > 0` control as the build arms, and it needs no rebuild,
so it runs on a CPU-only install.

A column produced under it stamps `par_driver_sabotage: true`, read back from
the shipped module rather than from the environment.
`_verify_reference.admit()` refuses such a column (its generic `*_sabotage`
rule), which is correct.

WHAT IT IS NOT. It is not a build. `verification_matrix.py::sabotage_signals`
classifies any truthy non-harness `*_sabotage` flag as kind `build`, so a
column carrying this flag would be read as `seen(build)` by the matrix, and
that would overstate it: nothing was compiled. Until the matrix grows a name
for a product-code arm, **a `par_driver_sabotage` column must not be committed
under `bench/results/`,** where the matrix scans it. Its hashes are recorded
as text in
`bench/results/identity_break/2026-09-20_par-sabotage-defines/README.md`
instead.
