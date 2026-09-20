# Thirteen `par-*` arms written, one watched, the rest owed a two-device box

`lane/par-sabotage-defines`, 2026-09-20. Apple M4, CPU host route only, one
core, `nice -n 19`, base fixture, `--repeats 2`, no GPU, no rental, no Metal
job. Every column here was produced by this branch's `tools/identity_break.py`
against the 32 host bindings built from this tree.

## The hole this lane was opened on

`docs/lanes/SABOTAGE_AUDIT_2026-09-16.md` put 28 `par-*` lanes in category (c):
"no host family and no host sabotage define reaches the lane at all". Four
`*_PARALLEL_SABOTAGE` defines existed (Cholesky, GMM, hierarchy, resample) and
had been watched failing the NATIVE multi-GPU checks under `training/checks/`,
never an `identity_break` cell. `lane/broken-par-sabotage-arms` closed eight
lanes on a CPU column earlier today. **Thirty-five `par-*` lanes were left with
no arm that could move a cell**, which is the largest single gap in the
verification story, because a sabotage nobody has watched fail is
indistinguishable from no sabotage.

## What is here

1. **Thirteen new build defines**, one per cooperative partition that no host
   binding restates, plus the four that already existed re-aimed at the lane
   cells for the first time. `docs/multi_gpu/PAR_SABOTAGE_ARMS.md` is the
   manifest: define, module, the build script that compiles it (computed with
   `tools/bincache.py`'s own import-closure resolver), the exact two-device
   command, and the one-device control that makes a moved cell attributable.
2. **Forty-four lane oracles rewritten.** Without this every one of the
   thirteen defines would have credited nothing even when working perfectly.
3. **`MOJOLEARN_PAR_DRIVER_SABOTAGE`**, the arm for the NON-cooperative
   drivers, whose partition is the driver's own Python and which therefore has
   no binding to rebuild.
4. **`par-forest-reg` closed**, on a real build sabotage, on this box.

## 1. WRITTEN IS NOT WATCHED

Thirty-four of the thirty-five lanes CANNOT be watched on this machine and
this file does not pretend otherwise. They refuse on a CPU column with the
reason the driver states itself -- "no CPU implementation of the cooperative
multi-GPU driver `<op>` yet: its shards are device row tiles, chunks or ranges
inside the GPU binding, which no host binding restates" -- or, for the three
byte-LM lanes, because the GPU binding is absent. The command that would watch
all of them in one pass is in `docs/multi_gpu/PAR_SABOTAGE_ARMS.md`, including
the step that matters most: the SAME sabotage bindings at ONE device must
reproduce the clean one-device column exactly, because every arm is guarded by
`rank > 0`.

## 2. The oracles, and why they had to move first

Each of these lanes held its sharded driver byte for byte to the plain call
through `_same_bytes`, which RAISES. Under any of these arms the two sides are
legitimately different computations, so the equality fires, the lane raises
before it can hash anything, and the cell reads REFUSED.
`verification_matrix.negative_control_moves` does not count a REFUSED cell, so
a control that was working perfectly would credit nothing. That is
`par-scaler`'s defect (2026-09-17) and the eight lanes of
`lane/broken-par-sabotage-arms` (2026-09-20).

`audit_bare_oracles.py` (committed in that lane's directory) reads
`tools/identity_break.py` as an AST:

| `tools/identity_break.py` | lanes with a bare body oracle | of them `par-*` |
|---|---|---|
| at the branch point (`origin/main` 36b77f669) | 49 | 44 |
| this branch | 5 | **0** |

The five left are `byte-lm-resident`, `embedding`, `embedding-sort`,
`ivf-extend` and `transformer-decode-session`, which are not `par-*` and are
out of this lane's scope.

## 3. The 34 lanes that are owed a two-device box, and the reason each gives

Every one of these ran REFUSED on the clean CPU column of this directory, and
the reason is the driver's own sentence, not a guess. The refusal count per
cooperative operation, read out of `cpu-apple-m4.par-sweep.clean.json`:

| refusing operation | lanes | the define now written for it |
|---|---|---|
| `gbdt_fit` | 7 (`par-boosting`, `-clf`, `-reg`, `-pointwise`, `par-border-types`, `par-feature-freq`, `par-ordered`) | `MOJOLEARN_GBDT_PARALLEL_SABOTAGE` |
| `ordered_rmse_fit` | 1 (`par-ordered-rmse`) | `MOJOLEARN_GBDT_PARALLEL_SABOTAGE` (same two modules; `_parallel_worker` routes both operations through `launch_feature_shards`) |
| `gram_fit` | 4 (`par-gram`, `-ols`, `-pca`, `-tsvd`) | `MOJOLEARN_GRAM_PARALLEL_SABOTAGE`, and `MOJOLEARN_QR_PARALLEL_SABOTAGE` for the tall-QR arm the PCA and TSVD paths take |
| `graph_fit` | 3 (`par-graph-agglomerative`, `-spectral`, `-umap`) | `MOJOLEARN_NEIGHBORS_PARALLEL_SABOTAGE` |
| `svm_fit` | 2 (`par-svm`, `par-svm-svr`) | `MOJOLEARN_SVM_PARALLEL_SABOTAGE` |
| `km_fit` | 2 (`par-kernel-ridge`, `par-nystroem`) | `MOJOLEARN_SVM_PARALLEL_SABOTAGE` (`kernel_methods/checks/kernel_matrix.mojo` reads `MOJOLEARN_SVM_DEVICE_COUNT`), and `MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE` for KernelRidge's factor |
| `solver_fit` | 2 (`par-cd`, `par-cd-elasticnet`) | `MOJOLEARN_SOLVER_PARALLEL_SABOTAGE` |
| `kmeans_fit` | 1 (`par-kmeans`) | `MOJOLEARN_KMEANS_PARALLEL_SABOTAGE` |
| `glm_fit` | 1 (`par-logistic`) | `MOJOLEARN_GLM_PARALLEL_SABOTAGE` |
| `dbscan_fit` | 1 (`par-dbscan`) | `MOJOLEARN_DBSCAN_PARALLEL_SABOTAGE` |
| `gp_fit` | 1 (`par-gp`) | `MOJOLEARN_GP_PARALLEL_SABOTAGE` |
| `iforest_fit` | 1 (`par-iforest`) | `MOJOLEARN_ISOLATION_FOREST_PARALLEL_SABOTAGE` |
| `forest_prepare` | 1 (`par-forest-pool`) | `MOJOLEARN_FOREST_POOL_PARALLEL_SABOTAGE` |
| `gmm_fit` | 1 (`par-gmm`) | `MOJOLEARN_GMM_PARALLEL_SABOTAGE` (existed; never aimed at a lane cell before) |
| `cholesky_fit` | 1 (`par-cholesky`) | `MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE` (existed) |
| `hdbscan_fit` | 1 (`par-hdbscan`) | `MOJOLEARN_HIERARCHY_PARALLEL_SABOTAGE` (existed) |
| `resample` | 1 (`par-resample`) | `MOJOLEARN_RESAMPLE_PARALLEL_SABOTAGE` (existed) |
| a missing GPU binding, not a driver | 3 (`par-byte-lm`, `-model-pool`, `-offload`) | `MOJOLEARN_BYTE_LM_PARALLEL_SABOTAGE`; those three are owed a `bindings/build_byte_lm.sh` build as well as a box |

34 lanes, 17 cooperative operations plus the byte-LM binding. The set of
refusing lanes and the set of lanes this branch writes an arm for are the
SAME 34: checked, not assumed.

## 4. The clean column did not move

Two questions, both answered by running rather than by reading. Every one of
the 59 `par-*` cells, `--repeats 2`, base fixture, the same 32 host bindings.

| pair | verdict changes | cells whose hash LIST moved | DIVERGENT parts |
|---|---|---|---|
| `origin/main`'s `tools/identity_break.py` against this branch's, same drivers (`cpu-apple-m4.par-sweep.clean.before-fix.json` against `cpu-apple-m4.par-sweep.clean.json`) | **0** | **0** | **0** |
| this branch's drivers before and after the `driver_read_shift` call sites were added, switch OFF | **0** | **0** | **0** |

The 44 rewritten oracles and the driver call sites are invisible to a clean
build, which is what a negative control has to be. `compare_columns.py`
self-checks before it prints an answer: it must classify a moved part, a moved
cell hash, a changed verdict, a clean pair and a string-valued (malformed)
column correctly, and it compares part LISTS, never strings.

## 5. WATCHED: the non-cooperative drivers' arm, on two worker processes

`MOJOLEARN_PAR_DRIVER_SABOTAGE=1 MOJOLEARN_HOST_ALLOW_SABOTAGE=1
MOJOLEARN_PAR_DEVICES=0,1`, against the same two-process CLEAN column. It is
NOT a device claim: on a CPU-only install a "device" is one worker PROCESS.
What it checks is the drivers' own partition and merge, which is exactly what
no host arm can reach.

| lane | cell clean | cell sabotaged | DIVERGENT parts |
|---|---|---|---|
| `par-arima` | `377790cb117abb61` | `846d2243aafeb725` | 4 of 4 |
| `par-forest` | `21178dc5457bf5dc` | `29ef11dd6241b5d2` | 2 of 2 |
| `par-forest-et` | `754d8c127ecfc04d` | `71beaa9bd09b56d1` | 1 of 1 |
| `par-forest-et-clf` | `c586b27a3b049614` | `a58069239c08a71a` | 2 of 2 |
| `par-forest-reg` | `f032017cfd1c73dd` | `db77aec34184b112` | 1 of 1 |
| `par-holtwinters` | `aa71e2e60daefca2` | `d2c6a46722ba4c28` | 9 of 10 (`criterion` unmoved) |
| `par-queries-kde` | `e0d6e3d0623d6112` | `7193aff5cd39cecb` | 1 of 1 |
| `par-queries-knn` | `4b0dd36744b84c37` | `13628600629edf29` | 4 of 4 |
| `par-queries-nn` | `6869ae45e01fcb7a` | `b02dc2bed36bd980` | 2 of 2 |
| `par-queries-radius` | `b4fa4ebf9623f5d0` | `99816c21f7710425` | 3 of 4 (`radius` unmoved) |
| `par-rbf-sampler` | `7ddfa0d92256da95` | `9256620ac37d5d2a` | 1 of 3 (`weights`, `offset` unmoved) |
| `par-reference-knn` | `4b0dd36744b84c37` | `7c721ecff7f5034f` | 3 of 4 |
| `par-reference-knn-reg` | `33e7da5c3fdd343c` | `99caaf9e1b3df5ac` | 1 of 1 |
| `par-scaler` | `c4cc661617a4b9a7` | `f2484b6918d5f0b5` | 4 of 4 |
| `par-scaler-minmax` | `646b46cf73ead58e` | `e1ffe0544cd0b233` | 6 of 6 |

15 cells, 15 verdict changes STABLE -> DIVERGENT, **44 DIVERGENT parts**. Both
repeats agree on every value, so `stable_digest` accepts them; at
`--repeats 1` it would refuse every one and the move would be silently
discarded, which is why every column here is `--repeats 2`. Not one cell reads
REFUSED: that is the 44 rewritten oracles doing their job, and before them
every one of these would have been a crash that credited nothing.

`par-mlp`, `par-samba` and `par-samba-clip` read REFUSED on BOTH two-process
columns with the CPU route's own by-name sentence ("no CPU implementation of
the cooperative multi-GPU driver `mlp_update`/`samba_update` across 2 devices
yet"). That is a declared limit of that route, not a defect, and it is why the
table has 15 rows and not 18.

### THE CONTROL that makes those 44 parts evidence

The same switch, the same build, the same lanes, at ONE device
(`cpu-apple-m4.par-sweep.sabotage-par-driver.1dev.json` against
`cpu-apple-m4.par-sweep.clean.json`):

| | verdict changes | cells moved | DIVERGENT parts |
|---|---|---|---|
| switch ON at one device | **0** | **0** | **0** |
| switch ON at two processes | 15 | 15 | 44 |

Without this row the 44 parts say only "these two runs differ". With it they
say the arm fired, and only above rank 0.

THIS ROW WAS RED FIRST, and that is the finding worth keeping. The switch was
gated on the shard INDEX, and it moved the SAME 15 cells at one device as at
two, because a Python-sharded driver cuts as many logical shards as its
`*_per_shard` argument asks for whatever the device count: `fit_forest` with
`trees_per_shard=4` makes four shards on ONE device. `DevicePool.map`
dispatches in waves of `len(devices)` and hands wave position `j` to worker
`j`, so the gate is `index % len(devices) > 0`, and
`test_cpu_training_par_classical` now holds it there.

## 6. WATCHED, AND A REAL BUILD: `par-forest-reg`

`par-forest-reg` was the one of the 35 that RUNS on a CPU column. It read
`declared` in the matrix -- the `rf` family's define reached it and nobody had
watched it move -- and it was one of the six lanes
`lane/broken-par-sabotage-arms` recorded as INERT under the `core` arm.

Three families built with their own defines into their own directory, the
other 29 the clean binary the clean column loaded:
`rf` and `trees` with `-D MOJOLEARN_HOST_SABOTAGE=1`, `forest` with
`-D MOJOLEARN_FOREST_HOST_SABOTAGE=1`. The column witnesses its own binary:
`host.families[*].sabotage` reads true for `_mojolearn_rf_host`,
`_mojolearn_trees_host` and `_mojolearn_forest_host`, and false for the other
29.

| lane | cell clean | cell sabotaged | DIVERGENT parts |
|---|---|---|---|
| **`par-forest-reg`** | `f032017cfd1c73dd` | `51c47ab53a694b17` | **1 of 1** (`predict`) |
| `par-forest` | `21178dc5457bf5dc` | `ed0bcfcef5b35833` | 2 of 2 |
| `par-forest-et` | `754d8c127ecfc04d` | `ea112e2954936a35` | 1 of 1 |
| `par-forest-et-clf` | `c586b27a3b049614` | `d630c867d817d8fd` | 2 of 2 |

Both sides STABLE on both repeats, so `stable_digest` accepts them and
`negative_control_moves` credits the move. `tools/verification_matrix.py`
reads it:

| | `par-*` lanes at `seen(build)` | `declared` | `none` |
|---|---|---|---|
| before this column | 24 | 1 (`par-forest-reg`) | 34 |
| after | **25** | **0** | 34 |

`par-forest-reg`: `seen(build)`, parts `batch, infer, model, train`, from
`cpu-apple-m4.sabotage-rf-trees-forest.par-sweep.json` against
`cpu-apple-m4.par-sweep.clean.json`, same commit.

## The honest count

| | lanes |
|---|---|
| `par-*` lanes with no arm that could move a cell, at the branch point | **35** |
| arms written on this branch | **35** |
| WATCHED locally, moving a real sabotage BUILD | **1** (`par-forest-reg`) |
| written, needs a two-device CUDA or HIP box | **34** |
| not written | **0** |

Thirty-four of thirty-five are a claim about code nobody has watched fail. The
command that would settle them is in `docs/multi_gpu/PAR_SABOTAGE_ARMS.md`,
one build pass and two columns, and it includes the one-device control without
which a moved cell proves only that two binaries differ.

Separately, 15 `par-*` lanes were watched moving under the Python-side
`MOJOLEARN_PAR_DRIVER_SABOTAGE` arm on two worker processes, with a clean
0-cell one-device control. That is a partition control, not a build control,
and section 5 says what it may and may not be counted as.

## What this does not say

Nothing here is a two-device DEVICE claim: on a CPU-only install each "device"
is one worker process. Nothing here says the thirteen new defines compile on a
GPU target -- this Mac built no GPU binding and none of the edited modules is
reachable from a host binding, so they are unexercised by every column in this
directory. Nothing here says an arm that fires is a GOOD arm: an arm that moves
a lane may still leave the interesting path alone.

## Files

* `cpu-apple-m4.par-sweep.clean.json` -- all 59 `par-*` lanes, base,
  `--repeats 2`, one device, 32 production host bindings. 25 ran, 34 refused.
* `cpu-apple-m4.par-sweep.clean.before-fix.json` -- the same, with
  `origin/main`'s `tools/identity_break.py`. The pair is the proof that the 44
  rewritten oracles moved nothing.
* `cpu-apple-m4.sabotage-rf-trees-forest.par-sweep.json` -- the same 59 lanes
  with `rf`, `trees` and `forest` built with their sabotage defines.
* `compare_columns.py` -- the reader. It self-checks on a moved part, a moved
  cell hash, a changed verdict, a clean pair and a malformed column before it
  prints anything, and it compares part LISTS.

The two-process and driver-sabotage columns are deliberately NOT committed
here; `docs/multi_gpu/PAR_SABOTAGE_ARMS.md` says why, and their hashes are in
section 5 above.
