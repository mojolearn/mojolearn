# SpectralClustering.predict: infer and batch cells (2026-09-15)

Branch `lane/spectral-predict`. DEVIATION 2860 is new capability. Neither cuML
nor scikit-learn labels new rows under a fitted SpectralClustering.

## The rule

The method is the Nystrom out-of-sample extension. The reference is Bengio,
Paiement, Vincent, Delalleau, Le Roux and Ouimet (NIPS 2003), section 3; Fowlkes
et al. (TPAMI 2004) use it for spectral grouping. The row is then assigned by
the fit's own k-means pass.

- **Stated in.** `spectral/host/spectral_predict_host.mojo` (CPU) and
  `spectral/impl/spectral_predict.mojo` (device).
- **Affinity of a new row.**
  - `nearest_neighbors`: its `n_neighbors` nearest training rows (the fit's
    k-NN), each at 0.5. That is the fit's `0.5 * (a + b)` for an edge with no
    reverse edge.
  - `precomputed`: the caller's `(n_new, n_train)` matrix.
- **Folds.** Ascending training index, seeded `+0.0`, through the identical
  primitives, one query at a time.
- **Ties.** A k-NN tie goes to the search's own (distance, index) order. A
  k-means tie goes to the lowest centroid index.
- **Threshold.** A used column with `|1 + theta| < 1e-3` is refused by name.
  Nothing is dropped, because the clustering fit keeps every column.
- **Opt-in.** `prediction_data=True` copies the eigenpairs, the degree scaling
  and the centroids out of the fit. The default fit is unchanged.

## Where each column ran

Three RunPod CPU pods, one at a time, each verified deleted
(`runpod_cpu_leg1.log`, `runpod_cpu_leg2.log`, `runpod_cpu_leg3.log`).

- **Leg 1** (7e072251b, 143 s billed). A setup run:
  - its merge refused mixed-fixture parts, and the record files were not
    shipped;
  - its GPU compile check found a partial move in `spectral_predict_binding`,
    fixed in 0a6957015.

  Its only kept outputs are the four saved models under `wheel/`.
- **Leg 2** (0a6957015, 219 s billed). Every shipped host family was built.
  - CPU columns for `spectral` and `spectral-precomputed` on nine fixtures,
    two repeats.
  - The record diff and the batch sabotage.
  - The training-row agreement (`agree/`).
  - The installed test wheel.
- **Leg 3** (33ac888dd, 227 s billed). The host sabotage column with the fixed
  arm, plus a production recheck on base and ties.

**Metal column** (Apple M4, `metal/`). Every Metal process ran through the
exclusive Metal slot, one short chunk at a time. Under the narrowed scope that
means `spectral` and `spectral-precomputed` on base and ties (two repeats),
plus a base-fixture spot check of `par-graph-spectral`.

- **The crash.** The first Metal binding (36355ac74) segfaulted on the first
  nearest_neighbors predict (`metal/probe.b2-segfault.txt`: a plain fit and a
  `prediction_data=True` fit succeeded). It had handed `knn_search` memory
  backed by Mojo `List`s.
- **The fix** (b886dbc97). The search is staged through
  `enqueue_create_host_buffer` buffers, as `create_connectivity_graph` and
  `umap/transform.mojo` stage theirs.
- **Scope of the fix.** The binding at b886dbc97 produced every Metal cell
  below. The fix touches only the device predict path. The CPU host columns,
  which never call `knn_search` on a device, stand at their commits.

## Results

| file | verdict |
|---|---|
| `cpu-x86.json` | cells=18 stable=18; infer, model and batch stable=18 |
| `diff.cpu-vs-record-owed.txt`, `owed.json` | `--require-columns 4 --owed-json` against the 166-lane apple-m4, nvidia-h100-sm_90a and amd-mi325x-gfx942 columns exits 0. Train IDENTICAL=18 (x4), so `prediction_data=True` moved no committed fit hash. The infer, model and batch parts are OWED=54 |
| `diff.cpu-vs-host-sabotage.txt` | Predict-only host sabotage (`-D MOJOLEARN_SPECTRAL_PREDICT_SABOTAGE=1`, embedding column 1 negated): train IDENTICAL=18, infer DIVERGENT=18, model IDENTICAL=18 (the saved file holds fit state), batch DIVERGENT=18 |
| `diff.cpu-vs-batch-sabotage.*.txt` | `MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1`, base: BATCH_MOVED on both lanes |
| `diff.leg2-vs-leg3-prod.txt` | Production CPU cells at 33ac888dd equal leg 2 at 0a6957015 on base and ties (IDENTICAL=4 train, 8 infer and model, 4 batch). The sabotage fix is in a comptime-false branch |
| `wheel/wheel_models.json` | Installed test wheel (15 shipped families, RECORD sha256 checked, extracted into `/tmp/target`, `MOJOLEARN_HOST_DIR` unset). Four saved models (both lanes, base and ties) through `mojolearn.host_model` and `SpectralClustering.load` give the saving process's label and embedding hashes: all_equal True |
| `metal/apple-m4.json` | Metal, base and ties: cells=4 stable=4; infer, model and batch stable=4 |
| `metal/diff.metal-vs-cpu.txt` | Metal against the x86 CPU column: train IDENTICAL=4, infer and model IDENTICAL=8, batch IDENTICAL=4 (the other seven fixtures are CPU only) |
| `metal/diff.metal-vs-record-train.txt` | Against the three 166-lane columns: `spectral` and `spectral-precomputed` train on base and ties IDENTICAL x4, so `prediction_data=True` moved no committed Metal fit hash. The REQUIRE FAIL lines name the seven fixtures this narrowed Metal column does not run |
| `metal/diff.par-graph-spectral-base-vs-record.txt` | `par-graph-spectral` base (the fit path this lane edited, no predict), train IDENTICAL x4 against the record |
| `metal/diff.metal-vs-batch-sabotage.*.txt` | `MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1` on Metal, base: BATCH_MOVED on both lanes |
| `metal/test_spectral_predict.metal.txt` | GREEN, 39 checks, on Metal. Its HOST arm is REPORT only here (no host dir given); the CPU host bytes are the pod columns |
| `sabotage-arm-check/` | Why the first sabotage arm was inert: column 0 is the trivial eigenvector, constant after the degree division. Negating it moves every query equally far from every centroid, and moved 0 of 4 label hashes; the column 1 arm moves 4 of 4 (M4 CPU host, the four saved models) |

## Training rows (self-consistency)

`predict(X_train) == labels_` is measured on the CPU reference fit (`agree/`).
Identical columns make the number the same on every device.

| fixture | spectral (2000 rows) | spectral-precomputed (1000 rows) |
|---|---|---|
| base | 0.9825 | 0.998 |
| ties | 0.9860 | 0.996 |
| hashed | 0.9820 | 0.999 |
| wide | 0.9930 | 0.997 |
| denormal | 0.9985 | 1.000 |
| denormal_ftz | 0.9985 | 1.000 |
| dupes | 0.9825 | 0.998 |
| odd | 0.9830 | 1.000 |
| negative | 0.9805 | 0.998 |

- **Totals.** 17773 of 18000 (0.9874) and 8986 of 9000 (0.9984).
- **Embedding gap.** The largest training-row `|extension - embedding_|` is
  1e-3 to 2e-3 (nearest_neighbors) and up to 1.1e-2 (precomputed).
- **Threshold margin.** The smallest `|1 + theta|` is 0.43, far from the
  threshold.

The test module's blob fixture reports:
- training-row agreement of 0.9917 (nearest_neighbors) and 1.0000
  (precomputed), and 1.0000 with every row duplicated;
- a training-row embedding gap of 2.0e-2.

## Owed

- The NVIDIA and AMD infer, model and batch cells of both lanes (`owed.json`,
  54 parts), at the next release record.
- The Metal infer, model and batch cells on the seven fixtures this narrowed
  column did not run (hashed, wide, denormal, denormal_ftz, dupes, odd,
  negative), and Metal train cells on those seven. Their CPU cells read
  IDENTICAL x4 on train against the record.
- The two-device `par-graph-spectral` cells.

No GPU box was rented.
