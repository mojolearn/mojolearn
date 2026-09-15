# DBSCAN and AgglomerativeClustering predict: infer and batch cells (2026-09-15)

Branch `lane/inference-transductive-predict`. DEVIATION 2740 is new
capability. Neither cuML nor scikit-learn labels new rows under these two
estimators.

**The rule.** One nearest-labeled-reference pass does both. On the device it
is `core/labeled_reference_predict.mojo`; on the host it is
`core/labeled_reference_host_predict.mojo`.

- **Distance.** This is the DBSCAN eps predicate's own accumulator:
  `_eps_acc`, imported unchanged, with the reference row flushed and features
  taken in ascending order.
- **DBSCAN.** A new row gets the label of the nearest core sample within eps,
  under the fit's metric. Ties go to the lowest (distance, core index). With no
  core sample within eps, the row is noise.
- **AgglomerativeClustering** (single linkage). A new row gets the cluster of
  its nearest training row. Ties go to the lowest cluster label, then the
  lowest index.

Both are opt-in through `prediction_data=True`. The default fit is unchanged.

## Where each column ran

- **Metal column.** Apple M4, commit ef1647619, one core in a shared Mac slot
  (nice 19, one-thread knobs, `-j 1`). The base, estimators and solver
  identical bindings were built from that tree.
  - Nothing in the GPU binding closure changed between ef1647619 and the merge
    bd9ef2eee (`git diff --name-only` over `bindings/_mojolearn*.mojo`,
    `dbscan`, `hierarchy`, `core`, `checks`, `neighbors`, `kde`,
    `decomposition`, `glm`, `gemm` and `cluster` lists 0 files). The column
    therefore stands for bd9ef2eee.
- **CPU column.** One RunPod CPU pod (`tools/runpod_cpu_leg.sh`, 8 vCPU, AMD
  EPYC 9654), commit bd9ef2eee, 179 s billed. The pod was deleted and verified
  gone (`runpod_cpu_leg.log`).
  - The core, estimators and solver host bindings were built on the pod.
  - The sabotage estimators host build used
    `-D MOJOLEARN_TRANSDUCTIVE_PREDICT_SABOTAGE=1`. It XORs 1 into every
    clustered predicted label and touches no fit. The production solver and
    core bindings sat beside it.

**Fixtures and lanes.**
- Fixtures: base, ties, dupes and denormal, with two repeats (one for the
  sabotage runs).
- Lanes: dbscan, dbscan-brute-l1, dbscan-weighted, agglomerative, par-dbscan
  and par-graph-agglomerative. The two par lanes run on Metal only, with one
  device.
- The lanes read `X[:6000, :4]`, which the dupes rewrite does not reach, so
  dupes hashes like base.

**Committed record.** The committed columns are the 166-lane record's
apple-m4, nvidia-h100-sm_90a and amd-mi325x-gfx942. They were trimmed in
scratch to these lanes and fixtures, keeping the fixture and held-out bytes
of those four fixtures.

## Results

| file | verdict |
|---|---|
| `apple-m4.json` | cells=24 stable=24; infer, model and batch stable=24 |
| `cpu-x86.json` | cells=16 stable=16; infer, model and batch stable=16 |
| `diff.metal-vs-record-train.txt` | train IDENTICAL=24 against all three committed GPU columns, so `prediction_data=True` moved no committed fit hash. The new infer, model and batch parts read ONE-COLUMN |
| `diff.cpu-vs-record-owed.txt`, `owed.json` | `--require-columns 4 --owed-json` exits 0: train IDENTICAL=16 x4, infer and model OWED=32, batch OWED=16 (48 owed parts) |
| `diff.metal-vs-cpu.txt` | Metal against CPU: train IDENTICAL=16, infer and model IDENTICAL=32, batch IDENTICAL=16 |
| `diff.cpu-vs-predict-sabotage.txt` | the predict-only host sabotage: train IDENTICAL=16, infer DIVERGENT=16, model IDENTICAL=16 (the saved file holds fit state, not predictions), batch DIVERGENT=16 |
| `diff.metal-vs-batch-sabotage.txt` | `MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1` on Metal: batch BATCH_MOVED=6 of 6 |
| `diff.cpu-vs-batch-sabotage.txt` | the same on the CPU column: BATCH_MOVED=4 of 4 |
| `test_transductive_predict.metal.txt` | GREEN, 43 checks, on Metal |
| `heldout_and_training_rows.txt` | how many held-out rows are labeled, and how training rows compare with their fitted labels |

**The test module's checks** include:
- a query exactly at eps (inside) and one ulp past it (noise);
- an exact tie between two clusters (lowest core index for DBSCAN, lowest
  label for agglomerative);
- duplicated rows;
- the L1 arm at eps and past it;
- uniform weights;
- a fit with no core sample;
- float64 queries, rows alone and reversed order;
- save and load;
- every refusal by name.

Its HOST arm ran against a pre-merge estimators host build. On those fixture
queries the CPU host binding returned the GPU bytes for both estimators. The
full CPU comparison is the pod column above.

The agglomerative model part read DIVERGENT Metal against CPU at 21c244b51,
because the saved file then carried `children_` and `n_boruvka_rounds_`. The
host binding runs Kruskal (rounds -1, its own merge orientation). The file
now holds the prediction state only, and the part reads IDENTICAL.

## Training rows (self-consistency), measured on Metal

- **DBSCAN, every lane and fixture.** Every core row and every noise row
  predicts its fitted label.
  - Core rows are guaranteed on both algorithms.
  - Noise rows are guaranteed on `algorithm='brute'`; the default `'rbc'` also
    matched here.
- **DBSCAN border rows** are not promised. All training rows agree at 1.0000
  except dbscan-brute-l1, at 0.9993 on base and dupes and 0.9995 on denormal.
- **Agglomerative.** Every training row agrees (1.0000). The rule guarantees
  it for every row without a distance-0 twin cut into a lower-labeled cluster.

## Held-out rows, honestly

The held-out predictions are thin on some fixtures:

- On base, the DBSCAN fits find one cluster, so labels are {-1, 0}, with 4 of
  256 rows noise (43 on brute L1).
- On denormal, every held-out row gets label 0.
- On base, the agglomerative fit (four clusters, single-linkage chaining)
  labels every held-out row 0.
- The ties fixture carries the discriminating cells. DBSCAN labels 131 of 256
  rows noise and spreads the rest over many cluster ids; agglomerative uses
  labels {0, 2}.

## Owed

- The NVIDIA and AMD infer, model and batch cells for the six lanes
  (`owed.json` lists the 48 CPU-column parts), at the next release record.
- The two-device par-dbscan and par-graph-agglomerative cells.
- The seven-runner CPU identity gate on main.

No GPU box was rented.
