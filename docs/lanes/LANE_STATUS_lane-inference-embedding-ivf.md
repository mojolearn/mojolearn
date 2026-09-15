# Lane status: lane/inference-embedding-ivf-cholesky

Andrew, Sep 15 2026: inference is where mojolearn will be used most. Goal:
public, bitwise-identical CPU inference loaded from GPU-built state for
Cholesky solve, embedding lookup and IVF search, then `IVFIndex.extend` on GPU
and CPU. No GPU box rented; Mac work one core.

## Stage 0: Cholesky solve on the CPU (this commit)

- `bindings/_mojolearn_linalg_host.mojo` exports the gp binding's three door
  names over `cholesky/host/chol_oracle.mojo`. linalg ships; gp does not.
- `python/mojolearn/_cholesky_impl.py`: CPU-only installs bind
  `_mojolearn_linalg`; `fit` is public on a CPU (`_CPU_FIT_IS_INFERENCE`,
  honored in `_cpu_reference.require_training`); `save`/`load`
  (`mojolearn-cholesky-1`); `HostCholesky`, returned by `host_model`.
- Manifest: the `cholesky` lane and `Cholesky` class moved from gp to linalg;
  `cholesky` is a public reference probe.
- Evidence: `bench/results/identity_break/2026-09-15_cholesky-cpu-inference/`
  (train, infer, batch IDENTICAL x4 = 9 each; model OWED x1 = 9; sabotage moves
  9 of 9 train and 9 of 9 owed).
- Tests: `python/mojolearn/tests/test_cholesky_cpu_inference.py`.

## Stage 1: embedding lookup and IVF search from saved GPU state (done)

- The IVF build and search are two binding calls (`ivf_flat_build`,
  `ivf_flat_search`) on the GPU and reference host bindings, with the contract
  in `bindings/ivf_index_arrays.mojo` and the arrays admitted by
  `ivf_validate_index_arrays`.
- `IVFIndex` and `Embedding` save and load. New shipped families
  `ivf_search` and `embedding_infer` serve the routes. `SEARCH_LOOKUP_RECORDED`
  names the Metal recording.
- Evidence: `bench/results/identity_break/2026-09-15_ivf-embedding-cpu-inference/README.md`.
  - train, infer and batch are IDENTICAL on 36 cells each, against the three
    GPU columns plus new Metal and CPU columns.
  - The IVF model cells are OWED x2 (18); the embedding model cells are n/a.
  - Sabotage moved 35 of 36 train cells and 18 of 18 owed cells.
  - The saved-model gate matched 27 fixtures with only the shipped bindings,
    matched them again on an installed wheel, and saw the sabotage mismatch.

## Stage 1 design as written before the work

Gate: `git merge-base --is-ancestor origin/lane/cpu-training-embedding-ivf
origin/main`. That branch adds the embedding and ivf host families as
training-only reference builds (`ships_in_wheel=False`).

Embedding lookup.
- Save format `mojolearn-embedding-1`: the weight `<f4` (V, d),
  `padding_idx` (-1 for none), `max_norm` refused at save unless None (a
  renorm on lookup writes the table), plan, tier.
- `Embedding.save`/`Embedding.load`; `forward(ids)` on a CPU-only install
  through the embedding host binding's gather. `backward` stays reference only
  (training), refused outside `reference_training()`.
- Ship: flip the embedding family to `ships_in_wheel=True`, or, if its binding
  also carries the backward, keep the binding and let `require_training` refuse
  the backward call by name. The wheel list change is one family.
- Identity: the `embedding` lane's probe gathers 512 held-out ids; add
  `save`/`load` so the model cell is hashed (OWED on the GPU columns).

IVF search on a built index. Today `IVFIndex.search` builds and searches in
one device call (policy 3, one identity card) and nothing is retained, so there
is no GPU-built index to save. Required:
- Mojo: `ivf_flat_build_host` already returns `IvfFlatIndex` (centers,
  center_norms, CSR offsets, ascending original ids, list data). Add binding
  entries `ivf_flat_build` (writes the index arrays to caller buffers, two
  calls: sizes, then fill) and `ivf_flat_search_index` (reads them back into
  an `IvfFlatIndex` and runs `ivf_flat_search_traced`) on the GPU binding and
  the host binding with identical contracts. The one-card build-and-search
  entry stays for the trace gates.
- Python: `IVFIndex.fit` builds (on a GPU) and keeps the arrays; `search`
  reads them; `save`/`load` as `mojolearn-ivf-flat-1` (all arrays with exact
  dtypes, n_lists, metric, dim, n_rows, tier, and the build parameters for the
  record). `fit` stays a GPU operation; on a CPU-only install `fit` refuses,
  `load` then `search` is public.
- Identity: the `ivf` and `ivf-euclidean` lanes gain a model cell (OWED); the
  train cell must keep its recorded hashes, which is the proof that splitting
  build from search changed no bit (the Metal column rerun on the M4 shows it).
- Ship: the ivf host binding's search half ships. The quantizer k-means stays
  in the binding (needed for extend's assignment, stage 2) but its build entry
  refuses outside the reference context.
- Installed-wheel check: build the inference wheel's host set, pip install
  into an isolated target, load a GPU-saved index and embedding table from
  fixtures committed under `bench/results/`, search and look up, compare bytes.

## Stage 2: `IVFIndex.extend` (done)

- The GPU path is `ivf_flat_extend` (the build's `predict`); the CPU path is
  `host_ivf_extend` (`host_assign`, plus a sabotage arm that sends every new
  row to the next list). Both use `extend_list_layout`, and the new ids are
  `n_rows, n_rows + 1, ...`.
- `ivf_flat_extend` is on the GPU binding, the reference ivf binding and the
  shipped ivf_search binding. There is a new lane `ivf-extend` with a batch
  declaration, and a new classical gate lane with its Metal recording.
- Evidence: `bench/results/identity_break/2026-09-15_ivf-extend/README.md`.
  - Metal and CPU agree on all 9 ivf-extend cells (OWED x2).
  - ivf and ivf-euclidean stay IDENTICAL to the records.
  - Host sabotage moves 54 of 54 owed parts; batch sabotage moves 9 of 9.
  - The saved GPU-extended index matched on the shipped binding and on an
    installed wheel.

## Stage 2 design as written before the work

Reference: cuVS `ivf_flat::extend` (`ivf_flat_build.cuh:180-345`). It predicts
labels for the new vectors with `kmeans::predict` against the FIXED centers
(when `adaptive_centers` is false), adds a histogram of the new labels to the
list sizes, resizes each list and inserts each vector with its caller-given id.

Behavior here:
- `extend(X_new, ids=None)`: ids default to `n_rows, n_rows + 1, ...`; given
  ids must be unique and not already in the index (refused by name).
- Assignment: the build's own assignment kernel and tie rule against the fixed
  centers (the `ivf.assign` stage: the pinned distance tile, the minimum with
  the lower list index winning an exact tie), so an extended row lands where a
  build with those centers would put it.
- Layout: merge into the CSR layout keeping ids ASCENDING within each list
  (DEVIATIONS 1783/1784), not appended in arrival order, so the index after
  extend is a function of the set of rows, not of insertion order or batch
  cuts. That is the determinism the reference does not promise.
- `center_norms` unchanged (centers fixed). `adaptive_centers` stays refused
  by name. Remove the extend row from `ivf/NOT_IMPLEMENTED.tsv`.
- GPU and CPU: the assignment runs through the same device kernel on the GPU
  and its host restatement in `ivf/host/ivf_host.mojo`; the CSR merge is host
  integer work shared by both.
- Identity: a new lane `ivf-extend` (build on 3072 rows, extend by 1024 in two
  batches, search 64 queries; train hashes the index arrays and the search)
  with a batch declaration that splits the extension rows (extend in one call
  versus several must give the same index bytes). Evidence: Metal and CPU
  agree, sabotage moves, batch sabotage moves the new batch cells; GPU columns
  OWED to the release record.

## Owed

- Stage 0 model cells on Apple, NVIDIA and AMD: the next release record.
- Stage 1 IVF model cells on Apple, NVIDIA and AMD: the next release record.
- The CPU identity gate workflow builds neither shipped inference family and
  does not check `SEARCH_LOOKUP_RECORDED`. That is owed to the workflow's
  owner; this lane does not edit workflows.
- Stage 2 ivf-extend cells on Apple (a record column), NVIDIA and AMD: the next release record.
- An intermittent Metal failure seen once on the M4 (2026-09-15 about 13:45 ET)
  needs its own lane.
  - What was seen: one identity_break process over ivf, ivf-euclidean and
    ivf-extend read correct cells on the first five `ivf` fixtures. It then read
    zero distances, `merge_probed_lists: list 0 appears at probe 0 and probe 1`
    and batch cells that moved, on every later cell. Another agent's Metal GBDT
    run was live on the same GPU.
  - What it was not: a rerun of the `ivf` lane alone, with that GBDT run still
    live, read 9 of 9 cells STABLE and IDENTICAL to the committed Apple, NVIDIA
    and AMD columns. No GPU-path Mojo source changed on main between the clean
    stage 1 runs and that failure.
  - The suspects are device state accumulating across many `DeviceContext`s in
    one long process, or device contention. The stage 2 Metal columns were
    therefore taken one process per lane. The failure itself is not diagnosed
    here.
