# Public CPU inference for saved IVF-Flat indexes and embedding tables

lane/inference-embedding-ivf-cholesky, 2026-09-15, stage 1.

- `IVFIndex.fit` builds the index through a new `ivf_flat_build` entry and
  `search` answers from the five index arrays through `ivf_flat_search`. This
  applies on the GPU binding and on the reference host binding, with one
  contract in `bindings/ivf_index_arrays.mojo`. Before this, the index never
  left the one-call build-and-search entry.
- `IVFIndex` and `Embedding` gain `save` and `load` (formats
  `mojolearn-ivf-flat-1` and `mojolearn-embedding-1`).
- Two new host families ship in the inference wheel and carry no training
  code:
  - `ivf_search` (`_mojolearn_ivf_search_host`: `ivf_flat_search` only).
  - `embedding_infer` (`_mojolearn_embedding_infer_host`: `embedding_forward`
    only).
- Each serves its route on a CPU-only install when the reference binding is
  not built (`host_surface.inference_routes`). `mojolearn.host_model` returns
  `HostIVFIndex` or `HostEmbedding` for a saved file.

## Columns

The GPU columns are the committed records:

- `2026-09-14_ivf-euclidean` (Apple M4, NVIDIA H100, AMD MI300X) for `ivf`
  and `ivf-euclidean`.
- `2026-09-15_embedding-sort` (Apple M4, NVIDIA H100, AMD MI325X) for
  `embedding` and `embedding-sort`.

The two new columns:

- `apple-m4.json`: the M4 Metal set rebuilt from this lane's tree in a shared
  Mac slot, two repeats.
- `cpu-x86.json`: one RunPod CPU pod (8 vCPU, `runpod/base:1.3.1-ubuntu2204`),
  host bindings built there through the binding cache, two repeats. The pod
  was deleted and verified gone (220 s billed). The lane commit was fa0e8ccb2;
  the three uncommitted tracked files it shipped were regenerated doc spans.

## Result (`diff.records-metal-cpu.txt`, `--require-columns 4 --owed-json`, exit 0)

| part | verdict |
|---|---|
| train, 36 cells (4 lanes x 9 fixtures) | IDENTICAL on all 36 across the three GPU columns, Metal and CPU |
| infer, 36 cells | IDENTICAL on all 36 |
| batch, 36 cells | IDENTICAL on all 36 |
| model, ivf and ivf-euclidean, 18 cells | OWED x2 (Metal and CPU agree; no GPU record saved an index) |
| model, embedding and embedding-sort, 18 cells | N/A (`n/a:input-table`) |

The train, infer and batch cells matching the records is the proof that
splitting build from search changed no bit, on Metal and on the CPU.

The embedding model cell hashes the saved file, and that file holds only the
table the lane passed in, so no sabotage can move it (the first leg's owed
check failed on exactly those 18 cells). The lanes now declare it n/a. The
saved-table lookup is gated below.

## Sabotage (`-D MOJOLEARN_HOST_SABOTAGE=1`, `diff.records-metal-cpu-sabotage.txt`, exit 1 as required)

- train: DIVERGENT on 35 of 36 cells. `ivf/ties` is an exact integer fold, as
  on the branch that added the host path.
- infer and model: DIVERGENT on 53, IDENTICAL on 1, N/A on 18.
- batch: DIVERGENT on 35 of 36.
- owed check (`owed-sabotage.txt`): 18 of 18 owed IVF model cells MOVED, verdict OK.
- The embedding forward gained a sabotage arm (it gathers the next row), so
  the lookup is seen to fail.

## Saved models from the GPU, answered on the CPU (`cpu-x86-leg/`)

`bench/results/classical_host/2026-09-15-apple-m4-ivf-embedding` was recorded
on the M4 Metal set by `tools/classical_host_gate.py record`. It holds 27
fixtures across `ivf`, `ivf-euclidean` and `embedding`, and each saved model
reloaded to the same bytes on Metal.

| check | host bindings | verdict |
|---|---|---|
| `check-wheel-families.json` | only the two shipped bindings | IDENTICAL, 27 fixtures; identity hashes EQUAL to every GPU column carrying the lane |
| `check-wheel-families.sabotage.json` | the two sabotage builds | EXPECTED MISMATCH SEEN |
| `check-installed-wheel.json` | a wheel built from this tree holding only the two bindings, pip-installed into a fresh venv, run from `/tmp` with `PYTHONPATH` unset | IDENTICAL, 27 fixtures; every ivf and embedding binary loaded from the venv (`wheel-check-binaries.log`) |

On the installed wheel, `IVFIndex.fit` refused with the CPU training refusal
(`wheel-fit-refusal.log`).

The symbol scan (`training_symbols.txt`) finds no `host_fit_main`,
`host_ivf_build` or `host_embedding_backward` in the shipped bindings, and
finds them in the reference bindings. pytest on the pod: 168 passed, 3
skipped, and the 2 record-presence tests were deselected because the pod
ships no records. The sabotage set passed 11 of 11.

The installed-wheel check is not the release packaging: the wheel is a
source `pip wheel` of this tree on the pod, not `packaging/linux/pack_wheel.py`
with the staged runtime closure.
