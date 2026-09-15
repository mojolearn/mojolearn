# IVFIndex.extend on the GPU and the CPU

lane/inference-embedding-ivf-cholesky, 2026-09-15, stage 2.

## What it does

`IVFIndex.extend(X)` follows cuVS `ivf_flat::extend` with
`adaptive_centers = false` as its reference.

- **Assignment.** Each new row goes to the built index's FIXED centres through
  the build's own assignment:
  - on the GPU, `ivf_flat_build.mojo::ivf_flat_extend` (the build's `predict`
    launch);
  - on the CPU, `ivf/host/ivf_host.mojo::host_ivf_extend` (`host_assign`).
  The tie rule is the same (the lower list id wins an exact tie).
- **Ids and layout.** `ivf/checks/list_layout.mojo::extend_list_layout`
  appends each row to its list under the ids `n_rows, n_rows + 1, ...`. The
  extended index is therefore the layout a build would give the concatenated
  rows under the same labels. One call and several calls over the same rows,
  in the same order, give the same bytes.
- **Refused.** Caller-chosen ids and `adaptive_centers` are not implemented
  (`ivf/NOT_IMPLEMENTED.tsv`).
- **Bindings.** `ivf_flat_extend` is registered on the GPU binding, the
  reference ivf host binding and the shipped `_mojolearn_ivf_search_host`.
  So a GPU-built index, saved and loaded on a CPU, is extended there.

## The lane

`ivf-extend` (tools/identity_break.py):

- It builds the `ivf` lane's index on rows 0..3072 and extends it by rows
  3072..4096 in one call, and from a clone of the same built index in two
  calls split at 3584.
- The five index arrays and the new rows' lists must be the same bytes, or
  the train cell reads REFUSED naming the array. Then 64 queries run with 4
  probes.
- Batch declaration: each of 64 held-out rows extended alone, and in pieces,
  must name the list it names inside the whole batch.

## Columns

| column | where |
|---|---|
| `apple-m4.json` | the M4 Metal set rebuilt from this lane's tree in a shared Mac slot. One process per lane, joined with `identity_break --merge`; see the lane status for the intermittent Metal failure that forced that |
| `cpu-x86.json` | one RunPod CPU pod (8 vCPU), host bindings built there, commit 2ce48b333. The pod was deleted and verified gone (387 s billed) |
| GPU records | `2026-09-14_ivf-euclidean` (Apple M4, NVIDIA H100, AMD MI300X) for `ivf` and `ivf-euclidean`. No record carries `ivf-extend` |

## Result (`diff.records-metal-cpu.txt`, `--require-columns 4 --owed-json`, exit 0)

| part | ivf, ivf-euclidean (18 cells) | ivf-extend (9 cells) |
|---|---|---|
| train | IDENTICAL x5 | OWED x2 (Metal and CPU agree) |
| infer | IDENTICAL | OWED x2 |
| model | OWED x2 (the saved index, stage 1) | OWED x2 |
| batch | IDENTICAL | OWED x2 |

## Negative controls

- **Host sabotage** (`-D MOJOLEARN_HOST_SABOTAGE=1`,
  `diff.records-metal-cpu-sabotage.txt`, exit 1 as required):
  - Train reads DIVERGENT on 26 of 27 cells, including all 9 `ivf-extend`
    cells. `ivf/ties` is an exact fold.
  - The owed check (`owed-sabotage.txt`) moved 54 of 54 owed cell parts,
    verdict OK.
  - The first stage 2 leg (commit c8337a534) read that check FAIL on 10
    cells: the nine `ivf-extend` batch cells and `ivf-extend/ties` train. The
    define never reached extend's assignment. `host_ivf_extend` gained a
    sabotage arm (every new row to the next list) before this column.
- **Batch sabotage** (`MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1`): BATCH_MOVED on
  9 of 9 `ivf-extend` cells on the CPU (`cpu-x86.batch-sabotage.json`) and on
  Metal (`apple-m4.batch-sabotage.json`).

## A GPU-extended saved index, answered on the CPU (`cpu-x86-leg/`)

`bench/results/classical_host/2026-09-15-apple-m4-ivf-extend` records nine
fixtures on the M4 Metal set. Each is an index built and extended on the GPU
and saved, with these extras:

- the loaded index extended again by 64 held-out rows;
- the new rows' lists and carried ids;
- a search after that.

| check | host bindings | verdict |
|---|---|---|
| `check-search-binding.json` | only `_mojolearn_ivf_search_host`, also over the stage 1 ivf and ivf-euclidean recordings | IDENTICAL, 9 fixtures of ivf-extend plus 18 of the stage 1 IVF lanes; identity hashes EQUAL to the three GPU columns where they carry the lane |
| `check-search-binding.sabotage.json` | its sabotage build | EXPECTED MISMATCH SEEN |
| `check-installed-wheel.json` | a wheel holding only that binding, pip-installed into a fresh venv, `PYTHONPATH` unset | IDENTICAL, 9 fixtures; the binary loaded from the venv |

pytest on the pod: 166 passed, 1 skipped, and 2 record-presence tests
deselected.
