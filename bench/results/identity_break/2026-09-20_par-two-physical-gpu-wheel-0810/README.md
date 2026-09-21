# Two physical GPUs against one, on the published 0.8.10 wheel (2026-09-20)

`python -m mojolearn verify --par` run from `pip install mojolearn==0.8.10` in a fresh
venv outside any checkout. Each `par-*` lane runs twice in one process, once with
`MOJOLEARN_PAR_DEVICES=0` and once with `0,1`; the columns are compared cell for cell and
every cell must show two distinct physical GPUs by UUID.

## NVIDIA, 2x GeForce RTX 4090 (`nvidia-2x-rtx4090/`)

GPU 0 `GPU-d9c3091b-b8ce-1d3a-5afe-0ce6876ad537`, GPU 1 `GPU-5c4c7be6-cdfa-9c7e-d40b-74d56b47de4f`.

| command | result |
|---|---|
| `verify --par --par-self-test` | exit 0 (lease 2 and lease 6) |
| `verify --par quick` | VERIFIED: 14 lanes, base fixture, 51 compared parts IDENTICAL, 0 DIVERGENT, 19 N/A, 224 s; 14 of 14 cells showed two distinct GPU UUIDs |
| `verify --par all`, one fixture per command over leases 2 to 6 | 531 lane/fixture cells, 1,935 compared parts IDENTICAL, 0 DIVERGENT, 0 MOVED, 0 ONE-COLUMN |

`SUMMARY.txt` has the per-lane table and every command's exit code and seconds.

Witness refusals in the full sweep: 25 cells, all in three lanes, `par-byte-lm` (9 fixtures),
`par-byte-lm-model-pool` (8) and `par-byte-lm-offload` (8). The verifier's words: "the two-device
column started NO device pool at all, so it ran on one device and agreeing with the one-device
column proves nothing." Those cells are refused, not counted as matches, and they make each
full-fixture command exit 4. This is OPEN: the three byte-level language-model drivers did not
start a second device pool on this install.

Lease 1 measured nothing: the body called `/usr/bin/python3` instead of the venv's interpreter, so
the verifier read CANNOT RUN (`No module named 'mojolearn'`). Fixed from lease 2 on.

Slowest lanes on one fixture (hashed, seconds): par-resample 302, par-graph-umap 81,
par-forecast-holtwinters 35, par-forecast-arima 31, par-border-types 31, par-ivf 27,
par-queries-knn 24, par-gmm 21, par-cholesky 18, par-reference-knn-reg 16. One fixture is about
950 s; `par-resample` is a third of it.

## AMD, 2x MI300X (`amd-2x-mi300x/`)

Nothing measured. Lease 1: the body refused a correct install on a stderr warning. Lease 2 was
torn down by the runner's exit trap before anything was pulled.

## Rule

`--par quick` plus the self test is the two-GPU check. `--par all` is not to be run again without
the maintainer's express permission (see `docs/VERIFY.md`).
