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
full-fixture command exit 4.

CLOSED in lease 7 (`nvidia-2x-rtx4090/lease7-byte-lm/`), and it was the witness, not the drivers.
The three byte-LM trainers hand their device list to the native binding, which opens one device
context per device inside the calling process (`training/byte_lm_parallel.mojo`,
`training/byte_lm_layer_pool.mojo`). They never start a `DevicePool`, and the witness watched only
`DevicePool._start`, so it counted zero pools on any install. `par-byte-lm-offload` is a third
case: `OffloadedByteLanguageModelTrainer` admits exactly one device and its lane passes
`_par_devices()[:1]`, so both of its columns run on the first device by design.

Lease 7, pod `iqo1phi7lxsks1`, GPU 0 `GPU-9e2cff5f-3fc0-a820-884d-495cb605b5a9` (PCI 01:00.0), GPU 1
`GPU-6c39fe9b-5877-2f4e-e654-15b0c13ac6da` (PCI C2:00.0). The same 0.8.10 wheel (sha256
`c8a09722c3a3fcbe91905d2d15b8489d960b008bb9776afa15b2400e115e778c`) in an activated venv, with ONE
file, `mojolearn/_verify_par.py`, replaced by the copy at commit `10dd372a7`; `overlay.diff` is the
whole difference and `overlay_sha256.txt` names both files.

| command | exit | seconds | result |
|---|---|---|---|
| `pytest test_verify_par.py` (CPU only, on the box) | 0 | under 1 | 40 passed |
| `verify --par --par-self-test` | 0 | 9 | SELF-TEST PASSED |
| `verify --par --par-self-test --json` | 0 | 8 | SELF-TEST PASSED |
| `verify --par --lanes par-byte-lm,par-byte-lm-model-pool,par-byte-lm-offload --fixtures base` | 0 | 6 | VERIFIED: 3 lanes, 15 parts, 2 compared, IDENTICAL 2, DIVERGENT 0, ONE-COLUMN 0, REFUSED 0, N/A 13, no witness refusal |

`par-byte-lm` opened one native session on devices 0,1 with 489,216 resident bytes on each;
`par-byte-lm-model-pool` opened one with 718,592 and 259,840. Both name the two GPU UUIDs above.
`par-byte-lm-offload` opened its driver on device 0 only; its `train` part, which would have read
IDENTICAL, reads N/A with the reason and is not counted as a match. 2 of 3 cells show two distinct
physical GPUs; the third is the one-device lane.

Lease 1 measured nothing: the body called `/usr/bin/python3` instead of the venv's interpreter, so
the verifier read CANNOT RUN (`No module named 'mojolearn'`). Fixed from lease 2 on.

Slowest lanes on one fixture (hashed, seconds): par-resample 302, par-graph-umap 81,
par-forecast-holtwinters 35, par-forecast-arima 31, par-border-types 31, par-ivf 27,
par-queries-knn 24, par-gmm 21, par-cholesky 18, par-reference-knn-reg 16. One fixture is about
950 s; `par-resample` is a third of it.

## AMD, 2x MI300X (`amd-2x-mi300x/`)

Leases 1 and 2 measured nothing. Lease 1: the body refused a correct install on a stderr warning.
Lease 2 was torn down by the runner's exit trap before anything was pulled.

Lease 3 (`amd-2x-mi300x/lease3-quick/`), pod `7g6tbuqw8853tb`, the plain 0.8.10 wheel (same sha256 as
above, no overlay) in an activated venv, `mojolearn.__file__` inside the venv, vendor `hip`,
`require_device_count(2): ok`. GPU 0 unique id `0x83cd823ce61dd094` (PCI 0000:85:00.0), GPU 1
`0x484960c35d6be285` (PCI 0000:A6:00.0), both AMD Instinct MI300X.

| command | exit | seconds | result |
|---|---|---|---|
| wheel download, install, environment report, interpreter checks | 0 | 6 in all | both devices listed by rocm-smi and rocminfo |
| `verify --par --par-self-test` | 124 | 900 | TIMED OUT, no lane line printed |
| `verify --par --par-self-test --json` | 124 | 900 | TIMED OUT, no document |
| `verify --par quick` | 124 | 583 | TIMED OUT at the body's budget, no document |
| `verify --par --lanes <the nine lanes> --fixtures base` | NOT STARTED | 0 | no budget left |

Lanes finished 0, parts compared 0, IDENTICAL 0, DIVERGENT 0, ONE-COLUMN 0, REFUSED 0, N/A 0. NO
AMD TWO-GPU RESULT EXISTS ON THIS WHEEL, and the nine lanes with no AMD two-device record
(par-border-types, par-causal-lm, par-cross-val, par-forecast-arima, par-forecast-holtwinters,
par-gpc-fit, par-gpc-predict, par-ivf, par-ordered) still have none.

`verify --par` HANGS on this box, in the first lane (`par-arima`), in all three commands.
`hang_observations.txt` has three read-only process samples: the one-device column had finished,
the two-device pool had started two workers, and the worker masked to device 0 sat in a kernel GPU
fence wait (`dma_fence_default_wait`) for at least eleven minutes at zero CPU with the GPU idle,
while device 0's VRAM was 99.7 % used and one process held 392 GB across both devices. The cause is
not established. `verify --par` runs both columns in one process, and the AMD two-device records on
main came from a leg that runs each column in its own process; that difference fits the samples and
is untested. Nothing points at the known MI300X SR-IOV stale read, which is wrong values, not a hang.

## Rule

`--par quick` plus the self test is the two-GPU check. `--par all` is not to be run again without
the maintainer's express permission (see `docs/VERIFY.md`).
