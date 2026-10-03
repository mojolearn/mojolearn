# FAST regressions vs 0.8.34 (lane apple-fast-regress, 2026-10-03)

Board: docs/apple-fast/BOARD_M3_FAST.md refresh 8 (main daaceede0) against the 0.8.34 FAST cells
(alpha-api-0.8.34-main-20261001). Found by reading the code path each board lane calls; the M3 A/Bs
below confirm or refute.

| row | 0.8.34 | main | culprit | where |
|---|---|---|---|---|
| layernorm synthetic | 26.8 | 76.8 | 1774263e0 (cpu-gpu-cleanup n-seq): `_pcopy` went from an 8-way split to one memcpy on the calling thread | sequence/exec_device.mojo `_pcopy` (every DeviceExec upload, download, sync) |
| adafactor synthetic | 184 | 683 | 1774263e0, same: a step moves five 64 MB arrays (P, G, the variance up; P, the variance down), each a serial copy plus a serial DMA. The kernels are unchanged under FAST (cb9749dff / b8584c9aa touch IDENTICAL's norms only) | sequence/exec_device.mojo `_pcopy`; sequence/pyapi.mojo `adafactor_step_py` uploads/downloads |
| optimizers (rmsprop, adagrad, adamax, nadam, lion) | 129-178 | 195-213 | 1774263e0 again (ledger 2026-10-03 optspeed finding); OPT_PIPE_DOWN/ZERO_OPEN (5990c5946) recovered the download half, the uploads are still `ex.upload` -> one memcpy | sequence/opt_resident.mojo `_upload_all` -> DeviceExec.upload |
| dynamic-optimized-theta taxi-hourly | 727 | 2087 | 95a09d1fd: SEQ_FAST_FMA (fused fma3) + THETA_REG made the FAST + Apple default. THETA_REG keeps the same operations, so the bits move only through the fused fmas; the synthetic row got faster (485 -> 377), so the likely mechanism is Nelder-Mead iterating longer on taxi-hourly (no fixed point, cycling toward the 1,000 cap). The FMA_OFF arm confirms it | sequence/ops.mojo `fma3` (SEQ_FAST_FMA); sequence/theta.mojo `op_theta` nelder_mead call (no cycle watch) |

## Fixes (FAST + Apple only, default OFF, for the M3 A/B; copies only or exact, no bit moves)

- `-D MOJOLEARN_SEQ_FAST_PIPE_UP` (sequence/exec_device.mojo `upload`): an upload of >= 2 chunks (8 MB) copies chunk i into its
  stage and queues its DMA at once, so the DMA overlaps the copy of chunk i + 1. Reaches layernorm, adafactor and the
  resident optimizers' parameter and gradient uploads.
- `-D MOJOLEARN_SEQ_FAST_PIPE_DOWN` (sequence/exec_device.mojo `download_async` / `sync` / `_pipe_down`): downloads of >= 2
  chunks are deferred to the sync, which (after the queue drains) streams them through two pinned 8 MB halves: the DMA of
  chunk i overlaps the read of chunk i - 1. opt_resident's OPT_PIPE_DOWN (318 -> 191 ms) for every x_sequence download.
- `-D MOJOLEARN_SEQ_FAST_THETA_SNAP` (sequence/theta.mojo `THETA_SNAP`, `op_theta`): the theta fits take the Nelder-Mead cycle
  watch GARCH already uses (sequence/nm.mojo `snap`): a state that returns bit for bit to an earlier one stops after the
  iterations left of its last lap, with the same final state, best vertex and iteration count. Snapshot = 16 floats of the 64
  the row reserves for Nelder-Mead.

No host threads come back (the cleanup's rule); no CPU route.

## Not done here (next if the arms do not reach 0.8.34)

- Adafactor's variance resident on the device (as ba3bfbdab did for the other optimizers): two of its five 64 MB transfers
  per step go away. Needs a handle kind with R/C-sized slots and the Python state mirror.
