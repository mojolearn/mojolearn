# FAST Apple rows slower than the best opponent (lane apple-fast-gap-optim, 2026-10-03)

Board: docs/apple-fast/BOARD_M3_FAST.md on main 18dc09a7a. Read from the board driver down to the kernels;
no local timing. The M3 A/Bs below confirm or refute each hypothesis.

## What each arm times

| row | ours (timed) | opponent (timed) |
|---|---|---|
| adagrad, rmsprop, adamax, nadam | `cls([p], ...)` then 10 x `opt.step([g])`, p and g NumPy (16,777,216 floats = 64 MB); tools/bench_board_algos.py:3592-3597 | torch.optim on MPS, p and the 10 grads already on the device (`g_dev`, input_home=device), one sync at the end; :3610-3617 |
| adafactor | the same 10 steps, one 1-D tensor: Adafactor keeps a FULL variance (n floats, 64 MB) | the same, device resident |
| layernorm | `layer_norm_forward(x)` + `layer_norm_backward(dy, x)`, x (16384, 1024) = 64 MB; :3424-3430 | F.layer_norm fwd + bwd on device tensors (bf16 eager best) |
| lr-exponential | 100,000 Python calls `sched.lr_at(k)`; tools/bench_board_extra.py:421 | torch ExponentialLR, 100,000 `get_last_lr()` + `step()` on the CPU |

Torch moves no host data inside its timed region; ours must take NumPy in and give NumPy back (the API is host
memory only), so every 64 MB array crosses the boundary each call.

## Time breakdown hypotheses (per step / call)

Costs on Apple (memory metal-transfer-costs-on-apple, M4; the M3 is similar in kind): raw or staged upload of
64 MB ~2-4 ms; DMA to a pinned stage ~2-3 ms; the single-thread READ of a pinned (write-combined) stage ~13-25 ms
per 64 MB; first touch of fresh host pages ~20 ms per 64 MB; one element-wise launch over 64 MB ~1 ms.

**Optimizers (19.5-20.5 ms a step; torch 1.4-3.3).** sequence/opt_resident.mojo `opt_resident_step_py` (:368):
- param + grad up: `_upload_all` (:289) -> DeviceExec.upload (sequence/exec_device.mojo:440), PIPE_UP chunked
  memcpy into a pinned stage + DMA: 2 x 64 MB, ~4-6 ms.
- one launch `opt_step` (:415), ~1 ms. State is already resident (ba3bfbdab), no state traffic.
- param down: `_pipe_download` (:312): 8 chunks of 8 MB, each a DMA into a pinned half, a `ctx.synchronize()`
  (:341) and a single-thread memcpy OUT OF write-combined memory (the read of chunk i-1 overlaps the DMA of
  chunk i). The read is the largest single cost, ~12-14 ms.
- the first chunk's DMA and the last chunk's read are never overlapped; 8 synchronizes a step.
Floor with one host thread and NumPy in/out: the 64 MB read-back. Matching torch (1.4 ms) needs either the
parameter resident on the device across steps (an API change: the caller would no longer see p updated in
place) or the multi-thread host copy 1774263e0 removed. Both are Andrew's decisions (ledger optspeed line).

**Adafactor (39 ms a step; torch 34.4).** sequence/pyapi.mojo `adafactor_step_py` (:519):
- 4 x `ex.alloc(n)` zero fills of 64 MB (P, G, S1, U; :538-542) of which 3 are overwritten by uploads at once.
- 3 x 64 MB up (P, G, the full variance; :546-548), 2 x 64 MB down (P, variance; :626-627). The variance's round
  trip (one up, one down) is ~40% of the step; 0.8.34 ran 184 ms with the same five transfers on 8 host threads.
- kernels: 2 sum-of-squares passes, AF_VEC, AF_DENOM, AF_APPLY, ~2-3 ms.
Fix: keep the variance on the device across steps, like ba3bfbdab did for the other optimizers (two of the five
64 MB transfers go away).

**LayerNorm (52.2 ms fwd+bwd; torch bf16 2.8).** sequence/pyapi.mojo `layer_norm_py` (:833) called twice:
- forward: x up (:848), Y written, y down (:920) INTO A FRESH `np.zeros` array (python/mojolearn/_x_sequence_norm.py:35):
  first touch page faults + WC read ~25-30 ms.
- backward: x up AGAIN (:848), forward recompute for mean/rstd (writes Y, never read), dy up (:872), dx down
  (:912) into a fresh `np.zeros` (:39): again faults + WC read.
- `ex.alloc` zero fills X, Y, DY, DX (64 MB each) of which X and DY are overwritten by uploads.
The two 64 MB downloads into fresh pages are most of the 52 ms.

**lr-exponential (225 ms / 100,000 = 2.25 us a call; torch-cpu 0.77 us).** Pure host Python by contract
(DEVIATION 5540: the exact rational value rounded once to float32). Per call python/mojolearn/_x_sequence_sched.py:
`lr_at` (:253) -> `_t` -> `_pow_value` (:207) -> `_GammaPow.at` (:140, a loop of one `_trim`) -> `_iv_f32` (:84)
-> `_dy_f32` (:60) twice -> `ldexp` twice: ~9 Python calls, interpreter bound. No GPU work belongs here (a
schedule is a scalar per step; the opponent is CPU torch).

## Candidates (default OFF; FAST + Apple comptime guard in Mojo, env var in Python)

| define / env | file | rows | what |
|---|---|---|---|
| `-D MOJOLEARN_OPT_FAST_MAP_DOWN` | sequence/opt_resident.mojo `_download_all` | optimizers | the param down through `DeviceBuffer.map_to_host` + one memcpy (tests whether the runtime's mapped read beats the pinned WC read) |
| `-D MOJOLEARN_OPT_FAST_RAW_DOWN` | sequence/opt_resident.mojo `_download_all` | optimizers | every chunk DMAd straight into the NumPy param (no stage, no host read), one synchronize |
| `-D MOJOLEARN_OPT_FAST_PIPE_CH=524288` | sequence/opt_resident.mojo `OPT_PIPE_CH` | optimizers | 2 MB pipeline chunks: the unhidden first DMA / last read shrink 4x (more synchronizes) |
| `-D MOJOLEARN_AF_FAST_RESIDENT` | sequence/opt_resident.mojo `adafactor_resident_*`, pyapi.mojo `adafactor_core`, python `Adafactor` | adafactor | the variance (row/col or full) lives on the device across steps; a step moves P, G up and P down only |
| `-D MOJOLEARN_AF_FAST_NOFILL` | sequence/pyapi.mojo `adafactor_step_py` | adafactor | P, G, S1, S2 bound by upload (`ex.bind`), no zero fill first |
| `-D MOJOLEARN_SEQ_FAST_MAP_DOWN` | sequence/exec_device.mojo `_pipe_down` | layernorm, adafactor | as OPT_FAST_MAP_DOWN for every pipelined x_sequence download |
| `-D MOJOLEARN_SEQ_FAST_RAW_DOWN` | sequence/exec_device.mojo `_pipe_down` | layernorm, adafactor | as OPT_FAST_RAW_DOWN |
| `-D MOJOLEARN_SEQ_FAST_PIPE_CH=524288` | sequence/exec_device.mojo `SEQ_PIPE_CH` | layernorm, adafactor | 2 MB chunks |
| `-D MOJOLEARN_LN_FAST_NOFILL` | sequence/pyapi.mojo `layer_norm_py` | layernorm | X and DY bound by upload, no zero fill |
| `MOJOLEARN_SCHED_FAST_INLINE=1` | python/mojolearn/_x_sequence_sched.py `ExponentialLR.lr_at` | lr-exponential | the forward walk (t = last + 1, base_lr > 0, gamma > 0) in one function: one enclosure product, both ends rounded inline; anything else falls to the existing path. Same bits by construction (the enclosure decides or the exact fallback does) |
| `MOJOLEARN_SCHED_FAST_P64=1` | same | lr-exponential | the enclosure at 64 bits instead of 128 (narrower ints; a wider interval only sends more steps to the exact fallback, never other bits) |

All Mojo candidates are copies or skipped fills: the same launches on the same values, no bit moves. Quality
(the board's column: max relative difference vs torch eager) cannot change.
