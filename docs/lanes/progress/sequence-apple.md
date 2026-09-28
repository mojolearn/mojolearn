# sequence-apple: progress (Apple Metal speed, 2026-09-28)

Branch `lane/sequence-apple` = origin/main (5b622763d) + origin/lane/algos-sequence
(merged: the family's step 0, the Adafactor Metal clamps, `sumsq_fold`, and the harness
`tools/sequence_speed.py`; that branch's NVIDIA gate is still owed, see
docs/lanes/progress/sequence.md). NOT merged to main: the gate runners merge it.

Home Mac for speed: m4-a (Apple M4, 10 GPU cores). Every timing below is one run of
`tools/sequence_speed.py` (HIGGS 1M rows from R2, staged to m4-a at
~/datasets/gbm-bench/higgs/higgs_speed.npz), each algorithm timed once after a small
warm-up, digest = sha256 of every output. Before and after on the same Mac.

## Profile (m4-a, IDENTICAL, before: 084fcaf5c, steward 1790580241118)

| algo | shape | fit s | infer s |
|---|---|---|---|
| lstm | 62500x16x28 h64 b512 1 epoch | 3.28 | 1.18 |
| gru | same | 2.67 | 0.90 |
| rnn | same | 1.51 | 0.34 |
| mlp | 1M x 28 (256,) | 1.73 | 0.44 |
| moe | 1M tokens | 0.54 | |
| layernorm | 1M x 28 fwd+bwd | 0.92 | |
| rmsprop, adagrad, lion, adamax, nadam | 4M params x 10 steps | 1.05 each | |
| lamb | same | 2.06 | |
| adafactor | 2000x2000 x 10 steps | 1.14 | |
| stl | 10000 series x 100 | 0.44 | |
| theta | same | 0.57 | |
| croston | same | 0.005 | |
| ets | same | 1.62 | |
| garch | same | 7.81 | |
| var | 4 x 1M | 0.93 | |

Where the time went:
- Optimizers: ~105 ms per step of 4M floats, nearly all host glue: `DeviceExec.upload`
  copied element by element into a staging buffer and synchronized after every copy
  (5 uploads + 4 downloads per step).
- Recurrent (LSTM/GRU/RNN): ~101 launches per batch (per time step: hidden GEMM, bias,
  cell; backward: cell, recurrent GEMM), 123 batches. RNN and LSTM fit took the same
  time on the M3 Ultra, so launches dominated there; on the M4 the cell math shows.
- GARCH: iteration-bound. Nelder-Mead's stop rule (population std of the simplex values
  < 1e-6) cannot be met in float32 at -loglik ~ 100, so 2350 of 10000 series run both
  2000-iteration runs to the cap and EVERY 32-series simdgroup contains one
  (per-simdgroup max = 4002 iterations, median series 2082).

## IDENTICAL changes (same bits: every digest equal before and after)

1. `DeviceExec.upload` / `download` (sequence/exec_device.mojo): memcpy instead of the
   element loop; an upload no longer synchronizes (its staging buffer lives until the
   next sync). Every device algorithm of the family goes through it.
2. Recurrent: one launch per time step, forward (`ops.mojo::op_cell_fwd_h`: the hidden
   GEMM, the b_hh add and `op_cell_fwd` verbatim) and backward (`op_cell_bwd_h`: the later
   step's recurrent GEMM folded into the start of this step's `op_cell_bwd`, the dead GEMM
   into h_0 after s = 0 dropped). `ops.mojo::gemm_dot` is now the lane's one GEMM fold
   (op_gemm and both fused steps); arm 5500 regenerated to reverse it.
3. Nelder-Mead fixed point (sequence/nm.mojo; GARCH, ETS, Theta): an iteration that
   changes no word of the simplex or its values leaves the loop in a fixed point; the
   loop jumps to max_iter (same result, same iteration count, same last evaluations).

Tried and reverted (no gain on m4-a, digests equal): GARCH s2[t-1] carried in a register
(7.83 -> 7.98 s) plus NLL terms staged 4 at a time (8.51 s); TPB 32 for GARCH (7.60 s,
noise); the recurrent forward reading W_ih / W_hh through a per-forward transpose
(LSTM 2.87 -> 2.98 s).

| algo | m4-a before | after (1+2) | digest |
|---|---|---|---|
| lstm fit / infer | 3.28 / 1.18 | 2.87 / 0.82 | 95db702a0612b4b9 same |
| gru fit / infer | 2.67 / 0.90 | 2.35 / 0.63 | f10b45313504c46a same |
| rnn fit / infer | 1.51 / 0.34 | 1.37 / 0.25 | 489b5e34e14e3287 same |
| mlp | 1.73 | 1.72 | same |
| moe | 0.54 | 0.39 | same |
| layernorm | 0.92 | 0.56 | same |
| rmsprop | 1.06 | 0.41 | same |
| adagrad | 1.06 | 0.45 | same |
| lion | 1.05 | 0.46 | same |
| adamax | 1.05 | 0.45 | same |
| nadam | 1.05 | 0.47 | same |
| lamb | 2.06 | 1.60 | same |
| adafactor | 1.14 | 0.95 | same |
| stl, theta, croston, ets, var | unchanged within noise | | same |
| garch (3, NM fixed point) | 7.83 | 1.37 | a27d94de9534677a same; iterations identical (mean 1737, 2350 capped) |
| ets, theta (3) | 1.62 / 0.57 | 1.58 / 0.56 | same (they do not stall this way) |

(1+2: steward 1790581085421 at a2f48dac3, recheck 1790581337120 at 92f6cb58f;
3: 1790581534292 at ca702b87e.)

Identity: batched family request (30 lanes, e2e_family_host_bits.patch) at ca702b87e:
1790582610273-sequence-ca702b87e5 on m2pro, m3ultra-b, m4-a, do-amd.

## FAST (m4-a, 92f6cb58f, steward 1790581337120; FAST digests differ by design)

lstm 2.90, gru 2.36, rnn 1.36, mlp 1.56, moe 0.38, layernorm 0.54, rmsprop 0.41,
lamb 1.03, adafactor 0.58, stl 0.35, theta 0.47, ets 1.68, garch 6.00, var 0.88.
With the NM fixed point (ca702b87e, 1790581534292): garch 6.00 -> 1.17 s (FAST digest
281bdc015afc7c45 unchanged: the exit is exact in either mode).
