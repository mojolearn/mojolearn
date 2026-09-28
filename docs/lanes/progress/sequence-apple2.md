# sequence-apple2: progress (Apple Metal speed round 2, 2026-09-28)

Branch `lane/sequence-apple2`, forked from lane/apple-merged 037daa353. Brief:
~/mojolearn-evidence/apple2_speed_brief.md. Round 1: docs/lanes/progress/sequence-apple.md.
Never merged to main or apple-merged here. No identity or sabotage runs (a later
combined run verifies on m2pro, NVIDIA, AMD and CPU).

## Tools

- `tools/sequence_apple_ab.sh`: the timing command of ONE steward speed job. Each
  variant (`SAB_VARIANTS="before=<sha> after=<sha>"`) is a git worktree built in
  every mode of `SAB_MODES`; the conductor's `tools/sequence_speed.py` times every
  variant (SEQ_SPEED_PYTHON = the variant's python/), same Mac, same job.
  `SAB_QUALITY` also runs `tools/sequence_quality.py` per variant and mode.
- `tools/sequence_quality.py`: the paired FAST quality check. LAMB and Adafactor
  (the harness's 4M-parameter, 10-step run) and LayerNorm's weight gradients and VAR's
  params against float64 NumPy statements of the same math (lower error is better);
  ETS: AAA damped on 10000 HIGGS series, fit 100, forecast 12 against the held out
  12 (NLL, MAE, sMAPE) with a sweep of the FAST stall stop.
- `tools/sequence_speed.py` also times ARIMA (1,1,1), Holt-Winters and KPSS now.

## Changes (see the table for what each measured)

IDENTICAL-exact (same arithmetic, same order; both modes):
1. LayerNorm `op_ln_bwd_w`: the column folds over 1M rows stage 16 rows of loads
   before folding (the round 1 trick; 28 threads each walked 1M strided rows).
2. `gemm_dot` (SHARED by every sequence GEMM): K >= 16384 stages 48 loads (VAR's
   normal equations: few cells, K = 1M).
3. `DeviceExec.download_async`: several device copies share one wait (optimizer,
   LAMB, Adafactor, LayerNorm drivers); `bind` skips the zero fill it uploads over.
4. MLP: X and y gathered in one launch; every layer's ||W||^2 in one launch.

FAST only (compiled out of IDENTICAL builds; each has a paired quality check):
5. LAMB and Adafactor norms: 4096 strided partials, then their ordered sum
   (`op_chunk_sumsq`), instead of one thread's 4M-long chain.
6. LayerNorm weight gradients: rows split into blocks (~8192 threads), then an
   ordered sum of the partials.
7. `gemm` split-K for products of at most 1024 cells over K >= 32768 (VAR).
8. ETS: Nelder-Mead stall stop (best value not down by more than rel |best| for
   W iterations): opt-in for ETS (no gain), default 50 at 1e-6 for GARCH.
9. Prophet, N >= 65536: the likelihood over point chunks on the device, L-BFGS
   and priors on the host (the IDENTICAL fit is ONE GPU thread: 1470 s at 1M points).

## Before / after

### Job 1: m4-a, steward 1790603854192, before 068959af0 (= 037daa353 sources), after bdb1e4bd8

One run each, before then after within a mode. Records: ~/mojolearn-evidence/sequence-apple2/j1.
The box slowed by about 2x during the job (the FAST phase and the later quality runs
of IDENTICAL both ran ~2x slower than the first IDENTICAL phase), so compare within a
mode only.

| algo | IDENTICAL before | after | x | bits | FAST before | after | x | bits |
|---|---|---|---|---|---|---|---|---|
| layernorm | 0.560 | 0.325 | 1.72x | same | 0.651 | 0.303 | 2.15x | moved (FAST split) |
| rmsprop | 0.449 | 0.331 | 1.35x | same | 0.727 | 0.368 | 1.97x | same |
| adagrad | 0.442 | 0.328 | 1.35x | same | 0.724 | 0.367 | 1.98x | same |
| lion | 0.425 | 0.326 | 1.30x | same | 0.732 | 0.365 | 2.00x | same |
| adamax | 0.429 | 0.333 | 1.29x | same | 0.729 | 0.381 | 1.91x | same |
| nadam | 0.432 | 0.329 | 1.31x | same | 0.719 | 0.376 | 1.91x | same |
| lamb | 1.877 | 1.785 | 1.05x | same | 1.669 | 0.519 | 3.21x | moved (FAST norms) |
| adafactor | 1.092 | 1.065 | 1.02x | same | 0.958 | 0.238 | 4.03x | moved (FAST norms) |
| var | 0.592 | 1.038 | 0.57x | same | 0.770 | 0.653 | 1.18x | moved (FAST split-K) |
| garch | 1.011 | 1.011 | 1.0x | same | 1.616 | 1.509 | 1.07x | moved (FAST stall 100) |
| ets | 1.600 | 1.602 | 1.0x | same | 3.929 | 3.661 | 1.07x | moved (FAST stall 100) |
| lstm, gru, rnn, mlp, moe, stl, theta, croston | within noise | | | same | within noise | | | same |
| prophet (1M points) | 1470 s | | | | | | | |

VAR IDENTICAL got slower with the 48-deep staging of long-K GEMM folds: reverted
(61b479c9f). The optimizer gain is the batched downloads (one wait instead of four).

FAST quality (tools/sequence_quality.py, same job; error against float64, lower is better):

| case | IDENTICAL | FAST before | FAST after |
|---|---|---|---|
| lamb max abs err | 3.65e-6 | 3.65e-6 | 1.26e-7 |
| adafactor max abs err | 7.2e-5 | 7.2e-5 | 1.09e-7 |
| layernorm dw rel err | 9.5e-6 | 9.5e-6 | 2.4e-7 (db: 132 -> 21 in the old, cancelling metric) |
| var params rel err | 0.0444 | 0.0444 | 0.000475 |

The two-pass sums are MORE accurate than one long float32 chain, so FAST quality
improves. ETS stall sweep (FAST after, 10000 series, holdout 12): every stop from
50 to 300 iterations at 1e-6 or 1e-7 left the mean iterations at 995 to 1001 and saved
no time (ETS's simplex still gains more than that at the 1000 cap), and nll moved
by 2e-5 at most (444.01043 off, 444.01045 at 100/1e-6); MAE and sMAPE equal. So the
ETS stop is OFF by default (opt-in `_fast_stall`). GARCH sweep: 50/1e-6 cut mean
iterations 1703 -> 248, fit 1.56 -> 1.19 s, loglik -138.31487 -> -138.31474 (better),
QLIKE 1.085557 -> 1.085526 (better); 100/1e-6 saved 7% with equal quality. Default
now 50/1e-6.

## Unproven

(pending)
