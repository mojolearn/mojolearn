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

### Job 2: m4pro-b, steward 1790606012720, before 068959af0, after 3c03eaabf (THE clean table)

Two alternating runs per arm, best shown; digests equal across runs. Prophet at
SEQ_PROPHET_N=32768 (below the FAST chunked threshold at that commit, so unchanged).
Records: ~/mojolearn-evidence/sequence-apple2/j2.

| algo | IDENTICAL before | after | x | bits | FAST before | after | x | bits |
|---|---|---|---|---|---|---|---|---|
| lstm (fit / infer) | 0.924 / 0.295 | 0.934 / 0.298 | 0.99x | same | 0.851 / 0.265 | 0.855 / 0.268 | 1.0x | same |
| gru | 0.747 / 0.232 | 0.749 / 0.235 | 1.0x | same | 0.692 / 0.208 | 0.694 / 0.210 | 1.0x | same |
| rnn | 0.378 / 0.101 | 0.380 / 0.104 | 1.0x | same | 0.354 / 0.092 | 0.357 / 0.095 | 1.0x | same |
| mlp | 0.868 / 0.165 | 0.846 / 0.168 | 1.03x | same | 0.741 / 0.159 | 0.727 / 0.162 | 1.02x | same |
| moe | 0.255 | 0.209 | 1.22x | same | 0.247 | 0.203 | 1.22x | same |
| layernorm | 0.606 | 0.179 | 3.38x | same | 0.564 | 0.123 | 4.60x | moved |
| rmsprop | 0.326 | 0.217 | 1.50x | same | 0.325 | 0.217 | 1.50x | same |
| adagrad | 0.367 | 0.222 | 1.66x | same | 0.355 | 0.224 | 1.58x | same |
| lion | 0.382 | 0.228 | 1.68x | same | 0.367 | 0.225 | 1.63x | same |
| adamax | 0.368 | 0.227 | 1.62x | same | 0.368 | 0.226 | 1.63x | same |
| nadam | 0.376 | 0.233 | 1.61x | same | 0.367 | 0.226 | 1.63x | same |
| lamb | 1.777 | 1.740 | 1.02x | same | 1.124 | 0.285 | 3.94x | moved |
| adafactor | 1.079 | 1.068 | 1.01x | same | 0.660 | 0.130 | 5.10x | moved |
| stl | 0.233 | 0.233 | 1.0x | same | 0.188 | 0.189 | 1.0x | same |
| theta | 0.295 | 0.296 | 1.0x | same | 0.206 | 0.204 | 1.0x | same |
| croston | 0.004 | 0.004 | 1.0x | same | 0.004 | 0.004 | 1.0x | same |
| ets | 0.543 | 0.549 | 0.99x | same | 0.423 | 0.419 | 1.0x | same |
| garch | 0.546 | 0.548 | 1.0x | same | 0.410 | 0.299 | 1.37x | moved |
| var | 0.658 | 0.659 | 1.0x | same | 0.630 | 0.474 | 1.33x | moved |
| prophet (32768) | 39.83 | 39.62 | 1.0x | same | 13.09 | 12.92 | 1.0x | same |

FAST quality, same job (FAST before -> FAST after; float64 reference errors, lower better):
lamb max abs err 3.65e-6 -> 1.26e-7; adafactor 7.20e-5 -> 1.09e-7; layernorm dw rel
9.5e-6 -> 2.4e-7, db err per sum|dy| 2.7e-6 -> 4.4e-7; var params rel 0.0444 -> 0.000475.
GARCH (10000 series, fit 100, holdout 12; no stop -> default 50/1e-6): loglik
-138.314873 -> -138.314741, QLIKE 1.085557 -> 1.085526, 0.406 -> 0.289 s. Sweep:
20/1e-6 and 30/1e-5 lose loglik (-138.3176, -138.3180): rejected; 30/1e-6 loses
loglik slightly (-138.31517): rejected; 50/1e-5: loglik -138.314814, QLIKE 1.085543,
0.256 s: both better than no stop, so the default became 50/1e-5 (c592aba84).
Prophet FAST chunked fit at 65536 points (quality run, fit seconds): 118.4 s -> 0.27 s,
objective -170422.72 -> -170422.28 (relative 2.6e-6 higher), in-sample RMSE
0.9932792 -> 0.9932782 (lower), 47 -> 41 L-BFGS iterations.

## Unproven

(pending)
