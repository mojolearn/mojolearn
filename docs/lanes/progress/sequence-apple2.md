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
   W iterations), default W = 100, rel = 1e-6 (pending the sweep).

## Before / after

(pending the first job)

## Unproven

(pending)
