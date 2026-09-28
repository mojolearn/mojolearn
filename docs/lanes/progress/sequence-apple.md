# sequence-apple: progress (Apple Metal speed, 2026-09-28)

Branch `lane/sequence-apple` = origin/main (5b622763d) + origin/lane/algos-sequence (the
family's step 0, the Adafactor Metal clamps, `sumsq_fold`, the harness
`tools/sequence_speed.py`) + origin/lane/merged (merged 2026-09-28, d5469618c, on
Andrew's instruction). Never merged to main here: the orchestrator merges the Apple
branches into lane/apple-merged and checks them once. Per Andrew (2026-09-28) this lane
ran NO identity or sabotage requests after that instruction; only speed measurements,
with the output digests compared before and after.

Home Mac: m4-a (Apple M4, 10 GPU cores). Every timing is one run of
`tools/sequence_speed.py` (HIGGS 1M rows from R2, staged to m4-a), each algorithm timed
once after a small warm-up; digest = sha256 of every output. Before and after on the same
Mac. Records: ~/mojolearn-evidence/sequence-apple/speed/.

## Result (m4-a; fit s, "fit / infer" for the networks)

Before = 084fcaf5c (IDENTICAL 1790580241118, FAST 1790584201626); after = 7281afb93
(1790586946276). Every digest is unchanged in both modes: the changes are exact.

| algo | IDENTICAL before | after | x | bits | FAST before | after | x | bits |
|---|---|---|---|---|---|---|---|---|
| lstm | 3.28 / 1.18 | 1.66 / 0.56 | 1.98x | same | 3.22 / 1.16 | 1.52 / 0.50 | 2.11x | same |
| gru | 2.67 / 0.90 | 1.31 / 0.44 | 2.04x | same | 2.61 / 0.88 | 1.21 / 0.39 | 2.16x | same |
| rnn | 1.51 / 0.34 | 0.60 / 0.18 | 2.51x | same | 1.47 / 0.33 | 0.56 / 0.17 | 2.60x | same |
| mlp | 1.73 / 0.44 | 1.08 / 0.31 | 1.60x | same | 1.58 / 0.43 | 0.96 / 0.30 | 1.65x | same |
| moe | 0.54 | 0.39 | 1.38x | same | 0.53 | 0.38 | 1.39x | same |
| layernorm | 0.92 | 0.55 | 1.66x | same | 0.95 | 0.54 | 1.75x | same |
| rmsprop | 1.06 | 0.43 | 2.44x | same | 1.09 | 0.44 | 2.47x | same |
| adagrad | 1.06 | 0.47 | 2.23x | same | 1.10 | 0.47 | 2.32x | same |
| lion | 1.05 | 0.48 | 2.18x | same | 1.07 | 0.47 | 2.25x | same |
| adamax | 1.05 | 0.48 | 2.20x | same | 1.06 | 0.48 | 2.22x | same |
| nadam | 1.05 | 0.50 | 2.12x | same | 1.06 | 0.48 | 2.22x | same |
| lamb | 2.06 | 1.59 | 1.30x | same | 1.64 | 1.03 | 1.59x | same |
| adafactor | 1.14 | 0.96 | 1.19x | same | 0.79 | 0.58 | 1.36x | same |
| stl | 0.44 | 0.43 | 1.03x | same | 0.37 | 0.35 | 1.04x | same |
| theta | 0.57 | 0.57 | 1.0x | same | 0.48 | 0.46 | 1.04x | same |
| croston | 0.005 | 0.005 | 1.0x | same | 0.005 | 0.005 | 1.0x | same |
| ets | 1.62 | 1.60 | 1.0x | same | 1.68 | 1.74 | 1.0x | same |
| garch | 7.81 | 0.99 | 7.92x | same | 5.99 | 0.74 | 8.04x | same |
| var | 0.93 | 0.58 | 1.60x | same | 0.90 | 0.57 | 1.58x | same |

FAST quality: no change here is a FAST-only approximation. Each keeps the arithmetic and
its order in both modes, and the FAST digests are unchanged too, so FAST output (and
quality) is exactly what it was. No paired quality run was needed.

## Where the time went (profile, before)

- Optimizers: ~105 ms per 4M-float step, nearly all host glue. `DeviceExec.upload` copied
  element by element into a staging buffer and synchronized after every copy (5 uploads
  and 4 downloads per step).
- Recurrent: ~101 launches per batch (per time step: hidden GEMM, bias, cell; backward:
  cell, recurrent GEMM), 123 batches. The long GEMM and column-sum folds (K = T*B = 8192
  in the weight gradients) waited out every load.
- GARCH: iteration-bound. Nelder-Mead's stop rule (std of the simplex values < 1e-6)
  cannot be met in float32 at -loglik ~ 100, so 2350 of 10000 series run both
  2000-iteration runs to the cap, and EVERY 32-series simdgroup holds one (per-simdgroup
  max = 4002). A capped simplex ends in a fixed point or a short cycle.
- ETS: also iteration-bound (9999 of 10000 series hit the 1000 cap), but its simplex
  keeps moving: no fixed point and no repeat was found, so it is unchanged.

## What changed (all IDENTICAL-exact: same arithmetic, same order)

1. `sequence/exec_device.mojo`: uploads and downloads by memcpy. An upload no longer
   synchronizes; its staging buffer lives until the next sync. (optimizers ~2.2x,
   layernorm, moe)
2. Recurrent, one launch per time step (`ops.mojo::op_cell_fwd_h`, `op_cell_bwd_h`).
   Forward: the hidden GEMM, the b_hh add and `op_cell_fwd` verbatim. Backward: the later
   step's recurrent GEMM, folded into the start of this step's `op_cell_bwd`; the dead
   GEMM into h_0 is not run. `ops.mojo::gemm_dot` is now the lane's one GEMM fold.
3. `gemm_dot` and `op_colsum`: loads staged 16 / 32 ahead of the fold (the `sumsq_fold`
   trick). LSTM 2.85 -> 1.66, RNN 1.38 -> 0.62, VAR 0.91 -> 0.58, MLP 1.72 -> 1.25.
4. MLP (`mlp.mojo::op_gemm_epi`, `op_colsum_div`): GEMM + bias + activation, GEMM + L2
   term, GEMM + activation derivative and column sum + mean each run as one launch; the
   followers' bodies run verbatim on the element just stored. MLP 1.25 -> 1.08.
5. `sequence/exec.mojo` (HostExec): the fused launches run on the host as the launches
   they fuse, so the GEMM keeps sequence-cpu's host kernel (`host_gemm`); same cells.
6. Nelder-Mead (`sequence/nm.mojo`), exact early exits:
   (a) a fixed point: an iteration that changes no word of the simplex or its values ->
       jump to max_iter. GARCH 7.83 -> 1.37.
   (b) Brent's cycle watch, when the caller passes a snapshot row (GARCH does:
       `GARCH_SNAP` floats at the end of its scratch row). On a repeat at period p, run
       only (max_iter - it) mod p more iterations (a full lap when that is 0, so the last
       sort is the full run's), then stop. The final state, best vertex, last evaluations
       and iteration count are the full run's. GARCH 1.36 -> 0.99 (FAST 1.20 -> 0.74).
   Theta and ETS use (a) only. ETS showed no gain from (b), so it passes no snapshot.

Arms regenerated for the moved code (they still apply; the orchestrator's one check
proves them): 5500 (reverses `gemm_dot`'s staged fold), 5501 (reverses `op_colsum`'s
staged fold). All 47 sequence patches `git apply --check` clean at 7281afb93.

## Tried and reverted (no gain on m4-a, digests equal)

- GARCH: s2[t-1] carried in a register (7.83 -> 7.98 s), plus NLL terms staged 4 at a time
  (8.51); TPB 32 (7.60, noise).
- Recurrent: W_ih / W_hh read through a per-forward transpose (2.87 -> 2.98); the G gate
  folds interleaved in one loop plus GEMM+bias for the input projection and head
  (1.649 -> 1.656).
- LAMB: ||p|| and ||u|| folded side by side (1.60 -> 1.58).
- Adafactor: alpha and the denominator in one thread (0.95 -> 0.98).
- ETS: skipping the unused log term under additive errors (1.59 -> 1.65); cycle snapshot
  (no repeat found).

## Other Macs

M3 Ultra before/after (IDENTICAL and FAST, one request) is 1790586969944 on m3ultra; see
below when it lands.
