# AMD post-round class flush

Branch `lane/amd-gemm-class`, based on main `1863520e9`.

GEMM is 81.3% of the measured AMD step; halving GEMM buys 40.65% of
that step. The 33.5 TFLOP/s contract ceiling is the H100 ceiling, not an
AMD hardware limit. No instruction-count speedup is claimed.

The AMD production default changes only the rounded result's
flush in `_tuned_step`. `MOJOLEARN_GEMM_LEGACY_CLASS_FLUSH` restores the old
spelling for controlled comparisons; the measured candidate previously used
the opt-in `MOJOLEARN_GEMM_CLASS_FLUSH`. The column capability lives in the kernel matrix.
Gather staging is already shipped on AMD; its fold retains software FTZ.
No scheduling or P=1 fold change. Production-default device qualification
is required before merging this flip to main.

`tools/gemm_class_probe_price_leg.sh` first builds a deliberately corrupted
probe and requires its gate to fail, then requires the clean probe's class,
shipped and software lanes to hash to `62a6b5621e27c707`, with zero class
mismatches and the normal boundary word. Only after that device proof does
it build the opt-in production path and bracket fixed-size prices.
It does not execute the closed wave-mode experiment.

Local evidence: one-worker nice-19 gfx942 ISA compilation, target and class
instruction matches; missing/duplicate/wrong-hash gate sabotage checks.
No local GPU work.

## Device proof and fixed-size prices

The class lane passed on RunPod MI300X, Hot Aisle MI300X, and RunPod H100:
`62a6b5621e27c707`, zero shipped/class mismatches, boundary `00800000`.
The deliberately corrupted device result was `00800001` at triple 400;
each gate rejected it. The closed wave-mode kernel was not launched.

All twelve output digests match across both builds, both repetitions, and
both vendors; `bench/results/gemm_class_flush/cross-column-price-gate.log`
prints the matches and deliberate corruption failures.

| Fixed-size GEMM sum | baseline brackets, ms | class brackets, ms |
|---|---:|---:|
| MI300X, Hot Aisle | 559.397 / 559.937 | 486.662 / 487.219 |
| H100, RunPod | 121.470 / 121.452 | 121.359 / 121.351 |

AMD's GEMM reduction is 12.995%. NVIDIA is INERT (under 0.1% timing noise).
These are weighted fixed-size GEMM microbenchmarks, NOT step measurements.
Against the supplied 81.3% share, the AMD result projects about 10.56% of the
old step if everything else holds; the measured late-window whole-step reduction is 8.43% (geometric mean).

RunPod AMD pod `36jpr8u3qw59mn` reported 201,007,382,528 of 206,141,652,992
VRAM bytes used at acceptance. HIP reported zero free bytes. Its seam probe
passed, but generic checks/card and the full price failed with out-of-memory
errors. It was terminated and verified 404. This is retained failure evidence,
not an admitted price. The replacement Hot Aisle MI300X passed all phases and
was deleted/verified absent after completing its work.

Training comparisons on both corpora and columns are pinned to `283577480`.
Each arm has separate pure and instrumented builds in the same directory,
700 training steps, and ten instrumented steps after step 700. Report the pure
steps 501–700, not a mixture with the early attention regime. All 700 loss
hashes and the final gradient/parameter/optimizer hashes match across every
build and both columns. These results precede the default flip. A detached measurement worktree preserves
that commit while this branch records evidence.

Two runner fixes accompany the work: strict R2 staging now stops the RunPod
payload on failure (injected failure exits before the payload; success
continues), and recovery pod names include the process ID so simultaneous
same-vendor legs cannot adopt each other's pod after a failed create. The
old collision occurred before any training started; both owed comparisons
were rerun under distinct pod names. All R2 corpus logs inspected so far show the pinned bytes and links.

## Completed pure training comparisons

Both columns ran measurement commit `283577480eecd8237189965131b0ffd62460aa7b`,
seed 93261, batch 1, length 2048, d_model 768, 12 layers, 12 heads and KV
heads, head_dim 64, intermediate 2048, vocab 50257, 162,147,840 parameters.
Each entry below is the median of pure steps 501–700 after a full 700-step
run. This is the late high-cost attention regime; enwik8 is still evolving
within that window, so these are not claimed to be steady-state measurements.

| Column / corpus | baseline ms | class ms | whole-step reduction |
|---|---:|---:|---:|
| MI300X / enwik8 | 861.667 | 791.067 | 8.193% |
| MI300X / Pile GitHub | 793.606 | 724.909 | 8.656% |
| H100 / enwik8 | 403.373 | 404.659 | -0.319% (INERT) |
| H100 / Pile GitHub | 327.097 | 327.142 | -0.014% (INERT) |

The change bought **8.425% of the AMD step**, geometric mean across corpora.
All 700 per-step loss hashes and the final gradients, parameters, optimizer
m/v and flags match across baseline/class and AMD/NVIDIA. Gate logs print
every witness and reject deliberately corrupted witnesses first.

NVIDIA baseline/class ELF allocated **sections** match exactly for both pure
and timer builds, built at the same path. Section-gate negative controls
change every section digest in turn and are rejected. The timing differences
above are noise, not a NVIDIA optimization. No opponent was rerun and no
stale opponent ratio was computed.

R2 staging was strict, with both pinned corpus hashes and destination links
verified in the stage logs. The H100 leases completed and were verified gone.
Both AMD instrumented runs completed with all gates passing and both leases
were deleted and verified absent. Production-default qualification is next.

## GEMM breakdown at the same measurement commit

These shares use the **instrumented step at 701–710** as denominator. The
pure 501–700 window above has different data and attention cost; its gap from
the instrumented window is not an isolated instrumentation-overhead estimate.
Rates use the twelve target GEMMs (1.518 TFLOP/step); shares also include
`norm_dW`. Raw runner summaries are retained as `summary.runner.json`;
`summary.json` and `summary.corrected.log` label these intervals correctly.

| Column / corpus | GEMM share before → after | GEMM TFLOP/s before → after |
|---|---:|---:|
| MI300X / enwik8 | 64.47% → 61.14% | 2.676 → 3.069 |
| MI300X / Pile GitHub | 69.10% → 66.00% | 2.691 → 3.082 |
| H100 / enwik8 | 28.17% → 28.18% | 12.438 → 12.437 |
| H100 / Pile GitHub | 34.25% → 34.26% | 12.436 → 12.430 |

The retained Apple identity card matches NVIDIA and AMD; every card row was
corrupted in turn and rejected before accepting those matches. Apple source
and dispatch are unchanged, and no full Apple column was run. The historical
Apple adversarial-boundary discrepancy documented in `LANE_STATUS_gemm-next.md`
is not repaired or redefined by this change.
