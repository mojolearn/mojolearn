# AMD post-round class flush

Branch `lane/amd-gemm-class`, based on main `1863520e9`.

GEMM is 81.3% of the measured AMD step; halving GEMM buys 40.65% of
that step. The 33.5 TFLOP/s contract ceiling is the H100 ceiling, not an
AMD hardware limit. No instruction-count speedup is claimed.

The opt-in `MOJOLEARN_GEMM_CLASS_FLUSH` changes only the rounded result's
flush in `_tuned_step`. The column capability lives in the kernel matrix.
Gather staging is already shipped on AMD; its fold retains software FTZ.
No scheduling or P=1 fold change. No production default flip.

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
old step if everything else holds; the actual step reduction remains owed.

RunPod AMD pod `36jpr8u3qw59mn` reported 201,007,382,528 of 206,141,652,992
VRAM bytes used at acceptance. HIP reported zero free bytes. Its seam probe
passed, but generic checks/card and the full price failed with out-of-memory
errors. It was terminated and verified 404. This is retained failure evidence,
not an admitted price. The replacement Hot Aisle MI300X passed all phases and
was deleted/verified absent after completing its work.

Training comparisons on both corpora and columns are pinned to `283577480`.
Each arm has separate pure and instrumented builds in the same directory,
700 untimed steps, and ten instrumented steps after step 700. Report the pure
steps 501–700, not a mixture with the early attention regime. All 700 loss
hashes and the final gradient/parameter/optimizer hashes must match. No flip
or merge before those results. A detached measurement worktree preserves
that commit while this branch records evidence.

Two runner fixes accompany the work: strict R2 staging now stops the RunPod
payload on failure (injected failure exits before the payload; success
continues), and recovery pod names include the process ID so simultaneous
same-vendor legs cannot adopt each other's pod after a failed create. The
old collision occurred before any training started and both legs are being
rerun. All R2 corpus logs inspected so far show the pinned bytes and links.
