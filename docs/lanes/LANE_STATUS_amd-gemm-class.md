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
No local GPU work. Device proof and prices are owed. Step comparisons need
at least 700 steps on each corpus; no short step results may be reported.
