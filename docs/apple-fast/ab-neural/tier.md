# tier.txt: the baselines (lane afn-tier, 2026-10-03)

No define. Each line builds the binding twice, IDENTICAL (raced as `ours`, MOJOLEARN_NUMERIC_MODE=identical)
and FAST with no MOJOLEARN_AFN_* define (raced as `ours-fast`, MOJOLEARN_NUMERIC_MODE=fast), alternated
twice at 3 rounds each, our arm alone (no torch opponent is re-run). Mechanism measured: the FAST tier as
main ships it on Apple, which is the IDENTICAL kernels with the pins of checks/numerics.mojo on the free
schedule; `AFN-DEF-SUMMARY ratio` below 1.0 means FAST's median is the lower one. Expected: at or a little
below 1.0 (the pins are cheap), and any lane above 1.0 is a FAST-slower-than-IDENTICAL row to fix first.
Risk: none to the product (nothing changes); a `median_ms=none` on an lm-* line means the byte LM FAST
build (afn-lm) is not in the merged branch yet. Quality: the output judge compares FAST's losses or
outputs to IDENTICAL's (f32 reassociation noise only; tolerances in tools/afn_ab.py), and on the samba,
mlp and block lanes tools/neural_fast_quality.py's pair rule at one seed. Board lanes: every GPU neural
lane (gemm, gemm-bf16, gemm-int8, transformer-forward, mamba1/2/3-forward, samba-forward,
samba-train-step, mlp-train-step, lm-forward, lm-train-step).
