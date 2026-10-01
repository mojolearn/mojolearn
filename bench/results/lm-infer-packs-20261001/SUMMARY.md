# lm-infer: key packs once a layer, causal fill per key (PR #22, lane/neural-pass16), 2026-10-01

AMD EPYC 9575F (Zen 5, 64 cores), CPU column. Host gate PASS 144/144; sweep PASS 864/864 (threads 1, 8, 64).

lm-infer cell, three alternating runs of 5 rounds (main through PR #17 vs this PR), mean_nll 9.017856651220221 in all:
main 190.0 / 180.5 / 187.5 ms; PR #22 146.6 / 191.7 / 146.8 ms. Medians 187.5 vs 146.8 ms (1.28x).
