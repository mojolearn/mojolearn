# NVIDIA FMA boundary correction and transformer final timing

The pre-existing `fma.rn.ftz` implementation flushed before rounding at a smallest-normal boundary: a=0x3f7fffff, b=0x00800000, acc=+0 yielded zero, whereas round-to-nearest then explicit flush yields 0x00800000. Direct imported attention `_step` and GEMM `_tuned_step` failed 48 of 10,648 finite adversarial triples on L40S driver580.159.03 (`boundary.log`).

The selected implementation computes FMA with RN and no FTZ, then applies `mul.rn.ftz` by exactly1.0 to the rounded value. This flushes subnormal inputs to that second instruction, preserves signedzero, and keeps the smallest normal unchanged. Attention register-tile score inputs were already flushed on shared-memory staging, so a preflushed helper avoids repeating operand flushing in that loop. Other uses preserve their existing operand flush semantics. No FAST or DETERMINISTIC paths change.

The selected production functions passed262,144 triples against NVIDIA software RN-then-flush (22 explicit edge words plus42 reproducible finite random words, full Cartesian product). The permanent `transformer/checks/attention_fma_boundary_check.mojo` also independently asserts the smallest-normal result. This is a NVIDIA-scoped correction: the concurrent Apple audit found the underlying software FMA itself can flush before rounding at this boundary. No universal cross-column closure is claimed and shared numerics/tree code is outside this lane.

## Public API measurements

Same dedicated L40S, original Sep7 public harness/seed7/shapes, five rounds; complete64MiB output pairs cmp equal for every experiment. The final joint candidate includes root's b308 GEMM dispatch improvement and corrected GEMM FMA, not just attention. Raw rounds and full SHA256 are retained.

| Variant | B8/L4096/D512 ms | B8/L1024/D2048 ms |
|---|---:|---:|
| Original session baseline2bdd7721 |426.012|264.311|
| Tile/direct-weight changes, old incorrect seam |175.726|180.683|
| RN + bitwise output flush, attention only |238.222|190.685|
| RN + bitwise flush, attention and current GEMM |262.591|283.326|
| Above plus preflushed attention score helper |231.330|269.663|
| Zero-result guarded recomputation, rejected |420.368|474.574|
| **RN then hardware multiply flush, selected** |**177.617**|**188.857**|

Final own A/B speedups are2.40x and1.40x. The Torch numerical admission failure documented in `../transformer_e2e_2026-09-09/README.md` still applies; these are not qualified opponent ratios. No additional opponent timing was performed during FMA experiments.

Initial final validation: selected native fused15/15, full public output cmp, focused HD128 output/gradients/cache/refusal checks all pass. `mul_production_boundary.log` uses an inherited 'Guarded' label but runs the selected RN+multiply production source; `mul_boundary_final.log` runs the permanent final check. Final GEMM7gate/19plan and16shape wide-split, transformer forward/backward reference cards, original Python surface, and5shape GEMM diagnostics all passed; logs are retained. `attn-mul-final.rc` and `attn-boundary-final.rc` are both0. Final Apple validation also passed: native fused15/15, both original reference-card comparisons, rebuilt IDENTICAL binding, 116/0 Python surface and focused HD128 gradient/cache/mutable-weight checks. These preserve existing Apple arithmetic; they do not close the separately recorded primitive underflow issue. Evidence is in apple/.

The rejected guarded diagnostic script reached both card comparisons and full surface, then its optional tuned probe failed because the shell was still in `python/`; the selected script fixes that working-directory mistake. This was not a product failure. Scripts and unsuccessful variants are retained to make experiment selection reviewable.

Final original HD64 window benchmark (DM1024/H16/KV4/HD64/IT4096,
B4/L4096/window2048, three measured rounds) was replayed through the final
binding: forward140.3ms; forward+backward493.7ms. This supersedes the earlier
154.7/444.9ms pre-correction values. The stored earlier SDPA35.9/114.3ms
reference has the same L40S/580.159.03 tuple but was measured on a different
physical pod; derived3.91x/4.32x ratios carry that explicit limitation.
No new opponent timing was performed.

Transferred pod o8q8dklahstpoz was deleted after all evidence fetches at
2026-09-09 22:13:12 EDT (DELETE HTTP204); absence verified22:13:17
(GET HTTP404). No rental remains owned by the attention lane.
