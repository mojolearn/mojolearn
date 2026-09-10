# Performance continuation: final September 9 state

Scope: our IDENTICAL implementations; tree sources and their numeric behavior were not changed. Existing FAST/DETERMINISTIC paths were preserved. Three delegated lanes handled Mamba3, attention/transformer and kNN/UMAP, with root handling integration, GEMM and Apple gates. All owned rentals were deleted and absence verified.

| Area | Final result | Comparison and limitation |
|---|---:|---|
| Mamba3 narrow B8/L4096/D512 |78.30ms|16.29x faster than Sept 7 ours; 4.78x admitted historical torch FP32 reference|
| Mamba3 wide B8/L1024/D2048 |149.87ms|11.02x faster than Sept 7 ours; 7.73x reference|
| Transformer narrow / wide |177.62 /188.86ms|Same L40S ours 426.01/264.31ms; 2.40x/1.40x faster; original and fresh torch numerical admission fail|
| Attention HD64 forward / forward+backward |140.3 /493.7ms|Matched L40S 580.159.03 stored SDPA 35.9/114.3ms gives 3.91x/4.32x; reference measured on another physical pod|
| kNN 400k index/4000 queries/k10 |40.67ms|Same H100 ours 66.54ms; pinned cuML 10.225ms:6.51x→3.98x|
| UMAP 1M |48.38s|Same L40S before 57.03s,15.18% less; one cold timed round each, no fresh opponent ratio|
| Gram 32x1M /Gram 128x100003 |0.4560 /0.4377ms|H100 shape-matched stored cuBLAS timing ratios 1.90x/4.56x; operand generators differ|
| Dense Llama t512 QKV /MLPup /MLPdown |1.952 /7.576 /6.755ms|H100 shape-matched stored cuBLAS timing ratios 5.22x/5.49x/5.70x; corrected arithmetic regresses earlier 2.1–2.7x figures|
| Apple kNN 400k/1000 queries/k15 |254.75ms|Correct underflow repair costs 39% versus 183.26ms; NVIDIA dispatch/timings unaffected|

Mamba1 decode is wired through `mamba_simple.mojo::mamba_step` in the public binding and native clause-D checks; `pixi run check-mamba-decode` exercises the formerly orphaned module. Host and device reference/decode/refusal/sabotage gates and the combined 102-check Python surface passed. Commit 2bdd7721 and `HANDOFF_mamba1_decode.md` contain that integration.

Mamba3 removes disabled-trace downloads, host element copies/zero-filled weight containers and full-buffer refusal readback. Integer device refusal preserves named order and first index. State/angle factoring and shared operand tiles preserve every ascending floating fold. Final default native traces, decode/continuation/refusal gates, full public output SHA and 102-check surface passed, including corrected GEMM. Final Apple evidence is retained. The large 1–1.5x torch target remains unmet; host/device transfers and projections are substantial residual costs.

Transformer HD128 uses a shared-memory-fitting register tile and direct per-call weight uploads. Mutable weights are reread/refused, no cache is hidden. Final native 15 cases, both reference cards, 116-check surface and focused HD128 gradients/cache/refusal checks passed on Apple and NVIDIA. Full 64 MiB output pairs match. The original Sep 7 torch comparison itself failed its numerical tolerance; no tolerance was relaxed and its 22–41x gap must not be described as an admitted comparison.

A new adversarial gate exposed a real error in NVIDIA's old single hardware FTZ FMA shortcut. The accepted replacement is RN FMA followed by hardware FTZ multiply by 1, preserving signed zero and the smallest normal rounding boundary. Imported production GEMM/attention functions pass 262,144 adversarial triples on L40S and H100. The initially faster but incorrect figures are retained only as history. Dense GEMM still needs work; no invented speedup is claimed.

The same boundary exposed a pre-existing Apple native-FMA issue. A kNN-local exact UInt64 repair, kept outside the common path, passed 396,584 independently generated integer RN-even oracle cases and the literal 8-case regression. It deliberately does not alter shared numerics or tree behavior. Other Apple FMA consumers remain an open numerical audit; ordinary bitwise fixture agreement does not close that issue. The inline repair that slowed kNN 13.3x was rejected; the retained correction has the 39% cost shown above.

The final Apple IDENTICAL base Python extension was rebuilt. Public kNN passed exact dyadic distance, tie-order and repeated-bit checks at k=1/10/15/33; logs and the smoke script are under `selector-final/apple-public-binding/`.

Evidence and detailed commands:

- `bench/results/mamba1_decode_wiring_2026-09-09/`
- `bench/results/mamba3/2026-09-09-statepass/`
- `bench/results/transformer_e2e_2026-09-09/` and `bench/results/attention_exact_fma_2026-09-09/`
- `bench/results/gemm_followup_2026-09-09/h100-corrected/`
- `bench/results/knn/2026-09-09-selector-resume/` and `2026-09-09-selector-final/`

Remaining targets: large Mamba3 parity, qualified transformer comparator, dense GEMM performance under correct arithmetic, broader Apple round-then-flush consistency, and a fresh 100k UMAP comparison if that exact row is needed. The 1M UMAP result does not refresh the old 100k 5.2x claim, which mixed GPU tuples.
