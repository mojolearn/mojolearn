# Transformer public end-to-end, 2026-09-09

Same dedicated NVIDIA L40S, driver 580.159.03, baseline checkout 2bdd7721 versus this lane's final changes. IDENTICAL only; original Sep 7 public API script, seed 7, five measured rounds, weights/refusals/output transfer included. No weight caching. Original script's historical note about a host weight List remains in the raw final log; the final implementation removes that List.

| Public shape | Baseline ms | Final ms | Speedup |
|---|---:|---:|---:|
| B8 L4096 D512 HD128 | 426.012 | 175.726 | 2.42x |
| B8 L1024 D2048 HD128 | 264.311 | 180.683 | 1.46x |

`medians.json` contains every measured round, including intermediate tile experiments. Baseline and final full output files each contain 16,777,216 float32 values (64 MiB). The final script ran `cmp` on each complete pair and exited zero. Full SHA256 matches are in `baseline_full_sha256.log` and `final_full_sha256.log`; short benchmark witnesses are not the sole identity check.

Changes: HD128 forward register tiling uses TQ16/BK32/KS64 (18,624 shared bytes), preserving ascending score and context FMA chains. HD64 retains its existing tile. A separate forward support predicate lets Apple HD128 forward fit while retaining the old backward fallback. IDENTICAL Python bindings upload each caller weight directly into device buffers, avoiding nine intermediate zero-filled/copied host Lists. Device weight validation keeps the same order and text and runs every call, including after caller mutation. Other numeric modes retain their original weight construction. The compile flag MOJOLEARN_TRANSFORMER_LEGACY_WEIGHT_COPY selects the old copy for diagnostic replay.

Final sampled wide profile: weight upload/refusal 20.9 ms, attention core 29.45 ms, MLP/residuals 62.34 ms, output transfer 32.24 ms. These single samples are explanatory, not timing medians.

Validation: NVIDIA fused 15/15 cases, hd128 public forward/backward all buffers, split cache state, repeated mutable-weight refusal, and original Python surface passed. Final NVIDIA original forward/backward cards both matched. Final Apple native 15/15, original forward/backward card comparisons, rebuilt IDENTICAL binding, 116-check Python surface and focused HD128 gradient/cache/mutable-weight checks all passed. Logs, cards and commands are in apple/.

## Opponent admission FAILED

The unchanged original-grid eager Torch FP32 comparator was measured once on this same L40S: torch 2.4.1+cu124, CUDA 12.4, TF32 explicitly disabled. Selection wrapper `grid_torch.py` only filters the arm and shapes. Input agreement passed all 20 tensors. Output admission FAILED both rows at the original rtol=0.0005, atol=0.00001: narrow max abs 0.0138197; wide 0.0278463. Raw torch medians 63.849/43.787 ms are retained for diagnosis but do not constitute an admitted opponent ratio. No tolerance was changed and no repeat opponent timing was used to select a favorable result.

This predates the change: original sibling `mojolearn-grid/bench/results/e1g/2026-09-07_174841-nvidia-speed-gemmseq/remote/logs/seq.transformer.torch.log` lines 375 and 423 explicitly report the same failed gate, narrow max abs 0.0138197 and wide 0.0285354. The original advertised 22–41x gap therefore also needs this qualification. Original RoPE uses NumPy power/sin/cos whereas the pinned implementation uses portable transcendental operations (DEVIATION 809); this is only a possible contributor, not a localized diagnosis.

A separate pre-existing issue was identified by the Mamba lane's adverse FMA probe: fused attention's unchanged hardware FTZ FMA helper is not equivalent to the software seam at every float32 rounding boundary, contrary to its historical docstring. This change does not introduce that seam; existing NVIDIA HD128 forward already used it. Ordinary and underflow fixture preservation here is not proof of universal seam equivalence. Root owns tracking that cross-lane numerical issue.
