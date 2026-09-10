# IDENTICAL GEMM follow-up, 2026-09-09

The retained change is plan 18: 128x128 split-K tiles with K step 16, preserving the existing leaf arithmetic and fold. NVIDIA dispatch selects it for complete 128-wide output tiles, at least 512 leaves, and at most 131072 output cells, subject to the existing 256 MiB workspace cap. Other numeric modes and columns retain their dispatcher.

On H100 PCIe, driver 580.126.09, seven interleaved rounds across 16 aligned/ragged-K shapes produced identical full-output digests with no poison left. New/old elapsed ratios range 0.632–0.684. The 128x128x100003 Gram case improved 0.5763 to 0.3892 ms (32.5% less time). Two largest ragged-K shapes exceed the automatic workspace cap and correctly retain plan 10; their forced-plan measurements are not automatic-dispatch claims. See h100/wide.log for every row.

L40S exploratory Gram128 measurements were 0.492 to 0.385 ms. Rectangular dense tiles and forced stack folds lost and were removed from product code. The saved rejected-candidate-sweep.patch documents the exploratory plan numbers (18/19 rectangular, 20 wide split, 21/22 stack-fold); retained product plan 18 is the former exploratory plan 20. These timings do not claim improvement for Gram32 or dense Llama GEMM.

Validation: final 19-plan NVIDIA device gate passed all seven gates; final Apple device and backward suites passed seven and ten gates respectively. H100 wide probe passed all 16 shapes. Raw logs are in h100/, l40s/, and apple/. All timings use our IDENTICAL mode. No new opponent timing was run, and this report makes no cross-GPU ratio claim.

## Final corrected arithmetic measurements

The initial measurements above predate the NVIDIA round-then-flush correction in8a2b855c. Final same H100/580.126.09 diagnostics are in h100-corrected/: Gram32x1M0.4560ms, Gram128x1000030.4377ms, QKVt5121.9517ms, MLPup7.5761ms, MLPdown6.7549ms. Seven timed rounds per arm; both arms deliberately select the same final dispatcher, so the printed 'speedup' near1 is a timing cross-check, not a before/after claim. Allfive digests match/no poison, and the imported production attention/GEMM boundary gate passes262144 triples on H100 as well asL40S.

Against stored H100 torch/cuBLAS FP32 timing rows0.240/0.096/0.374/1.379/1.185ms, these shape-matched timing ratios are1.90/4.56/5.22/5.49/5.70. The timing drivers use different deterministic operand generators; these are shape/model/driver latency comparisons, not numerical or exact-input admission claims. DenseGEMM is a regression from the earlier2.1–2.7x numbers, whose hardware arithmetic shortcut failed the boundary test. Gram32's earlier1.8x is likewise superseded. The corrected wide-tile old/new paired sweep onL40S remains faster; see ../attention_exact_fma_2026-09-09/mul_gemm_wide.log.

No new opponent timing was run. The shared H100 was deleted and verified absent after final evidence collection; cleanup log is in h100-corrected/pod-deletion.log.
