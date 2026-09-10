# IDENTICAL GEMM follow-up, 2026-09-09

The retained change is plan 18: 128x128 split-K tiles with K step 16, preserving the existing leaf arithmetic and fold. NVIDIA dispatch selects it for complete 128-wide output tiles, at least 512 leaves, and at most 131072 output cells, subject to the existing 256 MiB workspace cap. Other numeric modes and columns retain their dispatcher.

On H100 PCIe, driver 580.126.09, seven interleaved rounds across 16 aligned/ragged-K shapes produced identical full-output digests with no poison left. New/old elapsed ratios range 0.632–0.684. The 128x128x100003 Gram case improved 0.5763 to 0.3892 ms (32.5% less time). Two largest ragged-K shapes exceed the automatic workspace cap and correctly retain plan 10; their forced-plan measurements are not automatic-dispatch claims. See h100/wide.log for every row.

L40S exploratory Gram128 measurements were 0.492 to 0.385 ms. Rectangular dense tiles and forced stack folds lost and were removed from product code. The saved rejected-candidate-sweep.patch documents the exploratory plan numbers (18/19 rectangular, 20 wide split, 21/22 stack-fold); retained product plan 18 is the former exploratory plan 20. These timings do not claim improvement for Gram32 or dense Llama GEMM.

Validation: final 19-plan NVIDIA device gate passed all seven gates; final Apple device and backward suites passed seven and ten gates respectively. H100 wide probe passed all 16 shapes. Raw logs are in h100/, l40s/, and apple/. All timings use our IDENTICAL mode. No new opponent timing was run, and this report makes no cross-GPU ratio claim.
