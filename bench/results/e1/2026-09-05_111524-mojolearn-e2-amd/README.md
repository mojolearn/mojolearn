# AMD MI325X qualification, 2026-09-05

Source b715b12413599d1e932b24b0d9a27a15472eb6a8, IDENTICAL mode.
The retained native backward certificates match the same-source RTX 4090
certificate: 54 gradient tensors across five Mamba-1/2/3 cases. Both UMAP
stage profiles match the retained Apple captures (186 and 690 cells).
The UMAP Python surface passed; the two 64-point quality fixtures passed
trustworthiness 0.997712 and 0.994754, versus row-permuted controls
0.438393 and 0.454018. These are named-fixture results, not general proofs.

The baseline Mamba Python surface crashed with AMD GPU memory fault 134.
Commit 6dc93269 preserves Mamba-2 state allocations and copies uploaded
state into them. Supplemental clean-build checks returned 102/102 passing,
but the artifact transfer had already enumerated its files before the
supplemental diagnostics. Those later raw logs were not retained. This
baseline directory deliberately keeps the failure; durable qualification
of the fix and supplemental transformer/GEMV results requires another run.

The supplemental GEMV transpose candidate was not promoted: observed
2048-square timings were slower than the legacy kernel. No performance
claim is certified by this directory because that log was not retained.

Droplet 598010370 was deleted (HTTP 204), then independently checked
absent (GET HTTP 404 at approximately 11:38 UTC). No active AMD rental.
