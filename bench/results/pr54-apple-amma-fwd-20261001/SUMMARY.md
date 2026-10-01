# PR #54 on the M3 Ultra: Apple matrix-unit attention forward (Oct 1)

Measured on the M3 Ultra (Metal), comparing main to main + PR #54 (head 91e5c62f1, measure branch lane/neural-pass46-measure). Each tree was built from source on the box.
Raw files are in R2 at `measurements/2026-10-01/pr54b-metal.tar.gz`; summary-raw.txt holds the condensed lines.

## Bits
Main and the branch produce the same lm-forward digest (4a8e781b0739a038), the same transformer-forward digest (d5a2b289afdb5709) and the same loss trace. The attention harness has the same ctx/amax/denom digests as the earlier baseline.

## Time (ms, median after the first call; two runs each)

| lane | main | PR #54 |
|---|---|---|
| lm-forward (stage) | 77.2 / 78.2 | 66.7 / 66.9 |
| lm-train-step (stage) | 186.6 / 182.4 | 175.8 / 176.1 |
| transformer-forward (stage) | 11.9 / 11.6 | 10.3 / 10.3 |
| lm-forward (board race) | 71.9 | 60.9 |
| lm-train-step (board race) | 170.4 | 159.3 |
| transformer-forward (board race) | 11.1 | 10.1 |

Standalone attention harness (B1 L2048 nh6 hd64 causal), median: 4.51 / 3.33 ms, against 4.763 ms on the first head.

The code change is in the Apple matrix-unit forward kernel; the shipped TQ stays 32. NVIDIA and AMD digest confirmation: see pr54-offapple (nvc1-0005 and the AMD queue).

## NVIDIA confirmation (L40S nvc1-0005)
The branch build gives the same digests as main: lm-forward 4a8e781b0739a038, transformer-forward d5a2b289afdb5709, and the same losses. Times match main: lm-forward 49.6 / 49.2 ms, lm-train-step 41.2 / 41.1 ms, transformer-forward 2.39 / 2.40 ms. Only the condensed lines (nvidia-confirm-raw.txt) are kept, because the pod retired before the folder was uploaded.
