# PR #57 (lane/neural-pass49, pipelined staged download + fused scan) and #56 (scratch pool) on the M3 Ultra, 2026-10-01

~/pr5657m-metal: arms with origin/main (c49919ced, before #65) merged in. All arms: lm-forward digest 4a8e781b0739a038, same losses.

| arm | lm-forward stage (ms) | lm-train-step stage (ms) | forward race | train race |
|---|---|---|---|---|
| main | 70.7 / 70.2 | 196.0 / 193.8 | 63.6 | 185.7 |
| #57 (pass49) | 68.9 / 68.9 | 198.1 / 194.1 | 62.5 | 182.4 |
| #57 STAGE=0 | 95.1 / 94.8 | 198.2 / 195.6 | | |
| #57 chunk 1 MiB / 4 MiB | 74.7 / 67.3 | 209.1 / 195.5 | | |
| #56 (pass48) pool on / off | 69.4,70.1 / 70.7,70.0 | 198.6,195.2 / 200.9,198.6 | 63.9 | 183.7 |

#57: ~2% faster on Apple. #56: neutral (closed). An earlier run with unmerged arms (~20 commits behind) is superseded.
#57 decision waits on the L40S rerun on the #67 tree (nvc1-0013, lane/neural-pass49-on-61).

## NVIDIA L40S, #57 on the #67 tree (measure-only lane/neural-pass49-on-61, both arms with main 3befccd28 merged in)

| row | pass61 (#67) | pass49-on-61 (#67 + #57) |
|---|---|---|
| lm-forward stage (ms) | 17.6 / 17.5 (timers 17.4 / 17.8) | 15.6 / 15.9 (timers 16.0 / 15.7) |
| lm-train-step stage (ms) | 38.1 / 37.9 | 39.1 / 38.0 |

Same bits. Ticks: logits.scan 1.638 -> 0.010, logits.download 3.17 -> 3.45: forward -10%, train step neutral.
The merge of main into the arm conflicted only in bench/results/r2-index.tsv (gemm_identical.mojo resolved without markers; builds rc=0).
