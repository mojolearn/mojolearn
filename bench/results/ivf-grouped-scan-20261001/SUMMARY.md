# IVF identical scan grouped by list (PR #47, lane/neural-pass42 c57f56ed0), 2026-10-01

IVFIndex(1024 lists, 32 probes, k 10, 20 k-means iters) on 400,000 x 220, 4,000 queries; sha of (distances, indices)
95089ab768e8be7a in every run on both vendors. Restore = MOJOLEARN_IVF_SCAN_GROUPED=0.
| search (ms) | released 0.8.32 | branch (grouped) | branch ungrouped |
|---|---|---|---|
| AMD MI325X | 559 / 558 | 69.3 / 69.5 (8x vs 0.8.32) | 453 / 454 |
| NVIDIA L40S | 776 / 819 | 254 / 195 (3-4x) | 605 / 589 |
Fit unchanged (~3.4 s AMD, ~2.7 s NVIDIA).

`pixi run check-ivf` FAILS on both vendors with "check_filter_matches_oracle: FAIL an all-ones filter moved slot 1" --
on the branch, on current main, and on the 0.8.32 release source (32ad13888): a pre-existing break in the filtered
search path (or the check), not caused by #39 or #47. Handed to the author as a separate bug.
