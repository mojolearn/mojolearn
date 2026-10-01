# PR #57 download pipeline (lane/neural-pass49 1b2840f1e) on the MI325X, merged 559a77f01

ab_job amd p57 (pass14), main vs lane, plain x2 then timers. Same bits: lm-forward digest 4a8e781b0739a038 on every arm.

| arm | lm-forward ms | lm-train-step ms |
|---|---|---|
| main 1 / 2 | 23.204 / 23.318 | 58.362 / 58.591 |
| pass49 1 / 2 | 21.809 / 21.783 | 58.419 / 58.519 |

Forward -6%, train neutral. With L40S forward -10% and M3 ~-2% (earlier summaries), #57 merged.
Bulk: R2 measurements/2026-10-01/ab-pr57-amd.tar.gz.
