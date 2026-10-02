# Resident optimizer: pooled param/grad device buffers, staged download on Apple (PR #31, lane/neural-pass26 25193e918), 2026-10-01

optimizer_resident_check PASS (32 steps, every byte equal) for adam / sgd / adamw on the M3 Ultra, the L40S and (head
5b1f8aa84) the MI325X.

| | main | branch |
|---|---|---|
| M3 Ultra board cells sgd / adam / adamw (Metal, synthetic) | 431.6 / 442.6 / 437.5 ms | 111.0 / 115.4 / 122.0 ms (3.6-3.9x), digests identical |
| M3 Ultra resident step adam / sgd / adamw | ~17-18 ms | 10.3 / 10.3 / 10.2 ms |
| M3 Ultra, head f8a2d004d (raw download default) | | 348-363 ms; with MOJOLEARN_OPT_STAGE=1 111-112 ms: the staged download is the Apple lever |
| L40S resident step adam / sgd / adamw | 21.2 / 22.0 / 21.4 ms | 25.0 / 21.3 / 22.8 ms |

Open: Adam on the L40S reads slower on the branch in three runs (+18-40%); SGD and AdamW even. Follow-up asked of the
author. Also here: ivf-stages-l40s-20261001.log (MOJOLEARN_ANN_STAGES=1, released 0.8.32, 400,000 x 220, 1024 lists):
fit 2,805 ms (kmeans 1,085, upload 752, layout 204, copy_in 175, read_admit 195), search 790 ms.
