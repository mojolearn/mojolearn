# T3 rehearsal, 2026-09-23: the T3 recipe and driver, end to end, on rented boxes

Route A, two segments of 3 optimizer steps on the real T3 recipe
(`runs/t3/2026-09-22/recipe.json`: hash scheme `sliced-sha256-8.v2`, 5,000-step
schedule), run key `runs/t3-rehearsal/2026-09-23`, the published wheel
`mojolearn 0.8.15` installed on each box (no build on a box), driven by
`tools/lm_run_driver.py` from `spec.json`.

| segment | box | steps | s per step | hash s | arrival replay | checkpoints |
|---|---|---|---|---|---|---|
| A/1 | NVIDIA H100 80GB x2 (RunPod, Python 3.11 venv) | 0 to 3 | 34.7 | 9.4 | none (the seed) | 0, 1, 3 |
| A/2 | AMD MI325X x1 (DigitalOcean, Python 3.12 venv) | 3 to 6 | 138.9, 135.5, 135.5 | 2.6 | PASS (steps 2, 3 from the H100's checkpoint 1) | 4, 6 |

The MI325X reproduced the two-GPU H100's steps 2 and 3 bit for bit under the
new scheme, then trained on. Token parts arrived in 6 to 10 s each (20
parallel byte ranges per part).

## What it found and what changed (all on main)

1. `tools/lm_segment.py` crashed at its first state hash under the wheel's
   Python 3.11: `memoryview(Array)` raises below 3.12. Fixed (42b65751e): every
   byte view goes through the package's buffer helper.
2. One curl stream per token part sat at 130 KB/s for 25 minutes on the H100
   pod while range requests ran at 35 to 48 MB/s. The first run was rescued by
   hand with ranged fetches; the bodies now fetch every part as parallel byte
   ranges with a real speed floor (60bb8e959, 2122a5f99: a presigned GET URL
   refuses HEAD, so the size comes from Content-Range).
3. The 8x MI325X droplet (`gpu-mi325x8-2048gb`) is refused by DigitalOcean:
   "creating this droplet will exceed your GPU limit", with no droplet live.
   T3's AMD segments assume that size; the account limit must be raised, or
   the AMD segments split for one GPU (39 h per 1,000 steps at 139 s a step,
   over the 12 h lease). The rehearsal's AMD segment ran on one MI325X
   (`amd_size`, `amd_devices` in `spec.json`).

## Files

`spec.json`, `ledger.json`, `driver.log` (every attempt), and per segment
`status.txt`, `leg.txt`, `gpu.txt`, `binding.txt`, `wheel.sha256`,
`tokens_fetch.log`, `segment/{segment.json,chain.jsonl,manifest.tsv,log.txt}`,
`arrival/...` where a replay ran. Checkpoints are in R2 under
`runs/t3-rehearsal/2026-09-23/`.
