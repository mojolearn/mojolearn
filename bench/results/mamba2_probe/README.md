# DEVIATION 2712 probe runs (`tools/mamba2_step_probe.py`)

One `.npz` per run: every array the five call orders produce on the `base`
fixture with the mamba2 lane's weights and slabs, the first repeat's arrays and
a hash per repeat, plus the device and commit. Diff two with
`python3 tools/mamba2_step_probe.py diff A.npz B.npz`.

| file | box | commit | repeats | in-process moved | order-dependent parts |
|---|---|---|---|---|---|
| `2026-09-14-apple-m4.npz` | Apple M4 (Metal), this Mac, single-threaded | 256d09be1 | 2 | 0 | 0 |

A second Apple process the same minute diffed equal on all 86 arrays. The AMD
runs (a Hot Aisle MI300X and a DigitalOcean MI325X, `--repeats 20`, twice per
box in two processes) are owed; their diffs against this file and against
each other are the finding.
