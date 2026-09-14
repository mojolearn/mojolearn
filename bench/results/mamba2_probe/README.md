# DEVIATION 2712 probe runs (`tools/mamba2_step_probe.py`)

One `.npz` per run: every array the five call orders produce on the `base`
fixture with the mamba2 lane's weights and slabs, the first repeat's arrays and
a hash per repeat, plus the device and commit. Diff two with
`python3 tools/mamba2_step_probe.py diff A.npz B.npz`.

| file | box | commit | repeats | in-process moved | order-dependent parts |
|---|---|---|---|---|---|
| Apple M4 (Metal), this Mac, single-threaded, `--repeats 2` | 256d09be1 | 2 | 0 | 0 | NOT STORED (see below) |

The Apple reference is NOT committed: the file is 6.9 MB raw and 0.7 MB
compressed, and the run takes ten seconds on the Mac and is bit-repeatable (a
second Apple process the same minute diffed equal on all 86 arrays), so
regenerate it where the diff runs:

    MOJOLEARN_NUMERIC_MODE=identical python3 tools/mamba2_step_probe.py run apple-m4.npz --repeats 2

(A 6.9 MB copy was committed at cdcaf7890 by mistake and removed the next
commit; it stays in history, do not add another.) The AMD runs (a Hot Aisle
MI300X and a DigitalOcean MI325X, `--repeats 20`, twice per box in two
processes) are owed; keep their `.npz` OUTSIDE the tree too (a leg archive or
the R2 store) and record here only each run's box, commit, repeats, in-process
moved count, order-dependent parts, and the first DIFFER line of its Apple diff.
| amd-mi300x-hotaisle (Hot Aisle MI300X 8core VM enc1-gpuvm024, 22.04 ROCm container) | cdcaf7890 | 20 x 2 processes | 0 | none | Apple diff: equal=86 differ=0 of 86 (against an Apple reference regenerated at ad5ac66c2, and the two AMD processes equal=86); the race did NOT reproduce in a fresh process, where the 120-lane run at 65ae7612f had mamba2's step and backward differ on this VM type after about 100 lanes in one process; leg `bench/results/e1g/2026-09-14_140021-amd-mi300x-hotaisle-mamba2-probe` |
| amd-mi325x-do (DigitalOcean MI325X, 24.04 ROCm image) | cdcaf7890 | 20 x 2 attempted | n/a | n/a | both `run`s abort at the first launch: "Memory access fault by GPU node-1 ... Reason: Unknown", exit 134, no npz; the same fault killed the 120-lane run there right after mamba1; leg `bench/results/e1g/2026-09-14_134957-amd-mi325x-do-mamba2-probe` |
| amd-mi300x-hotaisle, warm variant (Hot Aisle MI300X 8core VM, 22.04 container) | 036f25681 | 20 x 3 processes: cold, `--warm poison:8`, `--warm lanes:<the 42 lanes before mamba2>` | 0 in each | none in each | cold vs poison equal=86 differ=0; cold vs lanes equal=86 differ=0; and cold equals the Apple reference (previous row's method). In the SAME leg, `tools/identity_break.py --lanes <the 42>,mamba2 --vendor amd-mi300x-warmprobe` (two fits per cell, one process) gave cells=387 stable=387 moved=0 with mamba2/base at cell 5b05a3ecbd70248e, step 3fa40f29231409be, backward f252b3fc19f5f9f5, the value every H100 and MI300X column on main carries; the diff read it DIVERGENT only because it was held against the Apple M4 column of 2026-09-14_120-lanes (b09925d3d8b074a2, step 16508d9095dbaa6a, backward 7c3fda003a3a0464), the one column on main that differs (correction, same day: DEVIATION 2712 is that Apple column, AMD never disagreed with the H100); leg `bench/results/e1g/2026-09-14_142050-amd-mi300x-hotaisle-mamba2-warm-probe` (the JSON is `remote/mamba2_probe/identity_42_mamba2.json`) |
| amd-mi300x-hotaisle 8core AND 13core, the instrumented identity lane (`MOJOLEARN_IDENTITY_DUMP_DIR`), back to back at 0ecdc1d05 | 0ecdc1d05 | identity_break over the 42 lanes plus mamba2, `--repeats 2`, plus a cold probe run per box | 0 (cells=387 stable=387 on each box) | n/a | 8core fit 0 vs 13core fit 0: equal=23 differ=0 of 23; 8core fit 0 vs the Apple probe: equal=21 differ=0 of 21 (every input, forward, prefill, step, backward gradient and carried state bit-equal); and both boxes' identity JSONs carry mamba2/base at cell 5b05a3ecbd70248e, step 3fa40f29231409be, backward f252b3fc19f5f9f5, equal to the H100 and to every other record on main. The DIVERGENT verdict I first wrote here was the diff holding these against the Apple M4 column of 2026-09-14_120-lanes, which carries b09925d3d8b074a2 (step 16508d9095dbaa6a, backward 7c3fda003a3a0464) and is the one column that differs: DEVIATION 2712 is a per-process wrong answer in that Apple M4 identity run at 65ae7612f, not anything on the HIP path (the commit message of f4c4188e9 says HIP; this row supersedes it). Legs `bench/results/e1g/2026-09-14_1442{34,39}-amd-mi300x-hotaisle-{8core,13core}-mamba2-dump` |
