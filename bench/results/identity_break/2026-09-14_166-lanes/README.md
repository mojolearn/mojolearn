# 166 lanes, three columns and two two-device columns, with the batch part, at 1eea14f80 (2026-09-14)

The record `python/mojolearn/host_surface.py` names as TRAINING_GPU_COLUMNS from
this commit on, replacing `2026-09-14_136-lanes`. Commit 1eea14f80 is main after
the harness lane (`lane/harness-sep14`, gate run 34901563040 green on seven
runners): the batch part of `tools/identity_break.py` (whole batch against each
of the first 16 rows alone, a 1/7/rest split, and sequence prefixes and forecast
horizons, `summary (batch):`), the coordinate descent predict asked in windows of
two because it refuses one row by name, and fifteen more `par-*` lanes for the
multi-GPU drivers the first sixteen missed (the query drivers, reference-sharded
KNN, the graph drivers, OrderedRMSE, the two-level feature-frequency fit, the
pointwise searcher, Holt-Winters series shards, the layer-owned and offloaded
byte-LM trainers, and pooled neural clipping). Nine hostile fixtures, two fits
per cell, train, infer, model and batch parts; every binding (37 build scripts)
rebuilt from this commit on each box. The four `par-*` lanes the multigpu lane
added to main after 1eea14f80 (170 lanes) are not in this record.

| column | box | how | cells |
|---|---|---|---|
| apple-m4 | Apple M4 (Metal), this Mac, one core, shared machine | 37 builds in a detached worktree at 1eea14f80, then `--vendor apple-m4`; see "the Apple column's processes" | 1494 stable, complete |
| nvidia-h100-sm_90a | RunPod pod with two H100s, one process per GPU (`CUDA_VISIBLE_DEVICES=0` and `=1`), one build | leg `bench/results/e1g/2026-09-14_223031-nvidia-2xh100-ib166-halves`, the lanes interleaved in two halves, joined with `--merge` (same binding digests) | 747 + 747 stable, 1006 s |
| amd-mi325x-gfx942 | DigitalOcean MI325X, three sequential legs | `...-223513-amd-mi325x-do-ib166-a` (693 cells, the 3037 s bound cut it after 77 of its 83 lanes), `...-232946-...-b1` (the six unfinished lanes plus 40, 414 cells, 826 s) and `...-235741-...-b2` (43 lanes, 387 cells, 540 s), joined with `--merge --allow-separate-builds`; complete by coverage | 1494 stable |
| nvidia-2xh100-sm_90a.par-devices-0-1 | RunPod pods with two H100s, `MOJOLEARN_PAR_DEVICES=0,1` | `...-223449-nvidia-2xh100-ib166-par2-old` (16 lanes, 848 s) and `...-230725-nvidia-2xh100-ib166-par2-new-b` (15 lanes, 980 s); the two builds were byte identical | 279 stable |
| amd-2xmi300x-gfx942.par-devices-0-1 | RunPod pods with two MI300X, `MOJOLEARN_PAR_DEVICES=0,1` | `...-223804-amd-2xmi300x-ib166-par2-0` (11 lanes, 676 s) and `...-230723-amd-2xmi300x-ib166-par2-rest` (20 lanes, 1264 s); 11 of 21 binding digests differ between the two builds, so `--allow-separate-builds` | 279 stable |

## Verdicts

`diff.three-columns.txt` (`--require-columns 3`, OK over every lane):
`summary: DIVERGENT=1, IDENTICAL=1493`, `summary (infer/model): IDENTICAL=1899,
N/A=1089`, `summary (batch): IDENTICAL=1188, N/A=306`. No MOVED, no BATCH_MOVED,
no RELOAD-MOVED, no REFUSED cell on any column.

The one DIVERGENT cell is `kmeans-sqrt/wide`, part `inertia` only (centers,
labels and scales agree): Apple and AMD hash 52ea06cbbcc24144, the H100
1a7e4ac5b8c0caaf. The H100 column stands alone; no earlier record carries the
lane. Brief: `docs/lanes/BRIEF_kmeans_sqrt_wide_h100_inertia_2026-09-14.md`.

`diff.batch.txt` is the batch rows of that diff: 1188 batch cells IDENTICAL x3
(the whole-batch hash equal on every vendor, and on every vendor every row alone,
the split and the prefixes the same bytes as the whole batch), 306 N/A with the
lane's reason. `apple-m4.batch-sabotage.json` is the part seen to fail at this
commit: with `MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1` all five cells it ran (lasso,
holtwinters, standard-scaler, byte-lm, par-queries-knn on `base`) read BATCH_MOVED
and the run exited 1. It is not evidence and no diff reads it.

`diff.par-two-devices-nvidia.txt` and `diff.par-two-devices-amd.txt` (the 31
`par-*` lanes, `--require-columns 3`): apple-m4, the one-device column and the
two-device column of each vendor read `summary: IDENTICAL=279`, `summary
(infer/model): IDENTICAL=360, N/A=198`, `summary (batch): IDENTICAL=207, N/A=72`.
`diff.par-five-columns.txt` puts both two-device columns beside the three
one-device columns: IDENTICAL x5 on all 279 training cells. What that says: the
drivers' cells on two H100s and on two MI300X GPUs are the same bits as one H100,
one MI325X and an Apple M4 running one device, for these fixtures and the
smallest shardings; it is not a throughput statement.

`diff.136-lanes-vs-166-lanes.txt`: the three 136-lane columns at 4048e1b51 beside
these three. Every training cell the old record carries is IDENTICAL x6, and so
is every infer cell. The 27 DIVERGENT model cells are the saved checkpoint BYTES
of samba, samba-untied-dropout-accum and par-samba: each record agrees with
itself on all three columns, and main changed the Samba checkpoint format after
4048e1b51 (b93f1bc52, "Stream Samba checkpoints beyond the JSON size limit").

## The Apple column's processes

The first Apple process (`apple-m4.proc1`, not committed) ran 106 lanes. At
`gbdt-multiclass/negative` its Metal command stream stopped executing: GBDT fits
refused by name through the DEVIATION 2002 canary ("the device did not execute
the fit's command stream"), and every later GPU lane read back infinities, zero
Jacobi rotations or MOVED hashes (kmeans-random/denormal, dbscan-brute-l1/base,
radius-minkowski-p3/odd). That process was stopped, its cells from
gbdt-multiclass on were dropped (the part file records the dropped lanes in
`truncated_note`), and those lanes reran at the same commit and build in six
fresh processes of at most 20 lanes (753, 165, 273, 73, 283 and 581 s), every
cell STABLE. The 52 lanes kept from the first process agree with both GPU
columns. The 136-lane Apple column ran every lane in one process without the
batch part; whether the batch part's many small calls exhaust something in a
long Metal process is not measured.

## Reproduce

    python3 tools/identity_break.py --diff apple-m4.json nvidia-h100-sm_90a.json amd-mi325x-gfx942.json --require-columns 3
    python3 tools/identity_break.py --merge part_a.json part_b.json --json column.json [--allow-separate-builds]

The raw leg archives stay under `bench/results/e1g/` in the harness worktree and
are not committed.
