# 178 lanes at df617c699 (2026-09-15): three complete columns, a PARTIAL record

**This record is partial, and nothing reads it.** `TRAINING_GPU_COLUMNS` in
`python/mojolearn/host_surface.py` and the CPU identity gate still name
`2026-09-14_166-lanes`. The run stopped on Andrew's order of 2026-09-15 ~14:00 UTC:
GPU records are required only for PyPI releases, and then only one Apple, one NVIDIA
and one AMD column; no two-device columns and no extra vendors. Legs already running
were allowed to finish; no box was rented after the order.

What is complete: the three one-device columns (Apple M4, NVIDIA H100, AMD MI325X), all
178 lanes of `tools/identity_break.py` at df617c699 (39 of them par-*), nine hostile
fixtures, two fits per cell, the train, infer, model and batch parts. What is not: the
two-MI300X column lacks five par lanes, and the gate switch was written, run locally and
then left unmerged (`unmerged-gate-switch.patch`).

| column | box | how | cells |
|---|---|---|---|
| apple-m4 | Apple M4 (Metal), this Mac, one core, shared machine | 44 builds in a detached worktree at df617c699, then twelve fresh `--vendor apple-m4` processes of 15 lanes each (270 to 667 s), joined with `--merge` | 1602 stable, complete |
| nvidia-h100-sm_90a | RunPod pod 2o8sm4yf2we58b, two H100s, one build | leg `bench/results/e1g/2026-09-15_123248-nvidia-2xh100-rec2-all`: four one-device processes at once, two pinned to each GPU (`CUDA_VISIBLE_DEVICES`), joined with `--merge` (one build) | 1602 stable, complete |
| amd-mi325x-gfx942 | DigitalOcean MI325X, two legs | `...-123228-amd-mi325x-do-rec2-a` (five processes at once on the one GPU, 806 s) and `...-130122-amd-mi325x-do-rec2-m` (rf-reg-gamma-ig and par-feature-freq alone), joined with `--merge --allow-separate-builds` (8 of 23 binding digests differ between the two builds; `_mojolearn_rf` does not) | 1602 stable, complete |
| nvidia-2xh100-sm_90a.par-devices-0-1 | RunPod two H100s, `MOJOLEARN_PAR_DEVICES=0,1` | 17 lanes from two par2 processes on the `rec2-all` pod (stopped between lanes, too slow sharing the GPUs with the one-device processes) and 22 lanes from one process on pod wnjvi7fhvdi0w1 (`...-131631-nvidia-2xh100-rec2-par2m`, 1725 s); the two builds are byte identical | 351 stable, all 39 par lanes |
| amd-2xmi300x-gfx942.par-devices-0-1 | RunPod two MI300X, `MOJOLEARN_PAR_DEVICES=0,1`, one process per pod | `...-131207-amd-2xmi300x-rec2-par2-A` (20 lanes, 2064 s) and `...-131339-amd-2xmi300x-rec2-par2-B` (cut by its poll deadline after 14 of 19 lanes); 11 of 23 digests differ, `--allow-separate-builds` | 306 stable, 34 of 39 par lanes |

Missing from the two-MI300X column: par-samba-clip, par-gmm, par-hdbscan,
par-kernel-ridge and par-rbf-sampler. The `--diff` NOTE lines call both two-device
columns INCOMPLETE because a par-only merge does not cover every lane; the NVIDIA one
carries all 39 par lanes.

## Verdicts

`diff.three-columns.txt` (`--require-columns 3`, OK over every lane):
`summary: IDENTICAL=1602`, `summary (infer/model): IDENTICAL=1998, N/A=1206`,
`summary (batch): IDENTICAL=1278, N/A=324`. No DIVERGENT, MOVED, BATCH_MOVED or
REFUSED cell on any column. `diff.batch.txt` is its batch rows.

`diff.par-two-devices-nvidia.txt` (apple-m4, the one-device H100, the two-H100 column,
the 39 par lanes): `summary: IDENTICAL=351`, infer/model `IDENTICAL=423, N/A=279`,
batch `IDENTICAL=261, N/A=90`.

`diff.par-two-devices-amd.txt` (apple-m4, the MI325X, the two-MI300X column, its 34
lanes): `summary: IDENTICAL=306`, infer/model `IDENTICAL=378, N/A=234`, batch
`IDENTICAL=225, N/A=81`. `diff.par-five-columns.txt` puts both two-device columns beside
the three one-device columns over those 34 lanes with `--require-columns 5`: the same
three summaries, IDENTICAL x5. The drivers' cells on two H100s and on two MI300X GPUs
are the same bits as one device on each vendor and the M4, for these fixtures and the
smallest shardings; it is not a throughput statement.

## Against the 166-lane record

`diff.166-lanes-vs-178-lanes.txt` puts the 166-lane record's three columns (1eea14f80)
beside these three: `summary: DIVERGENT=9, IDENTICAL=1593`, infer/model
`IDENTICAL=1998, N/A=1206`, batch `IDENTICAL=1278, N/A=324`. The twelve lanes the old
record lacks (ivf, ivf-euclidean, embedding, embedding-sort and eight par lanes) count as
IDENTICAL x3 there on the new columns alone. The nine DIVERGENT cells are
`kmeans-sqrt` on every fixture, and each is the merged fix 9fde8f5f7 (DEVIATIONS 2715
and 2716): the labels part moved on all nine fixtures (on `base`, old `961a4ac1fc219750`
on all three old columns, new `8864be3c295d6c4a` on all three new ones), and on `wide`
the old H100 inertia that stood alone is gone (old cell `8c66b352864fd354` on the M4 and
MI325X and `152655441c24b80a` on the H100, inertia `52ea06cbbcc24144` and
`1a7e4ac5b8c0caaf`; new cell `4bde18fe714ad92e` with inertia `52ea06cbbcc24144` on all
three). Read these hashes from the JSONs, not the table: with six JSONs carrying the
same three vendor labels twice, the diff prints five hashes on each `kmeans-sqrt` row
under six header names, so its hash columns do not line up with the header (not
investigated). Every other training, infer, model and batch cell of
the old record is unchanged. The new `kmeans-sqrt` cells equal the
`2026-09-14_kmeans-sqrt-fix` record's three columns on all nine fixtures (the eight
k-means lanes of that record diff IDENTICAL=72 across the six columns).

## Contention findings (processes sharing one box)

To fit the budget, several identity processes ran at once on one box. That cost more
than it saved, and it found one thing worth a brief:

- On the MI325X with five processes on one GPU, `rf-reg-gamma-ig/base` read MOVED (the
  second fit's inverse_gaussian part) and `par-feature-freq/denormal` REFUSED
  (`EOFError: Ran out of input`). Both lanes reran alone and read STABLE with the recorded
  hashes; the column takes them from that rerun and the dropped parts say so in
  `truncated_note`. Brief: `docs/lanes/BRIEF_rf_reg_gamma_ig_moved_under_contention_2026-09-15.md`.
- On two MI300X with four par2 processes at once (`...-123314-amd-2xmi300x-rec2-par2`),
  most par cells refused (`RuntimeError: GPU worker failed`, `EOFError`). Nothing from
  that leg is in this record.
- On two H100s with four one-device and two par2 processes at once, every cell read
  STABLE and IDENTICAL to the other columns, but the par2 processes were slow enough that
  they were stopped and finished on a second pod.

One par2 process per box is the working shape.

## The gate switch that was not merged

`unmerged-gate-switch.patch` (against df617c699) points `TRAINING_GPU_COLUMNS` and the
gate's GPU-column step at this directory, removes `TRAINING_FIX_LANES`,
`TRAINING_FIX_COLUMNS`, `fix_covered_lanes`, the `--fix-covered-lanes` and
`--training-fix-columns` flags and the kmeans-sqrt fix step, and rewrites the misc tests.
Locally on the M4, one core: the step as written passed on these columns; a copy
expecting `IDENTICAL=1601` failed, and the step fed the 166-lane columns failed
(`summary: DIVERGENT=1, IDENTICAL=1493`, exit 1); `test_host_surface` and
`test_cpu_training_misc` passed (112), and each of the three new tests failed on a
sabotaged input (the 166-lane columns, a restored `TRAINING_FIX_LANES`, a workflow still
reading `--training-fix-columns`). No CI gate ran it. If a release ever wants these three
columns, apply the patch after checking it against main.

## Cost

RunPod: two 2xH100 pods (43 and 38 minutes at $6.98/h, $9.48) and three 2xMI300X pods
(32, 41 and 56 minutes at $4.78/h, $10.24). DigitalOcean: two MI325X droplets (19 and 5
minutes). About 7.4 GPU hours, of which the four-process MI300X pod (about 1.1 GPU hours)
produced nothing usable.

## Reproduce

    python3 tools/identity_break.py --diff apple-m4.json nvidia-h100-sm_90a.json amd-mi325x-gfx942.json --require-columns 3
    python3 tools/identity_break.py --merge part_a.json part_b.json --json column.json [--allow-separate-builds]

The raw leg archives are not committed; copies are in `~/mojolearn-evidence/record2/`.
