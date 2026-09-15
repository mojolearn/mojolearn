# Lane status: lane/identity-record-next (the 178-lane identity record)

Updated 2026-09-15 13:25 UTC by the record2 agent. Work in progress.

## Goal

The next full identity record at main df617c699 (178 lanes, 39 of them par-*), to
replace `bench/results/identity_break/2026-09-14_166-lanes` as the CPU gate's GPU
columns (`TRAINING_GPU_COLUMNS` in `python/mojolearn/host_surface.py`), and to retire
`TRAINING_FIX_LANES`, `TRAINING_FIX_COLUMNS` and the kmeans-sqrt fix step.

## Done

- NVIDIA H100 one-device column: RunPod 2xH100 pod 2o8sm4yf2we58b, leg
  `bench/results/e1g/2026-09-15_123248-nvidia-2xh100-rec2-all` (four one-device
  processes, two per GPU): 1602 cells, every cell STABLE. Its two par2 processes were
  too slow sharing the GPUs and were stopped after 17 complete lanes.
- AMD MI325X column: DigitalOcean legs `2026-09-15_123228-amd-mi325x-do-rec2-a` (five
  processes on the one GPU) and `2026-09-15_130122-amd-mi325x-do-rec2-m` (rf-reg-gamma-ig
  and par-feature-freq alone, STABLE). Under five processes rf-reg-gamma-ig/base read MOVED
  and par-feature-freq/denormal REFUSED (EOFError); both lanes were dropped from those parts
  and taken from the lone rerun.
- The H100 and MI325X one-device columns diff IDENTICAL=1602 (infer/model IDENTICAL=1998,
  N/A=1206; batch IDENTICAL=1278, N/A=324).
- The first 2xMI300X par2 leg (`2026-09-15_123314-amd-2xmi300x-rec2-par2`, four par2
  processes at once) is NOT evidence: pooled workers failed (RuntimeError: GPU worker
  failed, EOFError) on most cells. Lesson: one par2 process per box.

Evidence copies: `~/mojolearn-evidence/identity-record-next/` (legs/ and scratch/, the
merged parts in scratch/parts/).

## Running at this update

- Apple M4 column: local, one core, `$SP/record2/apple_column.sh df617c6997cdf9f794c073c2d027dc68c9b1191f 15`
  in worktree `$SP/wt-apple-R` (12 fresh processes of 15 lanes), output `$SP/record2/applecol/`.
- 2xMI300X par2 halves, one process each: pods d6guwv4ll9lyii (`...-131207-amd-2xmi300x-rec2-par2-A`)
  and zzeusxtb0d7qmp (`...-131339-amd-2xmi300x-rec2-par2-B`).
- 2xH100 par2 makeup, the 22 missing par lanes in one process: `...-131631-nvidia-2xh100-rec2-par2m`.

$SP is `/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/4e8829df-5da1-4c85-8680-6b52df16132c/scratchpad`.
Leg dirs live under `$SP/wt-legs-R/bench/results/e1g/`.

## Next

1. Merge the parts: `python3 tools/identity_break.py --merge <parts> --json <column>.json [--allow-separate-builds]`
   (Apple chunks; NVIDIA par2 p1/p2 minus their unfinished lane plus par2m; AMD par2 A and B).
2. Write `bench/results/identity_break/2026-09-15_178-lanes/` with the three-column diff, both
   par two-device diffs, the five-column par diff, the batch diff and the diff against the 166-lane record.
3. Point `TRAINING_GPU_COLUMNS` and the gate's GPU-column step at it with exact counts (seen to fail
   with a wrong count first), remove `TRAINING_FIX_*` and the kmeans-sqrt fix step, update
   `python/mojolearn/tests/test_cpu_training_misc.py`.
4. Seven-runner gate on the branch, then merge to main.
