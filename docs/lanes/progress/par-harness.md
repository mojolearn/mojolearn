# lane/par-harness progress

Goal: the light lane check gives every `par-*` lane a real verdict (two-device
column vs one-device column, with the device witness) instead of a CPU refusal.
Branch lane/par-harness from origin/main 243f74dc7. Not merged to main.

## What changed

- `tools/algos_lane_check.py`: the `par-*` DEVICE AXIS.
  - `device_count(backend)`: metal is 1 by structure; CUDA/HIP ask
    `python/mojolearn/_gpu_witness.py` in a child (honors the queue slot's mask).
  - `par_axis(lane, backend)`: a `par-*` lane on a CUDA/HIP box with >= 2 GPUs.
  - `run_arm`: on the axis the "gpu" slot is the TWO-device run
    (`MOJOLEARN_PAR_DEVICES=0,1`) through `tools/par_witness_arm.py`, and the
    "cpu" slot is the ONE-device GPU run (`MOJOLEARN_PAR_DEVICES=0`); no CPU arm
    and no host binding. An ambient `MOJOLEARN_PAR_DEVICES` is dropped from
    every arm. Off the axis a `par-*` CPU arm may exit 1 with its JSON so
    `compare` can judge it.
  - `compare`: on the axis, AGREE needs bitwise equal columns AND a witness
    record with no refusal for every cell (`par_witness_verdict`), else
    WITNESS REFUSED; a cell refused on only one column is DISAGREE;
    ONE_DEVICE_BY_DESIGN (par-byte-lm-offload) that agrees is NOT APPLICABLE.
    Off the axis the by-design cooperative CPU refusal is NOT APPLICABLE
    ("needs 2 devices"), replacing KNOWN REFUSAL.
  - `PASSING = ("AGREE", "NOT APPLICABLE")`; the RESULT line names NOT APPLICABLE
    lanes separately. A sabotage cannot pass on a NOT APPLICABLE lane.
- `tools/par_witness_arm.py` (new): runs identity_break in process with one
  `_verify_par.PoolWitness` per cell (bounded by `identity_break.CellTimer`),
  writes `<column>.witness.json`.
- `tools/test_algos_lane_check_par.py` (new): verdict logic, arm commands on
  1- and 2-GPU boxes, the Mac NOT APPLICABLE path, per-cell witness bounds.
- `tools/par_harness/sabotage_par_read_shift.patch`: `driver_read_shift`
  returns 1 without the env switch, so every owner above rank 0 reads its
  slice one position early (inert at one device). Used on par-arima.
- `tools/par_harness/job.sh`: the one light job.

## Consumers

- The untracked consolidation driver (apple-merged tools/merged_check/merged_check.py)
  inherits the arm choice and the verdicts through `alc.run_arm` / `alc.compare`,
  but its `clean` counts every verdict other than AGREE as bad. It needs one
  change to accept `alc.PASSING`, and its jobs must be submitted with
  `--gpus 2` for the par-* lanes to be checked (a 1-GPU slot reads NOT APPLICABLE).
  Not edited here (not this lane's file).

## Job

(pending)

## Verdicts (2-GPU box, base fixture)

(pending)

## Sabotage

(pending)

## Unproven

- Fixtures other than base.
- The Mac path is shown by unit test only (no Mac job, by rule).
