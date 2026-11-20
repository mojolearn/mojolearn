# Multi-GPU driver checks

Optional. Not required for any merge or release. Not shipped to users.

(Andrew, 2026-11-20.) This is a developer suite in the repository. Nothing under `python/mojolearn/` belongs to it,
`python -m mojolearn verify` has no option for it, and no release gate, ledger row, admission step, CI trigger, merge
gate or push hook reads its result. `tools/test_multi_gpu_checks_optional.py` fails if a release, gate, admission,
workflow, hook or packaged file names it. The multi-GPU capability itself (the `mojolearn.parallel_*` drivers,
`ParallelNeuralTrainer`, the byte-LM trainers, `ParallelCausalLM`, `DistributedProgram`) is product code and is not
touched by this suite.

## What it checks

`tools/dev/multi_gpu_checks.py` runs each multi-GPU driver through its public entry on the installed wheels, on a box
with two or more GPUs of one vendor, in three arms, each in its own process:

| arm | devices | purpose |
| --- | --- | --- |
| `one` | one GPU (rotating over the box's GPUs) | the reference |
| `multi` | every GPU of the box | must give the same output digest as `one` |
| `fault` | every GPU, with a fault switched on | must give a different digest: the check is shown able to fail |

The digest is sha256 over each named output's raw bytes with its shape and dtype. One run per arm. A unit is `OK`
when `one` equals `multi`, the multi arm's work was seen on two or more ranks (for the units that have a placement
witness) and the fault was seen. `DIFFER` is the defect the suite exists to find. Anything that raises, times out or
returns nothing is `FAIL`, never a skip.

There are 54 units over 14 driver modules; `python3 tools/dev/multi_gpu_checks.py --list` prints them with their
witness and fault kind.

The fault arm is one of two kinds:

- `shift`: the drivers' own native negative control. With `MOJOLEARN_PAR_DRIVER_SABOTAGE=1` and
  `MOJOLEARN_HOST_ALLOW_SABOTAGE=1` every shard on an owner above rank 0 reads one position early. It does nothing on
  one device, so a differing digest also proves that a second owner did work. Used by the 15 units whose entries
  honour it (scalers, RBF sampler, forest fits, `ParallelQueries`, `ReferenceShardedNeighbors`, ARIMA and
  Holt-Winters fits).
- `input`: the other 39 units have no device-level fault switch in a shipped binary. Their fault arm runs the multi
  arm on a perturbed fixture. That shows the digest covers a computed result and that the verdict can read differ. It
  does not show that a device-level defect would be caught. A device-level fault for those drivers needs a native
  switch, which does not exist.

## What it does not check

- The plain one-device estimator. Both arms go through the driver, so a driver that is wrong on one device and equally
  wrong on N passes. The removed `par-*` lanes compared each driver with the plain fit; that comparison is not here.
- Mixed card models in one plan, two vendors in one box, Apple, and the CPU column.
- `OffloadedByteLanguageModelTrainer` (one device by design), `par-boosting-pointwise`, `par-border-types` and
  `par-ordered` configurations of `fit_boosting` (the driver is covered by the two boosting units).
- The Mojo-level `training/checks/*_parallel_check.mojo` programs. They need a source build on the box;
  `tools/two_device_checks.sh` runs them and the job below can call it as a second phase (`MGC_MOJO_CHECKS=1`).

## How to run it on a two-GPU pod

The orchestrator rents the pod and starts the job; lanes never do. The job runs under `tools/box_persist_run.sh`
through the usual wrappers, so its results reach R2 from the box.

```
# NVIDIA, two GPUs, the wheels of one build already on the box in <dir>
tools/nvidia_central.sh submit <lane> --gpus 2 --results <res> --env MGC_WHEELS=<dir> \
    <tree-on-box>/tools/ops/multi_gpu_checks_job.sh

# AMD
tools/do_amd_steward.sh run <tag> <res> env MGC_WHEELS=<dir> bash tools/ops/multi_gpu_checks_job.sh

# by hand on the box, with an interpreter that has the wheels installed
python3 tools/dev/multi_gpu_checks.py --python <venv>/bin/python --out <dir>
```

Useful settings of the job (environment): `MGC_PYTHON` (an interpreter instead of `MGC_WHEELS`), `MGC_UNITS`
(a comma-separated subset), `MGC_DEVICES`, `MGC_ARGS` (extra flags such as `--strict-spread`, `--no-fault`,
`--jobs N`, `--host-gb-per-job G`, `--gpu-gb-per-job G`).

Reading the result: `job/job.log` ends with the plan line, the summary line, every unit that is not `OK`, peak device
memory and `busy cores: mean X, max Y of N`. `out/multi_gpu_checks.json` holds every unit's verdict, digests, ranks,
seconds and peak memory; `out/arms/<unit>.<arm>.log` is the full log of one arm.

## Parallel plan

Every (unit, arm) is an independent process: 162 of them for the full list. They run at once on a pool whose size is
the smallest of the arms, the allocation's cores (cgroup quota and affinity, not the host's CPUs), the host memory
bound and the device memory bound. The `one` arms are spread over the GPUs; `multi` and `fault` arms use all of them.
The two per-arm memory figures (2 GiB of host memory, 1.5 GiB of device memory per GPU) are estimates. After the first
run, read "largest worker RSS" and `job/gpu-peak.txt` and pass measured values. If busy cores sit far below the
allocation, the device memory bound is the likely cause: lower `--gpu-gb-per-job` to the measured figure.

## History

Multi-GPU verification (59 `par-*` identity lanes, `verify --par`, two optional ledger rows) was retired on
2026-10-31: `docs/TOMBSTONES.md`, entry `MOJOLEARN_PAR_VERIFY`, patch `experiments/removed/MOJOLEARN_PAR_VERIFY.patch`.
That record is permanent and unchanged. This suite restores the driver calls of those lanes as recipes and none of
the verification machinery. The 531 reference cells and 5 lane revisions that the shipped verify table still carried
for the 59 lanes were moved, byte for byte, to `tools/dev/multi_gpu_retired_par_cells.json`; `RETIRED_PAR_LANES` in the
tool maps each of the 59 names to its unit or says why it has none. No tool reads those cells as a reference.
