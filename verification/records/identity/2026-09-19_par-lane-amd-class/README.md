# The AMD column for the thirteen `par-*` lanes, and the two-device half beside it

One lease. Pod `162bilnfsj8nmn`, 2x AMD Instinct MI300X (gfx942, amdgpu
6.10.5), RunPod SECURE EU-RO-1, `"gpuCount": 2` in the create request, $4.78/h,
16:39:09Z to 17:21:05Z = 42 minutes = $3.34. Body:
`tools/two_device_par_class_amd_leg.sh`, built from source on the box at commit
`331cdfaa32df4927dc3c5adc9a4c014820200289`. `rocminfo` reported
`visible_agents=2` and the body refuses PHASE TWO by name if it does not.

| file | `package.par_devices` | `_verify_reference.admit` | cells |
|---|---|---|---|
| `amd-amd-instinct-mi300x-gfx942.par-one.json` | `0` | ADMISSIBLE | 117 STABLE, 0 moved, 0 refused |
| `amd-amd-instinct-mi300x-gfx942.par-two.json` | `0,1` | `par_devices 0,1` | 117 STABLE, 0 moved, 0 refused |

117 = 13 lanes x 9 fixtures, `--repeats 2`, default fixture size, no sabotage
switch set anywhere in the run.

## A two-device column is not what closes a matrix gap

`python/mojolearn/_verify_reference.py:303` refuses any column whose
`package.par_devices` is not `"0"`, and `tools/verification_matrix.py:262`
builds its GPU coverage "from ADMITTED clean columns". A column recorded with
`MOJOLEARN_PAR_DEVICES=0,1` is therefore invisible, by construction, to the
count it appears to answer. The thirteen lanes were not waiting on a second GPU
to be COUNTED; they were waiting on a second DEVICE CLASS.

Eleven of them carried `nvidia` ALONE, all eleven from one column,
`bench/results/e1g/2026-09-17_203449-nvidia-refregen-par-lanes-b/remote/identity/`.
`par-one.json` is the AMD column they were missing and it takes those eleven to
two classes. The other two, `par-forest-pool` and `par-rbf-sampler`, already
carried `amd` and `nvidia`; the only class they lack is `apple`, which is a
Metal run on the one Mac and cannot be rented at any price. NO LANE HERE
REACHES THREE CLASSES AND NONE CAN WITHOUT AN APPLE COLUMN.

`par-two.json` is the drivers' own claim -- `identity_break._par_devices`:
"A two-device column ... must hash equal, cell for cell, to the one-device
column of the same commit; that equality is the drivers' whole claim". It is
worth recording only because it came off the SAME build on the SAME box at the
SAME commit as `par-one.json`, which is the pairing two leases cannot give.

## What the columns say

One device vs two, run on the box while it was still rented
(`identity_break.py --diff`, exit 0), and again here:

    summary:                 IDENTICAL=117
    summary (infer/model):   IDENTICAL=216, N/A=18
    summary (batch):         IDENTICAL=108, ONE-COLUMN=9

And the new AMD one-device column against the existing NVIDIA one-device column
over the eleven lanes they share:

    summary:                 IDENTICAL=99
    summary (infer/model):   IDENTICAL=180, N/A=18
    summary (batch):         IDENTICAL=99

## THE ONE THING THAT IS NOT CLEAN: par-queries-nn's batch part on two devices

`ONE-COLUMN=9` above is not a missing cell. It is nine REFUSED ones: on TWO
devices, and only on two, `par-queries-nn`'s batch part fails on every one of
the nine fixtures with

    batch: RuntimeError: GPU worker failed:
      File "python/mojolearn/_parallel_worker.py", line 386, in main
        response = (True, execute(request))

The same part on ONE device, same build, same box, produces a stable hash on
all nine (`batch_verdict: STABLE`), and this lane's train, infer and model
cells are IDENTICAL one-vs-two. Nothing here is flaky: the body runs strictly
one process at a time and the failure reproduced on 9 fixtures x 2 repeats =
18 serial attempts. The harness truncates the worker traceback at 300
characters, so the cause inside the worker is NOT known from this run and is
not guessed at here. `logs/column-two.log` in
`~/mojolearn-evidence/e1g/2026-09-19_amd-2gpu-par-class/` has what there is.

## Already discharged before this lease, and so not bought twice

`bench/results/identity_break/2026-09-19_hardware-gaps/` carries two-device
columns for eleven of the thirteen on BOTH vendors. All 99 of their train cells
equal the one-device NVIDIA column. What that set has no column for is any AMD
ONE-device run of these lanes, or either of `par-forest-pool` and
`par-rbf-sampler` at a recent commit. Those two gaps are what this lease bought.
