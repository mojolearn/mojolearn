# lane/lm-export-overlap on AMD gfx942, 2026-09-25

Branch `lane/lm-export-overlap-amd` (on `lane/lm-export-overlap`). This is
the AMD proof of the merged byte LM export lane (device state copied
straight into the caller's buffer) and the overlapped per-step hash of
`tools/lm_segment.py run`, plus the progress PUTs a resume reads from R2.
Scope is the light validation Andrew asked for: the NEW binding only,
everything once. The OLD digests are the H100's
(`bench/results/byte_lm_export_fast_2026-09-25/h100/export_old.json`,
step 101 from the same checkpoint).

## Where it ran

Hot Aisle's 2x MI300X offering had no stock from 17:10 UTC to 20:13 UTC
(and no MI300X offering of any size). The runner waited 170 minutes, then
refused and created nothing (`the 2gpu 2x MI300X spec showed no stock for
170 minutes`); afterwards the team's only VM was A/4's. The leg ran on the DigitalOcean fallback, one
AMD Instinct MI325X (gfx942, droplet 603695135, tor1), so **the two-device
pooled export is still not proven on AMD**. The body
(`tools/byte_lm_export_fast/leg_2gpu_amd.sh`) counts the GPU agents and
runs on devices 0,1 when it finds two. On this box it found one and ran
on device 0 (`do-mi325x/status.txt`). Both bindings were built on the box
from this branch's source (`bindings/build.sh` and
`bindings/build_byte_lm.sh`, `MOJOLEARN_GPU_ARCHS=gfx942`,
`MOJOLEARN_NUMERIC_MODE=identical`, column amd).

## 1. Export digests (box_export.py, NEW, one MI325X)

Every exported array after step 101 has the same sha256 as the H100's, for
both the H100 OLD binding and the H100 NEW binding. All three forms (fresh,
direct and `into=`) agree on all three repeats, and the chain digests equal
route A segment 1's line for step 101 (`do-mi325x/export_new_1dev.json`).

| array | sha256 (MI325X NEW = H100 OLD = H100 NEW) |
|---|---|
| parameters | `189a49eee383450f...` |
| m | `ceaca18311d713f0...` |
| v | `ef5df7ac3c40c184...` |
| flags | `360d579dbd14759b...` |
| gradient | `5caaccdbf79eec9b...` |

Chain digests: state `abc8b816b5c3fb15...`, gradient `25830bfc2016dc14...`.

Export seconds on the MI325X (medians of three):

| seconds | MI325X NEW | H100 OLD | H100 NEW |
|---|---|---|---|
| state, `export_raw()` (allocates) | 0.479 | 5.89 | 1.26 |
| state, binding call into buffers allocated once | 0.046 | 4.17 | 0.152 |
| state, `export_raw(into=...)` | 0.046 | (no `into=`) | 0.144 |
| gradient, `export_gradients()` (allocates) | 0.158 | 1.43 | 0.408 |
| gradient, binding call into a buffer allocated once | 0.019 | 0.900 | 0.052 |
| gradient, `export_gradients(into=...)` | 0.019 | (no `into=`) | 0.052 |
| allocating the four state arrays alone | 0.411 | 1.13 | 1.11 |

No OLD binding was built on AMD, so there is no AMD OLD timing.

## 2. One overlapped run, held to A-1's chain

`lm_segment.py run` (the default overlapped loop) from
`runs/t3/2026-09-22/A/1/ckpt_00000100.blm` (sha256 `80cd2126a89ba6d8...`),
10 steps, `--expect-chain` A-1's witness chain (sha256 `8a6fc036b896...`,
`do-mi325x/expect_chain.sha256`), under the T3 recipe with
`checkpoint_every` set to 2 (`recipe_ck2.json`; `checkpoint_every` is not
part of the data schedule a checkpoint carries), with presigned PUTs under
the scratch prefix `scratch/progress-proof-amd-2026-09-25-do/A/4/`.

Steps 101 to 106 each **agree with the expected chain**. Field by field,
every chain line equals A-1's line except the wall-clock and naming fields
(`seconds`, `hash_seconds`, `label`, `segment`, `route`, and `prev`, which
hashes the previous line with those fields in it): `step`, `lr_f32_hex`,
`losses_f32_hex`, `batch_index`, `hash_scheme`, `schema`, `state_sha256`,
`gradient_sha256` are equal on all six lines. States 101 to 103 are the
H100 replay's `abc8b816`, `a9421f91`, `fcdb48b8`, and gradients
`25830bfc`, `94c40a6e`, `19a43804`.

| step | seconds | `hash_seconds` (overlapped) |
|---|---|---|
| 101 | 71.45 | 0.716 |
| 102 | 72.12 | 0.678 |
| 103 | 72.16 | 3.477 |
| 104 | 69.36 | 0.095 |
| 105 | 72.75 | 3.441 |
| 106 | 69.40 | 0.081 |

Steps 103 and 105 are the steps after a checkpoint (102, 104). Their line
waited about 3.4 s for the checkpoint write of the step before. No
`--sync-hash` run was made (light scope), so there is no synchronous figure
on this box.

Checkpoints written by the overlapped writer: `ckpt_00000102.blm`
`b043a0015be6e0d9...` (1,945,780,168 bytes) and `ckpt_00000104.blm`
`517998d5a64e0132...` (1,945,780,167 bytes). There is no synchronous
checkpoint on AMD to compare them with.

## 3. Kill and resume from R2 (the progress PUTs)

After `progress: chain.progress.jsonl to step 105` the body sent SIGKILL
to the runner (18:43:20 UTC, exit 137, no survivors). The line for step
106 had already been written on the box, but nothing after the progress
PUT reached R2. R2 then held `ckpt_00000102.blm`, `ckpt_00000104.blm`,
`chain.progress.jsonl` (5,922 bytes, to step 105) and
`manifest.progress.tsv`, and no `chain.jsonl`, `manifest.tsv`,
`segment.json` or `ckpt_00000106.blm`.

On the Mac, `lm_run_driver.record_partial` ran against a scratch spec
(`t3_spec.json` with `run` set to the scratch prefix), a fresh ledger in
a scratch directory and an empty results directory, for the plan's A/4
entry with its range set to 100 to 110 (`do-mi325x/record_partial/`). It
found no chain at home, read the chain and manifest from R2's progress
copies, held both checkpoints to R2's object sizes, and recorded
**`resume_from` 104** with checkpoints 102 and 104 landed, the chain for
steps 101 to 105, and nothing refused. Nothing was written under
`runs/t3/`.

One thing seen and not chased: the manifest's progress PUT after step 105
took 72.8 s (the one after step 103 took 0.2 s). It ran on the upload
thread right after the 69.5 s upload of `ckpt_00000104.blm`.

## Not proven here

- The two-device pooled export (m and v from two contexts) and a
  two-device run on AMD. Hot Aisle had no MI300X stock during the window.
- An OLD AMD build, a `--sync-hash` run and `--control split` (all dropped
  by the light scope).

## Cost and teardown

DigitalOcean `gpu-mi325x1-256gb`, droplet 603695135, 18:29:02 to 18:43:50
UTC (861 s of lease). The droplet was deleted and confirmed gone (HTTP 404,
`do-mi325x/teardown.txt`). The Hot Aisle leg waiting for stock created no
VM, and at 20:14 UTC DigitalOcean listed no droplets on the account.

## Files

`do-mi325x/`: `status.txt` (the box's timeline), `export_new_1dev.json`,
`run-new-1dev/` (chain, log, manifest, checkpoint sha256), the input
digests, `files.new.sha256`, `byte_lm.new.so.sha256`, build log tails,
`host.txt`, `gpu.txt`, `mojo_version.txt`, `leg.txt`, `teardown.txt`, and
`record_partial/` (the driver's output, the scratch spec, ledger and
partial chain). On the box the files carried `2dev` in their names; they
were renamed `1dev` here because the box had one GPU. The full leg
directory stays in `~/mojolearn-evidence/lm-export-overlap-amd/do2/`.
