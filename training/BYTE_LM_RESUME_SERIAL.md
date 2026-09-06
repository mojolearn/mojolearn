# Bounded cross-vendor byte-LM continuation campaign

`tools/byte_lm_resume_serial.sh` is a source-authored helper for the root agent
on an already available remote Linux NVIDIA or AMD host. It performs no
installation, build, rental, architecture discovery, or transport. No subagent
may execute it. Its source has not been run or qualified by the author.

The campaign produces either a fresh first-64-step checkpoint or a foreign
checkpoint continuation through step128 followed by the one-step missing-
moments control. It preserves the existing fixed corpus, initialization,
optimizer, schedule, numerical kernels and capture format. A new directory is
required for every invocation, and an existing artifact is never replaced.

## Prerequisites

Each vendor must already have its complete successful continuous128 campaign
from `byte_lm_validation_serial.sh`, including:

- `full128/`, `step1/`, `gradient-oracle.json` and its raw reference archive;
- the original model/oracle root receipts and their command/log/result files;
- all twelve successful campaign jobs, `results.tsv`, `exit_code`, and the
  retained `bindings/_mojolearn_byte_lm.so`;
- `leg.txt` and `source_inventory.json` one directory above the campaign, as
  expected by `byte_lm_validation_admit.py`.

Use the baseline's existing Python environment and the same numerical source
tree. The helper compares the installed
`python/mojolearn/identical/_mojolearn_byte_lm.so` byte-for-byte with the
retained baseline binding, and checks its SHA against the continuous capture.
It does not rebuild or install a replacement. The same source inventory must
match exactly; vendor-specific binaries may differ across CUDA and HIP, but
each continuation must match its own vendor's baseline binary.

The helper itself and this document are new orchestration files, outside the
capture's numerical source inventory. They can be copied to the baseline
source tree without changing numerical source. The campaign retains the exact
helper source and generated file-check program, with SHA witnesses. If the
numerical source has changed, preflight refuses; do not loosen its comparison
to reuse an incompatible baseline.

## Root sequence

The environment and paths below are examples to fill from the actual retained
campaign. GPU architecture is an explicit root-provided value, never inferred.
Set `MOJOLEARN_PYTHON` to the existing baseline environment's absolute Python
path. These commands are for the remote host only.

First, on the NVIDIA host with its successful continuous campaign:

```sh
export MOJOLEARN_BYTE_LM_EXPECT_VENDOR=cuda
export MOJOLEARN_GPU_ARCHS=sm_ACTUAL_ARCH
export MOJOLEARN_PYTHON=/absolute/baseline/environment/bin/python
unset MOJOLEARN_BYTE_LM_FOREIGN_SHA256
bash tools/byte_lm_resume_serial.sh head64 \
  /absolute/nvidia/byte-lm-validation /absolute/new/nvidia-head
```

Preflight performs file-only admission of the baseline. The guarded head run
then trains the unchanged schedule through step64, writes its canonical full-
state checkpoint, and receives a root receipt only after the actual guard
exits zero. A final guarded file check compares every overlapping raw step
against the NVIDIA continuous run. Read `exit_code`, `results.tsv` and
`verify-head.json` together. The checkpoint SHA is retained in
`verify-head.json` as `head_checkpoint_sha256`.

The root transport controller must copy the **actual complete head campaign**
to AMD, preserving all receipt-relative files. Retain the source checkpoint
SHA before transfer, and compare it with the destination bytes. Do not replace
this transfer with locally regenerated weights or a same-seed reconstruction.
The helper does not transport files or rent another machine.

Next, on AMD with its independently successful continuous campaign and the
transferred NVIDIA head campaign:

```sh
export MOJOLEARN_BYTE_LM_EXPECT_VENDOR=hip
export MOJOLEARN_GPU_ARCHS=gfxACTUAL_ARCH
export MOJOLEARN_PYTHON=/absolute/baseline/environment/bin/python
export MOJOLEARN_BYTE_LM_FOREIGN_SHA256=ACTUAL_RETAINED_NVIDIA_HEAD_CHECKPOINT_SHA256
bash tools/byte_lm_resume_serial.sh resume128 \
  /absolute/amd/byte-lm-validation /absolute/new/amd-resume \
  /absolute/transferred/nvidia-head
```

The reverse direction works with the vendor arguments exchanged. One complete
direction satisfies this helper's bounded continuation experiment; it does
not imply that both directions have been executed.

## Transfer and state checks

Before any new model call, preflight verifies the foreign head's successful
capture receipt, foreign vendor, source/profile/configuration and root-pinned
checkpoint SHA. It compares all first64 raw steps against the receiving
vendor's continuous run. A mismatch stops before continuation.

The actual transferred checkpoint is copied exclusively into
`transferred-head64.checkpoint.json`, published read-only with mode0444, and
its bytes/SHA and source receipt linkage are retained in `preflight.json`.
Read-only permissions are an operational protection, not a claim of protection
against a privileged writer. Every capture loads these bytes through the
existing sealed-memfd checkpoint path and retains its own exact
`incoming.checkpoint.json`. Postchecks require all these bytes to equal the
foreign head's canonical checkpoint; the read-only copy and binding are
rechecked as well. The full checkpoint carries parameters, both moments,
flags, step, optimizer settings and the dataset schedule cursor.

The `resume128` run consumes batches64–127 and retains steps65–128. It is
checked against the corresponding receiving-vendor continuous steps and its
terminal checkpoint before the control is allowed to run.

The `zero-moments65` control loads the same foreign checkpoint, retains the
legitimate head state, then zeros only `m` and `v` and runs the same step65.
The existing comparator requires unchanged initial weights/flags/counters,
unchanged tokens/loss/gradients, and different post-step parameters **and**
first moments **and** second moments relative to legitimate step65. Nonzero
incoming moments and all canonical state signatures are checked. A control
that does not change those three post arrays is refused.

## Bounds and cancellation

Model and file-check jobs run serially through the existing vendor guard,
using its global exclusion lock, two child CPU cores/threads, 12GiB RSS cap,
GPU occupancy check, free-memory checks and GPU-memory ceiling. There is no
parallel model work. Default campaign time is2100 seconds, configurable through
`MOJOLEARN_BYTE_LM_RESUME_SECONDS` within120–2400 seconds. Each job's cap is
reduced by the remaining campaign deadline; no job starts with less than15
seconds remaining. The final30 seconds are reserved outside job allocation.

On TERM/HUP/INT the shell forwards cancellation to its active guard and waits
for its process-group cleanup before writing the campaign exit code. Each
model guard return status is recorded before a root receipt can be written.
No completed summary alone counts as successful teardown. Do not wrap the
whole helper inside another instance of the same guard: its per-job guard
already owns serialization and nested locks would refuse.

Capture tree and file bounds are enforced by the existing comparator. Each
capture contains at most64 new training steps in this helper, except the
one-step control. The helper also retains a copy of the already built binding.
The full baseline artifact is read, never duplicated as a new model run.

## Final root admission

The helper's local checks are not the full cross-vendor admission. Preserve
both original continuous campaigns, the full head campaign, and the full
receiving continuation campaign. Keep receipt-relative directory structure.
Under a single root-owned remote vendor guard, invoke the file-only comparator
with the original oracle reports and successful guard receipts:

```sh
"$MOJOLEARN_PYTHON" tools/amd_serial_guard.py --seconds 300 --rss-gib 12 -- \
  "$MOJOLEARN_PYTHON" tools/byte_lm_state_compare.py \
  --cuda "$NV_BASE/full128" --hip "$AMD_BASE/full128" \
  --head "$NV_HEAD/head64" --resume "$AMD_RESUME/resume128" \
  --control "$AMD_RESUME/zero-moments65" \
  --guard "cuda=$NV_BASE/byte-full128.receipt.json" \
  --guard "hip=$AMD_BASE/byte-full128.receipt.json" \
  --guard "head=$NV_HEAD/head64.receipt.json" \
  --guard "resume=$AMD_RESUME/resume128.receipt.json" \
  --guard "control=$AMD_RESUME/zero-moments65.receipt.json" \
  --oracle "cuda=$NV_BASE/gradient-oracle.json" \
  --oracle "hip=$AMD_BASE/gradient-oracle.json" \
  --oracle-guard "cuda=$NV_BASE/byte-gradient-oracle.receipt.json" \
  --oracle-guard "hip=$AMD_BASE/byte-gradient-oracle.receipt.json" \
  --output /absolute/new/byte-lm-comparison.json
```

Use the NVIDIA guard instead when running this final file check on NVIDIA.
Retain the final comparator guard log and actual exit status. Agreement with
missing prerequisites remains a nonzero diagnostic. Independent FP64
tolerance correctness, raw cross-vendor state identity, effective continuation
controls, and the predeclared10% held-out loss reduction are distinct claims.
The admitted scope is only these fixed retained trajectories and held-out
batches, not arbitrary transformer training, useful generation, or universal
bitwise certification.
