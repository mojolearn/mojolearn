# Root-only NVIDIA/AMD checkpoint-resume commands

These commands are authored, not executed. They exercise the existing fixed B2/L8/DM32/V64 training loop. No public training API is introduced. The separate `training/checks/cross_vendor_resume.mojo` executable leaves `checkpoint_check.mojo`'s default behavior unchanged.

Only the root agent may build, run, or compare. Use already-authorized remote NVIDIA/AMD machines; no Apple testing. Run one compiler or GPU process at a time with the controller's memory limits, deadline, and teardown policy. The Python evidence tool never provisions, executes subprocesses, or performs GPU work.

The native companion now embeds Python for Linux file admission: run from the repository root with its pinned Python runtime available. The helper opens input once, checks that the opened inode is a regular file of exactly 161,008 bytes before its bounded read, then seals a memfd capture. Both native decode and receipt hashing read those same immutable bytes. Output and receipt paths are reserved with `O_EXCL|O_NOFOLLOW` before GPU work; existing files, symlinks, and hardlinks refuse. Native writes use only the owned descriptor paths. Failures may leave newly reserved empty/partial files; choose fresh names for another attempt. Existing evidence is never overwritten. Directory/path identity and `fsync` checks run before success, but no power-loss recovery protocol is claimed.

## Build once on each remote vendor

Use a frozen checkout containing the same sources on both machines. Below, `/artifacts/training-resume` is a new campaign directory on that host. Adjust the known accelerator architecture in the build-command file, and preserve that exact recipe. The NVIDIA example `sm_89` must match the actual GPU; AMD uses its actual `gfx...` target instead. Do not infer architecture from these example names.

```sh
mkdir -p /artifacts/training-resume
cat > /artifacts/training-resume/build-command.txt <<'EOF'
python3 tools/nvidia_serial_guard.py --seconds 900 --rss-gib 12 -- pixi run mojo build -j 2 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 --target-accelerator sm_89 training/checks/cross_vendor_resume.mojo -o /artifacts/training-resume/resume-driver
EOF
python3 tools/training_cross_vendor_resume.py snapshot \
  --source-root . \
  --build-command-file /artifacts/training-resume/build-command.txt \
  --output /artifacts/training-resume/source.json
```

For AMD, write this corresponding guarded recipe into its `build-command.txt` before taking the source snapshot, replacing the example `gfx942` with the actual target:

```sh
python3 tools/amd_serial_guard.py --seconds 900 --rss-gib 12 -- \
  pixi run mojo build -j 2 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
  --target-accelerator gfx942 training/checks/cross_vendor_resume.mojo \
  -o /artifacts/training-resume/resume-driver
```

Root executes each retained recipe, which itself invokes the corresponding vendor guard:

```sh
bash /artifacts/training-resume/build-command.txt > /artifacts/training-resume/build.log 2>&1
```

Record the compiler version, locked environment identity, driver/runtime version, actual GPU model, and accelerator target in `/artifacts/training-resume/runtime.txt`; the evidence tool requires that retained file and does not discover devices itself.

## Six separate native processes

Use absolute, distinct, previously nonexistent output paths. Each successful invocation writes the checkpoint and `<checkpoint>.json` native receipt. Publish no evidence manifest for a failed process. Do not resume from a `head8` file generated with `steps_planned=8`: this driver deliberately records planned16 for every action.

On NVIDIA:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
MOJOLEARN_TRAIN_EXPECT_VENDOR=cuda \
MOJOLEARN_TRAIN_RESUME_ACTION=continuous16 \
MOJOLEARN_TRAIN_RESUME_OUTPUT=/artifacts/training-resume/cuda-continuous.ckptbin \
  python3 tools/nvidia_serial_guard.py --seconds 600 --rss-gib 12 -- \
  /artifacts/training-resume/resume-driver > /artifacts/training-resume/cuda-continuous.log 2>&1
```

Run the same command in a **new process** with action `head8`, output `cuda-head.ckptbin`, and log `cuda-head.log`. On AMD:

```sh
MOJOLEARN_TRAIN_EXPECT_VENDOR=hip \
MOJOLEARN_TRAIN_RESUME_ACTION=continuous16 \
MOJOLEARN_TRAIN_RESUME_OUTPUT=/artifacts/training-resume/hip-continuous.ckptbin \
  python3 tools/amd_serial_guard.py --seconds 600 --rss-gib 12 -- \
  /artifacts/training-resume/resume-driver > /artifacts/training-resume/hip-continuous.log 2>&1
```

Repeat with `head8` and output/log stem `hip-head`, retaining the AMD guard.

Transfer NVIDIA's `cuda-head.ckptbin` to AMD and AMD's `hip-head.ckptbin` to NVIDIA using the root's existing transfer method. Do not regenerate the input locally. Then, on NVIDIA:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
MOJOLEARN_TRAIN_EXPECT_VENDOR=cuda \
MOJOLEARN_TRAIN_RESUME_ACTION=resume16 \
MOJOLEARN_TRAIN_RESUME_INPUT=/artifacts/training-resume/hip-head.ckptbin \
MOJOLEARN_TRAIN_RESUME_OUTPUT=/artifacts/training-resume/cuda-from-hip.ckptbin \
  python3 tools/nvidia_serial_guard.py --seconds 600 --rss-gib 12 -- \
  /artifacts/training-resume/resume-driver > /artifacts/training-resume/cuda-from-hip.log 2>&1
```

On AMD:

```sh
MOJOLEARN_TRAIN_EXPECT_VENDOR=hip \
MOJOLEARN_TRAIN_RESUME_ACTION=resume16 \
MOJOLEARN_TRAIN_RESUME_INPUT=/artifacts/training-resume/cuda-head.ckptbin \
MOJOLEARN_TRAIN_RESUME_OUTPUT=/artifacts/training-resume/hip-from-cuda.ckptbin \
  python3 tools/amd_serial_guard.py --seconds 600 --rss-gib 12 -- \
  /artifacts/training-resume/resume-driver > /artifacts/training-resume/hip-from-cuda.log 2>&1
```

The native loader verifies exact layout, complete state, seed, AdamW descriptor bits, planned16, clean arm, and completed8 before creating a device context. Both continuations consume generated batches9–16 with optimizer `t=9..16`. Setting `MOJOLEARN_TRAIN_SEED` is optional; if set it must be the same decimal UInt64 on every process.

## Retain each completed leg

First retain each guarded native invocation as `<leg>.command.sh` (including its environment assignments, excluding the log redirection). Execute that file and capture the **guard's final exit status**, including cleanup, before recording artifacts. For example, after putting the NVIDIA uninterrupted snippet into `cuda-continuous.command.sh`:

```sh
guard_exit=0
bash /artifacts/training-resume/cuda-continuous.command.sh > /artifacts/training-resume/cuda-continuous.log 2>&1 || guard_exit=$?
python3 tools/training_cross_vendor_resume.py exit-record \
  --exit-code "$guard_exit" \
  --run-command-file /artifacts/training-resume/cuda-continuous.command.sh \
  --run-log /artifacts/training-resume/cuda-continuous.log \
  --binary /artifacts/training-resume/resume-driver \
  --checkpoint /artifacts/training-resume/cuda-continuous.ckptbin \
  --output /artifacts/training-resume/cuda-continuous.exit.json
```

Repeat this capture for every leg and negative control. `exit-record` executes nothing: it binds the root-supplied status to hashes of the exact command, log, binary, checkpoint, and native receipt. Admission requires the integer exit code to be exactly 0 and every digest to match. A stale native receipt or a process that prints COMPLETE before teardown failure cannot be admitted by itself.

Example for NVIDIA's uninterrupted leg:

```sh
python3 tools/training_cross_vendor_resume.py record \
  --source-root . --snapshot /artifacts/training-resume/source.json \
  --binary /artifacts/training-resume/resume-driver \
  --checkpoint /artifacts/training-resume/cuda-continuous.ckptbin \
  --vendor cuda --build-log /artifacts/training-resume/build.log \
  --run-log /artifacts/training-resume/cuda-continuous.log \
  --run-command-file /artifacts/training-resume/cuda-continuous.command.sh \
  --exit-code-file /artifacts/training-resume/cuda-continuous.exit.json \
  --runtime-info /artifacts/training-resume/runtime.txt \
  --output-dir /artifacts/training-resume/evidence/cuda-continuous
```

Apply the corresponding filenames/vendor for each of six legs. For `cuda-from-hip`, additionally pass `--input-checkpoint /artifacts/training-resume/hip-head.ckptbin`; for `hip-from-cuda`, pass the transferred NVIDIA head. Record copies checkpoint, native receipt, source snapshot, build/run logs, runtime information, and any incoming checkpoint into a new evidence directory. It validates the raw checkpoint structure and checksums independently, links native receipt hashes to retained bytes, and refuses changed source since the pre-build snapshot. The manifest retains the binary hash/size; keep the original executable with the campaign's build artifacts. Hashes and a root-supplied build recipe are provenance records, not proof that a particular compiler invocation produced a binary.

## Compare the complete transfer chain

Collect all six evidence directories at one remote analysis location. File comparison can occur without a GPU job, but remains root-only.

Before admitting the full resume claim, run an effective missing-moments control. On AMD, prepare a bounded corrupted copy of the transferred NVIDIA head; this command edits only a new file and repairs its normal codec checksums so the control reaches numerical continuation:

```sh
python3 tools/training_cross_vendor_resume.py zero-moments \
  --input /artifacts/training-resume/cuda-head.ckptbin \
  --output /artifacts/training-resume/cuda-head-zero-moments.ckptbin
```

Retain and execute the following as `hip-missing-moments.command.sh` through the same exit-status capture procedure:

```sh
MOJOLEARN_TRAIN_EXPECT_VENDOR=hip \
MOJOLEARN_TRAIN_RESUME_ACTION=resume16 \
MOJOLEARN_TRAIN_RESUME_INPUT=/artifacts/training-resume/cuda-head-zero-moments.ckptbin \
MOJOLEARN_TRAIN_RESUME_OUTPUT=/artifacts/training-resume/hip-missing-moments.ckptbin \
  python3 tools/amd_serial_guard.py --seconds 600 --rss-gib 12 -- \
  /artifacts/training-resume/resume-driver
```

Record it with vendor `hip`, the corresponding command/exit/log/output files, and `--input-checkpoint` pointing to the zero-moments input. The comparator verifies that only the originally nonzero moments changed, and that this continuation diverges from the uninterrupted reference. It also requires parameters to change from head8 to final16 on both normal vendor legs.

```sh
python3 tools/training_cross_vendor_resume.py compare \
  --cuda-continuous /artifacts/training-resume/evidence/cuda-continuous \
  --hip-continuous /artifacts/training-resume/evidence/hip-continuous \
  --cuda-head /artifacts/training-resume/evidence/cuda-head \
  --hip-head /artifacts/training-resume/evidence/hip-head \
  --cuda-from-hip /artifacts/training-resume/evidence/cuda-from-hip \
  --hip-from-cuda /artifacts/training-resume/evidence/hip-from-cuda \
  --hip-missing-moments /artifacts/training-resume/evidence/hip-missing-moments \
  --output /artifacts/training-resume/comparison.json
```

Exit0 means agreement, non-vacuity, and an effective retained negative control all passed; exit1 means a numerical/byte mismatch, vacuous updates, or ineffective control; exit2 means missing/inconsistent evidence or refusal. If `--hip-missing-moments` is omitted, otherwise-matching evidence produces `AGREEMENT_ONLY_PENDING_NEGATIVE_CONTROL`, exit3, and `full_resume_claim_admitted=false`. It must not be reported as a full resume pass. The comparator requires same source inventory and run descriptor for all legs, same binary within each vendor, complete per-step receipts, zero guard-exit evidence, and IDENTICAL native mode/vendor readback. It checks transferred input bytes against the opposite vendor's retained head, full final checkpoint bytes, and per-step loss bits/state digests. A mismatch reports its first byte and, for payload data, parameter name and element.

A PASS covers only this profile's state/loss equality and cross-vendor continuation. It is not a public API certificate, independent gradient-correctness proof, arbitrary model-size claim, or performance result. Corruption/configuration refusals and independent gradient tests remain separate root-only gates.
