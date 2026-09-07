# NVIDIA byte-LM common-source refresh — completed and retained recipe

**Completed September 7, 2026:** NVIDIA run2 at `eac39c36` passed all twelve jobs
and independent first-step FP64 gradient/AdamW admission. Root's comparison
against DigitalOcean AMD run6 found every retained raw state equal across 128
continuous steps, with the same expanded 258-file inventory and held-out loss
5.5412986 → 2.8436419. Both rentals were deleted 204 and verified absent 404.
[NVIDIA campaign](../bench/results/resume/2026-09-07-root-byte-lm-nvidia-common/README.md),
[expanded-source comparison](../bench/results/resume/2026-09-07-root-byte-lm-expanded-comparison/README.md).
New expanded-source resume and Metal remain unrun. Earlier bidirectional resume
is a separate qualified record with its supplemental transitive-source audit.

The launch recipe below is retained historical preparation, not a request to
repeat a completed rental. The author performed source/docs work only; root
executed the checks, models, comparison and teardown.

## Reference and scope

The retained NVIDIA reference is
[run2](../bench/results/resume/2026-09-06-root-byte-lm-nvidia/README.md),
source `d921eade0da5454e39e0cd103b9a1799286b41b9`, RTX4090/sm_89. Its twelve jobs,
independent FP64 first-step gradient/AdamW oracle and 128-step real-text learning
passed. The completed refresh records the expanded common source instead:

- Repository: `/tmp/mojolearn-byte-lm-do-20260907`.
- Exact source: `eac39c367beeddb8ba4792551d154673e654ce21`.
- Root verified the expanded 258-file source inventory against Metal snapshot
  `45cc2d9d`, including 45 Mamba transitive files. This is source provenance,
  not a new numerical certificate. The older 213-file subset was incomplete.
- Reuse that repository's existing `tools/gemm_remote_leg.sh`, **profile 5**,
  and pinned `tools/byte_lm_validation_serial.sh`. No new controller is needed.

The twelve jobs are byte-venv, byte-dependencies, dependencies,
dependency-freeze, guard-checks, comparator-fixtures, byte-build, retain-binding,
byte-host-mocks, byte-step1, byte-gradient-oracle and byte-full128. First-step
independent admission precedes full128; complete state, raw arrays and root
receipts must be fetched. This does not run resume or admit cross-vendor/Metal
agreement automatically.

## Retained root command recipe

Run the preparation checks on the common-source repository; keep separate,
new result directories for dry run and rental. Root must inspect the generated
remote body and retain its immutable controller copy before launch.

```sh
cd /tmp/mojolearn-byte-lm-do-20260907
MOJOLEARN_NVIDIA_CAMPAIGN=5 \
MOJOLEARN_GPU_ARCHS=sm_89 \
MOJOLEARN_BYTE_LM_PYTHON=python3 \
MOJOLEARN_GEMM_LEG_OUT=/absolute/NEW_NVIDIA_COMMON_SOURCE_DRYRUN \
tools/gemm_remote_leg.sh nvidia --payload mamba \
 --source-ref eac39c367beeddb8ba4792551d154673e654ce21 \
 --gpu 'NVIDIA GeForce RTX 4090' \
 --image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 \
 --minutes 60 --work-timeout 3000 --dry-run
```

After root's checks and confirmed AMD cleanup, use a different new output and
replace `--dry-run` with `--rent`. Supply the existing protected RunPod key file
through `MOJOLEARN_RUNPOD_KEY_FILE`; never paste a token into a command or log.
The image/GPU pair is the previous successful run2 configuration; current
availability and startup are not guaranteed. Root must confirm actual NVIDIA
CUDA/PyTorch runtime and sm_89 witnesses after launch.

The source transport defaults to the bounded SHA-checked SSH archive. If root
has verified the **exact** common-source commit is available in the public
`mojolearn/mojolearn` repository, optionally add
`MOJOLEARN_PUBLIC_GITHUB_SOURCE=mojolearn/mojolearn` to the same command. That
route fetches the pinned commit with two-thread/memory bounds and verifies the
source inventory before payload launch. It refuses an unavailable commit and
never silently falls back to branch HEAD. Public commit availability has not
been checked by this author. A slow SSH upload can consume substantial lease
time; do not change the pinned source merely to obtain a faster transfer.

## Safety and time budget

The reviewed common-source controller explicitly **unsets** all three Modular
pool overrides in its NVIDIA profile4/5/6 branch:

```sh
unset MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE
unset MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_ONLY
unset MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT
```

This preserves the qualified NVIDIA allocator defaults. **Do not apply AMD's
1 GiB pool-only settings on NVIDIA.** Two prior NVIDIA attempts refused small
allocations under that override. AMD's successful pool setting is vendor-specific.
The generated NVIDIA `leg.txt` must identify qualified runtime defaults.

Existing two-core affinity, two-thread/compiler limits, NVIDIA serial guard,
RSS/VRAM/host-reserve thresholds, global deadline, local and on-pod deletion
watchdogs, detached payload polling, fetch and verified deletion remain in force.
Lease cap is60minutes; nominal work cap3000seconds, reduced to the remaining
budget after bootstrap and subject to the controller's fetch/deletion reserve.
This is a budget, not a runtime prediction. No overlapping model/build job is
allowed. Never increase the limits or retry while a prior process/pod remains
unaccounted for.

## Retained acceptance checklist

- Root completed the source/file checks and dry run before the retained campaign.
- Confirm AMD DO completion/cleanup, GPU stock, protected credentials and the
  selected source transport before rental.
- Require all twelve statuses and exact-source/binding provenance to pass the
  existing `tools/byte_lm_validation_admit.py` on fetched `remote/byte-lm-validation`.
- Retain original failures, generated controller, source archive/inventory,
  complete raw capture and DELETE204/GET404 teardown evidence.
- Compare the new NVIDIA raw arrays with the completed common-source DO/Metal
  captures separately under root's file-only comparator. Earlier d921eade
  continuous/resume results do not automatically qualify the newer loader,
  source inventory or Metal route.

Root follow-up: the profile-5 bootstrap now runs locked Pixi installation
under the vendor RSS guard and stops before the campaign on failure. This
orchestration-only revision is `34fb352f`; all 213 numerical files still match
DO `0ba54976` and the Metal snapshot. Root passed 50 guarded dry-run checks
and a mocked bootstrap-failure check. That was a historical pre-rental checkpoint; the later completed run is recorded above.

Latest source follow-up: `eac39c36` adds transitive Mamba source inventory.
The earlier 50-check rehearsal covered orchestration `34fb352f`; root then
checked the expanded archive and completed NVIDIA run2. Run1's failed create
attempt remains retained separately. No mathematical kernel change is claimed
by this documentation update.
