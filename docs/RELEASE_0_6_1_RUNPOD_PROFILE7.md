# Root-only RunPod release build profile 7

The `0_6_1` in this file's name is historical (DEVIATION 2290): the build it
describes ships the version in `python/mojolearn/_version.py` (0.7.0 at the
time of writing) under the `release-linux3` assembly profile; 0.6.1 was never
published.

Authored source only; no controller dry run, syntax check, test, build, rental
or API call has been executed by the author. Root must review and exercise
the dry-run path before renting. This recipe does not qualify a wheel.

New explicit `MOJOLEARN_NVIDIA_CAMPAIGN=7` accepts only
`nvidia --payload mamba` with `MOJOLEARN_GPU_ARCHS=sm_89` or `sm_90`.
The payload name selects transport only: profile7 exits before any Mamba,
training, quality or benchmark campaign. Profiles0..6 retain their branches.

Root freezes the helper, controller and canonical release source in a clean
checkout, verifies existing leases/providers are clear, and chooses a pinned
RunPod image with Python3, curl, taskset, objdump and patchelf already present.
Missing patchelf/objdump records `NOT_STARTED_MISSING_PATCHELF_OR_OBJDUMP` and
exits; it never falls through to build_sets' installer. NVIDIA driver/runtime
must support the frozen Mojo toolchain. Use an actual RTX4090-class sm_89 host
for sm_89 and actual H100-class sm_90 host for sm_90; the build helper checks
physical architecture and rejects cross-compiling this profile.

Proposed root invocation, with root-selected pinned image/resource flags:

```sh
MOJOLEARN_NVIDIA_CAMPAIGN=7 MOJOLEARN_GPU_ARCHS=sm_89 \
  bash tools/gemm_remote_leg.sh nvidia --payload mamba \
  --source-ref FULL_FROZEN_COMMIT --minutes 60 --work-timeout 2700
```

The existing non-rent invocation is for root review. Only root adds `--rent`
after inspecting its output and provider state. Explicitly select the matching
GPU and pinned image using the controller's existing flags/environment; do
not assume its default GPU matches sm_89. Run sm_90 separately and serially.

Profile7 reuses profile5's vendor/RSS-guarded `pixi install --locked` with an
explicit `--environment default`. Bootstrap time counts against the same
work deadline. It then calls `release061_remote_build.sh` with at most2400
remaining work seconds, retaining an extra20 seconds around helper cleanup.
The helper imposes two-core affinity, serial binding builds, capped compiler
workers, resource-prefix tests and actual-device/source witnesses. Existing
detached-process, polling, lease watchdog, fetch reserve and teardown logic
are reused. No new unattended renewal or concurrent native build is added.

The archive selects every git-tracked canonical native-inventory file:
all eligible `.mojo`, binding/Linux-packaging/Python-package `.py`/`.sh`,
Pixi files and the qualification driver, with the same hidden/results/dist/
archive/upstream directory exclusions as final admission. It adds small
tool scripts and packaging metadata, but no binary corpus arrays or real-text
datasets. The full local canonical inventory is compared with the extracted
archive before upload and later with the actual build proof; missing files
from export-ignore, untracked source or selective archival fail loudly.
Installed qualification corpora must be supplied separately for the later
24-job final-wheel campaign; their absence is intentional in this build-only
transport and does not count as installed coverage.

Fetched evidence remains under `OUT/remote/release-build/`, including
`build/sets/cuda/ARCH/`, `build/build-provenance.json`, all build/readback/ISA
logs, preflight, root helper source, guard logs, commands, four status rows
and exit code. Local file admission requires all four jobs successful,
exact physical/source/architecture linkage and all45 fetched extension hashes.
It reports `BUILT_NOT_INSTALLED`. Root separately retains final fetch and
deletion receipts. The same exact final combined wheel still needs installed
qualification on actual sm_89, sm_90 and gfx942 before full admission.

Remaining risks to resolve in root review: cold locked Pixi installation may
consume too much lease time; image package/tool availability varies; three
sets plus logs may require substantial fetch reserve; current source
inventories can be invalidated by parallel edits; guard timeouts preserve a
failed or incomplete build, never a pass. Size is measured only by root after
packing. A two-rental estimate cannot cover both NVIDIA runtime architectures
on a typical single-GPU rental plus AMD.
