> Start at `docs/RELEASE_CHECKLIST.md`. This file is background for when a step there refuses.

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

## DEVIATION 2292: the uplink is part of the rental

The first three 0.7.0 build legs at `de719ac9` (2026-09-08, one MI325X and
two RunPod pods) produced no artifacts, and none of the three failed for a
reason on the box. All three lost this machine's network within about ninety
seconds of their boxes coming up. The AMD controller recorded
`Read from remote host 142.93.146.205: Can't assign requested address` and
`client_loop: send disconnect: Broken pipe` at 05:54:57, could not launch the
build (`rc=9`), and then logged `HTTP 000` against every DELETE and every
post-destroy GET from 06:31 to 08:58. The sm_90 controller's very first poll
went unanswered and every one after it did too, so it ran its full 2940 s
poll deadline against a box it could not see, fetched an empty directory and
a zero-byte console, and could not confirm its own terminate. The sm_89 leg
on the L40S failed identically.

Nothing in that evidence said the fault was local. Three hours of `HTTP 000`
in a controller log reads as a vendor outage, and it was this desk's link.
Both dead-man layers worked: every box was confirmed gone afterwards through
the API, and no rental outlived its lease.

The fix is not a retry. It is telling the two silences apart:

- `leg_uplink_down` / `uplink_down` probe three neutral hosts that are not a
  vendor API. If none answers, the fault is here.
- `leg_uplink_stable` / `uplink_stable` run that probe three times, spaced,
  in pre-flight **before any box is created**. A flapping link fails one of
  them and the leg refuses to rent, which costs twenty seconds against a
  one-hour rental.
- The NVIDIA poll loop re-checks after three unanswered polls and again at
  poll 30 and 60. It keeps polling either way, because the payload is
  detached and the link may return inside the lease, but the log and the
  leg record (`uplink_fault=`) now name the end that went quiet.
- The DigitalOcean teardown labels the first `HTTP 000` the same way and
  writes `uplink_fault=1` into the leg state, then keeps retrying the
  destroy.

A leg that comes home empty after this still costs a rental. What it no
longer costs is the next session's time re-deciding whose fault it was.
