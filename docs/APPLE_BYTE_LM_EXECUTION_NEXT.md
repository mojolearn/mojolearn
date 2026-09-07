# Next Apple byte-LM execution: root only, currently blocked

## September 7 update: Apple continuous training completed

Root built the Metal tiny byte-LM and ran all128 real-text training steps.
Complete raw states, held-out bytes and final checkpoint match NVIDIA and
DigitalOcean AMD bit for bit. Loss fell from5.5413 to2.8436 on all three.
Evidence: `bench/results/resume/2026-09-07-root-byte-lm-three-vendor/README.md`.
The user explicitly removed the fixed free-memory minimum; the separately
recorded user-tiny policy retained two-thread limits, a2GiB RSS cap, pressure,
swap/compression, CPU and deadline stops, watchdog and verified cleanup.
The earlier blocked/readiness sections below are historical, superseded for
continuous Metal training. Metal resume, separate MLP Metal, and installed
byte-LM wheel coverage remain open. No further paper work was performed.


## Authored first-step helper (later expanded source pin)

`tools/apple_byte_lm_first_step.py` now prepares the later expanded inventory:
Metal commit `45cc2d9dcd2279646f9a9716fdfab9d598320535`, exactly 258 source
files, checked against a supplied `common-source-expanded.json`. This supersedes
the older source pin below for this helper only. It changes no numerical files.

Root must review/test this **unexecuted source** before use. Invocation takes
`--source CHECKOUT --environment EXISTING_DEFAULT_ENV --common-source MANIFEST
--output NEW_CANONICAL_ABSOLUTE_DIRECTORY`; without `--run` it only prepares
evidence/configuration and checks source files/commit. Preparation is not admission.
Use a canonical output parent (`/private/tmp` on Darwin, rather than its `/tmp`
alias). A later explicit `--run` requires another fresh output directory.

With `--run`, the first candidate-runtime operation is preceded by retained guard
memory telemetry and unchanged 4 GiB reserve admission. Subsequent jobs each use
the existing macOS guard: compiler/Python/SDK readbacks, a 180-second/4-GiB
two-worker build, AIR inspection, a 60-second/2-GiB native witness, and one
180-second/4-GiB capture. No full128, retry, install, Pixi activation, or download
is included. It retains exact argv/environment, logs, exits, guard policy copies,
installed metadata, source inventory, binary witness, and the root capture receipt.

Limitations: source/metadata checks do not prove candidate-environment ABI
compatibility; AIR symbol presence alone does not prove every kernel executable;
fresh configured cache paths are not an OS-enforced filesystem sandbox. The
runtime may ignore configuration, so root must review private-cache behavior.
Fast probes with no retained guard sample fail the strict receipt policy. If
supervisor cleanup itself times out, the helper stops with incomplete evidence;
root must inspect/quarantine descendants rather than retry. The public binding
is exclusively linked into the otherwise fresh snapshot; it is not a release
wheel. No numerical or three-vendor claim follows from preparation or one step.

This is an **unexecuted preparation recipe**, not an environment compatibility,
build, learning, or identity result. Target source is the prepared checkout
`/tmp/mojolearn-byte-lm-metal-20260907` at `d17c1aa1`. No numerical source changes
are required by this recipe. The Mac must first satisfy the existing guard's
normal-pressure **4 GiB free/speculative reserve**. Do not bypass that admission,
drop its reserve, clear shared caches, or repeatedly launch while admission fails.
Only the main/root agent executes, one job at a time, after other campaigns stop.

## Dependencies found by reading files

The snapshot has no `.pixi` environment. Its `bindings/build_byte_lm.sh` calls
`pixi run mojo`; invoking that script normally can trigger local environment
creation. Do not use `pixi run`, `pixi install`, activation, or an implicit PATH
compiler for this attempt.

An existing candidate environment is
`/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/default`.
Its `conda-meta` records name Mojo/compiler/Python integration `1.0.0-release`,
MAX `26.5.0-3.13release`, Python `3.13.15-hf1cfe1e_101_cp313`, and NumPy
`2.5.2-py313hce9b930_0`. Corresponding version/build entries occur in the frozen
default environment's `pixi.lock`. Matching names do **not** establish package
bytes, ABI compatibility, or successful runtime loading. Retain the lock and
installed package metadata, then let root perform bounded readbacks before a
model call. The existing bench environment uses Python 3.14 and is not this recipe.

`etc/conda/activate.d/10-activate-max.sh` sets `MODULAR_HOME` and may perform
first-activation telemetry. Instead set the environment explicitly. The existing
`share/max/modular.cfg` contains absolute compiler/runtime paths **and an absolute
shared `cache_dir`**. Merely changing `MODULAR_HOME` while copying that file
unchanged would still reuse the old MAX cache.

The package selector supports missing-extension stubs if at least one selected
binding exists. `_mojolearn_byte_lm.so` is in that selector's registry, and its
mode/vendor/profile witnesses are read directly. No unrelated root working-tree
native libraries should be copied into the snapshot. If isolated import fails,
retain the failure and diagnose it; do not fill the package with newer binaries.

## Preparation and readbacks before build

Create a fresh evidence directory and a fresh private runtime/cache directory;
refuse existing destinations. Set task-specific variables (not `HOME`):

```sh
byte_src=/tmp/mojolearn-byte-lm-metal-20260907
byte_env=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/default
byte_out=/tmp/mojolearn-byte-lm-apple-attempt-CHOOSE_FRESH_ID
byte_python="$byte_env/bin/python"
byte_guard="$byte_src/tools/macos_serial_guard.py"
```

Root should retain `git rev-parse HEAD`, dirty status, compiler/runtime package
metadata, SDK identity, the exact commands, and copies of the frozen build script
and guard. Require the intended source revision and no unexpected inventoried
edits. Do not call version probes outside the guard just because they look cheap.

Prepare `$byte_out/runtime/modular.cfg` from the candidate environment's config:
preserve all absolute package/compiler/runtime/import/library paths, but set
`[max].cache_dir` to `$byte_out/runtime/.max_cache`. Ensure this path and
`$byte_out/runtime/cache` are new, ordinary private directories, with no symlinks
to previous caches. Retain both original and derived configuration. This is an
environment-only preparation, outside the source inventory. Root must verify
that the installed runtime honors the private configuration; this source audit
has not established that behavior. An unrecognized configuration or access to the
old cache blocks the attempt rather than authorizing a shared-cache reset.

For every guarded command set `MODULAR_HOME=$byte_out/runtime`,
`PATH=$byte_env/bin:/usr/bin:/bin:/usr/sbin:/sbin`,
`MOJOLEARN_NUMERIC_MODE=identical`, `MOJOLEARN_TARGET_COLUMN=apple`,
`PYTHONPATH=$byte_src/python`, `PYTHONNOUSERSITE=1`, and the ordinary
OMP/OpenBLAS/MKL/NumExpr thread limits to 2. Unset `MACOSX_DEPLOYMENT_TARGET`,
`MOJOLEARN_GPU_ARCHS`, and inherited `PYTHONHOME`. Do not source conda activation.

Root's initial guarded readbacks should retain the absolute compiler's version,
Python executable/version/platform, installed NumPy version/path, SDK version,
and candidate package/runtime paths. Use the same private configuration and
Python for the guard and its children. Each readback must pass guard admission,
with a fresh log/report and a deadline at most 60 seconds. Metadata matching alone
does not authorize an ABI or Metal support claim.

## Explicit equivalent build, then one step

After readbacks succeed, root may run the following compiler command **inside**
the macOS guard, from `$byte_src`. It is the Darwin branch of the frozen build
script with its `pixi run` prefix replaced by the absolute existing compiler.
Retain that distinction in provenance rather than claiming the script ran.
Resolve and retain `byte_sdk` through the guarded SDK readback first.

```sh
"$byte_python" "$byte_guard" --seconds 180 --rss-gib 4 \
  --report "$byte_out/build.guard.json" -- \
  "$byte_env/bin/mojo" build -j 2 --emit shared-lib \
  --target-cpu apple-m1 -D MOJOLEARN_COLUMN_APPLE \
  -Xlinker -platform_version -Xlinker macos -Xlinker 11.0 -Xlinker "$byte_sdk" \
  -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . -I bindings \
  bindings/_mojolearn_byte_lm.mojo -o "$byte_out/_mojolearn_byte_lm.so"
```

Keep stdout/stderr in a new `build.guard.log` and retain the actual guard exit
status, including teardown failure. A timeout is a failed bounded attempt, not
permission to raise limits or relaunch automatically. After success, publish
the artifact to the snapshot's `python/mojolearn/identical/` using an exclusive
hard link, refusing preexisting files or symlinks, and retain its SHA-256.

Existing Metal build conventions explain both `unset MACOSX_DEPLOYMENT_TARGET`
and the absence of `--target-accelerator`: either can suppress required Metal AOT
generation. Root should retain binary AIR symbol inspection and private-cache
artifacts; reject absent AOT payloads. The training binding's historical AIR-count
floor is not an established byte-LM count and must not be reused as one.

Next create a tiny host-only witness script importing
`mojolearn._mojolearn_byte_lm` and asserting `byte_lm_numeric_mode() == 1`,
`byte_lm_vendor() == 'metal'`, and `byte_lm_profile()` equals
`mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1`.
Retain the resolved imported binding path/hash and execute this script through
the same guard (`--seconds 60 --rss-gib 2`). No device/model operation belongs
in this witness script. Unexpected imports or witnesses stop the sequence.

Only then run the existing capture with a new output directory:

```sh
"$byte_python" "$byte_guard" --seconds 180 --rss-gib 4 \
  --report "$byte_out/step1.guard.json" -- \
  "$byte_python" "$byte_src/tools/byte_lm_real_text_capture.py" \
  --expected-vendor metal --steps 1 --action continuous \
  --output "$byte_out/step1"
```

Save the exact argv/command, combined guard log, actual exit status, result, and
cleanup/watchdog report. Use `tools/root_job_receipt.py` with `--vendor metal`,
`--job-kind capture`, the actual `--exit-code`, retained `--command-file`,
`--guard-log`, `--result`, and a fresh `--output`; receipt production itself is a
root-only bounded host task. Failed or incomplete captures remain diagnostics.
No pipe should replace the guard exit status with `tee`'s exit status.

The guard retains two-thread settings and sampled CPU enforcement; Darwin does
not supply the Linux hard-affinity guarantee. Its 4 GiB entry/2 GiB runtime
reserve, pressure/swap/compression limits, RSS cap, deadline, watchdog, and verified
cleanup all remain required. Any quarantine requires root investigation.

## What this can and cannot establish

One successful step establishes only that this fresh build can execute the fixed
step under the retained policy. It does not supply a Metal FP64 oracle, 128-step
learning, checkpoint resume, or three-vendor identity. The independent oracle
still requires CUDA/HIP. All three eventual continuous captures must use the
same complete numerical source inventory; old pre-portability CUDA/HIP cards
cannot simply be combined with the new Metal snapshot. Future 128-step or resume
execution needs a separately reviewed bounded plan after this first gate.

## Expanded transitive-source snapshot

Root advanced the prepared Metal checkout to `45cc2d9d` after adding Mamba
provenance. Its 258 source files match Linux `eac39c36`; the root record is
`bench/results/resume/2026-09-07-root-metal-preparation/common-source-expanded.json`.
This supersedes the earlier 213-file subset for new captures. No Metal build
or model was run; the unchanged memory admission still applies.
