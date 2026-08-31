# The Linux wheel: one name, two vendors, six binary sets

Design note, 2026-08-29. Decided by Andrew the same day. ONE PyPI name,
`mojolearn`, with the vendor -- and, since 2026-08-30, the GPU architecture
-- detected at import. Two wheels per release.

| wheel | tag | carries |
|---|---|---|
| macOS (shipping since 0.1.0) | `py3-none-macosx_11_0_arm64` | Metal, three tiers, ten extensions each |
| Linux (this note) | `py3-none-manylinux_<measured>_x86_64` | CUDA AND HIP, one set per GPU ARCHITECTURE, three tiers each (thirty extensions per architecture) |

Nothing in this note has run. Every script it names was written under the
no-run order of 2026-08-29 and is committed UNRUN; the commands in section 8
are what runs them, and section 9 lists every assumption a run has to
confirm. Any sentence here that a run contradicts is fixed in the commit that
records the run.

## 1. Why one name and not `mojolearn-cu12` and `mojolearn-rocm`

Because the size allows it. That was an extrapolation when this was written
and it is now MEASURED, 2026-08-30, and it held: six sets pack to about 77 MB
against PyPI's 100 MB per-file default. The MAX runtime closure, the one
number the extrapolation could not cover, came in at 3,200,416 bytes and is
BYTE-IDENTICAL between CUDA and HIP, so it ships once for both.

**77 MB is small for this category, which is the part worth keeping in mind
when the number sounds large.** Biggest Linux wheel on PyPI, read from the
JSON API the same day:

    nvidia-cublas-cu12   554 MB
    tensorflow           546 MB
    torch                502 MB
    cupy-cuda12x         141 MB
    jaxlib                84 MB
    mojolearn, 6 sets     77 MB

and ours carries six GPU architectures across two vendors in three numeric
tiers, where torch's 502 MB carries one CUDA version.

Two facts that keep this decision from being forced later. PyPI's 100 MB is a
per-project DEFAULT and a limit increase can be requested, so crossing it is
a form and not a redesign. And the wheel is about half duplication today (see
the `.text` measurement in section 9), so if MAX ever emits multi-architecture
binaries the same layout roughly halves with no change to the selector.

**A note on where bytes come from, because the intuition runs the wrong way.**
The Python is 25 files and 544 KB, and it ships ONCE for the whole wheel.
Every byte of Mojo is compiled per vendor, per architecture and per tier,
which is eighteen copies in a six-set wheel. Moving host-side logic from
Python into Mojo therefore GROWS the package by roughly eighteen times what
it saves. Keep on the GPU what must run on the GPU and leave the rest in
Python; that is the size-optimal rule here, and it is the opposite of the
usual instinct.

The user-facing reason is the same one that decided 0.1.0, which is `pip
install mojolearn` on any supported box, with the import name `mojolearn`,
and no extras.

## 2. Layout

The macOS layout is untouched. The Linux layout puts one directory per
accelerator API under the package and repeats the tier layout inside it.

```
python/mojolearn/                          macOS, unchanged
    _mojolearn*.so                         fast
    deterministic/_mojolearn*.so
    identical/_mojolearn*.so
    .dylibs/*.dylib                        MAX runtime (Mach-O closure)

python/mojolearn/                          Linux wheel
    cuda/sm_80/_mojolearn*.so              fast, one directory per
    cuda/sm_80/deterministic/_mojolearn*.so  GPU ARCHITECTURE
    cuda/sm_80/identical/_mojolearn*.so
    cuda/sm_90a/...
    hip/gfx942/_mojolearn*.so
    hip/gfx942/deterministic/_mojolearn*.so
    hip/gfx942/identical/_mojolearn*.so
    .libs/*.so             ONE shared MAX runtime closure, when every set's
                           closure is byte-identical (pack_wheel.py decides;
                           2026-08-30 measured the two vendors' identical)
    cuda/.libs/*.so        otherwise one per vendor
    hip/.libs/*.so
```

THE ARCHITECTURE LEVEL EXISTS BECAUSE ONE BUILD IS ONE ARCHITECTURE,
measured 2026-08-30 (bench/results/wheels/LEGS_2026-08-30.md): `mojo build`
takes exactly one `--target-accelerator` (a comma list parses and the
compiler rejects it), emits device code for that architecture and NO PTX, so
there is no JIT fallback -- the sm_90a-only 0.3.0 wheel installed cleanly on
an A40 and failed 27 of 29 lanes with CUDA_ERROR_NO_BINARY_FOR_GPU. A
portable wheel is therefore N single-architecture sets, and the import picks
one the same way it picks the vendor. A vendor directory with binaries
DIRECTLY in it (every set built before the axis) is the arch-less legacy
layout and keeps working unchanged.

The directory names are the accelerator APIs (`cuda`, `hip`, and `metal` as
a name the read-back returns, never a directory) and not the vendors, because
the kernel matrix already uses vendor names for its columns and a column is a
scheduling decision under one API (CDNA and RDNA are two columns under
`hip`). One directory per API.

A source checkout on a Linux box keeps building the FLAT layout
(`python/mojolearn/*.so`, exactly what every E1 leg has ever built) and the
selector recognizes it by the absence of a vendor directory with binaries in
it. Every existing leg script is unaffected.

## 3. The selector: a vendor axis on top of the tier axis

`python/mojolearn/_backend.py` had one axis, the tier, chosen by
`MOJOLEARN_NUMERIC_MODE` at import or by `set_numeric_mode()` and
`numeric_mode=` at run time. It now has two. `tier_dir(mode)` is the one
place both axes become a path, and the four bindings that carry private
loaders (`_linalg_impl`, `_metrics_impl`, `_tsa_impl`, `_svm_impl`) call it
rather than joining `pkg_dir, mode` themselves.

`_layout()` decides ONCE per process, by looking at the disk, not the
platform.

1. No vendor directory with binaries means FLAT. The package directory is the
   root and the vendor is whatever the loaded binaries say.
2. Otherwise the WHEEL layout, and a vendor directory is chosen, in this
   order.
   * `MOJOLEARN_VENDOR=cuda|hip` in the environment picks the directory. It
     must name a set the install carries or the import raises.
   * else the box probe, which is, for each carried vendor, `os.path.exists` on its
     device nodes (`/dev/nvidiactl`, `/dev/nvidia0`; `/dev/kfd`,
     `/dev/dri/renderD128`) and `ctypes.CDLL` on its driver library
     (`libcuda.so.1`; `libamdhip64.so.7|.6|.so`). Exactly one vendor with
     evidence is picked. Two with evidence raises and asks for
     `MOJOLEARN_VENDOR`. None raises the no-GPU refusal (section 4).

Then the ARCHITECTURE, one directory further down (2026-08-30), decided in
the same shape:

* `MOJOLEARN_GPU_ARCH=<sm_80|gfx942|...>` in the environment picks the
  directory. It must name a set the install carries or the import raises.
* else the device's own architecture, read WITHOUT loading an extension:
  for cuda, `cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR/MINOR)` through
  ctypes on the driver library the probe already loads; for hip,
  `gfx_target_version` out of `/sys/class/kfd/kfd/topology` (no ROCm
  library needed). An exact match wins. On cuda a device `sm_XY` also
  accepts a carried `sm_XYa` (the `a` restricts which devices, not this
  one), and failing both, the HIGHEST carried non-`a` architecture of the
  same major family not above the device -- NVIDIA documents cubin forward
  compatibility within a family (sm_80 code on sm_86/sm_89); our own
  measurement of it is OWED, section 9. On hip there is NO family rule:
  gfx code objects are ISA-exact, so anything but an exact match refuses,
  naming the device, every carried set and `MOJOLEARN_GPU_ARCH`.
* else (the device architecture cannot be read): exactly one carried
  architecture is used and says so in `gpu_arch_how()`; more than one
  refuses, naming `MOJOLEARN_GPU_ARCH`.

`gpu_arch()` and `gpu_arch_how()` report the choice; on the flat and
arch-less layouts both report that no choice was made.

Then, whichever way the directory was chosen, EVERY binary loaded from it
is asked what it was compiled for, and one that disagrees with the directory
is refused at import with the path, the directory and both answers. This is
the same refusal, in the same place, as the tier read-back that caught the
Aug 24 stale-`_MODULES` bug, and it exists for the same reason. A CUDA `.so`
under `hip/` imports cleanly on an NVIDIA box and touches no device until the
first fit, so nothing later would catch it.

The order of trust is therefore what the binary says, then what the
environment says, then what the box appears to have. `MOJOLEARN_VENDOR`
cannot relabel a binary; it can only open the other directory.

**This paragraph used to end "and that fails at the first device call with
the runtime's own error, which is the honest outcome for a forced wrong
choice". That was measured on 2026-08-30 and it is only half true.** In a
container with no device of either kind, against the finished wheel:

    MOJOLEARN_VENDOR=cuda   imports, and the first fit raises a clean Python
                            exception naming libnvidia-ml.so.1
    MOJOLEARN_VENDOR=hip    SIGILL during import, no traceback at all

The hip binary's module init takes a trap when the HIP runtime library is
absent entirely, so the promised runtime error never arrives and the process
simply dies. The two vendors do not degrade alike.

So the selector now REFUSES a forced vendor for which the box shows no
evidence at all, neither a device node nor a loadable driver library, and
says why. That refusal costs nothing real, because the case the override
exists for is a device that IS present and a probe that missed it, and such a
box has that vendor's driver library loadable, which is evidence.
`MOJOLEARN_VENDOR_FORCE=1` proceeds anyway for a genuinely mis-probed box,
and states that it may abort. The refusal is in Python, which is the only
place a message survives a trap in the binary below it.

## 4. The read-back from the binary

`checks/vendor.mojo` defines `COMPILED_VENDOR`, a compile-time constant
resolved from `std.sys.info.has_amd_gpu_accelerator()`,
`has_nvidia_gpu_accelerator()` and `has_apple_gpu_accelerator()`, the same
predicates `checks/kernel_matrix.mojo` uses to pick `DETECTED_COLUMN`,
folded to `hip`, `cuda`, `metal` or `none`. It is a property of the BUILD
TARGET, not a runtime query. No `DeviceContext` is opened at import, and the
answer is about the binary rather than about what the process can see.

Every one of the ten bindings exports it, the same shape as
`linalg_numeric_mode()` and `svm_numeric_mode()`:

| binding | export |
|---|---|
| `_mojolearn` | `mojolearn_vendor()` |
| `_mojolearn_<suffix>` | `<suffix>_vendor()` |

`_backend.read_vendor(module)` derives the name from the module name, so
adding a binding to `_MODULES` is enough. The Python surface follows.

* `mojolearn.vendor()` returns `'metal'`, `'cuda'` or `'hip'`, read back from
  the binaries of the current default tier and cross-checked against the
  chosen directory (a disagreement raises, exactly like `numeric_mode()`).
  `None` only for binaries built before the read-back existed.
* `estimator.vendor_used()` beside `numeric_mode_used()` returns the constant
  out of the binary that instance's `_bind()` resolves to.
* `python -m mojolearn verify` and `python -m mojolearn env` print a `vendor`
  line beside `numeric mode`, with how the directory was chosen.
* `mojolearn doctor` reports `vendor loaded:` and counts extensions per vendor
  and per tier.

On macOS the read-back says `metal` and the layout is flat; behavior is
unchanged except that `packaging/macos/smoke.py` now asserts the read-back.

## 5. The failure message on a Linux box with no supported GPU

There is no CPU path. When the wheel layout is present and neither vendor
has evidence, `import mojolearn` raises `ImportError` with the whole probe
table. For example

```
mojolearn: NO SUPPORTED GPU FOUND ON THIS BOX, and there is no CPU path in
this package. This install carries binary sets for ['cuda', 'hip'] under
/…/site-packages/mojolearn. What was looked for and what was found:
  cuda:
    /dev/nvidiactl               absent
    /dev/nvidia0                 absent
    libcuda.so.1                 not loadable
    libcuda.so                   not loadable
  hip:
    /dev/kfd                     absent
    /dev/dri/renderD128          absent
    libamdhip64.so.7             not loadable
    libamdhip64.so.6             not loadable
    libamdhip64.so               not loadable
  MOJOLEARN_VENDOR is not set.

A device node or a driver library for one of the sets above must be visible
to this process. In a container that means the GPU is passed through
(`--gpus all` for NVIDIA, `--device /dev/kfd --device /dev/dri` for AMD). If
the device is present and this probe is wrong, MOJOLEARN_VENDOR=cuda or
MOJOLEARN_VENDOR=hip picks the directory directly and the first fit reports
the runtime's own error.
```

Every path and library is named with its result. The message does not guess
at a cause beyond the container hint, and it names the escape.

**The escape's promise is conditioned on "if the device is present", and
outside that condition the two vendors do not behave alike.** On a box with
NO device and NO runtime of that vendor at all, `MOJOLEARN_VENDOR=cuda`
imports and then raises a clean Python exception at the first fit, naming the
library it could not open; `MOJOLEARN_VENDOR=hip` aborts the process with
SIGILL during import, with no traceback. Measured 2026-08-30 against the
finished 0.3.0 wheel (`bench/results/wheels/LEGS_2026-08-30.md`). Forcing a
vendor whose device is absent is outside what this paragraph offers, so the
message is not wrong, but the asymmetry is real and a user who mistypes the
variable on the wrong box gets a crash from one branch and an error from the
other.

**FIXED 2026-08-30, in the selector rather than in the binary.** A forced
vendor with no device node and no loadable driver library is refused in
Python before anything is dlopened, naming what it looked for and offering
`MOJOLEARN_VENDOR_FORCE=1` for a box whose device really is present and whose
probe is wrong. Making the hip binary itself raise the way the cuda one does
is still owed and belongs to MAX rather than to this package; the refusal
above means a user no longer meets it by accident.

## 6. Build and pack

Each vendor's sets are built on that vendor's box, with the existing
`bindings/build_*.sh` (gates off, as `tools/e1_bootstrap.sh` phase 9 already
runs them), by `packaging/linux/build_sets.sh`:

1. thirty builds, the three tiers as three parallel jobs (serial took about
   fifty minutes on a rented RTX 4090; the parallel time is unmeasured);
2. the vendor AND the architecture READ BACK from every binary (a bare
   `ExtensionFileLoader` import for the vendor, `strings` for the
   architecture), all thirty must agree on ONE of each, a typed
   `MOJOLEARN_GPU_ARCHS` must EQUAL the read-back (until 2026-08-30 only
   one of the ten build scripts even read that variable, so a leg that set
   it built one binding as asked and twenty-seven for the box's own
   device), and those answers name the set directory;
3. the sets MOVED out of `python/mojolearn/` into `sets/<vendor>/<arch>/`;
4. `packaging/linux/stage_libs.py` walks the ELF `DT_NEEDED` closure (pure
   Python reader, `patchelf` from a throwaway venv for the write), stages
   the MAX runtime under `sets/<vendor>/.libs/`, sets RUNPATH on every
   extension to BOTH candidate `.libs` locations (`$ORIGIN/.libs` and
   `$ORIGIN/../.libs` for the fast set, one level deeper for the tiers) so
   the packer can choose the layout on the Mac without an ELF tool, and
   verifies the closure statically;
5. driver-side libraries (`libcuda.so.1`, `libamdhip64`, `libhsa-runtime64`,
   anything `libnvidia-*`) are NEVER staged; they are the user's driver and
   are recorded in the manifest for the audit step to exclude by name;
6. `manifest.json` with every size and sha256, `SIZES.txt`, and
   `sets/<vendor>.tar.gz` for the fetch.

`packaging/linux/leg.sh <nvidia|amd>` is the one command per vendor. It sets
the phase-9 knobs that make the bootstrap run ONLY `packaging/linux/leg_diag.sh`
(`MOJOLEARN_P9_ONLY_DIAG=1`, new in `tools/e1_bootstrap.sh`, passed through
both legs), which runs `build_sets.sh` and then the on-box gates of section
7, and files the fetched `diag/` under `bench/results/wheels/<stamp>-<vendor>/`.

`packaging/linux/pack_wheel.py`, pure Python on the Mac, takes every
fetched `sets/<vendor>` directory (the same vendor may be given several
times -- each leg builds ONE architecture), refuses an arch-less set, a
duplicate (vendor, architecture), a set whose `readback.txt` or
`arch_readback.txt` disagrees with its directory, or one that lacks any of
the thirty binaries, decides the
`.libs` layout (shared when both closures match by name and sha256), writes
the wheel with a correct RECORD and a METADATA generated from
`python/pyproject.toml` in setuptools 84's field order (`--check-against` a
macOS wheel diffs the two), tags it `linux_x86_64` ON PURPOSE (PyPI refuses
that tag, so the audit cannot be skipped), and writes `SIZES-<v>-linux.json`.
Over 100 MB it stops and prints the numbers.

`packaging/linux/audit.sh` runs `auditwheel show` (the artifact), reads the
measured manylinux level out of it, runs `auditwheel repair --plat <that>`
with `--exclude` for every driver library and every staged library, and
`twine check`, each in `docker run --cpus 2`, one container at a time. The
tag is what `show` measured; `manylinux_2_28` is the expectation and nothing
more until then.

## 7. Gates, all on rented boxes, per "verify reach, not output"

| gate | script | what it proves |
|---|---|---|
| (a) smoke | `packaging/linux/smoke.py`, every tier | every lane in `tools/repeat_run_stability.py` fits; `numeric_mode()` reads back the tier; `vendor()` reads back the box's vendor; `vendor_used()` on one estimator per binding agrees |
| (b) sabotage | `packaging/linux/sabotage.py` | the other vendor's directory filled with THIS vendor's binaries is ignored by the probe and REFUSED by name when forced with `MOJOLEARN_VENDOR`; a poisoned directory is ignored and fails loud when forced; the correct set REMOVED raises the no-GPU refusal naming the found device and the set carried, and does not fall through to the other vendor; a fast binary under `deterministic/` is refused by the tier read-back on the new layout |
| (c) audit | `packaging/linux/audit.sh` | `show.txt`, `repair.txt`, `twine.txt` kept under `python/dist/audit/` |
| (d) no GPU | `packaging/linux/nogpu.py` | `CUDA_VISIBLE_DEVICES=`/`HIP_VISIBLE_DEVICES=` (expected to import: those hide devices from the runtime, not nodes from the filesystem, and the record says what the first fit did); `unshare -rm` with `/dev/null` bind-mounted over every probed path (the real test where `unshare` is permitted); docker with no device passed through where a daemon exists (the DigitalOcean droplet, not a RunPod pod). NOT TESTED is a recorded outcome, never a pass |
| install smoke | `packaging/linux/leg_diag_install.sh` via `leg.sh <vendor> install` | `pip install mojolearn==X.Y.Z` from TestPyPI into a clean venv on each vendor, cwd outside the checkout, smoke in all three tiers |

## 8. Release build flow, in order

See `docs/PYPI_RELEASE.md` section 9 for the same list with the paths every
step writes. In short, the NVIDIA leg (CUDA sets + gates) and the AMD leg (HIP
sets + gates), both fetched; `pack_wheel.py`; `audit.sh`; upload the repaired wheel
to TestPyPI beside the macOS wheel; an install-smoke leg on each vendor; then
PyPI.

## 9. What the two legs of 2026-08-30 measured, and what is still owed

Both legs ran green on 2026-08-30 (`bench/results/wheels/LEGS_2026-08-30.md`
is the record; `2026-08-30_112521-nvidia` and `2026-08-30_112523-amd` are the
directories). What this section used to list as unknown, and what came back:

**MEASURED, and no longer open:**

* `has_*_gpu_accelerator()` DOES fold to the expected constant inside a
  `--emit shared-lib` build. Thirty binaries read back `cuda` on the H100 and
  thirty read back `hip` on the MI325X, in `readback.txt` on each set.
* The Linux MAX runtime closure is 3,200,416 bytes, four libraries, and the
  two vendors' closures are BYTE-IDENTICAL (same sha256 on all four).
  `pack_wheel.py` therefore chose `shared mojolearn/.libs`, one copy for both
  vendors. `driver_libs_not_staged` is EMPTY on both: nothing driver-side is
  bundled, because the extensions carry no `DT_NEEDED` on `libcuda` or
  `libamdhip64` at all. MAX dlopens them, which is why the selector's probe
  has to load them with `ctypes.CDLL` rather than read an ELF header.
* The manylinux level is **`manylinux_2_35_x86_64`**, measured by
  `auditwheel show`, driven by `GLIBCXX_3.4.30` and `GLIBC_2.34` in the
  referenced versioned symbols, which is the Ubuntu 22.04 toolchain both
  boxes run. **glibc 2.35 puts Ubuntu 22.04 and Debian 12 in and leaves
  RHEL 9 out by one minor version** (it ships 2.34). Lowering the floor means
  building on an older base image; it is not a retag.
* `auditwheel repair` DOES honor `--exclude` for libraries it resolves
  through `$ORIGIN`. Zero `mojolearn.libs/` entries in the repaired wheel, so
  the staged closure is not shipped twice.
* Thirty parallel builds fit in one lease with room to spare: 915 s on the
  H100, 188 s on the MI325X. This document previously guessed "about fifty
  minutes" of serial building.
* The packed two-vendor wheel is 26,205,942 bytes, well under PyPI's 100 MB.
* Gate (d), the no-GPU refusal, is CLOSED, and NOT on a rented box. Both
  boxes are the wrong place to ask it: the RunPod pod is a container with
  seccomp-refused `unshare` and no docker daemon, and the DigitalOcean image
  ships without docker. `packaging/linux/nogpu_local.sh` runs the honest
  test on the Mac against a fetched set, in `python:3.12-slim` with no
  device passed through, and the finished wheel refuses by name with the
  whole probe table. The gate reads the probed names out of
  `_backend._PROBE` rather than retyping them.
* The finished wheel installs and behaves on a Linux box: `pip install`
  works, the default import refuses naming both sets, `MOJOLEARN_VENDOR=metal`
  is refused by name, and `MOJOLEARN_VENDOR=cuda` loads the binaries through
  the shared `.libs` RUNPATH outside any build environment.

**MEASURED 2026-08-30, the cost of an architecture, and how much of it is
waste.** Section sizes summed over all 30 binaries of one set:

                    CUDA set (65 MB)   HIP set (43 MB)
    .text            19 MB              19 MB     host CPU code
    .rodata          36 MB              15 MB     GPU device code

**The 19 MB of `.text` is IDENTICAL in both**, because it is the same Mojo
source compiled for x86-64, and it is identical across architectures of one
vendor for the same reason. So roughly 30 percent of every CUDA set and 45
percent of every HIP set is bytes the wheel already carries, and a zip does
not dedupe it because it compresses each member independently. Six sets
duplicate about 114 MB of host code.

It cannot be shared today. `mojo build` emits one COMPLETE shared library
targeting one GPU architecture, and there is no way to ask for a host
library plus swappable device images; that is the same limit that rejects a
comma-separated architecture list. The irreducible part is the `.rodata`,
which really is different machine code per architecture.

**If MAX ever gains multi-architecture output the wheel roughly halves**, from
about 324 MB uncompressed for three CUDA and three HIP sets to about 172 MB.
Recorded here as a measurement so whoever revisits it does not have to
re-derive it.

**STILL OWED:**

* **The cross-architecture question is now ANSWERED for the exact case and
  OWED for the family case.** Same-architecture-only was measured on the
  A40 (27 of 29 lanes down, LEGS_2026-08-30.md), which is what forced the
  architecture axis. What remains owed is the WITHIN-FAMILY measurement the
  selector's cuda fallback relies on: a set built `sm_80` (via
  `MOJOLEARN_GPU_ARCHS=sm_80`) install-smoked green on an sm_86 or sm_89
  box. NVIDIA documents that compatibility; this tree does not ship a claim
  on documentation alone, so until that leg is green the wheel's claim is
  "runs on the architectures it carries", and the family fallback is an
  escape hatch rather than a promise. The per-architecture build legs
  themselves (one leg per carried architecture, both vendors) are owed the
  same way. **The host CPU IS part of this question and this sentence used
  to say it was not.** 0.3.0's Linux wheel carried AVX-512 in its host code
  and died with SIGILL on an L40 whose host was an AMD EPYC 7773X. Fixed in
  `cfb665d2` by pinning `--target-cpu x86-64-v3` and gating with
  `packaging/linux/isa_baseline_linux.py`. See the correction at the end of
  `bench/results/wheels/LEGS_2026-08-30.md`.
* ~~**`MOJOLEARN_VENDOR=hip` aborts rather than raising when the HIP runtime
  is absent entirely.**~~ FIXED 2026-08-30 by refusing in the selector; see
  section 3 and section 5. The original finding, kept because the measurement
  stands and the underlying binary behaviour is unchanged: Forcing hip on a Linux box with no ROCm at all kills
  the process with SIGILL during module init, where forcing cuda on a box
  with no CUDA imports fine and then raises a clean Python exception at the
  first fit. Section 5's message offers exactly this override and promises
  "the first fit reports the runtime's own error"; on the hip branch that
  promise is not kept. Observed under x86 emulation on the Mac against the
  finished wheel; reproducing it on real x86-64 with no ROCm is owed, and
  then either the init path raises the way cuda's does or the message stops
  promising it. NOT on the default path: with `MOJOLEARN_VENDOR` unset the
  probe refuses first, which is verified against this same wheel.
* Whether the MAX runtime dlopens libraries that are in no `DT_NEEDED` (the
  ELF walk cannot see those). The clean-venv install smoke on a box without
  the pixi environment is the check.
* `nogpu.py`'s `namespace` route is fixed but UNRUN on a box. It covered
  device nodes by bind-mounting `/dev/null` over them until 2026-08-30 and
  could not pass, because a bind mount leaves the path there for
  `os.path.exists` to find; it now mounts a minimal tmpfs OVER `/dev`. That
  technique is verified in a privileged container on the Mac (`/dev/kfd`
  True before, False after) but has not run on a rented box.
