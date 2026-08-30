# The Linux wheel: one name, two vendors, six binary sets

Design note, 2026-08-29. Decided by Andrew the same day. ONE PyPI name,
`mojolearn`, with the vendor detected at import. Two wheels per release.

| wheel | tag | carries |
|---|---|---|
| macOS (shipping since 0.1.0) | `py3-none-macosx_11_0_arm64` | Metal, three tiers, ten extensions each |
| Linux (this note) | `py3-none-manylinux_<measured>_x86_64` | CUDA AND HIP, three tiers each, six sets, sixty extensions |

Nothing in this note has run. Every script it names was written under the
no-run order of 2026-08-29 and is committed UNRUN; the commands in section 8
are what runs them, and section 9 lists every assumption a run has to
confirm. Any sentence here that a run contradicts is fixed in the commit that
records the run.

## 1. Why one name and not `mojolearn-cu12` and `mojolearn-rocm`

Because the size allows it, on the numbers we have. The 0.1.0 macOS wheel is
10.3 MB compressed for TWO sets of ten `.so` plus 2.7 MB of MAX dylibs. Six
Linux sets extrapolate to roughly 30 MB compressed, under PyPI's 100 MB
per-file limit with room. The ONE number that extrapolation does not cover
is the size of the MAX runtime libraries a Linux extension links. The macOS
closure is four dylibs and 2.7 MB, and the Linux closure for CUDA and HIP has
never been measured. `packaging/linux/stage_libs.py` measures it and
`packaging/linux/pack_wheel.py` refuses to write a wheel over the limit. If
it refuses, the numbers are reported before any name is split, and the split
names are not written down anywhere until then.

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
    cuda/_mojolearn*.so                    fast
    cuda/deterministic/_mojolearn*.so
    cuda/identical/_mojolearn*.so
    hip/_mojolearn*.so
    hip/deterministic/_mojolearn*.so
    hip/identical/_mojolearn*.so
    .libs/*.so             ONE shared MAX runtime closure, when both vendors'
                           closures are byte-identical (pack_wheel.py decides)
    cuda/.libs/*.so        otherwise one per vendor
    hip/.libs/*.so
```

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

Then, whichever way the directory was chosen, EVERY binary loaded from it
is asked what it was compiled for, and one that disagrees with the directory
is refused at import with the path, the directory and both answers. This is
the same refusal, in the same place, as the tier read-back that caught the
Aug 24 stale-`_MODULES` bug, and it exists for the same reason. A CUDA `.so`
under `hip/` imports cleanly on an NVIDIA box and touches no device until the
first fit, so nothing later would catch it.

The order of trust is therefore what the binary says, then what the
environment says, then what the box appears to have. `MOJOLEARN_VENDOR`
cannot relabel a binary; it can only open the other directory on a box that
does not have that device, and that fails at the first device call with the
runtime's own error, which is the honest outcome for a forced wrong choice.

## 4. The read-back from the binary

`mojo_only/vendor.mojo` defines `COMPILED_VENDOR`, a compile-time constant
resolved from `std.sys.info.has_amd_gpu_accelerator()`,
`has_nvidia_gpu_accelerator()` and `has_apple_gpu_accelerator()`, the same
predicates `mojo_only/kernel_matrix.mojo` uses to pick `DETECTED_COLUMN`,
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

## 6. Build and pack

Each vendor's sets are built on that vendor's box, with the existing
`bindings/build_*.sh` (gates off, as `tools/e1_bootstrap.sh` phase 9 already
runs them), by `packaging/linux/build_sets.sh`:

1. thirty builds, the three tiers as three parallel jobs (serial took about
   fifty minutes on a rented RTX 4090; the parallel time is unmeasured);
2. the vendor READ BACK from every binary with a bare `ExtensionFileLoader`
   import, all thirty must agree, and that answer names the set directory;
3. the sets MOVED out of `python/mojolearn/` into `sets/<vendor>/`;
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

`packaging/linux/pack_wheel.py`, pure Python on the Mac, takes both
`sets/<vendor>` directories, refuses a set whose `readback.txt` disagrees
with its directory or that lacks any of the thirty binaries, decides the
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

**STILL OWED:**

* **Whether a set built on one machine's CPU and GPU runs on another's.**
  This is the big one and it now has TWO reasons. MAX embeds device kernels
  for the build target, and separately the host-side code carries AVX-512
  with NO `cpuid` dispatch anywhere in either set. Under x86 EMULATION on the
  Mac the hip set raises `Illegal instruction` during module init while the
  cuda set imports fine; AVX-512 was tested as the cause and RULED OUT, and
  the same hip set ran 29 smoke lanes in all three tiers on the MI325X that
  built it. That points at the emulator, but it is an inference. The
  install-smoke leg on a box that is NOT the build box, ideally a different
  NVIDIA GPU model, is the only evidence either way. Until then the wheel's
  claim is "runs on the architectures it was smoked on", named in the
  CHANGELOG entry that ships it.
* Whether the MAX runtime dlopens libraries that are in no `DT_NEEDED` (the
  ELF walk cannot see those). The clean-venv install smoke on a box without
  the pixi environment is the check.
* `nogpu.py`'s `namespace` route is fixed but UNRUN on a box. It covered
  device nodes by bind-mounting `/dev/null` over them until 2026-08-30 and
  could not pass, because a bind mount leaves the path there for
  `os.path.exists` to find; it now mounts a minimal tmpfs OVER `/dev`. That
  technique is verified in a privileged container on the Mac (`/dev/kfd`
  True before, False after) but has not run on a rented box.
