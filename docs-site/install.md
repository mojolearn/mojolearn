# Install

The quickest route is the [install picker](https://mojolearn.org/get-started), which gives the command
for your platform and GPU. This page has the details.

## Requirements

| | |
|---|---|
| Python | CPython 3.10, 3.11, 3.12, 3.13 and 3.14 |
| macOS | Apple silicon (arm64). One wheel with Apple Metal GPU support and CPU support. |
| Linux | x86-64, glibc 2.35 or newer. The core wheel plus the `mojolearn-nvidia` and `mojolearn-amd` GPU packages, installed automatically. |

## pip

```sh
pip install mojolearn
```

On Linux x86-64, pip also installs `mojolearn-nvidia` and `mojolearn-amd`, and mojolearn selects the
backend for the GPU it finds. Both platforms also train and predict on the CPU
(see [Devices and the CPU](user_guide/devices.md)).

Optional extras add NumPy without changing any numerical implementation:

```sh
pip install "mojolearn[verify]"   # NumPy for verification fixtures
pip install "mojolearn[numpy]"    # NumPy-backed conveniences (ANN filtering, TSNE, sequence APIs)
```

A plain `pip install mojolearn` is NumPy-free; APIs that need NumPy say so and name the install command.

## GPU architectures in this release

Device code is architecture specific. A GPU architecture the release does not carry needs a source
build; mojolearn never substitutes the CPU for a GPU it cannot serve.

| Vendor | Package | Architectures | Examples |
|---|---|---|---|
| Apple | `mojolearn` (macOS) | Apple silicon, built at the M1 instruction-set floor | M1 to M4 families |
| NVIDIA | `mojolearn-nvidia` | `sm_89` | L40S, L40, L4, RTX 4090, RTX 6000 Ada |
| AMD | `mojolearn-amd` | `gfx942` (exact match required) | Instinct MI300X, MI325X |

NVIDIA needs only the driver (`libcuda`); no CUDA toolkit is required. AMD needs the `amdgpu` driver
and the ROCm HIP runtime (`libamdhip64`, ROCm 6 or 7).

The 0.8.42 NVIDIA wheel contains native `sm_89` code and **no PTX fallback**. Its Ada binaries do not
run on A100 (`sm_80`), RTX 30-series/A40 (`sm_86`), H100/H200 (`sm_90`) or Blackwell GPUs.
The source loader supports an `sm_80` PTX set for compute capability 8.0 and newer when that set is
included in a build, but it is not shipped in this release because of PyPI's per-file size limit.

## Verify the results

```sh
python -m pip install "mojolearn[verify]"
python -m mojolearn verify --quick
```

The extra adds NumPy for verification. `verify` checks computations against the reference bits recorded for this release; see
[Verify on your hardware](user_guide/verify.md).

## Troubleshooting

If the package fails to load, run the optional diagnostics command:

```sh
mojolearn doctor
```

It reports the installed packages, GPU architecture sets, selected backend and numerical mode, and
whether the package imports. An import pass is not a numerical verification result. Older versions,
including 0.8.42, contain a stale support note claiming no Linux wheel is published; use the platform
and GPU requirements above instead of that note.

(from-source)=
## From source

A source build targets the GPU in the build machine unless `MOJOLEARN_GPU_ARCHS` says otherwise.
Building needs the Mojo toolchain, which the repository's `pixi` environment provides:

```sh
git clone https://github.com/mojolearn/mojolearn
cd mojolearn
pixi run probe        # reports the toolchain and the device it finds
```

The repository's [CONTRIBUTING](https://github.com/mojolearn/mojolearn/blob/main/CONTRIBUTING.md) guide
describes the build tasks. Check a source build's results with `verify`.
