# Install

The quickest route is the [install picker](https://mojolearn.org/get-started), which gives the command
for your platform and GPU. This page has the details.

## Requirements

| | |
|---|---|
| Python | CPython 3.10, 3.11, 3.12, 3.13 and 3.14 |
| macOS | Apple silicon (arm64). One wheel with Apple Metal GPU support and CPU support. |
| Linux | x86-64. The core wheel plus the `mojolearn-nvidia` and `mojolearn-amd` GPU packages, installed automatically. |

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

## Check your machine

```sh
mojolearn doctor
python -m mojolearn verify --quick
```

`mojolearn doctor` reports what the installed wheel supports on this machine. `verify` checks the
installed package against the reference bits recorded for this release; see
[Verify on your hardware](user_guide/verify.md).

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
