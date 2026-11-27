# Support matrix

What mojolearn {{release}} supports: platforms, Python versions, devices and numeric modes. Release notes
are in [Release notes](../community/release_notes.md).
`mojolearn doctor` reports what the installed wheel supports on your machine.

## Platforms and wheels

| Platform | Packages | GPU backend | GPU architectures | CPU |
|---|---|---|---|---|
| macOS arm64 | `mojolearn` | Apple Metal | Apple silicon, built at the M1 instruction-set floor | Apple silicon |
| Linux x86-64 | `mojolearn-core` plus `mojolearn-nvidia` and `mojolearn-amd` (`pip install mojolearn` installs all three) | NVIDIA CUDA, AMD HIP | CUDA native {{nvidia_arches}} and `sm_80` PTX, HIP {{amd_arches}} | x86-64 |

Both platforms carry the CPU bindings, so both train and predict on a machine with no GPU.

NVIDIA needs only the driver (`libcuda`); no CUDA toolkit is required. AMD needs the `amdgpu` driver and
the ROCm HIP runtime (ROCm 6 or 7).

The NVIDIA package carries native {{nvidia_arches}} code and `sm_80` PTX.
The loader prefers compatible native code. On other NVIDIA GPUs with compute
capability 8.0 or later, including A100 and H100, the driver compiles the bundled
PTX. AMD GPUs require an exact architecture match.

## Python versions

CPython 3.10, 3.11, 3.12, 3.13 and 3.14, on both platforms.

## Verified devices

| Vendor | Device | Architecture | Backend |
|---|---|---|---|
| Apple | M4 | Apple silicon | Metal |
| NVIDIA | RTX 4090 | `sm_89` | CUDA |
| NVIDIA | L40S | `sm_89` | CUDA |
| AMD | Instinct MI325X | `gfx942` | HIP |
| AMD | Instinct MI300X | `gfx942` | HIP |
| CPU | x86-64 (Linux) | x86-64 | CPU bindings |
| CPU | Apple silicon (macOS) | arm64 | CPU bindings |

Other GPUs of a carried architecture (for example the L40, L4 and RTX 6000 Ada on `sm_89`) load the
same code. Use [`verify`](../user_guide/verify.md) to check them.

## Numeric modes

| Mode | Contract |
|---|---|
| `identical` | The default. In `identical` mode, Mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs. |
| `fast` | Opt-in mode for throughput. Results can vary across runs and devices. |

On Apple GPUs, `fast` requires fast kernels for the selected operation. On Linux,
including CPU-only installs, `fast` uses the identical kernels when the installation
has no separate fast kernels. See [Numeric modes](../user_guide/numeric_modes.md).

## The CPU surface

On a CPU-only install, `fit` trains on the CPU for every estimator with a CPU binding and refuses by name
otherwise, and saved models predict on the CPU. Every CPU binding is checked bit for bit against the
recorded GPU results. See [Devices and the CPU](../user_guide/devices.md) and
[Saving and loading models](../user_guide/saving_models.md).

## What counts as verified

A cross-vendor result is accepted only when the same source, input bytes and numeric profile ran on
each vendor, each run recorded its device and toolchain, the raw-bit hashes agree, and refused
configurations are reported as refusals rather than passes. `python -m mojolearn verify` checks your installed package against the shipped reference
results.
