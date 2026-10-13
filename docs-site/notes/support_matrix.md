# Support matrix

What mojolearn 0.8.41 supports: platforms, Python versions, devices and numeric modes. Release notes
are in [Release notes](../community/release_notes.md).
`mojolearn doctor` reports what the installed wheel supports on your machine.

## Platforms and wheels

| Platform | Packages | GPU backend | GPU architectures | CPU |
|---|---|---|---|---|
| macOS arm64 | `mojolearn` | Apple Metal | Apple silicon, built at the M1 instruction-set floor | Apple silicon |
| Linux x86-64 | `mojolearn` plus `mojolearn-nvidia` and `mojolearn-amd`, installed automatically | NVIDIA CUDA, AMD HIP | CUDA `sm_89`; HIP `gfx942` | x86-64 |

Both platforms carry the CPU bindings, so both train and predict on a machine with no GPU.

NVIDIA needs only the driver (`libcuda`); no CUDA toolkit is required. AMD needs the `amdgpu` driver and
the ROCm HIP runtime (ROCm 6 or 7).

In 0.8.41 the NVIDIA package ships `sm_89` code only. A device loads the exact architecture the install
carries, or the highest carried architecture of the same NVIDIA family that does not exceed it, so
other NVIDIA generations (for example A100 and H100) are not served by this release. On AMD the match
must be exact. An architecture the wheel does not carry needs a source build. mojolearn never
substitutes the CPU for a GPU it cannot serve.

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
| `identical` | The default. In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs. |
| `deterministic` | Repeated runs on the same device and build give the same bits. No cross-vendor promise. |
| `fast` | Apple GPUs only. Tuned for throughput. No identity or repeatability promise. |

| Family | `identical` | `deterministic` | `fast` |
|---|---|---|---|
| Gradient boosting | yes | yes | Apple GPUs |
| Random forests | yes | yes | Apple GPUs |
| Extra Trees | yes | yes | Apple GPUs |
| Other tree models (decision trees, AdaBoost, random-trees embedding, isolation forest) | yes | no | Apple GPUs |
| Neural networks, GPU surface | yes | no | Apple GPUs |
| Every other family | yes | no | Apple GPUs |
| CPU bindings (all families) | yes | no | no |

`fast` is opt-in and tuned on Apple. `fast` is available on Apple GPUs only. Asking a family for a mode it does not ship raises
a named error. Bits may change between releases. See [Numeric modes](../user_guide/numeric_modes.md).

## The CPU surface

On a CPU-only install, `fit` trains on the CPU for every estimator with a CPU binding and refuses by name
otherwise, and saved models predict on the CPU. Every CPU binding is checked bit for bit against the
recorded GPU results. A few gradient boosting training configurations have no CPU path (sample weights,
CTR-encoded categorical columns, and some categorical and eval-set combinations outside symmetric
trees). See [Devices and the CPU](../user_guide/devices.md) and
[Saving and loading models](../user_guide/saving_models.md).

## What counts as verified

A cross-vendor result is accepted only when the same source, input bytes and numeric profile ran on
each vendor, each run recorded its device and toolchain, the raw-bit hashes agree, and refused
configurations are reported as refusals rather than passes. `python -m mojolearn verify` checks your installed package against the shipped reference
results.
