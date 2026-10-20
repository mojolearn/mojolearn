# Devices and the CPU

mojolearn runs on Apple, NVIDIA and AMD GPUs, and trains and predicts on the CPU when no GPU is
present. It picks the backend at import and never quietly substitutes one device for another.

## Which backend runs

```python
import mojolearn

print(mojolearn.vendor())     # 'metal', 'cuda', 'hip', or 'cpu' on a CPU-only install
print(mojolearn.gpu_arch())   # e.g. 'sm_89' or 'gfx942'; None on macOS
```

`vendor()` is read back from the loaded binaries, not guessed from the platform. `gpu_arch()` names the
architecture directory the process loaded from, and `gpu_arch_how()` says how it was chosen (exact
match, environment variable, and so on).

The backend is chosen in this order:

1. On macOS, the wheel carries the Apple Metal build.
2. On Linux, `MOJOLEARN_VENDOR` (`cuda` or `hip`) picks the backend when it is set.
3. Otherwise mojolearn probes the machine for an NVIDIA or AMD device and its driver libraries.

`MOJOLEARN_VENDOR=cpu`, set before import, runs the CPU bindings on purpose on any platform.

## Architecture matching

GPU device code is specific to an architecture. In {{release}} the wheels carry:

| Vendor | Package | Architectures | Examples |
|---|---|---|---|
| Apple | `mojolearn` (macOS arm64) | Apple silicon, built at the M1 instruction-set floor | M1 to M4 families |
| NVIDIA | `mojolearn-nvidia` | {{nvidia_arches}} | {{nvidia_examples}} |
| AMD | `mojolearn-amd` | {{amd_arches}} | {{amd_examples}} |

The loader matches the device against what the install carries:

- **NVIDIA:** an exact match, or else the highest carried architecture of the same compute-capability
  family that does not exceed the device. It never crosses a major version, so a device whose family
  has no carried architecture has no compatible build and the import refuses.
- **AMD:** the match must be exact. AMD code objects are ISA-specific, so a device other than
  {{amd_arches}} refuses.
- `MOJOLEARN_GPU_ARCH` names an architecture directory explicitly; it must be one the install carries.

A GPU architecture the release does not carry needs a [source build](../install.md#from-source).

## No CPU substitution for a GPU

If the machine has a GPU that mojolearn cannot serve, or a GPU without its package, the import
raises an error that names what it looked for, what it found and the package to install. It does not
fall back to the CPU, because a GPU machine silently computing on the host would mislabel every
result. Install the package for the GPU at the same version as `mojolearn-core`, for example:

```sh
pip install "mojolearn-nvidia=={{release}}"
```

To compute on the CPU on such a machine anyway, set `MOJOLEARN_VENDOR=cpu` before import.

## The CPU surface

Both wheels carry CPU bindings under `mojolearn/host/`, so a machine with no GPU still imports
mojolearn and `vendor()` answers `cpu`.

| Task | On a CPU-only install |
|---|---|
| Training | `fit` trains on the CPU for every estimator with a CPU binding. Others refuse by name. |
| Prediction from a saved model | Forests, gradient boosting, neighbors, linear models, PCA and SVD, scalers, kernel methods, SVMs, Gaussian mixtures and processes, HDBSCAN, UMAP transform, IVF-Flat search, ARIMA and Holt-Winters forecasts, among others. |
| Language models | `LanguageModelInference` runs the forward pass. |
| Numeric mode | `identical` only (`fast` is available on Apple GPUs only). |

Every CPU binding is checked bit for bit against the recorded GPU results. A model trained on any supported GPU can therefore be saved and
served on a CPU machine; see [Saving and loading models](saving_models.md).

Some configurations have no CPU path and refuse by name on a CPU-only install. Among them are gradient
boosting training with sample weights, with CTR-encoded categorical columns, and some categorical or
eval-set combinations outside symmetric trees. See the [support matrix](../notes/support_matrix.md#the-cpu-surface).

A refusal looks like any other Python error: it names the estimator or binding and the missing
function, and nothing else runs in its place.

`MOJOLEARN_CPU_THREADS` sets the number of host threads for forest and gradient boosting prediction on
the CPU (`0` or unset is one
per physical core, `1` is the calling thread). Each thread owns whole rows, so no output bit depends on
the thread count.

## Check your installation

```sh
mojolearn doctor
mojolearn doctor --json --output doctor.json
```

`mojolearn doctor` reports what the installed wheel supports on this machine: the version, the numeric
mode, the vendor and GPU architecture selected, and why. It runs in a separate module, so it still works
when importing `mojolearn` fails, which is when you most need it. It omits private environment data such
as usernames and home directories.

For an issue, `mojolearn bug-report reproducer.py` builds a support bundle for you to review before
uploading.

To confirm results, not just the installation, run the verifier; see
[Verify on your hardware](verify.md).
