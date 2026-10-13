# Frequently asked questions

## Why do different GPUs usually give different bits?

Floating-point addition is not associative, and GPU vendors differ in reduction order, fused
multiply-add contraction, denormal handling, square root, tie-breaking and elementary functions such as
`exp` and `log`. Recompiling the same source for another GPU does not remove those differences.
mojolearn fixes the operations that decide the output bits (reduction order, rounding and elementary
functions) and leaves tiling, thread layout, memory placement and scheduling free for each backend. Its
kernels are its own, written in Mojo, rather than vendor math libraries whose results depend on the
backend.

## What exactly does `identical` promise?

The same output bits for the same release, input bytes, model state, settings and supported
configuration on NVIDIA and AMD GPUs. Apple GPU and CPU results are recorded beside them. See [Numeric modes](../user_guide/numeric_modes.md).

## Does mojolearn match PyTorch or scikit-learn bit for bit?

No. The agreement is between mojolearn executions: the same release on different supported hardware.
mojolearn's estimators follow scikit-learn's interface conventions, but they compute in mojolearn's own
arithmetic. Weights trained in another framework reproduce under mojolearn's arithmetic, not that
framework's.

## Will bits change between releases?

They can. A release may change `identical` output bits, for example when a
kernel moves to a faster or more accurate fold order. The new bits must still agree on NVIDIA and AMD.
Compare runs made with the same release, and record `mojolearn.__version__` with your results.

## Is `identical` slower than `fast`?

`fast` runs on Apple GPUs only. There it can be faster: `identical` pins the operations that decide the
bits, while `fast` is tuned for Apple GPUs and makes no identity or repeatability promise. `identical` is
the default and the mode to report results in; `fast` is opt-in.

## Is `deterministic` the same as `identical`?

No. `deterministic` promises the same bits on repeated runs on one device and build, with no promise
across vendors. Only gradient boosting, random forests and Extra Trees offer it.

## Which GPUs are supported?

In {{release}}: Apple silicon GPUs (M1 to M4 families) on macOS; NVIDIA {{nvidia_arches}} GPUs (for example
{{nvidia_examples}}); and AMD {{amd_arches}} GPUs ({{amd_examples}}). Other NVIDIA and AMD architectures are
not served by the {{release}} wheels and need a source build. See the
[support matrix](support_matrix.md).

## Can I use mojolearn without a GPU?

Yes. Both wheels include CPU bindings. On a machine with no GPU, `mojolearn.vendor()` answers `cpu`,
`fit` trains on the CPU for every estimator with a CPU binding, and saved models predict on the CPU, all
in `identical` mode. Estimators without a CPU path refuse by name. See
[Devices and the CPU](../user_guide/devices.md).

## Why does mojolearn refuse instead of falling back to the CPU?

On a machine with a GPU it cannot serve, a silent CPU fallback would label host results as GPU results.
The import raises an error naming what it looked for and found instead. To compute on the CPU on
purpose, set `MOJOLEARN_VENDOR=cpu` before import. The same rule applies to modes: asking a family for a
mode it does not ship raises a named error rather than running a weaker mode.

## How do I check that my machine gives the published bits?

Install the verify extra and run the verifier:

```sh
python -m pip install "mojolearn[verify]"
python -m mojolearn verify --quick
```

`python -m mojolearn verify --cross-check` compares your GPU against your own CPU without relying on
the project's recorded table. See [Verify on your hardware](../user_guide/verify.md).

## How do I report a mismatch?

1. Confirm both runs used `identical` and the same release: compare `numeric_mode_used()` and
   `mojolearn.__version__`. Differences under `fast` (Apple GPUs only) are expected and are not defects.
2. Save a verifier report on each machine:
   `python -m mojolearn verify --all --json-out report.json`.
3. Run `mojolearn bug-report reproducer.py` with a minimal reproducer to build a support bundle, and
   review it before uploading.
4. Open an issue with the "Identity divergence" template at
   [github.com/mojolearn/mojolearn/issues](https://github.com/mojolearn/mojolearn/issues). It asks which
   comparison diverged, the hashes, the reproducer and the bundles.

## Can a model trained on one machine be used on another?

Yes. Save it with `save(path)` and load it on any supported device, or predict from it on a CPU with
`mojolearn.host_model(path)`. Under `identical`, the loaded model predicts the same bits. See
[Saving and loading models](../user_guide/saving_models.md).
