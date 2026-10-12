# The reproducibility contract

## What `identical` promises

Run the same mojolearn release on the same input bytes, model state and settings, in a supported
configuration, and NVIDIA and AMD GPUs return the same output bits. That covers predictions, fitted
models, training updates and saved checkpoints. This is the release policy: every release checks
the NVIDIA and AMD results against each other before it ships.

Apple GPU and CPU execution are also available. Their results are recorded beside the NVIDIA and AMD
results, and the configurations the [paper](https://mojolearn.org/paper.pdf) tested also agreed on Apple
and the CPU. The release requirement covers NVIDIA and AMD.

`identical` is the default numeric mode. See [Numeric modes](../user_guide/numeric_modes.md) for the others.

## What it does not promise

- **Agreement with other libraries.** Agreement is between mojolearn executions. A mojolearn result
  can differ from scikit-learn's, PyTorch's or the framework that originally trained an imported model.
- **Agreement across releases.** While mojolearn is in alpha, a release may change `identical` output
  bits, for example when a kernel needs a different summation order. Compare results produced by the
  same release, and keep the release version with any result you need to reproduce.
- **Every input and every GPU.** The claim covers certified fixtures and configurations, recorded
  per release. A GPU architecture the release does not carry is not covered.
- **Accuracy, fairness or fitness for a use.** Bitwise agreement is evidence about computational
  reproducibility only. Model quality needs its own evaluation.

## Why GPUs disagree, and what mojolearn fixes

Floating-point addition is not associative, so adding the same values in a different order can change
the last bits of the result. GPUs choose that order through thread layout, atomics and library
reductions, and compilers and math libraries make their own choices about fused multiply-add,
square root, `exp` and `log`. A one-bit difference can change a tree's best split or a training
update, and every later step inherits it.

For each pathway that can move a bit, `identical` does one of three things:

| Move | When | Example |
|---|---|---|
| Pin | The operation is deterministic once a parameter is fixed | The same reduction width on every vendor |
| Replace | The operation is order-dependent in itself | Floating-point atomics become fixed-point integer accumulation |
| Refuse | Neither is available yet | A configuration without an identity path raises a named error |

Thread layout, memory placement and scheduling may still differ between devices; only the operations
that determine the output bits are fixed. Neural training and inference use FP32 arithmetic
throughout. The full enumeration is the
[identity-path ledger](https://github.com/mojolearn/mojolearn/blob/main/IDENTITY_PATHS.md).

## What it costs

A fixed arithmetic order and portable math routines give up some of the throughput that
vendor-specific implementations use. The [measurements](https://mojolearn.org/measurements) record
timings and quality for complete implementations against other libraries on the same GPUs; a timing
ratio there does not isolate the cost of identity alone.

## Checking it yourself

`python -m mojolearn verify` compares the installed release against the reference bits recorded for
it on your hardware. To compare two machines, run the same script with the same release and inputs on
both and compare hashes of the outputs. See [Verify on your hardware](../user_guide/verify.md).

If two supported configurations disagree, that is a bug:
[open an issue](https://github.com/mojolearn/mojolearn/issues) with the release, both devices
(`mojolearn doctor` output) and a script that reproduces it.
