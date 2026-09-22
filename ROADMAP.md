# Roadmap

mojolearn's direction is fixed by its default contract. Every algorithm returns
the same bits on Apple, NVIDIA and AMD GPUs and on CPUs, in the `identical`
mode, and every claim to that effect is checked by the shipped verifier. The
work below extends that contract to more algorithms, more hardware and more
workloads, and makes each backend run closer to its hardware's best.

This file states direction, not commitments. It carries no dates. What has
shipped is in the [changelog](CHANGELOG.md), and what is verified today is in
the [support matrix](SUPPORT_MATRIX.md) and [identity paths](IDENTITY_PATHS.md).

## Current focus

- **Verified coverage for every algorithm.** Each public algorithm should be
  exposed by the verifier with recorded CPU and GPU references on all three
  GPU vendors. Remaining work is filling the outstanding vendor references,
  closing lanes with missing fixtures, and keeping CPU and GPU records in
  agreement.
- **Broader GPU coverage within NVIDIA, AMD and Apple.** More architectures
  per vendor in the released wheels, confirmed by installed-wheel checks
  rather than inferred from a GPU family name.
- **Performance tuning per backend.** The numerical contract fixes results,
  not tiling, thread layout or scheduling, so each backend can be tuned on its
  own. Priorities are fixed per-fit costs in tree learners, synchronization
  costs on Metal, and kernel geometry on AMD. Performance is judged on large
  real data (see [CONTRIBUTING.md](CONTRIBUTING.md#performance-claims)).
- **The CPU surface.** CPU-only installs train and predict. The aim is CPU
  training for every algorithm that has a GPU path, with the same bits, and
  sensible CPU defaults for gradient boosting.

## Planned work

- **Neural training.** Extend identical training for the transformer, Mamba
  and Samba stacks and the decoder language model to larger models and
  longer runs, and broaden serving of models trained in other frameworks.
- **Cross-vendor and multi-GPU training.** Move the live cross-vendor
  training API beyond experimental status, and extend multi-GPU execution
  plans to more estimators with a placement check for each.
- **Algorithm breadth.** New estimators enter in the `identical` mode and
  join the verifier when they land. The `fast` and `deterministic` modes stay
  limited to the tree learners.
- **Verifier usability.** Quicker checks, clearer coverage reports, and
  a verifier that states exactly what an installed wheel has and has not
  been checked for.
- **Releases.** Regular PyPI releases of the macOS arm64 wheel and the Linux
  x86-64 wheel carrying CUDA and HIP, each checked as the installed wheel on
  CPU and Apple GPU before publication.

## Out of scope

- GPU backends other than CUDA, HIP and Metal. Mojo targets those three.
- Reduced-precision arithmetic in place of FP32 for the identity guarantee.
  BF16 and INT8 remain storage formats.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the contributor quickstart in
[docs/START_HERE.md](docs/START_HERE.md).
