# Roadmap

mojolearn's direction is fixed by its default contract.
In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.
CPU coverage is documented per API, and claims are checked by the shipped verifier. The
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
  sensible CPU defaults for gradient boosting. After that, CPU speed: the
  host kernels multithreaded and vectorized under the same pinned-fold rule,
  inference first, judged on the bench board against the CPU learners on
  all cores.

## Planned work

For general neural model development, prioritize composable tensor operations
and automatic differentiation, then build graph compilation, export and reusable
distributed execution on that foundation. These are future capabilities;
the support matrix describes the released API.

- **General automatic differentiation.** Let users compose supported tensor
  operations into their own models and losses and obtain gradients
  automatically. Start with reverse-mode differentiation and reusable module
  and parameter interfaces. Reuse compatible forward and backward kernels;
  define saved-tensor lifetimes, mutation rules and a fixed gradient
  accumulation order for shared parameters and branching graphs. Keep graph
  recording, traversal, memory management and execution in compiled Mojo/native
  code. Check derivative correctness independently from cross-vendor identity,
  including complete optimizer steps on composed models.
- **General model compilation and export.** Capture supported user-defined
  model graphs, optimize them within the numerical contract, and export the
  graph, parameters and numerical profile for inference. Define operator,
  shape and control-flow coverage explicitly, and verify exported execution
  against the original model. Use supported Mojo/Modular interfaces; record
  missing upstream capabilities and wait for support rather than introducing
  unsupported compiler modes or Python execution fallbacks.
- **Broad distributed-training infrastructure.** Extend the existing
  cross-vendor and multi-GPU plans into reusable execution for user-composed
  models, including data parallelism, parameter sharding, tensor and pipeline
  parallelism, and distributed checkpoints. Qualify each supported plan on
  physical devices, with explicit gradient reduction order and complete
  optimizer-state restoration under the `identical` contract.
- **Developer tooling and ecosystem.** Add profiling of compute, transfers,
  synchronization and memory; gradient and graph debugging; and clear shape
  and operator diagnostics. Provide complete examples for custom models,
  losses, training and deployment, then broaden documented integrations with
  model and data tools as those paths are verified.
- **Neural training.** Extend identical training for the transformer, Mamba
  and Samba stacks and the decoder language model to larger models and
  longer runs.
- **Inference support and usability.** Make loading a saved model and
  obtaining predictions or generated text a short, complete workflow.
  First, provide an end-to-end example with a verified real checkpoint pinned
  to a revision: load the model and tokenizer together, accept prompt text,
  and return decoded text without requiring callers to construct token arrays.
  Add end-of-sequence stopping with a maximum-token limit, then streaming and
  reproducible sampling with explicit seeds and settings. Prioritize this
  complete workflow before expanding the architecture list. Keep saved-model
  prediction examples and the model/configuration support matrix clear.
- **Checkpoint and hybrid inference coverage.** Broaden verified checkpoint
  and tokenizer coverage for models trained in mojolearn and other frameworks.
  Qualify whole hybrid models through prompt processing and incremental
  decoding, including attention caches and recurrent state together. Loading
  an already-trained, supported architecture should require no retraining;
  architecture conversion is a separate task. Verify complete model outputs
  across vendors as well as the individual blocks.
- **Inference algorithms and performance.** Extend prediction and decoding
  methods, and improve repeated inference through state reuse, batching and
  memory management. Keep execution and orchestration in compiled Mojo/native
  code. Measure loading, prompt processing, token generation, prediction
  latency, throughput and memory use on real workloads, distinguishing cold
  and repeated use. Each new path joins the verifier before it is documented
  as supported, preserving the cross-vendor contract in `identical` mode.
  After the basic generation workflow, evaluate speculative decoding: a draft
  proposes tokens and the target model verifies them. Preserving the target's
  output distribution alone does not establish bitwise identity; acceptance
  also requires cross-vendor verification of outputs and sampling behavior.
- **Cross-vendor and multi-GPU training.** Move the live cross-vendor
  training API beyond experimental status, and extend multi-GPU execution
  plans to more estimators with a placement check for each. Use those verified
  plans as the foundation for the broader distributed interfaces above.
- **Algorithm breadth.** New estimators enter in the `identical` mode and
  join the verifier when they land. The `fast` mode stays
  Apple only.
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
