# Roadmap

mojolearn's direction is fixed by its default contract.
In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.
CPU coverage is documented per API, and claims are checked by the shipped verifier. The
work below extends that contract to more algorithms, more hardware and more
workloads, and makes each backend run closer to its hardware's best.

This file states direction, not commitments. It carries no dates. What has
shipped is in the [changelog](CHANGELOG.md), and what is verified today is in
the [support matrix](SUPPORT_MATRIX.md) and [identity paths](IDENTITY_PATHS.md).

## Implemented foundations

The following work is already implemented. Remaining work builds on these
capabilities; qualification applies to the particular configurations and
source or installed-package versions recorded in their evidence.

- **Custom models and automatic differentiation.** `Sequential` builds and
  trains models through a native training loop. `ops.Session` exposes
  composable tensor operations, reverse-mode differentiation, losses,
  optimizers and schedules. The tape and backward traversal run in Mojo.
  `Sequential` includes seeded dropout and validation-loss stopping with
  restoration of the matching parameter and optimizer state.
  Higher derivatives include vector-Jacobian, Jacobian-vector and
  Hessian-vector products, including LayerNorm. Tensor operations support up to eight dimensions,
  including broadcasting, axis reductions, batched matrix products,
  convolution and pooling. See [Build your own model](docs-site/user_guide/autograd.md)
  and the 0.8.54 and 0.8.55 [release entries](CHANGELOG.md).
- **Captured programs, export and resume.** Sessions capture supported
  inference operations into a program and replay them in one native call.
  Programs can be saved and loaded; `Sequential` supports inference export
  and training checkpoints, and sessions can save and restore state.
  `Program.fit` trains a captured graph through native minibatch and epoch
  loops. Program execution keeps fixed shapes by default and can accept
  changing dimensions at a fixed rank with `dynamic_shapes=True`, where
  the captured operations and parameters permit them. Broader graph
  optimization remains below.
- **Pretrained model import.** `import_model` accepts supported
  `torch.export` programs and ONNX models. Weight import accepts PyTorch
  state dictionaries and safetensors. The
  [retained custom-model and portability evidence](bench/results/model_portability/2026-10-20/README.md)
  records six matching custom-training/export digests and six pretrained
  models with matching outputs on Apple, NVIDIA and AMD, plus an
  NVIDIA-to-Apple saved-program reload. That record defines the tested inputs,
  source provenance and limits; installed-wheel qualification and comparative
  performance are separate.
- **Checkpoint inference and decoding.** The checkpoint loader, tokenizer,
  model prefill, carried decode state and greedy generation already exist.
  The current generation interface accepts token IDs and produces a fixed
  number of new tokens. The remaining text-generation workflow is described
  below.
- **Distributed execution for selected models.**
  [Ordered multi-GPU paths](docs/multi_gpu/README.md) already cover selected
  neural and classical workloads, and
  [live cross-vendor training](docs/CROSS_VENDOR_TRAINING.md) has
  [three-vendor execution evidence](bench/results/live_xvendor/2026-09-21/README.md).
  These are foundations for general distributed execution, with qualification
  scoped to the recorded models and configurations.
- **Tooling and distribution.** Custom-model examples, operation launch
  profiling, model-catalog verification and saved-model verification exist.
  The routine verifier batches checks within workers. Linux packaging
  separates the core, vendor payloads and NVIDIA PTX payload; remaining
  architecture coverage is checked through installed packages.

## Current focus

- **Verified coverage for every algorithm.** Each public algorithm should be
  exposed by the verifier with recorded CPU and GPU references on all three
  GPU vendors. Remaining work is filling the outstanding vendor references,
  closing lanes with missing fixtures, and keeping CPU and GPU records in
  agreement. Reuse accepted evidence and complete the missing checks,
  distinguishing source execution from installed-package qualification.
- **Broader GPU coverage within NVIDIA, AMD and Apple.** More architectures
  per vendor in the released wheels, confirmed by installed-wheel checks
  rather than inferred from a GPU family name.
- **Native execution throughout the public surface.** Complete the remaining
  native boundaries in existing paths, including model orchestration and
  distributed execution. Public Python methods hand buffers to compiled
  bindings; training, decoding, graph traversal and stopping decisions belong
  in Mojo. Verify the complete call, including data preparation, worker
  scheduling, callbacks and output handling. Check that model, graph and
  buffer ownership is released correctly when operations finish or fail.
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

## Remaining work

Composable operations, automatic differentiation, captured inference programs,
model import and export are the starting point for this work. Extend their
coverage and usability while keeping claims tied to verified execution.

- **Autograd coverage and model interfaces.** Broaden the supported
  compositions, reusable module interfaces and derivative checks around the
  existing engine. Check first and higher derivatives independently from
  cross-vendor identity, including shared parameters, branching graphs,
  saved-tensor lifetimes, mutation and complete optimizer steps.
- **Graph optimization and broader program coverage.** Build on captured
  programs and export with a native optimization pass pipeline beyond
  duplicate-operation elimination. Extend structural simplification,
  dead-operation removal and memory planning, with inspectable decisions
  for each pass. Add operator fusion where it preserves the numerical
  contract. Extend operator, shape and control-flow coverage explicitly;
  verify outputs, derivatives, aliases and checkpoint replay against the
  original model. Keep performance claims separate from correctness and
  measure complete workloads before promoting an optimization default.
  Use supported Mojo/Modular interfaces; record missing upstream capabilities
  and wait for support rather than introducing unsupported compiler modes or
  Python execution fallbacks.
- **General distributed execution.** Extend the existing model-specific
  cross-vendor and multi-GPU paths to user-composed models. Generalize data
  parallelism, parameter sharding and distributed checkpoints, and add tensor
  and pipeline parallelism where supported. Qualify each execution plan on
  physical devices, with explicit gradient reduction order and complete
  optimizer-state restoration under the `identical` contract. Existing
  selected-model evidence does not qualify these new plans.
- **Developer tooling and ecosystem.** Extend the existing launch profiler
  with transfer, synchronization and memory accounting, and add gradient and
  graph debugging. Broaden shape and operator diagnostics and the existing
  custom-model examples as new paths are verified, including integrations
  with model and data tools.
- **Neural training.** Extend identical training for the transformer, Mamba
  and Samba stacks and the decoder language model to larger models and
  longer runs.
- **Text-generation usability.** Build on the existing loader, tokenizer and
  greedy decoder with a complete text-in/text-out workflow using a verified
  checkpoint pinned to a revision. Load the model and tokenizer together,
  accept prompt text and return decoded text without requiring callers to
  construct token arrays. Add end-of-sequence stopping with a maximum-token
  limit, then streaming and reproducible sampling with explicit seeds and
  settings. Keep decoding, stopping and sampling in compiled Mojo/native
  code. Prioritize this workflow before expanding the architecture list.
- **Checkpoint and hybrid inference coverage.** Broaden verified checkpoint
  and tokenizer coverage for models trained in mojolearn and other frameworks.
  Qualify whole hybrid models through prompt processing and incremental
  decoding, including attention caches and recurrent state together. Loading
  an already-trained, supported architecture should require no retraining;
  architecture conversion is a separate task. Verify complete model outputs
  across vendors as well as the individual blocks.
- **Inference algorithms and performance.** Extend prediction and decoding
  methods, building on existing decode state and captured-program reuse to
  improve batching and memory management. Keep execution and orchestration
  in compiled Mojo/native code. Measure loading, prompt processing, token
  generation, prediction latency, throughput and memory use on real workloads,
  distinguishing cold and repeated use. Correctness evidence for imported
  models does not establish a performance result. Each new path joins the
  verifier before it is documented as supported, preserving the cross-vendor
  contract in `identical` mode.
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
- **Verifier usability.** Build on worker batching, the model catalog and
  saved-model verification with clearer coverage reports and shorter checks.
  State exactly what an installed package has and has not been checked for.
- **Releases.** Continue PyPI releases of the macOS arm64 wheel and the
  Linux x86-64 core, NVIDIA, AMD and NVIDIA PTX packages. Check the installed
  package combinations on their corresponding CPU and GPU targets before
  publication.

## Out of scope

- GPU backends other than CUDA, HIP and Metal. Mojo targets those three.
- Reduced-precision arithmetic in place of FP32 for the identity guarantee.
  BF16 and INT8 remain storage formats.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the contributor quickstart in
[docs/START_HERE.md](docs/START_HERE.md).
