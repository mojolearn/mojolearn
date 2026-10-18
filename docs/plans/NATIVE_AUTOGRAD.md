# Native autograd implementation branch

Engineering record, 2026-10-18. This is not a public support announcement.
Branch: `feature/native-autograd-20261018`, based on `95c41f70b` (0.8.52).
Worktree: `/Users/andrewhendel/CascadeProjects/mojolearn-autograd`.
Implementation commit: `4cea8ebd9`.

## Implemented scope

`autograd/tape.mojo` records a dynamically constructed graph of native FP32
matrix/scalar operations and computes first-order reverse-mode derivatives.
It supports arbitrary compositions of the implemented operations, branches,
shared parameters, explicit vector-Jacobian seeds, leaf-gradient accumulation,
detach, stale/foreign handle rejection, and reusable executor storage. Graph
construction, traversal, allocations, launches and accumulation run in Mojo.
There are no Python runtime callbacks, numerical fallbacks or training loops.

Operations: matrix multiplication; elementwise add/subtract/multiply/divide
with scalar, row and column broadcasting; ReLU, sigmoid, tanh, exp, log, sqrt
and scale; transpose and matrix reshape; sum and mean over either axis or all
elements; softmax/log-softmax; MSE and dense-target cross-entropy. This is a
first-order matrix API, not arbitrary-rank tensors, higher-order AD, graph
compilation/export, distributed training or PyTorch API compatibility.

Caller storage and storage lifetime (lane `ag-fusion`):

- `variable_into(ex, data, grad_destination, rows, cols)` records a
  differentiable leaf whose gradient buffer is caller-owned executor memory;
  a backward writes the gradient there directly (`accumulate=True` adds to
  what it holds). `variable` is unchanged.
- `unary_into(ex, x, op, output, scale)` and `binary_into(ex, x, y, op,
  output)` record an elementwise/softmax or broadcast binary operation whose
  value is written to caller storage; the values equal the plain methods'.
- Gradients are never zero-filled before use. A backward writes the first
  contribution to each gradient straight into it (the VJP or matmul GEMM
  destination is the parent's gradient) and adds later contributions from
  scratch in the original order, so the bits equal the old zero/copy/add
  schedule. An unreached differentiable leaf gets a zero gradient; an
  unreached intermediate has none (`gradient()` refuses), and so does any
  tensor before its first backward.
- `release(ex)` returns every tape-owned buffer to the executor through the
  new `Exec.free` (host: freed at once; device: dropped from lookup at once,
  storage released at the next `sync` or the end of the executor, after the
  queued kernels that use it). A slot that grows frees its old buffer. The
  tape holds no executor, so its destructor cannot free storage; buffers
  never released go when the executor ends. The MLP trainer releases its
  tape when the fit finishes.

The MLP tape path records each layer's output into the trainer's
activation buffers and lands parameter gradients in the trainer's gradient
vector, so the per-layer launch count drops from about 28 to 6 or 7 (the
derivation is at the top of `mlp_backward_autograd`). With the A/B define off
it is bit for bit the earlier tape path. The define
`MOJOLEARN_AUTOGRAD_MLP_SEED_SCALE` (default off) also folds the 1/B
normalization into the seed, which removes the bias divides and moves bits.

`BackwardRules` registers native custom VJPs. The tape handles ordering and
shared-input accumulation; each rule writes distinct per-input contributions.
`autograd/layers.mojo` adapts the existing LayerNorm forward/backward kernels.
The shared LayerNorm launch routines were extracted from `sequence/pyapi.mojo`
into `sequence/layernorm_core.mojo`; existing bindings and the resident entry
continue importing the same implementations. Autograd does not import the
Python binding module to reach those routines.

`sequence/mlp_fit.mojo` has a complete graph-based training path under
`-D MOJOLEARN_AUTOGRAD_MLP=1`. It records the forward composition, seeds the
existing combined loss/link derivative at the logits, and obtains parameter
gradients from the tape. The existing normalization, L2 penalty, optimizer,
shuffle, batching and stopping conventions remain intact. In particular, the
estimator's reported squared-loss normalization and its parameter-gradient
normalization retain their existing distinct conventions.

The new MLP path remains opt-in: full-dataset vendor performance evidence has
not been collected, so no new default is promoted. Transformer, Mamba,
recurrent and language-model trainers have not yet migrated to the tape.

## Validation

Environment: Mojo 1.0.0 (`ed45d567`), MAX 26.5.0, Mac16,12, macOS 26.5.2.
Complete logs and summaries are retained under
`/Users/andrewhendel/mojolearn-evidence/native-autograd-20261018/`.

CPU evidence: `cpu-final-02/summary.json` (all stages exit 0).
Reproducer and target-compilation evidence: `qualification/summary.json`.

- 35 independent float64 central-difference cases, 267 gradient entries;
  forward values also checked. Fixture generator and method manifest are in
  `autograd/checks/`; Python/NumPy are used only to generate test oracles.
- 10 invalid-operation refusals, shared-input accumulation and detach checks,
  disconnected-branch checks, and 30 repeated graphs without storage growth.
- 48 complete existing/autograd MLP fits: four hidden activations, two
  optimizers, three losses, full/partial minibatches, shuffle, regularization
  and learning-rate schedules. All 2,752 emitted parameters/losses match at
  the driver's printed precision, with matching executed epoch counts.
  This is a numerical regression comparison, not proof of bitwise model
  identity or a performance benchmark.
- A user-defined regression model trained by the tape reaches the example's
  loss-reduction acceptance criterion (loss 3.6120834 to 1.3218582e-10).
  Its data and updates stay native.

The full gradient/lifecycle executable also cross-compiles for NVIDIA sm_89
and AMD gfx942, with the corresponding vendor defines and the diagnostic GPU
define. Both builds exit 0. Those binaries have not been executed on either
vendor; compilation does not establish gradient correctness or identity.

Commands in an activated repository environment:

```sh
pixi run check-autograd
pixi run check-autograd-gpu
```

The GPU task intentionally runs the unqualified diagnostic path and currently
fails. It must not be reported as passing, skipped successfully, or covered
by the CPU results. CPU/GPU and Apple/NVIDIA/AMD bitwise identity for the new
engine remain unverified. No GPU performance measurements were attempted.

## GPU blocker and ask for Modular

The new combined operation compiles on Metal but the first addition returns
zeros. Existing upload/copy/fill operations work. A diagnostic retaining only
the forward half of the operation returns the expected addition. That
diagnostic is evidence, not a shipped replacement kernel.

With `MTL_DEBUG_LAYER=1`, the same compiled failing program aborts with:

> Read-only bytes are being bound at index 3 to a shader argument with write access enabled

Evidence:

- `gpu-pointer-fix-02/gpu-gradients-run.log`: ordinary execution failure.
- `gpu-metal-validation.log`: full Metal assertion and stack.
- `gpu-probe-08-codegen.log`: generated LLVM/AIR dumped through the supported
  `enqueue_function` diagnostic option. The helper takes the packed argument
  struct; the generated kernel describes argument 3 as a writable device
  buffer while the runtime supplies read-only bytes.
- `autograd/checks/gpu_arguments.mojo`: small caller that bypasses the tape,
  supplies live storage for every pointer slot, and expects `[6, 8]`. The
  committed probe returns `[6, 8]` on CPU and reproduces the numerical failure
  and Metal validation assertion on GPU (`qualification/` logs).

Reproduce from the worktree in its activated environment:

```sh
mojo build -j 2 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
  -D MOJOLEARN_AUTOGRAD_GPU_DIAGNOSTIC=1 \
  autograd/checks/gpu_arguments.mojo -o /tmp/autograd-gpu-arguments
MTL_DEBUG_LAYER=1 /tmp/autograd-gpu-arguments
```

To run the same caller on CPU, also define
`MOJOLEARN_AUTOGRAD_PROBE_CPU=1`. Without the GPU diagnostic define, the GPU
caller produces the intended explicit refusal; this was separately checked.

The evidence points to a compiler/runtime argument-classification problem;
Modular has not confirmed the diagnosis. Ask Modular to support this native
argument path correctly and verify the reproducer on their supported compiler
and runtime. No issue or message has been sent externally from this session.

Normal GPU calls to this new operation refuse explicitly. The test-only
`MOJOLEARN_AUTOGRAD_GPU_DIAGNOSTIC` define enables the original failing path
for reproduction; it does not fix or qualify it. Existing GPU operations and
the default MLP trainer retain their routes.

`AGENTS.md` says: "If Mojo or Modular does not support something yet, do not
build a workaround for it; wait for their support." Accordingly, no compiler
output is patched, no unsupported build mode is shipped, and no CPU/Python
fallback or alternate GPU implementation hides this failure.

## Remaining integration

Resolve the Modular blocker and run the independent gradient/graph checks on
Apple, NVIDIA and AMD. Extend native custom-rule adapters to the existing
Transformer/Mamba/recurrent blocks, then migrate their model-level backward
assembly. Qualify each changed trainer on its saved full-workload recipes
before promoting a default. Decide broader tensor ranks and public packaging
as explicit API work; this branch does not advertise those as complete.
