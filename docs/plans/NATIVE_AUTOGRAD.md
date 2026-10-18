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

The GPU task intentionally runs the unqualified diagnostic path; it failed
before the opcode split and its re-test is pending. It must not be reported as passing, skipped successfully, or covered
by the CPU results. CPU/GPU and Apple/NVIDIA/AMD bitwise identity for the new
engine remain unverified. No GPU performance measurements were attempted.

## Operation codes

Lane `ag-metal-folds` (2026-10-18) replaced the single combined body
(`OP_AUTOGRAD`, a runtime mode x operation switch) with one operation code
per body, as every other sequence operation is built. Codes are the
contiguous range `OP_AG_FIRST..OP_AG_LAST` (1000..1010) in
`autograd/ops.mojo`; `sequence/dispatch.mojo` asserts the range does not
collide with the recurrent scan codes. Every body is `@always_inline`.

| Code | Name | Threads | Slots read | Slots written |
|---|---|---|---|---|
| 1000 | `OP_AG_BIN_FWD` | output elements | p0 x, p1 y | p2 z |
| 1001 | `OP_AG_BIN_VJP` | output elements | p3 upstream; MULTIPLY p0, p1; DIVIDE p1, operand 1 also p2 z | p4 |
| 1002 | `OP_AG_UNARY_FWD` | elements | p0 x | p1 y |
| 1003 | `OP_AG_UNARY_VJP` | elements | p2 upstream; RELU/LOG p0 x; SIGMOID/TANH/EXP/SQRT p1 y | p3 |
| 1004 | `OP_AG_TRANSPOSE` | elements | p0 | p1 |
| 1005 | `OP_AG_REDUCE_PART` | blocks x outputs | p0 | p1 partials |
| 1006 | `OP_AG_REDUCE_FOLD` | outputs | p0 partials | p1 |
| 1007 | `OP_AG_REDUCE_VJP` | input elements | p0 upstream | p1 |
| 1008 | `OP_AG_SOFTMAX_FWD` | rows | p0 x | p1 y |
| 1009 | `OP_AG_SOFTMAX_VJP` | rows | p0 y, p1 upstream | p2 |
| 1010 | `OP_AG_ACCUMULATE` | elements | p0, p1 | p1 (element-wise) |

The binary and unary kinds (add/subtract/multiply/divide, relu/sigmoid/...)
stay a runtime integer inside their body. The tape binds only the slots a
body reads or writes; every other slot keeps the `Args()` placeholder, as
in the rest of the library. No launch binds one buffer to two slots: for
`x * x` the right operand is read through p0 (`i10 = 1`) and p1 is left
unbound. Softmax recomputes its exponentials instead of reading back its
output slot. A reshape's VJP is its upstream gradient itself (no launch);
transpose forward and VJP are the same body.

## Fold rule

Sums are blocked folds in a fixed order, the same on every executor
(HostExec runs the same bodies). `ag_fold_block(extent)` is the smallest
power of two R >= 64 with R * R >= extent (the `ln_fold_rows` rule): it
depends on the folded extent alone, never on a vendor or a benchmark shape.
`OP_AG_REDUCE_PART` adds each block's terms ascending from +0.0, one thread
per (block, output); `OP_AG_REDUCE_FOLD` adds each output's block partials
ascending from +0.0 and applies the mean factor, one thread per output. This
covers the total sum/mean (axis -1, used by `mean`, `mse_loss` and
`cross_entropy`), the column sum (axis 0, parallel over row blocks x
columns) and the row sum (axis 1). A broadcast operand's gradient (a
`(1, cols)`, `(rows, 1)` or scalar operand) is its elementwise VJP written
at the output's shape into tape scratch, then that blocked row/column/total
fold. An extent of at most 64 is one block. Partials come from the tape's
storage (`_alloc`), reused after `reset()`.

Bits: the total sum, the broadcast-operand gradients and folds over more
than 64 terms change bits against `4cea8ebd9` (a single chain before); the
gradient oracles are tolerance checks and are unchanged.

## GPU status: Apple passes; NVIDIA and AMD compiled, execution pending

Before the split, the combined operation compiled on Metal but returned
zeros, and under `MTL_DEBUG_LAYER=1` aborted with "Read-only bytes are being
bound at index 3 to a shader argument with write access enabled". The
orchestrator's bisect on 2026-10-18 (evidence under
`/Users/andrewhendel/mojolearn-evidence/native-autograd-20261018/orch-metal-folds-01/`
and `orch-metal-folds-02/`) separated two causes:

1. **Ours.** A pointer select in a kernel body, `a.p0 if a.i10 == 1 else a.p1`,
   compiles on Metal (Mojo 1.0.0 / MAX 26.5.0) to a pointer whose loads read
   zeros, with no error; the same body is correct on the host. The library's
   own OP_SUB through the same harness was correct, the autograd body returned
   its left operand for add, subtract and x + x alike, and replacing the select
   with a branch fixed all three (`control-run.log` vs `control-v2-run.log`).
   Rule: never select or return an FP inside a kernel body; branch and load.
2. **Not ours.** The validation assertion fires for every seq_kernel that
   touches pointer slot p3 or beyond, including the library's own OP_LN_FWD
   (`orch-metal-folds-02/ln-run.log`), and only under `MTL_DEBUG_LAYER=1`.
   Results are correct without the layer. The repository's Apple tools run the
   layer with `MTL_DEBUG_LAYER_ERROR_MODE=nslog` for this reason; do the same.

Results on the M4 (Metal), lane/ag-metal-folds at 368c22a79:

- `autograd/checks/gpu_arguments.mojo`: `[6, 8]` on Metal under the debug layer.
- `autograd/checks/gpu.mojo`: 97 of 97 cases pass; every case digest equals
  the CPU digest (`orch-metal-folds-02/{cpu,gpu}-digests.txt`, 0 differing).
- `autograd/checks/mlp_cpu.mojo`: 48 of 48 fits match the existing trainer.

NVIDIA (sm_89) and AMD (gfx942) cross-compiles are recorded below as they
land; execution on those vendors is queued through `lq`, and the autograd
identity claim (same bits on Apple, NVIDIA and AMD) waits for it.

DeviceExec still refuses the range without `MOJOLEARN_AUTOGRAD_GPU_DIAGNOSTIC`
until the NVIDIA and AMD runs pass; then the refusal is removed.

## Remaining integration

Re-test the split operation codes and run the independent gradient/graph
checks on Apple, NVIDIA and AMD. Extend native custom-rule adapters to the existing
Transformer/Mamba/recurrent blocks, then migrate their model-level backward
assembly. Qualify each changed trainer on its saved full-workload recipes
before promoting a default. Decide broader tensor ranks and public packaging
as explicit API work; this branch does not advertise those as complete.
