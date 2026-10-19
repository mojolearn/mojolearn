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
  schedule. A broadcast operand's blocked fold writes the parent's gradient
  directly for a first contribution; a reshape adds its upstream gradient
  straight from the node. An unreached differentiable leaf gets a zero gradient; an
  unreached intermediate has none (`gradient()` refuses), and so does any
  tensor before its first backward.
- `release(ex)` returns every tape-owned buffer to the executor through the
  new `Exec.free` (host: freed at once; device: dropped from lookup at once,
  storage released at the next `sync` or the end of the executor, after the
  queued kernels that use it). A slot that grows frees its old buffer. The
  tape holds no executor, so its destructor cannot free storage; buffers
  never released go when the executor ends. The MLP trainer releases its
  tape when the fit finishes.

Inference (lane `ag-forward`): `tape.no_grad()` puts a tape in inference
mode until `tape.record()`; the mode survives `reset()` and `release()`.
Every operation still allocates its value through the tape's reusable
storage (or writes caller storage for the `*_into` variants) and runs the
same launch helper, then `_record` keeps no parents, allocates no gradient
storage and marks the node inferred. `_record` never launches a value
kernel, so a recorded and a non-recorded forward give the same bits by
construction. `backward`, `backward_from` and `gradient()` refuse an
inferred tensor; inside a later recorded graph it is a constant.
`sequence/mlp_fit.mojo` `mlp_predict_autograd` predicts through
`mlp_forward_autograd` on an inference tape, chunked like `mlp_predict`;
`mlp_predict_py` uses it under `-D MOJOLEARN_AUTOGRAD_PREDICT` (default off,
A/B owed). `autograd/checks/infer_cpu.mojo` and `infer_gpu.mojo` check the
recorded/inference bit match on seven small models, the absence of graph
edges and gradient slots, chunked prediction against the full-batch
forward, and the refusals.

The MLP tape path records each layer's output into the trainer's
activation buffers and lands parameter gradients in the trainer's gradient
vector, so the per-layer launch count drops from about 28 to 8 or 9 (with the per-opcode bodies and the bias gradient's blocked fold) (the
derivation is at the top of `mlp_backward_autograd`). With the A/B define off
it is bit for bit the earlier tape path. The define
`MOJOLEARN_AUTOGRAD_MLP_SEED_SCALE` (fold the 1/B into the seed) was measured
as noise on NVIDIA and AMD and deleted 2026-10-18 (docs/TOMBSTONES.md).

`BackwardRules` registers native custom VJPs. The tape handles ordering and
shared-input accumulation; each rule writes distinct per-input contributions.
`autograd/layers.mojo` adapts the existing LayerNorm forward/backward kernels.
The shared LayerNorm launch routines were extracted from `sequence/pyapi.mojo`
into `sequence/layernorm_core.mojo`; existing bindings and the resident entry
continue importing the same implementations. Autograd does not import the
Python binding module to reach those routines.

`sequence/mlp_fit.mojo` trains through the tape by default (promoted
2026-10-18 on the A/B below; `-D MOJOLEARN_AUTOGRAD_MLP_OFF=1` restores the
hand-assembled backward). It records the forward composition, seeds the
existing combined loss/link derivative at the logits, and obtains parameter
gradients from the tape. The existing normalization, L2 penalty, optimizer,
shuffle, batching and stopping conventions remain intact. In particular, the
estimator's reported squared-loss normalization and its parameter-gradient
normalization retain their existing distinct conventions.

The MLP A/B (`~/mojolearn-evidence/native-autograd-20261018/AB_MLP_RESULTS_2026-10-18.md`,
HIGGS 1M x 28, one run per arm): NVIDIA L40S 0.334 -> 0.388 s, AMD MI325X
1.107 -> 0.602 s, average 0.85x; quality equal or better on 200k held-out
rows; each arm's digest the same on NVIDIA and AMD. Transformer, Mamba,

## Accumulation order

When a tensor feeds several consumers, its gradient is the sum of one
contribution per consumer edge, and the tape applies them in one canonical
order: decreasing consumer-node index (reverse record order), and within one
consumer that reads the tensor through several inputs (`x * x`), increasing
input index. The first contribution applied is written into the gradient
(with `accumulate=True` on a leaf, added to the gradient it already holds);
each later one is added with the tape's accumulate kernel
(`OP_AG_ACCUMULATE`, `add(gradient, contribution)` per element). The order
is a function of the recorded graph only, never of the executor, the vendor
or the launch geometry. So recording the same consumers in a different
order changes the bits exactly where float addition order matters (three or
more contributions whose partial sums round differently) and nowhere else.
Every node kind follows it: the plain VJPs, the linear node (its fused dX
epilogue applies only to a sole consumer), the loss nodes, reshapes and
custom rules (a rule writes distinct per-input destinations; the tape adds
them in input order). `_backward` asserts the order as it applies each
contribution and refuses if it is ever violated. The transformer graph
relies on it: q, k and v are recorded v, k, q so the reverse traversal sums
d(norm1) as (q + k) + v, the hand path's order.
`autograd/checks/order_cpu.mojo` and `order_gpu.mojo` check it against a
host oracle of the stated order (bit for bit, so the CPU, Metal, NVIDIA and
AMD digests are equal) with three to five consumers recorded in several
orders.

## Aliasing and mutation

Between record and backward, storage the tape has seen (every recorded
tensor's value and gradient) is read-only except through tape operations.
The caller-storage entries check this at record time (a scan over the
recorded nodes, host side, over graph structure only):

- `variable_into`: the gradient destination may not overlap any recorded
  value or gradient, nor the leaf's own data; the data may not overlap a
  recorded gradient (two leaves may read the same data).
- `linear_into`, `unary_into`, `binary_into`: the output may not overlap a
  recorded value or gradient; refused before anything is launched.
- `custom_into` and `binary_recorded`: the output and the gradient
  destination may not overlap a recorded value or gradient, nor each other.
- `backward_with_rules`, `backward_from`, `backward_from_preactivation`:
  the seed may not overlap a recorded gradient, except the seed-in-place
  case, where it is exactly the root's gradient range (custom_into's
  `grad_destination` passed as the seed); an in-place seed cannot also
  accumulate into a leaf root.

Inference mode (`no_grad`) keeps no gradients and skips these checks. The
plain operations, `variable` and `custom` take no caller output range (a
`custom` output may be a view of an input's value, as the recurrent last
step is). Mutation that the tape cannot see is the caller's to declare:
`tape.mark_mutated(x)` marks every tensor recorded before it that depends on
x (through x's consumers) stale, and a backward from a stale output
refuses; tensors recorded afterwards read x's new value and stay valid.
`tape.watch(ex, x)` keeps a host copy of x's value and
`tape.assert_unchanged(ex, x)` compares the current value with it bit for
bit (a download and a memcmp; a debug entry, never on a training path),
marking x mutated and raising when they differ. `reset()` clears the
marks and the copies.

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

NVIDIA L40S (pod nvlive, sm_89, `MOJOLEARN_COLUMN_NVIDIA=1`) and AMD MI325X
(DigitalOcean droplet, gfx942, `MOJOLEARN_COLUMN_AMD=1`), same commit: 97 of
97 cases pass on each, and every case digest equals the CPU and Metal digest
(`orch-nv-01/nv-digests.txt`, `orch-amd-01/amd-digests.txt`, 0 differing
against `orch-metal-folds-02/cpu-digests.txt`). The autograd ops are identical
on Apple, NVIDIA and AMD. The DeviceExec refusal and the diagnostic define are
gone (merged tree e6e72cebc and after).

## Remaining integration

Re-test the split operation codes and run the independent gradient/graph
checks on Apple, NVIDIA and AMD. Extend native custom-rule adapters to the existing
Transformer/Mamba/recurrent blocks, then migrate their model-level backward
assembly. Qualify each changed trainer on its saved full-workload recipes
before promoting a default. Decide broader tensor ranks and public packaging
as explicit API work; this branch does not advertise those as complete.
