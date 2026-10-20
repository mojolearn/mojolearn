# Higher-order gradients on the tape

Plan, 2026-10-19 (lane ag-higher-order-plan, ideas and plan only; no code).
Read with `NATIVE_AUTOGRAD.md` (the tape) and `AUTOGRAD_API.md` (the
public surface). Code references are to main at 486ac1cde (0.8.54).

Goal: `backward(create_graph=True)` records the backward pass itself as
tape nodes, so a gradient is a differentiable `Tensor` and a second (or
third) `backward` from any scalar built on it gives second derivatives:
Hessian-vector products, meta-learning through an update step,
physics-informed losses (derivatives of a network's output with respect to
its input, differentiated again). Same bits on Apple, NVIDIA and AMD by
construction; the GPU path stays parallel; no Python in the runtime; the
first-order path is untouched bit for bit.

## 0. What the tape does today (the facts the design rests on)

- `Tape._backward` (`autograd/tape.mojo:2024`) walks the recorded nodes in
  reverse creation order, marks the ancestors of the root, and for every
  active node launches that node kind's VJP kernel(s) straight into the
  parent's gradient buffer (first contribution) or into scratch followed by
  `OP_AG_ACCUMULATE` (later contributions), in the canonical order
  (decreasing consumer index, increasing edge). Gradients live in raw `FP`
  buffers (`_Node.grad`), not in nodes: they have no graph, so nothing can
  differentiate them. Scratch is rewound at the end (`_cursor`).
- The VJP bodies are fused kernels that are not tape operations:
  `OP_AG_BIN_VJP` / `OP_AGK_BIN_VJP_PART` (binary, folded for a broadcast
  operand), `OP_AG_UNARY_VJP` (`_unary_vjp_value`, `autograd/ops.mojo:611`),
  `OP_AG_SOFTMAX_VJP`, `OP_AG_REDUCE_VJP`, `OP_AG_DROPOUT_VJP`,
  `OP_LS_ELEM_VJP`, `OP_AGS_LOSS_VJP`, `OP_LS_CE_VJP`, `OP_P1_SCE_VJP`, the
  linear node's `OP_AGK_GEMM_EPI` epilogues (`EPI_L2GRAD`, `EPI_ACT_BWD`),
  `OP_ACT_BWD`, `OP_COLSUM(_DIV)`, the embedding scatter-add, and every
  `BackwardRules.apply` (`custom`, `custom_into`, `custom_into_shared`).
- Every forward op already is a parallel launch with a fixed fold order
  (`blocked_fold`, `autograd/fold.mojo`), and `_record` never launches a
  value kernel. Any value computed by tape ops is therefore identical
  across vendors by the existing argument (NATIVE_AUTOGRAD.md "Fold rule",
  "GPU status").
- Node kinds in use (`autograd/ops.mojo:161`): ADD 1, SUBTRACT 2,
  MULTIPLY 3, DIVIDE 4, RELU 10 .. SCALE 16, TRANSPOSE 20, RESHAPE 21,
  REDUCE 22, SOFTMAX 23, LOG_SOFTMAX 24, LOSS_ELEM 25, LOSS_CE 26,
  LOSS_SCE 27, LOSS_REDUCED 28, MATMUL 30, LINEAR 31, DROPOUT 35
  (`data_ops.mojo`), EMBEDDING 36 (`embedding_ops.mojo`), CUSTOM 40 + tag.
  Free below CUSTOM: 5-9, 17-19, 29, 32-34, 37-39.
- Opcode ranges in use (`sequence/dispatch.mojo` asserts them disjoint):
  1000-1010, 1040-1042, 1100-1101, 1110-1112, 1200-1211, 1300, 1400-1401,
  1600-1603, 1700, 2000-2001, 2100, 2200, 2300-2301, 2400, 2700-2708,
  2900-2914, 3500-3502. Free from 3600.
- `_backward`'s final `else` launches `OP_AG_UNARY_VJP` with `i0 = n.op`
  for any kind it does not name, and `_unary_vjp_value` returns `g * scale`
  for an unknown kind: an unknown node kind today gives a silently wrong
  gradient. Lane H1 turns that into a refusal.

## 1. Mechanism: the recorded backward

Design choice: a dual path. The first-order `_backward` keeps its fused
kernels, its in-place first-contribution schedule and its A/B-tuned
launches, untouched. `create_graph=True` runs a second traversal,
`backward_recorded`, that expresses every VJP through the tape's own
recorded operations (`matmul`, `multiply`, `transpose`, `sum`, ...). The
alternative, rewriting the first-order rules as recorded ops for everyone,
was rejected: it would multiply launches on the training path (a sigmoid
VJP becomes three launches instead of one, a linear layer's dX loses its
activation epilogue) and would change bits the trainers were A/B'd on.
With the dual path, `create_graph=False` is bit for bit today's code and
needs no define.

### 1.1 `backward_recorded` (new module `autograd/higher.mojo`, lane H1)

Free functions over the tape (no import cycle: `higher.mojo` imports
`tape.mojo`; `tape.mojo` never imports `higher.mojo`):

```
def backward_recorded[E: Exec](mut tape: Tape, mut ex: E, output: Tensor, seed: Tensor) raises
def backward_scalar_recorded[E: Exec](mut tape: Tape, mut ex: E, output: Tensor) raises
```

Traversal, host side, over graph structure only (`# driver-loop: over
graph nodes, reverse creation order`, as `_backward`):

1. Refusals before any launch: inferred root (`_INFERRED_REFUSAL`); stale
   root; root without differentiable ancestors; any active node with
   `op >= CUSTOM` (section 2, one error naming the family); `accumulate`
   is not a parameter of this entry (section 1.4).
2. Mark active ancestors and count consumers exactly as `_backward` does
   (reuse by extracting the two loops into a shared helper
   `Tape._mark_active(root) -> (active, consumers)`; one small edit of
   `tape.mojo`, lane H1).
3. `grad_node: List[Int]` (length `len(_nodes)`, -1): the index of the
   tensor that holds each active node's gradient so far. The root's entry
   is the seed tensor's index (`backward_scalar_recorded` records the seed
   as `tape.scalar(ex, 1.0)`, a constant leaf).
4. For each active non-leaf node `i`, in reverse order, and each edge
   `e` whose parent `p` requires a gradient: `c = lower_vjp(tape, ex, i,
   upstream = Tensor(grad_node[i]), e)` returns a recorded tensor at `p`'s
   shape. If `grad_node[p] < 0`, `grad_node[p] = c`; else `grad_node[p] =
   tape.add(ex, Tensor(grad_node[p]), c)`. The visiting order is the
   first-order path's canonical order, so the recorded sum `((c1 + c2) +
   c3)` is the arithmetic `OP_AG_ACCUMULATE` performs (`add(ld, ld)`, ftz
   on load and store): the same bits for the same contributions.
5. After the traversal: for every active node `tape._grad_node[i] =
   grad_node[i]` (a new `Tape` field, cleared by `reset()` and by the
   first-order `_backward`, which sets every entry to -1 at its start). An
   unreached differentiable leaf gets a recorded zero constant (`_fill`
   into `_alloc`, `variable(..., requires_grad=False)`). A leaf whose
   gradient is caller storage (`variable_into`, `caller_grad`) also gets
   `_copy(own_grad <- value of grad tensor)` so the trainer contract (the
   gradient lands in the caller's buffer) holds.
6. `Tape.gradient(x)` / `gradients(ex, x)` consult `_grad_node[x]` first
   and return that tensor's value buffer, so every existing reader keeps
   working after a recorded backward. New `Tape.grad_of(x) -> Tensor`
   returns the handle (refuses when `_grad_node[x] < 0`: "no recorded
   gradient: call backward with create_graph=True first", or the
   unreached-intermediate refusal `gradient()` gives today).
   `preactivation_gradient()` refuses after a recorded backward
   (`grad_preact` is never set by it).

Nothing in the recorded path writes `_Node.grad`, borrows buffers
(`AUTOGRAD_GRAD_ALIAS`), pairs GEMMs (`AUTOGRAD_PAIR`), fuses db
(`AUTOGRAD_PASS3_DB`) or rewinds scratch: every gradient is an ordinary
recorded node whose value lives in tape storage until `reset()`. So a
later first-order `backward` over the extended graph (the second
derivative) runs unchanged code: it allocates nothing new, writes
`own_grad` buffers (the alias reset at its start), and its result is read
with `gradient()` as always.

### 1.2 Lowering table: every VJP as recorded ops

`g` is the upstream gradient tensor (the node's shape), `x`, `y` the
operands' tensors (`Tensor(parent index)`: the tape already holds them),
`z` the node's own output tensor (`Tensor(i)`), `s(c)` a constant scalar
leaf `tape.scalar(ex, c)`. "fold" = `tape.sum(ex, ·, axis)` over the
broadcast axis of that operand (AXIS_ROWS for a `(1, cols)` operand,
AXIS_COLS for `(rows, 1)`, AXIS_ALL for a scalar), the same
`blocked_fold` the first-order `_binary_vjp` launches. Lane column: H1
core, H2 fused.

| Kind | Edge | Recorded VJP | New primitive | Lane |
|---|---|---|---|---|
| ADD | 0, 1 | `g`, folded if the operand broadcasts | none | H1 |
| SUBTRACT | 0 | `g` (folded) | none | H1 |
| SUBTRACT | 1 | `scale(g, -1)` (folded; fold of `-g`, as the kernel: sign of a zero sum) | none | H1 |
| MULTIPLY | 0 / 1 | `multiply(g, y)` / `multiply(g, x)` at the output's shape, folded | none | H1 |
| DIVIDE | 0 | `divide(g, y)`, folded | none | H1 |
| DIVIDE | 1 | `scale(divide(multiply(g, z), y), -1)`, folded (`-g z / y`, the kernel's `_bin_vjp_at` algebra) | none | H1 |
| RELU | 0 | `multiply(g, indicator(x))` | `indicator` constant | H1 |
| SIGMOID | 0 | `multiply(multiply(g, z), subtract(s(1), z))` (the kernel's order) | none | H1 |
| TANH | 0 | `multiply(g, subtract(s(1), multiply(z, z)))` | none | H1 |
| EXP | 0 | `multiply(g, z)` | none | H1 |
| LOG | 0 | `divide(g, x)` | none | H1 |
| SQRT | 0 | `divide(g, scale(z, 2))` | none | H1 |
| SCALE | 0 | `scale(g, f0)` | none | H1 |
| TRANSPOSE | 0 | `transpose(g)` (a view under `AUTOGRAD_TRANSPOSE_VIEW`, materialized where read, as today) | none | H1 |
| RESHAPE | 0 | `reshape(g, p.rows, p.cols)` | none | H1 |
| REDUCE (sum/mean, saved `i0` axis, `f0` factor) | 0 | `expand(g, p.rows, p.cols, axis, f0)` | `expand` node (kind EXPAND) | H1 |
| SOFTMAX | 0 | `multiply(z, subtract(g, sum(multiply(g, z), AXIS_COLS)))` | none | H1 |
| LOG_SOFTMAX | 0 | `subtract(g, multiply(exp(z), sum(g, AXIS_COLS)))` | none | H1 |
| MATMUL | 0 | `matmul(g, transpose(y))` (the view: same `_gemm` strides as the first-order `dX = G @ Y^T`) | none | H1 |
| MATMUL | 1 | `matmul(transpose(x), g)` | none | H1 |
| EXPAND (new) | 0 | `sum(g, axis)` then `scale(·, f0)` if `f0 != 1` (the REDUCE forward) | none | H1 |
| CLAMP (new, `f0` = bound) | 0 | `multiply(g, inside(x, f0))` (`inside` = indicator of `abs(x) < f0`, constant) | `inside` constant | H1 |
| LINEAR (saved `i0` act, `i1` bias, `f0` l2, `f1` divisor) | 0, 1, 2 | `dZ` from `z` (identity: `g`; RELU: `multiply(g, indicator(z))`, since `z > 0` iff the pre-activation is; SIGMOID/TANH: the unary rows above on `z`; SOFTMAX: the SOFTMAX row on `z`); dX = `matmul(dZ, transpose(W))`; dW = `matmul(transpose(x), dZ)`, `+ scale(W, l2)` when `l2 > 0`, `divide(·, s(divisor))` when `divisor > 0` (a division, as `EPI_L2GRAD`); db = `sum(dZ, AXIS_ROWS)`, `divide(·, s(divisor))` when `divisor > 0` | none | H2 |
| LOSS_ELEM (`i0` kind, `f0` delta) | 0 (p), 1 (t) | MSE: `multiply(scale(subtract(p, t), 2), g)`, and its negation for t. MAE: `multiply(sign(subtract(p, t)), g)`, negated for t. HUBER: `multiply(clamp(subtract(p, t), delta), g)`, negated for t (the derivative of the kernel's `0.5 q^2 + delta (|r| - q)` with `q = min(|r|, delta)` is `clip(r, -delta, delta)`). BCE_LOGITS: p: `multiply(subtract(sigmoid(p), t), g)`; t: `scale(multiply(p, g), -1)` | `sign` constant, `clamp` node | H2 |
| LOSS_REDUCED (`f1` factor) | 0, 1 | the LOSS_ELEM rows with `g` replaced by `expand(g, rows, cols, AXIS_ALL, f1)` | none | H2 |
| LOSS_CE (rows, codes) | 0 | `multiply(subtract(softmax(z_logits), onehot(codes, cols)), g)` (`g` is `(rows, 1)`, column broadcast) | `onehot` constant | H2 |
| LOSS_SCE (`f0` factor) | 0 | `multiply(subtract(softmax(logits), onehot(codes, cols)), scale(g, f0))` (`g` scalar) | `onehot` | H2 |
| DROPOUT (saved seed, counter, p) | 0 | `dropout(g, p, seed, counter)`: the VJP of dropout is dropout's own forward (`mask * s * g`) with the same `(seed, counter)`; re-derived mask, no stored mask | none | H2 |
| EMBEDDING | table | refuse in v1 (section 2); the lowering is a `scatter_add` node whose VJP is the embedding gather, listed as follow-up | - | - |
| CUSTOM + tag | any | refuse (section 2) | - | - |

Constants (`indicator`, `inside`, `sign`, `onehot`) are not node kinds:
each is a kernel launch into `_alloc` storage recorded through
`variable(ex, buf, rows, cols, requires_grad=False)`, exactly as `detach`
records a constant. They carry no graph, so the second derivative through
them is zero, which is the mathematics (relu'', |x|'', huber's clip
boundary, one-hot targets). `expand` and `clamp` are true node kinds
(their VJPs carry gradient) and need branches in the first-order
`_backward` so that the SECOND backward, which runs the first-order path
over them, knows them:

- `EXPAND = 32`: forward launches the existing `OP_AG_REDUCE_VJP` body
  (one thread per output element, the broadcast read; `autograd/ops.mojo`
  `op_ag_reduce_vjp`) with `f0` the factor; VJP in `_backward`: the
  blocked fold `_fold(dst, g, rows, cols, axis, f0)`.
- `CLAMP = 33`: forward `OP_HO_CLAMP_FWD`; VJP `OP_HO_CLAMP_VJP` (`g` where
  `|x| < f0`, else `+0.0`).

Both constants live in `autograd/higher_ops.mojo` (not `ops.mojo`), and
`_backward` gains `elif n.op == EXPAND` / `elif n.op == CLAMP` branches
plus a final `else: raise Error("autograd: no backward rule for operation
kind " + String(n.op))` in place of the silent unary fallthrough.

### 1.3 New kernels (all one thread per element, no host loops)

`autograd/higher_ops.mojo`, range `OP_HO_FIRST..OP_HO_LAST = 3600..3609`
(lane H1), registered in `sequence/dispatch.mojo` like `cast_ops`
(import line, the disjointness `comptime assert` block, the
`elif OP >= OP_HO_FIRST and OP <= OP_HO_LAST: apply_higher[OP](t, a)`
route):

| Code | Name | Slots | Body (per element t, `ld`/`st` with ftz) |
|---|---|---|---|
| 3600 | `OP_HO_INDICATOR` | p0 x, p1 dst | `1.0 if x > 0 else 0.0` |
| 3601 | `OP_HO_SIGN` | p0 x, p1 dst | `1.0 if x > 0 else (-1.0 if x < 0 else 0.0)` (`sign(0) = 0`, the MAE kernel's convention: verify against `OP_LS_ELEM_VJP`) |
| 3602 | `OP_HO_CLAMP_FWD` | p0 x, p1 dst, f0 bound | `min(max(x, -f0), f0)` |
| 3603 | `OP_HO_CLAMP_VJP` | p0 x, p2 upstream, p3 dst, f0 bound | `g if abs(x) < f0 else +0.0` |
| 3604 | `OP_HO_INSIDE` | p0 x, p1 dst, f0 bound | `1.0 if abs(x) < f0 else 0.0` |

`autograd/higher_loss_ops.mojo`, range `OP_HL_FIRST..OP_HL_LAST =
3610..3619` (lane H2):

| Code | Name | Slots | Body |
|---|---|---|---|
| 3610 | `OP_HL_ONEHOT` | p0 codes `(rows, 1)` as float values, p1 dst `(rows, cols)`, i1 cols | thread t = element: `1.0 if Int(codes[t // cols]) == t % cols else 0.0` |

Everything else is an existing kernel reached through an existing tape
op. Metal rules apply to every new body: typed `FP` slots only, a branch
and a load instead of a pointer select, `@always_inline`, no slot above
p3 unless the body needs it (the `MTL_DEBUG_LAYER` noise at p3+ is known,
NATIVE_AUTOGRAD.md "GPU status").

### 1.4 Semantics worth stating up front

- `accumulate=True` with `create_graph=True` is refused in v1 ("add the
  previous gradient tensor yourself"): a recorded gradient is a tensor,
  and accumulation across calls is `tape.add` by the caller.
- `backward_from_preactivation` has no create_graph form (the seed is
  dL/dZ of a fused linear node; the MLP trainer is its only user and never
  differentiates twice): refused with a clear error.
- Repeated backward: the second `backward` (first-order path, or
  `backward_recorded` again for third order) runs from any scalar built
  from `grad_of` tensors over the extended graph. It overwrites
  `gradient(x)` of every node it reaches (it is a new backward, as
  `backward_from` is today); the first-order gradient tensors from the
  recorded pass keep their values (they are node values, never gradient
  buffers). `grad_of(x)` keeps pointing at the recorded pass's tensor
  until the next recorded backward or `reset()`.
- A gradient tensor requires a gradient iff its lowering touched a tensor
  that does (automatic in `_record`): d(x @ W)/dW = `x^T @ seed` is a
  constant when `x` is data, so `backward` from a function of it refuses
  "no differentiable ancestors", which is correct (a zero Hessian). The
  `hvp` helper (section 5) checks `requires_grad` and returns zeros
  instead of raising.
- Memory: a recorded backward keeps roughly one tensor per contribution
  plus the lowering's intermediates until `reset()`; nothing is rewound.
  Launches: a recorded VJP is 1 to 5 launches where the fused kernel is 1.
  Both are the price of opting in; the first-order path pays nothing.
- First-order values under create_graph vs the fused path: equal wherever
  the lowering performs the same float32 operations in the same order
  (matmul VJPs: same `_gemm` strides; sigmoid/tanh/exp/log/sqrt/scale:
  the same `mul`/`sub`/`div` sequence; folds: the same `blocked_fold`;
  accumulation: the same `add` chain). They may differ where the fused
  kernel keeps an intermediate in a register that the recorded path
  stores (ftz flushes a subnormal intermediate), where the softmax VJP's
  row dot product (a serial chain in `OP_AG_SOFTMAX_VJP`) exceeds one
  fold block (64 columns; the recorded `sum(axis=1)` is the blocked fold),
  and where LINEAR's epilogue fused `+ l2 W` into the GEMM accumulator.
  This is allowed: it is one version, every vendor together; the checks
  report the match count as information, not a gate (section 6).

## 2. Which rules refuse create_graph, and why

Refused in v1, each with one error naming the family and the remedy
(`autograd: create_graph cannot differentiate through <family> (tag <n>):
it is a fused kernel with a first-order rule only; record the block with
tape operations, or take the first-order gradient`):

| Node | Family and tags | Why first-order-only |
|---|---|---|
| CUSTOM + 1 | LayerNorm (`layers.mojo` `NeuralRules`, `layer_norm_fwd_core`) | the backward is the sequence LayerNorm kernel; a composed form (`mean`, `subtract`, `multiply`, `sqrt`, `divide`) is the follow-up lane H5 |
| CUSTOM + 16..18 | recurrent (`rules_recurrent.mojo`: RNN_LINEAR, RNN_LAYER, RNN_LAST_STEP) | `op_cell_bwd_h`, the scans `OP_CELL_FWD/BWD_SCAN`, `wgrad_blocked` |
| CUSTOM + 32..38 | transformer (`rules_transformer.mojo`: RMSNorm, linear routes, RoPE, attention, SwiGLU, residual) | `bwd_rms_norm_routed`, `bwd_rope_kernel`, `bwd_attention_dispatch`, the S4-6 kernels |
| CUSTOM + 48..63 | Mamba-1/2 (`rules_mamba.mojo`) | selective-scan / SSD backward kernels, conv taps, gate/norm `_into` kernels |
| CUSTOM + 64..67 | language model (`rules_lm.mojo`: embedding scatter, head GEMM pair, CE backward, chunked head+CE) | `identical_*_backward_into` kernels |
| CUSTOM + 72..87 | Mamba-3 (`rules_mamba3.mojo` M3_BLOCK) | hand-chained block |
| CUSTOM + 88..95 | Samba (`rules_samba.mojo`) | composes the above |
| CUSTOM + 96..98 | CNN (`rules_cnn.mojo`) | conv block / linear / softmax-xent kernels |
| CUSTOM + 120, 121 | small MLP (`rules_small_mlp.mojo`) | fused linear and bias-act kernels |
| EMBEDDING (36) | `Tape.embedding` | the VJP is the run-sorted scatter-add (`embedding_backward_into`); the lowering needs a `scatter_add` node kind whose VJP is the gather (follow-up) |

The brief listed LINEAR and LOSS_SCE among the fused kernels to refuse.
Recommendation: lower them (section 1.2, lane H2) rather than refuse.
`Session.linear` and the losses are how every user model is written
(`Sequential` records `linear`; the session's `cross_entropy` is
`softmax_cross_entropy`); refusing them would leave create_graph usable
only on hand-composed `matmul`/`add` graphs. Their lowerings need no new
backward kernel (one constant, `onehot`), and the Hessian of softmax
cross-entropy (`diag(p) - p p^T` per row, the Gauss-Newton block) is the
most-asked second derivative in practice. Lane H1 ships with the H2 kinds
refused (its stub, section 7), so the refusal state exists and is tested
before H2 lands.

The refusal is decided before any launch: `backward_recorded` scans the
active set first (`_mark_active`), as `_backward` checks
`rules.supports` today.

## 3. IDENTICAL: same second-order words on every vendor, by construction

- Every recorded VJP is a composition of tape operations whose kernels are
  already in the cross-vendor digest suites (`autograd/checks/{cpu,gpu}.mojo`
  97 cases equal on CPU, Metal, NVIDIA and AMD; `fold`, `loss`, `data`,
  `fused` suites likewise). The new kernels are elementwise with no
  reduction (`indicator`, `sign`, `clamp`, `inside`, `onehot`): one
  thread, one word, ftz on load and store, `identical_*` arithmetic where
  any arithmetic exists (`min`/`max`/compare only).
- Every sum in a lowering is `blocked_fold` (through `Tape.sum`): the one
  fold order, a function of the extent alone (`ag_fold_block`, fold v2).
- Accumulation of several contributions is `tape.add` in the canonical
  order, a function of the recorded graph alone (NATIVE_AUTOGRAD.md
  "Accumulation order").
- The second backward is the existing first-order `_backward` over nodes
  of existing kinds plus EXPAND and CLAMP, whose VJPs are `blocked_fold`
  and an elementwise select.
- No rule keys on a size or a dataset; no vendor branch anywhere in the
  lowering (`higher.mojo` contains no `is_defined` vendor check).
- Therefore: for one recorded graph, the second-order words are the same
  on CPU, Metal, NVIDIA and AMD. The checks (section 6) make that a
  measured fact: `HIGHER_DIGEST` lines equal across the four columns.
- Bits may differ from the fused first-order path (section 1.4); that is a
  path difference inside one version, not a vendor difference, and it is
  reported, never gated.

## 4. GPU rules

- GPU only and parallel: every recorded VJP launches the same parallel
  kernels the forward does; the traversal is a host driver loop over
  graph nodes and node inputs (tagged `# driver-loop: over graph nodes` /
  `over node inputs`, as `_backward`), never over rows or elements. No
  download in the path. No one-thread, one-block or per-sample kernel: the
  constants are one thread per element; `onehot` is one thread per
  `(row, col)` element (never one thread per row looping over classes).
- No new fold: every reduction is `Tape.sum`. No serial chain anywhere.
- Scratch: recorded tensors come from `_alloc` (tape storage, reused after
  `reset()`); constants too. No `Exec.free` in the path.
- Metal: typed slots, branch-and-load, `@always_inline` bodies; new slots
  at p0..p3 only; `Args()` placeholders for unused slots.
- The `hvp`/`jvp`/`vjp` helpers (section 5) run in Mojo on the session's
  executor: the dot products are `tape.multiply` + `tape.sum`, the sum over
  parameters is `tape.add` of `(1, 1)` tensors in parameter order (a
  driver loop over the parameter list: model structure, tagged).
- No Hessian materialization helper in v1: `hessian(f, x)` would be a host
  loop over the elements of `x` (one HVP per basis vector), which the
  host-loop rule forbids. The GPU way is a batched seed (`backward` with a
  `(k, ...)` block of seeds, k rows of `v` at once, one launch per lowered
  op over the batch). It is listed as a follow-up, not built here.

## 5. Python surface (user voice; `python/mojolearn/ops.py`, lane H3)

`mojolearn.ops.Session` is the public home (`Sequential` trains
first-order only; it gets nothing here). Handles everywhere: the
helpers return handles, as every op does, and `numpy(h)` reads back; the
existing `grad(h)` keeps returning an array.

```python
with ops.Session(record=True) as s:
    x = s.tensor(X)                                   # data
    w = s.tensor(W0, requires_grad=True)              # parameters
    loss = s.mse(s.linear(x, w, activation="tanh"), s.tensor(Y))

    s.backward(loss, create_graph=True)              # gradients are tensors now
    g = s.grad_tensor(w)                              # a handle, differentiable
    v = s.tensor(V)                                   # the direction
    s.backward(s.sum(s.mul(g, v)))                   # second backward
    hv = s.grad(w)                                    # H v, as an array

    hv2 = s.numpy(s.hvp(loss, [w], [v])[0])          # the same in one call
```

Entries and docstrings (user voice: what it does, parameters, returns, a
short example; nothing about lanes, kernels or status):

- `Session.backward(loss, accumulate=False, create_graph=False)`.
  `create_graph=True` records the gradient computation so the gradients
  can be differentiated again; `accumulate=True` with it raises
  `ValueError` in the argument check (and Mojo refuses too).
- `Session.grad(h)`: unchanged (array of the last backward's gradient).
- `Session.grad_tensor(h)`: the gradient of `h` from the last
  `backward(..., create_graph=True)` as a tensor handle you can keep
  computing with; raises if the last backward did not record.
- `Session.vjp(output, inputs, v, create_graph=False)`: the
  vector-Jacobian product of `output` with respect to each of `inputs`
  for the seed `v` (a tensor of `output`'s shape); returns one handle per
  input. `create_graph=True` makes the results differentiable.
- `Session.jvp(output, inputs, u)`: the Jacobian-vector product of
  `output` along the directions `u` (one tensor per input, each input's
  shape); returns a handle of `output`'s shape. Implemented in Mojo by the
  double-VJP identity: with a dummy seed `w` (requires a gradient, value
  irrelevant since the VJP is linear in it), `vjp(output, inputs, w,
  create_graph=True)` gives `J^T w`; the gradient of `sum_i <J^T w, u_i>`
  with respect to `w` is `J u`.
- `Session.hvp(loss, params, v)`: the Hessian-vector product of a scalar
  `loss` with respect to `params` along `v` (one tensor per parameter,
  same shapes); returns one handle per parameter. Mojo: recorded backward,
  `d = sum_i sum(grad_of(p_i) * v_i)` (tape ops), first-order backward of
  `d`, the parameters' gradients. A parameter whose gradient tensor carries
  no graph gets a zero tensor (zero curvature), not an error.
- Error text for a refused block: "this operation has a first-order
  gradient only: layer_norm cannot be differentiated twice" (the Mojo
  message from section 2 passes through).

Mojo side (`autograd/ops_api.mojo` `OpsSession`, `autograd/pyapi.mojo`,
both bindings, `python/mojolearn/_surface_sequence.py` entry list):

- `backward_graph(h)`: `backward_scalar_recorded(self.tape, self.ex, t)`;
  refuses a non-recording session as `backward` does.
- `backward_seed(h, seed_h, create_graph)`: `backward_with_rules(...,
  NeuralRules())` or `backward_recorded` with the seed tensor.
- `grad_handle(h) -> Int`: registers `tape.grad_of(t)` as a session tensor
  (`_add`, not owned, shape from the tape) and returns its handle.
- `hvp(loss_h, params: List[Int], vs: List[Int]) -> List[Int]` and
  `jvp(out_h, inputs: List[Int], us: List[Int]) -> Int` as above, driver
  loops over the handle lists only.
- Binding entries: `ops_backward_graph(sid, h)`, `ops_backward_seed(sid, h,
  seed, create_graph)`, `ops_grad_handle(sid, h)`, `ops_hvp(sid, loss,
  params_list, v_list)`, `ops_jvp(sid, out, inputs_list, u_list)`;
  registered in `bindings/_mojolearn_x_sequence.mojo` and the host twin,
  listed in `_surface_sequence.py`. Python passes handles and lists of
  handles; no arithmetic, no loops over values.
- Lists cross the binding as today's `ops_op` list call does (`_handle_list`).

Docs: a "Second derivatives" section in
`docs-site/user_guide/autograd.md` (HVP, a physics-informed residual with
`d2u/dx2`, `jvp`), examples run by
`python/mojolearn/tests/test_autograd_public.py`-style doctests. Until the
H4 checks are green on Metal, NVIDIA and AMD the five new names are listed
in `docs-site/api_exclude.txt` (`Session.grad_tensor`, `Session.vjp`,
`Session.jvp`, `Session.hvp`, and `backward`'s `create_graph` noted on the
`Session.backward` line) with the reason "higher-order checks not yet
green on all three vendors"; the lane that removes them is the
orchestrator's, after the digests match.

Physics-informed example for the guide (per-row second derivative of a
network with one input): `x = s.tensor(X, requires_grad=True)` (`(n, 1)`),
`u = net(x)` `(n, 1)`; rows are independent, so
`s.backward(s.sum(u), create_graph=True)` gives `du/dx` per row in
`s.grad_tensor(x)`; `s.backward(s.sum(s.grad_tensor(x)))` gives
`d2u/dx2` per row in `s.grad(x)`; the residual `d2u/dx2 + f(x)` is built
from tape ops and trained with a third backward (the residual is itself a
function of the recorded second pass: use `create_graph=True` on the
second backward too).

## 6. Checks (lane H4), all small and fast

Convention: `autograd/checks/generate_higher_oracles.py` (Python and
NumPy as oracle generators only, float64; `-I`) writes the generated
`autograd/checks/higher.mojo` (`run_higher_checks[E](mut ex: E)`), driven
by `higher_cpu.mojo` (HostExec) and `higher_gpu.mojo` (DeviceExec).
Second derivatives come from nested float64 central differences of the
float64 forward along the direction `v` (`(grad(x + h v) - grad(x - h v))
/ 2h`, each `grad` itself central differences; `h` 1e-3 and 1e-4 agree to
1e-6 before a case is accepted), analytic where trivial. Tolerance:
`check_values`' `3e-4 + 3e-4 |expected|`. Every case prints a `PASS` line
and a `HIGHER_DIGEST <case> <digest>` line over its second-order words
(the FNV digest `check_values` already computes); the orchestrator diffs
the digest sets across CPU, Metal, NVIDIA and AMD (0 differing is the
gate). An informational `FIRST_ORDER_MATCH <n>/<m>` line counts the cases
whose recorded first-order gradient equals the fused path bit for bit.

Cases (small shapes, 2 x 3 and 3 x 2 matrices as the existing oracles):

1. Scalar chain `x*x*x` by two multiplies: `f'' = 6x`; a third backward
   (`backward_recorded` twice) gives `f''' = 6`.
2. Each unary through `sum(u(x))`: exp, log, sqrt, tanh, sigmoid, scale;
   relu gives the zero vector; `x / y` HVP over `(x, y)` jointly (mixed
   term).
3. Broadcast: `sum((x + row)^2)`, `sum((x * col)^2)`, `sum(x * s)` with a
   scalar leaf: the fold lowering and `expand`.
4. Matmul quadratic `sum((x @ W)^2)` HVP in `W` and in `x`; shared operand
   `sum(x @ x^T)`; transpose and reshape in the chain; `mean` over each
   axis.
5. Softmax: `sum(t * log_softmax(z))` HVP in `z` against the analytic
   `(p p^T - diag(p)) v` per row; `softmax` the same way.
6. (H2) Linear node: `mse(linear(x, W, b, act), y)` for act none, relu,
   sigmoid, tanh, softmax; with and without bias; `l2 > 0` and `divisor >
   0` variants (`Tape.linear`'s parameters); HVP in `(W, b)` and in `x`.
7. (H2) Losses: `loss_terms` + `mean` and `loss_reduced` for MSE, MAE
   (zero curvature), Huber (curvature `inside(r, delta)`; `r` chosen away
   from the boundary), BCE with logits; `cross_entropy_rows` + `mean` and
   `softmax_cross_entropy` with codes; `Tape.cross_entropy` dense.
8. (H2) Dropout: HVP through `dropout(x, 0.3, seed, counter)` composed
   with `sum(tanh(.)^2)`; the mask fixed by the seed, the oracle masks the
   same elements (the Philox mask is read back once from the forward value:
   `value != 0`).
9. Adjoint identity: `<v, J u> = <J^T v, u>` on a two-layer tanh network
   (`jvp` vs `vjp`), float64 tolerance.
10. Physics: `u = tanh(x @ w1 + b1) @ w2` with `x` `(n, 1)` requiring a
    gradient; `du/dx` and `d2u/dx2` per row against the closed form of the
    tanh network's derivatives.
11. Refusals (`PASS higher refusals=k`): `layer_norm` under create_graph
    (tag 1); `accumulate=True` with create_graph;
    `backward_from_preactivation` with create_graph; `grad_of` before any
    recorded backward; `grad_of` of an unreached intermediate; an
    inferred root; (before H2 lands, a LINEAR and a LOSS_SCE node refuse;
    after, those two refusal cases are replaced by case 6/7 coverage: the
    generator takes `--fused` to emit either set).
12. Lifecycle: 30 repeated `reset()` + record + recorded backward + second
    backward with no storage growth (`autograd/checks/lifecycle.mojo`'s
    pattern), and a `release()`.

Python (`python/mojolearn/tests/test_ops_higher.py`): the section 5
example through `Session`, `hvp` against a float64 NumPy nested-difference
oracle, `jvp` vs `vjp`, `grad_tensor` shapes, the refusals' error text,
and `OPS2_DIGEST` lines for cross-vendor diffing. `tools/check_autograd.py`
gains `--only higher` (stage names `<backend>-higher`, required
`PASS higher-order cases=N` and the refusal count). `tools/autograd_smoke.py`
gains `autograd-higher` (`higher_gpu.mojo`, 5 s, "does it start").
`pixi.toml`: `check-autograd-higher` task. The cross-vendor identity
record: `HIGHER_DIGEST` and `OPS2_DIGEST` sets equal on CPU, Metal, NVIDIA
and AMD, filed under `~/mojolearn-evidence/autograd-higher-<date>/`.

## 7. Lanes: four independent Opus lanes, disjoint files

Common to all: branch from main; never compile or run; commit after every
edit; the final reply lists every binding and check binary the
orchestrator must build. No trainer changes, so no A/B define anywhere:
`create_graph=False` is untouched. Seams are fixed here so the lanes need
not talk: names, signatures, constants, error strings as written above.

### H1 `ag-ho-core`: the recorded backward

Files (owner): `autograd/higher.mojo` (new: `backward_recorded`,
`backward_scalar_recorded`, `lower_vjp` for the H1 kinds, `expand`,
`clamp`, the constants `indicator`, `inside`, `sign`, the CUSTOM refusal),
`autograd/higher_ops.mojo` (new: EXPAND 32, CLAMP 33, `OP_HO_*`
3600..3604, `apply_higher`), `autograd/higher_fused.mojo` (new STUB:
`lower_fused_vjp[E](mut tape: Tape, mut ex: E, i: Int, upstream: Tensor,
edge: Int) raises -> Tensor` that raises the section 2 refusal for
LINEAR, LOSS_ELEM, LOSS_REDUCED, LOSS_CE, LOSS_SCE, DROPOUT and
EMBEDDING; H2 replaces this file whole), `autograd/tape.mojo` (edits:
`_grad_node` field + `reset()`; `_mark_active` extracted from `_backward`;
`grad_of`; `gradient`/`gradients` consult `_grad_node`; `_backward` sets
`_grad_node` to -1 at its start, gains the EXPAND and CLAMP branches and
the unknown-kind refusal), `sequence/dispatch.mojo` (the three `OP_HO`
lines). `lower_vjp` dispatches: H1 kinds inline; `op == LINEAR or op in
LOSS_* or op == DROPOUT or op == EMBEDDING` to `lower_fused_vjp`; `op >=
CUSTOM` refuses. Also the `sign` constant (used by H2 only) so H2 adds no
kernel to H1's file.

### H2 `ag-ho-fused`: linear, losses, dropout

Files: `autograd/higher_fused.mojo` (the real body, replacing the stub:
section 1.2 rows LINEAR, LOSS_ELEM, LOSS_REDUCED, LOSS_CE, LOSS_SCE,
DROPOUT; EMBEDDING keeps the refusal), `autograd/higher_loss_ops.mojo`
(new: `OP_HL_ONEHOT` 3610, `apply_higher_loss`, the `onehot` constant
helper), `sequence/dispatch.mojo` (the three `OP_HL` lines). Reads
`n.saved` as the first-order branches do (`_linear_vjp` for LINEAR's
`i0/i1/f0/f1`, `loss_terms`/`loss_reduced` for `i0/f0/f1`, `dropout_args`
for the dropout slots). Uses H1's `indicator`, `sign`, `clamp`, `expand`
by name from `autograd.higher`.

### H3 `ag-ho-session`: the public surface

Files: `autograd/ops_api.mojo` (`backward_graph`, `backward_seed`,
`grad_handle`, `hvp`, `jvp`), `autograd/pyapi.mojo` (the five `ops_*_py`
entries), `bindings/_mojolearn_x_sequence.mojo` and the host twin
(registration), `python/mojolearn/_surface_sequence.py` (entry list),
`python/mojolearn/ops.py` (`backward(create_graph=)`, `grad_tensor`,
`vjp`, `jvp`, `hvp`, docstrings), `docs-site/user_guide/autograd.md`
("Second derivatives" section), `docs-site/api_exclude.txt` (the five
lines). Calls `autograd.higher.backward_recorded` /
`backward_scalar_recorded` and `Tape.grad_of` by the names above.

### H4 `ag-ho-checks`: oracles, checks, smoke

Files: `autograd/checks/generate_higher_oracles.py`,
`autograd/checks/higher.mojo` (generated, committed),
`autograd/checks/higher_cpu.mojo`, `autograd/checks/higher_gpu.mojo`,
`python/mojolearn/tests/test_ops_higher.py`, `tools/check_autograd.py`
(`--only higher`), `tools/autograd_smoke.py` (`autograd-higher`),
`pixi.toml` (one task line). Writes against the section 1-5 names; the
generator's `--fused` switch emits case sets 6-7 (with H2) or their
refusal twins (without).

### Merge order and the one compile

1. H1, then H2 (its `higher_fused.mojo` replaces H1's stub: take H2's
   file on any conflict), then H3, then H4.
2. One compile of the merged tree on the cheap boxes: the sequence
   binding (device and host twin), `higher_cpu`, `higher_gpu`, and the
   existing `cpu`/`gpu`/`mlp_*`/`fused_*` check binaries (their digests
   must be unchanged: the first-order path is untouched; an unchanged
   `mlp_cases_digests.tsv` match is the proof).
3. Run `check-autograd-higher` on CPU and Metal, then `higher_gpu` and
   `test_ops_higher.py` on NVIDIA and AMD through `lq add <box> CMD`; diff
   the `HIGHER_DIGEST` and `OPS2_DIGEST` sets (0 differing).
4. Remove the five `api_exclude.txt` lines; the docs section publishes
   with the next publish command.

### Follow-ups (not in these lanes)

- H5 `ag-ho-layers`: composed LayerNorm (`layer_norm_composed` from
  primitives, a `Session(record=True, higher_order=True)` switch so
  `layer_norm` records it) and a `scatter_add` node kind for EMBEDDING
  (forward: the existing run-sorted scatter-add kernel; VJP: the gather).
  Depends on H1 and H3.
- Batched seeds (`k` seed rows per backward) for materialized Hessians and
  Jacobians without a host loop over elements.
- Fused derivative kernels for the recorded path (a `sigmoid_vjp` op whose
  own VJP is recorded), only if an A/B on an HVP workload shows the launch
  count matters.
- `Sequential` meta-learning (differentiating through `optim` steps) once a
  user asks; the optimizer step would be recorded as tape ops.
