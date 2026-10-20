# Arbitrary-rank tensors for the tape (plan, lane ag-rankn-plan, 2026-10-20)

Ideas and plan only; no code. Read with `docs/plans/NATIVE_AUTOGRAD.md` (the
tape, its fold rule and opcode registry) and `docs/plans/AUTOGRAD_API.md`
(the Python surface). Line references are to main 486ac1cde.

Today a tape tensor is a float32 matrix: `_Node.rows`, `_Node.cols`
(autograd/tape.mojo:121), `Tape._record(ex, value, rows, cols, ...)` (:702),
`OpsSession.rows/cols` (autograd/ops_api.mojo:137), `Session.tensor` refuses
`ndim > 2` (python/mojolearn/ops.py:206), `Sequential` takes `(n_samples,
n_features)` (python/mojolearn/_x_sequence_model.py:28). Users want
`[B, C, H, W]`, `[B, T, F]`, `[B, heads, Q, K]`.

## 0. Decisions in ten lines

1. Storage stays one flat row-major float32 device buffer per tensor. The
   rank-N shape is **metadata in side tables of the tape**, parallel to
   `_nodes`; `_Node` and every existing method keep `rows x cols`, where
   `rows = product of the leading dims` and `cols = the last dim` (the
   **matrix view**). Rank 2 is the special case: `shape = (rows, cols)`.
2. Every row-wise or elementwise 2-D op therefore already works on rank N
   through its matrix view (linear, softmax, log_softmax, layer_norm,
   cross_entropy, dropout, relu/sigmoid/tanh/exp/log/sqrt/scale, same-shape
   add/sub/mul/div, trailing-dims broadcasts, last-axis and all-axes sums):
   **zero new kernels, the 2-D words**. Only the shape metadata is set.
3. Free views: reshape, flatten, squeeze, unsqueeze (alias the buffer, the
   RESHAPE node as today). Pending strided views: permute (and the layout
   lane's narrow), the transpose-view mechanism generalised (`_pending`,
   `_materialize_at`, tape.mojo:504-521): a strided reader (bmm, the rank-N
   elementwise and fold kernels) reads them through strides and never
   copies; a dense reader materialises once with one copy launch.
4. New kernels are `seq_kernel[OP]` bodies with `Args` (sequence/dispatch.mojo),
   one thread per output word or per (output word, leaf block); shapes and
   strides travel in a **descriptor buffer** (int32 bit patterns in tape
   storage, as `Tape.ids` holds ids), bound to `Args.p11`.
5. One fold order for every rank-N sum, declared once (section 2.0): the
   reduced axes become one virtual trailing axis of extent R, the source is
   the virtual matrix `[outputs, R]`, and the sum is the tape's existing
   blocked fold v2 over that matrix along `AXIS_COLS` (leaf 64, tree fan-in
   64, `autograd/fold.mojo`). Only the leaf launch is new; the level and
   finish launches are the existing ones. When the reduced axes are a
   trailing block the virtual matrix is physical, so the words equal
   `Tape.sum(reshape(x, (outputs, R)), axis=1)` bit for bit (a check).
6. Batched matmul: one launch, thread = (batch, cell), the GEMM cell chain
   (`gemm_dot`, k ascending, one fma a term: `op_gemm`'s word), operands read
   through strides (batch strides, row stride, column stride). Backward for
   a batch-broadcast operand is one chain per word over (broadcast batch
   coordinates row-major, then k), no fold kernel. Never split-K.
7. New ops are **CUSTOM nodes with tags** (`Tape.custom`, tape.mojo:1553) and
   `BackwardRules` structs, one struct per lane in its own file, so lanes
   never touch `_backward`. The core lane alone edits tape.mojo (side
   tables, `_record_nd`, views, permute, shape inheritance).
8. Rank-2 callers get no new words: every existing call path is unchanged
   except metadata writes; the rank-N kernels are new words with one
   declared order on the host and every vendor. No define is needed (no
   behaviour change to any existing trainer); the orchestrator's A/B is the
   stored digests (`autograd/checks/mlp_cases_digests.tsv`, `OPS_DIGEST`,
   `MODEL_DIGEST`) staying equal.
9. Python: `Session.tensor` takes any rank up to 8; `shape(h)` returns the
   tensor's shape (unchanged for 0/1/2-D inputs); `reshape`, `permute`,
   `flatten`, `squeeze`, `unsqueeze`, `sum/mean/max(axis=int|tuple,
   keepdims)`, batched `matmul`, `concat`, `split`, `take`, `softmax(axis)`,
   `conv2d`, `max_pool2d`, `avg_pool2d`; `Sequential` gains `reshape`,
   `permute`, `flatten`, `conv2d`, `max_pool2d`, `avg_pool2d` and an
   `input_shape`.
10. Six Opus lanes, disjoint primary files, each its own opcode range above
    3502, its own tag block and its own `autograd/checks/rankn_*_cases.mojo`
    with digests equal on CPU, Metal, NVIDIA and AMD. Merge order: core,
    then fold / bmm / layout / conv in any order, then python.

## 1. Shape model

### 1.1 Where the shape lives

`_Node` keeps `rows`, `cols` (unchanged struct, unchanged copies). The core
lane adds to `Tape`:

```
var _shapes: List[ShapeN]      # one per node, appended by _record, cleared by reset
```

with, in a new file `autograd/rankn.mojo`:

```
comptime RN_MAX_RANK = 8
@fieldwise_init
struct ShapeN(ImplicitlyCopyable, Movable):
    var ndim: Int                            # 0 (a scalar) .. RN_MAX_RANK
    var d: InlineArray[Int, RN_MAX_RANK]     # dims, row-major; unused slots 1
    def count(self) -> Int                   # product of dims (1 for ndim 0)
    def rows(self) -> Int                    # product of d[0 .. ndim-2] (1 when ndim <= 1)
    def cols(self) -> Int                    # d[ndim-1] (1 when ndim == 0)
    def stride(self, k: Int) -> Int          # row-major stride of axis k
    def same(self, o: ShapeN) -> Bool
    def matrix(rows: Int, cols: Int) -> ShapeN   # static: the rank-2 shape
def rn_broadcast(a: ShapeN, b: ShapeN) raises -> ShapeN     # NumPy rule, right-aligned, 1 stretches
def rn_normalize_axis(axis: Int, ndim: Int) raises -> Int   # negative axes
```

`_record` appends `ShapeN.matrix(rows, cols)`; `_record_nd(ex, value, shape,
op, parents, saved, ...)` calls `_record(ex, value, shape.rows(),
shape.cols(), ...)` and then overwrites `_shapes[index] = shape`. `_shape`
(tape.mojo:693) already bounds `rows * cols`; the count of a rank-N tensor
is the same product, so the Int32 launch bound (`_args_i32_out`,
sequence/exec_device.mojo:216) is unchanged.

Public accessors on the tape: `shape_of(x) -> ShapeN`, `ndim(x) -> Int`,
`dim(x, k) -> Int`; `rows(x)` / `cols(x)` stay and mean the matrix view.

Shape inheritance for the existing ops (metadata only, no new launch, no
new word; the core lane makes these one-line insertions in tape.mojo):

| site (tape.mojo) | output ShapeN |
|---|---|
| `_unary_to` :1052, `dropout` :1101 | the input's |
| `_binary_to` :988 | when both inputs have the same ShapeN: that shape; when one input's matrix view is `(1, 1)`, `(1, cols)` or `(rows, 1)` of the other's: the other's shape (the trailing-dims broadcast of 2.1); otherwise rank 2 `(rows, cols)` as today |
| `softmax`/`log_softmax` :1094, `layer_norm` (autograd/layers.mojo) | the input's |
| `linear` :1165, `matmul` :1133 with a rank-N left operand | the left operand's leading dims + `(cols of the result)` |
| `_reduce` :1338 | `AXIS_ALL`: rank 2 `(1, 1)`; `AXIS_COLS`: leading dims + `(1,)`; `AXIS_ROWS`: rank 2 `(1, cols)` |
| `transpose` :1295 | rank 2 (a rank-N `transpose` is the core's `permute` swapping the last two axes) |
| everything else | rank 2 `(rows, cols)` |

Values and gradients are unchanged by any of this: a gradient buffer has
the node's `rows * cols` words either way.

### 1.2 Descriptor buffers

A kernel needs more integers than `Args` holds (twelve). Shapes and strides
travel in a descriptor: `Tape.descriptor(ex, words: List[Int]) -> FP`
allocates `len(words)` words from `_alloc` (tape storage, released with the
graph at `reset`), uploads the int32 bit patterns once (`ex.upload`, a
memcpy on the host, one small copy on a device: the cost of
`x_cnn/device.mojo put_prm`), and the caller binds it to `Args.p11`. Bodies
read it with `rn_word(p: FP, i: Int) -> Int` (bitcast, never `ld`: ftz would
zero a small integer's pattern, see autograd/embedding_ops.mojo `_ld_i`).
Layout, fixed and documented in `autograd/rankn.mojo`:

```
RN_DESC_SHAPE  : [ndim, d0 .. d7]                      9 words
RN_DESC_STRIDE : [s0 .. s7]                             8 words (0 on a broadcast axis)
```

A launch's descriptor is a concatenation of such records at offsets the
body's docstring names. Adjacent axes with compatible strides on every
operand are collapsed on the host before the upload (`rn_collapse`): a
metadata step over at most 8 axes, and it changes no word (each output
element reads the same operand words).

### 1.3 Views

| view | node kind | launch | gradient |
|---|---|---|---|
| reshape, flatten, squeeze, unsqueeze (`view_nd(ex, x, shape)`, count preserved) | RESHAPE (21) with a rank-N output shape; aliases `n.value` as `reshape` does (tape.mojo:1324) | none | the upstream buffer itself, borrowed as today (:2174, :2242) |
| permute (`permute(ex, x, axes)`), transpose of the last two axes | PERMUTE_N (37, a tape kind below CUSTOM) **pending**: `saved.p0` = source buffer, `saved.p11` = descriptor (source shape, permuted strides, inverse axes), `saved.i9 = 1` while pending | none until a dense reader calls `_use`: then `OP_RN_COPY_STRIDED` (one thread per output word, a copy) and `saved.i9 = 0` | `OP_RN_COPY_STRIDED` of the upstream through the inverse permutation into the operand's gradient (a copy: every word written once, `zeroed_destination` False) |
| narrow / split pieces (layout lane) | CUSTOM tag (section 2.5), pending, read through strides from a base offset | same copy kernel with a source offset | copy into the slab of the operand's gradient after a zero fill (2.5) |
| detach | as today | none | stops |

`_pending(i)` (tape.mojo:504) becomes `op == TRANSPOSE or op == PERMUTE_N
(or a NARROW tag) with saved.i9 == 1`; `_materialize_at` dispatches on the
kind. `_operand(i)` (:1312) generalises to `operand_nd(i) -> (FP, base
offset, ShapeN, strides)`: a dense node is `(value, 0, shape, row-major
strides)`, a pending view is its source with the view's strides. Strided
readers (bmm, OP_RN_BIN_*, OP_RN_REDUCE_PART, the slab copies) take their
operands from `operand_nd`, so `permute` feeding a batched matmul, a
broadcast add or a reduction costs no copy at all, exactly as the transpose
view feeding `matmul` today. A dense reader (softmax rows, layer_norm, the
2-D GEMM, conv) materialises first; the words are the same either way (a
copy moves bits).

When a copy is needed: materialising a permute or narrow for a dense
reader (one launch, `rows * cols` words), and `concat` (one slab copy per
operand). Reshape of a pending view materialises it first (a reshape of
strided memory is not a view in general; the core refuses nothing, it
copies once).

### 1.4 The core interface the other lanes code against

Lanes never compile, so fold / bmm / layout / conv write against these
names and the orchestrator compiles the merged tree once:

```
# autograd/rankn.mojo
RN_MAX_RANK, ShapeN (above), rn_broadcast, rn_normalize_axis, rn_collapse,
rn_word(p: FP, i: Int) -> Int, rn_desc_shape(mut words, s: ShapeN), rn_desc_strides(mut words, s: InlineArray[Int, RN_MAX_RANK])
# autograd/tape.mojo (core lane)
Tape.shape_of(x) -> ShapeN ; Tape.ndim(x) ; Tape.dim(x, k)
Tape._record_nd(ex, value, shape, op, parents, saved, leaf_grad=False, grad_storage=dummy_ptr(), caller_grad=False) -> Tensor
Tape.custom_nd(ex, inputs, output, shape, tag, saved) -> Tensor          # custom + the shape
Tape.custom_into_nd(ex, inputs, output, grad_destination, shape, tag, saved) -> Tensor
Tape.tensor_nd(ex, host values, shape, requires_grad) / variable_nd / variable_into_nd
Tape.view_nd(ex, x, shape) -> Tensor                                       # reshape family
Tape.permute(ex, x, axes: List[Int]) -> Tensor
Tape.operand_nd(ex, x) -> (FP, Int, ShapeN, InlineArray[Int, RN_MAX_RANK])  # buffer, base offset, shape, strides (no materialisation)
Tape.descriptor(ex, words: List[Int]) -> FP
Tape.values(ex, x) / gradients(ex, x) / value_of / gradient               # unchanged: rows * cols words
```

`BackwardRules.apply` (tape.mojo:180) is unchanged: `saved.p11` carries the
descriptor, so a rule has every shape it needs; `rows` / `cols` are the
matrix views.

## 2. Operations and their backward rules under IDENTICAL

### 2.0 The two declared orders

**(F) The virtual-matrix fold.** Every rank-N sum (a reduction over an axis
set A, a broadcast operand's gradient, a mean) is defined as: order the kept
axes row-major (they index the outputs, O of them) and the reduced axes
row-major (they index the terms, R of them); the sum is the tape's blocked
fold v2 (`autograd/fold.mojo`: leaves of 64 consecutive terms added
ascending from +0.0, `OP_AGF_LEVEL` fan-in 64 in place, `OP_AG_REDUCE_FOLD`
with the factor) over the virtual matrix `[O, R]` along `AXIS_COLS`. Only
the leaf launch is new (`OP_RN_REDUCE_PART`, `OP_RN_BIN_VJP_PART`): thread
`t = (o, leaf b)` folds terms `b * 64 .. min(b * 64 + 64, R) - 1` of output
o, each term's source index decoded from (o, r) through the descriptor, and
writes the partial at `o * nb + b`, the `AXIS_COLS` layout
`_ag_reduce_part_v2` writes (autograd/ops.mojo:655-658). The level and
finish launches then run exactly as `OP_AGK_BIN_VJP_PART`'s path runs them
today (a caller that writes the leaf partials itself, autograd/fold.mojo
header), with `axis = AXIS_COLS, rows = O, cols = R`. One order on the host
and every vendor by construction (the same bodies through
sequence/dispatch.mojo). Mean: `factor = identical_div(1, R)` applied by the
finish launch as `_reduce` does (tape.mojo:1347).

When A is a trailing block of axes the virtual matrix is the physical
buffer, so the result equals `Tape.sum(view_nd(x, (O, R)), axis=1)` word for
word. When A is every leading axis the result equals `Tape.sum(matrix view,
axis=0)` only if the `AXIS_ROWS` layout's chain is the same chain: it is
(block-major partials, the same 64 ascending terms per leaf, the same tree),
so that equality is also a check, not a requirement.

**(C) Copies and scatters.** Views, permutes, concat, split, max's gradient
and conv's layout kernels move or place words; every destination word is
written by exactly one thread and no two writes race, so there is no order
to declare. They are identical by construction; the checks compare them
against the composition they stand for.

### 2.1 Elementwise with broadcasting (fold lane)

`rankn_binary(tape, ex, x, y, op)` for ADD / SUBTRACT / MULTIPLY / DIVIDE:

- Output shape `rn_broadcast(sx, sy)`.
- Route: if `sx.same(sy)`, or one operand's matrix view is `(1,1)`,
  `(1, cols)` or `(rows, 1)` of the other's (a trailing-dims broadcast:
  `[B,T,F] + [F]`, `[B,T,F] + [B,T,1]`, a scalar), call `Tape._binary_to`:
  the existing `OP_AG_BIN_FWD` / `OP_AGK_BIN_VJP_PART` kernels and words,
  shape inherited (1.1). Otherwise (a middle-axis broadcast such as
  `[B,H,T,T] + [1,1,T,T]` or `[B,1,F] + [B,T,F]`) the rank-N kernels below.
- Forward `OP_RN_BIN_FWD`: thread t = output word t; decode t over the
  output shape, operand offsets from the operands' strides (0 on a broadcast
  axis, from `operand_nd`, so a permuted operand is read in place); value
  `_bin_value(op, x, y)` (autograd/ops.mojo:394, the same add/sub/mul/div,
  `ld`/`st` ftz). The same word per element as `OP_AG_BIN_FWD`.
- Backward: an operand with the output's shape takes `OP_RN_BIN_VJP`
  (elementwise, `_bin_vjp_at`'s arithmetic, written in place: a copy-shaped
  write, order (C)); a broadcast operand takes `OP_RN_BIN_VJP_PART`, the
  leaf of order (F) over its broadcast axes with the term
  `_bin_vjp_at`-style value at each output element, then the fold's level
  and finish launches. `zeroed_destination` False on both.
- Node: CUSTOM tag `RN_BINARY` (136 + op), `saved.p11` descriptor = output
  shape, x strides, y strides; `saved.i1 = op`.

Unary ops, scale and dropout on a rank-N tensor need nothing new (1.1).

### 2.2 Axis reductions: sum, mean, max (fold lane)

`rankn_sum(tape, ex, x, axes, mean)`: output shape = x's with the reduced
axes set to 1 (keepdims; the Python squeezes with a free view unless
`keepdims=True`).

- Route: A = every axis -> `Tape.sum(x, AXIS_ALL)`; A = the last axis ->
  `Tape.sum(matrix view, AXIS_COLS)`; A = every leading axis -> `Tape.sum(matrix
  view, AXIS_ROWS)`; otherwise `OP_RN_REDUCE_PART` (order F) + the existing
  level and finish launches. A reduction over a trailing block of axes may
  also go through `Tape.sum(view_nd(x, (O, R)), axis=1)`; the lane picks the
  existing path whenever it applies, and the check asserts the rank-N path
  equals it.
- Backward `OP_RN_BCAST_VJP`: thread per source word, reads the upstream at
  the output coordinates (reduced axes collapsed), times the factor:
  `OP_AG_REDUCE_VJP`'s rule generalised, order (C). The existing routes keep
  the existing VJP.
- Node: CUSTOM tag `RN_REDUCE` (141, sum) / 142 (mean).

`rankn_max(tape, ex, x, axes)` returns the max and keeps the argmax (int32
patterns in tape storage) in `saved.p10`:

- Semantics, order-free so that no tree is a bit version: the combine of
  two (value, index) partials takes the larger value; a NaN wins over any
  number; equal values take the lower linear index. The result is then
  independent of the fold structure; the lane still uses the (F) structure
  (`OP_RN_MAX_PART` leaves of 64, `OP_RN_MAX_LEVEL` in place over (value,
  index) partial pairs, `OP_RN_MAX_FOLD`) so the parallelism is the fold's.
  This differs from the softmax row max's "first maximum, a NaN in column 0
  sticks" scan (autograd/row_ops.mojo header) on NaN inputs only; max is a
  new op, not that scan, and says so in its docstring.
- Backward `OP_RN_MAX_VJP`: thread per output word writes the upstream into
  the operand's gradient at the stored argmax; the outputs partition the
  source so no two threads write one word; `zeroed_destination` True (the
  tape zero-fills the destination, tape.mojo:166).
- Node: CUSTOM tag `RN_MAX` (143).

### 2.3 Batched matrix product (bmm lane)

`rankn_matmul(tape, ex, x, y)`: `x [..., M, K]`, `y [..., K, N]`, batch dims
broadcast (equal or 1; a rank-2 operand has none), output `broadcast(batch)
+ [M, N]`. A rank-N `x` with a rank-2 `y` is `Tape.matmul(matrix view, y)`
with the shape inherited (1.1): the existing GEMM, its route and its words
(this covers `[B,T,F] @ W[F,O]`, the common case, and `linear`).

- Forward `OP_RN_BMM`: thread t = (batch b = t // (M N), cell mn); b
  decoded over the output batch dims; operand offsets from batch strides (0
  where broadcast), row and column strides (`operand_nd`: a permuted or
  transposed operand is read through its strides, no copy, as `matmul`
  reads a transpose view, tape.mojo:1146); the word is `gemm_dot(...)` with
  k ascending from +0.0, one fma a term (sequence/ops.mojo:469): `op_gemm`'s
  chain, hence `op_gemm`'s word, hence the tiled and thin routes' word
  (sequence/gemm_tiled.mojo states that contract). The batched product never
  takes the split-K order (`gemm_split_shape`, sequence/gemm_split.mojo:116,
  is consulted by `DeviceExec._launch` for `OP_GEMM` only): a per-slice
  `Tape.matmul` of a split-K shape has different bits, so the EQ2D check
  uses shapes outside that rule.
- Backward, for an operand whose batch dims all equal the output's: dX[b] =
  G[b] @ Y[b]^T and dY[b] = X[b]^T @ G[b] are the same kernel with swapped
  strides (as tape.mojo:2268-2275). For an operand with broadcast batch
  axes (a weight shared across heads or batch), its gradient word is **one
  chain over (the broadcast batch coordinates row-major, then k) ascending**
  from +0.0: `OP_RN_BMM` with the broadcast axes appended to the contraction
  (descriptor: a list of contraction axes with their strides on each
  operand). No fold kernel, no scratch, a declared order. With every batch
  axis broadcast this is the stacked-rows GEMM `(stack_b X_b)^T (stack_b
  G_b)`; when the operand is also rank 2 the lane routes to `Tape.matmul`'s
  backward instead (the same stacked chain, the existing kernel).
- Phase 2 (its own A/B, same words): a tiled batched kernel on NVIDIA and
  AMD, `seq_gemm_tiled_kernel` with the batch on `block_idx.y`, dispatched
  by `seq_gemm_route(M, N, K, False)` per slice as `OP_AGK_GEMM_EPI` is
  dispatched; the host and Apple keep the cell body. The long broadcast
  chain of the backward may later take a split order: a bit version,
  recorded as one.
- Node: CUSTOM tag `RN_BMM` (152); `saved.p0/p1` operands, `saved.p11`
  descriptor (output shape, each operand's strides, contraction list).

### 2.4 Permute (core lane)

Section 1.3. `permute(ex, x, axes)` records a pending PERMUTE_N view; a
strided reader costs nothing; a dense reader costs one `OP_RN_COPY_STRIDED`
(thread per output word: decode, read the source at the permuted strides,
store). The VJP is the same kernel with the inverse permutation into the
operand's gradient, every word written once. `transpose(x)` on a rank-N
tensor is `permute` swapping the last two axes; on rank 2 it is the existing
`transpose` (unchanged words, unchanged view).

### 2.5 Concat and split (layout lane)

- `rankn_concat(tape, ex, xs, axis)`: output dims = the operands' with the
  axis summed. One `OP_RN_SLAB_COPY` launch per operand (`# driver-loop: over
  concat operands`, model structure), thread per operand word, writing the
  operand into its slab (destination offset along the axis). Operands are
  read through `operand_nd` strides. Backward: one slab copy per operand
  from the upstream into the operand's gradient (every word once,
  `zeroed_destination` False, order C). Node: CUSTOM tag `RN_CONCAT` (160);
  up to `Args`' pointer slots minus two operands a node (p0..p9 operands,
  p10 output, p11 descriptor); more operands chain two concat nodes (the
  Python splits the list).
- `rankn_narrow(tape, ex, x, axis, start, length)` is a pending strided view
  (1.3) with a base offset: `split(x, axis, sizes)` is a list of narrows, a
  free view each. A dense reader materialises with `OP_RN_SLAB_COPY`. VJP:
  the first contribution to the operand's gradient zero-fills it
  (`zeroed_destination` True) and copies the upstream into the slab; a later
  contribution goes through the tape's scratch + `OP_AG_ACCUMULATE` path as
  any second consumer does (the scratch is zero outside the slab; adding
  +0.0 to a word is that word under ftz, so the result is the slab placed:
  the same words as an in-place slab write). Phase 2: an in-place slab add
  for disjoint narrows of one operand (`OP_RN_SLAB_ADD`), same words, fewer
  passes, its own A/B. Node: CUSTOM tag `RN_NARROW` (161).

### 2.6 Take and scatter-add along an axis (layout lane)

`rankn_take(tape, ex, x, ids: IdsHandle, axis)`: the output replaces
`dim[axis]` by the id count. Definition: permute `axis` to the front (a
view), materialise the `[D_axis, rest]` matrix if the permute moved
anything, `Tape.embedding(table = that matrix, ids)` (autograd/
embedding_ops.mojo: the gather and the run-sorted scatter-add VJP, the
embedding profile's words and plan rule `EMB_AUTO_SORT_MIN_CELLS`, a work
rule), then a permute view back. The lane adds no kernel; it adds the
composition and the executor-side id check `Tape.embedding` already runs
(`OP_EMB_IDS_CHECK`). `axis = 0` on a rank-2 tensor is `embedding(x, ids)`
itself, so `Session.gather_rows` can at last carry a gradient: the Python
lane routes `gather_rows` on a recording tensor to `take(x, idx, 0)`
(`OP_GATHER_ROWS` and `OP_EMB_GATHER` both copy through ftz loads: the same
words).

### 2.7 Along the last axis: softmax, log_softmax, layer_norm, linear, losses

Nothing new. On a rank-N tensor the matrix view's rows are the leading
dims, so `Tape.softmax`, `log_softmax`, `layer_norm` (autograd/layers.mojo),
`linear`, `cross_entropy`, the losses and `dropout` run as they do on 2-D
tensors and the output inherits the shape (1.1). Along another axis the
fold lane defines `rankn_softmax(x, axis)` as permute (view) ->
materialise -> `Tape.softmax` -> permute back (view): the 2-D words on the
permuted matrix. The layer-norm's `(1, features)` weight is the last dim.

### 2.8 conv2d and pools as rank-4 ops (conv lane)

`x [N, C, H, W]` (NCHW, x_cnn's layout), `w [OC, C, KH, KW]`, `b [OC]`,
`stride`, `padding`; `max_pool2d(x, k, stride)`, `avg_pool2d(x, k, stride)`.
Reuse is at the element-body level: every kernel below is a `seq_kernel`
body whose arithmetic is x_cnn's element function, read against the same
parameter words (`CP_*` / `PP_*`, x_cnn/ops.mojo) held in the descriptor
(an FP buffer read with `rn_word`, never an int-cast pointer: the Metal
rule), so each word is x_cnn's word for that element. The x_cnn entries
themselves (`conv_block_forward_into`, x_cnn/device.mojo:3257) are not
called: they run on `cnn_ctx()` with `put`/`fetch` and a `synchronize` per
entry, and fuse pool(relu(conv)) as one block; the tape wants separate,
queued, resident ops.

- conv2d forward: `OP_RN_IM2COL` (thread per `cols[r, j]` word, rows = N OH
  OW, j < C KH KW; `im2col_val`'s rule, zero padding) -> `OP_GEMM` through
  the Exec, `y2 [rows, OC] = cols @ W^T` with W read as `B(k, n) = W[n, k]`
  through strides -> `OP_RN_CONV_OUT` (`out[n, oc, oh, ow] = y2[r, oc] +
  bias[oc]`, one add, x_cnn/ops.mojo:237). The GEMM's word is `op_gemm`'s
  chain over C KH KW terms in im2col order on the sequence GEMM's routes:
  **not** CNNClassifier's `device_gemm` plan (x_cnn/device.mojo:794), so a
  rank-4 conv2d is a new op with its own words; CNNClassifier keeps its
  path and its digests (x_cnn/checks/tape_check.mojo).
- conv2d backward: `OP_RN_CONV_GROWS` (`grow[r, oc] = dout[n, oc, oh, ow]`,
  a copy, :261), dW `[OC, C KH KW] = grow^T @ cols` (OP_GEMM, strides), db =
  `blocked_fold(grow, rows, OC, AXIS_ROWS, 1)` (the existing fold: the
  tape's column-sum order), dcols `= grow @ W` (OP_GEMM), dx =
  `OP_RN_COL2IM` (thread per x word, its taps added in the fixed order of
  x_cnn/neural_col2im.mojo `_col2im_pages`). `cols` is kept from the forward
  in tape storage (`Tape.reserve`), as the resident block keeps
  `save_cols`.
- max_pool2d: `OP_RN_MAXPOOL_FWD` (`maxpool_fwd_at`'s rule; the winner index
  as int32 patterns in tape storage, `saved.p10`), `OP_RN_MAXPOOL_VJP`
  (thread per x word, `maxpool_bwd_val`: the sum over the windows that chose
  it in window order; `zeroed_destination` False). avg_pool2d:
  `OP_RN_AVGPOOL_FWD` / `OP_RN_AVGPOOL_VJP` (`avgpool_fwd_at` /
  `avgpool_bwd_at`'s rules, the divisor x_cnn's).
- Rules: `autograd/rules_conv2d.mojo` `Conv2dRules(BackwardRules)` with tags
  `RN_CONV2D` 176, `RN_MAXPOOL2D` 177, `RN_AVGPOOL2D` 178 (the x_cnn header
  reserves 99-111 for per-layer rules on cnn_ctx; these are tape-executor
  rules, so they take the rank-N block instead). relu between them is the
  tape's unary op.
- Conv1d and the pools' 1-D forms are the 2-D ops on `[N, C, 1, L]`
  (free views), Python only.

### 2.9 Operation table

| op | node | forward launches | VJP launches | order | zeroed dest. |
|---|---|---|---|---|---|
| reshape / flatten / squeeze / unsqueeze | RESHAPE | 0 | 0 (borrowed upstream) | - | - |
| permute | PERMUTE_N 37 (pending) | 0, or 1 `OP_RN_COPY_STRIDED` at a dense reader | 1 `OP_RN_COPY_STRIDED` | C | no |
| add/sub/mul/div, trailing broadcast | existing (ADD..DIVIDE) | 1 `OP_AG_BIN_FWD` | existing | existing | existing |
| add/sub/mul/div, middle-axis broadcast | CUSTOM 136-139 | 1 `OP_RN_BIN_FWD` | same-shape operand 1 `OP_RN_BIN_VJP`; broadcast operand 1 `OP_RN_BIN_VJP_PART` + level(s) + finish | F | no |
| sum / mean, last / all / leading axes | existing REDUCE | existing fold | existing | existing | existing |
| sum / mean, other axis sets | CUSTOM 141 / 142 | 1 `OP_RN_REDUCE_PART` + level(s) + finish | 1 `OP_RN_BCAST_VJP` | F | no |
| max / argmax | CUSTOM 143 | `OP_RN_MAX_PART` + `OP_RN_MAX_LEVEL`(s) + `OP_RN_MAX_FOLD` | 1 `OP_RN_MAX_VJP` | order-free | yes |
| matmul, rank-N @ rank-2 | existing MATMUL | 1 `OP_GEMM` | existing | existing | existing |
| matmul, batched | CUSTOM 152 | 1 `OP_RN_BMM` | 1 or 2 `OP_RN_BMM` | chain (b, k) ascending | no |
| concat | CUSTOM 160 | 1 `OP_RN_SLAB_COPY` per operand | 1 per operand | C | no |
| narrow / split | CUSTOM 161 (pending) | 0, or 1 at a dense reader | 1 copy (+ scratch/accumulate for a second consumer) | C | yes |
| take along an axis | composition (permute, embedding) | 0-1 copy + `OP_EMB_*` | `OP_EMB_*` | embedding's | embedding's |
| softmax / log_softmax / layer_norm / linear / losses, last axis | existing | existing | existing | existing | existing |
| softmax, other axis | composition | 1 copy + softmax + view | the composition's | existing | existing |
| conv2d | CUSTOM 176 | `OP_RN_IM2COL`, `OP_GEMM`, `OP_RN_CONV_OUT` | `OP_RN_CONV_GROWS`, 2 `OP_GEMM`, fold, `OP_RN_COL2IM` | GEMM chain + tap order | no |
| max_pool2d / avg_pool2d | CUSTOM 177 / 178 | 1 | 1 | C / tap order | no |

## 3. How the existing 2-D ops stay bit-identical

- No existing kernel body changes. No existing `Tape` method changes its
  launches or arguments; the core lane adds metadata writes (`_shapes`) and
  the pending-view generalisation, whose `TRANSPOSE` branch is today's code.
- A rank-2 `Tensor` records `ShapeN.matrix(rows, cols)` and never reaches a
  rank-N kernel: every `rankn_*` entry routes rank-2 inputs (and the
  trailing-broadcast, last-axis and rank-N-by-rank-2 cases of 2.1-2.3,
  2.7) to the existing method. "No new words for existing calls" is
  therefore a property of the routing, checked by the stored digests:
  `autograd/checks/mlp_cases_digests.tsv` (48 cases), the view / fused /
  gemm / loss / order suites, `OPS_DIGEST` (python/mojolearn/tests/
  test_ops_session.py), `MODEL_DIGEST` (test_autograd_model.py), the
  transformer / Mamba / CNN arm checks. All must be unchanged after the
  merge; a change is a routing bug.
- The rank-N kernels are new words. Where a rank-N result can be stated as
  a composition of 2-D ops (a trailing-axes reduction, a bmm slice outside
  the split-K rule, a permute materialised then reshaped, a concat of two
  pieces then split), the lane's check asserts bit equality with that
  composition (`_same`, autograd/checks/view_cases.mojo:44): evidence, not a
  contract. The contract is: equal digests on CPU, Metal, NVIDIA and AMD.
- Bits may move between versions (phase-2 tiled bmm, slab add, a split
  order for the broadcast chain): every column together, recorded as a bit
  version in the kernel's comment and in `docs/apple-fast/EXPERIMENTS.md`.

## 4. GPU rules for every lane

- Parallel over output words, or over (output word, leaf block) for a fold.
  No one-thread, one-block or per-sample kernel; no serial chain over a
  data dimension except the declared fold leaf (64 terms) and the GEMM k
  chain (which the 2-D GEMM also has). The bmm backward's broadcast chain
  is the stacked-rows GEMM's K and is parallel over K x N words.
- No host loop over rows, elements, batch items or heads. The only host
  loops are `# driver-loop: over concat operands`, `# driver-loop: over
  layers` in the model trainer, and the fold's `# driver-loop: over tree
  levels` that `blocked_fold` already has. The batch of a bmm is a grid
  dimension, not a loop. The descriptor is built from at most 8 axes
  (`# small-loop(ndim: tensor rank, at most 8): shape metadata, no data`).
- Descriptors are uploaded once per recorded node (metadata), never per
  element; a check that reads a value back (ids bounds) reads one or two
  words as `OP_EMB_IDS_CHECK` does.
- Launch counts and every `Args` integer stay inside Int32 (`_args_i32_out`).
- Metal: typed `FP` pointers only; operand selection by branch, never a
  pointer select (autograd/ops.mojo `_ld_right`); every function a body
  reaches `@always_inline`; unused `Args` slots keep the `Args()`
  placeholder; no buffer bound to two slots of one launch (the `x * x` case
  binds once, `i10 = 1`, as `_binary_to` does).
- Identical numerics: `ld`/`st` (ftz), `add/sub/mul/div`, `identical_div`,
  `identical_exp`, as autograd/ops.mojo. Integers (ids, argmax, winners,
  descriptors) are bit patterns read with `bitcast`, never `ld`.
- No dimension targeting: the fold leaf is the fixed 64 of fold v2; the
  bmm route is `seq_gemm_route`'s cost rule; the take plan is
  `EMB_AUTO_SORT_MIN_CELLS` (work); `rn_collapse` collapses by stride
  compatibility. No rule keys on a size.
- Wait for Modular: nothing here needs a new compiler feature.

## 5. The Python surface (user voice; docstrings carry no status notes)

`mojolearn.ops.Session`:

- `tensor(x, requires_grad=False)`: "x is a scalar or an array of up to 8
  dimensions of bool, integers, float16, float32 or float64, in any memory
  layout, used as float32." A scalar is `(1, 1)`, a 1-D array one column,
  as today; an array of 3 or more dimensions keeps its shape.
- `shape(h)`: "The shape of tensor h as a tuple." `(rows, cols)` for the
  tensors `shape` returns today; the full shape otherwise. `numpy(h)`
  returns an array of that shape.
- `reshape(x, shape)`: "x's elements, row-major, in a new shape with the
  same number of elements; -1 infers one dimension." `flatten(x, start=1)`,
  `squeeze(x, axis=None)`, `unsqueeze(x, axis)`: free views.
- `permute(x, axes)`: "x with its axes reordered." `transpose(x, a=-2,
  b=-1)`: "x with two axes swapped; the matrix transpose by default."
- `sum(x, axis=None, keepdims=False)`, `mean(...)`: "axis is None (all
  elements), an integer or a tuple of axes." `max(x, axis, keepdims=False)`:
  "The largest value along axis (a NaN wins; ties take the first position).
  Its gradient flows to that position." `argmax` as a non-differentiable
  int32 result (`numpy` returns int32 for it).
- `matmul(a, b)`: "The matrix product of the last two axes; leading axes
  are batch axes and broadcast."
- `concat(xs, axis)`, `split(x, sizes_or_sections, axis)`.
- `take(x, indices, axis=0)`: "The slices of x at indices along axis.
  Indices are any integer array; one outside [0, x.shape[axis]) raises. In a
  recording session the gradient of x sums each slice's gradients in
  ascending index position." `gather_rows` on a recording tensor becomes
  `take(x, indices, 0)` and so carries a gradient.
- `softmax(x, axis=-1)`, `log_softmax(x, axis=-1)`; `layer_norm` over the
  last axis; `linear` on any rank ("the last axis is the feature axis");
  `cross_entropy(logits, codes)` with logits `[..., classes]` and codes of
  the leading shape.
- `conv2d(x, weight, bias=None, stride=1, padding=0)`: "x is `(N, C, H, W)`,
  weight `(out_channels, C, kh, kw)`, bias `(out_channels,)`."
  `max_pool2d(x, kernel_size, stride=None)`, `avg_pool2d(...)`.

Argument passing: the packed block holds 8 ints (`OPS_PACK_INTS`); a shape
or axes list of up to 8 plus an axis exceeds it, so the shape ops use the
list entry `ops_op` (already there), or the python lane raises
`OPS_PACK_INTS` to 24 (both paths run the same `op`). New codes continue
from 33 (`OPS_RESHAPE_N` 33, `OPS_PERMUTE` 34, `OPS_SUM_N` 35, `OPS_MEAN_N`
36, `OPS_MAX` 37, `OPS_MATMUL_N` 38, `OPS_CONCAT` 39, `OPS_NARROW` 40,
`OPS_TAKE` 41, `OPS_SOFTMAX_N` 42, `OPS_LOG_SOFTMAX_N` 43, `OPS_CONV2D` 44,
`OPS_MAXPOOL2D` 45, `OPS_AVGPOOL2D` 46, `OPS_ARGMAX` 47). `ops_shape_py`
returns the full dims list; `ops_upload` takes a descriptor of any rank
(`CastIn` today is rows x cols with two strides: the python lane either
requires C order for ndim > 2, as `embedding` does for its ids, ops.py:401,
or extends `CastIn` with per-axis strides; the plan takes the C-order
requirement first, the extension as a follow-up).

`mojolearn.Sequential`:

- `Sequential(input_shape=None)` or `fit(X, ...)` with X of any rank: the
  description records the input shape after the sample axis.
- New layers: `reshape(shape)`, `flatten()` (to `(n_samples, -1)`),
  `permute(axes)`, `conv2d(out_channels, kernel_size, stride=1, padding=0,
  bias=True)`, `max_pool2d(kernel_size, stride=None)`,
  `avg_pool2d(kernel_size, stride=None)`. `linear` acts on the last axis.
  `params_` adds a conv layer's weight `(out, in, kh, kw)` row-major then
  its bias.
- Description version 2 (`autograd/model.mojo parse_model` keeps version 1):
  header `[2, ndim, d0 .. d7, layer count]`, per layer `(code, arg0 .. arg5,
  0)`; `ModelSpec` carries `in_shape` / `out_shape` as `ShapeN` lists instead
  of widths; `model_forward` records the rank-N ops. The conv layers stay in
  `docs-site/api_exclude.txt` until NVIDIA and AMD reference records exist
  for them (the CNN rows are excluded today for that reason); the view,
  reduction, bmm and layout ops are documented once their check digests
  match on the three vendors.

## 6. Lanes

Six Opus lanes, from main after the core lane's interface (1.4) is fixed in
this plan; they code against it without compiling. Each lane: its own
primary files, its own opcode range, its own tag block, its own check
`autograd/checks/rankn_<lane>_cases.mojo` with `run_rankn_<lane>_checks[E:
Exec](mut ex: E)`, `rankn_<lane>_cpu.mojo` (HostExec) and
`rankn_<lane>_gpu.mojo` (DeviceExec) mains on the pattern of
`autograd/checks/view_*.mojo`, printing one `RANKN <lane> <case> digest
<d>` line per case plus `PASS` and raising on any `_same` or
`check_values` failure. Cases use small, odd, non-board shapes (one-leaf,
multi-level and tree-level folds; a non-square bmm; a 3-operand concat).
Shared registration files are append-only touches at distinct anchors:
`sequence/dispatch.mojo` (the range import, fence asserts and `elif`
branch, as every family adds), `autograd/ops_api.mojo` (the python lane
only), `python/mojolearn/ops.py` (the python lane only).

Orchestrator check command per lane and executor (the `native` stage of
tools/check_autograd.py:60):
`mojo build -j 1 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . autograd/checks/rankn_<lane>_{cpu,gpu}.mojo -o <out>/rankn_<lane>_{cpu,gpu}; <out>/rankn_<lane>_{cpu,gpu} > <log>; grep -c '^RANKN' <log>; grep -m 5 FAIL <log>`,
then `diff` the `RANKN` lines of the CPU, Metal, NVIDIA and AMD logs: equal
or the lane is not done. Add each new `rankn_*` pair to
tools/check_autograd.py (python lane).

### Lane 1: ag-rankn-core (merges first)

- Files: `autograd/rankn.mojo` (new: `ShapeN`, `rn_*`, descriptor layout),
  `autograd/rankn_ops.mojo` (new: `OP_RN_COPY_STRIDED` 3600; range
  3600..3609, `OP_RNC_FIRST/LAST`), `autograd/tape.mojo` (the `_shapes` side
  table, `_record_nd`, `custom_nd`, `custom_into_nd`, `tensor_nd`,
  `variable_nd`, `variable_into_nd`, `shape_of`/`ndim`/`dim`, `view_nd`,
  `permute` with PERMUTE_N 37, `operand_nd`, `descriptor`, the pending-view
  generalisation of `_pending`/`_materialize_at`/`_use`, the shape
  inheritance lines of 1.1, PERMUTE_N's VJP branch in `_backward` beside
  TRANSPOSE's), `sequence/dispatch.mojo` (range).
- Check `rankn_core_cases.mojo`: a `[3, 5, 7]` tensor's `view_nd` round
  trips and gradient pass-through; `permute` read dense equals the
  composition of two 2-D transposes and a reshape where one exists; permute
  VJP is the inverse permute of the seed; a permute feeding `Tape.matmul`
  through `operand_nd` equals the materialised product (as view_cases does
  for the transpose view); shape inheritance through relu, softmax, linear
  and the trailing-broadcast add; every existing suite's digests unchanged
  (the orchestrator's existing stages).
- Opcode range 3600..3609; kind 37.

### Lane 2: ag-rankn-fold

- Files: `autograd/rankn_fold.mojo` (new: `rankn_binary`, `rankn_sum`,
  `rankn_mean`, `rankn_max`, `rankn_softmax(axis)`, `RankNFoldRules
  (BackwardRules)`), `autograd/rankn_fold_ops.mojo` (new bodies:
  `OP_RN_BIN_FWD` 3700, `OP_RN_BIN_VJP` 3701, `OP_RN_BIN_VJP_PART` 3702,
  `OP_RN_REDUCE_PART` 3703, `OP_RN_BCAST_VJP` 3704, `OP_RN_MAX_PART` 3705,
  `OP_RN_MAX_LEVEL` 3706, `OP_RN_MAX_FOLD` 3707, `OP_RN_MAX_VJP` 3708; range
  3700..3719), `sequence/dispatch.mojo` (range). Reuses `_bin_value`,
  `_bin_vjp_at`'s arithmetic, `_fold_chain`, `ag_part_finish`, the fold's
  level and finish launches, `identical_div`.
- Check `rankn_fold_cases.mojo`: middle-axis broadcast add/mul/div forward
  equals the materialised (`full` + copy) same-shape op word for word; its
  broadcast-operand gradient equals `Tape.sum(reshape, axis=1)` of the
  elementwise VJP when the broadcast axes trail (EQ2D) and passes the
  finite-difference check otherwise (autograd/checks/gradients.mojo's
  tolerance form via `check_values`); `rankn_sum` over a trailing block
  equals `Tape.sum(view_nd, axis=1)`, over the leading block equals
  `Tape.sum(matrix view, axis=0)`, over a middle axis against the host
  oracle built elementwise in Mojo (never a NumPy product: the Oct 19 rule);
  mean factors; max/argmax with ties and one NaN case; `rankn_softmax(axis=1)`
  of a `[4, 6, 5]` equals the permute-softmax-permute composition.
- Tags 136..151 (`RN_BINARY` 136 + op kind, `RN_REDUCE_SUM` 141,
  `RN_REDUCE_MEAN` 142, `RN_MAX` 143).

### Lane 3: ag-rankn-bmm

- Files: `autograd/rankn_bmm.mojo` (new: `rankn_matmul`, `RankNBmmRules`),
  `autograd/rankn_bmm_ops.mojo` (new: `OP_RN_BMM` 3800; 3801 reserved for
  the phase-2 tiled dispatch; range 3800..3809), `sequence/dispatch.mojo`.
  Reuses `gemm_dot`, `seq_gemm_route` (phase 2), `operand_nd`.
- Check `rankn_bmm_cases.mojo`: `[2, 3, 23, 37] @ [2, 3, 37, 19]` forward
  and both gradients equal, slice by slice, `Tape.matmul` on the slices
  (shapes outside `gemm_split_shape`); a transposed right operand through
  `permute` equals the materialised one; a `[1, 3, K, N]` broadcast weight's
  gradient against the host oracle (the stacked chain, written in Mojo);
  a four-line attention `softmax(q @ k^T * s) @ v` on `[2, 4, 11, 8]`
  against the composition of per-head 2-D ops.
- Tag 152 (`RN_BMM`).

### Lane 4: ag-rankn-layout

- Files: `autograd/rankn_layout.mojo` (new: `rankn_concat`, `rankn_narrow`,
  `rankn_split`, `rankn_take`, `RankNLayoutRules`),
  `autograd/rankn_layout_ops.mojo` (new: `OP_RN_SLAB_COPY` 3900, 3901
  reserved for `OP_RN_SLAB_ADD`; range 3900..3909), `sequence/dispatch.mojo`.
  Reuses `Tape.embedding`, `Tape.ids`, `embedding_refuse_ids`, `permute`,
  `operand_nd`.
- Check `rankn_layout_cases.mojo`: concat of three `[3, k, 5]` pieces then
  split returns each piece bit for bit and the gradients place the seed's
  slabs; a second consumer of one narrow accumulates in the tape's order
  (equal to the 2-D add of two reshapes); `take` along axis 1 of `[3, 7,
  5]` equals the permute-embedding-permute composition and its gradient the
  embedding's (`emb_backward_oracle`'s words for the touched cells); an id
  out of range is refused on the executor.
- Tags 160..175 (`RN_CONCAT` 160, `RN_NARROW` 161).

### Lane 5: ag-rankn-conv

- Files: `autograd/rankn_conv.mojo` (new: `rankn_conv2d`, `rankn_max_pool2d`,
  `rankn_avg_pool2d`), `autograd/rules_conv2d.mojo` (new: `Conv2dRules`),
  `autograd/rankn_conv_ops.mojo` (new bodies: `OP_RN_IM2COL` 4000,
  `OP_RN_CONV_OUT` 4001, `OP_RN_CONV_GROWS` 4002, `OP_RN_COL2IM` 4003,
  `OP_RN_MAXPOOL_FWD` 4004, `OP_RN_MAXPOOL_VJP` 4005, `OP_RN_AVGPOOL_FWD`
  4006, `OP_RN_AVGPOOL_VJP` 4007; range 4000..4019),
  `sequence/dispatch.mojo`. Reuses the arithmetic of x_cnn/ops.mojo
  `im2col_val`, the NCHW layout rule (:237, :261), `_col2im_pages`'s tap
  order, `maxpool_fwd_at` / `maxpool_bwd_val`, `avgpool_fwd_at` /
  `avgpool_bwd_at`, the `CP_*` / `PP_*` parameter layout; `OP_GEMM` through
  the Exec; `blocked_fold` for the bias.
- Check `rankn_conv_cases.mojo`: a `[2, 3, 9, 11]` input, `[4, 3, 3, 3]`
  kernel, stride 1 and 2, padding 0 and 1: forward against a host oracle
  written elementwise in Mojo; dx, dW, db by finite differences
  (`check_values`); max and avg pools 2x2 and 3x3 stride 2 likewise; digests
  across the four executors. Optional evidence: the conv forward's words
  against x_cnn's host twin (x_cnn/host/ops_host.mojo) on the same case.
- Tags 176..191 (`RN_CONV2D` 176, `RN_MAXPOOL2D` 177, `RN_AVGPOOL2D` 178).

### Lane 6: ag-rankn-python (merges last)

- Files: `autograd/ops_api.mojo` (`shapes: List[ShapeN]`, the new codes,
  `RankNRules`: one struct delegating by tag block to `NeuralRules`,
  `RankNFoldRules`, `RankNBmmRules`, `RankNLayoutRules`, `Conv2dRules`),
  `autograd/pyapi.mojo` (`ops_shape_py` full dims; `ops_upload_py` any
  rank), `python/mojolearn/ops.py` (the methods of section 5),
  `python/mojolearn/_cast_in.py` (C-order rank-N descriptors),
  `autograd/model.mojo` (description v2, `in_shape`/`out_shape`, the new
  layer codes 9 Reshape, 10 Permute, 11 Flatten, 12 Conv2d, 13 MaxPool2d,
  14 AvgPool2d in `model_forward`), `python/mojolearn/_x_sequence_model.py`
  (`input_shape`, the new layers), `python/mojolearn/tests/
  test_ops_rankn.py` (`RANKN_DIGEST` lines) and a rank-4 case in
  `test_autograd_model.py`, `tools/check_autograd.py` (the five `rankn_*`
  pairs), `docs-site/api_exclude.txt` (conv layers until records),
  `docs-site/user_guide/autograd.md` (a shapes section).
- No opcode range (no kernel). Its check is the two Python tests' digest
  lines equal on CPU, Metal, NVIDIA and AMD, plus `OPS_DIGEST` and
  `MODEL_DIGEST` unchanged.

### Merge order and what the orchestrator builds

1. core (small; compile; every existing autograd suite's digests
   unchanged: `tools/check_autograd.py --backend both`, the view / fused /
   gemm / loss / order / embedding suites, `OPS_DIGEST`, `MODEL_DIGEST`).
2. fold, bmm, layout, conv in any order or together (independent files;
   each adds its block to sequence/dispatch.mojo). Compile once; run the
   four `rankn_*` pairs on CPU, Metal, NVIDIA, AMD; diff the `RANKN` lines.
3. python (depends on all five). Compile the sequence binding and its host
   twin (`bindings/_mojolearn_x_sequence.mojo`, `_mojolearn_x_sequence_host.mojo`:
   the only bindings that import autograd/ops_api.mojo and autograd/model.mojo);
   run `test_ops_session.py`, `test_ops_rankn.py`, `test_autograd_model.py`,
   `test_autograd_public.py` on the four executors.

## 7. Registry

Opcode ranges (all above `OP_CAST_LAST` 3502, autograd/cast_ops.mojo:29;
each fenced in sequence/dispatch.mojo as the cast family is, :354):

| range | family | lane |
|---|---|---|
| 3600..3609 | `OP_RNC_*` views (copy strided) | core |
| 3700..3719 | `OP_RNF_*` broadcast, reductions, max | fold |
| 3800..3809 | `OP_RNB_*` batched matmul | bmm |
| 3900..3909 | `OP_RNL_*` slab copy / add | layout |
| 4000..4019 | `OP_RNV_*` conv2d, pools | conv |

Node kinds and custom tags: PERMUTE_N 37 (tape kind, core); tags 128..135
core (reserved), 136..151 fold, 152..159 bmm, 160..175 layout, 176..191
conv; 99..111 stay x_cnn's. Session op codes 33..47 (python).

## 8. Out of scope, owed, and the decisions taken

- Strided (non-C-order) rank-N uploads: C order required first (the
  `embedding` ids precedent); the `CastIn` extension is a follow-up of the
  python lane, not a blocker.
- Tensors above rank 8: refused with a message; no use case asked for more.
- Integer tensors as first-class tensors: not in this plan (ids and
  argmax are int32 patterns in float storage, as today).
- Phase-2 speed work, each its own A/B with the define pattern of
  CLAUDE.md "Experiments": the tiled batched GEMM (3801), the in-place slab
  add (3901), a split order for the bmm broadcast chain (a bit version),
  reading NCHW gradients through two-stride operands instead of
  `OP_RN_CONV_GROWS`.
- Attention and the transformer trainer keep their fused kernels
  (autograd/rules_transformer.mojo); rank-N ops are for user compositions
  and `Sequential`, and the bmm check's four-line attention is a
  correctness case, not a speed claim.
- Decided here, not asked: CUSTOM tags instead of new tape kinds (lanes
  stay out of `_backward`); descriptors over `Args` ints (rank 8 needs
  more than twelve); the order-free max (no bit version can ever come from
  a fold change); the stacked chain for bmm's broadcast gradient (no fold
  kernel, one declared order); conv2d on the sequence GEMM, not
  CNNClassifier's plan (the tape's routes and words).
