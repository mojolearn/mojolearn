# Importing PyTorch and ONNX models for inference (plan, lane ag-importer-plan, 2026-10-20)

Ideas and plan only: no code was written for this document and nothing was
compiled or run. It is the brief the Opus implementation lanes code against.
Read with docs/plans/AUTOGRAD_API.md (the session and the Program design
note), docs/plans/AUTOGRAD_RANKN.md (the rank-N operation set this plan
lowers onto) and the two sibling briefs it depends on,
briefs-20261020/ag-program.md (the captured Program) and
briefs-20261020/ag-weight-import.md (weights into session parameters).

## 0. Decisions in twelve lines

1. **An imported model is a Program plus weights.** The importer never adds
   an executor, an interpreter or a second runtime: it replays the source
   graph once, eagerly, through `ops.Session` (one Mojo call per op), then
   `Session.capture` (lane ag-program) turns that node list into a Program;
   `Program.run` is one binding call. The weights are session parameters
   uploaded by the cast path of lane ag-weight-import.
2. **ONNX first, `torch.export` second, TorchScript never.** ONNX is a
   static, self-describing file with a frozen op set and a specification per
   opset; it needs no torch at import time, and a test can write one without
   torch or onnx installed. `torch.export.ExportedProgram` is read in
   process, lowered to the same intermediate graph and the same table.
   TorchScript is deprecated upstream and has dynamic control flow by
   construction.
3. **One intermediate graph (`GraphIR`)** in ONNX's vocabulary (op names,
   attribute names, NCHW, `[out, in]` linears). The ONNX reader fills it
   from a file; the torch.export reader fills it from an `ExportedProgram`
   after `run_decompositions()`; one lowering table serves both.
4. **Every op is lowered or refused by name.** A graph with an op outside
   the table, or inside it with an attribute outside the supported subset,
   is refused with one error listing every offending node (op, node name,
   reason). Nothing is approximated, skipped, or replaced by a fallback.
5. **Python touches shapes and names, never element values.** Shape
   inference and the folding of integer shape subgraphs (`Shape`, `Gather`
   on shapes, `Concat` of ints, `Range`, `ConstantOfShape`) are description
   work in Python. Every computation whose inputs are tensor element values,
   including one-element constants and BatchNorm's folded scale, runs as
   session ops on the device in an import-time **prologue** whose results
   become parameters. The lowering raises if a value computation would need
   a Python number.
6. **Static shapes.** A Program is fixed-shape. `import_model(path,
   shapes={name: shape})` resolves dynamic axes once; the import's eager
   pass validates every op on placeholder tensors of those shapes, so a
   shape error names the node at import time, not at run time.
7. **Identity by construction.** The Program replays the tape's kernels
   (blocked folds with declared orders, `identical_*` numerics), so the
   same bits come out on Apple, NVIDIA and AMD, and on the host executor,
   for any imported model. Torch or onnxruntime output is a **reference**
   for closeness: a tool reports the distance, nothing gates on it.
8. **Missing kernels are elementwise.** The present tape and rank-N ops
   express the structural operations (matmul, conv, pools, norms, softmax,
   reshapes, broadcasts, gather, concat/split, attention patterns). What
   is missing is a set of elementwise kinds (erf, gelu, clip, leaky relu,
   elu, hard sigmoid/swish, softplus, abs, neg, floor, ceil, round, sin,
   cos, pow, min, max, where, comparisons), one new family in one file,
   plus `expand` and `argmax` through existing kernels.
9. **Weights follow ag-weight-import.** float32 bit for bit; float16 and
   bfloat16 widened on the device; float64 cast on the device; int64
   indices to int32 on the device with the embedding range check; the
   `[out, in]` to `[in, out]` transpose of a `Gemm` weight is one strided
   copy in the prologue, so `Program.run` carries no layout work.
10. **Python surface:** `mojolearn.import_model(source, *, shapes=None,
    numeric_mode=None) -> Program`; the Program has `run`, `inputs`,
    `outputs`, `save`; `mojolearn.importers.supported_ops()` lists the
    table. `fast` is Apple only in every docstring.
11. **Five Opus lanes, disjoint files:** the ONNX reader and IR, the
    lowering table, the elementwise kernels, the torch.export reader, and
    docs with the closeness tool. Merge order: reader and kernels (any
    order), then lowering (after ag-program and ag-weight-import), then
    torch.export and docs.
12. **Checks are torch-free fixtures.** The tests write small ONNX files
    with a 60-line protobuf encoder of their own (no `onnx`, no `torch`),
    import them, and compare the imported Program's output word for word
    with an eager `ops.Session` composition of the same weights, printing
    `IMPORT_DIGEST` lines that must be equal on CPU, Metal, NVIDIA and AMD.

## 1. The target: a Program and its weights

Today (main 33196ba55) `ops.Session` (python/mojolearn/ops.py,
autograd/ops_api.mojo) runs one binding call per op over tensors of up to
rank 8 (docs/plans/AUTOGRAD_RANKN.md, merged), with `Sequential`
(python/mojolearn/_x_sequence_model.py, autograd/model.mojo) as the one
"whole model in one call" path, limited to its own layer codes.

Lane ag-program adds the captured Program: `prog = s.capture(inputs=[h...],
outputs=[h...])` records the session's node list between the named inputs
and outputs (shapes, op kinds, saved args, parameter handles) into a Mojo
Program object; `prog.run(*arrays)` / `s.run(prog, arrays)` is one binding
call that uploads the inputs through the `CastIn` descriptor, replays every
node on the device with fresh intermediates, parameters read in place, and
downloads the outputs. Inference only; the description plus the weights
save and load through the checkpoint writer (training/checkpoint.mojo,
autograd/model_checkpoint.mojo; a new section kind beside
`CKPT_SECTION_SEQUENTIAL` 1 and `CKPT_SECTION_SESSION` 2), loadable on a
CPU-only install and run by the host executor.

Lane ag-weight-import adds `Session.load_weights(path, names=[(handle,
key), ...])` and the layout codes: Python matches names and hands buffers
plus a layout code; transposes, reshapes and dtype widening run in Mojo
(autograd/cast_ops.mojo, autograd/rankn_ops.mojo `OP_RN_COPY_STRIDED`,
`f16_to_f32_bits` and the bf16 shift beside it).

The importer is the third piece and the one that needs no new runtime:

```
source file / object
  -> GraphIR (nodes in topological order, initializers as raw buffers,
     value shapes, graph inputs and outputs)                          [Python, description only]
  -> refusal pass (every unsupported op or attribute, one error)      [Python, names only]
  -> Session(record=False):
       parameters  = initializers uploaded through the cast path      [Mojo: cast kernels]
       prologue    = parameter-only subgraphs run once                 [Mojo: tape kernels]
       placeholders= graph inputs as uploaded zeros of the resolved shapes
       body        = the remaining nodes, one session call each        [Mojo: tape kernels]
  -> prog = s.capture(inputs=placeholders, outputs=outputs)           [ag-program]
  -> Program (owns the session; run / save / inputs / outputs)
```

Everything the body records is a tape node the Program already knows how
to replay. The importer adds no node kind, no replay rule and no binding
entry of its own; its only Mojo is the elementwise family of section 3.3.

## 2. Which source format first, and why

| | ONNX (`.onnx`, opset 13+) | `torch.export` (`ExportedProgram`) | TorchScript | `nn.Module` directly |
|---|---|---|---|---|
| Static graph | yes, by definition | yes (control flow is specialized or exported as HOPs we refuse) | no: `prim::If`, `prim::Loop`, Python semantics | no |
| Self-describing | yes: shapes, dtypes, initializers, opset versions in one file | yes in process; the on-disk form (`torch.export.save`) stores weights as a torch pickle | partly | no |
| Needs torch at import | no | yes (in process) | yes | yes |
| Testable without torch | yes: a test writes a `.onnx` with a tiny protobuf encoder | no: tests are skipped unless torch is installed | no | no |
| Op set size | ~190 ops; the inference subset in use is ~60 | core ATen after `run_decompositions()`: ~180, the usual models reach ~50 | thousands of aten overloads | n/a |
| Who exports to it | torch (`torch.onnx.export(dynamo=True)`, built on torch.export), HF Optimum, TensorFlow, JAX, scikit-learn | torch 2.x | legacy torch | - |
| Status upstream | stable, versioned | the current PyTorch path | deprecated | - |

**ONNX first.** It gives the whole importer, including its checks and the
refusal policy, without a torch dependency anywhere in the package, and it
covers PyTorch users through torch's own exporter (which is torch.export
underneath). **torch.export second**, in process: the user already has the
object, `run_decompositions()` shrinks the op set to core ATen, and the
state_dict tensors hand over their memory with no arithmetic. The aten to
IR table is small because the IR speaks ONNX's names. TorchScript and raw
`nn.Module` tracing are not planned: the former is deprecated and dynamic,
the latter would mean writing a tracer.

Opset floor: 13. Below 13 `Softmax`, `LogSoftmax`, `Squeeze`/`Unsqueeze`
(axes as attributes), `Split` and `ReduceSum` have different semantics;
rather than carry two meanings per op the importer refuses `opset < 13` by
version with the hint to run `onnx.version_converter` once. Only the
default domain (`""` / `ai.onnx`) is lowered; `ai.onnx.ml`, `com.microsoft`
and every other domain are refused by name (section 3.4 lists the fused
Microsoft ops we may decompose later).

## 3. The operation table

Vocabulary: "direct" = one or more existing session calls, no new kernel,
no new words (the kernels and their orders are the ones the autograd check
suites already hold equal on CPU, Metal, NVIDIA and AMD); "composition" =
several existing calls (new words, but only existing kernels);
"prologue" = parameter-only, run once at import; "L3" = a kind of the new
elementwise family (section 3.3); "refuse" = by name, with the estimated
cost of a later kernel.

Session calls referenced below are the ones in python/mojolearn/ops.py
on main: `matmul` (rank-2 and batched with NumPy batch broadcasting),
`add/sub/mul/div` (same shape, scalar, row, column, trailing and
middle-axis broadcasts), `relu/sigmoid/tanh/exp/log/sqrt/scale`,
`transpose/permute`, `reshape/flatten/squeeze/unsqueeze`, `sum/mean/max`
(axis int or tuple, keepdims), `softmax/log_softmax(axis)`, `linear`,
`layer_norm` (last axis, weight and bias), `dropout`, `embedding`, `take`,
`concat`, `split`, `conv2d(stride, padding)`, `max_pool2d/avg_pool2d
(kernel_size, stride)`, `full`, `copy`, `detach`.

### 3.1 Direct and composition (no new kernel)

| ONNX op (opset 13+) | aten (core ATen) | Lowering | Supported subset; the rest is refused by name |
|---|---|---|---|
| MatMul | mm, bmm, matmul | `matmul` | any ranks that broadcast NumPy's way; 1-D operands via unsqueeze/squeeze |
| Gemm | addmm, mm (+ t) | prologue: `transpose` of B when `transB=1` (one strided copy), `scale` of B by `alpha` when `alpha != 1`, `scale` of C by `beta`; body: `linear(A, B', C')` when C is `[N]` or `[1, N]`, else `matmul` + `add` | `transA=0` (transA=1 is a body `transpose`, allowed) |
| Add, Sub, Mul, Div | add, sub, mul, div (Tensor and Scalar) | `add/sub/mul/div` | shapes that broadcast (scalar, trailing, middle axis); a python scalar operand becomes `scale`/`add` of a `full` constant |
| Relu, Sigmoid, Tanh, Exp, Log, Sqrt | relu, sigmoid, tanh, exp, log, sqrt | the unary call | - |
| Neg | neg | `scale(x, -1.0)` | - |
| Reciprocal | reciprocal | `div(full(1.0), x)` | - |
| Pow | pow.Tensor_Scalar | exponent 2: `mul(x, x)`; 0.5: `sqrt`; 3: `mul(mul(x,x),x)`; -1: Reciprocal; other exponents and tensor exponents: L3 `pow` | constant exponent in {2, 0.5, 3, -1} direct |
| Softmax, LogSoftmax | _softmax, _log_softmax | `softmax(axis)` / `log_softmax(axis)` | any axis |
| LayerNormalization (opset 17) | native_layer_norm | `layer_norm(x, scale, bias)`; a missing bias becomes `full((1, d), 0.0)` in the prologue | `axis == -1` (or the last axis), `stash_type` default; other axes: a `permute` around it when the normalized axes are one axis, else refuse |
| RMSNormalization (opset 23) and the Pow/ReduceMean/Add/Sqrt/Div/Mul pattern | the same pattern after decomposition | composition: `mul(x,x)`, `mean(axis=-1, keepdims)`, `add(eps)`, `sqrt`, `div`, `mul(weight)` | last axis |
| BatchNormalization (inference) | _native_batch_norm_legit_no_training | prologue: `s = div(scale, sqrt(add(var, eps)))`, `t = sub(B, mul(mean, s))`, both reshaped `[1, C, 1, 1]`; body: `add(mul(x, s), t)` (middle-axis broadcast) | `training_mode=0`; rank 4 (rank 2 and 3 by reshaping to `[N, C, 1, 1]`) |
| InstanceNormalization | instance_norm | composition: `mean(axis=(2,3), keepdims)`, `sub`, `mul`, `mean`, `add(eps)`, `sqrt`, `div`, `mul(scale[1,C,1,1])`, `add(bias)` | rank 4 |
| GroupNormalization (opset 18) | group_norm | `reshape([N, G, -1])`, the InstanceNorm composition over the last axis, `reshape` back, then scale and bias per channel | rank 4 |
| Transpose | permute, transpose, t | `permute(perm)` | - |
| Reshape | view, reshape, _unsafe_view | `reshape(shape)` | static shape (an initializer or a folded shape subgraph); `allowzero=0` |
| Flatten | flatten | `flatten(start=axis)` | - |
| Squeeze, Unsqueeze | squeeze, unsqueeze | `squeeze(axes)` / `unsqueeze(axis)` | static axes |
| Expand | expand | L3 `expand` (a zero-stride copy through `OP_RN_COPY_STRIDED`, no new body) | static shape |
| Concat | cat | `concat(xs, axis)` | - |
| Split | split_with_sizes, split | `split(x, sizes, axis)`, one output per piece | static sizes |
| Slice | slice | `split` along each sliced axis and the piece kept (a free pending view until a dense reader) | `steps == 1`, static starts and ends; negative steps refused |
| Gather (constant indices) | index_select, select | `take(x, indices, axis)` | indices an initializer or a folded shape value |
| Gather axis 0 on a 2-D table with graph-input indices | embedding | `embedding(table, ids)`; the ids become an integer input of the Program (int64 cast to int32 on the device, range-checked by `OP_EMB_IDS_CHECK`) | table a parameter |
| ReduceSum, ReduceMean, ReduceMax | sum.dim_IntList, mean.dim, amax | `sum/mean/max(axis, keepdims)` | static axes; `noop_with_empty_axes=0` |
| ReduceMin | amin | `scale(max(scale(x, -1), axis), -1)` | - |
| ReduceL2 | linalg_vector_norm (ord 2) | `sqrt(sum(mul(x, x), axis))` | - |
| GlobalAveragePool, GlobalMaxPool | adaptive_avg_pool2d (output 1x1), amax | `mean(axis=(2,3), keepdims=True)` / `max(...)` | rank 4 |
| Conv (2-D) | convolution | `conv2d(x, w, b, stride, padding, groups)` | any `group` dividing both channel counts (grouped and depthwise: lane ag-import-conv-gaps, one OP_RN_BMM launch over every group), `dilations=(1,1)`, equal strides, any non-negative pads (equal pads lower as `padding`, unequal ones as `pad` then `padding=0`), `auto_pad` NOTSET, VALID, SAME_UPPER or SAME_LOWER resolved to explicit pads from the static shapes at lowering, `transposed=False`; `kernel_shape` matches the weight |
| Conv (1-D) | convolution (3-D input) | `unsqueeze(x, 2)`, `unsqueeze(w, 2)`, `conv2d` with `kh=1`, `squeeze(2)` | the same subset; padding on W only |
| MaxPool, AveragePool (2-D) | max_pool2d_with_indices (first output), avg_pool2d | `max_pool2d(k, stride)` / `avg_pool2d(k, stride)` | square `kernel_shape`, equal strides, `ceil_mode=0` (or windows that fit), `dilations=1`, `storage_order=0`; pads and `auto_pad` as Conv, each pad below the kernel size. With pads: MaxPool = `pad(-inf)` then `max_pool2d`; AveragePool `count_include_pad=1` = `pad(0)` then `avg_pool2d`; `count_include_pad=0` (the default) = the window sum (a depthwise `conv2d` of `pad(0)` with ones, taps in (kh, kw) order) divided by a count map (the same convolution of a zero-padded ones image, exact integers, built on the device once and kept as a program parameter) |
| Dropout | dropout (eval), native_dropout | identity (the input handle; inference import) | `training_mode` absent or false |
| Identity, Constant | clone, alias, detach, lift_fresh_copy | the input handle / an initializer | - |
| Cast | _to_copy, to.dtype | to float16/bfloat16/float32/float64: identity (storage is float32; a widening already happened at upload); to int on element values: refuse (shape subgraphs fold statically) | - |
| Shape, Size, ConstantOfShape, Range, Gather/Concat/Slice/Unsqueeze/Cast on shape values | sym_size, arange on sizes | folded statically in Python (integers from the resolved shapes); never a tensor | the subgraph must be integer-only and reachable from shapes or constants |
| Where on a static condition | where (constant cond) | the selected branch handle | condition an initializer (e.g. a causal mask as a selected constant); a data-dependent condition: L3 `where` |
| Attention pattern: MatMul(Q, Transpose(K)), Div/Mul by sqrt(d), Add(mask), Softmax(-1), MatMul(., V), with Reshape/Transpose for heads | the decomposition of scaled_dot_product_attention | `matmul` (batched), `scale`, `add` (leading-axis broadcast of a `[1, 1, T, T]` mask parameter), `softmax(axis=-1)`, `matmul` | the mask an initializer or a Where/Trilu folded in the prologue; a data-dependent mask input is a float input added with broadcasting |
| Trilu on a constant | triu/tril on a constant | prologue: refuse unless the input is an initializer (then it is an `embedding`-free constant lowered through L3 `where` on a static index mask the importer uploads) | see note below |
| Silu / Mul(x, Sigmoid(x)) | silu | composition `mul(x, sigmoid(x))` (zero kernels); a fused L3 `silu` kind (`identical_silu` exists) is a later launch-count A/B, not needed for coverage | - |
| Gelu (opset 20) | gelu | L3 `gelu` (`identical_gelu_erf`, `identical_gelu_tanh`), both `approximate` values | - |
| Erf (the pre-opset-20 Gelu pattern) | erf | L3 `erf` (`identical_erf`) | - |

Trilu note: a causal mask reaches the graph either as an initializer
(direct: upload it) or as `Trilu(ConstantOfShape(...))` with static
arguments. For the second form the importer allocates the `[T, T]` 0/1
pattern in Python and uploads it as a constant: the entries are positions
(row >= column), description rather than data arithmetic, like the shape
integers. A `Trilu` whose input is a graph value is refused. If the
position pattern is judged compute after all, the fallback is an L3
`trilu` kind (one body, small).

### 3.2 Attention and transformer blocks

Everything a decoder layer exported by torch or Optimum contains is in 3.1:
embedding, RMSNorm or LayerNorm patterns, Gemm/MatMul projections, head
reshapes and transposes, the batched QK^T, the scale, the additive mask, the
softmax, the PV product, the output projection, SiLU/GELU MLPs and residual
adds. Rotary embeddings export as Mul/Add/Concat/Split over precomputed
`cos`/`sin` initializers (direct) or over `Sin`/`Cos` of a position Range
(L3 `sin`/`cos`, or a prologue when positions are static). KV caches export
as extra inputs and outputs with Concat along the time axis: direct, but
the Program is fixed-shape, so a cached decode step is a second Program
imported with the cache shapes (section 8).

The fused `Attention` op (opset 23) and the `com.microsoft` fused ops
(`MultiHeadAttention`, `Attention`, `SkipLayerNormalization`,
`SkipSimplifiedLayerNormalization`, `BiasGelu`, `FastGelu`, `FusedMatMul`,
`Gelu`, `EmbedLayerNormalization`, `RotaryEmbedding`) are refused by name in
the first release. Each is a composition of 3.1 with no kernel; they are a
follow-up lane's table rows once a real exported model asks for them.

### 3.3 New kernels: the elementwise family (lane 3)

One new file autograd/import_ops.mojo (kernels, `OP_IMP_*` 4600..4619) and
autograd/import_api.mojo (the tape calls, `ImportRules` BackwardRules,
custom tags 192..207), registered in sequence/dispatch.mojo at the next
anchor and exposed as session codes 48.. in autograd/ops_api.mojo and
python/mojolearn/ops.py. Every body is one thread per output word,
`@always_inline`, `identical_*` numerics with ftz as autograd/ops.mojo's
unary body does; the kind is a runtime integer of the body as `RELU` ..
`SCALE` are today. Backward rules are included so a recording session can
use the same ops (one VJP body per family, kind-switched); the importer
itself only needs the forward.

| Kind | Body | Numerics on hand (checks/numerics.mojo) | Cost |
|---|---|---|---|
| `erf`, `gelu` (erf), `gelu` (tanh) | `OP_IMP_UNARY_FWD` 4600 / `_VJP` 4601 | `identical_erf`, `identical_gelu_erf`, `identical_gelu_tanh` | small: kinds in one body |
| `softplus`, `silu`(fused, optional), `leaky_relu(alpha)`, `elu(alpha)`, `hard_sigmoid(alpha, beta)`, `hard_swish`, `clip(min, max)`, `abs`, `floor`, `ceil`, `round` (half to even), `sin`, `cos`, `sign`, `reciprocal` | the same two bodies, float args in `Args` f-slots | `identical_softplus`, `identical_silu`, `identical_clamp`, `identical_sin`, `identical_cos`; floor/ceil/round/abs/sign are exact | small |
| `pow(x, y)` tensor or scalar exponent, `minimum`, `maximum`, `where(cond, a, b)` (cond as 0/1 float words), `equal`, `greater`, `less`, `greater_equal`, `less_equal`, `not`, `and`, `or` (0/1 results) | `OP_IMP_BIN_FWD` 4602 / `_VJP` 4603 (`where` reads three slots: `_ld_right`'s branch-and-load pattern, never a pointer select), broadcasting through the rank-N descriptor as `OP_RN_BIN_FWD` does | `identical_pow`, `identical_fmax` | small; the broadcast descriptor is the fold lane's |
| `expand(x, shape)` | no body: `OP_RN_COPY_STRIDED` with zero strides on broadcast axes (autograd/rankn_ops.mojo) | - | tiny: a tape call and a session code |
| `argmax(x, axis)` (forward only, int32 words in float storage) | `OP_IMP_ARGMAX` 4604 over the rank-N max partials of autograd/rankn_fold.mojo (`OP_RN_MAX_*` already carry the winning position for the VJP; argmax is that position written out) | order-free (first position on ties, NaN wins, as `max`) | small if the position is already in the partials, medium otherwise |
| `pad(x, pads, value)` (constant pad outside conv) | `OP_IMP_PAD` 4605: one thread per output word, source index or the fill | exact | small |

The VJPs: unary by `y` or `x` as autograd/ops.mojo chooses per kind; the
binary family's broadcast VJP reuses the fold lane's `OP_RN_BIN_VJP_PART`
and levels (same declared order F); `where`'s VJP is `where(cond, g, 0)` and
`where(cond, 0, g)`; comparisons, `argmax`, `floor/ceil/round/sign` pass no
gradient (piecewise constant); `expand`'s VJP is the fold lane's broadcast
reduction; `pad`'s VJP is a narrow. Check: autograd/checks/import_cases.mojo
with cpu/gpu mains on the `rankn_*` pattern, `IMPORT <kind> <case> digest
<d>` lines, odd non-board shapes, every kind once forward and once through a
gradient against a closed form.

### 3.4 Refused by name (first release), with the cost of a later kernel

| ONNX op | Why not now | Later cost |
|---|---|---|
| Resize, Upsample, GridSample | interpolation kernels | nearest: small (a gather by computed index); linear/cubic: medium |
| Pad (non-constant modes: reflect, edge, wrap) | index mapping per mode | small each |
| Tile | `expand` + `reshape` cannot tile along a non-leading axis without a copy | small (strided copy with modular index) |
| CumSum | a scan with a declared order | medium: core/device_scan.mojo's blocked scan, order declared once |
| TopK, NonZero, NonMaxSuppression, Unique | sort/select with data-dependent output shapes | TopK: medium (metrics/impl/binary_ranking.mojo's bitonic sort); NonZero/NMS: data-dependent shape, no Program form |
| ArgMin | not asked | small once argmax exists (negate) |
| ScatterND, ScatterElements, GatherND, GatherElements (data-dependent indices) | collision order must be declared; indices are data | medium: sort pairs then fold per destination as autograd/embedding_ops.mojo does |
| Einsum | equation parser plus a general contraction | the common two-operand equations are bmm with permutes: medium; general: high |
| Loop, If, Scan, SequenceAt and every control-flow op | the Program is a straight line; AUTOGRAD_API.md's `IF`/`LOOP` design note is not implemented | high; a data-dependent `If` on a shape is never a Program |
| LSTM, GRU, RNN | the recurrent trainers are not tape ops | high (or lower to an unrolled composition for a fixed T: medium, no kernel) |
| QuantizeLinear, DequantizeLinear, MatMulInteger, DynamicQuantizeLinear, QLinearConv | integer tensors are not first-class in the tape; lowbit weights have their own path (`mojolearn.lowbit`) | medium: dequantize in the prologue to float32 (exact by construction), then direct |
| DepthToSpace, SpaceToDepth | reshape/permute compositions exist but the 6-D permute of a rank-4 pending view needs rank 6 strided copy: supported by rank 8 | small (composition only; verify) |
| LRN, Softsign, Mish, ThresholdedRelu, Selu, Celu, HardMax, Shrink, Atan/Asin/... | rare | small each, L3 kinds |
| Mod, IsNaN, IsInf, BitShift, bitwise ops | rare | small each |
| Det, Inverse, Cholesky-style linear algebra | not elementwise, not a tape op | high; out of scope |
| Random* ops, Multinomial, Bernoulli | inference imports hold no RNG; Philox exists for dropout only | n/a |
| `ai.onnx.ml` domain (TreeEnsemble, LinearClassifier, ...) | a different product (scikit-learn exports); mojolearn has its own estimators | out of scope |
| `com.microsoft` and other domains | fused ops, see 3.2 | composition only, no kernel |
| Any op whose inputs have a data-dependent shape | no Program form | n/a |
| Opset < 13 | changed semantics (Softmax, Squeeze, Split, ReduceSum) | the user runs `onnx.version_converter` once |

Refusal message, one for the whole graph:

```
UnsupportedModel: 3 nodes of model.onnx cannot be imported:
  /layer.0/attn/Resize (Resize): no kernel; see mojolearn.importers.supported_ops()
  /layer.0/conv (Conv): dilations=(2, 2) (only 1 is supported)
  /head/TopK (TopK): no kernel
```

`UnsupportedModel` subclasses `ValueError`; the list is in graph order and
complete, so a user sees every gap at once.

## 4. Layout and dtype rules

Layouts, fixed by the IR's vocabulary (ONNX's), checked once at lowering:

- **Images are NCHW.** `Conv`, pools and the norms read `(N, C, H, W)`;
  this is the tape's rank-4 convention too (autograd/rankn_conv.mojo). An
  NHWC source graph carries its own `Transpose` nodes; nothing is inferred.
- **Conv weights are `[out, in, kh, kw]`**, the tape's layout: uploaded as
  they are.
- **Linear weights.** `Gemm` with `transB=1` (torch's `nn.Linear`, weight
  `[out, in]`) becomes one strided copy to `[in, out]` in the prologue; the
  Program sees the `[in, out]` parameter and `linear` runs with no
  transpose. `MatMul(x, W)` with a `[in, out]` initializer uploads as is.
  `transA` is a body transpose (rare; allowed).
- **LayerNorm scale and bias** are `[d]`, uploaded as `(1, d)`;
  **BatchNorm** statistics fold in the prologue to `[1, C, 1, 1]`.
- **Embedding tables** are `[vocab, dim]`, the tape's layout.
- **Broadcasting** is NumPy's (ONNX's), which the rank-N ops implement:
  scalar, trailing, leading (a `[1, 1, T, T]` mask against `[B, H, T, T]`)
  and middle axis (`[1, C, 1, 1]` against NCHW).
- **Shapes are static.** `shapes={input_name: tuple}` resolves every
  symbolic axis; a graph input with an unresolved axis is refused by name
  with the axes listed. `Program.run` refuses an array of another shape.

Dtypes, all on the device through the cast path (core/cast_in.mojo,
autograd/cast_ops.mojo, the ag-weight-import additions):

| Source element type | Initializer (weight) | Graph input at `run` | Rule |
|---|---|---|---|
| float32 | uploaded bit for bit | `CastIn` float32 | the storage type |
| float16 | `f16_to_f32_bits` on the device (exact) | `CastIn` NumPy 23 (exact) | widened, never rounded |
| bfloat16 | the bf16 shift on the device (ag-weight-import; exact) | as uint16 bits with the bf16 code | widened, never rounded |
| float64 | `OP_CAST_F32` on the device (rounds to nearest even once) | `CastIn` float64 | the one place a bit is dropped, stated in the docstring |
| int64 / int32 indices (Gather ids, embedding ids) | `OP_CAST_I32` on the device, then `OP_EMB_IDS_CHECK` range check | int inputs through the integer `CastIn` | values above int32 refused by the cast's own check |
| int64 shape constants (Reshape, Slice, axes) | read in Python as description integers | n/a | never uploaded |
| bool (masks, Where conditions) | uploaded as 0/1 float32 words through `CAST_KIND_BOOL` | `CastIn` bool | L3 `where` reads 0/1 |
| int8 / uint8 / float8 / int4 weights | refused by name | refused | the dequantize ops are refused (3.4) |
| string, complex | refused by name | refused | - |

External data (`TensorProto.data_location == EXTERNAL`, `location`,
`offset`, `length`): the reader memory-maps the named file beside the model
and hands the slice, no copy before the device upload. A sharded `.onnx`
with external weights is the common case for anything above 2 GB.

Outputs: float32 arrays of the recorded shapes; an output the graph marks
int64 (ArgMax) comes back as int32 words (the `argmax` convention of the
session), stated in the docstring.

## 5. Identity, and torch as a reference

Identity needs no importer work: the Program replays tape nodes whose
kernels are the ones the autograd check suites hold equal on CPU, Metal,
NVIDIA and AMD (blocked folds with one declared order per reduction,
`identical_*` elementwise numerics, the GEMM chain `k` ascending, the
order-free max). An imported model's output hash is therefore the same on
the three vendors and on the host executor by construction, and the checks
only confirm it: every import test prints `IMPORT_DIGEST <fixture>
<digest>` and the orchestrator diffs the lines across the four executors.
The L3 family adds kinds under the same rules (one thread per word,
`identical_*`, ftz) and its own `IMPORT` digest lines.

The source framework's bits are a **reference**, never a gate: torch and
onnxruntime sum in a different order, fuse differently, and use their own
`erf`/`exp`/`tanh`. tools/import_compare.py (lane 5; torch or onnxruntime
optional, never imported by the package or its tests) runs the source
model and the imported Program on the same inputs and prints, per output,
`max |diff|`, `max rel diff`, the count of words beyond 1e-5 relative, and
the first divergent index; for a classifier also the argmax agreement
rate. The numbers go into the lane's final reply and into
docs/apple-fast/EXPERIMENTS.md only when a speed A/B accompanies them;
they never decide a merge. A closeness result that looks like a bug (a
whole output off by orders of magnitude, a wrong class on every row) is a
lowering error to fix (standing order: fix what is found), not a tolerance
to widen.

Quality on the inference board (lane ag-inference-board, `program-run`):
the imported Program's accuracy or perplexity is shown beside the torch
twin's on the same checkpoint, time beside time, never as a gate.

### 5.1 Torch twins (inventory, lane ag-torch-twins, 2026-10-20)

Every public surface that claims closeness to torch or reads from it, what
compares it against torch, and where that runs. None is a gate: torch's bits
are a reference. Identity is decided by the digest lines across CPU, Metal,
NVIDIA and AMD. The Mac test environment (the default pixi environment) has
no torch, so every torch test there skips with `no torch in this
environment (<error>)`; the `skgpu` pixi environment and the board venvs
have torch, and box runs set `MOJOLEARN_REQUIRE_TORCH=1`, which turns that
skip into a failure.

| surface | compared against torch by | runs where | role |
|---|---|---|---|
| `import_model(ExportedProgram)` | python/mojolearn/tests/test_import_torch.py: the reader on every shared fixture; the torch.export import equals the ONNX import word for word; `IMPORT_TORCH_REFERENCE` max abs/rel to torch's forward, printed | Mac (skips without torch), boxes with torch | reference (torch distance printed); `IMPORT_DIGEST torch_*` is the identity line |
| `import_model(ExportedProgram)`, does it start | tools/autograd_smoke.py `import-torch-smoke`: linear, layer norm, linear exported and run once; `RUN torch <v>` + `SMOKE_DIGEST`, or `SKIPPED no torch` | Apple smoke set (tools/apple_smoke_set.py), NVIDIA/AMD gate smoke (tools/gate_smoke_box.py) | smoke only, no torch comparison |
| `import_model` closeness report | tools/import_compare.py `--torch FILE.py` | on demand, anywhere torch is installed | reference report |
| `Sequential.load_state_dict(module.state_dict())`, `Session.load_weights(mapping)` | python/mojolearn/tests/test_torch_state_dict.py (added by this lane): a real torch state_dict in float32/bfloat16/float16 loads to torch's words exactly; `TORCH_STATE_DICT_REFERENCE` forward distance, printed. python/mojolearn/tests/test_weight_import.py covers the same entries torch-free (NumPy mappings with `source='torch'`) | Mac (skips without torch), boxes with torch | reference; `TORCH_STATE_DICT_DIGEST` is the identity line |
| board `sequential-fit`, `sequential-predict` | tools/bench_board_neural.py TorchFitArm: the same torch.nn network, torch.optim.Adam, same sizes, batch, lr and seed | board boxes (NVIDIA, AMD, Apple); opponents scored once | reference: quality beside time |
| board `ops-session-step` | the same step in torch autograd + torch.optim.SGD, same weights; each gradient's max difference from ours | board boxes | reference |
| board `program-run` | the same forward in torch under `no_grad`; max error from a float64 forward and from ours | board boxes | reference |
| board `mlp-train-step`, `mlp-predict`, `lm-*`, `transformer-*`, `mamba*-forward`, `samba-*`, `gemm` | the twins of docs/BENCH_BOARD.md's neural table (`F.linear` MLP, `torch_lm_step_opponent.py`, `LlamaEager`, the Mamba references) | board boxes | reference (the Mamba ones against pure-PyTorch references) |
| board RNN/LSTM/GRU/CNN fit lanes | tools/bench_board_algos.py torch twins starting from ours' exported initial weights and row orders | board boxes | reference |
| `CausalLM` (`causal-lm-generate*`) | transformers `LlamaForCausalLM.generate` on the same checkpoint; new ids equal or first divergence | board boxes | reference |

## 6. The Python surface (user voice)

```python
import numpy as np, mojolearn

prog = mojolearn.import_model("resnet_small.onnx", shapes={"input": (8, 3, 32, 32)})
prog.inputs          # [("input", (8, 3, 32, 32), "float32")]
prog.outputs         # [("logits", (8, 10), "float32")]
logits = prog.run(x)          # x: an array of the input's shape; any float dtype, used as float32
prog.save("resnet_small.mlprog")

served = mojolearn.ops.Program.load("resnet_small.mlprog")   # any install, GPU or CPU
served.run(x)
```

`mojolearn.import_model(source, *, shapes=None, numeric_mode=None)`
(python/mojolearn/importers/__init__.py, exported lazily from
python/mojolearn/__init__.py as `ops` is):

> Import a trained model for inference. ``source`` is the path of an ONNX
> file (opset 13 or later) or a ``torch.export.ExportedProgram``. The
> model's weights are copied to the device once and its operations run as
> one call per :meth:`Program.run`, with the same results on Apple, NVIDIA
> and AMD GPUs and on the CPU. ``shapes`` fixes the size of any input axis
> the model leaves open (such as the batch), by input name; the imported
> model then accepts arrays of exactly those shapes. ``numeric_mode`` is
> ``'identical'`` (the default) or ``'fast'`` (Apple GPUs only). Raises
> :class:`UnsupportedModel` listing every operation the import does not
> support, by node name.

`mojolearn.importers.supported_ops()`: the table of section 3.1 as a list
of `(op, supported_subset)` rows, so a user can check a model's ops before
exporting; `mojolearn.importers.UnsupportedModel`.

`Program` is lane ag-program's object; the importer gives it `inputs` and
`outputs` (names, shapes, dtypes of the source graph), keeps the owning
`Session` alive on the Program, and uses its `run` and `save` unchanged.
`Program.load` is ag-program's. Docstrings carry no status notes; until the
`IMPORT_DIGEST` lines match on the three vendors `import_model`,
`importers` and `Program` stay in docs-site/api_exclude.txt (ag-program
lists `Program`; the docs lane removes the lines when the orchestrator
says the digests are green).

Not offered: a `fit` on an imported model (inference only; a recording
import is a later plan once ag-program's recording Program exists),
`fast` on NVIDIA, AMD or the CPU, any numpy fallback when the binding is
missing.

## 7. The GraphIR the lanes code against

Lanes never compile or run, so the interface is fixed here
(python/mojolearn/importers/_graph.py, lane 1 owns the file; lanes 2 and 4
import from it):

```python
@dataclass(frozen=True)
class TensorInfo:          # an initializer or a graph value with data
    name: str
    dtype: str             # "f32" "f16" "bf16" "f64" "i64" "i32" "bool" (others refused by the reader)
    shape: tuple[int, ...]
    data: memoryview | None    # raw little-endian bytes, row-major; None for a graph value

@dataclass(frozen=True)
class Node:
    op: str                # ONNX op name ("Gemm"); the torch reader maps aten names onto these
    domain: str            # "" for the default domain
    name: str              # the node name, for messages
    inputs: tuple[str, ...]    # value names; "" for an absent optional input
    outputs: tuple[str, ...]
    attrs: dict[str, int | float | str | tuple[int, ...] | tuple[float, ...] | TensorInfo]

@dataclass
class GraphIR:
    opset: int
    inputs: tuple[tuple[str, str, tuple[int | str, ...]], ...]   # (name, dtype, shape with symbolic axes as str)
    outputs: tuple[tuple[str, str, tuple[int | str, ...]], ...]
    nodes: tuple[Node, ...]                 # topological order
    initializers: dict[str, TensorInfo]
    source: str                             # path or "torch.export", for messages

def resolve_shapes(g: GraphIR, shapes: dict[str, tuple[int, ...]]) -> dict[str, tuple[int, ...]]:
    """Every value's static shape from the inputs' resolved shapes; integer
    shape subgraphs folded; raises UnsupportedModel on an unresolved axis or
    a data-dependent shape, by node name."""
```

The lowering (lane 2) exposes `lower(g: GraphIR, value_shapes, session) ->
Lowered(input_handles, output_handles, parameter_handles)` and
`import_model` wraps it with `Session` and `capture`. The ONNX reader (lane
1) returns a `GraphIR`; the torch.export reader (lane 4) returns a
`GraphIR`; neither knows the session.

Reading a model file is an input step before the runtime: the reader's
file handling is tagged `# cpu-route: model file parsed on the host` at
its entry (the hook's class for text and file handling). It never
computes on element values; the only bytes it touches are copied by the
device upload.

## 8. Lanes

Five Opus lanes from main after ag-program and ag-weight-import merge (lane
1 and lane 3 can start now; they touch neither). Each lane: its own
primary files, its own tests, no compile, commits after every edit, the
_common.md rules. Shared files are append-only at distinct anchors:
sequence/dispatch.mojo (lane 3 only), autograd/ops_api.mojo and
python/mojolearn/ops.py (lane 3 only), python/mojolearn/__init__.py and
docs-site/api_exclude.txt (lane 5 only), tools/autograd_smoke.py and
tools/check_autograd.py (lane 5 only).

### Lane 1: ag-import-onnx (reader and IR; merges first, with lane 3)

(Lane ag-native-onnx-reader, 2026-10-20: the reader and the static shape resolution of a read
graph now run in Mojo, autograd/import_proto.mojo, import_reader.mojo and import_resolve.mojo,
through the x_sequence binding entries onnx_read / onnx_info / onnx_copy_blob / onnx_resolve /
onnx_close; importers/onnx.py only wraps the tables in the GraphIR classes. The Python reader
below is the test oracle, python/mojolearn/tests/_onnx_reader_oracle.py and
_onnx_proto_oracle.py. Step 2 moves _lower.py into Mojo against this graph.)

(Lane ag-native-lowering, 2026-10-20, step 2: the lowering runs in Mojo too,
autograd/import_lower.mojo, through the x_sequence entry onnx_import(handle, session, names, flat,
buffer addresses, source): shape resolution, the refusal pass (the same texts, in the same order,
before any upload), the uploads through the cast and weight-import paths, the import-time
prologue and its persists, one session call per operation in the Python lowering's order, the
capture and the release of unread uploads. importers.import_model is read + one onnx_import call
+ the ImportedProgram wrapper. A GraphIR built in Python (torch.export) goes to Mojo as onnx_info's
words through onnx_from_words and takes the same lowering. The Python lowering is the test oracle,
python/mojolearn/tests/_lower_oracle.py; tests/test_import_lower_native.py asserts both give the
same refusals, the same saved program bytes and the same run outputs. The op table is
importers/_table.py.)

- Files: python/mojolearn/importers/__init__.py (the package, `UnsupportedModel`,
  `supported_ops` placeholder reading lane 2's table lazily),
  python/mojolearn/importers/_graph.py (section 7), python/mojolearn/importers/_onnx_proto.py
  (a minimal protobuf wire reader for ModelProto, GraphProto, NodeProto,
  AttributeProto, TensorProto incl. external data, ValueInfoProto,
  TypeProto, OperatorSetIdProto; varints and length-delimited fields only;
  no `onnx`, no `protobuf` dependency; unknown fields skipped),
  python/mojolearn/importers/onnx.py (`read(path) -> GraphIR`, topological
  sort, dtype refusals by name), python/mojolearn/tests/_onnx_fixture.py
  (the test-side encoder: `make_model(nodes, inputs, outputs,
  initializers, opset)` writing a valid `.onnx`), python/mojolearn/tests/test_import_onnx.py.
- Checks: round trip of every fixture the other lanes use (linear + layernorm +
  linear; conv + relu + maxpool + flatten + gemm; a two-head attention
  block with a constant causal mask; batchnorm + relu; a sharded external
  data file); `resolve_shapes` on a dynamic batch; refusals: opset 12 by
  version, an `ai.onnx.ml` node by domain, a float8 initializer by name, an
  unresolved axis by input name. Pure Python, runs on every install.

### Lane 2: ag-import-lower (the table; merges after lane 1, lane 3, ag-program, ag-weight-import)

- Files: python/mojolearn/importers/_lower.py (the `LOWER` table of 3.1,
  the prologue/body split, the refusal pass collecting every unsupported
  node before any upload, the weight upload through ag-weight-import's
  cast path and layout codes, `capture`), python/mojolearn/importers/_program.py
  (`import_model`, the Program's `inputs`/`outputs`, session ownership),
  python/mojolearn/tests/test_import_lower.py.
- Codes against: ag-program's `Session.capture` / `Program.run` / `save`
  as its brief states them; ag-weight-import's cast and layout entry; lane
  3's session methods by the names in 3.3 (`erf`, `gelu`, `clip`, `where`,
  `expand`, `argmax`, `pad`, ...); lane 1's `GraphIR`.
- Checks (`IMPORT_DIGEST <fixture> <digest>` lines): each fixture of lane 1
  imported, run on random inputs with a fixed seed, and compared **word for
  word** with an eager `ops.Session` composition of the same operations on
  the same uploaded weights (the reference forward); repeat-run equality;
  `Program.save` then `Program.load` runs equal; the BatchNorm prologue
  equals the eager fold; a Gemm `transB` weight equals the eager
  `transpose`; refusals by node name (`Resize`, `Conv group=2`, a
  data-dependent `Reshape`), each listing every offending node in one
  error. Digests must be equal on CPU, Metal, NVIDIA and AMD.

### Lane 3: ag-import-kernels (the elementwise family; merges first, with lane 1)

- Files: autograd/import_ops.mojo (bodies 4600..4605, `OP_IMP_FIRST`/`OP_IMP_LAST`
  4600..4619, `apply_import`), autograd/import_api.mojo (tape calls,
  `ImportRules`, tags 192..207 as `IMP_TAG_FIRST`/`IMP_TAG_LAST`, the
  `expand` view through `OP_RN_COPY_STRIDED`, `argmax` over the fold
  lane's max partials), autograd/checks/import_cases.mojo +
  import_cpu.mojo + import_gpu.mojo; append-only anchors in
  sequence/dispatch.mojo (import, fence asserts against every earlier
  range, `elif` branch), autograd/ops_api.mojo (codes 48.. and their
  `elif` arms: `OPS_ERF` 48, `OPS_GELU` 49, `OPS_SOFTPLUS` 50, `OPS_SILU`
  51, `OPS_LEAKY_RELU` 52, `OPS_ELU` 53, `OPS_HARD_SIGMOID` 54,
  `OPS_HARD_SWISH` 55, `OPS_CLIP` 56, `OPS_ABS` 57, `OPS_FLOOR` 58,
  `OPS_CEIL` 59, `OPS_ROUND` 60, `OPS_SIN` 61, `OPS_COS` 62, `OPS_SIGN` 63,
  `OPS_RECIPROCAL` 64, `OPS_POW` 65, `OPS_MINIMUM` 66, `OPS_MAXIMUM` 67,
  `OPS_WHERE` 68, `OPS_COMPARE` 69 (kind: eq, gt, lt, ge, le, ne),
  `OPS_LOGICAL` 70 (kind: not, and, or), `OPS_EXPAND` 71, `OPS_ARGMAX` 72
  (47 stays reserved as the rank-N plan left it), `OPS_PAD` 73), and
  python/mojolearn/ops.py (one method per code, user-voice docstrings).
- Checks: `IMPORT <kind> <case> digest <d>` lines on odd shapes, forward
  and gradient per kind, `where` with a broadcast condition, `expand` on a
  middle axis equal to `mul(full(1), x)` word for word, `argmax` ties and
  NaN; python/mojolearn/tests/test_ops_import_kinds.py with `OPS_IMPORT_DIGEST`
  lines. Bindings the orchestrator builds: `bindings/_mojolearn_x_sequence.mojo`
  and `_mojolearn_x_sequence_host.mojo` (the only bindings importing
  autograd/ops_api.mojo), plus the two check mains.

### Lane 4: ag-import-torch (merges after lane 2)

- Files: python/mojolearn/importers/torch_export.py (`read(ep) -> GraphIR`:
  `ep.run_decompositions()`, walk `ep.graph_module.graph` in order, map
  `ep.graph_signature` to inputs/parameters/buffers/outputs, aten targets
  to IR ops by a table (`aten.mm/bmm/matmul` MatMul, `aten.addmm` Gemm,
  `aten.convolution` Conv with the subset checks, `aten.native_layer_norm`
  LayerNormalization (first output), `aten._softmax` Softmax,
  `aten.view/reshape/_unsafe_view` Reshape, `aten.permute/transpose/t`
  Transpose, `aten.cat` Concat, `aten.split_with_sizes/slice` Split/Slice,
  `aten.embedding` Gather axis 0, `aten.expand` Expand, `aten.sum/mean/amax`
  Reduce*, `aten.max_pool2d_with_indices` MaxPool, `aten.avg_pool2d`
  AveragePool, `aten._native_batch_norm_legit_no_training`
  BatchNormalization, `aten.gelu/erf/silu/...` the L3 kinds,
  `aten.clone/alias/detach` Identity, `sym_size` and friends to shape
  folding; everything else refused by the aten name), tensors from
  `ep.state_dict` and `ep.constants` as buffers (`detach().cpu()` then the
  memory as bytes; bfloat16 via `view(torch.int16)` bits with the bf16
  code; no arithmetic), `import_model` dispatching on the object type.
  python/mojolearn/tests/test_import_torch.py, skipped unless torch
  imports.
- Checks: the same fixtures built as `nn.Module`s exported with
  `torch.export.export`, imported, equal word for word to the ONNX import
  of the same weights (lane 1's encoder writes the ONNX twin from the
  state_dict) and to the eager reference; `IMPORT_DIGEST` lines equal
  across the four executors on a box with torch. The torch reference
  closeness is printed, not asserted.

### Lane 5: ag-import-docs (merges last)

- Files: docs-site/user_guide/autograd.md section "Import a model" with a
  named block `<!-- block: import -->` (the test writes the fixture file
  the block names, as `test_guide_export_and_load` does for `export`),
  python/mojolearn/tests/test_autograd_public.py (the block's test and its
  `PUBLIC_DIGEST`), python/mojolearn/__init__.py (`import_model`,
  `importers` lazy exports), docs-site/api_exclude.txt (`import_model`,
  `importers`, `UnsupportedModel`, the L3 session methods until their
  digests are green), tools/import_compare.py (section 5),
  tools/autograd_smoke.py `probe_import` (5 s, does it start, fixture
  written in process), tools/check_autograd.py (the `import_{cpu,gpu}`
  pair), docs/plans/AUTOGRAD_API.md one paragraph linking here.

### Merge order and what the orchestrator builds

1. Lane 1 (pure Python; `pytest python/mojolearn/tests/test_import_onnx.py`
   on the laptop's CPU install is enough) and lane 3 (compile the sequence
   binding and its host twin plus `autograd/checks/import_{cpu,gpu}.mojo`
   with `-D MOJOLEARN_NUMERIC_IDENTICAL=1`; run on CPU, Metal, NVIDIA, AMD;
   diff the `IMPORT` and `OPS_IMPORT_DIGEST` lines).
2. ag-program and ag-weight-import (their own briefs).
3. Lane 2: no Mojo; run `test_import_lower.py` on the four executors and
   diff `IMPORT_DIGEST`.
4. Lane 4 (on a box with torch: nv or amd through `lq add <box> CMD`) and
   lane 5 (pytest on the four, `test_autograd_public.py` block `import`),
   in any order.

## 9. Registry

| Range or block | Family | Lane |
|---|---|---|
| 4600..4619 | `OP_IMP_*` elementwise kinds, argmax, pad (4600..4605 used, the rest reserved) | 3 |
| custom tags 192..207 | `IMP_TAG_*` (autograd/rankn.mojo `RN_TAG_END` is 192) | 3 |
| session codes 48..73 | the L3 methods (47 stays reserved) | 3 |
| checkpoint section kind 3 | the Program (ag-program; the importer adds no kind) | ag-program |
| python/mojolearn/importers/ | the package | 1, 2, 4 |

## 10. Out of scope, owed, and the decisions taken

- **Dynamic shapes and KV-cache decode.** A Program is fixed-shape. A
  generate loop needs one Program per sequence length or a
  shape-polymorphic Program (ag-program's follow-up); the importer then
  imports once per shape set. Until then `CausalLM` stays the way to run a
  decoder with a cache.
- **Training an imported model** waits for a recording Program (the `IF`/
  `LOOP` design note in AUTOGRAD_API.md). Every L3 kind ships its VJP so
  nothing in this plan blocks it.
- **Fused Microsoft and opset-23 attention ops**: a follow-up table-only
  lane (3.2), after one real exported model asks for them.
- **Quantized models** (`lowbit` weights, Q/DQ ops): a later lane that
  dequantizes in the prologue and, for int8 linears, reuses
  `mojolearn.lowbit`'s packed path.
- **`torch.export.save` files** (zip with a torch-pickled state_dict): not
  read; the in-process object is the path. Reading the pickle without torch
  would mean an unpickler of our own.
- **TorchScript, `nn.Module` tracing, TensorFlow SavedModel, JAX**: not
  planned; every one of them reaches ONNX today.
- **Speed A/Bs owed once something imports:** a fused `silu` kind versus
  `mul(x, sigmoid(x))` (launch count), a fused `gelu` versus the erf
  composition, the Program's intermediate buffer reuse (ag-program). Each
  is a `-D MOJOLEARN_AUTOGRAD_FAST_<NAME>` define with one row in
  docs/apple-fast/EXPERIMENTS.md.
- **Decided here, not asked:** ONNX's names as the IR vocabulary (one
  table for two readers); a protobuf reader of our own instead of the
  `onnx` dependency (the format is five message types and the package pins
  protobuf and numpy); opset floor 13; one error listing every unsupported
  node; a Python-built 0/1 mask constant counts as description, with L3
  `where` on an uploaded index pattern as the fallback if reviewed
  otherwise; `expand` through the existing strided copy instead of a
  `mul(full(1), x)` composition (no new words for a view); the elementwise
  family in its own file and opcode range rather than more kinds in
  autograd/ops.mojo (no lane edits the shared tape files).
