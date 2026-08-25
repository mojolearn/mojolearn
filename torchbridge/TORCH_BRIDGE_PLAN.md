# The PyTorch bridge for `mojolearn.identical.gemm.fp32.v1`

**NOTHING IN THIS DIRECTORY HAS EVER BEEN COMPILED OR EXECUTED.** Not the
Mojo file, not the Python file, not one line of either. This document, and
the two source files it describes, are a design and a first implementation
written from a source read. Every claim below that is a MEASUREMENT is
attributed to the lane that measured it; everything else is construction, and
where the construction rests on an API whose exact spelling this lane could
not verify from the pinned environment, the file says so in those words at
the point of use.

DEVIATIONS 1200 through 1213. Written 2026-08-25.

Owed, in one list, with the long form in the last section of this document.

- Nothing here has been compiled. The first thing anyone does with these
  files is find out whether they compile, and the second is run gate T7.
- `python/mojolearn/identical/_mojolearn_linalg.so` is not built in this
  tree, so the HOST lane cannot run in identical mode today.
- The `.mojopkg` build step for the CUSTOM OP lane does not exist, and
  without it that lane compiles the FAST arithmetic. Gate T7 exists to
  make that loud rather than silent.
- Gates T1 through T9 are specified in section 8 and NONE of them is
  written.

---

## 0. What this bridge is for, and the one sentence version

This repository has a measured property. One Mojo source produces
byte-identical FP32 output on Apple Metal, NVIDIA CUDA and AMD HIP, for a
GEMM (60 card stages, three vendors, `gemm/README.md` and `E3_RESULTS.md`
round 11) and for a Mamba-1 block. **None of it is reachable from PyTorch.**
There is no torch bridge anywhere in the tree.

The bridge makes exactly this true and nothing more.

> A PyTorch user can call `mojolearn.torch_ops.identical_matmul(a, b)` inside
> a normal training loop, on an M4, an H100 or an MI325X, and the output
> tensor holds the same bits on all three. So does the gradient with respect
> to each operand, at a fixed token count.

What it does not make true is in section 6, and section 6 is not optional
reading. Contract section 11 is the authority and this bridge may not promise
more than it does.

---

## 1. The two lanes, and why there are two

**DEVIATION 1200.** The bridge has TWO transport lanes to the same certified
kernel, they are selected by the device the caller's tensor is on, and the
lane that ran is REPORTED BACK rather than inferred.

    lane          torch device      transport                    copies
    CUSTOM_OP     cuda (incl. ROCm) max.experimental.torch,       3 device
                                    dlpack, device resident       to device
                                                                  (see 2.2)
    HOST          cpu, mps          numpy -> mojolearn.linalg     4 host
                                    -> identical_gemm_host        round trip
                                                                  (see 2.3)

Two lanes is a cost, not a preference. It exists because of a single fact
read out of the pinned environment, and the fact is the most consequential
thing this lane found.

### 1.1 `max.experimental.torch` cannot see an Apple GPU

`.pixi/envs/default/lib/python3.13/site-packages/max/experimental/torch/torch.py`,
lines 672-688, is the whole of the torch-device translation.

```python
def max_device_ref(device: torch.device) -> DeviceRef:
    type = device.type
    index = device.index or 0
    if type == "cpu":
        return DeviceRef.CPU(index)
    elif type == "cuda":
        return DeviceRef.GPU(index)
    else:
        raise TypeError(f"Unable to convert {type} to a MAX device type.")

def max_device(device: torch.device) -> Device:
    DeviceType = {"cuda": Accelerator, "cpu": CPU}[device.type]
    return DeviceType(device.index)
```

There is no `"mps"` arm in either function. A PyTorch tensor on an Apple GPU
lives on `torch.device("mps")`. So:

1. `ops.identical_gemm(out, a, b)` with `mps` tensors raises `TypeError` from
   inside MAX, not from us, with a message that names neither PyTorch nor
   Metal usefully.
2. Moving those tensors to `cpu` first does not fix it. `max_tensor_type`
   (line 690) reads the device off the tensor, `MaxOp.graph` builds the graph
   at that device, and a CPU-typed graph dispatches the custom op with
   `target = "cpu"`. **The Metal GPU is never touched.** The answer would
   still be correct, and under this profile it would still be BIT IDENTICAL
   (that is the whole point of the profile), so no output check anywhere
   could see that the GPU was skipped. See section 7.1; this is the sharpest
   instance in the bridge of "looks right while being wrong".

Therefore the custom-op lane is a CUDA/ROCm lane in the pinned environment,
and Apple reaches the certified kernel only through the host-pointer surface
that already exists (`gemm/host_entry.mojo::identical_gemm_host`, which
constructs its own `DeviceContext()` and does reach Metal, because that is
how `python/mojolearn/_mojolearn_linalg.so` runs today).

PyTorch built for ROCm reports `device.type == "cuda"`, so AMD is on the
custom-op lane with NVIDIA and no `"hip"` arm is needed. That is an inherited
PyTorch convention and it is the one claim in this subsection this lane did
NOT verify against a ROCm install; gate T8 finds out.

### 1.2 The consequence for the three-vendor card, stated before anyone runs one

**Apple runs a different lane from NVIDIA and AMD.** A three-vendor card built
by running "the bridge" on three boxes would be comparing HOST lane on Apple
against CUSTOM_OP lane on the other two, and a pass would then be evidence
about the kernel (which is already certified) rather than about the bridge.

So the card runs the HOST lane on all three vendors as its spine, and the
CUSTOM_OP lane on NVIDIA and AMD as a second row compared against the same
spine. One lane common to three vendors, one vendor common to two lanes. Gate
T8 is written that way and says so. Dropping the CPU-lane row on NVIDIA
because "it obviously agrees" is the exact shortcut that would make the card
decorative.

---

## 2. The memory boundary

### 2.1 What the kernel wants

`gemm/mojo_only/gemm_identical.mojo::identical_gemm_into(ctx, c, a, b, ws, m,
n, k, op)` takes `DeviceBuffer[DType.float32]` for all four buffers, is
ASYNCHRONOUS, and requires a caller-owned workspace of at least
`identical_gemm_workspace_max_floats(m, n, k)` floats. Its docstring records
what guessing that number cost this repository once already -- a one-float
workspace against a SPLITK dispatch returned right answers at 64 x 4 and
regions of `+0.0` at 64 x 64.

`gemm/host_entry.mojo::identical_gemm_host(ctx, c_ptr, a_ptr, b_ptr, m, n, k,
op)` takes raw HOST pointers, allocates and copies both ways, calls the
synchronizing `identical_gemm` (which sizes its own workspace), and waits.

### 2.2 CUSTOM_OP lane -- the pointer crosses, and then we stage it anyway

Two halves, and the second is the one that disappoints.

**Torch to MAX is a borrow.** `MaxOp.custom_op_def`'s `callable` (torch.py
line 517) converts every argument with `Buffer.from_dlpack(t)`, or on CUDA
with the faster `Buffer._from_dlpack(data, device, stream)` path at line 502.
dlpack does not copy. The tensor's own allocation is what the graph sees.

**MAX to our kernel is a copy, three of them, and this lane could not avoid
it.** `identical_gemm_into` and `identical_gemm` take
`DeviceBuffer[DType.float32]`, and `DeviceBuffer` has **no public constructor
that wraps a foreign device pointer**. Its documented construction routes are
`DeviceContext.enqueue_create_buffer`, `create_buffer_sync`, the reference
copy constructor and `create_sub_buffer`, and that is the whole list
(`max/gpu/host/device_context/DeviceBuffer`). A custom op receives tensor
arguments, not `DeviceBuffer`s. So `torchbridge/identical_ops.mojo`

1. allocates three `DeviceBuffer`s with `ctx.enqueue_create_buffer`,
2. fills two of them with the DEVICE-TO-DEVICE overload
   `ctx.enqueue_copy(dst_ptr, src_ptr, size)` -- the one documented as
   *"an async copy of `size` elements from a device pointer to another device
   pointer"*, no host round trip,
3. calls the certified `identical_gemm`,
4. and copies the output buffer back to the tensor's own memory the same way.

That is `m*k + n*k + m*n` floats of device-to-device traffic per call, on top
of the kernel's own. It is bandwidth, not correctness, and it does not touch
one bit of the arithmetic. **Removing it needs a pointer-taking entry point in
`gemm/`, which is not this lane's directory.** Section 10 item 11.

The alternative -- reimplementing the launch inside the custom op against raw
pointers -- is refused outright. That would be a second implementation of a
bit-exact contract, which is the one thing `gemm/host_entry.mojo` says in its
own header that this family cannot afford.

Consequences the bridge has to honor, each of which is a rule below.

- **The output tensor is destination-passing.** `torch.library.custom_op(...,
  mutates_args=mutated_args)` at torch.py line 529, with `mutated_args` being
  the first `num_outputs` parameter names (line 493). The caller allocates
  `out` and the op writes into it. `torch_ops.py` allocates it with
  `a.new_empty((m, n))` so device, dtype and stream ownership stay PyTorch's.
- **Strides are dropped on the floor.** `max_tensor_type` (line 690) builds
  `TensorType(dtype, shape, device=device)`. There is NO stride in a
  `TensorType`. A non-contiguous torch tensor handed across this boundary is
  read as if it were contiguous. That is not a slow path, it is a WRONG
  ANSWER, and it is why section 3 refuses rather than repairs.
- **The workspace.** `identical_gemm_into` needs one and a custom op has no
  caller to own it. `torchbridge/identical_ops.mojo` therefore calls the
  SYNCHRONIZING `identical_gemm`, which allocates, sizes and frees its own
  workspace and waits before returning. That costs a synchronize per matmul
  and it is the correct first version -- an asynchronous op that allocated a
  workspace inside itself would hit `[[mojo-buffer-freed-at-last-use]]`
  exactly as `identical_gemm`'s own docstring describes, and would free the
  workspace before the kernel ran. Making the op asynchronous is a real
  optimization and it is OWED, not skipped for lack of interest.
- **The staged buffers must outlive the wait.**
  `[[mojo-buffer-freed-at-last-use]]` -- a `DeviceBuffer` is dead at its
  `.unsafe_ptr()`, and staging through `ctx.enqueue_copy(dst_ptr, src_ptr,
  size)` takes `.unsafe_ptr()` on every one of them. `identical_ops.mojo`
  therefore carries `_ = a` / `_ = b` / `_ = c` lines after the final
  `ctx.synchronize()`, exactly as `gemm/host_entry.mojo` does and for exactly
  that reason. A small fixture cannot see this hazard, which is why it is
  written down rather than discovered.

### 2.3 HOST lane -- four copies, counted out loud

`torch tensor -> numpy -> device -> compute -> device -> numpy -> torch`.

    1. t.detach().cpu().numpy()          device to host   (0 on a cpu tensor)
    2. enqueue_copy host to device       host to device   (host_entry.mojo)
    3. enqueue_copy device to host       device to host   (host_entry.mojo)
    4. torch.from_numpy(...).to(device)  host to device   (0 on a cpu tensor)

Four for an `mps` tensor, two for a `cpu` tensor. This is the price of the
Apple lane in the pinned environment and it is not hidden anywhere. Any
performance number quoted for this bridge must name the lane; the two lanes
are not comparable and a mixed average of them is meaningless.

`_linalg_impl.matmul` is the entry, not a second front door of our own. Its
guards, its refusals, its `require_identical()` and its op mapping are
reused as they stand. **The bridge adds no numerical surface. If it ever
needs to, that is a change to `_linalg_impl.py`, in that file, with that
file's gates.**

### 2.4 What the bridge does NOT do to your memory

- It never calls `.contiguous()` for you. DEVIATION 1201, section 3.1.
- It never calls `.float()`, `.to(torch.float32)` or `.double()` for you.
  DEVIATION 1202, section 4.
- It never moves a tensor between devices for you on the CUSTOM_OP lane. On
  the HOST lane it must (that lane IS a host round trip), and it says so in
  the returned metadata rather than in a docstring nobody reads.
- It never retains a pointer past the call. On the CUSTOM_OP lane MAX owns
  that; on the HOST lane the numpy arrays are held in locals across the call
  exactly as `_arrays.py` requires.
- It DOES stage three device buffers inside the custom op, per 2.2. That is
  the one copy in this bridge that is neither refused nor optional, and it is
  in the table in section 1 rather than in a footnote.

---

## 3. Layout, strides and contiguity

Contract section 2 is unconditional. *"Every operand and the output are
row-major and fully contiguous. No leading dimension, no stride, no offset,
no sub-matrix view."*

### 3.1 A non-contiguous input is REFUSED, not repaired. DEVIATION 1201.

The alternative is one line, `a = a.contiguous()`, and it is the single worst
line this bridge could contain.

- It is a full copy of the operand, allocated on the caller's device, that
  the caller did not write and cannot see. On a 4096 x 4096 fp32 activation
  that is 64 MB per call.
- It produces the RIGHT ANSWER. Every numerical gate in this repository would
  pass with it in place. There is no output any check can look at that
  distinguishes "the user passed contiguous memory" from "the bridge quietly
  made some". A performance claim measured with that line in the source is a
  claim about a different program than the one the user thinks they wrote.
- It has no upper bound on how often it fires. A user who transposes in a
  loop pays for it every iteration and sees nothing.

So `identical_matmul` raises `ValueError` naming the operand, its shape, its
strides, and the exact expression that fixes it. A caller who wants the copy
writes `a.contiguous()` in their own source, where they can price it.

`allow_copy=True` is offered as an explicit keyword. Passing it is a
statement. Omitting it is not.

### 3.2 What is already contiguous and needs nothing

`F.linear(x, W)` computes `x @ W.T` with `x` of shape `(tokens, in_features)`
and `W` of shape `(out_features, in_features)`, both C-contiguous as PyTorch
stores them. That is `OP_NT` with both operands passed exactly as stored.
**The most common linear layer in PyTorch needs no copy and no transpose**,
which is the same saving contract section 0.2 counts for the native `OP_TN`
arm. This is worth stating because it is the reason refusing in 3.1 is cheap
rather than obstructive.

### 3.3 The output

`out` is allocated by the bridge with `a.new_empty((m, n))`, which is
C-contiguous, float32 and on `a`'s device. A user-supplied `out=` is accepted
and checked -- float32, contiguous, right shape, right device, not a view
that aliases either input. It is never made contiguous for the caller, for
3.1's reason plus one more -- copying into a non-contiguous `out` after the
fact would make `out` a lie about where the result was produced, which is
`_linalg_impl.matmul`'s own argument for the same refusal.

---

## 4. dtype policy. Refuse, do not cast. DEVIATION 1202.

The profile is FP32 and only FP32. Contract section 1 makes that a hard
requirement and section 0.5 excludes FP16, BF16, TF32 and float64 by name.

    torch.float32   accepted
    torch.float16   REFUSED by name
    torch.bfloat16  REFUSED by name
    torch.float64   REFUSED by name
    everything else REFUSED by name

Casting would be worse than refusing in both directions and the reasons
differ, so both are stated.

- **Down** from float64 or up from float16, the cast changes the input bits
  before the profile ever sees them. The product this module sells is the
  caller's control over exactly which bits go in. A cast performed inside the
  bridge takes that control away and returns a number that is identical
  across vendors and is not the answer to the question the caller asked.
- **A mixed-precision training loop is the likely caller**, and under
  `torch.autocast` a linear layer's inputs arrive as bfloat16 without the
  user having typed anything. Silently upcasting them to fp32 would make the
  bridge appear to work inside `autocast` while running an arithmetic the
  surrounding model does not use, at 2x the memory traffic, invisibly. The
  refusal message says `autocast` by name for exactly this reason and tells
  the caller to wrap the call in `torch.autocast(enabled=False)` if fp32 is
  what they want here.

`_linalg_impl._operand` makes the same call for the same reason and this is
deliberately the same policy in the same words.

---

## 5. Device policy

    torch device        CUSTOM_OP lane   HOST lane   result
    cuda:N (NVIDIA)     yes              yes         accepted, CUSTOM_OP
    cuda:N (ROCm/AMD)   yes              yes         accepted, CUSTOM_OP
    mps                 NO (see 1.1)     yes         accepted, HOST, 4 copies
    cpu                 compiles, but    yes         accepted, HOST
                        targets the CPU
    anything else       no               no          REFUSED by name

Three rules follow and each is a refusal or a report rather than a fallback.

1. **Both operands must be on the same device.** PyTorch would raise for
   `torch.matmul`; the bridge raises first, with both devices named, because
   a device mismatch that reaches dlpack is a MAX-internal error message.
2. **`mps` is routed to the HOST lane explicitly and the routing is
   reported.** DEVIATION 1204. It is never silently promoted to the
   custom-op lane, because section 1.1 shows that lane would run on the CPU
   and return identical bits while doing so.
3. **A `cpu` tensor is accepted and runs the HOST lane**, which means it
   still runs on the GPU that `DeviceContext()` picks. That is a surprise
   worth documenting -- a CPU tensor in, a GPU computation, a CPU tensor out
   -- and it is the right behavior, because this bridge's product is the GPU
   profile and running an unpinned CPU matmul under the same function name
   would be the FAST-build failure again in a different costume.

The device that actually ran is in the metadata `identical_matmul` can return
(`return_meta=True`) and in `torch_ops.profile()`. **Never inferred from the
input tensor.** See section 7.1 for why that distinction is the whole ball
game here.

---

## 6. What identity buys the caller, stated precisely and no wider

### 6.1 What is bought

For ONE call to `identical_matmul(a, b, transpose_a=..., transpose_b=...)`
with fp32 2-D contiguous operands, in a process that loaded the IDENTICAL
build, the output tensor's bits are a pure function of

- the input bits,
- `k`, the contracted extent,
- and the profile `mojolearn.identical.gemm.fp32.v1`.

Not of `m`, not of `n`, not of the launch geometry, not of the block count,
not of how many rows shared the call, and not of the vendor. That is contract
sections 6, 7 and 0.3, and it is what the 60-stage three-vendor card measured
at leg 11.

For the BACKWARD pass, the same statement holds for `dA`, `dB` and `dbias`
individually, at a fixed shape, by `gemm/mojo_only/gemm_backward.mojo`'s
construction -- that file contains no arithmetic and routes onto the same
forward kernel.

### 6.2 What is NOT bought, itemized

Contract section 11 is the authority. The items that reach a torch caller in
particular.

1. **Not an identical model.** An identical GEMM gives identical linear
   layers. Every norm, softmax, activation, RoPE, residual add, attention
   score and reduction between your layers is untouched by this profile and
   most of them are order dependent. `IDENTICAL_GEMM_PLAN.md` says it as
   "deterministic linear layers, NOT deterministic models".
2. **Not an identical training run.** `gemm/IDENTICAL_BACKWARD_PLAN.md`
   section 4 lists what else identical training needs -- an identical RNG
   (dropout, init, shuffling), an identical optimizer step, identical
   scatter-shaped gradients, identical multi-GPU reduction. None is in scope
   here and none is provided by this bridge.
3. **Not dropout, not data order, not the optimizer.** Named separately from
   item 2 because they are the three a reader assumes are covered.
4. **Not NaN payload bits** (contract 9.1). If your output can contain NaN,
   compare those cells as "is NaN" and not by bits.
5. **Not order independence of what you do next.** A `min`, `max`, `argmin`
   or `topk` taken over this output reintroduces order dependence, because
   `-0.0 == +0.0` compares equal (contract 9.2(e)).
6. **Not a match with `torch.matmul`.** This is a different, pinned summation
   order. It will not equal cuBLAS, MPS or `a @ b` bit for bit, and contract
   7.4 is explicit that sameness rather than accuracy is what is bought.
7. **Not a certified answer at your shape.** The card is 62 shapes and 60
   stages. Outside it the property is CONSTRUCTION -- strong construction,
   contract 0.3 -- and not a measurement. Say so if you quote this in a
   paper.
8. **Not a performance claim.** Contract 11.5, plus section 2.3 of this
   document -- the two lanes are not comparable and this bridge has never
   been run.
9. **Not a `torch.compile` guarantee.** DEVIATION 1209. The op is a real
   kernel launch behind `torch.library.custom_op`; it is opaque to Inductor,
   it will graph-break unless the surrounding code is written for it, and
   nothing in this lane has tested it under `torch.compile`, CUDA graph
   capture, or a non-default stream. The custom-op lane's dlpack path reads
   `torch.cuda.current_stream` (torch.py line 504); what happens under
   capture is unknown and unmeasured.

### 6.3 The gradient-accumulation clause, which is the one a trainer needs

`gemm_backward_b_call` routes `dB` at `k' = m`, the FORWARD's batch
dimension. So **the weight gradient's bits are a function of the token
count**, and contract section 6 requires the leaf size to depend on `k`.

That reads like "microbatching destroys identity" and the backward lane
MEASURED that it does not, provided the split is aligned.

> `T = 512` split 256/256 moved **0 of 35 gradient cells**. `T = 384` split
> 256/128 moved **0 of 35**. `T = 300` split 150/150, `T = 512` split
> 200/312 and `T = 384` split 192/192 moved **31, 30 and 31** cells. `dA`
> moved 0 cells under every split, aligned or not. Host and device agreed.
> (`gemm/IDENTICAL_BACKWARD_PLAN.md` sections 3.2 and 5.2, gate G5,
> 2026-08-25.)

The mechanism, which is why this is a rule and not a coincidence -- v1 holds
the leaf size `L` at `K_LEAF_MIN = 128` for every `k` up to 131,072, and
folds ADJACENT leaves in a fixed balanced binary tree. A split at a token
boundary that is both a LEAF boundary and a SUBTREE boundary of that tree
reproduces the unsplit tree exactly, as long as the cross-microbatch
accumulation is spelled with the contract's own flushed add.

**So `IdenticalMatmul.backward` states an alignment requirement of its
caller.** DEVIATION 1207.

> If you accumulate `dB` across microbatches, every microbatch's token count
> must be a multiple of 128, and the partition of the total must line up with
> the balanced tree's subtrees. `torch_ops.accumulation_split_is_aligned(
> total_tokens, [t0, t1, ...])` answers it for a given schedule. If it
> returns False your gradient is still bit-identical ACROSS VENDORS at that
> schedule -- it is simply a different number from the unsplit one, and your
> reproducibility claim has to name the schedule.

The bridge does NOT enforce this. It cannot -- it sees one call and has no
idea what the caller does with the gradient afterwards. It provides the
predicate, states the rule, and gate T6 measures that the rule holds through
the bridge rather than only inside the Mojo check.

`+=` into a PyTorch `.grad` buffer is a plain float add, NOT the contract's
flushed add. On a flush-to-zero backend they coincide; on one that is not
they do not, and that is a real gap between the measured Mojo result and what
a torch training loop actually executes. It is OWED and named in section 10.

---

## 7. Batched inputs, and the operations that map onto them

### 7.1 A 3-D or higher input is REFUSED. DEVIATION 1203.

Batched GEMM is DEFERRED by contract section 0.3, with its caller evidence.
The bridge does not quietly loop over the batch dimension.

The temptation is strong and the contract itself supplies the argument that
makes the loop LOOK free.

> Under sections 6 and 7 the arithmetic for cell `(i, j)` depends on `k` and
> the profile alone. So a batched GEMM whose per-cell arithmetic is this
> contract's cannot produce different bits from a loop of unbatched ones.

That is true, and it is exactly why the loop must not be hidden. A host-side
loop over `B` batch items is `B` kernel launches, `B` synchronizes on the
custom-op lane, and on the host lane `4B` copies. It would be numerically
indistinguishable from a real batched GEMM and arbitrarily slower, and the
caller would have no way to find out. `torch.bmm`-shaped work is refused with
a message that says so and shows the reshape for the case where the batch
dimension is contractible into `m` (a `(B, T, K) @ (K, N)` linear layer is
one `OP_NN` at `m = B*T`, no loop, no copy, and that IS the common case).

### 7.2 The orientation table

`transpose_a` and `transpose_b` describe the ARRAYS passed, matching
`_linalg_impl.matmul` word for word so that the two front doors cannot drift.

    transpose_a  transpose_b   op      C = ...    a.shape   b.shape
    False        False         OP_NN   a @ b      (m, k)    (k, n)
    False        True          OP_NT   a @ b.T    (m, k)    (n, k)
    True         False         OP_TN   a.T @ b    (k, m)    (k, n)
    True         True          REFUSED -- not one of the contract's three

What common torch expressions map to. **The bridge does not intercept any of
these; the user calls `identical_matmul` explicitly.** The table says which
call reproduces which expression.

    torch expression              bridge call
    a @ b            (2-D)        identical_matmul(a, b)
    torch.matmul(a, b) (2-D)      identical_matmul(a, b)
    torch.mm(a, b)               identical_matmul(a, b)
    F.linear(x, W)               identical_matmul(x, W, transpose_b=True)
    F.linear(x, W, bias)         the above, then `+ bias` in torch, which is
                                 NOT in the profile (contract section 10
                                 excludes bias and epilogues)
    a.T @ b                      identical_matmul(a, b, transpose_a=True)
                                 with `a` STORED (k, m) -- passing `a.T` as
                                 the tensor is a non-contiguous view and is
                                 refused per 3.1
    a @ b.T                      identical_matmul(a, b, transpose_b=True)
    a.T @ b.T                    REFUSED. DEVIATION 1211, mirroring 913.
                                 Write `identical_matmul(b, a).T.contiguous()`
                                 yourself, where the extra step is visible.
    torch.bmm / einsum / 3-D     REFUSED, section 7.1
    torch.mv, a @ v              OP_NT at n == 1, as a 2-D (1, k) operand;
                                 gemv is not a fourth operation (contract 0.1)

### 7.3 The op numbering, which is a trap, and which convention this boundary speaks

**This boundary speaks the `gemm_oracle.mojo` convention, and nothing else.**

    gemm/mojo_only/gemm_oracle.mojo   OP_NN = 0   OP_NT = 1   OP_TN = 2
    bench/gemm_shapes.mojo            OP_NT = 0   OP_TN = 1   OP_NN = 2

Those are a full permutation of each other. `gemm/host_entry.mojo` imports
`OP_NN, OP_NT, OP_TN` from `gemm_oracle` and validates against them;
`bindings/_mojolearn_linalg.mojo`'s `params[3]` is that same numbering;
`_linalg_impl.OP_NN/OP_NT/OP_TN` are 0/1/2 in that order. The bridge's
`OP_NN/OP_NT/OP_TN` are those same three integers imported FROM
`_linalg_impl` rather than retyped, so there is one literal in the Python
tree and not two.

There is already a gate in this repository whose entire job is to catch a raw
pass-through between the two conventions,
`check_op_encodings_are_not_interchangeable`. Gate T4 is this bridge's
version of it and section 8 says what it must do that a naive version would
not -- **run the cube shape**, because at `m == n == k` a permuted op code
returns a plausible full matrix and no shape error.

---

## 8. The gate plan. How anyone would ever prove this bridge is identical.

### 8.0 The two ways this bridge can be wrong while every number looks right

Both are consequences of the property the bridge exists to deliver, and both
are why an output check cannot certify a wrapper.

**(a) The profile is device independent, so a wrongly routed device is
invisible.** If the bridge silently runs on the CPU, or on the wrong GPU, or
runs the host lane while claiming the custom-op lane, the OUTPUT IS THE SAME
BITS. That is not a bug in the profile; it is the profile working. It means
no comparison of numbers, on any number of vendors, can tell you which device
ran. Only a read-back of what the process actually did can. Gate T9.

**(b) The FAST build makes no identity claim and returns the same bits on
Apple.** `_linalg_impl`'s module docstring already carries this -- the two
builds coincide at both pinned seams on Metal, so a caller comparing numbers
on one Mac learns nothing. The custom-op lane makes it worse, because that
lane JIT-compiles `torchbridge/identical_ops.mojo` through
`KernelLibrary.load_paths` and **this lane found no way to pass
`-D MOJOLEARN_NUMERIC_IDENTICAL=1` through that path.** Unless the library is
built ahead of time into a `.mojopkg` with the define, the custom-op lane
compiles the FAST arithmetic and calls it identical. Gate T7, and DEVIATION
1205, exist for precisely this.

Everything below is written against those two.

### T1 `check_bridge_dtype_refusals_fire`
float16, bfloat16, float64 and int32 operands each raise `TypeError` naming
the dtype. **Reach, not output** -- a refusal branch that is never taken is
indistinguishable from one that is absent. Sabotage arm `BRIDGE_CAST_DTYPE`
inserts `a = a.float()`; T1 must fail. Also asserts the message names
`autocast`, because that is the caller who hits it without typing a dtype.

### T2 `check_bridge_does_not_copy`
Two halves and both are required.

- The REFUSAL half. `x.t()`, `x[:, ::2]` and `x.expand(...)` each raise
  `ValueError` naming the strides. Sabotage arm `BRIDGE_SILENT_CONTIGUOUS`
  inserts `a = a.contiguous()`; T2 must fail. Note that with that sabotage in
  place every NUMERICAL gate in this document still passes.
- The NO-COPY half, which is the part a refusal test cannot give you.
  On the custom-op lane, record `a.data_ptr()` and `b.data_ptr()` before the
  call and assert the buffers MAX read were those addresses. Absent a hook
  into dlpack, the available proxy is that no new allocation appeared --
  `torch.cuda.memory_allocated()` before and after must differ by exactly the
  output tensor. Weaker than a pointer identity check and the gate must say
  so in its own failure text rather than claiming more.

### T3 `check_bridge_bits_equal_the_numpy_surface`
The same fixture through `mojolearn.linalg.matmul` (numpy) and through
`torch_ops.identical_matmul` (both lanes) must be **BIT EQUAL**, compared as
`t.view(torch.int32)` against `arr.view(np.int32)`, never `allclose` and
never `torch.testing.assert_close`. This is the gate that catches identity
lost in the wrapper while the kernel is untouched.

Shapes must include at least one with `m`, `n`, `k` pairwise distinct AND one
cube `m == n == k`. The cube is not redundant; it is the only shape at which a
swapped `m`/`n` or a permuted op code is a silent wrong answer rather than an
exception.

Sabotage arms, all of which produce a full matrix of plausible floats.
`BRIDGE_SWAP_MN`, `BRIDGE_PERMUTE_OP` (map through `bench/gemm_shapes.mojo`'s
numbering), `BRIDGE_ROUTE_TORCH_MATMUL` (call `torch.matmul` instead).

### T4 `check_bridge_op_encoding_is_the_oracle_convention`
Asserts `torch_ops.OP_NN, OP_NT, OP_TN` are the same objects as
`_linalg_impl`'s, and separately runs all three orientations on ONE logical
pair of matrices transposed on the host, requiring bit equality (contract
section 3 makes that a requirement, not an observation). Run at the cube
shape as well as the distinct shape, for T3's reason. This is the bridge's
`check_op_encodings_are_not_interchangeable`.

### T5 `check_bridge_backward_is_the_routing_table`
For each of the three forward orientations, compare the bridge's `dA`, `dB`
and `dbias` against `gemm/mojo_only/gemm_backward.mojo`'s own gates' expected
values BY BITS, and against central finite differences to a tolerance (the
finite-difference arm is what separates `SAB_BWD_UNTRANSPOSED` at a square
symmetric fixture, per that file's own note). Shapes must have `m`, `n`, `k`
pairwise distinct so an ignored `dc_side` is a shape error somewhere.

### T6 `check_bridge_gradient_split_alignment`
Reproduce the backward lane's measured G5 result THROUGH THE TORCH PATH.
`T = 512` split 256/256 and `T = 384` split 256/128 must move 0 gradient
cells; `T = 300` split 150/150, `T = 512` split 200/312 and `T = 384` split
192/192 must move 31, 30 and 31. **Both halves are required.** A bridge that
ignored the split entirely -- concatenating the microbatches internally --
would pass the aligned cases and fail the misaligned ones, which is how the
gate tells "aligned splits are free" from "the split was not honored".
Also asserts `accumulation_split_is_aligned` agrees with the measurement on
all five, so the predicate is checked against a number rather than against
its own docstring.

### T7 `check_bridge_reports_its_compiled_numeric_mode`
The custom-op library exposes a second op, `identical_gemm_mode_probe`, that
writes `GLOBAL_NUMERIC_MODE` into a one-element output tensor. The gate

- asserts `torch_ops.numeric_mode()` reads that probe and not the environment
  variable,
- builds the library WITHOUT the define and asserts `require_identical()`
  RAISES,
- builds it WITH the define and asserts it does not,
- and asserts the two builds are distinguished by the probe and NOT by
  comparing outputs, since on Apple they coincide (`_linalg_impl` docstring,
  contract 4.1).

**This is the most important gate in the list**, because section 8.0(b) is
the failure mode with no other symptom.

### T8 `check_bridge_is_identical_across_vendors`
The card. One fixture file, one script, three boxes.

    row  vendor       lane        must equal
    1    Apple M4     HOST        row 2, row 3
    2    NVIDIA H100  HOST        row 1, row 3
    3    AMD MI325X   HOST        row 1, row 2
    4    NVIDIA H100  CUSTOM_OP   row 2
    5    AMD MI325X   CUSTOM_OP   row 3

Rows 1-3 are the spine, one lane on three vendors. Rows 4-5 tie the second
lane to the spine on the vendors that can run it. Compared as hex of the
float32 bits, for the forward output and for `dA`, `dB` and `dbias`. Every
row records the numeric mode the probe reported (T7) beside its hash; a row
whose probe says `fast` is not a card row, it is a note.

Dropping rows 2 and 3 as "obviously equal to row 1" is the shortcut that
would leave the card measuring the kernel instead of the bridge. Section 1.2.

### T9 `check_bridge_reports_the_lane_and_device_it_actually_used`
`identical_matmul(..., return_meta=True)` returns the lane, the device, the
numeric mode and the profile string, and each is read back from what ran
rather than from the input tensor. Sabotage arm `BRIDGE_LIE_ABOUT_LANE`
forces the host lane while reporting `CUSTOM_OP`; T9 must fail. Sabotage arm
`BRIDGE_FORCE_CPU_TARGET` routes an `mps` tensor to the custom-op lane (which
section 1.1 says lands on the CPU); T9 must fail **and every other gate in
this document must still pass**, which is the demonstration of 8.0(a).

---

## 9. Deviation block

| # | what |
|---|---|
| 1200 | Two transport lanes, CUSTOM_OP and HOST, selected by device and REPORTED rather than inferred |
| 1201 | Non-contiguous input refused, never silently made contiguous |
| 1202 | Non-float32 input refused by name, never cast; message names `autocast` |
| 1203 | 3-D and higher input refused; batched GEMM deferred per contract 0.3, no hidden host loop |
| 1204 | `mps` routed to the HOST lane explicitly, because `max_device_ref` has no mps arm and the custom-op lane would silently target the CPU |
| 1205 | The compiled numeric mode is read back from the kernel through a probe op, not from the environment; the custom-op lane's JIT has no visible define hook |
| 1206 | Backward implemented as a `torch.autograd.Function` calling the same forward op three times through the routing table, with no second arithmetic |
| 1207 | The microbatch alignment requirement stated as a caller contract, with `accumulation_split_is_aligned` as the predicate |
| 1208 | The bias-gradient ones vector built as exactly `1.0` and cached per token count, per `identical_gemm_backward_bias_ones_floats` |
| 1209 | No `torch.compile`, CUDA-graph or non-default-stream promise of any kind |
| 1210 | The gate plan, T1 through T9, with the card structured as one lane across three vendors plus one vendor across two lanes |
| 1211 | `transpose_a=True, transpose_b=True` refused by name, mirroring DEVIATION 913 |
| 1212 | A caller-supplied `out=` is checked for dtype, contiguity, shape, device and aliasing, and never repaired |
| 1213 | `torch_ops.profile()` reports what the process is holding, mirroring `_linalg_impl.profile()`, with the lane and the probe result added |

1214 through 1229 are reserved to this lane and unused.

---

## 10. OWED, AND WHY I DID NOT DO IT HERE

This lane may write only `torchbridge/__init__.mojo`,
`torchbridge/identical_ops.mojo`, `torchbridge/TORCH_BRIDGE_PLAN.md` and
`python/mojolearn/torch_ops.py`. Everything below needs a file this lane may
not touch, or a run this lane may not perform.

1. **Nothing here has been compiled or executed.** No `mojo`, no `pixi`, no
   `python`, not even an import check. The first owed item is finding out
   whether `torchbridge/identical_ops.mojo` compiles at all.

2. **`python/mojolearn/identical/_mojolearn_linalg.so` does not exist in this
   tree.** It is one of five of the ten extension modules missing from
   `python/mojolearn/identical/`. Until it is built with
   `MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_linalg.sh`, the HOST
   lane raises `ImportError` in identical mode, which means the Apple lane of
   this bridge cannot run at all. Not this lane's file and not this lane's
   run.

3. **There is no build script for the custom-op kernel library, and without
   one the CUSTOM_OP lane is the FAST arithmetic.** `CustomOpLibrary` accepts
   a `.mojo` path or a `.mojoc`/`.mojopkg`, and JIT-compiles the former; this
   lane found no way to pass `-D MOJOLEARN_NUMERIC_IDENTICAL=1` through
   `KernelLibrary.load_paths`. The fix is a `bindings/build_torchbridge.sh`
   modeled on `build_linalg.sh` that runs `mojo package -D
   MOJOLEARN_NUMERIC_IDENTICAL=1 -I . torchbridge -o
   python/mojolearn/identical/torchbridge.mojopkg`, with the AIR-blob floor
   and the `minos` read-back that script already carries. `bindings/` is not
   this lane's directory. **Gate T7 is what stops this hole from being
   silent in the meantime, and until T7 runs, the CUSTOM_OP lane's identity
   claim is unsupported.**

4. **`_backend._MODULES` needs no change** -- `_mojolearn_linalg` is already
   in it as of DEVIATION 869 -- but `_linalg_impl._load`'s comment still says
   otherwise and says the collapse into `from . import _mojolearn_linalg` is
   in a hand-off note. That is a stale comment in a file this lane may not
   edit. Recorded here rather than fixed.

5. **`python/mojolearn/__init__.py` does not export `torch_ops`.** It should
   not import it eagerly -- torch must stay an optional dependency, which is
   why `torch_ops.py` imports torch lazily inside functions -- so the wiring
   is `mojolearn.torch_ops` as a lazily imported submodule, plus an entry in
   `__all__`. One edit, someone else's file.

6. **The cross-microbatch accumulator is a plain torch `+=`, not the
   contract's flushed add.** Section 6.3. The measured G5 alignment result
   used the contract's own `ftz`-flushed add; a torch training loop
   accumulating into `.grad` does not. On a flush-to-zero backend the two
   coincide and on one that is not they do not. Closing this needs either a
   pinned accumulate op in `gemm/` or a documented restriction, and `gemm/`
   is not this lane's directory.

7. **An asynchronous custom op.** Section 2.2 -- the op calls the
   synchronizing `identical_gemm` because a custom op has no caller to own a
   workspace, so every matmul costs a synchronize. A persistent per-shape
   workspace owned by the op library would remove it. That is a real design
   change to the op's contract and it needs the buffer-lifetime hazard
   (`[[mojo-buffer-freed-at-last-use]]`) worked through with a gate, not a
   quick edit.

8. **The `mps` gap is upstream.** `max_device_ref` and `max_device` in
   `max/experimental/torch/torch.py` have no `"mps"` arm. That is Modular's
   file in the pinned environment, not ours. Worth reporting upstream; the
   workaround is section 1's HOST lane and it costs four copies per call.

9. **A `torch.nn.Module` wrapper (`IdenticalLinear`) is not written.** It
   would be the obvious thing a user wants and it is deliberately deferred
   until T1 through T9 are green, because a Module makes the call implicit
   and every failure mode in section 8.0 is one that hides better when the
   call is implicit.

10. **All nine gates are unwritten.** Section 8 is a specification. No gate
    file exists, no sabotage arm exists, and until a sabotage arm has been
    fired and shown to bite, each of these gates is a claim that a passing
    check has never been made to fail.

11. **A pointer-taking entry point in `gemm/` would delete the custom-op
    lane's three staging copies.** Section 2.2. `identical_gemm` and
    `identical_gemm_into` require `DeviceBuffer`, `DeviceBuffer` cannot wrap a
    foreign device pointer, and a custom op only ever holds pointers. The
    shape of the fix is a sibling of `gemm/host_entry.mojo` --
    `identical_gemm_device_ptrs(ctx, c_ptr, a_ptr, b_ptr, ws_ptr, m, n, k,
    op)` -- containing no arithmetic, doing what `identical_gemm_with_plan`
    already does with the buffers it is handed. `gemm/` is not this lane's
    directory, and this is a change to the certified family's public surface,
    so it needs that lane's gates and not a drive-by edit.

12. **This lane never verified the `ManagedTensorSlice` accessor spellings.**
    `torchbridge/identical_ops.mojo` needs, from an `InputTensor` /
    `OutputTensor`, a raw device pointer and the two runtime dimensions. The
    published `extensibility` docs list the module but this lane could not
    fetch the `managed_tensor_slice` page (404 on every published form of the
    URL) and the `.mojoc` package is compressed. Every such use in that file
    carries a `SPELLING NOT VERIFIED` comment naming the alternatives. That
    is the single most likely reason the file does not compile.
