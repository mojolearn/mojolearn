# The training step, composed -- `mojolearn.identical.train.step.fp32.v1`

**NOTHING DESCRIBED HERE HAS BEEN COMPILED OR EXECUTED.** This plan and its
two companion files (`training/mojo_only/train_loop.mojo`,
`training/mojo_only/train_step_check.mojo`) were written on 2026-08-25 by the
training-loop lane, DEVIATIONS 1550 through 1589. No `mojo` process has read
any of the three. No device has run a step. No checkpoint hash has ever been
computed, on any vendor, and therefore **no two checkpoint hashes have ever
been compared.** Every sentence below that says a clause "catches" something
or an arm "must move" is a PREDICTION with no measurement behind it.

Section 12 lists what is owed. Read that before believing anything else.

---

## 0. Why this file exists, and what it is not

Every identical op in this repository is gated IN ISOLATION. A GEMM is
bit-identical on three vendors over 60 stages. A Mamba block is over 17. The
Llama decoder block forward passed five clauses and thirteen sabotage arms
last night, and cross-entropy and the optimizer each passed six clauses on
their first execution.

None of that is the claim. The claim is

> run N steps of the same model on an M4, an H100 and an MI325X, and the
> checkpoint hashes match.

**Two ops that are each bitwise identical can still produce a divergent
run.** Three mechanisms make that possible and none of them is visible to a
per-op gate.

1. **Absorption.** Op A's output feeds op B. If A is identical, B is
   identical on A's output. That composes fine. What does not compose is a
   quantity that is identical only because a fixture kept it in a benign
   range. A training run walks the parameters somewhere no fixture chose,
   and the first subnormal, the first cancellation to `-0.0`, the first
   `logdenom` at a magnitude the loss fixture never held, arrives at step 40
   and not at step 1.
2. **Accumulation across steps.** `m` and `v` are carried state. A one-ULP
   divergence in a gradient at step 3 is not corrected at step 4; it is
   integrated. The per-step gates all start from fresh state and cannot see
   this at all -- `optimizer_check` found exactly this shape of hole in its
   own arms (`OPT_SAB_MOMENT_LERP` is bit-inert at every first step, because
   `m_prev` is `+0.0`).
3. **Step-count invariance.** `beta1^t` and `beta2^t` are computed on the
   host, once per step, from `t`. The two spellings the contract separates
   agree exactly through `t = 6` and first differ at `t = 7`. A composed run
   is the only thing that reaches `t = 7`.

Only running it finds out. This plan builds the harness that runs it and,
more importantly, builds the machinery that makes a green result MEAN
something.

**This is not a training library.** It is an instrument. It trains a model
small enough to fit inside a one-hour GPU lease, on data it generates itself,
for the sole purpose of producing a number that can be compared across three
machines.

---

## 1. The step, and what is actually gated

The step, in order. `M = B * L` token-major rows throughout.

| # | stage | entry point | status |
|---|-------|-------------|--------|
| 1 | data | `train_loop.mojo::train_batch_ids` (this lane) | NEW, ungated |
| 2 | embed | `embedding/mojo_only/embedding_identical.mojo::identical_embedding_forward_into` | **WRITTEN, NO GATE** |
| 3 | block forward | `transformer/ported/transformers/models/llama/modeling_llama.mojo::llama_decoder_layer_forward` | **GATED** (5 clauses, 13 arms fired 2026-08-24) |
| 4 | lm_head | `gemm/mojo_only/gemm_identical.mojo::identical_gemm_into`, `OP_NT` at `(M, V, d_model)` | **GATED**, three vendors |
| 5 | loss | `training/mojo_only/loss.mojo::identical_ce_forward_into` | **GATED** (6 clauses, first execution 2026-08-25) |
| 6 | loss backward | `training/mojo_only/loss.mojo::identical_ce_backward_into` | **GATED** (same 6 clauses) |
| 7 | lm_head backward | `gemm/mojo_only/gemm_backward.mojo::identical_gemm_backward_a_into` and `_b_into` | **GATED** |
| 8 | block backward | `transformer/mojo_only/transformer_backward.mojo::llama_decoder_layer_backward` | **WRITTEN, NO GATE** |
| 9 | embedding backward | `embedding/mojo_only/embedding_identical.mojo::identical_embedding_backward_into` | **WRITTEN, NO GATE** |
| 10 | pack gradients | `train_loop.mojo::train_copy_range_kernel` (this lane) | NEW, ungated |
| 11 | optimizer | `training/mojo_only/optimizer.mojo::identical_optimizer_step` | **GATED** (6 clauses, first execution 2026-08-25) |
| 12 | digest | `train_loop.mojo::snapshot` and `::digest_of_lists` (this lane) | NEW, ungated |

### 1.1 The headline caveat, and it is not a footnote (DEVIATION 1570)

**Two of the twelve stages have never been gated at all, and both are on the
critical path.** `transformer_backward.mojo` and `embedding_identical.mojo`
are written, they parse, and nothing has ever compared either of them to its
oracle on any hardware.

So state precisely what three matching checkpoint hashes would and would not
demonstrate.

* They WOULD demonstrate that stages 8 and 9 produce the same bits on three
  vendors on this fixture. That is a real and separately interesting result,
  and it is the result the cross-vendor run is for.
* They would NOT demonstrate that stages 8 and 9 are CORRECT. Three machines
  computing the same wrong gradient agree perfectly.

Correctness against the oracle is the other half, and it lives on ONE device
in `train_step_check.mojo` clause (a), which compares every intermediate of
one step against a host oracle composed of the existing oracles. **A run of
the loop without clause (a) having passed is not evidence of anything.** The
plan's ordering requirement is therefore

> clause (a) green on one device, THEN the cross-vendor loop.

Not the other way around, and not both at once on three rented machines.

### 1.2 What the step does NOT do

* No KV cache is carried between steps. Every step builds a fresh
  `LlamaKVCache` at `pos0 = 0` and prefills (DEVIATION 1554). A carried cache
  would make a step depend on history through a second channel, and
  disentangling a cache divergence from a parameter divergence in a
  checkpoint hash is not possible.
* The `lm_head` is UNTIED from the embedding table (DEVIATION 1555). Tying
  them makes one parameter tensor receive gradient from two sites, and the
  order those two contributions are summed in is a real arithmetic decision
  with two answers. v1 does not make it.
* One decoder layer. Stacking layers adds nothing to the identity question
  that the first layer does not already contain, and multiplies the run cost.
* No dropout anywhere. The ported block has none.
* No microbatching, no gradient accumulation, no distributed anything, one
  device, one stream.

---

## 2. The model, and the FROZEN parameter order (DEVIATION 1550)

The parameters live in ONE flat `DeviceBuffer[DType.float32]`. `offsets` is
the `List[Int]` of length `J + 1` that `identical_optimizer_step` already
takes, `j` IS the `param_id`, and its ascending order is the cross-tensor
summation order of optimizer contract clause 3.3.

**This order is part of the checkpoint hash specification.** Changing it
changes every hash this profile has ever produced, and it may not be changed
without a new profile version.

| `param_id` | tensor | shape | count at the v1 shape |
|---|---|---|---|
| 0 | `embed` | `[V, d_model]` | 2048 |
| 1 | `norm1_w` | `[d_model]` | 32 |
| 2 | `w_q` | `[n_heads*head_dim, d_model]` | 1024 |
| 3 | `w_k` | `[n_kv*head_dim, d_model]` | 512 |
| 4 | `w_v` | `[n_kv*head_dim, d_model]` | 512 |
| 5 | `w_o` | `[d_model, n_heads*head_dim]` | 1024 |
| 6 | `norm2_w` | `[d_model]` | 32 |
| 7 | `w_gate` | `[intermediate, d_model]` | 2048 |
| 8 | `w_up` | `[intermediate, d_model]` | 2048 |
| 9 | `w_down` | `[d_model, intermediate]` | 2048 |
| 10 | `lm_head` | `[V, d_model]` | 2048 |

`J = 11`, `n_total = 13376`.

The v1 shape, and every one of these is FROZEN for the profile.

```
V            = 64     vocabulary
d_model      = 32
n_heads      = 4
n_kv         = 2
head_dim     = 8      (d_model == n_heads*head_dim, head_dim even)
intermediate = 64
B            = 2      batch
L            = 8      sequence length,  M = B*L = 16
rms eps      = 1e-6   / 0x358637BD   (transformer contract, frozen)
rope theta   = 10000  / 0x461C4000   (transformer contract, frozen)
```

Everything is small on purpose. The largest buffer in the step is `dlogits`
at `[M, V] = 1024` floats. A hundred steps is minutes on any of the three
machines, and the whole run fits inside one lease with room to spare
(`[[rented-gpus-self-expire]]`).

### 2.1 The flat buffer is the state; the per-tensor buffers are a view (DEVIATION 1553)

`LlamaDeviceWeights` owns nine `DeviceBuffer`s and `identical_optimizer_step`
wants one flat buffer. Rather than teach either side about the other, the
loop keeps the flat buffer as THE parameter state and refreshes the
per-tensor buffers from it at the top of every step (`unpack`), then copies
the eleven gradient buffers into a flat `grad` at the bottom (`pack`).

Both are pure element copies through `train_copy_range_kernel`. There is no
arithmetic in either, so neither can move a bit -- **but a wrong offset gives
plausible, in-bounds, wrong numbers that are identical on all three
vendors**, which is the single most dangerous defect class in this file.
Clause (f) is the assertion that catches it and it is not optional.

---

## 3. The checkpoint hash (DEVIATION 1551), spec `mojolearn.identical.train.ckpt.v1`

Specified here the way `core/identity_trace.mojo` specifies a card, and using
that file's hash so a digest from either instrument means the same kind of
thing.

### 3.1 The function

FNV-1a64. Offset basis `0xCBF29CE484222325`, prime `0x100000001B3`, folded
ONE BYTE AT A TIME, over the LITTLE-ENDIAN bytes of the elements IN INDEX
ORDER. Byte at a time on purpose -- a word-at-a-time variant is a different
function, and `tools/identity_trace_diff.py` recomputes this from a `.bin`
dump, so the writer and the reader must be the same spelling.

This is `core/identity_trace.mojo::fnv1a64_bytes` and the loop calls that
function rather than restating it.

### 3.2 What is hashed, in what order

Four digests per checkpoint. `h(seed, buf, n)` means the fold of `n` elements
of `buf` starting from `seed`.

```
h_param  = h(FNV_OFFSET, param,   n_total)
h_m      = h(FNV_OFFSET, m_state, n_total)
h_v      = h(FNV_OFFSET, v_state, n_total)
h_all    = h(h(h(FNV_OFFSET, param, n_total), m_state, n_total), v_state, n_total)
```

`h_all` is the CHAINED fold -- param, then `m`, then `v`, one continuous byte
stream -- and it is the single number two machines compare. The other three
exist so that a mismatch has an address rather than a verdict.

Plus one digest per parameter tensor, each a FRESH fold from the offset
basis, so a mismatch localizes to a `param_id`.

```
h_p[j]   = h(FNV_OFFSET, param + offsets[j], offsets[j+1] - offsets[j])   for j in 0..J-1
```

Note that concatenating the eleven tensor ranges in ascending `j` over a
contiguous flat buffer is the same byte sequence as `param[0 : n_total]`, so
`h_param` and the chained per-tensor fold coincide today. The per-tensor form
is written out anyway because it is what defines the hash if the layout ever
stops being contiguous, and because it is what defines `h_p[j]`.

### 3.3 What is NOT hashed, and this list is the specification

The digest is a pure function of the parameter and optimizer-state BITS and
of nothing else. Specifically it does not read, and may never read

* the iteration count `t`, in any representation
* the wall clock, the elapsed time, or any timing
* the device, the vendor, the driver version, the SM or CU count
* the block size, the grid shape, the plan `choose_gemm_plan` picked, or the
  workspace size
* the loss value, the learning rate, the seed, or any configuration field
* the length of the run, or how many times the loop was entered
* any pointer, any allocation address, any buffer CAPACITY

The last one is a real hazard and not a hypothetical. Every count above is
`n_total` or a tensor extent, **never `len(buf)`**. A buffer allocated with
slack and hashed to its capacity folds uninitialized memory into the digest,
which differs run to run on ONE machine and would make the instrument report
divergence everywhere. That is `core/identity_trace.mojo` rule 3, and clause
(g) below is what would catch a violation of it.

### 3.4 How it is emitted

Four records per step onto an `IdentityTrace`, and the tags are machine
independent (rule 2). The step index is zero padded to four digits
(DEVIATION 1575) so lexical and numeric order agree and two traces of the
same length align tag for tag.

```
train.s%04d.data.ids       i32  B*(L+1)   the batch, BEFORE it reaches a float
train.s%04d.loss           f32  1
train.s%04d.ckpt.tensors   u32  2*J       h_p[0..J-1], high word then low
train.s%04d.ckpt.digest    u32  8         h_param, h_m, h_v, h_all, same
```

Plus a step-zero checkpoint recorded BEFORE any step runs, carrying only the
two `ckpt.*` tags at `s0000`. Steps are numbered 1..N and `t` is one-based,
which is `optimizer_step_oracle`'s own convention.

**The record count is therefore exactly `2 + 4*N`, and the gate asserts it**
(DEVIATION 1563). A trace with zero records compared against a trace with
zero records is three machines agreeing about nothing, and that failure is
otherwise completely silent.

**The digests are recorded as `uint32` HIGH-WORD-THEN-LOW pairs and NOT as
`uint64`** (DEVIATION 1567), for a reason found by reading
`core/identity_trace.mojo::_dtype_name` -- it handles `int64` and NOT
`uint64`, so a `u64` record would emit the dtype `?` and
`tools/identity_trace_diff.py`'s parser accepts exactly six names. Splitting
into two `u32` words needs no conversion that could trap and no dtype the
differ does not know. **This is the first defect the plan's own standard
predicted and then found**, before any compiler ran, and it is exactly the
class listed in section 4.1.

Never as text. `String(Float32)` does not round trip in this toolchain
(`[[mojo-string-float-roundtrip]]`), and although a digest is an integer, a
value that went through a decimal round trip anywhere could report agreement
across a real difference -- the worst failure an instrument of this kind can
have.

`h_all` is also printed to stdout as sixteen lowercase hex digits and written
to a sidecar file named by `MOJOLEARN_TRAIN_CKPT`, so a three-vendor
comparison is `diff` on three one-line files and does not require the trace
differ at all.

---

## 4. What would make a matching hash VACUOUS

**This is the section that matters most.** A run whose parameters barely move
produces the same digest on any hardware and proves nothing. Three matching
hashes can be three machines agreeing that nothing happened.

The rule this lane adopts, and DEVIATION 1559 is the rule.

> A run that has not executed the negative controls reports **VACUOUS**, not
> PASS. The word PASS does not appear in the output of a run that produced no
> divergence when it was supposed to.

### 4.1 The failure-mode table

Every row is a way to get three matching digests while demonstrating
nothing. The right column is the assertion that catches it, and if a row had
no assertion it would be listed as owed rather than left out.

| # | failure | why the digests still match | caught by |
|---|---------|------------------------------|-----------|
| V1 | the loop ran zero steps | nothing was computed anywhere | `steps_run == N` assert, and N5 |
| V2 | parameters do not move (lr too small, gradients structurally zero) | the digest is of the initial state, which is a function of the seed alone | N5, per tensor |
| V3 | the digest reads the wrong buffer (a stale per-tensor view, a scratch copy) | reads something nothing updates | N3 plus N5 |
| V4 | the seed is never used; data is a constant | all three machines generate the same constant | **N1** |
| V5 | the step index never reaches the data generator | every step trains on the same batch, deterministically | **N2** |
| V6 | `t` is frozen at 1 | bias correction is constant; the run is still deterministic | N6, plus the direct `t >= 7` requirement in 5.2 |
| V7 | gradients packed at the wrong offsets | deterministic and wrong, identically on every vendor | clause (f), plus `h_p[j]` addressing |
| V8 | an op raised and the harness swallowed it | the digest is of the pre-failure state | refusal count asserted zero on the clean arm; no bare catch |
| V9 | the trace was disabled, so no digests were emitted | zero records compared to zero records | record count `== 2 + 4*N`, and an empty trace is refused |
| V10 | a sabotage `-D` was misspelled | `is_defined` returns False, the build is clean, and "the arm did not bite" is recorded as a pass | `MOJOLEARN_TRAIN_EXPECT_ARM` (DEVIATION 1566) |
| V11 | the digest folded uninitialized tail memory | it would NOT match; it would differ everywhere, including twice on one machine | clause (g), same-process repeat |
| V12 | the model is so small that every gradient underflows | parameters do not move, see V2 | N5's absolute floor, stated in 5.3 |

V10 is a repository scar and not a hypothetical. `tools/gemm_ladder.sh:71`
recorded a clean build with a misspelled `-D` as an arm that did not bite,
which is the exact inverse of the truth. `optimizer_check.mojo` closed the
same hole with `MOJOLEARN_OPT_EXPECT_SABOTAGE` and this lane copies that
spelling deliberately.

### 4.2 The negative controls

Each one is a run that SHOULD diverge and MUST produce a different `h_all`,
or a run that should NOT move and must produce an identical one. They are
selected by `MOJOLEARN_TRAIN_ARM`.

---

**N1 -- `seed_plus_one`. The seed must reach the data.**

Same everything, `MOJOLEARN_TRAIN_SEED` incremented by one.
`h_all` after step N MUST differ from the clean run's.

Catches V4 (the seed was never used), V1 (zero steps -- the digest would be
the initial state, which does not depend on the seed either, so N1 fires),
and a data buffer that was allocated and never uploaded.

Does NOT catch V5. A run that trains on step 1's batch forever still responds
to the seed.

---

**N2 -- `data_reverse`. The step index must reach the data.**

Same seed, but the per-step batches are presented in reverse step order --
step 1 gets what the clean run gave step N, and so on. `h_all` after step N
MUST differ.

Catches V5, and it also catches an optimizer whose update happens to be
order-free (it must not be; Adam's `m` and `v` are order dependent by
construction).

**This control is VACUOUS at `N == 1` and the gate REFUSES to run it there**
(DEVIATION 1560) rather than reporting a pass. Reversing a one-element
sequence is the identity, so at `N == 1` this arm cannot fire, and an arm
that cannot fire reported as an arm that passed is exactly the class of
defect that last night's first gate runs found twice. The refusal names the
reason.

---

**N3 -- `ulp`. The digest must read the buffer the model actually computes with.**

Before step 1, the LAST float of `param_id = 4` (`w_v`) is replaced by its
one-ULP neighbor -- next-representable away from zero, done on the BITS
(`bits + 1` for a positive value, `bits + 1` on the magnitude for a negative
one) and never by adding a small float. `h_all` after ONE step MUST differ.

Catches V3 (the digest reads a stale copy), a forward that does not actually
read `w_v`, and a parameter that never got written back into the flat state.

**The arm must ALSO move the recorded `train.s0001.loss`** (DEVIATION 1561),
and the gate asserts both. If the digest moves but the loss does not, the
perturbation reached the checkpoint by being COPIED there rather than by
being computed with, and the arm has demonstrated the opposite of what it
was armed for. This is the "reached but inert" rule
(`[[reached-but-inert]]`) pointed at a control instead of at a kernel.

`w_v` is chosen rather than an embedding row because every token's forward
pass reads `w_v`, so the perturbation cannot land on a row the batch never
touched. The LAST float rather than the first because an off-by-one that
skips the tail of a tensor is a defect this positions the arm to catch.

---

**N4 -- `zero_lr`. An update must not happen when it should not.**

`lr = 0.0`, `weight_decay = 0.0`, clipping off, AdamW, N steps.

* `h_param` after N steps MUST EQUAL `h_param` at step 0.
* `h_m` and `h_v` MUST both DIFFER from step 0.

At `lr = 0` AdamW's `decay_mul` is `1 - lr*wd = 1` exactly and `step_size` is
`0 / bc1 = +0.0`, so `p` is provably unchanged bit for bit while `m` and `v`
integrate normally. This is the inverse control (DEVIATION 1576) -- it proves
the harness is not simply scribbling on the parameters, and it proves `m` and
`v` are genuinely written, which N6 then leans on.

If `h_param` moves here, either `lr` did not reach the kernel or something
outside the optimizer is writing parameters.

---

**N5 -- movement. The vacuity control, and it is not a divergence control.**

After N steps of the CLEAN run, the gate asserts all of the following.

1. `n_changed`, the count of flat parameter indices whose BITS differ from
   step 0, is at least `0.99 * n_total`.
2. For EVERY `param_id` in 0..10, at least one element changed. A whole-model
   "something moved" check passes with ten of eleven tensors frozen, and a
   frozen tensor is exactly what a structurally-zero gradient looks like
   (`dw_norm1` never written, an embedding backward that only ever fills row
   zero).
3. `max |p_j(N) - p_j(0)|` over each tensor is at least `1e-4`.
4. **The embedding separation** (DEVIATION 1562). Among `param_id = 0`'s
   rows, take one token id that APPEARS in step 1's batch and one that does
   NOT. Both must move -- weight decay moves every parameter -- but they must
   move by DIFFERENT amounts. If they moved identically, the embedding
   gradient contributed nothing and the whole of stage 9 is inert while the
   digest still changes every step.

Clause 4 is the one that separates decay-only motion from gradient motion,
and it is the reason `weight_decay` is nonzero on the clean arm. With
`weight_decay = 0` the absent rows would be exactly frozen and clause 2 would
be satisfied by the present rows alone.

---

**N6 -- `split_4_6`. Step-count composition.**

Run 10 steps in one process. Then, in the SAME process, from the same
initial state, run 4 steps, take a digest, and run 6 more from the carried
state. `h_all` after both must be EQUAL.

Catches V6 (`t` frozen or restarted), an SGD `buf_initialized` flag that
resets, and a data generator that restarts its stream.

**Stated honestly:** this is a resume WITHIN one process and not from a file.
There is no checkpoint serialization in v1 (DEVIATION 1569), so
`OPT_SAB_RESUME_REINIT`'s real form -- a checkpoint that drops the momentum
flags on the way to disk -- is not reachable from here and this control does
not test it.

---

**N7 -- batch splitting is NOT a control and must not be added.**

Someone will want to assert that two microbatches of 8 produce the same
digest as one batch of 16. **That claim is false and asserting it would be a
gate that fails for a correct implementation.** `gemm_backward_b_call`'s `k`
IS the token count, so a weight gradient's contraction length changes when
the batch is split, `contract_partition` picks a different `(L, P)`, and the
bits legitimately move. The embedding backward's carry (`accumulate = True`)
reproduces a split bit-exactly and the weight gradients do not, and the two
facts sitting next to each other is what makes this worth writing down rather
than leaving to be rediscovered. DEVIATION 1568.

### 4.3 The rule the report obeys

The clean run's digest is reported alongside the arm results, always, in one
block. A digest printed on its own is not a result. The output shape is

```
clean      h_all=<16 hex>   moved=<n_changed>/<n_total>   tensors_moved=11/11
N1 seed    h_all=<16 hex>   DIFFERS   (required)
N2 order   h_all=<16 hex>   DIFFERS   (required)
N3 ulp     h_all=<16 hex>   DIFFERS   (required)   loss_moved=yes (required)
N4 zerolr  h_param SAME (required)    h_m DIFFERS (required)  h_v DIFFERS (required)
N6 split   h_all=<16 hex>   SAME      (required)
```

and any row whose requirement is not met makes the whole run FAIL, including
the rows whose requirement is SAME.

---

## 5. Making the parameters move, and proving they did

### 5.1 The configuration

```
optimizer      AdamW  (OPT_ADAMW)
lr             1e-3
beta1          0.9
beta2          0.999
eps            1e-8
weight_decay   0.01
max_norm       0.0     -- clipping OFF (DEVIATION 1558)
loss           CeConfig(V, ignore_index = -100, REDUCTION_MEAN, eps = 0.0, num_items = 0)
steps N        8 by default, MOJOLEARN_TRAIN_STEPS overrides
```

Clipping is off on the clean arm because with clipping ON one parameter's
update is a function of every gradient in the model (optimizer contract 3.5),
which is the reference's own semantics and not a defect -- but it couples
every tensor to every other and destroys the per-tensor localization that
`h_p[j]` exists to provide. `clip_on` is a separate arm.

### 5.2 Why N = 8 and not 1 (DEVIATION 1557)

Two of the optimizer's own arms are structurally invisible below `t = 7`, and
`optimizer_check.mojo` recorded both.

* `OPT_SAB_MOMENT_LERP` is bit-inert whenever `m_prev` is exactly `+0.0`,
  which is EVERY first step from fresh state.
* The `beta^t` spellings the contract separates agree exactly through
  `t = 6` and first differ at `t = 7`.

So a composed run that stops at `t = 4` cannot exercise either, and 8 covers
both with one step of margin. `MOJOLEARN_TRAIN_STEPS` runs the same binary at
1 or at 100; 1 is for clause (a)'s single-step oracle comparison and is not a
number to report identity from.

### 5.3 The check that proves they moved

The arithmetic below is ON PAPER and has not been run.

Adam's per-element step is `step_size * m_hat / (sqrt(v_hat) + eps)`, and for
a gradient of roughly constant sign the ratio is close to `1`, so each
parameter moves on the order of `lr` per step. Over 8 steps that is about
`8e-3`. The weights are initialized in `[-0.5, 0.5]`, where the FP32 ULP is
about `6e-8`. The predicted motion is therefore roughly five orders of
magnitude above the last bit -- not a rounding wobble, a real displacement.

N5's floor is set at `1e-4`, which is two orders below the prediction and
four above the ULP. If the observed motion is anywhere near `1e-4` the
prediction was wrong and the fixture is weak, and the FIRST run must print
the per-tensor max displacement before any identity claim is believed.

---

## 6. Data, and determinism without a clock (DEVIATION 1556)

Ids come from splitmix64 and from nothing else. No RNG reads the clock, the
process id, an address, or the environment beyond one integer seed.

```
key      = splitmix64(seed ^ (step_index * 0x9E3779B97F4A7C15))
ids[i]   = Int32( (splitmix64(key + UInt64(i)) >> 33) % UInt64(V) )
```

for `i` in `0 .. B*(L+1) - 1`. Each row of `B` draws `L + 1` ids; the first
`L` are the inputs and the last `L` are the next-token targets. There is
therefore NO ignored row and `count == M` exactly, which keeps `ce_divisor`
on its simple arm and keeps `ignore_index` off the measured path.

**`+` and never `&+`.** `x &+ k` computes `x & k` in this toolchain, with no
compile error (`[[mojo-amp-plus-is-bitwise-and]]`), and it produced wrong
hashes in this repository twice on 2026-08-25 alone. Every addition inside
the generator is a plain `+` on `UInt64` and DEVIATION 1565 is the note that
says so at the call site.

The `>> 33` before the modulus takes the HIGH bits, because splitmix64's low
bits are the weaker half and `% V` at `V = 64` would otherwise be reading
them directly.

`train.s%04d.data.ids` is a recorded stage (DEVIATION 1572) so that a data
divergence localizes BEFORE it reaches a float. Two runs whose `data.ids`
already differ have a generator bug, not an arithmetic one, and that is a
completely different investigation.

---

## 7. What is NOT claimed

Written out because a claim with no boundary is a claim nobody can check.

1. **"Bitwise identical" means GIVEN THE SAME DRAWS.** In v1 the draws are a
   deterministic function of one integer seed and the step index, so the
   condition is satisfied by construction rather than by assumption. It stops
   being satisfied the moment anything stochastic enters -- dropout, a
   sampler, a shuffled loader, a data-parallel split whose order depends on
   arrival. None of those exists here and adding any of them would need this
   sentence rewritten, not just extended.
2. **FP32 only.** No bf16, no fp16, no mixed precision, no tf32. The GEMM
   profile is `mojolearn.identical.gemm.fp32.v1` and nothing here reaches
   outside it.
3. **One architecture, one layer, one shape.** Section 2's table. A different
   `d_model` is a different run and its digests are not comparable to these.
4. **One optimizer configuration per run.** The digest does not encode the
   configuration, so comparing two digests produced under different `lr` is
   meaningless and the harness cannot detect it. The card's header line
   records the configuration and a comparison must check it by eye.
5. **No claim about batch splitting.** Section 4.2 N7.
6. **No claim about resuming from a file.** Section 4.2 N6.
7. **No performance number.** A traced run drains the queue and copies four
   buffers to the host every step; `identical_optimizer_step` additionally
   downloads all four of `param`, `grad`, `m` and `v` for its refusal pass
   (DEVIATION 1573). That is a control-plane change of exactly the kind
   `HOST_AND_DEVICE.md` refuses to certify a timing under. **Any timing taken
   from this harness is fiction and quoting one is a defect.**
8. **No claim that the composed step is correct.** Section 1.1.

---

## 8. `train_step_check.mojo` -- the gate

Seven clauses. Each names what it would catch, because a clause that cannot
name its failure is decoration.

**(a) ONE STEP vs the composed host oracle, BITWISE.**
The host side is assembled from the oracles that already exist and no second
spelling of anything is written.

```
emb_forward_oracle                    embedding/mojo_only/embedding_oracle.mojo
transformer_block_oracle              transformer/mojo_only/transformer_oracle.mojo
gemm_oracle                           gemm/mojo_only/gemm_oracle.mojo        (lm_head, OP_NT)
ce_forward_oracle / ce_backward_oracle  training/mojo_only/loss_oracle.mojo
gemm backward via gemm_oracle           the a-call and b-call shapes from gemm_backward.mojo
transformer_block_backward_oracle     transformer/mojo_only/transformer_backward_oracle.mojo
emb_backward_oracle                   embedding/mojo_only/embedding_oracle.mojo
optimizer_step_oracle                 training/mojo_only/optimizer_oracle.mojo
```

Every INTERMEDIATE is compared, not only the final parameters (DEVIATION
1574). The gate reports the FIRST stage whose bits move, because a
disagreement in `param.out` alone tells you a step is wrong and nothing about
where.

The hazard this clause carries, and it is real. The device path uses
`LlamaDims` / `LlamaDeviceWeights` (in `transformer/ported/`) and the oracle
path uses `TransformerDims` / `TransformerWeights` (in
`transformer/mojo_only/`), because the two halves of that lane were written
by different agents and deliberately do not import each other. **Two shape
structs describing one model is a place to put a different number in each**
(DEVIATION 1552), so the check builds BOTH from one table of integers and
asserts field for field that they agree before it runs anything.

**(b) The negative controls** N1, N2, N3, N4 of section 4.2, each with its
required verdict, and N2 refused rather than passed at `N == 1`.

**(c) Movement**, N5, all four sub-clauses.

**(d) Step-count composition**, N6.

**(e) Hygiene.** `steps_run == N`; record count `== 2 + 4*N`; an empty trace
is an error and not a pass; refusals counted and asserted zero on the clean
arm; `MOJOLEARN_TRAIN_EXPECT_ARM` compared against the armed name.

**(f) The offset round trip.** Fill the flat buffer with a value that is a
distinct function of its index (`f(i) = f32_from_bits(0x3F800000 + i)` --
distinct per cell and in a range where every value is normal), unpack into
the eleven per-tensor buffers, pack back into a second flat buffer, and
assert equality on all `n_total` cells. Catches V7. **A uniform fill would
pass this with every offset wrong**, which is
`[[uniform-test-data-hides-permutation]]` applied to a copy.

**(g) Same-device determinism.** Run the whole N-step loop TWICE in one
process from the same seed and assert all four digests identical. Catches
V11 -- uninitialized memory folded into a hash differs on ONE machine, which
no cross-vendor comparison can see because it differs on all three.

---

## 9. The cross-vendor run, once clause (a) is green

Three legs, one command each, one hour of lease each
(`[[rented-gpus-self-expire]]`, `tools/runpod_guard.sh`).

```
MOJOLEARN_TRAIN_STEPS=8 MOJOLEARN_TRAIN_CKPT=/tmp/apple.ckpt  \
  MOJOLEARN_IDENTITY_TRACE=/tmp/apple.trace  <run train_loop>
```

then the same on the H100 and the MI325X, then

```
diff /tmp/apple.ckpt /tmp/nvidia.ckpt
python tools/identity_trace_diff.py /tmp/apple.trace /tmp/nvidia.trace
```

The `diff` gives the verdict; the trace differ gives the address of the first
step whose loss or digest moved, and `MOJOLEARN_IDENTITY_TRACE_DUMP=ckpt`
gives the raw bytes.

`train_step_check.mojo`'s clauses (b) through (g) run on ONE device and are a
precondition for the legs, not part of them. The legs cost GPU money; the
vacuity controls do not.

---

## 10. The build modes

`NUMERIC_IDENTICAL` is the mode the claim is about. `NUMERIC_FAST` runs the
same code on the same path with the pins compiled away, so it is a correct
training step that makes NO identity claim. Both are the same source, and the
run's card header records which one produced it. Reading the mode back from
the run rather than trusting the build command is
`[[mojolearn-shared-checkout-mode-flip]]` and the loop prints it.

There is no `if apple` in either new file and there is nowhere one could go.
Every kernel this lane adds is a byte copy. Every arithmetic decision belongs
to an op that already has a contract.

---

## 11. Sabotage arms this lane adds

None in a kernel. The two new kernels (`train_copy_range_kernel`,
`train_ulp_perturb_kernel`) contain no arithmetic, so there is nothing in
them to sabotage -- an arm that moves no float is a comment.

The arms live in the HARNESS and are the negative controls of section 4.2,
driven by `MOJOLEARN_TRAIN_ARM`. That is the same decision
`optimizer_check.mojo` made for `OPT_SAB_RESUME_REINIT` and
`OPT_SAB_MICROBATCH_SERIAL` (DEVIATION 1473) and for the same reason -- both
describe a defect in the CALLER's behavior, and a `comptime` switch inside a
kernel cannot express one.

---

## 12. OWED, and none of it is optional

1. **Compile all three files.** None has been read by a compiler. Section 13
   lists what this lane is least confident about.
2. **A gate for `transformer/mojo_only/transformer_backward.mojo`.** It is on
   the critical path of every step and nothing has compared it to its oracle.
   Section 1.1.
3. **A gate for `embedding/mojo_only/embedding_identical.mojo`.** Same. Its
   `PLAN_SORT` is additionally not written, so the embedding contract's
   plan-invariance clause -- the strongest evidence available that its
   arithmetic does not read the plan -- cannot run at all.
4. **Run clause (a) before any leg is rented.**
5. **A device-side non-finite refusal for the optimizer.**
   `opt_refuse_device_inputs` downloads all four buffers EVERY STEP. At this
   model size that is 214 KB per step and affordable; at any real size it is
   not, and a run that sets `MOJOLEARN_OPT_TRUST_INPUTS=1` to escape it is a
   DELIBERATE downgrade of the profile that must appear in the banner.
6. **A `kernel_matrix.mojo` row for this lane's `TRAIN_TPB`.** It is a literal
   128 (matching `LLAMA_TPB`) and the portable baseline column's cap is 128,
   so it happens to be legal -- by coincidence and not by construction. The
   optimizer lane owes the same fix for `OPT_TPB = 256`, which is NOT legal
   on that column.
7. **A checkpoint file format**, if resume-from-disk is ever to be tested.
   N6 tests resume within a process and says so.
8. **A second architecture** and a second shape, before the phrase "one
   source, three vendors" is used about training rather than about a GEMM.
9. **Print the per-tensor max displacement on the first run** and check it
   against section 5.3's paper arithmetic before believing any digest.
   `train_loop.mojo::main` prints it unconditionally for this reason.
10. **A device-buffer form of `llama_decoder_layer_backward`** (DEVIATION
    1577). That entry point takes its incoming gradient as a HOST
    `List[Float32]`, so every step downloads `d_h` and the callee re-uploads
    it. 512 floats here and irrelevant; a bus round trip in the hot path at
    any real size. Not in this lane's write set.
11. **`train_loop.mojo` and `train_step_check.mojo` share the parameter
    layout in three places** -- `train_offsets`, `unpack_params` and
    `pack_grads`. Three spellings of one layout is three chances to get it
    wrong, and clause (f) round-trips the pair rather than reading them. A
    single table that all three consume is the real fix and is owed.

DEVIATIONS 1578 through 1589 are unallocated and reserved for what the first
compile finds.

---

## 13. What this lane is least confident compiles

In order.

1. **`ctx.enqueue_function` with a kernel that takes an `Int32` offset plus
   two pointers.** The spelling is copied from `loss.mojo` and
   `optimizer.mojo` and has never been compiled either.
2. **Passing a STRUCT FIELD as a `mut` argument**, which
   `train_loop.mojo::train_step` does constantly (`tb.logits` into
   `identical_ce_forward_into`). `optimizer_oracle.mojo` avoids this
   deliberately and calls it "the kind of place-expression borrow this file
   cannot check without a compiler"; `transformer_backward.mojo` relies on it
   (`_route_a(ctx, bst.d_mlp_gated, ...)`). This lane follows the latter,
   because the alternative is a `train_step` with thirty-five parameters.
   **If the build fails, that is the first place to look.**
3. **`record_host[DType.uint64]` does not exist usefully** -- see 3.4. Caught
   before the compiler ran, and the loop records `u32` pairs instead.
4. **`List` copies.** Nothing copies implicitly in this toolchain. Every
   assignment of a `List` or a struct in the new files is `.copy()` or `^`,
   and a missed one is a compile error rather than a wrong answer, which is
   the good direction.
5. **String slicing.** `s[:n]` is refused; `s[byte=0:n]` works. `len(String)`
   is unsupported; `len(s.as_bytes())` and `s.byte_length()` work. The env
   integer parser hand-rolls its digits for this reason rather than calling
   anything that might not exist.
6. **`DeviceBuffer` lifetimes.** A buffer is dead at its `.unsafe_ptr()`.
   Every `_into` call in the loop is followed by the buffers staying live in
   a struct field or by an explicit `_ =` past the `synchronize()`. The
   places this is most likely wrong are the eleven per-tensor gradient
   buffers inside `LlamaBackwardStages`, which the pack step reads AFTER the
   backward returns.
