# The training step, composed. `mojolearn.identical.train.step.fp32.v1`

Written 2026-08-25. DEVIATIONS 1550 through 1589 are this lane's.

## STATUS

**COMPILED, RUN, AND ALL SEVEN CLAUSES GREEN ON TWO COLUMNS, APPLE AND AMD.
THE CHECKPOINT HASHES MATCH ACROSS THE TWO. NO NVIDIA.**

`training/checks/train_loop.mojo` and `training/checks/train_step_check.mojo`
first compiled and ran at commit `5ce6eb17`. On 2026-08-28 the gate ran on an
Apple M4 and on an AMD MI325X at `steps N = 8`, `seed = 6085033151980924784`,
`n_total = 13376`, `J = 11`, mode `NUMERIC_IDENTICAL`, no sabotage armed, and
produced **the same digests on both boxes**.

    clean   h_all = 463245ce6c97e68d      whole(8) == split(4+4), both boxes
    N1 seed h_all = ed3015b9520337a2      DIFFERS, required
    N2 order h_all = 70b9b1a63630586a     DIFFERS, required
    N3 ulp  h_all = 940e6decf0a26127      DIFFERS, required

    (a) oracle 0 stages moved   (b) controls 0 failed   (c) movement 0 failed
    (d) composition 0 failed    (e) hygiene 0 failed    (f) layout 0 failed
    (g) determinism 0 failed

Clause (a) compared thirteen stages device against the host oracle BITWISE.
Clause (c) moved 13,376 of 13,376 parameter cells, every one of the eleven
tensors, with the embedding present-row and absent-row displacements
differing. The Apple leg also emitted `training-step.identical.card`, 34
records, md5 `bd703178`; **the AMD leg emitted the log and no card.**

**One MEASURED defect, DEVIATION 1580, on BOTH boxes.** N3's report line reads
*"REPORTED, NOT ASSERTED: the digest moved and step 1's loss did NOT. clean
4.7812595/0x40990014 vs ulp 4.7812595/0x40990014. The perturbation was carried
into the checkpoint without ever being computed with."* **That is the
`reached-but-inert` failure in the control itself**, and it is the reason 3.3
asserts both halves.

**What is still not closed.**

- **NO NVIDIA LEG**, so the three-machine claim in section 0 is not earned.
- **Two of the twelve stages have never been gated in isolation and both are
  on the critical path.** `transformer/checks/transformer_backward.mojo` has
  no run recorded on any column, and `embedding/checks/embedding_identical.
  mojo` has clause (a) only and no sabotage arm ever built.
- No `pixi.toml` task; the legs drove the gate by path.
- No checkpoint file format, so clause (d) tests resume WITHIN one process.

---

## 0. Why this file exists

Every identical op in this repository is gated IN ISOLATION. **None of that is
the claim.** The claim is

> run N steps of the same model on an M4, an H100 and an MI325X, and the
> checkpoint hashes match.

**Two ops that are each bitwise identical can still produce a divergent run**,
by three mechanisms none of which is visible to a per-op gate.

1. **Absorption.** What does not compose is a quantity that is identical only
   because a fixture kept it in a benign range. A training run walks the
   parameters somewhere no fixture chose, and the first subnormal, the first
   cancellation to `-0.0`, the first `logdenom` at a magnitude the loss
   fixture never held, arrives at step 40 and not at step 1.
2. **Accumulation across steps.** `m` and `v` are carried state, so a one-ULP
   divergence in a gradient at step 3 is not corrected at step 4, it is
   integrated. **The per-step gates all start from fresh state and cannot see
   this at all**, and `optimizer_check` found exactly this shape of hole in its
   own arms, since `OPT_SAB_MOMENT_LERP` is bit-inert at every first step.
3. **Step-count invariance.** The two `beta^t` spellings the optimizer
   contract separates agree exactly through `t = 6` and first differ at
   `t = 7`. **A composed run is the only thing that reaches `t = 7`.**

**This is not a training library. It is an instrument.** It trains a model
small enough to fit inside a one-hour GPU lease, on data it generates itself,
for the sole purpose of producing a number that can be compared across
machines.

---

## 1. The step, and what is gated underneath it

`M = B * L` token-major rows throughout.

| # | stage | entry point | status |
|---|-------|-------------|--------|
| 1 | data | `train_loop.mojo::train_batch_ids` | this lane |
| 2 | embed | `embedding_identical.mojo::identical_embedding_forward_into` | clause (a) on Apple + AMD, **no sabotage arm ever built** |
| 3 | block forward | `modeling_llama.mojo::llama_decoder_layer_forward` | **GATED**, 5 clauses, 13 arms, three columns |
| 4 | lm_head | `gemm_identical.mojo::identical_gemm_into`, `OP_NT` at `(M, V, d_model)` | **GATED**, three vendors |
| 5, 6 | loss and its backward | `loss.mojo::identical_ce_forward_into` / `_backward_into` | clause (a) and (e) on Apple + AMD |
| 7 | lm_head backward | `gemm_backward.mojo::identical_gemm_backward_a_into` / `_b_into` | ten gates green Apple + AMD, no sabotage fired |
| 8 | block backward | `transformer_backward.mojo::llama_decoder_layer_backward` | **WRITTEN, NO RUN RECORDED** |
| 9 | embedding backward | `embedding_identical.mojo::identical_embedding_backward_into` | as stage 2 |
| 10 | pack gradients | `train_loop.mojo::train_copy_range_kernel` | this lane |
| 11 | optimizer | `optimizer.mojo::identical_optimizer_step` | clause (a) and (e) on Apple + AMD |
| 12 | digest | `train_loop.mojo::snapshot`, `::digest_of_lists` | this lane |

### 1.1 The headline caveat (DEVIATION 1570)

**Stage 8 has never been compared to its oracle on any hardware, and stage 9
has been compared under one clause with no arm ever fired.** So state
precisely what matching checkpoint hashes would and would not demonstrate.

* They WOULD demonstrate that those stages produce the same bits on the
  columns that ran. That is a real result and it is what the legs are for.
* **They would NOT demonstrate that those stages are CORRECT. Three machines
  computing the same wrong gradient agree perfectly.**

Correctness against the oracle is the other half and it lives in clause (a),
which compares every intermediate of one step against a host oracle composed
of the existing oracles. **A run of the loop without clause (a) having passed
is not evidence of anything**, so the ordering requirement is clause (a) green
on one device, THEN the cross-vendor loop. Not the other way around and not
both at once on three rented machines.

### 1.2 What the step does NOT do

**No KV cache is carried between steps**, DEVIATION 1554; every step builds a
fresh `LlamaKVCache` at `pos0 = 0` and prefills, because a carried cache would
make a step depend on history through a second channel and disentangling a
cache divergence from a parameter divergence in a checkpoint hash is not
possible. **The `lm_head` is UNTIED from the embedding table**, DEVIATION
1555, because tying them makes one parameter tensor receive gradient from two
sites and the order those two contributions are summed in is a real arithmetic
decision with two answers that v1 does not make. One decoder layer, because
stacking adds nothing to the identity question the first layer does not
contain. No dropout, since the ported block has none. **No microbatching, no
gradient accumulation, no distributed anything, one device, one stream.**

---

## 2. The model, the FROZEN parameter order, and the checkpoint hash

**DEVIATION 1550.** The parameters live in ONE flat
`DeviceBuffer[DType.float32]`; `offsets` is the `List[Int]` of length `J + 1`
that `identical_optimizer_step` already takes, `j` IS the `param_id`, and its
ascending order is the cross-tensor summation order of optimizer contract
clause 2.3. **This order is part of the checkpoint hash specification.
Changing it changes every hash this profile has ever produced and may not be
done without a new profile version.**

    param_id  tensor    shape                        count at the v1 shape
    0         embed     [V, d_model]                 2048
    1         norm1_w   [d_model]                      32
    2         w_q       [n_heads*head_dim, d_model]  1024
    3         w_k       [n_kv*head_dim, d_model]      512
    4         w_v       [n_kv*head_dim, d_model]      512
    5         w_o       [d_model, n_heads*head_dim]  1024
    6         norm2_w   [d_model]                      32
    7         w_gate    [intermediate, d_model]      2048
    8         w_up      [intermediate, d_model]      2048
    9         w_down    [d_model, intermediate]      2048
    10        lm_head   [V, d_model]                 2048
                                            J = 11,  n_total = 13376

The v1 shape, every value FROZEN for the profile.

    V = 64   d_model = 32   n_heads = 4   n_kv = 2   head_dim = 8
    intermediate = 64   B = 2   L = 8   M = 16
    rms eps = 1e-6 / 0x358637BD      rope theta = 10000 / 0x461C4000

Everything is small on purpose. The largest buffer in the step is `dlogits` at
1024 floats, a hundred steps is minutes on any of the three machines, and the
whole run fits inside one lease with room to spare.

**DEVIATION 1553. The flat buffer is the state; the per-tensor buffers are a
view.** The loop refreshes the per-tensor buffers from the flat one at the top
of every step and copies the eleven gradient buffers into a flat `grad` at the
bottom, both pure element copies through `train_copy_range_kernel`. There is
no arithmetic in either, so neither can move a bit, **but a wrong offset gives
plausible, in-bounds, wrong numbers that are identical on all three vendors**,
which is the single most dangerous defect class in this file. Clause (f) is
the assertion that catches it and it is not optional.

## 3. The checkpoint hash (DEVIATION 1551), spec `mojolearn.identical.train.ckpt.v1`

FNV-1a64, offset basis `0xCBF29CE484222325`, prime `0x100000001B3`, folded ONE
BYTE AT A TIME over the LITTLE-ENDIAN bytes of the elements IN INDEX ORDER.
**Byte at a time on purpose**, because a word-at-a-time variant is a different
function and `tools/identity_trace_diff.py` recomputes this from a `.bin`
dump, so the writer and the reader must be the same spelling. The loop calls
`core/identity_trace.mojo::fnv1a64_bytes` rather than restating it.

    h_param  = h(FNV_OFFSET, param,   n_total)
    h_m      = h(FNV_OFFSET, m_state, n_total)
    h_v      = h(FNV_OFFSET, v_state, n_total)
    h_all    = h(h(h(FNV_OFFSET, param, n), m_state, n), v_state, n)
    h_p[j]   = h(FNV_OFFSET, param + offsets[j], offsets[j+1] - offsets[j])

`h_all` is the CHAINED fold, param then `m` then `v`, one continuous byte
stream, and **it is the single number two machines compare.** The other three
and the per-tensor digests exist so that a mismatch has an address rather than
a verdict. Concatenating the eleven tensor ranges in ascending `j` over a
contiguous flat buffer is the same byte sequence as `param[0 : n_total]`, so
`h_param` and the chained per-tensor fold coincide today; the per-tensor form
is written out anyway because it is what defines the hash if the layout ever
stops being contiguous.

**What is NOT hashed, and this list is the specification.** The digest is a
pure function of the parameter and optimizer-state BITS and of nothing else.
It does not read, and may never read, the iteration count `t` in any
representation; the wall clock or any timing; the device, vendor, driver
version, SM or CU count; the block size, grid shape, the plan
`choose_gemm_plan` picked, or the workspace size; the loss value, the learning
rate, the seed, or any configuration field; the length of the run; **or any
pointer, allocation address or buffer CAPACITY.** The last is a real hazard.
Every count is `n_total` or a tensor extent, **never `len(buf)`**: a buffer
allocated with slack and hashed to its capacity folds uninitialized memory
into the digest, which differs run to run on ONE machine and would make the
instrument report divergence everywhere. Clause (g) is what catches that.

**How it is emitted.** Four records per step, tags machine independent, the
step index zero padded to four digits (DEVIATION 1575) so lexical and numeric
order agree and two traces of the same length align tag for tag.

    train.s%04d.data.ids      i32  B*(L+1)   the batch, BEFORE it reaches a float
    train.s%04d.loss          f32  1
    train.s%04d.ckpt.tensors  u32  2*J       h_p[0..J-1], high word then low
    train.s%04d.ckpt.digest   u32  8         h_param, h_m, h_v, h_all

plus a step-zero checkpoint recorded BEFORE any step runs, carrying only the
two `ckpt.*` tags. Steps are numbered 1..N and `t` is one-based. **The record
count is therefore exactly `2 + 4*N`, and the gate asserts it** (DEVIATION
1563), because a trace with zero records compared against a trace with zero
records is machines agreeing about nothing and that failure is otherwise
completely silent. At `N = 8` that is 34, which is what both legs reported and
what the Apple card carries.

**The digests are recorded as `uint32` HIGH-WORD-THEN-LOW pairs and NOT as
`uint64`** (DEVIATION 1567), found by reading
`core/identity_trace.mojo::_dtype_name`, which handles `int64` and NOT
`uint64`, so a `u64` record would emit the dtype `?` and the differ's parser
accepts exactly six names. **This is the first defect the plan's own standard
predicted and then found, before any compiler ran.** Never as text:
`String(Float32)` does not round trip in this toolchain, and a value that went
through a decimal round trip anywhere could report agreement across a real
difference, which is the worst failure an instrument of this kind can have.

`h_all` is also printed as sixteen lowercase hex digits and written to a
sidecar named by `MOJOLEARN_TRAIN_CKPT`, so a cross-vendor comparison is
`diff` on three one-line files and does not require the trace differ.

---

## 4. What would make a matching hash VACUOUS

**This is the section that matters most.** A run whose parameters barely move
produces the same digest on any hardware and proves nothing. **DEVIATION 1559
is the rule: a run that has not executed the negative controls reports
VACUOUS, not PASS.** The word PASS does not appear in the output of a run that
produced no divergence when it was supposed to.

### 4.1 The failure-mode table

| # | failure | why the digests still match | caught by |
|---|---|---|---|
| V1 | the loop ran zero steps | nothing was computed | `steps_run == N`, and N5 |
| V2 | parameters do not move | the digest is of the initial state, a function of the seed alone | N5, per tensor |
| V3 | the digest reads the wrong buffer | reads something nothing updates | N3 plus N5 |
| V4 | the seed is never used; data is constant | all machines generate the same constant | **N1** |
| V5 | the step index never reaches the data generator | every step trains on the same batch | **N2** |
| V6 | `t` is frozen at 1 | bias correction constant, run still deterministic | N6, plus the `t >= 7` requirement |
| V7 | gradients packed at the wrong offsets | deterministic and wrong, identically on every vendor | clause (f), plus `h_p[j]` addressing |
| V8 | an op raised and the harness swallowed it | the digest is of the pre-failure state | refusal count asserted zero; no bare catch |
| V9 | the trace was disabled | zero records compared to zero records | record count `== 2 + 4*N` |
| V10 | a sabotage `-D` was misspelled | `is_defined` returns False, the build is clean, and "the arm did not bite" is recorded as a pass | `MOJOLEARN_TRAIN_EXPECT_ARM` (DEVIATION 1566) |
| V11 | the digest folded uninitialized tail memory | it would NOT match, it would differ everywhere including twice on one machine | clause (g) |
| V12 | every gradient underflows | parameters do not move, see V2 | N5's absolute floor |

**V10 is a repository scar and not a hypothetical.** `tools/gemm_ladder.sh:71`
recorded a clean build with a misspelled `-D` as an arm that did not bite,
which is the exact inverse of the truth. `optimizer_check.mojo` closed the same
hole with `MOJOLEARN_OPT_EXPECT_SABOTAGE` and this lane copies that spelling
deliberately.

### 4.2 The negative controls

Selected by `MOJOLEARN_TRAIN_ARM`. The clean run's digest is reported alongside
the arm results, always, in one block; **a digest printed on its own is not a
result**, and any row whose requirement is not met makes the whole run FAIL,
including the rows whose requirement is SAME.

**N1 `seed_plus_one`.** Same everything with the seed incremented. `h_all`
MUST differ. Catches V4, V1, and a data buffer allocated and never uploaded.
**Does NOT catch V5**, since a run that trains on step 1's batch forever still
responds to the seed.

**N2 `data_reverse`.** Same seed, per-step batches presented in reverse step
order. `h_all` MUST differ. Catches V5, and an optimizer whose update happens
to be order-free. **VACUOUS at `N == 1` and the gate REFUSES to run it there**
(DEVIATION 1560) rather than reporting a pass, because reversing a
one-element sequence is the identity and an arm that cannot fire reported as
an arm that passed is exactly the class of defect the first gate runs found
twice.

**N3 `ulp`.** Before step 1 the LAST float of `param_id = 4` (`w_v`) is
replaced by its one-ULP neighbor, done on the BITS and never by adding a small
float. `h_all` after ONE step MUST differ. Catches V3, a forward that does not
actually read `w_v`, and a parameter never written back into the flat state.
**The arm must ALSO move the recorded `train.s0001.loss`** (DEVIATION 1561),
and the gate asserts both: **if the digest moves and the loss does not, the
perturbation reached the checkpoint by being COPIED there rather than by being
computed with, and the arm has demonstrated the opposite of what it was armed
for.** `w_v` is chosen because every token's forward reads it, so the
perturbation cannot land on a row the batch never touched, and the LAST float
so an off-by-one that skips a tensor's tail is caught.

**MEASURED, BOTH BOXES, DEVIATION 1580.** The digest moved and step 1's loss
did NOT. **This control is currently in its own failure mode and the run
reports it as REPORTED, NOT ASSERTED rather than passing it.** It is the
largest open finding in this lane.

**N4 `zero_lr`.** `lr = 0`, `weight_decay = 0`, clipping off, AdamW.
`h_param` after N steps MUST EQUAL `h_param` at step 0, and `h_m` and `h_v`
MUST both DIFFER. At `lr = 0` AdamW's `decay_mul` is `1 - lr*wd = 1` exactly
and `step_size` is `0 / bc1 = +0.0`, so `p` is provably unchanged bit for bit
while `m` and `v` integrate normally. **The inverse control** (DEVIATION
1576), proving the harness is not simply scribbling on the parameters and that
`m` and `v` are genuinely written.

**N5 movement, the vacuity control and not a divergence control.** After N
steps of the CLEAN run, `n_changed >= 0.99 * n_total`; **for EVERY `param_id`
at least one element changed**, because a whole-model "something moved" check
passes with ten of eleven tensors frozen and a frozen tensor is exactly what a
structurally-zero gradient looks like; `max |p_j(N) - p_j(0)|` at least `1e-4`
per tensor; and **the embedding separation** (DEVIATION 1562), where a token
id that APPEARS in step 1's batch and one that does NOT must both move,
because weight decay moves every parameter, **and must move by DIFFERENT
amounts**, because if they moved identically the embedding gradient
contributed nothing and the whole of stage 9 is inert while the digest still
changes every step. That last clause is why `weight_decay` is nonzero on the
clean arm.

**N6 `split_4_6`.** Run 10 steps in one process, then from the same initial
state run 4, take a digest, and run 6 more from the carried state. `h_all`
after both must be EQUAL. Catches V6, an SGD `buf_initialized` flag that
resets, and a data generator that restarts its stream. **Stated honestly: this
is a resume WITHIN one process and not from a file.** There is no checkpoint
serialization in v1 (DEVIATION 1569), so `OPT_SAB_RESUME_REINIT`'s real form,
a checkpoint that drops the momentum flags on the way to disk, is not
reachable from here and this control does not test it.

**N7 batch splitting is NOT a control and must not be added.** Somebody will
want to assert that two microbatches of 8 produce the same digest as one batch
of 16. **That claim is false and asserting it would be a gate that fails for a
correct implementation.** `gemm_backward_b_call`'s `k` IS the token count, so
a weight gradient's contraction length changes when the batch is split,
`contract_partition` picks a different `(L, P)`, and the bits legitimately
move. **The embedding backward's carry reproduces a split bit-exactly and the
weight gradients do not**, and the two facts sitting next to each other is what
makes this worth writing down. DEVIATION 1568.

## 5. Making the parameters move, and proving they did

    optimizer AdamW   lr 1e-3   beta1 0.9   beta2 0.999   eps 1e-8
    weight_decay 0.01   max_norm 0.0 (clipping OFF, DEVIATION 1558)
    loss CeConfig(V, ignore_index = -100, REDUCTION_MEAN, eps = 0.0)
    steps N = 8 by default, MOJOLEARN_TRAIN_STEPS overrides

**Clipping is off on the clean arm** because with it ON one parameter's update
is a function of every gradient in the model, which is the reference's own
semantics and not a defect, but it couples every tensor to every other and
destroys the per-tensor localization `h_p[j]` exists to provide.

**Why `N = 8` and not 1, DEVIATION 1557.** Two of the optimizer's own arms are
structurally invisible below `t = 7`: `OPT_SAB_MOMENT_LERP` is bit-inert
whenever `m_prev` is exactly `+0.0`, which is EVERY first step from fresh
state, and the `beta^t` spellings agree exactly through `t = 6`. So a composed
run that stops at `t = 4` cannot exercise either, and 8 covers both with one
step of margin. `MOJOLEARN_TRAIN_STEPS = 1` is for clause (a)'s single-step
oracle comparison and is not a number to report identity from.

**The predicted displacement, ON PAPER.** Adam's per-element step is
`step_size * m_hat / (sqrt(v_hat) + eps)` and for a gradient of roughly
constant sign the ratio is close to `1`, so each parameter moves on the order
of `lr` per step, about `8e-3` over 8 steps, against weights initialized in
`[-0.5, 0.5]` where the FP32 ULP is about `6e-8`. That is roughly five orders
of magnitude above the last bit. **N5's floor is set at `1e-4`, two orders
below the prediction and four above the ULP, so if the observed motion is
anywhere near `1e-4` the prediction was wrong and the fixture is weak.** The
2026-08-28 legs measured per-tensor maxima between `0.0063` and `0.0077`,
which is the predicted band.

---

## 6. Data, and determinism without a clock (DEVIATION 1556)

Ids come from splitmix64 and nothing else. **No RNG reads the clock, the
process id, an address, or the environment beyond one integer seed.**

    key    = splitmix64(seed ^ (step_index * 0x9E3779B97F4A7C15))
    ids[i] = Int32( (splitmix64(key + UInt64(i)) >> 33) % UInt64(V) )

for `i` in `0 .. B*(L+1) - 1`. Each row of `B` draws `L + 1` ids; the first `L`
are the inputs and the last `L` the next-token targets, **so there is NO
ignored row, `count == M` exactly, and `ignore_index` stays off the measured
path.**

**`+` and never `&+`.** `x &+ k` computes `x & k` in this toolchain with no
compile error, and it produced wrong hashes in this repository twice on
2026-08-25 alone. Every addition inside the generator is a plain `+` on
`UInt64` and DEVIATION 1565 says so at the call site. The `>> 33` before the
modulus takes the HIGH bits, because splitmix64's low bits are the weaker half
and `% V` at `V = 64` would otherwise read them directly.

`train.s%04d.data.ids` is a recorded stage (DEVIATION 1572) so a data
divergence localizes BEFORE it reaches a float. **Two runs whose `data.ids`
already differ have a generator bug, not an arithmetic one, and that is a
completely different investigation.**

---

## 7. What is NOT claimed

1. **"Bitwise identical" means GIVEN THE SAME DRAWS.** In v1 the draws are a
   deterministic function of one integer seed and the step index, so the
   condition is satisfied by construction rather than by assumption. **It
   stops being satisfied the moment anything stochastic enters**, dropout, a
   sampler, a shuffled loader, a data-parallel split whose order depends on
   arrival. Adding any of them needs this sentence rewritten, not extended.
2. **FP32 only.** No bf16, fp16, mixed precision or tf32.
3. **One architecture, one layer, one shape.** A different `d_model` is a
   different run and its digests are not comparable to these.
4. **One optimizer configuration per run.** The digest does not encode the
   configuration, so comparing two digests produced under different `lr` is
   meaningless **and the harness cannot detect it.** The card's header line
   records the configuration and a comparison must check it by eye.
5. **No claim about batch splitting**, 4.2 N7.
6. **No claim about resuming from a file**, 4.2 N6.
7. **No performance number.** A traced run drains the queue and copies four
   buffers to the host every step, and `identical_optimizer_step` additionally
   downloads all four of `param`, `grad`, `m` and `v` for its refusal pass
   (DEVIATION 1573). **Any timing taken from this harness is fiction and
   quoting one is a defect.**
8. **No claim that the composed step is correct**, 1.1.
9. **Two columns is not three**, and the missing one is NVIDIA.

---

## 8. `train_step_check.mojo`, the gate, and 9. the cross-vendor run, 10. the build modes, 11. the arms

**Seven clauses**, each naming what it would catch, because a clause that
cannot name its failure is decoration.

**(a) ONE STEP against the composed host oracle, BITWISE.** The host side is
assembled from the oracles that already exist (`emb_forward_oracle`,
`transformer_block_oracle`, `gemm_oracle` for the lm_head, `ce_forward_oracle`
/ `ce_backward_oracle`, the gemm backward a-call and b-call shapes,
`transformer_block_backward_oracle`, `emb_backward_oracle`,
`optimizer_step_oracle`) and **no second spelling of anything is written.**
Every INTERMEDIATE is compared, not only the final parameters (DEVIATION
1574), and the gate reports the FIRST stage whose bits move.

*The hazard this clause carries is real.* The device path uses `LlamaDims` /
`LlamaDeviceWeights` and the oracle path uses `TransformerDims` /
`TransformerWeights`, because the two halves of that lane were written by
different agents and deliberately do not import each other. **Two shape
structs describing one model is a place to put a different number in each**
(DEVIATION 1552), so the check builds BOTH from one table of integers and
asserts field for field that they agree before it runs anything.

**(b)** the negative controls N1 through N4, each with its required verdict,
N2 refused rather than passed at `N == 1`.
**(c)** movement, N5, all four sub-clauses.
**(d)** step-count composition, N6.
**(e)** hygiene: `steps_run == N`; record count `== 2 + 4*N`; an empty trace
is an error and not a pass; refusals counted and asserted zero on the clean
arm; `MOJOLEARN_TRAIN_EXPECT_ARM` compared against the armed name.
**(f)** the offset round trip. Fill the flat buffer with a value that is a
distinct function of its index (`f(i) = f32_from_bits(0x3F800000 + i)`, distinct
per cell and every value normal), unpack into the eleven per-tensor buffers,
pack back into a second flat buffer, and assert equality on all `n_total`
cells. Catches V7. **A uniform fill would pass this with every offset wrong**,
which is `[[uniform-test-data-hides-permutation]]` applied to a copy.
**(g)** same-device determinism. Run the whole N-step loop TWICE in one
process from the same seed and assert all four digests identical. Catches V11,
uninitialized memory folded into a hash, **which no cross-vendor comparison
can see because it differs on all three.**

**The modes.** `NUMERIC_IDENTICAL` is the mode the claim is about;
`NUMERIC_FAST` runs the same code on the same path with the pins compiled
away, so it is a correct training step that makes NO identity claim. The run's
card header records which one produced it, and **the loop prints the mode back
rather than trusting the build command.** There is no `if apple` in either new
file and nowhere one could go: every kernel this lane adds is a byte copy, and
every arithmetic decision belongs to an op that already has a contract.

**Sabotage arms, none in a kernel.** The two new kernels contain no
arithmetic, so there is nothing in them to sabotage; **an arm that moves no
float is a comment.** The arms live in the HARNESS and are 3.2's negative
controls, driven by `MOJOLEARN_TRAIN_ARM`, the same decision
`optimizer_check.mojo` made for its two caller-behavior arms.

**The cross-vendor run, per leg.**

    MOJOLEARN_TRAIN_STEPS=8 MOJOLEARN_TRAIN_CKPT=/tmp/<box>.ckpt \
      MOJOLEARN_IDENTITY_TRACE=/tmp/<box>.trace  <run train_loop>
    diff /tmp/apple.ckpt /tmp/nvidia.ckpt
    python tools/identity_trace_diff.py /tmp/apple.trace /tmp/nvidia.trace

The `diff` gives the verdict; the trace differ gives the address of the first
step whose loss or digest moved. **Clauses (b) through (g) run on ONE device
and are a precondition for the legs, not part of them.** The legs cost GPU
money; the vacuity controls do not.

---

## 12. OWED, and none of it is optional

1. **RUN IT ON NVIDIA.** Two boxes agreed; the claim in section 0 names three.
2. **DEVIATION 1580.** N3's ulp arm moves the digest and does not move the
   loss, on both boxes. Until that is understood, the control is in the
   failure mode it was written to detect and it is reported rather than
   asserted.
3. **A gate for `transformer/checks/transformer_backward.mojo`.** It is on the
   critical path of every step, it is registered as
   `check-transformer-backward`, and nothing has compared it to its oracle on
   any column.
4. **Sabotage arms for `embedding/checks/embedding_identical.mojo`**, of which
   not one has ever been built, and its `PLAN_SORT`, without which the
   embedding lane's plan-invariance clause cannot run at all.
5. **A device-side non-finite refusal for the optimizer that does not download
   four buffers every step.** At this model size 214 KB per step is
   affordable; at any real size it is not, **and a run that sets
   `MOJOLEARN_OPT_TRUST_INPUTS=1` to escape it is a DELIBERATE downgrade of
   the profile that must appear in the banner.**
6. **A `kernel_matrix.mojo` row for `TRAIN_TPB`.** It is a literal 128
   matching `LLAMA_TPB` and the portable baseline column's cap is 128, **so it
   happens to be legal by coincidence and not by construction.** The optimizer
   lane owes the same fix for `OPT_TPB = 256`, which is NOT legal on that
   column.
7. **A `pixi.toml` task.**
8. **A checkpoint file format**, if resume-from-disk is ever to be tested.
9. **A second architecture and a second shape**, before the phrase "one
   source, three vendors" is used about training rather than about a GEMM.
10. **A device-buffer form of `llama_decoder_layer_backward`** (DEVIATION
    1577). That entry point takes its incoming gradient as a HOST
    `List[Float32]`, so every step downloads `d_h` and the callee re-uploads
    it. 512 floats here and irrelevant; a bus round trip in the hot path at any
    real size.
11. **`train_loop.mojo` and `train_step_check.mojo` share the parameter layout
    in three places**, `train_offsets`, `unpack_params` and `pack_grads`.
    Three spellings of one layout is three chances to get it wrong, and clause
    (f) round-trips the pair rather than reading them. **A single table that
    all three consume is the real fix.**

DEVIATIONS 1578 through 1589 were reserved for what the first compile would
find. The first compile and the first run happened at `5ce6eb17`; 1580 is what
they cost, and the rest of the reservation stands for the NVIDIA leg.
