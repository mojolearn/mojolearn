# The backward pass of one Llama decoder block under
# `mojolearn.identical.transformer.fp32.v1`

Opened 2026-08-25. DEVIATIONS 1400 through 1449 are this lane's.

## STATUS

**COMPILED AND RUN 2026-09-03; THE GATE REFUSED TO CERTIFY, AND THAT IS THE
CORRECT OUTCOME.** All three files compiled cleanly on the FIRST attempt and
every preflight assertion passed. The gate then refused: its `d_out` fixture
cannot separate a fused multiply-add chain from an unfused one, so THREE
sabotage arms are unfalsifiable. **THE BLOCKER IS A FIXTURE, NOT THE
ARITHMETIC.** No column carries a certified card, and this is ONE APPLE
BOX.

**DEVIATION 1536, written 2026-09-03, NOT YET RUN.** The `d_out` generator's
exponent draw is narrowed from 17 values (`2^-8` through `2^8`) to three
(`2^-1` through `2^1`, magnitudes in `[0.5, 4)`). **The refusal's own stated
diagnosis -- "every product in the row is exactly representable" -- IS
WRONG**, and the correction is in `transformer_backward_check.mojo`'s THE
BINADE BUDGET block: `fixture_x` carries a 23-bit significand and the
generator a 24-bit one, so the products need 47 bits and essentially all of
them round. What fired is ABSORPTION -- with a 17-binade draw one partial of
a `d_model` fold can be `2^18` times another and swallows the one-`ulp`
difference the double rounding of any smaller term opens, so both spellings
land on the dominant term's rounding. Same scar as `checks/numerics.mojo`
row 9 (2026-08-23) and the same answer as `checks/ieee_arith_check.mojo`
ARM 6. **RUN OWED: `pixi run check-transformer-backward`.** Nothing here is
a verdict until that runs; `guard_d_out_separates` measures both clauses on
every run and refuses rather than reporting a pass.

    transformer/checks/transformer_backward_oracle.mojo   the host oracle
    transformer/checks/transformer_backward.mojo          the device spelling
    transformer/checks/transformer_backward_check.mojo    4,237 lines, main at :3851
    transformer/checks/transformer_fixture.mojo           1,169 lines, shared
    pixi.toml:1092   check-transformer-backward, registered since 2026-08-31

All four files exist and all of them compile. **No `transformer.backward.*`
card appears anywhere under `bench/results/`, on any column**, because the
gate has not certified. Every predicted count in section 5 is still on
paper.

**For contrast, and it is not evidence about the backward.** The FORWARD
profile in this directory ran on three columns on 2026-08-28 and its three
`transformer.identical.card` files are byte-identical, md5
`8ce661b469681b18fb5cf4d566ad78ff`, 28 of 28 card records on 262,634 cells, 13
fixture cases, 8 repeated launches, batch composition and sequence length
invariance each with its own firing control, decode equals prefill with a
misalignment control, 26 non-finite plants read back off the device, and all
13 sabotage arms firing at the stage each seam writes.

**This lane also needs a one-line edit to a file it may not touch.** The
forward card records `input.x` but `LlamaDeviceStages` holds no buffer for it;
the forward launcher uploads `x`, uses it and lets it die. So
`llama_decoder_layer_backward` takes `x_dev` as an EXPLICIT ARGUMENT and the
caller owns it. **CROSS-LANE REQUEST: an `x` field on `LlamaDeviceStages`,
written by the forward launcher, would delete that argument.** Parking `x` in a
backward scratch field instead would have been the wrong fix and nearly was:
at the point the first norm's backward runs, every scratch buffer still holds
a live gradient term, and reusing one would have read a gradient where a block
input belongs, giving a plausible, in-bounds, wrong `dx`.

---

## 0. The four findings

- **The prior art's shape does not survive.** `gemm/checks/gemm_backward.mojo`
  contains no arithmetic of its own, because the gradient of a matmul is two
  more matmuls at a different transpose. That was mechanically verifiable and
  it is why that lane was cheap. **This lane cannot be that lane.**
- **The organizing rule is inherited, not invented, DEVIATION 1401.**
  > **Route a derivative fold through gemm v1 if and only if its contraction
  > LENGTH is a configuration quantity. Pin it as a serial ascending chain
  > seeded `+0.0` if its length is a launch quantity**, the kv length, the
  > query count, or the token count of an activation gradient.

  Token counts appear as `k'` in every WEIGHT gradient and there the rule does
  not apply, because a weight gradient IS a sum over the batch and no spelling
  makes it otherwise; those are routed and their token dependence is DECLARED
  rather than defended. Applied mechanically, that rule and nothing else
  produces section 2.1's class column. It is worth stating as a rule because the
  alternative, deciding seam by seam, is how a lane ends up with two fold
  shapes and a clause it cannot explain.
- **The largest finding: S11's backward may NOT be routed.** The QK product was
  routed through gemm v1 (`OP_NT`, `k = head_dim`) precisely because
  `head_dim` is the same integer in prefill and decode (contract DEVIATION
  808). Its `dA` contracts over `n` and its `dB` over `m`, and for S11 those
  are the KV LENGTH and the QUERY COUNT. **Both are path dependent, so `dq`
  and `dk` are new pinned serial chains and not gemm calls.** DEVIATIONS 1402
  and 1403. Section 1.1 is the worked example.
- **The second finding runs the other way and is good news.** S19, the
  attention-weighted value sum, was deliberately NOT a gemm call (DEVIATION
  807), and its `dW` derivative contracts over `head_dim`, which IS path
  invariant. **So the forward's most hand-written seam has the backward that
  routes most cleanly**, DEVIATION 1405. The asymmetry between 1402 and 1405
  looks like an inconsistency and is the opposite of one; both are the same
  rule applied to whichever axis the derivative contracts.

---

## 1. What a backward call is, and what it is given

**A backward call takes** the incoming gradient `d(residual2.out)` at
`[M, dm]`, the block's weights, and the forward's SAVED stages for this same
call. It returns the eleven parameter gradients, the input gradient `dx`, and
`d(k_cache)` / `d(v_cache)` over the full `[0, S)` range.

### 1.1 The saved set, exactly (DEVIATION 1421)

Read from `TransformerStages` and nothing else. `input.x` for the RMSNorm 1
backward and the `dw_norm1` product; `norm1.sumsq` and `norm2.sumsq`, from
which `rstd` is RECOMPUTED; `norm1.out` and `norm2.out` as the `A` operand of
the four `dW` gemm calls; `q_rope.out` as the `dk_cache` chain's other
operand; `kv.k_cache` and `kv.v_cache`; `attn.weights` for the `dv_cache`
chain and the softmax backward's `y`; `attn.ctx` as `dW_o`'s `A` operand;
`residual1.out`; `gate_proj.out` as the SiLU derivative's argument, **not
`silu.out`**, 3.3(a); `up_proj.out` and `silu.out` for the two S21 products;
`mlp.gated` as `dW_down`'s `A` operand; and the rope table, read at the SAME
absolute positions.

**Two absences are findings and are stated as such.** `q_proj.out` and
`k_proj.out` are NOT NEEDED, because the RoPE backward is a linear map that
reads only the TABLE, so the pre-rotation activations are not on the backward
path at all. And **`attn.scores`, `attn.masked`, `attn.max`, `attn.exp` and
`attn.denom` are NOT NEEDED**, because the closed-form softmax backward reads
only `y` and `dy`: five materialized `[B,nh,L,S]` and `[B,nh,L]` buffers the
forward computed and the backward never touches, which is the clearest
consequence of DEVIATION 1406.

**The eager path already paid for this.** Contract section 6 accepted a
materialized score buffer as the cost of not fusing the stages away and argued
for it on instrument grounds. The backward now gets `attn.weights` for free
where a fused-softmax forward would have had to recompute it. **That is a cost
the forward paid that the backward collects, and it is the only place in this
lane where the reference-quality choice is also the cheap one.**

**What is NOT given.** No loss, no optimizer, no accumulation buffer, no
autograd tape. The incoming gradient arrives the way `dC` arrives in
`gemm_backward.mojo`, from somewhere this lane does not specify.

---

## 2. The seam table: ROUTING or NEW ARITHMETIC

**The two words are defined before the table, because a table whose column
headings are vague agrees with whatever the reader already thought.**

> **ROUTING.** The derivative is a call into an operation whose arithmetic is
> ALREADY CERTIFIED, at a shape that certificate covers, with no fold order
> and no fusion decision left to this lane. Two things qualify, a
> `mojolearn.identical.gemm.fp32.v1` cell and a `checks/numerics.mojo`
> primitive applied elementwise. A routed seam adds no clause and needs no
> sabotage of its own beyond the ones its host profile already carries.
>
> **NEW ARITHMETIC.** This lane must choose something: a fold ORDER, a fusion
> decision, a spelling with more than one legal association, or a sign
> convention. It owes a pinned order, a stated reason, a card stage, and a
> named sabotage with a predicted inert case.

Under those definitions an elementwise `pinned_mul` is ROUTING, because
DEVIATION 720's whole purpose is that no codegen may contract it, and a
two-operand add is ROUTING, because IEEE addition is bitwise commutative for
every non-NaN input and NaN is refused. **A THREE-operand sum is NEW
ARITHMETIC**, because `(a+b)+c` and `a+(b+c)` are different numbers.

`fold?` names the axis the derivative contracts over and whether its LENGTH is
the same in a prefill and a decode step. **That column is the whole argument.**

| # | forward seam | derivative | fold axis / path invariant? | class | pinned as | sabotage |
|---|---|---|---|---|---|---|
| **B23** | S23 `r2 = r1 + dp` | `d dp = g`, `d r1 += g` | none | ROUTING | bit copy, no rounding | -- |
| **B5d** | S5 `dp = gt @ Wdown^T` | `d gt`, `dW_down` | `dm` YES / `M` **NO** | ROUTING | `gemm_backward_a_call` / `_b_call` at `OP_NT` | inherits `BWD_UNTRANSPOSED`, `BWD_OPERAND_ORDER` |
| **B21** | S21 `gt = si * u` | `d si`, `d u` | none | ROUTING | two `pinned_mul` | -- |
| **B20** | S20 `si = silu(g)` | `d g = dsi * silu'(g)` | none | **NEW** | 3.3, five roundings, `identical_sigmoid` recomputed from `gate_proj.out` | `B20_SIGMOID_FROM_SILU`, `B20_SILU_DERIV_ALT_ASSOC`, `B20_SILU_DERIV_FUSED` |
| **B5gu** | S5 `g,u = n2_out @ W^T` | `d n2_out`, `dW_gate`, `dW_up` | `inter` YES / `M` **NO** | ROUTING + a 2-term add | gemm table; the fan-in add is unseeded, 2.4 | `B_FANIN_ZERO_SEED` |
| **B1-4b** | S1-S4 rmsnorm 2 | `d r1 +=`, `dW_norm2` | `dm` YES / `M` **NO** | **NEW (the `c` fold) + ROUTING (`dW`)** | 3.4 | `B01_DOT_UNFUSED`, `B01_DOT_DESCENDING`, `B_RSTD_RECOMPUTE_DESCENDING` |
| **B22** | S22 `r1 = x + o` | `d o`, `d x +=` | none | ROUTING | bit copy | -- |
| **B5o** | S5 `o = ctx @ Wo^T` | `d ctx`, `dW_o` | `dm` YES / `M` **NO** | ROUTING | gemm table | inherited |
| **B19w** | S19's `y` argument | `dy[t,j] = sum_d dctx[t,d] v[j,d]` | **`head_dim` YES** | **ROUTING** | gemm v1 `OP_NT` at `(L, S, hd)`, DEVIATION 1405 | `B10_DW_VIA_CHAIN`, which CANNOT FIRE at the gate shape |
| **B19v** | S19's `v` argument | `dv[j,d] = sum_{h,t} y[h,t,j] dctx[h,t,d]` | **query axis `L` NO** | **NEW** | serial ascending over `(h in group, t)`, `+0.0`, FUSED. DEVIATION 1404 | `B19_DV_VIA_GEMM` |
| **B18** | S14-S18 softmax | `dS = y*(dy - z)`, `z = sum_j dy_j y_j` | **kv axis `S` NO** | **NEW** | closed form, serial ascending over the ABSOLUTE key index from `+0.0`, FUSED. DEVIATIONS 1406, 1407 | `B18_SOFTMAX_DECOMPOSED`, `B18_ZFOLD_UNFUSED`, `B18_ZFOLD_DESCENDING` |
| **B13** | S13 `masked = scores + mask` | `d scores = d masked` | none | ROUTING (exact identity) | bit copy, NOT a zeroing | `B13_MASK_ZEROES_GRAD` |
| **B12** | S12 `scores = cell*scale` | `d cell = d scores * scale` | none | ROUTING | one `pinned_mul` per score | `B12_SCALE_INTO_DQ` |
| **B11q** | S11 `cell = qr @ kr^T` | `dq[t,d] = sum_j dcell[t,j] k[j,d]` | **kv axis `S` NO** | **NEW** | serial ascending over the ABSOLUTE key index, `+0.0`, FUSED. DEVIATION 1402 | `B11_DQ_VIA_GEMM` |
| **B11k** | same | `dk[j,d] = sum_{h,t} dcell[h,t,j] qr[h,t,d]` | **query axis `L` NO** | **NEW** | serial ascending over `(h in group, t)`, `+0.0`, FUSED. DEVIATION 1403 | `B11_DK_VIA_GEMM` |
| **Bkv** | the KV append | slice `[pos0, S)` out of `dk_cache` / `dv_cache` | none | ROUTING | bit copy; `[0, pos0)` is a HANDOFF, DEVIATION 1417 | -- |
| **B9-10** | S9, S10 RoPE | the TRANSPOSE rotation | none | **NEW SPELLING, INHERITED ROUNDING** | 3.2 | `B09_ROPE_TRANSPOSE_SIGN`, `B09_ROPE_HALVES_ADJACENT`, `B10_ROPE_BWD_FUSED` |
| **B5qkv** | S5 `q,k,v = n1_out @ W^T` | `d n1_out`, `dW_q/k/v` | `qw`/`kw` YES / `M` **NO** | ROUTING + a **3-term** fan-in | gemm table; the 3-term sum is NEW, 2.4 | `B_FANIN_ORDER_QKV_REVERSED`, `B_FANIN_ZERO_SEED` |
| **B1-4a** | S1-S4 rmsnorm 1 | `dx +=`, `dW_norm1` | `dm` YES / `M` **NO** | **NEW + ROUTING** | 3.4 | as B1-4b |

**Twenty rows. Eleven are ROUTING, six are NEW ARITHMETIC**, and the six are
the softmax's `z` fold (kv axis, not path invariant), the RMSNorm backward's
`c` fold (d_model, path invariant), `dq`'s chain (kv axis), `dk`'s chain
(query axis), `dv`'s chain (query axis) and the SiLU derivative (no fold, five
roundings, one order). Plus **three fan-in accumulations** of 2, 2, 2 and 3
terms, of which only the three-term one is order dependent, and **one new
spelling with an inherited rounding budget**, the RoPE transpose.

**Four of the six new folds are the same fold shape**, serial ascending over
one axis, seeded `+0.0`, one `identical_mul_add` per term. That is contract
S1's shape and contract S19's shape, and this lane introduces no seventh.

**The eleven routed rows are not filler.** All seven weight-matrix gradients,
all five activation gradients through a projection, both RMSNorm weight-vector
gradients through the GEMM lane's OWN bias-gradient trick (DEVIATION 851)
lifted intact, the attention-weight gradient, and every copy. **The routing is
done by CALLING `gemm_backward_a_call` and `gemm_backward_b_call`, not by
restating their table.** First, the table is six rows and a lane that retypes
it gets a fifty percent chance of a transpose error that is bit identical on
three vendors and wrong. Second, and load bearing, **calling them means this
lane's weight gradients inherit G1 and G2 and inherit `SAB_BWD_UNTRANSPOSED`
and `SAB_BWD_OPERAND_ORDER` as live arms through a new entry point**, which is
exactly what the gemm plan asks for when it says the forward sabotages must be
shown to fail THROUGH the backward entry points.

### 2.3 Why the prior art's shape does not survive, worked

For a forward `OP_NT` at `(m, n, k)` the gemm table gives
`dA = OP_NN(dC, B) @ (m, k, n)` with `k' = n` and
`dB = OP_TN(dC, A) @ (n, k, m)` with `k' = m`. For a projection that is fine,
since `n` is a layer width and `m` the token count. Apply the same two lines
to S11, whose forward call is `OP_NT` at `(L, S, head_dim)` per
`(batch, head)`.

    dq = OP_NN(dcell, k_head) @ (L, head_dim, S)     k' = S    the KV LENGTH
    dk = OP_TN(dcell, q_head) @ (S, head_dim, L)     k' = L    the QUERY COUNT

**Neither is `head_dim`.** The one property that licensed routing S11, that
its contraction length is the same integer in prefill and decode, is exactly
the property a derivative destroys, because `dA` contracts over the output
width and `dB` over the batch, **and S11's output width IS the kv axis.**

Under gemm v1, `contract_leaf_size` holds at 128 up to `k = 131072`, so a `dq`
routed through the gemm folds a row of 257 keys as `P = 3` leaves under a
balanced tree, and the same query's `dq` in a decode step at `S = 200` folds
as `P = 2`. **Different trees over the same numbers. The masked `+0.0` tail,
which is bitwise inert in a serial ascending chain, is NOT inert under a tree
whose SHAPE changes with the length.** The forward lane wrote that sentence
about S19 and used it to refuse a gemm call there; the backward lane finds the
same sentence forces the refusal at S11 too, and the pair of them is why the
routing fraction is eleven of twenty rather than twenty of twenty.

## 3. The new arithmetic, seam by seam, with the order pinned

### 3.2 RoPE backward, the transposed rotation (DEVIATION 1412)

The forward acts on the pair `(a_i, a_{i+half})` at column `ci = i` as
`out_i = a_i*c - a_{i+half}*s` and `out_{i+half} = a_{i+half}*c + a_i*s`, a
rotation `M` with `M^T = M^{-1}`, so the exact derivative is `M^T`. Written in
the forward's own `rotate_half` idiom, that is the SAME code with **the
negation moved from the lower half to the upper half**.

    forward     j <  half:  rot = -x[j+half]      j >= half:  rot = +x[j-half]
    backward    j <  half:  rot = +dout[j+half]   j >= half:  rot = -dout[j-half]

**Everything else is identical**: the same `ci = j mod half` column, the same
`cos` and `sin` rows at the same ABSOLUTE position, two `pinned_mul`, one
UNFUSED add `ftz(ftz(pa) + ftz(pb))`, the same three roundings. Contract
DEVIATION 811's refusal of an fma applies here for the same reason it applied
there: an fma is one rounding where the structure has three, and it is the
natural thing for a kernel author to write.

**So the rounding BUDGET is inherited, the SPELLING is new, and the one thing
this lane must pin is the sign convention.** It is one character in two
branches and it produces a plausible, correctly-shaped, wrong gradient.
`B09_ROPE_TRANSPOSE_SIGN` is that character, and it is INERT at absolute
position 0, where `sin` is exactly `+0.0` and `cos` exactly `1.0`.

**One sentence that should stop a reader expecting the wrong thing.** Our
backward is the exact transpose of the EXACT rotation, spelled with the
forward's rounding budget. **It is NOT the numerical adjoint of the ROUNDED
forward map, because a rounded map has no adjoint.** `rope_backward(rope(x))`
does not return `x`, and no clause here says it does.

### 3.3 SiLU backward (DEVIATION 1411)

`silu(x) = x / (1 + exp(-x))`, ONE division, contract S20, and NOT
`x * sigmoid(x)`. Pinned per element, with `x` the SAVED `gate_proj.out`.

    sg  = ftz(identical_sigmoid(ftz(x)))         DEVIATION 743, portable_sigmoidf
    r1  = ftz(ftz(1.0) - ftz(sg))                1 - sg          SUBTRACT
    r2  = ftz(pinned_mul(ftz(x), r1))            x * (1 - sg)    PRODUCT
    r3  = ftz(ftz(1.0) + ftz(r2))                1 + that        UNFUSED ADD
    r4  = ftz(pinned_mul(sg, r3))                sg * that       PRODUCT
    dx  = ftz(pinned_mul(ftz(dsi), r4))          dy * silu'      PRODUCT

Five roundings after the sigmoid, left to right, no fma anywhere. **Three
things are decisions, not transcription.**

**(a) `sg` is RECOMPUTED from `gate_proj.out`, never reconstructed from
`silu.out`.** `silu(x) = x * sigmoid(x)` is true in the reals and FALSE in
Float32 under this profile, because the forward's SiLU is one division and
`x * sg` is a division followed by a product. So `sg = silu_out / x` is a
different number, it is `0/0` at `x = 0`, and it costs a division to get a
worse answer. `portable_sigmoidf` and `portable_siluf` share the SAME
`d = portable_expf(-x) + 1.0` and only the numerator differs, so the
recomputed sigmoid is exactly `1/d` against the forward's `x/d`, the cleanest
relationship available. Sabotage `B20_SIGMOID_FROM_SILU`, **INERT for every
`x` at or above about 17**, where `1 + exp(-x)` rounds to exactly `1.0` so
`silu(x) == x` exactly and `silu_out / x == 1.0 == sigmoid(x)`. **A fixture
whose gate activations are all large is blind to this arm.**

**(b) The association is `sg * (1 + x*(1-sg))` and not `sg + x*sg*(1-sg)`.**
Equal in the reals, different in the last bit on ordinary inputs. Chosen
because it is the spelling with four operations instead of five and because it
is, as far as this lane can tell without a checkout, the one ATen writes.
Sabotage `B20_SILU_DERIV_ALT_ASSOC`, INERT wherever `sg` is exactly `1.0` and
at `x = 0`.

**(c) `1 + x*(1-sg)` is UNFUSED.** An `fma(x, r1, 1.0)` is one rounding where
this is two. Sabotage `B20_SILU_DERIV_FUSED`, INERT wherever the product is
exactly representable.

**THE REFERENCE'S OWN SPELLING COULD NOT BE VERIFIED, AND THAT IS A STATED GAP
RATHER THAN A DECISION MADE ON EVIDENCE.** There is no PyTorch checkout in
`/Users/andrewhendel/CascadeProjects/upstream/`, verified again on 2026-08-25;
the directory holds `cccl`, `cuml`, `cuvs`, `curand-headers`, `mamba`,
`modular`, `raft`, `scikit-learn` and `transformers` and no torch. So ATen's
`silu_backward` could not be read. This is contract 5.4's gap in a second
place and it is the second thing to check when a checkout lands.

### 3.4 RMSNorm backward (DEVIATIONS 1408, 1409, 1410, 1420)

**The reference has no fused RMSNorm backward to mirror.** `LlamaRMSNorm.
forward` (modeling_llama.py:62-67, re-read 2026-08-25) is six lines of
ordinary tensor ops, so autograd differentiates the ELEMENTWISE GRAPH node by
node. **This lane mirrors that graph rather than the algebraically-collapsed
formula**, for the same reason contract S10 refuses an fma: the reference
rounds where it rounds, and being righter than the reference is not the goal.

    rstd  RECOMPUTED, not saved:
          mean = ftz(identical_div(sumsq[t], Float32(dm)))
          rstd = ftz(identical_rsqrt(ftz(mean + RMS_EPS)))
    per j ASCENDING, and the fold in the same pass:
          dh_j = ftz(pinned_mul(ftz(dy_j), ftz(w_j)))          PRODUCT
          c    = ftz(identical_mul_add(dh_j, ftz(x_j), c))     FUSED, from +0.0
    the rsqrt / mean / pow tail, per ROW, once:
          r2 = ftz(pinned_mul(rstd, rstd)) ; r3 = ftz(pinned_mul(r2, rstd))
          cr3 = ftz(pinned_mul(c, r3))
          da  = ftz(pinned_mul(NEG_HALF, cr3))      rsqrt backward, -0.5*dr*r^3
          dv  = ftz(identical_div(da, Float32(dm))) mean backward, ONE division
    per j ASCENDING again:
          dx1_j = ftz(pinned_mul(dh_j, rstd))                  the x branch
          tx_j  = ftz(pinned_mul(TWO, ftz(x_j)))               pow(2) backward
          dx2_j = ftz(pinned_mul(dv, tx_j))
          dx_j  = ftz(ftz(dx1_j) + ftz(dx2_j))                 UNFUSED ADD
    the weight gradient, ROUTED:
          inner = ftz(pinned_mul(ftz(x), rstd))   recompute of S3
          P     = ftz(pinned_mul(ftz(dy), inner))
          dW[1 x dm] = gemm v1 OP_NN at (1, dm, M) with ones[1 x M] . P

**(a) `c` is a SERIAL ASCENDING FUSED chain over `d_model` from `+0.0`,
DEVIATION 1408**, contract S1's shape unchanged. `d_model` is a configuration
quantity, so no path-invariance argument forces this; the reasons are that it
gives the block ONE fold shape instead of two, that it keeps the norm out of
every launch-geometry argument, and that a reader can put contract S1 beside
it. **`pinned_block_sum` is REFUSED** for exactly the reasons contract 5.3
refuses it at S17: it is a halving tree, it pairs by stride, and it is a
different sum that would pass every launch-invariance gate.

**(b) `rstd` is RECOMPUTED from the saved `sumsq`, DEVIATION 1420.** A
recompute is bit exact, since `identical_div` and `identical_rsqrt` are pure
functions and the input bits are the card's own; saving `rstd` would add a
stage the forward does not have. Recomputing the SUM OF SQUARES instead of
reading `sumsq` is `B_RSTD_RECOMPUTE_DESCENDING`, **which must move nothing
when spelled ascending and must move `bwd.norm2.dx` when spelled descending**,
and that is what makes "a recompute is exact" a checked statement.

**(c) `-0.5` and `2.0` are SPELLED, not elided.** Both are exact scalings by a
power of two and both change the value, so unlike the `* attention_scaling` of
contract S8 (exactly `1.0` and bit inert on every input) they cannot be
dropped. They go through `pinned_mul` so no codegen may contract them into a
neighboring add. **They also do not cancel**: `-0.5 * z` then `* 2.0` is `-z`
for every finite `z` whose halving does not fall into the flush region, and
the two-step spelling is kept because the graph has two steps.

**(d) The weight gradient is the GEMM lane's ones-vector trick and its cost is
stated, DEVIATION 1410.** `dW_norm[j] = sum_t dy inner` is a
Hadamard-then-reduce, not a matmul, so the obvious reading is that it needs a
new pinned fold. It does not, for exactly the reason
`gemm/IDENTICAL_BACKWARD_PLAN.md` gives about `db`. **The cost this lane pays
that the GEMM lane did not: two roundings per term instead of one.**
`db[j] = sum dC[i,j]` had no product to round; `dW_norm[j] = sum dy*inner`
does, so the routed form rounds the product and then the add where a
hand-written `fma(dy, inner, acc)` would round once. **The routed form is
pinned anyway**, because one arithmetic under one certificate is worth more
than one rounding per term, and because the hand-written form would be a SIXTH
new fold in a lane that has five.

### 3.5 The fan-in accumulations (DEVIATION 1413)

| activation | terms | order pinned |
|---|---|---|
| `norm1.out` | 3 (q, k, v paths) | `((dq_term + dk_term) + dv_term)`, FORWARD-USE order, ASCENDING |
| `norm2.out` | 2 (gate, up) | gate then up; order irrelevant |
| `residual1.out` | 2 | norm branch then residual branch; irrelevant |
| `input.x` | 2 | norm branch then residual branch; irrelevant |

**The rule is FORWARD-USE ORDER**, not card order: the gradient terms
accumulate in the order the forward CONSUMED the activation. It is well
defined at every fan-in site, it agrees with card order at the only site where
the order matters, and it is what an autograd tape does. **A two-term fan-in
has no order to pin**, because IEEE addition is bitwise commutative on every
non-NaN input and NaN is refused at the door; the clause is written down
anyway, because "it does not matter" is exactly the kind of sentence that
turns out to matter. **A three-term fan-in does**, and `d(norm1.out)` is
pinned as `q`, then `k`, then `v` (`LlamaAttention.forward` :250, :251, :252),
left associative. Sabotage `B_FANIN_ORDER_QKV_REVERSED`.

**No `+0.0` SEED, and this is the decision.** The accumulation starts at the
FIRST term. The arithmetic reason: `+0.0 + x` equals `x` for every value
except `x = -0.0`, where it gives `+0.0`, IDENTITY_PATHS row 39's one-liner,
**so a seed LAUNDERS a negative-zero first term, and a negative zero is
reachable in a gradient**, since every masked attention cell produces one. The
reference reason: an autograd engine's `AccumulateGrad` installs the first
incoming gradient as the buffer and adds each subsequent one; it does not
allocate zeros first. Sabotage `B_FANIN_ZERO_SEED`, **predicted to move ZERO
cells on every fixture that does not PLANT a `-0.0`. Vacuous without a plant,
said out loud so nobody fires it, sees nothing and deletes it.**

---

## 4. The softmax backward, which gets its own section

### 4.1 The form

    z_row = sum over j ASCENDING, ABSOLUTE key index, from +0.0:
                z = ftz(identical_mul_add(ftz(dy_j), ftz(y_j), z))     FUSED
    dS_j  = ftz(pinned_mul(ftz(y_j), ftz(ftz(dy_j) - ftz(z))))
            one SUBTRACT, one PRODUCT, UNFUSED

where `y` is the forward's `attn.weights` and `dy` is `bwd.d_attn_weights`.
Two roundings per output cell plus one per fold term. `dS` is the gradient of
`attn.masked`, and the mask backward is the identity, so it is also the
gradient of `attn.scores`.

### 4.2 Why there is no max backward, no exp backward and no division backward (DEVIATION 1406)

**`softmax` is ONE autograd node**, and the checkout contains the evidence
directly: `transformers/src/transformers/pytorch_utils.py:50-58` defines
`softmax_backward_data(parent, grad_output, output)` whose body is
`from torch import _softmax_backward_data; return _softmax_backward_data(
grad_output, output, parent.dim, output.dtype)`. **A private ATen symbol
taking `(grad_output, output, dim, dtype)` and NOT taking the input, the max,
the exponentials or the denominator is a closed-form derivative of the whole
op.** The reference never differentiates through `max`, `exp` or the division.

**Writing the decomposed graph would be a different answer, and worse in a
specific way.** `y = softmax(s)` is invariant to `m`, so `dL/dm` is
analytically EXACTLY zero. Autograd does not know that: it computes
`dm = -sum_j d(s-m)_j`, a sum of terms that cancel in the reals and do not
cancel in Float32, and then SCATTERS that nonzero residue onto whichever
element the max selected. **That scatter is unpinnable under this profile**,
for three reasons. `identical_fmax` returns a VALUE, not an index, and
contract 5.1 leaves the fold TOPOLOGY free because the operation is exactly
associative, so a free topology and a defined argmax are incompatible. Ties
are reachable, since a masked row's tail is a run of identical `-FLT_MAX`
values and a row with two equal maxima is one plant away. And **`+0.0` against
`-0.0` is a MEASURED three-vendor split**, row 39, `max(+0.0, -0.0)` being
`-0.0` on Apple and `+0.0` on NVIDIA and AMD, so an argmax over such a row is
a vendor decision.

So the decomposed graph would move a numerically-nonzero quantity to an index
no clause in this profile defines. **REFUSED, and the refusal is a correctness
improvement that happens to also be the reference's behavior, which is the
only kind this lane accepts.** Sabotage `B18_SOFTMAX_DECOMPOSED`.

### 4.3 The `z` fold follows contract 5.3, and the argument carries

Contract 5.3's load-bearing reason for a serial ascending chain was that a
tail of exactly `+0.0` terms is bitwise inert in a chain seeded `+0.0` and is
NOT inert under gemm v1's `P = f(k)` topology. **That survives
differentiation, and it needs checking rather than assuming, because the terms
are different terms.** At a masked cell `y_j` is exactly `+0.0` (contract 7.1)
while `dy_j` is an ordinary nonzero number, so the fold term is
`fma(dy_j, +0.0, acc) = acc + (±0.0) = acc`, provided `acc` is not `-0.0`, and
the `+0.0` seed forbids that. **Inert.** So `z`, and everything downstream of
it, is independent of the kv length. FUSED, matching S1 and S19, because it is
a sum of products; `pinned_block_sum` is refused for the third time in this
profile and for the same reason.

**What contract 5.4's gap costs the backward.** Contract 5.4 records that
ATen's softmax could not be read and pins the DIVISION at S18 on rounding-count
grounds rather than evidence. The backward inherits that gap in a different
shape: the closed form `(dy - z) * y` reads only `y`, so **if the forward's
S18 were ever changed to a reciprocal multiply, the backward form would not
change at all**; it would simply be the derivative of a different forward.
**The gap does not compound**, which is the rare case where an open question
in the forward does not propagate.

---

## 5. State, decode, length, batch, and the chunk theorem

### 5.1 The masked tail is still bitwise inert, and everything rests on it

The backward needs contract 7.1's two facts about DIFFERENT quantities, so
they are proved again rather than cited.

**(i) A `+0.0`-seeded fma chain never holds `-0.0`.** `fma(a, b, acc)` returns
`-0.0` only when the exact value `a*b + acc` is a zero of negative sign, and
under round-to-nearest that happens only if both `a*b` and `acc` are `-0.0`,
since an exact cancellation of two nonzero opposites gives `+0.0`. The seed is
`+0.0` and the property is preserved at every step.

**(ii) Every masked cell's contribution is a signed zero, at all four folds.**
At `z`, `y_j` is exactly `+0.0` by contract 7.1. At `dq` and `dk`,
`dS_j = pinned_mul(+0.0, dy_j - z)` is `±0.0`, the mask backward is the
identity, and `dcell = pinned_mul(dS, scale)` keeps the zero. At `dv`,
`y_{t,j}` is exactly `+0.0`.

Combined, **every masked term is bitwise inert in every one of the four
chains.** So `z` and `dq` fold over the KEY axis and are therefore independent
of the kv length, exactly as `attn.denom` and `attn.ctx` are in the forward,
and that holds by construction with the gate existing to catch the
construction being violated. **`dk` and `dv` fold over the QUERY axis and are
NOT.**

### 5.2 The honest replacement for "decode equals prefill"

The forward's clause (d) says a token's output bits do not depend on how the
sequence was chopped up. **The backward cannot say that about every output and
it must not pretend to**, because two of its outputs are sums over the tokens
in the call.

    dq, and every activation gradient reachable only through dq
        -> INDEPENDENT of the kv length and of the sequence chopping.
    dk_cache, dv_cache, and every WEIGHT gradient
        -> A SUM OVER THE QUERIES IN THIS CALL. A key at slot j is read by
           every query at position >= j, and the queries in later calls are not
           in this call. So a per-call backward computes a PARTIAL gradient and
           the caller must add the partials. Clause (c-bwd) asserts they MOVE,
           as a negative property, the way gemm G5 asserts dB moves.

> **A backward pass's INPUT gradients can be made chopping-invariant. Its
> PARAMETER gradients cannot, because they ARE the sum over the chopping.**

The `[0, pos0)` slice, the gradient owed to tokens from earlier calls, is
available to whoever assembles a multi-call backward. **Assembling it is out of
scope, DEVIATION 1417.** What this lane hands over is a named, recorded,
complete partial.

### 5.3 The carried-accumulator chunk theorem (DEVIATION 1416)

**This is the property this lane gets for free that the GEMM lane cannot
have.** Every fold pinned in section 3 is a serial ascending chain seeded
`+0.0`. Split such a chain at ANY index, run the first piece, STORE the
accumulator, and run the second piece with that stored value as its SEED. The
result is the unsplit chain's result, bit for bit, **because it is literally
the same sequence of operations in the same order**, and `ftz` is idempotent
so the store and reload change nothing.

Contrast with the GEMM lane's aligned-split finding, and note two corrections
this document must not repeat loosely. **That lane's aligned-split measurement
used only TWO pieces**, and over two pieces a serial running sum and a
balanced tree are the same operation, so the measurement does NOT establish
that the accumulator must be the tree. And **the alignment rule is stricter
than "every boundary is a subtree boundary"**: accumulation across pieces is
LEFT ASSOCIATIVE, so at `P = 8` with four pieces on subtree edges it computes
`((n0+n1)+n2)+n3` while the unsplit tree computes `(n0+n1)+(n2+n3)`. **Two
aligned pieces are safe; four are not, even on subtree edges.**

Neither correction touches the theorem, because a carried accumulator does not
accumulate PARTIALS at all and there is nothing to associate. **The serial
chain's chunk property is exact for any number of pieces at any boundary**,
and that is a real advantage of the reference-quality fold shape over the
partitioned one, in the one place where the partitioned one is otherwise
strictly better.

**The price, stated rather than hidden.** The theorem covers the FOLDS. It
does not cover the eleven ROUTED rows, every one a gemm v1 call whose
activation-gradient `k'` is a layer width (fine) or whose weight gradient `k'`
is the token count (not fine).

> A backward chunked over the query axis, with carried accumulators, is bit
> exact for `z`, `dq`, `dk`, `dv` and everything downstream. It is bit exact
> for the ELEVEN weight gradients only at gemm v1's aligned two-piece splits,
> and never for four pieces. **A training run must declare its chunk
> schedule.**

### 5.4 Batch composition

Nothing in section 3 reads `B`, so the activation gradients are batch
composition invariant by construction and the gate exists to catch an
execution plan violating the construction. **The weight gradients are NOT, and
the gate asserts they move**, since their `k'` is `M = B*L` and summing over
more tokens IS the operation. `IDENTICAL_SSM_NOTES` records that batch
invariance for attention has been pursued in public (Thinking Machines,
*Defeating Nondeterminism in LLM Inference*); **batch invariance of a weight
gradient is not a thing anyone can have** and this document should not be read
as claiming a weaker version of a published result. What is this lane's is the
CROSS-VENDOR half, and only after a leg.

---

## 6. The card, the clauses, the sabotages

### 6.1 The stages, in card order (DEVIATION 1418)

One record per stage per backward call, in strictly the order the backward
computes them, which is the reverse of the forward's.
`tools/identity_trace_diff.py` aligns two traces by their TAG SEQUENCES before
it compares a single hash, so the strings and the order are both part of the
instrument and a reordered pair produces a WRONG ALIGNMENT rather than a
bigger diff.

     0 bwd.in.d_residual2   [M,dm]      the incoming gradient, as given
     1 bwd.d_down_proj_out  [M,dm]      B23, a copy
     2 bwd.d_mlp_gated      [M,inter]   B5d dA
     3 bwd.dW_down          [dm,inter]  B5d dB
     4 bwd.d_silu_out       [M,inter]   B21
     5 bwd.d_up_proj_out    [M,inter]   B21
     6 bwd.d_gate_proj_out  [M,inter]   B20
     7 bwd.dW_gate          [inter,dm]  B5gu dB
     8 bwd.dW_up            [inter,dm]  B5gu dB
     9 bwd.d_norm2_out      [M,dm]      B5gu dA x2, 2-term fan-in
    10 bwd.norm2.dot        [M]         B1-4b, the c fold
    11 bwd.dW_norm2         [dm]        B1-4b, the ones gemm
    12 bwd.norm2.dx         [M,dm]      B1-4b
    13 bwd.d_residual1      [M,dm]      B22, 2-term fan-in
    14 bwd.d_o_proj_out     [M,dm]      B22, a copy
    15 bwd.d_attn_ctx       [M,qw]      B5o dA
    16 bwd.dW_o             [dm,qw]     B5o dB
    17 bwd.d_attn_weights   [B,nh,L,S]  B19w, the ROUTED gemm
    18 bwd.attn.zdot        [B,nh,L]    B18, the z fold
    19 bwd.d_attn_masked    [B,nh,L,S]  B18, dS
    20 bwd.d_attn_scores    [B,nh,L,S]  B13, an exact identity
    21 bwd.d_qk_cell        [B,nh,L,S]  B12
    22 bwd.d_q_rope         [M,qw]      B11q, the dq chain
    23 bwd.d_k_cache        [B,nkv,S,hd] B11k, the dk chain, FULL [0,S)
    24 bwd.d_v_cache        [B,nkv,S,hd] B19v, the dv chain, FULL [0,S)
    25 bwd.d_k_rope         [M,kw]      the [pos0,S) slice, a copy
    26 bwd.d_v_proj_out     [M,kw]      the [pos0,S) slice, a copy
    27 bwd.d_q_proj_out     [M,qw]      B9-10, the RoPE transpose
    28 bwd.d_k_proj_out     [M,kw]      B9-10, the RoPE transpose
    29 bwd.dW_q             [qw,dm]     B5qkv dB
    30 bwd.dW_k             [kw,dm]     B5qkv dB
    31 bwd.dW_v             [kw,dm]     B5qkv dB
    32 bwd.d_norm1_out      [M,dm]      B5qkv dA x3, 3-term fan-in
    33 bwd.norm1.dot        [M]         B1-4a, the c fold
    34 bwd.dW_norm1         [dm]        B1-4a, the ones gemm
    35 bwd.norm1.dx         [M,dm]      B1-4a
    36 bwd.d_x              [M,dm]      2-term fan-in, THE OUTPUT

Thirty-seven stages. **Four choices a reader will question.**
`bwd.in.d_residual2` is a stage, because **two cards whose INPUTS differ are
diffing their fixtures**, so compare it before comparing any output stage.
`bwd.d_attn_scores` is recorded even though it is bitwise equal to
`bwd.d_attn_masked` by construction, and that is the whole reason: the
equality is what `B13_MASK_ZEROES_GRAD` breaks, and **an unrecorded identity is
an unchecked one.** `bwd.norm1.dot` and `bwd.norm2.dot` are recorded because
**a fold whose only evidence is the value it feeds cannot be localized**, and
the forward lane's own scar is that thirteen moved stages were absorbed by a
residual add and an output-only gate called the sabotage inert. And
`bwd.d_k_cache` and `bwd.d_v_cache` are recorded over `[0, S)`, the USED
prefix and not the allocation.

### 6.2 What "identical" is gated to mean, mirroring contract section 10

Every clause below is still a specification and not a report.

**(a)** device backward card equals host backward oracle card, bitwise, at
every stage and every shape. FAST arms RECORDED, not asserted, where they are
vendor shaped.
**(b)** the same bits on every one of 8 repeated launches.
**(c)** BATCH COMPOSITION and SEQUENCE LENGTH invariance **of the activation
gradients**, each half with its own firing control; **AND the negative half**,
that `bwd.dW_*`, `bwd.d_k_cache` and `bwd.d_v_cache` MUST MOVE under a change
of batch composition, with the host oracle predicting the moved cell count
BEFORE the device is asked. **A gate that only asserts the positive half would
pass on an implementation that computed constants.**
**(d)** THE CHUNK CLAUSE, replacing the forward's decode-equals-prefill. A
backward chunked over the query axis with CARRIED accumulators reproduces the
unchunked activation gradients bitwise at every split point including
misaligned ones; and the UNCARRIED control, two partial chains added at the
end, MUST MOVE at any chunk of 2 or more terms. **Without the negative control
this clause passes for ever on a broken chunker.**
**(e)** the row-39 audit. The incoming gradient is REFUSED BY NAME if it holds
a NaN or an infinity, tested BY BITS and not by compares because Metal flushes
compare operands. Plus the signed-zero audit at the masked cells of
`bwd.d_attn_scores`, whose predicted sign is `sign(dy_j - z)`, and at the
fan-in sites.
**(f)** CORRECTNESS, a different question from identity that no other clause
asks, because **a transpose error is bit identical on three vendors.** The
linear and bilinear seams admit an EXACT integer check: on a fixture whose
every operand is a nonzero integer in `[-8, 8]` with partial sums under
`2^24`, the derivative is exactly representable and can be written down
directly, with no epsilon to tune. That covers every gemm route, `dq`, `dk`,
`dv`, `dy`, the mask, the residuals, the fan-ins, and **the RoPE transpose AT
ABSOLUTE POSITION 0 ONLY**, where `cos` is exactly `1.0` and `sin` exactly
`+0.0`. **The nonlinear seams do not.** `silu'`, the `rsqrt` tail and the
softmax closed form carry transcendentals, so exact integers do not survive
them, and gating those needs a FLOAT64 DIRECTIONAL-DERIVATIVE reference at a
stated tolerance. **No such reference exists for the BACKWARD.**
`transformer_oracle.mojo` deliberately carries no float64 reference, and
`transformer/corpus/` exists (committed `82173423`, 2026-08-25) but is a
FORWARD per-stage corpus whose `gen_corpus.py` has never been executed and
which has no case data on disk. **So a float64 backward reference is owed from
scratch, and until it lands clause (f) covers the routing and not the
calculus.**
**(g)** every clause above falsifiable by a NAMED sabotage that fails a gate.

### 6.3 The sabotage set, with each arm's predicted INERT case

**The inert column is the point of the table.** An arm with no stated inert
case is an arm nobody has thought about, and the forward lane's experience is
that two of its thirteen arms pass clause (a) by construction and would have
been deleted as pointless if clause (d) had been written late.

| arm | first stage it must move | predicted INERT case |
|---|---|---|
| `B11_DQ_VIA_GEMM` | clause (c-bwd)/(d-bwd), NOT clause (a) at a fixed shape | **INERT at every `S <= 128`**, where `P(S) = 1` and gemm v1's cell IS the whole-`k` ascending chain from `+0.0`. Needs `S >= 129`; the profile's `L = 257` fixture is the only one that reaches it |
| `B11_DK_VIA_GEMM` | same | **INERT at `L <= 128` AND `n_rep == 1` together.** At `n_rep == 2` the routed form must accumulate two per-head gemms, a different association, so it fires on shape alone |
| `B19_DV_VIA_GEMM` | same | identical to `B11_DK_VIA_GEMM`. The backward twin of `S19_VALUE_SUM_VIA_GEMM`, inheriting its warning verbatim: without clause (d) it looks inert and gets deleted |
| `B10_DW_VIA_CHAIN` | `bwd.d_attn_weights` | **CANNOT FIRE AT THE GATE SHAPE AT ALL.** It replaces the routed `k' = head_dim` gemm with a hand chain, and the two agree bit for bit whenever `P(head_dim) == 1`, i.e. `head_dim <= 128`; the fixtures are `head_dim` 16 and 24. **DEVIATION 1405 is therefore pinned by ARGUMENT and not by a fired arm**, recorded here rather than discovered later |
| `B18_SOFTMAX_DECOMPOSED` | `bwd.d_attn_masked` | INERT at `S == 1`, where `y = 1`, `z = dy` and both forms give `0`, so a decode-against-empty-cache fixture is blind to it. **AND THE ARM IS WEAKER THAN THE CLAUSE IT GUARDS**: it separates the ASSOCIATION and NOT the argmax, because a device kernel cannot scatter an argmax residue without a float ATOMIC and an atomic is what this whole file refuses. **The argmax half of DEVIATION 1406 is unfirable by construction**, which is itself an argument for the refusal |
| `B18_ZFOLD_UNFUSED` | `bwd.attn.zdot` | INERT at `S == 1` and on any row whose products are all exactly representable |
| `B18_ZFOLD_DESCENDING` | `bwd.attn.zdot` | INERT at `S == 1` only. **NOT inert at `S == 2`**, because an fma keeps the second product exact and the two orders round different quantities; a two-term fma chain is order dependent where a two-term add chain is not |
| `B01_DOT_UNFUSED` | `bwd.norm2.dot` | INERT at `d_model == 1` and on exactly-representable rows |
| `B01_DOT_DESCENDING` | `bwd.norm2.dot` | INERT at `d_model == 1` only, same fma argument |
| `B_RSTD_RECOMPUTE_DESCENDING` | `bwd.norm2.dx` | INERT at `d_model == 1`. **The ASCENDING recompute must move NOTHING**, which is the arm's other half |
| `B20_SIGMOID_FROM_SILU` | `bwd.d_gate_proj_out` | **INERT for every `gate_proj.out >= about 17`.** Raises on a `0/0` at `x == 0`, which the refusal catches |
| `B20_SILU_DERIV_ALT_ASSOC` | `bwd.d_gate_proj_out` | INERT wherever `sg` is exactly `1.0` and at `x == 0` |
| `B20_SILU_DERIV_FUSED` | `bwd.d_gate_proj_out` | INERT wherever `x * (1 - sg)` is exactly representable |
| `B09_ROPE_TRANSPOSE_SIGN` | `bwd.d_q_proj_out` | **INERT AT ABSOLUTE POSITION 0**, per CELL and not per fixture, **so the gate must COUNT moved cells and RAISE if the count equals the position-0 population** |
| `B09_ROPE_HALVES_ADJACENT` | `bwd.d_q_proj_out` | INERT at `head_dim == 2`, and at position 0 |
| `B10_ROPE_BWD_FUSED` | `bwd.d_q_proj_out` | INERT at position 0 |
| `B12_SCALE_INTO_DQ` | `bwd.d_qk_cell` | **INERT AT EVERY POWER-OF-FOUR `head_dim`**, since at 16 and 64 the scale is exactly `0.25`/`0.125` and exact scaling commutes with the fma chain bitwise. **The `head_dim = 24` fixture contract section 3 already requires is what makes this arm fire at all** |
| `B13_MASK_ZEROES_GRAD` | `bwd.d_attn_scores` | Moves ONLY masked cells whose `dS` is `-0.0`, where `dy_j - z < 0`. Predicted to move roughly half the masked cells, **and the oracle must predict the exact count before the device is asked** |
| `B_FANIN_ZERO_SEED` | `bwd.d_norm1_out` | **PREDICTED TO MOVE ZERO CELLS ON EVERY UNPLANTED FIXTURE.** Vacuous without a planted `-0.0` |
| `B_FANIN_ORDER_QKV_REVERSED` | `bwd.d_norm1_out` | INERT wherever any two of the three terms are zero |
| `BWD_UNTRANSPOSED` (gemm lane's) | `bwd.dW_down` and the four routed activation gemms | the gemm lane's predicted mask re-derived at THESE shapes. INERT on any route whose correct op already is `OP_NN`; that lane measured 4 of 6 and the count here differs because every forward call in this profile is `OP_NT` |
| `BWD_OPERAND_ORDER` (gemm lane's) | same | Every forward call here is `OP_NT`, whose `dA` and `dB` both put `dC` LEFT, so **this arm is INERT through this lane's entry points entirely.** Recorded as such rather than run and reported green |

**Twenty-two arms, of which four are predicted inert at the current gate shape
and two of those cannot be made to fire without changing the profile's
fixtures.** That is the honest count and it is the number a reader should
carry, not "twenty-two arms exist".

---

## 7. Cost

Analysis. No number below is measured, and this profile publishes no timing.

**Memory.** 37 stage buffers against the forward's 30, with four large ones at
`[B, nh, L, S]` against the forward's six of that shape, so a backward call's
peak footprint is roughly 1.7x a forward call's, plus the saved forward stages
held for the whole interval. **The eager path is what makes this affordable and
also what makes it large**; a fused-softmax forward would save the four buffers
and lose every localizing stage on the card.

**Recomputation, three, all bit-exact re-executions of pinned seams**: `rstd`
from `sumsq`, `inner` from `x` and `rstd`, and `sigmoid` from `gate_proj.out`.
The sigmoid is the expensive one and it is the price of 3.3(a).

**Arithmetic.** Four chains of length `S` or `L` where the forward had two,
plus eleven gemm calls where the forward had seven plus `B*n_heads`, with the
per-head loop now run TWICE, so the launch count grows.

**The identity cost is REAL, POSITIVE and UNMEASURED, on every vendor. This
document does not say identity is free anywhere.** The `dq`, `dk` and `dv`
chains are one thread per output cell walking a dependent chain, latency bound
by construction, **and the routed alternative is faster and wrong.** The
RMSNorm weight gradient pays two roundings per term where a hand fold pays one,
and the SiLU derivative pays a fresh sigmoid per cell.

**One execution hazard, mirroring the GEMM lane's.** `choose_gemm_plan` picks
SPLITK only when `m' * n' <= 4096`, and every weight gradient here has `k'`
equal to the token count with `m' * n'` the weight matrix, so a middling weight
matrix with a large token count sits just the wrong side of that threshold.
Analysis, not measurement, it moves no bit, **and the fix if a measurement
finds it matters is a threshold change and never a different leaf rule.**

---

## 8. Deviations 1400-1449

    grep -rhoE "DEVIATIONS? 14[0-4][0-9](-14?[0-9]+)?" . | sort -u

| # | what | state |
|---|---|---|
| 1400 | the backward composition `transformer.bwd.fp32.v1` and its IDENTITY_PATHS row | RESERVED, no row written |
| 1401 | the organizing rule, section 0 | SPENT |
| 1402 | `dq` pinned as a serial ascending chain over the ABSOLUTE key index; gemm v1 REFUSED | SPENT, both code files |
| 1403 | `dk` pinned as a serial ascending chain over `(head in group, query)`; gemm v1 REFUSED | SPENT |
| 1404 | `dv` pinned likewise, the mirror of contract DEVIATION 807 | SPENT |
| 1405 | the attention-weight gradient ROUTED through gemm v1 `OP_NT` at `(L, S, head_dim)`, and the admission that its sabotage cannot fire at the gate shape | SPENT |
| 1406 | the softmax backward as the CLOSED FORM, and the refusal of the decomposed graph with its argmax scatter | SPENT |
| 1407 | the `z` fold pinned serial ascending over the ABSOLUTE key index from `+0.0`, FUSED | SPENT |
| 1408 | the RMSNorm `c` fold pinned serial ascending over `d_model` from `+0.0`, FUSED | SPENT |
| 1409 | the RMSNorm elementwise tail spelled node by node against the reference's autograd GRAPH, and `-0.5` and `2.0` spelled rather than elided | SPENT |
| 1410 | the two RMSNorm weight-vector gradients as a gemm v1 `OP_NN` against a ones vector, lifting DEVIATION 851, and the two-roundings-per-term cost | SPENT |
| 1411 | the SiLU derivative's five roundings and the refusal to reconstruct `sigmoid` from `silu.out` | SPENT |
| 1412 | the RoPE backward as the transposed rotation, the sign convention, the inherited rounding budget | SPENT |
| 1413 | the fan-in accumulations, NO `+0.0` seed, FORWARD-USE order, `q,k,v` ascending | SPENT |
| 1414 | the mask backward as an exact identity, and the refusal to zero the masked cells | SPENT |
| 1415 | the scale backward applied to the finished `dS`, not folded into `dq`'s chain | SPENT |
| 1416 | the carried-accumulator chunk theorem, and its NON-application to the eleven routed weight gradients | SPENT |
| 1417 | `dk_cache` / `dv_cache` emitted over the full `[0, S)` as a HANDOFF; a multi-call backward REFUSED as out of scope | SPENT |
| 1418 | the 37-stage backward card and its tags | SPENT |
| 1419 | the 22-arm sabotage set and the inert-case column | SPENT |
| 1420 | the recomputation set and the claim, gated, that a recompute is bit exact | SPENT |
| 1421 | the saved-stage set, and the finding that the closed-form softmax backward reads NONE of `attn.scores`, `attn.masked`, `attn.max`, `attn.exp`, `attn.denom` | SPENT |
| 1422 | the device file's launch geometry, one thread per output cell, no float atomic, no cross-thread float reduction, no shared memory | SPENT |
| 1423 | the incoming gradient refused nonfinite BY BITS before any recorded stage | SPENT |
| 1424 | the GQA head-group fold order inside `dk` and `dv`, heads ascending outer, queries ascending inner, ONE chain and not a sum of per-head partials | SPENT |
| 1425 | the oracle reuses `gemm_backward_a_call` / `_b_call` rather than restating the table, so the weight gradients inherit G1, G2 and the two `SAB_BWD_*` arms | SPENT |
| 1426 | `transformer_backward.mojo` lives in `checks/` and not `impl/`, because there is no upstream file to mirror; HuggingFace ships no backward and autograd generates it | SPENT |
| 1427 | the device backward reuses the forward device module's types and restates four private plumbing helpers locally rather than importing underscore-prefixed symbols | SPENT |
| 1428 | the device backward calls the synchronizing `identical_gemm` rather than the `_into` forms, so it carries no caller-owned workspace; **the cost is that gemm gate G7's workspace-sizing coverage is NOT inherited** | SPENT |
| 1430-1439 | RESERVED for the gate file. **It EXISTS, 4,237 lines, and it compiles and runs; these numbers are unspent because the gate has not yet certified** | |
| 1440-1449 | RESERVED for the float64 directional-derivative reference and the backward corpus, neither of which exists | |

---

## 9. Phases, and 10. what this buys

Phase A, this document plus the oracle and the device file, is DONE and
ungated. Every phase after it, compiling the two files, the fixture, clause
(a)'s 37-stage comparison, clause (f)'s exact-integer arm, clauses (b), (c) and
(e), clause (d)'s chunk theorem, the 22 sabotage arms, the float64 reference
and the three-vendor leg, is OWED and is listed in section 11. **Nothing after
the sabotage phase should be quoted as evidence and nothing before the leg is a
cross-vendor claim.** What this lane buys is one row of the gemm lane's ladder,
identical gradients for a BLOCK, moved from "not built" to "constructed and
ungated", and nothing else.

---

## 11. Not claimed, and what is owed

- **NOTHING HERE HAS BEEN RUN OR OBSERVED.** No card on any column.
- **Not identical training**, not identical models, not "bit-identical AI
  inference" and not its training equivalent. This lane moves one row of the
  gemm lane's ladder from "not built" to "constructed and ungated" and moves
  nothing else. **Every arrow in that ladder is an ONLY IF.**
- **Not a cross-vendor claim.** Not one line of THIS profile has run on one
  device, let alone three. The forward's three-column result is not evidence
  about the backward, and the standing lesson is that Apple and AMD agreed bit
  for bit through 302 stages while NVIDIA diverged at `tree001.winners.scores`.
- **Not agreement with PyTorch.** The fold orders, transcendentals, division
  and SiLU association are OURS, and ATen could not be read at all.
- **Not a verified reference spelling for two seams.** ATen's `silu_backward`
  and `_softmax_backward_data` are both PINNED ON ROUNDING-COUNT GROUNDS AND
  NOT ON EVIDENCE. Both are the first things to check when a checkout lands.
- **Not a correctness gate for the calculus.** Clause (f) covers the routing
  and the linear seams by exact integers; the nonlinear seams need a float64
  directional-derivative reference and there is none. **The bits are gated and
  the CALCULUS is not, yet.**
- **Not a multi-call backward**, DEVIATION 1417.
- **Not FlashAttention's backward**, not SDPA's, not a fused kernel of any
  kind.
- **Not batch-composition invariance of a weight gradient.** 4.4 says the
  opposite and the gate asserts it.
- **Not a performance number.**
- **One pin cannot be falsified at the gate shape**, DEVIATION 1405's
  `B10_DW_VIA_CHAIN`, inert for every `head_dim <= 128`, so the routed
  attention-weight gradient rests on section 2.1's argument and on gemm v1's own
  certificate and not on a fired arm.

**What T7 buys, and it is worth saying how.** The gemm lane's honest fraction
said the norm and attention backward family was the largest remaining piece
and that attention was "genuinely hard rather than merely unwritten, because
the standard fused attention backward accumulates `dK` and `dV` across thread
blocks with float atomics." **That obstacle is gone here.** This lane's `dk`
and `dv` are one thread per OUTPUT cell walking `(head in group, query)`, so
no two threads ever write the same address and **there is no float atomic
anywhere in this backward**, nor any shared memory, warp primitive,
cross-block reduction or `pinned_reduce` call. That is not cleverness; it is
the direct consequence of refusing the fused shape, and the price is the eager
path's memory and the chains' latency. **T7 was hard for FlashAttention's
backward. It is merely slow for the eager one.**

**OWED, in priority order.**

1. **A `d_out` FIXTURE THAT SEPARATES FUSED FROM UNFUSED.** This is what
   blocks certification: three of the 22 sabotage arms are unfalsifiable
   against the present fixture, so the gate refuses rather than printing a
   tick over them. DEVIATION 1536 is the attempt at it -- narrower binade
   budget, written 2026-09-03 -- and it is WRITTEN AND NOT RUN. This item
   closes when `guard_d_out_separates` prints SEPARATES on both clauses,
   not when the edit lands.
2. **The 22 sabotage arms, each checked against its predicted INERT case as
   well as its predicted firing.** This is the phase that decides whether any
   of the above is evidence.
3. **The float64 directional derivative**, clause (f)'s calculus half and the
   largest single owed item. `transformer/corpus/` is a FORWARD corpus, has
   never been executed, and has no case data on disk.
4. The three-vendor leg.
5. An IDENTITY_PATHS row, which DEVIATION 1400 reserves and which must not be
   written until item 4.

**What was least likely to compile** -- all of it compiled first try on
2026-09-03: the oracle's use of `gemm_backward_a_call`'s five-element
`Tuple` return, which the gemm lane's own header flags as one of the two
things most likely to need a syntax adjustment; the device file's per-head
gather-gemm-scatter loop, which allocates nothing but holds four buffers alive
across an `identical_gemm` call that synchronizes internally; and every `mut`
argument on a struct field, the shape that produces the `MutUntrackedOrigin`
versus `MutAnyOrigin` unification errors this repository has lost time to.
