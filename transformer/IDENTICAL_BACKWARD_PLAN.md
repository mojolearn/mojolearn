# The backward pass of one Llama decoder block under
# `mojolearn.identical.transformer.fp32.v1`

Opened 2026-08-25, the transformer BACKWARD lane, DEVIATIONS 1400-1449.

This document answers one question. *What does it take to make the gradient
of one Llama-shaped decoder block bit identical across Apple, NVIDIA and
AMD, how much of it is routing onto arithmetic that is already pinned, and
what new folds does this lane have to pin and gate itself?*

The forward is GATED. `transformer/checks/transformer_check.mojo` passes
clauses (a) through (e) on an M4: 13 fixture cases, 30 of 30 stages bit
identical on 262,634 cells, 8 repeated launches, batch composition and
sequence length invariance each with its own firing control, decode equals
prefill with a misalignment control, and 26 non-finite plants read back off
the device. All 13 sabotage arms fire, each at the stage its own seam
writes. **Nothing in THIS document has been compiled or executed.** Every
sentence below that says what a seam DOES is a statement about what the
source says; every sentence that says what two runs will AGREE ON is a
PREDICTION, and the point of writing the predictions down before the run is
that a disagreement afterwards is a FINDING rather than something
rationalized.

The three files this lane produced are

    transformer/IDENTICAL_BACKWARD_PLAN.md            this file
    transformer/checks/transformer_backward_oracle.mojo   the host oracle
    transformer/checks/transformer_backward.mojo          the device spelling

and **none of the three has been compiled, run, or had a single bit of its
output observed.** No gate file exists. No fixture exists. No card has been
emitted on any vendor. Section 11 lists what is owed.

---

## 0. The short answer, before the detail

- **The prior art's shape does not survive.**
  `gemm/checks/gemm_backward.mojo` contains no arithmetic of its own,
  because the gradient of a matmul is two more matmuls at a different
  transpose. That was mechanically verifiable and it is why that lane was
  cheap. **This lane cannot be that lane**, and section 2 is the table that
  says exactly where and why.
- **The organizing rule is one sentence and it is inherited, not invented.**
  Contract section 7.2 licenses routing a fold through gemm v1 when the
  CONTRACTION LENGTH is the same in prefill and decode, and refuses it when
  the length differs, because gemm v1's partition `P` is a pure function of
  `k`. **A derivative swaps which axis is contracted.** So a forward seam
  that was safe to route can have a backward that is not, and the single
  most important consequence of that sentence is section 2.3.
- **The largest finding: S11's backward may NOT be routed.** The QK product
  was routed through gemm v1 (`OP_NT`, `k = head_dim`) precisely because
  `head_dim` is the same integer in both paths (contract DEVIATION 808).
  Its `dA` contracts over `n` and its `dB` contracts over `m`, and for S11
  those are the KV LENGTH and the QUERY COUNT. Both are path dependent.
  **`dq` and `dk` are therefore new pinned serial chains, not gemm calls**,
  and the arm that proves it (`B11_DQ_VIA_GEMM`) is inert at every kv
  length at or below 128. DEVIATIONS 1402 and 1403.
- **The second finding runs the other way and is good news.** S19, the
  attention-weighted value sum, was deliberately NOT a gemm call
  (DEVIATION 807), and its `dW` derivative contracts over `head_dim`, which
  IS path invariant. So the forward's most hand-written seam has the
  backward that routes most cleanly. DEVIATION 1405. The asymmetry between
  1402 and 1405 looks like an inconsistency and is the opposite of one: both
  are the same rule applied to whichever axis the derivative contracts.
- **The softmax has no max backward, no exp backward and no division
  backward, and refusing to write them is a decision with a name.**
  `softmax` is ONE autograd node in the reference, not a graph
  (`torch._softmax_backward_data`, cited in the checkout at
  `transformers/src/transformers/pytorch_utils.py:50-58`), and its
  derivative is the closed form `dS = y * (dy - z)` with
  `z = sum_j dy_j y_j`. The decomposed graph is a DIFFERENT answer: it
  computes a max gradient that is analytically zero, numerically nonzero,
  and SCATTERED THROUGH AN ARGMAX whose tie break `identical_fmax` does not
  even define. DEVIATION 1406, and section 4.2 is the argument.
- **The masked tail stays bitwise inert in the backward**, by exactly
  contract 7.1's argument pointed the other way, and that is what makes a
  length-invariance clause possible for the ACTIVATION gradients at all.
  Section 5.1.
- **The chunk theorem is STRONGER here than in the GEMM lane and it is
  free.** Every fold this lane pins is a serial ascending chain seeded
  `+0.0`. A serial chain split anywhere and RESUMED WITH THE CARRIED
  ACCUMULATOR reproduces the unsplit chain bit for bit, because it is the
  same sequence of operations. Section 5.3, and it is stated beside the
  price: every WEIGHT gradient in the table is a gemm v1 call whose `k` is
  the token count, and those do not have the property.
- **This buys identical gradients for one block. It does not buy identical
  training.** Section 10.

---

## 1. What a backward call is, and what it is given

The forward, from `transformer_oracle.mojo::transformer_block_oracle`, with
`M = B*L`, `dm = d_model`, `qw = n_heads*head_dim`, `kw = n_kv*head_dim`,
`inter = intermediate`, `S = pos0 + L` the kv length after the append.

    n1_sumsq, n1_out = rmsnorm(x, w1)                        S1-S4
    q, k, v          = n1_out @ {Wq,Wk,Wv}^T                 S5   OP_NT k=dm
    qr, kr           = rope(q, pos), rope(k, pos)            S9,S10
    k_cache, v_cache = append(kr, v)                         copies
    cell             = qr_head @ kr_head^T                   S11  OP_NT k=hd
    scores           = cell * scale                          S12
    masked           = scores + maskval                      S13
    mx, e, denom, y  = softmax(masked)                       S14-S18
    ctx[t,d]         = sum_j y[t,j] * v_cache[j,d]           S19  chain
    o                = ctx @ Wo^T                            S5
    r1               = x + o                                 S22
    n2_sumsq, n2_out = rmsnorm(r1, w2)                       S1-S4
    g, u             = n2_out @ {Wgate,Wup}^T                S5
    si               = silu(g)                               S20
    gt               = si * u                                S21
    dp               = gt @ Wdown^T                          S5
    r2               = r1 + dp                               S23

**A backward call takes** the incoming gradient `d(residual2.out)` at
`[M, dm]`, the block's weights, and the forward's SAVED stages for this same
call. It returns the eleven parameter gradients, the input gradient `dx`,
and `d(k_cache)` / `d(v_cache)` over the full `[0, S)` range.

**The saved set, exactly** (DEVIATION 1421). Read from
`TransformerStages` and nothing else:

| saved stage | why the backward needs it |
|---|---|
| `input.x` | the RMSNorm 1 backward's `x`, and the `dw_norm1` product |
| `norm1.sumsq`, `norm2.sumsq` | `rstd` is RECOMPUTED from these, section 3.4 |
| `norm1.out`, `norm2.out` | the `A` operand of the four `dW` gemm calls |
| `q_proj.out`, `k_proj.out` | nothing. **NOT NEEDED**, and saying so matters: the RoPE backward is a linear map that reads only the TABLE, so the pre-rotation activations are not on the backward path at all |
| `q_rope.out` | the `dk_cache` chain's other operand |
| `kv.k_cache`, `kv.v_cache` | the `dq` chain and the `dW` gemm |
| `attn.weights` | the `dv_cache` chain and the softmax backward's `y` |
| `attn.scores`, `attn.masked`, `attn.max`, `attn.exp`, `attn.denom` | nothing. **The closed-form softmax backward reads only `y` and `dy`.** Five materialized `[B,nh,L,S]` and `[B,nh,L]` buffers that the forward computed and the backward never touches, which is the clearest single consequence of DEVIATION 1406 |
| `attn.ctx` | the `A` operand of `dW_o` |
| `residual1.out` | the RMSNorm 2 backward's `x` |
| `gate_proj.out` | the SiLU derivative's argument. **Not `silu.out`**, section 3.3 |
| `up_proj.out`, `silu.out` | the two S21 products |
| `mlp.gated` | the `A` operand of `dW_down` |
| the rope table | the RoPE backward reads the SAME `cos`/`sin` rows at the SAME absolute positions |

**ONE SAVED TENSOR IS MISSING ON THE DEVICE AND IT IS NAMED RATHER THAN
WORKED AROUND.** The forward card records `input.x`, but
`LlamaDeviceStages` does not hold a buffer for it -- the forward launcher
uploads `x`, uses it and lets it die. So `llama_decoder_layer_backward`
takes `x_dev` as an EXPLICIT ARGUMENT and the caller owns it.
**CROSS-LANE REQUEST**: an `x` field on `LlamaDeviceStages`, written by the
forward launcher, would delete that argument. It is a one-line edit to a
file this lane may not touch. Parking `x` in a backward scratch field
instead would have been the wrong fix and nearly was: at the point the
first norm's backward runs, every scratch buffer still holds a live
gradient term, and reusing one would have read a gradient where a block
input belongs -- a plausible, in-bounds, wrong `dx`.

**The eager path already paid for this.** Contract section 6 accepted a
materialized `[B, n_heads, L, S]` score buffer as the cost of not fusing the
stages away, and argued for it on instrument grounds. The backward now gets
`attn.weights` for free, where a fused-softmax forward would have had to
recompute it. That is a cost the forward paid that the backward collects,
and it is worth recording because it is the only place in this lane where
the reference-quality choice is also the cheap one.

**What is NOT given.** No loss. No optimizer. No accumulation buffer. No
autograd tape. The incoming gradient arrives the way `dC` arrives in
`gemm_backward.mojo`, from somewhere this lane does not specify.

---

## 2. The seam table: ROUTING or NEW ARITHMETIC

**The two words are defined before the table, because a table whose column
headings are vague is a table that agrees with whatever the reader already
thought.**

> **ROUTING.** The derivative is a call into an operation whose arithmetic
> is ALREADY CERTIFIED, at a shape that certificate covers, with no fold
> order and no fusion decision left to this lane. Two things qualify: a
> `mojolearn.identical.gemm.fp32.v1` cell, and a `checks/numerics.mojo`
> primitive applied elementwise. A routed seam adds no clause to the
> contract and needs no sabotage of its own beyond the ones its host
> profile already carries.
>
> **NEW ARITHMETIC.** This lane must choose something: a fold ORDER, a
> fusion decision, a spelling with more than one legal association, or a
> sign convention. A new-arithmetic seam owes a pinned order, a stated
> reason, a card stage, and a named sabotage with a predicted inert case.

Under those definitions an elementwise `pinned_mul` is ROUTING (the fusion
question is settled by `pinned_mul` existing at all -- DEVIATION 720's whole
purpose is that no codegen may contract it), and a two-operand add is
ROUTING (IEEE addition is bitwise commutative for every non-NaN input, and
NaN is refused). A THREE-operand sum is NEW ARITHMETIC, because
`(a+b)+c` and `a+(b+c)` are different numbers.

### 2.1 The table

`fold?` names the axis the derivative contracts over and whether its LENGTH
is the same in a prefill and in a decode step. That column is the whole
argument; everything else follows from it.

| # | forward seam | derivative | fold axis / path invariant? | class | pinned as | sabotage |
|---|---|---|---|---|---|---|
| **B23** | S23 `r2 = r1 + dp` | `d dp = g`, `d r1 += g` | none | ROUTING (copy) | bit copy, no rounding | -- |
| **B5d** | S5 `dp = gt @ Wdown^T` | `d gt`, `dW_down` | `dm` YES / `M` **NO** | **ROUTING** | `gemm_backward_a_call` / `_b_call` at `OP_NT`, then `gemm_oracle` / `identical_gemm` | inherits `BWD_UNTRANSPOSED`, `BWD_OPERAND_ORDER` |
| **B21** | S21 `gt = si * u` | `d si = dgt*u`, `d u = dgt*si` | none | ROUTING | two `pinned_mul` | -- |
| **B20** | S20 `si = silu(g)` | `d g = dsi * silu'(g)` | none | **NEW** | section 3.3, five roundings, `identical_sigmoid` recomputed from `gate_proj.out` | `B20_SIGMOID_FROM_SILU`, `B20_SILU_DERIV_ALT_ASSOC`, `B20_SILU_DERIV_FUSED` |
| **B5gu** | S5 `g,u = n2_out @ W^T` | `d n2_out`, `dW_gate`, `dW_up` | `inter` YES / `M` **NO** | ROUTING + a 2-term add | gemm table; the fan-in add is unseeded, section 3.5 | `B_FANIN_ZERO_SEED` |
| **B1-4b** | S1-S4 rmsnorm 2 | `d r1 +=`, `dW_norm2` | `dm` YES / `M` **NO** | **NEW (the `c` fold) + ROUTING (`dW`)** | section 3.4 | `B01_DOT_UNFUSED`, `B01_DOT_DESCENDING`, `B_RSTD_RECOMPUTE_DESCENDING` |
| **B22** | S22 `r1 = x + o` | `d o = dr1`, `d x += dr1` | none | ROUTING (copy) | bit copy | -- |
| **B5o** | S5 `o = ctx @ Wo^T` | `d ctx`, `dW_o` | `dm` YES / `M` **NO** | ROUTING | gemm table | inherited |
| **B19w** | S19's `y` argument | `dy[t,j] = sum_d dctx[t,d] v[j,d]` | **`head_dim` YES** | **ROUTING** | gemm v1 `OP_NT` at `(L, S, hd)`, DEVIATION 1405 | `B10_DW_VIA_CHAIN` (see 6.2: CANNOT FIRE at the gate shape) |
| **B19v** | S19's `v` argument | `dv[j,d] = sum_{h,t} y[h,t,j] dctx[h,t,d]` | **query axis `L` NO** | **NEW** | serial ascending over `(h in group, t)`, `+0.0`, FUSED. DEVIATION 1404 | `B19_DV_VIA_GEMM` |
| **B18** | S14-S18 softmax | `dS = y*(dy - z)`, `z = sum_j dy_j y_j` | **kv axis `S` NO** | **NEW** | closed form, serial ascending over the ABSOLUTE key index from `+0.0`, FUSED. DEVIATIONS 1406, 1407 | `B18_SOFTMAX_DECOMPOSED`, `B18_ZFOLD_UNFUSED`, `B18_ZFOLD_DESCENDING` |
| **B13** | S13 `masked = scores + mask` | `d scores = d masked` | none | ROUTING (exact identity) | bit copy, NOT a zeroing | `B13_MASK_ZEROES_GRAD` |
| **B12** | S12 `scores = cell*scale` | `d cell = d scores * scale` | none | ROUTING | one `pinned_mul` per score | `B12_SCALE_INTO_DQ` |
| **B11q** | S11 `cell = qr @ kr^T` | `dq[t,d] = sum_j dcell[t,j] k[j,d]` | **kv axis `S` NO** | **NEW** | serial ascending over the ABSOLUTE key index, `+0.0`, FUSED. DEVIATION 1402 | `B11_DQ_VIA_GEMM` |
| **B11k** | same | `dk[j,d] = sum_{h,t} dcell[h,t,j] qr[h,t,d]` | **query axis `L` NO** | **NEW** | serial ascending over `(h in group, t)`, `+0.0`, FUSED. DEVIATION 1403 | `B11_DK_VIA_GEMM` |
| **Bkv** | the KV append | slice `[pos0, S)` out of `dk_cache` / `dv_cache` | none | ROUTING (copy) | bit copy; `[0, pos0)` is a HANDOFF, DEVIATION 1417 | -- |
| **B9-10** | S9, S10 RoPE | the TRANSPOSE rotation | none | **NEW SPELLING, INHERITED ROUNDING** | section 3.2. Two `pinned_mul`, one UNFUSED add, the same table rows | `B09_ROPE_TRANSPOSE_SIGN`, `B09_ROPE_HALVES_ADJACENT`, `B10_ROPE_BWD_FUSED` |
| **B5qkv** | S5 `q,k,v = n1_out @ W^T` | `d n1_out`, `dW_q/k/v` | `qw`/`kw` YES / `M` **NO** | ROUTING + a **3-term** fan-in | gemm table; the 3-term sum is NEW, section 3.5 | `B_FANIN_ORDER_QKV_REVERSED`, `B_FANIN_ZERO_SEED` |
| **B1-4a** | S1-S4 rmsnorm 1 | `dx +=`, `dW_norm1` | `dm` YES / `M` **NO** | **NEW + ROUTING** | section 3.4 | as B1-4b |

### 2.2 The count, honestly

Twenty rows. **Eleven are ROUTING** (including all seven projection weight
gradients, all five projection activation gradients, every copy and every
elementwise product). **Six are NEW ARITHMETIC**, and they are exactly:

    the softmax's z fold                 kv axis      NOT path invariant
    the RMSNorm backward's c fold        d_model      path invariant
    dq's chain                           kv axis      NOT path invariant
    dk's chain                           query axis   NOT path invariant
    dv's chain                           query axis   NOT path invariant
    the SiLU derivative                  no fold      five roundings, one order

plus **three fan-in accumulations** (2, 2, 2 and 3 terms), of which only the
three-term one at `d(norm1.out)` is order dependent, and **one new spelling
with an inherited rounding budget**, the RoPE transpose.

**Four of the six new folds are the same fold shape**: serial ascending over
one axis, seeded `+0.0`, one `identical_mul_add` per term. That is contract
S1's shape and contract S19's shape, and this lane introduces no seventh.

### 2.3 Why the prior art's shape does not survive, in one worked example

`gemm/IDENTICAL_BACKWARD_PLAN.md` section 2.2 gives, for a forward `OP_NT`
at `(m, n, k)`:

    dA = OP_NN(dC, B) @ (m, k, n)     k' = n
    dB = OP_TN(dC, A) @ (n, k, m)     k' = m

and its own observation is that **`k'` is `n` for every `dA` and `m` for
every `dB`**. For a projection that is fine: `n` is a layer width and `m` is
the token count, and the token dependence is a stated, understood cost.

Apply the same two lines to S11. The forward call is `OP_NT` at
`(m, n, k) = (L, S, head_dim)`, one per `(batch, head)`. So

    dq = OP_NN(dcell, k_head) @ (L, head_dim, S)     k' = S      the KV LENGTH
    dk = OP_TN(dcell, q_head) @ (S, head_dim, L)     k' = L      the QUERY COUNT

**Neither is `head_dim`.** The one property that licensed routing S11 --
that its contraction length is the same integer in a prefill and in a decode
step -- is exactly the property a derivative destroys, because `dA`
contracts over the output width and `dB` over the batch, and S11's output
width IS the kv axis.

Under gemm v1, `P = ceil(k / contract_leaf_size(k))` and
`contract_leaf_size` holds at 128 up to `k = 131072`. So a `dq` routed
through the gemm folds a row of 257 keys as `P = 3` leaves under a balanced
tree, and the same query's `dq` in a decode step at `S = 200` folds as
`P = 2`. Different trees over the same numbers. The masked `+0.0` tail,
which is bitwise inert in a serial ascending chain, is NOT inert under a
tree whose SHAPE changes with the length.

That is contract 7.2's sentence, arrived at from the other side. The forward
lane wrote it about S19 and used it to refuse a gemm call there. **The
backward lane finds the same sentence forces the refusal at S11 too**, and
the pair of them is the reason this file's routing fraction is eleven of
twenty rather than twenty of twenty.

### 2.4 What DOES route, and it is not nothing

The eleven routed rows are not filler. They are:

- **All seven weight-matrix gradients** (`dW_q`, `dW_k`, `dW_v`, `dW_o`,
  `dW_gate`, `dW_up`, `dW_down`), each a `gemm_backward_b_call` row.
- **All five activation gradients through a projection** (`d n1_out` in
  three pieces, `d n2_out` in two, `d ctx`, `d gt`), each a
  `gemm_backward_a_call` row.
- **Both RMSNorm weight-vector gradients**, `dW_norm1` and `dW_norm2`,
  through the GEMM lane's OWN bias-gradient trick (DEVIATION 851) lifted
  intact: `dw[1 x dm] = ones[1 x M] . P[M x dm]`, an `OP_NN` at `(1, dm, M)`
  where `P[t,j] = pinned_mul(dy[t,j], inner[t,j])`. Section 3.4 carries the
  one thing that is NOT inherited: the ones trick makes the term
  `fma(1.0, p, acc)`, which is TWO roundings per term (the product `p`, then
  the add), where a hand-written `fma(dy, inner, acc)` would be one. This
  lane pins the two-rounding routed form, DEVIATION 1410.
- **The attention-weight gradient `dy`**, the one attention seam that
  routes, DEVIATION 1405.
- **Every copy**: both residual backward splits, the mask backward, the KV
  slice.

**The routing is done by CALLING `gemm_backward_a_call` and
`gemm_backward_b_call`**, not by restating their table. Two reasons and the
second is the load bearing one. First, the table is six rows and a lane that
retypes it gets a fifty percent chance of a transpose error that is bit
identical on three vendors and wrong (`gemm/IDENTICAL_BACKWARD_PLAN.md`'s
G2 exists to say so). Second, **calling them means this lane's weight
gradients inherit G1 and G2**, the two host-only gates that already assert
the table is the table, and inherit `SAB_BWD_UNTRANSPOSED` and
`SAB_BWD_OPERAND_ORDER` as live arms through a new entry point -- which is
precisely what that plan's section 5 asks for when it says the forward
sabotages must be shown to fail THROUGH the backward entry points as well.

---

## 3. The new arithmetic, seam by seam, with the order pinned

### 3.1 The organizing rule (DEVIATION 1401)

> **Route a derivative fold through gemm v1 if and only if its contraction
> LENGTH is a configuration quantity. Pin it as a serial ascending chain
> seeded `+0.0` if its length is a launch quantity -- the kv length, the
> query count, or the token count of an activation gradient.**
>
> Token counts appear as `k'` in every WEIGHT gradient and there the rule
> does not apply, because a weight gradient IS a sum over the batch and no
> spelling can make it otherwise. Those are routed, and their token
> dependence is DECLARED rather than defended (section 5.4).

Applied mechanically, that rule and nothing else produces the `class` column
of section 2.1. It is worth stating as a rule because the alternative --
deciding seam by seam -- is how a lane ends up with two fold shapes and a
clause it cannot explain.

### 3.2 RoPE backward: the transposed rotation (DEVIATION 1412)

The brief asks whether this is "exactly the forward's two-rounding add run
backwards, or a new spelling". **It is both, and the distinction is worth
being precise about.**

The forward (`transformer_oracle.mojo::apply_rope_into`, seams S9 and S10)
acts on the pair `(a_i, a_{i+half})` at column `ci = i` as

    out_i        = a_i * c - a_{i+half} * s          M = [[c, -s],
    out_{i+half} = a_{i+half} * c + a_i * s               [s,  c]]

`M` is a rotation, `M^T = M^{-1}`, and the exact derivative is `M^T`:

    da_i        =  dout_i * c + dout_{i+half} * s
    da_{i+half} =  dout_{i+half} * c - dout_i * s

Written in the forward's own `rotate_half` idiom, that is the SAME code with
**the negation moved from the lower half to the upper half**:

    forward     j <  half:  rot = -x[j+half]      j >= half:  rot = +x[j-half]
    backward    j <  half:  rot = +dout[j+half]   j >= half:  rot = -dout[j-half]

**Everything else is identical.** The same `ci = j mod half` column, the
same `cos` and `sin` rows at the same ABSOLUTE position, two `pinned_mul`
calls, one UNFUSED add `ftz(ftz(pa) + ftz(pb))`. Same three roundings.
Contract DEVIATION 811's refusal of an fma applies here for the same reason
it applied there: an fma is one rounding where the structure has three, and
it is the natural thing for a kernel author to write.

So: **the rounding BUDGET is inherited, the SPELLING is new, and the one
thing this lane must pin is the sign convention.** It is one character in
two branches and it produces a plausible, correctly-shaped, wrong gradient.
`B09_ROPE_TRANSPOSE_SIGN` is that character, and section 6 records that it
is INERT at absolute position 0, where `sin` is exactly `+0.0` and `cos` is
exactly `1.0`.

**One sentence that should stop a reader expecting the wrong thing.** Our
backward is the exact transpose of the EXACT rotation, spelled with the
forward's rounding budget. It is NOT the numerical adjoint of the ROUNDED
forward map, because a rounded map has no adjoint. `rope_backward(rope(x))`
does not return `x`, and no clause in this document says it does.

### 3.3 SiLU backward (DEVIATION 1411)

`silu(x) = x / (1 + exp(-x))`, ONE division, contract S20, and NOT
`x * sigmoid(x)` -- that refusal is the forward's and it has a consequence
here that is easy to miss.

Pinned, with every rounding numbered, per element, `x` being the SAVED
`gate_proj.out` and `dsi` the incoming gradient:

    sg  = ftz(identical_sigmoid(ftz(x)))         DEVIATION 743, portable_sigmoidf
    r1  = ftz(ftz(1.0) - ftz(sg))                1 - sg          SUBTRACT
    r2  = ftz(pinned_mul(ftz(x), r1))            x * (1 - sg)    PRODUCT
    r3  = ftz(ftz(1.0) + ftz(r2))                1 + that        UNFUSED ADD
    r4  = ftz(pinned_mul(sg, r3))                sg * that       PRODUCT
    dx  = ftz(pinned_mul(ftz(dsi), r4))          dy * silu'      PRODUCT

Five roundings after the sigmoid, left to right, no fma anywhere.

**Three things are decisions, not transcription.**

**(a) `sg` is RECOMPUTED from `gate_proj.out`, never reconstructed from
`silu.out`.** `silu(x) = x * sigmoid(x)` is true in the reals and FALSE in
Float32 under this profile, because the forward's SiLU is one division and
`x * sg` is a division followed by a product. So `sg = silu_out / x` is a
different number, it is `0/0` at `x = 0`, and it costs a division to get a
worse answer. `portable_sigmoidf` and `portable_siluf` share the SAME
`d = portable_expf(-x) + 1.0`; only the numerator differs. So the recomputed
sigmoid is exactly `1/d` against the forward's `x/d`, which is the cleanest
relationship available. Sabotage `B20_SIGMOID_FROM_SILU`, **INERT for every
`x` at or above about 17**, where `1 + exp(-x)` rounds to exactly `1.0`, so
`silu(x) == x` exactly and `silu_out / x == 1.0 == sigmoid(x)`. A fixture
whose gate activations are all large is blind to this arm.

**(b) The association is `sg * (1 + x*(1-sg))` and not `sg + x*sg*(1-sg)`.**
The two are equal in the reals and differ in the last bit on ordinary
inputs. This one is chosen because it is the spelling with four operations
instead of five and because it is, as far as this lane can tell without a
checkout, the one ATen writes. Sabotage `B20_SILU_DERIV_ALT_ASSOC`, INERT
wherever `sg` is exactly `1.0` (again `x >= ~17`) and at `x = 0`.

**(c) `1 + x*(1-sg)` is UNFUSED.** An `fma(x, r1, 1.0)` is one rounding
where this is two. Sabotage `B20_SILU_DERIV_FUSED`, INERT wherever the
product `x*(1-sg)` is exactly representable.

**THE REFERENCE'S OWN SPELLING COULD NOT BE VERIFIED, AND THAT IS A STATED
GAP RATHER THAN A DECISION MADE ON EVIDENCE.** There is no PyTorch checkout
in `/Users/andrewhendel/CascadeProjects/upstream/` -- verified again on
2026-08-25, the directory holds `cccl`, `cuml`, `cuvs`, `curand-headers`,
`mamba`, `modular`, `raft`, `scikit-learn` and `transformers` and no torch.
So ATen's `silu_backward` could not be read. This is contract section 5.4's
gap in a second place, it is recorded here in the same words, and it is the
second thing to check when a checkout lands.

### 3.4 RMSNorm backward (DEVIATIONS 1408, 1409, 1410, 1420)

**The reference has no fused RMSNorm backward to mirror.**
`LlamaRMSNorm.forward` (modeling_llama.py:62-67, re-read in the checkout on
2026-08-25) is six lines of ordinary tensor ops, so autograd differentiates
the ELEMENTWISE GRAPH node by node. This lane mirrors that graph rather
than the algebraically-collapsed formula, for the same reason contract S10
refuses an fma: the reference rounds where it rounds, and being righter than
the reference is not the goal.

The forward graph, per row `t`:

    v = x.pow(2)   ->   m = v.mean(-1)   ->   a = m + eps
    r = rsqrt(a)   ->   h = x * r        ->   y = w * h

Backward, pinned:

    rstd  RECOMPUTED, not saved:
          mean = ftz(identical_div(sumsq[t], Float32(dm)))
          rstd = ftz(identical_rsqrt(ftz(mean + RMS_EPS)))

    per j ASCENDING, and the fold in the same pass:
          dh_j = ftz(pinned_mul(ftz(dy_j), ftz(w_j)))          PRODUCT
          c    = ftz(identical_mul_add(dh_j, ftz(x_j), c))     FUSED, from +0.0

    the rsqrt / mean / pow tail, per ROW, once:
          r2  = ftz(pinned_mul(rstd, rstd))
          r3  = ftz(pinned_mul(r2, rstd))
          cr3 = ftz(pinned_mul(c, r3))
          da  = ftz(pinned_mul(NEG_HALF, cr3))      rsqrt backward, -0.5 * dr * r^3
          dv  = ftz(identical_div(da, Float32(dm))) mean backward, ONE division

    per j ASCENDING again:
          dx1_j = ftz(pinned_mul(dh_j, rstd))                  h = x*r, the x branch
          tx_j  = ftz(pinned_mul(TWO, ftz(x_j)))               pow(2) backward's 2*x
          dx2_j = ftz(pinned_mul(dv, tx_j))
          dx_j  = ftz(ftz(dx1_j) + ftz(dx2_j))                 UNFUSED ADD

    the weight gradient, ROUTED:
          inner_{t,j} = ftz(pinned_mul(ftz(x_{t,j}), rstd_t))  recompute of S3
          P_{t,j}     = ftz(pinned_mul(ftz(dy_{t,j}), inner_{t,j}))
          dW[1 x dm]  = gemm v1 OP_NN at (1, dm, M) with ones[1 x M] . P

**Four decisions.**

**(a) `c` is a SERIAL ASCENDING FUSED chain over `d_model` from `+0.0`,
DEVIATION 1408.** It is contract S1's shape unchanged. `d_model` is a
configuration quantity, so no path-invariance argument forces this; the
reasons are that it gives the block ONE fold shape instead of two, that it
keeps the norm out of every launch-geometry argument (one thread per token
row, no block fold), and that a reader can put contract S1 beside it.
`core/pinned_reduce.mojo::pinned_block_sum` is REFUSED here for exactly the
reasons contract 5.3 refuses it at S17: it is a halving tree, it pairs by
stride, and it is a different sum that would pass every launch-invariance
gate. FUSED because S1 is fused and because both are sums of products.
Sabotages `B01_DOT_UNFUSED` and `B01_DOT_DESCENDING`.

**(b) `rstd` is RECOMPUTED from the saved `sumsq`, DEVIATION 1420.** A
recompute is bit exact: `identical_div` and `identical_rsqrt` are pure
functions and the input bits are the card's own. Saving `rstd` instead would
add a stage the forward does not have and a buffer the forward does not
write. Recomputing the SUM OF SQUARES, rather than reading `sumsq`, is the
sabotage `B_RSTD_RECOMPUTE_DESCENDING` -- it must move nothing when spelled
ascending, and move `bwd.norm2.dx` when spelled descending, which is what
makes "a recompute is exact" a checked statement rather than a believed one.

**(c) `-0.5` and `2.0` are SPELLED, not elided.** Both are exact scalings by
a power of two and both change the value, so unlike the `* attention_scaling`
of contract S8 (which is exactly `1.0` and bit inert on every input) they
cannot be dropped. They go through `pinned_mul` so that no codegen may
contract them into a neighboring add, and so that a reader is not left
wondering whether the author elided them on purpose. **They also do not
cancel.** `-0.5 * z` then `* 2.0` is `-z` for every finite `z` whose halving
does not fall into the flush region, and the two-step spelling is kept
because the graph has two steps.

**(d) The weight gradient is the GEMM lane's ones-vector trick and its cost
is stated, DEVIATION 1410.** `dW_norm[j] = sum_t dy_{t,j} inner_{t,j}` is a
Hadamard-then-reduce, not a matmul, so the obvious reading is that it needs
a new pinned fold. It does not, for exactly the reason
`gemm/IDENTICAL_BACKWARD_PLAN.md` section 3.3 gives about `db`: materialize
`P = dy * inner` and the reduction becomes `ones . P`, an `OP_NN` at
`(1, dm, M)` whose leaf is `fma(ftz(1.0), ftz(P), acc)`, which is the
contract's ascending flushed chain and the contract's balanced tree with
nothing new to certify. **The cost this lane pays that the GEMM lane did
not: two roundings per term instead of one.** `db[j] = sum dC[i,j]` had no
product to round; `dW_norm[j] = sum dy*inner` does, so the routed form
rounds the product and then rounds the add, where a hand-written
`fma(dy, inner, acc)` would round once. The routed form is pinned anyway,
because one arithmetic under one certificate is worth more than one rounding
per term, and because the hand-written form would be a SIXTH new fold in a
lane that has five. It is recorded here rather than hidden.

### 3.5 The fan-in accumulations (DEVIATION 1413)

Four activations fan out in the forward and their gradients fan in:

| activation | terms | order pinned |
|---|---|---|
| `norm1.out` | 3 (q, k, v paths) | `((dq_term + dk_term) + dv_term)`, FORWARD-USE order, ASCENDING |
| `norm2.out` | 2 (gate, up) | gate then up (LMLP:175) -- order irrelevant, see below |
| `residual1.out` | 2 (norm2 backward's `dx`, and `d residual2`) | norm branch (LDL:321) then residual branch (LDL:323); irrelevant |
| `input.x` | 2 (norm1 backward's `dx`, and `d residual1`) | norm branch (LDL:306) then residual branch (LDL:317); irrelevant |

**The rule is FORWARD-USE ORDER**, not card order: the gradient terms
accumulate in the order the forward CONSUMED the activation. It is well
defined at every fan-in site, it agrees with card order at the only site
where the order matters, and it is what an autograd tape does.

**A two-term fan-in has no order to pin**, because IEEE addition is bitwise
commutative on every non-NaN input and NaN is refused at the door. The
clause is still written down, because "it does not matter" is exactly the
kind of sentence that turns out to matter.

**A three-term fan-in does.** `d(norm1.out)` is pinned as the forward-use
order `q`, then `k`, then `v` (`LlamaAttention.forward` :250, :251, :252),
left associative. Sabotage `B_FANIN_ORDER_QKV_REVERSED`.

**No `+0.0` SEED, and this is the decision.** The accumulation starts at the
FIRST term, not at `+0.0`. Two reasons. The arithmetic reason: `+0.0 + x`
equals `x` for every value except `x = -0.0`, where it gives `+0.0` --
IDENTITY_PATHS row 39's one-liner -- so a seed LAUNDERS a negative-zero
first term, and a negative zero is reachable in a gradient (every masked
attention cell produces one, section 5.1). The reference reason: an autograd
engine's `AccumulateGrad` installs the first incoming gradient as the buffer
and adds each subsequent one; it does not allocate zeros first. Sabotage
`B_FANIN_ZERO_SEED`, and section 6 records that it is **predicted to move
ZERO cells on every fixture that does not PLANT a `-0.0`** -- an arm that is
vacuous without a plant, said out loud so that nobody fires it, sees nothing
and deletes it.

---

## 4. The softmax backward, which gets its own section

### 4.1 The form

    z_row = sum over j ASCENDING, ABSOLUTE key index, from +0.0:
                z = ftz(identical_mul_add(ftz(dy_j), ftz(y_j), z))     FUSED

    dS_j  = ftz(pinned_mul(ftz(y_j), ftz(ftz(dy_j) - ftz(z))))
            one SUBTRACT, one PRODUCT, UNFUSED

where `y` is the forward's `attn.weights` and `dy` is `bwd.d_attn_weights`.

Two roundings per output cell plus one per fold term. `dS` is the gradient
of `attn.masked`; the mask backward is the identity, so it is also the
gradient of `attn.scores`.

### 4.2 Why there is no max backward, no exp backward and no division backward (DEVIATION 1406)

Contract 5.3 pins the FORWARD denominator as a serial ascending chain and
the brief asks whether the backward fold follows the same argument. **It
does, and the argument is stronger here.** But the prior question is which
fold there even is, and the answer is that the decomposed graph is not the
reference.

**`softmax` is ONE autograd node.** The checkout contains the evidence
directly: `transformers/src/transformers/pytorch_utils.py:50-58` defines
`softmax_backward_data(parent, grad_output, output)` and its body is
`from torch import _softmax_backward_data; return _softmax_backward_data(
grad_output, output, parent.dim, output.dtype)`. A private ATen symbol
taking `(grad_output, output, dim, dtype)` and NOT taking the input, the
max, the exponentials or the denominator, is a closed-form derivative of the
whole op. The reference never differentiates through `max`, `exp` or the
division.

**Writing the decomposed graph would be a different answer, and worse in a
specific way.** Take it node by node and the max backward is the problem.
`y = softmax(s)` is invariant to `m`, so `dL/dm` is analytically EXACTLY
zero. Autograd does not know that: it computes `dm = -sum_j d(s-m)_j`, a sum
of terms that cancel in the reals and do not cancel in Float32, and then
SCATTERS that nonzero residue onto whichever element the max selected.

That scatter is unpinnable under this profile, and saying why is the point:

1. **`identical_fmax` returns a VALUE, not an index.** Contract 5.1 pins the
   max as an order-free fold over a total-order key and explicitly leaves
   the fold TOPOLOGY free, because the operation is exactly associative. An
   argmax is not associative and its answer under ties depends on the fold
   shape, so a free topology and a defined argmax are incompatible.
2. **Ties are reachable.** A masked row's tail is a run of identical
   `-FLT_MAX` values; a row with two equal maxima is one plant away.
3. **`+0.0` against `-0.0` is a MEASURED three-vendor split.**
   IDENTITY_PATHS row 39, 2026-08-23: `max(+0.0, -0.0)` is `-0.0` on Apple
   and `+0.0` on NVIDIA and AMD. Contract 5.1 says a row of attention scores
   reaches both zero signs easily. An argmax over such a row is a vendor
   decision.

So the decomposed graph would move a numerically-nonzero quantity to an
index no clause in this profile defines. **REFUSED**, and the refusal is a
correctness improvement that happens to also be the reference's behavior,
which is the only kind this lane accepts. Sabotage
`B18_SOFTMAX_DECOMPOSED`.

### 4.3 The `z` fold follows 5.3, and here is why the argument carries

Contract 5.3's two reasons for the serial ascending chain were (1) one fold
shape for the block, and (2) the load-bearing one, that a tail of exactly
`+0.0` terms is bitwise inert in a chain seeded `+0.0` and is NOT inert
under gemm v1's `P = f(k)` topology.

Reason (2) survives differentiation **and it needs checking rather than
assuming, because the terms are different terms.** Section 5.1 does the
check: at a masked cell `y_j` is exactly `+0.0` (contract 7.1) while `dy_j`
is an ordinary nonzero number, so the fold term is
`fma(dy_j, +0.0, acc) = acc + (+-0.0) = acc`, provided `acc` is not `-0.0`,
and the `+0.0` seed forbids that. **Inert.** So the same theorem holds and
`z` -- and therefore every activation gradient downstream of it -- is
independent of the kv length.

FUSED, matching S1 and S19, because it is a sum of products.
`pinned_block_sum` is refused for the third time in this profile and for the
same reason.

### 4.4 What section 5.4's gap costs the backward

Contract 5.4 records that ATen's softmax could not be read and pins the
DIVISION at S18 on grounds of rounding count rather than evidence. The
backward inherits that gap in a different shape: the closed form
`(dy - z) * y` reads only `y`, so **if the forward's S18 were ever changed
to a reciprocal multiply, the backward form would not change at all** -- it
would simply be the derivative of a different forward. The gap does not
compound. That is worth one sentence because it is the rare case where an
open question in the forward does not propagate.

---

## 5. State, decode, length, batch, and the chunk theorem

### 5.1 The masked tail is still bitwise inert, and everything rests on it

Contract 7.1 proves the forward's masked attention weight is exactly `+0.0`
and that a `+0.0`-seeded chain never holds `-0.0`. The backward needs the
same two facts about DIFFERENT quantities, so they are proved again rather
than cited.

**(i) A `+0.0`-seeded fma chain never holds `-0.0`.** `fma(a, b, acc)`
returns `-0.0` only when the exact value `a*b + acc` is a zero of negative
sign, and under round-to-nearest that happens only if both `a*b` and `acc`
are `-0.0` (an exact cancellation of two nonzero opposites gives `+0.0`).
The seed is `+0.0` and the property is preserved at every step, so `acc` is
never `-0.0` and `acc + (+-0.0) == acc` at every step.

**(ii) Every masked cell's contribution is a signed zero, at all four
folds.** At a masked `(t, j)`:

| fold | term | why it is a signed zero |
|---|---|---|
| `z` | `fma(dy_j, y_j, acc)` | `y_j` is exactly `+0.0`, contract 7.1 |
| `dq` | `fma(dcell_{t,j}, k_{j,d}, acc)` | `dS_j = pinned_mul(+0.0, dy_j - z)` is `+-0.0`; the mask backward is the identity; `dcell = pinned_mul(dS, scale)` keeps the zero |
| `dk` | `fma(dcell_{t,j}, qr_{t,d}, acc)` | same `dcell` |
| `dv` | `fma(y_{t,j}, dctx_{t,d}, acc)` | `y_{t,j}` is exactly `+0.0` |

Combined with (i), **every masked term is bitwise inert in every one of the
four chains.** So:

- **`z` and `dq` fold over the KEY axis and are therefore independent of the
  kv length**, exactly as `attn.denom` and `attn.ctx` are in the forward.
  A query token's `dq` is the same bits whether the sequence has 4 keys or
  257, and the same bits in a decode step as in the prefill that produced
  it. That is the backward's clause (c) and its clause (d), and it holds by
  construction, and the gate exists to catch the construction being
  violated.
- **`dk` and `dv` fold over the QUERY axis and are NOT.** Section 5.2.

### 5.2 The honest replacement for "decode equals prefill"

The forward's clause (d) says a token's output bits do not depend on how the
sequence was chopped up. **The backward cannot say that about every output
and it must not pretend to**, because two of its outputs are sums over the
tokens in the call.

    dq, and every activation gradient reachable only through dq
        -> INDEPENDENT of the kv length and of the sequence chopping.
           Clause (c-bwd) and (d-bwd) assert it.

    dk_cache, dv_cache, and every WEIGHT gradient
        -> A SUM OVER THE QUERIES IN THIS CALL. A key at slot j is read by
           every query at position >= j, and the queries in later calls are
           not in this call. So a per-call backward computes a PARTIAL
           gradient for dk and dv, and the caller must add the partials.
           Clause (c-bwd) asserts they MOVE, as a negative property, the way
           gemm G5 asserts dB moves.

This is the transformer form of `gemm/IDENTICAL_BACKWARD_PLAN.md`'s section
3.2 finding that `dB`'s `k` is the token count. It is the same fact wearing
attention's clothes, and the version of it a reader should carry away is:

> **A backward pass's INPUT gradients can be made chopping-invariant. Its
> PARAMETER gradients cannot, because they ARE the sum over the chopping.**

`dk_cache` and `dv_cache` are emitted over the FULL `[0, S)` range, so the
`[0, pos0)` slice -- the gradient owed to tokens from earlier calls -- is
available to whoever assembles a multi-call backward. **Assembling it is
out of scope** (DEVIATION 1417). What this lane hands over is a named,
recorded, complete partial.

### 5.3 The carried-accumulator chunk theorem (DEVIATION 1416)

**This is the property this lane gets for free that the GEMM lane cannot
have, and it is worth the section.**

Every fold pinned in section 3 is a serial ascending chain seeded `+0.0`.
Split such a chain at ANY index `c`, run the first piece, STORE the
accumulator, and run the second piece with that stored value as its SEED.
The result is the unsplit chain's result, bit for bit, because it is
literally the same sequence of operations in the same order. `ftz` is
idempotent, so the store and reload change nothing.

Contrast, carefully, with the GEMM lane's aligned-split finding, and note
two corrections found on 2026-08-25 that this document must not repeat
loosely:

1. **That lane's aligned-split measurement used only TWO pieces**, and over
   two pieces a serial running sum and a balanced tree are the same
   operation. So the measurement does NOT establish that the accumulator
   must be the tree; it establishes that two-piece accumulation at an
   aligned boundary reproduces the unsplit bits, and both spellings do.
2. **The alignment rule is stricter than "every boundary is a subtree
   boundary."** Accumulation across pieces is LEFT ASSOCIATIVE. At `P = 8`
   with four pieces on subtree edges the accumulation computes
   `((n0 + n1) + n2) + n3` while the unsplit tree computes
   `(n0 + n1) + (n2 + n3)`. Different. **Two aligned pieces are safe;
   four are not, even on subtree edges.**

Neither correction touches the theorem above, because a carried accumulator
does not accumulate PARTIALS at all -- there is nothing to associate. The
serial chain's chunk property is exact for any number of pieces at any
boundary, and that is a real advantage of the reference-quality fold shape
over the partitioned one, in the one place where the partitioned one is
otherwise strictly better.

**The price, stated rather than hidden.** The theorem covers the FOLDS. It
does not cover the eleven ROUTED rows, every one of which is a gemm v1 call
whose activation-gradient `k'` is a layer width (fine) or whose weight
gradient `k'` is the token count (not fine). So:

> A backward chunked over the query axis, with carried accumulators, is bit
> exact for `z`, `dq`, `dk`, `dv` and everything downstream of them. It is
> bit exact for the ELEVEN weight gradients only at gemm v1's aligned
> two-piece splits, and never for four pieces. **A training run must declare
> its chunk schedule.**

### 5.4 Batch composition

Nothing in section 3 reads `B`. The activation gradients are therefore batch
composition invariant by construction and the gate exists to catch an
execution plan violating the construction, exactly as contract 7.4 says for
the forward.

**The weight gradients are NOT, and the gate asserts they move.** Their
`k'` is `M = B*L`, so a launch that carries three sequences produces a
different `dW` from three launches of one, and no spelling makes it
otherwise -- summing over more tokens IS the operation. `IDENTICAL_SSM_NOTES`
records that batch invariance for attention has been pursued in public
(Thinking Machines' *Defeating Nondeterminism in LLM Inference*); **batch
invariance of a weight gradient is not a thing anyone can have** and this
document should not be read as claiming a weaker version of a published
result. What is this lane's is the CROSS-VENDOR half, and only after a leg.

---

## 6. The card, the clauses, the sabotages

### 6.1 The stages, in card order (DEVIATION 1418)

One record per stage per backward call, tags prefixed by the driver, in
strictly the order the backward computes them, which is the reverse of the
forward's. `tools/identity_trace_diff.py` aligns two traces by their TAG
SEQUENCES before it compares a single hash, so the strings and the order are
both part of the instrument and a reordered pair produces a WRONG ALIGNMENT
rather than a bigger diff.

     0  bwd.in.d_residual2      [M, dm]        the incoming gradient, as given
     1  bwd.d_down_proj_out     [M, dm]        B23, a copy
     2  bwd.d_mlp_gated         [M, inter]     B5d dA
     3  bwd.dW_down             [dm, inter]    B5d dB
     4  bwd.d_silu_out          [M, inter]     B21
     5  bwd.d_up_proj_out       [M, inter]     B21
     6  bwd.d_gate_proj_out     [M, inter]     B20
     7  bwd.dW_gate             [inter, dm]    B5gu dB
     8  bwd.dW_up               [inter, dm]    B5gu dB
     9  bwd.d_norm2_out         [M, dm]        B5gu dA x2, 2-term fan-in
    10  bwd.norm2.dot           [M]            B1-4b, the c fold
    11  bwd.dW_norm2            [dm]           B1-4b, the ones gemm
    12  bwd.norm2.dx            [M, dm]        B1-4b
    13  bwd.d_residual1         [M, dm]        B22, 2-term fan-in
    14  bwd.d_o_proj_out        [M, dm]        B22, a copy
    15  bwd.d_attn_ctx          [M, qw]        B5o dA
    16  bwd.dW_o                [dm, qw]       B5o dB
    17  bwd.d_attn_weights      [B,nh,L,S]     B19w, the ROUTED gemm
    18  bwd.attn.zdot           [B,nh,L]       B18, the z fold
    19  bwd.d_attn_masked       [B,nh,L,S]     B18, dS
    20  bwd.d_attn_scores       [B,nh,L,S]     B13, an exact identity
    21  bwd.d_qk_cell           [B,nh,L,S]     B12
    22  bwd.d_q_rope            [M, qw]        B11q, the dq chain
    23  bwd.d_k_cache           [B,nkv,S,hd]   B11k, the dk chain, FULL [0,S)
    24  bwd.d_v_cache           [B,nkv,S,hd]   B19v, the dv chain, FULL [0,S)
    25  bwd.d_k_rope            [M, kw]        the [pos0,S) slice, a copy
    26  bwd.d_v_proj_out        [M, kw]        the [pos0,S) slice, a copy
    27  bwd.d_q_proj_out        [M, qw]        B9-10, the RoPE transpose
    28  bwd.d_k_proj_out        [M, kw]        B9-10, the RoPE transpose
    29  bwd.dW_q                [qw, dm]       B5qkv dB
    30  bwd.dW_k                [kw, dm]       B5qkv dB
    31  bwd.dW_v                [kw, dm]       B5qkv dB
    32  bwd.d_norm1_out         [M, dm]        B5qkv dA x3, 3-term fan-in
    33  bwd.norm1.dot           [M]            B1-4a, the c fold
    34  bwd.dW_norm1            [dm]           B1-4a, the ones gemm
    35  bwd.norm1.dx            [M, dm]        B1-4a
    36  bwd.d_x                 [M, dm]        2-term fan-in, THE OUTPUT

**Thirty-seven stages.** Four notes on choices a reader will question.

- **`bwd.in.d_residual2` is a stage.** Two cards whose INPUTS differ are
  diffing their fixtures, and `gemm/IDENTICAL_BACKWARD_PLAN.md`'s G9 makes
  the same point about `bwd.<route>.in.dc`. **Compare it before comparing
  any output stage.**
- **`bwd.d_attn_scores` is recorded even though it is bitwise equal to
  `bwd.d_attn_masked` by construction.** That is the whole reason: the
  equality is what `B13_MASK_ZEROES_GRAD` breaks, and an unrecorded identity
  is an unchecked one.
- **`bwd.norm1.dot` and `bwd.norm2.dot` are recorded** for the same reason
  the forward records `norm1.sumsq`: a fold whose only evidence is the value
  it feeds cannot be localized, and the forward lane's own scar is that
  thirteen moved stages were absorbed by a residual add and an output-only
  gate called the sabotage inert.
- **`bwd.d_k_cache` and `bwd.d_v_cache` are recorded over `[0, S)`, the
  USED prefix and not the allocation.** `core/identity_trace.mojo` rule 3.
  Their `[0, pos0)` half is the handoff of section 5.2 and it is on the card
  so that a multi-call assembler has something to diff.

### 6.2 What "identical" is gated to mean, mirroring contract section 10

**No gate file exists. Every clause below is a specification for a file
nobody has written.**

**(a)** device backward card equals host backward oracle card, bitwise, at
every stage and every shape. FAST arms RECORDED, not asserted, where they
are vendor shaped.

**(b)** the same bits on every one of 8 repeated launches.

**(c)** BATCH COMPOSITION and SEQUENCE LENGTH invariance **of the activation
gradients**, `bwd.d_x`, `bwd.d_q_rope`, `bwd.d_norm1_out`,
`bwd.d_attn_ctx`, each half with its own firing control; **AND the negative
half**, that `bwd.dW_*`, `bwd.d_k_cache` and `bwd.d_v_cache` MUST MOVE under
a change of batch composition, with the host oracle predicting the moved
cell count BEFORE the device is asked. A gate that only asserts the positive
half would pass on an implementation that computed constants.

**(d)** THE CHUNK CLAUSE, replacing the forward's decode-equals-prefill. A
backward chunked over the query axis with CARRIED accumulators reproduces
the unchunked activation gradients bitwise, at every split point including
misaligned ones; and the UNCARRIED control (two partial chains added at the
end) MUST MOVE at any chunk of 2 or more terms. Without the negative
control this clause passes for ever on a broken chunker.

**(e)** the row-39 audit. The incoming gradient is REFUSED BY NAME if it
holds a NaN or an infinity, tested BY BITS and not by compares because Metal
flushes compare operands. Plus the signed-zero audit: the masked cells of
`bwd.d_attn_scores`, whose predicted sign is `sign(dy_j - z)`, and the
fan-in sites of section 3.5.

**(f)** CORRECTNESS, which is a different question from identity and which
no other clause asks. **A transpose error is bit identical on three
vendors.** This lane's answer, and its limit:

- **The linear and bilinear seams admit an EXACT integer check.** On a
  fixture whose every operand is a nonzero integer in `[-8, 8]` with the
  partial sums bounded under `2^24`, the derivative is exactly
  representable and can be written down directly, one plain ascending sum
  per cell, with no epsilon to tune. That covers every gemm route, `dq`,
  `dk`, `dv`, `dy`, the mask, the residuals, the fan-ins, and **the RoPE
  transpose AT ABSOLUTE POSITION 0 ONLY**, where `cos` is exactly `1.0` and
  `sin` exactly `+0.0`. This is `gemm/IDENTICAL_BACKWARD_PLAN.md` G2's
  method, DEVIATION 1051, lifted.
- **The nonlinear seams do not.** `silu'`, the `rsqrt` tail and the softmax
  closed form carry transcendentals, so exact integers do not survive them.
  Gating those needs a FLOAT64 DIRECTIONAL-DERIVATIVE reference at a stated
  tolerance. **It does not exist.** `transformer_oracle.mojo` deliberately
  carries no float64 reference (its header says why), `transformer/corpus/`
  does not exist, and a float64 backward written now would be a second
  unreviewed implementation of a gradient whose first implementation has
  not been compiled. **IT IS OWED**, it is section 11's largest item, and
  until it lands **clause (f) covers the routing and not the calculus.**

**(g)** every clause above falsifiable by a NAMED sabotage that fails a
gate.

### 6.3 The sabotage set, with each arm's predicted INERT case

**The inert column is the point of the table.** An arm with no stated inert
case is an arm nobody has thought about, and the forward lane's own
experience is that two of its thirteen arms pass clause (a) by construction
and would have been deleted as pointless if clause (d) had been written
late.

| arm | first stage it must move | predicted INERT case -- what makes it pass while gating nothing |
|---|---|---|
| `B11_DQ_VIA_GEMM` | clause (c-bwd)/(d-bwd), NOT clause (a) at a fixed shape | **INERT at every `S <= 128`**, where `P(S) = 1` and gemm v1's cell IS the whole-`k` ascending chain from `+0.0`. Needs `S >= 129`; the profile's `L = 257` fixture is the only one that reaches it |
| `B11_DK_VIA_GEMM` | same | **INERT at `L <= 128` AND `n_rep == 1` together.** At `n_rep == 2` the routed form must accumulate two per-head gemms, a different association, so it fires on shape alone |
| `B19_DV_VIA_GEMM` | same | identical to `B11_DK_VIA_GEMM`. This is the backward twin of `S19_VALUE_SUM_VIA_GEMM` and inherits its warning verbatim: without clause (d) it looks inert and gets deleted |
| `B10_DW_VIA_CHAIN` | `bwd.d_attn_weights` | **CANNOT FIRE AT THE GATE SHAPE AT ALL.** It replaces the routed `k' = head_dim` gemm with a hand chain, and the two agree bit for bit whenever `P(head_dim) == 1`, i.e. `head_dim <= 128`. The profile's fixtures are `head_dim` 16 and 24. **DEVIATION 1405 is therefore pinned by ARGUMENT and not by a fired arm**, and that is recorded here rather than discovered later |
| `B18_SOFTMAX_DECOMPOSED` | `bwd.d_attn_masked` | INERT at `S == 1`, where `y = 1`, `z = dy` and both forms give `0`. A decode-against-empty-cache fixture is blind to it. **AND THE ARM IS WEAKER THAN THE CLAUSE IT GUARDS**: it separates the ASSOCIATION (`y*dy - y*z` against `y*(dy - z)`, the node-by-node form) and NOT the argmax, because a device kernel cannot scatter an argmax residue without a float ATOMIC and an atomic is what this whole file refuses. **The argmax half of DEVIATION 1406 is unfirable by construction**, which is itself an argument for the refusal and is recorded rather than papered over |
| `B18_ZFOLD_UNFUSED` | `bwd.attn.zdot` | INERT at `S == 1` (`fma(a,b,+0)` and `ftz(+0 + ftz(a*b))` agree) and on any row whose products are all exactly representable |
| `B18_ZFOLD_DESCENDING` | `bwd.attn.zdot` | INERT at `S == 1` only. **NOT inert at `S == 2`**, because an fma keeps the second product exact and the two orders round different quantities -- a two-term fma chain is order dependent where a two-term add chain is not |
| `B01_DOT_UNFUSED` | `bwd.norm2.dot` | INERT at `d_model == 1` and on exactly-representable rows |
| `B01_DOT_DESCENDING` | `bwd.norm2.dot` | INERT at `d_model == 1` only, same fma argument |
| `B_RSTD_RECOMPUTE_DESCENDING` | `bwd.norm2.dx` | INERT at `d_model == 1`. The ASCENDING recompute must move NOTHING, and that is the arm's other half: it is what makes "a recompute is bit exact" checked |
| `B20_SIGMOID_FROM_SILU` | `bwd.d_gate_proj_out` | **INERT for every `gate_proj.out >= about 17`**, where `1 + exp(-x)` rounds to `1.0`, `silu(x) == x` exactly and the reconstruction returns exactly `1.0`. Raises on a `0/0` at `x == 0`, which the refusal catches |
| `B20_SILU_DERIV_ALT_ASSOC` | `bwd.d_gate_proj_out` | INERT wherever `sg` is exactly `1.0` (the same `x >= ~17`) and at `x == 0` |
| `B20_SILU_DERIV_FUSED` | `bwd.d_gate_proj_out` | INERT wherever `x * (1 - sg)` is exactly representable |
| `B09_ROPE_TRANSPOSE_SIGN` | `bwd.d_q_proj_out` | **INERT AT ABSOLUTE POSITION 0**, where `sin` is exactly `+0.0`. Per CELL, not per fixture, so the gate must COUNT moved cells and RAISE if the count equals the position-0 population |
| `B09_ROPE_HALVES_ADJACENT` | `bwd.d_q_proj_out` | INERT at `head_dim == 2`, and at position 0 |
| `B10_ROPE_BWD_FUSED` | `bwd.d_q_proj_out` | INERT at position 0 (`fma(rot, +0.0, a)` and `ftz(a + ftz(rot * +0.0))` agree for `a != -0.0`) |
| `B12_SCALE_INTO_DQ` | `bwd.d_qk_cell` | **INERT AT EVERY POWER-OF-FOUR `head_dim`.** At 16 and 64 the scale is exactly `0.25` / `0.125`, exact scaling commutes with the fma chain bitwise, and moving the scale to the far side of the chain changes nothing. **The `head_dim = 24` fixture contract section 3 already requires is what makes this arm fire at all** |
| `B13_MASK_ZEROES_GRAD` | `bwd.d_attn_scores` | Moves ONLY masked cells whose `dS` is `-0.0`, i.e. where `dy_j - z < 0`. Predicted to move roughly half the masked cells, and **the oracle must predict the exact count before the device is asked** |
| `B_FANIN_ZERO_SEED` | `bwd.d_norm1_out` | **PREDICTED TO MOVE ZERO CELLS ON EVERY UNPLANTED FIXTURE.** It fires only where the FIRST accumulated term is `-0.0`, which needs a planted `-0.0` in `dq` or in the incoming gradient. **Vacuous without a plant**, said out loud |
| `B_FANIN_ORDER_QKV_REVERSED` | `bwd.d_norm1_out` | INERT wherever any two of the three terms are zero |
| `BWD_UNTRANSPOSED` (gemm lane's) | `bwd.dW_down`, and the four routed activation gemms | The gemm lane's own predicted mask, re-derived at THESE shapes. It is INERT on any route whose correct op already is `OP_NN`; that lane measured 4 of 6 and the count here differs because every forward call in this profile is `OP_NT` |
| `BWD_OPERAND_ORDER` (gemm lane's) | same | Every forward call here is `OP_NT`, whose `dA` and `dB` both put `dC` LEFT, so **this arm is INERT through this lane's entry points entirely.** Recorded as such rather than run and reported green |

**Twenty-two arms, of which four are predicted inert at the current gate
shape and two of those cannot be made to fire without changing the profile's
fixtures.** That is the honest count and it is the number a reader should
carry, not "twenty-two arms exist".

---

## 7. Cost

Analysis. No number below has been measured, and this profile publishes no
timing figure (contract section 11).

**Memory.** The backward allocates 37 stage buffers against the forward's
30, and its four large ones are `[B, nh, L, S]`: `bwd.d_attn_weights`,
`bwd.d_attn_masked`, `bwd.d_attn_scores`, `bwd.d_qk_cell`. Against the
forward's SIX buffers of that shape, that is a 4/6 increase on the dominant
term, so a backward call's peak footprint is roughly 1.7x a forward call's,
plus the saved forward stages which must be held for the whole interval.
**The eager path is what makes this affordable and it is also what makes it
large**; a fused-softmax forward would save the four buffers and lose every
localizing stage on the card, and contract section 6's argument for not
doing that applies here unchanged.

**Recomputation.** Three, all bit-exact re-executions of pinned seams:
`rstd` from `sumsq` (two host-cheap operations per row), `inner` from
`x` and `rstd` (one product per cell, twice, once for each norm), and
`sigmoid` from `gate_proj.out` (one `portable_expf` plus one division per
MLP cell). The sigmoid is the expensive one and it is the price of DEVIATION
1411(a).

**Arithmetic.** The backward's dominant term is the same order as the
forward's: four chains of length `S` or `L` where the forward had two, plus
eleven gemm calls where the forward had seven-plus-`B*n_heads`. The
`B*n_heads` per-head gemm loop is now run TWICE (once for `dy` at
`(L, S, hd)`), so the launch count grows.

**The identity cost is REAL, POSITIVE and UNMEASURED, on every vendor.**
This document does not say identity is free anywhere. The `dq`, `dk` and
`dv` chains are one thread per output cell walking a dependent chain, which
is latency bound by construction, and the routed alternative is faster and
wrong. The RMSNorm weight gradient pays two roundings per term where a hand
fold pays one. The SiLU derivative pays a fresh sigmoid per cell. What that
costs against a FAST backward has not been measured on any device and will
not be quoted until it is, alternated inside one thermal window.

**One execution hazard, named because it mirrors the GEMM lane's 6.3.**
`choose_gemm_plan` picks SPLITK only when `m' * n' <= 4096`. Every weight
gradient here has `k'` equal to the token count and `m' * n'` equal to the
weight matrix, so a middling weight matrix with a large token count -- say
`dW_gate` at `intermediate = 300`, `d_model = 32`, `M` in the thousands --
sits just the wrong side of that threshold. Analysis, not measurement, it
moves no bit, and the fix if a measurement finds it matters is a threshold
change and never a different leaf rule.

---

## 8. Deviations 1400-1449

| # | what | state |
|---|---|---|
| 1400 | the backward composition `transformer.bwd.fp32.v1` and its IDENTITY_PATHS row | RESERVED, no row written |
| 1401 | the organizing rule: route a fold whose contraction LENGTH is a configuration quantity, pin one whose length is a launch quantity | SPENT, this document |
| 1402 | `dq` pinned as a serial ascending chain over the ABSOLUTE key index; gemm v1 REFUSED, because S11's backward contracts over the kv length and not over `head_dim` | SPENT, both code files |
| 1403 | `dk` pinned as a serial ascending chain over `(head in group, query)` ascending; gemm v1 REFUSED | SPENT |
| 1404 | `dv` pinned likewise, the mirror of contract DEVIATION 807 | SPENT |
| 1405 | the attention-weight gradient ROUTED through gemm v1 `OP_NT` at `(L, S, head_dim)`, and the admission that its sabotage cannot fire at the gate shape | SPENT |
| 1406 | the softmax backward as the CLOSED FORM, and the refusal of the decomposed graph with its argmax scatter | SPENT |
| 1407 | the `z` fold pinned serial ascending over the ABSOLUTE key index from `+0.0`, FUSED | SPENT |
| 1408 | the RMSNorm backward's `c` fold pinned serial ascending over `d_model` from `+0.0`, FUSED, contract S1's shape | SPENT |
| 1409 | the RMSNorm backward's elementwise tail spelled node by node against the reference's autograd GRAPH, and `-0.5` and `2.0` spelled rather than elided | SPENT |
| 1410 | the two RMSNorm weight-vector gradients spelled as a gemm v1 `OP_NN` against a ones vector, lifting gemm DEVIATION 851, and the two-roundings-per-term cost that lift carries | SPENT |
| 1411 | the SiLU derivative's five roundings, and the refusal to reconstruct `sigmoid` from `silu.out` | SPENT |
| 1412 | the RoPE backward as the transposed rotation, the sign convention, and the inherited rounding budget | SPENT |
| 1413 | the fan-in accumulations: NO `+0.0` seed, FORWARD-USE order, `q,k,v` ascending at the three-term site | SPENT |
| 1414 | the mask backward as an exact identity, and the refusal to zero the masked cells | SPENT |
| 1415 | the scale backward applied to the finished `dS`, not folded into `dq`'s chain | SPENT |
| 1416 | the carried-accumulator chunk theorem, and its NON-application to the eleven routed weight gradients | SPENT |
| 1417 | `dk_cache` / `dv_cache` emitted over the full `[0, S)` as a HANDOFF; a multi-call backward REFUSED as out of scope | SPENT |
| 1418 | the 37-stage backward card and its tags | SPENT |
| 1419 | the 22-arm sabotage set and the inert-case column | SPENT |
| 1420 | the recomputation set (`rstd`, `inner`, `sigmoid`) and the claim, gated, that a recompute is bit exact | SPENT |
| 1421 | the saved-stage set, and the finding that the closed-form softmax backward reads NONE of `attn.scores`, `attn.masked`, `attn.max`, `attn.exp`, `attn.denom` | SPENT |
| 1422 | the device file's launch geometry: one thread per output cell, no float atomic, no cross-thread float reduction, no shared memory | SPENT, `transformer_backward.mojo` |
| 1423 | the incoming gradient refused nonfinite BY BITS before any recorded stage | SPENT |
| 1424 | the GQA head-group fold order inside `dk` and `dv`: heads ascending outer, queries ascending inner, ONE chain and not a sum of per-head partials | SPENT |
| 1425 | the backward oracle reuses `gemm_backward_a_call` / `_b_call` rather than restating the six-row table, so this lane's weight gradients inherit G1, G2 and the two `SAB_BWD_*` arms | SPENT, oracle |
| 1426 | `transformer_backward.mojo` lives in `checks/` and not in `impl/`, because there is no upstream file to mirror: HuggingFace ships no backward and autograd generates it | SPENT |
| 1427 | the device backward reuses `LlamaDims`, `LlamaDeviceWeights`, `LlamaKVCache` and `LlamaDeviceStages` from the forward device module, and restates the four private plumbing helpers locally rather than importing underscore-prefixed symbols | SPENT, device |
| 1428 | the device backward calls the synchronizing `identical_gemm` rather than `identical_gemm_backward_*_into`, so it carries no caller-owned workspace; the cost is that gemm gate G7's workspace-sizing coverage is NOT inherited | SPENT, device |
| 1429 | reserved | |
| 1430-1439 | RESERVED for the gate file `transformer_backward_check.mojo`, which does not exist | |
| 1440-1449 | RESERVED for the float64 directional-derivative reference and the backward corpus, neither of which exists | |

Read the block with the pattern that answers the question, since the
singular form has bitten this repository before:

    grep -rhoE "DEVIATIONS? 14[0-4][0-9](-14?[0-9]+)?" . | sort -u

and note that a range is reported by its first number only.

---

## 9. Phases

| phase | content | depends on | size |
|---|---|---|---|
| **A** | this document, the oracle, the device file | the gated forward | DONE, UNCOMPILED, UNGATED |
| **B** | compile both files. **Nothing below is meaningful until this passes**, and section 11 names what is most likely to break | A | unknown, and it is the honest first task |
| **C** | a backward fixture: the forward fixture's 13 cases plus a `d(residual2.out)` generator, plus the `-0.0` plant `B_FANIN_ZERO_SEED` needs | B | small |
| **D** | clause (a): host oracle against device, 37 stages | C | moderate |
| **E** | clause (f)'s EXACT INTEGER arm, host only, over the routing and the linear seams | C | moderate, and it can run beside D |
| **F** | clauses (b), (c), (e) | D | moderate |
| **G** | clause (d), the chunk theorem, with its uncarried control | D | moderate; needs a chunked driver |
| **H** | the 22 sabotage arms, each with its predicted inert case CHECKED as well as its predicted firing | D, F, G | large. **This is the phase that decides whether any of the above is evidence** |
| **I** | the float64 directional-derivative reference, clause (f)'s calculus half | D | large, and it is the largest single owed item |
| **J** | the three-vendor leg | F, G, H | one rented hour per vendor, guard armed FIRST |

B through E are parallelizable. Nothing after H should be quoted as
evidence, and nothing before J is a cross-vendor claim.

---

## 10. What this buys, and the ladder it does not climb

`gemm/IDENTICAL_BACKWARD_PLAN.md` section 3.1's ladder, with this lane's
rows filled in:

    identical backward GEMM                    CLOSED on Apple + AMD, no
                                               sabotage arm fired anywhere
      -> identical dA, dB for ONE linear layer CLOSED, same caveat
      -> identical gradients for a BLOCK       THIS LANE. Construction only.
                                               Every activation backward,
                                               norm backward and softmax
                                               backward now HAS a pinned
                                               form. None has been compiled.
      -> identical WEIGHTS after one step      NOT BUILT. Needs the loss
                                               reduction, the optimizer step
                                               and the accumulation order.
      -> identical weights after N steps       NOT BUILT. Needs the RNG on
                                               top of everything above.
      -> identical weights on a multi-GPU run  REFUSED. NCCL and RCCL choose
                                               their summation order at run
                                               time.

**Every arrow is an ONLY IF.** This lane moves the third row from "not
built" to "constructed and ungated" and moves nothing else. The gemm lane's
own honest fraction said T6 plus T7, the norm and attention backward family,
was the largest remaining piece and that T7 was "genuinely hard rather than
merely unwritten, because the standard fused attention backward accumulates
`dK` and `dV` across thread blocks with float atomics."

**That obstacle is gone here, and it is worth saying how.** This lane's
`dk` and `dv` are one thread per OUTPUT cell walking `(head in group,
query)`, so no two threads ever write the same address and **there is no
float atomic anywhere in this backward**, nor any shared memory, warp
primitive, cross-block reduction or `pinned_reduce` call. That is not
cleverness; it is the direct consequence of refusing the fused shape, and
the price is the eager path's memory and the chains' latency, both priced in
section 7. T7 was hard for FlashAttention's backward. It is merely slow for
the eager one.

---

## 11. Not claimed, and what is owed

- **NOTHING HERE HAS BEEN COMPILED, RUN, OR OBSERVED.** No gate file exists.
  No fixture exists. No card has been emitted. Every predicted count in
  section 6.3 is on paper.
- **Not identical training.** Not identical models. Not "bit-identical AI
  inference", which the GEMM charter forbids, and not its training
  equivalent. Section 10.
- **Not a cross-vendor claim.** Not one line of this has run on one device,
  let alone three, and the standing lesson is that two backends agreeing
  closes nothing: Apple and AMD agreed bit for bit through 302 stages while
  NVIDIA diverged at `tree001.winners.scores`.
- **Not agreement with PyTorch.** The fold orders, the transcendentals, the
  division and the SiLU derivative's association are OURS. A reader who
  takes "identical" to mean "equal to torch" has taken more than is offered,
  and ATen could not be read at all -- there is no PyTorch checkout.
- **Not a verified reference spelling for two seams.** ATen's
  `silu_backward` and ATen's `_softmax_backward_data` are both PINNED ON
  ROUNDING-COUNT GROUNDS AND NOT ON EVIDENCE, exactly as contract 5.4 pins
  S18's division. Both are the first things to check when a checkout lands.
- **Not a correctness gate for the calculus.** Clause (f) covers the routing
  and the linear seams by exact integers. The nonlinear seams need a float64
  directional-derivative reference and **there is none**. A gradient checked
  only against a tolerance is checked against a tolerance; this lane deals
  in bits, and the honest statement is that the bits are gated and the
  CALCULUS is not, yet.
- **Not a multi-call backward.** DEVIATION 1417. `dk_cache` and `dv_cache`
  are handed over complete over `[0, S)` and assembling them across calls is
  somebody else's function.
- **Not FlashAttention's backward**, not SDPA's, not a fused kernel of any
  kind. Contract section 6 excluded the forward shapes and every word
  applies.
- **Not batch-composition invariance of a weight gradient.** Section 5.4
  says the opposite and the gate asserts it.
- **Not a performance number.** None has been taken.
- **One pin cannot be falsified at the gate shape.** DEVIATION 1405's
  `B10_DW_VIA_CHAIN` is inert for every `head_dim <= 128`, so the routed
  attention-weight gradient rests on section 2.1's argument and on gemm v1's
  own certificate, not on a fired arm. Recorded in section 6.3 and repeated
  here so it is not lost in a table.

**Owed, in priority order.** (0) A one-line cross-lane edit: an `x` field
on `LlamaDeviceStages` (section 1). (1) Compile both files. (2) The gate file and
the fixture. (3) The 22 sabotage arms, each checked against its predicted
INERT case as well as its predicted firing. (4) The float64 directional
derivative. (5) The three-vendor leg. (6) An IDENTITY_PATHS row, which
DEVIATION 1400 reserves and which must not be written until (5).

**What is least likely to compile**, so that the first failure is expected
rather than surprising: the oracle's use of `gemm_backward_a_call`'s
five-element `Tuple` return (the gemm lane's own header flags it as one of
the two things most likely to need a syntax adjustment, and it has never
been compiled either); the device file's per-head gather-gemm-scatter loop,
which allocates nothing but does hold four buffers alive across an
`identical_gemm` call that synchronizes internally; and every `mut` argument
on a struct field, which is the shape that produces the
`MutUntrackedOrigin` versus `MutAnyOrigin` unification errors this
repository has lost time to and which neither file's author can settle
without a compiler.
