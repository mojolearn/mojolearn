# The backward passes of Mamba-2 and Mamba-3, scoped

Written 2026-09-03. A SCOPE document, not a contract and not a plan.

## STATUS, and what this document is

**NOTHING DESCRIBED HERE EXISTS.** No Mamba-2 backward and no Mamba-3
backward has been designed, declared, written, compiled or run in this
repository. Both forward contracts say so themselves and neither hedges
(`mamba/IDENTICAL_MAMBA2_CONTRACT.md`:399, "No BF16/FP16, no training, no
backward"; `mamba/IDENTICAL_MAMBA3_CONTRACT.md`:455, "no training/backward").
This document answers one question, asked by the orchestrator, which is what
those two backwards would actually require and how much of
`mamba/IDENTICAL_BACKWARD_PLAN.md` carries over.

**Everything below is DERIVED ON PAPER from the three contracts and from
upstream source read for this document.** Not one number here was measured.
The repository's own standing rule applies to this file harder than to most,
because a scope document has no gate at all
(`[[mojotrees-code-not-source-of-truth]]`).

What this document does NOT do, deliberately.

- It allocates **no deviation numbers.** Mamba-1's backward spent 1070-1089
  (`IDENTICAL_BACKWARD_PLAN.md`:754). Mamba-3's forward records 832-839 as
  unclaimed (`IDENTICAL_MAMBA3_CONTRACT.md`:5-6). Neither sibling has a
  backward band and this document does not open one.
- It states **no hours, no effort estimate and no schedule.**
- It **amends no contract.** All three forward profiles are consumed here and
  none is edited.

The counting rule, stated so the integers can be argued with. One row per
operation that either moves data with no arithmetic (a COPY), is answered by
a primitive this repository already owns and calls unchanged at a new site
(category **a**), routes to `mojolearn.identical.gemm.fp32.v1` or to a v1
ones-vector reduction (category **b**), or needs a pinned order of its own
(category **c**, split into **c-elementwise**, **c-inherited** and
**c-topology** exactly as `IDENTICAL_BACKWARD_PLAN.md`:186-203 splits it).
**A different split gives a different integer.** The ratio between the
categories is the finding, never the integer.

---

## 1. The Mamba-1 finding, tested against both siblings

`IDENTICAL_BACKWARD_PLAN.md`:52-56 states the finding that this whole
document is built to check.

> **The hardest seam is not the one that looks hardest.** The reverse
> recurrence stays inside one thread and inherits the forward's structural
> launch invariance for free. The hard seam is that **`B` and `C` are shared
> across channels**, so `dB` and `dC` contract over `d_inner`, an axis no
> float crosses in the forward.

And :539-545 spells the mechanism.

> In Mamba-1, `A` and `D` are per channel but `B` and `C` come out of
> `x_proj` per TOKEN at `[M, N]` and are shared by every one of the
> `d_inner` channels. The forward reads them, which costs nothing. The
> backward SUMS over the channels that read them, and every term of that sum
> lives in a different thread under the forward's grid.

### 1.1 Mamba-2. YES, at three sites and at two different lengths.

The forward contract makes the sharing a stated constant rather than an
inference. `IDENTICAL_MAMBA2_CONTRACT.md`:90.

> | ngroups (G) | 1 | mamba2.py:47; B and C are shared across heads by COPY
> (broadcast, not arithmetic) |

"Broadcast, not arithmetic" is the forward's whole reason for calling it a
copy, and it is exactly the sentence that makes the backward a contraction.
The three sites, with the axis each one crosses.

| site | forward seam | backward contracts over | length |
|---|---|---|---|
| `dC` from the state read-out | S18, `Y_off = (C · h_prev) ⊙ exp(dA_cs)` (`:141`) | heads AND headdim, because `C[i,n]` is read by every `(h, p)` | `H * P = d_inner` |
| `dB` from the chunk state | S15, `B_decay = B ⊙ decay` (`:138`), `decay` per head | heads | `H` |
| `dG` from the mask product | S13, `M = G ⊙ L` (`:136`), `G` per group and `L` per head | heads | `H` |

**The `dC` site is Mamba-1's T3 at the same contracted length and with the
same spelling available.** `dC[i,n] = sum over (h,p) of d_dot[i,h,p] *
h_prev[h,p,n]` is a gemm v1 cell at `(1, N, d_inner)` per token, which is
what T3 pins (`IDENTICAL_BACKWARD_PLAN.md`:294-300). Nothing new is needed
here beyond calling it.

**What is NEW is that Mamba-2 splits the contraction into two lengths.** The
`dB` and `dG` folds cross only the HEAD axis, length `H = d_inner / 64`,
which is sixty four times shorter than Mamba-1's. `IDENTICAL_BACKWARD_PLAN.md`
:304-307 warns that at `di > 128` T3's fold "is NOT a single chain, it is
`contract_partition(di)` leaves combined by v1's balanced tree". At `H` the
partition question is a different question with a different answer, and a
lane that copies T3's spelling to the `H` sites without recomputing
`contract_partition(H)` gets a wrong answer that passes every launch gate.

### 1.2 Mamba-3. YES, and the contract already says why it is worse.

`IDENTICAL_MAMBA3_CONTRACT.md`:166 does not merely restate Mamba-2's row, it
adds the clause that changes the backward.

> | ngroups = num_bc_heads (G) | 1 | :50, :95; B/C shared across heads by
> COPY, but rotation is PER HEAD (dt is per-head), so post-rotation K/Q
> DIFFER per head **the broadcast is a copy, what follows it is arithmetic** |

Read the backward implication out of that sentence. The head fold cannot be
done early. Each head applies its own rotation at its own `θ[t,h,r]` (S13,
`:220`), so `dq_pre[t,h,n]` and `dk_pre[t,h,n]` differ per head and the sum
over heads can only happen AFTER every head's rotation has been transposed.
The fold therefore sits between the rotation backward and the B/C RMSNorm
backward (S21, `:228`), at length `H`, once for `B` and once for `C`.

**Mamba-3's version of the finding is worse than Mamba-1's in kind and
better in length.** Worse, because in Mamba-1 the shared quantity is read
directly and the contraction is a plain weighted sum; in Mamba-3 there is a
per-head, data-dependent, ROUNDED transformation between the shared value and
its consumers, so the head fold's terms are each the output of a three
rounding rotation. Better, because `H` is shorter than `d_inner`.

### 1.3 The reverse recurrence, still easy, and now shorter

`IDENTICAL_BACKWARD_PLAN.md`:529-537 says T1 is easy because a thread owns
one `(b, d)` pair and holds `N` values in registers. Both siblings inherit
that, and both make it CHEAPER, because the recurrence they run backward is
not over tokens.

Mamba-2's S17 is serial over CHUNKS (`IDENTICAL_MAMBA2_CONTRACT.md`:140,
`h_c = ftz(fma(scale_c, h_{c-1}, chunk_states_c))`), and Mamba-3's S20 is the
same shape (`IDENTICAL_MAMBA3_CONTRACT.md`:227, cited as "mamba2 S17's fused
answer to the identical decay·state+increment question"). The backward of a
first order linear recurrence is a first order linear recurrence run
backwards, so both reverse over `C = ceil(L/Q)` chunks rather than over `L`
tokens. Same topology, one to two orders of magnitude fewer steps.

Mamba-3 breaks the pattern in one place, and it is the place nobody expects.
Its ANGLE recurrence S10 (`:217`) is serial over TOKENS with no chunk reset,
where Mamba-2's cumsum S11 (`:134`) explicitly resets ("no cross-chunk carry
the chunk boundary is a hard reset, that is the algorithm"). So Mamba-3's
backward carries **one whole-sequence reverse chain that Mamba-2 does not
have**, and section 5 says what that costs.

---

## 2. Mamba-2. The operation inventory

Derived by the chain rule from `IDENTICAL_MAMBA2_CONTRACT.md` section 4's
pinned forward, seam by seam, in the backward's own direction. `M = B*L`,
`dm = d_model`, `di = 2*dm`, `H = di/P`, `P = 64`, `N = 128`, `G = 1`,
`Q = 256`, `C = ceil(L/Q)`, `CD = di + 2N`.

    copies                                    6
    (a)  existing primitive, no new order      3
    (b)  routable to gemm v1                  20
    (c)  genuinely new arithmetic             41
         of which  c-elementwise              19
                   c-inherited order           3
                   c-TOPOLOGY                 19
                                        -------
                                              70

Beside Mamba-1's 4 / 2 / 13 / 23 out of 42
(`IDENTICAL_BACKWARD_PLAN.md`:195-203). **The routed fraction barely moves
(29 percent against 31 percent) and the topology count more than doubles,
19 against 8.** The chunked forward does not turn the backward into a routing
problem any more than the sequential one did. Nineteen c-topology rows is not
nineteen new topologies. It is SIX new ones at six sites, plus six Mamba-1
topologies reused at thirteen sites, section 2.3.

### 2.1 The table

`k'` is the contracted length where there is one. Sites marked **T-n** reuse a
Mamba-1 topology; sites marked **M2-n** are new here.

| # | operation | reduces over | cat | note |
|---|---|---|---|---|
| C1 | `d_op = dres` | none | copy | |
| C2 | `d_gn = d_op . W_out` | `dm` | b | gemm `dA` |
| C3 | `dW_out = d_op^T . gnorm.out` | **tokens** | b | gemm `dB`, `k'=M` |
| C4 | `d_yn = pinned_mul(d_gn, w_gn[j])` | none | c-elem | |
| C5 | `pWg = pinned_mul(d_gn, inner_g)` | none | c-elem | pre-product for C6 |
| C6 | `dw_gn[j] = sum_t pWg` | **tokens** | b | ones-vector `OP_NN` |
| C7 | `dot_g[t] = sum_j d_yn * y_g` | `d_ssm = di` | c-inh | S1's fold at a NEW length |
| C8 | gated-norm `dy_g` closed form | none | **T7** | `rstd^3` association |
| C9 | recompute `identical_silu(z)` | none | a | MUST match S8 |
| C10 | `sig_z = identical_sigmoid(z)` | none | a | new site |
| C11 | `silu'(z)` composite | none | c-elem | 3 roundings |
| C12 | `dY = pinned_mul(dy_g, silu_z)` | none | c-elem | |
| C13 | `dz = ((dy_g * Y) * silu')` | none | c-elem | 4 factors |
| C14 | `dskip = dY` | none | copy | S20 |
| C15 | `pD = pinned_mul(dskip, x_post)` | none | c-elem | pre-product |
| C16 | `dD[h] = sum over (t,p) of pD` | **tokens AND `P`** | **M2-1** | a TWO AXIS fold, 5.2 |
| C17 | `dx_D = pinned_mul(dskip, D[h])` | none | c-elem | |
| C18 | `dY` split to `dY_diag`, `dY_off` | none | copy | S19 |
| C19 | `d_dot18 = pinned_mul(dY_off, e_i)` | none | c-elem | |
| C20 | `de18[i,h] = sum_p dY_off * dot18` | `P` | b | gemm cell, `k'=P` |
| C21 | `dC_18[i,n] = sum_{h,p} d_dot18 * h_prev` | **`d_inner`** | **T3** | the shared-channel fold |
| C22 | `dh_prev[h,p,n] = sum_i d_dot18 * C` | `Q` | b | gemm cell, `k'=Q` |
| C23 | the reverse CHUNK recurrence `dh_c` | **chunks** | **T1** | new axis, 1.3 |
| C24 | `dscale_c = sum_{p,n} dh_c * h_{c-1}` | **`P*N`** | **M2-2** | the longest crossing fold |
| C25 | `d_dacsl_c = pinned_mul(dscale_c, scale_c)` | none | c-elem | `exp'` is `exp` |
| C26 | `dcstate_c = dh_c` | none | copy | |
| C27 | `dBdec[i,h,n] = sum_p dcstate * X_d` | `P` | b | gemm cell |
| C28 | `dX_d_16 = sum_n dcstate * B_decay` | `N` | b | gemm cell, `k'=128` |
| C29 | `ddecay[i,h] = sum_n dBdec * B` | `N` | b | gemm cell |
| C30 | `dB_15[i,n] = sum_h dBdec * decay` | **`H`** | **T3** | at the SHORT length |
| C31 | `d_diff = pinned_mul(ddecay, decay)` | none | c-elem | |
| C32 | `d_dacsl += sum_{i in chunk} d_diff` | `Q` | b | ones-vector, `k'=Q` |
| C33 | `dM[i,j,h] = sum_p dY_diag * X_d[j]` | `P` | b | gemm cell |
| C34 | `dX_d_14[j] = sum_{i>=j} M[i,j,h] * dY_diag` | `Q` | **M2-3** | the TRANSPOSED mask, 5.2 |
| C35 | `dL = pinned_mul(dM, G)` | none | c-elem | |
| C36 | `dG[i,j] = sum_h dM * L` | **`H`** | **T3** | |
| C37 | `dseg = pinned_mul(dL, L)` | none | c-elem | |
| C38 | `dC_12[i,n] = sum_j dG * B[j,n]` | `Q` | b | gemm cell |
| C39 | `dB_12[j,n] = sum_i dG * C[i,n]` | `Q` | b | gemm cell |
| C40 | `d_dacs` join, three paths | none | **T6** | ORDER |
| C41 | `d_dA` from the SEGSUM triangle | **`Q x Q`** | **M2-4** | 2.3, the largest new fold |
| C42 | `d_dA` from the cumsum, reverse in-chunk | `Q` | **M2-5** | |
| C43 | `dx_ssm = pinned_mul(dX_d, dt)` | none | c-elem | |
| C44 | `ddt_x[l,h] = sum_p dX_d * x` | `P` | b | gemm cell |
| C45 | `ddt_A = pinned_mul(d_dA_tok, A[h])` | none | c-elem | |
| C46 | `dA_head[h] = sum_t d_dA_tok * dt` | **tokens** | b | ones-vector, 2.2 |
| C47 | `dA_log[h] = pinned_mul(dA_head, A[h])` | none | c-elem | `A = -exp(A_log)` |
| C48 | `ddt` join, x path and A path | none | **T6** | ORDER |
| C49 | the CLAMP derivative | none | **M2-6** | S9's `identical_clamp`, 5.2 |
| C50 | `ddt_raw` through `softplus'` | none | c-elem | DEVIATION 1078's question |
| C51 | `db_dt[h] = sum_t ddt_raw` | **tokens** | b | ones-vector |
| C52 | `dB` join, S12 path and S15 path | none | **T6** | ORDER |
| C53 | `dC` join, S12 path and S18 path | none | **T6** | ORDER |
| C54 | `sig_conv = identical_sigmoid(conv)` | none | a | |
| C55 | `dconv = pinned_mul(dsilu, silu'(conv))` | none | c-elem | |
| C56 | `dhin` four tap correlation over `CD` | `K` | c-inh | mamba1 B31 |
| C57 | `dcw[d,k] = sum_t dconv * hin_k` | **tokens** | b | four ones-vector calls |
| C58 | `dcb[d] = sum_t dconv` | **tokens** | b | ones-vector |
| C59 | `dsilu_xBC = concat(dx | dB | dC)` | none | copy | |
| C60 | `dzxbcdt = concat(dz | dxBC | ddt_raw)` | none | copy | |
| C61 | `d_norm_out = dzxbcdt . W_in` | `2di+2GN+H` | b | gemm `dA` |
| C62 | `dW_in = dzxbcdt^T . norm.out` | **tokens** | b | gemm `dB`, `k'=M` |
| C63 | `dinner = pinned_mul(d_norm_out, w_norm)` | none | c-elem | |
| C64 | `pW = pinned_mul(d_norm_out, inner)` | none | c-elem | pre-product |
| C65 | `dw_norm[j] = sum_t pW` | **tokens** | b | ones-vector |
| C66 | `drstd[t] = sum_j dinner * x` | `dm` | c-inh | S1's fold |
| C67 | block-norm `dx_norm` closed form | none | **T7** | second site |
| C68 | `dx = dres + dx_norm` | none | **T8** | THE ABSORPTION SITE |
| C69 | `dx` join into the `x` column of `dsilu` | none | **T6** | S10 path and S20 path |
| C70 | the parameter fold over `b` | **batch** | **T5** | shared by nine parameters |

### 2.2 Two of Mamba-1's eight topologies are DELETED by the chunked form

**T2, the `h[t-1]` checkpoint, does not exist as a decision.**
`IDENTICAL_BACKWARD_PLAN.md`:280-290 spends a whole deviation refusing
upstream's `a = h[t] - dbu[t]` recovery and pricing `N = 16` times the
activation footprint to avoid it. In Mamba-2 the incoming state is a CARDED
FORWARD STAGE. `IDENTICAL_MAMBA2_CONTRACT.md`:239.

> `pass.states [B, C, H, P, N]  S17, the state ENTERING each chunk (h_{c-1})`

The quantity T2 fights to obtain is already materialized, at
`B * C * H * P * N` rather than at `M * di * N`, which is a factor of `Q`
fewer entries per batch. **There is nothing to refuse and nothing to price.**

**T4, the descending `dA` fold, is deleted too, and the plan predicted it.**
`IDENTICAL_BACKWARD_PLAN.md`:322-330 declines to route `dA` to gemm v1
because the routed form needs "a THIRD buffer of `M * di * N` floats", and
closes with the instruction to revisit.

> **If a blocked variant ever makes the memory affordable this decision
> should be revisited and the routed form is strictly preferable**, because
> it deletes a declared order in favor of a certified one.

Mamba-2 makes it affordable by construction. Its `A` is per HEAD
(`IDENTICAL_MAMBA2_CONTRACT.md`:86-89, `A = -exp(A_log)` at `[H]`, S5 at
`:128`), so the pre-product buffer for `dA_head` is `[M, H]`, not
`[M, di, N]`. **C46 routes, T4 dies, and one hand declared fold direction
leaves the profile.** This is the single cleanest inheritance in this
document, and it is one the Mamba-1 plan wrote down before it was earned.

### 2.3 The six new topologies, named

Named, not declared. A declaration needs a direction, a seed and a fusion
decision, and that belongs in a contract this document is not.

- **M2-1, the two axis `dD` fold.** `D` is per head at `[H]`
  (`:95`, `D_has_hdim = False`) and S20 spells `pinned_mul(x[l,h,p_], D[h])`
  (`:143`), so `dD[h]` sums over TOKENS and over `P`. Every other parameter
  gradient in all three models crosses exactly one axis. This one crosses
  two, and the ORDER of the two folds is a free choice that moves bits.
- **M2-2, the `(P, N)` crossing fold for `dscale_c`.** `dscale_c = sum over
  (p, n) of dh_c * h_{c-1}`, contracted length `P * N`, once per `(b, h,
  chunk)`. It is routable, as gemm v1 at `(1, 1, P*N)`, and routing it is the
  right answer for T5's own reason (`IDENTICAL_BACKWARD_PLAN.md`:340-345, no
  threadgroup float atomics on Metal and 32 KB of threadgroup memory).
- **M2-3, the transposed triangular contraction.** S14's forward reads
  `M[i,j]` for `j <= i` (`:137`); `dX_d[j]` reads `M[i,j]` for `i >= j`. This
  is the conv tap reversal hazard in a new dress and it carries the same
  trap, `IDENTICAL_BACKWARD_PLAN.md`:153-159. **A transposed implementation
  is bit identical on any fixture whose `M` is symmetric**, and the padded
  positions are exact zeros (`IDENTICAL_MAMBA2_CONTRACT.md`:107-111) which
  makes a naive fixture MORE symmetric, not less.
- **M2-4, the segsum triangle backward.** S11 builds `seg[i][j]` for `j < i`
  as a serial ascending rebuild (`:134`) and S12 exponentiates it. So
  `dseg[i,j]` is a full `Q x Q` triangle per `(chunk, head)` and
  `d_dA[k] = sum over all (i,j) with j < k <= i of dseg[i,j]`. That is a two
  dimensional reverse prefix fold over a triangle. **Nothing in this
  repository declares a fold of that shape and it is the largest single new
  item of the Mamba-2 backward.**
- **M2-5, the in-chunk reverse cumsum** for the `dA_cs` path, length `Q`.
- **M2-6, the clamp derivative.** `identical_clamp` exists
  (`checks/numerics.mojo`:2273) and DEVIATION 788
  (`IDENTICAL_MAMBA2_CONTRACT.md`:371-377) defines its forward zero-sign and
  NaN behavior. **No document defines its DERIVATIVE, and the interesting
  case is the boundary.** At `x == lo` exactly, does the gradient pass or
  vanish. That is a selection on floats, which is row 13's hazard
  (`:211-214`), and it is unanswered.
And one REUSED topology whose site count is a finding in its own right.
**T6, the join order, at FIVE sites.** Mamba-1 has one join
(`IDENTICAL_BACKWARD_PLAN.md`:347-357) and spends a deviation on it. Mamba-2
has five, C40, C48, C52, C53 and C69, and every one is a free order that
moves the last bit. Mamba-3 has six.

---

## 3. Mamba-2. The hard seams, with the reason each is hard

| seam | why it is hard |
|---|---|
| **The segsum triangle (M2-4)** | A two dimensional reverse prefix fold, no precedent in the repository, and the only backward operation whose OUTPUT count is smaller than its input count by a factor of `Q`. The natural spellings (row major reverse, column major reverse, a suffix-sum trick) are three different numbers. |
| **`dC` over `d_inner` (T3)** | Mamba-1's own hardest seam, unchanged, at the same length. Section 1.1. |
| **`dB` and `dG` over `H` (T3 at a new length)** | The SAME topology at a length short enough that `contract_partition(H)` may be one leaf where `contract_partition(di)` is many. Copying T3's spelling without recomputing the partition is a wrong answer that passes every launch gate. |
| **`dscale_c` over `P*N` (M2-2)** | The longest crossing fold in the backward, at 8192 terms under the profile constants. Refuses a threadgroup tree on Metal for `IDENTICAL_BACKWARD_PLAN.md`:340-345's reason. |
| **`CHUNK_SIZE` is now MORE load-bearing, not less** | DEVIATION 783 (`:330-337`) says `Q` "is part of the arithmetic" for the forward, because "which terms flow through the intra-chunk GEMMs versus the serial pass IS the summation order". In the backward `Q` is additionally the LENGTH of four new folds (C32, C34, C41, C42). A backward built at `Q = 256` and a forward built at `Q = 256` are one profile; a backward that autotunes `Q` is not a backward for this forward at all. Upstream does exactly that, section 6. |
| **The five join orders (T6 x 5)** | Each one is INERT on any fixture where one contribution dominates, which is `IDENTICAL_BACKWARD_PLAN.md`:355-357's warning about `SAB_BWD_JOIN_ORDER` multiplied by five. Five quiet unfalsifiable arms is the `adv_softplus_guard` failure mode (`:408-424`) at scale. |
| **The transposed triangular contraction (M2-3)** | Bit identical to the wrong answer on a symmetric fixture, and the profile's own zero padding pushes fixtures toward symmetry. |
| **The clamp derivative (M2-6)** | Undefined at the boundary by every document. |
| **The conv backward, inherited UNREAD** | `IDENTICAL_BACKWARD_PLAN.md`:599-603 records that `causal_conv1d` "IS NOT ON DISK" and that every statement about the conv backward is INFERRED. Mamba-2 runs the same conv over a WIDER channel set, `CD = d_ssm + 2·G·N` (`:129`), so it inherits the gap and widens the surface it applies to. |

---

## 4. Mamba-3. The operation inventory

Derived the same way from `IDENTICAL_MAMBA3_CONTRACT.md` section 4. `P = 64`,
`N = 128`, `Q = 64`, `R = num_rope_angles = 32`, `H = d_inner / P`. The
resumption correction S22 (`:229`) is EXCLUDED from the count because it does
not run in an unbroken prefill, only on the DEVIATION 831 continuation path
(`IDENTICAL_MAMBA3_CONTRACT.md`:252-262).

    copies                                    8
    (a)  existing primitive, no new order      5
    (b)  routable to gemm v1                  22
    (c)  genuinely new arithmetic             63
         of which  c-elementwise              30
                   c-inherited order           3
                   c-TOPOLOGY                 30
                                        -------
                                              98

The routed fraction FALLS to 22 percent, from Mamba-1's 31 and Mamba-2's 29.
**Mamba-3's backward is the least routable of the three**, and the reason is
that everything Mamba-3 adds over Mamba-2 is per token elementwise arithmetic
with a data-dependent parameter, which has no gemm shape at all. Thirty
c-topology rows is EIGHT new topologies at ten sites, plus five of Mamba-2's
at six sites, plus six of Mamba-1's at fourteen sites.

### 4.1 The table

Marks are **T-n** for a Mamba-1 topology, **M2-n** for one Mamba-2
introduces, **M3-n** for one new here. The resumption correction S22 (`:229`)
is EXCLUDED for the reason given above the count.

| # | operation | reduces over | cat | note |
|---|---|---|---|---|
| D1 | `d_op = dres` | none | copy | S23 |
| D2 | `d_gate = d_op . W_out` | `dm` | b | gemm `dA` |
| D3 | `dW_out = d_op^T . gate.out` | **tokens** | b | `k'=M` |
| D4 | recompute `identical_silu(z)` | none | a | MUST match S19 |
| D5 | `sig_z = identical_sigmoid(z)` | none | a | new site |
| D6 | `silu'(z)` composite | none | c-elem | |
| D7 | `dY = pinned_mul(d_gate, silu_z)` | none | c-elem | |
| D8 | `dz = ((d_gate * Y) * silu')` | none | c-elem | |
| D9 | `dY_pre = dY` | none | copy | S18 |
| D10 | `dcoef[i,h] = sum_p dY * v` | `P` | b | serves `dD` AND `dqkγ`, 4.2 |
| D11 | `dD[h] = sum_t dcoef` | **tokens** | b | ones-vector |
| D12 | `dqkγ = dcoef` | none | copy | |
| D13 | `dv_18 = pinned_mul(dY, coef)` | none | c-elem | |
| D14 | `d_dot17 = pinned_mul(dY_pre, e_i)` | none | c-elem | S17 |
| D15 | `de17[i,h] = sum_p dY_pre * dot17` | `P` | b | gemm cell |
| D16 | `dq_rot_17 = sum_p d_dot17 * h` | `P` | b | gemm cell |
| D17 | `dh_in_17 = sum_i d_dot17 * q_rot` | `Q` | b | gemm cell |
| D18 | `dM[i,j,h] = sum_p dY_pre * v[j]` | `P` | b | S16 |
| D19 | `dv_16[j] = sum_{i>j} M * dY_pre` | `Q` | **M2-3** | TRANSPOSED strict mask |
| D20 | `ds = pinned_mul(dM, L)` | none | c-elem | |
| D21 | `dL = pinned_mul(dM, s)` | none | c-elem | |
| D22 | `dseg = pinned_mul(dL, L)` | none | c-elem | `exp'` |
| D23 | `dq_rot_16 = sum_j ds * k_scaled` | `Q` | b | gemm cell |
| D24 | `dk_scaled_16 = sum_{i>j} ds * q_rot` | `Q` | **M2-3** | second site |
| D25 | reverse CHUNK recurrence `dh_c` | **chunks** | **T1** | S20 |
| D26 | `dscale_c = sum_{p,n} dh_c * h_{c-1}` | **`P*N`** | **M2-2** | |
| D27 | `d_dacsl_c = pinned_mul(dscale_c, scale_c)` | none | c-elem | |
| D28 | `dincr = dh_c` | none | copy | |
| D29 | `dk_scaled_20 = sum_p dincr * v_dec` | `P` | b | gemm cell |
| D30 | `dv_dec = sum_n dincr * k_scaled` | `N` | b | gemm cell |
| D31 | `dv_20 = pinned_mul(dv_dec, exp(d_rev))` | none | c-elem | |
| D32 | `d_drev[i,h] = sum_p dv_dec * v` | `P` | b | gemm cell |
| D33 | the `exp` chain into `d_dacs`, `d_dacsl` | none | c-elem | |
| D34 | `d_dacs` join, three paths | none | **T6** | ORDER |
| D35 | `d_ADT` from the SEGSUM triangle | **`Q x Q`** | **M2-4** | |
| D36 | `d_ADT` from the in-chunk reverse cumsum | `Q` | **M2-5** | |
| D37 | `dk_rot_15 = pinned_mul(dk_scaled, scale)` | none | c-elem | S15 |
| D38 | `dscale[i,h] = sum_n dk_scaled * k_rot` | `N` | b | gemm cell |
| D39 | `dqk = pinned_mul(dqkγ, γ)` | none | c-elem | S14 |
| D40 | `dγ_14 = pinned_mul(dqkγ, qk)` | none | c-elem | |
| D41 | `dq_pre_14 = pinned_mul(dqk, k_pre)` | none | c-elem | BYPASSES the rotation |
| D42 | `dk_pre_14 = pinned_mul(dqk, q_pre)` | none | c-elem | same |
| D43 | `dq_rot` join, S16 and S17 | none | **T6** | ORDER |
| D44 | the transposed rotation, `q` | none | **M3-1** | section 7.1 |
| D45 | the transposed rotation, `k` | none | **M3-1** | second site |
| D46 | `dθ` from `q`'s pair | none | **M3-2** | section 7.2 |
| D47 | `dθ` from `k`'s pair | none | **M3-2** | second site |
| D48 | `dθ` join, `q` then `k` | none | **M3-7** | ORDER |
| D49 | pairs `>= R`, structural identity | none | copy | DEVIATION 828 |
| D50 | recompute `portable_cosf(θ)` | none | a | never upstream's approx |
| D51 | recompute `portable_sinf(θ)` | none | a | same |
| D52 | `dq_pre` join, rotation path and S14 path | none | **T6** | ORDER |
| D53 | `dk_pre` join, rotation path and S14 path | none | **T6** | ORDER |
| D54 | `dq_bias[h,n] = sum_t dq_pre` | **tokens** | b | ones-vector, 11.4 |
| D55 | `dk_bias[h,n] = sum_t dk_pre` | **tokens** | b | ones-vector |
| D56 | `dC_head[t,n] = sum_h dq_pre` | **`H`** | **T3** | section 1.2 |
| D57 | `dB_head[t,n] = sum_h dk_pre` | **`H`** | **T3** | second site |
| D58 | `dinner_C = pinned_mul(dC_head, w_Cn)` | none | c-elem | S21 |
| D59 | `pWc = pinned_mul(dC_head, inner_C)` | none | c-elem | pre-product |
| D60 | `dw_Cnorm[n] = sum_t pWc` | **tokens** | b | ones-vector |
| D61 | `dot_C[t] = sum_n dinner_C * C` | `N` | c-inh | S1's fold at `N` |
| D62 | `dC_pre` closed form | none | **T7** | |
| D63 | `dinner_B = pinned_mul(dB_head, w_Bn)` | none | c-elem | |
| D64 | `pWb = pinned_mul(dB_head, inner_B)` | none | c-elem | |
| D65 | `dw_Bnorm[n] = sum_t pWb` | **tokens** | b | |
| D66 | `dot_B[t] = sum_n dinner_B * B` | `N` | c-inh | |
| D67 | `dB_pre` closed form | none | **T7** | second site |
| D68 | `dθ` reverse cumsum over the SEQUENCE | **`L`** | **M3-3** | 5.2 item 6 |
| D69 | `d_inc = dθ_t` | none | copy | the mod's slope is exactly 1 |
| D70 | `d_a = pinned_mul(d_inc, dt)` | none | c-elem | |
| D71 | `ddt_10[t,h] = sum_r d_inc * a` | `R = 32` | b | gemm cell |
| D72 | recompute `identical_tanh(angle_raw)` | none | a | |
| D73 | `d_angle_raw = d_a * π * (1 - tanh²)` | none | c-elem | association pinned |
| D74 | `dγ` join, S9 and S14 | none | **T6** | ORDER |
| D75 | `dβ' = dscale` | none | copy | |
| D76 | the SHIFTED scatter of `dβ'` to `t+1` | none | **M3-4** | the off-by-one |
| D77 | `dσ` join, γ leg and β' leg | none | **T6** | ORDER |
| D78 | `dσ` elementwise contributions | none | c-elem | |
| D79 | `dtrap_raw = dσ * σ(1-σ)` | none | c-elem | S8 |
| D80 | `dA_tok = pinned_mul(d_ADT, dt)` | none | c-elem | S7 |
| D81 | `ddt_7 = pinned_mul(d_ADT, A)` | none | c-elem | |
| D82 | `ddt` FOUR WAY join | none | **M3-6** | ORDER, S7/S9γ/S9β'/S10 |
| D83 | `ddd_dt` through `softplus'` | none | c-elem | DEVIATION 1078 |
| D84 | `db_dt[h] = sum_t ddd_dt` | **tokens** | b | ones-vector |
| D85 | the CLAMP derivative | none | **M2-6** | S5, and here the bound BINDS |
| D86 | `heavy_tail'` piecewise | none | **M3-5** | branch on `dd_A`'s sign |
| D87 | `ddd_A` negate | none | c-elem | exact |
| D88 | `d_in = concat` of eight columns | none | copy | |
| D89 | `d_norm_out = d_in . W_in` | `d_in_proj` | b | gemm `dA` |
| D90 | `dW_in = d_in^T . norm.out` | **tokens** | b | `k'=M` |
| D91 | `dinner = pinned_mul(d_norm_out, w_norm)` | none | c-elem | S1-S3 |
| D92 | `pW = pinned_mul(d_norm_out, inner)` | none | c-elem | |
| D93 | `dw_norm[j] = sum_t pW` | **tokens** | b | ones-vector |
| D94 | `drstd[t] = sum_j dinner * x` | `dm` | c-inh | |
| D95 | block-norm `dx_norm` closed form | none | **T7** | THIRD site in this model |
| D96 | `dx = dres + dx_norm` | none | **T8** | THE ABSORPTION SITE |
| D97 | `dv` join, S16 and S18 and S20 | none | **M3-8** | THREE WAY, ORDER |
| D98 | the parameter fold over `b` | **batch** | **T5** | nine parameters |

### 4.2 What is NEW, seam by seam, with what its derivative needs

Category (a) grows by three because the backward reaches for
`portable_cosf`, `portable_sinf` and `identical_tanh` at new sites, all of
which exist (`IDENTICAL_MAMBA3_CONTRACT.md`:147-149).

| forward seam | what the backward needs | covered by anything? |
|---|---|---|
| **S5, data-dependent A** (`:212`), `-heavy_tail(dd_A)` then `clamp(max=-A_floor)` | The piecewise derivative. `heavy_tail(x) = 1+x` for `x >= 0`, `1/(1-x)` for `x < 0`, so `heavy_tail'(x)` is exactly `1` on the right branch and `1/(1-x)^2` on the left. **Two spellings and they are different floats**, `pinned_mul(hv, hv)` reusing the forward's already-rounded value against `identical_div(1, ftz((1-x)*(1-x)))`. Plus the clamp mask. | NO. New elementwise, and the repository's own rule (`IDENTICAL_BACKWARD_PLAN.md`:656-659, "a recomputed forward quantity must be spelled with the same function the forward used") argues for the first spelling and does not decide it. |
| **S9, the trapezoid** (`:216`), `scale_t = γ_t + β'_{t+1}` from a SHIFTED load | The backward SCATTERS across the token boundary. `dscale_t` feeds `dt_t`, `dσ_t`, AND `dt_{t+1}`, `dσ_{t+1}`. So `ddt` is a FOUR way join (S7's ADT path, S9's γ leg, S9's shifted β' leg, S10's angle leg) and the shifted leg is an off-by-one. | NO, and it is `IDENTICAL_BACKWARD_PLAN.md`:160-168's hazard verbatim. **"Getting this wrong produces a plausible gradient that is bit identical on any fixture where all `da` are equal"** becomes, here, bit identical on any fixture where `dt` and `σ(trap)` are uniform across tokens. |
| **S10, the angle recurrence** (`:217`), `θ_t = mod2π(θ_{t-1} + inc_t)` serial over TOKENS | `mod2π(x) = x - 2π*floor(x/2π)` is piecewise affine with slope exactly 1 away from the wraps, so the backward is a PURE REVERSE CUMULATIVE SUM of `dθ` over the whole sequence, then `d_inc_t = dθ_t`, `d_a = pinned_mul(d_inc, dt)`, `ddt += pinned_mul(d_inc, a)`, and `d(angle_raw) = d_a * π * (1 - tanh²)`. | PARTLY. The reverse chain is T1's topology with the coefficient pinned to exactly `1.0`, so it degenerates to a fold whose fusion question disappears. `tanh'` is new elementwise. **The whole-sequence length is the finding, section 5.** |
| **S11, cos/sin** (`:218`) | Recompute at the carded `θ` with the SAME functions the forward used. DEVIATION 828 (`:391-405`) refuses upstream's three trig spellings; the backward inherits that refusal unchanged and must not import upstream's `cos_approx`/`sin_approx`. | YES for the primitives, NO for the site. |
| **S13, the rotation** (`:220`) | Two things, not one. A transposed rotation on the input, and a term through `θ`. Section 7 is entirely about this. | HALF. `transformer/checks/transformer_backward.mojo`:996 `bwd_rope_kernel` covers the input half's IDIOM and covers neither its PAIRING nor the `θ` term. |
| **S14, the pre-rotation QK dot** (`:221`, DEVIATION 830) | `dq_pre` and `dk_pre` each acquire a SECOND path that bypasses the rotation entirely, because S14 contracts the PRE rotation values. So the join at `dq_pre` mixes one term that came through three roundings of rotation with one that did not, and the join ORDER is a pin. Also `dγ` gets a leg here. | NO. |
| **S16, the strict-causal attention** (`:223`) | `dq_rot`, `dk_scaled`, `dv` and `dseg`, exactly the transformer's `bwd_dq`/`bwd_dk`/`bwd_dv` shapes. The diagonal is STRUCTURALLY excluded forward (DEVIATION 830, `:421-433`), so no gradient flows through it and the backward's mask is the same strict `>`. | STRUCTURE yes, SPELLING no. Section 6.2 explains why the transformer's kernels cannot be called. |
| **S18, the diagonal plus D in ONE add** (`:225`) | Because `t = ftz(D[h] + qkγ_i)` is ONE value multiplied by `v`, a single coefficient gradient `dcoef[i,h] = sum_p dY * v` serves BOTH `dD` and `dqkγ`. **Mamba-3 therefore has no two axis `dD` fold**, where Mamba-2 does (M2-1). One seam decision in the forward removed a topology from the backward. | Its own, and it is cheaper than Mamba-2's. |
| **S21, the B/C RMSNorm** (`:228`) | The SECOND and THIRD RMSNorm backward in this model, at row length `N = 128`, per `(b, l, g)`, fed by the head fold of section 1.2. | T7's topology, three sites in this model. |

### 4.3 The eight new topologies of Mamba-3, beyond Mamba-2's six

- **M3-1, the transposed rotation.** Section 7.
- **M3-2, the `θ` derivative of the rotation.** Section 7.
- **M3-3, the whole-sequence reverse `θ` chain.** Section 5.
- **M3-4, the trapezoid's shifted scatter.** The off-by-one, table above.
- **M3-5, the heavy-tail piecewise derivative.** A branch on the sign of the
  INPUT, evaluated in the backward, which means the backward must read
  `dd_A` and not merely `A.out`. The forward's own branchless spelling is
  declared bit-equal to the branch spelling (`:212`); **that argument does
  not transfer to the derivative** and must be redone or refused.
- **M3-6, the four-way `ddt` join.**
- **M3-7, the `dθ` join across q and k.**
- **M3-8, the three-way `dv` join** (S16's attention path, S18's diagonal
  path, S20's state path).

Plus five of the six in 2.3. M2-1, the two axis `dD` fold, is DELETED by
S18's single add, 4.2.

---

## 5. The numerically dangerous seams

`IDENTICAL_BACKWARD_PLAN.md`:505-516 states the rule that governs this whole
section.

> **That sentence cannot be written about the backward.** T3 contracts over
> `d_inner` and the `(b, d)` grid gives each `d` to a different thread; T5
> contracts over `b`; every category (b) token reduction contracts over `M`
> ... So launch invariance stops being a property of the kernel's SHAPE and
> becomes a property of each crossing fold's PARTITION, under one rule stated
> once.
>
> > **Every crossing fold's partition is a pure function of the length of the
> > axis it reduces, and of nothing else.**

### 5.1 The crossing folds, and what each one's length is a function of

| fold | Mamba-1 | Mamba-2 | Mamba-3 | length is |
|---|---|---|---|---|
| shared-channel B/C contraction (T3) | `d_inner` | `d_inner` and `H` | `H` | a PROFILE CONSTANT |
| in-chunk contractions (S12/S14/S16/S18 backwards) | none | `Q`, `N`, `P` | `Q`, `N`, `P` | PROFILE CONSTANTS |
| segsum triangle | none | `Q x Q` | `Q x Q` | a PROFILE CONSTANT |
| state-pass `dscale` | none | `P*N` | `P*N` | PROFILE CONSTANTS |
| reverse state recurrence | `L`, in thread | `C = ceil(L/Q)`, in thread | `C`, in thread | a function of `L` |
| reverse ANGLE chain | none | none | **`L`, in thread** | a function of `L` |
| batch fold (T5) | `B` | `B` | `B` | the launch composition |
| token-axis parameter folds | `M` | `M` | `M` | the MICROBATCH SCHEDULE |

**The finding is favourable and it should be said plainly. Every crossing
fold in both siblings that is not a parameter gradient has a length that is a
PROFILE CONSTANT.** `Q`, `N`, `P` and `H` do not move with the batch, the
sequence, the launch or the vendor, because the contracts freeze them
(`IDENTICAL_MAMBA2_CONTRACT.md`:86-98,
`IDENTICAL_MAMBA3_CONTRACT.md`:163-176). So `contract_partition` of each one
is a compile-time answer and 4.3's rule is satisfiable by construction
rather than by care. That is a strictly stronger position than Mamba-1's,
whose T3 fold is over a model width and whose T4 fold is over the sequence.

The parameter gradients keep Mamba-1's answer unchanged.
`IDENTICAL_BACKWARD_PLAN.md`:439-445 says asking a sum over the batch to be
invariant to its own terms "is not a coherent request", and :447-457 lists
ten structurally impossible microbatch equivalences. Both siblings have the
same shape of parameter list and inherit that verbatim. The clause worth
gating is the split one, activation gradients invariant and parameter
gradients gated at fixed `(B, L)`, and a backward profile document for either
sibling must split it (`:813-819`).

### 5.2 Where the reduction order is harder to pin than the forward's

**Mamba-2.**

1. **`dD`'s two axis fold (M2-1).** The only operation in either sibling
   whose ORDER involves choosing between two axes rather than a direction
   within one. Fold `P` first then tokens, or tokens first then `P`, and the
   two are different numbers.
2. **The segsum triangle (M2-4).** A fold whose OUTPUT index appears in both
   loop bounds. The partition rule as stated covers a fold over one axis of
   known length; a triangle is two axes with a dependent bound and the rule
   needs restating before it can be applied.
3. **`dB` and `dG` at length `H`.** `H = d_inner / 64`. At `d_model = 32`,
   the gate shape (`:103`), `H = 1` and **the fold is a single term, so every
   sabotage arm against its order is INERT.** A gate whose shape set is
   `d_model` in `{32, 64}` gives `H` in `{1, 2}` and can never witness a
   partition defect. This is `IDENTICAL_BACKWARD_PLAN.md`:688's lesson, "each
   gate must assert its arm's predicted cell count, not merely that the arm
   ran", pointed at a specific shape.
4. **The transposed triangular contraction (M2-3)**, inert on a symmetric
   fixture.
5. **Every token-axis parameter fold**, ten of them, all `k' = M`.

**Mamba-3, all of the above minus `dD`, plus four.**

6. **The whole-sequence reverse `θ` chain (M3-3).** Mamba-3's angle state
   crosses chunk boundaries by construction (`:217`, `θ₋₁ = angle state in`)
   where Mamba-2's cumsum resets (`:134`). So this ONE chain has length `L`
   and every `dθ_t` depends on every token from `t` to `L-1`.
   `IDENTICAL_BACKWARD_PLAN.md`:459-471 proves sequence-length invariance
   structurally impossible for exactly this reason, and Mamba-3 is the only
   one of the three where it bites on a quantity that is not the SSM state.
   **Consequence, and it should be stated in any Mamba-3 backward contract's
   section 5.** `IDENTICAL_MAMBA3_CONTRACT.md`:242-251 earns decode-equals-
   prefill for the FORWARD from prefix stability, "a prefill of length t
   gives token s < t the scale `γ_s + β'_{s+1}` using only tokens <= t". The
   backward has no prefix property at all, so **there is no decode backward
   for Mamba-3 for the same reason there is none for Mamba-1**, and the
   trapezoid does not rescue it.
7. **The `dθ` join across q and k (M3-7)** and the **`dq_pre` join across the
   rotation path and the S14 path.** Both are two-term joins whose terms have
   different rounding histories, and both are INERT whenever one term
   dominates.
8. **The trapezoid's shifted scatter (M3-4)**, inert on uniform `dt`.
9. **Both rotations, section 7.**

**No operation in either backward needs an atomic**, and the reason is
Mamba-1's, restated. Every crossing fold has a private-slot spelling
(`IDENTICAL_BACKWARD_PLAN.md`:332-345, T5) and Metal's absence of threadgroup
float atomics forced that spelling to exist anyway. Upstream needs
seventeen atomics between the two models. Section 6.

---

## 6. Upstream, read rather than assumed

`IDENTICAL_BACKWARD_PLAN.md`:58-62 records that upstream's Mamba-1 backward
is not reproducible run to run on one device. **Both siblings are worse, and
Mamba-3 is worse in a way that is new.** Read from
`/Users/andrewhendel/CascadeProjects/upstream/mamba` at the pin the contracts
name, `e9594ce`.

### 6.1 Mamba-2's backward exists upstream and its `dB` fold is partitioned by the SM COUNT

This is the sharpest finding in this document.

    mamba_ssm/ops/triton/ssd_chunk_state.py:937-940
      sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
      nheads_per_program = max(min(math.ceil(batch * nchunks * nheads / sm_count),
                                   nheads_ngroups_ratio), 1)
      nsplits = triton.cdiv(nheads_ngroups_ratio, nheads_per_program)
      dB = torch.empty(batch, seqlen, nsplits, ngroups, dstate, ...)

The same four lines appear at `ssd_chunk_scan.py`:1485-1488 for `dC` and at
`ssd_chunk_scan.py`:1534-1537 for `dcb`. **These are exactly the three sites
section 1.1 identifies as the shared-channel contraction.** Upstream splits
that fold into `nsplits` partial buffers and the split count is a function of
`torch.cuda.get_device_properties(...).multi_processor_count`. Their `dB`,
`dC` and `dcb` summation order is therefore a function of the GPU's SM count,
which is not merely vendor-varying but SKU-varying inside one vendor. An
A100 and an H100 running the same weights on the same tokens compute
different `dB` bits.

Twelve live `tl.atomic_add` calls sit beside it, on `dD`
(`ssd_combined.py`:248), `ddt` (`:254`, `ssd_chunk_state.py`:363,
`ssd_chunk_scan.py`:737), `dA` and `ddt_bias` (`ssd_chunk_state.py`:167,
`:174`), and `ddA_cumsum` at five sites (`ssd_chunk_state.py`:370, `:371`,
`:480`, `:617`, `ssd_chunk_scan.py`:604, `:1256`).

And upstream ships TWO implementations of the `ddA_cumsum` seam and names one
of them unstable, `_chunk_scan_bwd_ddAcs_unstable_kernel`
(`ssd_chunk_scan.py`:889) against `_chunk_scan_bwd_ddAcs_stable_kernel`
(`:1098`). **That is M2-4, the segsum triangle backward, and upstream's own
naming records that they found it numerically delicate.**

`mamba_ssm/utils/determinism.py`:21-27 gates their deterministic fallbacks
behind a `MAMBA_DETERMINISTIC` environment variable, which
`FEATURE_PARITY.md`:173 already cites as evidence the ambition is real and
theirs is not there yet.

### 6.2 Mamba-3's backward AUTOTUNES ITS CHUNK SIZE

`mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py` exists, 1788 lines, and its
first decorated kernel opens with this.

    :22-26
      @triton.autotune(
          configs=[
              triton.Config({"CHUNK_SIZE": cs}, num_stages=s, num_warps=w, maxnreg=r)
              ...

The candidate set is `for cs in [32, 64]` (`:26`), and the Mamba-3 forward
runs at 64 (`IDENTICAL_MAMBA3_CONTRACT.md`:171). **So upstream's backward may
select a chunk boundary its own forward did not use, inside one training
step.** The same autotune block appears at `:1419-1428`. **`CHUNK_SIZE` is a
tuned parameter of their backward.** The Mamba-2 forward contract's DEVIATION 783
(`IDENTICAL_MAMBA2_CONTRACT.md`:330-337) says `CHUNK_SIZE` "is a profile
constant with exactly the standing of gemm v1's `K_LEAF_MIN`/`MAX_LEAVES`",
and the Mamba-3 contract inherits that standing at `Q = 64` (`:171`). So
upstream's backward selects its summation boundary by benchmark, per shape,
per device, at run time. That is the forward fragility note
(`IDENTICAL_MAMBA3_CONTRACT.md`:101-110, the Blackwell `num_stages > 1`
silent corruption) reappearing on the backward as a design choice rather than
a bug.

Five `tl.atomic_add` sites follow, on `dK` and `dK_bias` (`:1138`, `:1142`),
on `dAngles` (`:1167`), and on `dDT` and `dTrap` (`:1619`, `:1620`), with a
comment at `:1064` conceding the interaction.

    # NOTE: Do not autotune this kernel. It overwrites dK, dK_bias, dAngles
    # via atomic adds and autotuning will lead to multiple overwrites.

**Upstream's own comment says their autotuner and their atomics are
incompatible and that the guard is a convention.**

`dq_bias_partial` and `dk_bias_partial` at `:1286` show they also carry the
private-slot idiom, so the two spellings coexist in one file.

### 6.3 What we must COPY, and what we must NOT

COPY, in both siblings. The structure of the reverse recurrence over chunks.
The identification of which quantity each seam's derivative needs, which is
what sections 2 and 4 transcribe. And, for Mamba-3, the FACT that the θ path
exists at all, since it is easy to write a rotation backward that omits it.

DO NOT COPY. The seventeen atomics. The three `sm_count` head-fold splits.
The autotuned `CHUNK_SIZE`. The `cos_approx`/`sin_approx` recompute
(`mamba3_siso_bwd.py`:980-981, `:1128-1129`), which is refused by DEVIATION
828 on the forward side and must be refused again here for the identical
reason. And the general rule
(`IDENTICAL_BACKWARD_PLAN.md`:656-659) applies unchanged, a recomputed
forward quantity must be spelled with the same function the forward used.

---

## 7. Mamba-3's rotation, specifically

The forward S13 pins three roundings per component and the 2026-09-03 run
record says what it cost to enforce that (`IDENTICAL_MAMBA3_CONTRACT.md`
:596-618).

> S13's rotation combined two products as `a*b - c*d` over `pinned_mul`, and
> `pinned_mul(a, b)` is `identical_mul_add(a, b, -0.0)` mathematically
> `a * b`, so a backend may simplify it and then re-contract the difference
> into a single fma. ONE rounding where S13 asks for three.

And the lesson (`:615-618`).

> **THE LESSON.** A row of a numeric contract that states a property no
> spelling defends is a comment. S13 read UNFUSED for the whole of its life
> and nothing enforced it.

### 7.1 The input half. The same hazard, at four sites per pair instead of two

The forward, `:220`.

    ro0 = ftz(ftz(pinned_mul(x0,c)) - ftz(pinned_mul(x1,s)))
    ro1 = ftz(ftz(pinned_mul(x0,s)) + ftz(pinned_mul(x1,c)))

The rotation matrix is orthogonal, so the exact derivative with respect to
the input is the transpose, which is the rotation by the negated angle.

    dx0 = ftz(ftz(pinned_mul(d0,c)) + ftz(pinned_mul(d1,s)))
    dx1 = ftz(ftz(pinned_mul(d1,c)) - ftz(pinned_mul(d0,s)))

Upstream writes exactly this shape (`mamba3_siso_bwd.py`:1012-1015).

    dq0 = dQ_in_r0 * cos_angle + dQ_in_r1 * sin_angle
    dq1 = -dQ_in_r0 * sin_angle + dQ_in_r1 * cos_angle

**Every one of those four lines is a sum or difference of two products and
every one is re-contractable into an fma.** The barrier idiom applies
verbatim and for the same mechanism, `ftz`'s bitcast-compare-select gives the
`fmul` a second use and hands the add a `select` (`:608-613`). Two points
that are NOT a restatement of the forward's finding.

1. **The plus form is as exposed as the minus form and is harder to catch.**
   The forward bug surfaced as a one ULP disagreement on FOUR card stages and
   was caught only because AMD produced the unfused answer while Apple and
   NVIDIA contracted (`:596-606`). Nothing guarantees a three-column split
   next time. **The barrier must be spelled because the contract asks for it,
   not because a column disagreed.**
2. **Two tensors, so eight sites per rotated pair.** `q` and `k` both rotate
   (`:220` serves both), and the backward runs each. Applied at all 14 sites
   the forward fix touched (`:611`), the backward's site count is larger.

### 7.2 The angle half, which the forward has no counterpart for

`θ` in Mamba-3 is DATA DEPENDENT. It comes from `in_proj` through
`angle_raw`, `identical_tanh`, `π` and `dt`, accumulated serially (S10,
`:217`). So the rotation's derivative has a second term that the transformer
lane's RoPE backward does not have at all, because there `cos`/`sin` are a
precomputed table at absolute positions
(`transformer/checks/transformer_backward.mojo`:996-1055).

    dθ = d0 * d(ro0)/dθ + d1 * d(ro1)/dθ
       = d0 * (-x0·s - x1·c) + d1 * (x0·c - x1·s)

Upstream spells that literally (`mamba3_siso_bwd.py`:1044-1046).

    dtheta_q = dQ_in_r0 * (-Q_r0*sin - Q_r1*cos) + dQ_in_r1 * (Q_r0*cos - Q_r1*sin)
    dtheta_k = dK_in_r0 * (-K_r0*sin - K_r1*cos) + dK_in_r1 * (K_r0*cos - K_r1*sin)
    dtheta   = dtheta_q + dtheta_k

**There are two spellings and they are different floats, and choosing between
them is the first decision a Mamba-3 backward makes.**

- **Spelling (i), upstream's.** Six products, three sum-or-difference-of-two-
  products combinations, all contractable. Recomputes the rotation from
  `x0`, `x1`, `c`, `s`.
- **Spelling (ii), the collapsed one.** `(x0·c - x1·s)` IS the forward's
  `ro0` and `-(x0·s + x1·c)` IS `-ro1`, so
  `dθ = ftz(ftz(pinned_mul(d1, ro0)) - ftz(pinned_mul(d0, ro1)))`. Two
  products, one difference, ONE contraction site.

Spelling (ii) is available only because the forward CARDS the rotated values
pre-scale, `rot.q [M,H,N]` and `rot.k [M,H,N] (PRE-scale)`
(`IDENTICAL_MAMBA3_CONTRACT.md`:308-309), and S15 confirms the carried k
state is the pre-scale value (`:222`). The repository's own recompute rule
(`IDENTICAL_BACKWARD_PLAN.md`:656-659) points at (ii) and the house
transliteration rule points at (i). **This document does not decide it. It
names it as a deviation and observes that (ii) has one third the contraction
hazard and reuses a value the card already holds, which is the argument T2
lost and B4 won.** Whichever is picked, the other is the required-RED
sabotage arm.

Three further hazards specific to the angle half.

- **`dθ_q + dθ_k` is a JOIN and its order is a pin** (M3-7). Upstream pins q
  then k (`:1046`). Predicted INERT whenever one tensor's gradient dominates.
- **`cos` and `sin` must be recomputed with `portable_cosf` and
  `portable_sinf`**, never with anything algebraically equal. `θ` is a card
  stage (`angle.theta`, `:307`) so the recompute is exact by construction, in
  the shape of DEVIATION 1420 in the transformer lane
  (`transformer_backward.mojo`:858-865). Upstream recomputes with the PTX
  approximations, which the profile already refuses.
- **The unrotated pairs are a STRUCTURAL COPY and a sabotage arm against them
  is INERT BY CONSTRUCTION.** `rope_fraction = 0.5` leaves pairs 32 through
  63 unrotated (`:167`), and DEVIATION 828 records that the computed and
  structural spellings coincide bit for bit because `cos(+0.0) = 1.0` and
  `sin(+0.0) = +0.0` exactly (`:401-405`). The backward inherits that
  coincidence. **Any gate that expects an arm there to fire must declare it
  VACUOUS in advance**, the way the mamba3 forward declared
  `FOLD_SERIAL_ZERO_SEED` vacuous by construction (`:528-530`).

### 7.3 What the transformer's RoPE backward does and does not give us

`transformer/checks/transformer_backward.mojo`:996 is a real, written,
transposed-rotation backward with an explicit refusal of the fma
(`SAB_B10_ROPE_BWD_FUSED`, `:1085-1090`) and the `ftz(ftz(pa) + ftz(pb))`
spelling this document recommends. **It cannot be called from a Mamba-3
backward and must not be copied verbatim, for three reasons.**

1. **The pairing is wrong.** It rotates HALVES, `(j, j+half)` (`:1023-1026`),
   which is HuggingFace's checkpoint layout. Mamba-3 pins INTERLEAVED pairs
   `(2i, 2i+1)` (`:220`), and `ROTATE_HALF_SPLIT` is a mamba3 sabotage arm
   precisely against the halves spelling (`:356`).
2. **It has no angle term**, section 7.2.
3. **It is not gated.** `transformer_backward.mojo`:8-16 says the file
   "COMPILED AND RUN 2026-09-03, NOT YET GATED" and that the check "REFUSED
   TO CERTIFY".

What DOES transfer is the idiom and the sign-convention warning
(`:1025-1032`), that the transpose is one character in two branches and
produces "a plausible correctly-shaped wrong gradient that is bit identical
on three vendors", INERT at the position where `sin` is exactly `+0.0`. In
Mamba-3 that inert position is `θ = 0`, which is every token of a fixture
whose `angle_raw` is zero, and the mamba3 gate set must plant nonzero angles
for the same reason `DIAG_INCLUDE_SUBTRACT` needs them (`:353`).

---

## 8. What can be REUSED, and the trap in the word

The parent question named six pieces as "all already gated". **Three of the
six are not gated, and the states differ enough to matter.**

| piece | where | actual state |
|---|---|---|
| gemm backward routing (dgrad/wgrad/bias) | `gemm/checks/gemm_backward.mojo`:1-45 | **GATED on Apple and AMD**, and `IDENTICAL_BACKWARD_PLAN.md`:16 adds "no sabotage arm ever fired" |
| RMSNorm backward | `transformer/checks/transformer_backward.mojo`:801, :824, :932 | **GATED THREE-VENDOR 2026-09-03**, all six clauses on Apple, NVIDIA and AMD, card md5 7eee8da90ecb4dba1d154991e2e67e30 identical on all three. It DID refuse to certify on its first three runs; the cause was the operator running an identity gate in FAST mode, where `ftz` is the identity and the unfused arm gets contracted into the fused one, so no fixture could separate them |
| SiLU backward | same file, `:713-787` | same, NOT GATED |
| softmax backward | same file, `:1101`, `:1180` | same, NOT GATED, **and it has NO SITE in either sibling** |
| AdamW | `training/checks/optimizer.mojo` | compiled, run and gated, **on ONE device, not three** (`:5-22`) |
| embedding backward | `embedding/checks/embedding_check.mojo` | clauses (a),(b),(c),(e),(f) print PASS; clause (d) is half gated (`:2685-2689`). **No site inside a block**, model level only |

### 8.1 What routes cleanly

- **Every projection gradient.** In-proj and out-proj, both siblings, through
  `gemm_backward.mojo`'s six-row table entered at the block's shapes. This is
  `mamba/checks/mamba_backward.mojo`'s whole existing job (`:14-21`) and it
  generalizes without change.
- **Every token-axis parameter reduction**, as a pre-product buffer plus a v1
  ones-vector `OP_NN` at `(1, W, M)`. Nine parameters in Mamba-2, nine in
  Mamba-3.
- **Every in-chunk contraction**, because their contracted axes are profile
  constants. Section 5.1.
- **The SiLU derivative's association.** `transformer_backward.mojo`:782-786
  spells `sg * (1 + x*(1 - sg))` and `IDENTICAL_BACKWARD_PLAN.md`:91 and :397
  spell `sig * (1 + z*(1 - sig))` left to right. **These are the same
  spelling.** It is the one clean reuse in this table.

### 8.2 The trap. Two RMSNorm backwards already exist in this repository and they are different numbers

`IDENTICAL_BACKWARD_PLAN.md` derives the RMSNorm backward twice and says so.
Nodes `S1'` through `S3'` (`:130-134`) and then, at `:137-138`, "collapse to
the closed form and are spelled that way in topology T7". T7 is at `:359-372`.

    c3   = ftz(pinned_mul(ftz(pinned_mul(rstd, rstd)), rstd))
    s    = ftz(identical_div(c3, Float32(dm)))
    t2   = ftz(pinned_mul(ftz(pinned_mul(s, x[t,j])), drstd[t]))
    t1   = ftz(pinned_mul(rstd, dinner[t,j]))
    dx_norm[t,j] = ftz(t1 - t2)

The transformer lane spells the SAME derivative NODE BY NODE
(`transformer_backward.mojo`:824-931 and :932-989).

    r2 = pinned_mul(rstd, rstd);  r3 = pinned_mul(r2, rstd)
    cr3 = pinned_mul(c, r3);  da = pinned_mul(-0.5, cr3)
    dv = identical_div(da, dm)
    dx1 = pinned_mul(dh_j, rstd);  tx = pinned_mul(2.0, x_j)
    dx2 = pinned_mul(dv, tx);  dx = ftz(ftz(dx1) + ftz(dx2))

and defends the choice explicitly, "**`-0.5` AND `2.0` ARE SPELLED, NOT
ELIDED**" (`:865-869`). **T7 folds `-0.5` and `2.0` into a subtraction; the
transformer keeps them as two exact power-of-two scalings and an addition.
They compute the same real number through a different sequence of roundings
and they are different floats.**

This is the parent's warning made concrete. **Mamba-1 needs one RMSNorm
backward, Mamba-2 needs two (block norm and gated norm), Mamba-3 needs three
(block norm, B norm, C norm), and the transformer already has one.** Seven
sites. If each lane picks locally, this repository will carry two different
RMSNorm gradients and no gate anywhere will notice, because each is
self-consistent against its own oracle. **A backward profile that does not
pin ONE of the two spellings across all seven sites has already failed.**

The same class of trap sits at three smaller sites.

- `identical_silu` is the ONE-division spelling in all three mamba profiles
  (`IDENTICAL_MAMBA2_CONTRACT.md`:130, `IDENTICAL_MAMBA3_CONTRACT.md`:226).
  Its derivative needs `sigmoid` SEPARATELY, which is a new call, and
  `IDENTICAL_BACKWARD_PLAN.md`:638-648 records that upstream's own backward
  breaks exactly here and multiplies by a `silu(z)` its own forward never
  produced.
- `softplus'` has two spellings, a multiply by `identical_sigmoid` or
  upstream's single division, and the distinguishing band is `biased` in
  roughly `[8, 14]` (`IDENTICAL_BACKWARD_PLAN.md`:408-424). All three models
  hit it (mamba1 S14, mamba2 S9, mamba3 S6). **Answer once.**
- `exp'` is `exp`, so every decay derivative reuses the forward's rounded
  value rather than recomputing. Both siblings have four such sites.

### 8.3 What does NOT route, and the precedent that says why

The transformer lane's largest finding is directly load bearing here
(`transformer_backward.mojo`:1294-1304, DEVIATION 1402).

> **A DERIVATIVE SWAPS WHICH AXIS IS CONTRACTED**: `dA` contracts over the
> OUTPUT WIDTH, and S11's output width is the KV AXIS. So the routed `dq`
> would be an `OP_NN` at `(l, hd, s)` with `k' = S`, and `P = f(S)` builds
> one tree at `S = 257` and a different one at `S = 200`.

**Both siblings ESCAPE that finding and the reason is the chunking.** The
axis a Mamba-2 or Mamba-3 intra-chunk derivative swaps into is `Q`, a profile
constant, never a sequence length. `contract_partition(Q)` builds one tree at
every `L`. This is the strongest structural argument in favour of the
siblings' backwards over Mamba-1's and it should lead any summary of this
document. It is also the reason DEVIATION 783's standing matters more in the
backward than in the forward, section 3.

---

## 9. What is SHARED between all three backwards, and could be built once

Twelve items. Each is used by at least two of the three, most by all three,
and every one of them is currently unbuilt.

| # | the shared piece | Mamba-1 | Mamba-2 | Mamba-3 | transformer |
|---|---|---|---|---|---|
| 1 | the gemm backward routing table | yes | yes | yes | yes, and BUILT |
| 2 | the ones-vector token reduction, pre-product plus `OP_NN` at `(1,W,M)` | 6 sites | 10 | 9 | yes |
| 3 | **T5**, the private-slot batch fold with NO atomic | yes | yes | yes | applicable |
| 4 | **T7**, ONE RMSNorm backward spelling | 1 site | 2 | 3 | 1 |
| 5 | **T8**, the residual absorption join and its per-stage gate discipline | yes | yes | yes | yes |
| 6 | **T3**, the shared-channel contraction, SHAPE PARAMETERIZED by the crossed length | `di` | `di`, `H` | `H` | no |
| 7 | the reverse first-order linear recurrence, ONE combine operator | over `L` | over `C` | over `C`, and over `L` at coefficient 1 | no |
| 8 | `silu'` composite plus the `sigmoid` recompute | yes | 2 sites | 1 site | yes |
| 9 | `softplus'`, one answer to DEVIATION 1078's question | yes | yes | yes | no |
| 10 | `exp'` as the forward's rounded value reused | yes | 4 sites | 4 sites | no |
| 11 | a backward `refuse_nonfinite` for `dres`, which does not exist anywhere (`IDENTICAL_BACKWARD_PLAN.md`:811-812) | owed | owed | owed | owed |
| 12 | the SPLIT invariance clause, activation gradients invariant and parameter gradients not (`:813-819`) | owed | owed | owed | owed |

**Items 3, 4, 6, 7, 11 and 12 are the ones worth extracting deliberately.**
The rest fall out of ordinary code sharing. Item 4 is the one that will be
got wrong if it is not extracted, section 8.2. Item 6 is the one that will be
got wrong if it IS extracted carelessly, because the same topology at length
`H` and at length `d_inner` may have different partitions, section 5.2.

---

## 10. The recommended ORDER, with the reasoning

**Step 0. Close the forward legs before starting any backward.**

Neither sibling has a three-vendor forward at ONE commit today. Mamba-2's
NVIDIA column is at a different commit and the single-commit card is owed
(`IDENTICAL_MAMBA2_CONTRACT.md`:543-548). Mamba-3's NVIDIA column is unrun
(`:624-625`). `FEATURE_PARITY.md`:173 already names the Mamba-2 backward's
trigger as "the Mamba-2 FORWARD profile gated on three vendors AND the
Mamba-1 backward gates green", and neither half holds. A backward built on a
forward that has not closed inherits an open question and cannot make its own
sentence.

**Step 1. The Mamba-1 backward, phases A through D of its own ladder**
(`IDENTICAL_BACKWARD_PLAN.md`:741-748).

Because it is the smallest, because its plan already exists and its
topologies are already declared, because its forward IS closed on three
vendors (`:14`), and above all because **five of its eight topologies (T1,
T3, T5, T7, T8) are exactly the pieces both siblings need.** Building them
inside a lane that has a written plan is cheaper than inventing them inside a
lane that does not.

One carve-out. **Do NOT let T3's memory problem block the shared kit.**
`IDENTICAL_BACKWARD_PLAN.md`:564-581 calls the blocked T3 "the largest single
open item of the lane, ahead of everything in section 7". That problem is
Mamba-1 SPECIFIC. Section 2.2 shows the chunked forward materializes state
only at chunk boundaries, so the siblings never face it. Sequencing the
shared kit behind a Mamba-1-only memory design would be sequencing it behind
a problem the consumers do not have.

**Step 2. A deliberate shared-kit pass, before Mamba-2.**

Section 9's items 3, 4, 6, 7, 11 and 12, each with its own gate. Item 4 in
particular has to be reconciled against the transformer lane's existing
node-by-node spelling and the reconciliation is a deviation, not a cleanup.

**Step 3. Mamba-2's backward.**

It adds M2-1 through M2-6 and DELETES T2 and T4 (section 2.2). Its
launch-invariance argument is CLEANER than Mamba-1's because every crossing
fold that is not a parameter gradient has a profile-constant length (section
5.1). It has an upstream backward to read statement by statement, which
Mamba-3's plan will want as precedent.

**Step 4. Mamba-3's backward, last.**

Everything in step 3 is a prerequisite and Mamba-3 adds strictly on top,
M3-1 through M3-8. This is the same staggering argument the Mamba-3 FORWARD
contract already makes for itself (`IDENTICAL_MAMBA3_CONTRACT.md`:28-34 and
:465-472, "the shared substrate must exist ONCE, in the Mamba-2
implementation round, and be certified"), and it holds a fortiori for the
backward, which shares MORE substrate than the forward does.

**The one ordering constraint that is not negotiable, at every step. The
FIXTURE work comes before the kernels.**

The precedent is two days old. `transformer/checks/transformer_backward.mojo`
:8-16 records that the file and its oracle and its check "compiled cleanly on
the FIRST attempt and the check's preflight assertions all passed. The check
then REFUSED TO CERTIFY: its `d_out` fixture cannot separate a fused
multiply-add chain from an unfused one, so three sabotage arms are
unfalsifiable", leaving twenty arms of which "**not one has yet been shown
able to fail**" (`:22-25`). Both siblings' backwards are dense with
sum-or-difference-of-two-products seams whose only sabotage arms are exactly
the fused-versus-unfused arms that fixture failure kills. Section 7 is the
whole rotation. **Design the `d_out` fixture that separates them first.**

**An alternative order, and why it is rejected.** Build Mamba-2's backward
first, because its forward is more mature and its chunked form is cheaper in
memory. Rejected because five topologies would then be invented inside the
Mamba-2 lane while Mamba-1's plan sits declaring the same five, and the two
would have to be reconciled afterwards. That is the situation the Mamba-2
forward contract forbids for itself in one sentence
(`IDENTICAL_MAMBA2_CONTRACT.md`:30-32), "the two Mamba lanes may not answer
the same question twice".

---

## 11. What I could not determine from the documents

Listed rather than guessed.

1. **Upstream's backward ARITHMETIC, statement by statement.** Section 6
   reports their STRUCTURE, read for this document, kernel names, atomics,
   `sm_count` splits, autotuned `CHUNK_SIZE`, and the two rotation
   expressions. **I did not transliterate their associations, fold
   directions, seeds or fusion decisions**, which is what
   `IDENTICAL_BACKWARD_PLAN.md` section 6 did for Mamba-1 and what a real
   backward lane owes for each sibling.
2. **The memory cost of either backward.** No document prices one. The card
   stage shapes give the algebra (`IDENTICAL_MAMBA2_CONTRACT.md`:223-248,
   `IDENTICAL_MAMBA3_CONTRACT.md`:296-323) and the direction of the trade is
   visible, the state term shrinks as `1/Q` where the `[B,C,H,Q,Q]` mask
   terms grow as `Q`, so `Q` trades one against the other. **I have not
   priced it and will not invent a model shape to price it at.**
3. **`identical_clamp`'s DERIVATIVE at the boundary.** The function exists
   (`checks/numerics.mojo`:2273) and DEVIATION 788 defines its forward
   zero-sign and NaN behavior. What the gradient does at `x == lo` exactly is
   defined nowhere. Mamba-2's S9 and Mamba-3's S5 both need it and Mamba-3's
   bound structurally BINDS (`IDENTICAL_MAMBA3_CONTRACT.md`:150).
4. **The shape of Mamba-3's `q_bias` and `k_bias`.** The contract calls them
   "learned per-head biases ... ones-initialized (:121-122)" (`:63-64`) and
   S12 spells `qb[h,n]` (`:219`), so `[H, N]` is the reading, but the
   constants table (`:161-176`) does not list it and I did not read
   `mamba3.py`. **The reduction axis of `dq_bias` depends on it.**
5. **Whether the forward implementations RETAIN the intermediates a backward
   would read.** Both contracts allow `seg.L` and `cb.G` to be ELIDED from
   the card under a size cap (`IDENTICAL_MAMBA2_CONTRACT.md`:250-252,
   `IDENTICAL_MAMBA3_CONTRACT.md`:325). A backward needs `L` and `G` PRESENT
   as device buffers regardless of whether they are carded, and no document
   says whether the implementations keep them.
6. **Whether a Float64 gradient reference is obtainable for either.**
   `IDENTICAL_BACKWARD_PLAN.md`:794-803 records that Mamba-1 has none,
   because "`torch.autograd` over einsums" has no fixed contraction order.
   Mamba-2 ships an explicit backward so a float64 reference may be
   constructible from its autograd Function; I did not check whether that
   Function admits float64. **Mamba-3 is worse and the contract says why**,
   `:74-79`, "No second reference. HF transformers @ `d56c55b` has NO mamba3
   model ... This is a ONE-REPOSITORY profile". A Mamba-3 backward would be a
   one-repository, one-implementation cross-check, weaker than either
   sibling's.
7. **Which deviation band either backward would spend.** None is allocated
   and this document allocates none.
8. **Whether the Mamba-3 Python surface's ten-piece state and consumed
   `pending` (`FEATURE_PARITY.md`:378-386, DEVIATION 794) has any backward
   analogue.** No document treats it and I did not derive one.
9. **The `causal_conv1d` backward, still.** `IDENTICAL_BACKWARD_PLAN.md`
   :599-603 records it is not on disk. Mamba-2 inherits the gap over a wider
   channel set. Mamba-3 does not inherit it, because Mamba-3 has no conv
   (`IDENTICAL_MAMBA3_CONTRACT.md`:42-44).

---

## 12. Documentation defects found on the way

Per `[[fix-docs-on-discovery]]`. Recorded here because this document may not
edit the files they live in.

1. **`IDENTICAL_MAMBA3_CONTRACT.md`:150 is STALE.** It says `identical_clamp`
   "DOES NOT EXIST. Already requested by mamba2 DEVIATION 788". It exists, at
   `checks/numerics.mojo`:2273, with a gate section at `:2511`. The
   `check-portable-nn` arm for it is still listed as owed in the Mamba-2
   contract's STILL OWED block (`:491`), so the honest correction is EXISTS,
   ARM OWED, not EXISTS.
2. **`FEATURE_PARITY.md` section 4 has no Mamba-3 backward row.** The table
   at `:169-175` carries "Mamba-1 backward", the microbatch REFUSE row,
   "Mamba-2 backward" and "chunked training scan". Mamba-3's forward has a
   row in section 5 (`:195`) and its backward has none anywhere. The Mamba-3
   backward's disposition is therefore unrecorded rather than deferred.
3. **The Mamba-2 backward row's trigger is stated as met-able and is not
   met** (`FEATURE_PARITY.md`:173). Both halves are open, section 10 step 0.
   The row is not wrong, but a reader scanning it will not learn that.

---

## 13. The four sentences a summary should carry

1. **Mamba-1's hardest seam recurs in both siblings and the contracts already
   say so.** `B` and `C` are shared across heads by COPY in Mamba-2
   (`IDENTICAL_MAMBA2_CONTRACT.md`:90) and in Mamba-3, where the contract
   adds that "the broadcast is a copy, what follows it is arithmetic"
   (`IDENTICAL_MAMBA3_CONTRACT.md`:166). T3 is the answer at three sites in
   Mamba-2 and two in Mamba-3, at two different lengths, and the length
   change is itself a hazard.
2. **The chunked forward makes the backward's launch-invariance argument
   EASIER, not harder.** Every crossing fold in either sibling that is not a
   parameter gradient has a length that is a PROFILE CONSTANT, `Q`, `N`, `P`
   or `H`. The transformer lane's largest finding, that a derivative swaps
   which axis is contracted and can land on a sequence length
   (`transformer/checks/transformer_backward.mojo`:1294-1304), does not bite
   here. Chunking bought that.
3. **Upstream's siblings are further from reproducible than upstream's
   Mamba-1 was.** Their Mamba-2 `dB`, `dC` and `dcb` folds are partitioned by
   `torch.cuda.get_device_properties(...).multi_processor_count`
   (`ssd_chunk_state.py`:937-940 and `ssd_chunk_scan.py`:1485-1488, :1534-1537),
   so their summation order varies by GPU SKU inside one vendor, and their
   Mamba-3 backward AUTOTUNES `CHUNK_SIZE` (`mamba3_siso_bwd.py`:22-26),
   which the forward profile pins as part of the arithmetic. Seventeen
   atomics sit beside those.
4. **The word "reuse" is the trap.** This repository already contains two
   different, self-consistent RMSNorm backwards, T7's collapsed closed form
   (`IDENTICAL_BACKWARD_PLAN.md`:359-372) and the transformer's node-by-node
   spelling with `-0.5` and `2.0` kept explicit
   (`transformer/checks/transformer_backward.mojo`:865-869). Seven sites
   across the four models need one of them. Same function, different number,
   and no existing gate can see the difference.
