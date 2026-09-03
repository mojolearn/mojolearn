# The backward pass under `mojolearn.identical.mamba1.fp32.v1`

Opened 2026-08-25, the mamba BACKWARD lane, DEVIATIONS 1070 through 1089.

## STATUS

**PHASE A IS DONE: THE ROUTING LAYER COMPILES, UNMODIFIED, IN ALL FOUR
CONFIGURATIONS (2026-09-03).** The clean build and each of the three
sabotage arms, through `pixi run build-mamba-backward-probe` plus the
three `-D` arms. Section 10 item 1 predicted the five-element `Tuple`
returns and the `comptime if` early returns would need adjustment;
NEITHER NEEDED ANY EDIT -- every construct already had a compiling twin
in `gemm/checks/gemm_backward.mojo`.

**PHASE C IS WRITTEN AND UNCOMPILED: THE ARITHMETIC EXISTS AS OF
2026-09-03.** The 23 new arithmetic operations of section 3.2 and all eight
topologies of 3.4 now have kernels, in two files that mirror the forward's
own ownership split:

    mamba/impl/mamba_ssm/ops/selective_scan_backward.mojo
        T1 T2 T3 T4 T5 and B13-B21, seams S5'-S11'
    mamba/impl/transformers/models/mamba/modeling_mamba_backward.mojo
        T6 T7 T8 and B4-B11, B22, B26, B30-B34, B37-B42

**NOTHING IN THIS LANE HAS BEEN RUN, AND NO GRADIENT EXISTS.** The probe
gates compilation only: it asserts no value, compares no bits and
launches no kernel. **The two kernel files above have never been through a
compiler either**, the 26-stage host oracle does not exist, and gates
MB1-MB10 are specified and unbuilt. Compiling a file is the first rung of
the ladder and not a result about gradients. No gate in section 7
exists, no backward fixture has been built and no device has executed a
backward call. Every sentence about behavior is a prediction until a gate
prints.

| thing | state |
|---|---|
| forward `mojolearn.identical.mamba1.fp32.v1` | IDENTITY_PATHS row 55, clauses (a) through (f) gated. **THREE COLUMNS**: `mamba.identical.card` md5 `f072dd22`, 18 records, byte-identical on Apple M4, NVIDIA (runpod) and AMD MI325X across the 2026-08-24 through 2026-08-28 legs. The sentence that stood here, that mamba had two columns and AMD had no mamba column at all, is deleted as false |
| `mojolearn.identical.gemm.fp32.v1` | IDENTITY_PATHS row 40, CLOSED on three vendors |
| `gemm/checks/gemm_backward.mojo` | gated on Apple and AMD, no sabotage arm ever fired |
| the derivation of section 1, the classification of section 2 | derived here |
| the eight new topologies of 2.3 | DECLARED here; **KERNELS WRITTEN 2026-09-03, NEVER COMPILED, NEVER RUN**. `mamba_backward_topology_site()` is the index |
| `mamba/checks/mamba_backward.mojo` | the routing layer, COMPILED 2026-09-03 in all four configurations, never run. The row that said **NEVER COMPILED** contradicted this document's own STATUS block and is deleted as false |
| the gates MB1 through MB10 | SPECIFIED, NOT BUILT. `mamba/checks/` holds no backward check file |

The completion claim this lane may make when the gates are green on three
vendors is one sentence, written to be as narrow as the forward lane's.

> **Cross-vendor bit-identical FP32 gradients for one Mamba-1 block, at a
> fixed batch composition and a fixed sequence length, under the declared
> profile.**

Read every qualifier. Not identical training. Not identical models. Not
identical gradients under a different microbatch schedule, which 3.1 proves
impossible. Not a streaming or decode backward, which 4.2 proves impossible.
Not Mamba-2.

**The profile is not amended.** `mamba/IDENTICAL_MAMBA_CONTRACT.md` is
consumed here and never edited. What this document proposes is a SEPARATE
profile name, `mojolearn.identical.mamba1.bwd.fp32.v1`, which CONSUMES the
forward profile and adds clauses of its own.

**Four findings, before the detail.**

- **No new transcendental is needed.** Every derivative is a rational function
  of quantities the forward already computes. `identical_exp`,
  `identical_sigmoid`, `identical_silu`, `identical_div`, `identical_rsqrt`,
  `identical_mul_add` and `ftz` are the whole toolbox and all seven ship.
- **A great deal of new ORDER is needed.** Of 42 backward operations, four are
  copies, two are pure reuse, thirteen route to gemm v1, and **twenty three
  are genuinely new arithmetic**, of which eleven are elementwise spellings
  whose association must be declared, four are folds inherited from an
  existing forward clause, and **eight are fold or recurrence topologies
  nothing in this repository declares today**. Section 3.4 is the real output
  of this lane.
- **The hardest seam is not the one that looks hardest.** The reverse
  recurrence stays inside one thread and inherits the forward's structural
  launch invariance for free. The hard seam is that **`B` and `C` are shared
  across channels**, so `dB` and `dC` contract over `d_inner`, an axis no
  float crosses in the forward. Section 5.
- **Upstream's own backward is not reproducible even run to run on one
  device.** `selective_scan_bwd_kernel.cuh` accumulates `dA`, `dB`, `dC`, `dD`
  and `ddelta_bias` with five `gpuAtomicAdd` calls, and
  `selective_scan_bwd_cuda` picks its block shape from `params.seqlen` through
  a table that is DIFFERENT under `USE_ROCM`. Its summation order is a
  function of the sequence length, the arrival order of blocks and the vendor.

---

## 1. Status, and what is being claimed

See the STATUS block above; it is this section.

---

## 2. The backward, derived

Token-major throughout, `M = B * L` rows, the oracle's own layouts. `dm` is
`d_model`, `di` is `d_inner = 2 * dm`, `r` is `dt_rank`, `N` is `D_STATE = 16`,
`K` is `D_CONV = 4`.

The forward's seams and intermediate names are `mamba/IDENTICAL_MAMBA_CONTRACT.md` section 2, S1 through S17, and are not
restated here. The backward reuses the forward's own ROUNDED intermediates,
so a name that does not exist in the forward cannot be reused. `h[-1,d,n]` is
the carried state, zeros on prefill.

The backward, given `dres[M, dm]`, with every association written out.

    S16'   dO[t,c]     = dres[t,c]                          a copy
           dx_res      = dres                               a copy
    S17d'  dg          = dO . W_out                         gemm dA, k' = dm
           dW_out      = dO^T . g                           gemm dB, k' = M
    S12'   dsk[t,d]    = dg[t,d] * silu(z[t,d])
           dz[t,d]     = dg[t,d] * sk[t,d] * silu'(z[t,d])
           silu'(z)    = sig(z) * (1 + z*(1 - sig(z)))
    S11'   dy[t,d]     = dsk[t,d]                           a copy
           du_D[t,d]   = dsk[t,d] * D[d]
           dD[d]       = sum over (b,l) of dsk[t,d]*u[t,d]  REDUCES OVER TOKENS
    S9'    dh[t,d,n]   = dy[t,d]*Cm[t,n] + da[t+1,d,n]*dh[t+1,d,n]
                                                            REVERSE RECURRENCE
           dh[L]       = 0                                  see below
    S10'   dCm[t,n]    = sum over d of dy[t,d]*h[t,d,n]     REDUCES OVER d_inner
    S8'    d_dbu       = dh
           du_s[t,d]   = sum over n of dh[t,d,n]*dbb[t,d,n] fold over N
           d_dbb[t,d,n]= dh[t,d,n] * u[t,d]
    S7'    dBm[t,n]    = sum over d of d_dbb[t,d,n]*delta[t,d]
                                                            REDUCES OVER d_inner
           ddelta_B    = sum over n of d_dbb[t,d,n]*Bm[t,n] fold over N
    S6'    d_da[t,d,n] = dh[t,d,n] * h[t-1,d,n]
           d_arg       = d_da * da[t,d,n]                   exp' is exp
    S5'    ddelta_A    = sum over n of d_arg*A[d,n]         fold over N
           dA[d,n]     = sum over (b,l) of d_arg*delta[t,d] REDUCES OVER TOKENS
    S15'   dA_log[d,n] = dA[d,n] * A[d,n]                   because A = -exp(A_log)
    S14'   ddelta      = ONE interleaved fold over n, B term then A term
                        (DEVIATION 1083; this line used to read
                        `ddelta_B + ddelta_A`, which is a DIFFERENT number
                        and contradicted 3.2's row B18. See below)
           ddtp[t,d]   = ddelta[t,d] * softplus'(biased[t,d])
           softplus'   = biased <= 20 ? sig(biased) : 1
           db_dt[d]    = sum over (b,l) of ddtp[t,d]        REDUCES OVER TOKENS
    S17c'  ddtl        = ddtp . W_dt                        gemm dA, k' = r
           dW_dt       = ddtp^T . dtl                       gemm dB, k' = M
           dXP         = concat(ddtl | dBm | dCm)           a copy
    S17b'  du_x        = dXP . W_x                          gemm dA, k' = r+2N
           dW_x        = dXP^T . u                          gemm dB, k' = M
           du          = du_D + du_s + du_x                 A THREE WAY JOIN
    silu'  dconv[t,d]  = du[t,d] * silu'(conv[t,d])
    S13'   dhin[b,p,d] = sum over k of cw[d,k]*dconv[b,p+3-k,d]
                                       fold over K, dropped where p+3-k >= L
           dcw[d,k]    = sum over (b,l) of dconv*hin[b,l-3+k,d]  OVER TOKENS
           dcb[d]      = sum over (b,l) of dconv[t,d]            OVER TOKENS
           dP          = concat(dhin | dz)                  a copy
    S17a'  dnrm        = dP . W_in                          gemm dA, k' = 2di
           dW_in       = dP^T . nrm                         gemm dB, k' = M
    S4'    dinner[t,j] = dnrm[t,j] * w_norm[j]
           dw_norm[j]  = sum over (b,l) of dnrm*inner[t,j]  REDUCES OVER TOKENS
    S3'    drstd[t]    = sum over j of dinner[t,j]*x[t,j]   fold over dm
           dx_a[t,j]   = dinner[t,j] * rstd[t]
    S2'    dmean[t]    = drstd[t] * (-0.5 * rstd[t]^3)
           dss[t]      = dmean[t] / dm
    S1'    dx_b[t,j]   = dss[t] * 2 * x[t,j]
    S16''  dx[t,j]     = dres[t,j] + dx_a[t,j] + dx_b[t,j]  THE ABSORPTION SITE

`S1'` through `S3'` collapse to the closed form and are spelled that way in
topology T7, because the intermediate `dss` has no use of its own.

### 2.3 The three checks that were done rather than assumed

**The RMSNorm backward, derived rather than recalled.** With
`nrm = w * x * rstd`, `rstd = (ss/dm + eps)^{-1/2}` and `ss = sum_j x^2`, we
have `d rstd / d ss = -0.5 * rstd^3 / dm` and `d ss / d x_j = 2 x_j`, so
`d rstd / d x_j = -rstd^3 * x_j / dm`. With `dinner_j = dnrm_j * w_j` and
`R = sum_j dinner_j * x_j`,

    dx_norm[t,j] = rstd[t]*dinner[t,j] - (rstd[t]^3 / dm) * x[t,j] * R[t]

**The only reduction is `R`, over the FEATURE axis, per row**, the same axis
and the same length as S1's own fold. Section 3 uses that.

**The conv backward's index arithmetic.** Forward, `conv[b,l,d]` reads
`hin[b, l-3+k, d]` for `k` in `[0, 4)`, so `hin[b,p,d]` is read by output
position `l = p + 3 - k`, in range only for `k` with `0 <= p + 3 - k < L`. The
backward is a CORRELATION in the opposite direction and the tap dropped at the
START of the forward sequence is dropped at the END of the backward one. **A
reversed-tap implementation is bit identical on a symmetric fixture and wrong
on every other**, which is why MB2's sabotage exists.

**The off-by-one on `da`.** `h[t]` influences `y[t]` through `Cm[t]` and
`h[t+1]` through `da[t+1]`, NOT through `da[t]`. Upstream spells the shift
explicitly as `thread_reverse_data[i-1].x = delta_a_exp` at index `i`
(`selective_scan_bwd_kernel.cuh:255`), with the last element of the block taken
from the neighboring thread's `smem_delta_a`. **Getting this wrong produces a
plausible gradient that is bit identical on any fixture where all `da` are
equal**, so MB7's fixture must plant unequal `delta` across tokens and the
gate must REFUSE to pass if it does not.

### 2.4 The gradient of the carried state is DROPPED

`selective_scan_fn`'s docstring says "the gradient of the last state is not
considered in the backward pass" and the kernel seeds its reverse scan with
`make_float2(1.f, 0.f)` at the final chunk. This lane adopts that, so
`dh[L, d, n] = 0` and no gradient flows out of a block call into the state a
previous call produced. It is the right scope decision for one block and it is
the reason 3.2's answer is what it is. **Nobody should read "the backward is
correct" as "the backward does truncated BPTT correctly"**, because it does no
BPTT at all. The same is true of the conv window; `dhin` at a pre-sequence
position would flow into the previous call's output and is dropped.

---

## 3. The classification, and the count that is the point of this lane

Each operation is (a) an already-pinned operation this repository owns and
calls unchanged, (b) routable to gemm v1, or (c) genuinely new arithmetic
needing a pinned order. There is no fourth bucket and no "basically the same
as". A pure data movement is a copy and is counted separately. Category (c) is
subdivided: **c-elementwise**, no fold but the association or spelling is a
free choice that moves the last bit; **c-inherited**, a fold whose order is
already declared by a forward seam and reused verbatim; **c-topology**, a fold
or recurrence nothing in this repository declares.

    copies                                    4
    (a)  existing primitive, no new order     2
    (b)  routable to gemm v1                 13
    (c)  genuinely new arithmetic            23
         of which  c-elementwise             11
                   c-inherited order          4
                   c-TOPOLOGY                 8
                                       -------
                                             42

Counted by contract seam instead, so it can be read beside contract section 7,
**not one of S1 through S17 is answered entirely by (a) or (b)**. S17 comes
closest, since all eight of its gemm calls route, but even S17 leaves the
concatenations that build `dXP` and `dP` and the join at `du`.

**The mamba backward is not a routing problem the way the gemm backward was.**
That lane could write a 595 line file with no multiply in it. This one cannot.

### 3.2 The operation table

`k'` is the contracted length where there is one.

| # | operation | reduces over | cat | note |
|---|---|---|---|---|
| B1 | `dO = dres` | none | copy | |
| B2 | `dg = dO . W_out` | `dm` | b | gemm `dA`, forward `OP_NT` |
| B3 | `dW_out = dO^T . g` | **tokens** | b | gemm `dB`, `k' = M` |
| B4 | recompute `silu(z)` | none | a | `identical_silu`, MUST match S12 |
| B5 | `dsk = pinned_mul(dg, silu_z)` | none | c-elem | |
| B6 | `sig_z = identical_sigmoid(z)` | none | a | new SITE, existing primitive |
| B7 | `silu'(z)` composite | none | c-elem | FOUR roundings after the sigmoid -- this row said three and undercounted the `1 - sig` subtraction -- association pinned; the spelling is `transformer/checks/transformer_backward.mojo::bwd_silu_backward_kernel`'s, reused rather than re-derived |
| B8 | `dz = dg * sk * silu'(z)` | none | c-elem | 4 factors, order pinned |
| B9 | `dy = dsk` | none | copy | |
| B10 | `du_D = pinned_mul(dsk, D[d])` | none | c-elem | |
| B11 | `pD = pinned_mul(dsk, u)` | none | c-elem | pre-product for B12 |
| B12 | `dD[d] = sum_t pD` | **tokens** | b | ones-vector `OP_NN` at `(1, di, M)` |
| B13 | the reverse recurrence `dh` | **sequence** | **c-top T1** | |
| B14 | `h[t-1]` availability | none | **c-top T2** | a storage decision, 4.1 |
| B15 | `dCm[t,n] = sum_d dy*h` | **d_inner** | **c-top T3** | |
| B16 | `dBm[t,n] = sum_d w*dh` | **d_inner** | **c-top T3** | same topology |
| B17 | `du_s = sum_n dh*dbb` | `N` | c-inh | S10's ascending fused chain |
| B18 | `ddelta = sum_n (B path + A path)` | `N` | c-inh | interleaving PINNED, DEVIATION 1083. ONE accumulator, `n` ascending, per step the B term then the A term, two `fma`s into one register -- upstream's own shape (`ddelta_vals[i] += ddelta_u * u + dx * A * a`, one statement per state index). `SAB_BWD_DDELTA_TWO_FOLDS` is the two-fold reading section 2.2 used to spell |
| B19 | `dA` fold over `t` | **sequence** | **c-top T4** | direction is the choice |
| B20 | parameter fold over `b` | **batch** | **c-top T5** | **`dA` ONLY.** The row said it was shared by `dD`, `db_dt`, `dcw` and `dcb` too, and that does not survive this table's own rows B12, B23, B32 and B33: those four are category (b), routed to ones-vector v1 GEMMs at `k' = M`, and `M = B * L` already folds the batch. Only `dA` declines the routing (T4) |
| B21 | `dA_log = pinned_mul(dA, A)` | none | c-elem | |
| B22 | `ddtp` = softplus' applied | none | c-elem | DIV or MUL, 2.4 |
| B23 | `db_dt[d] = sum_t ddtp` | **tokens** | b | ones-vector `OP_NN` |
| B24 | `ddtl = ddtp . W_dt` | `di` | b | gemm `dA` |
| B25 | `dW_dt = ddtp^T . dtl` | **tokens** | b | gemm `dB`, `k' = M` |
| B26 | `dXP = concat(ddtl, dBm, dCm)` | none | copy | |
| B27 | `du_x = dXP . W_x` | `r + 2N` | b | gemm `dA` |
| B28 | `dW_x = dXP^T . u` | **tokens** | b | gemm `dB`, `k' = M` |
| B29 | `du = du_D + du_s + du_x` | none | **c-top T6** | a three way join, ORDER |
| B30 | `dconv = du * silu'(conv)` | none | c-elem | |
| B31 | `dhin` four tap correlation | `K` | c-inh | S13's tap order, reversed index |
| B32 | `dcw[d,k] = sum_t dconv*hin_k` | **tokens** | b | four ones-vector calls |
| B33 | `dcb[d] = sum_t dconv` | **tokens** | b | ones-vector `OP_NN` |
| B34 | `dP = concat(dhin, dz)` | none | copy | |
| B35 | `dnrm = dP . W_in` | `2di` | b | gemm `dA` |
| B36 | `dW_in = dP^T . nrm` | **tokens** | b | gemm `dB`, `k' = M` |
| B37 | `dinner = pinned_mul(dnrm, w_norm)` | none | c-elem | |
| B38 | `pW = pinned_mul(dnrm, inner)` | none | c-elem | pre-product for B39 |
| B39 | `dw_norm[j] = sum_t pW` | **tokens** | b | ones-vector `OP_NN` |
| B40 | `drstd[t] = sum_j dinner*x` | `dm` | c-inh | S1's ascending fused chain |
| B41 | `dx_norm` closed form | none | **c-top T7** | `rstd^3` association |
| B42 | `dx = dres + dx_norm` | none | **c-top T8** | THE ABSORPTION SITE |

### 3.4 The eight new topologies, declared

Declarations in the contract's own vocabulary. **No kernel exists for any of
them.**

**T1, the reverse recurrence, DEVIATION 1070.** REVERSE over `t` from `L - 1`
to `0`, seed `dh[L,d,n] = +0.0`, one rounding per step, FUSED.

    contrib   = ftz(pinned_mul(dy[t,d], Cm[t,n]))
    dh[t,d,n] = ftz(identical_mul_add(da[t+1,d,n], dh[t+1,d,n], contrib))

FUSED mirrors S9, whose clause says fusion is pinned because only fusion has a
portable spelling. If wrong, every `dh` from `t = L-2` downward moves and nine
card stages move with it. **The cost of being wrong is the whole lane.**

**T1's SEED IS AN OMITTED OPERATION, DEVIATION 1082, AND THIS PARAGRAPH IS
NEW.** The line above reads `fma(da[t+1], dh[t+1], contrib)` with
`dh[L] = +0.0`, and at `t = L - 1` there is no `da[L]`: the forward computed
`L` values of `da` and this recurrence asks for an `L + 1`-th. The omission is
not cosmetic, because the two closures differ on a real input class --
`(+0.0) + (-0.0)` is `+0.0`, so folding a stored `+0.0` in would LAUNDER a
negative zero wherever `dy[L-1,d] * Cm[L-1,n]` is one. PINNED as the omitted
form, `dh[L-1,d,n] = ftz(contrib)`, which is the strongest statement of a
`+0.0` seed and is the same argument `gemm_identical.mojo::_fold_drain`
already makes for `P == 1` ("performs NO addition"). `SAB_BWD_T1_SEED_ADD` is
the other closure and is **PREDICTED BITWISE INERT on any fixture without a
planted negative-zero `dy*Cm` at the last token**, which is every hashed
fixture.

**T2, `h[t-1]` availability, DEVIATION 1071.** PINNED as an explicit
checkpoint. The forward stores `h[t,d,n]` for every `t` in `[-1, L)` into a
caller-owned buffer of `B * di * (L+1) * N` floats and the backward READS
`h[t-1]` from it.

    REFUSED alternative, upstream's:  a = h[t] - dbu[t]
                                      (selective_scan_bwd_kernel.cuh:290)

Since `h[t] = round(da*h[t-1] + dbu[t])`, that subtraction is exact only when
no rounding occurred, and it suffers unbounded relative cancellation whenever
`|da * h[t-1]| << |dbu[t]|`, which is the normal case early in a sequence
where `h` is near zero and `dbu` is not. It is a memory optimization with an
accuracy cost upstream does not state. **Do not port their bugs.** The price of
refusing it is `N = 16` times the activation footprint, priced in 5.3.
`SAB_BWD_H_SUBTRACT` keeps upstream's spelling so MB9 can MEASURE the ulp
distance rather than assert that it matters.

**T3, the `d_inner` contraction for `dBm` and `dCm`, DEVIATION 1072.** Serial
ASCENDING `d` from `+0.0`, one FUSED multiply-add per term, which is bit for
bit gemm v1 at `(m', n', k') = (1, N, di)` per token.

    dCm[t,n] = fold over d of fma(dy[t,d],  h[t,d,n],  acc)
    dBm[t,n] = fold over d of fma(w[t,d],  dh[t,d,n],  acc)
    with      w[t,d] = ftz(pinned_mul(delta[t,d], u[t,d]))

`w` is pre-formed so the leaf is one `fma`; the alternative pre-forms
`R[t,d,n] = pinned_mul(dh, u)`, which costs `N` times the memory and one extra
rounding per `(t,d,n)` rather than per `(t,d)`. **If `di > 128` the fold is
NOT a single chain**, it is `contract_partition(di)` leaves combined by v1's
balanced tree, and an implementation that ignores that is a different answer at
every real model width. **`di` is the reduced length, never a launch quantity.**

**T4, the `dA` fold over `t`, DEVIATION 1073.** PINNED DESCENDING in `t`, the
direction the reverse pass already walks, from `+0.0`, FUSED.

    for t from L-1 down to 0:
        acc = ftz(identical_mul_add(d_arg[t,d,n], delta[t,d], acc))

**This is the one place this lane pins a DESCENDING fold** and every other
fold in both profiles is ascending. Ascending would need either a second pass
over `M * di * N` stored per-token products or a second traversal.
`SAB_BWD_DA_ASCENDING` is the other direction, and MB4 must show it bites,
otherwise this deviation is unmeasured and should be re-decided in favor of
consistency.

**`dA` IS ROUTABLE TO GEMM v1 AND IS DELIBERATELY NOT ROUTED**, the one place
this plan declines a category (b) answer. With
`q[t, d*N + n] = pinned_mul(d_arg[t,d,n], delta[t,d])` materialized at
`[M, di*N]`, `dA` is a ones-vector `OP_NN` at `(1, di*N, M)` and its fold order
would be v1's certified one rather than a hand-declared direction. Declined
because `q` is a THIRD buffer of `M * di * N` floats on top of T2's `h` and
T3's `dh`. **If a blocked variant ever makes the memory affordable this
decision should be revisited and the routed form is strictly preferable**,
because it deletes a declared order in favor of a certified one.

**T5, the parameter fold over `b`, DEVIATION 1074.** Every parameter gradient
a thread can only compute per `(b, d)` is written to a private slot
`partial[b, d, ...]` with NO atomic anywhere, and a SECOND kernel folds over
`b` ASCENDING from `+0.0` with a plain flushed add. **ITS ONLY SITE IS
`dA`.** This paragraph said it was shared by `dD`, `db_dt`, `dcw` and `dcb`
as well; that is deleted as false, because section 3.2 routes all four to
ones-vector v1 GEMMs at `k' = M` and `M = B * L` already folds the batch.
Only `dA` declines the routing, which is T4's own decision. The kernel is
written width-generic anyway so a future unrouted parameter has a home rather
than a second declaration. The alternative is a float `atomicAdd`, which is
what upstream does five times, REFUSED for IDENTITY_PATHS rows 1 and 2's
reason. The cost is `B` times the parameter footprint and one extra launch.

**Metal check, done rather than assumed.** This fold never enters threadgroup
memory. Metal has float `atomicAdd` and warp shuffles but NO threadgroup float
atomics, and 32 KB of threadgroup memory, so a fold needing scratch
proportional to `d_inner` inside a threadgroup would not fit at any real
width. Routing through GLOBAL private slots and a second kernel is the
spelling with no ceiling.

**T6, the three way join at `du`, DEVIATION 1075.** PINNED ORDER, stated once
and never a scheduler's choice.

    du[t,d] = ftz(ftz(ftz(du_D[t,d]) + du_s[t,d]) + du_x[t,d])

D-skip first, then the scan path, then the x_proj path, following the
forward's own data flow since `u` is consumed by S11, then the scan, then
S17b. Any permutation is a different answer. `SAB_BWD_JOIN_ORDER` permutes it,
**predicted INERT on any fixture where one contribution dominates the other
two**, so MB4's fixture must plant the three at comparable magnitude and the
gate must raise VACUOUS rather than pass if it cannot.

**T7, the RMSNorm backward closed form, DEVIATION 1076.**

    c3   = ftz(pinned_mul(ftz(pinned_mul(rstd, rstd)), rstd))
    s    = ftz(identical_div(c3, Float32(dm)))
    t2   = ftz(pinned_mul(ftz(pinned_mul(s, x[t,j])), drstd[t]))
    t1   = ftz(pinned_mul(rstd, dinner[t,j]))
    dx_norm[t,j] = ftz(t1 - t2)

`rstd^3` is `(rstd*rstd)*rstd`, left associated, never `portable_powf`.
`c3 / dm` is hoisted per row because it is loop invariant in `j` and a hoisted
flushed division moves no bits. The subtraction is UNFUSED because the two
terms are separately rounded products in every reference spelling. The
alternative folds the `-1/dm` into the fold as a negative scale, which is one
fewer rounding and a different answer.

**T8, the residual join at `dx`, DEVIATION 1077. THE ABSORPTION SITE.**

    dx[t,j] = ftz(ftz(dres[t,j]) + dx_norm[t,j])

One add, `dres` on the left. This is the exact twin of S16, and IDENTITY_PATHS
row 55 records what happened there: at the shape where a sabotage arm was
STRONGEST, thirteen of sixteen intermediate stages moved, `out_proj.out`
differed on 23 of 64 cells, and `residual.out` was STILL BIT IDENTICAL,
because the residual add put a value of order 1e-3 beside one of order 1 and
rounded the difference away. **An output-only gate called that arm inert.**
`dx` is the backward's `residual.out`, so any gate that compares only `dx`
will report a broken norm backward as green. That is why section 5 is per
stage and for no other reason.

### 3.5 The eleven elementwise associations, in one place

Each is one line and each moves a bit if spelled the other way. Listed
together because a reader skimming for "the hard part" will skip them, and
eleven quiet wrong answers are worse than one loud one.

| op | pinned | the alternative that is wrong |
|---|---|---|
| B5 | `pinned_mul(dg, identical_silu(z))` | `pinned_mul(dg, pinned_mul(z, sig))`, upstream's |
| B7 | `sig * (1 + z*(1 - sig))` left to right | `sig + z*sig*(1-sig)` |
| B8 | `((dg * sk) * silu')` | `dg * (sk * silu')`, and upstream's four factor chain |
| B10 | `pinned_mul(dsk, D[d])` | fusing it into B29's join |
| B11 | `pinned_mul(dsk, u)` | fusing into the ones-vector leaf |
| B21 | `pinned_mul(dA, A)` | recomputing `-exp(A_log)`, bit equal but a second call |
| B22 | below | below |
| B30 | `pinned_mul(du, silu'(conv))` | `identical_div` of the silu quotient form |
| B37 | `pinned_mul(dnrm, w_norm[j])` | folding `w_norm` into B35's gemm |
| B38 | `pinned_mul(dnrm, inner)` | using `nrm/w` to recover `inner`, a division |
| B41 | T7 | T7 |

**B22 has an upstream disagreement, DEVIATION 1078.** `softplus'(x) =
sigmoid(x)` and there are two spellings.

    PINNED    ddtp = ftz(pinned_mul(ddelta, identical_sigmoid(biased)))
    upstream  ddtp = ddelta / (1 + expf(-biased))    bwd_kernel.cuh:454-457

Upstream spells ONE DIVISION where `identical_sigmoid` is
`portable_divf(1, 1+e)` then a product, two roundings against one. The guard is
the same in both, `biased <= 20`, and the operand is the PRE-softplus biased
value in both. The pinned choice is the multiply, because the forward already
owns `identical_sigmoid` as row 52's seam. `SAB_BWD_S14_DIVISION` is
upstream's, and **the distinguishing band is `biased` in roughly `[8, 14]`; a
fixture that only straddles 20 passes this vacuously. That has already bitten
this lane once**, on `adv_softplus_guard`, where 256 inputs spanned
`[19.87, 20.10]` with ZERO cells in the distinguishing band and a rebuild
under `S14_THRESHOLD_10` produced 23 of 23 byte-identical dumps. **BITWISE
INERT.** Do not repeat it.

---

## 4. The three invariance questions, answered separately

The argument in each case is the reduction axis and nothing else.

### 4.1 Batch invariance. HALF YES, HALF STRUCTURALLY IMPOSSIBLE.

**Activation gradients. YES**, and it is the backward twin of contract clause
(c). `du`, `ddelta`, `dz`, `dconv`, `dhin`, `dnrm`, `dinner`, `dx`, `dBm`,
`dCm` and `dh` reduce over the feature axis, the state axis `N`, the tap axis
`K`, the `d_inner` axis and the sequence axis. **Not one reduces over the
batch.** This is the clause worth gating and worth claiming.

**Parameter gradients. NO, and the question is malformed.** `dW_in`, `dW_x`,
`dW_dt`, `dW_out`, `dA_log`, `dD`, `db_dt`, `dcw`, `dcb` and `dw_norm` are
each DEFINED as a sum over the batch, and asking a sum to be invariant to its
own terms is not a coherent request. The coherent, weaker clause to gate is
that at a fixed batch CONTENT and token order the bits do not depend on launch
geometry, which 3.3 says is achievable.

**Microbatch equivalence. STRUCTURALLY IMPOSSIBLE, in TEN places.** Four are
literal gemm `dB` calls with `k' = M` and inherit the gemm plan's finding
verbatim, B3, B25, B28 and B36. **The other six are NOT matmuls**, so a reader
who has absorbed the gemm finding will assume they are exempt and they are
not: B12 `dD`, B23 `db_dt`, B32 `dcw`, B33 `dcb`, B39 `dw_norm`, and B19 plus
B20 for `dA_log`. Each is a token-axis fold whose partition is a function of
the token count, either because it routes to a v1 ones-vector GEMM at `k' = M`
or because T4's sequence fold and T5's batch fold are sums whose term count is
the batch and sequence size. **The microbatch schedule is part of a training
run's numerical specification**, and a reproducibility claim must name the
factor.

### 4.2 Sequence-length invariance. STRUCTURALLY IMPOSSIBLE, AND NOT A ROUNDING PROBLEM.

This finding has no counterpart in the gemm lane and it is worse than the
microbatch one.

The forward's clause (d) is prefill equals decode bitwise per token, and it
holds because the recurrence is CAUSAL; IDENTITY_PATHS row 55 records that
gate D earned it at the composition point, 2,152 cells at `d_model` 8 and
4,168 at 16. **The backward's state recurrence runs the other way**, since
`dh[t] = dy[t]*Cm[t] + da[t+1]*dh[t+1]` depends on every token from `t` to
`L - 1`. Truncate the sequence and `dh[t]` is a DIFFERENT NUMBER, not a
differently rounded one, and everything downstream inherits that. There is no
prefix property to preserve and therefore nothing to pin.

1. **There is no decode backward.** A per-token streaming backward is not a
   different implementation of the same function; it does not exist. Training
   a Mamba block requires the whole sequence in hand.
2. **Every parameter gradient's bits depend on `L`** as well as on `B`.
3. **The only sequence-shaped invariance available is CHUNK INVARIANCE**, and
   it is worth claiming precisely because upstream does not have it. If the
   reverse pass is ever segmented, the segmented answer must equal the
   unsegmented one bitwise, which requires the boundary to carry the exact
   `dh` and the chunk count never to reach the arithmetic. Upstream fails this
   by construction, since `selective_scan_bwd_cuda` derives
   `kNThreads * kNItems` from `params.seqlen` through one table on CUDA and a
   DIFFERENT table under `USE_ROCM`, and the `cub::BlockScan` and
   `cub::BlockReduce` fold shapes follow `kNThreads`. **A block count is a
   summation order**, and theirs is a function of the sequence length and the
   vendor. This lane's declared shape is one thread per `(b, d)` walking the
   whole sequence serially, which has ONE chunk by construction, so chunk
   invariance is trivially true today and becomes a real clause only if a
   chunked variant is ever built.

**Lead with this in any summary.** A limitation found on paper is worth more
than one found on a rented GPU next week, and this one would have been found
on the GPU as a gate that could not be written.

### 4.3 Launch invariance. YES, but the forward's structural argument does NOT survive.

The forward's argument is the strongest sentence in that lane.

> A thread owns a whole `(b, d)` recurrence ... no float crosses a thread
> boundary anywhere in this file. There is no shared memory, no reduction, no
> atomic. So `block_size` and the grid decide only WHICH thread holds a pair,
> never the sequence of values accumulated into it.

**That sentence cannot be written about the backward.** T3 contracts over
`d_inner` and the `(b, d)` grid gives each `d` to a different thread; T5
contracts over `b`; every category (b) token reduction contracts over `M`,
which is spread across the whole grid. So launch invariance stops being a
property of the kernel's SHAPE and becomes a property of each crossing fold's
PARTITION, under one rule stated once.

> **Every crossing fold's partition is a pure function of the length of the
> axis it reduces, and of nothing else.** `contract_partition(di)` for T3, the
> batch index for T5, `contract_partition(M)` for the ones-vector reductions.
> No `block_size`, no grid dimension, no `n_chunks`, no occupancy heuristic
> appears in any of them.

That is gemm contract 6.1's rule wearing a different hat, and the reason it is
affordable here is Metal's forcing function: with no threadgroup float atomics
and 32 KB of threadgroup memory the cheap wrong answer was never available on
one of the three columns, so the fold had to go through global memory and a
second kernel anyway. **That is the mechanism, and it is why the identity is
cheap here rather than free.**

---

## 5. The hardest seam, and why it is not the one that looks hardest

### 5.1 The reverse recurrence is easy, which is a surprise

T1 walks `t` from `L - 1` to `0` inside a single thread owning one `(b, d)`
pair, holding `N = 16` values of `dh` in registers exactly as the forward holds
`N` values of `h`. No cross-thread communication, no shared memory, no atomic.
**It inherits the forward's structural launch invariance and batch-composition
invariance unchanged**, and its only new decisions are a direction, a seed and
a fusion, one line each. The brief predicted this would be the crux. It is
not.

### 5.2 The hard seam is that `B` and `C` are shared across channels

In Mamba-1, `A` and `D` are per channel but `B` and `C` come out of `x_proj`
per TOKEN at `[M, N]` and are shared by every one of the `d_inner` channels.
The forward reads them, which costs nothing. The backward SUMS over the
channels that read them, and every term of that sum lives in a different
thread under the forward's grid. Four structures were considered.

1. **A float `atomicAdd` per term**, upstream's answer. Arrival order.
   **REFUSED**, rows 1 and 2.
2. **A threadgroup tree over `d`.** Needs all `d_inner` contributions for one
   `(b, t)` resident in one threadgroup. **REFUSED on Metal**, which has no
   threadgroup float atomics and 32 KB of memory, so at `d_inner = 2048` the
   scratch alone is 8 KB per `n`, and it is a block-shaped fold besides.
3. **A fixed-point Int32 accumulator**, rows 1 and 2's own replacement.
   **REJECTED, not refused**, because gradients have a far wider dynamic range
   than a histogram bin and row 8 warns the scale must not itself be derived
   from a float reduction. It stays on the table as the fallback if structure
   4's memory is unaffordable.
4. **Materialize `h` and `dh` and contract per token. PINNED, T3.** `dCm[t,:]`
   is `dy[t,:]` at `1 x di` times `h[t]` at `di x N`, which is gemm v1 at
   `(1, N, di)` exactly, per token. The contracted length is `d_inner`, a
   LAYER WIDTH, bounded and independent of the batch. No atomic, no threadgroup
   memory, no launch quantity anywhere.

**The price of structure 4, stated in the same sentence as the claim.** `h`
and `dh` are each `M * d_inner * N * 4` bytes with `N = 16`.

| shape | `M` | `d_inner` | `h` + `dh` |
|---|---|---|---|
| the corpus's largest, `comp_b2_l257_d8` | 514 | 32 | 2.1 MB |
| `d_model` 16, `L` 257, `B` 3 | 771 | 64 | 6.3 MB |
| a 130M parameter Mamba block, `L` 2048, `B` 8 | 16,384 | 1,536 | 3.2 GB |
| the same at `L` 8192 | 65,536 | 1,536 | 12.9 GB |

**So the declared T3 spelling is a GATE-SCALE construction.** At model scale it
needs a blocked variant that streams `d` in tiles, which is not built and is
not free to design: **the tile size has to be `contract_partition(d_inner)`'s
leaf boundary and not a memory budget.** If a future lane picks the tile from
available VRAM, the answer becomes a function of the machine and every claim
here is void. **This is the largest single open item of the lane, ahead of
everything in section 7.** `h`'s buffer is needed by T2 anyway, so T3 pays for
`dh` and not for both.

### 5.4 Why S9's fusion does not create a second hazard

S9 pins `h = fma(da, h, dbu)`, one rounding. A reader might expect the
backward to need `h` and `dbu` separately in order to differentiate a fused
operation, and it does not: `d/d(da) = h[t-1]` and `d/d(dbu) = 1` regardless of
whether the forward fused them, because fusion changes the ROUNDING of the
result and not the derivative of the exact function. The pinned backward
differentiates the exact bilinear form and rounds its own products through
`pinned_mul`.

---

## 6. Upstream, symbol by symbol

Read from `/Users/andrewhendel/CascadeProjects/upstream/mamba` at `e9594ce`.

**`causal_conv1d` IS NOT ON DISK.** It is a separate repository
(Dao-AILab/causal-conv1d) and is not checked out. Every statement here about
the conv backward is derived from forward seam S13 and from 2.3's index
arithmetic and is marked INFERRED. Nothing about `causal_conv1d_bwd_function`'s
rounding order, bias-gradient fold or tap order has been read.

**What we must COPY.** The reverse recurrence with the same combine operator
as the forward (`SSMScanOp` is `(a0,b0) . (a1,b1) -> (a1*a0, a1*b0 + b1)`,
`selective_scan_common.h:141-145`, run through `BlockReverseScan`); the
backward of a first-order linear recurrence is a first-order linear recurrence
run backwards. The off-by-one on `da`, 2.3. `du` seeded with the D-skip path
then accumulated over the state index (`du_vals[i] = D_val * dout_vals[i]` at
:217, `+= ddelta_u * delta_vals[i]` inside the state loop); the join is theirs
and T6 pins its ORDER. The two-term `ddelta`,
`ddelta_vals[i] += ddelta_u * u + dx * A * a`; a backward with only the `B`
path is a plausible wrong answer. The softplus derivative's guard and its
operand, `delta_val <= 20` on the PRE-softplus biased value, reloaded and
re-biased rather than inverted. Recomputation instead of storage where it is
bit identical (`checkpoint_lvl >= 1`). And dropping the last state's gradient.

**What we must NOT copy.**

1. **Five float `atomicAdd` calls.** `dA` at :483, `dB` and `dC` at :486, :491
   and at :306, :307 for the variable case, `dD` at :475, `ddelta_bias` at
   :480. **The single largest reason upstream's backward cannot have the
   property this lane is trying to build.**
2. **A launch shape chosen from the sequence length, and a different one per
   vendor.** `selective_scan_bwd_cuda` at :537-559 selects
   `(kNThreads, kNItems)` from `params.seqlen` through five branches, with a
   DIFFERENT table under `USE_ROCM`. Their backward is not sequence invariant,
   not launch invariant and not cross-vendor invariant, by construction and on
   purpose, for occupancy.
3. **`cub::BlockScan<..., BLOCK_SCAN_RAKING>` and `cub::BlockReduce` as the
   fold.** CUB is OPEN and therefore a port candidate, so the objection is not
   that we may not read it; a raking scan's fold topology is a function of the
   block width, and a block count is a summation order.
4. **`a = thread_data[i].y - delta*u*B`** to recover `da * h[t-1]`. T2.
   Unbounded cancellation, undocumented by them, MEASURED by MB9 rather than
   asserted.
5. **The `silu(z)` recomputation inconsistency, the sharpest one.** The
   forward kernel computes the gate as `out_vals[r][i] *= z_val / (1 +
   expf(-z_val))` (`selective_scan_fwd_kernel.cuh:298`); the backward
   recomputes the SAME quantity as
   `z_sigmoid_val = 1.0f/(1.0f+expf(-z_val)); z_silu_vals[i] = z_val *
   z_sigmoid_val;` (`selective_scan_bwd_kernel.cuh:193-194`). **A single
   quotient and a reciprocal-then-product are not the same float**, so
   upstream's own backward multiplies by a `silu(z)` its own forward never
   produced. This repository has MEASURED that exact difference on the forward
   side: `SAB_S12_MUL_SIGMOID` bit `gate.out` on 15 of 64 cells,
   IDENTITY_PATHS row 55. Our B4 calls `identical_silu`, full stop.
6. **`exp2f(delta * A * M_LOG2E)` for `delta_a_exp`.** Both their forward and
   backward do it so they are self-consistent, but seam S6 pins
   `identical_exp(delta * A)` and DEVIATION 722 already refused the exp2
   substitution as a different function with an extra rounding. The backward's
   recomputed `da` must be OUR forward's `da` or the recomputation is not a
   recomputation.

**The general rule this produces, and it is worth promoting.** *A recomputed
forward quantity must be spelled with the same function the forward used, not
with an algebraically equal one.* Upstream breaks it once, at `silu(z)`, and
that single break makes their gradient inconsistent with their own output.

---

## 7. The gates, the sabotages and the negative controls, SPECIFIED AND NOT BUILT

A pin with no fixture separating it from the unpinned spelling is a belief,
and **an arm that has never been shown to fail is an arm nobody has tested.**

**Per stage or not at all.** T8 is why; an output-only comparison of `dx` will
call a thirteen-stage divergence inert.

**The card, twenty six stages**, upstream to downstream in the backward's own
direction. Sixteen activation gradients then ten parameter gradients, and the
split is exactly the split 4.1 draws, deliberately, so a gate can assert a
clause over one group and not the other.

    bwd.dres [M,dm]   bwd.dg [M,di]    bwd.dsk [M,di]   bwd.dz [M,di]
    bwd.dh [M,di,N]   bwd.dCm [M,N]    bwd.dBm [M,N]    bwd.du_s [M,di]
    bwd.ddelta [M,di] bwd.ddtp [M,di]  bwd.du [M,di]    bwd.dconv [M,di]
    bwd.dhin [M,di]   bwd.dnrm [M,dm]  bwd.drstd [M]    bwd.dx [M,dm]
    bwd.dW_in [2di,dm]   bwd.dW_x [r+2N,di]   bwd.dW_dt [di,r]
    bwd.dW_out [dm,di]   bwd.dA_log [di,N]    bwd.dD [di]
    bwd.db_dt [di]       bwd.dcw [di,K]       bwd.dcb [di]
    bwd.dw_norm [dm]

**The sabotage switches, with the INERT column that is the point of the
table.** The lesson that produced the column is 3.5's `adv_softplus_guard`
finding: the arm ran, reached its branch, and moved nothing. Each gate must
assert its arm's predicted cell count, not merely that the arm ran.

| switch | breaks | first moved stage | predicted INERT when |
|---|---|---|---|
| `SAB_BWD_S9B_FORWARD` | `dh` walked ascending in `t` | `bwd.dh` | `L == 1` |
| `SAB_BWD_S9B_DA_OFFSET` | `da[t]` where `da[t+1]` belongs | `bwd.dh` | `L == 1`, or all `delta` equal across `t` |
| `SAB_BWD_S9B_UNFUSED` | T1's `fma` split into mul then add | `bwd.dh` | `L == 1` |
| `SAB_BWD_H_SUBTRACT` | upstream's `h[t] - dbu[t]` | `bwd.ddelta` | `h[t-1] == 0`, i.e. `t == 0` only |
| `SAB_BWD_S14_DIVISION` | softplus' as upstream's single division | `bwd.ddtp` | `biased` outside `[8, 14]` |
| `SAB_BWD_S10_N_DESCENDING` | `du_s` and `ddelta` folded `n` descending | `bwd.du_s` | `N == 1`, or fewer than 2 nonzero terms |
| `SAB_BWD_DA_ASCENDING` | T4's `t` fold reversed | `bwd.dA_log` | `L == 1` |
| `SAB_BWD_JOIN_ORDER` | T6's three way join permuted | `bwd.du` | one contribution dominates |
| `SAB_BWD_S1B_FOLD_DESCENDING` | `drstd` folded `j` descending | `bwd.drstd` | `dm == 1` |
| `SAB_BWD_S13_TAPS_REVERSED` | B31's correlation index reversed | `bwd.dhin` | conv weights symmetric in `k` |
| `SAB_BWD_DBDC_ATOMIC` | `dBm`, `dCm` by float atomicAdd | **MB4, not MB3** | `d_inner == 1` |
| ~~`SAB_BWD_RSTD3_ASSOC`~~ | **WITHDRAWN, VACUOUS BY CONSTRUCTION.** `(r*r)*r` and `r*(r*r)` are the SAME two roundings -- one product `p = r*r`, then one product `p*r` -- because IEEE multiplication is commutative, so the arm is bit identical on EVERY input and not merely on powers of two. An arm that cannot be witnessed may not be credited (the mamba3 lane's DEVIATION 834 rule) | -- | always |
| `SAB_BWD_RSTD3_POW` | `rstd^3` as `identical_pow(rstd, 3.0)`, the spelling T7 forbids by name. THE REPLACEMENT for the withdrawn row above, DEVIATION 1086 | `bwd.dx` | never predicted inert; `identical_pow` is a different function |
| `SAB_BWD_T1_SEED_ADD` | T1's seed folded in as a stored `+0.0` instead of omitted (DEVIATION 1082) | `bwd.dh` | no PLANTED `-0.0` in `dy*Cm` at `t = L-1`, i.e. every hashed fixture |
| `SAB_BWD_DDELTA_TWO_FOLDS` | `ddelta` as two folds over `n` joined by one add, section 2.2's reading (DEVIATION 1083) | `bwd.ddelta` | one of the two paths is zero for every `n`, which includes `t == 0` on prefill |
| `SAB_BWD_S12_MUL_SIGMOID` | B4 recomputes `silu(z)` as `z*sig(z)`; the same defect upstream's own backward has | `bwd.dsk` | `z == 0` |
| `SAB_BWD_SILU_DERIV_ALT_ASSOC` | B7 as `sig + z*sig*(1-sig)` | `bwd.dz` | not predicted; the transformer lane's twin has never fired either |
| `SAB_BWD_S12B_ASSOC` | B8 as `dg * (sk * silu')` | `bwd.dz` | not predicted |
| `SAB_BWD_INNER_FROM_NRM` | B38 recovers `inner` as `norm.out / w` | `bwd.dw_norm` | `w_norm[j]` a power of two |
| `SAB_BWD_PARAM_ATOMIC` | T5's `b` fold by float atomicAdd | **MB4, not MB3** | `B == 1` |

Note the two rows whose failure lands on MB4 rather than MB3. **An atomic
accumulation is bit identical to a pinned fold on any run where the arrival
order happens to match**, so a single-launch oracle comparison can pass with
the atomic in place. Only a repeat-launch or launch-geometry gate can see it.

**The ten gates, each an OWED item.** What each asserts, and the vacuity guard
without which it is not evidence.

| gate | where | asserts, and the guard |
|---|---|---|
| MB1 `check_backward_routing_is_the_table` | host | the fourteen gemm-routed calls return the shapes 3.2 names, `m`, `n`, `k` pairwise distinct at every call, each returned `(m', n')` equal to the target buffer's shape computed independently. It catches the error the gemm lane found most likely, a router that assumes `dC` is always the left operand; three of the six routings put it on the right. `SAB_BWD_UNTRANSPOSED` and `SAB_BWD_OPERAND_ORDER` fired THROUGH this lane's entry points |
| MB2 `check_backward_is_the_derivative` | host | that the answer is the RIGHT derivative, which no other gate asks, because **a transposed conv backward is bit identical on three boxes**. **The gemm lane got a tolerance-free version of this and this lane cannot have one**, since `exp`, `softplus`, `silu` and `rsqrt` make no step size exact. The split: the BILINEAR sub-seams (B2, B3, B24, B25, B27, B28, B31, B32, B33, B35, B36, B10, B12, B42) get a central difference at `h = 1` on an integer fixture asserted BITWISE; the transcendental seams get a Float64 directional derivative against `mamba_block_ref64` at a tolerance calibrated PER CASE from that function's own self test, **never a fixed epsilon**, because `adv_gate_saturation` passed only at `rtol=1e-3` and **the tolerance was the defect, not the block** (torch's own FP32 failed the same four stages at `1e-7` where values reached 7.25e8). GUARD: refuse to pass unless `dm`, `di`, `r`, `L` are pairwise distinct where they can be, the conv weights are asymmetric in `k`, and `delta` varies across `t`; and DEMONSTRATE that requirement by running `SAB_BWD_S13_TAPS_REVERSED` on a `k`-symmetric weight, showing it PASSES, and printing that as the reason |
| MB3 `check_backward_device_matches_oracle` | device | all twenty six stages, per cell, bitwise, against the host backward oracle through the same seam functions. `B` in `{1,2,3}`, `L` in `{1,4,16,64,257}`, `d_model` in `{8,16}`, both modes. GUARD: **`L = 1` must be present and MARKED**, since four of the fourteen sabotages are predicted inert there and a ledger that omits it reports four false negatives |
| MB4 `check_backward_is_launch_invariant` | device | one byte pattern across block widths 32, 64, 128, 256, two grid paddings, eight repeat launches with fresh dispatches, and a poisoned workspace. The ONLY gate that can see the two atomic arms, plus `SAB_BWD_DA_ASCENDING` and `SAB_BWD_JOIN_ORDER` |
| MB5 `check_backward_activation_gradients_are_batch_invariant` | device | 4.1's positive half over the sixteen activation stages, from ONE sliced `dres`. GUARD: rows 0 and 1 of the `B = 2` run must DIFFER on every batched stage, or a broken row slicer makes this pass for ever on every vendor. **The ten PARAMETER stages are EXCLUDED and the exclusion is ASSERTED**: the gate must FAIL if a parameter stage is bit identical across batch compositions |
| MB6 `check_parameter_gradients_are_not_microbatch_invariant` | device | 3.1 MEASURED. Positive half, the ten stages identical across launches and plans at fixed `(B, L)`; negative half, `dParam(T)` differs from the two accumulated halves. GUARD: the two arms must FIRST agree in exact arithmetic on an integer fixture, or the number measured is a bug in the split; if they agree on the rounding-sensitive fixture, raise VACUOUS. Record cells moved and ulps for each of the ten; expect the six non-GEMM parameters to move more than the four GEMM ones |
| MB7 `check_the_recurrence_is_actually_reversed` | device | the structural gate for T1, because MB3 compares against an oracle a shared misconception would corrupt on both sides. Perturbing `dres` at token `s` MUST move `bwd.dh` at every `t <= s` and MUST NOT at any `t > s`; perturbing the LAST token must move every `t`, and token 0 exactly one. A forward-direction implementation passes the first half and fails the second, so both are asserted. GUARD: **refuse to run at `L == 1`**, and require `L >= 8` with `delta` varying across `t` |
| MB8 `record_the_absorption_at_the_backward_residual` | device | a RECORDING gate, not an asserting one. On a fixture with `dx_norm` of order 1e-3 against `dres` of order 1, record per arm how many stages moved and how many cells of `bwd.dnrm` and `bwd.dx` moved. Prediction, stated so it can be wrong: arms moving ten or more intermediate stages leave `bwd.dx` bit identical on most cells |
| MB9 `price_the_upstream_h_recovery` | either | turns 5's item 4 from an accusation into a number. Max and median ulp distance between the checkpointed read and `h[t] - dbu[t]` per case, plus the fraction of cells in the cancellation band. **If the distance is zero everywhere, say so and downgrade T2's refusal to a preference.** A lane that only confirms its own suspicion is a lane to re-audit |
| MB10 the card and the three vendor leg | device | twenty six per-stage hashes in the order above. GUARD: **the card must read `MOJOLEARN_IDENTITY_TRACE` and must FAIL when it cannot write.** DEVIATION 970's lesson is that `mamba_check.mojo` wrote to a hardcoded `TRACE_PATH` and ignored the variable, so the Apple column had a GREEN CHECK AND NO CARD and the gate could not tell the difference |

**The independent cross check is a cross-lane request and it does not exist.**
Every gate above compares us against us. The forward lane's strongest
non-self-referential result came from `mamba/corpus/`, torch Float64 through
`selective_scan_ref` in the HF block order, and it found a real disagreement,
the `dt_proj.bias` naming clash, that no internal gate could have found. The
backward needs the same instrument: `mamba/corpus/gen_corpus.py` would have to
dump `torch.autograd.grad` outputs in Float64 for the same sixteen cases. That
file belongs to another lane.

---

## 8. The phase ladder

Phase A, this document plus `mamba/checks/mamba_backward.mojo`, is DONE and
COMPILED (2026-09-03, all four configurations), ungated.

Phase C, the arithmetic, is WRITTEN and UNCOMPILED (2026-09-03): the
elementwise kernels, the reverse recurrence, T3's contraction, T4's and T5's
folds and the RMSNorm closed form are all in the two files named in the STATUS
block, with nineteen sabotage arms. **Not one of them has been compiled and
not one has run.** The next rung is `pixi run build-mamba-backward-probe`
clean plus each of the nineteen arms.

Phase B, the host backward oracle, is NOT started and it is the blocking item:
the kernels have nothing normative to be compared against, so MB1, MB2 and MB3
cannot be written. Everything after that is OWED and is listed in section 10:
MB1 and MB2 host only, MB3 on the device, MB7 and MB9 for the recurrence, the
invariance and recording gates, the card, the three-vendor leg, and the
blocked T3 for model scale.
**B, C and D are independently parallelizable once A exists; E through G are
serial, because each one's outputs are the next one's inputs.**

---

## 9. Deviation block

1070 to 1078 are SPENT in this document and in `mamba/checks/mamba_backward.mojo`.
Read the block with the pattern that answers the question, since a range is
reported by its first number only.

    grep -rhoE "DEVIATIONS? 10[7-8][0-9](-10?[0-9]+)?" . | sort -u

| # | what | if it is wrong | state |
|---|---|---|---|
| 1070 | T1, the reverse recurrence, direction, `+0.0` seed, FUSED | every `dh` below `t = L-2` and nine card stages | SPENT |
| 1071 | T2, the explicit `h[t-1]` checkpoint, refusing upstream's subtraction | `N`-times memory spent for nothing, if MB9 records zero ulps | SPENT |
| 1072 | T3, `dBm` and `dCm` as per-token `(1, N, di)` v1 contractions over materialized `h` and `dh` | the whole seam, and 5.3 is the price | SPENT |
| 1073 | T4, the `dA` fold DESCENDING in `t`, against every other fold in both profiles | `dA_log` and a consistency argument lost | SPENT |
| 1074 | T5, the private-slot batch fold with no atomic anywhere | five parameter stages, and MB4 is the only gate that sees it | SPENT |
| 1075 | T6, the three way `du` join, D then scan then xproj | `bwd.du` and every stage after it; the arm may be INERT | SPENT |
| 1076 | T7, the RMSNorm closed form and `rstd^3`'s association | `bwd.dx`, absorbed at T8 on most fixtures | SPENT |
| 1077 | T8, the residual join, `dres` left, one flushed add | nothing visible, which is exactly the hazard | SPENT |
| 1078 | B22, softplus' as a multiply by `identical_sigmoid` | `bwd.ddtp` and everything upstream, only in the `[8,14]` band | SPENT |
| 1079 | the fourteen gemm routings and their workspace sizing | out of bounds writes, which a small shape hides | SPENT |
| 1080 | the six token-axis reductions as ones-vector v1 GEMMs, and the pre-product buffers | six parameter stages | SPENT |
| 1081 | T2's checkpoint PRODUCER: the forward recurrence RE-RUN by `selective_scan_checkpoint_kernel` rather than stored by the certified forward kernel, plus the four buffer layouts | a SECOND SPELLING of S5-S9 drifts from the first; the OWED gate is that the checkpoint's last slot equals the forward's `scan.h` bitwise | SPENT |
| 1082 | T1's seed is an OMITTED operation, not a stored `+0.0` folded in | a `-0.0` at `t = L-1` is laundered; visible only on a PLANTED fixture | SPENT |
| 1083 | `ddelta` is ONE interleaved fold over `n`, resolving this document's own contradiction between 2.2 and 3.2 | `bwd.ddelta` and everything downstream of it | SPENT |
| 1084 | T4 is a SECOND descending pass over `t` reading the materialized `dh`, not a fold inside T1's walk | nothing numerical; `d_arg` is recomputed bit-equal, the cost is `identical_exp` calls | SPENT |
| 1085 | `silu'` is `transformer/checks/transformer_backward.mojo::bwd_silu_backward_kernel`'s chain, transcribed, and B7's row said 3 roundings where there are 4 | `bwd.dz` and `bwd.dconv` | SPENT |
| 1086 | `SAB_BWD_RSTD3_ASSOC` withdrawn as vacuous by construction, replaced by `SAB_BWD_RSTD3_POW` | T7's `rstd^3` clause has no falsifier at all | SPENT |
| 1087 | this repository has TWO self-consistent RMSNorm backwards with different associations and no gate can distinguish them | a future unification silently moves one lane's card | SPENT |
| 1088 | the blocked T3 for model scale | | RESERVED |
| 1089 | unallocated | | RESERVED |

---

## 10. Open questions, OWED, and things a reader should not assume

1. **DONE for the routing layer, OWED for the kernels.**
   `mamba/checks/mamba_backward.mojo` compiled 2026-09-03 unmodified in all
   four configurations. The two kernel files written the same day have NEVER
   been through a compiler, in any configuration. The commands are
   `pixi run build-mamba-backward-probe` clean and once per arm; the arm list
   is in `pixi.toml` beside that task.
2. **The host backward oracle**, twenty six stages, beside
   `mamba_block_oracle`. It is the normative answer and everything compares to
   it. Then MB1 and MB2 host-only, then the device path.
3. **T3's memory, 4.2. The largest unpriced item.** The blocked variant's
   tiling must be `contract_partition(d_inner)`'s leaf boundary, never a VRAM
   budget.
4. **A three-vendor leg.** The forward now has three columns, so the
   dependency that used to block this (an AMD forward column) is discharged.
5. **`causal_conv1d` is not on disk.** `dcw`'s fold order and `dcb`'s
   bias-gradient spelling upstream are UNREAD.
6. **The backward has no determinate torch reference.** The forward could say
   "the reference's order is the profile's" because `selective_scan_ref` is
   readable line by line. Its backward is `torch.autograd` over einsums, and
   einsum's backward grouping depends on the contraction path the optimizer
   picks, so there is no fixed order to transcribe. This lane's answer is to
   derive every association by the chain rule FROM THE PINNED FORWARD, so the
   backward's order is a theorem of the forward's rather than a second source
   of truth. The CUDA kernel's groupings differ in four places, `du`,
   `ddelta`'s A path, `dB` and `dA`, and those are the sabotage arms rather
   than the profile.
7. **The `-0.0` question is inherited and unexamined for the backward.** The
   forward's `pinned_mul` preserves a negative zero product and is SETTLED
   independently, 125 negative zeros surviving S3, S4 and S12. No fold here
   takes a `min`, `max`, `argmin` or `argmax`, so row 13's selection hazard has
   no site in this lane, **but an OPTIMIZER downstream is exactly such a
   consumer and a `max(grad, 0)` clamp meets row 39 in full. Clamps must be
   spelled value-first.**
8. **`refuse_nonfinite` has no backward counterpart yet.** `dres` needs the
   same by-bits refusal the forward gives its inputs, and it is not written.
9. **The forward's clause (c) is stated over rows and the backward needs it
   stated over two groups.** Applied literally to a backward it is false for
   ten of twenty six stages and true for sixteen, for 3.1's reason and not
   because of a defect. **Any backward profile document must split the
   clause.** This is a drafting note for whoever writes
   `mojolearn.identical.mamba1.bwd.fp32.v1`; this document is not that
   contract.
10. **One claim in a file this lane may not edit is stale.**
    `gemm/IDENTICAL_BACKWARD_PLAN.md` row T5 said GELU is REFUSE because
    `checks/numerics.mojo` has no `portable_erff` and no `portable_tanhf`.
    Both exist, at `:1672` and `:1465`, with `portable_gelu_erf:1722`,
    `portable_gelu_tanh:1778`, `identical_erf:2009` and both
    `identical_gelu_*` at `:2023` and `:2039`, DEVIATIONS 820-825. The row's
    move should be PIN.
