# The backward pass under `mojolearn.identical.mamba1.fp32.v1`

Opened 2026-08-25, the mamba BACKWARD lane (DEVIATIONS 1070-1089). This
document answers one question, asked by the lane brief. *What does it take to
make the backward pass of a Mamba-1 block bit identical across Apple, NVIDIA
and AMD GPUs, and how much of it already exists?*

The shape of this document is `gemm/IDENTICAL_BACKWARD_PLAN.md`'s, on purpose,
and section 3.2 of that document is the finding this lane went looking to
inherit. It does inherit it, ten times over.

The short answer, before the detail.

- **The Mamba-1 backward needs no new transcendental.** Every derivative in
  the block is a rational function of quantities the forward already computes
  through pinned primitives. `identical_exp`, `identical_sigmoid`,
  `identical_silu`, `identical_div`, `identical_rsqrt`, `identical_mul_add`
  and `ftz` are the whole toolbox and all seven already ship. Nothing needs
  to be added to `original/numerics.mojo`. That is the good news and it is
  the only unqualified good news in this document.
- **It does need a great deal of new ORDER.** Section 3 classifies 42
  backward operations. Four are copies, two are pure reuse, thirteen route to
  `mojolearn.identical.gemm.fp32.v1`, and **twenty three are genuinely new
  arithmetic**. Of those twenty three, eleven are elementwise spellings whose
  ASSOCIATION has to be declared, four are folds whose order is inherited
  from an existing forward clause, and **eight are new fold or recurrence
  topologies that nothing in this repository declares today**. Section 3.4 is
  the list of eight and it is the real output of this lane.
- **SEQUENCE-LENGTH INVARIANCE IS STRUCTURALLY IMPOSSIBLE, AND IT IS NOT A
  ROUNDING PROBLEM.** The forward's clause (d), prefill equals decode bitwise
  per token, is a theorem of a CAUSAL recurrence. The backward's state
  recurrence is ANTI-CAUSAL. `dh[t]` depends on every token from `t` to
  `L - 1`, so a prefix's backward is a different mathematical object and not
  a differently rounded one. There is no decode backward, there is no
  streaming backward, and no amount of pinning produces one. Section 4.2.
- **The lane inherits the gemm plan's microbatching limitation in TEN places
  and only four of them are matmuls.** `dW_in`, `dW_x`, `dW_dt`, `dW_out`,
  `dA_log`, `dD`, `d(dt_proj.bias)`, `d(conv1d.weight)`, `d(conv1d.bias)` and
  `d(norm.weight)` every one contracts over the token axis. A microbatched
  training step is not bit equal to an unsplit one for any of them, on any
  vendor, ever. Section 4.1.
- **The hardest seam is not the one that looks hardest.** The reverse
  recurrence stays inside one thread and inherits the forward's structural
  launch invariance for free. The hard seam is that **`B` and `C` are shared
  across channels**, so `dB` and `dC` contract over `d_inner`, an axis no
  float crosses in the forward. That crossing has no cheap identical
  spelling, Metal's absent threadgroup float atomics and 32 KB ceiling rule
  out the obvious one, and the priced answer costs `2 * M * d_inner * 16`
  floats of scratch. Section 5.
- **Upstream's own backward is not reproducible even run to run on one
  device**, and it is worth being blunt about that because it is the state of
  the art this lane is measured against. `selective_scan_bwd_kernel.cuh`
  accumulates `dA`, `dB`, `dC`, `dD` and `ddelta_bias` with five
  `gpuAtomicAdd` calls, and `selective_scan_bwd_cuda` picks its block shape
  from `params.seqlen` through a table that is DIFFERENT under `USE_ROCM`.
  Its summation order is therefore a function of the sequence length, the
  arrival order of blocks and the vendor. Section 6.
- **Upstream's backward also recomputes `silu(z)` with a different spelling
  from its own forward.** The forward kernel writes `z / (1 + expf(-z))`
  (`selective_scan_fwd_kernel.cuh:298`), the backward writes
  `z_val * z_sigmoid_val` (`selective_scan_bwd_kernel.cuh:194`). Those are
  not the same float. This repository already has a sabotage switch named for
  exactly that mistake on the forward side, `SAB_S12_MUL_SIGMOID`, and it
  bit 15 of 64 cells of `gate.out`. Section 6.2. Do not port it.
- **NOTHING BELOW HAS RUN.** This document is CONSTRUCTION plus a routing and
  declaration layer. No gate in section 7 exists, no backward fixture has
  been built, no device has executed a backward call, and no line of
  `mamba/original/mamba_backward.mojo` has been compiled. **Every sentence
  about behavior is a prediction until a gate prints.**

---

## 1. Status, and what is being claimed

| thing | state |
|---|---|
| forward `mojolearn.identical.mamba1.fp32.v1` | IDENTITY_PATHS row 55. Clauses (a) through (f) gated on Apple. **TWO vendors, not three**, leg 12 at `1b7c916`, Apple M4 (Metal) and NVIDIA RTX 4090 (CUDA), 17 of 17 card stages bit identical. **AMD has no mamba column at all.** |
| `mojolearn.identical.gemm.fp32.v1` | IDENTITY_PATHS row 40, CLOSED on three vendors |
| `gemm/original/gemm_backward.mojo` | derived and coded by the gemm backward lane, **UNGATED** |
| the backward derivation of section 2 | derived here, **UNGATED, UNCOMPILED** |
| the seam classification of section 3 | derived here |
| the eight new topologies of section 3.4 | DECLARED here, **NO KERNEL WRITTEN** |
| the routing layer `mamba/original/mamba_backward.mojo` | written here, **NEVER COMPILED** |
| everything in section 7 | SPECIFIED, NOT BUILT |

The completion claim this lane may make when the gates of section 7 are green
on three vendors is exactly one sentence, and it is written to be as narrow
as the forward lane's.

> **Cross-vendor bit-identical FP32 gradients for one Mamba-1 block, at a
> fixed batch composition and a fixed sequence length, under the declared
> profile.**

Read every qualifier. Not identical training. Not identical models. Not
identical gradients under a different microbatch schedule, which section 4.1
proves impossible. Not a streaming or decode backward, which section 4.2
proves impossible. Not Mamba-2. And not one word of it until the gates print
on three boxes, which for this lane means an AMD column the FORWARD does not
have yet either.

**The profile is not amended.** `mamba/IDENTICAL_MAMBA_CONTRACT.md` is
consumed here and never edited. Section 9 of that contract says "No training,
no backward" and that sentence stays true of v1. What this document proposes
is a SEPARATE profile name, `mojolearn.identical.mamba1.bwd.fp32.v1`, which
CONSUMES the forward profile and adds clauses of its own. Anything that would
require a forward seam decision to move is a `mamba1.fp32.v2` and is
announced loudly, never spelled quietly here.

---

## 2. The backward, derived

### 2.1 Notation, taken from the forward and not invented

Token-major throughout, `M = B * L` rows, exactly the oracle's layouts
(`mamba/original/mamba_oracle.mojo::MambaStages`). `dm` is `d_model`, `di`
is `d_inner = 2 * dm`, `r` is `dt_rank`, `N` is `D_STATE = 16`, `K` is
`D_CONV = 4`.

The forward, restated from contract section 2 with every intermediate named,
because the backward reuses the forward's own rounded intermediates and a
name that does not exist in the forward cannot be reused.

    ss[t]          = sum_j x[t,j]^2                      S1
    rstd[t]        = rsqrt(ss[t]/dm + eps)               S2
    inner[t,j]     = x[t,j] * rstd[t]                    S3
    nrm[t,j]       = w_norm[j] * inner[t,j]              S4
    P[t,c]         = sum_j nrm[t,j] * W_in[c,j]          S17a   c in [0, 2di)
    hin            = P[:, 0:di]        z = P[:, di:2di]  a split, not a seam
    conv[b,l,d]    = cb[d] + sum_k cw[d,k]*hin[b,l-3+k,d]  S13
    u[t,d]         = silu(conv[t,d])                     silu.out
    XP[t,c]        = sum_d u[t,d] * W_x[c,d]             S17b
    dtl | Bm | Cm  = XP columns                          a split
    dtp[t,d]       = sum_p dtl[t,p] * W_dt[d,p]          S17c
    biased[t,d]    = dtp[t,d] + b_dt[d]                  S14 add
    delta[t,d]     = softplus(biased[t,d])               S14
    A[d,n]         = -exp(A_log[d,n])                    S15
    da[t,d,n]      = exp(delta[t,d] * A[d,n])            S5, S6
    dbb[t,d,n]     = delta[t,d] * Bm[t,n]                S7
    dbu[t,d,n]     = dbb[t,d,n] * u[t,d]                 S8
    h[t,d,n]       = fma(da, h[t-1,d,n], dbu)            S9
    y[t,d]         = sum_n fma(Cm[t,n], h[t,d,n], acc)   S10
    sk[t,d]        = y[t,d] + u[t,d]*D[d]                S11
    g[t,d]         = sk[t,d] * silu(z[t,d])              S12
    O[t,c]         = sum_d g[t,d] * W_out[c,d]           S17d
    res[t,c]       = x[t,c] + O[t,c]                     S16

`h[-1, d, n]` is the carried state, zeros on prefill (contract section 5).

### 2.2 The backward, seam by seam, with every association written out

Given `dres[M, dm]` arriving from downstream. `d<name>` is the gradient of
the scalar loss with respect to `<name>`.

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
           dh[L-1+1]   = 0                                  see 2.4
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
    S14'   ddelta      = ddelta_B + ddelta_A                the two paths join
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
                                                            fold over K, dropped
                                                            where p+3-k >= L
           dcw[d,k]    = sum over (b,l) of dconv*hin[b,l-3+k,d]
                                                            REDUCES OVER TOKENS
           dcb[d]      = sum over (b,l) of dconv[t,d]       REDUCES OVER TOKENS
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

`S1'` through `S3'` collapse to the standard closed form and are spelled that
way in section 3.4 topology T8 rather than as three separate rounded steps,
because the intermediate `dss` has no use of its own.

### 2.3 The three checks that were done rather than assumed

**Check one, the RMSNorm backward, derived rather than recalled.**
`nrm = w * x * rstd` with `rstd = (ss/dm + eps)^{-1/2}` and `ss = sum_j x^2`.
`d rstd / d ss = -0.5 * rstd^3 / dm`, and `d ss / d x_j = 2 x_j`, so
`d rstd / d x_j = -rstd^3 * x_j / dm`. With `dinner_j = dnrm_j * w_j` and
`R = sum_j dinner_j * x_j`, the closed form is

    dx_norm[t,j] = rstd[t]*dinner[t,j] - (rstd[t]^3 / dm) * x[t,j] * R[t]

The only reduction is `R`, over the FEATURE axis, per row, which is the same
axis and the same length as S1's own fold. That matters and section 4 uses it.

**Check two, the conv backward's index arithmetic.** Forward,
`conv[b,l,d]` reads `hin[b, l-3+k, d]` for `k` in `[0, 4)`. So `hin[b,p,d]`
is read by output position `l = p + 3 - k`, which is in range only for
`k` such that `0 <= p + 3 - k < L`. The backward is therefore a CORRELATION
in the opposite direction and the tap that is dropped at the START of the
forward sequence is dropped at the END of the backward one. A reversed-tap
implementation is bit identical on a symmetric fixture and wrong on every
other, which is why gate MB7's sabotage exists.

**Check three, the off-by-one on `da`.** `h[t]` influences `y[t]` through
`Cm[t]` and `h[t+1]` through `da[t+1]`, NOT through `da[t]`. Upstream is
explicit about this and spells it as
`thread_reverse_data[i-1].x = delta_a_exp` at index `i`, a deliberate shift by
one slot (`selective_scan_bwd_kernel.cuh:255`), with the last element of the
block taken from the neighboring thread's `smem_delta_a`. Getting this wrong
produces a plausible gradient that is bit identical on any fixture where all
`da` are equal, so gate MB7's fixture must plant unequal `delta` across tokens
and the gate must REFUSE to pass if it does not.

### 2.4 The gradient of the carried state is DROPPED, and that is upstream's
### decision, adopted and named

`selective_scan_fn`'s own docstring says "the gradient of the last state is
not considered in the backward pass", and the kernel seeds its reverse scan
with `make_float2(1.f, 0.f)` at the final chunk. This lane adopts that, so
`dh[L, d, n] = 0` and no gradient flows out of a block call into the state a
previous call produced. It is the right scope decision for one block, and it
is also the reason section 4.2's answer is what it is. It is recorded here so
that nobody later reads "the backward is correct" as "the backward does
truncated BPTT correctly", because it does no BPTT at all.

The same is true of the conv window. `dhin` at a pre-sequence position would
flow into the previous call's output and is dropped.

---

## 3. The classification, and the count that is the point of this lane

### 3.1 The rule used

Each operation is (a) an already-pinned operation this repository owns and
calls unchanged, (b) routable to `mojolearn.identical.gemm.fp32.v1`, or (c)
genuinely new arithmetic that needs a pinned order of its own. There is no
fourth bucket and no "basically the same as". A pure data movement is a copy
and is counted separately, because a copy moves bits untouched
(contract section 4) and cannot be wrong in a way that a gate would have to
measure numerically.

Category (c) is subdivided, because "new" hides a large range.

- **(c-elementwise)** no fold at all, but the ASSOCIATION or the SPELLING is a
  free choice that moves the last bit.
- **(c-inherited)** a fold whose order is already declared by a forward seam
  and is reused verbatim. New arithmetic, no new topology.
- **(c-topology)** a fold or a recurrence that nothing in this repository
  declares. **These eight are the real cost of this lane.**

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
| B7 | `silu'(z)` composite | none | c-elem | 3 roundings, association pinned |
| B8 | `dz = dg * sk * silu'(z)` | none | c-elem | 4 factors, order pinned |
| B9 | `dy = dsk` | none | copy | |
| B10 | `du_D = pinned_mul(dsk, D[d])` | none | c-elem | |
| B11 | `pD = pinned_mul(dsk, u)` | none | c-elem | the pre-product for B12 |
| B12 | `dD[d] = sum_t pD` | **tokens** | b | ones-vector `OP_NN` at `(1, di, M)` |
| B13 | the reverse recurrence `dh` | **sequence** | **c-top T1** | |
| B14 | `h[t-1]` availability | none | **c-top T2** | a storage decision, see 5.2 |
| B15 | `dCm[t,n] = sum_d dy*h` | **d_inner** | **c-top T3** | v1 order, no v1 entry |
| B16 | `dBm[t,n] = sum_d w*dh` | **d_inner** | **c-top T3** | same topology |
| B17 | `du_s = sum_n dh*dbb` | `N` | c-inh | S10's ascending fused chain |
| B18 | `ddelta = sum_n (B path + A path)` | `N` | c-inh | interleaving pinned, see 3.4 |
| B19 | `dA` fold over `t` | **sequence** | **c-top T4** | direction is the choice |
| B20 | parameter fold over `b` | **batch** | **c-top T5** | shared by `dA`, `dD`, `db_dt`, `dcw`, `dcb` |
| B21 | `dA_log = pinned_mul(dA, A)` | none | c-elem | |
| B22 | `ddtp = softplus'` applied | none | c-elem | DIV or MUL, see 3.4 |
| B23 | `db_dt[d] = sum_t ddtp` | **tokens** | b | ones-vector `OP_NN` at `(1, di, M)` |
| B24 | `ddtl = ddtp . W_dt` | `di` | b | gemm `dA` |
| B25 | `dW_dt = ddtp^T . dtl` | **tokens** | b | gemm `dB`, `k' = M` |
| B26 | `dXP = concat(ddtl, dBm, dCm)` | none | copy | |
| B27 | `du_x = dXP . W_x` | `r + 2N` | b | gemm `dA` |
| B28 | `dW_x = dXP^T . u` | **tokens** | b | gemm `dB`, `k' = M` |
| B29 | `du = du_D + du_s + du_x` | none | **c-top T6** | a three way join, ORDER |
| B30 | `dconv = du * silu'(conv)` | none | c-elem | |
| B31 | `dhin` four tap correlation | `K` | c-inh | S13's tap order, reversed index |
| B32 | `dcw[d,k] = sum_t dconv*hin_k` | **tokens** | b | four ones-vector `OP_NN` calls |
| B33 | `dcb[d] = sum_t dconv` | **tokens** | b | ones-vector `OP_NN` at `(1, di, M)` |
| B34 | `dP = concat(dhin, dz)` | none | copy | |
| B35 | `dnrm = dP . W_in` | `2di` | b | gemm `dA` |
| B36 | `dW_in = dP^T . nrm` | **tokens** | b | gemm `dB`, `k' = M` |
| B37 | `dinner = pinned_mul(dnrm, w_norm)` | none | c-elem | |
| B38 | `pW = pinned_mul(dnrm, inner)` | none | c-elem | pre-product for B39 |
| B39 | `dw_norm[j] = sum_t pW` | **tokens** | b | ones-vector `OP_NN` at `(1, dm, M)` |
| B40 | `drstd[t] = sum_j dinner*x` | `dm` | c-inh | S1's ascending fused chain |
| B41 | `dx_norm` closed form | none | **c-top T7** | `rstd^3` association, two terms |
| B42 | `dx = dres + dx_norm` | none | **c-top T8** | THE ABSORPTION SITE |

### 3.3 The count

    copies                                    4
    (a)  existing primitive, no new order     2
    (b)  routable to gemm v1                 13
    (c)  genuinely new arithmetic            23
         of which  c-elementwise             11
                   c-inherited order          4
                   c-TOPOLOGY                 8
                                       -------
                                             42

Counted by contract seam instead of by operation, so it can be read beside
contract section 7's list, every one of S1 through S17 has a backward and
**not one of the seventeen is answered entirely by (a) or (b)**. S17 comes
closest, since all eight of its gemm calls route, but even S17 leaves the
concatenations that build `dXP` and `dP` and the join at B29.

The honest headline is that **the mamba backward is not a routing problem the
way the gemm backward was.** The gemm backward lane could write a 595 line
file with no multiply in it. This lane cannot, and
`mamba/original/mamba_backward.mojo` is short because it routes what routes
and DECLARES the rest without writing kernels, which is the correct shape for
a lane whose finding is that most of the work is new.

### 3.4 The eight new topologies, declared

These are declarations in the contract's own vocabulary. No kernel exists for
any of them. Each names what is pinned, what the alternatives are, and what
it costs if the choice is wrong.

**T1, the reverse recurrence (S9', DEVIATION 1070).**
Fold direction REVERSE over `t` from `L - 1` down to `0`. Seed
`dh[L, d, n] = +0.0` (section 2.4). One rounding per step, FUSED.

    contrib   = ftz(pinned_mul(dy[t,d], Cm[t,n]))
    dh[t,d,n] = ftz(identical_mul_add(da[t+1,d,n], dh[t+1,d,n], contrib))

FUSED mirrors S9, whose clause says fusion is pinned because only fusion has
a portable spelling. The alternative is the reference's two roundings, which
is `SAB_BWD_S9B_UNFUSED`. If wrong, every `dh` from `t = L - 2` downward
moves, and everything derived from `dh` moves with it, which is `du_s`,
`ddelta`, `dBm`, `dA` and through them nine card stages. The cost of being
wrong is the whole lane.

**T2, the `h[t-1]` availability (DEVIATION 1071).**
PINNED as an explicit checkpoint. The forward pass stores `h[t, d, n]` for
every `t` in `[-1, L)` into a caller-owned buffer of
`B * di * (L + 1) * N` floats, and the backward READS `h[t-1]` from it.

    REFUSED alternative, and it is upstream's:
        a = h[t] - dbu[t]        (selective_scan_bwd_kernel.cuh:290)

Upstream recovers `da * h[t-1]` by subtracting `dbu[t]` from `h[t]`. Since
`h[t] = round(da*h[t-1] + dbu[t])`, that subtraction is exact only when no
rounding occurred, and it suffers unbounded relative cancellation whenever
`|da * h[t-1]| << |dbu[t]|`, which is the normal case early in a sequence
where `h` is near zero and `dbu` is not. It is a memory optimization with an
accuracy cost that upstream does not state. This is the "do not port their
bugs" rule with a name on it. The price of refusing it is `N = 16` times the
activation footprint, and section 5.3 prices it in bytes.
`SAB_BWD_H_SUBTRACT` is upstream's spelling, kept so gate MB9 can MEASURE the
ulp distance rather than assert that it matters.

**T3, the `d_inner` contraction for `dBm` and `dCm` (DEVIATION 1072).**
Serial ASCENDING `d` from `+0.0`, one FUSED multiply-add per term, which is
bit for bit `mojolearn.identical.gemm.fp32.v1` at `(m', n', k') = (1, N, di)`
per token. FUSED is forced by the routing and is consistent with S10, which
pins exactly this shape for a sum of products.

    dCm[t,n] = fold over d of fma(dy[t,d],  h[t,d,n],  acc)
    dBm[t,n] = fold over d of fma(w[t,d],  dh[t,d,n],  acc)
    with      w[t,d] = ftz(pinned_mul(delta[t,d], u[t,d]))

`w` is pre-formed so the leaf is one `fma`. The alternative pre-forms
`R[t,d,n] = pinned_mul(dh, u)` instead, which is algebraically the same and
costs `N` times the memory and one extra rounding per `(t,d,n)` rather than
per `(t,d)`. If `di > 128` the fold is NOT a single chain, it is
`contract_partition(di)` leaves combined by v1's balanced tree, and an
implementation that ignores that is a different answer at every real model
width. **`di` is the reduced length, never a launch quantity.**

**T4, the `dA` fold over `t` (DEVIATION 1073).**
PINNED DESCENDING in `t`, the direction the reverse pass already walks, from
a `+0.0` seed, FUSED.

    dA[d,n] accumulates, for t from L-1 down to 0:
        acc = ftz(identical_mul_add(d_arg[t,d,n], delta[t,d], acc))

This is the one place where this lane pins a DESCENDING fold, and every other
fold in both profiles is ascending. The reason is that ascending would need
either a second pass over `M * di * N` stored per-token products or a second
traversal, and the price is stated rather than hidden. `SAB_BWD_DA_ASCENDING`
is the other direction, and gate MB4 must show it bites, otherwise this
deviation is unmeasured and should be re-decided in favor of consistency.

**`dA` IS ROUTABLE TO GEMM v1 AND IS DELIBERATELY NOT ROUTED, which is the
one place this plan declines a category (b) answer.** With the per-token term
`q[t, d*N + n] = pinned_mul(d_arg[t,d,n], delta[t,d])` materialized at
`[M, di*N]`, `dA` is a ones-vector `OP_NN` at `(1, di*N, M)`, exactly like the
six reductions of section 3.2, and its fold order would then be v1's ascending
leaves and balanced tree rather than a hand-declared direction. It is declined
because `q` is a THIRD buffer of `M * di * N` floats on top of T2's `h` and
T3's `dh`, and section 5.3 already names that memory as the largest open item
of the lane. The in-thread fold costs nothing. If phase K's blocked variant
ever makes the memory affordable, this decision should be revisited and the
routed form is strictly preferable, because it deletes a declared order in
favor of a certified one.

**T5, the parameter fold over `b` (DEVIATION 1074).**
Every parameter gradient that a thread can only compute per `(b, d)` is
written to a private slot `partial[b, d, ...]`, with NO atomic anywhere, and a
SECOND kernel folds over `b` ASCENDING from `+0.0` with a plain flushed add.
Shared by `dA`, `dD`, `db_dt`, `dcw` and `dcb`.

The alternative is a float `atomicAdd`, which is what upstream does five
times, and it is REFUSED for the reason IDENTITY_PATHS rows 1 and 2 refuse
it. Arrival order is not reproducible run to run on one device, never mind
across three. The cost of the private-slot form is `B` times the parameter
footprint, which is trivial, and one extra kernel launch.

**Metal check, done rather than assumed.** This fold never enters threadgroup
memory. Metal has float `atomicAdd` and warp shuffles but NO threadgroup float
atomics, and 32 KB of threadgroup memory, so a fold that needed a scratch
proportional to `d_inner` inside a threadgroup would not fit at any real
width. Routing it through GLOBAL private slots and a second kernel is the
spelling that has no ceiling, and it is the same move the ones-vector
reductions of category (b) make.

**T6, the three way join at `du` (DEVIATION 1075).**
PINNED ORDER, stated once and never a scheduler's choice.

    du[t,d] = ftz(ftz(ftz(du_D[t,d]) + du_s[t,d]) + du_x[t,d])

D-skip first, then the scan path, then the x_proj path. The order follows the
forward's own data flow, since `u` is consumed by S11 then by the scan then by
S17b. Three floats added in three roundings, and any permutation is a
different answer. `SAB_BWD_JOIN_ORDER` permutes it. **Predicted to be INERT on
any fixture where one contribution dominates the other two**, which is the
absorption hazard applied to a sabotage, so gate MB4's fixture for this arm
must plant the three at comparable magnitude and the gate must raise VACUOUS
rather than pass if it cannot.

**T7, the RMSNorm backward closed form (DEVIATION 1076).**
Two terms, and the second's association is the choice.

    c3   = ftz(pinned_mul(ftz(pinned_mul(rstd, rstd)), rstd))
    s    = ftz(identical_div(c3, Float32(dm)))
    t2   = ftz(pinned_mul(ftz(pinned_mul(s, x[t,j])), drstd[t]))
    t1   = ftz(pinned_mul(rstd, dinner[t,j]))
    dx_norm[t,j] = ftz(t1 - t2)

`rstd^3` is `(rstd*rstd)*rstd`, left associated, never `portable_powf`.
`c3 / dm` is a division by an exact power-relevant integer through
`identical_div`, hoisted per row because it is loop invariant in `j` and a
hoisted flushed division moves no bits. The subtraction is UNFUSED because
the two terms are separately rounded products in every reference spelling.
The alternative folds the `-1/dm` into the fold as a negative scale, which is
one fewer rounding and a different answer.

**T8, the residual join at `dx` (DEVIATION 1077). THE ABSORPTION SITE.**

    dx[t,j] = ftz(ftz(dres[t,j]) + dx_norm[t,j])

One add, order pinned with `dres` on the left. This is the exact twin of S16,
and IDENTITY_PATHS row 55 records what happened there. At the shape where a
sabotage arm was STRONGEST, thirteen of sixteen intermediate stages moved,
`out_proj.out` differed on 23 of 64 cells, and `residual.out` was STILL BIT
IDENTICAL, because the residual add put a value of order 1e-3 beside one of
order 1 and rounded the difference away. **An output-only gate called that arm
inert.** `dx` is the backward's `residual.out`. Any gate that compares only
`dx` will report a broken norm backward as green, and section 7 is per stage
for that reason and no other.

### 3.5 The eleven elementwise associations, in one place

Each is one line and each moves a bit if spelled the other way. They are
listed together because a reader skimming for "the hard part" will skip them,
and eleven quiet wrong answers are worse than one loud one.

| op | pinned | the alternative that is wrong |
|---|---|---|
| B5 | `pinned_mul(dg, identical_silu(z))` | `pinned_mul(dg, pinned_mul(z, sig))`, upstream's, see 6.2 |
| B7 | `sig * (1 + z*(1 - sig))` left to right | `sig + z*sig*(1-sig)` |
| B8 | `((dg * sk) * silu')` | `dg * (sk * silu')`, and upstream's four factor chain |
| B10 | `pinned_mul(dsk, D[d])` | fusing it into B29's join |
| B11 | `pinned_mul(dsk, u)` | fusing into the ones-vector leaf, which the GEMM already does |
| B21 | `pinned_mul(dA, A)` | recomputing `-exp(A_log)`, which is bit equal but is a second call |
| B22 | see below | see below |
| B30 | `pinned_mul(du, silu'(conv))` | `identical_div` of the silu quotient form, which has no derivative form |
| B37 | `pinned_mul(dnrm, w_norm[j])` | folding `w_norm` into B35's gemm |
| B38 | `pinned_mul(dnrm, inner)` | using `nrm/w` to recover `inner`, a division |
| B41 | T7 above | T7 above |

**B22 is the one that has an upstream disagreement (DEVIATION 1078).**
`softplus'(x) = sigmoid(x)`, and there are two spellings.

    PINNED   ddtp = ftz(pinned_mul(ddelta, identical_sigmoid(biased)))
    upstream ddtp = ddelta / (1 + expf(-biased))     bwd_kernel.cuh:454-457

Upstream spells it as ONE DIVISION rather than a multiply by a sigmoid, and
those are not the same float, because `identical_sigmoid` itself is
`portable_divf(1, 1+e)` and then a product, which is two roundings where
upstream has one. The guard is the same in both, `biased <= 20` takes this
arm and above it the derivative is exactly `1.0`, and the operand is the
PRE-softplus biased value in both. The pinned choice is the multiply, because
the profile's forward already owns `identical_sigmoid` as row 52's seam and
because a division by a reconstructed `1 + exp(-x)` re-derives a quantity
`identical_sigmoid` already computes. `SAB_BWD_S14_DIVISION` is upstream's.
The distinguishing band is the same one contract section 4's softplus note
names, `biased` in roughly `[8, 14]`, and a fixture that only straddles 20
passes this vacuously. That has already bitten this lane once, on
`adv_softplus_guard`, where 256 inputs spanned `[19.87, 20.10]` with ZERO
cells in the distinguishing band and a rebuild under `S14_THRESHOLD_10`
produced 23 of 23 byte-identical dumps. **BITWISE INERT.** Do not repeat it.

---

## 4. The three invariance questions, answered separately

The argument in each case is the reduction axis and nothing else.

### 4.1 Batch invariance. HALF YES, HALF STRUCTURALLY IMPOSSIBLE.

Two different claims hide under one word and they must be separated.

**Activation gradients. YES, achievable, and it is the backward twin of
contract clause (c).** `du`, `ddelta`, `dz`, `dconv`, `dhin`, `dnrm`,
`dinner`, `dx`, `dBm`, `dCm` and `dh` reduce over the feature axis, the state
axis `N`, the tap axis `K`, the `d_inner` axis and the sequence axis. **Not
one of them reduces over the batch.** So a sequence's activation gradients can
be bit identical whether it shares the launch with zero, one or two others,
for the same structural reason the forward's clause (c) holds. This is the
clause worth gating and worth claiming, and it is the clause vLLM's
batch-invariant mode does not offer for this architecture in either direction.

**Parameter gradients. NO, and the question is malformed.** `dW_in`, `dW_x`,
`dW_dt`, `dW_out`, `dA_log`, `dD`, `db_dt`, `dcw`, `dcb` and `dw_norm` are
each DEFINED as a sum over the batch. Asking a sum to be invariant to its own
terms is not a coherent request. The coherent, weaker clause, and the one to
gate, is that **at a fixed batch CONTENT and a fixed token order the bits do
not depend on launch geometry**, which section 4.3 says is achievable.

**Microbatch equivalence. STRUCTURALLY IMPOSSIBLE, in TEN places.** This is
the gemm plan's section 3.2 inherited whole.

> The forward is batch invariant because the frozen contract forbids the leaf
> size from depending on `m`. The weight gradient contracts over the batch, so
> the batch arrives as `k`, where the same contract REQUIRES the leaf size to
> depend on it.

Four of the ten are literal gemm `dB` calls with `k' = M` and inherit the
gemm plan's finding verbatim. B3, B25, B28 and B36. **The other six are new
and they are NOT matmuls**, so a reader who has already absorbed the gemm
finding will assume they are exempt and they are not. B12 `dD`, B23 `db_dt`,
B32 `dcw`, B33 `dcb`, B39 `dw_norm` and B19 plus B20 for `dA_log`. Each is a
token-axis fold whose partition is a function of the token count, either
because it routes to a v1 ones-vector GEMM at `k' = M`, or because T4's
sequence fold and T5's batch fold are both sums whose term count is the batch
and sequence size.

The consequence, stated as bluntly as the gemm plan states it. `dParam` over
`T` tokens is not the same bits as `dParam` over two halves accumulated, and
cannot be under any partition, because those are two different sums of the
same terms in a different order. **The microbatch schedule is part of a
training run's numerical specification.** Two runs of the same code at
different gradient accumulation factors are two different numerical
experiments and a reproducibility claim must name the factor.

### 4.2 Sequence-length invariance. STRUCTURALLY IMPOSSIBLE, AND IT IS NOT A ROUNDING PROBLEM.

This is the finding that has no counterpart in the gemm lane and it is worse
than the microbatch one, because it is not about rounding at all.

The forward's clause (d) is prefill equals decode bitwise per token, and it
holds because the recurrence is CAUSAL. Token `t`'s output is a function of
tokens `0` through `t` only, so truncating the sequence after `t` cannot
change it, and DEVIATION 721's bias seed makes the one place the two paths
could have differed identical by construction. IDENTITY_PATHS row 55 records
that gate D earned this at the composition point, 2,152 cells at `d_model` 8
and 4,168 at 16.

**The backward's state recurrence runs the other way.** From section 2.2,

    dh[t,d,n] = dy[t,d]*Cm[t,n] + da[t+1,d,n]*dh[t+1,d,n]

so `dh[t]` depends on every token from `t` to `L - 1`. Truncate the sequence
and `dh[t]` is a DIFFERENT NUMBER, not a differently rounded one. Every
quantity downstream of `dh` inherits that, which is `du_s`, `ddelta`, `dBm`,
`dA`, and through `du` and the conv backward, `dW_in`, `dx` and everything
else. There is no prefix property to preserve and therefore nothing to pin.

Three consequences that follow immediately and that somebody will otherwise
have to be told twice.

1. **There is no decode backward.** A per-token streaming backward is not a
   different implementation of the same function, it does not exist. Training
   a Mamba block requires the whole sequence in hand.
2. **Every parameter gradient's bits depend on `L`** as well as on `B`,
   because its token fold has `M = B * L` terms.
3. **The only sequence-shaped invariance available is CHUNK INVARIANCE**, and
   it is worth claiming precisely because upstream does not have it. If the
   reverse pass is ever segmented, the segmented answer must equal the
   unsegmented one bitwise, which requires the segment boundary to carry the
   exact `dh` and the chunk count never to reach the arithmetic. Upstream
   fails this by construction, since `selective_scan_bwd_cuda` derives
   `kNThreads * kNItems` from `params.seqlen` through one table on CUDA and a
   DIFFERENT table under `USE_ROCM`, and the `cub::BlockScan` and
   `cub::BlockReduce` fold shapes follow `kNThreads`. **A block count is a
   summation order**, and theirs is a function of the sequence length and the
   vendor. This lane's declared shape is one thread per `(b, d)` walking the
   whole sequence serially, which has ONE chunk by construction, so chunk
   invariance is trivially true today and becomes a real clause only if the
   chunked variant of section 8 phase J is ever built.

**Lead with this in any summary.** A limitation found on paper is worth more
than one found on a rented GPU next week, and this one would have been found
on the GPU as a gate that could not be written.

### 4.3 Launch invariance. YES, achievable, but the forward's structural argument does NOT survive.

The forward's argument is quoted from
`mamba/derived/mamba_ssm/ops/selective_scan_interface.mojo`'s kernel docstring
and it is the strongest sentence in that lane.

> A thread owns a whole `(b, d)` recurrence ... no float crosses a thread
> boundary anywhere in this file. There is no shared memory, no reduction, no
> atomic. So `block_size` and the grid decide only WHICH thread holds a pair,
> never the sequence of values accumulated into it.

**That sentence cannot be written about the backward.** Three of the eight new
topologies force a float across a thread boundary.

- T3, `dBm` and `dCm`, contract over `d_inner`, and the `(b, d)` grid gives
  each `d` to a different thread. This is the crossing with no cheap fix and
  section 5 is about it.
- T5, every parameter gradient, contracts over `b`.
- Every category (b) token reduction contracts over `M`, which is by
  construction spread across the whole grid.

So launch invariance stops being a property of the kernel's SHAPE and becomes
a property of each crossing fold's PARTITION. The answer is YES, achievable,
under one rule stated once.

> **Every crossing fold's partition is a pure function of the length of the
> axis it reduces, and of nothing else.** `contract_partition(di)` for T3,
> the batch index for T5, `contract_partition(M)` for the ones-vector
> reductions. No `block_size`, no grid dimension, no `n_chunks`, no
> occupancy heuristic appears in any of them.

That rule is `gemm/IDENTICAL_FP32_CONTRACT.md` section 6.1's rule wearing a
different hat, and the reason it is affordable here is Metal's forcing
function. With no threadgroup float atomics and 32 KB of threadgroup memory,
the cheap wrong answer was never available on one of the three columns, so
the fold had to go through global memory and a second kernel anyway. **That is
the mechanism, and it is why the identity is cheap here rather than free.**

---

## 5. The hardest seam, and why it is not the one that looks hardest

### 5.1 The reverse recurrence is easy, which is a surprise

T1 walks `t` from `L - 1` to `0` inside a single thread that owns one
`(b, d)` pair, holding `N = 16` values of `dh` in registers, exactly as the
forward holds `N` values of `h`. It performs no cross-thread communication,
allocates no shared memory and takes no atomic. **It inherits the forward's
structural launch invariance and batch-composition invariance unchanged**, and
its only new decisions are a direction, a seed and a fusion, all three of
which are one line each in section 3.4.

The brief predicted this would be the crux. It is not. It is the cheapest of
the eight topologies to declare and the second cheapest to implement.

### 5.2 The hard seam is that `B` and `C` are shared across channels

In Mamba-1, `A` and `D` are per channel, `[d_inner, N]` and `[d_inner]`, but
`B` and `C` come out of `x_proj` per TOKEN, `[M, N]`, and are shared by every
one of the `d_inner` channels. The forward reads them, which costs nothing.
The backward SUMS over the channels that read them.

    dCm[t,n] = sum over d in [0, d_inner) of dy[t,d] * h[t,d,n]
    dBm[t,n] = sum over d in [0, d_inner) of w[t,d] * dh[t,d,n]

Every term of that sum lives in a different thread under the forward's grid,
and there are `M * N` such sums, each `d_inner` long. Four structures were
considered and three are rejected.

1. **A float `atomicAdd` per term.** Upstream's answer,
   `gpuAtomicAdd(dB_cur + i*kNThreads, dB_vals[i])`. Arrival order.
   **REFUSED**, IDENTITY_PATHS rows 1 and 2's rule.
2. **A threadgroup tree over `d`.** Needs all `d_inner` contributions for one
   `(b, t)` resident in one threadgroup. **REFUSED on Metal**, which has no
   threadgroup float atomics and 32 KB of threadgroup memory, so at
   `d_inner = 2048` the scratch alone is 8 KB per `n` and the tree is a
   block-shaped fold besides, which is the defect this whole ledger exists to
   prevent.
3. **A fixed-point Int32 accumulator**, IDENTITY_PATHS rows 1 and 2's own
   replacement. Integer addition is associative so arrival order stops
   mattering. **REJECTED HERE**, not refused, because gradients have a far
   wider dynamic range than a histogram bin and row 8 warns that the scale
   must not itself be derived from a float reduction. It stays on the table
   as the fallback if structure 4's memory is unaffordable.
4. **Materialize `h` and `dh` and contract per token.** PINNED, T3. With
   `h[M, di, N]` and `dh[M, di, N]` in global memory, `dCm[t, :]` is the
   product of `dy[t, :]` at `1 x di` with `h[t]` at `di x N`, which is
   `mojolearn.identical.gemm.fp32.v1` at `(m', n', k') = (1, N, di)`,
   exactly, per token. The contracted length is `d_inner`, a LAYER WIDTH,
   bounded and independent of the batch. The arithmetic is certified. No
   atomic, no threadgroup memory, no launch quantity anywhere.

Structure 4 is the right answer and it costs memory, which is the price of
the identity and must be said in the same sentence as the claim.

### 5.3 The price of structure 4, in bytes, stated

`h` and `dh` are each `M * d_inner * N * 4` bytes with `N = 16`.

| shape | `M` | `d_inner` | `h` + `dh` |
|---|---|---|---|
| the gate corpus's largest, `comp_b2_l257_d8` | 514 | 32 | 2.1 MB |
| `d_model` 16, `L` 257, `B` 3 | 771 | 64 | 6.3 MB |
| a 130M parameter Mamba block, `L` 2048, `B` 8 | 16,384 | 1,536 | 3.2 GB |
| the same at `L` 8192 | 65,536 | 1,536 | 12.9 GB |

**So the declared T3 spelling is a GATE-SCALE construction.** It is correct,
it is identical, and at model scale it needs a blocked variant that streams
`d` in tiles. That variant is not built and it is not free to design, because
the tiling must not reach the arithmetic, which means the tile size has to be
`contract_partition(d_inner)`'s leaf boundary and not a memory budget. If a
future lane picks the tile from available VRAM, the answer becomes a function
of the machine and every claim in this document is void.

Recorded as the largest single open item of this lane, ahead of everything in
section 7.

`h`'s buffer is needed by T2 anyway, so T3 pays for `dh` and not for both.
That is the one piece of good news in this subsection.

### 5.4 Why S9's fusion does not create a second hazard

S9 pins `h = fma(da, h, dbu)`, ONE rounding, because only fusion has a
portable spelling. A reader might expect the backward to need `h` and `dbu`
separately in order to differentiate a fused operation, and it does not.
`d/d(da) = h[t-1]` and `d/d(dbu) = 1` regardless of whether the forward fused
them, because fusion changes the ROUNDING of the result and not the
derivative of the exact function. The pinned backward differentiates the exact
bilinear form and rounds its own products through `pinned_mul`. That is the
same choice every autograd system makes and it is worth stating because it is
the one place where "the backward of a fused op" sounds like it should need a
new rule and does not.

---

## 6. Upstream, symbol by symbol

Read from `/Users/andrewhendel/CascadeProjects/upstream/mamba` at `e9594ce`,
the same pin contract section 1 names.

**`causal_conv1d` IS NOT ON DISK.** It is a separate repository
(Dao-AILab/causal-conv1d) and is not checked out under
`/Users/andrewhendel/CascadeProjects/upstream/`. Every statement in this
document about the conv backward is therefore derived from the FORWARD
contract seam S13 and from the index arithmetic of section 2.3 check two, and
is marked INFERRED. Nothing about `causal_conv1d_bwd_function`'s rounding
order, its bias-gradient fold or its tap order has been read. That is a
cross-lane request in the report and a real gap.

### 6.1 What upstream does that we must copy

1. **The reverse recurrence with the same combine operator as the forward.**
   `SSMScanOp` is `(a0,b0) . (a1,b1) -> (a1*a0, a1*b0 + b1)`
   (`selective_scan_common.h:141-145`) and the backward runs it through
   `BlockReverseScan`. The backward of a first-order linear recurrence is a
   first-order linear recurrence run backwards. That is the design and it is
   right.
2. **The off-by-one on `da`.** Section 2.3 check three. Their explicit shift,
   `thread_reverse_data[i-1].x = delta_a_exp`, is the correct semantics and is
   the thing a from-scratch implementation gets wrong.
3. **`du` seeded with the D-skip path and then accumulated over the state
   index.** `du_vals[i] = D_val * dout_vals[i]` at line 217 and
   `du_vals[i] += ddelta_u * delta_vals[i]` inside the state loop. The join
   at `du` is real and its existence is theirs. Our T6 pins its ORDER, which
   theirs does not need to because theirs is already a fixed sequence.
4. **The two-term `ddelta`.** `ddelta_vals[i] += ddelta_u * u + dx * A * a`.
   Both paths are required and a backward with only the `B` path is a
   plausible wrong answer.
5. **The softplus derivative's guard and its operand.** `delta_val <= 20`
   evaluated on the PRE-softplus biased value, reloaded from `delta_ptr` and
   re-biased rather than inverted from the post-softplus value.
6. **Recomputation instead of storage where the recomputation is bit
   identical.** `checkpoint_lvl >= 1` drops `conv1d_out` and `delta` in the
   forward and recomputes them in the backward with the SAME kernels. That is
   sound and this lane adopts it, with one added rule that upstream violates
   and 6.2 names.
7. **Dropping the last state's gradient.** Section 2.4.

### 6.2 What upstream does that we must NOT copy

1. **Five float `atomicAdd` calls.** `dA` at line 483, `dB` and `dC` at 486
   and 491 and at 306 and 307 for the variable case, `dD` at 475,
   `ddelta_bias` at 480. Arrival order, therefore not reproducible run to run
   on ONE device, never mind three. This is the single largest reason
   upstream's backward cannot have the property this lane is trying to build.
2. **A launch shape chosen from the sequence length, and a different one per
   vendor.** `selective_scan_bwd_cuda` at lines 537-559 selects
   `(kNThreads, kNItems)` from `params.seqlen` through five branches, and the
   `#else` arm under `USE_ROCM` uses a DIFFERENT table. `cub::BlockScan`,
   `BlockReverseScan` and `BlockReduce` all take `kNThreads` as a template
   parameter, so the fold shape follows. Their backward is not sequence
   invariant, not launch invariant and not cross-vendor invariant, by
   construction and on purpose, for occupancy.
3. **`cub::BlockScan<..., BLOCK_SCAN_RAKING>` and `cub::BlockReduce` as the
   fold.** CUB is OPEN and therefore a port candidate under VENDOR_LIBS.md's
   FOLLOW-THEIR-DISPATCH banner, so the objection is not that we may not read
   it. The objection is that a raking scan's fold topology is a function of
   the block width, and a block count is a summation order.
4. **`a = thread_data[i].y - delta*u*B` to recover `da * h[t-1]`.** T2.
   Unbounded cancellation, undocumented by them, refused by us and MEASURED
   by gate MB9 rather than merely asserted.
5. **The `silu(z)` recomputation inconsistency, and this is the sharpest
   one.** The forward kernel computes the gate as
   `out_vals[r][i] *= z_val / (1 + expf(-z_val))`
   (`selective_scan_fwd_kernel.cuh:298`). The backward recomputes the SAME
   quantity as
   `z_sigmoid_val = 1.0f/(1.0f+expf(-z_val)); z_silu_vals[i] = z_val * z_sigmoid_val;`
   (`selective_scan_bwd_kernel.cuh:193-194`), then multiplies `dout` by it.
   **A single quotient and a reciprocal-then-product are not the same float.**
   So upstream's own backward multiplies by a `silu(z)` that its own forward
   never produced. This repository has already measured that exact difference
   on the forward side. `SAB_S12_MUL_SIGMOID` is `z * sigmoid(z)` in place of
   the pinned quotient and IDENTITY_PATHS row 55 records it biting `gate.out`
   on 15 of 64 cells. Our B4 calls `identical_silu`, full stop.
6. **`exp2f(delta * A * M_LOG2E)` for `delta_a_exp`.** Both their forward and
   their backward do it, so they are self-consistent, but contract seam S6
   pins `identical_exp(delta * A)` and DEVIATION 722 already refused the
   exp2 substitution as a different function with an extra rounding. The
   backward's recomputed `da` must be OUR forward's `da`, or the recomputation
   is not a recomputation. `SAB_S5_EXP2` already exists on the forward side
   and the backward inherits the requirement rather than restating it.

**The general rule this section produces, and it is worth promoting.**
*A recomputed forward quantity must be spelled with the same function the
forward used, not with an algebraically equal one.* Upstream breaks it once,
at `silu(z)`, and that single break makes their gradient inconsistent with
their own output. Every recomputation in this lane, `da`, `dbb`, `silu(z)`,
`silu(conv)`, `delta` and `conv`, calls the forward's own function.

---

## 7. The gates, the sabotages and the negative controls

Specified, not written. A pin with no fixture that separates it from the
unpinned spelling is a belief, and **an arm that has never been shown to fail
is an arm nobody has tested.** Every gate below names what it asserts, what
sabotage must break it, and what makes it VACUOUS.

**Per stage or not at all.** Section 3.4's T8 is why. The backward's
`residual.out` is `dx`, and the forward lane's absorption finding says an
output-only comparison of `dx` will call a thirteen-stage divergence inert.
The card of gate MB10 is the instrument, not a convenience.

### 7.0 The backward card's stages

Mirroring contract section 7's ordering discipline, upstream to downstream in
the backward's own direction.

    bwd.dres        [M, dm]          the input gradient, as given
    bwd.dg          [M, di]          B2
    bwd.dsk         [M, di]          B5
    bwd.dz          [M, di]          B8
    bwd.dh          [M, di, N]       B13, the reverse recurrence
    bwd.dCm         [M, N]           B15
    bwd.dBm         [M, N]           B16
    bwd.du_s        [M, di]          B17
    bwd.ddelta      [M, di]          B18
    bwd.ddtp        [M, di]          B22
    bwd.du          [M, di]          B29, the three way join
    bwd.dconv       [M, di]          B30
    bwd.dhin        [M, di]          B31
    bwd.dnrm        [M, dm]          B35
    bwd.drstd       [M]              B40
    bwd.dx          [M, dm]          B42, the absorption site
    bwd.dW_in       [2di, dm]        B36
    bwd.dW_x        [r+2N, di]       B28
    bwd.dW_dt       [di, r]          B25
    bwd.dW_out      [dm, di]         B3
    bwd.dA_log      [di, N]          B21
    bwd.dD          [di]             B12
    bwd.db_dt       [di]             B23
    bwd.dcw         [di, K]          B32
    bwd.dcb         [di]             B33
    bwd.dw_norm     [dm]             B39

Twenty six stages. Sixteen activation gradients and ten parameter gradients,
and the split is exactly the split section 4.1 draws, which is deliberate so
that a gate can assert a clause over one group and not the other.

### 7.1 The sabotage switches

| switch | breaks | predicted first moved stage | predicted INERT when |
|---|---|---|---|
| `SAB_BWD_S9B_FORWARD` | `dh` walked ascending in `t` | `bwd.dh` | `L == 1` |
| `SAB_BWD_S9B_DA_OFFSET` | `da[t]` used where `da[t+1]` belongs | `bwd.dh` | `L == 1`, or all `delta` equal across `t` |
| `SAB_BWD_S9B_UNFUSED` | T1's `fma` split into mul then add | `bwd.dh` | `L == 1` |
| `SAB_BWD_H_SUBTRACT` | upstream's `h[t] - dbu[t]` for `h[t-1]` | `bwd.ddelta` | `h[t-1] == 0`, i.e. `t == 0` only |
| `SAB_BWD_S12_MUL_SIGMOID` | `silu(z)` as `z * sig(z)`, upstream's | `bwd.dsk` | `z == 0` |
| `SAB_BWD_S14_DIVISION` | softplus' as upstream's single division | `bwd.ddtp` | `biased` outside `[8, 14]` |
| `SAB_BWD_S10_N_DESCENDING` | `du_s` and `ddelta` folded `n` descending | `bwd.du_s` | `N == 1`, or fewer than 2 nonzero terms |
| `SAB_BWD_DA_ASCENDING` | T4's `t` fold reversed | `bwd.dA_log` | `L == 1` |
| `SAB_BWD_JOIN_ORDER` | T6's three way join permuted | `bwd.du` | one contribution dominates |
| `SAB_BWD_S1B_FOLD_DESCENDING` | `drstd` folded `j` descending | `bwd.drstd` | `dm == 1` |
| `SAB_BWD_S13_TAPS_REVERSED` | B31's correlation index reversed | `bwd.dhin` | conv weights symmetric in `k` |
| `SAB_BWD_DBDC_ATOMIC` | `dBm`, `dCm` by float atomicAdd | **MB4, not MB3** | `d_inner == 1` |
| `SAB_BWD_RSTD3_ASSOC` | `rstd^3` right associated | `bwd.dx` | `rstd` a power of two |
| `SAB_BWD_PARAM_ATOMIC` | T5's `b` fold by float atomicAdd | **MB4, not MB3** | `B == 1` |

**The INERT column is the point of the table.** Every one of those entries is
a prediction of a fixture that would make the arm bitwise inert, and the
lesson that produced the column is the mamba forward lane's own. Its
`adv_softplus_guard` corpus case was chosen to test the S14 threshold, had
zero inputs in the distinguishing band, and produced 23 of 23 byte-identical
dumps under `S14_THRESHOLD_10`. **The arm ran, reached its branch, and moved
nothing.** Each gate below must assert its arm's predicted cell count, not
merely that the arm ran.

Note the two rows whose failure lands on MB4 rather than MB3. An atomic
accumulation is bit identical to a pinned fold on any run where the arrival
order happens to match, so a single-launch oracle comparison can pass with
the atomic in place. Only a repeat-launch or a launch-geometry gate can see
it. That is a different failure MODE and is worth demonstrating on purpose.

### 7.2 MB1, `check_backward_routing_is_the_table` (HOST ONLY)

Asserts that `mamba_backward.mojo`'s fourteen gemm-routed calls return the
shapes section 3.2 names, with `m`, `n` and `k` pairwise distinct at every
call so a transposed router cannot pass. Asserts separately that each
returned `(m', n')` equals the target buffer's shape computed independently
from the forward dimensions.

Cheap, instant, needs no device, and it is the gate that catches the class of
error the gemm backward lane found most likely, which is a router that
assumes `dC` is always the left operand. Three of the six gemm routings put
it on the right.

Sabotages that must fail. `SAB_BWD_UNTRANSPOSED` and `SAB_BWD_OPERAND_ORDER`
inherited from `gemm_backward.mojo`, invoked THROUGH this lane's entry points
rather than through the gemm lane's, which is the proof that this path
actually reaches the gemm contract's arithmetic and not some other path that
happens to agree.

### 7.3 MB2, `check_backward_is_the_derivative` (HOST ONLY)

**This gate answers a different question from every other gate here.** Bit
identity says the answer is the same on three boxes. It does not say the
answer is the RIGHT derivative, and a transposed conv backward is bit
identical on three boxes.

**The gemm backward lane got an exact, tolerance-free version of this gate
and this lane cannot have one.** A GEMM is bilinear, so a central difference
at `h = 1` on an integer fixture is EXACTLY the derivative and can be asserted
bitwise. The Mamba block contains `exp`, `softplus`, `silu` and `rsqrt`, so no
step size makes the central difference exact. Pretending otherwise would be
the worst kind of green check. The honest replacement has two halves.

- **The bilinear sub-seams get the exact treatment.** The four matmuls, the
  conv tap chain, the D skip, the gate product and the residual add are each
  bilinear or affine in their own operands with the others held fixed, so a
  central difference on an integer fixture at `h = 1` is exact and is asserted
  BITWISE. This covers B2, B3, B24, B25, B27, B28, B31, B32, B33, B35, B36,
  B10, B12 and B42.
- **The transcendental seams get a Float64 directional derivative against
  `mamba_block_ref64`, with a per-stage tolerance calibrated from that
  function's own self test.** Never a fixed epsilon. The corpus lesson is
  explicit here. `adv_gate_saturation` passed only at `rtol=1e-3` and **the
  tolerance was the defect, not the block**, because torch's own FP32 failed
  the same four stages at `1e-7` where values reached 7.25e8. Tolerance is
  calibrated PER CASE or the gate is measuring the tolerance.

**Vacuity guard, required.** The check must REFUSE to pass unless `dm`, `di`,
`r` and `L` are pairwise distinct where they can be, the conv weights are
asymmetric in `k`, and `delta` varies across `t`. At a symmetric fixture a
reversed tap order and an off-by-one on `da` both compute the same numbers.

**Self test, and it is the interesting half.** The gate should DEMONSTRATE its
own fixture requirement rather than assert it. Run `SAB_BWD_S13_TAPS_REVERSED`
on a `k`-symmetric conv weight, show that it PASSES, and print that as the
reason the constraint exists. The gemm backward plan specifies the same move
and this lane's own forward history is the precedent.

### 7.4 MB3, `check_backward_device_matches_oracle` (DEVICE)

All twenty six stages, per cell, bitwise, against a host backward oracle
written the way `mamba_block_oracle` is written, through the same seam
functions. Shapes `B` in `{1, 2, 3}`, `L` in `{1, 4, 16, 64, 257}`, `d_model`
in `{8, 16}`, both numeric modes, matching the forward's gate coverage
(contract section 3) so that a backward divergence can be compared against a
forward run at the same shape.

**`L = 1` must be present and must be marked.** It is the shape at which four
of the fourteen sabotages are predicted inert, and a gate whose sabotage
ledger does not record that will report four false negatives.

### 7.5 MB4, `check_backward_is_launch_invariant` (DEVICE)

One byte pattern across block widths 32, 64, 128 and 256, across two grid
paddings, across eight repeat launches with fresh dispatches, and across a
poisoned workspace. This is the gate that section 4.3 exists to make
assertable, and it is the ONLY gate that can see `SAB_BWD_DBDC_ATOMIC` and
`SAB_BWD_PARAM_ATOMIC`.

Sabotages that must fail. Those two, plus `SAB_BWD_DA_ASCENDING` and
`SAB_BWD_JOIN_ORDER`, which move a bit only when the accumulation order moves.

### 7.6 MB5, `check_backward_activation_gradients_are_batch_invariant` (DEVICE)

Section 4.1's positive half. A sequence's sixteen ACTIVATION gradient stages
must be bit identical whether it shares the launch with zero, one or two
others, from ONE sliced `dres` so that row zero's input is identical by
construction.

**Negative control, load bearing.** Rows 0 and 1 of the `B = 2` run must
DIFFER on every batched stage. Without it a broken row slicer makes this
clause pass for ever on every vendor. The forward lane's clause (c) carries
exactly this control and IDENTITY_PATHS row 55 records it firing on 15 of 15
batched stages.

**The ten PARAMETER stages are explicitly EXCLUDED from this gate and the
exclusion is asserted rather than implied.** The gate must FAIL if a
parameter stage is bit identical across batch compositions, because that
would mean the parameter gradient is not accumulating the extra sequences.

### 7.7 MB6, `check_parameter_gradients_are_not_microbatch_invariant` (DEVICE)

A gate on a NEGATIVE property, so section 4.1 is measured rather than
asserted. Three arms.

- At a fixed `(B, L)` the ten parameter stages are bitwise identical across
  launches, plans and repeat runs. The positive half.
- `dParam` at `T` tokens DIFFERS from `dParam(first T/2) + dParam(second T/2)`.
  The negative half, and the point.
- **Vacuity guard, required.** The two arms must first be shown to AGREE in
  exact arithmetic on an integer fixture, so the difference then measured on a
  rounding-sensitive fixture is a rounding-order difference and not a bug in
  the split. If they agree on the rounding-sensitive fixture the gate raises
  VACUOUS rather than passing.

Record the measured difference as a number, cells moved and ulps, for each of
the ten. That number is the blast radius of a microbatch schedule change and
somebody will ask for it. Expect the six non-GEMM parameters to move by more
than the four GEMM ones, because their folds are shorter and their terms are
more nearly equal in magnitude.

### 7.8 MB7, `check_the_recurrence_is_actually_reversed` (DEVICE)

The structural gate for T1, and it exists because MB3 compares against an
oracle that a shared misconception would corrupt on both sides.

Two arms, both on a fixture with `L >= 8` and `delta` varying across `t`.

- Perturbing `dres` at token `s` MUST move `bwd.dh` at every `t <= s` and MUST
  NOT move it at any `t > s`. Assert both directions and print the exact
  count.
- Perturbing `dres` at the LAST token must move `bwd.dh` at every `t`, and
  perturbing it at token `0` must move `bwd.dh` at exactly one `t`.

A forward-direction implementation passes the first half of arm one and fails
the second, which is why both halves are asserted. **The gate must refuse to
run at `L == 1`**, where the two directions are the same computation.

### 7.9 MB8, `record_the_absorption_at_the_backward_residual` (DEVICE)

A RECORDING gate, not an asserting one, on the T8 join.

Build a fixture where `dx_norm` is of order 1e-3 against a `dres` of order 1,
run every sabotage arm, and record for each one how many stages moved, how
many cells of `bwd.dnrm` moved, and how many cells of `bwd.dx` moved. The
prediction, stated so it can be wrong, is that arms which move ten or more
intermediate stages will leave `bwd.dx` bit identical on most cells.

The forward lane measured exactly this. Thirteen stages moved, `out_proj.out`
differed on 23 of 64 cells, `residual.out` was still bit identical. If the
backward reproduces it, the per-stage card is not a convenience, it is the
only instrument that can see the clause at all, and that sentence belongs in
the paper.

### 7.10 MB9, `price_the_upstream_h_recovery` (HOST or DEVICE)

The gate that turns section 6.2 item 4 from an accusation into a number.
Compute `da * h[t-1]` both ways on the corpus fixtures, the checkpointed read
and upstream's `h[t] - dbu[t]`, and record the maximum and median ulp
distance, per case, along with the fraction of `(t, d, n)` cells where
`|da*h[t-1]| < |dbu[t]| / 2^10`, which is the cancellation band.

If the distance is zero everywhere, say so and downgrade T2's refusal to a
preference. **A lane that only confirms its own suspicion is a lane to
re-audit**, and this is the cheapest place in the whole plan to be wrong in
public.

### 7.11 MB10, the backward card and the three vendor leg

`bench/mamba_bwd_card_main.mojo`, twenty six per-stage hashes in the section
7.0 order, following the forward card's pattern and DEVIATION 970's lesson,
which is that `mamba_check.mojo` wrote its card to a hardcoded `TRACE_PATH`
and ignored `MOJOLEARN_IDENTITY_TRACE`, so the Apple column had a GREEN CHECK
AND NO CARD and the gate could not tell the difference. **The backward card
must read the environment variable and must FAIL when it cannot write.**

Nothing here is a cross-vendor claim until the card is byte identical on
Apple, NVIDIA and AMD. The standing lesson is that two columns agreeing closes
nothing, and this lane starts from a forward that has TWO columns and no AMD.

### 7.12 The independent cross check, which is a cross-lane request

Every gate above compares us against us. The forward lane's strongest
non-self-referential result came from `mamba/corpus/`, torch Float64 through
`selective_scan_ref` in the HF block order, and it found a real disagreement,
the `dt_proj.bias` naming clash, that no internal gate could have found.

The backward needs the same instrument, and it does not exist.
`/Users/andrewhendel/CascadeProjects/mojolearn/mamba/corpus/gen_corpus.py`
would have to dump `torch.autograd.grad` outputs in Float64 for the same
sixteen cases. That file belongs to another lane. The exact request is in the
report and this lane may not make the edit.

---

## 8. The phase ladder

Phases A and B are host only and safe beside anything. Everything from C
touches the device and goes through `tools/with_build_lock.sh`.

| phase | content | depends on | size |
|---|---|---|---|
| **A** | this document plus `mamba_backward.mojo` | nothing | DONE, ungated, uncompiled |
| **B** | the host backward oracle, twenty six stages, beside `mamba_block_oracle` | A | moderate. It is the normative answer and everything compares to it |
| **C** | MB1 and MB2, host only, plus the two self tests | B | small |
| **D** | the device backward for the routed and the c-elementwise operations only, and MB3 restricted to the stages they produce | B | moderate |
| **E** | T1, T2 and T4, the reverse recurrence kernel, plus MB7 and MB9 | D | moderate. One kernel, one thread per `(b,d)`, no cross-thread float |
| **F** | T3, `dBm` and `dCm`, plus the `h` and `dh` buffers | E | **large, and section 5.3 is why** |
| **G** | T5, the parameter folds, plus the ten parameter stages | F | moderate |
| **H** | MB4, MB5, MB6, MB8, the invariance and recording gates | G | moderate. MB6 needs a fixture designed to separate |
| **I** | MB10, the backward card | G | small |
| **J** | the three vendor leg, and it needs an AMD FORWARD column first | H and I | one rented hour per vendor, guard armed FIRST |
| **K** | the blocked T3 for model scale | J | large, unpriced, section 5.3 |

**Phase J has a dependency the gemm lane did not have.** The mamba forward is
green on two vendors, not three. A backward leg on AMD is a first mamba
column on AMD, so the forward's own AMD leg has to run first or the backward
leg is comparing a backward against a forward that has never been checked on
that box.

B, C and D are independently parallelizable once A exists. E through G are
serial, because each one's outputs are the next one's inputs, which is the
same reason section 4 of the gemm plan orders its table.

---

## 9. Deviation block

Numbers 1070 to 1089 are this lane's. 1070 to 1078 are SPENT in this document
and in `mamba/original/mamba_backward.mojo`. The rest are reserved against
the phase they belong to.

| number | what | if it is wrong | state |
|---|---|---|---|
| 1070 | T1, the reverse recurrence, direction, `+0.0` seed, FUSED | every `dh` below `t = L-2` and nine card stages | SPENT |
| 1071 | T2, the explicit `h[t-1]` checkpoint, refusing upstream's subtraction | `N`-times memory spent for nothing, if MB9 records zero ulps | SPENT |
| 1072 | T3, `dBm` and `dCm` as per-token `(1, N, di)` v1 contractions over materialized `h` and `dh` | the whole seam. It is the hardest one and 5.3 is the price | SPENT |
| 1073 | T4, the `dA` fold DESCENDING in `t`, against every other fold in both profiles | `dA_log`, `bwd.dA_log`, and a consistency argument lost | SPENT |
| 1074 | T5, the private-slot batch fold with no atomic anywhere | five parameter stages, and MB4 is the only gate that sees it | SPENT |
| 1075 | T6, the three way `du` join in D-then-scan-then-xproj order | `bwd.du` and every stage after it, and the arm may be INERT | SPENT |
| 1076 | T7, the RMSNorm backward closed form and `rstd^3`'s association | `bwd.dx`, absorbed at T8 on most fixtures | SPENT |
| 1077 | T8, the residual join, `dres` left, one flushed add | nothing visible, which is exactly the hazard | SPENT |
| 1078 | B22, softplus' as a multiply by `identical_sigmoid`, refusing upstream's single division | `bwd.ddtp` and everything upstream, only in the `[8,14]` band | SPENT |
| 1079 | the fourteen gemm routings and their workspace sizing | out of bounds writes, which a small shape hides | SPENT |
| 1080 | the six token-axis reductions as ones-vector v1 GEMMs, and the pre-product buffers they need | six parameter stages | SPENT |
| 1081 | the host backward oracle | phase B | RESERVED |
| 1082 | MB1 and MB2, and the bilinear-exact plus Float64-tolerance split | phase C | RESERVED |
| 1083 | the routed and elementwise device path, MB3 | phase D | RESERVED |
| 1084 | the reverse recurrence kernel, MB7, MB9 | phase E | RESERVED |
| 1085 | the `h` and `dh` materialization and the T3 kernel | phase F | RESERVED |
| 1086 | the parameter fold kernels, MB4, MB5, MB6, MB8 | phases G and H | RESERVED |
| 1087 | the backward card, MB10 | phase I | RESERVED |
| 1088 | the blocked T3 for model scale | phase K | RESERVED |
| 1089 | unallocated | | RESERVED |

Read the block with the pattern that actually answers the question, since a
range is reported by its first number only.

    grep -rhoE "DEVIATIONS? 10[7-8][0-9](-10?[0-9]+)?" . | sort -u

---

## 10. Open questions, and things a reader should not assume

- **Nothing here has run.** Not one gate, not one fixture, not one device
  call, not one compile. `mamba/original/mamba_backward.mojo` mirrors
  `gemm_backward.mojo`'s spellings but has never been through the compiler,
  and the five-element `Tuple` returns and the `comptime if` early returns are
  the two things most likely to need adjustment.
- **The largest unpriced item is section 5.3.** T3's memory makes the declared
  spelling gate-scale. The blocked variant is phase K and its tiling must be
  `contract_partition(d_inner)`'s leaf boundary, never a VRAM budget.
- **`causal_conv1d` is not on disk.** Every conv-backward statement is
  INFERRED from S13 and from the index arithmetic. `dcw`'s fold order and
  `dcb`'s bias-gradient spelling upstream are UNREAD.
- **The backward has no determinate torch reference.** The forward could say
  "the reference's order is the profile's" because `selective_scan_ref` is
  readable line by line. Its backward is `torch.autograd` over einsums, and
  einsum's backward grouping depends on the contraction path the optimizer
  picks, so there is no fixed order to transcribe. This lane's answer is to
  derive every backward association by the chain rule FROM THE PINNED FORWARD,
  reusing the forward's own rounded intermediates, so the backward's order is
  a theorem of the forward's rather than a second source of truth. The CUDA
  kernel's groupings differ from that in four places, `du`, `ddelta`'s A path,
  `dB` and `dA`, and they are the sabotage arms rather than the profile.
- **The `-0.0` question is inherited and unexamined for the backward.** The
  forward's `pinned_mul` preserves a negative zero product and is SETTLED
  independently, 125 negative zeros surviving S3, S4 and S12. A gradient is a
  place where signed zeros are more common than in a forward pass, because a
  zero `dres` cell is normal. No fold in this lane takes a `min`, `max`,
  `argmin` or `argmax`, so row 13's selection hazard has no site here either,
  but an OPTIMIZER downstream is exactly such a consumer and a `max(grad, 0)`
  clamp meets row 39 in full. Clamps must be spelled value-first. Named here
  so it is not discovered later.
- **`refuse_nonfinite` has no backward counterpart yet.** The forward refuses
  a NaN or infinity in any input BY BITS before any recorded stage. `dres`
  needs the same refusal for the same reason, NaN payloads are vendor-shaped,
  and it is not written.
- **Two claims in files this lane may not edit are stale and are REPORTED
  rather than fixed.**
  1. `gemm/IDENTICAL_BACKWARD_PLAN.md` section 4 row T5 says "**GELU is
     REFUSE today**: `original/numerics.mojo` has no `portable_erff` and no
     `portable_tanhf`." Both exist, at `original/numerics.mojo:1672` and
     `:1465`, along with `portable_gelu_erf` at `:1722`, `portable_gelu_tanh`
     at `:1778`, `identical_erf` at `:2009` and both `identical_gelu_*` at
     `:2023` and `:2039`, added by the transformer-block lane as DEVIATIONS
     820-825. The row's conclusion is wrong and its move should be PIN.
  2. `mamba/IDENTICAL_MAMBA_CONTRACT.md` section 9 says "Nothing
     cross-vendor until the leg runs." The leg ran. IDENTITY_PATHS row 55
     records leg 12 at `1b7c916`, Apple M4 against NVIDIA RTX 4090, 17 of 17
     stages bit identical. What is still true is a DIFFERENT sentence, that
     the second column runs ONE fixture and that AMD has no mamba column at
     all. The contract is a certified artifact and belongs to its owner.
- **The forward's clause (c) is stated over rows and the backward needs it
  stated over two groups.** Contract section 8(c) says "a row's bits identical
  whether its sequence shares the launch with 0, 1 or 2 others". Applied
  literally to a backward it is false for ten of twenty six stages and true
  for sixteen, and the reason is section 4.1, not a defect. Any backward
  profile document must split the clause. This is a drafting note for whoever
  writes `mojolearn.identical.mamba1.bwd.fp32.v1`, and this document is not
  that contract.
