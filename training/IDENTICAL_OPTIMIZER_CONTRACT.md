# The IDENTICAL FP32 optimizer-step contract

# PROFILE `mojolearn.identical.optimizer.fp32.v1`

## STATUS

**COMPILED, RUN AND CARDED ON TWO COLUMNS, APPLE AND AMD. NO NVIDIA.**

- **First execution, `ecd1a436`, and the run at `b90f52ab`.** The gate file
  this contract once said was not in the lane's write set was written and ran.
  **Clause (f) MEASURED a real defect**: with clipping off, a non-finite
  planted in a PARAMETER reached `param.out`, because the only device-side
  refusal lived in `identical_clip_grad_norm`, which does not run when
  clipping is off and covered `grad` alone. **DEVIATION 1496** moved the
  refusal to the first statement of `identical_optimizer_step` and made it
  call the oracle's own function. **Two arms could NOT be made non-vacuous
  and that is recorded as DEVIATIONS 1493 and 1613 rather than papered over.**
- **The two-column legs, 2026-08-28.** `training-optimizer.identical.card`, 18
  records, md5 `97d160b0`, **byte-identical on an Apple M4 and an AMD
  MI325X.** **Clause (a) PASS on both, 33 cases across every step, on all
  382,822 compared cells. Clause (e) ran on both**, host only, always on.
  **Clauses (b), (c), (d) and (f) SKIPPED on both.**

What is NOT closed, from the run's own SCOPE line. NO NVIDIA LEG. An
INDEPENDENT reference, because section 16's corpus does not exist, so every
clause is our device against our oracle and **seven of the twenty-three stages
are the SAME HOST CODE on both sides**, since `device_step_scalars` calls the
oracle's own `step_scalars` on the clean path. **PyTorch parity, which 5.2 and
5.3 make IMPOSSIBLE by construction and which nothing here should be read as
claiming.** The GEOMETRY half of clause (b), because `OPT_TPB` is a comptime
literal (DEVIATION 1479). The thousand-step case unless
`MOJOLEARN_OPT_CHECK_T1000` was set. FAST mode. And clause (d)'s control 2 IS
the `OPT_SAB_RESUME_REINIT` arm (DEVIATION 1473), so a leg that skips clause
(d) has one fewer ARM and not merely one fewer clause; likewise a leg that
skips (f) has not looked at the device-side refusal once, and (f) carries the
MEASURED finding that the DEVICE path refuses only `clip.total_norm` and does
not refuse at all when clipping is off (DEVIATION 1478).

**The v1 GEMM this contract DELEGATES its reductions to HAS run on three
vendors at leg 11 (`144aa5b`). That measurement is the GEMM's, not this
lane's.**

Also owed. No `pixi.toml` task, so the gate runs by path. No `IDENTITY_PATHS`
rows for the optimizer step or the global-norm clip. No `DERIVATION_MAP.tsv`
or `NOT_IMPLEMENTED.tsv` entries for `training/`.

DEVIATIONS 1170 through 1189 are this lane's.

**Companion documents, and this contract is not readable without the first.**
`gemm/IDENTICAL_FP32_CONTRACT.md`, because **this contract DELEGATES every
reduction it performs to that profile**, so its sections 5, 6, 7 and 9 are
load bearing here. `archive/plans/gemm/IDENTICAL_BACKWARD_PLAN.md`, whose row T9 is this
lane. `IDENTITY_PATHS.md` rows 9, 10, 12, 13, 39, 49 and 50.

---

## 0. What is in scope, and what the inventory came from

### 0.1 The three algorithms

| kind | reference | what it is here |
|---|---|---|
| `OPT_SGD` | `torch.optim.SGD` | plain SGD, momentum, dampening, Nesterov, coupled L2 decay |
| `OPT_ADAM` | `torch.optim.Adam` | Adam with COUPLED decay, folded into the gradient |
| `OPT_ADAMW` | `torch.optim.AdamW` | Adam with DECOUPLED decay, applied to the parameter |
| `clip_grad_norm` | `torch.nn.utils.clip_grad_norm_`, `norm_type = 2` | a two-level global L2 norm and one unconditional rescale. **Where all the difficulty is** |

FP32 master weights, FP32 gradients, FP32 optimizer state. No second precision
anywhere.

### 0.2 Excluded, each with the reason

**Excluded, each with the reason.** `amsgrad`, because `v_max = max(v_max, v)`
is row 13's selection hazard plus row 39's NaN canonicalization; the spelling
it would take exists (`identical_fmax`, DEVIATION 824) so this is a small
addition, and it is excluded because an unreached branch is not a working
guard. `foreach` and `fused` multi-tensor apply, because a multi-tensor kernel
changes which elements share a block, which under section 7 cannot reach the
arithmetic, **but "it cannot reach the arithmetic" is exactly the kind of
claim that has to be gated rather than asserted**, and that gate is owed.
`maximize`. Learning-rate schedules that carry state, since a schedule is
admitted only as a pure function of the integer step index. Mixed precision,
loss scaling and BF16/FP16 master weights, by gemm contract 0.5 and 1. Sharded
or multi-GPU optimizers, because an all-reduce is a summation order chosen at
runtime from the topology; **a claim about a run under this profile must
contain the words "single GPU".** Sparse and `index_add`-shaped updates, which
are T10 and a different lane. And the backward pass itself.

### 0.3 The two references, and which one is normative

**The two references.** `optimizer_oracle.mojo::optimizer_step_oracle` is
**NORMATIVE**, host, scalar, single threaded, built from
`checks/numerics.mojo`'s actual helpers rather than local copies so it cannot
drift into a second opinion. **PyTorch is the DESIGN reference and is not a
bit reference**, and 5.1 and 5.2 make bit parity impossible by construction.

**CITATION HEALTH WARNING.** There is no PyTorch checkout in
`/Users/andrewhendel/CascadeProjects/upstream/`, so every PyTorch spelling
cited below is cited FROM MEMORY of the documented pseudocode and of
`torch/optim/adam.py`'s single-tensor path, which violates
`read-their-source-against-ours`. Where a clause depends on which of two
PyTorch spellings is real, the clause states BOTH and names its own answer as
the profile's, which is the construction that survives the citation being
wrong.

---

## 1. Dtypes, hyperparameters, and the bit-pattern rule

Every value is `Float32`. Parameters, gradients, the momentum buffer, `m`,
`v`, every host scalar derived per step, and every hyperparameter.

**Hyperparameters are BIT PATTERNS, not decimal strings.** A card records
`lr`, `beta1`, `beta2`, `eps`, `weight_decay`, `momentum`, `dampening` and
`max_norm` as eight-hex-digit patterns, and two runs are comparable only when
those patterns are equal. It matters more than it looks: `0.999` narrowed from
a float64 parse is `0x3F7FBE77` and so is a direct `Float32` literal, but the
DERIVED quantity `1 - beta2` is not the same number by the two routes, and 5.2
measures the gap.

| name | decimal | bits |
|---|---|---|
| `beta1` | 0.9 | `0x3F666666` |
| `beta2` | 0.999 | `0x3F7FBE77` |
| `eps` | 1e-8 | `0x322BCC77` |
| `CLIP_EPS` | 1e-6 | `0x358637BD` |

**`CLIP_EPS` is fixed by the profile and is not caller supplied**, because
`clip_grad_norm_` hardcodes `1e-6` and a caller who could change it could
change the bits of every parameter in the model through one scalar.

**There is no float64 anywhere, on the device or on the host.** On the device
that is a hardware fact. On the HOST it is a deliberate choice and 5.1 is the
reason.

## 2. Why an elementwise step is not automatically identical

**Why an elementwise step is not automatically identical.** The update is
elementwise, so no float crosses a thread boundary and there is no summation
order in it. Everything hard is in the parts that are NOT the update. The
gradient norm is a fold across many tensors of many lengths whose partition,
tree and cross-tensor order are three separate free choices, and the third is
the one a reader skips. `sqrt` lowers to an APPROXIMATE PTX sqrt on NVIDIA,
180,714 of 2^20 hashed patterns off by one ulp with 176,577 on NORMAL inputs
(DEVIATION 258), which was the single NVIDIA miss in an otherwise closed lane
(DEVIATION 550), and Adam takes a square root per element per step. An `rsqrt`
intrinsic is a per-vendor approximation. `m / denom` and `m * (1/denom)` are
different bits whenever `denom` is not a power of two. Bias correction by
repeated multiplication and by a general `pow` are different numbers and the
difference GROWS with `t`. Adam's `v` is a running average of squared
gradients and goes very small; `g` at 1e-25 gives `g*g` at 1e-50, not
representable as a normal `Float32` at all. And Adam and AdamW differ in
exactly the ORDER weight decay is applied in and in nothing else.

---

## 3. The gradient-norm clip, which is the hardest clause here

### 3.1 The reference's shape is TWO LEVELS, and that is load bearing

`clip_grad_norm_` does not compute one flat sum of squares. It computes a
per-tensor L2 norm, stacks those, and takes the L2 norm of THAT.

    norms      = [ vector_norm(g_j, 2) for j in parameters ]
    total_norm = vector_norm( stack(norms), 2 )
    coef       = clamp( max_norm / (total_norm + 1e-6), max = 1.0 )
    for j:  g_j *= coef

**In exact arithmetic the two-level and flat forms are the same number. In
`Float32` they are not**, because each per-tensor `sqrt` rounds and each
result is squared again inside the outer norm. **CLAUSE 3.1.**
`total_norm = sqrt( SUM_j ( sqrt(sumsq_j) )^2 )`, not `sqrt( SUM_j sumsq_j )`.
`COPY, DO NOT IMPROVE` at a place where the improvement is tempting and is one
line. Sabotage `OPT_SAB_CLIP_FLAT_NORM`, and **at `J = 1` the two forms agree
for most `s`, so the fixture must carry at least three tensors of different
lengths whose norms differ by several binades.**

### 3.2 The per-tensor sum of squares IS a v1 GEMM, and is not written here

**CLAUSE 3.2.** `sumsq_j = gemm_oracle( g_j, g_j, OP_NT, 1, 1, N )[0]`, and on
the device the same call through `identical_gemm`. Not a fold written in this
lane, not `pinned_block_sum`, not a `sum` helper. At `m = n = 1` and `OP_NT`
the operand accessors reduce to `a[p]` and `b[p]`, so the call really is the
sum of squares and not a shape that happens to be near one. Then
`norm_j = ftz( identical_sqrt( ftz(sumsq_j) ) )`.

Three things follow from delegating rather than writing. **The partition is
`contract_leaf_size(N)` and nothing else**, not the block count, the vendor,
the occupancy, or how many other tensors are in the launch. **The fold is v1's
fixed balanced tree over adjacent leaves with the odd tail carried**,
including the no-padding and no-stride-pairing clauses and their fixtures. And
**it inherits a MEASUREMENT**, the three-vendor v1 device card at leg 11; a
fold written fresh here would inherit nothing and would owe its own leg.

*What would make this pass while gating nothing.* Any fixture whose tensors
all have `N <= 128`, where `P == 1`, the tree has no arithmetic node, and the
v1 answer IS the serial ascending chain, so a hand-written serial fold passes.
**At least one tensor must have `N > 128`, and one should have an `N` that is
not a multiple of 128; `N = 300` gives `P = 3` with a 44-element last leaf**,
the gemm lane's own ragged fixture. Sabotage `OPT_SAB_CLIP_SERIAL_FOLD`.

### 3.3 The cross-tensor fold, and why PARAMETER ORDER is in the profile

**CLAUSE 3.3.** `total_sumsq = gemm_oracle( norms, norms, OP_NT, 1, 1, J )[0]`
and `total_norm = ftz( identical_sqrt( ftz(total_sumsq) ) )`, where `norms` is
indexed by `param_id` ASCENDING.

**`param_id` is part of the profile.** An integer assigned once when the model
is registered, written into the checkpoint. **Not** the iteration order of a
dictionary, not the order the optimizer happened to receive the tensors in,
not a device pointer order, and not a name sort that could change when a layer
is renamed. Two runs with the same parameters in a different `param_id` order
are two different numerical experiments and the card must show it.

**This is the clause most likely to be waved past, and the reason is that it
looks like bookkeeping. It is not. It is the cross-tensor summation order, and
a summation order is the thing this repository exists to pin.**

*What would make this pass while gating nothing.* A fixture with `J = 2`.
Reversing two elements swaps the two children of ONE tree node and `a + b`
equals `b + a` bitwise, so **a two-tensor fixture cannot see a parameter-order
sabotage at all.** `J >= 3` is required and `J = 5` is better, because `P = 5`
is the smallest `P` that carries twice. Sabotage `OPT_SAB_CLIP_PARAM_ORDER`.

### 3.4 The coefficient, the clamp, and the multiply that may not be skipped

**CLAUSE 3.4a.** `denom = ftz( total_norm + CLIP_EPS )` and
`coef = ftz( identical_div(max_norm, denom) )`, a true divide.

**CLAUSE 3.4b.** `coef_c = coef if coef < 1.0 else 1.0`, a compare-select
against the exact `1.0`, on a value section 8 has already established is
finite and non-negative, so the compare has one meaning on every column.

**CLAUSE 3.4c.** `g_j[i] = ftz( identical_mul( coef_c, ftz(g_j[i]) ) )` for
every element, **UNCONDITIONALLY, including when `coef_c` is exactly `1.0`.** A
"skip the rescale when no clipping is needed" optimization is FORBIDDEN.

The reason is precise, because the optimization looks obviously safe.
`identical_mul(1.0, x)` is `fma(1.0, x, -0.0)`, which returns `x` exactly for
every finite `x` including both signed zeros, so for a NORMAL gradient the
multiply and the skip agree. **They do NOT agree for a SUBNORMAL gradient**,
because the multiply's operand flush and result flush turn it into a signed
zero and the skip leaves the original bit pattern in the buffer.

And the honest half. **The difference is CARD VISIBLE and DOWNSTREAM INERT.**
The `clip.grad` stage hash differs between the two spellings; the optimizer's
own first act is `ftz` on the gradient load, so by the time the value reaches
`m` and `v` the two have converged. **So this clause protects the card and any
OTHER consumer of the gradient buffer, a logger, a second optimizer, a
gradient-statistics pass, and it does not protect the parameter update.** That
is `reached-but-inert` stated at the clause rather than discovered by whoever
writes the gate.

*What would make this pass while gating nothing.* Any fixture whose gradients
are all normal. **The fixture must plant a subnormal gradient cell, and it
must compare `clip.grad` and not only `param.out`.** Sabotage
`OPT_SAB_CLIP_SKIP_AT_ONE`.

### 3.5 What clipping costs the independence claim, stated rather than hidden

**Without clipping, one parameter's update bits are independent of every other
parameter in the model.** That is a real property and clause (c) gates it.
**With clipping, it is FALSE, by construction and by the reference's own
semantics**, since `coef_c` is a function of every gradient in the model. Not
a defect and not repairable; it is what a GLOBAL norm clip means. **So a
parameter-count-invariance gate must run with clipping OFF**, and with it ON
the claim is the narrower one, that the answer is a pure function of the
parameter registry and the gradients, and the registry is part of the run's
identity.

---

## 4. `sqrt`, `rsqrt`, and the division

**CLAUSE 4a. Every square root is `identical_sqrt`**, which under IDENTICAL is
`portable_sqrtf`, correctly rounded by construction and measured 0 mismatches
against a float64 reference over 2^20 patterns. Not defensive: `std.math.sqrt`
on NVIDIA is an approximate PTX sqrt, DEVIATION 258, and there are three square
roots in a clipped Adam step, at `norm_j`, at `total_norm` and at `sqrt(v)`
per element per step.

**CLAUSE 4b. `rsqrt` is never called, in either mode.** Not `std.math.rsqrt`,
not `identical_rsqrt`, not a Newton refinement. The intrinsic is a per-vendor
approximation, PTX `rsqrt.approx` at about 2 ulp. And `identical_rsqrt`, which
IS pinned, is `1 / portable_sqrtf(x)`, two correctly rounded operations, and
**DEVIATION 741 measured it off the correctly rounded `rsqrt` on 134,858 of
520,133 positive-normal lanes**. So even the pinned `rsqrt` is a DIFFERENT
NUMBER from `sqrt` then `div`, and the profile has to say which it means. It
means `sqrt` then `div`, because that is the shape 7.2's denominator has, with
an `eps` added between them. *A handful of round `v` values will pass a
sabotage here; the fixture needs at least a few thousand hashed `v`, on all
three columns before anything is concluded.* Sabotage `OPT_SAB_RSQRT`.

**CLAUSE 4c. Every division is `identical_div`, a true divide**, never a
reciprocal followed by a multiply. Three per step, `lr / bc1` and
`max_norm / denom` on the host and `m / denom` per element. `check-division`
characterized Apple's division as correctly rounded on the whole normal class
over 2^20 pairs, and `check-ieee-arith` measured div 0 wrong on the H100 and
the MI325X over its own patterns; that is not a certificate for the other two
columns and `portable_divf`'s docstring says so. *`x * (1/d)` is EXACT when
`d` is a power of two, so a fixture whose denominators are powers of two
passes vacuously.* Sabotage `OPT_SAB_RECIP_MUL`.

**CLAUSE 4d. `eps` is added OUTSIDE the square root**, to the bias-corrected
denominator. The alternative `sqrt(v_hat + eps)` is what several reference
implementations do and is a different number. *Whenever `v_hat` is much larger
than `eps^2` the two agree to the last bit, and an ordinary fixture has `v`
around 1e-4 against `eps^2` at 1e-16. **The fixture must plant `v` in the
1e-20 to 1e-12 band**, which a real run reaches on a dead unit and a synthetic
fixture never reaches by accident.* Sabotage `OPT_SAB_EPS_INSIDE_SQRT`.

---

## 5. Bias correction, and the step-count-invariance clause

### 5.1 `beta^t` is a pure function of the INTEGER `t`, by binary exponentiation

**CLAUSE 5.1.** `beta1^t` and `beta2^t` are computed on the host, once per
step per parameter group, by LSB-first binary exponentiation over
`identical_mul`.

    pow_int_f32(base, t):
        acc = 1.0 ; b = base ; e = t
        while e > 0:
            if (e & 1) != 0:  acc = ftz(identical_mul(acc, b))
            b = ftz(identical_mul(b, b))
            e = e >> 1
        return acc

**It uses only the profile's own arithmetic**, so the same function evaluated
on a DEVICE returns the same bits, and there is no float64. **It is not a
general `pow`**; `portable_powf` is `exp(p * log(x))` through two Cephes
polynomials, accurate to a few ulp and NOT exact even at `t = 1`, and `t` here
is an integer. **AND IT IS A PURE FUNCTION OF `t`, SO STEP-COUNT INVARIANCE IS
STRUCTURAL**: there is no running `beta_pow` state to checkpoint, nothing to
reconstruct on resume, and no way for a resumed run to disagree with a
continuous one. The gate then verifies what the construction promises rather
than being the only thing standing between the profile and a silent drift.

The alternative this refuses is the running product `b_t = b_{t-1} * beta`,
which an implementation reaches for because it is one multiply per step. It is
`t - 1` sequential roundings and a different number.

*PREDICTED, DERIVED OFF-REPOSITORY, NOT MEASURED.* In `Float32` with the
profile's flush, `pow_int_f32(beta, t)` and the running product **first differ
at `t = 7`**, for `beta = 0.9` and `beta = 0.999` alike. **A sabotage gate that
runs to `t = 4` is VACUOUS**, and `t = 1` through `t = 6` agree exactly, which
is precisely the range a hand-written fixture stops at. Run to at least
`t = 8` and ideally to `t = 1000`.

*Also PREDICTED.* With the flush, `beta1^t` reaches exactly `+0.0` at `t = 829`
for `beta1 = 0.9` and `beta2^t` at `t = 87,295` for `beta2 = 0.999`, after
which `bc1` and `bc2` are exactly `1.0`. That is the mathematically correct
`Float32` answer and a pure function of `t` on every column, so it is admitted
rather than special-cased. Sabotages `OPT_SAB_POW_RUNNING` and
`OPT_SAB_POW_EXPLOG`.

### 5.2 What PyTorch computes instead, and why the profile does not follow it

PyTorch computes `1 - beta1 ** step` in Python, FLOAT64 on the host, narrowing
to `Float32` only when the scalar reaches the kernel. The profile could have
matched that and does not, for three reasons. **Float64 does not exist on the
device**, so a float64 host route makes the bias correction the one quantity a
device kernel cannot recompute or check, where every other scalar in 7.1 can
be. **Host float64 underflow is a runtime setting, not a constant**: FTZ and
DAZ are mode bits a linked library can set, and `beta2^t` spends thousands of
steps in the subnormal band, so a quantity whose bits depend on an MXCSR bit
set by whichever BLAS loaded first is not a quantity this profile can pin. And
**it is not the profile's arithmetic**; one float64 exception is one place for
a second arithmetic to live.

**The consequence, stated in this direction rather than buried. The profile's
`bc1` and `bc2` are NOT PyTorch's numbers, so a run under this profile is not
bit-comparable with a PyTorch run and no gate should be written as though it
could be.** The property purchased is sameness across vendors, not agreement
with a reference.

### 5.3 `1 - beta` is a `Float32` subtraction here, and the gap is measurable

**CLAUSE 5.3.** `c1 = ftz(1.0 - beta1)` and `c2 = ftz(1.0 - beta2)`, both in
`Float32`, from the `Float32` hyperparameter. *PREDICTED, DERIVED
OFF-REPOSITORY, NOT MEASURED.*

| quantity | this profile | PyTorch's float64 route |
|---|---|---|
| `1 - 0.9` | `0x3DCCCCD0` = 0.10000002384185791 | `0x3DCCCCCD` = 0.10000000149011612 |
| `1 - 0.999` | `0x3A831200` = 0.0009999871253967285 | `0x3A83126F` = 0.0010000000474974513 |
| `1 - 0.99` | `0x3C23D700` = 0.009999990463256836 | `0x3C23D70A` = 0.009999999776482582 |

At `beta2 = 0.999` the two differ by about **1.3e-5 relative**, which is not a
last-bit difference, it is the third significant decimal of the coefficient
that drives `v`. **This is the second reason PyTorch parity is not available
and it is a larger one than 4.2.** Recorded here rather than in a code comment
because a reader who sees `1.0 - beta2` in the oracle will assume it is the
obvious thing, and the obvious thing is a deliberate choice with a measurable
cost.

---

## 6. Denormals, the seam table

Row 10's policy. Under IDENTICAL `ftz` flushes any value below `2^-126` to a
zero of ITS OWN SIGN; under FAST it compiles away. Metal flushes, CUDA honors
denormals by default. **Adam's `v` reaches one on any dead unit.**

Every seam flushes. The unit is "a value a kernel writes for another kernel or
the host to read", plus every stored INTERMEDIATE of a pinned expression,
because an intermediate cannot be reached from the outside on a backend that
does not flush.

    CLIP   C1 each gradient as loaded ; C2 each sumsq (inherited, gemm 5g) ;
           C3 norm_j ; C4 total_sumsq (inherited) ; C5 total_norm ;
           C6 total_norm + CLIP_EPS ; C7 coef ; C8 each rescaled gradient
    HOST   H1 every intermediate of pow_int_f32 ; H2 bc1, bc2 ;
           H3 step_size ; H4 rt_bc2 ; H5 c1, c2 ; H6 decay_mul (AdamW)
    ELEM   O1 g loaded ; O2 p loaded ; O3 m, v loaded ; O4 the decayed
           gradient (Adam) or decayed parameter (AdamW) ; O5 beta1*m_prev ;
           O6 m ; O7 g*g ; O8 beta2*v_prev ; O9 v ; O10 sqrt(v) ;
           O11 sqrt(v)/rt_bc2 ; O12 denom + eps ; O13 q = m/denom ;
           O14 p stored ; S1 the momentum buffer loaded and stored

**O7 is the expensive one to get wrong and the cheap one to get right.** A
gradient at 1e-25 is a perfectly ordinary normal `Float32` and its square is
1e-50, not representable as a normal at all. Flush it and `v` picks up exactly
`c2 * 0`, the FTZ answer on every column. Carry it and Metal disagrees with
CUDA from that step onward, **forever, because `v` is a running state.**

*What would make an `ftz` sabotage pass while gating nothing.* Every fixture
whose gradients are within a few binades of 1.0. **Plant a gradient near
1e-25, where the VALUE is normal and the SQUARE is not.** Not reachable from a
uniform random fixture. Sabotage `OPT_SAB_FTZ_LATE`.

## 7. The op sequences, written out

### 7.1 The host scalars, computed ONCE per step per parameter group

    b1t = pow_int_f32(beta1, t)      b2t = pow_int_f32(beta2, t)
    bc1 = ftz(1.0 - b1t)             bc2 = ftz(1.0 - b2t)
    step_size = ftz(identical_div(lr, bc1))
    rt_bc2    = ftz(identical_sqrt(bc2))
    c1 = ftz(1.0 - beta1)            c2 = ftz(1.0 - beta2)
    decay_mul = ftz(1.0 - ftz(identical_mul(lr, wd)))      AdamW only
    neg_lr    = -lr                                        SGD only, exact
    c_damp    = ftz(1.0 - dampening)                       SGD only

**CLAUSE 9.2.1. A kernel may not recompute them.** They WOULD agree if it did,
because every step is a pinned primitive, but a per-element recomputation is
where a `powf` creeps in, it is `O(log t)` work per element instead of per
step, and having one producer is what makes "the same scalars reached every
element" a structural fact rather than a check. `t` is one-based.

### 7.2 Adam and AdamW, per element

    O1   g  = ftz(grad[i])                       already clipped
    O2   p  = ftz(param[i])
    O3   mp = ftz(m[i]) ; vp = ftz(v[i])
    O4a  ADAM  (only when wd != 0)  g = ftz(identical_mul_add(wd, p, g))  FUSED
    O4b  ADAMW (only when wd != 0)  p = ftz(identical_mul(decay_mul, p))  PRODUCT
         and the gradient is NOT touched
    O5   ms = ftz(identical_mul(beta1, mp))                       PRODUCT
    O6   m  = ftz(identical_mul_add(c1, g, ms))                   FUSED
    O7   g2 = ftz(identical_mul(g, g))                            PRODUCT
    O8   vs = ftz(identical_mul(beta2, vp))                       PRODUCT
    O9   v  = ftz(identical_mul_add(c2, g2, vs))                  FUSED
    O10  s  = ftz(identical_sqrt(v))
    O11  sd = ftz(identical_div(s, rt_bc2))
    O12  dn = ftz(sd + eps)                                       add
    O13  q  = ftz(identical_div(m, dn))
    O14  p  = ftz(identical_mul_add(-step_size, q, p))            FUSED
    store ftz(p), ftz(m), ftz(v)

PRODUCT is `identical_mul`, `identical_mul_add(a, b, -0.0)`. **The `-0.0`
addend is load bearing and a `+0.0` would be a bug**, because
`(-0.0) + (+0.0)` is `+0.0` under round-to-nearest and would launder the sign.

Four contested decisions inside that sequence.

**CLAUSE 9.2.2a. The moment recurrences are a PRODUCT then an FMA**, mirroring
`exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)`, two roundings in that
order. The alternative is `lerp`, `m + c1 * (g - m)`, which recent PyTorch
uses in some paths and which is a different number. Sabotage
`OPT_SAB_MOMENT_LERP`, **bit-inert at every first step from fresh state
because `m_prev` is `+0.0`.**

**CLAUSE 9.2.2b. The `v` term associates as `c2 * (g * g)`**, forming the square
first. `addcmul_` does not say which way it associates and `(c2 * g) * g` is
the other reading. Sabotage `OPT_SAB_SQ_ASSOC`.

**CLAUSE 9.2.2c. The bias correction is applied to the DENOMINATOR and to the
STEP SIZE, not to `m` and `v`.** The documented-pseudocode alternative forms
`m_hat` and `v_hat` explicitly, which is one more rounding and a different
number. The profile takes the implementation shape because that is what a real
training run computes. Sabotage `OPT_SAB_MHAT_FORM`.

**CLAUSE 9.2.2d. O14 is ONE fused rounding**, not a rounded product followed by
a rounded subtract. Only fusion has a portable spelling.

*This is the most important non-vacuity note in the document.*
`check-ieee-arith` scored Metal as UNFUSED over 2^20 HASHED patterns and the
verdict was WRONG: **zero of those 2^20 patterns separate a fused `a*b + c`
from an unfused one**, because random exponents put the product and the addend
so far apart that both spellings round identically. **The fixture must be
BUILT TO SEPARATE**, which means `step_size * q` must have low bits the
addition to `p` would otherwise discard, so `p` and `step_size * q` must be
within a few binades of each other with the product's tail nonzero. **A random
fixture will report a fused-versus-unfused sabotage as INERT and the report
will be false.** Sabotage `OPT_SAB_UNFUSED_UPDATE`.

### 7.3 SGD with momentum, per element

    S1   g = ftz(grad[i]) ;  S2  p = ftz(param[i])
    S3   if wd != 0:  g = ftz(identical_mul_add(wd, p, g))        FUSED
    S4   if momentum != 0:
            if the buffer is UNINITIALIZED:  b = g       A COPY, NO ARITHMETIC
            else: bs = ftz(identical_mul(momentum, ftz(buf[i])))  PRODUCT
                  b  = ftz(identical_mul_add(c_damp, g, bs))      FUSED
            store ftz(b)
            if nesterov: g = ftz(identical_mul_add(momentum, b, g))  FUSED
            else:        g = b                           A COPY, NO ARITHMETIC
    S5   p = ftz(identical_mul_add(neg_lr, g, p))                 FUSED

**CLAUSE 9.2.3a. The first momentum step is a COPY**, `b_1 = g` and not
`c_damp * g`, which is PyTorch's `buf = grad.clone()`, so the dampening factor
is not applied on the step that creates the buffer. *At `dampening = 0.0`,
`c_damp` is exactly `1.0` and `identical_mul(1.0, g)` returns `g`, so **the
default configuration cannot see this clause at all.** The fixture must set
`dampening != 0` and compare at `t = 1`, then at `t = 2` and beyond to show the
divergence persists.* Sabotage `OPT_SAB_MOMENTUM_FIRST_STEP`.

**CLAUSE 9.2.3b. The "is the buffer initialized" flag is CHECKPOINT STATE.** A
resume that reinitializes it recomputes `b` as a copy of the current gradient
instead of continuing the recurrence, and the two runs diverge at that step and
never reconverge. Sabotage `OPT_SAB_RESUME_REINIT`, **which IS clause (d)'s
control 2 (DEVIATION 1473), so skipping clause (d) drops the arm as well.**

**CLAUSE 9.2.3c. Nesterov is `g + momentum * b`, in that operand order**,
matching `grad.add(buf, alpha = momentum)`. The transposed reading is a
different number. Sabotage `OPT_SAB_NESTEROV_ORDER`.

### 7.4 The order weight decay is applied in, Adam against AdamW

| | where the decay lands | what it touches |
|---|---|---|
| Adam | into the GRADIENT, before the moments (O4a) | `g`, so it enters `m` and `v` and is itself smoothed and normalized |
| AdamW | onto the PARAMETER, before the update (O4b) | `p`, and the gradient is untouched, so the decay is not adapted by `v` |

**CLAUSE 9.2.4a.** AdamW's decay is a MULTIPLY, `p * (1 - lr*wd)`, matching
`param.mul_(1 - lr * weight_decay)`. The additive reading `p - lr*wd*p` is a
different number and is the other spelling in circulation. Sabotage
`OPT_SAB_DECAY_ADD_FORM`. **CLAUSE 9.2.4b.** AdamW's decayed `p` is the base of
O14.

*At `weight_decay = 0.0` the two algorithms are the SAME ARITHMETIC*, and
`torch.optim.Adam`'s default is `weight_decay = 0`. **This is the single most
likely vacuous gate in the lane.** Every weight-decay fixture must set
`wd != 0`, and should set it so `lr * wd` is not a power of two so 5.5a
separates too. Sabotage `OPT_SAB_ADAMW_AS_ADAM`.

---

## 8. NaN, infinity and signed zero

**CLAUSE 8a. A non-finite value in a gradient, a parameter or a state is
REFUSED BY NAME before any recorded stage**, tested BY BITS and not by
compares, because Metal flushes COMPARE operands (row 49) so a compare-written
test has two meanings. The reason is row 39: NaN PAYLOADS are vendor shaped,
three payloads for one IEEE answer, so a certified card may never contain a
computed NaN. **This is where DEVIATION 1496 landed**, and DEVIATION 1478 is
the measured finding it corrected.

**It has a real cost and the contract states it.** Production training loops
routinely SKIP a step whose gradient norm is non-finite, and that behavior is
useful. Under this profile it is not arithmetic, it is a CONTROL decision, and
it belongs outside the pinned region as an explicit recorded branch, a
`skipped` flag in the card with the step index, so two runs can be compared on
whether they skipped the same steps. **Folding it into the arithmetic as a
`select` would put a NaN inside a certified stage.**

**CLAUSE 8b. `-0.0` is an admitted value everywhere and the seams preserve it
where the reference does.** The GEMM leaves seed `+0.0`, so a sum of squares
of an all-zero tensor is `+0.0` on every vendor by `(+0) + (-0) = +0`. A
`-0.0` gradient reaching O7 gives `g*g = +0.0`, so `v` never picks up a
negative zero from a squared gradient.

**CLAUSE 8c. There is exactly ONE compare-select in the profile**, the clamp
at 3.4b, on a value 6a has established is finite and non-negative. **So row
13's selection hazard has no reachable site in this lane**, and that is what
makes `amsgrad`'s exclusion load bearing, since `max(v_max, v)` would create
the first one.

---

## 9. Gradient accumulation, what alignment the step requires of its caller

The step imposes NO alignment requirement on its own arithmetic. It imposes
one on whoever produced the gradient, and without it multi-step identical
training is not available at all.

**The measured fact this is built on.** `archive/plans/gemm/IDENTICAL_BACKWARD_PLAN.md`,
measured 2026-08-25 by gate G5. A weight gradient contracts over the TOKEN
dimension, so the token count is the GEMM's `k`, and a split at a boundary
that is both a LEAF and a SUBTREE boundary of v1's balanced tree, accumulated
with the contract's own flushed add, reproduces the unsplit bits exactly.
`T=512` split 256/256 and `T=384` split 256/128 moved 0 of 35 cells; `T=300`
split 150/150, `T=512` split 200/312 and `T=384` split 192/192 moved 31, 30
and 31.

**CLAUSE 9.2.** For `A` microbatches over `T` tokens, accumulation is
bit-identical to the unsplit computation when ALL of the following hold, with
`L = contract_leaf_size(T)`.

1. `contract_leaf_size(T / A) == contract_leaf_size(T)`. The leaf rule holds
   `L` at 128 for every `k` in `(128, 131072]`, so both the full token count
   and the microbatch size land in the same band of the rule.
2. `T mod L == 0`, so `P = T / L` exactly and there is no ragged last leaf.
3. `A` divides `P`.
4. `A` is a power of two.
5. The cross-microbatch combination is the v1 BALANCED TREE over the `A` pieces
   in ascending microbatch index, `ftz(ftz(x) + ftz(y))` at every node, **not a
   running serial sum.**

Conditions 3 and 4 together make each microbatch's leaf range a COMPLETE
SUBTREE: with `A` a power of two and `P` divisible by `A`, the `A` piece roots
are exactly the nodes of level `log2(A)` of the unsplit tree.
`optimizer_oracle.mojo::microbatch_split_is_identical(t_tokens, a)` computes
the predicate on the host, so a trainer can ask rather than guess.

**MEASURED on both columns, clause (e), host only.**

    T=512:  A=1 aligned  A=2 aligned  A=3 no  A=4 aligned  A=8 no
    T=384:  A=1 aligned  A=2 no       A=3 no  A=4 no       A=8 no
    T=300:  none aligned at any A
    T=256:  A=1 aligned  A=2 aligned  A=3 no  A=4 no       A=8 no

**Condition 5 is what the `A = 2` measurements CANNOT establish, and that is
the point.** Over two pieces a serial running sum and a balanced tree are the
same operation. At `A = 4` they separate, and the run measured exactly that.

    T=512 A=2 (ALIGNED): unsplit 0x4b800001  tree 0x4b800001 MATCH
                                             serial 0x4b800001 MATCH
    T=512 A=4 (ALIGNED): unsplit 0x4b800001  tree 0x4b800001 MATCH
                                             serial 0x4b800000 MOVED

**So `T = 512` with `A = 4` is the fixture that closes condition 5**, and it
is inside the range the gemm device check already sweeps. Sabotage
`OPT_SAB_MICROBATCH_SERIAL`, **inert at `A <= 2`.** The ONE-BINADE hashed pair
at `A = 4` was REPORTED and did NOT separate (tree and serial agree), which is
why **DEVIATION 1493 replaced it as the falsifier.**

**The consequence.** The microbatch count is part of a run's numerical
specification unless clause 9.2 holds. A run at `A = 3` and the same run at
`A = 1` are two different numerical experiments and a reproducibility claim
must name `A`. When clause 7 does hold, `A` is free, which is a designable
property, is cheap to arrange, and is the thing that makes identical training
across different hardware budgets possible at all.

---

## 10. The stages, in card order

    step.t              [1] Int      the step index, one-based
    input.param.<j>     [N_j]        as given
    input.grad.<j>      [N_j]        as given, before clipping
    clip.sumsq          [J]          2.2, one v1 GEMM per tensor
    clip.norm           [J]          2.2
    clip.total_sumsq    [1]          2.3, one v1 GEMM over the J norms
    clip.total_norm     [1]          2.3
    clip.coef           [1]          3.4b, after the clamp
    clip.grad.<j>       [N_j]        3.4c, the rescaled gradient
    sched.pow1 / pow2 / bc1 / bc2 / step_size / rt_bc2   [1] each   4.1, 5.2
    sched.decay_mul     [1]          5.2, AdamW only
    adam.gwd.<j>        [N_j]        O4a, Adam with wd != 0 only
    adamw.pdec.<j>      [N_j]        O4b, AdamW with wd != 0 only
    adam.m / adam.v / adam.denom / adam.q   [N_j] each   O6, O9, O12, O13
    sgd.buf.<j> / sgd.dir.<j>        [N_j] each   S4, SGD with momentum only
    param.out.<j>       [N_j]        O14 or S5

Twenty-three stages; the shipped configuration wants 18 of them, which is why
the card carries 18 records. **`adam.gwd` and `adamw.pdec` have NO PRODUCER,
DEVIATION 1477.**

The `sched.*` stages are one-element buffers and are on the card on purpose.
**They are the cheapest possible early-divergence address**: if a cross-vendor
run differs at `sched.pow2` the problem is section 4 and not a single one of
the millions of elementwise updates downstream.

---

## 11. What "identical" is gated to mean

(a) **Device equals host oracle, bitwise, at every stage and every shape.**
Asserted under IDENTICAL; RECORDED, not asserted, under FAST.
(b) **Launch invariance**, the same bits on eight repeated launches and across
a sweep of grid and block geometries. For the elementwise update this is
structural, no float crosses a thread boundary, and the gate verifies what the
construction promises; for the clip's reductions it is inherited from the v1
GEMM. **The GEOMETRY half is not gated, DEVIATION 1479, because `OPT_TPB` is a
comptime literal.**
(c) **Parameter-count invariance, WITH CLIPPING OFF.** One tensor's
`param.out` bits identical whether stepped alone or alongside two others. With
clipping ON this is false by construction, 2.5, and the clause REFUSES a
clipping-on case by name rather than quietly measuring something else.
(d) **STEP-COUNT INVARIANCE.** A run reaching step `t` continuously and a run
resumed from a checkpoint written after step `t - 1` produce bitwise identical
`param.out`, `adam.m`, `adam.v` and `sgd.buf` at step `t`. The checkpoint must
carry, and the gate must round-trip, `t`; every parameter,
gradient-independent state and hyperparameter bit pattern; **the
momentum-buffer-initialized flag per parameter** (7.3b); the `param_id`
assignment and its order (3.3); and any schedule's own state if the schedule
is not a pure function of `t`. The gate runs `2t` steps continuously,
checkpoints at `t`, resumes and compares at `2t`, **with `t = 8` at minimum
because 5.1's spellings agree through `t = 6`.**
(e) **Microbatch-alignment behavior.** At an `A` satisfying clause 9.2, zero
cells move; at an `A` that does not, the number is REPORTED, not asserted, and
the host oracle says the number before the device is asked. **`A = 4` is
required.**
(f) **The row-39 audit of section 6**, including that the refusal path
actually executes, since an unreached refusal is an untested guard.
(g) **Every clause falsifiable by a named sabotage that FAILS a gate**, each
accompanied by the fixture property that makes it non-vacuous.

The STATUS block says which of those ran on which columns.

## 12. The sabotage set

The "must not pass on" column is the fixture property without which the arm
goes inert, and it is the column to read first.

| switch | what it breaks | clause | must not pass on |
|---|---|---|---|
| `OPT_SAB_POW_RUNNING` | `beta^t` by running product | 6 | `t <= 6`. The two agree exactly there. Run to `t >= 8`, ideally 1000 |
| `OPT_SAB_POW_EXPLOG` | `beta^t` via `portable_powf` | 6 | nothing; it separates at `t = 1` |
| `OPT_SAB_EPS_INSIDE_SQRT` | `sqrt(v_hat + eps)` | 4d | `v` near 1e-4. Plant `v` in 1e-20 to 1e-12 |
| `OPT_SAB_RSQRT` | `rsqrt` instead of sqrt-then-div | 4b | a handful of round `v`. Needs thousands of hashed `v`, on all three columns |
| `OPT_SAB_RECIP_MUL` | `m * (1/dn)` | 4c | power-of-two denominators. It is EXACT there |
| `OPT_SAB_ADAMW_AS_ADAM` | decay into the gradient instead of the parameter | 7.4 | `wd == 0`. The two algorithms are then identical |
| `OPT_SAB_DECAY_ADD_FORM` | `p - lr*wd*p` instead of `p*(1-lr*wd)` | 7.4a | `lr*wd` a power of two; tiny `p` |
| `OPT_SAB_MOMENT_LERP` | `m + c1*(g - m)` | 7.2a | `g` and `m` in the same binade with no low bits; **and every first step, where `m_prev` is `+0.0`** |
| `OPT_SAB_SQ_ASSOC` | `(c2*g)*g` | 7.2b | round `g`. Needs hashed `g` |
| `OPT_SAB_MHAT_FORM` | explicit `m_hat`, `v_hat`, `lr *` at the end | 7.2c | nothing known, but it is a 1-ulp-class arm; report the cell count, do not assume |
| `OPT_SAB_UNFUSED_UPDATE` | O14 as a rounded product then a rounded add | 7.2d | **random exponents. ZERO of 2^20 hashed patterns separated fused from unfused. The fixture must be BUILT to separate** |
| `OPT_SAB_FTZ_LATE` | flush only at the store | 6 | gradients near 1.0. Plant one near 1e-25 so the SQUARE is subnormal |
| `OPT_SAB_CLIP_SKIP_AT_ONE` | skip the rescale when `coef_c == 1.0` | 3.4c | normal gradients. Plant a SUBNORMAL cell, and compare `clip.grad`, not `param.out` |
| `OPT_SAB_CLIP_FLAT_NORM` | one flat sum of squares | 3.1 | `J == 1`. Needs `J >= 3` with norms several binades apart |
| `OPT_SAB_CLIP_PARAM_ORDER` | reverse the `param_id` order | 3.3 | **`J == 2`. A swap of two children of one node is bitwise inert. Needs `J >= 3`, prefer 5** |
| `OPT_SAB_CLIP_SERIAL_FOLD` | hand-written serial fold instead of the v1 GEMM | 3.2 | **every `N <= 128`, where `P == 1` and the tree is empty. Needs `N > 128`, and `N = 300` for the ragged leaf** |
| `OPT_SAB_CLIP_BLOCK_PARTITION` | the clip partition reads `block_dim` | 3.2 | a single launch geometry. Needs the geometry sweep |
| `OPT_SAB_MICROBATCH_SERIAL` | serial running accumulation across microbatches | 9.2 | **`A <= 2`. A tree and a serial sum coincide over two pieces. Needs `A = 4`** |
| `OPT_SAB_MOMENTUM_FIRST_STEP` | `b_1 = c_damp * g` | 7.3a | **`dampening == 0`, which is the default. Needs `dampening != 0`** |
| `OPT_SAB_NESTEROV_ORDER` | `b + momentum*g` | 7.3c | `momentum == 0`; also inert when `b == g`, so not at `t = 1` |
| `OPT_SAB_RESUME_REINIT` | drop the buffer-initialized flag on resume | 7.3b, 11(d) | `momentum == 0`; needs a checkpoint at `t >= 1` and a compare at `t + 1` |
| `OPT_SAB_SCALARS_PER_ELEMENT` | recompute `pow_int_f32` inside the kernel | 7.1 | **everything. This arm is EXPECTED to be bit-inert**, because the recomputation uses the same pinned primitives. It is a REACH probe, not a bit probe, and its bit result must be reported as INERT and not as a pass |

The last row is deliberate. Clause 5.2's ban is a DESIGN rule rather than a
numerical one, and **an arm whose predicted result is "no bits move" is worth
having only when it is labelled that way in advance.**

---

## 13. Clause-to-code index

| clause | function | file |
|---|---|---|
| 0 | `OptimizerConfig`, `OPT_*` constants | `training/checks/optimizer_oracle.mojo` |
| 3.2 | `clip_tensor_sumsq_oracle` | same |
| 2.2 device | `identical_clip_grad_norm` | `training/checks/optimizer.mojo` |
| 3.3 | `clip_grad_norm_oracle` | oracle |
| 2.4 | `clip_coefficient` | oracle |
| 3, 5.1 | `identical_sqrt`, `identical_div`, `identical_mul`, `identical_mul_add`, `ftz` | `checks/numerics.mojo` |
| 6 | `pow_int_f32` | oracle |
| 7.1 | `StepScalars`, `step_scalars` | oracle |
| 5.3 | `adam_element_oracle` / `adam_update_kernel` | oracle / device |
| 5.4 | `sgd_element_oracle` / `sgd_update_kernel` | oracle / device |
| 6a | `refuse_nonfinite` | oracle |
| 9.2 | `microbatch_split_is_identical` | oracle |
| NORMATIVE | `optimizer_step_oracle` | oracle |
| 12 | `optimizer_sabotage_name` and the `SAB_*` switches | device |
| every fixture | `check_*` | `training/checks/optimizer_check.mojo` |

---

## 14. Not claimed

1. **Not "bit-identical training".** This profile pins ONE optimizer step
   given a gradient. Closing T9 closes T9.
2. **No PyTorch parity, and two clauses make it impossible.** 5.2 (the bias
   correction is not float64 here) and 5.3 (`1 - beta2` differs by about
   1.3e-5 relative). **A gate that compares against PyTorch is measuring the
   gap, not the profile.**
3. **Two columns is not three.** The missing one is NVIDIA, the column whose
   `sqrt` this contract's clause 4a exists for.
4. **No performance number**, and no claim about what the pins cost. In
   particular the per-element `ftz` at fourteen seams and the `sqrt`-plus-`div`
   in place of one `rsqrt` are real instruction counts and nobody has counted
   them. `IDENTITY IS NOT FREE` is a standing rule.
5. **No NaN payload claim.** 6a refuses non-finite inputs instead.
6. **No claim that the clip's two-level norm is more accurate than a flat
   one.** It is the reference's shape. It is probably slightly worse.
7. **No amsgrad, no foreach or fused apply, no mixed precision, no sharded or
   multi-GPU optimizer, no stateful learning-rate schedule.**
8. **Single GPU.**
9. **The PyTorch spellings cited here are cited from memory**, not from a
   checkout.

---

## 15. Deviation block

Range 1170 to 1189.

| # | what |
|---|---|
| 1170 | the profile, FP32 master weights and state, three algorithms plus the clip pass, hyperparameters as bit patterns |
| 1171 | `pow_int_f32`, `beta^t` as LSB-first binary exponentiation over `identical_mul`, a pure function of the integer `t`; NOT a running product, NOT `portable_powf`, NOT PyTorch's float64 `beta ** t`. The clause that makes step-count invariance structural |
| 1172 | `eps` OUTSIDE the square root, with `step_size = lr/bc1` and `rt_bc2 = sqrt(bc2)` as host scalars |
| 1173 | every division a true `identical_div`; `rsqrt` never called in either mode |
| 1174 | the moment recurrences as PRODUCT then FMA, and the `c2 * (g*g)` association |
| 1175 | the parameter update as ONE fused rounding |
| 1176 | AdamW's decoupled decay as `p * (1 - lr*wd)` applied BEFORE the moments and never to the gradient; Adam's coupled decay as `fma(wd, p, g)` |
| 1177 | the SGD-with-momentum spelling, including the first-step COPY and the Nesterov operand order |
| 1178 | the per-tensor sum of squares IS a v1 GEMM call at `m = n = 1`, `k = N`, `OP_NT` |
| 1179 | the cross-tensor combination is the reference's TWO-LEVEL form over ASCENDING `param_id`, and `param_id` order is part of the profile |
| 1180 | `CLIP_EPS` as a profile constant, the clamp spelling, the UNCONDITIONAL rescale, and the refusal of a non-finite total norm |
| 1181 | the microbatch alignment predicate, and the sharpening that condition 5 needs `A = 4` |
| 1182 | the `ftz` seam table, including O7's squared-gradient case |
| 1183 | the checkpoint contents that make step-count invariance checkable, including the momentum-buffer-initialized flag and the `param_id` assignment |
| 1184 | the device execution plan, one thread per element, block size a scheduling row, the clip's reductions delegated entire |
| 1185 | the FAST arms, same code, same partition, same order, both pins compiled away, no identity claim |
| 1186 | the sabotage set and its non-vacuity column |
| 1187-1189 | RESERVED, unused |

Cited from elsewhere and never redefined: 258, 550, 741, 824, 826, 1473, 1477,
1478, 1479, 1493, 1496, 1613.

## 16. OWED, AND WHY I DID NOT DO IT HERE

1. **THE CROSS-VENDOR LEG. Everything was measured on TWO columns and the
   missing one is NVIDIA.** This is the largest item by a wide margin.
2. **Clauses (b), (c), (d) and (f) on the two existing columns**, remembering
   that (d) carries the `OPT_SAB_RESUME_REINIT` arm and (f) carries DEVIATION
   1478's finding. And the GEOMETRY half of (b), which needs `OPT_TPB` to stop
   being a comptime literal, DEVIATION 1479.
3. **A `pixi.toml` task**, `check-optimizer`, and the sabotage arms as
   `-D MOJOLEARN_OPT_SABOTAGE_*` builds.
4. **`IDENTITY_PATHS.md` rows** for the optimizer step and for the global-norm
   clip, in the file's PIN / REPLACE / REFUSE form, with the number taken at
   commit time rather than reserved here.
5. **A PyTorch source read.** Clone `pytorch/pytorch` at a pinned tag, mirror
   `torch/optim/{sgd,adam,adamw}.py` and `torch/nn/utils/clip_grad.py`, and
   re-check every "the reference spells it" sentence. **Where the source
   disagrees with a clause here, the CLAUSE is what stands**, since the
   profile is normative and PyTorch is the design reference, **but the citation
   must be corrected in the same commit and the deviation renumbered as a
   deliberate departure rather than an accidental one.**
6. **A `kernel_matrix.mojo` row for `OPT_TPB`.** It is a literal 256 in
   `optimizer.mojo`. Block size is a SCHEDULING row and free in both modes, so
   no bit depends on it, **but `column_max_block_size(COLUMN_SPEC_BASELINE)` is
   128, so a literal 256 would be REFUSED on the portable-floor column rather
   than resolved downward.**
7. **A shared home for `refuse_nonfinite`.** `mamba_oracle.mojo:57`,
   `loss_oracle.mojo:167` and `embedding`'s copy are the others.
   `checks/numerics.mojo` is the canonical home and **four copies of one
   predicate have four chances to drift.**
8. **`DERIVATION_MAP.tsv` and `NOT_IMPLEMENTED.tsv` entries for `training/`.**
9. **A corpus.** `mamba/corpus` is the model, planted adversarial cases with
   names. This lane needs `adv_subnormal_square` (5.1), `adv_dead_unit_v`
   (3d), `adv_dampening_first_step` (5.4a), `adv_param_order_five` (3.3) and
   `adv_pow_step_1000` (4.1). **None exists**, which is why every clause is our
   device against our oracle and seven of twenty-three stages are the same host
   code on both sides.
