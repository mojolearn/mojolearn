# The IDENTICAL FP32 optimizer-step contract

# PROFILE `mojolearn.identical.optimizer.fp32.v1`

**STATUS, AND IT IS THE FIRST THING A READER NEEDS. CORRECTED 2026-08-31:
COMPILED AND RUN, ON ONE DEVICE.** This banner used to say "Nothing in this
lane has been compiled or executed", that neither
`training/mojo_only/optimizer_oracle.mojo` nor
`training/mojo_only/optimizer.mojo` had been through a compiler, that every
PREDICTED number was a prediction about what the gate would print, and that
nothing here had run on any GPU on one vendor or on three. **The gate file it
said was not in this lane's write set was written at `ecd1a436` and RAN at
`b90f52ab`**, which falsified the first three of those. The run measured a
real defect: with clipping off, a non-finite planted in a PARAMETER reached
`param.out`, because the only device-side refusal lived in
`identical_clip_grad_norm`, which does not run when clipping is off, and
covered `grad` alone. DEVIATION 1496 moved the refusal to the first statement
of `identical_optimizer_step` and made it call the oracle's own function.
PREDICTED numbers that no clause covered are still predictions. **THE LAST
CLAUSE OF THE OLD BANNER SURVIVES AND IS THE IMPORTANT ONE: this has run on
ONE VENDOR and not on three.** Section 16 lists what is owed.

The companion documents, and this contract is not readable without the
first of them.

- `gemm/IDENTICAL_FP32_CONTRACT.md` -- profile `mojolearn.identical.gemm.
  fp32.v1`. **This contract DELEGATES every reduction it performs to that
  one** (section 3.2), so its sections 5, 6, 7 and 9 are load bearing here.
- `gemm/IDENTICAL_BACKWARD_PLAN.md` -- sections 3.2, 4.2 and 5.2. Row T9
  of its ladder is this lane, and 4.2 is the five-line brief this document
  expands.
- `mamba/IDENTICAL_MAMBA_CONTRACT.md` -- the shape of a seam table and of
  a "Not claimed" section; this document follows it.
- `IDENTITY_PATHS.md` rows 9, 10, 12, 13, 39, 49 and 50, which are the
  pins every clause below is built from.

---

## 0. What is in scope, and what the inventory came from

### 0.1 The three algorithms

| kind | reference | what it is here |
|---|---|---|
| `OPT_SGD` | `torch.optim.SGD` | plain SGD, momentum, dampening, Nesterov, coupled L2 decay |
| `OPT_ADAM` | `torch.optim.Adam` | Adam with COUPLED decay, the decay folded into the gradient |
| `OPT_ADAMW` | `torch.optim.AdamW` | Adam with DECOUPLED decay, the decay applied to the parameter |

Plus one preprocessing pass that is not an optimizer and is specified here
because it is where all the difficulty is.

| pass | reference | what it is here |
|---|---|---|
| `clip_grad_norm` | `torch.nn.utils.clip_grad_norm_`, `norm_type = 2` | a two-level global L2 norm and one unconditional rescale |

FP32 master weights, FP32 gradients, FP32 optimizer state. There is no
second precision anywhere in the profile.

### 0.2 Excluded, each with the reason

- **`amsgrad`.** It maintains `v_max = max(v_max, v)`, which is
  IDENTITY_PATHS row 13's selection hazard (`-0.0` and `+0.0` compare
  equal, so which one survives is decided by ORDER) plus the NaN
  canonicalization row 39 measured three vendor answers for. The spelling
  it would take already exists and is `identical_fmax` (DEVIATION 824),
  so this is a small addition and not a hard one. It is excluded because
  an unreached branch is not a working guard, and this lane cannot reach
  anything.
- **`foreach` and `fused` multi-tensor apply.** A multi-tensor kernel
  changes which elements share a thread block. Under section 7 that
  cannot reach the arithmetic, so the exclusion is about scope and not
  about bits -- but the claim "it cannot reach the arithmetic" is exactly
  the kind of claim that has to be gated rather than asserted, and the
  gate is owed.
- **`maximize`.** A sign flip on the gradient. Cheap and unreached.
- **Learning-rate schedules that carry state.** A schedule is admitted
  only as a pure function of the integer step index (section 5). A
  schedule that carries a running state, `ReduceLROnPlateau` being the
  common one, moves the step-count-invariance obligation onto that state
  and section 11(d) then covers the schedule's state too.
- **Mixed precision, loss scaling, master-weight copies in BF16 or FP16.**
  Excluded by gemm contract 0.5 and 1, inherited.
- **Sharded, offloaded or multi-GPU optimizers (ZeRO, FSDP).** An
  all-reduce is a summation order chosen at runtime from the topology.
  `IDENTICAL_BACKWARD_PLAN.md` 4.4 REFUSES this and so does this contract.
  **A claim about a run under this profile must contain the words "single
  GPU".**
- **Sparse gradients and `index_add`-shaped updates.** That is T10, a
  REPLACE and not a PIN, and it is a different lane.
- **The backward pass itself.** T1 and T2 are the gemm lane's.

### 0.3 The two references, and which one is normative

`training/mojo_only/optimizer_oracle.mojo::optimizer_step_oracle` is
**NORMATIVE**. It is host, scalar, single threaded, and it is built from
`mojo_only/numerics.mojo`'s actual helpers rather than from local copies
of them, so it cannot drift into a second opinion about what IDENTICAL
means. That is `gemm/mojo_only/gemm_oracle.mojo`'s construction and the
consequence is the same one -- under `NUMERIC_FAST` both helpers compile
away and the oracle is then the FAST spelling of the same loops, not the
contract.

PyTorch is the **DESIGN reference and is not a bit reference.** Two
clauses below make bit parity with PyTorch impossible by construction
(section 5.2 and section 5.3), and both differences are recorded with
numbers rather than waved at. Nothing in this profile claims to reproduce
a PyTorch training run, and no gate should be written as though it could.

**CITATION HEALTH WARNING.** There is no PyTorch checkout in
`/Users/andrewhendel/CascadeProjects/upstream/`, so every PyTorch spelling
cited below is cited FROM MEMORY of the documented pseudocode and of
`torch/optim/adam.py`'s single-tensor path. `PORTING_RULES.md`'s standing
requirement is to read their source, and this lane could not. Section 16
owes that read. Where a clause depends on which of two PyTorch spellings
is real, the clause states BOTH and names its own answer as the profile's,
which is the construction that survives the citation being wrong.

---

## 1. Dtypes, hyperparameters, and the bit-pattern rule

Every value in the profile is `Float32`. Parameters, gradients, the
momentum buffer, `m`, `v`, every host scalar derived per step, and every
hyperparameter.

**Hyperparameters are BIT PATTERNS, not decimal strings.** A run's card
records `lr`, `beta1`, `beta2`, `eps`, `weight_decay`, `momentum`,
`dampening` and `max_norm` as eight-hex-digit `Float32` patterns, and two
runs are comparable only when those patterns are equal. This is
`[[mojo-string-float-roundtrip]]` applied to configuration rather than to
output. It matters more than it looks -- `0.999` narrowed from a float64
parse is `0x3F7FBE77`, and so is a direct `Float32` literal, but the
DERIVED quantity `1 - beta2` is not the same number by the two routes and
section 5.3 measures the gap.

The profile's default patterns.

| name | decimal | `Float32` bits |
|---|---|---|
| `beta1` | 0.9 | `0x3F666666` |
| `beta2` | 0.999 | `0x3F7FBE77` |
| `eps` | 1e-8 | `0x322BCC77` |
| `CLIP_EPS` | 1e-6 | `0x358637BD` |
| `1.0` | 1.0 | `0x3F800000` |

`CLIP_EPS` is fixed by the profile and is not a caller-supplied number,
because `torch.nn.utils.clip_grad_norm_` hardcodes `1e-6` and a caller who
could change it could change the bits of every parameter in the model
through one scalar.

There is no float64 anywhere in this profile, on the device or on the
host. On the device that is a hardware fact (`mojolearn-hardware-limits`).
On the HOST it is a deliberate choice with a reason that is not obvious,
and section 5.2 is that reason.

---

## 2. Why an elementwise step is not automatically identical

An Adam update is elementwise, so no float crosses a thread boundary and
there is no summation order in it. That sentence is true and it is what
makes this lane look easy. Everything that is actually hard is in the
parts of a step that are NOT the update.

1. **The gradient norm.** `sqrt(sum(g^2))` over every parameter tensor in
   the model is a fold across many tensors of many different lengths. Its
   partition, its tree, and the order the per-tensor results are combined
   in are three separate free choices, and the third one is the one a
   reader skips. Section 3.
2. **`sqrt`.** Mojo's `std.math.sqrt` lowers to an APPROXIMATE PTX sqrt on
   NVIDIA -- 180,714 of 2^20 hashed patterns off by one ulp with 176,577
   of those on NORMAL inputs (DEVIATION 258). That was the single NVIDIA
   miss in an otherwise closed lane (DEVIATION 550). Adam takes a square
   root per element per step. Section 4.
3. **`rsqrt`.** An `rsqrt` intrinsic is a per-vendor approximation. PTX
   `rsqrt.approx` is about 2 ulp. It is never called. Section 4.
4. **The division.** `m / denom` and `m * (1/denom)` are different bits
   whenever `denom` is not a power of two. Section 4.
5. **Bias correction.** `1 - beta^t` by repeated multiplication and by a
   general `pow` are different numbers, and the difference GROWS with `t`.
   A run resumed from a checkpoint must reproduce a continuous run bit for
   bit, and the bias correction is the one place where the arithmetic can
   depend on HOW the step index was reached rather than on what it is.
   Section 5.
6. **Denormals.** Adam's `v` is a running average of squared gradients and
   goes very small. `g` at 1e-25 gives `g*g` at 1e-50, which is not
   representable as a normal `Float32` at all. Section 6.
7. **The order weight decay is applied in.** Adam and AdamW differ in
   exactly this and in nothing else. Section 7.4.

---

## 3. The gradient-norm clip, which is the hardest clause here

### 3.1 The reference's shape is TWO LEVELS, and that is load bearing

`torch.nn.utils.clip_grad_norm_` does not compute one flat sum of squares
over the whole model. It computes a per-tensor L2 norm, stacks those into
a vector, and takes the L2 norm of THAT.

    norms      = [ vector_norm(g_j, 2) for j in parameters ]
    total_norm = vector_norm( stack(norms), 2 )
    coef       = max_norm / (total_norm + 1e-6)
    coef       = clamp(coef, max = 1.0)
    for j:  g_j *= coef

**In exact arithmetic the two-level form and the flat form are the same
number. In `Float32` they are not**, because each per-tensor `sqrt` rounds
and then each result is squared again inside the outer norm. The profile
takes the REFERENCE's two-level form. This is `COPY, DO NOT IMPROVE`
applied to a place where the improvement is tempting and is one line.

**CLAUSE 3.1.** `total_norm = sqrt( SUM_j ( sqrt(sumsq_j) )^2 )`, not
`sqrt( SUM_j sumsq_j )`. Sabotage `OPT_SAB_CLIP_FLAT_NORM`.

*What would make this pass while gating nothing.* A fixture with ONE
parameter tensor. At `J = 1` the two forms are `sqrt(s)` against
`sqrt(sqrt(s)^2)`, which agree for most `s` because squaring a rounded
square root usually lands back in the same binade. **The fixture must
carry at least three tensors of different lengths and of norms that differ
by several binades.**

### 3.2 The per-tensor sum of squares IS a v1 GEMM, and is not written here

**CLAUSE 3.2.** For a parameter tensor of `N` elements,

    sumsq_j = gemm_oracle( g_j, g_j, OP_NT, m = 1, n = 1, k = N )[0]

and on the device the same call goes through `identical_gemm`. Not a fold
written in this lane. Not `pinned_block_sum`. Not a `sum` helper.

Three things follow from delegating rather than writing, and they are the
whole reason for the clause.

1. **The partition is `contract_leaf_size(N)` and nothing else.** Not the
   block count, not the vendor, not the occupancy, not how many other
   tensors are in the launch. gemm contract section 6, inherited entire.
2. **The fold is v1's fixed balanced tree over adjacent leaves with the
   odd tail carried.** gemm contract 7.2, inherited entire, including
   the "no padding" and "no stride pairing" clauses and their fixtures.
3. **It inherits a MEASUREMENT.** The v1 device card was bit-identical
   Apple M4 against NVIDIA H100 against AMD MI325X at leg 11, commit
   `144aa5b` (gemm contract section 11 item 2). A fold written fresh in
   this lane would inherit nothing and would owe its own three-vendor leg.

At `m = n = 1` and `op = OP_NT` the GEMM's operand accessors reduce to
`a[p]` and `b[p]`, so the call really is the sum of squares and not a
shape that happens to be near one. `_a_at(a, OP_NT, 0, p, 1, N)` is
`a[0*N + p]` and `_b_at(b, OP_NT, p, 0, 1, N)` is `b[0*N + p]`.

Then `norm_j = ftz( identical_sqrt( ftz(sumsq_j) ) )`.

*What would make this pass while gating nothing.* Any fixture whose
tensors all have `N <= 128`. There `P == 1`, the tree has no arithmetic
node, and the v1 answer IS the serial ascending chain, so a hand-written
serial fold passes. **At least one tensor must have `N > 128`, and one
should have an `N` that is not a multiple of 128 so the ragged last leaf
is exercised. `N = 300` gives `P = 3` with a 44-element last leaf, which
is the gemm lane's own ragged fixture.** Sabotage
`OPT_SAB_CLIP_SERIAL_FOLD`.

### 3.3 The cross-tensor fold, and why PARAMETER ORDER is in the profile

**CLAUSE 3.3.** The outer norm is the same construction over the vector of
per-tensor norms.

    total_sumsq = gemm_oracle( norms, norms, OP_NT, m = 1, n = 1, k = J )[0]
    total_norm  = ftz( identical_sqrt( ftz(total_sumsq) ) )

where `norms` is indexed by `param_id` ASCENDING.

**`param_id` is part of the profile.** It is an integer assigned once when
the model is registered, it is written into the checkpoint, and it is not
the iteration order of a dictionary, not the order the optimizer happened
to receive the tensors in, not a device pointer order and not a name sort
that could change when a layer is renamed. Two runs with the same
parameters in a different `param_id` order are two different numerical
experiments and the card must show it.

This is the clause the brief says is most likely to be waved past, and the
reason it is easy to wave past is that it looks like bookkeeping. It is
not. It is the cross-tensor summation order, and a summation order is the
thing this whole repository exists to pin.

*What would make this pass while gating nothing.* A fixture with `J = 2`.
Reversing two elements swaps the two children of ONE tree node, and
`a + b` equals `b + a` bitwise, so **a two-tensor fixture cannot see a
parameter-order sabotage at all.** `J >= 3` is required, and `J = 5` is
better because it also exercises the odd-tail carry at two levels (gemm
contract 7.2.2 notes `P = 5` is the smallest `P` that carries twice).
Sabotage `OPT_SAB_CLIP_PARAM_ORDER`.

### 3.4 The coefficient, the clamp, and the multiply that may not be skipped

**CLAUSE 3.4a.** `denom = ftz( total_norm + CLIP_EPS )`, `CLIP_EPS` the
profile constant `0x358637BD`. `coef = ftz( identical_div(max_norm,
denom) )`. The division is a true divide (section 4).

**CLAUSE 3.4b.** The clamp is `coef_c = coef if coef < 1.0 else 1.0`,
a compare-select against the exact `Float32` `1.0`. Written with `<` on a
value that section 8 has already established is finite and non-negative,
so the compare has one meaning on every column.

**CLAUSE 3.4c.** `g_j[i] = ftz( identical_mul( coef_c, ftz(g_j[i]) ) )`
for every element of every tensor, **UNCONDITIONALLY, including when
`coef_c` is exactly `1.0`.** A "skip the rescale when no clipping is
needed" optimization is FORBIDDEN.

The reason is precise and it is worth spelling out because the
optimization looks obviously safe. `identical_mul(1.0, x)` is
`fma(1.0, x, -0.0)`, which returns `x` exactly for every finite `x`
including both signed zeros. So for a NORMAL gradient the multiply and the
skip agree. They do NOT agree for a SUBNORMAL gradient, because the
multiply's operand flush and result flush turn it into a signed zero and
the skip leaves the original bit pattern in the buffer.

And here is the honest half, which belongs in the contract and not in a
footnote. **The difference is CARD VISIBLE and DOWNSTREAM INERT.** The
`clip.grad` stage hash differs between the two spellings. The optimizer's
own first act is `ftz` on the gradient load (seam O1), so by the time the
value reaches `m` and `v` the two spellings have converged. So this clause
protects the card and protects any OTHER consumer of the gradient buffer
-- a logger, a second optimizer, a gradient-statistics pass -- and it does
not protect the parameter update. That is exactly the distinction
`reached-but-inert` asks for, stated at the clause rather than discovered
by whoever writes the gate.

*What would make this pass while gating nothing.* Any fixture whose
gradients are all normal. **The fixture must plant a subnormal gradient
cell**, and it must compare the `clip.grad` stage and not only
`param.out`. Sabotage `OPT_SAB_CLIP_SKIP_AT_ONE`.

### 3.5 What clipping costs the independence claim, stated rather than hidden

**Without clipping, one parameter's update bits are independent of every
other parameter in the model.** The update is elementwise and reads only
that tensor's own gradient and state. That is a real and useful property
and section 11(f) gates it.

**With clipping, that property is FALSE, by construction and by the
reference's own semantics.** `coef_c` is a function of every gradient in
the model, so adding a parameter tensor, removing one, or changing one
element of one of them changes every parameter's update. This is not a
defect and it is not repairable -- it is what a GLOBAL norm clip means.

The consequence for gates. A parameter-set-invariance gate must be run
with clipping OFF. With clipping ON the claim is the narrower one, which
is that the answer is a pure function of the parameter registry and the
gradients, and the registry is part of the run's identity.

---

## 4. `sqrt`, `rsqrt`, and the division

**CLAUSE 4a. Every square root is `identical_sqrt`.** Under IDENTICAL that
is `portable_sqrtf`, one arithmetic, correctly rounded by construction and
measured 0 mismatches against a float64 reference over 2^20 patterns.
Under FAST it is the stdlib's device path verbatim.

This is not defensive. `std.math.sqrt` on NVIDIA is an approximate PTX
sqrt, measured 180,714 of 2^20 patterns off by one ulp with 176,577 of
those on normals (DEVIATION 258), and an unrouted `sqrt` was the single
NVIDIA miss in the unsupervised lane (DEVIATION 550). There are three
square roots in a clipped Adam step, at `norm_j`, at `total_norm` and at
`sqrt(v)` per element, and the third one runs once per parameter per step.

**CLAUSE 4b. `rsqrt` is never called, in either mode.** Not
`std.math.rsqrt`, not `identical_rsqrt`, not a Newton refinement. The
denominator is built from a `sqrt` and a `div`.

The temptation is real, because `1/sqrt(v)` looks like the natural
spelling and because a hardware `rsqrt` is one instruction. Two reasons it
is refused. The intrinsic is a per-vendor approximation, PTX
`rsqrt.approx` at about 2 ulp. And `identical_rsqrt`, which IS pinned, is
`1 / portable_sqrtf(x)` -- two correctly rounded operations, and DEVIATION
741 measured it off the correctly rounded `rsqrt` on 134,858 of 520,133
positive-normal lanes. So even the pinned `rsqrt` is a DIFFERENT NUMBER
from `sqrt` followed by `div`, and the profile has to say which one it
means. It means `sqrt` then `div`, because that is the shape section 7.2's
denominator has, with an `eps` added between them.

*What would make a `rsqrt` sabotage pass while gating nothing.* DEVIATION
741's own numbers say that about a quarter of positive normals separate
the pin from the intrinsic. **A handful of round `v` values will pass. The
fixture needs at least a few thousand hashed `v` values**, and the
intrinsic arm must be run on all three columns before anything is
concluded, which is the DEVIATION 258 lesson. Sabotage `OPT_SAB_RSQRT`.

**CLAUSE 4c. Every division is `identical_div`, a true divide.** Never a
reciprocal followed by a multiply. There are three divisions per step,
`lr / bc1` and `max_norm / denom` on the host and `m / denom` per element.

`identical_div` is `portable_divf`, which is row 10's flush model around
ONE hardware division. `check-division` characterized Apple's division as
correctly rounded on the whole normal class over 2^20 pairs, and
`check-ieee-arith` measured div 0 wrong on the H100 and the MI325X over
its own patterns. That is not a certificate for the other two columns and
`portable_divf`'s own docstring says so, which is the right place for the
named seam to be if a vendor turns out to be wrong.

*What would make a reciprocal-multiply sabotage pass while gating
nothing.* `x * (1/d)` is EXACT when `d` is a power of two. **A fixture
whose denominators are powers of two passes vacuously.** Plant
denominators that are not. Sabotage `OPT_SAB_RECIP_MUL`.

**CLAUSE 4d. `eps` is added OUTSIDE the square root**, to the
bias-corrected denominator. Section 7.2 writes the exact position. The
alternative, `sqrt(v_hat + eps)`, is what several reference
implementations do and it is a different number.

*What would make an eps-position sabotage pass while gating nothing.*
Whenever `v_hat` is much larger than `eps^2` the two spellings agree to
the last bit, and a fixture of ordinary gradients has `v` around 1e-4.
With `eps = 1e-8`, `eps^2` is 1e-16. **The fixture must plant `v` in the
1e-20 to 1e-12 band**, which is a small-but-nonzero gradient that a real
training run reaches on a dead unit and a synthetic fixture never reaches
by accident. Sabotage `OPT_SAB_EPS_INSIDE_SQRT`.

---

## 5. Bias correction, and the step-count-invariance clause

### 5.1 `beta^t` is a pure function of the INTEGER `t`, by binary exponentiation

**CLAUSE 5.1.** `beta1^t` and `beta2^t` are computed on the host, once per
step per parameter group, by LSB-first binary exponentiation over
`identical_mul`.

    pow_int_f32(base, t):
        acc  = 1.0
        b    = base
        e    = t
        while e > 0:
            if (e & 1) != 0:  acc = ftz(identical_mul(acc, b))
            b = ftz(identical_mul(b, b))
            e = e >> 1
        return acc

Three properties, and the third is the one the whole section is for.

1. **It uses only the profile's own arithmetic.** Every step is
   `identical_mul`, which is `fma(a, b, -0.0)` under IDENTICAL, so the
   same function evaluated on a DEVICE returns the same bits. No float64,
   so `mojolearn-hardware-limits` is satisfied and so is the host FTZ
   hazard in 5.2.
2. **It is not a general `pow`.** `portable_powf` is `exp(p * log(x))`
   through two Cephes polynomials, accurate to a few ulp and NOT exact
   even at `t = 1`. `t` here is an integer and integer powers do not need
   a transcendental.
3. **IT IS A PURE FUNCTION OF `t`, SO STEP-COUNT INVARIANCE IS
   STRUCTURAL.** There is no running `beta_pow` state to checkpoint,
   nothing to reconstruct on resume, and no way for a resumed run to
   disagree with a continuous one about what `beta1^t` is. The gate then
   verifies what the construction promises, rather than being the only
   thing standing between the profile and a silent drift.

The alternative that this clause exists to refuse is the running product
`b_t = b_{t-1} * beta`, which is what an implementation reaches for
because it is one multiply per step. It is `t - 1` sequential roundings
and it is a different number.

*PREDICTED, DERIVED OFF-REPOSITORY, NOT MEASURED.* In `Float32` with the
profile's flush, `pow_int_f32(beta, t)` and the running product first
differ at **`t = 7`**, for `beta = 0.9` and for `beta = 0.999` alike.
**A sabotage gate that runs to `t = 4` is VACUOUS**, and `t = 1` through
`t = 6` agree exactly, which is precisely the range a hand-written fixture
stops at. The gate must run to at least `t = 8` and should run to
`t = 1000`.

*Also PREDICTED.* With the flush, `beta1^t` reaches exactly `+0.0` at
`t = 829` for `beta1 = 0.9` and `beta2^t` at `t = 87,295` for
`beta2 = 0.999`, after which `bc1` and `bc2` are exactly `1.0` and stay
there. That is the mathematically correct `Float32` answer -- `0.9^829` is
below the smallest normal -- and it is a pure function of `t` on every
column, so it is admitted rather than special-cased. The monotonicity
argument is that `base` in the loop is `beta^(2^j)`, and once that flushes
to zero every `t` with that bit set has `beta^t <= beta^(2^j)`, which is
also below the flush threshold.

Sabotages `OPT_SAB_POW_RUNNING` and `OPT_SAB_POW_EXPLOG`.

### 5.2 What PyTorch computes instead, and why the profile does not follow it

PyTorch computes `bias_correction1 = 1 - beta1 ** step` in Python, which
is FLOAT64 arithmetic on the host, and narrows to `Float32` only when the
resulting scalar reaches the kernel.

The profile could have matched that. It does not, for three reasons.

- **Float64 does not exist on the device**, so a float64 host route makes
  the bias correction the one quantity a device kernel cannot recompute or
  check. Every other scalar in section 7.1 can be.
- **Host float64 underflow is a runtime setting, not a constant.** x86 and
  arm CPUs honor denormals by default, but FTZ and DAZ are mode bits that
  a linked library can set, and `beta2^t` spends thousands of steps in the
  subnormal band before it reaches zero. A quantity whose bits depend on
  an MXCSR bit set by whichever BLAS happened to load first is not a
  quantity this profile can pin.
- **It is not the profile's arithmetic.** Everything else here is built
  from `identical_mul`, `identical_mul_add`, `identical_div`,
  `identical_sqrt` and `ftz`. One float64 exception is one place for a
  second arithmetic to live.

**The consequence, and it must be stated in this direction rather than
buried.** The profile's `bc1` and `bc2` are NOT PyTorch's numbers, so a
run under this profile is not bit-comparable with a PyTorch run and no
gate should be written as though it could be. That is the same position
`portable_logf` is in -- it is not `raft::log` either -- and the property
purchased is sameness across vendors, not agreement with a reference.

### 5.3 `1 - beta` is a `Float32` subtraction here, and the gap is measurable

**CLAUSE 5.3.** `c1 = ftz(1.0 - beta1)` and `c2 = ftz(1.0 - beta2)`, both
in `Float32`, from the `Float32` hyperparameter.

*PREDICTED, DERIVED OFF-REPOSITORY, NOT MEASURED.*

| quantity | this profile | PyTorch's float64 route |
|---|---|---|
| `1 - 0.9` | `0x3DCCCCD0` = 0.10000002384185791 | `0x3DCCCCCD` = 0.10000000149011612 |
| `1 - 0.999` | `0x3A831200` = 0.0009999871253967285 | `0x3A83126F` = 0.0010000000474974513 |
| `1 - 0.99` | `0x3C23D700` = 0.009999990463256836 | `0x3C23D70A` = 0.009999999776482582 |

At `beta2 = 0.999` the two `c2` values differ by about **1.3e-5 relative**,
which is not a last-bit difference -- it is the third significant decimal
of the coefficient that drives `v`. This is the second reason PyTorch
parity is not available and it is a larger one than 5.2.

It is recorded here rather than in a code comment because a reader who
sees `1.0 - beta2` in the oracle will assume it is the obvious thing, and
the obvious thing is a deliberate choice with a measurable cost.

---

## 6. Denormals -- the seam table

Row 10's policy. Under IDENTICAL `ftz` flushes any value of magnitude
below `2^-126` to a zero of ITS OWN SIGN; under FAST it compiles away.
Metal flushes, CUDA honors denormals by default, so without the flush the
same step diverges on any path a denormal can reach. Adam's `v` reaches
one on any dead unit.

Every seam below flushes. The unit is "a value a kernel writes for another
kernel or the host to read", plus, per `ftz`'s own docstring, every stored
INTERMEDIATE of a pinned expression, because an intermediate cannot be
reached from the outside on a backend that does not flush.

| # | seam | spelling |
|---|---|---|
| C1 | each gradient element as loaded by the clip pass | `ftz(g[i])` |
| C2 | each `sumsq_j` as returned by the GEMM | inherited, gemm 5g |
| C3 | `norm_j` | `ftz(identical_sqrt(ftz(sumsq_j)))` |
| C4 | `total_sumsq` | inherited, gemm 5g |
| C5 | `total_norm` | `ftz(identical_sqrt(ftz(total_sumsq)))` |
| C6 | `denom = total_norm + CLIP_EPS` | `ftz(...)` |
| C7 | `coef` | `ftz(identical_div(max_norm, denom))` |
| C8 | each rescaled gradient as stored | `ftz(identical_mul(coef_c, ftz(g[i])))` |
| H1 | `b1t`, `b2t`, every intermediate of `pow_int_f32` | `ftz` at every `identical_mul` |
| H2 | `bc1`, `bc2` | `ftz(1.0 - b_t)` |
| H3 | `step_size` | `ftz(identical_div(lr, bc1))` |
| H4 | `rt_bc2` | `ftz(identical_sqrt(bc2))` |
| H5 | `c1`, `c2` | `ftz(1.0 - beta)` |
| H6 | `decay_mul` for AdamW | `ftz(1.0 - ftz(identical_mul(lr, wd)))` |
| O1 | gradient as loaded | `ftz(g[i])` |
| O2 | parameter as loaded | `ftz(p[i])` |
| O3 | `m` and `v` as loaded | `ftz(m[i])`, `ftz(v[i])` |
| O4 | the decayed gradient (Adam) or the decayed parameter (AdamW) | `ftz(...)` |
| O5 | `beta1 * m_prev` | `ftz(identical_mul(...))` |
| O6 | `m` as computed and as stored | `ftz(identical_mul_add(...))` |
| O7 | `g * g` | `ftz(identical_mul(g, g))` |
| O8 | `beta2 * v_prev` | `ftz(identical_mul(...))` |
| O9 | `v` as computed and as stored | `ftz(identical_mul_add(...))` |
| O10 | `sqrt(v)` | `ftz(identical_sqrt(v))` |
| O11 | `sqrt(v) / rt_bc2` | `ftz(identical_div(...))` |
| O12 | `denom = ... + eps` | `ftz(...)` |
| O13 | `q = m / denom` | `ftz(identical_div(m, denom))` |
| O14 | the parameter as stored | `ftz(identical_mul_add(-step_size, q, p))` |
| S1 | the momentum buffer as loaded and as stored | `ftz(...)` |

O7 is the expensive one to get wrong and the cheap one to get right. A
gradient at 1e-25 is a perfectly ordinary normal `Float32`, and its square
is 1e-50, which is not representable as a normal at all. Flush it and `v`
picks up exactly `c2 * 0`, which is the FTZ answer on every column. Carry
it and Metal disagrees with CUDA from that step onward, forever, because
`v` is a running state.

*What would make an `ftz` sabotage pass while gating nothing.* Every
fixture whose gradients are within a few binades of 1.0. **Plant a
gradient near 1e-25, where the VALUE is normal and the SQUARE is not.**
That is the one input class that separates a late flush from a per-step
flush, and it is not reachable from a uniform random fixture. Sabotage
`OPT_SAB_FTZ_LATE`.

---

## 7. The op sequences, written out

FMA contraction is PER SEAM (`mojolearn-lossguide-lane`). A seam marked
FUSED is one rounding through `identical_mul_add`. A seam marked PRODUCT
is one rounding through `identical_mul`, which is
`identical_mul_add(a, b, -0.0)` -- the spelling no codegen may contract
into a neighboring add and which preserves a `-0.0` product. **The `-0.0`
addend is load bearing and a `+0.0` would be a bug**, because
`(-0.0) + (+0.0)` is `+0.0` under round-to-nearest and would launder the
sign. That is gemm fixture F6a's lesson and DEVIATION 826's docstring.

### 7.1 The host scalars, computed ONCE per step per parameter group

    b1t       = pow_int_f32(beta1, t)                      (5.1)
    b2t       = pow_int_f32(beta2, t)
    bc1       = ftz(1.0 - b1t)
    bc2       = ftz(1.0 - b2t)
    step_size = ftz(identical_div(lr, bc1))
    rt_bc2    = ftz(identical_sqrt(bc2))
    c1        = ftz(1.0 - beta1)                           (5.3)
    c2        = ftz(1.0 - beta2)
    decay_mul = ftz(1.0 - ftz(identical_mul(lr, wd)))      AdamW only
    neg_lr    = -lr                                        SGD only, exact
    c_damp    = ftz(1.0 - dampening)                       SGD only

**CLAUSE 7.1. These are computed on the host and passed to the kernel as
`Float32` arguments. A kernel may not recompute them.** They WOULD agree
if it did, because every step is a pinned primitive -- but a per-element
recomputation is where a `powf` creeps in, it is `O(log t)` work per
element instead of per step, and having one producer is what makes "the
same scalars reached every element" a structural fact rather than a
check.

`t` is one-based. The first step of a run is `t = 1`.

### 7.2 Adam and AdamW, per element

    O1   g  = ftz(grad[i])                       already clipped, section 3
    O2   p  = ftz(param[i])
    O3   mp = ftz(m[i]) ; vp = ftz(v[i])

    O4a  ADAM  (coupled decay, only when wd != 0.0)
             g = ftz(identical_mul_add(wd, p, g))                 FUSED
    O4b  ADAMW (decoupled decay, only when wd != 0.0)
             p = ftz(identical_mul(decay_mul, p))                 PRODUCT
         -- and the gradient is NOT touched. Section 7.4.

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

Four decisions inside that sequence are contested, and each is named here
so nobody has to derive which line is a choice.

**CLAUSE 7.2a. The moment recurrences are a PRODUCT then an FMA**, which
mirrors `exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)` -- two
roundings, in that order. The alternative is `lerp`, `m + c1 * (g - m)`,
which recent PyTorch uses in some paths and which is a different number.
Sabotage `OPT_SAB_MOMENT_LERP`.

**CLAUSE 7.2b. The `v` term associates as `c2 * (g * g)`**, forming the
square first. `addcmul_(grad, grad, value = 1 - beta2)` does not say which
way it associates and `(c2 * g) * g` is the other reading. Sabotage
`OPT_SAB_SQ_ASSOC`.

**CLAUSE 7.2c. The bias correction is applied to the DENOMINATOR and to
the STEP SIZE, not to `m` and `v`.** O11 divides by `rt_bc2` and
`step_size` already carries `1/bc1`. The documented-pseudocode alternative
forms `m_hat = m / bc1` and `v_hat = v / bc2` explicitly and then
`p -= lr * m_hat / (sqrt(v_hat) + eps)`, which is one more rounding and a
different number. The profile takes the implementation shape because that
is what a real training run computes. Sabotage `OPT_SAB_MHAT_FORM`.

**CLAUSE 7.2d. The parameter update at O14 is ONE fused rounding.**
`fma(-step_size, q, p)`, not a rounded product followed by a rounded
subtract. Only fusion has a portable spelling (gemm contract section 4).

*What would make an O14 fusion sabotage pass while gating nothing.* This
is the most important non-vacuity note in the document. `check-ieee-arith`
scored Metal as UNFUSED over 2^20 HASHED patterns and the verdict was
WRONG -- **zero of those 2^20 patterns separate a fused `a*b + c` from an
unfused one**, because random exponents put the product and the addend so
far apart that both spellings round identically. The fixture must be
BUILT TO SEPARATE, which means `step_size * q` must have low bits that the
addition to `p` would otherwise discard, which means `p` and
`step_size * q` must be within a few binades of each other with the
product's tail nonzero. A random fixture will report a fused-versus-
unfused sabotage as INERT and the report will be false.

### 7.3 SGD with momentum, per element

    S1   g = ftz(grad[i])
    S2   p = ftz(param[i])
    S3   if wd != 0.0:  g = ftz(identical_mul_add(wd, p, g))      FUSED
    S4   if momentum != 0.0:
            if the buffer is UNINITIALIZED for this parameter:
                b = g                                     A COPY, NO ARITHMETIC
            else:
                bs = ftz(identical_mul(momentum, ftz(buf[i])))    PRODUCT
                b  = ftz(identical_mul_add(c_damp, g, bs))        FUSED
            store ftz(b)
            if nesterov:  g = ftz(identical_mul_add(momentum, b, g))  FUSED
            else:         g = b                           A COPY, NO ARITHMETIC
    S5   p = ftz(identical_mul_add(neg_lr, g, p))                 FUSED
    store ftz(p)

**CLAUSE 7.3a. The first momentum step is a COPY.** `b_1 = g`, not
`c_damp * g`. That is PyTorch's `buf = grad.clone()` and it means the
dampening factor is not applied on the step that creates the buffer.

*What would make a first-step sabotage pass while gating nothing.* At
`dampening = 0.0`, `c_damp` is exactly `1.0` and `identical_mul(1.0, g)`
returns `g` for every finite `g`, so **the default configuration cannot
see this clause at all.** The fixture must set `dampening != 0` and must
compare at `t = 1`, and then at `t = 2` and beyond to show the divergence
persists. Sabotage `OPT_SAB_MOMENTUM_FIRST_STEP`.

**CLAUSE 7.3b. The "is the buffer initialized" flag is CHECKPOINT STATE.**
A resume that reinitializes it recomputes `b` as a copy of the current
gradient instead of continuing the recurrence, and the two runs diverge at
that step and never reconverge. Section 11(d) and sabotage
`OPT_SAB_RESUME_REINIT`.

**CLAUSE 7.3c. Nesterov is `g + momentum * b`, in that operand order**,
matching `grad.add(buf, alpha = momentum)`. The transposed reading
`b + momentum * g` is a different number. Sabotage
`OPT_SAB_NESTEROV_ORDER`.

### 7.4 The order weight decay is applied in -- Adam against AdamW

This is the entire difference between the two algorithms and it is an
ORDER, not a coefficient.

| | where the decay lands | what it touches |
|---|---|---|
| Adam | into the GRADIENT, before the moments (O4a) | `g`, so it enters `m` and `v` and is itself smoothed and normalized |
| AdamW | onto the PARAMETER, before the update (O4b) | `p`, and the gradient is untouched, so the decay is not adapted by `v` |

**CLAUSE 7.4a.** AdamW's decay is a MULTIPLY, `p * (1 - lr*wd)`, matching
`param.mul_(1 - lr * weight_decay)`. The additive reading
`p - lr*wd*p` is a different number and is the other spelling in
circulation. Sabotage `OPT_SAB_DECAY_ADD_FORM`.

**CLAUSE 7.4b.** AdamW's decayed `p` is the base of O14. The decay
happens first and the Adam update lands on the decayed value, not on the
original.

*What would make an Adam-versus-AdamW sabotage pass while gating nothing.*
**At `weight_decay = 0.0` the two algorithms are the SAME ARITHMETIC**, so
the default configuration -- and `torch.optim.Adam`'s default is
`weight_decay = 0` -- cannot distinguish them. This is the single most
likely vacuous gate in the lane. Every weight-decay fixture must set
`wd != 0`, and it should set `wd` to a value where `lr * wd` is not a
power of two so 7.4a separates too. Sabotage `OPT_SAB_ADAMW_AS_ADAM`.

---

## 8. NaN, infinity and signed zero

**CLAUSE 8a. A non-finite value in a gradient, a parameter or a state is
REFUSED BY NAME before any recorded stage.** `refuse_nonfinite` in the
oracle, tested BY BITS and not by compares, because Metal flushes COMPARE
operands (row 49) so a compare-written test has two meanings.

The reason is row 39. NaN PAYLOADS are vendor shaped -- three payloads for
one IEEE answer were measured -- so a certified card may never contain a
computed NaN, and a cross-vendor gate would have to compare NaN cells as
"is NaN" rather than by bits, which is a hole in a bitwise claim.

**This has a real cost and the contract states it rather than pretending
otherwise.** Production training loops routinely SKIP a step whose
gradient norm is non-finite, and that behavior is useful. Under this
profile it is not arithmetic, it is a CONTROL decision, and it belongs
outside the pinned region as an explicit recorded branch -- a `skipped`
flag in the card, with the step index -- so that two runs can be compared
on whether they skipped the same steps. Folding it into the arithmetic as
a `select` would put a NaN inside a certified stage.

**CLAUSE 8b. `-0.0` is an admitted value everywhere and the seams preserve
it where the reference does.** `identical_mul`'s `-0.0` addend keeps a
negative zero product. The GEMM leaves seed `+0.0`, so a sum of squares of
an all-zero tensor is `+0.0` on every vendor by IEEE's
`(+0) + (-0) = +0`. A `-0.0` gradient reaching O7 gives `g*g = +0.0`, so
`v` never picks up a negative zero from a squared gradient.

**CLAUSE 8c. There is exactly ONE compare-select in the profile**, the
clamp at 3.4b, and it is on a value section 8a has already established is
finite and non-negative. So row 13's selection hazard -- `-0.0` and
`+0.0` comparing equal, order deciding which survives -- has no reachable
site in this lane. That statement is what makes `amsgrad`'s exclusion
(0.2) load bearing, since `max(v_max, v)` would create the first one.

---

## 9. Gradient accumulation -- what alignment the step requires of its caller

The optimizer step imposes NO alignment requirement on its own arithmetic.
It imposes one on whoever produced the gradient, and this section states
it because without it multi-step identical training is not available at
all and a trainer would not know what to arrange.

### 9.1 The measured fact this is built on

`IDENTICAL_BACKWARD_PLAN.md` 3.2 point 2 and 5.2, measured 2026-08-25 by
gate G5. A weight gradient `dB` contracts over the TOKEN dimension, so the
token count is the GEMM's `k`. An accumulation split at a token boundary
that is both a LEAF boundary and a SUBTREE boundary of v1's balanced tree,
accumulated with the contract's own flushed add, reproduces the unsplit
bits exactly.

    T = 512 split 256/256   ->  0 of 35 cells moved
    T = 384 split 256/128   ->  0 of 35 cells moved
    T = 300 split 150/150   ->  31 cells moved
    T = 512 split 200/312   ->  30 cells moved
    T = 384 split 192/192   ->  31 cells moved

### 9.2 The sufficient condition, stated as a predicate

**CLAUSE 9.2.** For `A` microbatches over `T` tokens, gradient
accumulation is bit-identical to the unsplit computation when ALL of the
following hold. `L` is `contract_leaf_size(T)` from
`gemm/mojo_only/gemm_oracle.mojo`.

1. `contract_leaf_size(T / A) == contract_leaf_size(T)`. The leaf rule
   holds `L` at 128 for every `k` in `(128, 131072]`, so this says both
   the full token count and the microbatch size land in the same band of
   the rule.
2. `T mod L == 0`. No ragged last leaf, so `P = T / L` exactly.
3. `A` divides `P`.
4. `A` is a power of two.
5. The cross-microbatch combination is the v1 BALANCED TREE over the `A`
   pieces in ascending microbatch index, with `ftz(ftz(x) + ftz(y))` at
   every node -- not a running serial sum.

`optimizer_oracle.mojo::microbatch_split_is_identical(t_tokens, a)`
computes this predicate on the host, so a trainer can ask rather than
guess, and a gate can enumerate the `A` values that qualify at a given
`T`.

Conditions 3 and 4 together are what make each microbatch's leaf range a
COMPLETE SUBTREE. With `A` a power of two and `P` divisible by `A`, the
`A` piece roots are exactly the nodes of level `log2(A)` of the unsplit
tree, and the tree over them is the remainder of the unsplit tree. That is
the argument 5.2 works at `A = 2` and this is its generalization.

### 9.3 The sharpening the existing measurement cannot supply

**Clause 9.2 condition 5 is NOT established by the measured evidence, and
saying so is the point of this subsection.** Every measured aligned case
in 9.1 is `A = 2`. **Over two pieces a serial running sum and a balanced
tree are the same operation**, so the G5 measurement cannot distinguish
them and a reader who concludes "the accumulator just has to be the
flushed add" is over-reading it.

At `A = 4` they separate. The tree computes
`ftz(ftz(p0+p1) + ftz(p2+p3))`; a running sum computes
`ftz(ftz(ftz(p0+p1) + p2) + p3)`. **`T = 512` with `A = 4` is the fixture
that closes this**, and it is inside the range the gemm device check
already sweeps. Sabotage `OPT_SAB_MICROBATCH_SERIAL`.

### 9.4 The consequence for a training run's identity

**The microbatch count is part of a run's numerical specification unless
clause 9.2 holds.** A run at `A = 3` and the same run at `A = 1` are two
different numerical experiments, and a reproducibility claim must name
`A`. When 9.2 does hold, `A` is free, which is a designable property and
is cheap to arrange -- and it is the thing that makes identical training
across different hardware budgets possible at all.

---

## 10. The stages, in card order

One record per stage per step, hashed by `core/identity_trace.mojo`'s
FNV-1a64 over the buffer's little-endian bytes in index order. Tags carry
no machine property. `J` is the number of parameter tensors, `N_j` the
element count of tensor `j`.

    step.t              [1] Int      the step index, one-based
    input.param.<j>     [N_j]        as given
    input.grad.<j>      [N_j]        as given, before clipping
    clip.sumsq          [J]          3.2, one v1 GEMM per tensor
    clip.norm           [J]          3.2
    clip.total_sumsq    [1]          3.3, one v1 GEMM over the J norms
    clip.total_norm     [1]          3.3
    clip.coef           [1]          3.4b, after the clamp
    clip.grad.<j>       [N_j]        3.4c, the rescaled gradient
    sched.pow1          [1]          5.1
    sched.pow2          [1]          5.1
    sched.bc1           [1]          7.1
    sched.bc2           [1]          7.1
    sched.step_size     [1]          7.1
    sched.rt_bc2        [1]          7.1
    sched.decay_mul     [1]          7.1, AdamW only
    adam.gwd.<j>        [N_j]        O4a, Adam with wd != 0 only
    adamw.pdec.<j>      [N_j]        O4b, AdamW with wd != 0 only
    adam.m.<j>          [N_j]        O6
    adam.v.<j>          [N_j]        O9
    adam.denom.<j>      [N_j]        O12
    adam.q.<j>          [N_j]        O13
    sgd.buf.<j>         [N_j]        S4, SGD with momentum only
    sgd.dir.<j>         [N_j]        S4, the direction actually stepped
    param.out.<j>       [N_j]        O14 or S5

The `sched.*` stages are one-element buffers and they are in the card on
purpose. They are the cheapest possible early-divergence address -- if a
cross-vendor run differs at `sched.pow2` the problem is section 5 and not
a single one of the millions of elementwise updates downstream.

---

## 11. What "identical" is gated to mean

(a) **Device equals host oracle, bitwise, at every stage and every
shape.** The oracle is normative. Asserted under IDENTICAL; RECORDED, not
asserted, under FAST, for the reason the metrics lane's leg-11 lesson
gives.

(b) **Launch invariance.** The same bits on eight repeated launches, and
the same bits across a sweep of grid and block geometries. For the
elementwise update this is structural -- no float crosses a thread
boundary -- and the gate verifies what the construction promises. For the
clip's reductions it is inherited from the v1 GEMM, whose own
`check_device_is_launch_invariant` covers it.

(c) **Parameter-count invariance, WITH CLIPPING OFF.** One tensor's
`param.out` bits are identical whether it is stepped alone or alongside
two others. With clipping ON this is false by construction and section 3.5
says why.

(d) **STEP-COUNT INVARIANCE.** A run that reaches step `t` continuously
and a run resumed from a checkpoint written after step `t - 1` produce
bitwise identical `param.out`, `adam.m`, `adam.v` and `sgd.buf` at step
`t`. The checkpoint must carry, and the gate must round-trip, all of --
`t`; every parameter, gradient-independent state and hyperparameter bit
pattern; the momentum-buffer-initialized flag per parameter (7.3b); the
`param_id` assignment and its order (3.3); and any learning-rate
schedule's own state if the schedule is not a pure function of `t` (0.2).
The gate runs `2t` steps continuously, checkpoints at `t`, resumes, and
compares at `2t`. `t = 8` at minimum, because section 5.1's spellings
agree through `t = 6`.

(e) **Microbatch-alignment behavior.** At an `A` satisfying clause 9.2,
zero cells move. At an `A` that does not, the number of cells that move is
REPORTED, not asserted, and the host oracle is what says the number before
the device is asked. `A = 4` is required (9.3).

(f) **The row-39 audit of section 8**, including that the refusal path
actually executes -- an unreached refusal is an untested guard.

(g) **Every clause above falsifiable by a named sabotage that FAILS a
gate**, and every sabotage accompanied by the fixture property that makes
it non-vacuous. Section 12.

`training/mojo_only/optimizer_check.mojo` is the gate file. ~~**It does not
exist.**~~ **It was written at `ecd1a436` and RAN at `b90f52ab`, correction
made 2026-08-31.** Six clauses on one device, and clause (f) measured a real
device-side defect that DEVIATION 1496 fixed. Section 16. **One vendor, not
three.**

---

## 12. The sabotage set

Each is a comptime switch read by `is_defined`, off in every build that
does not name it, following `gemm/mojo_only/gemm_identical.mojo`'s
convention. The "must not pass on" column is the fixture property without
which the arm goes inert, and it is the column to read first.

| switch | what it breaks | clause | must not pass on |
|---|---|---|---|
| `OPT_SAB_POW_RUNNING` | `beta^t` by running product | 5.1 | `t <= 6`. The two agree exactly there. Run to `t >= 8`, ideally 1000 |
| `OPT_SAB_POW_EXPLOG` | `beta^t` via `portable_powf` | 5.1 | nothing; it separates at `t = 1`. Cheap arm, keep it |
| `OPT_SAB_EPS_INSIDE_SQRT` | `sqrt(v_hat + eps)` | 4d | `v` near 1e-4. Plant `v` in 1e-20 to 1e-12 |
| `OPT_SAB_RSQRT` | `identical_rsqrt` or the intrinsic instead of sqrt-then-div | 4b | a handful of round `v`. Needs thousands of hashed `v`, on all three columns |
| `OPT_SAB_RECIP_MUL` | `m * (1/dn)` | 4c | power-of-two denominators. It is EXACT there |
| `OPT_SAB_ADAMW_AS_ADAM` | decay into the gradient instead of the parameter | 7.4 | `wd == 0`. The two algorithms are then identical |
| `OPT_SAB_DECAY_ADD_FORM` | `p - lr*wd*p` instead of `p*(1-lr*wd)` | 7.4a | `lr*wd` a power of two; tiny `p` |
| `OPT_SAB_MOMENT_LERP` | `m + c1*(g - m)` | 7.2a | `g` and `m` in the same binade with no low bits |
| `OPT_SAB_SQ_ASSOC` | `(c2*g)*g` | 7.2b | round `g`. Needs hashed `g` |
| `OPT_SAB_MHAT_FORM` | explicit `m_hat`, `v_hat`, `lr *` at the end | 7.2c | nothing known to make it inert, but it is a 1-ulp-class arm -- report the cell count, do not assume |
| `OPT_SAB_UNFUSED_UPDATE` | O14 as a rounded product then a rounded add | 7.2d | **random exponents. ZERO of 2^20 hashed patterns separated fused from unfused in `check-ieee-arith`. The fixture must be BUILT to separate** |
| `OPT_SAB_FTZ_LATE` | flush only at the store | 6 | gradients near 1.0. Plant one near 1e-25 so the SQUARE is subnormal |
| `OPT_SAB_CLIP_SKIP_AT_ONE` | skip the rescale when `coef_c == 1.0` | 3.4c | normal gradients. Plant a SUBNORMAL cell, and compare `clip.grad`, not `param.out` |
| `OPT_SAB_CLIP_FLAT_NORM` | one flat sum of squares | 3.1 | `J == 1`. Needs `J >= 3` with norms several binades apart |
| `OPT_SAB_CLIP_PARAM_ORDER` | reverse the `param_id` order | 3.3 | **`J == 2`. A swap of two children of one node is bitwise inert. Needs `J >= 3`, prefer 5** |
| `OPT_SAB_CLIP_SERIAL_FOLD` | hand-written serial fold instead of the v1 GEMM | 3.2 | **every `N <= 128`. `P == 1` there and the tree is empty. Needs `N > 128`, and `N = 300` for the ragged leaf** |
| `OPT_SAB_CLIP_BLOCK_PARTITION` | the clip partition reads `block_dim` | 3.2 | a single launch geometry. Needs the geometry sweep |
| `OPT_SAB_MICROBATCH_SERIAL` | serial running accumulation across microbatches | 9.2(5) | **`A <= 2`. A tree and a serial sum coincide over two pieces. Needs `A = 4`** |
| `OPT_SAB_MOMENTUM_FIRST_STEP` | `b_1 = c_damp * g` | 7.3a | **`dampening == 0`, which is the default. Needs `dampening != 0`** |
| `OPT_SAB_NESTEROV_ORDER` | `b + momentum*g` | 7.3c | `momentum == 0`; also inert when `b == g`, so not at `t = 1` |
| `OPT_SAB_RESUME_REINIT` | drop the buffer-initialized flag on resume | 7.3b, 11(d) | `momentum == 0`; needs a checkpoint at `t >= 1` and a compare at `t + 1` |
| `OPT_SAB_SCALARS_PER_ELEMENT` | recompute `pow_int_f32` inside the kernel | 7.1 | **everything. This arm is EXPECTED to be bit-inert, because the recomputation uses the same pinned primitives. It is in the set as a REACH probe, not a bit probe -- it proves the kernel could reach the scalars, and its bit result must be reported as INERT and not as a pass** |

The last row is deliberate. `sabotage-when-required` says a sabotage is
required when a bound was chosen or a path is new, and clause 7.1's ban is
a DESIGN rule rather than a numerical one. An arm whose predicted result
is "no bits move" is worth having only when it is labelled that way in
advance, which is what this row does.

---

## 13. Clause-to-code index

| clause | function | file |
|---|---|---|
| 1 | `OptimizerConfig`, `OPT_*` constants | `training/mojo_only/optimizer_oracle.mojo` |
| 3.2 | `clip_tensor_sumsq_oracle` | `training/mojo_only/optimizer_oracle.mojo` |
| 3.2 device | `identical_clip_grad_norm` | `training/mojo_only/optimizer.mojo` |
| 3.3 | `clip_grad_norm_oracle` | `training/mojo_only/optimizer_oracle.mojo` |
| 3.4 | `clip_coefficient` | `training/mojo_only/optimizer_oracle.mojo` |
| 4, 6 | `identical_sqrt`, `identical_div`, `identical_mul`, `identical_mul_add`, `ftz` | `mojo_only/numerics.mojo` |
| 5.1 | `pow_int_f32` | `training/mojo_only/optimizer_oracle.mojo` |
| 7.1 | `StepScalars`, `step_scalars` | `training/mojo_only/optimizer_oracle.mojo` |
| 7.2 | `adam_element_oracle` | `training/mojo_only/optimizer_oracle.mojo` |
| 7.2 device | `adam_update_kernel` | `training/mojo_only/optimizer.mojo` |
| 7.3 | `sgd_element_oracle` | `training/mojo_only/optimizer_oracle.mojo` |
| 7.3 device | `sgd_update_kernel` | `training/mojo_only/optimizer.mojo` |
| 8a | `refuse_nonfinite` | `training/mojo_only/optimizer_oracle.mojo` |
| 9.2 | `microbatch_split_is_identical` | `training/mojo_only/optimizer_oracle.mojo` |
| NORMATIVE | `optimizer_step_oracle` | `training/mojo_only/optimizer_oracle.mojo` |
| 12 | `optimizer_sabotage_name` and the `SAB_*` switches | `training/mojo_only/optimizer.mojo` |
| every clause's fixture | `check_*` | `training/mojo_only/optimizer_check.mojo` -- ~~**DOES NOT EXIST**~~ **EXISTS AND RAN, `ecd1a436` + `b90f52ab`, ONE DEVICE ONLY** (corrected 2026-08-31) |

---

## 14. Not claimed

1. **Not "bit-identical training".** This profile pins ONE optimizer step
   given a gradient. `IDENTICAL_BACKWARD_PLAN.md` 3.1 is an AND of every
   arrow in the ladder, and T3 through T7 and T10 are not built. Closing
   T9 closes T9.
2. **No PyTorch parity, and two clauses make it impossible.** Section 5.2
   (the bias correction is not float64 here) and section 5.3 (`1 - beta2`
   differs by about 1.3e-5 relative). A gate that compares against PyTorch
   is measuring the gap, not the profile.
3. **Nothing cross-vendor.** No leg has run. The v1 GEMM this contract
   delegates its reductions to HAS run on three vendors at leg 11, and
   that measurement is the GEMM's, not this lane's.
4. ~~**Nothing has been compiled.**~~ **Compiled and run on one device,
   `ecd1a436` and `b90f52ab`. See the corrected status banner.** What is not
   done is item 3 above, the cross-vendor leg.
5. **No performance number**, and no claim about what the pins cost.
   `IDENTITY IS NOT FREE` is a standing rule -- conforming costs on every
   vendor, and this lane has priced nothing. In particular the per-element
   `ftz` at fourteen seams and the `sqrt`-plus-`div` in place of one
   `rsqrt` are real instruction counts and nobody has counted them.
6. **No NaN payload claim.** Section 8a refuses non-finite inputs instead.
7. **No claim that the clip's two-level norm is more accurate than a flat
   one.** It is the reference's shape. It is probably slightly worse.
8. **No amsgrad, no foreach or fused apply, no mixed precision, no
   sharded or multi-GPU optimizer, no stateful learning-rate schedule.**
   Section 0.2.
9. **Single GPU.** Section 0.2 and `IDENTICAL_BACKWARD_PLAN.md` 4.4.
10. **The PyTorch spellings cited here are cited from memory**, not from a
    checkout. Section 0.3's health warning and section 16.

---

## 15. Deviation block

Range 1170 to 1189, assigned to this lane.

| # | what |
|---|---|
| 1170 | The profile `mojolearn.identical.optimizer.fp32.v1` -- FP32 master weights and state, three algorithms plus the clip pass, hyperparameters as bit patterns |
| 1171 | `pow_int_f32` -- `beta^t` as LSB-first binary exponentiation over `identical_mul`, a pure function of the integer `t`; NOT a running product, NOT `portable_powf`, NOT PyTorch's float64 `beta ** t`. The clause that makes step-count invariance structural |
| 1172 | `eps` OUTSIDE the square root, added to the bias-corrected denominator, with `step_size = lr/bc1` and `rt_bc2 = sqrt(bc2)` as host scalars |
| 1173 | Every division a true `identical_div`, never a reciprocal-multiply; `rsqrt` never called in either mode |
| 1174 | The moment recurrences as PRODUCT then FMA, and the `c2 * (g*g)` association |
| 1175 | The parameter update as ONE fused rounding, `fma(-step_size, q, p)` |
| 1176 | AdamW's decoupled decay as `p * (1 - lr*wd)` applied BEFORE the moments and never to the gradient; Adam's coupled decay as `fma(wd, p, g)` |
| 1177 | The SGD-with-momentum spelling, including the first-step COPY `b_1 = g` and the Nesterov operand order |
| 1178 | The per-tensor sum of squares IS a v1 GEMM call at `m = n = 1`, `k = N`, `OP_NT`, not a fold written in this lane |
| 1179 | The cross-tensor combination is the reference's TWO-LEVEL form over ASCENDING `param_id`, and `param_id` order is part of the profile |
| 1180 | `CLIP_EPS = 1e-6` as a profile constant, the clamp spelling, the UNCONDITIONAL rescale, and the refusal of a non-finite total norm |
| 1181 | The microbatch alignment predicate of clause 9.2, and the sharpening that condition 5 needs `A = 4` because the measured `A = 2` cases cannot see it |
| 1182 | The `ftz` seam table of section 6, including O7's squared-gradient case |
| 1183 | The checkpoint contents that make step-count invariance checkable, including the momentum-buffer-initialized flag and the `param_id` assignment |
| 1184 | The device execution plan -- one thread per element, block size a scheduling row, the clip's reductions delegated entire to `identical_gemm` |
| 1185 | The FAST arms -- same code, same partition, same order, both pins compiled away, no identity claim |
| 1186 | The sabotage set of section 12 and its non-vacuity column. RESERVED for the gate file, which is not written |
| 1187 | RESERVED, unused |
| 1188 | RESERVED, unused |
| 1189 | RESERVED, unused |

---

## 16. OWED, AND WHY I DID NOT DO IT HERE

This lane's write set was exactly three new files --
`training/mojo_only/optimizer.mojo`, `training/mojo_only/optimizer_oracle.
mojo` and this document. Everything below needs a file outside that set
and is therefore described rather than done.

1. ~~**`training/mojo_only/optimizer_check.mojo`.** The gate file. Every
   sabotage in section 12 and every fixture property in its last column
   is specified and none is implemented. **Without it this contract is
   prose and nothing in it has been falsified.** This is the largest owed
   item by a wide margin.~~ **PAID, corrected 2026-08-31.** The gate file was
   written at `ecd1a436` and RAN at `b90f52ab`. This contract is no longer
   prose: six clauses were falsifiable and passed, the sabotage arms were
   built and fired, and clause (f) found a real defect in the device entry
   point (DEVIATION 1496). Two arms could NOT be made non-vacuous and that is
   recorded as DEVIATIONS 1493 and 1613 rather than papered over. **What
   replaces this as the largest owed item is item 3 of section 15: the
   CROSS-VENDOR LEG. Everything above was measured on ONE DEVICE.**
2. **A `pixi.toml` task.** `check-optimizer`, and the sabotage arms as
   `-D MOJOLEARN_OPT_SABOTAGE_*` builds, following the
   `check-portable-nn` and `gemm_device_check` patterns. `pixi.toml` is
   outside the write set.
3. **`IDENTITY_PATHS.md` rows.** This lane needs rows for the optimizer
   step and for the global-norm clip, in the file's PIN / REPLACE / REFUSE
   form. The next free row number is 60 at the time of writing, but the
   transformer and loss lanes are writing rows concurrently, so the number
   must be taken at commit time and not reserved here.
4. **`IDENTICAL_BACKWARD_PLAN.md` section 4.2 and the T9 row.** They say
   "the composition and the operation ORDER are unwritten". That is now
   false and the sentence should be replaced with a pointer here, in the
   same commit that lands this file, per `fix-docs-on-discovery`.
5. **A PyTorch source read.** There is no PyTorch checkout under
   `/Users/andrewhendel/CascadeProjects/upstream/`. Every spelling in
   sections 3.1, 7.2, 7.3 and 7.4 is cited from memory of the documented
   pseudocode and of `torch/optim/adam.py`'s single-tensor path, which
   violates `read-their-source-against-ours`. Clone `pytorch/pytorch` at a
   pinned tag, mirror `torch/optim/{sgd,adam,adamw}.py` and
   `torch/nn/utils/clip_grad.py` into `training/ported/`, and re-check
   every "the reference spells it" sentence. **Where the source disagrees
   with a clause here, the CLAUSE is what stands -- the profile is
   normative and PyTorch is the design reference -- but the citation must
   be corrected in the same commit and the deviation renumbered as a
   deliberate departure rather than an accidental one.**
6. **A `kernel_matrix.mojo` row for the optimizer's block size.**
   `OPT_TPB` is currently a literal 256 in `optimizer.mojo`. Block size is
   a SCHEDULING row and free in both modes, so no bit depends on it -- but
   `column_max_block_size(COLUMN_SPEC_BASELINE)` is 128, so a literal 256
   would be REFUSED on the portable-floor column rather than resolved
   downward. `mojo_only/kernel_matrix.mojo` is outside the write set.
7. **A shared home for `refuse_nonfinite`.** `mamba/mojo_only/
   mamba_oracle.mojo:57` carries a function of the same name and the same
   body, and this lane now carries a third. `mojo_only/numerics.mojo` is
   the canonical home for row-39 helpers and is outside the write set.
   Three copies of one predicate have three chances to drift, which is
   `identical_mul`'s own docstring's complaint about `pinned_mul`.
8. **`PORTED_MAP.tsv` and `UNPORTED.tsv` entries for `training/`.** The
   directory has no mapping rows. Neither file is in the write set.
9. **`training/__init__.mojo` and `training/mojo_only/__init__.mojo`
   already exist and were NOT touched**, per the lane brief.
10. **A corpus.** The mamba lane's `mamba/corpus` is the model -- planted
    adversarial cases with names (`adv_signed_zeros`, `adv_softplus_
    guard`). This lane needs `adv_subnormal_square` (section 6),
    `adv_dead_unit_v` (clause 4d), `adv_dampening_first_step` (7.3a),
    `adv_param_order_five` (3.3) and `adv_pow_step_1000` (5.1). None
    exists.
