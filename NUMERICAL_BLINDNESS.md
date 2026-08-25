# NUMERICAL BLINDNESS

**When two different spellings of the same arithmetic agree bit for bit, and
why conventional numerical hygiene steers a test author straight into the
inputs where they do.**

DEVIATIONS 1600-1649. Written 2026-08-25.

---

## 0. STATUS. NOTHING HERE HAS BEEN EXECUTED

**This file has never been run and neither has `tools/inert_condition_audit.py`,
its companion.** No number below came from running a gate, a build, a
benchmark or the auditor. Every count in this document was obtained by
READING the tree -- `grep`, `sed`, and the arm doc blocks themselves -- and
every derived condition was derived on paper. The measurements quoted in
section 2 and section 5 were made by OTHER lanes and are cited to the file
that records them; this file measured nothing.

The auditor is a REPORTING tool. It does not gate, it never raises on an
uncharacterized arm, and it exits 0 whatever it finds. It also cannot verify
anything -- see section 8.

---

## 1. THE CLAIM UNDER TEST

This repository gates bitwise identity with SABOTAGE ARMS. An arm is a
`comptime` switch that changes exactly one numerical decision, and a gate is
supposed to detect the change. That is mutation testing (DeMillo, Lipton and
Sayward, 1978) and the classical obstacle is the EQUIVALENT MUTANT -- a mutant
no test can kill because it is semantically identical to the original.

**The arms here are not equivalent mutants.** `scale * (q . k)` and
`sum_i (scale * q_i) k_i` are genuinely different functions. They coincide only
ON PARTICULAR INPUTS. The claim to be tested is twofold.

1. Those inputs form a CHARACTERIZABLE class, often in closed form.
2. Worse, conventional numerical-testing hygiene steers authors directly into
   that class. Every blinding condition in section 2 is a property a careful
   numerical author WANTS.

The second half is the inversion, and it is the only part that is a candidate
for novelty. Section 9 gives the honest verdict, which is narrower than the
framing above and is partly negative.

**The failure mode this document exists to name.** An arm fired on a blind
fixture reports GREEN. Nobody logs anything. The arm is then counted as
coverage, and the clause it was supposed to gate is licensed by a measurement
that could not have come out any other way. That is strictly worse than having
no arm, because an absent arm is visible in a census and a vacuous one is not.

---

## 2. THE TAXONOMY

Each row gives the algebraic condition under which two spellings of the same
quantity agree BIT FOR BIT in FP32 round-to-nearest-even, and the repository
instance where it bit us, where one exists. Rows marked ABSTRACT have no
measured instance in this tree and are derivations only.

### 2.1 Operands exactly representable -- collapses FMA contraction against separate multiply-add

**Condition.** `fma(a, b, c) == fl(fl(a*b) + c)` whenever `a*b` is exactly
representable in FP32, because then `fl(a*b) == a*b` and the fused form's
single rounding and the unfused form's second rounding round the same real
number. Sufficient (not necessary) condition on the inputs: `a` and `b` each
carry at most 12 significant bits, so their product carries at most 24.

**Instance. DEVIATION 1147, MEASURED.** `gemm/mojo_only/gemm_unpinned_price.mojo`'s
first fixture emitted integers scaled by `2^-4`. Every product was exactly
representable, so FMA contraction was bit-neutral and `ftz` could never fire.
**Sixteen of twenty rows had two genuinely different kernels produce IDENTICAL
BITS.** The timing half of the experiment was unaffected (1.554x then, 1.522x
after the fix) and the bit half was vacuous. It was visible at all only
because the driver reports bit equality as INERT rather than as agreement.

**Instance. `SAB_UNFUSED_UPDATE`, STATED, not yet measured.** The optimizer
lane calls this "the most important inert case in the file" and gives the
reason. `check-ieee-arith` scored Metal as UNFUSED over `2^20` HASHED
patterns and the verdict was WRONG, because ZERO of those patterns separate a
fused `a*b + c` from an unfused one -- random exponents put the product and
the addend so far apart that both spellings round identically. **A random
fixture reports this arm inert and the report is false.** This is the
absorption row (2.4) and this row acting together, and it is the cleanest
demonstration in the tree that sample size does not help.

### 2.2 All values in one binade -- collapses a balanced tree against a serial chain

**Condition.** A reassociation of `sum_i x_i` is bitwise inert exactly when
every partial sum reached by BOTH orders is exactly representable. A
sufficient condition an author can check on the host: all `x_i` in
`[2^e, 2^(e+1))` with at most `24 - ceil(log2 n)` significant bits each, so no
prefix sum needs more than 24.

**Instance. DEVIATION 1493, MEASURED.** `optimizer_check` clause (e) used four
partials all in `[1, 2)`. A balanced tree and a serial chain agreed. Replaced
with `2^24, 1, 1, 1`, which separate by one ULP -- the smallest possible
separating fixture for this row, and it is four numbers.

**Instance. `SAB_S1_FOLD_DESCENDING`, MEASURED, and it is the useful shape of
this row.** The mamba block's ascending-vs-descending sum of squares bit **1 of
4 rows at `d_model = 8` and 3 of 4 at `d_model = 16`**. The blindness is PER
ROW and it shrinks with fold length rather than vanishing at any threshold.
Eight terms was simply too short a fold for most rows to round differently in
the two directions.

**The trap inside this row.** The bound is TREE-SPECIFIC and the two lanes in
this tree do not have the same tree. See DEVIATION 1603 in section 6.

### 2.3 No subnormal reachable -- collapses `ftz` against no `ftz`

**Condition.** `ftz(x) == x` for every `x` with `|x| >= 2^-126` and for every
zero, infinity and NaN. So a flush pin is bitwise inert on any fixture whose
INTERMEDIATES all stay normal -- and intermediates, not inputs, is the whole
difficulty, since `x*x` for `x ~ 1e-25` is not representable as a normal.

**Instance. DEVIATION 1147 again**, same fixture, second mechanism. `ftz`
never fired because nothing underflowed.

**Instance. `SAB_FTZ_LATE`, STATED.** "INERT on any fixture whose gradients are
within a few binades of 1.0. Plant a gradient near 1e-25, where the VALUE is a
perfectly ordinary normal and the SQUARE is not representable as one." That
sentence is the closed form of this row and it was written by the lane, not by
this file.

**The vendor twist, and it is why this row is not simply a fixture problem.**
`SAB_NO_FLUSH_ACC` in the embedding lane is **inert on Apple entirely**,
because Metal flushes anyway, so the pinned and unpinned spellings agree there
by hardware. Its sibling `SAB_GATHER_NO_FLUSH` is NOT inert on Apple, because
a gather performs no arithmetic and a raw copy of a subnormal survives even an
FTZ backend. Two arms, one clause, opposite Apple visibility. A single-vendor
column cannot distinguish "the pin works" from "the hardware made the pin
redundant".

### 2.4 One term dominates (absorption) -- collapses dropping a term

**Condition.** `fl(a + b) == a` whenever `|b| <= ulp(a)/2` and the tie rounds
to `a`, i.e. roughly `|b| / |a| < 2^-24`. A term smaller than the accumulator
by 24 binades is invisible, and so is any change to it.

**Instance. DEVIATION 1492, MEASURED.** `loss_check` clause (d)'s control
dropped a tail exponential and did not move. **Four GEMM plans agreeing would
have read as a strong result and meant nothing.**

**Instance, and the strongest one in the tree. The mamba lane's block
measurement.** At `d_model = 16`, with THIRTEEN of sixteen stages moved and
`out_proj.out` differing on 23 of 64 cells, **`residual.out` is STILL bitwise
identical**, because the residual add puts an `out_proj` of order 1e-3 beside
an input of order 1. At the shape where the arm is STRONGEST, an output-only
gate calls it inert. Four of the six mamba block arms are invisible in the
block's output at that shape.

`S14_THRESHOLD_10` is the miniature of it -- 13 cells of `softplus.out`, 13 of
`scan.y`, ONE cell of `skip.out`, zero of `gate.out`, `out_proj.out` and
`residual.out`.

### 2.5 Divisor a power of two -- collapses reciprocal-multiply against true divide

**Condition.** `x * (1/d) == x / d` bitwise whenever `1/d` is exactly
representable, which for a finite normal `d` holds exactly when `d = 2^j`
(and, degenerately, for the values where both sides are zero or infinite).
Everywhere else the reciprocal form rounds twice and the divide once.

**Instance. STATED in three lanes and DERIVED in a fourth.**
`SAB_MEAN_RECIPROCAL_MUL` and `SAB_GRAD_RECIPROCAL_MUL` (loss) and
`SAB_RECIP_MUL` (optimizer) all state it. The loss lane names the cheapest
separating divisor -- `count = 3`. **`SAB_S18_RECIPROCAL_MUL` in the
transformer block does not state it** (DEVIATION 1605).

**And the honest other half.** `optimizer_check` records that it could NOT
build a separating INERT fixture for `SAB_RECIP_MUL`, because the Adam
denominator is `ftz(ftz(div(ftz(sqrt(v)), rt_bc2)) + eps)` and every term is
data dependent. **The condition is in closed form and is still not reachable
by construction.** Closed form does not imply constructible, and that is a
real limit on the claim in section 1.

### 2.6 Scale a power of two -- collapses scale-into-operand against scale-after

**Condition.** For `s = 2^j` and any operands that do not over- or underflow,
`s * fl(sum_i fl(q_i k_i)) == fl(sum_i fl((s q_i) k_i))`, because
multiplication by a power of two is exact and therefore commutes with every
product and every partial sum. The two spellings agree on **ALL inputs**, not
merely on a lucky fixture. For `s` not a power of two they differ generically.

**Instance. DEVIATION 1102, MEASURED, and it is the model instance for this
whole document.** `S12_SCALE_INTO_Q` was pointed at `head_dim = 16`, where the
attention scale is `1/sqrt(16) = 0.25` exactly. It moved NOTHING and the check
raised, which was the correct outcome. Repointed at `head_dim = 24`, where
`1/sqrt(24) = 0x3e5105eb` is inexact, it moved 15 of 30 stages and stayed
inert on the original -- so the arm now carries a two-sided reach proof.

**The generalization the lane did not state.** The condition is not "head_dim
16". It is `1/sqrt(head_dim) = 2^j`, i.e. **`head_dim` a power of FOUR** --
1, 4, 16, 64, 256. `head_dim = 64` is the most common value in a real model
and it is blind. `head_dim = 128` is not. The backward twin
`SAB_B12_SCALE_INTO_DQ` DOES state the power-of-four form.

**The preflight already knew.** `transformer_check`'s own preflight printed
`head_dim 16 -> 0x3e800000 ... EXACT, blind to the spelling`. The expectation
table did not use it. **The gap between a preflight that computes the
separating condition and an expectation table that ignores it is exactly what
firing the arm found**, and it is the single strongest piece of evidence for
the mechanization proposal in section 10.

### 2.7 Fewer than three terms -- collapses any fold order

**Condition.** `fl(a + b) == fl(b + a)` for every `a` and `b` including both
zero signs, so a reassociation over two terms is inert unconditionally. For an
ADJACENT-PAIR tree the bound is stronger -- at three terms the tree IS the
chain node for node -- but that extension is spelling-specific.

**Instance. `SAB_FOLD_BALANCED_TREE`, STATED with a proof.** "INERT AT EVERY
RUN OF LENGTH <= 3, PROVABLY. At `R = 2` and `R = 3` the balanced tree IS the
ascending chain, node for node." And the operational half, which is the part
worth copying -- "**`R_max` is a property of the DATA**, so a gate must ASSERT
`emb_max_run_length >= 4` rather than hope for it."

**The exception that breaks the naive generalization.**
`SAB_B01_DOT_DESCENDING` states "INERT at `d_model == 1` ONLY. **NOT inert at
`d_model == 2`**: an fma keeps the second product exact, so a two-term fma
chain is order dependent where a two-term ADD chain is not." The two-term rule
holds for `+` and fails for the fma chain. Any tool that applied "fewer than
three terms is inert" as a blanket rule would be wrong here.

### 2.8 A single element -- collapses any reduction seed

**Condition.** A seed `s` is invisible whenever `fl(s + x) == x`. For
`s = +0.0` that is every `x` except `x = -0.0`, where `fl((+0.0) + (-0.0))` is
`+0.0` in round-to-nearest and the seedless answer is `-0.0`. So a seed clause
is falsifiable at exactly one input class -- a cell whose SOLE contributor is a
negative zero.

**Instance. `SAB_SEED_SEEDLESS` and `SAB_SINGLE_RUN_BYPASS`, both STATED, and
the embedding lane keeps BOTH on purpose** -- "they are two different wrong
spellings that agree on the same single input, and a gate that carried only
one would not know which clause it had proved."

**Instance. DEVIATION 1490, MEASURED, and it is this row inverted.**
`base_n1_v1` was declared `L_MAX_SEED_ZERO`'s INERT case, when at `V == 1` the
arm actually FIRES. **The claim was backwards.** A stated condition is not a
checked condition.

### 2.9 Two rows the taxonomy in the brief did not have, and both have measured instances

**(i) A HARNESS-CONDITIONED inert case.** `SAB_IGNORED_ROW_SKIPPED` and
`SAB_EMPTY_ROW_SKIPPED` skip a store rather than writing it. They are **ALWAYS
inert if the gate pre-fills the output with zeros and never inert if it
POISONS.** The blinding condition is a property of the TEST HARNESS, not of
the data, and a fixture audit -- including this one -- structurally cannot see
it. `emb.dw` at the shipped shape is 124,160 of 128,256 empty rows, so this is
most of the output. **DEVIATION 1608.**

**(ii) A PROBABILISTIC condition asserted as an invariant. DEVIATION 1491,
MEASURED.** `assert_row_has_a_positive_logit` asserted `L_MAX_SEED_ZERO`'s
inert condition as though it were structural. A row of 8 hashed logits is
all-negative about 1 in 256 times, so over 300 rows the assertion holds about
31% of the time. **The inert condition was correct and its status was wrong.**
An inert condition that holds with probability `1 - p` per row is not an inert
condition, it is a flake, and it is the hardest kind to read because the arm
simply looks inert.

**(iii) The control's own prose containing its refutation. DEVIATION 1494,
MEASURED.** A control that "forgot the step index" was applied to SGD, which
never reads the step index -- bias correction is an Adam mechanism. Not a
numeric blindness at all. A REACH blindness, and it is worth listing beside
the others because it produces the identical symptom, a control that does not
move.

---

## 3. THE PER-ARM TABLE

115 arms across five lanes, every `comptime` sabotage switch in `gemm/`,
`mamba/`, `transformer/`, `training/` and `embedding/`. See section 4 for why
this is 115 and not the 127 the brief expected.

Classification is exactly one of FIRED / INERT-STATED / INERT-DERIVED /
UNKNOWN, by the precedence FIRED > STATED > DERIVED > UNKNOWN. **FIRED here
uses the WIDE definition** -- a measurement recorded anywhere, including an
in-source ledger. Under the narrow definition (a run log in `bench/results/`)
FIRED is ZERO across all five lanes. See DEVIATION 1602.

### 3.1 gemm, forward -- `gemm/mojo_only/gemm_identical.mojo`

| arm | changes | inert condition | class |
|---|---|---|---|
| `SAB_LEAF_READS_LAUNCH` | leaf boundary `L` scaled by block size | `P(k) == 1`, or every leaf-chain partial exactly representable | FIRED |
| `SAB_FOLD_STRIDE` | fold pairs by stride, not adjacent leaf | `P <= 2`; also every node sum exact | FIRED |
| `SAB_PAD_PLUS_ZERO` | odd tail padded with `+0.0`, not carried | `P` a power of two AND no carried partial is `-0.0` | FIRED |
| `SAB_FOLD_SERIAL` | serial ascending fold, not balanced tree | `P <= 3` for the adjacent-pair tree | FIRED |
| `SAB_NODE_ORDER` | partial written at block-arrival address | `P == 1`, single-block launch, or all partials bitwise equal | FIRED |
| `SAB_LEAF_ROTATE` | leaf `t` visits `(t + block_idx) mod P` | `P == 1`, one folding block, or all partials bitwise equal | FIRED |

**All six conditions in this table were DERIVED here.** Not one of the six
states an inert condition in its own doc block, despite being the most-run and
longest-standing arms in the tree. DEVIATION 1607.

### 3.2 gemm, backward -- `gemm/mojo_only/gemm_backward.mojo`

| arm | changes | inert condition | class |
|---|---|---|---|
| `SAB_BWD_UNTRANSPOSED` | every backward call routed `OP_NN` | square shape with a symmetric fixture; only a finite-difference check or a non-square shape separates | INERT-STATED |
| `SAB_BWD_OPERAND_ORDER` | `dC` always the left operand | a gate exercising only forward `OP_NN`'s `dA` (3 of 6 calls are correct) | INERT-STATED |
| `SAB_BWD_BIAS_AXIS` | bias gradient reduces the wrong axis | STATED as `m == n` gives the right LENGTH; the true bitwise condition, DERIVED, is `m == n` AND `dC` symmetric | INERT-STATED |

Ledger is PREDICTED 2026-08-25, not measured. `gemm_backward_check.mojo` says
so in its first line.

### 3.3 mamba, selective scan -- `mamba/ported/mamba_ssm/ops/selective_scan_interface.mojo`

| arm | changes | inert condition | class |
|---|---|---|---|
| `SAB_S8_CUDA_PAIRING` | `B*(delta*u)` not `(delta*B)*u` | all three products exactly representable, or any factor a power of two, or any factor zero | FIRED |
| `SAB_S9_UNFUSED` | two roundings, not one fused | STATED -- every `L == 1` shape, where `h` is the zero state and `deltaA` is multiplied by zero | FIRED |
| `SAB_S11_D_FIRST` | seed the fold with `D*u` | every prefix sum exact in both orders, or `D*u == +0.0` with no `-0.0` to launder | FIRED |
| `SAB_S10_DESCENDING` | fold `n` descending | `DSTATE <= 2`, or every partial sum exact | FIRED |
| `SAB_S5_EXP2` | `exp2(A*LOG2E*delta)` not `exp(delta*A)` | STATED -- every `L == 1` shape, same reason as S9 | FIRED |

Measured 2026-08-23, Apple M4, first-failing shape recorded per arm. **The
file's own note is the important part** -- "A FIXTURE WHOSE ONLY SHAPE IS
`L = 1` CANNOT SEE S5, S6 OR S9 AT ALL, and that is structural rather than
unlucky." Gate D's decode arm is exactly an `L = 1` call.

### 3.4 mamba, block -- `mamba/ported/transformers/models/mamba/modeling_mamba.mojo`

| arm | changes | inert condition | class |
|---|---|---|---|
| `SAB_S14_THRESHOLD_10` | softplus guard 20 -> 10 | STATED -- a fixture that only straddles 20 passes VACUOUSLY; the distinguishing range is delta in about `[8, 14]` | FIRED |
| `SAB_S13_BIAS_LAST` | sum taps from `+0.0`, bias after | every prefix sum of (bias, four tap products) exact in both orders | FIRED |
| `SAB_S13_TAPS_REVERSED` | walk taps newest first | `K <= 2`, or every partial sum exact both directions | FIRED |
| `SAB_S1_FOLD_DESCENDING` | sum of squares descending | `d_model <= 2`, or every partial sum exact. PER ROW | FIRED |
| `SAB_S12_MUL_SIGMOID` | `z*sigmoid(z)` not `z/(1+exp(-z))` | `1 + exp(-z)` rounds to exactly 1.0 (about `z >= 17`), `z == +/-0.0`, or `1 + exp(-z)` a power of two | FIRED |
| `SAB_S17_OP_NUMBERING` | `OP_NT` read as `OP_NN` | `op(A) == A` elementwise -- A symmetric, or a shape where transpose is the identity | FIRED |

**Measured, and `S14_THRESHOLD_10` has a second measurement worth more than
the first.** `adv_softplus_guard`'s 256 softplus inputs span `[19.87, 20.10]`
with ZERO cells in the `[8, 14]` distinguishing band, and a rebuild under the
arm produced 23 of 23 byte-identical dumps. **The arm is bitwise inert on the
corpus case built to exercise its clause.** The only fixture that falsifies
that clause is still one we wrote.

### 3.5 mamba, backward and decode

| arm | changes | inert condition | class |
|---|---|---|---|
| `SAB_BWD_PROJ_SWAP` | `dA`/`dB` routings swapped | STATED -- at `dt_proj` with `M == di` and `r == di` the shapes coincide | INERT-STATED |
| `SAB_BWD_WS_FROM_FORWARD` | workspace sized from forward shape | DERIVED -- forward workspace at least the backward's requirement at every call; allocation slack absorbs it | INERT-DERIVED |
| `SAB_BWD_CONV_TAP_SLOT` | tap reduction in wrong slot order | STATED -- conv weight symmetric in `k` | INERT-STATED |
| `SABOTAGE_BIAS_LAST` (`mamba_simple`) | decode conv bias last | DERIVED -- twin of `SAB_S13_BIAS_LAST`, same condition | INERT-DERIVED |
| `SABOTAGE_NO_CARRY` (`mamba_simple`) | state carry cut | STATED -- "token 0 legitimately agrees: at token 0 the carried state IS the zero cache" | INERT-STATED |

### 3.6 transformer, block -- `transformer/ported/transformers/models/llama/modeling_llama.mojo`

| arm | changes | inert condition | class |
|---|---|---|---|
| `SAB_S1_FOLD_DESCENDING` | sum of squares descending | DERIVED -- `d_model <= 2` or every partial sum exact | INERT-DERIVED |
| `SAB_S07_ROPE_RELATIVE_POSITION` | rotary table indexed from slice start | STATED -- inert in prefill, where `pos0 == 0` | INERT-STATED |
| `SAB_S09_ROPE_HALVES_SWAPPED` | `cat(x2,-x1)` not `cat(-x2,x1)` | DERIVED -- `sin == +0.0`, i.e. absolute position 0, PER CELL | INERT-DERIVED |
| `SAB_S10_ROPE_FUSED` | three roundings contracted to one fma | DERIVED -- position 0, or `q*cos` exactly representable | INERT-DERIVED |
| `SAB_S12_SCALE_INTO_Q` | pre-scale `q`, not scale the dot | STATED after DEVIATION 1102 -- `head_dim` a power of FOUR | FIRED |
| `SAB_S13_MASK_NEG_INF` | `-inf` not `finfo.min` | STATED -- `s + (-FLT_MAX) == -FLT_MAX` exactly for `\|s\| < 2^103`; also LAUNDERED by S16 on every fixture, so visible only at `attn.masked` | INERT-STATED |
| `SAB_S13_MASK_SELECT` | select, not `+0.0` add | STATED -- every input except a score at `-0.0` | INERT-STATED |
| `SAB_S14_MAX_PLAIN_COMPARE` | `a>b?a:b` not `identical_fmax` | STATED -- every row not carrying both zero signs | INERT-STATED |
| `SAB_S17_DENOM_HALVING_TREE` | halving tree not serial chain | STATED -- must move at kv length 3 or more, so inert at `s <= 2`. See DEVIATION 1603 | INERT-STATED |
| `SAB_S18_RECIPROCAL_MUL` | `e*(1/denom)` not `e/denom` | DERIVED -- `denom` an exact power of two. Three other lanes STATE this for their own copies | INERT-DERIVED |
| `SAB_S19_VALUE_SUM_VIA_GEMM` | value sum routed through gemm | STATED -- `P(k) == 1`, i.e. `k <= 128`; passes clause (a) at a fixed length by construction | INERT-STATED |
| `SAB_S20_SILU_MUL_SIGMOID` | `z*sigmoid(z)` not the quotient | DERIVED -- same as `SAB_S12_MUL_SIGMOID`. The BACKWARD twin states the threshold | INERT-DERIVED |
| `SAB_S05_OP_NUMBERING` | `OP_NT` read as `OP_NN` | DERIVED -- `op(A) == A` elementwise | INERT-DERIVED |

### 3.7 transformer, backward -- `transformer/mojo_only/transformer_backward.mojo`

**All twenty state an inert condition.** This is the best-documented arm set in
the tree and the table below is a transcription, not a derivation.

| arm | changes | stated inert condition | class |
|---|---|---|---|
| `SAB_B01_DOT_UNFUSED` | product-then-add not fma | `d_model == 1`, exactly-representable rows | INERT-STATED |
| `SAB_B01_DOT_DESCENDING` | same fold descending | `d_model == 1` ONLY; NOT inert at 2, because an fma keeps the second product exact | INERT-STATED |
| `SAB_B_RSTD_RECOMPUTE_DESCENDING` | `rstd` refolded descending | `d_model == 1`; the ascending half must move NOTHING | INERT-STATED |
| `SAB_B09_ROPE_TRANSPOSE_SIGN` | forward sign convention in backward | absolute position 0, PER CELL; gate must COUNT moved cells | INERT-STATED |
| `SAB_B09_ROPE_HALVES_ADJACENT` | `(2i, 2i+1)` pairing | `head_dim == 2` and position 0 | INERT-STATED |
| `SAB_B10_ROPE_BWD_FUSED` | three roundings into one fma | position 0 | INERT-STATED |
| `SAB_B10_DW_VIA_CHAIN` | hand chain not routed gemm cell | **CANNOT FIRE AT THE GATE SHAPE AT ALL** -- agree whenever `P(head_dim) == 1`, i.e. `head_dim <= 128`, and the fixtures are 16 and 24 | INERT-STATED |
| `SAB_B11_DQ_VIA_GEMM` | `dq` routed `OP_NN` | every `S <= 128` | INERT-STATED |
| `SAB_B11_DK_VIA_GEMM` | `dk` routed `OP_TN` | `L <= 128` AND `n_rep == 1` together | INERT-STATED |
| `SAB_B19_DV_VIA_GEMM` | `dv` routed the same way | inherits S19's warning verbatim | INERT-STATED |
| `SAB_B12_SCALE_INTO_DQ` | scale folded into the `dq` chain | **every power-of-four `head_dim`** -- at 16 and 64 the scale is exactly `0.25` / `0.125` | INERT-STATED |
| `SAB_B13_MASK_ZEROES_GRAD` | `+0.0` at masked cells | every masked cell whose `dS` is not `-0.0`; the oracle must predict the exact count first | INERT-STATED |
| `SAB_B18_SOFTMAX_DECOMPOSED` | decomposed autograd graph | `s == 1` | INERT-STATED |
| `SAB_B18_ZFOLD_UNFUSED` | `z` fold product-then-add | `s == 1` | INERT-STATED |
| `SAB_B18_ZFOLD_DESCENDING` | `z` fold descending | `s == 1` only | INERT-STATED |
| `SAB_B20_SIGMOID_FROM_SILU` | `silu_out / gate_out` | **every gate activation at or above about 17**, where `1 + exp(-x)` rounds to exactly 1.0; also `0/0` at a zero gate | INERT-STATED |
| `SAB_B20_SILU_DERIV_ALT_ASSOC` | `sg + x*sg*(1-sg)` | `sg` exactly 1.0, and `x == 0` | INERT-STATED |
| `SAB_B20_SILU_DERIV_FUSED` | one rounding not two | wherever `x*(1-sg)` is exactly representable | INERT-STATED |
| `SAB_B_FANIN_ZERO_SEED` | fan-in seeded `+0.0` | **PREDICTED TO MOVE ZERO CELLS ON EVERY UNPLANTED FIXTURE**; vacuous without a planted `-0.0` | INERT-STATED |
| `SAB_B_FANIN_ORDER_QKV_REVERSED` | `v,k,q` not `q,k,v` | wherever any two of the three terms are zero | INERT-STATED |

### 3.8 training, loss -- `training/mojo_only/loss.mojo`

| arm | changes | inert condition | class |
|---|---|---|---|
| `SAB_MAX_PLAIN_COMPARE` | plain compare not `identical_fmax` | STATED -- every row not carrying both zero signs | INERT-STATED |
| `SAB_MAX_SEED_ZERO` | fold seeds `+0.0` not `-inf` | STATED -- any row containing a positive logit. **The direction was applied backwards, DEVIATION 1490, and the assertion was probabilistic, DEVIATION 1491** | INERT-STATED |
| `SAB_MAX_TOPK_PREFIX` | max over first 32 classes | STATED -- argmax inside the first 32 | INERT-STATED |
| `SAB_EXP_STDLIB` | `std.math.exp` | STATED -- under `NUMERIC_FAST`, and at `shift == +0.0` | INERT-STATED |
| `SAB_LOG_STDLIB` | `std.math.log` | STATED -- under FAST, and at `denom == 1.0` where both return `+0.0` | INERT-STATED |
| `SAB_NLL_VIA_ADDBACK` | `(m + logdenom) - x_y` | STATED -- a centered row; separated by a `+1e6` common offset | INERT-STATED |
| `SAB_NLL_VIA_LOG_W` | `nll = -log(w[y])` | STATED -- wherever `e[y]` is normal; separated below `-87.33655` | INERT-STATED |
| `SAB_NEG_VIA_ZERO_SUB` | `0.0 - x` | STATED -- every input except `x == +0.0` | INERT-STATED |
| `SAB_SMOOTH_FOLDED_CONSTANT` | host-folded `eps/V` | STATED -- `eps == 0` | INERT-STATED |
| `SAB_SMOOTH_FUSED_COMBINE` | fused combine | STATED -- `eps == 0` | INERT-STATED |
| `SAB_SMOOTH_ALWAYS_SPELLED` | smoothing spelled at `eps == 0` | STATED -- every `eps != 0` AND every row whose `nll` is not `-0.0` | INERT-STATED |
| `SAB_MEAN_RECIPROCAL_MUL` | `total * (1/divisor)` | STATED -- divisor an exact power of two; `count = 3` is the cheapest separator | INERT-STATED |
| `SAB_IGNORED_ROW_NEG_ZERO` | ignored row writes `-0.0` | STATED -- must move `ce.row` and must NOT move `ce.total`, since `(+0.0) + (-0.0)` is `+0.0` | INERT-STATED |
| `SAB_IGNORED_ROW_SKIPPED` | ignored row's stores skipped | STATED -- **ALWAYS inert if the gate pre-fills with zeros.** Harness-conditioned, section 2.9(i) | INERT-STATED |
| `SAB_W_VIA_EXP_LOGP` | `w = exp(logp)` | **"Never inert on an ordinary row"** -- an assertion that NO inert condition exists. Honest UNKNOWN | UNKNOWN |
| `SAB_GRAD_SIGN` | `t - w` not `w - t` | STATED -- only a row where `w == t` at every class, reached at `V == 1` | INERT-STATED |
| `SAB_GRAD_TARGET_OFF_BY_ONE` | target at `y+1 mod V` | STATED -- `V == 1` | INERT-STATED |
| `SAB_GRAD_DIVISOR_IS_N` | divides by `N` not `ce_divisor` | STATED -- no ignored row under MEAN, and every SUM | INERT-STATED |
| `SAB_GRAD_RECIPROCAL_MUL` | `d * (1/divisor)` | STATED -- divisor an exact power of two | INERT-STATED |
| `SAB_DENOM_SERIAL_CHAIN` | hand chain not routed gemm | STATED -- every `V <= 128`, where `P == 1` | INERT-STATED |
| `SAB_REDUCE_SERIAL` | batch reduction hand-chained | STATED -- every `N <= 128` | INERT-STATED |

**One arm the contract names has no switch anywhere.** `L_NLL_VIA_MAX_LOGSOFTMAX`,
contract 4.2(c), DEVIATION 1457. A pinned decision with no falsifier in the
tree. A census of declarations structurally cannot see it, which is
false-negative item 5 in the auditor's own list.

### 3.9 training, optimizer -- `training/mojo_only/optimizer.mojo`

| arm | changes | inert condition | class |
|---|---|---|---|
| `SAB_EPS_INSIDE_SQRT` | `sqrt(v_hat + eps)` | STATED -- unless `v` is planted in the 1e-20 to 1e-12 band | INERT-STATED |
| `SAB_RSQRT` | `rsqrt` and a multiply | STATED, and it is a POPULATION statement -- about three quarters of positive-normal inputs agree (DEVIATION 741, 134,858 of 520,133 lanes) | INERT-STATED |
| `SAB_RECIP_MUL` | `m * (1/denom)` | STATED -- power-of-two denominators. **And `optimizer_check` could not construct one**, so the arm is a smoke test rather than a reach proof | INERT-STATED |
| `SAB_ADAMW_AS_ADAM` | decay folded into the gradient | STATED -- `weight_decay == 0`, **which is the reference's own default.** "The single most likely vacuous gate in the lane" | INERT-STATED |
| `SAB_DECAY_ADD_FORM` | `p - lr*wd*p` | STATED -- `lr*wd` a power of two; weak at tiny `p` | INERT-STATED |
| `SAB_MOMENT_LERP` | the `lerp_` spelling | STATED -- `g` and `m` in the same binade with no low bits to lose | INERT-STATED |
| `SAB_SQ_ASSOC` | `(c2*g)*g` not `c2*(g*g)` | STATED -- round `g` | INERT-STATED |
| `SAB_MHAT_FORM` | explicit `m_hat`/`v_hat` | **"No fixture property is known to make this inert, which is itself a claim that has not been checked."** Honest UNKNOWN | UNKNOWN |
| `SAB_UNFUSED_UPDATE` | rounded product then rounded add | STATED at length -- see 2.1. `2^20` hashed patterns separate NONE of it | INERT-STATED |
| `SAB_FTZ_LATE` | flush only at the store | STATED -- gradients within a few binades of 1.0 | INERT-STATED |
| `SAB_CLIP_SKIP_AT_ONE` | skip rescale at coefficient 1.0 | STATED -- normal gradients; `fma(1.0, x, -0.0)` returns `x` exactly | INERT-STATED |
| `SAB_CLIP_FLAT_NORM` | flat not two-level norm | STATED -- `J == 1` | INERT-STATED |
| `SAB_CLIP_PARAM_ORDER` | reversed `param_id` fold | STATED -- `J == 2`, since reversing two elements swaps one node's children | INERT-STATED |
| `SAB_CLIP_SERIAL_FOLD` | serial fold not v1 gemm | STATED -- every `N <= 128` | INERT-STATED |
| `SAB_CLIP_BLOCK_PARTITION` | partition reads the launch | STATED -- a single launch geometry | INERT-STATED |
| `SAB_POW_RUNNING` | running product not `pow_int_f32` | STATED -- `t <= 6`, PREDICTED off-repository and not measured | INERT-STATED |
| `SAB_POW_EXPLOG` | `exp(t*log(beta))` | separates at `t = 1`; no inert half in the ledger. Honest UNKNOWN | UNKNOWN |
| `SAB_SCALARS_PER_ELEMENT` | host scalars recomputed per element | STATED -- **ALWAYS inert by design.** A REACH probe declared in advance, and reported as inert rather than counted as a pass | INERT-STATED |
| `SAB_MOMENTUM_FIRST_STEP` | `b_1 = c_damp * g` | STATED -- `dampening == 0`, which is the default | INERT-STATED |
| `SAB_NESTEROV_ORDER` | `b + momentum*g` | STATED -- `momentum == 0`, and `t == 1` where `b == g` | INERT-STATED |

Two more arms contract section 12 names are modelled in the GATE rather than
by a `-D`, because both describe a defect in the CALLER --
`OPT_SAB_RESUME_REINIT` and `OPT_SAB_MICROBATCH_SERIAL`, DEVIATION 1473. They
are not counted in the 115.

### 3.10 embedding -- `embedding/mojo_only/embedding_identical.mojo`

| arm | changes | inert condition | class |
|---|---|---|---|
| `SAB_FOLD_DESCENDING` | run walked descending | STATED -- run length `<= 1`, bitwise-equal contributors, every exactly-representable fixture | INERT-STATED |
| `SAB_FOLD_BALANCED_TREE` | balanced tree not chain | STATED WITH A PROOF -- every run of length `<= 3` | INERT-STATED |
| `SAB_SEED_SEEDLESS` | chain starts from first contributor | STATED -- every input except a cell whose SOLE contributor is `-0.0` | INERT-STATED |
| `SAB_SINGLE_RUN_BYPASS` | `R == 1` stores the contributor | STATED -- same inert mask as `SEED_SEEDLESS`, and both are kept for exactly that reason | INERT-STATED |
| `SAB_EMPTY_ROW_SKIPPED` | empty run's store skipped | STATED -- always inert if the gate pre-fills with zeros. Harness-conditioned | INERT-STATED |
| `SAB_EMPTY_ROW_NEG_ZERO` | empty run stores `-0.0` | **"Never inert on `emb.dw`"** -- asserts NO inert condition. Honest UNKNOWN | UNKNOWN |
| `SAB_FOLD_READS_LAUNCH` | fold rotates by `block_dim.x` | STATED as "never inert on a run of length `>= 2`", i.e. inert at run length `<= 1` | INERT-STATED |
| `SAB_RANK_BY_ARRIVAL` | rank from an atomic counter | STATED -- no duplicate ids, and a single-block launch where arrival order IS position order | INERT-STATED |
| `SAB_SORT_TIE_REVERSED` | ties in reverse position order | STATED -- no duplicate ids, or duplicates carrying bitwise-equal rows | INERT-STATED |
| `SAB_PAD_ROW_CONTRIBUTES` | `padding_idx` position contributes | STATED -- entirely inert with no `padding_idx` position; and it MUST move `counts`/`run_begin`/`perm` and MUST NOT move `dw` | INERT-STATED |
| `SAB_PAD_ROW_NEG_ZERO` | pad row filled `-0.0` | STATED -- no `padding_idx` | INERT-STATED |
| `SAB_NO_FLUSH_ACC` | accumulator flushed only at the end | STATED -- no subnormal intermediate, and **INERT ON APPLE ENTIRELY** | INERT-STATED |
| `SAB_GATHER_NO_FLUSH` | gather copies raw | STATED -- no subnormal weight, and **NOT inert on Apple** | INERT-STATED |
| `SAB_GATHER_CLAMP_OOR` | out-of-range id clamped | STATED -- no out-of-range id, which is every ordinary fixture | INERT-STATED |
| `SAB_ACCUM_BY_ADD` | accumulate seeds `+0.0` | STATED -- any split leaving every row's contributors on one side, and every exactly-representable fixture. **No kernel reads this switch** -- the wrong spelling lives in the caller | INERT-STATED |
| `SAB_ACCUM_REFILLS` | accumulate refills with `+0.0` | STATED -- a single-microbatch gate | INERT-STATED |

---

## 4. THE HEADLINE NUMBERS

**115 arms enumerated, not 127.** DEVIATION 1601.

| lane | arms | FIRED | INERT-STATED | INERT-DERIVED | UNKNOWN |
|---|---|---|---|---|---|
| gemm | 9 | 6 | 3 | 0 | 0 |
| mamba | 16 | 11 | 3 | 2 | 0 |
| transformer | 33 | 1 | 26 | 6 | 0 |
| training | 41 | 0 | 38 | 0 | 3 |
| embedding | 16 | 0 | 15 | 0 | 1 |
| **ALL** | **115** | **18** | **85** | **8** | **4** |

**Arms with a VERIFIED SEPARATING FIXTURE -- 18 of 115 (16%).** These are the
gemm forward six, the mamba scan five, the mamba block six, and
`S12_SCALE_INTO_Q`. Every other arm's separating fixture is a prediction.

**Arms with a stated inert condition -- 103 of 115 (90%), counting the 18
fired ones that also state or record a condition.** Not 24. The brief's
estimate is stale by a wide margin, and the difference is almost entirely the
transformer backward, loss, optimizer and embedding lanes, which state a
condition on essentially every arm they declare. **If the "24 of 127" figure
is being carried into a paper, it is wrong and should be replaced.**

**Arms nobody has characterized -- 4 of 115 (3.5%).** And all four are HONEST
unknowns, in the sense that the arm says so about itself rather than being
silent -- `SAB_MHAT_FORM` ("no fixture property is known to make this inert,
which is itself a claim that has not been checked"), `SAB_POW_EXPLOG`,
`SAB_W_VIA_EXP_LOGP` and `SAB_EMPTY_ROW_NEG_ZERO`.

### 4.1 Reconciling 115 against 127

The auditor counts DECLARATIONS. Adding everything a reasonable person might
also count gets close to but not to 127.

* 115 `comptime` sabotage declarations in the five lanes.
* +1 `L_NLL_VIA_MAX_LOGSOFTMAX`, contract-named with no switch (DEVIATION 1457).
* +2 `OPT_SAB_RESUME_REINIT` and `OPT_SAB_MICROBATCH_SERIAL`, driven by
  `MOJOLEARN_OPT_GATE_ARM` rather than by a `-D` (DEVIATION 1473).
* +6 the gemm forward arms RE-ROUTED through the loss lane's gate as `G_*`
  rows, which `loss_check.mojo`'s ledger counts separately and which are the
  same six switches seen through a second entry point.
* = 124.

The remaining three are not accounted for. The likely explanations are that
the 127 predates the census, counts the non-profile lanes' integer-parameterized
arms, or counts a lane's ledger rows rather than its switches. **The census
number is 115 and it is reproducible with one `grep`; the 127 is not.**

### 4.2 The FIRED number under the narrow definition is ZERO

**DEVIATION 1602.** `bench/results/` contains 134 result documents and a tree
of run logs. **Not one of them names a sabotage arm from these five lanes.**
The only sabotage macro appearing anywhere under `bench/results/` is
`MOJOLEARN_CD_SABOTAGE_SOFT_SWAP`, from a different lane.

The 18 FIRED arms were measured. Their measurements were written into source
comments (`mamba_check.mojo`'s ledger, `selective_scan_interface.mojo`'s) and
into `IDENTITY_PATHS.md` prose. **A results harness cannot find any of it, a
diff cannot summarize it, and a second run cannot be compared to the first.**
That is a finding about the RECORD rather than about the arms, and it is
cheap to fix -- one file per sabotage round under `bench/results/`, with the
arm name, the shape, the first stage moved and the cell count, which is
exactly what the in-source ledgers already contain.

### 4.3 Which taxonomy rows a repository instance supports

| row | repository instance | status |
|---|---|---|
| operands exactly representable | DEVIATION 1147 (16 of 20 rows identical); `SAB_UNFUSED_UPDATE`'s `2^20`-pattern verdict | **MEASURED, twice** |
| all values in one binade | DEVIATION 1493 (four partials in `[1,2)`); `S1_FOLD_DESCENDING` 1-of-4 at `d_model` 8 | **MEASURED, twice** |
| no subnormal reachable | DEVIATION 1147 (`ftz` never fired); `SAB_NO_FLUSH_ACC` inert on Apple by hardware | **MEASURED once, STATED once** |
| one term dominates (absorption) | DEVIATION 1492; the mamba `residual.out` result | **MEASURED, twice, and the second is the strongest result in the document** |
| divisor a power of two | STATED in three lanes, no measured instance | **DERIVED / STATED, never fired** |
| scale a power of two | DEVIATION 1102 | **MEASURED** |
| fewer than three terms | `SAB_FOLD_BALANCED_TREE`'s proof; `SAB_B01_DOT_DESCENDING`'s counterexample at 2 | **DERIVED with a proof, never fired** |
| a single element | DEVIATION 1490 (the claim inverted); `SEED_SEEDLESS` / `SINGLE_RUN_BYPASS` | **MEASURED as a mis-statement, never fired as an arm** |
| harness-conditioned (new, 2.9(i)) | `IGNORED_ROW_SKIPPED`, `EMPTY_ROW_SKIPPED` | **STATED, never fired** |
| probabilistic invariant (new, 2.9(ii)) | DEVIATION 1491 | **MEASURED** |

**Five of the eight original rows have a measured repository instance. Three
do not** -- power-of-two divisor, fewer-than-three-terms, and single-element
seeds are derivations plus lane statements, with no arm yet fired that
demonstrates the blindness. Saying otherwise would be an overclaim.

---

## 5. DEVIATIONS 1600-1649

| # | finding |
|---|---|
| 1600 | This census. 115 arms, five lanes, classified. Never executed. |
| 1601 | The arm count is 115, not 127. Section 4.1 reconciles to 124 and cannot find the last three. |
| 1602 | ZERO sabotage results from these five lanes exist under `bench/results/`. Every measurement is filed in a source comment or in `IDENTITY_PATHS.md`. |
| 1603 | "Balanced tree" means two different trees in this repository. The embedding lane's ADJACENT-PAIR tree is the chain at `R <= 3`; the transformer's STRIDE-halving tree folds `(0+2)` then `+1` at `s = 3` and separates. **A condition derived from one and applied to the other is wrong.** Found by deriving `s <= 3` for `S17_DENOM_HALVING_TREE` and then reading the implementation, which says `s = 3` separates in its own comment. |
| 1604 | `SAB_S09_ROPE_HALVES_SWAPPED` states no inert condition; its backward twin `SAB_B09_ROPE_HALVES_ADJACENT` states "position 0" for the same arithmetic. An asymmetry in the record, not in the arithmetic. |
| 1605 | `SAB_S18_RECIPROCAL_MUL` states no inert condition. `SAB_MEAN_RECIPROCAL_MUL`, `SAB_GRAD_RECIPROCAL_MUL` and `SAB_RECIP_MUL` all state the same power-of-two condition for the same operation. |
| 1606 | `SAB_S12_MUL_SIGMOID` and `SAB_S20_SILU_MUL_SIGMOID` state no threshold; `SAB_B20_SIGMOID_FROM_SILU` states "at or above about 17" for the same `1 + exp(-x)` rounding. |
| 1607 | None of the six gemm FORWARD arms states an inert condition, despite being the oldest and most-run arm set in the tree. All six conditions in 3.1 were derived here. |
| 1608 | A HARNESS-CONDITIONED inert case is a distinct class the taxonomy did not have. `*_SKIPPED` arms are always inert against a zero-filled buffer and never inert against a poisoned one. No fixture audit can see it. |
| 1609 | **THE CONFOUND.** The seven measured instances are not independent samples. Section 7. |
| 1610 | The recommended discipline, section 10. |
| 1611 | A bug in this document's own auditor, found while writing it. `INERT_RE` matched the word "inert" inside "Never inert on an ordinary row" -- an assertion that NO inert condition exists -- and would have scored three arms STATED in the flattering direction. Fixed with `NEGATED_INERT_RE`. **The census tool had the same failure mode as the census subject.** |
| 1612 | `S12_SCALE_INTO_Q`'s condition is `head_dim` a power of FOUR, not "head_dim 16". `head_dim = 64` is the commonest real value and is blind. The lane recorded the instance; the generalization is stated here. |
| 1613 | Closed form does not imply constructible. `SAB_RECIP_MUL`'s inert condition is exact and `optimizer_check` could not build a fixture satisfying it, because the Adam denominator's every term is data dependent. This is the sharpest limit on the section 1 claim. |
| 1614 | Three of the eight taxonomy rows have no fired arm demonstrating them in this tree -- power-of-two divisor, fewer-than-three-terms, single-element seed. Section 4.3. |

Numbers 1615 through 1649 are unallocated.

---

## 6. PRIOR ART, HONESTLY

The framing in section 1 is much older than this repository in almost every
part. What follows is what is old, said plainly, followed by the narrow
residue that might not be.

**Mutation testing and the equivalent mutant, 1978.** DeMillo, Lipton and
Sayward. The equivalent-mutant problem -- a mutant no test can kill because it
is semantically identical -- is the founding difficulty of the field and has a
literature spanning constraint-based detection, coverage analysis, compiler
optimization, program-equivalence heuristics and, recently, LLM classifiers.
Equivalent mutants are reported at 4% to 39% of generated mutants in real
software. **Everything about "a mutant that cannot be killed" is old.**

**Constraint-based test generation for FP, and this is the closest prior
art.** IBM holds patents on generating floating-point test cases under
operand constraints -- mask-constrained FP add/subtract test-case generation
(US 7,028,067), range-constrained FP add/subtract test generation (US
8,965,944), and FP test-vector generation (US 5,572,664). The idea that you
compute the input class that exercises a specific FP behavior BEFORE running
anything, and then generate operands satisfying it, is the core of that work.
**The "computable on the host before the device runs" half of the claim is
anticipated by this line, and the anticipation is close enough that it should
be cited rather than worked around.**

**Differential and randomized FP testing.** FLiT (PRUNERS) detects
result-variability across compilers, hardware and environments for
user-supplied kernels. Varity (Laguna, IPDPS 2020) generates random FP
programs and checks numerical inconsistency between CPUs and GPUs; a 2024
follow-up applies it to NVIDIA-vs-AMD differences in CUDA and HIP. LLM4FP (SC
'25 workshops) generates programs that trigger cross-compiler FP
inconsistencies. **The cross-vendor bitwise-difference framing is well
established.** What these do is generate PROGRAMS and check for divergence;
what they do not do is take a FIXED pair of spellings and solve for the input
class on which they agree.

**Error-inducing FP input generation.** FPGen (ICSE 2020) generates inputs
that expose large FP errors, using bitwise operations to formulate inaccuracy
checks. Herbie rewrites FP expressions for accuracy and uses sampled inputs to
compare candidate rewritings. s3fp and its relatives search for
maximum-error inputs. **All of these search for inputs that make two
expressions DIFFER MOST. The complement -- characterizing where they agree
exactly -- is the same problem read backwards and is not obviously novel,
though the papers are pointed the other way.**

**Property-based and metamorphic testing.** The observation that a test
oracle can be satisfied vacuously, and that a metamorphic relation can hold
for uninteresting reasons, is standard in both literatures. Shrinking in
property-based testing produces MINIMAL failing inputs, which is structurally
what "the cheapest separating divisor is `count = 3`" is.

**Reproducibility literature.** FP non-associativity as a source of
irreproducibility on GPUs is thoroughly documented. Deterministic reduction
libraries (ReproBLAS and the exact-accumulation line), vLLM's batch-invariant
mode, and the HPC reproducibility papers all pin summation orders for the same
reason this repository does. **Pinning a fold order is not novel.**

### 6.1 The residue, and it is narrow

What I could not find anticipated, after the reading above, is the specific
combination of three things.

1. **For BITWISE-equivalence testing specifically** -- not accuracy, not
   error bounds, but "do these two spellings produce the same 32 bits" -- the
   blinding input class for a large family of realistic spelling pairs
   (reassociation, contraction, flush, scale placement, seed, reciprocal) is
   available in CLOSED FORM rather than by search. Sections 2.1 through 2.8
   are those closed forms and they are all one line each.
2. **Those forms are computable on the HOST, cheaply, BEFORE the device runs**,
   which makes them a preflight assertion rather than a post-hoc diagnosis.
   `training/mojo_only/loss_check.mojo`'s preflight is the mechanized form and
   it is the strongest evidence in this repository -- it asserts the fixture
   constants' bits, the three `splitmix64` copies' agreement, the leaf rule
   against the gemm oracle at 16 lengths, and `identical_exp(+0.0) == 1.0`
   exactly, all before any device call. `transformer_check`'s preflight
   printing `head_dim 16 -> 0x3e800000 ... EXACT, blind to the spelling` is
   the same idea one step short of being wired to the expectation table.
3. **The blinding class is the same class conventional numerical hygiene
   recommends.** Round numbers, one binade, no subnormals, exact
   representations, powers of two, small fixtures. This is the inversion and
   it is the only part I would defend in a paper.

### 6.2 And the honest negative

**Point 2 is substantially anticipated by the IBM constraint-based FP
test-generation patents.** They compute the input constraints that exercise a
target FP behavior and generate operands satisfying them, before execution.
Ours differs in what the constraint is FOR -- theirs targets an FP unit's
corner cases, ours targets the agreement set of two spellings -- but the
mechanism is the same mechanism. **If a reviewer finds those patents, point 2
does not survive as novelty on its own.**

**Point 1 is weaker than it reads.** The closed forms are elementary
IEEE-754. Any FP-numerics specialist would produce section 2's conditions on
demand, and several appear as folklore in compiler `-ffast-math`
documentation, in the Goldberg paper's descendants, and in the discussion of
why compilers may not reassociate. What is not standard is having them
COLLECTED and applied as a test-design checklist, and "collected" is a
weak claim.

**Point 3 is the one to lead with**, and even there the honest form is not
"nobody knew" but "everybody knows each piece and the checklist is not
standard practice". I looked for a paper that says outright "the fixtures a
careful numerical author builds are exactly the fixtures that blind a bitwise
differential test" and did not find one. That is a survey of maybe two hours
of searching, not an exhaustive one, and absence of a hit is weak evidence.

**Verdict.** The candidate contribution should be stated as an EMPIRICAL one,
not a theoretical one. The theory is old. What this repository has that the
literature does not obviously have is **a measured incidence rate on a real
codebase** -- seven independent-looking blindings found by firing gates that
had never executed, out of a set of about thirty arms actually run. That
number is worth publishing and the taxonomy is worth publishing as the
instrument that produced it. The closed forms are the method section, not the
result. And section 7 is why even the incidence rate needs a caveat printed
next to it.

---

## 7. WHAT THIS DOES NOT SHOW

**THE CONFOUND, AND IT IS THE FIRST THING A REVIEWER WILL FIND. DEVIATION 1609.**

The seven measured instances -- DEVIATIONS 1102, 1147, 1490, 1491, 1492, 1493,
1494 -- came from lanes written by agents briefed similarly within a few hours
of each other on 2026-08-25. **They are not independent samples.** They may
share a bias introduced by the briefs, in at least three ways.

1. **The briefs may have taught the finding.** If several briefs contained
   language about vacuous gates, absorption or exact fixtures, then the agents
   went looking for exactly what they found, and the "discovery rate" is a
   measurement of the brief and not of the codebase.
2. **The briefs may have taught the fixture style.** Agents briefed alike
   write alike. If the blinding fixtures were themselves produced by a shared
   habit -- hashed values in `[1, 2)`, small integer scales, powers of two --
   then the incidence rate measures that habit, not numerical practice at
   large. Note that this cuts BOTH ways and the sharper reading is the
   uncomfortable one: the habit those agents share may be exactly the habit a
   careful human numerical author has, which is the thesis. It is also exactly
   what a shared brief would produce. **The two hypotheses make the same
   prediction and this data cannot separate them.**
3. **Selection on the outcome.** Seven findings are reported. Nobody counted
   the arms that were fired and behaved as predicted, so the denominator is
   soft. The estimate "about thirty arms actually run" in section 6.2 is my
   reading of the ledgers, not a tally anybody kept.

**What would fix it.** A blind replication -- take the arms that have never
been fired, have someone with no exposure to this document predict for each
whether the CURRENT fixture separates it, then fire them all and score the
predictions. That is a real experiment, it is cheap, and until it is run the
incidence rate is an anecdote with a taxonomy attached.

**Other limits.**

* **The census measures documentation, not correctness.** 90% of arms state a
  condition. DEVIATION 1490 is a stated condition that was inverted, so the
  right reading of "90%" is "90% of arms have a claim attached", and this
  document did not check the claims. Section 8.
* **One column.** Every measurement here is Apple, plus the gemm and mamba
  cross-vendor legs. `SAB_NO_FLUSH_ACC` being inert on Apple by hardware is a
  reminder that per-vendor blindness is a separate axis this document does not
  cover.
* **The derivations are not proofs.** Eight arms carry a condition I derived
  on paper and nobody checked. DEVIATION 1603 is the case where I derived one
  and it was WRONG until I read the implementation, and I only caught it
  because that implementation happened to carry a comment about `s = 3`.
* **This document has never been executed and neither has its auditor.**

---

## 8. WHAT THE AUDITOR CANNOT DO

Restated from `tools/inert_condition_audit.py`, because it belongs in both
places.

**A regex over source can FIND a stated condition. It cannot VERIFY one.** The
tool reads text. It does not evaluate FP arithmetic, does not know what values
a fixture holds, does not know which shapes a gate runs at, and cannot tell a
correct inert condition from a confidently wrong one.

Six failure modes, each of which has already occurred in this tree.

1. A stated condition that is FALSE (DEVIATION 1490). Scored INERT-STATED.
2. A stated condition that is right and UNMET. Statedness is not coverage.
3. A condition that is true and INCOMPLETE (DEVIATION 1102 / 1612).
4. A derivation from the WRONG MODEL (DEVIATION 1603).
5. An arm with NO SWITCH, invisible to a census of declarations (DEVIATION 1457).
6. An arm whose switch exists and whose kernel ignores it on purpose
   (`SAB_ACCUM_BY_ADD`).

And a seventh, found while writing the tool -- **a negated statement counted as
a positive one** (DEVIATION 1611). Three arms say "never inert" and the first
draft scored all three as stating a condition.

**The INERT-STATED count of 85 is an upper bound on what is known and says
nothing about what is right.**

---

## 9. THE RECOMMENDED DISCIPLINE

What a lane must do before an arm counts as coverage. Each numbered item
exists because its absence produced a specific defect above.

**1. STATE THE INERT CONDITION AT THE DECLARATION, IN CLOSED FORM WHERE ONE
EXISTS.** Not "needs a long fixture" -- `P(k) == 1`, i.e. `k <= 128`. Not
"head_dim 16 is blind" -- `head_dim` a power of four. The closed form
generalizes and the instance does not (DEVIATION 1612).

**2. IF THERE IS NO INERT CONDITION, SAY SO IN THOSE WORDS.** "Never inert on
an ordinary row" is a complete and useful answer. Silence is not. Four arms in
this tree are honest unknowns and they read better than a hedge would.

**3. COMPUTE THE CONDITION ON THE HOST, IN A PREFLIGHT, BEFORE ANY DEVICE
CALL, AND RAISE IF THE FIXTURE SATISFIES IT.** This is the whole proposal and
`training/mojo_only/loss_check.mojo`'s preflight is the model. It asserts
fixture constants by bits, the three `splitmix64` copies against each other,
the leaf rule against the gemm oracle at 16 lengths, and two `identical_exp`
theorems -- all before the device is touched, so a bad constant fails in a
second rather than after a case sweep.

The preflight for an arm should print, per arm and per case, one of

    ARM x CASE -> SEPARATING   (condition C is violated: <the number>)
    ARM x CASE -> BLIND        (condition C holds: <the number>)

and the expectation table should be BUILT from that, not written beside it.
`transformer_check`'s preflight printed `head_dim 16 -> 0x3e800000 ... EXACT,
blind to the spelling` and its expectation table pointed the arm at
`head_dim 16` anyway. **The gap between a preflight that knows and a table
that does not is DEVIATION 1102.**

**4. EVERY ARM CARRIES BOTH HALVES -- A WITNESS CASE AND AN INERT CASE -- OR IS
LABELLED A SMOKE TEST.** An arm with a witness and no inert half proves it can
move bits somewhere and does not prove it is aimed where it is pointed.
`optimizer_check` gets this right for `OPT_SAB_RECIP_MUL` -- it says the word
"smoke test" rather than counting the arm as a reach proof.

**5. AN ARM WHOSE PREDICTED RESULT IS "NO BITS MOVE" IS DECLARED THAT WAY IN
ADVANCE.** `SAB_SCALARS_PER_ELEMENT` is a REACH probe whose bit result is
reported as INERT and never counted as a pass. Declaring it after the fact is
rationalization; declaring it before is a design.

**6. THE INERT HALF IS AN INVARIANT OR IT IS NOT AN INERT HALF.** If the
condition holds with probability `1 - p` per row it is a flake, not a
condition, and the assertion must be stated over the population with its
probability (DEVIATION 1491).

**7. COMPARE STAGES, NOT OUTPUTS.** Absorption is not a bug and it is
everywhere. Four of six mamba block arms are invisible in the block's output
at the shape where they are strongest. An output-only gate would have licensed
every one of those clauses.

**8. POISON EVERY BUFFER BEFORE THE CALL.** The `*_SKIPPED` class is always
inert against a zero-filled allocation. This is the one blinding condition a
fixture audit cannot detect, so it has to be a harness rule (DEVIATION 1608).

**9. FILE THE MEASUREMENT WHERE A HARNESS CAN FIND IT.** One document per
sabotage round under `bench/results/`, with arm, shape, first stage moved and
cell count. The in-source ledgers already contain exactly that and are
invisible to every tool in `tools/` (DEVIATION 1602).

**10. A CONDITION DERIVED FROM ANOTHER LANE'S SPELLING IS NOT PORTABLE.** Two
"balanced trees" in this repository have different inert bounds. Read the
implementation before quoting a bound across a lane boundary (DEVIATION 1603).

---

## 10. SOURCES

Prior-art reading behind section 6.

- [Using Constraints for Equivalent Mutant Detection](https://arxiv.org/pdf/1207.2234)
- [Equivalent Mutants in the Wild (ISSTA 2024)](https://dl.acm.org/doi/10.1145/3650212.3680310)
- [LLMs for Equivalent Mutant Detection](https://arxiv.org/pdf/2408.01760)
- [Interval Constraint-Based Mutation Testing of Numerical Specifications](https://pure.mpg.de/rest/items/item_3345868_2/component/file_3345869/content)
- [Generation of mask-constrained floating-point addition and subtraction test cases (US 7,028,067)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7028067)
- [Generation of test cases with range constraints for floating point add and subtract (US 8,965,944)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8965944)
- [System for generating floating point test vectors (US 5,572,664)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5572664)
- [FLiT](https://github.com/PRUNERS/FLiT)
- [Varity (Laguna, IPDPS 2020)](https://www.osti.gov/servlets/purl/1664653)
- [Testing GPU Numerics: NVIDIA vs AMD](https://arxiv.org/pdf/2410.09172)
- [LLM4FP (SC '25 workshops)](https://arxiv.org/pdf/2509.00256)
- [FPGen: Efficient Generation of Error-Inducing Floating-Point Inputs (ICSE 2020)](https://web.cs.ucdavis.edu/~rubio/includes/icse20.pdf)
- [Impacts of floating-point non-associativity on reproducibility](https://arxiv.org/pdf/2408.05148)
- [Enabling Bitwise Reproducibility for the Unstructured Computational Motif](https://www.mdpi.com/2076-3417/14/2/639)
- [Mixing Condition Numbers and Oracles for Accurate Floating-point Debugging](https://arxiv.org/pdf/2503.11884)
