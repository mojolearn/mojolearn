# lane/lowbit-quality: progress

Brief: `~/mojolearn-evidence/lowbit-units/brief.md`, section Lane B and its
updates of 02:50Z, 03:25Z and 03:35Z. Plan: `docs/lanes/LOWBIT_UNITS_PLAN.md`.
Lane files: `~/mojolearn-evidence/lowbit-quality/`. Records:
`bench/results/lowbit_quality/2026-09-29_nvc1-rtx4090/`.
Box: shared pod nvc1 (2x RTX 4090), lane `lowbit-quality`. Nothing runs on
the laptop; it edits files, talks to the pod and formats JSON records.

## State (2026-09-29T03:40Z)

| Step | State |
|---|---|
| Model and both texts staged from R2 onto nvc1, pins checked on the box | done (`stage.log`) |
| Host binding `_mojolearn_linalg_host` built on the pod | done, `/root/lowbit-quality-work/host` |
| Quantizer cross-check | PASS in job nvc1-0006 (FAILED once in nvc1-0005, cause below) |
| This forward against `transformers` | PASS, sabotage arm fails |
| Inference, six arms and the attention switch, enwik8 | done, job nvc1-0006 |
| Inference, six arms and the attention switch, pile_github | done, job nvc1-0008 step 1 |
| Width sweep, both texts | running, job nvc1-0008 steps 2 and 3 |
| Training plan (noise floor, then arms) | running, job nvc1-0007 |
| Finalists F1 and F2, both texts, then their training arms | queued, job nvc1-0010 |
| F1 and bf16 inference, then bf16 and 15-bit training at five seeds | queued, job nvc1-0011 |
| Task evaluation of the finalists | OWED: the R2 store holds no task set (below) |

## Decisions that bind this lane (Andrew, through the brief)

- int8 as a model's arithmetic and the int8-attention mix (finalist F2) are
  DROPPED. Dropped means only that the flag does not offer them. Their
  measurement is finished and reported; rows carry the note.
- No new int8 rescue arm. Every arm already planned or queued is run.
- Nothing is cancelled, queued or running.
- Training order: bf16 on both operands, forward and backward, first; then
  15-bit the same way.

## The cancelled job

nvc1-0009 (`finalists.sh`, queued, not started) was cancelled at 03:32:18Z
on the orchestrator's 04:30Z instruction. The orchestrator withdrew that
instruction as its error. The script was restored byte for byte (sha256
0a61b14eb13889d1d3abfee20ebb6fe5e23d1e4d1837b4886854522cb28161c7, the same
on the pod and at commit 1aa6da891) and resubmitted as nvc1-0010 at
03:35:42Z. No measurement was lost; the job lost its place in line.
Code that skipped int8 arms was committed (352f71a29) and removed
(9dcf5100d) without ever being synced to the pod, so no job ran it.

## The pass rule

The relative perplexity change AND the upper end of its interval are both
under 1 percent, on TWO held-out texts of different kind. A profile measured
on one text reads OWED, not PASS.

The interval: within each of the 200 windows of a text the per-position
difference of nll (arm minus baseline) is averaged; the interval is the mean
of those window means plus and minus 1.96 of their standard error (95
percent, normal approximation), mapped through exp(x) - 1. The windows are
what is resampled. It bounds the sampling error on these texts and says
nothing about other text or about tasks.

Top-1 agreement is the share of scored positions, each with the true context
supplied, where the arm's top token equals the baseline's. It is not a rate
of changed tokens in generated text: free-running generation diverges from
the first changed token on.

## The evaluation texts

| Text | Key in the R2 store | Bytes used | sha256 of those bytes | Ids |
|---|---|---|---|---|
| enwik8 (English, wiki markup) | `corpus/enwik8/input.txt` | [99000022, 99374116) | a2bf2dea5108207b0653ffb0aa46cc550ba9cd6e67e82ff2d1123fa890c0880d | 102400 ids, sha256 d5be2324ffc522bc306eaa9baa63e626c9fe8df690eec7c552b4117f1ad8a56f |
| pile_github (source code) | `corpus/pile_github/input.txt` | [96000030, 96240053) | 91cc52d4b375dd0b7a8b0a256f1b5b5261967eba025a44ce3b6072633e3919ef | 102400 ids |

Each starts at the first byte after the first newline at or after byte
99,000,000 (enwik8) or 96,000,000 (pile_github). Strict UTF-8, no invalid
byte in either range. Tokenized by the model's own `tokenizer.json`, no
special token added. 200 windows of 512 ids that do not overlap; every
position but the first of a window is scored from the ids before it in the
same window: 102,200 scored positions per text. Baseline perplexity 7.937106
on enwik8 and 3.361991 on pile_github.

## Quantizer cross-check (job nvc1-0006): PASS

`quantizer_check.json`. `mojolearn.lowbit.pack` ran through the host binding
(the compiled seams of `checks/numerics.mojo`), built on the pod.

- 14 real tensors (5 weights, 9 activations captured where they enter a
  product) and one planted tensor of edge cases: int8 codes, exponents, bf16
  bits and materialized float32 values all equal, bit for bit, to
  `lowbit.pack_one` / `materialize_one`, and the codes equal to the
  pure-Python spelling `_quantize_int8_py`.
- The int8 product equals the host oracle `gemm_int8_oracle` bit for bit on
  five real operand pairs (three projections, QK, PV) and the planted pair.
- L-5: the float64 to float32 cast equals the 20-bit/12-bit spelling of
  `i32_to_f32_pinned` on 1,012,492 integers.
- The arms that must fail, seen failing: ties away from zero, truncation and
  a scale one exponent off each disagree with `pack_one` on the real
  tensors; the sabotaged product disagrees with the oracle; truncating
  conversion disagrees with L-5 on 486,192 values. Ties away from zero
  differs on the WEIGHTS only (they are bf16, so exact ties are common:
  50,889 in one 960x960 weight) and on no activation, which has none.

## Inference on enwik8 (job nvc1-0006)

Numerical floor: the same float32 operands accumulated in float64 move
perplexity by +1e-8 relative. That is the most the order of a float32
accumulation can move this metric.

| Arm | Profile | Attention products | Change | Upper end | Top-1 agreement | On enwik8 |
|---|---|---|---|---|---|---|
| a | `fp32.v1` | | 0 | | 1.0000 | baseline |
| b | `bf16f32.v1` (bf16 weights) | no | 0, bit-equal to the baseline | | 1.0000 | under the bar |
| c | bf16 both operands | no | +0.0093% | +0.0168% | 0.9962 | under the bar |
| c | bf16 both operands | yes | +0.0085% | +0.0179% | 0.9953 | under the bar |
| d | `int8i32.v1` | no | +32.18% | +34.62% | 0.7659 | MISS |
| d | `int8i32.v1` | yes | +31.61% | +33.91% | 0.7653 | MISS |
| e | 15-bit both operands | no | -0.0027% | +0.0017% | 0.9979 | under the bar |
| e | 15-bit both operands | yes | -0.0015% | +0.0033% | 0.9980 | under the bar |
| f | 15-bit weights, int8 activations | no | +27.28% | +29.30% | 0.7844 | MISS |
| f | 15-bit weights, int8 activations | yes | +27.59% | +29.61% | 0.7825 | MISS |

Arm b is bit-equal to the baseline BY CONSTRUCTION on this model: its
safetensors are bf16, so rounding the weights to bf16 changes no bit. It
says nothing about a model whose weights are float32.

Follow-up rows, each its own rule under its own name (enwik8 only):

| Row | Change | Upper end | Top-1 agreement |
|---|---|---|---|
| int8 weights, fp32 activations (what `weight_format="int8"` ships) | +1.59% | +1.71% | 0.9327 |
| fp32 weights, int8 activations | +27.10% | +29.08% | 0.7847 |
| 15-bit weights, fp32 activations | -0.0007% | -0.0002% | 0.9998 |
| fp32 weights, 15-bit activations | -0.0001% | +0.0048% | 0.9981 |
| int8 weights, 15-bit activations | +1.59% | +1.71% | 0.9327 |
| int8 weights, 12-bit activations | +1.66% | +1.78% | 0.9317 |
| int8 weights, 10-bit activations | +3.90% | +4.10% | 0.9199 |
| int8 both, activation scale saturating by one exponent | +213.9% | | 0.5325 |
| int8 both, activation scale saturating by two exponents | perplexity 44,426 | | 0.0266 |
| int8 both, scale not a power of two (absmax to 127) | +6.12% | +6.43% | 0.8840 |
| `int8i32.v1`, LM head fp32 | +25.27% | | 0.8084 |
| `int8i32.v1`, the three SwiGLU products fp32 | +6.50% | | 0.8639 |
| `int8i32.v1`, down_proj fp32 | +30.37% | | 0.7750 |
| `int8i32.v1`, the four attention projections fp32 | +34.96% | | 0.7667 |
| `int8i32.v1`, down_proj 15-bit | +30.85% | | 0.7754 |
| `int8i32.v1`, down_proj and LM head 15-bit | +23.58% | | 0.8222 |
| projections fp32, QK and PV int8 | +0.595% | +0.666% | 0.9636 |
| projections fp32, QK and PV 15-bit | +0.0011% | +0.0020% | 0.9995 |

What the rows say:
- The int8 ACTIVATION codes cause nearly all of the int8 loss (+27%). The
  int8 WEIGHT codes alone cost +1.6%, which also misses.
- No selective fp32 or 15-bit product family rescues `int8i32.v1`.
- A saturating activation scale is far worse: the largest values of a row
  carry the signal.
- 15-bit codes cost nothing measurable on either operand or on both.

No new int8 rescue arm is added (orchestrator, 03:35Z). The seven follow-up
rows inside job nvc1-0008 (the finer and the saturating int8 WEIGHT scale,
and int8 weights with named products on 15 bits) were planned and queued
before that, so they run and are reported.

## Inference on both texts: the six arms and the attention switch

| Profile | Attention products | enwik8: change (upper end) | pile_github: change (upper end) | Top-1 agreement | Verdict |
|---|---|---|---|---|---|
| `bf16f32.v1` | no | 0 | 0 | 1.0000, 1.0000 | PASS, bit-equal to the baseline |
| bf16 both | no | +0.0093% (+0.0168%) | +0.0000% (+0.0061%) | 0.9962, 0.9976 | PASS |
| bf16 both | yes | +0.0085% (+0.0179%) | +0.0127% (+0.0224%) | 0.9953, 0.9967 | PASS |
| `int8i32.v1` | no | +32.18% (+34.62%) | +28.89% (+31.86%) | 0.7659, 0.8501 | MISS; dropped |
| `int8i32.v1` | yes | +31.61% (+33.91%) | +27.89% (+30.68%) | 0.7653, 0.8489 | MISS; dropped |
| 15-bit both | no | -0.0027% (+0.0017%) | +0.0046% (+0.0089%) | 0.9979, 0.9986 | PASS |
| 15-bit both (the F1 configuration) | yes | -0.0015% (+0.0033%) | +0.0055% (+0.0099%) | 0.9980, 0.9987 | PASS |
| 15-bit weights, int8 activations | no | +27.28% (+29.30%) | +22.59% (+24.85%) | 0.7844, 0.8648 | MISS; dropped |
| 15-bit weights, int8 activations | yes | +27.59% (+29.61%) | +23.35% (+25.64%) | 0.7825, 0.8626 | MISS; dropped |
| int8 weights, fp32 activations | no | +1.59% (+1.71%) | +1.06% (+1.16%) | 0.9327, 0.9630 | MISS; dropped |

## Training (job nvc1-0007, running)

Byte LM, profile `mojolearn.byte-lm.b32-l128-d128-h4-kv2-hd32-ff256-v256-blocks4.fp32.v3`,
the repo's initializer and AdamW settings, enwik8 bytes (training windows
from [0, 90,000,000), validation 512 fixed windows from byte 95,000,000).
Equal-step point 4000, runs continue to 6000.

Self-test of the custom products: PASS. Float32 kinds through the forward
only path and the forward and backward path agree with torch's own autograd
(cosine 1.0, relative norm error 1.4e-7); the arm with the weight gradient
negated fails (cosine -0.06).

Baseline, five seeds, validation loss at step 4000: 1.48804, 1.47899,
1.46952, 1.47935, 1.46688. Noise floor (sample standard deviation over the
seeds) 0.0085 nats, 0.853 percent of perplexity. It is close to the bar, so a
training verdict rests on the interval of the PAIRED differences (arm minus
the baseline of the same seed), which is tighter than the floor.

Interim table, 03:36Z, three seeds per arm (five are queued in nvc1-0011):

| Profile | Products | Mode | Change at step 4000 | Interval (t, 2 degrees of freedom) | Inside the noise | Steps to the baseline's final loss | Verdict |
|---|---|---|---|---|---|---|---|
| bf16 both | projections | forward + backward | +0.001% | -0.616% to +0.621% | yes | median 4100 | PASS |
| 15-bit both | projections | forward | -0.229% | -0.877% to +0.424% | yes | median 4000 | PASS |
| 15-bit both | projections | forward + backward | -0.184% | -1.400% to +1.046% | yes | median 4000 | UNDERPOWERED at three seeds |

## The backward quantization rule: what the training simulation does

The rule (brief, 06:50Z): every product quantizes its own operands from
their float32 values, along that product's own contracted extent. Codes are
never carried from one product to another and never transposed.

**The simulation follows the rule in all three products. It differs in
none.** The code is `QMatmulNT` in `bench/lowbit_quality/byte_lm_train.py`.
Every product is a call of `arith.product_nt(A, B, ...)`, which computes
`A @ B^T` and scales each row of each operand it is handed along that row's
last extent, the contracted one. What is handed over is float32: the saved
forward operands `X` and `W` as they were before any rounding
(`ctx.save_for_backward`), and the incoming gradient `dY` as autograd
delivers it. No code and no exponent of the forward product is kept.

| Product | Call in the simulation | Left operand is scaled | Right operand is scaled |
|---|---|---|---|
| Forward, `Y = X W^T` | `product_nt(X, W)` | X by token (each row of X over the features) | W by output feature (each row of W over the input features) |
| Weight gradient, `dW = dY^T X` | `product_nt(dY^T, X^T)`, the transposes taken of the float32 tensors | dY by output feature over all tokens | X by input feature over all tokens |
| Input gradient, `dX = dY W` | `product_nt(dY, W^T)`, the transpose taken of the float32 tensor | dY by token over the output features | W by input feature over the output features (the columns of W) |

"All tokens" is every row of the batch: a projection's `X` is flattened to
`[batch * length, features]` before the product, so the weight gradient
contracts over 32 * 128 = 4096 tokens at this shape.

The attention products follow the same rule through the same code, with
the leading extents (batch, head) kept:

| Product | Left operand is scaled | Right operand is scaled |
|---|---|---|
| Forward `S = Q K^T` | Q by query position over the head channels | K by key position over the head channels |
| `dQ = dS K` | dS by query position over the key positions | K by head channel over the key positions |
| `dK = dS^T Q` | dS by key position over the query positions | Q by head channel over the query positions |
| Forward `O = P V` | P by query position over the key positions | V by head channel over the key positions |
| `dP = dO V^T` | dO by query position over the head channels | V by key position over the head channels |
| `dV = P^T dO` | P by key position over the query positions | dO by head channel over the query positions |

Two things a reader comparing line by line should know:

- In the FORWARD ONLY arms the two backward products are float32 products
  of the saved float32 operands. Nothing is quantized there, so the rule
  has nothing to act on.
- The self-test holds the orientation of the backward calls: with float32
  kinds the forward and backward path equals torch's own autograd (cosine
  1.0, relative norm error 1.4e-7), and the arm with the weight gradient
  negated fails.

Vectors. This lane exported none, forward or backward. Lane C exported the
forward vectors itself, from a copy of `arith.py` pinned at blob 17337a5c
(this branch at 59f445c6c). `arith.py` has changed since that pin by 18
added lines, none of them arithmetic (the dropped note and a method that
reads it); its blob is now c60a455c. Backward vectors are exported by
`bench/lowbit_quality/backward_export.py` in Lane C's file format, from one
real training step, with the operands exactly as `QMatmulNT` hands them to
the three products.

## Failures, with cause

| What | Cause | State |
|---|---|---|
| Jobs nvc2-0005 and nvc2-0006 exit 4 after 5 seconds, before any run: `SyntaxError: name 'ZERO_CODE_WIDTHS' is used prior to global declaration` | A `global` statement placed after the name's use in `main`. The laptop check parsed the file and did not compile it. The same file was in nvc1's tree from 03:46Z to 03:55Z; no job started from it. | fixed (d44962ac3); files are compiled on the pod before a submit; resubmitted as nvc2-0007 and nvc2-0008 |
| Job nvc1-0005 (smoke) exit 1: `QUANTIZER CHECK FAIL`, `materialized_bits_differing` 27,461 of 921,600 on the first weight | A negative value that rounds to zero was `-0.0` in the simulation's float64 code and dequantized to `-0.0`; the int8 code 0 dequantizes to `+0.0`. Codes, exponents and the product were already equal. | fixed (`arith.py`: an integer code has one zero); PASS in nvc1-0006 |

## Owed

- A TASK EVALUATION of the finalists. The R2 store holds corpora
  (enwik8, pile_github, fineweb-edu), tabular and ANN benchmark sets, the
  model and opponent wheels. It holds NO language task set. Nothing was
  downloaded onto the pod. The orchestrator stages one.
- The second text for every arm that is under the bar on enwik8, the width
  table on both texts, the finalists on both texts, and the training table:
  running or queued.
- Vectors exported from this simulation for Lane C's host oracle cross-check
  (the brief's Lane C gate 6): not asked of this lane yet, not done.
