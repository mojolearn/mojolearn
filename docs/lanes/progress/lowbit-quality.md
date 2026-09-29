# lane/lowbit-quality: progress

Brief: `~/mojolearn-evidence/lowbit-units/brief.md`, section Lane B and its
updates of 02:50Z, 03:25Z and 03:35Z. Plan: `docs/lanes/LOWBIT_UNITS_PLAN.md`.
Lane files: `~/mojolearn-evidence/lowbit-quality/`. Records:
`bench/results/lowbit_quality/2026-09-29_nvc1-rtx4090/`.
Box: shared pod nvc1 (2x RTX 4090), lane `lowbit-quality`. Nothing runs on
the laptop; it edits files, talks to the pod and formats JSON records.

## State (2026-09-29T03:20Z)

| Step | State |
|---|---|
| Model and both texts staged from R2 onto nvc1, pins checked on the box | done (`stage.log`) |
| Host binding `_mojolearn_linalg_host` built on the pod | done, `/root/lowbit-quality-work/host` |
| Quantizer cross-check | PASS in job nvc1-0006 (FAILED once in nvc1-0005, cause below) |
| This forward against `transformers` | PASS, sabotage arm fails |
| Inference, six arms and the attention switch, enwik8 | done, job nvc1-0006 |
| Inference, second text (pile_github) and the width sweep | running, job nvc1-0008 |
| Finalists F1 and F2, both texts, then training | queued, job nvc1-0009 |
| Training plan (noise floor, then arms in verdict order) | running, job nvc1-0007 |
| Task evaluation of the finalists | OWED: the R2 store holds no task set (below) |

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

The int8 rescue arms stop here (orchestrator, 03:35Z). Seven follow-up rows
(the finer and the saturating int8 WEIGHT scale, and int8 weights with named
products on 15 bits) were inside job nvc1-0008 when that instruction came.
The job was submitted before it and a submitted job is never cancelled, so
those rows run; none was added after.

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
seeds) 0.0084 nats, 0.84 percent of perplexity. It is close to the bar, so a
training verdict needs the interval over the seeds, not the mean alone.

## Failures, with cause

| What | Cause | State |
|---|---|---|
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
