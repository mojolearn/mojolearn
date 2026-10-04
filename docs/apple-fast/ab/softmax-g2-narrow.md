# Softmax-specific narrow G2 candidate

Source-only, default OFF, not compiled or measured. Base manager
`5474c2ade7c32e9f54390f882768026ede26b720`. No production default changes.

The actual `glm_base.mojo::linear_fwd` multiclass branch repeatedly launches
NT GEMM in softmax objective/gradient evaluations and in prediction. Its
output is rows x classes, often narrow. This adapter changes only that call,
retaining weight transpose, bias addition, loss, backward pass, optimizer,
stopping criteria, precision, buffers and refusals. Binary C=1 GEMV is intact.

The shared direct `scoped_kernel[32,32,False]` is reused without cloning tile
machinery. Candidate scope: FAST+Apple, M>=4096, C2..16, D128..512, contiguous
NT buffers, valid capacities, no output/input alias and int32-safe products.
No split/atomic path, allocation, extra copy or synchronization is introduced.
The selection depends only on known shape/layout metadata, not data or labels.

Resident G2 at 32768x8x220 recorded1.203750->0.865625ms against SDK, with exact
output words in that screen. That is rationale only: no actual softmax gain or
quality inference is transferred. Neighbor/window coverage is still owed.
Prior G1/G5 small-row downstream gates do not test this G2/window. KMeans was
rejected as a plain-adapter target during source review: real assignment is
fused SIMT, its unfused generic GEMM routine is only for checks. This candidate
does not force KMeans through a materialized matrix to manufacture reach.

## Build and strict quality contract

Binding: estimators (`bindings/build_estimators.sh`), FAST, M2 only.
A: `-D MOJOLEARN_SOFTMAX_G2_AUDIT`.
B: `-D MOJOLEARN_SOFTMAX_G2_AUDIT -D MOJOLEARN_SOFTMAX_FAST_G2_NARROW`.
Build default OFF as an isolation check before any future default decision.
No core/IDENTICAL prerequisite: helper directly loads a verified estimators
artifact and calls actual public `qn_fit`/`qn_decision_function` bindings.
Audit exports: `softmax_g2_state`, `softmax_g2_reset`, `softmax_g2_count(0..2)`
(eligible,selected,fallback), `softmax_g2_last(0..4)` (M,C,D,eligible,selected).

`tools/softmax_g2_quality.py` has capture/compare modes, only on M3 Ultra:

```
MOJOLEARN_NUMERIC_MODE=fast python tools/softmax_g2_quality.py capture   --source FULL_SHA --binding VERIFIED_SO --binding-sha256 HASH   --arm 0 --case anchor --output FRESH_A_NPZ
# Arm1, separate fresh process, B artifact, same case/source and fresh output.
python tools/softmax_g2_quality.py compare --a A_NPZ --b B_NPZ --output FRESH_JSON
```

All declared cases required before claiming the proposed window: anchor,
odd,lower,upper,rows-out,features-low,features-high,classes-out. Actual fit and
train/query scoring are captured, with exact expected route counts/metadata.
Fresh reservation prevents replay under the same capture tag. Fixture hashes,
source and binary hashes are recorded. NO_REACH is a failure, not admission.

Independent FP64 evaluates each arm's fitted train/holdout cross-entropy,
training gradient norm, holdout classification error, returned-score maximum
error and objective error. Every metric requires B<=A with ZERO allowance;
retcode must match. No coefficient bit-match requirement or relaxed numerical
threshold is substituted. Cases outside eligibility remain control evidence;
rows-out still exercises eligible query scoring. No timings or opponent jobs.
Binary GEMV isolation also needs a predeclared binary caller control before
promotion; it is not silently certified by these multiclass-only captures.

Manager owns source/artifact manifests, pinned helper allowlist and preflight,
serial staging/queue, builds, eventual one-call timings, and promotion. This
source change is ready for M2 compiler review, not queue-ready. Full board
reach and benefit remain unknown; no claim that the351 count will move.
