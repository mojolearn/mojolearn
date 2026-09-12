# Bounded real-text byte LM capture

`tools/byte_lm_real_text_capture.py` is authored source, unexecuted by its implementation agent. Root exclusively builds/runs/tests/measures on the authorized remote NVIDIA/AMD device; no Apple testing. The script contains no compiler, provisioning, subprocess, timing or inline independent model.

The corpus and manifest are read and verified before constructing the public trainer. It requires the pinned Tiny Shakespeare bytes with SHA-256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`, length 1,115,394, and exactly the supplied manifest schedule. Training is the first 65,536 bytes; the heldout range is the next 8,192. Step `s` (zero-based), row `b`, starts at `(s*64+b*32)%65504` and consumes 33 bytes. All 128 actual token batches and eight actual heldout batches are retained as raw int32 before the first step. No sampling, token normalization or inferred ordering occurs. Those figures are the default two-row shape. `--shape` selects a different committed shape, its own manifest and its own schedule, described under "Training shape" below; the corpus, its SHA-256 and both byte ranges are pinned independently of the shape and do not move.

Initialization is explicit and versioned. Every flat index goes through the specified 32-bit integer avalanche with seed tag `0x42595445`; its upper eight bits minus 128 are divided by 1024, giving exact dyadic FP32 weights in approximately ±0.125. Both blocks' RMSNorm scales are exactly one. The divisor was changed before any real-LM execution, based on source-level sensitivity analysis of the nonlinear gradient control; gradient tolerances, heldout threshold and optimizer configuration are unchanged. The complete resulting initial bytes and their SHA-256 are retained; no unexecuted hash is invented in this document. AdamW is fixed at learning rate .003, betas .9/.999, epsilon 1e-8, weight decay .01, no clipping. The actual normalized FP32 scalar values are checked against public state.

## Runs and artifacts

The public `SmallByteLanguageModelTrainer` supplies state snapshots, gradients, forward-only evaluation, and its explicitly separate JSON checkpoint format. Native checkpoint v1 is not silently substituted. Every output directory must be new.

- `--steps 1`: captures one actual-text step compatible with the independent two-block oracle. Heldout evaluation and learning admission are not run.
- Default `--steps 128 --action continuous`: captures all 128 steps, plus initial/final evaluation on exactly eight fixed heldout batches.
- `--action head64`: captures steps 1–64 and saves the public checkpoint.
- `--action resume128 --resume-checkpoint ...`: requires a step-64 checkpoint with exactly the same profile, optimizer, initialization identity, corpus/schedule and cursor; continues steps 65–128. The bounded input is captured once, sealed in a Linux memfd, and loaded from those exact bytes. An identical retained copy and transfer SHA identify what was actually loaded.

Each `stepNNNNNN/` directory contains raw pre-state, 20 gradients, post-state, flags, loss and IDs plus `capture.json` in the schema documented by `BYTE_LM_GRADIENT_ORACLE.md`. Pre-state files are exclusive hardlinks to the previous retained post-state, avoiding duplicated bytes: a full 128-step run uses approximately 72 MB of raw numerical data, plus modest manifests/checkpoints. Hardlinks must be preserved or accounted for when transferring/archiving artifacts; dereferencing/copying every logical file duplicates the pre-state data. Sources, actual loaded binding hash, runtime witnesses, corpus manifest, initialization and schedule hashes are retained. Source/binary changes during the run refuse admission.

Every heldout call is checked for complete parameter/moment/flag/config/cursor invariance. The aggregate is `math.fsum` of eight actual FP32 batch means divided by eight, over 512 targets. The predeclared numerical learning criterion is final mean / initial mean ≤ .9. Only a continuous 128-step run is eligible for this criterion; a resumed run's initial evaluation is at step 64 and cannot be relabeled as step-zero loss. The summary records the observed criterion separately and keeps `learning.admitted=false` until root retains successful guard exit and independent gradient evidence. This is a small next-byte-learning criterion, not a useful-generation or contextual-reasoning claim.

Root-only NVIDIA examples, using fresh artifact paths and the installed/selected public extension:

```sh
MOJOLEARN_NUMERIC_MODE=identical OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 tools/nvidia_serial_guard.py --seconds 600 --rss-gib 12 -- \
  python3 tools/byte_lm_real_text_capture.py --steps 1 \
  --expected-vendor cuda --output /artifacts/byte-lm-one-step-new

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 tools/nvidia_serial_guard.py --seconds 300 --rss-gib 12 -- \
  python3 tools/byte_lm_gradient_oracle.py \
  /artifacts/byte-lm-one-step-new/step000001 --expected-vendor cuda \
  --output /artifacts/byte-lm-one-step-oracle-new.json

MOJOLEARN_NUMERIC_MODE=identical OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 tools/nvidia_serial_guard.py --seconds 1800 --rss-gib 12 -- \
  python3 tools/byte_lm_real_text_capture.py --steps 128 \
  --expected-vendor cuda --output /artifacts/byte-lm-continuous-new
```

For AMD use its serial guard, expected vendor `hip`, and fresh paths. Root must retain exact command/logs, process and complete guard exit statuses, and runtime/source/binary artifacts. A produced summary is not sufficient evidence if teardown later fails. Outputs refuse existing paths; failures may leave new partial directories and cannot be resumed in-place.

## Training shape

DEVIATION 2682. The shape is no longer a literal repeated across the harness, the manifest and the verifiers. `tools/byte_lm_shape.py` derives every shape-dependent value from nine integers and is the single source of shape truth; it fails at import if anything it derives for the default drifts from the literals the certified `b2-l32` run was produced and admitted with. `tools/byte_lm_real_text_capture.py`, `tools/byte_lm_gradient_oracle.py`, `tools/byte_lm_state_compare.py`, `tools/byte_lm_cpu_train_gate.py` and `tools/byte_lm_validation_admit.py` take `--shape`. Omitting it is the default two-row shape, unchanged.

Identity is claimed **per shape**. Nine weight gradients contract over the token count, so 128 tokens in a step is a different sum from 64 tokens in a step and not two of them accumulated. A second shape is therefore a second capture, a second oracle run, a second comparison and a second certificate.

`--shape 4,32` is the one committed second shape. Its schedule is pinned in `training/corpus/tinyshakespeare/manifest-b4-l32.json`, batch 4, context 32, 128 planned steps, over the same pinned corpus bytes and the same train and heldout ranges. Its heldout schedule is four batches of four rows starting at 65536, 65664, 65792 and 65920, which reads the same 512 target bytes as the default's eight batches of two rows, so the heldout mean of the two shapes is a mean over the same amount of text. Each shape writes its own manifest name and its own output tree, so neither run can read the other's schedule.

**No four-row capture exists.** The manifest's `learning_gate` records `status` as predeclared and unexecuted, which is accurate. No GPU has produced a `b4-l32` capture, no oracle has checked one, no comparator has compared one, and the `b4-l32` certificate is owed. `tools/byte_lm_validation_serial.sh` adds the second shape's three jobs only when `MOJOLEARN_BYTE_LM_SHAPE=4,32` is set, under their own job names and in their own subdirectories, after the certified default's jobs have completed. Setting that variable buys the opportunity to capture the shape on a rented box. It is not evidence that the shape was captured.

## Remaining admission work

The harness captures head64/resume128 and the missing-moments control described below. The separate `tools/byte_lm_state_compare.py` is authored for complete raw-state comparison, actual checkpoint transfer, and effective missing-moments controls. It requires continuous NVIDIA and AMD runs, plus one head/resume/control direction per invocation; both directions require two comparisons. Root guard receipts and independent gradient/AdamW oracle receipts are prerequisites for admission. These tools have not yet qualified this model. One independently checked step is not all-step reference coverage, and no language-learning result has been observed while authoring this harness. A restarted-step control remains separate work.

`tools/byte_lm_validation_serial.sh`, selected by remote campaign profile 5,
orders the build, host API checks, one-step capture, independent oracle and
128-step capture under the vendor's serial guard. It stops at the first
failure. `tools/byte_lm_validation_admit.py` checks fetched artifacts against
the frozen source inventory and retained binding, matches the independent
capture to the full run's first step, and requires the predeclared learning
gate. Its single-vendor verdict cannot certify cross-vendor identity.

## Effective missing-moments resume control (source addition)

`--action zero-moments65 --resume-checkpoint <actual-head64-checkpoint>` uses the same bounded, immutable sealed incoming capture as legitimate resume. It first admits and retains the complete original state at step 64 under `legitimate-head64/`. Both incoming moment arrays must contain nonzero values. Only `m` and `v` are then replaced by exact FP32 zeros through the public state-loading interface; parameters, flags, configuration, metadata and cursor hashes must remain unchanged. The altered state is retained under `initial/` and used for **only step 65**, with the identical fixed token batch. No heldout evaluation or learning gate runs in this control action.

`summary.json` records `action="zero-moments65"`, `completed_steps=65`, one step record, the original incoming checkpoint hash, and a `control` object containing `legitimate_state`, `altered_state`, `legitimate_state_directory="legitimate-head64"`, `changed_fields=["m","v"]`, and `effective="PENDING_COMPARATOR"`. The normal `step000065/capture.json` and raw arrays provide all before/after evidence. Root's comparator must tie the legitimate state to the actual head64 file, require exact equality of step-65 tokens/loss/gradients against the legitimate continuous trajectory, and require updated parameters/first moments/second moments to differ. Preparing or executing a control is not evidence that it was effective until that comparison succeeds.

The harness also now explicitly refuses unexpected public registry shapes/order, non-FP32/non-int32 raw arrays, malformed fixed array sizes, and scalar losses that do not exactly represent FP32. It never silently converts a wrong gradient dtype into a plausible capture. Existing outputs still refuse; partial artifacts can remain after failures and have no successful summary/guard admission. Control and gradient runs must be separate guarded root jobs; no subagent executes them.
