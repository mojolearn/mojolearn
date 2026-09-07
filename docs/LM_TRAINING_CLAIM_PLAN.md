# Real-text training claim and paper acceptance plan

The target is a reproducible, end-to-end **small language-model training**
experiment: 34,944 FP32 parameters, two decoder blocks, pinned real byte data,
falling held-out loss, complete per-step state equality and actual cross-vendor
checkpoint continuation. NVIDIA/AMD continuous 128-step raw agreement is now
retained evidence; bidirectional NVIDIA/AMD resume is admitted; Metal remains pending.

## Execution and admission order

1. **NVIDIA passed:** all 12 jobs and file admission at `d921eade`, including
   independent first-step FP64 gradient/AdamW checks and controls before the
   128-step learning run. Held-out loss fell 5.5413 → 2.8436.
   [Retained evidence](../bench/results/resume/2026-09-06-root-byte-lm-nvidia/README.md).
2. **AMD and continuous agreement passed:** AMD run 3 passed all 12 jobs from
   the same `d921eade` numerical source, with a 1 GiB runtime pool. All 128 raw
   steps, held-out batches and final checkpoint bytes match NVIDIA, including
   parameters, gradients, moments, flags, counters and losses. NVIDIA-to-AMD checkpoint continuation is separately admitted below.
   [Raw comparison and provenance](../bench/results/resume/2026-09-06-root-byte-lm-cross-vendor/README.md).
3. **NVIDIA → AMD passed:** actual foreign checkpoint, fresh-process continuation,
   raw continuous comparison and effective missing-moments control are admitted.
   **AMD → NVIDIA passed:** the separate reverse head/resume/control also passed
   full raw comparison with no missing prerequisites.
4. **Pending:** root adds Apple only after NVIDIA/AMD resume passes, with bounded memory,
   time and two CPU cores/threads (never above three). NVIDIA+AMD is a two-vendor
   result; only actual matching Apple evidence makes it three-vendor.
5. Measure actual training-step cost under a declared boundary. The current
   fixed byte-LM trainer is IDENTICAL-only: a FAST comparison first needs an
   implemented, independently checked matching training path. Use exactly one
   matched external NVIDIA training implementation if such a comparison is run.
6. Prepare arXiv v2 after the acceptance evidence exists. The requested target
   is the week the result lands, conditional on the gates and paper checks;
   a prepared driver or manuscript paragraph cannot stand in for execution.

Subagents never test, build, run models, measure or provision. Root runs one
numerical/build job at a time. No threshold or fixture is selected after seeing
a favorable result without documenting that change as a new experiment.

## API publication is a separate claim

PyPI **0.6.0** macOS/AMD alpha-api artifacts are published; exact index filenames
and SHA256 values are retained in
[publication verification](../bench/results/releases/2026-09-06-alpha-api/pypi-0.6.0-verification.json).
Their Alpha classifier and inherited-native provenance remain explicit, and
NVIDIA is source-build-only. API exposure does not certify every training mode,
shape, published binary, resume behavior or hardware vendor.

## Scope and performance wording

The small model size is explicit. The construction argument for underlying
pinned operations is separate from empirical coverage of a complete training
program. It does not certify arbitrary architectures, unsupported shapes,
larger models, future compilers or different optimizers without further work.

Report the training result alongside **measured step cost** and the relevant
kernel work. Table 6's historical operation ratios cannot be substituted for
an unmeasured whole-training overhead, and kernel optimization gaps should not
be described as unavoidable costs of the numerical contract.

## Initial literature check — not a priority certificate

- [RepDL](https://arxiv.org/html/2510.09180v1) describes a working library for
  bit-level reproducible training and inference. Its
  [repository](https://github.com/microsoft/RepDL) includes a MNIST training
  example and trained-model/logit hashes. Calling it merely a proposal would
  misrepresent that implementation. The precise supported and demonstrated
  hardware matrix must be audited before comparison.
- [Verde/RepOps](https://arxiv.org/html/2502.19405v1), section 4, reports FP32
  language-model training/inference benchmarks on four NVIDIA GPU variants.
  That is relevant prior training work, but that reported matrix is not an
  NVIDIA/AMD/Apple experiment.
- [Meganeura](https://arxiv.org/html/2608.01563v1) explicitly distinguishes its
  strict-f32 arithmetic permissions from operation-order or bitwise identity.
  Its cross-vendor training evidence addresses a different numerical claim.

This initial primary-source check does not establish that nobody has published
the proposed result. Audit current versions, artifacts and relevant additional
work before writing “first.” Prefer the concrete verified contribution over a
universal priority statement if the search cannot substantiate the latter.

## Metal preparation blockers (source-only audit)

The current byte-LM build, native/Python vendor admission, capture platform
checks, Linux sealed-memfd checkpoint loader and receipt/comparator policy
accept CUDA/HIP only. Metal needs explicit build/admission support and an
immutable-bytes checkpoint decoder; reopening a temporary path is insufficient.
Keep the independent FP64 oracle on NVIDIA/AMD. Metal can be compared exactly
to those independently checked captures without claiming a Metal FP64 oracle.

Before local execution, provide a dedicated macOS supervisor with an exclusive
lock, short deadline, process-group cleanup, two compiler/runtime threads and
unified-memory pressure limits. Darwin thread caps are not Linux CPU affinity
and must not be presented as an enforceable two-core affinity. Begin with one
step and full-state comparison before expanding to 128. Any portability edits
change the source inventory: freeze a common final source and rerun the relevant
NVIDIA/AMD legs, or disclose and separately validate the portability overlay.
Do not silently weaken exact-inventory comparison against `d921eade`.

### macOS supervisor review before any model launch

The authored `tools/macos_serial_guard.py` has not been approved for model
execution. Source review identified PGID-only descendant tracking, no verified
whole-group extinction before lock release, CPU undercount for short-lived
children, missing coordination with `/tmp/cbsym-build.lock`, and sampled
rather than hard deadline enforcement. Source fixes and nine root-run mocks now cover the identified branches;
actual OS supervision remains unvalidated before local builds/models. Its conservative memory checks are useful but cannot be
presented as hard CPU affinity or per-process Metal memory accounting.
NVIDIA/AMD checkpoint work proceeds independently under the existing remote
guards; this review is not a Metal execution result.

### First checkpoint portability slice: host checks passed

`SmallByteLanguageModelTrainer.from_checkpoint_bytes` now accepts an exact
immutable bounded byte capture and shares existing file-loader validation.
Root's selected checkpoint/PCA dispatch host checks passed (16 tests), with
no native model execution. [Retained logs and source hashes](../bench/results/resume/2026-09-06-root-byte-lm-host-portability/README.md).
Capture integration, Metal admission/build support and safe local supervision
remain open. This source change is not part of the frozen current resume
campaign and requires a future common source inventory before recertification.
