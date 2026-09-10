# Mamba3 public-path optimization continuation

Branch `lane/mamba3-statepass`, base `9eb32597`. Root integrates per hunk;
its Mamba1 binding/decode changes must be preserved.

Validated IDENTICAL production changes:

1. Factor independent S20 decay/increment work from the serial chunk recurrence.
   Every output keeps the original ascending floating-point fold. Independent
   increments occupy `pass_states` temporarily; chunk-decay scratch occupies
   `qk_s` before its ordinary producer. No scratch allocation was added.
2. Replace Mamba3 host element loops with byte-preserving copies. Mamba1/2 and
   other numeric modes retain their original helpers.
3. Compute independent angle increments in parallel; the wrapped theta
   recurrence remains serial and consumes exactly the same increments.
4. Stop `_record_work_slice` before any device download when traces are disabled.
   The previous implementation downloaded/repacked hundreds of MiB per call
   only to discard them in disabled `IdentityTrace.record_list_f32`.
5. Replace full nonfinite-buffer downloads with an integer first-bad-index
   reduction. Named-buffer order, first offending index, NaN/Inf wording,
   pending-state checks, and fresh public-call weight validation are unchanged.

Legacy A/B flags are `MOJOLEARN_MAMBA3_LEGACY_STATEPASS`,
`MOJOLEARN_MAMBA3_LEGACY_HOST_COPY`,
`MOJOLEARN_MAMBA3_LEGACY_ANGLE_INCREMENT`,
`MOJOLEARN_MAMBA3_LEGACY_TRACE_SLICES`, and
`MOJOLEARN_MAMBA3_LEGACY_REFUSAL`.
`MOJOLEARN_MAMBA3_PHASE_TIMERS` compiles in explicit synchronization/timing;
ordinary production builds contain no timer calls.

Additional defaults selected at `8cda8cee` after NVIDIA native, continuation,
refusal, full Python surface, and complete output-SHA gates:

- Direct public NumPy addresses through pinned buffers, avoiding intermediate
  Lists on input/state/output paths and zero-filled host weight containers.
  Fresh device weights still undergo the original ordered validation.
- Ystate and QKS cooperative operand tiles, 5,248 shared bytes under the
  existing kernel-matrix shared-memory row. Each output owns all 128 fold
  leaves in their original order. QKS structural +0 cells remain unchanged.
- State increment shares independently rounded decayed V operands across
  the 128 N owners, using 512 shared bytes. The full Q fold and padded +0
  terms remain private to each output owner.

New A/B controls are `MOJOLEARN_MAMBA3_LEGACY_DIRECT_TRANSFER`,
`MOJOLEARN_MAMBA3_LEGACY_YSTATE`, `MOJOLEARN_MAMBA3_LEGACY_QKS`, and
`MOJOLEARN_MAMBA3_LEGACY_INCREMENT_V`. `LEGACY_HOST_COPY` also disables
new direct transfers so original host-path baseline builds remain reachable.

On H100 before the jointly corrected GEMM seam, ystate fell ~17→1.67 ms,
QKS ~8.6→1.61 ms, state increment ~11.6→8.0 ms. Core totals fell ~49→22–23 ms.
Full public medians at this stage: narrow 76.232392 ms, wide 143.328134 ms.
Joint final H100 with root's corrected GEMM/matrix passed all gates. Final
public medians: smoke 1.631379 ms, narrow 78.300729 ms, wide 149.868213 ms.
Against the admitted historical same-model/driver torch reference these are
1.10×, 4.78×, 7.73×. Against September 7 ours, large rows improved 16.29×/11.02×.
Parent owns final Apple integration and the GEMM/matrix source files.
The same-pod old-path timing samples are also archived; host-heavy baseline
jitter is substantial, so those medians should not be presented without it.

The optional hardware-FTZ arithmetic experiment was rejected by an adversarial
boundary gate and withdrawn from runtime calls. See evidence README; this
lane's preservation of existing output bits is not a repair of the separately
identified pre-existing cross-column arithmetic boundary issue.

Evidence: `bench/results/mamba3/2026-09-09-statepass/README.md`, with parent
Apple evidence at its `apple/README.md`.
The 1–1.5× target is not yet met.
