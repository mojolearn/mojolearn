# 0.8.7 public surface and verification audit

Status: **not comprehensively qualified and not ready to publish on this evidence**.
Audited 2026-09-17. This supersedes the missing-artifact statements in the earlier
0.8.7 handoff, not the release admission requirements.

## Scope and completed changes

The expanded source inventory has 214 harness lanes and 201 public API entries.
These are different quantities; aliases, result containers, helpers and parameter
variants prevent either from being a count of distinct algorithms. The old
inventory omitted public parallel, pooled and offloaded modules. The generator
now includes those modules, preserves module-qualified names to avoid confusing
serial and parallel APIs, and follows harness helpers to find indirect calls.
The matrix is historical evidence, not current-wheel qualification.
Refused, moved and empty cells no longer count as backend coverage merely
because their lane name appears in a record.

Four existing classes are now also exported from `mojolearn`:
`ParallelNeuralTrainer`, `ParallelByteLanguageModelTrainer`,
`PooledByteLanguageModelTrainer`, and `OffloadedByteLanguageModelTrainer`.
Their existing submodule APIs and device requirements remain their contracts.
This adds discoverability; it does not certify all their device configurations.
Native-only implementations, including internal spectral embedding, are not
ready public algorithms merely because a Mojo implementation exists.

The new `metrics-homogeneity-completeness` lane checks the combined public metric
API's result order and beta forwarding. Both it and corpus-wide BPE training
now declare why per-row batch invariance does not apply. All 214 lanes have a
batch declaration; a declaration is not proof that every applicable check passed.
The CPU manifest now includes the two existing GP optimizer routes and the
new `ordered-gradient-sum` lane, which checks the public helper against an
independent Float32 left fold and detects input mutation.

## Measured development evidence

[Retained diagnostics](../../bench/results/release087-coverage-probe/2026-09-17/README.md)
are deliberately excluded from reference admission by the `probe` path. They
use modified source and prebuilt native binaries, not newly qualified wheels.

- Combined metric: nine CPU fixtures, two executions each, all stable.
- Combined metric: Metal base fixture, two executions, stable; CPU and Metal
  agree on `672e75f003fce299`. Scheduler plus execution took 3.38 seconds.
- Real sabotaged CPU metric binding: base hash `fdd8000292e1a4ed`, different
  from control. Focused tests also reject swapped results and ignored beta.
- GP optimization with and without restarts: nine CPU fixtures each, two
  executions; all 72 train/infer/model/batch comparisons match bundled references.
  They remain public-verifier candidates because those references lack CUDA/HIP
  GPU witnesses. This does not establish global optimizer optimality.
- Ordered-gradient sum: all nine CPU fixtures pass twice; Metal base matches
  CPU (`a742b6c812b94347`), with two executions in 0.55 seconds including queue.
- Focused coverage, host manifest and CPU GP/metrics tests: 191 passed.
- The applicability selfcheck passes; root exports resolve to the same classes
  as their established submodule imports on a CPU-only import.

## Published package versus release candidates

PyPI inspection found version 0.8.5, not 0.8.7. Its two wheels contain the older
single-fixture verifier, not `_verify_all.py`, the comprehensive reference table,
or the new model bundle. The older `--all` means all divergent stages, not all
algorithms. Only the byte-LM CPU host binding is included in those wheels.
Source documentation must not be read as a statement about that published wheel.

Both locally assembled 0.8.7 candidates carry the old freeze commit
`1cbb4630f7cce23853acf2b249cc65d7bcc44b3a` and 32 CPU host bindings:

| candidate | SHA-256 |
|---|---|
| macOS arm64 | `206181595bff847049f598896b29e749cc08dcc68ae83e1c74a1f207cb8c391d` |
| manylinux x86_64 | `e43eba9940f446c869cc8de2be3253857a8308c4a08846f6f57f5e41efed1803` |

The macOS candidate is under `~/mojolearn-wt/release-087-macos/python/dist/`.
Its second build log shows successful installed surface checks on Python
3.10–3.14. The first failed build is not the final outcome.
Linux artifacts and logs are under `~/mojolearn-evidence/release-0.8.7/`.
Neither candidate includes the changes in this audit.

Existing NVIDIA identity evidence covers 162 lanes × nine fixtures, with all
1,458 training cells stable, from an installed wheel at the old freeze.
The CPU column covers 171 lanes × nine fixtures but has 18 REFUSED cells:
both categorical CTR saved-model lanes lack their input model fixtures.
That CPU run imported the source checkout rather than the installed wheel.
Its `complete: true` reports completion, not successful coverage.
The AMD recording attempt failed SSH host-key verification; it proves no cells.

## Remaining release work, in order

1. Resolve every source API entry to an identity lane, an explicitly named
   alternative gate, or a documented result-container/helper contract. The
   matrix now covers `parallel_training.ordered_sum_gradients`;
   `TrainedBpeVocabulary` is a returned artifact whose rendered bytes are checked
   in `bpe-trainer`. Saved-model host APIs have separate forest/classical gates.
   Include pure host BPE and fold construction in the release scope even though
   GPU agreement cannot provide an independent oracle for their Python code.
2. Freeze the expanded source and regenerate matching wheel payloads. Run the
   existing packaging/admission tools against the final candidate bytes, not
   the older freeze. Ensure saved-model CTR fixtures are available through the
   installed package path, without requiring a source checkout.
3. Record every applicable lane/fixture/property against the installed wheels
   on CPU, Metal, CUDA and HIP. Multi-device claims require real multi-device
   hardware; one-GPU fallback does not qualify them. Keep local CPU work within
   three cores and follow the bounded Metal/installed-wheel qualification rules.
4. Verify training, held-out inference, serialization/reload and batch invariance
   where applicable. Neural contracts additionally need their applicable
   step/full, recurrent-state, gradient, batch-scale and ragged checks; four
   historical columns alone do not cover these properties. Record explicit
   reasons for inapplicability and require stable repeats and negative controls.
5. Regenerate reference hashes from admitted, matching records, resolve conflicts
   and stale/missing references, then replay the public verifier from each final
   installed wheel. Pending/candidate lanes must not silently disappear behind
   an apparently comprehensive success. Promote them only with the evidence
   required by `host_surface.py`.
6. Run the existing Linux and macOS release admission checks, retain artifact
   digests and records, and only then publish 0.8.7. Current audit results do not
   authorize claiming every exposed API or backend is qualified.
