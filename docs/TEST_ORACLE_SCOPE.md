# Oracle scope and the oversized suite

The CPU verifier grew to cover internal training across the host families.
Those fits are enabled only inside `_cpu_reference.reference_training()`.
They are not a new public CPU estimator-training promise. Public CPU inference,
GPU training, and the published host trainer retain their existing boundaries.

The expensive mistake is using the entire verification matrix as the routine
unit of work. `identity_break` combines several contracts in one lane/fixture
cell, defaults to nine fixtures, and fits twice. Adding a contract to that
harness makes every full invocation more expensive. CPU coverage expansion
also enlarges the shared lane registry; an unscoped run traverses it. This does
not mean every CPU-only check is applicable to every GPU algorithm.

| Comparison | What it establishes | When to run it |
| --- | --- | --- |
| Two training hashes | Repeatability on the selected implementation | Changed algorithm, base fixture |
| Inference and save/reload | In-memory and serialized-model consistency | Changes to that estimator or serialization |
| Whole batch versus rows/splits/prefixes | Batch invariance | Changes to batching, shapes, or relevant arithmetic |
| Sampler versus teacher, joins/leaves | Stateful decoding and log-probability consistency | Changes to state, decoding, or their dependencies |
| CPU versus frozen GPU records | Agreement with those recorded implementations and revisions | Backend qualification and releases |
| Independent reference/corpus checks | Algorithm-specific numerical correctness | Changes to the numerical implementation |

Repeatability is not an independent correctness oracle. Batch and decode probes
also compare execution paths within the implementation; they test relations,
not an external mathematical answer. Keep the independent algorithm checks and
frozen cross-vendor records. Do not bless new expected hashes merely because a
new CPU run is stable.

The decode probe is especially expensive on Metal. `_probe_rlpair` runs a
whole-batch decode, five single-row decodes, teacher-forced whole/split/row
calls, and a changing-batch trajectory. `_rlpair_block` wraps the bare block in
embedding, normalization and a projection head; each token also computes the
library cross-entropy. This repeats for both fits and every selected fixture.
Those are useful state/batch contracts, but they are not a prerequisite for an
unrelated estimator edit. The new routine runner selects them explicitly and
keeps their recorded hashes unchanged.

## A reproduced false pass in the oracle checker

`cpu_identity_gate_check.do_column` checked the training verdict and ignored
all reported sub-verdicts. A record with training STABLE and inference MOVED
returned success. The new negative control failed on the old checker with
`0 != 1`, then passed after the repair. The checker now rejects failed
inference, reload, batch, decode and extra-part verdicts. Explicit N/A remains
allowed so an inapplicable or deliberately skipped contract is not fabricated
as coverage. This was a verdict-checking defect, not the cause of long runtime.

## Routine scope

Use `test-algo --lane NAME` for two fits and core inference checks, on CPU,
with one-minute execution and queue limits. Select `--probe-group batch` or
`rlpair` only when relevant, or `all` for separate bounded jobs. Use
`--exhaustive` to select all nine fixtures for that algorithm. Every record
says which probes were skipped. This is iteration coverage, not release
qualification. Fixture floors are unchanged. Apple releases now use the
installed-wheel gates instead of requiring the full Apple identity matrix;
see [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md#5b-apple-qualification-once-per-pypi-update).

On 2026-09-16 the Transformer base/core job took 0.72 seconds on the available
CPU oracle binaries. That is a scoped-job measurement, not a claim that it
replaces the nine-hour release matrix. Native Metal and AMD decode validation
is recorded separately in [NEURAL_METAL_DECODE.md](NEURAL_METAL_DECODE.md).

The split Transformer core, batch and decode jobs matched every enabled training,
inference, model, batch and decode hash from a combined CPU run of the same
fixture. The targeted tooling regression suite passed 66 tests.
