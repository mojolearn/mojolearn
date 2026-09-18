# Expanded 0.8.7 implementation and proof

User direction, 2026-09-18: close the 23 ordinary holds, six kernel saved-model
debts and whole loaded-CausalLM proof; implement useful multi-GPU capabilities;
expose useful completed algorithms in the upcoming PyPI wheel. GEMM belongs to
another active lane. Do not spend this effort on 0.8.5 or backward compatibility.

The intended candidate is rebuilt from the integrated expanded source, with
matching native artifacts and qualification. The earlier frozen candidate is
preserved as evidence, not used as a reason to omit the newly authorized APIs.
No new implementation inherits a hardware certificate from old wheel bytes.

## Owners

- Root: `lane/next-wheel-coverage`: integration, wheel exposure/completeness,
  CV follow-up and hardware orchestration.
- `lane/qualify-ordinary-23`: all 23 holds, reference admission and six kernel
  saved-model recording routes. Sole writer for reference table promotion.
- `lane/causal-lm-distributed-proof`: whole-model proof and explicit distributed
  loaded-model execution, including honest state/residency limits.
- `lane/classical-distributed`: forecast prediction, GPC distribution and IVF
  distributed index search/storage.
- GEMM: existing external lane; not modified here.

Each feature lane commits/pushes checkpoints; root merges reviewed/tested work.
Local numerical work uses the shared slot and one math worker; no agent creates
a rental independently. Native controls, metadata placement, actual execution
and wheel qualification remain separate evidence claims.

## Wheel completeness checkpoint

Both wheel builders now independently audit the candidate against source public
exports and all package Python implementations, including newly added nested
packages. They require the current embedded verifier and comparator too. Missing
or stale bytes fail the build; Linux removes the rejected candidate. Reports are
retained beside the built wheel. This checks packaging, not numerical correctness
or native binding freshness (the existing native build/qualification gates own
those requirements).

Five focused tests cover a complete payload, omission of a new subpackage,
stale implementation/verifier bytes, a missing generated comparator and strict
CLI exit status. All passed; shell syntax and Python compilation passed.

The existing local 0.8.7 candidate correctly fails source completeness: it lacks
the new BpeTokenizer/corpus public exports, four new Python implementation files,
and carries older bytes for several changed implementations. This is expected
for an older candidate and establishes why a fresh expanded build is required.
No 0.8.5 artifact was examined for this implementation task.

## Remaining

Integrate each feature checkpoint, exercise real GPU paths with bounded guarded
jobs, admit matching references only after their actual checks pass, build the
expanded 0.8.7 candidate, run installed API/verifier gates and qualify exact wheel
bytes. No publication or all-holds-closed claim is made by this checkpoint.
