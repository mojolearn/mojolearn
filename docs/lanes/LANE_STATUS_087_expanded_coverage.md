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

## Integrated proof and API checkpoints

- Six kernel variants are now selectable in the classical saved-model gate,
  with correctly scaled held-out probes. All 54 local GPU-recorded models
  (six variants, nine fixtures) passed CPU replay using explicitly hashed
  retained binaries. Independent current NVIDIA/AMD and installed-wheel
  receipts remain owed; no ordinary hold was removed.
- The 23-hold capture runner records full properties twice, checkpoints by
  lane/stage, validates installed wheel bytes when requested, and has a total
  time budget. Its remote wrapper captures families as soon as their bindings
  are built, starting with kernels, preserving early results if later work fails.
- Forecast scheduling and GPC class-level scheduling are integrated and exposed
  through public modules. Partition/failure tests pass; real CPU-native logical
  partition evidence and physical GPU qualification are separate lane work.
- Whole loaded-CausalLM capture is reachable through `verify-causal-lm`, with
  separate capture/comparison actions, non-overwriting output and the same
  conservative CPU budget as other verifier commands. Read-only coverage lists
  this supplemental check and the still-experimental three-route small profile.
- Integration checks: 34 targeted capture/kernel/LM/GPC/resource tests, four
  new CLI tests, then 47 coverage/resource/CLI tests passed. The package import
  inventory reports every current root module reachable. No full native sweep
  was substituted for missing hardware evidence.
- Experimental `models.ParallelCausalLM` is integrated with explicit layer
  ownership, process-local weights/state and host activation transport. It
  remains host-cache backed and still needs physical GPU/capacity qualification.
  The public proof CLI accepts the explicit layer map. New focused tests after
  integration: 33 passed, including strict capture comparison and CLI routing.

## Remaining execution

Additional integrated work: disjoint IVF search/storage with a native partial
candidate API, and expanded whole-model profile v2 covering 24 cases across
all seven supported checkpoint families. IVF has 14 real Apple GPU logical
partition checks and forecast/GPC have 11 real CPU-native partition checks;
these do not claim two-device execution. Root exports DistributedIVFIndex and
all new modules in the package/API inventory. Rebuilding IVF is mandatory for
the new entry points; old binaries cannot supply them.

Review follow-up: validate the original IVF offset sequence before clipping it
into local shards. Otherwise a negative first offset or oversized final offset
could be silently normalized by partitioning instead of rejected as in ordinary
search. Twenty binding-free IVF checks pass, including four corrupt global
layouts rejected before any GPU worker is created. Valid input arithmetic and
native source are unchanged by this follow-up.

Integrate each feature checkpoint, exercise real GPU paths with bounded guarded
jobs, admit matching references only after their actual checks pass, build the
expanded 0.8.7 candidate, run installed API/verifier gates and qualify exact wheel
bytes. No publication or all-holds-closed claim is made by this checkpoint.

## Installed distributed verifier and three-class kernel checkpoint

Integrated qualification through `60cf3a4ff` and the shipped distributed runner
through `830098980`. Six kernel variants now have admitted matching CPU/Apple/AMD
references (54 cells / 216 numerical parts); NVIDIA and installed replay still
hold default admission. AMD recovery adds the preprocessing dependency required
by normalized GP routes. Hardware rentals remain owned by their respective lane
controllers with independent deadlines.

`python -m mojolearn verify-distributed` now exposes the shipped runner, fresh
checkpoint outputs, read-only comparisons, installed RECORD validation and the
CPU thread budget. Its numerical/placement and transport-control results do not
claim independent physical kernel execution or native arithmetic sabotage.
CLI/resource/coverage integration: 46 tests passed. Wheel import inventory:
103 package modules, all reachable. New artifacts still require an expanded
0.8.7 build and installed hardware replay before publication.

## CV integration and NVIDIA infrastructure interruption

Merged CV through `0ff8d0a30`, preserving both worker inventory and fold execution
dispatch. Added public `parallel_model_selection` exposure. Combined scheduler,
driver-witness, distributed verifier, capture-source and recovery tests: 99 passed.
The separate distributed API suite also passed 91 tests with 14 native IVF tests
skipped outside their dedicated native job.

The first NVIDIA two-L40S pod disappeared before CV capture; provider GET returned
404 while its lease still had about 45 minutes remaining. Its owner stopped
polling, verified absence and cancelled its local watchdog. Another GEMM lane's
NVIDIA pod is active and was untouched. The deletion cause is unproven; no NVIDIA
numerical pass is claimed. Replacement awaits ownership coordination. AMD and
local installed-wheel preparation continue independently.

## User-requested checkpoint: no new pods or long builds

The user requested saving everything promptly because usage points are low.
No new rentals or fresh macOS build will start. The existing AMD lease finished;
its owner fetched the records, received DELETE 204 and independently verified
GET 404/list absence at 18:23:16 UTC. All 23 ordinary routes match the CPU-backed
references across 990 numerical parts. Two initial normalized-GP dependency
refusals are preserved alongside successful corrected captures.

Merged the 54 kernel portable models and expanded wheel qualification gates.
The bundle now contains 58 models; source CPU inference passed all 116 model/batch
parts with two repetitions. Root integrated gate tests: 19 passed. The workflow
YAML test could not be collected in the root test interpreter because PyYAML is
absent; the owning lane's gate/orchestration tests are separately recorded.

Merged the installed CV verifier through `9dc7fbb0f` and exposed it as
`python -m mojolearn verify-cross-validation`; CLI/resource/coverage tests:
48 passed. Hardware qualification remains explicit and separate from software
integration. Follow `EXPANDED_087_BUILD_AND_PROOF.md` for the deferred fresh
source-pinned build. Outstanding: NVIDIA records, exact rebuilt-wheel replay,
physical two-GPU execution evidence and release qualification. No wheel was
published by this work.

AMD whole-loaded-model closure (`c4abc7a56`): all 24 v2 cases passed and all
144 parts match both CPU and Apple. The exact supplemental source-file hashes,
original remote source attribution, raw capture and comparisons are retained;
this is source-capsule evidence, not an installed-wheel or two-GPU certificate.
Final wheel import inventory includes 106 reachable package modules.
