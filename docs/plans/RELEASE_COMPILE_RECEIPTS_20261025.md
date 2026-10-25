# Linux release compilation admission

The readiness ledger now admits the actual Linux release producer, `tools/linux_surface_qualification.sh`, for its expanded native binding compile units. A development group/cache pass is a different producer and cannot satisfy these requirements.

Before a fresh leg launches, `release.py` atomically begins its selected requirements and records their original run IDs, source revisions and fingerprints in the leg launch provenance. Refusal occurs before logs or output directories move and before any process spawns. Normal reuse never forces another execution. If all selected units already have retained passes, the controller requires restoration/reconciliation of their artifacts instead of launching another build. Recognized provisioning-only failures may continue the same begun attempt on the next provider; ordinary compiler failures cannot use that path.

Before invoking `build_sets.sh`, the build producer writes `compile-inputs.json`: frozen source inventory identity, binding identities, actual installed package names and compiler bytes hash, and an explicit allowlist of compiler/target/build flags. It never captures a full environment, pip configuration, credentials or download URLs. Unknown build input names require explicit capture-policy reconciliation; their values are not printed. The original build proof binds this capture and each staged set manifest by hash. The capture is inside the existing retained release-build tree and travels with that tree.

The begun configuration must supply `dependency_context.compiler_sha256["linux-64"]`, obtained from the actual build toolchain before the run. Each expanded compile unit retains that expected executable hash, and admission requires equality with the actual captured compiler bytes. Package names and a lock hash alone do not establish executable identity. Missing expected bytes remain a provenance reconciliation task; this implementation does not invent them from a completed historical build.

After the complete leg exits zero, the controller constructs a report referencing that original capture, launch, complete build proof, set manifest and exit record. The receipt parser independently derives coverage from those records and verifies the consumed bytes:

- Binding units require the exact begun source/build/toolchain identity, a unique merged-module owner (or an explicit per-binding artifact), the retained index bytes and actual native artifact hash.

Runtime-library and adapter compile units remain unsupported: portable math uses a C compiler and adapter registration uses C++, and those compiler executable bytes have not yet been bound to begun inputs. The Mojo hash does not qualify their full toolchains. Their original build files remain retained, but the ledger keeps these units at `complete-provenance` until that support is added. Installed-wheel adapter smoke remains separate.

Every subsequent validation rereads the evidence through the shared portable resolver. Publication bundles include the new parser policy hash. Counters come from the selected native artifact; caller-supplied counts cannot supply missing coverage.

Only the concrete begun `unit.build_flags` configuration is admitted. Dependency expansion retains `compile_variants`; a separately declared multi-variant switch matrix requires its own supported producer/units, and one build cannot establish both enabled and disabled variants. Unknown or additional coverage IDs refuse admission. Apple compile, Linux runtime-library and C++ adapter compile units, other producer kinds, incomplete/failed reports and historical builds lacking the original before-build/launch provenance remain unqualified or require reconciliation. No current fingerprint is attached to an old build, and missing provenance never authorizes a rerun.

Both begin and result admission are atomic batches, with one ledger evaluation/write per leg. A retained ordinary nonzero build-producer outcome records the original begun runs as failed before the controller raises, with immutable copies of the failed launch, capture, proof, exit and log. It admits no successful compile coverage. Missing proof, provisioning-only work, timeouts and signal interruptions remain unfinished for reconciliation. Provider continuation requires the original nonempty active run mapping and exactly the original selected scope; it cannot restart a completed failed run.

## Validation

Evidence: `~/mojolearn-evidence/release-compile-receipts-20261025/`.

Combined ledger/receipt/dependency/controller/publisher suite: 204 passed, 54 subtests. Final focused compile/dependency suite: 64 passed. The final controller subset had 108 passes and one new fixture missing a `command` attribute; its retained failure was repaired and the regression passes in the final focused suite. Negative cases include development reports, missing capture or expected compiler identity, changed source/flags/packages/compiler bytes/identities, incomplete or ambiguous merged membership, failed exit, mismatched launch, corrupt artifacts, undeclared coverage and unsupported targets/variants. Tests also cover actual expansion preserving variants, selected Pixi environment capture, atomic batch refusal/admission, original failed-run and successful-peer retention, interrupted work, exact provider continuation scope, refusal before file rotation/spawn, credential exclusion and reuse after only the source commit changes. Shell syntax and `git diff --check` passed.

No lane build, rental, upload, publication or canonical-ledger mutation occurred. The new capture path still needs its first actual release leg; these tool tests are not release compilation or runtime qualification.

Final scope review additionally required refusing runtime-library and C++ adapter compile units because their C/C++ executable identities are not bound. That narrowed parser/ledger/dependency suite passed 145 tests and 38 subtests. These units remain unqualified rather than inheriting the Mojo-only proof.
