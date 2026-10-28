# Neural provider ownership and next-stage library inventory

Source base: `904bd41c93fa1cf9012f174ade1e7065d5e2beb1`. This is a shared vendor-neutral layout migration; root performs local Mac compilation and public runtime validation. No GPU campaign, release, main update or performance default is changed by this lane.

## Delivered ownership changes

36 product boundary modules move into `providers/python/`, using an origin-prefixed name ending in `_pyapi.mojo`: 24 from autograd, 10 from sequence, one from training and one from core. `migration.json` records every old/new path, pre-move source hash and 121 direct import sites. Public module registrations and APIs remain unchanged. A source regression reverses import/path substitutions and checks each moved body against its original hash. No arithmetic moves into the provider directory and no runtime callback is claimed removed by these relocations.

The seven native execution imports previously routed through `bindings` were not Python callbacks: six imported the pure classical conversion helper, and one imported a sampler reexport. The classical conversion implementation moves byte-for-byte to `core/classical_cast.mojo`. Generation imports point directly to the existing `core/generation_policy_impl.mojo`; the redundant binding reexport is deleted. This is genuine native ownership, not a provider helper renamed to look native. A separate commit contains these changes so root can integrate/build incrementally.

`tools/check_neural_provider_layout.py` checks **all six local native trees** (autograd, sequence, training, transformer, mamba, core), following imports beyond those directories. It rejects provider/bindings imports, Python imports, Python helper tokens and direct CPython call symbols. 706 source roots reach 1080 modules with zero violations. Exactly three `training/checks` Python-using fixtures remain testing exclusions, not product exemptions. Their paths are enumerated in the checker and manifest. Comments/documentation may still describe Python boundaries; textual mentions are not runtime dependencies.

Upstream standard/MAX/native packages are explicit external dependency boundaries, whose internals this source guard does not inspect. The expanded closure reaches `nn.topk`, supplied by installed `lib/mojo/nn.mojoc`; its exact local path, byte count and SHA256 are retained in `external_native.json`. This is an exact native-module allowlist entry, not a blanket provider or unresolved-import exception.

## What this does not prove

Zero imports in these local execution trees does not mean the whole library is Python-free. Generation/Samba orchestration still lives in Python-facing binding modules and calls provider state/GIL helpers around native descriptors. Framework capture/admission, foreign-buffer compatibility paths, optional owned/tape routes and nonresident training dispatch require their own operation-level evidence. The relocation does not silently turn these into native-only runtime owners. `library_inventory.json` lists concrete remaining orchestration-to-provider edges and the required ownership inversion; moving those entire loops into another folder would not fix them.

The same machine-readable inventory maps 40 additional native family directories to existing Python-facing sources and binding entries, and prioritizes bounded work: finish neural public validation; shared classical input/preprocessing boundaries; tree/ensemble/meta-estimator orchestration; linear/SVM/clustering/decomposition; neighbor/manifold/time-series owners. These are static ownership findings, not measured callback costs or approved algorithm changes.

## Validation and integration

67 source tests plus 3 subtests passed, including architecture negative controls, lossless relocation, neural metadata/dispatch, path admission and native execution/import guards. Relocation required updating the active 29-recipe source inventories. Two previously reproduced stale import tests now provide the optional-NumPy fixture and check the actual pinned native tuple iterator; the generation candidate source test now uses the metadata refactor's `head_profile` name. No test was weakened to accept Python/provider backedges.

`affected_groups.json` records 12 binding entry closures intersecting moved source, resolving the build's `-I bindings` imports. This is a conservative source-impact map, not an instruction to repeat unchanged runtime/identity work. Root retains responsibility for current-main additional consumers, development-group compilation, installed/local public probes and source/binary provenance. Preserve existing numerical evidence wherever relevant inputs remain equivalent; do not report source checks as compiled ABI correctness or performance evidence.

Complete logs and summaries: `~/mojolearn-evidence/neural-provider-layout-20261027/`. The earlier read-only raw inventory is retained under `~/mojolearn-evidence/provider-inventory-20261027/`.
