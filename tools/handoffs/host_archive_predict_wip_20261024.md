# Host archive inference: finished on lane/host-archive-native-20261024

Status (2026-10-24): the draft below was completed on
`lane/host-archive-native-20261024` (uncompiled; the orchestrator's single
compile owns validation). Completed: availability checks (DBSCAN,
agglomerative/spectral prediction_data, SVC probability), `_bind` mode
refusals per class, per-recipe binding export checks, setup-time host
binding loads moved to `_classical_host.py`, the GP classifier gather on
the core host binding, raw (non-`__getattribute__`) attribute reads and
native builtins on the GMM/DBSCAN/agglomerative/spectral/GP/GPC/ARIMA/HW
predict paths, and tests in
`python/mojolearn/tests/test_host_archive_predict_native.py`. The original
draft notes follow for history.

# Unfinished host archive inference draft — do not merge into the candidate

Preservation-only handoff from Codex dev_builds. Base is
`50de0f93a59fc72229f4551a92e06139cb8ad769`; the WIP branch is
`wip/codex-host-archive-predict-20261024`. Work stopped for confirmed CV runtime
failures, then the owner requested committing all drafts for Claude. This
branch has not been compiled or executed and has no new regression tests.

The original defect is in `classical_native.mojo`: host archive prediction
reconstructs a model and calls its Python `predict`/`predict_proba` method.
Putting that call inside Mojo does not satisfy the no-Python runtime rule.

## Draft contents

- `_native_host_archive_predict.py`: import-time fixed routing metadata for
  the saved host classes, their native owners, contexts and compiled entries.
- `host_archive_predict_native.mojo`: native reconstruction, route selection,
  class/method/module checks and calls through retained native owner entries.
- `host_knn_predict_native.mojo`: draft whole native host KNN classification,
  probability and regression orchestration using existing host kernel ABIs.
- `classical_native.mojo`: draft routes replacing Python method invocation.
- `_classical_host.py`: setup-time connector call after class registration.

The static inventory found 35 base format entries representing 34 classes,
plus HostForest and HostGBDT. Current expansion doors define no additional
`classical_host_formats` implementations. Classes without prediction methods
must keep their AttributeError behavior. Saved KNN currently rejects string
label metadata; do not invent a new format or narrow an existing contract.

## Important unfinished work

1. **Transitive callbacks remain.** Existing native GMM, DBSCAN,
   agglomerative/spectral and related helpers still read model attributes
   through Python `getattr`/`hasattr`. Some saved classes inherit Python
   `PublicOutputsMixin.__getattribute__`. Use admitted raw fields on the
   reachable prediction paths; calling their compiled entry alone is not a
   complete repair. GP label-kind checks still use numbers ABCs and some label
   output paths construct array.array. Forecast/archive helper closures still
   contain runtime stdlib imports and array constructors. Reuse the qualified
   loaded-module and native array-storage helpers where appropriate.
2. **Output parity needs explicit proof.** The old archive-to-Python method
   callback introduced an internal mojolearn frame, selecting internal output
   behavior. Direct reentry changes frame depth. The draft deliberately routes
   Linear/Ridge to `predict_score` and host inference with `outside=False` to
   preserve the old archive dtype/container behavior; this has not been tested.
3. **Review setup and error metadata.** Missing host binding exceptions become
   strings in setup recipes. The native recipe consumer currently recognizes
   only the ImportError prefix in one place; other recorded errors need correct
   native propagation. Verify import order, optional missing-binding behavior,
   dependency-free import, class ancestry/descriptor protection and CPU host
   routes in GPU packages. Module validation currently scans every retained
   prediction module per operation, not only the selected recipe.
4. **KNN is uncompiled.** Check all native signatures, shape/type conversions,
   pointer-owner lifetimes, integer class metadata, multioutput probability
   blocks, sentinel output buffers, class-mismatch messages and query-tile
   metadata. Existing host core exports only `knn_classify`/`knn_regress`, not
   resident variants. Preserve integer-label dtype and output shape semantics.
5. **Qualification is wholly pending.** Author actual public native-audit,
   old-boundary parity and refusal tests across every saved prediction family,
   including absent methods, DBSCAN prediction_data, forecast scalar arguments,
   descriptor/binding replacement and CPU routing from GPU packages. Keep
   numerical/reference evidence separate from callback compliance.

The independent held findings remain open: archive pathname I/O audit/signal
behavior, foreign materialization, hidden Arrow ownership/finalizers, and cold
sklearn tags. This draft does not repair or waive them. No source deletion,
surface removal, callback fallback, compiler patch or unsupported build mode
is authorized.

## Evidence and continuation rules

Mac evidence root:
`/Users/andrewhendel/mojolearn-evidence/codex-recovery-20261023/`.

- `host-archive-predict/host-methods.json`: AST-resolved saved method inventory.
- `policy-refresh-f101a7d8c/REVIEW.md`, `report.json`, and
  `source-provenance.json`: all five held policy findings and source identity.
- `dev-builds-final-handoff/`: preservation commit/push receipts and worktree
  inventory.
- `cv-row-ownership-50de/REVIEW.md`: separate finished CV source repair
  `ff7e3a57f42a48a77954f09622af837b1941cfc0`; do not confuse it with this WIP.

Keep logs out of context: save complete output to files, use targeted rg/grep
with bounded surrounding lines and short tails, and summarize exit status,
coverage, failures, and evidence paths. Expand only relevant diagnostic blocks;
never hide failures or infer full success from filtered output. Propagate this
reminder to delegated agents. Root owns integration and validation; source
lanes do not run compilers, tests, GPU jobs, SSH or rentals.
