# Native model-selection admission/dispatch split

The model-selection host compiler stalled on the full owner at freezes `753f11cea` and `29a913409`, and on a reduced export of the actual native clone implementation. A source reduction localized the wait to an unnecessary cycle: clone retrieved constructor parameters through general estimator dispatch, whose separate `__admit__` operation calls clone again.

`msel_estimator_call` still validates the exact connected estimator plan and handles `__admit__` through the complete native clone implementation. Ordinary operations enter `_msel_admitted_dispatch` after that validation. Cloning already has the validated plan, so its known `get_params` lookup enters the same ordinary dispatcher directly. The constructor parameter lookup therefore cannot re-enter admission.

The shared dispatcher preserves the complete existing method body: compiled-entry checks, method recipes, keywords, selected binding and mode, forest fit lifecycle, method-specific entry overrides, scorer response and returned value. Constructor paths for forests, fixed native storage schemas and compiled initialization remain intact. Nested parameter cloning, buffers, containers, native estimator admission and callback refusals remain intact. This change introduces no compiler flags, toolchain modifications, fallback or diagnostic behavior omission.

Both helper call sites obtain their plan from `msel_estimator_plan`. That checks exact connected class identity, class ancestry, method/storage identities and protected instance fields. The clone branch uses only the import-connected `get_params` recipe; the shared dispatcher still verifies both the ordinary compiled entry and any method-entry override. No Python estimator method is called.

## Retained diagnostic sequence

Evidence root: `~/mojolearn-evidence/model-selection-stall-20261025/`.

- Original freezes: controller exit124 after300s without CPU progress;197 warnings each, no error diagnostics. Logs are identical after normalizing the generated-root path.
- V1: failed with missing `gemm.f16_widen` because the diagnostic snapshot omitted namespaces. Preserved as an unsuccessful reduction; it says nothing about the stall.
- V2: complete frozen snapshot, actual clone-only export; exit124 after67.1s, no CPU progress60s,47warnings, no errors.
- V3: ordinary integer recursion, PythonObject recursion with defaults, explicit-argument object recursion and actual estimator-plan-only export all compiled successfully. These controls do not support a claim that ordinary recursion is unsupported.
- V4: removing only general dispatch's admission branch compiled successfully in18.3s. Removing only parameter cloning still stalled, exit124 after67.7s. Both variants intentionally omit behavior and must never ship or execute. Product source retains both behaviors through separate call paths.

## Validation and remaining qualification

Source comparison confirms the ordinary dispatch body, parameter-cloning body and constructor routes are byte-identical; the admission branch is preserved. `tools/test_model_selection_connection.py`:3passed. `git diff --check` passed. No lane compilation or reduced-binary execution occurred.

The orchestrator must compile the integrated model-selection host once and run the existing `python/mojolearn/tests/test_runtime_model_selection_partition.py` and `test_native_prep_no_callbacks.py` against that integrated artifact. These cover public model-selection routing/native errors and native admission/clone use by RFE, IterativeImputer and related owners, including callback refusals and supported fit parameters. Compilation and runtime qualification of this final repair remain pending at this commit. The reduced compile passes do not qualify any release artifact.
