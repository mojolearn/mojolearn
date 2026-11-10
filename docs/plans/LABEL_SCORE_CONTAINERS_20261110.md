# Native label score containers

Branch `lane/mojo-finite-tests-20261110`; source-only work, no lane builds or tests.
This batch depends on native label driver/admission `0268d8b446` and `521abd4e6c`.

## Accuracy program plan

`core/labels_accuracy_plan.mojo` constructs the same PAIR_KEY, GROUP_SORT,
GROUP_SUM, OFF_DIFF and COUNT stages in `MetricProgramPlan`. It preserves the
stage order, offsets, three group outputs, weighted total group, 256-row soft
binary64 leaves and fixed fold tree. No tuned kernels or dispatch switches
changed. The existing runner still owns ranges, resident input admission,
download ranges and output allocation.

`bindings/classical_labels.mojo` now supplies the native plan and a native List
of Python buffer-owner pins to the existing `_metrics_run_plan`. The private
`_Program`, Python input tuples/stage lists/skip pairs, program namespace object
and result `buffer_info` call are removed. Weight copies remain exactly where
needed by the existing Array input contract. This is not yet a claim that the
shared metrics runner's descriptor/connection path is Python-free.

`checks/native_tests/labels_accuracy_suite.mojo` has 40 independent native
fixtures over the real host planner and COUNT kernels: weighted and unweighted
fractions with independent integer-count oracles, class renumbering, missing
codes, neighboring leaf boundaries, multiple fold levels, zero denominator,
all-hit and invalid-code cases, and Int32 arena overflow. Fixtures run through
the existing host pool with private arenas. Thin entry and pixi task are added.
No runtime result or performance claim is inferred from their source.

Root run owed: `bash tools/compile_slot.sh pixi run check-native-label-accuracy`.
Product affected roots/groups are the same 13-root shared-label closure recorded
in `NATIVE_LABEL_DRIVER_20261110.md`; additionally the new native fixture imports
`x_metrics.host.program`. Keep installed score/weighted score/public mutable
classes and invalid input precedence tests. Full evidence is under
`~/mojolearn-evidence/python-mojo-migration-20261110/label-score-plan/`.

## Remaining work

The score-label object admission/representation, Python sets/sorted union/index
dictionaries, and generic classes fallback remain in this commit. Their typed
native migration is the next separate batch. Public Python seam tests stay.
