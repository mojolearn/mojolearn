# Native forest auxiliary orchestration — 2026-10-25

This chunk moves `ForestProtocol.apply`, `predict_with_leaves` and
`staged_predict`, plus their private combined helper, into the existing compiled
tree owner. The six public owners are RandomForestClassifier,
RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
DecisionTreeClassifier and DecisionTreeRegressor: 18 public method bodies.

The unchanged `x_trees_forest_auxiliary` entry in `xtrees/api_pyapi.mojo` already
implements every host/device tree walk and ordered reduction. The new owner
only admits storage, converts X through the existing native tree input path,
allocates requested buffers and supplies the same six arguments. RF and
DecisionTree retain the RF input convention; ExtraTrees retain theirs. Leaf IDs
remain tree-relative Int32 `(rows, trees)`. Raw final means remain Float32
`(rows, outputs)` and eager increasing-tree mean prefixes remain Float32
`(rows, trees, outputs)`. Classifier means remain undecoded votes. Allocation is
limited to requested outputs; no inference-engine or numerical-default change
is made.

The orchestrator captured the actual old public behavior at
`e087692fd764c443b5db746cb9eaac053373e4c6`, before the port. All six owners fitted
the same explicit 12-by-2 fixture; all 18 operations returned successfully, with
unfitted and wrong-width refusals retained. Even external callers with NumPy
loaded received `mojolearn._array.Array`, or a tuple of Arrays. This chunk keeps
that measured boundary. Applying normal predict's NumPy conversion or regressor
widening here would change the established auxiliary contract.

Baseline evidence:
`~/mojolearn-evidence/forest-aux-native-20261025/legacy-public/legacy-public.json`,
SHA256 `0c51bf46cd83f1db87556ccc71780db26ec157638c1202432b078939c6f46f14`.
The capture retains exact types, shapes, dtypes, output bytes and hashes,
constructor/effective settings, input/target bytes and refusal details. Its
successful legacy execution does not claim Python-free runtime compliance.

Native admission retains the old fitted RuntimeError and rank/empty/width
refusals. Custom input/model-buffer protocols are refused before hooks. Before
the address-only kernel call, model storage must have the native flat dtypes
and lengths required by its offsets; spare trailing storage remains accepted.
Dimension checks repeat the existing Int32/address-width ABI limits before
output allocation. These are storage checks, not dimension-specific routes.

Source validation passes 16 tests across the new public-door/connection suite
and existing tree registration suite. The first new test run found a test-only
AST selector error (attempting literal evaluation of an exception-class tuple);
the repaired selector passes, and both logs remain retained. Python syntax and
diff checks pass. Source dependency analysis finds exactly `host/runtime_classical`,
with no unresolved closures. The logical entry is
`_mojolearn_runtime_classical_host`, assembled in `_mojolearn_native_host` for
release. No device/kernel source, binary, experiment switch or arithmetic is
changed. Lane compilation and new runtime execution remain prohibited.

After the consolidated owner build, run the 43 public runtime cases:

```sh
python -m pytest -q python/mojolearn/tests/test_native_forest_auxiliary.py
```

They cover every owner/method against both the retained Python orchestration
and an independent tiny-tree reference, exact output bytes and prefix order,
external/internal Array results, output ownership, supported input layouts,
unfitted/width/rank/empty/nonfinite refusals, input/model-buffer callback counters,
and the private combined output request. The old implementation lives only in
`tests/_legacy_forest_auxiliary.py`; its work is outside native-call audits.

The orchestrator can also rerun retained `capture_legacy.py` on the new freeze
and use `compare_legacy.py` to compare all 54 successful/error method results.
Both scripts and full logs live in the evidence directory above. Matching old
behavior is separate from passing the new callback/runtime audit. No runtime,
identity, performance or release qualification is claimed by source checks.
