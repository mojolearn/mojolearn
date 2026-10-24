# Native dispatch precision debt

The selector retains every dependency it cannot safely exclude. Native owners
shared by several estimator families therefore cause extra verification work
when the selector cannot prove which operations a cached function pointer runs.
This is a tooling precision limitation, not a numerical or runtime failure.

Path selections expose `precision_limitations` in their JSON output. The CLI
prints a summary and `--why` prints the Python reference and native dispatcher.
Each record names the changed paths retained through that conservative export
closure. A generator record also names the generated outputs that carry its
unresolved dependency. Provenance is followed per export, so sharing a library does not
itself create a diagnostic path. Unknown native branches and aliases remain
broad; these records are analysis evidence, not runtime execution traces. Reporting never excludes a
lane, hides an unattributed path, or admits a numerical mismatch.

Two concrete boundaries remain unproved as of 2026-10-23:

- Mamba's low-bit conversion path reaches `_linalg_impl`. Its public functions
  pass literal `family_lin_*` operations to a shared native owner, and its module
  dictionary is retained as `family_lin_state`. The selector does not yet carry
  callable identity, operation arguments and context slots across those Python
  and Mojo boundaries. The unresolved owner also reaches the QN branch.
- Selecting `LinearRegression` retains import-time archive registration in
  `linear_model.py`. `_native_archive_bridge.connect()` caches and returns
  `(entry, context)`; registration discards the entry with
  `_, context = connect()`. Proving that the cached entry cannot be invoked
  requires return-slot and global-cache provenance. Without that proof, the
  generic archive owner retains forest dependencies.

Neither example proves that Mamba executes QN or OLS executes a forest. A
lane-wide list of observed operation strings cannot safely resolve the gap:
an opaque consumer or another escaped reference to the same entry must keep a
separate unknown edge. A sound implementation must preserve direct-call and
escaped uses even when they share a name, and must track cache writes,
namespace access, tuple/dictionary aliases and native context reads.

The required real-tree tests check attribution, inclusion of the affected
family and visible evidence for extra selections. Synthetic tests still
require proven narrow helper/import paths. The original desired real-tree
exclusions and duplicated numerical/generator bounds remain in three **strict expected-failure** tests named
`test_pending_precision_*`, plus `test_shared_binding_paths_exclude_unrelated_lanes`; only their explicit
exclusion failure is expected. (That fourth one passed for a day after 8188d819d only because the selector had
lost the edge to the classical owner: `_mojolearn_tokenizer_host`'s `classical_native_call` became a stub that
`load_host_module` aliases to `_mojolearn_runtime_classical_host`. `host_routes` follows
`HOST_IMPORT_ALIASES` since 2026-10-24, and the exclusion is pending again.)

Since 2026-10-24 an export is reached from a native body only through a call site (`_mojo_export_refs`: a
whole-identifier string literal, an attribute name, or a call in code), never through a word in a comment,
docstring or message. Before that, `glm/host/qn_oracle.mojo` selected mamba1 and transformer with no
native-dispatch record, because `_mojolearn_estimators_host`'s docstring lists its exports.
Other exceptions remain failures, and an unexpected pass requires removing the
resolved debt marker. Passing required tests with these expected failures is
not a claim that real-tree narrowing is complete.

The 2026-10-24 selector (call-site exports, routed import aliases) selects
490 / 490 / 490 lanes for k-means / host GBDT / host forest; 480 of each carry
unresolved dispatcher records (the shared classical owner), 10 do not. Their
desired counts remain 84 / 55 / 93. (2026-10-23: 555 / 470 / 470, 540 / 470 /
470 with records.)
The required checks retain essential family coverage, refuse unexplained
widening, and validate those records. The independent forest-inference count
54 and neural-inference count 89 (remeasured 2026-10-24) remain ordinary exact assertions.

The neighbors generator writes two driver files and two binding files. Its
selection must include the union of all four outputs, including callers that
only use a binding's metadata. The current union is 524 lanes (469 through
drivers, 55 additional binding callers), so the original less-than-half bound
has the same unresolved native-owner dependency. That bound remains a strict
precision regression; the exact four-output set, union, seam-check narrowing
and output-specific provenance remain required checks. Quoted marker examples
are not generated outputs.

The shared-file guard remains active. Six new carrier files are traced to
commit c1de021978's `_array` -> `_user_objects` -> `tokenizer` import chain,
then the tokenizer's trainer, fixture module and two binding names. Six named
numerical paths in the unresolved owner closure are treated separately:
`cluster/__init__.mojo`, `cluster/host/__init__.mojo`,
`cluster/host/kmeans_oracle.mojo`, `core/ic_min_device.mojo`,
`core/knn_host_predict.mojo`, and `core/rows_bytes_device.mojo`. Each requires
its reported OLS dependency while shared. Any other newly shared file fails
normally; this is not permission to widen the reviewed file set.

Retained investigation: `~/mojolearn-evidence/codex-recovery-20261023/native-failures/`
contains `selector-75e59297a.json`, `python-bridge-paths.json`, the source-only
factory investigations, and the parked `selector-factory-slots-exploratory.patch`.
The unproved cache/slot relaxation is not part of the selector.
The final membership/count audit is retained in `final-selector-paths.json`;
it is source analysis, not native execution or runtime proof.
