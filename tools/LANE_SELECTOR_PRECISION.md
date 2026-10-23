# Native dispatch precision debt

The selector retains every dependency it cannot safely exclude. Native owners
shared by several estimator families therefore cause extra verification work
when the selector cannot prove which operations a cached function pointer runs.
This is a tooling precision limitation, not a numerical or runtime failure.

Path selections expose `precision_limitations` in their JSON output. The CLI
prints a summary and `--why` prints the Python reference and native dispatcher.
Each record names the changed paths retained through that conservative export
closure. Provenance is followed per export, so sharing a library does not
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
exclusions remain in two **strict expected-failure** tests named
`test_pending_precision_*`; only their explicit exclusion failure is expected.
Other exceptions remain failures, and an unexpected pass requires removing the
resolved debt marker. Passing required tests with these expected failures is
not a claim that real-tree narrowing is complete.

Retained investigation: `~/mojolearn-evidence/codex-recovery-20261023/native-failures/`
contains `selector-75e59297a.json`, `python-bridge-paths.json`, the source-only
factory investigations, and the parked `selector-factory-slots-exploratory.patch`.
The unproved cache/slot relaxation is not part of the selector.
