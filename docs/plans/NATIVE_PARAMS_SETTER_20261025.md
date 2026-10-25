# Native linear parameter setters — 2026-10-25

This source chunk moves `set_params` for 19 linear estimators to one compiled
entry. It does not qualify their Python constructors or any remaining Python
setter, and has not been compiled or run against a native artifact by this lane.

Covered classes: SGDClassifier, SGDRegressor, PoissonRegressor, GammaRegressor,
TweedieRegressor, HuberRegressor, BayesianRidge, ARDRegression, Lars, LassoLars,
QuantileRegressor, Perceptron, PassiveAggressiveClassifier,
PassiveAggressiveRegressor, RidgeClassifier, RidgeCV, LassoCV, ElasticNetCV and
LogisticRegressionCV.

The setter reuses their existing field-only model-selection constructor plans.
It borrows parameter objects, validates all root names before mutation, routes
nested updates to connected native setters, preserves estimator identity, and
resets the same owned fitted fields without deleting unrelated user fields.
These constructors store options unchanged; their existing native fit validation
and numeric-mode capture timing are preserved. No Python constructor, getter,
setter, descriptor or callback is invoked from the native setter. Unsupported
Python callback or lifetime owners are refused before release. The shared native
ownership admission covers weak references, native buffers and NumPy object-array
ownership; container traversal in this setter is iterative and cycle-aware.

Reset metadata distinguishes unrelated user names ending in `_` from owned
learned state. A subsequent fit can establish fitted state by its actual learned
fields, including legacy owners that do not write an explicit fitted flag.

## Scope still owed

The shared Python `ParamsMixin.set_params` remains unchanged for 110 endpoints
(129 before this chunk). Across all 196 public setter endpoints the source
inventory reports 38 native boundaries and 158 prohibited source paths, including
48 setters outside that shared mixin. The exact residual class list is retained
in `~/mojolearn-evidence/native-params-setter-20261025/setter-endpoint-summary.json`.
PCA/HostPCA and composite/native-constructor families are not migrated here.
Other classes' own setters, constructors and transitive execution still require
their separate work. A source classification does not establish runtime purity.

## Validation and compile impact

Source-only tests: 11 passed across
`tools/test_native_params_setter_connection.py` and
`tools/test_native_params_metadata_connection.py`. They check all 19 constructor
assignment schemas, native plan connections, owned fitted fields, thin wrappers,
export indentation, borrowed references and reset metadata ordering. Python
syntax checks and `git diff --check` pass. Full logs and endpoint reports are
under `~/mojolearn-evidence/native-params-setter-20261025/`.

Direct compiled binding affected: `_mojolearn_tokenizer_host`. It gains
`params_setter_native_call` and imports `params_setter_native.mojo`; the shared
`mode_metadata_native.mojo` fitted-state change is consumed there and by its
parameter/attribute metadata owners. No numerical kernel or identity arithmetic
changed. Integrated dependency analysis must include any later consumers.
The inherited tokenizer getter indentation fix is already integrated separately.

After the consolidated build, run from the intended installed/source-matched
environment (retain full output and exit status):

```sh
python -m pytest -q python/mojolearn/tests/test_native_params_setter.py
```

This pending runtime suite exercises all 19 public setters plus empty updates,
nested identity/routing, invalid roots, numeric-mode capture, changed class and
instance hooks, direct finalizers, weak-reference callbacks, NumPy object-array
owners, cyclic containers, and fit/reset/refit behavior. The profiling probes
must have zero extra Python frames, prohibited/unresolved native calls, wrapper
work and unobserved threads; independent callback counters supplement profiling.
No compiled runtime result or release qualification pass is claimed here.

## Integrated connection review

At integration `6dd569e26`, native public attribute slots install before the
model-selection layout snapshots. The setter connector then retains those plan
objects and the shared getter context without mutating any class; the final
getter connection captures the installed identities afterwards. Lazy sequence
connection installs new attribute slots before new getter plans and does not
refresh existing snapshots. The reset marker is an ordinary private instance
field: native attribute access leaves it unchanged, and fitted-state reads use
raw native instance storage. Refit restores the selected owner's learned fields.

Review found and repaired a lifetime-admission regression for replacing existing
built-in CV splitters on the four covered CV estimators. Their existing native
constructor plans now admit their storage before release, including the exact
splitter class stored by repeated splitters. This adds no splitter setter;
`cv__n_splits` retains its existing AttributeError when the splitter has no
`set_params`. Incoming foreign metaclasses are also excluded from keyed plan
lookups before their hash hooks could execute.

The integrated source suite passes 22 tests and 2 subtests across setter, getter,
attribute and mode connection checks. The pending public runtime suite now has
40 cases, including eight splitter-replacement cases, the existing nested error
and a metaclass-hash negative control. Full evidence is under
`~/mojolearn-evidence/native-params-setter-integration-review-20261025/`.
