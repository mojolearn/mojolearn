# Full-reset parameter setter candidate

This is a source candidate, not release qualification. The 77 selected
`ParamsMixin.set_params` implementations now have one native entry and retained
field-only constructor schemas. No native build or public runtime test was run
on this lane. A source inventory classification of `native_boundary` does not
establish transitive callback freedom.

The selected constructors are unchanged. The native setter reproduces their
ordered field writes, including six classes with fixed extra fields and eight
with repeated writes. It borrows parameter references, keeps the existing
instance dictionary, validates all root names before nested mutation, and
clears all previous dictionary state only after ownership admission. It does
not allocate a replacement through a Python constructor. Numeric-mode wrapper
events preserve the fresh-constructor rule and ordinary inherited class values.

The existing nineteen linear setters retain their distinct fitted-field reset
contract. The 46 declared shallow setters remain unchanged. The 33 constructors
that require validation or native initialization are outside this migration.
Twenty-four explicit host inference subclasses retain their old setter rather
than inheriting a new refusal from a changed base; their remaining runtime debt
is not exempted or claimed fixed.

## Required public runtime qualification

`python/mojolearn/tests/test_native_full_reset_params.py` declares 268 cases.
Run the KernelRidge pilot first, then the class-wide constructor-oracle,
dictionary identity, error, nested mutation, descriptor and lifetime controls.
The constructor oracle executes only in test setup, outside the audited setter.
The test file must not be treated as passing until the integrated binary runs it.

The new Mojo source and its registration invalidate only the development
`host/tokenizer` group. The group-closure source test verifies this. No arithmetic,
device kernel, compiler option or production build layout is changed here.

## Inherited nested-setter blockers

The existing model-selection registry connects native setter ABIs for
LinearRegression, Ridge, PCA, scalers, LogisticRegression and LDA. The new
composition can forward these exact entries without calling a Python method.
Their export identities alone do not establish safe transitive execution:

* `classical_metadata.mojo:classical_set_params`,
  `scaler_lifecycle_native.mojo`'s setter, and
  `logistic_admission.mojo:logistic_set_params` clear arbitrary instance fields
  without the full ownership admission added by this candidate. The new nested
  lifetime controls deliberately expose this remaining obligation. An external
  reference in a test is not a justification for omitting lifetime admission.
* The logistic setter imports `builtins` during execution. This needs a supported
  import-time native connection, plus an import-hook negative control, before
  claiming that nested path is callback-free.
* LDA performs shallow assignment and retains unrelated fitted/user fields. Its
  test must preserve that behavior; applying full-reset semantics would be wrong.
* Forest setters are absent from that registry and their existing native setter
  imports `builtins` and clears instance state without full ownership admission.
  A supported nested forest route remains owed.
* Optimizer setters also import `builtins`; search setters use the existing
  cloning helper. Neither is silently connected as a callback-free,
  reference-preserving nested route. Supported built-in combinations remain
  release blockers; refusing them is not completion of this migration.

Resolve the affected child paths and run the positive and refusal controls
before release. Do not weaken the tests, alter the nineteen existing setters,
claim 77 fully qualified methods from source classification, or restart broad
qualification before the bounded public-runtime checks pass.
