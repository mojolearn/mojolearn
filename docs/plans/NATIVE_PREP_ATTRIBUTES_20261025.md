# Prep attribute availability — 2026-10-25

This source chunk extends the existing native public attribute owner to the
six prep classifiers whose `fit_transform` availability was still implemented
by a Python descriptor: BernoulliNB, CategoricalNB, ComplementNB, GaussianNB,
MultinomialNB, and QuadraticDiscriminantAnalysis. The other 18 classes covered
by the shared prep parameter getter already meet the dynamic attribute
installer's descriptor requirements. All 24 have installed-boundary probes;
this change does not establish that their constructors, setters or method
bodies are native.

The import-time connector admits only the public prep classes whose resolved
getter is `_PrepBase.get_params`, and only the two exact built-in
`hasattr(self, 'transform')` predicates from `_PrepBase` and `_Classifier`.
It preserves fit/output installers' copied descriptors and wrapped method
identities. The compiled owner reads ordinary instance and class storage,
preserves non-data descriptor shadowing, checks `transform` presence without
calling it, and creates a bound method using CPython's native method owner.
It never invokes `available_if.__get__`, its predicate, or the method body.
Mutated predicate identities/code and descriptor hooks remain explicit errors
before Python execution. Arbitrary properties and other families' availability
predicates retain their existing unported ownership.

Fitted Array output continues through the existing native Array-to-NumPy view
owner, with its ownership and read-only rules. No arithmetic or buffer kernel
changes are part of this chunk. Ordinary Python function binding also uses the
exact CPython method constructor, because reading `function.__get__` dynamically
can invoke an instance attribute supplied by application code.

## Qualification

Direct Mojo binding impact: `tokenizer_host`, through
`bindings/public_attributes_native.mojo`. No lane compile or runtime execution
is claimed. Root compiles the next immutable source freeze. Existing integrated
attribute/parameter runtime probes currently fail on direct native dictionary
field returns; the owning lanes have reported generated argument-state
corruption as a compiler investigation. This extension does not repair or
work around that blocker, and its runtime cases remain unqualified.

Source checks:

```sh
python -m pytest -q tools/test_native_prep_attribute_connection.py \
  tools/test_native_public_attribute_connection.py \
  tools/test_native_prep_params_connection.py \
  tools/test_native_params_metadata_connection.py
```

Installed runtime probes after that compile:

```sh
python -m pytest -q python/mojolearn/tests/test_native_prep_attributes.py \
  python/mojolearn/tests/test_native_prep_params.py \
  python/mojolearn/tests/test_native_public_attributes.py
```

The new prep file contains 37 cases: 24 actual slots/public getter expressions,
six classifier availability/instance shadowing cases, and seven subclass,
mutation, callback-refusal and storage ownership cases. NumPy must be installed
for the ownership case; an import skip is missing coverage. The getter-body
suite contains 30 additional cases. Existing public-attribute cases cover
NumPy-free output, lazy NumPy loading, caller markers, buffer slices and other
native owner classes. No optional scikit-learn tag behavior is changed.

The strict static endpoint report still cannot symbolically resolve the dynamic
attribute installer. It therefore retains all 24 inherited attribute source
rows. No source exemption, fixture pass, or whole-class compliance count is
introduced. At this chunk's initial source base the report remains failed:
3,773 total endpoints, 1,168 prohibited source and 442 unresolved. Runtime
receipts and a later sound installer-analysis improvement are separate work.

Retained lane logs and selected endpoint report:
`~/mojolearn-evidence/native-prep-attributes-20261025/`.
