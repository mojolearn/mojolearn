# Declared parameter assignment — 2026-10-25

The preprocessing and decomposition bases have the same shallow setter
contract: process keywords in insertion order, assign supplied references,
preserve fitted state, and reject undeclared names without rolling back earlier
assignments. They do not call constructors, coerce values or route nested
parameters. Their distinct invalid-name messages remain explicit native plan
metadata. This differs from the existing 19-class linear setter's constructor
and fitted-state reset contract.

Both public setter bodies now call `declared_params_native_call` once. Import
connection follows the final parameter metadata snapshot and retains exact
declared schemas, setter/storage hooks and message styles. Passive subclasses
retain the built-in contract; changed setter, dictionary or parameter descriptor
hooks refuse before execution. An empty update remains an identity/no-op even
when fields are missing. Updating one declared field does not require other
constructor fields to exist.

Replacement checks the previous owner's lifetime before each write. This keeps
the original partial-mutation order while preventing Python finalizers or weak
reference callbacks. Existing native estimator plans admit their pinned class
layouts and recursively check owned fields. Only already-registered native
archive finalizers qualify; arbitrary native-looking descriptors do not.
Loaded modules, native estimator classes and the six existing SelectKBest score
functions are independently retained at connection, so replacing their field
reference cannot destroy them. New parameter references are stored without
executing callbacks; assigning the identical old reference needs no release.

The affected preprocessing setters are BernoulliNB, Binarizer, CategoricalNB,
ComplementNB, GaussianNB, IterativeImputer, KBinsDiscretizer, LabelBinarizer,
LabelEncoder, MultiLabelBinarizer, MultinomialNB, Normalizer, OneHotEncoder,
OrdinalEncoder, PolynomialFeatures, PowerTransformer,
QuadraticDiscriminantAnalysis, QuantileTransformer, RFE, SelectKBest,
SimpleImputer, SplineTransformer, TargetEncoder and VarianceThreshold.

The affected decomposition setters are AlternatingLeastSquares, CCA,
ClassicalMDS, DictionaryLearning, EllipticEnvelope, FactorAnalysis, FastICA,
GaussianRandomProjection, IncrementalPCA, Isomap, LatentDirichletAllocation,
LocallyLinearEmbedding, MDS, MinCovDet, MiniBatchDictionaryLearning,
MiniBatchSparsePCA, NMF, PLSCanonical, PLSRegression, SparseCoder, SparsePCA and
SparseRandomProjection. Source tests reconcile both lists with the exported
classes that inherit each setter, including inherited class schemas.

The native operation is a separate tokenizer owner. Source closure analysis
finds exactly one affected development group, `host/tokenizer`, with no
unresolved closures. The logical entry is `_mojolearn_tokenizer_host`; the
release assembly includes it in `_mojolearn_native_host`. Numerical source,
runtime_classical, runtime_model_selection and the other native groups are
unchanged. The new required native entry must be built before importing the
updated package. No lane compilation or runtime qualification was performed.

Source validation passes 22 tests across the new setter connector, existing
linear setter and both getter families. Complete output, source closure proof
and handoff are retained under
`~/mojolearn-evidence/native-prep-params-set-20261025/`.

After the consolidated tokenizer build, run:

```sh
python -m pytest -q python/mojolearn/tests/test_native_declared_params_setter.py
python -m pytest -q python/mojolearn/tests/test_native_params_setter.py python/mojolearn/tests/test_native_prep_params.py python/mojolearn/tests/test_native_decomp_params.py
```

The new file stages 69 runtime cases: all 46 setters; per-family ordered
mutation, exact messages, nested-name refusal, missing-field repair, passive
subclasses and descriptor/setattr/dictionary callback controls; finalizer and
weakref controls; replacement of native built-in estimators and score functions;
and preservation of real fitted RFE, IterativeImputer and random-projection
state. Native Ridge replacement deliberately drops the last application owner,
including a fitted model, so finalizer admission is exercised within the audited
operation. Constructor and method lookup are test setup; their separate native
qualification remains owed where applicable. Source checks establish no runtime
or release PASS, and these 46 setter bodies do not qualify whole classes.
