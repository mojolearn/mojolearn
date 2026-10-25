# Prep parameter reads — 2026-10-25

This chunk replaces the `_PrepBase.get_params` dictionary comprehension with
one call to the existing `params_metadata_native_call` entry. Its declared prep
schema already implements shallow reads in declaration order, retaining the
original parameter objects. The historically ignored `deep` argument remains
ignored, including objects with Python truth hooks. Constructor, setter and
scikit-learn tag behavior are unchanged.

The frozen `c00c4d8003c84cc74c60fab29687a4da72e24899` inventory identifies 24
affected public getter bodies: BernoulliNB, Binarizer, CategoricalNB, ComplementNB,
GaussianNB, IterativeImputer, KBinsDiscretizer, LabelBinarizer, LabelEncoder,
MultiLabelBinarizer, MultinomialNB, Normalizer, OneHotEncoder, OrdinalEncoder,
PolynomialFeatures, PowerTransformer, QuadraticDiscriminantAnalysis,
QuantileTransformer, RFE, SelectKBest, SimpleImputer, SplineTransformer,
TargetEncoder and VarianceThreshold.

The subsequent source inventory changes exactly those 24 endpoints from
`prohibited_source` to `native_boundary`. This is getter-body progress only:
all 24 inherited `__getattribute__` paths and setters, and 23 constructors,
still have prohibited Python source. The native runtime probes receive bound
methods; they do not qualify the earlier attribute lookup. Whole-class or
whole-product compliance is not established by this change.

No Mojo file, compiled ABI, arithmetic or parameter connection schema changes.
The existing tokenizer metadata entry is sufficient, so this chunk requires
no new binding compilation. Connection runs at import after native attribute
installation and captures the new getter identity before product operations.

Source validation: 13 tests pass across prep parameter connection, shared
parameter connection and public attribute connection suites. Python syntax
and `git diff --check` pass. Complete evidence and the exact inventory delta:
`~/mojolearn-evidence/native-prep-params-read-20261025/`.

After the consolidated build, run in a fresh source-matched process:

```sh
python -m pytest -q python/mojolearn/tests/test_native_prep_params.py
```

The 30 pending cases cover all 24 declared parameter schemas, reference and
order preservation, unchanged fitted/user state, ignored `deep`, no nested
callback expansion, array/callable references, missing-field AttributeError,
passive subclasses, changed properties and Python getter refusal. They require
pytest and the built package; the test file has no skip or optional NumPy
dependency. Constructors run only as test setup. No runtime result, identity
measurement or release qualification pass is claimed by this source lane.
