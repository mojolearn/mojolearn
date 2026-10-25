# Decomposition parameter reads — 2026-10-25

This chunk replaces `_expansion_decomp._Base.get_params` with one call to the
existing `params_metadata_native_call` entry. The existing declared schema reads
parameters in declaration order, retains their object identities and ignores
`deep`. The connection already recognizes this getter and pins its final class,
method and descriptor identities during package import.

The 22 affected getter bodies are AlternatingLeastSquares, CCA, ClassicalMDS,
DictionaryLearning, EllipticEnvelope, FactorAnalysis, FastICA,
GaussianRandomProjection, IncrementalPCA, Isomap, LatentDirichletAllocation,
LocallyLinearEmbedding, MDS, MinCovDet, MiniBatchDictionaryLearning,
MiniBatchSparsePCA, NMF, PLSCanonical, PLSRegression, SparseCoder, SparsePCA and
SparseRandomProjection. The source test reconciles this list against public
exports and inherited getter definitions, including inherited parameter schemas.

No Mojo, numerical source, native entry or constructor schema changes. Existing
binaries can be reused; this chunk invalidates zero native compile groups.
Constructors, setters and public attribute lookup remain separate qualification
boundaries. Getter-body progress does not establish whole-class compliance.

Source validation passes 10 tests across decomposition, preprocessing and shared
parameter metadata connection suites. The first source-test attempt used system
Python, which has no pytest installed; the successful run uses the existing
project test environment. Complete logs and handoff are retained at
`~/mojolearn-evidence/native-decomp-params-read-20261025/`.

The orchestrator must run the 30 public runtime cases against the updated Python
source and retained native binaries:

```sh
python -m pytest -q python/mojolearn/tests/test_native_decomp_params.py
```

These cover all 22 classes, shallow declaration order, borrowed parameters,
unchanged fitted/user state, ignored `deep`, no nested callback expansion,
buffer and callable identity, missing-field AttributeError, passive subclasses,
property and attribute-hook refusal, and independently returned dictionaries.
Constructors and bound-method lookup occur in test setup; the native audit
surrounds the public getter operation. No runtime or release qualification is
claimed by source-only validation.

The separate 24-class preprocessing setter recommendation remains pending for
the next necessary native freeze. Its state-preserving assignment contract must
not be replaced with the existing linear setter's fitted-state reset behavior.
