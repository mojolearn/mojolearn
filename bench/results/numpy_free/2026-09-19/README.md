# NumPy restricted to independent testing

The product's remaining NumPy uses (token corpus preparation/batching and
parallel RBFSampler) now use Array buffers. Wheel metadata has no NumPy
requirement, including extras; NumPy itself is not bundled. Optional identity
and verification commands still use independently installed NumPy as their
oracle. Their source, fixtures and reference hashes are unchanged.

Validation: 26 focused tests passed (independent NumPy byte comparisons,
blocked product imports, malformed worker results/cleanup, and wheel dependency
negative controls). A clean installed macOS candidate with no NumPy ran the real
compiled tokenizer, corpus preparation/batching and Apple RBFSampler: three row
shards matched the ordinary transform bit for bit. Empty-input refusal remains.
See installed-smoke.json. Linux and macOS candidates passed the wheel audit
(124 and 66 native files respectively), including the existing libm guard.
Linux execution was not repeated for this Python-only change.

Candidates are private repacks of the previous libm-free candidates, replacing
only the two product modules, dependency metadata and this contract document;
RECORD was regenerated. All native binary bytes are unchanged. See candidates.json
for exact wheel hashes and changed members. No PyPI publication was performed.
The existing hardware matrix remains retained; this is not new hardware evidence.

Reproduce focused tests with the current native helper available:

```sh
python -m pytest -q python/mojolearn/tests/test_numpy_free_features.py packaging/portable_math/test_wheel.py
python packaging/portable_math/wheel.py --audit-only path/to/candidate.whl
```

Local build/smoke scripts and private wheels are retained at
`/Users/andrewhendel/mojolearn-evidence/numpy-free-2026-09-19/`.
