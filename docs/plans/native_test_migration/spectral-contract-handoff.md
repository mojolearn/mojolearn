# Spectral native request contracts

Dependency `f0201aa23b` is the pilot's independently pickable API/provider.
The58 new CPU cases are authored and unrun. Import
`checks.native_tests.spectral_contract_suite.run_checks`, or use
`check-native-spectral-contract` from the repository root. This is additive to
the separate CPU14/GPU8 dense preparation slice, not numerical Spectral coverage.

The suite independently checks both sides of the fixed native descriptor:
28 driver calls against prescribed raw fields, nine provider calls against
prescribed typed fields, nine exact error propagations,11 malformed envelopes
and one provider table. Routes, Boolean flags, distinct component widths,
binary64 tolerance and integer handles beyond binary64 precision are covered.
Every fixture has private storage and runs through `host_parallelize`.

Nothing removed: original Python numerical and boundary coverage remains.
No product/kernel, algorithm, default, benchmark or timing changes. Fake data
addresses are never dereferenced; these mocks establish no numerical or public
admission behavior. Actual backend and owner coverage awaits the pilot's next
dependency. Source-only evidence is in `E/pca-native-tests/spectral-contract-static.json`.
