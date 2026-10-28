# peer2-preprocessing 2026-10-28 handoff

## Prep: lane/peer2-preprocessing-20261027 @ 2c746e42cd (pushed, clean; hooks passed)
- c8297427b7 merges origin/main. Conflicts: bindings/prep_stateless_native.mojo, where the lane's Normalizer dispatch was kept, and tests/array_metadata_owner_probe_pyapi.mojo, where main's newer version was taken.
- 0ad355f218 commits the paused Normalizer fit/fit_transform draft as it was: prep_normalizer_fit, prep_fit_error, prep_target/category/arrow/metadata providers, and the transform provider changes.
- 43cfea91c5 contains the compile fixes:
  - core/prep_boundary_impl.mojo:67,94,123 now use `bitcast[DType.*]`.
  - bindings/prep_fit_error.mojo:45 now reads `BaseException.__dict__["__str__"]`.
  - bindings/prep_normalizer_fit.mojo:74-104: the required-y default now follows an exact-string `_estimator_type`, as `_check_y` does. A classifier type with a y is refused explicitly.
- 2c746e42cd adds the receipt and experiments/peer2_preprocessing_boundary_20261027/NORMALIZER_FIT.txt, which covers the review, the narrow refusals and the open gaps.
- The 9608dee4e5, ed41316e9c and f10fbb18a0 semantics, and the e00a8b393c prep_stateless_c context admission, are in the compiled source.
- Compile receipt: experiments/peer2_preprocessing_boundary_20261027/compile_prep_20261028.json.
  - Source 43cfea91c5, clean tree. Root compile_prep.mojo leads to prep_stateless_c.
  - Mac CPU, identical, -O0, -j10. Exit 0, 23.3 s. sha256 938ff814...
  - Covers prep_stateless_c and all its native callees.
  - Does not cover the Python glue or production module registration. No runtime, identity or vendor run.

## PCA: lane/peer2-decomposition-20261027 @ 213e36ec71 (pushed, clean; hooks passed)
- 9f3952c3a8 merges origin/main. The 5 PCA conflicts were resolved to the lane's versions. Main's PCA commits there were older cherry-picks of the same lane work, and the resolved files are byte-equal to the pre-merge HEAD.
- c6b5f296f9 is a compile fix at providers/python/pca_fit_scalar_pyapi.mojo:233: `view[slice]` replaces `py_call(view.__getitem__, ...)`.
- New root experiments/peer2_decomposition_20261027/pca_public_compile.mojo leads to the public pca_c entry: fit, fit_transform, query, params and archive.
- The head carries the 3bd993dec5, f5b4781287, 29ad490369 and e4fa50751b commits, which until now were local and unpushed, plus fa442d9bf2 (training_scalar dtype reuse).
- Compile receipt: experiments/peer2_decomposition_20261027/pca_public_compile_20261028.json.
  - Source c6b5f296f9, clean tree. Mac CPU, identical, -O0, -j10. Exit 0.
  - Does not cover the solver kernels in _mojolearn_estimators, which are called through connected entries, or the Python glue. No runtime, identity or vendor run.

## Box, evidence
- No box rented. All compiles ran on the Mac through compile_slot.sh.
- Evidence dir: ~/mojolearn-evidence/peer2-preprocessing-20261028/ (prep-attempt1-3, pca-attempt1-3, merge and push logs).

## Pending / held
- Neither branch is merged to main; the orchestrator merges. No runtime, identity or vendor run on either.
- Normalizer, still open (listed in NORMALIZER_FIT.txt):
  - Small-fit context is applied only to pending exceptions of exactly ValueError, ZeroDivisionError or Exception, where the old code used isinstance.
  - A non-number fitting depth raises a different TypeError message.
  - Arrow, categorical and nullable target promotion are not verified.
  - NumPy longdouble scalar lists are not verified.
  - Other X pandas extensions are not covered.
  - The other preprocessing classes (Binarizer, PolynomialFeatures, VarianceThreshold, selector fit, Quantile, Power) still use the Python wrappers.
- PCA, still open:
  - Shared CPython >6000-digit int parsing goes through the Python `_pylong` module. Needs the root's shared repair.
  - pca_c does not yet admit a frozen provider context before classical_connect_numpy.
  - IPCA and the other decomposition operations are not done.
