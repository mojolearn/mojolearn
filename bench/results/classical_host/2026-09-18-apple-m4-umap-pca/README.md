# Fresh installed-wheel UMAP and whitened full PCA saved models

Recorded on Metal from the installed 0.8.7 wheel at aff968968's source
(full commit aff968968f3ffc45844e36f3fd6260391eb829e5), after the UMAP row-local
transform fix. Nine UMAP and nine PCA fixtures; host-check.json records CPU
inference equality for all 18. Wheel SHA and capture details are in
../../identity_break/2026-09-18_installed-apple-properties/README.md.

This is measured replacement evidence for the stale September 15 UMAP
expectations. It does not silently rewrite historical records, refresh their
old GPU columns, or declare the release gate passed.
