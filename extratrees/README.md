# Extra Trees

GPU extremely randomized trees. Provenance and current scope are defined by `DERIVATION_MAP.tsv`
and `NOT_IMPLEMENTED.tsv`; implementation history remains available in Git.

Tree construction needs fixtures for random-state mapping, split ties, missing values, and leaf reduction.
Only measured FAST-path improvements should be retained.

## GPU-only training migration

Public Python classifiers and regressors retain `device="gpu"`; `device="cpu"`
is now refused at construction, parameter updates and refit validation. The
native Python binding independently requires GPU selector 1 before reading
input pointers. Existing GPU signatures, selector layout and defaults are unchanged.

Native `fit_extra_trees_classifier` / `fit_extra_trees_regressor` convenience
functions now create a GPU context and call their existing `_device` variants.
Callers already supplying a context continue using those variants. Explicit
`*_reference` routines remain only for independent host-oracle checks and the
clearly labelled reference benchmark; they are not a public CPU backend.
Historical CPU/GPU parity checks describe reference validation, not a supported
CPU training option. Prediction GPU migration is tracked separately.
