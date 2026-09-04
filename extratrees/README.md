# Extra Trees

GPU extremely randomized trees. Provenance and current scope are defined by `DERIVATION_MAP.tsv`
and `NOT_IMPLEMENTED.tsv`; implementation history remains available in Git.

Tree construction needs fixtures for random-state mapping, split ties, missing values, and leaf reduction.
Only measured FAST-path improvements should be retained.
