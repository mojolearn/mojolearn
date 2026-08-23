# E2U: the unsupervised sub-feature matrix

`tools/e2u_matrix_fit.py` is the unsupervised twin of `tools/e2_matrix_fit.py`:
one subprocess per cell, one identity card per cell, a sha256 over every
caller-visible output, REFUSED cells recorded as results, and a REACH table
that says which parameter moved the answer. It covers `mojolearn.KMeans`,
`NearestNeighbors`, `DBSCAN`, `PCA`, `TruncatedSVD` and `LinearRegression`
through the Python surface, which is the path a user takes.

## Run it

    # FAST (the shipped default binaries)
    PYTHONPATH=python python3 tools/e2u_matrix_fit.py bench/results/e2u/<stamp>/fast1
    PYTHONPATH=python python3 tools/e2u_matrix_fit.py bench/results/e2u/<stamp>/fast2

    # IDENTICAL (needs python/mojolearn/identical/*.so:
    #   MOJOLEARN_NUMERIC_MODE=identical bash bindings/build.sh
    #   MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_estimators.sh)
    MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python python3 tools/e2u_matrix_fit.py bench/results/e2u/<stamp>/identical1
    MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python python3 tools/e2u_matrix_fit.py bench/results/e2u/<stamp>/identical2

`--only SUBSTR` runs the cells whose name contains SUBSTR (the reach table
is then partial and the exit code ignores it). Each run prints one line per
cell and a REACH table, and writes `e2u_cells.json` (the record:
`numeric_mode`, commit, platform, input hashes, every cell, the reach
verdicts) plus `<cell>.card` and `<cell>.cell.json` per cell. The record is
rewritten after every cell, so a killed run still leaves one.

The whole matrix is 67 cells (2026-08-23); on an M4 a pass takes a few
minutes in either mode (the first run's four passes are
`bench/results/e2u/2026-08-23_081643/`, with the four verdict tables).

## Judge it

`tools/e2_matrix_diff.py` reads `e2u_cells.json` exactly as it reads
`e2_cells.json` (additive change of 2026-08-23: the file name and the
unsupervised output keys were added to its lists; nothing else moved).

    # run-to-run control, one mode: every card byte-identical is the pass
    python3 tools/e2_matrix_diff.py bench/results/e2u/<stamp>/identical1:ID1 bench/results/e2u/<stamp>/identical2:ID2
    python3 tools/e2_matrix_diff.py bench/results/e2u/<stamp>/fast1:FAST1 bench/results/e2u/<stamp>/fast2:FAST2

    # cross-vendor, IDENTICAL: the Apple directory is the reference column
    python3 tools/e2_matrix_diff.py <apple>/identical1:APPLE <amd>/identical1:AMD --write

Under IDENTICAL two runs on one box must come back IDENTICAL on every
cell (the exit code is 0 only then). Under FAST they need not -- the cells
that differ between two FAST runs are the FAST-unstable list and a finding
to report, not a failure of the driver. FAST vs IDENTICAL is informational
(two binaries, two arithmetics; `knn_k300` is REFUSED in one and runs in
the other).

## The reach table

`REACH` in the driver declares pairs of cells that differ in ONE parameter
and whether the outputs are expected to differ or to be invariant:

    REACHES             expected to differ, and did
    INVARIANT           expected the same, and was (the side field -- the
                        batch count, the query tile -- shows the parameter
                        reached the kernel)
    INERT               expected to differ and did not: a parameter that is
                        accepted and ignored. A BUG (E2's ET max_features
                        finding is the precedent). Exit code 1.
    BROKEN-INVARIANCE   expected the same and differed. Exit code 1.
    REFUSED             the "differ-or-refused" / "same-or-refused" pairs,
                        where one side refuses by design under IDENTICAL (a
                        DBSCAN propagation cap that binds, DEVIATION 507)

## What a cell records

| kind | hashes | side fields |
|---|---|---|
| kmeans | labels, centroids, inertia | n_iter, inertia_value, sum_scale, weight_scale |
| knn | distances, indices | used_query_tile |
| dbscan | labels | n_clusters, n_noise, n_iter (propagation passes), batches (from the card header) |
| pca | components, explained_variance, explained_variance_ratio, singular_values, mean, noise_variance, transformed | |
| tsvd | components, singular_values, transformed | |
| ols | coef, intercept, predictions | intercept_value |

plus `card`, `stages`, `seconds`, or `refused` / `error` / `crashed`.

## Wiring

Not yet a phase of `tools/e1_bootstrap.sh`; the coordinator wires it. The
pixi task lines it wants are in the report that introduced this file.
