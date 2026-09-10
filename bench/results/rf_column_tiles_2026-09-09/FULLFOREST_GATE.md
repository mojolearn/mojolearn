# Native full-forest tile gate, Apple M4, 2026-09-09

PASS: all 11 cases have equal reference/tile2/tile4 model, prediction, and
(where enabled) OOB fingerprints within each of FAST, DETERMINISTIC, IDENTICAL.
This is 99 completed case fits across 9 native builds; each case trains 3 trees.
No comparison of hashes across different numeric modes is required.

Run command:

```sh
tools/check_rf_column_tiles.sh bench/results/rf_column_tiles_2026-09-09
```

The gate verifies compiled mode and flag readback. Per-case launch logs prove
18–24 tiled histogram launches in each eligible case, and zero in the
17-class/128-bin capacity fallback and 257-bin search fallback. Reference runs
have no tile launches. The 256-bin boundary does take the tiled path.

Cases cover classification (2, 5, 17 classes), regression, weighted and
unweighted objectives, bootstrap on/off, OOB predictions, a leaf-count cap,
and thirteen features with heterogeneous cardinalities (ten-column round plus
three-column tail). Sampled-label gathering is enabled by the existing RF
default. The leaf-count cap is not a claim of a separate best-first growth API.

`summary.txt` reports all runs; hyphenated `*.log` and `*.launches.log` are the
raw full-forest outputs. `toolchain.txt` records compiler version. Binary hashes
and local ignored build paths are in `fullforest_binary_manifest.json`.
The underscored timing logs in this directory belong to the separate kernel
benchmark. Launch logging is enabled here: these runs make no timing claim.

The model hash folds every sparse-node field used by the reference helper,
leaf values, tree IDs and output dimensions, prediction bits, and OOB values.
It is not a hash of a serialized object including incidental metadata/padding.
Local-only validation; no NVIDIA validation or remote execution occurred.

Compiler and launch logs are stored as `.log.gz`; their raw contents are preserved.
