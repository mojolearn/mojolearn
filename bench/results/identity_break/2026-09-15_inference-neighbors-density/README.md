# Public CPU inference for the k-NN, radius and KDE option lanes (2026-09-15)

Branch `lane/inference-neighbors-density`. Eighteen lanes that already had internal CPU
reference code now serve PUBLIC CPU inference from a model saved on a GPU:
knn-sqeuclidean, knn-manhattan, knn-chebyshev, knn-cosine, knn-minkowski-p3, knn-rbc,
knn-clf-distance, knn-reg-distance, radius, radius-manhattan, radius-chebyshev,
radius-minkowski-p3, kde-tophat-sqeuclidean, kde-epanechnikov-l1, kde-exponential-chebyshev,
kde-linear-cosine, kde-cosine-minkowski and kde-weighted.

What changed: `RadiusNeighbors.save` and `RadiusNeighbors.load` (format `mojolearn-radius-1`)
and `HostRadiusNeighbors`; the k-NN rbc refusal names the classifier and regressor by class,
so `NearestNeighbors(algorithm='rbc')` loads and searches on the host (the core host binding
exports `rbc_knn_search`); the classical gate and the manifest carry the lanes. No binding
and no kernel changed; the core and estimators host bindings already ship.

Where it ran: the Apple M4, one core (nice 19, one-thread knobs, `-j 1`), shared machine, one
process at a time. Fixtures base, ties and dupes. The Metal identical bindings were copied from
a sibling worktree built at the same Mojo sources; every recorded identity hash that the
166-lane record carries matched its Apple column, which is the check that the copy was right.
Host bindings built fresh from this tree.

| file | verdict |
|---|---|
| `bench/results/classical_host/2026-09-15-apple-m4-neighbors-density/` | 54 models saved by the Metal classes, each reloaded on Metal to the same bits |
| `classical_check_host.txt`, `check_apple-m4_host.json` | `gate verdict IDENTICAL (54 fixtures, 3 GPU columns, exit 0)`: every probe SHA-256, dtype and shape equals the recording, and every identity hash equals the Apple M4, NVIDIA H100 and AMD MI325X infer cells of the 166-lane record |
| `classical_check_sabotage.txt`, `check_apple-m4_sabotage.json` | `-D MOJOLEARN_HOST_SABOTAGE=1`: `gate verdict EXPECTED MISMATCH SEEN`, 50 of 54 identity hashes DIFFER |
| `cpu-apple-m4.json` | the CPU column (CPU-only package copy, reference fits), infer, model and batch stable=54 |
| `diff.four-columns.base-ties-dupes.txt` | against the 166-lane record cut to the three fixtures run: `summary: IDENTICAL=54`, `summary (infer/model): IDENTICAL=96, OWED=12`, `summary (batch): IDENTICAL=54`, `require-columns 4 ... OK (12 OWED)` |
| `owed.json` | the 12 OWED parts: the model cell of the four radius lanes on each fixture (no GPU column saved a RadiusNeighbors before this branch) |
| `diff.four-columns.txt` | the same CPU column against the whole record: the same verdicts, plus 408 `REQUIRE FAIL` lines that each say the CPU column has no cell, all on the six fixtures not run |
| `cpu-apple-m4.sabotage.json`, `diff.four-columns.base-ties-dupes.sabotage.txt` | `summary: DIVERGENT=51, IDENTICAL=3`, `summary (infer/model): DIVERGENT=50, IDENTICAL=46, ONE-COLUMN=12`, `summary (batch): DIVERGENT=50, IDENTICAL=4` |
| `installed_wheel_check.txt` | the recordings checked from an isolated installed test wheel (see the iforest, GMM and HDBSCAN directory's README for the wheel) |

The cells the sabotage does not move are the infer and batch cells of knn-cosine, knn-rbc,
radius and radius-manhattan on `ties`, the integer fixture, where the reversed distance fold
changes no bit; the classical check names the same four fixtures. The IDENTICAL model cells
under sabotage are file hashes of the saved index, which no sabotage reaches.

No box was rented (the release-only GPU rule). Owed to the release record: NVIDIA and AMD
recordings of these 54 models, the radius model cells, and the six fixtures not run here.
