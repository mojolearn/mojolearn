# Three-vendor identity at the shipped default, with inference and saved-model columns (2026-09-13)

`tools/identity_break.py` at b65ccc2a (the three-column tool; the bindings were built from
bfde0442 on every box, the 0.8.4 release commit), every public lane (28), nine hostile
fixtures, each cell fitted twice in one process. Three columns per cell: `train` is the
hash the tool always took (outputs of the just-fitted model on its training rows), `infer`
is the same outputs on a held-out draw of the fixture (seed 1) the model never saw, and
`model` is the sha256 of the saved model bytes with a reload check for the eight forest
and GBDT lanes that have save/load.

| column | box | commit of the bindings | cells | verdict |
|---|---|---|---|---|
| apple-m4 | Apple M4, this Mac, Metal | bfde0442 | 252 train, 189 infer, 72 model | stable=252 moved=0 refused=0 |
| nvidia-h100-sm_90a | RunPod H100 80GB HBM3, sm_90a (`bench/results/e1g/2026-09-13_195632-nvidia-h100-identity-three-columns`) | b65ccc2a (same native source) | 252, 189, 72 | stable=252 moved=0 refused=0 |
| amd-mi325x-gfx942 | DigitalOcean MI325X, gfx942 (`bench/results/e1g/2026-09-13_195718-amd-mi325x-do-identity-three-columns`) | b65ccc2a (same native source) | 252, 189, 72 | stable=252 moved=0 refused=0 |

`diff.apple-nvidia-amd.txt`: `summary: IDENTICAL=252` and `summary (infer/model):
IDENTICAL=261, N/A=243`, exit 0. No DIVERGENT, MOVED or RELOAD-MOVED cell on any column.

What this closes. Before today k-means, PCA, OLS, k-NN, KDE and the four GBDT lanes had
only an AMD witness at the 0.8.x default (docs/lanes/BRIEF_amd_confirmations_2026-09-12.md);
every lane now has the three-vendor diff at the shipped default. And no gate had separated
inference from training: the `infer` column is predictions on rows the model never saw, and
the `model` column is the saved bytes themselves, equal on all three vendors for RF, ET and
the four GBDT lanes.

What it does not say. `infer` is n/a for the transductive lanes (DBSCAN, agglomerative,
spectral), for k-means (fit only), Holt-Winters (a forecast has no new rows), and for the
GEMM and metrics functions; `model` is n/a where no save/load exists. Fixtures are at most
20,000 rows by 16 columns. The NVIDIA leg's byte-LM binding build failed on the box
(`gate.txt`: `failed= build_byte_lm`); no identity_break lane uses it. The NVIDIA JSON's
`vendor` field reads `box-arch` because the RunPod runner passes no environment to the body;
the box is named in its leg directory. All three JSONs carry `"commit": ""` for the same
reason (MOJOLEARN_COMMIT was never exported into the body). Since the CPU training lane
(2026-09-13, phase 0) `tools/identity_break.py` REFUSES both: a `--vendor` that is not a box
label (`^[a-z0-9][a-z0-9_.-]*$`, never `box-arch` or another placeholder) and a run with no
commit witness (MOJOLEARN_COMMIT, `git rev-parse HEAD`, or a COMMIT / commit.txt file at the
repository root). These three files are kept as recorded; the next GPU columns cannot repeat
the defect. Their cells' hashed bytes are unchanged by that tool change, and `--diff` still
reads them.
