# The CPU column, phase 1b: agglomerative, et-clf, et-reg, iforest (2026-09-14)

The fourth column of `tools/identity_break.py` for the four lanes of
`docs/lanes/BRIEF_cpu_training_2026-09-13.md` phase 1 that the 2026-09-13
merge (c4883127f) left open, run on this Mac (Apple M4, macOS 26.5, Mojo
1.0.0, host builds `--target-cpu apple-m1`) through the CPU-only package
path: no GPU set under `python/mojolearn/`, the host bindings under
`python/mojolearn/host/` (`_mojolearn_core_host`, `_mojolearn_solver_host`,
`_mojolearn_trees_host`, `_mojolearn_svm_host`), so `mojolearn.vendor()`
is `cpu` and `--vendor cpu-apple-m4`. Every JSON carries `commit`
6796ceff9 (the branch point; `MOJOLEARN_COMMIT`) and the `host` object.
Each was diffed against the three 2026-09-14 GPU columns
(`../2026-09-14_46-lanes/{apple-m4,nvidia-h100-sm_90a,amd-mi300x-gfx942}.json`)
with `--require-columns 4 --lanes <lane>`. The lane's verdict is its nine
train rows, its infer and model rows, and the `require-columns` line; the
whole-column summary counts the other 42 lanes' three GPU columns too.

| file | lanes | verdict |
|---|---|---|
| `cpu-apple-m4.agglomerative.json`, `diff.agglomerative.txt` | agglomerative | 9 train rows `IDENTICAL x4`; infer `n/a:transductive`, model `n/a:no-save` on all four; `require-columns 4 over ['agglomerative']: OK` |
| `cpu-apple-m4.et.json`, `diff.et.txt` | et-clf, et-reg | 18 train, 18 infer and 18 model rows `IDENTICAL x4` (the model column's RELOAD check predicts through the trees host binding's `et_predict`); `require-columns 4 over ['et-clf', 'et-reg']: OK` |
| `cpu-apple-m4.iforest.json`, `diff.iforest.txt` | iforest | 9 train and 9 infer rows `IDENTICAL x4`; model `n/a:no-save` on all four; `require-columns 4 over ['iforest']: OK` |

The sabotage arm: the same bindings built with `-D MOJOLEARN_HOST_SABOTAGE=1`
into a scratch directory, loaded through `MOJOLEARN_HOST_DIR` with
`MOJOLEARN_HOST_ALLOW_SABOTAGE=1`.

| file | lanes | what the define moves | verdict |
|---|---|---|---|
| `cpu-apple-m4.sabotage.agglomerative-et.json`, `diff.sabotage.agglomerative-et.txt` | agglomerative (et-clf, et-reg with the FIRST hook placement) | `linkage_oracle.mojo::host_kruskal` walks the sorted edge keys descending (the maximum spanning tree) | agglomerative 9 of 9 `DIVERGENT`, `parts differ: labels`; the ET lanes stayed `IDENTICAL x4` because the first hook sat in `uniform_threshold`, which the restated device search never calls (recorded, not hidden) |
| `cpu-apple-m4.sabotage.et.json`, `diff.sabotage.et.txt` | et-clf, et-reg | `pcg_rng.mojo::SplitKey.generator` burns one extra draw on every keyed stream | 18 of 18 train, infer and model cells `DIVERGENT` |
| `cpu-apple-m4.sabotage.iforest.json`, `diff.sabotage.iforest.txt` | iforest | `xorwow.mojo::curand_uniform` advances one extra step before every split fraction | 9 of 9 train and infer cells `DIVERGENT`, `parts differ: scores,predict` |

Timings on this Mac under the two-thread cap, whole lane (nine fixtures,
two fits each, plus the infer and reload probes): agglomerative 24 s
(2,000 by 4), et-clf plus et-reg 15 s together (16 trees, depth 8, 20,000
by 16), iforest 8 s (16 trees, 256 samples, scored on 20,000 rows). The
brief's "time first" question for the ET reference and the isolation
forest oracle is answered: neither is within two orders of magnitude of a
runner's 60 minutes.
