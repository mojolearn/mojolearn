# bench: progress

Lane `bench` of the algorithm expansion: make `tools/bench_board.py` race every
new algorithm, measure nothing. Worktree `~/mojolearn-wt/algos-bench`, branch
`lane/algos-bench`. Evidence: `~/mojolearn-evidence/algos-bench/`.

## Where it stands

- NEW family `algos` (`tools/bench_board_algos.py`, wired into
  `tools/bench_board.py`, in the default families): 156 algorithm lanes, 295
  races (taxi + Istella-S, or the lane's own data: text, taxi-hourly,
  taxi-zones, synthetic). Before: 93 races on every vendor. After: 388 races
  (Apple 1,704 fit cells, NVIDIA 1,496, AMD 1,343). Merge 1 (861bd78ea): 375
  races; merge 2 aligned the board to the classes the lanes exported
  (solve/lstsq/randomized_svd, LSTM/GRU/RNN Classifier/Regressor, AutoARIMA
  search/fit, STL/VAR statsmodels shapes, Conv forward/backward/set_weights,
  explainers, DARTRegressor, PLSCanonical, SelectKBest/RFE support); 207 of
  the 295 races find their class in the source tree at that merge. The existing
  93 races' plan lines are unchanged (dry-run diff before/after, evidence
  `before_*.txt` / `after_*.txt`).
- Our arms on every algos race: `ours` (IDENTICAL), `ours-fast` (Apple),
  `ours-cpu`. A class the installed wheel does not export reads
  `SKIPPED: not built yet` (worker event `skipped`, cell status
  `bench_board.ALGOS_SKIPPED`); opponents still race.
- Tests: `tools/test_bench_board_algos.py` (new) plus the family counts in
  `test_bench_board.py` / `test_bench_board_cpu_mem.py` (the cpu_mem counts
  were stale at 88 races on main; fixed to 93 with the pre-expansion
  families named). 86 pass on the pod.
- Plumbing smoke (opponent side, `--max-rows 2000`, one round, labelled
  plumbing, no times quoted): `~/mojolearn-evidence/algos-bench/plumbing_smoke.py`,
  logs `smoke_nvidia.log` (all opponents), `smoke_rapids.log` + `smoke2_rapids.log`
  (cuML/cuVS/cuGraph from a clean rapids venv: the pod image's torch-cu124
  libraries break libcuml inside a --system-site-packages venv), `smoke2.log`,
  `smoke5.log`, `smoke_rerun4.log` (reruns after fixes), `smoke_ours_skip.log`
  (every ours/ours-cpu arm against the 0.8.22 wheel reads skipped).
- Named refusals left in the smoke (opponent-side facts, not board bugs):
  torch 2.4 on the pod has no `torch.optim.Adafactor` and its inductor cannot
  compile `adaptive_max_pool2d`; the PyPI implicit wheel has no CUDA
  (`implicit-gpu` refuses); cuML PowerTransformer fails its bracket on
  2,000-row Istella; scikit-learn BayesianGaussianMixture and cuVS CAGRA refuse
  2,000-row Istella (collapsed / duplicate rows); cuML AutoARIMA exceeded the
  smoke's 900 s round cap.

## Next

- As lanes add classes or options (option parity), align `LANES` params and
  the ours adapters; re-read CURRENT DIRECTIVES in the plan after each merge.

## Adding a class name or contract

A lane whose class is exported under another name: add it to the lane's
`ours` candidates in `LANES` (`tools/bench_board_algos.py`). The class
contract the board calls is in that file's docstring.

## Not in R2 (reported, not fetched)

An image set (CIFAR/ImageNet) and an implicit-feedback set
(MovieLens/Last.fm). CNN layers run on seeded tensors; ALS on taxi-zones and
text counts.
