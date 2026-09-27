# bench: progress

Lane `bench` of the algorithm expansion: make `tools/bench_board.py` race every
new algorithm, measure nothing. Worktree `~/mojolearn-wt/algos-bench`, branch
`lane/algos-bench`. Evidence: `~/mojolearn-evidence/algos-bench/`.

## Where it stands

- NEW family `algos` (`tools/bench_board_algos.py`, wired into
  `tools/bench_board.py`, in the default families): 162 algorithm lanes, 303
  races (taxi + Istella-S, or the lane's own data: text, taxi-hourly,
  taxi-zones, synthetic). Before: 93 races on every vendor. After: 396 races
  (Apple 1,748 fit cells, NVIDIA 1,540, AMD 1,379). Merges: 1 (861bd78ea) 375
  races; 2 (b3d708c85) 388, the board aligned to the classes the lanes
  exported; 3: 396, races for MaxPool1d/AvgPool1d/BatchNorm1d/CNNClassifier/
  ClassicalMDS/MiniBatchDictionaryLearning, `mojolearn.refine` in the refine
  race, and the cuVS host-copy fix (`classical_two_datasets._to_host` read a
  pylibraft device_ndarray as garbage: every cuvs-gpu recall, the classical2
  `ivf` lane's included, read ~0; now copy_to_host). 288 of 303 races find
  their class in the source tree at merge 3. The existing 93 races' plan is
  unchanged at every merge (`before_*.txt` vs `after_existing_*.txt`).
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

- `tools/test_lane_select.py` on the pod before merge 3: 59 pass, 1 fails
  (`test_the_wider_mojo_walk_did_not_widen_the_narrow_answers`:
  cluster/host/kmeans_oracle.mojo answers 54 lanes, not 47). Not this lane:
  this branch differs from origin/main only in the three bench files; told
  main.

- Merge 4: graph races on the dense-adjacency PageRank / connected_components
  / Louvain (20,000-node graph), LayerNorm through layer_norm_forward/backward.
  Pre-merge on the pod: bench tests 96 pass; test_host_surface (run with a
  stub `mojolearn` package, no binaries on this pod) 192 pass, 4 fail in
  x_decomp / x_trees manifests and trees-dt-clf pending (not bench files);
  told main.

- Session 2, merge 5: the forecast races call the classes as exported
  (statsforecast's shape: `Cls(**params).fit(Y).predict(h)["mean"]`; AutoARIMA
  keeps cuML's `forecast(h)`). theta -> `Theta(season_length=24,
  decomposition_type="multiplicative")`; croston -> `CrostonClassic`;
  damped-ets -> `ETS(season_length=1, model="AAN", damped=True)` with every
  arm on the non-seasonal damped model (ours refuses seasonal ETS; statsforecast
  AutoETS(model="AAN", damped=True), statsmodels ExponentialSmoothing(
  trend="add", damped_trend=True, seasonal=None)). ivf-filter -> `IVFPQIndex.
  search(filter=)` (IVFIndex has no filter) against faiss IndexIVFPQ +
  IDSelectorBatch; cuVS dropped from that race by name (its Python IVF-PQ
  search takes no filter, only IVF-Flat/CAGRA/brute force do). Plan: 396 races,
  Apple 1,748 fit cells, NVIDIA 1,538, AMD 1,379; the 93 existing races' dry
  run is byte-identical on apple, nvidia and amd (`drydiff.sh`,
  `drydiff_s2m1.log`). Plumbing smoke with the bindings built on the pod
  (A40, `smoke_s2_ours.log`, `smoke_s2_rapids.log`): every ours / ours-cpu /
  statsforecast / statsmodels / faiss arm of the four races ran; ours-cpu
  bits equal ours on all. Lane check on the pod: sequence-theta,
  sequence-croston, sequence-ets, x-ann-filter AGREE.
  GARCH (merged 5baf7a3f3): the race calls arch's shape as exported,
  `GARCH(p=1, q=1, mean="Constant", dist="normal").fit(Y, horizon=h)`,
  `.forecast(h)`, `.loglikelihood_`; smoke `smoke_s2_garch.log` (ours,
  ours-cpu bits equal, arch-cpu all ran; sequence-garch AGREE on the pod).
  On the merged tree (main 3a7d5e185): bench tests 62 pass, test_host_surface
  196 pass; dry run of the 93 existing races byte-identical on apple, nvidia,
  amd (`drydiff_s2m5.log`). test_lane_select skipped per CURRENT DIRECTIVES
  item 0 (bench-only diff); last run: the single kmeans_oracle 54 vs 47
  failure of merge 3, not bench.
- Still guessed (classes not merged): Prophet, the MoE block. When the
  sequence lane merges them: read the class, fix the `prophet` / `moe`
  entries in LANES and their builders (`_build_ts`, the layer builder), build
  with `tools/algos_lane_check.sh <lane>` on the pod and smoke with
  `~/mojolearn-evidence/algos-bench/smoke_s2.py` (OURS=/root/ourspy runs the
  source tree; `drydiff.sh` is the before/after plan diff).

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
