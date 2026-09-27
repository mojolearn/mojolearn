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
- Session 3, merge 6: the `prophet` and `moe` races call the classes as
  exported: `ProphetForecaster(...).fit(ds, Y).predict(future ds)` (ds hourly
  from 2024-01-01, max_iter=10000 = prophet's Stan iter) vs prophet-cpu;
  `MoEBlock(hidden_size, intermediate_size, num_experts, top_k,
  norm_topk_prob)` loading torch's weights in HF's fused layout, forward only
  on every arm, vs the six torch arms. Dry run of the 93 existing races
  byte-identical on apple, nvidia, amd (`drydiff_s3.log`). Lane check on the
  pod: sequence-prophet, sequence-moe AGREE (`lanecheck_s3.log`). Plumbing
  smoke (`smoke_s3.log`; OURS=/root/ourspy for prophet, OURS=/root/ourstorchpy
  and THEIRS=/usr/bin/python3 for moe, since /root/opp has no torch): every
  arm of both races ran, ours-cpu bits equal ours. On the merged tree: bench
  tests 62 pass; test_host_surface 196 pass after fixing main's
  byte-lm-host-train revision (72a64f8b9 named no size; now in
  NON_SIZE_REVISIONS of tools/identity_break.py). test_lane_select skipped
  (bench-only diff plus identity_break.py, not a trigger path).
- Session 4, option parity (item 3), merge 7: read every lane's option-parity
  merge on main (prep b2de7bf0e, c73e49116, ac4abdac1, cfc7646d5; cnn
  0e2798963; cluster e670b8db2; decomp ec539d941, 18d267abe; trees
  00d0138d5; linear 651e359c8). None of their new options is needed by a race:
  every race already runs the opponents' default for them. A constructor
  default census on the pod (`default_diff.py`, `default_diff2.py`: ours vs
  scikit-learn 1.7.2, torch.nn and cuML on every parameter a race does not
  set) found three races NOT at matched settings, now set in `LANES`:
  kbins `quantile_method='linear'` on ours and scikit-learn (the pinned
  1.7.2's default and cuML's np.percentile edges; ours defaults to
  'averaged_inverted_cdf'); tsne `init='random'` on every arm (ours refused
  'pca', so its arm could not have run; cuML has only random); sgd-reg cuML
  `power_t=0.25` (scikit-learn's and ours; cuML defaults to 0.5). Other census
  diffs are the same semantics under another spelling (MDS metric, AdaBoost
  algorithm, Calibrated ensemble, Lars eps) or output dtype (encoders, ours
  float32). Still unmatched, by missing options (NOT_IMPLEMENTED rows):
  categorical-nb min_categories (ours_drop), damped-ets seasonal, tsne
  Barnes-Hut (refused under IDENTICAL). Dry run: 93 existing races
  byte-identical, and the whole plan byte-identical, on apple, nvidia, amd
  (`drydiff_s4.log`). Bench tests 62 pass, test_host_surface 196 pass
  (`premerge_s4.log`). test_lane_select skipped (bench-only diff).
- Every algos class is now in the source tree; no race is guessed.
- damped-ets stays non-seasonal on every arm: main's ETS still refuses
  seasonal components (`_x_sequence_ets.py`). When sequence adds them, restore
  `season_length=24` with a seasonal model on ours, statsforecast AutoETS and
  statsmodels ExponentialSmoothing(seasonal="add", seasonal_periods=24).
- Pod note: run `tools/dev_pod.sh sync bench ~/mojolearn-wt/algos-bench` with
  `MOJOLEARN_DEVPOD_ALLOW_SELF=1` when running the tool from this worktree.

## Next

- Option parity again as more lanes merge options: re-run
  `~/mojolearn-evidence/algos-bench/default_diff.py` / `default_diff2.py` on
  the pod (`/root/opp/bin/python` for sklearn, `/usr/bin/python3` for torch,
  `/root/rapids/bin/python` for cuML, cwd /root/mojolearn) and set in `LANES`
  only what makes a race's settings match. When prep adds CategoricalNB
  min_categories, drop the `ours_drop`; when sequence adds seasonal ETS,
  restore the seasonal damped-ets race (below).

## Adding a class name or contract

A lane whose class is exported under another name: add it to the lane's
`ours` candidates in `LANES` (`tools/bench_board_algos.py`). The class
contract the board calls is in that file's docstring.

## Not in R2 (reported, not fetched)

An image set (CIFAR/ImageNet) and an implicit-feedback set
(MovieLens/Last.fm). CNN layers run on seeded tensors; ALS on taxi-zones and
text counts.
