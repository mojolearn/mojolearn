# The benchmark board

`tools/bench_board.py` is one script, the same file on every box. It times
mojolearn from the installed PyPI wheel against the opponent libraries on the
same box, in the same run, with the same settings, interleaved round by round,
and records quality next to every time. See the script's docstring for the
full contract. In short:

| box | modes | opponents |
|---|---|---|
| Apple (Metal) | `fast` and `identical`, interleaved in one race (trees, classical, classical2); `identical` only (neural) | CatBoost, XGBoost, LightGBM, scikit-learn, umap-learn, statsmodels and faiss-cpu on the CPU; torch on MPS (classical, neural) |
| NVIDIA | `identical` | CatBoost, XGBoost and LightGBM GPU arms, cuML, cuVS, torch CUDA (the rosters of `tools/bench_all_ours.sh`); scikit-learn and statsmodels on the CPU where cuML has no such estimator |
| AMD | `identical` | XGBoost ROCm where the image has it, otherwise the CPU learners on all cores (scikit-learn, umap-learn, statsmodels, faiss-cpu included); torch ROCm |

- Families: trees (`gbdt-symmetric`, `gbdt-depthwise`, `gbdt-lossguide`, `rf`,
  `et`, `iforest`) and classical (`kmeans`, `pca`, `ols`, `knn`, `kde`, `svc`,
  `dbscan`, `hdbscan`) on taxi and Istella-S, classical2 (23 lanes, below) on
  taxi and Istella-S or seeded synthetic series, and neural
  (`lm-train-step`, `lm-forward`, `gemm`) on inputs the driver builds from
  seed 7.
- Neural is `identical` only on every vendor, because the wheel builds its
  neural surface in that tier only. `--modes fast` with the neural family is
  refused by name; a FAST-only Apple run passes
  `--families trees,classical,classical2`.
- One seed (7). Five timed rounds after one warm-up (`--rounds`).
- Output: one directory with `board.json` (box fingerprint and every cell) and
  `BOARD.md`. Bulky state (venv, wheel download, classical blocks) goes in
  `--cache`, which defaults to `<out>/cache`. Rerunning the same command
  resumes: finished races are skipped and failed ones are retried
  (`--skip-failed` turns that off). A resume on a different box or a
  different wheel is refused.
- Data is never downloaded. taxi and Istella-S come from R2
  (`docs/REMOTE_DATA_R2.md`). Without them the script refuses and prints the
  staging command. A neural-only run (`--families neural`) needs no dataset.
- Always check the plan first: `python3 tools/bench_board.py --dry-run
  --vendor apple` (or `nvidia`, `amd`). The current plan has 75 races on
  every vendor (44 of them classical2, 3 neural). That comes to 242 cells on
  Apple, 172 on NVIDIA and 176 on AMD (classical2 alone: 134, 94 and 90).
  Inference adds 122 cells on Apple, 84 on NVIDIA and 100 on AMD (below);
  `--no-infer` times training only.

## Inference

Training is not the only clock. After a race's fit rounds, every arm predicts
with its own last fitted model from those rounds (no fit is retimed), on the
same rows and in the same output kind, one warm-up and then the timed rounds,
arms interleaved. These are separate cells (`infer_cells` in `board.json`)
with their own ratios, under each race's fit table and in "Inference at a
glance". Our FAST and IDENTICAL predictions on the same rows are compared bit
for bit.

Trees (`forest_speed_arm.py --infer`; the flag is off by default and the
driver's output is unchanged without it) time two batches: `test`, the
held-out split the accuracy column scores, and `large`, the first 1,000,000
training rows (capped at the training rows). Every clock is host rows in and
host predictions out. The output is P(class 1) on the binary tasks and the
anomaly score for iforest.

| arm | the timed call | why this path |
|---|---|---|
| ours | `predict_proba(X)[:, 1]`; iforest `score_samples(X)` | the public surface. iforest rebuilds its forest inside every scoring call (DEVIATION 874), so its clock includes a build |
| XGBoost | `Booster.inplace_predict(X)`; on a CUDA booster, `inplace_predict(cupy.asarray(X))` then `cupy.asnumpy` | XGBoost documents in-place prediction as its fastest path; host rows on a CUDA booster fall back to a DMatrix, so the rows go up and the result comes back inside the clock |
| LightGBM | `Booster.predict(X)` | its predict runs on the CPU whatever device trained it |
| CatBoost | `predict_proba(X, task_type=...)`, GPU on the `-gpu` arm | CatBoost's own GPU apply. If a build refuses it, the arm applies on the CPU and says so |
| scikit-learn | `predict_proba(X)` or `score_samples(X)`, `n_jobs=-1` | its only path |
| cuML | RF converted once to FIL outside the clock (a model load), then FIL `predict_proba(X)`; IsolationForest `score_samples(X)` | FIL is cuML's forest inference |

Each arm's call is printed under its table (the driver's FSPEED-INFER-PATH
line). Quality: the FSPEED-ACC metric recomputed from the timed output on the
held-out rows (`<metric>_matches_fit` says it equals the fit-time value), and
on our FAST arm `bits_equal_vs_ours_identical` and
`max_abs_diff_vs_ours_identical`.

Classical (`classical_two_datasets.py race --infer`, off by default there
too): kmeans `predict`, pca `transform`, ols `predict` and svc `predict` on
the eval rows (the 500,000 test rows for kmeans, pca and ols; 10,000 for
svc). The clock span is the fit's: ours takes host rows and returns host
results; the torch and cuML arms upload the rows before their clock, which
ends at the device synchronize (`SPAN-ASYMMETRIC`). Quality: kmeans
`eval_inertia` and `label_agreement_own_centers`, pca
`transform_max_rel_err_own_fp64`, ols `r2_eval`, `rmse_eval` and
`predict_max_rel_err_own_fp64`, svc `accuracy_eval`, each against a float64
NumPy evaluation of the arm's own fitted model, plus `bits_equal_vs_ours` on
every arm (on `ours-fast` it is the FAST against IDENTICAL check). kNN
(`kneighbors`) and KDE (`score_samples`) already time inference as their race;
DBSCAN and HDBSCAN have no predict.

Not covered yet: categorical (criteo) frames in the trees inference phase; a
single-row latency batch; ONNX, Treelite and other export paths; the
classical2 family's predict calls as separate inference cells (its lanes
define their own clocks in `tools/bench_board_more.py`); svc
`decision_function`.

## The classical2 family

`tools/bench_board_more.py` races the wheel's remaining classical estimators.
It uses the classical racer's worker protocol, interleaving and JSON shape,
and its block helpers (stride samples, the Istella sentinel clean, the fit
rows' standardization). Every estimator's binding ships a FAST tier in 0.8.22
(`mojolearn._backend._CLASSICAL_FAST`), so on Apple every lane races `ours`
(IDENTICAL) and `ours-fast` (FAST) side by side. The tier is read back from
the binary. `_mojolearn_solver` (Lasso, ElasticNet) and `_mojolearn_tsa`
(ExponentialSmoothing) carry no numeric-mode constant in 0.8.22, so for
those the tier is read from the directory the loaded binary sits in, and the
cell says so.

| lane | ours | Apple and AMD opponents | NVIDIA opponents | quality |
|---|---|---|---|---|
| `umap` | `UMAP` | umap-learn seeded (one thread, its rule) and unseeded (every core) | cuML UMAP (exact kNN) | trustworthiness k=15 |
| `spectral-embedding` | `SpectralEmbedding` | scikit-learn | cuML, scikit-learn | trustworthiness k=15 |
| `gmm` | `GaussianMixture` | scikit-learn | scikit-learn (cuML has none) | held-out mean log-likelihood, BIC |
| `logreg`, `linearsvc` | `LogisticRegression`, `LinearSVC` | scikit-learn | cuML | held-out accuracy (and log loss) |
| `ridge`, `lasso`, `elasticnet`, `linearsvr` | the same names | scikit-learn | cuML | held-out R2, RMSE |
| `tsvd` | `TruncatedSVD` | scikit-learn (arpack) | cuML | explained-variance ratio sum, reconstruction error |
| `knn-clf`, `knn-reg` | `KNeighborsClassifier`, `KNeighborsRegressor` | scikit-learn | cuML | held-out accuracy, R2 |
| `spectral`, `agglomerative` | `SpectralClustering`, `AgglomerativeClustering` | scikit-learn | cuML (and scikit-learn for spectral) | silhouette, ARI vs ours, cluster count |
| `gpr`, `gpc` | `GaussianProcessRegressor`, `GaussianProcessClassifier` | scikit-learn | scikit-learn (cuML has none) | held-out RMSE, R2, log predictive density; accuracy, log loss |
| `svr`, `kernel-ridge` | `SVR`, `KernelRidge` | scikit-learn | cuML | held-out R2, RMSE |
| `nystroem`, `rbf-sampler` | `Nystroem`, `RBFSampler` | scikit-learn | scikit-learn (cuML has none) | kernel approximation error |
| `arima` | `ARIMA` (one batched fit) | statsmodels (one fit per series, joblib over every core) | cuML, statsmodels | mean llf, mean AIC, forecast and in-sample RMSE |
| `ets` | `ExponentialSmoothing` | statsmodels | cuML, statsmodels | forecast and in-sample RMSE |
| `ivf` | `IVFIndex` | faiss-cpu IVF-Flat | cuVS `ivf_flat` | recall@10 vs float64 brute force |

The rows, every matched parameter, what is inside the clock, and each
mismatch that could not be avoided (with its one-line reason) are in
`LANE_CONFIG` in the driver. The board copies them into every cell's
`settings.lane_config` and prints them under each race. The main ones:

- Sizes are the classical racer's: 1,000,000 fit and 100,000 held-out stride
  rows for the linear lanes and TruncatedSVD. The O(n^2) and O(n^3) lanes take
  a documented stride subset, the same rows for every arm: UMAP and
  SpectralEmbedding 20,000 rows, kNN 200,000 fit rows and 4,000 queries,
  spectral and agglomerative 10,000, GaussianMixture 100,000 and 20,000, GP
  3,000 and 3,000, SVR and KernelRidge 10,000 and 10,000. The IVF lane reads
  the knn lane's block (400,000 index rows, 4,000 queries).
- The time series are synthetic, as in the repo's own ARIMA quality work: 64
  ARMA(1,1) series of 2,100 points and 64 hourly series with a period-24
  season of 1,488 points, all from `default_rng(7)`. The last 100 and 48
  points of each are held out for the forecast error.
- The GP regressor uses `alpha=2**-20`, the one ridge IDENTICAL accepts besides
  0, and carries its noise in a `WhiteKernel(1e-2)` on every arm. Without it
  the float32 factor of the kernel matrix does not exist on taxi's
  near-duplicate rows, and ours refuses to predict from that fit.
- umap-learn with `random_state=7` runs one thread (its own rule), so the
  board also races it unseeded on every core, and says so.
- cuML's Holt-Winters has only its heuristic initialization. Ours runs
  `initialization_method='estimated'`, its default and statsmodels'.

Opponent pins (`MORE_PINS` in `tools/bench_board.py`): umap-learn 0.5.12,
pynndescent 0.6.0, numba 0.67.0, statsmodels 0.15.0 and faiss-cpu 1.15.1 on
Apple and AMD, and statsmodels 0.15.0 on NVIDIA. cuML and cuVS come from the
rapids set in `tools/opponent_wheels.sh`. They are installed only when
classical2 is planned. The dry run and the board list each opponent that is
not planned on a vendor, with the reason (for example faiss-gpu, and cuML's
missing GaussianMixture, GP, Nystroem and RBFSampler).

## The neural family

`tools/bench_board_neural.py` times the wheel's public Python API against torch
on the same GPU (MPS on Apple, CUDA on NVIDIA, ROCm on AMD). The torch arm,
`torch-eager-fp32`, is torch eager in float32 with TF32 off. That is the
`eager_fp32` column of `tools/torch_lm_step_opponent.py`, which the repo
treats as the opponent's fast setting at our precision. Its compile, TF32 and
bf16 columns are not raced here.

| lane | ours | torch | clock (both sides) | quality |
|---|---|---|---|---|
| `lm-train-step` | `LanguageModelTrainer(resident=True, step_result='lean').train_step(ids)` | the twin model of `tools/torch_lm_step_opponent.py`; forward, mean cross entropy, backward, `torch.optim.AdamW`, `loss.item()` | ids to the device, one full step, the loss back on the host | `loss_first_step`, `loss_last_step`, and `loss_last_abs_diff_vs_ours` on torch |
| `lm-forward` | `LanguageModelTrainer(resident=True).logits(ids)` | the same twin under `no_grad`, logits to the host | ids to the device, the forward, float32 logits back on the host | `mean_nll`, and `max_abs_diff_vs_ours` (logits) on torch |
| `gemm` | `mojolearn.linalg.matmul(a, b)` | `a.to(dev) @ b.to(dev)`, `.cpu()` | both operands to the device, the product, C back on the host | `max_rel_err_vs_fp64`, and `max_abs_diff_vs_ours` on torch |

Both LM arms start from the same parameters (`default_rng(7).normal(0, .02)`,
+1 on every norm) and read the same batches, so their losses and logits are
comparable. The byte stream is the installed mojolearn package's own `.py`
sources, sorted and concatenated, and its sha256 is recorded. Training
continues across rounds, so round r is step r + 1 on both sides. AdamW uses
the trainer defaults (lr 1e-3, betas 0.9 and 0.999, eps 1e-8, weight decay
0.01) on both sides.

`--neural-shape full` (the default) is the 20,453,376-parameter control shape
of `tools/torch_lm_step_opponent.py` (batch 1, length 2048, d_model 384, 6
heads, head_dim 64, intermediate 1024, 8 layers, vocab 8192) and a 4096 cube
GEMM. It was chosen to fit a 16 GB Apple M4 beside torch (the parameters
and both Adam moments are about 250 MB per side); it has not yet been run at
that size on the Mac. The GPT-3-small target shape (162 M parameters, vocab
50257) is left off the board. `--neural-shape small` is a plumbing smoke (batch 2, length 64,
d_model 64, 2 layers, vocab 256; a 256 cube GEMM), and the board says SMOKE.
`--rows` does not apply to neural lanes.

Not covered yet: the Mamba blocks, `TransformerBlock` on its own, `SambaStack`
and `SmallMLPTrainer`. In the classical2 family: `RadiusNeighbors`, the
preprocessing scalers, `Cholesky`, and the `parallel_*` and `Distributed*`
wrappers. Taxi-derived time series are not used.

## Remote Mac (Apple Metal)

Do not run this on the orchestrating laptop. The work is heavy, and that
machine is shared. The steps below are for a separate benchmark Mac,
`bench@bench-mac`, whose home directory is `/Users/bench`.

On the Mac that holds `~/.mojolearn_r2`, stage the data and ship the tree:

```sh
MOJOLEARN_STAGE_BOX_HOME=/Users/bench sh tools/dataset_store.sh stage "bench@bench-mac" \
    gbm-bench/taxi/taxi_speed.npz gbm-bench/istella/istella_speed.npz
ssh bench@bench-mac 'rm -rf ~/mojolearn-board && mkdir -p ~/mojolearn-board'
git archive --format=tar HEAD | ssh bench@bench-mac 'tar -x -C ~/mojolearn-board'
git rev-parse HEAD | ssh bench@bench-mac 'cat > ~/mojolearn-board/SHIPPED_COMMIT.txt'
```

On the benchmark Mac, inside `tmux`, run the plan and then the board.
`--base-python` must be an interpreter the wheel supports.

```sh
cd ~/mojolearn-board
python3 tools/bench_board.py --dry-run --mojolearn-version 0.8.22
python3 tools/bench_board.py --mojolearn-version 0.8.22 \
    --base-python /opt/homebrew/bin/python3.12 \
    --out ~/board-runs/apple-0.8.22 --cache ~/board-cache \
    2>&1 | tee -a ~/board-runs/apple-0.8.22.log
```

If the run stops, rerun the same command. Back on the orchestrator, fetch the
result:

```sh
rsync -a bench@bench-mac:board-runs/apple-0.8.22/ "$HOME/mojolearn-evidence/bench-board/apple-0.8.22/"
```

## NVIDIA leg (DigitalOcean H100)

The body is `tools/bench_board_leg.sh`. The runner stages taxi and Istella-S
from R2 into `/root/datasets/gbm-bench` before the body starts. A full board
takes many hours, so it uses a segment lease rather than the one-hour cap.
The RunPod runner `tools/gemm_remote_leg.sh` is capped at 60 minutes and
passes no environment to the body, so it can only run a small `--lanes`
subset there.

```sh
MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token \
MOJOLEARN_GEMM_LEG_EXTRA=tools/bench_board_leg.sh \
MOJOLEARN_GEMM_LEG_OUT=$HOME/mojolearn-evidence/bench-board/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100 \
MOJOLEARN_DO_EXTRA_ENV='MOJOLEARN_BOARD_VERSION=0.8.22' \
bash tools/do_extra_leg.sh nv --segment-lease 720 --dollar-cap 60 --skip-gates
```

The result arrives at `<leg out>/remote/bench-board/`. NVIDIA uses the image's
CUDA torch through `--system-site-packages`. cuML is installed from the pinned
`rapids-*` set in `tools/opponent_wheels.sh`.

## AMD leg (Hot Aisle MI300X)

The `13core` spec is the one whose CPU opponent rows are comparable. The body
builds a Python 3.12 venv, because the pinned torch ROCm wheels are cp312.

```sh
MOJOLEARN_HOTAISLE_SPEC=13core MOJOLEARN_HOTAISLE_LANE=bench-board \
MOJOLEARN_GEMM_LEG_EXTRA=tools/bench_board_leg.sh \
MOJOLEARN_GEMM_LEG_OUT=$HOME/mojolearn-evidence/bench-board/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi300x \
MOJOLEARN_HOTAISLE_EXTRA_ENV='MOJOLEARN_BOARD_VERSION=0.8.22' \
bash tools/hotaisle_leg.sh amd --rent --segment-lease 900 --dollar-cap 60 --skip-gates
```

The DigitalOcean MI325X works the same way:
`bash tools/do_extra_leg.sh amd --segment-lease 900 --dollar-cap 60 --skip-gates`,
with the environment passed in `MOJOLEARN_DO_EXTRA_ENV`.

## Knobs the leg body reads

Values contain no spaces, and lists are separated by commas.

| variable | effect |
|---|---|
| `MOJOLEARN_BOARD_VERSION` | the mojolearn version to install (required) |
| `MOJOLEARN_BOARD_ROWS` | row cap for a smoke run; the board is then marked SMOKE |
| `MOJOLEARN_BOARD_LANES` / `_FAMILIES` / `_DATASETS` / `_ROUNDS` | narrow the plan |
| `MOJOLEARN_BOARD_NEURAL_SHAPE` | `full` (default) or `small` for a neural smoke |
| `MOJOLEARN_BOARD_NO_INFER` | `1` times training only (no inference cells) |
| `MOJOLEARN_BOARD_OUT`, `MOJOLEARN_BOARD_CACHE` | result directory (fetched) and cache (not fetched; the classical and classical2 blocks live here) |

A smoke leg, for example:
`MOJOLEARN_DO_EXTRA_ENV='MOJOLEARN_BOARD_VERSION=0.8.22 MOJOLEARN_BOARD_ROWS=20000 MOJOLEARN_BOARD_ROUNDS=1 MOJOLEARN_BOARD_LANES=rf,kmeans'`.

## Reading the board

`BOARD.md` gives times, ratios and quality, and never states a direction. A
ratio column is our median divided by the opponent's median. Our FAST and
IDENTICAL arms are never divided by each other, because that ratio is the
cost of identity and not a result (ENGINEERING_RULES 0b-iii). The "Quality at
a glance" table puts our FAST value, our IDENTICAL value and each opponent's
value side by side for every lane and dataset; "Inference at a glance" does
the same for the inference medians per batch, with whether our FAST and
IDENTICAL predictions agree bit for bit. For comparability, trees carry
`FSPEED-FIT-VERDICT` and classical and neural lanes carry the clock span
(`SPAN-ASYMMETRIC` names an opponent whose clock excludes an upload or a fit
that ours includes). A missing arm shows as `UNKNOWN` or `REFUSED(reason)`,
never as a blank.
