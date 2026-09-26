# The benchmark board

`tools/bench_board.py` is one script, the same file on every box. It times
mojolearn from the installed PyPI wheel against the opponent libraries on the
same box, in the same run, with the same settings, interleaved round by round,
and records quality next to every time. See the script's docstring for the
full contract. In short:

| box | modes | opponents |
|---|---|---|
| Apple (Metal) | `fast` and `identical`, interleaved in one race (trees, classical, classical2); `identical` only (neural) | CatBoost, XGBoost, LightGBM, scikit-learn, umap-learn, statsmodels and faiss-cpu on the CPU; torch on MPS (classical, neural) and on the CPU (neural `*-infer`) |
| NVIDIA | `identical` | CatBoost, XGBoost and LightGBM GPU arms, cuML, cuVS, torch CUDA (the rosters of `tools/bench_all_ours.sh`); scikit-learn and statsmodels on the CPU where cuML has no such estimator |
| AMD | `identical` | XGBoost ROCm where the image has it, otherwise the CPU learners on all cores (scikit-learn, umap-learn, statsmodels, faiss-cpu included); torch ROCm |

- Families: trees (`gbdt-symmetric`, `gbdt-depthwise`, `gbdt-lossguide`, `rf`,
  `et`, `iforest`) and classical (`kmeans`, `pca`, `ols`, `knn`, `kde`, `svc`,
  `dbscan`, `hdbscan`) on taxi and Istella-S, classical2 (23 lanes, below) on
  taxi and Istella-S or seeded synthetic series, and neural (16 lanes,
  below) on inputs the driver builds from seed 7.
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
  --vendor apple` (or `nvidia`, `amd`). The current plan has 88 races on
  every vendor (44 of them classical2, 16 neural). That comes to 312 cells on
  Apple, 261 on NVIDIA and 246 on AMD (classical2 alone: 134, 94 and 90;
  neural alone: 76, 95 and 76).

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

`tools/bench_board_neural.py` times the wheel's public neural Python API
against torch on the same box. Our arm is IDENTICAL (the only tier the neural
surface builds). Following the rule that our IDENTICAL is compared with the
opponent's fastest supported setting (`bench/OPPONENT_REFERENCE.md`), every
lane races one torch arm per fast setting torch supports there. These are the
columns of `tools/torch_lm_step_opponent.py`, with the setting in the arm name:

| arm | torch setting | planned on |
|---|---|---|
| `torch-eager-fp32` | eager, float32, TF32 off | every vendor |
| `torch-compile-fp32` | `torch.compile` (inductor, default mode), TF32 off | every vendor |
| `torch-eager-tf32` | eager, float32 matmuls in TF32 | NVIDIA only |
| `torch-compile-tf32` | compile with TF32 on | NVIDIA only |
| `torch-eager-bf16` | bf16 autocast mixed precision (parameters, gradients and AdamW state stay float32) | every vendor, probed on the device |
| `torch-compile-bf16` | compile inside the same autocast | every vendor, probed on the device |

TF32 is an NVIDIA CUDA tensor-core matmul mode. On ROCm and MPS torch accepts
the flag and changes nothing, so those arms are not planned there, and the
board says so in "Not covered". An arm that torch cannot run on the box is
refused by name in its cell and never falls back to another setting or
device. Examples are bf16 on a GPU without it, or `torch.compile` failing on
MPS. The TF32 and bf16 arms run at another precision than ours, and their
quality columns show how far each one lands from our output.

On Apple (torch 2.13, M4, small smoke), `torch.compile` and bf16 autocast both
work on MPS for the LM, GEMM, transformer, Mamba-2 and MLP lanes. Inductor's
Metal code generator fails on the Mamba-3 reference, and so on Samba, which
contains it. Those four compile cells are REFUSED by name. On the CPU,
compile and bf16 work for every lane.

| lane | ours (public API) | runs on | torch twin |
|---|---|---|---|
| `lm-train-step` | `LanguageModelTrainer(resident=True, step_result='lean').train_step(ids)` | GPU | `tools/torch_lm_step_opponent.py` `build_model` (SDPA), `torch.optim.AdamW` |
| `lm-forward` | `LanguageModelTrainer(resident=True).logits(ids)` | GPU | the same twin, logits |
| `gemm` | `mojolearn.linalg.matmul(a, b)` | GPU | `a @ b` |
| `transformer-forward` | `TransformerBlock(weights, n_heads, n_kv_heads, head_dim).forward(x)` | GPU | `tools/speed_torch_seq.py` `LlamaEager.block(sdpa=True)` |
| `transformer-infer` | `TransformerBlockInference(...).forward(x)` | CPU | the same, on the CPU |
| `mamba1-forward` / `mamba1-infer` | `Mamba1Block` / `Mamba1BlockInference(weights).forward(x)` | GPU / CPU | `mamba/corpus/gen_corpus.py` `block_forward` (mamba_ssm's `selective_scan_ref`, a per-token loop; eager arms only) |
| `mamba2-forward` / `mamba2-infer` | `Mamba2Block` / `Mamba2BlockInference(weights).forward(x)` | GPU / CPU | `gen_corpus.py` `m2_forward` (chunked SSD reference) |
| `mamba3-forward` / `mamba3-infer` | `Mamba3Block` / `Mamba3BlockInference(weights).forward(x)` | GPU / CPU | `gen_corpus.py` `m3_forward` (SISO reference) |
| `samba-train-step` | `SambaStack(config, weights).train_step(inputs, targets)` | GPU | embedding, then per layer `m3_forward` or `LlamaEager.block`, then RMSNorm and the tied head; mean CE, `torch.optim.AdamW` |
| `samba-forward` | `SambaStack(config, weights).forward(inputs)` | GPU | the same stack, logits |
| `samba-infer` | `SambaInference(config, weights).forward(inputs)` | CPU | the same stack on the CPU |
| `mlp-train-step` | `SmallMLPTrainer(w1, b1, w2, b2).train_step(X, y)` | GPU | `F.linear`, ReLU, `F.linear`; mean CE, `torch.optim.AdamW` |
| `mlp-infer` | `MLPInference(w1, b1, w2, b2).predict_logits(X)` | CPU | the same MLP on the CPU |

The `*-infer` lanes are the public CPU inference classes, which run on the
host binding. They race `torch-cpu-<setting>` arms (eager and compile, fp32
and bf16). Every twin already existed in the repo, and none was written for
the board. In the Apple smoke, each fp32 torch twin matched our output to
about 1e-7 relative (Samba about 7e-7), and each train step's losses matched
to about 1e-6. The Mamba opponents are pure-PyTorch reference
implementations, not mamba-ssm's fused CUDA or Triton kernels (the board does
not install mamba-ssm). A Mamba ratio is therefore against a reference, and
the board says so. The Mamba-1 reference needs `einops`, which the board
installs (`einops==0.8.1`) when the neural family is planned.

Every arm starts from the same inputs, built by the conductor from seed 7.
The LM parameters are `normal(0, .02)` with +1 on every norm, and the batches
come from the byte stream (the installed mojolearn package's own `.py`
sources, sorted and concatenated, sha256 recorded). The transformer weights
are `normal(0, .02)` with +1 on the norms. The Mamba weights are uniform over
`gen_corpus.py`'s default range for each tensor. Samba follows
`SambaStack`'s initializer rules, and our worker refuses unless
`SambaConfig.registry()` matches the conductor's list. The MLP gets
uniform(+-1/sqrt(fan_in)) weights, standard normal X and labels 0..2. Every
clock is host in, host out, synchronized. Training state stays on the device
between steps on both sides, so round r is step r + 1. AdamW uses lr 1e-3,
betas 0.9 and 0.999, eps 1e-8 and weight decay 0.01 on both sides (Samba
without clipping).

Quality: train lanes report `loss_first_step`, `loss_last_step`, and on each
torch arm `loss_first_abs_diff_vs_ours` and `loss_last_abs_diff_vs_ours`.
Forward lanes report `max_abs_diff_vs_ours` and `max_rel_diff_vs_ours` (max
abs difference over max abs of ours) on each torch arm. The logit lanes add
`mean_nll`, and `gemm` adds `max_rel_err_vs_fp64`.

`--neural-shape full` (the default) has these shapes:
- LM: the 20,453,376-parameter control shape of
  `tools/torch_lm_step_opponent.py` (batch 1, length 2048, d_model 384, 6
  heads, head_dim 64, intermediate 1024, 8 layers, vocab 8192).
- GEMM: a 4096 cube.
- Transformer block: that shape's block.
- Mamba blocks: batch 1, length 2048, d_model 384.
- Samba: batch 2, length 512, d_model 384, vocab 256, layers
  mamba3+attention+mamba3+attention, 6 heads, intermediate 1024.
- MLP: 256 rows.
- The CPU lanes cap the length at 512.

The full shape has not yet been run on the Mac. The GPT-3-small target shape
is left off the board. `--neural-shape small` is a plumbing smoke, and the
board says SMOKE. On an Apple M4 all 16 small lanes (76 cells, one round)
took about 3 minutes. `--rows` does not apply to neural lanes.

Not covered in the neural family:
- the blocks' backward (the VJPs), decode `step`, ragged `lengths` and
  carried state;
- `SmallMLPTrainer.predict_logits`;
- mamba-ssm's fused kernels;
- `torch.compile` on the Mamba-1 reference (a per-token Python loop).

Not covered in the classical2 family: `RadiusNeighbors`, the preprocessing
scalers, `Cholesky`, and the `parallel_*` and `Distributed*` wrappers.
Taxi-derived time series are not used.

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
| `MOJOLEARN_BOARD_OUT`, `MOJOLEARN_BOARD_CACHE` | result directory (fetched) and cache (not fetched; the classical and classical2 blocks live here) |

A smoke leg, for example:
`MOJOLEARN_DO_EXTRA_ENV='MOJOLEARN_BOARD_VERSION=0.8.22 MOJOLEARN_BOARD_ROWS=20000 MOJOLEARN_BOARD_ROUNDS=1 MOJOLEARN_BOARD_LANES=rf,kmeans'`.

## Reading the board

`BOARD.md` gives times, ratios and quality, and never states a direction. A
ratio column is our median divided by the opponent's median. Our FAST and
IDENTICAL arms are never divided by each other, because that ratio is the
cost of identity and not a result (ENGINEERING_RULES 0b-iii). The "Quality at
a glance" table puts our FAST value, our IDENTICAL value and each opponent's
value side by side for every lane and dataset. For comparability, trees carry
`FSPEED-FIT-VERDICT` and classical and neural lanes carry the clock span
(`SPAN-ASYMMETRIC` names an opponent whose clock excludes an upload or a fit
that ours includes). A missing arm shows as `UNKNOWN` or `REFUSED(reason)`,
never as a blank.
