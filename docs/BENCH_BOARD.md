# The benchmark board

`tools/bench_board.py` is one script, the same file on every box. It times
mojolearn from the installed PyPI wheel against the opponent libraries on the
same box, in the same run, with the same settings, interleaved round by round,
and records quality next to every time. See the script's docstring for the
full contract. In short:

| box | modes | opponents |
|---|---|---|
| Apple (Metal) | `fast` and `identical`, interleaved in one race (trees, classical); `identical` only (neural) | CatBoost, XGBoost, LightGBM, scikit-learn on the CPU; torch on MPS (classical, neural) |
| NVIDIA | `identical` | CatBoost, XGBoost and LightGBM GPU arms, cuML, torch CUDA (the rosters of `tools/bench_all_ours.sh`) |
| AMD | `identical` | XGBoost ROCm where the image has it, otherwise the CPU learners on all cores; torch ROCm |

- Families: trees (`gbdt-symmetric`, `gbdt-depthwise`, `gbdt-lossguide`, `rf`,
  `et`, `iforest`) and classical (`kmeans`, `pca`, `ols`, `knn`, `kde`, `svc`,
  `dbscan`, `hdbscan`) on taxi and Istella-S, and neural (`lm-train-step`,
  `lm-forward`, `gemm`) on inputs the driver builds from seed 7.
- Neural is `identical` only on every vendor, because the wheel builds its
  neural surface in that tier only. `--modes fast` with the neural family is
  refused by name; a FAST-only Apple run passes `--families trees,classical`.
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
  --vendor apple` (or `nvidia`, `amd`). The current plan has 31 races on
  every vendor (3 of them neural, 2 cells each). That comes to 106 cells on
  Apple, 78 on NVIDIA and 84 on AMD.

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
and `SmallMLPTrainer`.

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
| `MOJOLEARN_BOARD_OUT`, `MOJOLEARN_BOARD_CACHE` | result directory (fetched) and cache (not fetched) |

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
