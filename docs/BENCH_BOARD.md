# The benchmark board

`tools/bench_board.py` is one script, the same file on every box. It times
mojolearn from the installed PyPI wheel against the opponent libraries on the
same box, in the same run, with the same settings, interleaved round by round,
and records quality next to every time. See the script's docstring for the
full contract. In short:

| box | modes | opponents |
|---|---|---|
| Apple (Metal) | `fast` and `identical`, interleaved in one race | CatBoost, XGBoost, LightGBM, scikit-learn on the CPU; torch on MPS (classical) |
| NVIDIA | `identical` | CatBoost, XGBoost and LightGBM GPU arms, cuML, torch CUDA (the rosters of `tools/bench_all_ours.sh`) |
| AMD | `identical` | XGBoost ROCm where the image has it, otherwise the CPU learners on all cores; torch ROCm |

- Families: trees (`gbdt-symmetric`, `gbdt-depthwise`, `gbdt-lossguide`, `rf`,
  `et`, `iforest`) and classical (`kmeans`, `pca`, `ols`, `knn`, `kde`, `svc`,
  `dbscan`) on taxi and Istella-S. Neural and GEMM lanes are not covered yet.
  Their drivers time Mojo binaries built from source, not the wheel.
- One seed (7). Five timed rounds after one warm-up (`--rounds`).
- Output: one directory with `board.json` (box fingerprint and every cell) and
  `BOARD.md`. Bulky state (venv, wheel download, classical blocks) goes in
  `--cache`, which defaults to `<out>/cache`. Rerunning the same command
  resumes: finished races are skipped and failed ones are retried
  (`--skip-failed` turns that off). A resume on a different box or a
  different wheel is refused.
- Data is never downloaded. taxi and Istella-S come from R2
  (`docs/REMOTE_DATA_R2.md`). Without them the script refuses and prints the
  staging command.
- Always check the plan first: `python3 tools/bench_board.py --dry-run
  --vendor apple` (or `nvidia`, `amd`). The current plan has 26 races on
  every vendor. That comes to 94 cells on Apple, 68 on NVIDIA and 74 on AMD.

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
python3 tools/bench_board.py --dry-run --mojolearn-version 0.8.18
python3 tools/bench_board.py --mojolearn-version 0.8.18 \
    --base-python /opt/homebrew/bin/python3.12 \
    --out ~/board-runs/apple-0.8.18 --cache ~/board-cache \
    2>&1 | tee -a ~/board-runs/apple-0.8.18.log
```

If the run stops, rerun the same command. Back on the orchestrator, fetch the
result:

```sh
rsync -a bench@bench-mac:board-runs/apple-0.8.18/ "$HOME/mojolearn-evidence/bench-board/apple-0.8.18/"
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
MOJOLEARN_DO_EXTRA_ENV='MOJOLEARN_BOARD_VERSION=0.8.18' \
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
MOJOLEARN_HOTAISLE_EXTRA_ENV='MOJOLEARN_BOARD_VERSION=0.8.18' \
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
| `MOJOLEARN_BOARD_OUT`, `MOJOLEARN_BOARD_CACHE` | result directory (fetched) and cache (not fetched) |

A smoke leg, for example:
`MOJOLEARN_DO_EXTRA_ENV='MOJOLEARN_BOARD_VERSION=0.8.18 MOJOLEARN_BOARD_ROWS=20000 MOJOLEARN_BOARD_ROUNDS=1 MOJOLEARN_BOARD_LANES=rf,kmeans'`.

## Reading the board

`BOARD.md` gives times, ratios and quality, and never states a direction. A
ratio column is our median divided by the opponent's median. Our FAST and
IDENTICAL arms are never divided by each other, because that ratio is the
cost of identity and not a result (ENGINEERING_RULES 0b-iii). The "Quality at
a glance" table puts our FAST value, our IDENTICAL value and each opponent's
value side by side for every lane and dataset. For comparability, trees carry
`FSPEED-FIT-VERDICT` and classical lanes carry the clock span
(`SPAN-ASYMMETRIC` names an opponent whose clock excludes an upload or a fit
that ours includes). A missing arm shows as `UNKNOWN` or `REFUSED(reason)`,
never as a blank.
