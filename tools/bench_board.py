#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE BENCHMARK BOARD: one script, the same file on every box.

    python3 tools/bench_board.py --mojolearn-version 0.8.18 --out ~/board-run
    python3 tools/bench_board.py --dry-run                 # the plan, nothing run
    python3 tools/bench_board.py --out ~/board-run --render-only
    python3 tools/bench_board.py --families neural --neural-shape small ...   # neural smoke

WHAT IT DOES
------------
On ONE box (a Mac with Apple Metal, an NVIDIA Linux box, an AMD Linux box) it
times mojolearn from the INSTALLED PyPI wheel against the opponent libraries in
the SAME run, on the SAME box, with the SAME settings, interleaved round by
round (one warm-up, then `--rounds` timed rounds), and records QUALITY beside
every time. It writes ONE result directory: `board.json` (the box fingerprint
and every cell) and `BOARD.md` rendered from it.

  * Vendor: detected (Metal on macOS; nvidia-smi / rocm-smi on Linux), or
    `--vendor`. Apple runs BOTH numeric modes, `fast` and `identical`,
    interleaved in the same race, for trees, classical AND (since the Apple
    FAST neural pass, 2026-10-03) the neural family's GPU lanes: the wheel
    builds its neural surface FAST on Apple, and the neural FAST arm is
    `ours-fast`. NVIDIA and AMD run `identical` ONLY, so `--modes fast` with
    the neural family is refused by name there. The neural CPU lanes (the
    *-infer lanes and lm-host-train-step) never race in any mode.
  * mojolearn comes from `pip install mojolearn==<V>` into a venv this script
    creates (or `--python-env` an interpreter that already has it). No source
    build. The wheel's sha256 is recorded.
  * Opponents are pinned from tools/opponent_wheels.sh; `--opponent-wheels DIR`
    installs them offline from a set staged out of R2.
  * DEFAULT = OUR GPU ARM(S) ONLY (Andrew, Oct 3 2026). Opponents are scored
    once and stored (the opponent store); a default run races only our arms
    and joins the opponents the store already holds. An opponent missing from
    the store is NOT raced unless `--with-opponents` (or `--retime-opponents`)
    asks; the race records it under `skipped_opponents`.
  * Datasets: taxi and Istella-S, the decoded caches staged from R2
    (docs/REMOTE_DATA_R2.md). This script NEVER downloads data; a missing file
    is a refusal that prints the staging command. `--rows N` shrinks for a
    smoke test and the board then says it is a smoke run.
  * ONE seed (7, the drivers' own), no multi-seed statistics.
  * Resumable: every race (family, lane, dataset, rows) is written to
    board.json as it finishes, and a rerun skips the finished ones.

WHAT IT REUSES (it re-implements no measurement)
------------------------------------------------
  trees      bench/speed/forest_speed_arm.py (tools/speed_gbdt_arm.py pins every
             parameter per opponent; FSPEED lines; FSPEED-FIT-VERDICT). The Apple
             FAST arm is its `--ours-ab numeric_mode='fast'`, interleaved.
  classical  tools/classical_two_datasets.py prep + race (CTD JSON; quality from
             one float64 NumPy function per lane; CTD-SPAN). The Apple FAST arm
             is its `ours-fast` arm; torch runs on MPS there.
  classical2 tools/bench_board_more.py prep + race (the classical racer's worker
             protocol and JSON shape): UMAP, GaussianMixture, the linear models,
             TruncatedSVD, k-NN classifier and regressor, spectral and
             agglomerative clustering, GP regressor and classifier, ARIMA,
             ExponentialSmoothing, IVFIndex, SVR, KernelRidge, Nystroem,
             RBFSampler and SpectralEmbedding against scikit-learn, umap-learn,
             statsmodels, FAISS, cuML and cuVS. The Apple FAST arm is its
             `ours-fast` arm. Every lane's settings and mismatches are its
             LANE_CONFIG.
  neural     tools/bench_board_neural.py race (the classical racer's worker
             protocol and JSON shape): the wheel's public neural surface
             (LanguageModelTrainer train step and logits, linalg.matmul,
             TransformerBlock, Mamba1/2/3Block, SambaStack train step and
             forward, SmallMLPTrainer, and the CPU *Inference classes) against
             torch at every fast setting it supports on the box: eager and
             torch.compile, fp32, TF32 (NVIDIA CUDA only) and bf16 autocast,
             one arm each, the precision in the arm name (bench_board_neural's
             `opponents`). The torch models are the repo's twins
             (tools/torch_lm_step_opponent.py, tools/speed_torch_seq.py
             LlamaEager, mamba/corpus/gen_corpus.py). `--neural-shape full` is
             the board; `small` is a smoke. The Apple FAST arm is its
             `ours-fast` arm (MOJOLEARN_NUMERIC_MODE=fast in that worker, the
             same FAST binaries the wheel ships on Apple; quality columns
             against `ours`, the IDENTICAL arm, like every opponent).
  inference  tools/bench_board_infer.py: the trees driver's `--infer` phase
             (FSPEED-INFER lines, batches `test` and `large`) and the classical
             racer's `race --infer` (kmeans/pca/ols/svc), each arm predicting
             with its own model from the race's fit rounds; `--no-infer` skips.
  parsing    tools/bench_all_summarize.py's FSPEED parser.
  rosters    tools/bench_all_ours.sh's per-lane NVIDIA rosters.

NEURAL, NOT COVERED: bench_board_neural.NOT_COVERED and NOT_PLANNED (the
Mamba opponents are pure-PyTorch references, not mamba-ssm's fused kernels;
the blocks' backward and decode are not raced; TF32 exists on NVIDIA CUDA
only; the GPT-3-small target shape is off the board). tools/speed_gemm_arm.py,
tools/speed_torch_seq.py's timing harness and bench/model/harness.py time
source-built Mojo binaries, not the wheel, and are not used here (the board
reuses speed_torch_seq.py's LlamaEager twin only).

THE BOARD NEVER STATES A DIRECTION. It prints times, ratios and quality
numbers only (CONTRIBUTING.md, "never say we are faster"). A ratio column is
`ours median / that arm's median`: below 1.0 our median time is the lower one.
"""
import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

SCHEMA = "mojolearn-bench-board/1"
#: Fields added to schema /1 without breaking a resume: per cell `peak_host_mb`,
#: `peak_gpu_mb` and `memory`. An old record's `ours-cpu` cells and
#: `ratio_ours_cpu_over` are dropped on load (strip_our_cpu).
SEED = 7                 # the drivers' own seed (lane_config seed=7); one seed only
DEFAULT_ROUNDS = 1
TREE_ROW_FLOOR = 1_000_000

#: gbdt-symmetric-1000 is gbdt-symmetric at 1000 trees (CatBoost's own default
#: iteration count; oblivious trees are weaker per tree), same arms and datasets.
TREE_LANES = ("gbdt-symmetric", "gbdt-symmetric-1000", "gbdt-depthwise", "gbdt-lossguide",
              "rf", "et", "iforest")
CLASSICAL_LANES = ("kmeans", "pca", "ols", "knn", "kde", "svc", "dbscan", "hdbscan")
FAMILIES = ("trees", "classical", "classical2", "neural", "algos")
NEURAL_SHAPES = ("full", "small")
DATASETS = ("taxi", "istella")
VENDORS = ("apple", "nvidia", "amd")
MODES = ("fast", "identical")
VENDOR_API = {"apple": "metal", "nvidia": "cuda", "amd": "hip"}

#: Where the loaders read the decoded caches (under GBM_BENCH_DATA) and the R2
#: key each is staged from (bench/results/dataset_store/manifest.tsv).
DATA_FILES = {"taxi": "taxi/taxi_speed.npz", "istella": "istella/istella_speed.npz"}
R2_KEYS = {"taxi": "gbm-bench/taxi/taxi_speed.npz",
           "istella": "gbm-bench/istella/istella_speed.npz"}

# ---------------------------------------------------------------------------
# Rosters. The opponent set is per vendor AND per lane; it is never a global
# list (bench_all_ours.sh records what a global list did: every rf/et/iforest
# cell silently ran ours-only).
# ---------------------------------------------------------------------------

#: Tree opponents, exact arm names built by tools/speed_gbdt_arm.py.
TREE_OPPONENTS = {
    # Apple: no vendor GPU path exists, so the opponents are their CPU learners
    # on every core (resolve_devices' `auto` is cpu on the Mac).
    "apple": {
        "gbdt-symmetric": ("catboost-cpu",),
        "gbdt-symmetric-1000": ("catboost-cpu",),
        "gbdt-depthwise": ("catboost-cpu", "xgboost-cpu"),
        "gbdt-lossguide": ("catboost-cpu", "xgboost-cpu", "lightgbm-cpu"),
        "rf": ("sklearn-rf-cpu", "lightgbm-cpu"),
        "et": ("sklearn-et-cpu", "lightgbm-cpu"),
        "iforest": ("sklearn-iforest-cpu",),
    },
    # NVIDIA: the vendor GPU path only; bench_all_ours.sh's tree_arms_for.
    "nvidia": {
        "gbdt-symmetric": ("catboost-gpu",),
        "gbdt-symmetric-1000": ("catboost-gpu",),
        "gbdt-depthwise": ("catboost-gpu", "xgboost-gpu"),
        "gbdt-lossguide": ("catboost-gpu", "xgboost-gpu", "lightgbm-cuda"),
        "rf": ("cuml-rf-gpu",),
        "et": ("sklearn-et-cpu",),          # no GPU ExtraTrees exists anywhere
        "iforest": ("cuml-iforest-gpu",),
    },
    # AMD (CONTRIBUTING.md, comparing against libraries without a GPU path):
    # a library with an AMD GPU path runs on the GPU, one without runs on this
    # box's CPU on every core. xgboost-gpu needs a ROCm build of XGBoost; the
    # pinned PyPI wheel is CUDA, so that arm refuses by name unless the image
    # carries one, and xgboost-cpu stands beside it.
    "amd": {
        "gbdt-symmetric": ("catboost-cpu",),
        "gbdt-symmetric-1000": ("catboost-cpu",),
        "gbdt-depthwise": ("catboost-cpu", "xgboost-gpu", "xgboost-cpu"),
        "gbdt-lossguide": ("catboost-cpu", "xgboost-gpu", "xgboost-cpu", "lightgbm-cpu"),
        "rf": ("sklearn-rf-cpu", "lightgbm-cpu"),
        "et": ("sklearn-et-cpu", "lightgbm-cpu"),
        "iforest": ("sklearn-iforest-cpu",),
    },
}


# ---------------------------------------------------------------------------
# THE GBDT TASK LANES (lane bench-board-gbdt-tasks). Ranking, multiclass and
# categorical fits of the public GradientBoosting, raced in the trees family
# through the same driver. Each lane runs only the board datasets it has a
# task for, mapped to the driver's dataset name; the objectives and every
# mismatch are tools/speed_gbdt_arm.py's TASK_LANES (the race's lane_config).
# ---------------------------------------------------------------------------

TREE_TASK_LANES = ("gbdt-rank-yetirank", "gbdt-rank-pairlogit", "gbdt-multiclass",
                   "gbdt-categorical", "gbdt-ordered")

#: board dataset -> the driver's --dataset, per task lane. Istella-S is the
#: learning-to-rank set (query ids from istella_rank.npz); multiclass is
#: taxi's tip-share band (4 classes) and Istella-S's grade (5 classes);
#: categorical is taxi with its id columns declared categorical. criteo, the
#: categorical set the driver also knows, is not in the R2 store, so it is
#: not a board dataset (docs/BENCH_BOARD.md, "The GBDT task lanes").
TREE_TASK_DATASETS = {
    "gbdt-rank-yetirank": {"istella": "istellarank"},
    "gbdt-rank-pairlogit": {"istella": "istellarank"},
    "gbdt-multiclass": {"taxi": "taximc", "istella": "istellamc"},
    "gbdt-categorical": {"taxi": "taxicat"},
    "gbdt-ordered": {"taxi": "taxi", "istella": "istella"},
}

#: data files a driver dataset reads beyond its board dataset's own cache
#: (keys of DATA_FILES).
TREE_TASK_EXTRA_DATA = {"istellarank": ("istella-rank",)}

#: The rosters. The libraries run each lane's closest objective (TASK_LANES);
#: LightGBM has no pairwise logistic objective, so gbdt-rank-pairlogit has no
#: LightGBM arm. Same device policy as the gbdt lanes.
_TASK_APPLE = ("catboost-cpu", "xgboost-cpu", "lightgbm-cpu")
_TASK_NVIDIA = ("catboost-gpu", "xgboost-gpu", "lightgbm-cuda")
_TASK_AMD = ("catboost-cpu", "xgboost-gpu", "xgboost-cpu", "lightgbm-cpu")


def _no_lgbm(arms):
    return tuple(a for a in arms if not a.startswith("lightgbm-"))


for _v, _arms in (("apple", _TASK_APPLE), ("nvidia", _TASK_NVIDIA), ("amd", _TASK_AMD)):
    for _lane in TREE_TASK_LANES:
        TREE_OPPONENTS[_v][_lane] = _no_lgbm(_arms) if _lane == "gbdt-rank-pairlogit" else _arms
    # Ordered boosting: CatBoost only, as gbdt-symmetric
    TREE_OPPONENTS[_v]["gbdt-ordered"] = TREE_OPPONENTS[_v]["gbdt-symmetric"]
DATA_FILES["istella-rank"] = "istella/istella_rank.npz"
R2_KEYS["istella-rank"] = "gbm-bench/istella/istella_rank.npz"


def tree_task_datasets(lane, datasets):
    """The board datasets a trees lane runs: every requested one for the
    original lanes, only those with a task for a task lane."""
    table = TREE_TASK_DATASETS.get(lane)
    return list(datasets) if table is None else [d for d in datasets if d in table]


def tree_driver_dataset(lane, dataset):
    return TREE_TASK_DATASETS.get(lane, {}).get(dataset, dataset)


def race_data_keys(race):
    """DATA_FILES keys a race needs (its dataset and any side file)."""
    keys = [race["dataset"]]
    if race["family"] == "trees":
        keys += list(TREE_TASK_EXTRA_DATA.get(tree_driver_dataset(race["lane"], race["dataset"]), ()))
    return keys


def tree_devices(vendor, lane):
    """forest_speed_arm.py --devices for this vendor and lane."""
    if vendor == "apple":
        return "cpu"
    if vendor == "amd":
        return "cpu,gpu"
    return "cpu,gpu" if lane == "et" else "gpu"


#: Classical opponents, exact arm names built by tools/classical_two_datasets.py.
CLASSICAL_OPPONENTS = {
    "apple": {
        "kmeans": ("sklearn-cpu", "torch-gpu"),
        "pca": ("sklearn-cpu", "torch-gpu"),
        "ols": ("sklearn-cpu", "torch-gpu"),
        "knn": ("sklearn-cpu", "torch-gpu"),
        "kde": ("sklearn-cpu",),
        "svc": ("sklearn-cpu",),
        "dbscan": ("sklearn-cpu",),
        "hdbscan": ("sklearn-cpu",),
    },
    "nvidia": {                              # bench_all_ours.sh's classical_arms_for
        "kmeans": ("cuml-gpu", "torch-gpu"),
        "pca": ("cuml-gpu", "torch-gpu"),
        "ols": ("cuml-gpu", "torch-gpu", "torch-gpu-eigh"),
        "knn": ("cuml-gpu", "torch-gpu"),
        "kde": ("cuml-gpu",),
        "svc": ("cuml-gpu",),
        "dbscan": ("cuml-gpu",),             # cuml-gpu-rbc overflows at 1M rows
        "hdbscan": ("cuml-gpu",),
    },
    "amd": {                                 # tools/classical_hotaisle_leg.sh's arms
        "kmeans": ("sklearn-cpu", "torch-gpu"),
        "pca": ("sklearn-cpu", "torch-gpu"),
        "ols": ("sklearn-cpu", "torch-gpu", "torch-gpu-eigh"),
        "knn": ("sklearn-cpu", "torch-gpu"),
        "kde": ("sklearn-cpu",),
        "svc": ("sklearn-cpu",),
        "dbscan": ("sklearn-cpu",),
        "hdbscan": ("sklearn-cpu",),
    },
}

def family_lanes(fam):
    return {"trees": TREE_LANES + TREE_TASK_LANES, "classical": CLASSICAL_LANES,
            "classical2": MORE_LANES,
            "neural": NEURAL_LANES, "algos": ALGOS_LANES}[fam]


#: classical2 opponent pins, per vendor, installed beside the trees set when
#: the family is planned. umap-learn's numba and pynndescent are pinned with
#: it so the resolver cannot move them. NVIDIA races cuML and cuVS (the
#: rapids set) and statsmodels; an opponent that is not pinned for a vendor
#: is named in bench_board_more.NOT_PLANNED, never dropped silently.
MORE_PINS = {
    "apple": ["umap-learn==0.5.12", "pynndescent==0.6.0", "numba==0.67.0",
              "statsmodels==0.15.0", "faiss-cpu==1.15.1"],
    "amd": ["umap-learn==0.5.12", "pynndescent==0.6.0", "numba==0.67.0",
            "statsmodels==0.15.0", "faiss-cpu==1.15.1"],
    "nvidia": ["statsmodels==0.15.0"],
}


def check_neural_modes(families, modes, vendor="apple"):
    """FAST for the neural family is the Apple tier (the Apple FAST neural
    pass, 2026-10-03: the wheel builds its neural surface FAST on Apple and
    the board races it as `ours-fast`). On NVIDIA and AMD the neural surface
    is raced IDENTICAL only, so a neural plan without identical is refused
    there by name. The neural CPU lanes never race FAST (or anything else)."""
    if "neural" in families and "identical" not in modes and vendor != "apple":
        raise SystemExit(
            "bench_board: REFUSING --modes %s for the neural family on %s: the wheel's neural "
            "surface (LanguageModelTrainer, linalg.matmul, the transformer and Mamba blocks, the "
            "Samba stack, the MLP) is raced IDENTICAL only on NVIDIA and AMD; its FAST tier is "
            "the Apple tier (the `ours-fast` arm on an Apple box). Pass --families "
            "trees,classical for a FAST-only run here, or add identical." % (",".join(modes), vendor))


#: Per-round ceilings for the classical racer. A DBSCAN round on Istella-S is
#: about 337 s on an H100 (bench_all_ours.sh); a CPU opponent on a Mac is
#: longer, so the defaults here are generous. Long runs are expected.
def classical_round_seconds(lane, dataset, vendor):
    if lane in ("dbscan", "hdbscan"):
        return 3600 if dataset == "istella" else 1800
    return 1800 if vendor == "apple" else 900


#: DBSCAN eps and min_samples: the cuML benchmark's (eps=3, min_samples=2 on
#: every dataset, tools/bench_board_harness.py), constants of the racer
#: (classical_two_datasets.DBSCAN_EPS / DBSCAN_MIN_SAMPLES) since 2026-09-29.

#: Classical block each lane reads (classical_two_datasets.BLOCK_OF).
CLASSICAL_BLOCK = {"kmeans": "big", "pca": "big", "ols": "big", "knn": "knn",
                   "kde": "kde", "svc": "svc", "dbscan": "dbscan", "hdbscan": "dbscan"}

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

#: Torch pins. Mac: the version the skgpu pixi env solved (pixi.lock
#: osx-arm64/pytorch-2.13.0). AMD: the ROCm 6.4.1 wheels (cp312, hash-pinned)
#: of tools/classical_two_datasets_leg.sh, so the venv must be Python 3.12
#: there (tools/bench_board_leg.sh arranges it). NVIDIA: the Mac's torch
#: version built for CUDA 12.9 (torch==2.13.0+cu129 from the PyTorch cu129
#: index, which requires cuda-toolkit 12.9.1, inside the RAPIDS 26.8 sets'
#: cuda-toolkit 12.*), in a CLEAN venv. 2026-09-29: the image's torch
#: 2.4.1+cu124 through --system-site-packages broke cuML: the image's
#: dist-packages/nvidia/__init__.py is a regular package, which shadows the
#: venv's `nvidia` namespace, so nvidia.libnvcomp did not import, libcudf.so
#: did not load, and every cuML arm refused (ImportError: libcudf.so).
#: --system-site-packages is refused on NVIDIA for that reason.
AMD_TORCH_ROCM = (
    "https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/pytorch_triton_rocm-3.2.0%2Brocm6.4.1."
    "git6da9e660-cp312-cp312-linux_x86_64.whl"
    "#sha256=1d97c15798bf178299328032141a21d9777e7cdef59d5a7e3ac74e297c17198e "
    "https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/torch-2.6.0%2Brocm6.4.1.git1ded221d-"
    "cp312-cp312-linux_x86_64.whl"
    "#sha256=6b141e1a03148b007c6217519cd9947d760123ded5caebadffec22cba7358d2d")
DEFAULT_TORCH_SPEC = {"apple": "torch==2.13.0", "nvidia": "torch==2.13.0+cu129",
                      "amd": AMD_TORCH_ROCM}
#: an EXTRA index for the default torch spec (PyPI stays the main index)
DEFAULT_TORCH_EXTRA_INDEX = {"nvidia": "https://download.pytorch.org/whl/cu129"}
#: OUR CPU IS NEVER RACED (Andrew, Oct 2 2026, hard rule). `ours-cpu` is the
#: name old records used for our IDENTICAL estimator under MOJOLEARN_VENDOR=cpu;
#: nothing plans, runs, times, ratios or renders it now, and strip_our_cpu drops
#: it (and any other cell of ours on the CPU) from a record on load and render.
#: The host column gives same-bits digests only (lq ID, tools/aft_idcheck.sh).
CPU_ARM = "ours-cpu"


#: The neural family's one extra pin: einops, which mamba/corpus/gen_corpus.py's
#: verbatim selective_scan_ref (the Mamba-1 torch twin) imports (pixi.toml
#: carries einops >= 0.8 for the same reference).
NEURAL_PINS = ["einops==0.8.1"]


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_tool(name):
    """A tools/ module by path (tools/ is not a package)."""
    path = os.path.join(HERE, name + ".py")
    spec = importlib.util.spec_from_file_location("bench_board_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: The classical2 driver's tables (its module imports nothing beyond the
#: standard library at import time, so the orchestrator can read them).
MORE = _load_tool("bench_board_more")
MORE_LANES = MORE.LANE_ORDER

#: The algorithm-expansion driver's tables (standard library only at import):
#: one race per new algorithm and dataset, its opponents per vendor, its R2
#: keys; our side SKIPS by name ("SKIPPED: not built yet") until the lane's
#: class is in the installed wheel.
ALGOS = _load_tool("bench_board_algos")
ALGOS_LANES = ALGOS.LANE_ORDER

#: The neural driver's tables (standard library only at import time): its
#: lanes, the data each reads (no R2 dataset: inputs are built from seed 7)
#: and the torch arms per vendor and lane, one per fast setting torch
#: supports there (eager/compile x fp32/tf32/bf16; TF32 on NVIDIA CUDA only;
#: the CPU *Inference lanes race torch on the CPU).
NEURAL = _load_tool("bench_board_neural")
NEURAL_LANES = NEURAL.LANES
NEURAL_DATA = dict(NEURAL.DATA_OF)
NEURAL_OPPONENTS = {v: {lane: NEURAL.opponents(v, lane) for lane in NEURAL_LANES} for v in VENDORS}

#: INFERENCE cells (tools/bench_board_infer.py): after a race's fit rounds each
#: arm predicts with its own fitted model, timed and raced the same way. Trees
#: and the classical kmeans/pca/ols/svc lanes; on unless --no-infer. The cells
#: live in a race record's `infer_cells`, apart from the fit `cells`.
INFER = _load_tool("bench_board_infer")

#: NVIDIA's harnesses (tools/bench_board_harness.py): which lanes take their
#: values, and from which file and commit (standard library only).
HARNESS = _load_tool("bench_board_harness")

#: The opponent store (tools/bench_board_store.py): an opponent is measured once
#: per key (box, device, library version, settings, data, ...) and reused.
STORE = _load_tool("bench_board_store")
WATCHDOG = _load_tool("bench_board_watchdog")

#: Per-arm memory and the our-CPU refusal (standard library only at import).
PROBE = _load_tool("bench_board_probe")

#: The two clocks (tools/board_clock_audit.py, standard library only): per cell
#: the whole-operation clock (incl. the host-to-device copy) and the kernel-only
#: clock where stored fields give them, and each opponent ratio on its clock
#: (kernel for a torch GPU arm, whole otherwise; AGENTS.md measurement item 6).
#: Derived at render time only; the stored cells and ratios are untouched.
CLOCKS = _load_tool("board_clock_audit")


def _bb():
    """This module's helpers, for bench_board_infer (it imports nothing from here)."""
    return types.SimpleNamespace(**globals())


# ---------------------------------------------------------------------------
# Vendor and modes
# ---------------------------------------------------------------------------

def detect_vendor(system=None, which=shutil.which, exists=os.path.exists):
    """'apple', 'nvidia' or 'amd'. NVIDIA wins when both tools exist (a CUDA
    image with rocm-smi installed is still an NVIDIA box), as in
    speed_gbdt_arm.accel_vendor. None when nothing is recognized."""
    system = system or platform.system()
    if system == "Darwin":
        return "apple"
    if which("nvidia-smi"):
        return "nvidia"
    if which("rocm-smi") or exists("/dev/kfd"):
        return "amd"
    return None


def modes_for(vendor, requested=None):
    """Apple: fast AND identical. NVIDIA and AMD: identical only.
    `requested` may NARROW that set (e.g. identical only on Apple); it may
    never add fast to NVIDIA or AMD, which is refused by name."""
    allowed = ("fast", "identical") if vendor == "apple" else ("identical",)
    if not requested:
        return list(allowed)
    want = [m.strip().lower() for m in requested.split(",") if m.strip()]
    bad = [m for m in want if m not in allowed]
    if bad:
        raise SystemExit(
            "bench_board: REFUSING --modes %s on %s: this box runs %s only. FAST is "
            "the Apple tier on this board; NVIDIA and AMD are measured IDENTICAL."
            % (",".join(want), vendor, ",".join(allowed)))
    # stable order: fast before identical
    return [m for m in MODES if m in want]


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def rows_tag(rows):
    return "full" if not rows else str(int(rows))


def race_id(family, lane, dataset, rows, shape=None):
    if family == "neural":
        return "neural/%s/%s/shape=%s" % (lane, dataset, shape or "full")
    return "%s/%s/%s/rows=%s" % (family, lane, dataset, rows_tag(rows))


def our_arms(family, modes, lane=None):
    """driver arm name -> numeric mode, for our arms in one race. Our GPU
    only: the board never races our CPU (Andrew, Oct 2 2026)."""
    if family == "neural":
        # identical on every vendor; FAST (`ours-fast`) only where modes_for
        # admits fast, which is the Apple box (the Apple FAST neural tier)
        out = {}
        if "identical" in modes:
            out["ours"] = "identical"
        if "fast" in modes:
            out["ours-fast"] = "fast"
        return out
    if family == "classical2" and lane and not MORE.has_fast(lane):
        # an estimator whose binding ships no FAST tier is identical only
        return {"ours": "identical"} if "identical" in modes else {}
    if family == "trees":
        if modes == ["fast", "identical"]:
            return {"ours": "identical", "ours-ab": "fast"}
        return {"ours": modes[0]}
    out = {}
    if "identical" in modes:
        out["ours"] = "identical"
    if "fast" in modes:
        out["ours-fast"] = "fast"
    return out


def _is_cpu_arm(arm):
    return re.search(r"-cpu(?:-|$)", arm) is not None


def gpu_opponents_first(opp):
    """Andrew, Oct 2 2026: when a race has any GPU opponent, only GPU opponents
    race and every CPU opponent is dropped (sklearn included). A race with no
    GPU opponent at all keeps its CPU opponents."""
    gpu = [a for a in opp if not _is_cpu_arm(a)]
    return gpu if gpu else list(opp)


def _ours_runs_on_cpu(family, lane, arm):
    """True when a planned `ours*` arm would time our CPU: the ours-cpu arm, or a
    neural lane whose public class is the host binding."""
    if arm == CPU_ARM or PROBE.is_our_cpu_arm(arm):
        return True
    return arm_library(arm) == "mojolearn" and family == "neural" and \
        NEURAL.DEVICE_OF.get(lane) == "cpu"


def enforce_gpu_only(races):
    """LOCKED (Andrew, Oct 2 2026): the board races only OUR GPU, and our GPU races
    GPU opponents only unless a race has no GPU opponent at all. Any plan that
    breaks this stops the board before anything runs. There is no switch."""
    bad = []
    for r in races:
        for a in r["our_arms"]:
            if _ours_runs_on_cpu(r["family"], r["lane"], a):
                bad.append("%s: our arm %s runs on the CPU" % (r["id"], a))
        opp = r.get("opponents") or []
        if any(_is_cpu_arm(a) for a in opp) and any(not _is_cpu_arm(a) for a in opp):
            bad.append("%s: CPU opponents %s race beside GPU ones" % (
                r["id"], ",".join(a for a in opp if _is_cpu_arm(a))))
    if bad:
        raise SystemExit("bench_board: GPU-only board violated (our CPU never races; CPU "
                         "opponents only where no GPU opponent exists):\n  " + "\n  ".join(bad))
    return races


def plan_races(vendor, modes, families=FAMILIES, lanes=None, datasets=DATASETS, rows=None,
               neural_shape="full"):
    check_neural_modes(families, modes, vendor)
    races = []
    for fam in families:
        for lane in family_lanes(fam):
            if lanes and lane not in lanes:
                continue
            if fam == "neural":
                if NEURAL.DEVICE_OF.get(lane) == "cpu":
                    continue                  # ours would run on the CPU: never raced, no mode
                # one race per lane: its own data, not taxi/Istella; IDENTICAL
                # everywhere, plus the FAST arm (`ours-fast`) on Apple
                ours = our_arms(fam, modes, lane)
                if not ours:
                    continue
                opp = gpu_opponents_first(NEURAL_OPPONENTS[vendor][lane])
                ds = NEURAL_DATA[lane]
                races.append({
                    "id": race_id(fam, lane, ds, None, neural_shape),
                    "family": fam, "lane": lane, "dataset": ds, "rows": None,
                    "shape": neural_shape,
                    "modes": sorted(set(ours.values()), key=MODES.index), "our_arms": ours,
                    "opponents": list(opp), "arms": list(ours) + list(opp),
                })
                continue
            if fam == "algos":
                ours = our_arms(fam, modes, lane)
                if not ours:
                    continue
                opp = gpu_opponents_first(ALGOS.opponents(vendor, lane))
                # taxi/Istella follow --datasets; a lane's own data (text,
                # taxi-hourly, synthetic, ...) runs whatever --datasets says
                for ds in [d for d in ALGOS.datasets_of(lane) if d not in DATASETS or d in datasets]:
                    races.append({
                        "id": race_id(fam, lane, ds, rows),
                        "family": fam, "lane": lane, "dataset": ds, "rows": rows,
                        "modes": sorted(set(ours.values()), key=MODES.index), "our_arms": ours,
                        "ours_class": list(ALGOS.LANES[lane]["ours"]),
                        "opponents": list(opp), "arms": list(ours) + list(opp),
                    })
                continue
            if fam == "classical2":
                ours = our_arms(fam, modes, lane)
                if not ours:
                    continue
                opp = gpu_opponents_first(MORE.OPPONENTS[vendor][lane])
                own = MORE.datasets_of(lane)
                # a lane on its own synthetic data runs once, whatever
                # --datasets says; taxi/Istella lanes follow --datasets
                dss = [d for d in datasets if d in own] if set(own) <= set(DATASETS) else list(own)
                for ds in dss:
                    races.append({
                        "id": race_id(fam, lane, ds, rows),
                        "family": fam, "lane": lane, "dataset": ds, "rows": rows,
                        "modes": sorted(set(ours.values()), key=MODES.index), "our_arms": ours,
                        "opponents": list(opp), "arms": list(ours) + list(opp),
                    })
                continue
            for ds in (tree_task_datasets(lane, datasets) if fam == "trees" else datasets):
                opp = gpu_opponents_first(
                    (TREE_OPPONENTS if fam == "trees" else CLASSICAL_OPPONENTS)[vendor][lane])
                ours = our_arms(fam, modes, lane)
                races.append({
                    "id": race_id(fam, lane, ds, rows),
                    "family": fam, "lane": lane, "dataset": ds, "rows": rows,
                    "modes": list(modes), "our_arms": ours,
                    "opponents": list(opp),
                    "arms": list(ours) + list(opp),
                })
    for race in races:
        unsupported = unsupported_our_arms(race)
        if unsupported:
            race["unsupported_arms"] = unsupported
    return enforce_gpu_only(races)


def unsupported_our_arms(race):
    """Public API capabilities, never benchmark-shape exceptions.

    _linalg_impl._lowbit_binding calls require_identical() for BF16/INT8
    on every shape. No FAST implementation is exposed by these APIs.
    """
    if race.get("family") != "neural" or race.get("lane") not in ("gemm-bf16", "gemm-int8"):
        return {}
    return {arm: "UNSUPPORTED(FAST low-bit API requires IDENTICAL: _lowbit_binding.require_identical)"
            for arm, mode in race.get("our_arms", {}).items() if mode == "fast"}


def plan_summary(races):
    by_fam = {}
    for r in races:
        f = by_fam.setdefault(r["family"], {"races": 0, "cells": 0})
        f["races"] += 1
        f["cells"] += len(r["arms"])
    return {"races": len(races), "cells": sum(len(r["arms"]) for r in races),
            "unsupported_races": sum(bool(r.get("our_arms")) and set(r["our_arms"]) <= set(unsupported_our_arms(r)) for r in races),
            "by_family": by_fam}


# ---------------------------------------------------------------------------
# Library / device labels for an arm name
# ---------------------------------------------------------------------------

def arm_library(arm):
    if arm in ("ours", "ours-ab", "ours-fast", "ours-base", CPU_ARM):
        return "mojolearn"
    head = arm.split("-", 1)[0]
    return {"sklearn": "scikit-learn", "umap": "umap-learn", "hf": "tokenizers"}.get(head, head)


def arm_device(arm, vendor):
    if arm == CPU_ARM:
        return "cpu"
    if arm_library(arm) == "mojolearn":
        return "gpu"
    if arm.endswith("-cpu") or "-cpu-" in arm:
        return "cpu"
    return "gpu"


# ---------------------------------------------------------------------------
# Opponent pins (tools/opponent_wheels.sh is the single source)
# ---------------------------------------------------------------------------

def opponent_pins(path=None):
    """{set name: (extra index or None, [requirement, ...])} parsed from the
    setspec block of tools/opponent_wheels.sh, so the pins cannot drift."""
    path = path or os.path.join(HERE, "opponent_wheels.sh")
    out = {}
    with open(path) as fh:
        for line in fh:
            m = re.match(r'^"([A-Za-z0-9_.-]+)\t(\S+)\t([^"]+)"', line.strip())
            if m:
                idx = None if m.group(2) == "-" else m.group(2)
                out[m.group(1)] = (idx, m.group(3).split())
    return out


def opponent_requirements(vendor, pins):
    """[(extra index or None, [reqs])] to install for this vendor."""
    trees = next((v for k, v in pins.items() if k.startswith("trees-")), None)
    rapids = next((v for k, v in pins.items() if k.startswith("rapids-")), None)
    groups = []
    if trees:
        groups.append(trees)
    if vendor == "nvidia" and rapids:
        groups.append(rapids)
    return groups


# ---------------------------------------------------------------------------
# Small process helpers
# ---------------------------------------------------------------------------

def capture(cmd, timeout=60, env=None):
    """stdout of `cmd`, or None if it cannot run. Never raises."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           check=False, env=env)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    return p.stdout.strip()


#: What the host-memory watchdog killed (tools/bench_board_watchdog.py), in
#: order; run_race reads the entries its race added.
HOST_MEMORY_KILLS = []


def run_logged(cmd, env, log_path, timeout, cwd=REPO, nice=0):
    """Run `cmd` with stdout+stderr to `log_path`, in its own process group,
    killed as a GROUP at `timeout` seconds (macOS has no timeout(1), and a
    driver that forks leaves children behind a plain kill). Returns rc;
    124 means the timeout fired."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if nice:
        cmd = ["nice", "-n", str(nice)] + list(cmd)
    with open(log_path, "a") as log:
        log.write("\n=== %s %s\n" % (now_utc(), " ".join(shlex.quote(c) for c in cmd)))
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                cwd=cwd, start_new_session=True)
        dog = None
        try:
            dog = WATCHDOG.HostMemoryWatchdog(proc.pid, log).start()
        except Exception as exc:      # noqa: BLE001 - recorded; the race still runs
            log.write("\n=== bench_board: host-memory watchdog not started: %r\n" % (exc,))
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            proc.wait()
            log.write("\n=== bench_board: KILLED at the %d s race ceiling\n" % timeout)
            return 124
        finally:
            if dog is not None:
                HOST_MEMORY_KILLS.extend(dog.stop())


def child_env(ctx, extra=None):
    env = dict(os.environ)
    for k in THREAD_ENV:              # a cap inherited from a shell throttles CPU arms silently
        env.pop(k, None)
    env.pop("PYTHONPATH", None)       # nothing may shadow the installed wheel
    env["MOJOLEARN_BENCH_INSTALLED"] = "1"
    env["GBM_BENCH_DATA"] = ctx["data_root"]
    env["MOJOLEARN_REPO_COMMIT"] = ctx.get("commit") or "unknown"
    env["PYTHONUNBUFFERED"] = "1"
    env["MOJOLEARN_BOARD_VENDOR"] = ctx["vendor"]     # tools/bench_board_probe.py
    if ctx["vendor"] == "nvidia" and ctx.get("ptxas"):
        env.setdefault("MODULAR_NVPTX_COMPILER_PATH", ctx["ptxas"])
    if ctx.get("artifact_manifest"):
        env["MOJOLEARN_BOARD_ARTIFACT_MANIFEST"] = ctx["artifact_manifest"]
        if ctx.get("receipt_dir"):
            env["MOJOLEARN_BOARD_RECEIPTS"] = ctx["receipt_dir"]
    if extra:
        env.update(extra)
    # Resource policy wins over shell and per-run diagnostic overrides.
    _load_tool("cpu_quota").apply_cpu_quota(env)
    return env


# ---------------------------------------------------------------------------
# The box fingerprint
# ---------------------------------------------------------------------------

def repo_sync(repo=REPO):
    """The patch sync's provenance (tools/dev_pod.sh writes .git/devpod_synced
    on a pod): {commit, worktree_dirty, base, patch_sha256, ...}, or None when
    this tree was not patch-synced or its HEAD is no longer that sync's base."""
    gd = capture(["git", "-C", repo, "rev-parse", "--absolute-git-dir"], timeout=20)
    head = capture(["git", "-C", repo, "rev-parse", "HEAD"], timeout=20)
    if not gd or not head:
        return None
    try:
        with open(os.path.join(gd, "devpod_synced")) as fh:
            rec = dict(l.strip().split("=", 1) for l in fh if "=" in l)
    except OSError:
        return None
    if rec.get("base") != head or not rec.get("commit"):
        return None
    return rec


def repo_commit(repo=REPO):
    """The repo commit this script shipped in: on a patch-synced pod the synced
    commit (repo_sync; the box's HEAD is only the merge base), git, else
    SHIPPED_COMMIT.txt (legs ship `git archive`, which has no .git), else the
    environment."""
    sync = repo_sync(repo)
    if sync:
        return sync["commit"] + ("-dirty" if sync.get("worktree_dirty") == "1" else "")
    c = capture(["git", "-C", repo, "rev-parse", "HEAD"], timeout=20)
    if c:
        dirty = capture(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"], timeout=20)
        return c + ("-dirty" if dirty else "")
    p = os.path.join(repo, "SHIPPED_COMMIT.txt")
    if os.path.exists(p):
        with open(p) as fh:
            return fh.read().strip() or None
    return os.environ.get("MOJOLEARN_REPO_COMMIT")


def _cpu_model():
    s = capture(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=10)
    if s:
        return s
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _mem_bytes():
    s = capture(["sysctl", "-n", "hw.memsize"], timeout=10)
    if s and s.isdigit():
        return int(s)
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _os_name():
    if platform.system() == "Darwin":
        return "macOS " + (platform.mac_ver()[0] or "?")
    try:
        with open("/etc/os-release") as fh:
            for line in fh:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return platform.system()


def gpu_info(vendor):
    info = {"vendor": vendor, "api": VENDOR_API.get(vendor)}
    if vendor == "nvidia":
        s = capture(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                     "--format=csv,noheader"], timeout=30)
        if s:
            first = [x.strip() for x in s.splitlines()[0].split(",")]
            info.update(name=first[0], driver=first[1] if len(first) > 1 else None,
                        memory=first[2] if len(first) > 2 else None,
                        count=len(s.splitlines()))
        info["cuda_driver_raw"] = capture(["nvidia-smi"], timeout=30)
        if info["cuda_driver_raw"]:
            info["cuda_driver_raw"] = info["cuda_driver_raw"].splitlines()[:4]
    elif vendor == "amd":
        s = capture(["rocm-smi", "--showproductname"], timeout=30)
        if s:
            for line in s.splitlines():
                if "Card Series" in line and ":" in line:
                    info["name"] = line.rsplit(":", 1)[1].strip()
                    break
        d = capture(["rocm-smi", "--showdriverversion"], timeout=30)
        if d:
            m = re.search(r"Driver version:\s*(\S+)", d)
            info["driver"] = m.group(1) if m else d.splitlines()[-1].strip()
        v = capture(["cat", "/opt/rocm/.info/version"], timeout=10)
        info["rocm"] = v
    elif vendor == "apple":
        s = capture(["system_profiler", "SPDisplaysDataType", "-json"], timeout=60)
        try:
            gpus = json.loads(s)["SPDisplaysDataType"] if s else []
            g = gpus[0] if gpus else {}
            info["name"] = g.get("sppci_model") or g.get("_name")
            info["cores"] = g.get("sppci_cores")
            info["metal"] = g.get("spdisplays_mtlgpufamilysupport")
        except (ValueError, KeyError, IndexError, TypeError):
            pass
        info["driver"] = "macOS " + (platform.mac_ver()[0] or "?")
    return info


def python_packages(python):
    s = capture([python, "-m", "pip", "list", "--format=json", "--disable-pip-version-check"],
                timeout=120)
    try:
        return {p["name"].lower(): p["version"] for p in json.loads(s)} if s else {}
    except (ValueError, KeyError, TypeError):
        return {}


def box_fingerprint(ctx):
    python = ctx["python"]
    pyver = capture([python, "-c", "import sys,platform;print(sys.version.split()[0], platform.python_implementation())"],
                    timeout=60)
    mj = capture([python, "-c", "import importlib.metadata as m;print(m.version('mojolearn'))"],
                 timeout=60)
    script = os.path.abspath(__file__)
    return {
        "host": {"hostname": platform.node(), "machine": platform.machine(),
                 "cpu_model": _cpu_model(), "cpu_count": os.cpu_count(),
                 "memory_bytes": _mem_bytes()},
        "os": {"name": _os_name(), "system": platform.system(),
               "release": platform.release(), "platform": platform.platform()},
        "gpu": gpu_info(ctx["vendor"]),
        "python": {"executable": python, "version": pyver,
                   "orchestrator": sys.version.split()[0]},
        "packages": python_packages(python),
        "mojolearn": {"version": mj, "requested": ctx.get("mojolearn_version"),
                      "wheel": ctx.get("wheel")},
        "repo": {"commit": ctx.get("commit"), "sync": repo_sync(),
                 "script": os.path.relpath(script, REPO),
                 "script_sha256": sha256_file(script)},
        "env": {k: os.environ.get(k) for k in THREAD_ENV + ("MODULAR_NVPTX_COMPILER_PATH",)},
    }


def pinned_package_names():
    """Every library the board pins (the opponents' sets, per family, and
    torch): their installed versions are part of an NVIDIA resume key."""
    names = {"torch"}
    reqs = [r for _, rs in opponent_pins().values() for r in rs]
    reqs += [r for rs in MORE_PINS.values() for r in rs] + list(NEURAL_PINS)
    reqs += [r for rs in ALGOS.PINS.values() for r in rs]
    reqs += [r for rs in ALGOS.RAPIDS_EXTRA.values() for r in rs]
    for r in reqs:
        m = re.match(r"^([A-Za-z0-9_.-]+)", r)
        if m:
            names.add(m.group(1).lower())
    return sorted(names)


def box_key(box):
    """What must not change between a run and its resume: the same box and
    the same library bytes. A board mixing two boxes is the patchwork that
    produced a wrong CatBoost headline (bench_all_ours.sh).

    NVIDIA (2026-09-29): a shared RunPod pod deletes
    itself when idle, and each race holds ours and its opponents measured
    together in one run, so a race is valid on its own. An NVIDIA board
    therefore resumes on a NEW pod when the GPU model, the driver major
    version, the wheel and every pinned library version are the same; the
    hostname is not in its key, and every race records the host it ran on
    (race_host). A different GPU model refuses: start a new --out. The
    opponent store keeps its own machine key (a reused opponent must come
    from the same machine)."""
    w = (box.get("mojolearn") or {}).get("wheel") or {}
    gpu = box.get("gpu") or {}
    key = {"vendor": gpu.get("vendor"),
           "gpu": gpu.get("name"),
           "hostname": (box.get("host") or {}).get("hostname"),
           "mojolearn": (box.get("mojolearn") or {}).get("version"),
           "wheel_sha256": w.get("sha256")}
    if box.get("artifact_identity"):
        key["artifact_identity"] = box["artifact_identity"]
        key["artifact_hardware"] = box.get("artifact_hardware")
    if gpu.get("vendor") == "nvidia":
        del key["hostname"]
        key["driver_major"] = str(gpu.get("driver") or "").split(".")[0] or None
        pk = box.get("packages") or {}
        key["pinned"] = {n: pk.get(n) for n in pinned_package_names() if pk.get(n)}
    return key


def race_host():
    """Where a race ran: the hostname and, on RunPod, the pod id."""
    return {"hostname": platform.node(), "pod_id": os.environ.get("RUNPOD_POD_ID")}


# ---------------------------------------------------------------------------
# The environment: venv, the wheel, the opponents
# ---------------------------------------------------------------------------

def cache_dir(args, out):
    """Bulky reusable state (venv, downloaded wheel, classical blocks) lives
    here, NOT beside board.json, so fetching a result directory home never
    drags gigabytes of blocks and a torch install with it."""
    return os.path.abspath(os.path.expanduser(args.cache or os.path.join(out, "cache")))


def setup_python(args, vendor, out, log):
    """Returns (python, wheel record or None). Creates the venv and installs
    unless --python-env/--skip-install say otherwise. The wheel is DOWNLOADED
    first so its bytes can be hashed, then installed from that file."""
    if args.python_env:
        python = args.python_env
        venv_dir = None
    else:
        venv_dir = os.path.abspath(args.venv or os.path.join(cache_dir(args, out), "venv"))
        python = os.path.join(venv_dir, "bin", "python")
        if not os.path.exists(python):
            cmd = [args.base_python, "-m", "venv"]
            if args.system_site_packages:
                cmd.append("--system-site-packages")
            cmd.append(venv_dir)
            rc = run_logged(cmd, None, log, 1800)
            if rc != 0:
                raise SystemExit("bench_board: venv creation failed (rc %d); see %s" % (rc, log))
    wheel = None
    if args.skip_install:
        return python, wheel
    pip = [python, "-m", "pip", "install", "--no-input", "--disable-pip-version-check"]
    wheels = os.path.join(cache_dir(args, out), "wheels")
    os.makedirs(wheels, exist_ok=True)
    if args.mojolearn_wheel:
        wfile = os.path.abspath(args.mojolearn_wheel)
    else:
        if not args.mojolearn_version:
            raise SystemExit("bench_board: --mojolearn-version is required to install the wheel")
        rc = run_logged([python, "-m", "pip", "download", "--no-input", "--no-deps",
                         "--disable-pip-version-check", "-d", wheels,
                         "mojolearn==%s" % args.mojolearn_version], None, log, 3600)
        cands = sorted(f for f in os.listdir(wheels)
                       if f.startswith("mojolearn-%s-" % args.mojolearn_version) and f.endswith(".whl"))
        if rc != 0 or not cands:
            raise SystemExit("bench_board: could not download mojolearn==%s (rc %d); see %s"
                             % (args.mojolearn_version, rc, log))
        wfile = os.path.join(wheels, cands[-1])
    wheel = {"file": os.path.basename(wfile), "sha256": sha256_file(wfile),
             "bytes": os.path.getsize(wfile)}
    if run_logged(pip + [wfile], None, log, 3600) != 0:
        raise SystemExit("bench_board: installing %s failed; see %s" % (wfile, log))
    # torch FIRST: the opponent sets that depend on torch (torch-geometric,
    # gpytorch, ...) then resolve against the pinned build instead of pulling
    # PyPI's newest torch (another CUDA major on Linux)
    torch_spec = args.torch_spec if args.torch_spec is not None else DEFAULT_TORCH_SPEC[vendor]
    if torch_spec:
        cmd = list(pip)
        if args.torch_index_url:
            cmd += ["--index-url", args.torch_index_url]
        elif args.torch_spec is None and DEFAULT_TORCH_EXTRA_INDEX.get(vendor):
            cmd += ["--extra-index-url", DEFAULT_TORCH_EXTRA_INDEX[vendor]]
        if run_logged(cmd + shlex.split(torch_spec), None, log, 3600) != 0:
            print("bench_board: torch install failed (%s); torch arms will refuse by name"
                  % torch_spec, flush=True)
    for idx, reqs in opponent_requirements(vendor, opponent_pins()):
        cmd = list(pip)
        if args.opponent_wheels:
            cmd += ["--no-index", "--find-links", os.path.abspath(args.opponent_wheels)]
        elif idx:
            cmd += ["--extra-index-url", idx]
        # A failed opponent install is NOT fatal: that arm then refuses by
        # name inside its race, which is visible on the board.
        rc = run_logged(cmd + reqs, None, log, 3600)
        if rc != 0:
            print("bench_board: opponent install rc %d for %s (their arms will refuse by name)"
                  % (rc, " ".join(reqs)), flush=True)
    if "classical2" in (args.families or ""):
        cmd = list(pip)
        if args.opponent_wheels:
            cmd += ["--no-index", "--find-links", os.path.abspath(args.opponent_wheels)]
        rc = run_logged(cmd + MORE_PINS[vendor], None, log, 3600)
        if rc != 0:
            print("bench_board: classical2 opponent install rc %d for %s (their arms will refuse "
                  "by name)" % (rc, " ".join(MORE_PINS[vendor])), flush=True)
    if "algos" in (args.families or ""):
        cmd = list(pip)
        if args.opponent_wheels:
            cmd += ["--no-index", "--find-links", os.path.abspath(args.opponent_wheels)]
        rc = run_logged(cmd + ALGOS.PINS[vendor], None, log, 3600)
        if rc != 0:
            print("bench_board: algos opponent install rc %d for %s (their arms will refuse "
                  "by name)" % (rc, " ".join(ALGOS.PINS[vendor])), flush=True)
        extra = ALGOS.RAPIDS_EXTRA.get(vendor)
        if extra:
            idx = next((v[0] for k, v in opponent_pins().items() if k.startswith("rapids-")), None)
            cmd = list(pip)
            if args.opponent_wheels:
                cmd += ["--no-index", "--find-links", os.path.abspath(args.opponent_wheels)]
            elif idx:
                cmd += ["--extra-index-url", idx]
            if run_logged(cmd + extra, None, log, 3600) != 0:
                print("bench_board: %s install failed (the cugraph-gpu arms will refuse by name)"
                      % " ".join(extra), flush=True)
    if "neural" in (args.families or ""):
        cmd = list(pip)
        if args.opponent_wheels:
            cmd += ["--no-index", "--find-links", os.path.abspath(args.opponent_wheels)]
        rc = run_logged(cmd + NEURAL_PINS, None, log, 3600)
        if rc != 0:
            print("bench_board: neural opponent install rc %d for %s (the Mamba-1 torch arms will "
                  "refuse by name)" % (rc, " ".join(NEURAL_PINS)), flush=True)
    return python, wheel


#: Run in the board's interpreter: can the installed wheel's GPU set load on
#: this device? (2026-09-29: an A40 is sm_86 and 0.8.25 carries sm_89 and
#: sm_90a only; the wheel fell back to its CPU set and 18 NVIDIA races
#: refused our arm one by one.)
_GPU_SET_PROBE = r"""
import importlib.util, sys
if importlib.util.find_spec("mojolearn") is None:
    print("NO-MOJOLEARN"); sys.exit(0)
try:
    import mojolearn._backend as b
except Exception as e:
    print("%s: %s" % (type(e).__name__, e)); sys.exit(3)
try:
    b.tier_dir("identical")
except Exception as e:
    print("%s: %s" % (type(e).__name__, e)); sys.exit(3)
print("OK")
"""


def gpu_set_refusal(python, vendor):
    """None when our GPU set loads here (or on Apple, or no mojolearn is
    installed in the interpreter, as in the unit tests); else the wheel's own
    reason."""
    if vendor not in ("nvidia", "amd"):
        return None
    try:
        p = subprocess.run([python, "-c", _GPU_SET_PROBE], capture_output=True, text=True,
                           timeout=300)
    except (OSError, subprocess.TimeoutExpired) as e:
        return "the probe did not run: %s" % e
    if p.returncode == 0:
        return None
    return ((p.stdout or "") + (p.stderr or "")).strip()[-2000:] or "rc %d" % p.returncode


def setup_arm_venvs(args, vendor, out, log):
    """{arm: interpreter} for the arms whose library needs its own clean venv
    (tools/bench_board_algos.py ARM_VENVS), created and installed under the
    cache. A failed install is not fatal: the arm then refuses by name in its
    race, in the board's venv."""
    if args.skip_install or args.python_env or "algos" not in (args.families or ""):
        return {}
    got = {}
    for arm, reqs in sorted(ALGOS.ARM_VENVS.get(vendor, {}).items()):
        vdir = os.path.join(cache_dir(args, out), "venv-" + arm)
        py = os.path.join(vdir, "bin", "python")
        if not os.path.exists(py) and run_logged([args.base_python, "-m", "venv", vdir], None,
                                                 log, 1800) != 0:
            print("bench_board: venv for %s failed (it refuses by name)" % arm, flush=True)
            continue
        cmd = [py, "-m", "pip", "install", "--no-input", "--disable-pip-version-check",
               "--extra-index-url", ALGOS.ARM_VENV_INDEX] + list(reqs)
        if run_logged(cmd, None, log, 3600) != 0:
            print("bench_board: %s install failed for %s (it refuses by name)"
                  % (" ".join(reqs), arm), flush=True)
            continue
        got[arm] = py
    return got


def data_status(data_root, datasets, verify=False):
    """{dataset: {path, present, bytes, pinned_bytes, sha256_ok}}."""
    pins = {}
    manifest = os.path.join(REPO, "bench", "results", "dataset_store", "manifest.tsv")
    try:
        with open(manifest) as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) >= 3:
                    pins[f[0]] = (f[1], f[2])
    except OSError:
        pass
    out = {}
    for ds in datasets:
        path = os.path.join(data_root, DATA_FILES[ds])
        rec = {"path": path, "present": os.path.isfile(path), "r2_key": R2_KEYS[ds]}
        pin = pins.get(R2_KEYS[ds])
        if pin:
            rec["pinned_bytes"] = int(pin[0])
            rec["pinned_sha256"] = pin[1]
        if rec["present"]:
            rec["bytes"] = os.path.getsize(path)
            rec["size_ok"] = (rec["bytes"] == rec.get("pinned_bytes")) if pin else None
            if verify and pin:
                rec["sha256_ok"] = sha256_file(path) == pin[1]
        out[ds] = rec
    return out


# ---------------------------------------------------------------------------
# Result file
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# --invalidate-memory: a GPU figure read with the wrong counter, withdrawn
# ---------------------------------------------------------------------------

#: The GPU counters that are only valid for a torch arm (tools/bench_board_probe.py).
TORCH_GPU_METHODS = ("torch.cuda.max_memory_allocated", "torch.mps.driver_allocated_memory")


def _wrong_gpu_counter(cell):
    """A GPU cell of a non-torch arm whose GPU figure came from torch's
    allocator (before bench_board_probe took the arm's library, 2026-09-29)."""
    m = cell.get("memory") or {}
    return (cell.get("device") == "gpu"
            and str(cell.get("library") or "") not in PROBE.TORCH_LIBRARIES
            and str(m.get("gpu_method") or "").startswith(TORCH_GPU_METHODS))


def _withdraw_gpu_figure(cell, note):
    cell["peak_gpu_mb"] = None
    m = cell.setdefault("memory", {})
    m["peak_gpu_mb"] = None
    m["warmup_gpu_mb"] = None
    m["gpu_method_withdrawn"] = m.get("gpu_method")
    m["gpu_method"] = note


def invalidate_memory(out, prefixes, reason, fixed_at, store_path=None):
    """Mark the GPU memory of every finished cell whose figure came from the
    wrong counter (_wrong_gpu_counter) as not measured, in the races whose id
    starts with one of `prefixes` (comma list; "all" for every race). The
    times are untouched: the probe only read memory. The opponent store gets
    a corrected copy of each such record (the store keeps the latest per
    key). Idempotent: a withdrawn figure no longer carries a torch counter.
    Returns (board cells marked, store records corrected)."""
    rpath = os.path.join(out, "board.json")
    result = load_result(rpath)
    if result is None:
        raise SystemExit("--invalidate-memory: no board.json under %s" % out)
    want = [x.strip() for x in (prefixes or "").split(",") if x.strip()]
    note = "not measured (%s; fixed at %s)" % (reason, fixed_at)
    n_board = 0
    for rid, rec in (result.get("races") or {}).items():
        if not ("all" in want or any(rid.startswith(x) for x in want)):
            continue
        for c in rec.get("cells") or []:
            if _wrong_gpu_counter(c):
                _withdraw_gpu_figure(c, note)
                n_board += 1
    n_store = 0
    if store_path and os.path.exists(store_path):
        for r in STORE.load(store_path).values():
            k = r.get("key") or {}
            rid = "%s/%s/%s" % (k.get("family"), k.get("lane"), k.get("dataset"))
            cell = r.get("cell") or {}
            if ("all" in want or any(rid.startswith(x) for x in want)) and _wrong_gpu_counter(cell):
                fixed = json.loads(json.dumps(r))
                _withdraw_gpu_figure(fixed["cell"], note)
                fixed["corrected_at"] = now_utc()
                STORE.append(store_path, fixed)
                n_store += 1
    if n_board:
        result.setdefault("corrections", []).append(
            {"at": now_utc(), "what": "peak_gpu_mb withdrawn", "prefixes": want, "note": note,
             "cells": n_board, "store_records": n_store})
        save_result(rpath, result)
        write_board(out, result)
    return n_board, n_store


# ---------------------------------------------------------------------------
# --invalidate-arm: an arm that did not run what its name says, refused after the fact
# ---------------------------------------------------------------------------

def _refuse_cell(cell, reason):
    cell["status"] = "REFUSED(%s)" % reason
    cell["withdrawn"] = {"median_ms": cell.get("median_ms"), "min_ms": cell.get("min_ms"),
                         "max_ms": cell.get("max_ms"), "times_ms": cell.get("times_ms"),
                         "quality": cell.get("quality")}
    for k in ("median_ms", "min_ms", "max_ms", "warmup_ms", "quality", "peak_gpu_mb", "peak_host_mb",
              "ratio_ours_identical_over", "ratio_ours_fast_over"):
        if k in cell:
            cell[k] = None
    cell["times_ms"] = []
    cell["rounds"] = 0


def invalidate_arm(out, arm, reason, store_path=None, vendor=None):
    """Mark every finished cell of `arm` (fit and inference, run or stored) as
    REFUSED(reason): its times and quality are withdrawn (kept under
    `withdrawn`), the rest of the race stays, and the ratios are recomputed
    without it. For an arm whose measurement was not what its name says (on
    do-amd, 2026-09-29, `xgboost-gpu` trained on the host CPU), so the races
    need not be run again. The opponent store gets a refused copy of each of
    its records of the arm (of `vendor` when given). Idempotent. Returns
    (board cells refused, store records refused)."""
    rpath = os.path.join(out, "board.json")
    result = load_result(rpath)
    if result is None:
        raise SystemExit("--invalidate-arm: no board.json under %s" % out)
    n_board = 0
    for rid, rec in (result.get("races") or {}).items():
        hit = False
        for c in rec.get("cells") or []:
            if c.get("arm") == arm and not str(c.get("status", "")).startswith("REFUSED"):
                _refuse_cell(c, reason)
                n_board += 1
                hit = True
        infer = rec.get("infer_cells") or []
        for c in infer:
            if c.get("arm") == arm and not str(c.get("status", "")).startswith("REFUSED"):
                _refuse_cell(c, reason)
                n_board += 1
                hit = True
        if hit:
            rec["cells"] = add_ratios(rec.get("cells") or [])
            if infer:
                INFER._ratios(infer, _bb())
    n_store = 0
    if store_path and os.path.exists(store_path):
        for r in STORE.load(store_path).values():
            k = r.get("key") or {}
            cell = r.get("cell") or {}
            if k.get("arm") != arm or (vendor and k.get("vendor") != vendor) \
                    or str(cell.get("status", "")).startswith("REFUSED"):
                continue
            fixed = json.loads(json.dumps(r))
            _refuse_cell(fixed["cell"], reason)
            for ic in fixed.get("infer_cells") or []:
                _refuse_cell(ic, reason)
            fixed["corrected_at"] = now_utc()
            STORE.append(store_path, fixed)
            n_store += 1
    if n_board:
        result.setdefault("corrections", []).append(
            {"at": now_utc(), "what": "arm refused after the fact", "arm": arm, "reason": reason,
             "cells": n_board, "store_records": n_store})
        save_result(rpath, result)
        write_board(out, result)
    return n_board, n_store


def load_result(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return strip_our_cpu(json.load(fh))


def save_result(path, result):
    result["updated"] = now_utc()
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Trees: run forest_speed_arm.py, parse its FSPEED lines
# ---------------------------------------------------------------------------

_KV = re.compile(r"(\S+?)=(.*?)(?=\s+\S+?=|$)")


def _kv(text):
    return {m.group(1): m.group(2).strip() for m in _KV.finditer(text.strip())}


def parse_tree_log(path):
    """{arms: {arm: rec}, verdict, verdict_line, shape, notes, bindings}."""
    summ = _load_tool("bench_all_summarize")
    arms, verdict, shape = summ.parse_tree_log(path)
    bindings, warm, notes, verdict_line, mem, libs = {}, {}, [], None, {}, {}
    states = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            head, _, rest = line.rstrip("\n").partition(" ")
            if head == "BENCH_BINDING":
                f = _kv(rest)
                bindings[f.get("arm")] = f
            elif head == "FSPEED-WARMUP":
                f = _kv(rest)
                try:
                    warm[f.get("arm")] = float(f["ms"])
                except (KeyError, ValueError):
                    pass
            elif head == "FSPEED-NOTE":
                notes.append(rest[:300])
            elif head == "FSPEED-LIBRARY":
                try:
                    d = json.loads(rest)
                    libs[d.get("arm")] = d
                except ValueError:
                    pass
            elif head == "FSPEED-STATE":
                d = json.loads(rest)
                states[d["arm"]] = d["receipt"]
            elif head == "FSPEED-FIT-VERDICT":
                verdict_line = rest[:300]
            elif head == "FSPEED-MEM":
                f = _kv(rest)
                try:
                    r = int(f.get("round"))
                except (TypeError, ValueError):
                    continue
                num = {}
                for k in ("host_mb", "gpu_mb", "children_mb"):
                    try:
                        num[k] = float(f[k])
                    except (KeyError, ValueError):
                        num[k] = None
                num["host_method"] = f.get("host_method")
                num["gpu_method"] = f.get("gpu_method")
                mem.setdefault(f.get("arm"), {})[r] = num
    return {"arms": arms, "verdict": verdict, "verdict_line": verdict_line,
            "shape": shape, "notes": notes, "bindings": bindings, "warmup": warm,
            "mem": {a: [rs[k] for k in sorted(rs)] for a, rs in mem.items()},
            "libraries": libs, "state_receipts": states}


def tree_cmd(ctx, race):
    ours = race["our_arms"]
    primary = ours.get("ours", "identical")
    cmd = [ctx["python"], "-u", ctx["tree_driver"], "--lane", race["lane"],
           "--dataset", tree_driver_dataset(race["lane"], race["dataset"])]
    if race["rows"]:
        cmd += ["--rows", str(int(race["rows"]))]
    if race["opponents"]:
        cmd += ["--devices", tree_devices(ctx["vendor"], race["lane"]),
                "--arms", ",".join(race["opponents"])]
    else:
        cmd += ["--ours-only"]
    if "ours-ab" in ours:
        cmd += ["--ours-ab", "numeric_mode='%s'" % ours["ours-ab"]]
    if ctx.get("opponents_only"):
        if ours or not race["opponents"]:
            raise ValueError("opponents-only tree command requires only opponents")
        cmd += ["--opponents-only"]
    cmd += ["--mem"]
    if ctx.get("infer"):
        cmd += INFER.driver_args(race)
    env = {"MOJOLEARN_NUMERIC_MODE": primary,
           "MOJOLEARN_SPEED_EXPECTED_VENDOR": VENDOR_API[ctx["vendor"]],
           "MOJOLEARN_SPEED_ROUNDS": str(ctx["rounds"]),
           "MOJOLEARN_SPEED_SIZE": "shipped",
           "MOJOLEARN_SPEED_BUDGET_S": str(ctx["arm_budget_s"]),
           "MOJOLEARN_SPEED_DEADLINE_S": str(ctx["race_deadline_s"])}
    return cmd, env


def _status(ms, refused, rounds):
    if refused and not ms:
        return "REFUSED(%s)" % refused
    if not ms:
        return "UNKNOWN(no rounds)"
    if len(ms) < rounds:
        return "PARTIAL(%d/%d rounds)" % (len(ms), rounds)
    return "ok"


def _from_wheel(path, ctx):
    if not path:
        return "UNKNOWN(no binding path)"
    repo_py = os.path.join(REPO, "python") + os.sep
    if os.path.abspath(path).startswith(repo_py):
        return "NOT-FROM-WHEEL(%s)" % path
    return "wheel" if "site-packages" in path or "dist-packages" in path else "UNKNOWN(%s)" % path


def tree_cells(ctx, race, parsed):
    cells = []
    seen = set()
    rounds = ctx["rounds"]
    names = list(race["arms"]) + [a for a in parsed["arms"] if a not in race["arms"]]
    for arm in names:
        if arm in seen or arm in ("all", "*-cpu"):
            continue
        seen.add(arm)
        a = parsed["arms"].get(arm)
        mode = race["our_arms"].get(arm)
        cell = base_cell(ctx, race, arm, mode)
        if a is None:
            cell["status"] = "UNKNOWN(not in log)"
            cells.append(cell)
            continue
        ms = a["ms"]
        cell.update(times_ms=ms, warmup_ms=parsed["warmup"].get(arm),
                    median_ms=float(statistics.median(ms)) if ms else None,
                    min_ms=min(ms) if ms else None, max_ms=max(ms) if ms else None,
                    rounds=len(ms), status=_status(ms, a["refused"], rounds),
                    quality=dict(a["acc"]),
                    hash=(a["hashes"][-1] if a["hashes"] else None),
                    hash_stable=(len(set(a["hashes"])) == 1) if a["hashes"] else None,
                    state_receipts=([parsed["state_receipts"][arm]]
                                    if arm in parsed.get("state_receipts", {}) else []),
                    fit=a["fit"], shape=parsed["shape"],
                    comparability={"fit_verdict": parsed["verdict"] or "UNKNOWN",
                                   "fit_verdict_line": parsed["verdict_line"]},
                    verdict=parsed["verdict"] or "UNKNOWN")
        cell.update(memory_fields(parsed.get("mem", {}).get(arm)))
        lib = (parsed.get("libraries") or {}).get(arm) or {}
        cell["library_version"] = lib.get("version")
        cell["device_name"] = lib.get("device_name")
        if mode:
            b = parsed["bindings"].get(arm) or {}
            cell["binding"] = b
            cell["mode_witness"] = b.get("compiled")
            cell["installed_wheel"] = _from_wheel(b.get("path"), ctx)
            if b and (b.get("compiled") != mode or b.get("resolved") != mode):
                cell["status"] = "MODE-MISMATCH(requested %s, compiled %s)" % (mode, b.get("compiled"))
        cells.append(cell)
    return strip_our_cpu_cells(cells)


def memory_fields(samples):
    """peak_host_mb / peak_gpu_mb and the `memory` record of one cell from
    its per-round samples (warm-up first; tools/bench_board_probe.py)."""
    if not samples:
        return {"peak_host_mb": None, "peak_gpu_mb": None,
                "memory": {"host_method": "not sampled", "gpu_method": "not sampled"}}
    m = PROBE.summarize(samples)
    return {"peak_host_mb": m["peak_host_mb"], "peak_gpu_mb": m["peak_gpu_mb"], "memory": m}


def is_our_cpu_cell(c):
    """A cell of ours that ran on the CPU: the old `ours-cpu` arm, any arm of
    ours on the host (bench_board_probe.is_our_cpu_arm), or a mojolearn cell
    whose device is the CPU (an old neural *-infer / host train lane)."""
    if c.get("library") != "mojolearn" and not str(c.get("arm", "")).startswith("ours"):
        return False
    return c.get("arm") == CPU_ARM or PROBE.is_our_cpu_arm(c.get("arm")) \
        or c.get("device") == "cpu"


def strip_our_cpu_cells(cells):
    """Our CPU is never on a board (Andrew, Oct 2 2026): drop every cell of ours
    on the CPU and every our-CPU ratio, whatever an old record holds."""
    out = []
    for c in cells or []:
        if is_our_cpu_cell(c):
            continue
        c.pop("ratio_ours_cpu_over", None)
        out.append(c)
    return out


def strip_our_cpu(result):
    """strip_our_cpu_cells over a whole board record, in place. A race whose
    every cell of ours ran on the CPU (and has none on the GPU) is dropped."""
    races = (result or {}).get("races") or {}
    for rid in list(races):
        rr = races[rid]
        if rr.get("infer_cells"):
            rr["infer_cells"] = strip_our_cpu_cells(rr["infer_cells"])
        cells = rr.get("cells")
        if not cells:
            continue
        had_ours = any(c.get("library") == "mojolearn" for c in cells)
        kept = strip_our_cpu_cells(cells)
        if had_ours and not any(c.get("library") == "mojolearn" for c in kept):
            del races[rid]
            continue
        rr["cells"] = kept
    return result


# ---------------------------------------------------------------------------
# Classical: prep once, race per (lane, dataset); read the race JSON
# ---------------------------------------------------------------------------

def classical_data_dir(ctx, rows):
    base = ctx["ctd_data"] or os.path.join(ctx["out"], "ctd-data")
    return os.path.join(base, "rows-" + rows_tag(rows))


def ensure_classical_prep(ctx, races):
    """Untimed block prep, once per box and row cap; skipped when every block
    the races need already has its JSON record."""
    need = {}
    for r in races:
        if r["family"] == "classical":
            need.setdefault(r["rows"], set()).add((r["lane"], r["dataset"]))
    for rows, pairs in need.items():
        d = classical_data_dir(ctx, rows)
        missing = sorted(p for p in pairs if not os.path.exists(
            os.path.join(d, "%s-%s.json" % (CLASSICAL_BLOCK[p[0]], p[1]))))
        if not missing:
            continue
        lanes = sorted({p[0] for p in missing})
        dss = sorted({p[1] for p in missing})
        cmd = [ctx["python"], "-u", ctx["classical_driver"], "prep", "--data", d,
               "--lanes", ",".join(lanes), "--datasets", ",".join(dss)]
        if rows:
            cmd += ["--max-rows", str(int(rows))]
        log = os.path.join(ctx["out"], "logs", "classical-prep-rows-%s.log" % rows_tag(rows))
        print("bench_board: classical prep rows=%s lanes=%s datasets=%s"
              % (rows_tag(rows), ",".join(lanes), ",".join(dss)), flush=True)
        rc = run_logged(cmd, child_env(ctx), log, 6 * 3600, nice=ctx["nice"])
        if rc != 0:
            print("bench_board: classical prep rc %d (see %s); its races will fail by name"
                  % (rc, log), flush=True)


def classical_cmd(ctx, race):
    lane, ds = race["lane"], race["dataset"]
    rsec = ctx["round_seconds"] or classical_round_seconds(lane, ds, ctx["vendor"])
    cmd = [ctx["python"], "-u", ctx["classical_driver"], "race",
           "--lane", lane, "--dataset", ds,
           "--data", classical_data_dir(ctx, race["rows"]),
           "--out", os.path.join(ctx["out"], "raw", "classical", "rows-" + rows_tag(race["rows"])),
           "--work", os.path.join(ctx["out"], "work"),
           "--root", REPO,
           "--arms", ",".join(race["arms"]),
           "--rounds", str(ctx["rounds"]),
           "--round-seconds", str(rsec),
           "--warmup-seconds", str(max(rsec, 600)),
           "--ours-python", shlex.quote(ctx["python"]),
           "--theirs-python", shlex.quote(ctx["python"])]
    if ctx.get("infer"):
        cmd += INFER.driver_args(race)
    env = {}
    ceiling = 600 + max(rsec, 600) * len(race["arms"]) + rsec * ctx["rounds"] * len(race["arms"]) + 900
    return cmd, env, ceiling


def more_data_dir(ctx, rows):
    base = ctx.get("more_data") or os.path.join(ctx["out"], "more-data")
    return os.path.join(base, "rows-" + rows_tag(rows))


def ensure_more_prep(ctx, races):
    """classical2 block prep, untimed, once per box and row cap; skipped when
    every block the races need already has its JSON record."""
    need = {}
    for r in races:
        if r["family"] == "classical2":
            need.setdefault(r["rows"], set()).add((r["lane"], r["dataset"]))
    for rows, pairs in need.items():
        d = more_data_dir(ctx, rows)
        missing = sorted(p for p in pairs if not os.path.exists(
            os.path.join(d, "%s-%s.json" % (MORE.block_of(p[0]), p[1]))))
        if not missing:
            continue
        lanes = sorted({p[0] for p in missing})
        dss = sorted({p[1] for p in missing if p[1] in DATASETS}) or ["taxi"]
        cmd = [ctx["python"], "-u", ctx["more_driver"], "prep", "--data", d,
               "--lanes", ",".join(lanes), "--datasets", ",".join(dss)]
        if rows:
            cmd += ["--max-rows", str(int(rows))]
        log = os.path.join(ctx["out"], "logs", "classical2-prep-rows-%s.log" % rows_tag(rows))
        print("bench_board: classical2 prep rows=%s lanes=%s datasets=%s"
              % (rows_tag(rows), ",".join(lanes), ",".join(dss)), flush=True)
        rc = run_logged(cmd, child_env(ctx), log, 6 * 3600, nice=ctx["nice"])
        if rc != 0:
            print("bench_board: classical2 prep rc %d (see %s); its races will fail by name"
                  % (rc, log), flush=True)


def more_round_seconds(rows):
    return 600 if rows else 3600


def more_cmd(ctx, race):
    """tools/bench_board_more.py race for one classical2 (lane, dataset)."""
    rsec = ctx["round_seconds"] or more_round_seconds(race["rows"])
    cmd = [ctx["python"], "-u", ctx["more_driver"], "race",
           "--lane", race["lane"], "--dataset", race["dataset"],
           "--data", more_data_dir(ctx, race["rows"]),
           "--arms", ",".join(race["arms"]),
           "--rounds", str(ctx["rounds"]),
           "--out", os.path.join(ctx["out"], "raw", "classical2", "rows-" + rows_tag(race["rows"])),
           "--work", os.path.join(ctx["out"], "work"),
           "--ours-python", shlex.quote(ctx["python"]),
           "--theirs-python", shlex.quote(ctx["python"]),
           "--ready-seconds", str(rsec), "--warmup-seconds", str(rsec),
           "--round-seconds", str(rsec)]
    n = len(race["arms"])
    ceiling = 600 + rsec * n * 2 + rsec * ctx["rounds"] * n + 900
    return cmd, {}, ceiling


def more_json_path(ctx, race):
    return os.path.join(ctx["out"], "raw", "classical2", "rows-" + rows_tag(race["rows"]),
                        "%s-%s.json" % (race["lane"], race["dataset"]))


def algos_data_dir(ctx, rows):
    base = ctx.get("algos_data") or os.path.join(ctx["out"], "algos-data")
    return os.path.join(base, "rows-" + rows_tag(rows))


def ensure_algos_prep(ctx, races):
    """algos block prep, untimed, once per box and row cap (the driver skips
    every block whose JSON record exists)."""
    need = {}
    for r in races:
        if r["family"] == "algos":
            need.setdefault(r["rows"], set()).add((r["lane"], r["dataset"]))
    for rows, pairs in need.items():
        lanes = sorted({p[0] for p in pairs})
        dss = sorted({p[1] for p in pairs if p[1] in DATASETS}) or ["taxi"]
        cmd = [ctx["python"], "-u", ctx["algos_driver"], "prep", "--data", algos_data_dir(ctx, rows),
               "--lanes", ",".join(lanes), "--datasets", ",".join(dss)]
        if rows:
            cmd += ["--max-rows", str(int(rows))]
        log = os.path.join(ctx["out"], "logs", "algos-prep-rows-%s.log" % rows_tag(rows))
        print("bench_board: algos prep rows=%s lanes=%d datasets=%s"
              % (rows_tag(rows), len(lanes), ",".join(dss)), flush=True)
        rc = run_logged(cmd, child_env(ctx), log, 6 * 3600, nice=ctx["nice"])
        if rc != 0:
            print("bench_board: algos prep rc %d (see %s); its races will fail by name"
                  % (rc, log), flush=True)


def algos_cmd(ctx, race):
    """tools/bench_board_algos.py race for one algos (lane, dataset)."""
    rsec = ctx["round_seconds"] or more_round_seconds(race["rows"])
    cmd = [ctx["python"], "-u", ctx["algos_driver"], "race",
           "--lane", race["lane"], "--dataset", race["dataset"],
           "--data", algos_data_dir(ctx, race["rows"]),
           "--arms", ",".join(race["arms"]),
           "--rounds", str(ctx["rounds"]),
           "--out", os.path.join(ctx["out"], "raw", "algos", "rows-" + rows_tag(race["rows"])),
           "--work", os.path.join(ctx["out"], "work"),
           "--ours-python", shlex.quote(ctx["python"]),
           "--theirs-python", shlex.quote(ctx["python"]),
           "--ready-seconds", str(rsec), "--warmup-seconds", str(rsec),
           "--round-seconds", str(rsec)]
    for arm, py in sorted((ctx.get("arm_python") or {}).items()):
        if arm in race["arms"]:
            cmd += ["--arm-python", "%s=%s" % (arm, py)]
    if race["rows"]:
        cmd += ["--smoke-rows", str(int(race["rows"]))]
    n = len(race["arms"])
    ceiling = 600 + rsec * n * 2 + rsec * ctx["rounds"] * n + 900
    return cmd, {}, ceiling


def algos_json_path(ctx, race):
    return os.path.join(ctx["out"], "raw", "algos", "rows-" + rows_tag(race["rows"]),
                        "%s-%s.json" % (race["lane"], race["dataset"]))


ALGOS_SKIPPED = "SKIPPED: not built yet"


def algos_skips(cells, r):
    """An `ours` arm whose class the installed wheel does not export reads
    SKIPPED, never an error (the algorithm lanes merge classes all day)."""
    arms = (r or {}).get("arms") or {}
    for c in cells:
        a = arms.get(c["arm"]) or {}
        if a.get("status") == "skipped":
            c["status"] = ALGOS_SKIPPED
            c["skip_reason"] = a.get("error")
    return cells


def neural_round_seconds(shape):
    return 600 if shape == "small" else 1800


def neural_cmd(ctx, race):
    """tools/bench_board_neural.py race for one neural lane."""
    rsec = ctx["round_seconds"] or neural_round_seconds(race.get("shape"))
    cmd = [ctx["python"], "-u", ctx["neural_driver"], "race",
           "--lane", race["lane"], "--shape", race.get("shape") or "full",
           "--arms", ",".join(race["arms"]),
           "--rounds", str(ctx["rounds"]),
           "--out", os.path.join(ctx["out"], "raw", "neural", "shape-" + (race.get("shape") or "full")),
           "--work", os.path.join(ctx["out"], "work"),
           "--ours-python", shlex.quote(ctx["python"]),
           "--theirs-python", shlex.quote(ctx["python"]),
           "--ready-seconds", str(rsec), "--warmup-seconds", str(rsec),
           "--round-seconds", str(rsec)]
    n = len(race["arms"])
    ceiling = 600 + rsec * n * 2 + rsec * ctx["rounds"] * n + 900
    return cmd, {}, ceiling


def neural_json_path(ctx, race):
    return os.path.join(ctx["out"], "raw", "neural", "shape-" + (race.get("shape") or "full"),
                        "%s-%s.json" % (race["lane"], race["dataset"]))


def classical_json_path(ctx, race):
    return os.path.join(ctx["out"], "raw", "classical", "rows-" + rows_tag(race["rows"]),
                        "%s-%s.json" % (race["lane"], race["dataset"]))


def classical_cells(ctx, race, r):
    cells = []
    rounds = ctx["rounds"]
    shapes = (r.get("block") or {}).get("arrays", {})
    first = shapes.get("X") or shapes.get("index") or {}
    # the neural racer names its shape itself (no classical block)
    shape = r.get("shape") or "x".join(str(s) for s in first.get("shape", [])) or None
    qual = r.get("quality") or {}
    spans = r.get("spans") or {}
    ours_span = (((r.get("arms") or {}).get("ours") or {}).get("span")
                 or spans.get("ours") or {})
    names = list(race["arms"]) + [a for a in (r.get("arms") or {}) if a not in race["arms"]]
    for arm in names:
        a = (r.get("arms") or {}).get(arm)
        mode = race["our_arms"].get(arm)
        cell = base_cell(ctx, race, arm, mode)
        if a is None:
            cell["status"] = "UNKNOWN(not in race json)"
            cells.append(cell)
            continue
        ms = a.get("ms") or []
        info = a.get("info") or {}
        refused = None if a.get("status") == "ok" else "%s: %s" % (
            a.get("status"), json.dumps(a.get("error"))[:200])
        q = {k: v for k, v in (qual.get(arm) or {}).items() if k != "reference"}
        span = a.get("span") or spans.get(arm) or {}
        asym = []
        if arm_library(arm) != "mojolearn":
            if span.get("input_home") == "device" and ours_span.get("input_home") == "host":
                asym.append("upload_outside_its_clock")
            if span.get("pre_clock_fit") and not ours_span.get("pre_clock_fit"):
                asym.append("fit_before_its_clock")
        verdict = ("SPAN-ASYMMETRIC(%s)" % "+".join(asym)) if asym else "LIKE-FOR-LIKE-SPAN"
        cell.update(times_ms=ms, warmup_ms=a.get("warmup_ms"),
                    median_ms=float(statistics.median(ms)) if ms else None,
                    min_ms=min(ms) if ms else None, max_ms=max(ms) if ms else None,
                    rounds=len(ms), status=_status(ms, refused, rounds),
                    # the last round digest; a race whose rounds carry none (the neural training
                    # lanes) falls back to its saved-outputs digest (bench_board_neural.outputs_digest)
                    quality=q, hash=(a.get("digests") or [None])[-1] or a.get("outputs_digest"),
                    hash_stable=a.get("digest_stable"), shape=shape,
                    state_receipts=a.get("state_receipts", []), operations=a.get("operations", []),
                    device=info.get("device", cell["device"]),
                    device_name=info.get("device_name") or info.get("cpu_model"),
                    library_version=info.get("version"),
                    comparability={"span": span, "span_asymmetry": asym},
                    verdict=verdict)
        cell.update(memory_fields(a.get("mem")))
        if mode:
            cell["mode_witness"] = info.get("numeric_mode_used")
            cell["installed_wheel"] = _from_wheel(info.get("module_path"), ctx)
            if info and info.get("numeric_mode_used") not in (None, mode):
                cell["status"] = "MODE-MISMATCH(requested %s, read back %s)" % (
                    mode, info.get("numeric_mode_used"))
        cells.append(cell)
    return strip_our_cpu_cells(cells)


# ---------------------------------------------------------------------------
# Cells and ratios
# ---------------------------------------------------------------------------

def base_cell(ctx, race, arm, mode):
    lib = arm_library(arm)
    if _ours_runs_on_cpu(race["family"], race["lane"], arm):
        raise SystemExit("bench_board: refusing to record %s for %s: our CPU never races "
                         "(GPU-only board, Andrew Oct 2 2026)" % (arm, race["id"]))
    return {
        "family": race["family"], "lane": race["lane"], "dataset": race["dataset"],
        "rows": race["rows"], "rows_tag": rows_tag(race["rows"]),
        "neural_shape": race.get("shape"),
        "arm": arm, "library": lib,
        "mode": mode if lib == "mojolearn" else "opponent",
        "device": arm_device(arm, ctx["vendor"]),
        "settings": race_settings(ctx, race),
        "times_ms": [], "warmup_ms": None, "median_ms": None, "min_ms": None,
        "max_ms": None, "rounds": 0, "status": "UNKNOWN",
        "quality": {}, "hash": None, "hash_stable": None, "verdict": "UNKNOWN",
        "comparability": {},
        # peak memory over the timed rounds; the method is in `memory`
        "peak_host_mb": None, "peak_gpu_mb": None, "memory": {},
    }


_SETTINGS_CACHE = {}


def race_settings(ctx, race):
    key = (race["family"], race["lane"], race.get("shape"))
    if key not in _SETTINGS_CACHE:
        s = {"seed": SEED, "rounds": ctx["rounds"], "warmup_rounds": 1,
             "interleaved": True, "rows_cap": race["rows"]}
        if race["family"] == "neural":
            s["driver"] = "tools/bench_board_neural.py"
            s["numeric_mode"] = (
                "identical (`ours`); fast (`ours-fast`, the Apple FAST neural tier) on Apple"
                if "fast" in (race.get("modes") or []) else
                "identical (`ours`; the neural FAST tier is Apple only)")
            s["opponent_mode"] = ("torch at every fast setting it supports on this box, one arm "
                                  "each (eager/compile x fp32/tf32/bf16 autocast; "
                                  "tools/torch_lm_step_opponent.py COLUMNS); the arm name "
                                  "carries the setting")
            s["shape"] = race.get("shape")
            s["shape_dims"] = NEURAL.shape_text(race["lane"], race.get("shape") or "full")
            s.update(NEURAL.lane_settings(race["lane"]))
        elif race["family"] == "algos":
            s["driver"] = "tools/bench_board_algos.py"
            s["lane_config"] = ALGOS.lane_config(race["lane"])
        elif race["family"] == "classical2":
            s["driver"] = "tools/bench_board_more.py"
            s["block"] = MORE.block_of(race["lane"])
            s["lane_config"] = MORE.LANE_CONFIG[race["lane"]]
        elif race["family"] == "trees":
            s["driver"] = "bench/speed/forest_speed_arm.py"
            s["devices"] = tree_devices(ctx["vendor"], race["lane"])
            try:
                spec = _load_tool("speed_gbdt_arm")
                s["lane_config"] = spec.lane_config(race["lane"], "shipped")
            except Exception as exc:          # noqa: BLE001  (numpy absent here)
                s["lane_config"] = "tools/speed_gbdt_arm.py lane_config (%s)" % exc.__class__.__name__
        else:
            s["driver"] = "tools/classical_two_datasets.py"
            s["block"] = CLASSICAL_BLOCK[race["lane"]]
            s["shape_rule"] = ("the lane's own shape (kmeans/pca/ols 4,000,000 rows or the "
                               "Istella train split; knn 400,000 x 4,000 queries, k=64; kde "
                               "100,000 x 2,000; svc 10,000 + 10,000; dbscan 1,000,000)")
            if race["lane"] == "dbscan":
                s["dbscan_eps_min_samples"] = "eps=3, min_samples=2 (cuML benchmark DBSCAN)"
        # where this lane's values come from: an NVIDIA harness, or the board's own
        hid = ("algos/" + race["lane"]) if race["family"] == "algos" else race["lane"]
        src = HARNESS.harness_source(hid) if race["family"] != "neural" else None
        s["config"] = src or "the board's own settings (no NVIDIA harness entry)"
        if race["family"] != "neural":
            s["seed"] = _params_mod().seed_for(hid)     # 42 where cuML's benchmark sets it
        _SETTINGS_CACHE[key] = s
    return dict(_SETTINGS_CACHE[key])


def ours_of(cells, which):
    """Our GPU cell of one kind: 'identical' or 'fast'. A cell of ours on the
    CPU is never picked (Andrew, Oct 2 2026)."""
    if which not in ("identical", "fast"):
        raise ValueError("ours_of: %r: our CPU is never on a board" % (which,))
    for c in cells:
        if c["library"] != "mojolearn" or is_our_cpu_cell(c):
            continue
        if c["mode"] == which:
            return c
    return None


def add_ratios(cells):
    """ratio_ours_identical_over / ratio_ours_fast_over: our GPU median divided
    by this OPPONENT arm median, computed only when both sides completed every
    round. Our CPU is never in a ratio (Andrew, Oct 2 2026)."""
    ok = [c for c in cells if c["status"] == "ok" and c["median_ms"]]
    ours_id = ours_of(ok, "identical")
    ours_fast = ours_of(ok, "fast")
    done = {id(c) for c in ok}
    for c in cells:
        c["ratio_ours_identical_over"] = None
        c["ratio_ours_fast_over"] = None
        c.pop("ratio_ours_cpu_over", None)
        # Opponents only. FAST over IDENTICAL is the cost of identity, an
        # internal number that never reaches a board (ENGINEERING_RULES 0b-iii).
        if id(c) not in done or c["library"] == "mojolearn":
            continue
        if ours_id and c is not ours_id:
            c["ratio_ours_identical_over"] = ours_id["median_ms"] / c["median_ms"]
        if ours_fast and c is not ours_fast:
            c["ratio_ours_fast_over"] = ours_fast["median_ms"] / c["median_ms"]
    return cells


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The opponent store: keys, lookup before a race, append after it
# ---------------------------------------------------------------------------

def _machine(box):
    return ((box.get("gpu") or {}).get("name") or (box.get("host") or {}).get("cpu_model")
            or (box.get("host") or {}).get("machine"))


def _device_text(box, arm, vendor, device_name=None):
    """'gpu (NVIDIA H100 80GB HBM3)' / 'cpu (Apple M3 Ultra)': the arm's device and
    the name its own worker reported (a CPU arm: the box's CPU model)."""
    dev = arm_device(arm, vendor)
    name = device_name if dev == "gpu" else (box.get("host") or {}).get("cpu_model")
    return "%s (%s)" % (dev, name) if name else None


_R2_TO_DATA = {v: k for k, v in R2_KEYS.items()}


def race_data_files(ctx, race):
    """[(name, path)] of the data files a race reads; [] for seeded data."""
    fam, lane, ds = race["family"], race["lane"], race["dataset"]
    if fam == "neural":
        return []
    if fam == "algos":
        out = []
        for key in ALGOS.r2_keys(lane, ds):
            if key in _R2_TO_DATA:
                out.append((_R2_TO_DATA[key], os.path.join(ctx["data_root"], DATA_FILES[_R2_TO_DATA[key]])))
            else:
                out.append((key, ALGOS.corpus_path(key)))
        return out
    return [(k, os.path.join(ctx["data_root"], DATA_FILES[k])) for k in race_data_keys(race)
            if k in DATA_FILES]


def race_data_sha(ctx, race, box):
    """sha256 of what the race reads (one hash over several files); the neural
    family's inputs come from the installed wheel's sources, so its wheel; a
    seeded synthetic set is named as such. None when a file is missing."""
    if race["family"] == "neural":
        w = ((box.get("mojolearn") or {}).get("wheel") or {}).get("sha256")
        v = (box.get("mojolearn") or {}).get("version")
        return "mojolearn-sources:%s" % (w or v) if (w or v) else None
    files = race_data_files(ctx, race)
    if not files:
        return "synthetic (seed %d)" % SEED
    cache = ctx.setdefault("data_sha", {})
    shas = []
    for name, path in files:
        if name not in cache:
            cache[name] = sha256_file(path) if path and os.path.isfile(path) else None
        if cache[name] is None:
            return None
        shas.append(cache[name])
    return shas[0] if len(shas) == 1 else STORE.sha256_json(shas)


def opponent_key(box, race, arm, settings, data_sha, rounds, params=None, version=None,
                 device_name=None):
    """The store key of one opponent arm. `params` is its canonical read-back
    (BOARD-PARAMS), `version` and `device_name` what its own worker reported."""
    vendor = (box.get("gpu") or {}).get("vendor")
    return {"box": (box.get("host") or {}).get("hostname"), "machine": _machine(box),
            "vendor": vendor, "device": _device_text(box, arm, vendor, device_name),
            "os": (box.get("os") or {}).get("platform"),
            "library": arm_library(arm), "library_version": version,
            "family": race["family"], "lane": race["lane"], "dataset": race["dataset"],
            "rows": race.get("rows"), "neural_shape": race.get("shape"), "arm": arm,
            "params_sha256": STORE.sha256_json(params) if params is not None else None,
            "settings_sha256": STORE.sha256_json(settings) if settings else None,
            "data_sha256": data_sha, "rounds": rounds}


_PARAMS_ONLY_CMD = {"classical": "classical_cmd", "classical2": "more_cmd", "algos": "algos_cmd",
                    "neural": "neural_cmd"}


def params_probe(ctx, race, arms):
    """Construct `arms` (no fit, no timed round) through the race's own driver
    (--params-only) and read back what each got: {arm: {"params": canonical
    read-back, "version", "device_name"}}. An arm that did not construct is
    absent (it then runs normally)."""
    probe_dir = os.path.join(ctx["out"], "raw", "params-only", race["id"].replace("/", "."))
    os.makedirs(probe_dir, exist_ok=True)
    sub = dict(race, arms=list(arms), opponents=list(arms))
    log = os.path.join(probe_dir, "probe.log")
    out = {}
    if race["family"] == "trees":
        cmd, extra = tree_cmd(ctx, sub)
        cmd = cmd + ["--params-only"]
        run_logged(cmd, child_env(ctx, extra), log, ctx["race_deadline_s"] + 900, nice=ctx["nice"])
        try:
            with open(log, errors="replace") as fh:
                text = fh.read()
        except OSError:
            return {}
        reps = _params_mod().parse_lines(text)
        rep = reps[-1] if reps else {}
        libs = {}
        for line in text.splitlines():
            if line.startswith("FSPEED-LIBRARY {"):
                try:
                    d = json.loads(line[len("FSPEED-LIBRARY "):])
                    libs[d.get("arm")] = d
                except ValueError:
                    pass
        for arm in arms:
            a = (rep.get("arms") or {}).get(arm)
            if a is None or arm not in libs:
                continue
            out[arm] = {"params": a.get("params"), "version": libs[arm].get("version"),
                        "device_name": libs[arm].get("device_name")}
        return out
    cmd, extra, ceiling = globals()[_PARAMS_ONLY_CMD[race["family"]]](ctx, sub)
    cmd = list(cmd)
    cmd[cmd.index("--out") + 1] = probe_dir
    cmd.append("--params-only")
    for f in os.listdir(probe_dir):
        if f.endswith(".params.json"):
            os.remove(os.path.join(probe_dir, f))
    run_logged(cmd, child_env(ctx, extra), log, ceiling, nice=ctx["nice"])
    found = [f for f in os.listdir(probe_dir) if f.endswith(".params.json")]
    if not found:
        return {}
    r = load_result(os.path.join(probe_dir, found[0])) or {}
    rep = r.get("params_check") or {}
    for arm in arms:
        a = (r.get("arms") or {}).get(arm) or {}
        p = (rep.get("arms") or {}).get(arm)
        if a.get("status") not in (None, "ok", "params_only") or p is None:
            continue
        info = a.get("info") or {}
        out[arm] = {"params": p.get("params"), "version": info.get("version"),
                    "device_name": info.get("device_name")}
    return out


def successful_opponent_cell(cell):
    """Admission for missing-only runs: finite scores, warmup, no quality error."""
    def finite(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
    def quality_ok(v):
        if isinstance(v, dict):
            return "error" not in v and v.get("finite") is not False and all(quality_ok(x) for x in v.values())
        if isinstance(v, list):
            return all(quality_ok(x) for x in v)
        return not isinstance(v, (int, float)) or math.isfinite(v)
    times = cell.get("times_ms") or []
    return (cell.get("status") == "ok" and cell.get("rounds") == len(times)
            and len(times) >= 1 and all(finite(t) for t in times) and finite(cell.get("median_ms"))
            and finite(cell.get("warmup_ms")) and quality_ok(cell.get("quality")))


def successful_opponent_inference(race, arms, cells):
    sub = dict(race, arms=list(arms))
    expected = set(INFER.plan_cells(sub))
    if race["family"] == "algos" and ALGOS.has_infer(race["lane"]):
        expected = {(a, "Xq") for a in arms}
    by_key = {(c.get("arm"), c.get("batch")): c for c in cells}
    return all(k in by_key and successful_opponent_cell(by_key[k]) for k in expected)


def stored_opponents(ctx, race):
    """{arm: stored record} for the race's opponents the store holds under the
    same key, read-back included: a candidate (the fields known before
    construction match) is constructed through its driver and reused only when
    its read-back, library version and device equal the stored ones. None with
    --retime-opponents or without a store."""
    if not ctx.get("store_path") or ctx.get("retime") or not ctx.get("box"):
        return {}
    store = STORE.load(ctx["store_path"])
    if not store:
        return {}
    settings = race_settings(ctx, race)
    dsha = race_data_sha(ctx, race, ctx["box"])
    cands = [a for a in race.get("opponents") or []
             if STORE.candidates(store, opponent_key(ctx["box"], race, a, settings, dsha,
                                                     ctx["rounds"]))]
    if not cands:
        return {}
    probe = params_probe(ctx, race, cands)
    out = {}
    for arm in cands:
        pr = probe.get(arm)
        if pr is None:
            continue
        key = opponent_key(ctx["box"], race, arm, settings, dsha, ctx["rounds"],
                           params=pr.get("params"), version=pr.get("version"),
                           device_name=pr.get("device_name"))
        hit = STORE.lookup(store, key)
        if hit is not None:
            cell = hit.get("cell") or {}
            if ctx.get("retime_cpu_opponents") and cell.get("device") == "cpu" and hit.get("commit") != ctx["commit"]:
                continue
            if ctx.get("opponents_only") and (not successful_opponent_cell(cell)):
                continue
            if ctx.get("opponents_only") and ctx.get("infer") and not successful_opponent_inference(race, [arm], hit.get("infer_cells") or []):
                continue
            out[arm] = hit
    return out


def store_opponents(ctx, race, rec):
    """Append every opponent cell measured in this race to the store, keyed by
    the race's own read-back and each arm's reported version and device;
    returns how many were stored."""
    if not ctx.get("store_path") or not ctx.get("box"):
        return 0
    settings = race_settings(ctx, race)
    dsha = race_data_sha(ctx, race, ctx["box"])
    params = ((rec.get("params") or {}).get("arms") or {})
    n = 0
    for c in rec.get("cells") or []:
        if c.get("library") == "mojolearn" or c.get("stored"):
            continue
        key = opponent_key(ctx["box"], race, c["arm"], settings, dsha, ctx["rounds"],
                           params=(params.get(c["arm"]) or {}).get("params"),
                           version=c.get("library_version"), device_name=c.get("device_name"))
        if STORE.missing(key):
            c["store"] = "not stored (key fields missing: %s)" % ", ".join(STORE.missing(key))
            continue
        infer = [ic for ic in rec.get("infer_cells") or [] if ic.get("arm") == c["arm"]]
        STORE.append(ctx["store_path"], STORE.record(
            key, c, measured_at=rec.get("finished"), commit=ctx.get("commit"),
            params=params.get(c["arm"]), infer_cells=infer))
        n += 1
    return n


def _manifest_pins():
    """{R2 key: (bytes, sha256)} from bench/results/dataset_store/manifest.tsv."""
    pins = {}
    try:
        with open(os.path.join(REPO, "bench", "results", "dataset_store", "manifest.tsv")) as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) >= 3 and f[1].isdigit():
                    pins[f[0]] = (int(f[1]), f[2])
    except OSError:
        pass
    return pins


def _venv_version(python, library, started):
    """(version, why) of `library` in the board's recorded venv, read from its
    dist-info, and only when that dist-info was last written before the board
    started (the venv then still holds what the board imported)."""
    import glob
    import importlib.metadata as md
    if not python:
        return None, "no recorded venv"
    venv = os.path.dirname(os.path.dirname(python))
    sites = glob.glob(os.path.join(venv, "lib", "python*", "site-packages"))
    if not sites:
        return None, "the recorded venv %s is not on this box" % venv
    name = PROBE.IMPORT_NAME.get(library, library)
    try:
        started_ts = datetime.datetime.strptime(started, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None, "the board's start time is unknown"
    for site in sites:
        for dist in md.distributions(path=[site]):
            dname = (dist.metadata.get("Name") or "").lower()
            top = (dist.read_text("top_level.txt") or "").split()
            if dname != library.lower() and not dname.startswith(library.lower() + "-") \
                    and name not in top:
                continue
            info = getattr(dist, "_path", None)
            mtime = os.path.getmtime(str(info)) if info is not None else None
            if mtime is None:
                return None, "the dist-info of %s has no time" % dname
            if mtime >= started_ts:
                return None, "%s changed in the venv after the board started" % dname
            return dist.version, None
    return None, "%s is not in the recorded venv" % library


def backfill_store(board_json, store_path):
    """Import the opponent cells of an existing board.json into the store, the
    key and provenance from the board's box, config and each cell's settings,
    read-back and reported version. Returns (imported, {reason: skipped})."""
    res = load_result(board_json)
    if res is None:
        raise SystemExit("bench_board: cannot read %s" % board_json)
    box = res.get("box") or {}
    cfg = res.get("config") or {}
    data = cfg.get("data") or {}
    shas = dict(cfg.get("data_sha256") or {})
    for name, d in data.items():         # a pinned file whose size matched its pin
        if name not in shas and d.get("size_ok") and d.get("pinned_sha256"):
            shas[name] = d["pinned_sha256"]
    pins = _manifest_pins()
    ctx = {"data_root": cfg.get("data_root") or "", "data_sha": shas}
    started = res.get("created")
    python = (box.get("python") or {}).get("executable")
    imported, skipped = 0, {}

    def skip(why):
        skipped[why] = skipped.get(why, 0) + 1

    for rid, rr in sorted((res.get("races") or {}).items()):
        race = {"family": rr.get("family"), "lane": rr.get("lane"), "dataset": rr.get("dataset"),
                "rows": rr.get("rows"), "shape": rr.get("shape")}
        if not race["family"]:
            continue
        dsha, why_data = None, None
        try:
            files = race_data_files(ctx, race)
        except Exception as exc:        # noqa: BLE001
            files, why_data = None, "the race's data files cannot be named (%s)" % exc
        if files is not None:
            for name, path in files:
                if name in shas:
                    continue
                pin = pins.get(name)      # a corpus key: its pin, when the staged file matches
                if pin and path and os.path.isfile(path) and os.path.getsize(path) == pin[0]:
                    shas[name] = pin[1]
            missing = [n for n, _ in files if n not in shas]
            if missing:
                why_data = "no sha256 for %s (not recorded, and no pinned file of the pinned size here)" \
                    % ", ".join(missing)
            else:
                dsha = race_data_sha(ctx, race, box)
        params = ((rr.get("params") or {}).get("arms") or {})
        for c in rr.get("cells") or []:
            if c.get("library") == "mojolearn" or c.get("stored"):
                continue
            if why_data:
                skip(why_data)
                continue
            version = c.get("library_version")
            if not version:
                version, why = _venv_version(python, c.get("library"), started)
                if not version:
                    skip("library version: " + why)
                    continue
            key = opponent_key(box, race, c.get("arm"), c.get("settings"), dsha,
                               (c.get("settings") or {}).get("rounds") or cfg.get("rounds"),
                               params=(params.get(c.get("arm")) or {}).get("params"),
                               version=version, device_name=c.get("device_name"))
            miss = STORE.missing(key)
            if miss:
                skip("key fields missing: " + ", ".join(miss))
                continue
            infer = [ic for ic in rr.get("infer_cells") or [] if ic.get("arm") == c.get("arm")]
            STORE.append(store_path, STORE.record(
                key, c, measured_at=rr.get("finished"), commit=(box.get("repo") or {}).get("commit"),
                params=params.get(c.get("arm")), infer_cells=infer))
            imported += 1
    return imported, skipped


def note_host_memory(rec, kills, arms):
    """The host-memory watchdog killed something of this race: the race fails
    by name (HOST MEMORY) and the killed arm's cell says so."""
    if not kills:
        return
    rec["host_memory"] = list(kills)
    named = []
    for k in kills:
        arm = WATCHDOG.arm_of(k.get("command") or "", arms)
        named.append("%s at %.1f GB (%s)" % (arm or "the driver", k["mb"] / 1024.0, k["why"]))
        for c in rec.get("cells") or []:
            if arm is not None and c.get("arm") == arm:
                c["status"] = "HOST-MEMORY(killed at %.1f GB: %s)" % (k["mb"] / 1024.0, k["why"])
    rec["status"] = "failed"
    rec["failure"] = "HOST MEMORY: the watchdog killed " + "; ".join(named)
    print("bench_board:   %s" % rec["failure"], flush=True)


def run_race(ctx, race):
    """Run one race and return its record (status, rc, log, cells). Opponents
    the store already holds for this key are not run; their stored cells join
    the race (tools/bench_board_store.py). Default is our arm(s) only: an
    opponent the store does not hold is skipped unless ctx["with_opponents"]
    (--with-opponents / --retime-opponents)."""
    stored = stored_opponents(ctx, race)
    skipped = [] if (ctx.get("with_opponents") or ctx.get("retime")) else [
        a for a in race.get("opponents") or [] if a not in stored]
    full = race
    if stored or skipped:
        drop = set(stored) | set(skipped)
        race = dict(race, opponents=[a for a in race["opponents"] if a not in drop],
                    arms=[a for a in race["arms"] if a not in drop])
    if stored:
        print("bench_board:   stored (not run): %s" % ", ".join(
            "%s [%s]" % (a, STORE.source_text(r)) for a, r in sorted(stored.items())), flush=True)
    if skipped:
        print("bench_board:   opponents not stored, not raced (ours-only default; "
              "--with-opponents races them): %s" % ", ".join(skipped), flush=True)
    unsupported = unsupported_our_arms(race)
    declared_race = race
    if unsupported:
        race = dict(race, arms=[a for a in race["arms"] if a not in unsupported],
                    our_arms={a: m for a, m in race["our_arms"].items() if a not in unsupported})
    mark = len(HOST_MEMORY_KILLS)
    if ctx.get("artifact_identity"):
        import tempfile
        ctx = dict(ctx)
        root = os.path.join(ctx["out"], "provenance")
        os.makedirs(root, exist_ok=True)
        ctx["receipt_dir"] = tempfile.mkdtemp(prefix="race-", dir=root)
    if race["arms"]:
        rec = _run_race(ctx, race)
    else:
        rec = dict(id=race["id"], family=race["family"], lane=race["lane"], dataset=race["dataset"],
                   rows=race["rows"], our_arms=declared_race["our_arms"], cells=[],
                   started=now_utc(), finished=now_utc(), status="unsupported", rc=0,
                   params_check="NOT APPLICABLE (public API unsupported)")
    if unsupported:
        rec["unsupported_arms"] = unsupported
        for arm, reason in unsupported.items():
            cell = base_cell(ctx, declared_race, arm, declared_race["our_arms"][arm])
            cell["status"] = reason
            rec["cells"].append(cell)
    if ctx.get("artifact_identity"):
        try:
            rows = _load_tool("bench_board_provenance").read_receipts(
                ctx["receipt_dir"], ctx["artifact_identity"], race["our_arms"])
            if any(row.get("hardware") != ctx["artifact_hardware"] for row in rows):
                raise ValueError("Worker ran on a different physical GPU or driver")
            rec["worker_provenance"] = rows
        except (OSError, ValueError) as exc:
            rec.update(status="failed", failure="WORKER PROVENANCE: " + str(exc))
            for cell in rec.get("cells", []) + (rec.get("infer_cells") or []):
                if cell.get("arm") in race["our_arms"]:
                    _refuse_cell(cell, "WORKER PROVENANCE: " + str(exc))
        rec["provenance_dir"] = os.path.relpath(ctx["receipt_dir"], ctx["out"])
    note_host_memory(rec, HOST_MEMORY_KILLS[mark:], race["arms"])
    if ctx.get("opponents_only") and any(c.get("library") == "mojolearn" or str(c.get("arm", "")).startswith("ours") for c in rec["cells"]):
        raise RuntimeError("opponents-only safety violation: own arm appeared")
    if ctx.get("opponents_only") and not race["arms"]:
        rec["status"] = "done"
        rec["no_execution"] = True
    rec["arms"] = [a for a in full["arms"] if a not in skipped]
    rec["stored_arms"] = sorted(stored)
    rec["skipped_opponents"] = list(skipped)
    for c in rec["cells"]:
        if c.get("library") != "mojolearn":
            c["source"] = "measured this run"
    if ctx.get("infer"):
        rec.setdefault("infer_cells", [])
    for arm, r in sorted(stored.items()):
        rec["cells"].append(STORE.stored_cell(r))
        if r.get("infer_cells") and rec.get("infer_cells") is not None:
            for ic in r["infer_cells"]:
                ic = dict(ic, source=STORE.source_text(r))
                rec["infer_cells"].append(ic)
    rec["cells"] = add_ratios(rec["cells"])
    if ctx.get("opponents_only"):
        cells = {c["arm"]: c for c in rec["cells"]}
        rec["status"] = "done" if rec["rc"] == 0 and all(successful_opponent_cell(cells.get(a, {})) for a in full["opponents"]) and (not ctx.get("infer") or successful_opponent_inference(full, full["opponents"], rec.get("infer_cells") or [])) else "failed"
    rec["stored_now"] = store_opponents(ctx, full, rec)
    return rec


def _run_race(ctx, race):
    """Run one race and return its record (status, rc, log, cells)."""
    rec = {"id": race["id"], "family": race["family"], "lane": race["lane"],
           "dataset": race["dataset"], "rows": race["rows"], "arms": race["arms"],
           "our_arms": race["our_arms"], "started": now_utc()}
    tag = "%s.%s.rows-%s" % (race["lane"], race["dataset"], rows_tag(race["rows"]))
    if race["family"] == "trees":
        cmd, extra = tree_cmd(ctx, race)
        log = os.path.join(ctx["out"], "raw", "trees", tag + ".log")
        if os.path.exists(log):
            os.replace(log, log + ".previous")
        rc = run_logged(cmd, child_env(ctx, extra), log, ctx["race_deadline_s"] + 900,
                        nice=ctx["nice"])
        rec.update(command=cmd, env=extra, log=os.path.relpath(log, ctx["out"]), rc=rc)
        parsed = parse_tree_log(log)
        rec["notes"] = parsed["notes"]
        rec["fit_verdict_line"] = parsed["verdict_line"]
        cells = tree_cells(ctx, race, parsed)
        if ctx.get("infer"):
            rec["infer_cells"] = INFER.tree_cells(_bb(), ctx, race, log, cells)
    elif race["family"] == "neural":
        cmd, extra, ceiling = neural_cmd(ctx, race)
        tag = "%s.%s.shape-%s" % (race["lane"], race["dataset"], race.get("shape") or "full")
        log = os.path.join(ctx["out"], "logs", "neural." + tag + ".log")
        jpath = neural_json_path(ctx, race)
        if os.path.exists(jpath):
            os.replace(jpath, jpath + ".previous")
        rc = run_logged(cmd, child_env(ctx, extra), log, ceiling, nice=ctx["nice"])
        rec.update(command=cmd, env=extra, log=os.path.relpath(log, ctx["out"]), rc=rc,
                   race_json=os.path.relpath(jpath, ctx["out"]), shape=race.get("shape"))
        r = load_result(jpath) if os.path.exists(jpath) else None
        if r is None:
            cells = [dict(base_cell(ctx, race, a, race["our_arms"].get(a)),
                          status="UNKNOWN(no race json, rc %d)" % rc) for a in race["arms"]]
        else:
            rec["inputs"] = r.get("inputs")
            cells = classical_cells(ctx, race, r)
    elif race["family"] == "algos":
        cmd, extra, ceiling = algos_cmd(ctx, race)
        log = os.path.join(ctx["out"], "logs", "algos." + tag + ".log")
        jpath = algos_json_path(ctx, race)
        if os.path.exists(jpath):
            os.replace(jpath, jpath + ".previous")
        rc = run_logged(cmd, child_env(ctx, extra), log, ceiling, nice=ctx["nice"])
        rec.update(command=cmd, env=extra, log=os.path.relpath(log, ctx["out"]), rc=rc,
                   race_json=os.path.relpath(jpath, ctx["out"]))
        r = load_result(jpath) if os.path.exists(jpath) else None
        if r is None:
            cells = [dict(base_cell(ctx, race, a, race["our_arms"].get(a)),
                          status="UNKNOWN(no race json, rc %d)" % rc) for a in race["arms"]]
        else:
            rec["lane_config"] = r.get("lane_config")
            cells = algos_skips(classical_cells(ctx, race, r), r)
            if ctx.get("infer"):
                rec["infer_cells"] = algos_skips(INFER.classical_cells(_bb(), ctx, race, r), r)
    elif race["family"] == "classical2":
        cmd, extra, ceiling = more_cmd(ctx, race)
        log = os.path.join(ctx["out"], "logs", "classical2." + tag + ".log")
        jpath = more_json_path(ctx, race)
        if os.path.exists(jpath):
            os.replace(jpath, jpath + ".previous")
        rc = run_logged(cmd, child_env(ctx, extra), log, ceiling, nice=ctx["nice"])
        rec.update(command=cmd, env=extra, log=os.path.relpath(log, ctx["out"]), rc=rc,
                   race_json=os.path.relpath(jpath, ctx["out"]))
        r = load_result(jpath) if os.path.exists(jpath) else None
        if r is None:
            cells = [dict(base_cell(ctx, race, a, race["our_arms"].get(a)),
                          status="UNKNOWN(no race json, rc %d)" % rc) for a in race["arms"]]
        else:
            rec["lane_config"] = r.get("lane_config")
            cells = classical_cells(ctx, race, r)
    else:
        cmd, extra, ceiling = classical_cmd(ctx, race)
        log = os.path.join(ctx["out"], "logs", "classical." + tag + ".log")
        jpath = classical_json_path(ctx, race)
        if os.path.exists(jpath):
            os.replace(jpath, jpath + ".previous")
        rc = run_logged(cmd, child_env(ctx, extra), log, ceiling, nice=ctx["nice"])
        rec.update(command=cmd, env=extra, log=os.path.relpath(log, ctx["out"]), rc=rc,
                   race_json=os.path.relpath(jpath, ctx["out"]))
        r = load_result(jpath) if os.path.exists(jpath) else None
        if r is None:
            cells = [dict(base_cell(ctx, race, a, race["our_arms"].get(a)),
                          status="UNKNOWN(no race json, rc %d)" % rc) for a in race["arms"]]
        else:
            cells = classical_cells(ctx, race, r)
            # the classical driver's settings and mismatch lines reach BOARD.md
            rec["lane_config"] = r.get("lane_config")
            if ctx.get("infer"):
                rec["infer_cells"] = INFER.classical_cells(_bb(), ctx, race, r)
        # the arms' saved outputs are only for the conductor's quality pass
        work = os.path.join(ctx["out"], "work")
        if os.path.isdir(work):
            for f in os.listdir(work):
                if f.startswith("%s-%s-" % (race["lane"], race["dataset"])):
                    try:
                        os.remove(os.path.join(work, f))
                    except OSError:
                        pass
    rec["cells"] = add_ratios(cells)
    rec["finished"] = now_utc()
    rec["status"] = "done" if rc == 0 else "failed"
    rec["host"] = race_host()
    attach_params(ctx, rec)
    return rec


def _params_mod():
    spec = importlib.util.spec_from_file_location("bench_board_params",
                                                  os.path.join(HERE, "bench_board_params.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def attach_params(ctx, rec):
    """tools/bench_board_params.py: the driver printed one BOARD-PARAMS line
    per race (every arm's parameters, read back from the constructed object).
    The report goes on the race and each arm's resolved parameters on its
    cell. A REFUSED report fails the race by name, whatever the driver's rc.
    No line means the driver ran without the check: `params_check` says so."""
    try:
        with open(os.path.join(ctx["out"], rec["log"]), errors="replace") as fh:
            reports = _params_mod().parse_lines(fh.read())
    except (OSError, KeyError, TypeError):
        reports = []
    if not reports:
        rec["params_check"] = "NOT CHECKED (the driver printed no BOARD-PARAMS line)"
        return
    rep = reports[-1]
    rec["params"] = rep
    rec["params_check"] = rep.get("verdict")
    for c in rec.get("cells") or []:
        arm = rep.get("arms", {}).get(c.get("arm"))
        if arm is not None:
            c["params"] = arm
    if rep.get("verdict") == "REFUSED":
        rec["status"] = "failed"
        rec["failure"] = "PARAMS REFUSED: " + " | ".join(rep.get("problems") or [])


def render_params(rr):
    """The resolved parameters of every arm side by side, under the race."""
    L = [""]
    rep = rr.get("params")
    if not rep:
        L.append("parameters: %s" % clean(rr.get("params_check") or "NOT CHECKED"))
        return L
    arms = list(rep.get("arms") or {})
    names = sorted({p for a in arms for p in rep["arms"][a].get("params", {})})
    L.append("parameters (tools/bench_board_params.py, read back from each constructed arm; "
             "reference `%s`, seed %s): %s" % (rep.get("reference"), rep.get("seed"),
                                                clean(rep.get("verdict"))))
    L.append("")
    L.append("| parameter | %s |" % " | ".join(arms))
    L.append("|---|%s" % "---|" * len(arms))
    L.append("| library (source) | %s |" % " | ".join(
        "%s (%s)" % (rep["arms"][a].get("library"), rep["arms"][a].get("source")) for a in arms))
    for n in names:
        L.append("| %s | %s |" % (n, " | ".join(
            clean(json.dumps(rep["arms"][a]["params"][n])) if n in rep["arms"][a].get("params", {})
            else "-" for a in arms)))
    for e in rep.get("exceptions") or []:
        L.append("")
        L.append("accepted difference: %s %s: %s" % (e.get("arm"), e.get("param"), clean(e.get("reason"))))
    for pr in rep.get("problems") or []:
        L.append("")
        L.append("REFUSED: %s" % clean(pr))
    return L


def all_cells(result):
    out = []
    for rid in sorted(result.get("races", {})):
        out.extend(result["races"][rid].get("cells", []))
    return out


# ---------------------------------------------------------------------------
# The board (Markdown). Numbers and ratios only; no direction words.
# ---------------------------------------------------------------------------

_BANNED = re.compile(r"\b(faster|slower)\b", re.I)


def clean(text):
    """Driver-sourced free text (a refusal reason, a note) goes through here:
    the board states no direction, whoever wrote the sentence."""
    s = " ".join(str(text).split())
    s = s.replace("|", "/")
    return _BANNED.sub(lambda m: "[direction word removed]", s)


def _f(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if v != 0 and (abs(v) >= 1e6 or abs(v) < 1e-3):
            return "%.4g" % v
        return "%.*f" % (nd, v)
    return clean(v)


def _q(q):
    if not q:
        return "-"
    return ", ".join("%s=%s" % (k, _f(v, 6) if isinstance(v, float) else clean(v))
                     for k, v in sorted(q.items()))


def _arm_label(c):
    if c["library"] == "mojolearn":
        return "mojolearn %s" % (c["mode"] or "?").upper()
    return c["arm"]


QUALITY_NOTE = {
    "auc": "higher is better", "accuracy": "higher is better", "logloss": "lower is better",
    "rmse": "lower is better", "r2": "higher is better", "inertia": "lower is better",
    "explained_variance_ratio_sum": "higher is better", "recall_at_10": "higher is better", "recall_at_k": "higher is better",
    "mean_log_likelihood": "higher is better",
    "ndcg10": "higher is better", "ndcg5": "higher is better", "map": "higher is better",
    "mlogloss": "lower is better",
    "loss_first_step": "same init and batches on every arm",
    "loss_last_step": "same init and batches on every arm",
    "loss_last_abs_diff_vs_ours": "0 is our value exactly",
    "loss_first_abs_diff_vs_ours": "0 is our value exactly",
    "max_rel_diff_vs_ours": "0 is our output exactly",
    "mean_nll": "lower is better", "max_abs_diff_vs_ours": "0 is our output exactly",
    "max_rel_err_vs_fp64": "lower is better",
    "trustworthiness_k15": "higher is better, 1 at most",
    "bic": "lower is better", "silhouette": "higher is better",
    "ari_vs_ours": "1 is our partition exactly",
    "relative_reconstruction_error": "lower is better",
    "kernel_rel_error": "lower is better",
    "mean_log_predictive_density": "higher is better",
    "forecast_rmse": "lower is better", "insample_rmse": "lower is better",
    "mean_llf": "higher is better", "mean_aic": "lower is better",
    "bits_equal_vs_ours_identical": "yes is the product's promise",
}


#: The clock columns beside the stored ratios (tools/board_clock_audit.py).
CLOCK_HEADER = ("whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | "
                "ours FAST / arm (clock)")
CLOCK_COLUMNS = 5


def _clock_ratio(r):
    if not r or r.get("value") is None:
        return "-"
    return "%s (%s)" % (_f(r["value"], 3), clean(r["label"]))


def clock_cells(c):
    """The five clock cells of one board row, joined with ` | ` (the caller's
    format string supplies the outer bars). Derived by CLOCKS.annotate_cells;
    a cell rendered without it shows dashes."""
    k = c.get("clock") or {}
    copy_txt = "-"
    if k.get("copy_ms") is not None:
        copy_txt = "%s (%s)" % (_f(k["copy_ms"], 2), clean(k.get("copy_source")))
    elif k.get("stored") in ("whole", "kernel") and k.get("median_ms") is not None:
        copy_txt = "- (stored %s)" % k["stored"]
    return " | ".join((_f(k.get("whole_ms")), _f(k.get("kernel_ms")), copy_txt,
                       _clock_ratio(c.get("ratio_ours_identical_clock")),
                       _clock_ratio(c.get("ratio_ours_fast_clock"))))


def render_board(result):
    result = copy.deepcopy(result)
    # Historical page-only races are measurements too; include them in the main board.
    result.setdefault("races", {}).update(result.get("extra_races") or {})
    result = strip_our_cpu(result)
    CLOCKS.annotate_result(result)      # derived `clock` + ratio-clock fields, this copy only
    box = result.get("box") or {}
    cfg = result.get("config") or {}
    gpu = box.get("gpu") or {}
    mj = box.get("mojolearn") or {}
    w = mj.get("wheel") or {}
    L = []
    L.append("# mojolearn benchmark board")
    L.append("")
    L.append("Generated %s from `board.json` (schema `%s`)." % (result.get("updated") or now_utc(), SCHEMA))
    L.append("")
    if cfg.get("smoke"):
        why = []
        if cfg.get("rows"):
            why.append("`--rows %s` is below the 1,000,000-row tree floor or the classical lane "
                       "shapes" % cfg.get("rows"))
        if cfg.get("neural_shape") == "small" and "neural" in (cfg.get("families") or []):
            why.append("`--neural-shape small` is a plumbing shape")
        L.append("> SMOKE RUN: %s. These numbers are plumbing checks, not results."
                 % ("; ".join(why) or "a reduced shape"))
        L.append("")
    # Board-level header notes a result assembler sets (tools/main_board_ingest.py: the
    # rolling main board's label, cell span and rules). Release boards carry none.
    for note in result.get("board_notes") or []:
        L.extend(["> " + clean(note), ""])
    for note in result.get("opponent_resource_notes", []):
        L.extend(["> " + note, ""])
    L.append("## Box")
    L.append("")
    L.append("| field | value |")
    L.append("|---|---|")
    host = box.get("host") or {}
    rows = [
        ("vendor / API", "%s / %s" % (gpu.get("vendor"), gpu.get("api"))),
        ("GPU", gpu.get("name")), ("GPU driver", gpu.get("driver")),
        ("CPU", "%s (%s logical cores)" % (host.get("cpu_model"), host.get("cpu_count"))),
        ("memory bytes", host.get("memory_bytes")),
        ("OS", (box.get("os") or {}).get("name")),
        ("Python", (box.get("python") or {}).get("version")),
        ("mojolearn", "%s (wheel %s, sha256 %s)" % (mj.get("version"), w.get("file"), w.get("sha256"))),
        ("script commit", (box.get("repo") or {}).get("commit")),
        ("patch sync", ("synced commit %s over base %s, patch sha256 %s" % (
            sy.get("commit"), sy.get("base"), sy.get("patch_sha256")))
         if (sy := (box.get("repo") or {}).get("sync")) else "-"),
        ("modes", ", ".join(cfg.get("modes") or [])),
        ("rounds", "%s timed after 1 warm-up, arms interleaved round by round" % cfg.get("rounds")),
        ("seed", SEED),
    ]
    for k, v in rows:
        L.append("| %s | %s |" % (k, _f(v)))
    pk = box.get("packages") or {}
    opp = ["catboost", "xgboost", "lightgbm", "scikit-learn", "cuml-cu12", "cuvs-cu12", "torch",
           "umap-learn", "pynndescent", "numba", "statsmodels", "faiss-cpu", "numpy"]
    L.append("| opponent versions | %s |" % ", ".join(
        "%s %s" % (p, pk[p]) for p in opp if p in pk) or "-")
    L.append("")
    L.append("## How to read this board")
    L.append("")
    L.append("- Times are wall milliseconds of the public fit call (trees) or the lane's timed "
             "call (classical), median of the timed rounds; min..max beside it.")
    L.append("- `ours IDENTICAL / arm` is our IDENTICAL median divided by that opponent's median; "
             "`ours FAST / arm` likewise. Below 1.0 our median time is the lower one, above 1.0 "
             "the higher one. A ratio is shown only when both arms completed every round in this run, "
             "and only against an opponent: our two modes are never divided by each other here.")
    L.append("- Two clocks (AGENTS.md measurement item 6): `whole ms` is the operation including the "
             "host-to-device copy of its inputs, `kernel ms` the same with the inputs already on the "
             "device; `copy ms` comes only from a stored field, named beside it (`upload_ms_separate`: "
             "our separate upload probe, kernel = median - copy; `upload_ms_untimed`: an opponent's "
             "pre-clock upload, whole = median + copy; `cpu-arm`: no device copy exists). A clock the "
             "stored fields cannot give is `-`, never estimated. `ours IDENTICAL / arm (clock)` reads "
             "a torch GPU arm on kernel/kernel and every other arm on whole/whole; when that clock is "
             "missing on a side it falls back to the other common clock (labelled), and with no "
             "common clock it is the two stored medians labelled MIXED with each side's clock.")
    L.append("- Quality comes from the drivers: FSPEED-ACC for trees (held-out rows), one float64 "
             "NumPy function per lane for classical.")
    L.append("- Comparability: trees carry FSPEED-FIT-VERDICT (total leaves within 10% across "
             "arms is COMPARABLE); classical carry the clock span (SPAN-ASYMMETRIC names an arm "
             "whose clock excludes an upload or a fit that ours includes).")
    L.append("- Classical, wave 2 (`classical2`, tools/bench_board_more.py): the same worker "
             "protocol as classical; every lane's parameters, rows, timed span and each "
             "unavoidable mismatch with its reason are in the cells' `settings.lane_config`. "
             "Quality is one float64 NumPy function per lane over each arm's saved outputs.")
    L.append("- Neural: our IDENTICAL arm only (the neural surface builds no other tier, on any "
             "vendor) against torch at every fast setting it supports on this box, one arm each, "
             "the setting in the arm name: `torch-eager-fp32` (TF32 off), `torch-compile-fp32` "
             "(torch.compile, inductor), `torch-eager-tf32` / `torch-compile-tf32` (NVIDIA CUDA "
             "only), `torch-eager-bf16` / `torch-compile-bf16` (bf16 autocast mixed precision). "
             "TF32 and bf16 arms are ANOTHER PRECISION than ours; their quality columns show how "
             "far. An arm torch cannot run on this box is REFUSED by name in its cell. Every "
             "clock is host in, host out, synchronized. Every arm starts from the same "
             "parameters and reads the same inputs, so losses and outputs are comparable; "
             "`max_abs_diff_vs_ours` / `max_rel_diff_vs_ours` are the arm's output against ours.")
    L.append("- `installed_wheel` confirms our binding loaded from site-packages, not the repo tree.")
    L.append("- Our CPU is never raced or reported: the board races only our GPU, against GPU "
             "opponents; a race keeps CPU opponents only when it has no GPU opponent (Andrew, "
             "Oct 2 2026). A cell of ours on the CPU in an old record is dropped before rendering.")
    L.append("- Memory: `peak host MB` and `peak GPU MB` are the highest per-round peaks over the "
             "timed rounds, read outside the clock; each arm's method is listed under its table "
             "(host: the resettable peak RSS on Linux, the peak physical footprint on macOS, which "
             "holds Metal buffers too; GPU: torch's own counter for torch arms, the driver's "
             "per-process figure for the rest, none on Apple).")
    L.append("- Inference: after a race's fit rounds each arm predicts with its own fitted model "
             "(no fit retimed), same rows, same output kind, one warm-up then the timed rounds "
             "interleaved. Trees: batch `test` (the held-out split) and `large` (1,000,000 "
             "training rows, capped at the training rows), host rows in and host predictions "
             "out on every arm; each arm's call is printed under its table. Classical: kmeans "
             "predict, pca transform, ols predict and svc predict on the eval rows, with the "
             "fit's clock span. Ratios are per batch, ours over each opponent.")
    L.append("")
    races = result.get("races") or {}
    planned = result.get("plan") or []
    done = sum(1 for r in races.values() if r.get("status") == "done")
    failed = sum(1 for r in races.values() if r.get("status") == "failed")
    unsupported = sum(1 for r in races.values() if r.get("status") == "unsupported")
    cells = all_cells(result)
    st = {}
    for c in cells:
        k = c["status"].split("(")[0]
        st[k] = st.get(k, 0) + 1
    L.append("## Coverage")
    L.append("")
    L.append("Races: %d planned, %d done, %d failed, %d unsupported, %d pending. Cells: %d (%s)."
             % (len(planned), done, failed, unsupported, max(0, len(planned) - done - failed - unsupported), len(cells),
                ", ".join("%s %d" % kv for kv in sorted(st.items())) or "none"))
    icov = INFER.coverage(races)
    if icov:
        L.append("")
        L.append(icov)
    L.append("")

    # Quality at a glance
    L.append("## Quality at a glance")
    L.append("")
    L.append("Per lane and dataset: our FAST value, our IDENTICAL value, and each opponent's.")
    L.append("")
    L.append("| family | lane | dataset | metric | ours FAST | ours IDENTICAL | opponents |")
    L.append("|---|---|---|---|---|---|---|")
    for rid in sorted(races):
        rc = races[rid].get("cells") or []
        metrics = []
        for c in rc:
            for m, v in (c.get("quality") or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and m not in metrics:
                    metrics.append(m)
        for m in metrics:
            fast = ours_of(rc, "fast")
            ident = ours_of(rc, "identical")
            opps = [c for c in rc if c["library"] != "mojolearn"]
            q = lambda c: _f((c.get("quality") or {}).get(m), 6) if c else "-"   # noqa: E731
            L.append("| %s | %s | %s | %s%s | %s | %s | %s |" % (
                races[rid]["family"], races[rid]["lane"], races[rid]["dataset"], m,
                (" (%s)" % QUALITY_NOTE[m]) if m in QUALITY_NOTE else "",
                q(fast), q(ident),
                "; ".join("%s %s" % (c["arm"], q(c)) for c in opps) or "-"))
    L.append("")
    L.extend(INFER.render_glance(_bb(), races))

    for fam in FAMILIES:
        fam_races = [races[r] for r in sorted(races) if races[r]["family"] == fam]
        if not fam_races:
            continue
        L.append("## %s" % {"trees": "Trees", "classical": "Classical",
                             "classical2": "Classical, wave 2", "neural": "Neural",
                             "algos": "Algorithm expansion"}[fam])
        L.append("")
        for rr in fam_races:
            rc = rr.get("cells") or []
            shape = next((c.get("shape") for c in rc if c.get("shape")), None)
            if fam == "neural":
                L.append("### %s / %s (neural shape %s: %s)" % (
                    rr["lane"], rr["dataset"], _f(rr.get("shape") or "full"), _f(shape)))
            else:
                L.append("### %s / %s (rows %s, shape %s)" % (rr["lane"], rr["dataset"],
                                                             rows_tag(rr["rows"]) if "rows" in rr else "unrecorded", _f(shape)))
            L.append("")
            rh = rr.get("host") or {"hostname": (box.get("host") or {}).get("hostname")}
            L.append("race: %s, driver rc %s, log `%s`, ran on %s%s" % (
                rr.get("status"), rr.get("rc"), rr.get("log"), _f(rh.get("hostname")),
                " (pod %s)" % rh["pod_id"] if rh.get("pod_id") else ""))
            L.append("")
            L.append("| arm | library | device | mode | median ms | min..max ms | rounds | "
                     "ours IDENTICAL / arm | ours FAST / arm | " + CLOCK_HEADER + " | peak host MB | "
                     "peak GPU MB | quality | hash stable | comparability | installed_wheel | "
                     "status |")
            L.append("|---|---|---|---|---|---|---|---|---|" + "---|" * CLOCK_COLUMNS
                     + "---|---|---|---|---|---|---|")
            for c in rc:
                L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | "
                         "%s | %s | %s |" % (
                             _arm_label(c), c["library"], c["device"], c["mode"],
                             _f(c["median_ms"]),
                             "%s..%s" % (_f(c["min_ms"]), _f(c["max_ms"]))
                             if c["min_ms"] is not None else "-",
                             c["rounds"],
                             _f(c.get("ratio_ours_identical_over"), 3),
                             _f(c.get("ratio_ours_fast_over"), 3),
                             clock_cells(c),
                             _f(c.get("peak_host_mb")), _f(c.get("peak_gpu_mb")),
                             _q(c.get("quality")), _f(c.get("hash_stable")),
                             clean(c.get("verdict")), clean(c.get("installed_wheel", "-")),
                             clean(c["status"]) + (" (%s)" % clean(c["source"])
                                                   if c.get("source") else "")))
            L.extend(render_memory_methods(rc))
            lc = rr.get("lane_config") or {}
            if lc:
                L.append("")
                L.append("settings: %s. Rows%s: %s. Timed: %s." % (
                    clean(lc.get("params")),
                    " (the full board; this run caps them at --rows %s)" % rr["rows"] if rr.get("rows") else "",
                    clean(lc.get("rows")), clean(lc.get("timed"))))
                for mm in lc.get("mismatches") or []:
                    L.append("")
                    L.append("mismatch: %s" % clean(mm))
            if rr.get("fit_verdict_line"):
                L.append("")
                L.append("FSPEED-FIT-VERDICT: `%s`" % clean(rr["fit_verdict_line"]))
            src = next((c.get("settings", {}).get("config") for c in rr.get("cells") or []
                        if c.get("settings", {}).get("config")), None)
            if src:
                L.append("")
                L.append("config: %s" % clean(src if isinstance(src, str) else
                                              "%s, %s (%s)" % (src.get("harness"), src.get("entry"),
                                                               src.get("url"))))
            L.extend(render_params(rr))
            L.extend(INFER.render_race(_bb(), rr))
            L.append("")
    L.append("## Not covered by this board")
    L.append("")
    vendor = (gpu.get("vendor") or cfg.get("vendor"))
    L.append("- Classical, wave 2: RadiusNeighbors, the preprocessing scalers, HDBSCAN's "
             "prediction data, Cholesky and the parallel_* and Distributed* wrappers are "
             "public and not raced here; taxi-derived time series are not used (the ARIMA "
             "and ExponentialSmoothing lanes fit seeded synthetic series, as the repo's own "
             "ARIMA quality work does).")
    for why in MORE.NOT_PLANNED.get(vendor, []):
        L.append("- Classical, wave 2, not planned on this vendor: %s" % clean(why))
    for why in INFER.NOT_COVERED:
        L.append("- %s" % clean(why))
    for why in cpu_not_covered(cfg):
        L.append("- %s" % clean(why))
    for why in NEURAL.NOT_COVERED:
        L.append("- Neural: %s" % clean(why))
    for why in NEURAL.NOT_PLANNED.get(vendor, []):
        L.append("- Neural, not planned on this vendor: %s" % clean(why))
    L.append("")
    return "\n".join(L) + "\n"


def render_memory_methods(cells):
    """One line per distinct (host method, GPU method) pair, naming the arms."""
    groups = {}
    for c in cells:
        m = c.get("memory") or {}
        key = (m.get("host_method") or "not sampled", m.get("gpu_method") or "not sampled")
        groups.setdefault(key, []).append(c["arm"])
    if not groups or set(groups) == {("not sampled", "not sampled")}:
        return []
    out = []
    for (host, gpu), arms in groups.items():
        out.append("")
        out.append("memory, %s: host %s; GPU %s" % (", ".join(arms), clean(host), clean(gpu)))
    return out


def cpu_not_covered(cfg):
    """The board's 'Not covered' lines for the CPU arm and for memory."""
    out = []
    out.append("Our CPU: never raced or reported; the board races only our GPU (Andrew, Oct 2 "
               "2026). The host column gives same-bits digests only (lq ID).")
    lanes = sorted(l for l in NEURAL_LANES if NEURAL.DEVICE_OF.get(l) == "cpu")
    if lanes:
        out.append("Neural, not planned: %s: ours runs the CPU binding, and our CPU is never "
                   "raced, in no numeric mode (the Apple FAST neural tier is the GPU lanes "
                   "only)." % ", ".join(lanes))
    out.append("Memory: GPU memory on Apple has no per-process counter (Metal buffers are inside "
               "the host footprint); the trees driver runs every arm in one process, so its GPU "
               "figure is the process total; a figure taken at the round's end misses a buffer "
               "freed inside the round; inference cells carry memory only on the classical lanes.")
    return out


def write_board(out, result):
    with open(os.path.join(out, "BOARD.md"), "w") as fh:
        fh.write(render_board(result))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The smoke check: every planned race, small, must produce a time, a quality
# and a MATCHED parameter check on every arm before a full board may start
# ---------------------------------------------------------------------------

SMOKE_SCHEMA = "mojolearn-bench-board-smoke/1"
#: the per-step bound in a smoke run (warm-up, round, trees arm budget), seconds
SMOKE_STEP_S = 300
SMOKE_ROWS = 2000
#: The board and driver files a smoke result vouches for (with the vendor).
SMOKE_FILE_GLOBS = ("tools/bench_board*.py", "tools/speed_gbdt_arm.py",
                    "tools/classical_two_datasets.py", "bench/speed/forest_speed_arm.py")
#: Arms the plan races although they refuse by name on that vendor. A smoke
#: race passes with such an arm only when its cell is a named REFUSED(...).
#: (vendor glob, race key glob family/lane/dataset, arm glob, why)
PLANNED_REFUSALS = [
    ("nvidia", "trees/*", "lightgbm-cuda",
     "the pip LightGBM wheel has no CUDA build (tools/speed_gbdt_arm.py lightgbm_arms)"),
    ("amd", "*", "xgboost-gpu",
     "the pinned XGBoost wheel is CUDA; the image carries no ROCm build"),
    ("nvidia", "classical2/spectral*", "cuml-gpu",
     "cuML SpectralClustering / SpectralEmbedding are absent from the pinned cuML 26.8.0"),
]
#: Race keys whose arms produce no comparable output (a random mask), so a
#: smoke pass needs a time but no quality value.
QUALITY_EXEMPT = ("algos/dropout2d/*",)


def smoke_files_sha256(repo=REPO):
    import glob
    h = hashlib.sha256()
    for pat in SMOKE_FILE_GLOBS:
        for path in sorted(glob.glob(os.path.join(repo, pat))):
            h.update(os.path.relpath(path, repo).encode())
            h.update(sha256_file(path).encode())
    return h.hexdigest()


def smoke_key(race):
    """family/lane/dataset: the same race in a smoke run and a full run."""
    return "%s/%s/%s" % (race["family"], race["lane"], race["dataset"])


def shard_races(races, shard):
    """'i/n': the i-th of n round-robin shares of the races, ordered by race id."""
    try:
        i, n = (int(x) for x in shard.split("/"))
    except (AttributeError, ValueError):
        raise SystemExit("--shard wants i/n (1 <= i <= n), got %r" % (shard,))
    if not 1 <= i <= n:
        raise SystemExit("--shard wants i/n (1 <= i <= n), got %r" % (shard,))
    ordered = sorted(races, key=lambda r: r["id"])
    return [r for k, r in enumerate(ordered) if k % n == i - 1]


def _planned_refusal(vendor, key, arm):
    import fnmatch
    for v, kg, ag, why in PLANNED_REFUSALS:
        if fnmatch.fnmatch(vendor, v) and fnmatch.fnmatch(key, kg) and fnmatch.fnmatch(arm, ag):
            return why
    return None


def smoke_verdict(race, rec, vendor):
    """[(arm, reason)] of what keeps this race from passing ([] = PASS)."""
    import fnmatch
    if rec is None:
        return [("-", "not run")]
    fails = []
    unsupported = unsupported_our_arms(race)
    all_unsupported = bool(race.get("our_arms")) and set(race["our_arms"]) <= set(unsupported)
    if rec.get("status") == "unsupported" and all_unsupported:
        cells = {c.get("arm"): c for c in rec.get("cells") or []}
        if rec.get("unsupported_arms") != unsupported or any(
                cells.get(a, {}).get("status") != reason or cells[a].get("median_ms") is not None
                for a, reason in unsupported.items()):
            return [("-", "missing explicit unsupported capability record")]
        return []
    if rec.get("status") != "done":
        fails.append(("-", "status %s%s" % (rec.get("status"), (": " + str(rec.get("failure")))
                                            if rec.get("failure") else "")))
    if rec.get("params_check") != "MATCHED":
        fails.append(("-", "BOARD-PARAMS %s" % rec.get("params_check")))
    key = smoke_key(race)
    cells = {c.get("arm"): c for c in rec.get("cells") or []}
    exempt = any(fnmatch.fnmatch(key, g) for g in QUALITY_EXEMPT)
    skipped = set(rec.get("skipped_opponents") or []) & set(race.get("opponents") or [])
    for arm in race["arms"]:
        if arm in skipped:
            continue
        c = cells.get(arm)
        if c is None:
            fails.append((arm, "no cell"))
            continue
        status = str(c.get("status"))
        if arm in unsupported and status == unsupported[arm] and c.get("median_ms") is None:
            continue
        why = _planned_refusal(vendor, key, arm)
        if why and status.startswith("REFUSED"):
            continue
        if status != "ok":
            fails.append((arm, "status %s" % status[:200]))
            continue
        if not isinstance(c.get("median_ms"), (int, float)):
            fails.append((arm, "no time"))
        quality = c.get("quality") or {}
        if "error" in quality:
            fails.append((arm, "quality error: " + str(quality["error"])))
        if not exempt and not any(v is not None for k, v in quality.items() if k != "error"):
            fails.append((arm, "no quality value"))
    return fails


def write_smoke(out, races, result, vendor, shard):
    """smoke.json and the SMOKE PASS/FAIL lines; returns the number failing."""
    verdicts = {}
    for r in races:
        fails = smoke_verdict(r, result["races"].get(r["id"]), vendor)
        verdicts[smoke_key(r)] = {"id": r["id"], "pass": not fails,
                                  "unsupported": unsupported_our_arms(r),
                                  "eligible": not (bool(r.get("our_arms")) and set(r["our_arms"]) <= set(unsupported_our_arms(r))),
                                  "failures": [{"arm": a, "reason": w} for a, w in fails]}
    bad = sorted(k for k, v in verdicts.items() if not v["pass"])
    unsupported_count = sum(not v["eligible"] for v in verdicts.values())
    eligible_count = len(verdicts) - unsupported_count
    doc = {"schema": SMOKE_SCHEMA, "created": now_utc(), "commit": repo_commit(),
           "files_sha256": smoke_files_sha256(), "vendor": vendor, "shard": shard,
           "artifact_identity": (result.get("config") or {}).get("artifact_identity"),
           "modes": sorted({m for r in races for m in r.get("our_arms", {}).values()}),
           "rows": SMOKE_ROWS, "total": len(verdicts), "eligible": eligible_count,
           "unsupported": unsupported_count,
           "passed": sum(v["pass"] and v["eligible"] for v in verdicts.values()),
           "races": verdicts}
    with open(os.path.join(out, "smoke.json"), "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    if bad:
        print("SMOKE FAIL %d/%d" % (len(bad), len(verdicts)), flush=True)
        for k in bad:
            for f in verdicts[k]["failures"]:
                print("SMOKE-FAIL %s arm=%s %s" % (verdicts[k]["id"], f["arm"], f["reason"]), flush=True)
    else:
        print("SMOKE PASS %d/%d eligible (%d unsupported, %d planned)"
              % (eligible_count, eligible_count, unsupported_count, len(verdicts)), flush=True)
    return len(bad)


def smoke_gate(out, vendor, races, extra_paths=(), artifact_identity=None):
    """None when every planned race passed a smoke run of the same board and
    driver files on this vendor (the union of every smoke.json found: the
    <out>-smoke* directories beside --out and --smoke-json), else why not."""
    import glob
    paths = list(extra_paths or ()) + sorted(glob.glob(os.path.abspath(out) + "-smoke*/smoke.json"))
    want = smoke_files_sha256()
    passed, seen = set(), []
    wanted_modes = sorted({m for r in races for m in r.get("our_arms", {}).values()})
    for p in paths:
        try:
            with open(p) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        if doc.get("schema") != SMOKE_SCHEMA or doc.get("vendor") != vendor \
                or doc.get("files_sha256") != want:
            continue
        if doc.get("modes") is not None and doc["modes"] != wanted_modes:
            continue
        if doc.get("artifact_identity") != artifact_identity:
            continue
        seen.append(p)
        passed |= {k for k, v in (doc.get("races") or {}).items() if v.get("pass")}
    missing = sorted({smoke_key(r) for r in races} - passed)
    if not missing:
        return None
    return ("no SMOKE PASS for %d of %d planned races with these board and driver files on %s "
            "(smoke results read: %s): %s%s. Run `tools/bench_board.py --smoke --out %s` (or its "
            "--shard halves) first, or pass --no-smoke-gate."
            % (len(missing), len({smoke_key(r) for r in races}), vendor, ", ".join(seen) or "none",
               ", ".join(missing[:12]), " ..." if len(missing) > 12 else "", out))


def build_parser():
    p = argparse.ArgumentParser(prog="bench_board", description=__doc__.split("\n")[0])
    p.add_argument("--out", default=None, help="the run's result directory (resumable)")
    p.add_argument("--vendor", default="auto", choices=("auto",) + VENDORS)
    p.add_argument("--modes", default=None,
                   help="narrow the vendor's modes (Apple: fast,identical; others: identical)")
    p.add_argument("--families", default=",".join(FAMILIES))
    p.add_argument("--lanes", default=None, help="comma list; default every lane of the families")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--rows", default="full",
                   help="full (default) or a row cap for a smoke test (the board says SMOKE)")
    p.add_argument("--neural-shape", default="full", choices=NEURAL_SHAPES,
                   help="neural family shape: full (the board; the 20.45 M-parameter LM "
                        "control shape, 4096^3 GEMM, L2048 blocks) or small (a smoke; the board "
                        "says SMOKE). --rows does not apply to neural lanes")
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                   help="timed rounds after one warm-up (default 1)")
    p.add_argument("--mojolearn-version", default=None, help="pip install mojolearn==<V>")
    p.add_argument("--mojolearn-wheel", default=None,
                   help="install this wheel file instead of downloading one (sha256 recorded)")
    p.add_argument("--cache", default=None,
                   help="venv, wheel download and classical blocks (default <out>/cache; "
                        "a leg puts it outside the fetched directory)")
    p.add_argument("--venv", default=None, help="venv to create or reuse (default <cache>/venv)")
    p.add_argument("--python-env", default=None,
                   help="use this existing interpreter instead of a venv")
    p.add_argument("--base-python", default=sys.executable,
                   help="interpreter that creates the venv (default: this one)")
    p.add_argument("--system-site-packages", action="store_true",
                   help="venv sees the image's packages (the CUDA or ROCm torch on a rented box)")
    p.add_argument("--skip-install", action="store_true",
                   help="install nothing; the interpreter already holds mojolearn and the opponents")
    p.add_argument("--opponent-wheels", default=None,
                   help="offline opponent install from this staged wheel dir (tools/opponent_wheels.sh)")
    p.add_argument("--torch-spec", default=None,
                   help="pip spec for torch; default per vendor: Apple torch==2.13.0, AMD the "
                        "hash-pinned ROCm 6.4.1 cp312 wheels, NVIDIA none (the image's torch)")
    p.add_argument("--torch-index-url", default=None)
    p.add_argument("--data-root", default=os.environ.get(
        "GBM_BENCH_DATA", os.path.join(os.path.expanduser("~"), "datasets", "gbm-bench")),
        help="GBM_BENCH_DATA: where the R2-staged taxi/istella npz caches live")
    p.add_argument("--ctd-data", default=None,
                   help="classical block dir (default <cache>/ctd-data); prep is untimed and once")
    p.add_argument("--verify-data", action="store_true", help="sha256 the dataset files too")
    p.add_argument("--arm-budget-s", type=int, default=3600,
                   help="trees: per-arm budget (MOJOLEARN_SPEED_BUDGET_S)")
    p.add_argument("--race-deadline-s", type=int, default=6 * 3600,
                   help="trees: process deadline per race (MOJOLEARN_SPEED_DEADLINE_S)")
    p.add_argument("--round-seconds", type=int, default=0,
                   help="classical per-round ceiling (0: per lane default)")
    p.add_argument("--nice", type=int, default=0)
    p.add_argument("--skip-failed", action="store_true",
                   help="on resume, do not retry races that failed (default: retry them)")
    p.add_argument("--no-infer", action="store_true",
                   help="time training only: skip the inference cells (trees, and the classical "
                        "kmeans/pca/ols/svc lanes) that are timed after each race's fit rounds")
    p.add_argument("--no-cpu-arm", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--rerun", default=None,
                   help="comma list of race id prefixes (e.g. trees/rf/,trees/et/) to run again "
                        "although done; the old record is kept under `superseded`")
    p.add_argument("--rerun-before", default=None,
                   help="with --rerun: only races that finished before this UTC time "
                        "(ISO, e.g. 2026-09-29T13:00:00Z)")
    p.add_argument("--opponent-store", default=None,
                   help="the opponent store, JSONL (default <out>/../opponent-store.jsonl): an "
                        "opponent already measured for the same key is not run again; its stored "
                        "cell joins the race (tools/bench_board_store.py)")
    p.add_argument("--retime-opponents", action="store_true",
                   help="measure every opponent again (and store the new measurement); "
                        "implies --with-opponents")
    p.add_argument("--retime-cpu-opponents", action="store_true", help="fresh CPU opponents for this harness; GPU opponents stay missing-only")
    p.add_argument("--opponents-only", action="store_true", help="run missing or failed opponent cells only; never run our arms")
    p.add_argument("--with-opponents", action="store_true",
                   help="also race the opponents the store does not hold (default: our GPU "
                        "arm(s) only; stored opponent cells still join the race)")
    p.add_argument("--invalidate-memory", default=None, metavar="PREFIXES",
                   help="before the run (or alone with --render-only): in the finished races whose "
                        "id starts with one of these (comma list, or all), withdraw every GPU memory "
                        "figure a non-torch arm read from torch's allocator; needs "
                        "--invalidate-reason and --invalidate-fixed-at")
    p.add_argument("--invalidate-reason", default=None)
    p.add_argument("--invalidate-arm", action="append", default=[], metavar="ARM",
                   help="before the run (or alone with --render-only): mark every finished cell "
                        "of this arm REFUSED(--invalidate-arm-reason), times withdrawn and ratios "
                        "recomputed, for an arm that did not run what its name says (repeatable)")
    p.add_argument("--invalidate-arm-reason", default=None)
    p.add_argument("--invalidate-fixed-at", default=None, metavar="COMMIT")
    p.add_argument("--backfill-store", default=None, metavar="BOARD_JSON",
                   help="import the opponent cells of an existing board.json into the store, "
                        "print how many were imported and skipped, and exit")
    p.add_argument("--artifact-manifest", help="pin installed numerical artifacts and require actual worker path receipts; mandatory for forced PTX")
    p.add_argument("--smoke", action="store_true",
                   help="the measurement check: every planned race at --rows %d, one round, the "
                        "small neural shape, into <out>-smoke (never the board's own directory), "
                        "no opponent store; writes smoke.json and prints SMOKE PASS n/n or SMOKE "
                        "FAIL k/n (exit 1)" % SMOKE_ROWS)
    p.add_argument("--shard", default=None, metavar="I/N",
                   help="run the I-th of N round-robin shares of the planned races (by race id), "
                        "so two boxes can run halves")
    p.add_argument("--smoke-json", action="append", default=[],
                   help="a smoke.json the gate reads besides <out>-smoke*/smoke.json (repeatable; "
                        "e.g. another box's shard)")
    p.add_argument("--no-smoke-gate", action="store_true",
                   help="start a full board without a SMOKE PASS for every planned race (recorded "
                        "in board.json as overridden)")
    p.add_argument("--guarded-smoke-override-reason", default=None,
                   help="explicit operator reason for skipping smoke on an artifact-guarded full run; runtime provenance remains mandatory")
    p.add_argument("--dry-run", action="store_true", help="print the plan and run nothing")
    p.add_argument("--render-only", action="store_true", help="re-render BOARD.md from board.json")
    p.add_argument("--tree-driver", default=os.path.join(REPO, "bench", "speed", "forest_speed_arm.py"),
                   help=argparse.SUPPRESS)
    p.add_argument("--classical-driver", default=os.path.join(HERE, "classical_two_datasets.py"),
                   help=argparse.SUPPRESS)
    p.add_argument("--more-driver", default=os.path.join(HERE, "bench_board_more.py"),
                   help=argparse.SUPPRESS)
    p.add_argument("--more-data", default=None,
                   help="classical2 block dir (default <cache>/more-data); prep is untimed and once")
    p.add_argument("--neural-driver", default=os.path.join(HERE, "bench_board_neural.py"),
                   help=argparse.SUPPRESS)
    p.add_argument("--algos-driver", default=os.path.join(HERE, "bench_board_algos.py"),
                   help=argparse.SUPPRESS)
    p.add_argument("--algos-data", default=None,
                   help="algos block dir (default <cache>/algos-data); prep is untimed and once")
    return p


_DRIVER_KEY = {"trees": "tree_driver", "classical": "classical_driver", "classical2": "more_driver",
               "neural": "neural_driver", "algos": "algos_driver"}


def _driver_has_params_check(ctx, family):
    """Does this family's driver call tools/bench_board_params.py? A finished
    race whose record has no MATCHED check is run again once it does, so every
    race on the board ends up checked (2026-09-29: same seed, same tuning
    parameters, enforced). Only the driver this run uses is read: the
    trees driver (bench/speed/forest_speed_arm.py) calls the check itself, and
    reading tools/speed_gbdt_arm.py beside it made a driver without the check
    (a stub, an older driver) rerun its finished races on every resume."""
    paths = [ctx.get(_DRIVER_KEY.get(family, ""), "")]
    for p in paths:
        try:
            with open(p, errors="replace") as fh:
                if "bench_board_params" in fh.read():
                    return True
        except OSError:
            pass
    return False


def _rerun_wanted(args, race_id, prev):
    """--rerun: a finished race whose id starts with one of the prefixes and
    that finished before --rerun-before (UTC, ISO) runs again. The earlier
    record is kept under `superseded` in board.json. The time bound makes a
    resumed rerun job skip the races it already reran."""
    if not args.rerun:
        return False
    if not any(race_id.startswith(p.strip()) for p in args.rerun.split(",") if p.strip()):
        return False
    return str(prev.get("finished") or "") < (args.rerun_before or "9999")


def parse_rows(text):
    if text in (None, "", "full"):
        return None
    n = int(text)
    if n <= 0:
        raise SystemExit("--rows must be positive or 'full'")
    return n


def _csv(text, allowed, what):
    vals = [v.strip() for v in (text or "").split(",") if v.strip()]
    bad = [v for v in vals if v not in allowed]
    if bad:
        raise SystemExit("unknown %s: %s (known: %s)" % (what, ",".join(bad), ",".join(allowed)))
    return vals


def print_plan(vendor, modes, races, args, rows, data):
    s = plan_summary(races)
    print("BENCH-BOARD PLAN (dry run: nothing installed, nothing run)")
    print("vendor=%s api=%s modes=%s rounds=%d warmup=1 seed=%d rows=%s"
          % (vendor, VENDOR_API[vendor], ",".join(modes), args.rounds, SEED, rows_tag(rows)))
    if any(r["family"] == "neural" for r in races):
        print("neural: ours %s; opponents torch %s on the GPU, %s on the CPU for the "
              "*-infer lanes; shape %s (LM %s; GEMM %s)" % (
                  "IDENTICAL + FAST (ours-fast, the Apple tier)" if "fast" in modes else "IDENTICAL",
                  "/".join(NEURAL.GPU_SETTINGS[vendor]), "/".join(NEURAL.CPU_SETTINGS),
                  args.neural_shape, NEURAL.shape_text("lm-train-step", args.neural_shape),
                  NEURAL.shape_text("gemm", args.neural_shape)))
        for why in NEURAL.NOT_PLANNED[vendor]:
            print("neural not planned: %s" % why)
    print("mojolearn=%s (from the installed wheel; %s)" % (
        args.mojolearn_version or "<--mojolearn-version>",
        "python %s" % args.python_env if args.python_env else "venv %s" % (args.venv or "<cache>/venv")))
    for idx, reqs in opponent_requirements(vendor, opponent_pins()):
        print("opponents pinned: %s%s" % (" ".join(reqs), (" (index %s)" % idx) if idx else ""))
    ts = args.torch_spec if args.torch_spec is not None else DEFAULT_TORCH_SPEC[vendor]
    print("torch: %s%s" % (ts or "the image's own (use --system-site-packages)",
                           " (extra index %s)" % DEFAULT_TORCH_EXTRA_INDEX[vendor]
                           if args.torch_spec is None and DEFAULT_TORCH_EXTRA_INDEX.get(vendor) else ""))
    if any(r["family"] == "classical2" for r in races):
        print("classical2 opponents pinned: %s" % " ".join(MORE_PINS[vendor]))
        for why in MORE.NOT_PLANNED[vendor]:
            print("classical2 NOT PLANNED: %s" % why)
    algos = [r for r in races if r["family"] == "algos"]
    if algos:
        print("algos opponents pinned: %s%s" % (" ".join(ALGOS.PINS[vendor]), (
            " + %s (rapids index)" % " ".join(ALGOS.RAPIDS_EXTRA[vendor])
            if ALGOS.RAPIDS_EXTRA.get(vendor) else "")))
        for why in ALGOS.not_planned(vendor):
            print("algos NOT PLANNED: %s" % why)
        built = ALGOS.source_exports()
        for key in sorted({k for r in algos for k in ALGOS.r2_keys(r["lane"], r["dataset"])
                           if k.startswith("corpus/")}):
            p = ALGOS.corpus_path(key)
            print("data corpus %s %s (R2 key %s)" % ("present" if os.path.isfile(p) else "MISSING",
                                                    p, key))
    for ds, rec in data.items():
        print("data %-8s %s %s (R2 key %s)" % (ds, "present" if rec["present"] else "MISSING",
                                              rec["path"], rec["r2_key"]))
    print("")
    for r in races:
        if r["family"] == "algos":
            cls = [c for c in r["ours_class"] if c.split(".")[-1] in built or c in built]
            print("RACE %-40s arms=%s class=%s[%s] rows=%s data=%s" % (
                r["id"], ",".join("%s[%s]" % (a, r["our_arms"][a]) if a in r["our_arms"] else a
                                  for a in r["arms"]),
                "|".join(r["ours_class"]), "in source" if cls else "not built yet: SKIPPED",
                ALGOS.rows_text(r["lane"], r["rows"]),
                ",".join(ALGOS.r2_keys(r["lane"], r["dataset"])) or "seed 7"))
            continue
        print("RACE %-40s arms=%s" % (r["id"], ",".join(
            "%s[%s]" % (a, r["our_arms"][a]) if a in r["our_arms"] else a for a in r["arms"])))
    print("")
    for fam, f in sorted(s["by_family"].items()):
        print("family %-10s races=%d cells=%d" % (fam, f["races"], f["cells"]))
    print("TOTAL races=%d cells=%d eligible=%d unsupported=%d"
          % (s["races"], s["cells"], s["races"] - s["unsupported_races"], s["unsupported_races"]))
    print("our CPU: never raced (the board races only our GPU)")
    print("memory: peak_host_mb and peak_gpu_mb per arm and cell (tools/bench_board_probe.py)")
    if not args.no_infer:
        inf = {}
        for r in races:
            n = len(INFER.plan_cells(r)) if r["family"] != "algos" else (
                len(r["arms"]) if ALGOS.has_infer(r["lane"]) else 0)
            if n:
                inf[r["family"]] = inf.get(r["family"], 0) + n
        print("INFER cells=%d (%s; each arm's own model after the fit rounds; --no-infer skips)"
              % (sum(inf.values()), ", ".join("%s %d" % kv for kv in sorted(inf.items())) or "none"))


def validate_guarded_smoke_override(args, artifact_identity):
    reason = (args.guarded_smoke_override_reason or "").strip()
    if args.guarded_smoke_override_reason is not None and (not reason or not args.no_smoke_gate or args.smoke):
        raise SystemExit("bench_board: override reason requires a full run with --no-smoke-gate")
    if artifact_identity and args.no_smoke_gate and not reason:
        raise SystemExit("bench_board: guarded path comparison cannot bypass smoke without an explicit --guarded-smoke-override-reason")
    return reason or None


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    if args.smoke:
        # small and once, into its own directory, never the board's
        args.rows, args.rounds, args.neural_shape = str(SMOKE_ROWS), 1, "small"
        # a hang fails fast: at the smoke's rows anything needing minutes is already a
        # failure (the full board's 600 s steps and 1 h trees budget cost 10 min per hung
        # arm in the 2026-09-29 proof); the bound still covers a first compile
        args.round_seconds = args.round_seconds or SMOKE_STEP_S
        args.arm_budget_s = min(args.arm_budget_s, SMOKE_STEP_S)
        args.race_deadline_s = min(args.race_deadline_s, 4 * SMOKE_STEP_S)
        if not args.dry_run and not args.out:
            raise SystemExit("bench_board: --smoke needs --out (it writes <out>-smoke)")
        if args.out and not os.path.abspath(args.out).rstrip("/").endswith("-smoke"):
            args.out = os.path.abspath(os.path.expanduser(args.out)).rstrip("/") + "-smoke"

    if args.backfill_store:
        board_json = os.path.abspath(os.path.expanduser(args.backfill_store))
        store = args.opponent_store or os.path.join(os.path.dirname(os.path.dirname(board_json)),
                                                    "opponent-store.jsonl")
        imported, skipped = backfill_store(board_json, os.path.abspath(os.path.expanduser(store)))
        print("bench_board: backfill %s -> %s: imported %d opponent cells, skipped %d"
              % (board_json, store, imported, sum(skipped.values())), flush=True)
        for why, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print("bench_board:   skipped %d: %s" % (n, why), flush=True)
        return 0

    if args.invalidate_memory:
        if not (args.out and args.invalidate_reason and args.invalidate_fixed_at):
            raise SystemExit("--invalidate-memory needs --out, --invalidate-reason and "
                             "--invalidate-fixed-at")
        iout = os.path.abspath(os.path.expanduser(args.out))
        store = os.path.abspath(os.path.expanduser(
            args.opponent_store or os.path.join(os.path.dirname(iout), "opponent-store.jsonl")))
        nb, ns = invalidate_memory(iout, args.invalidate_memory, args.invalidate_reason,
                                   args.invalidate_fixed_at, store)
        print("bench_board: --invalidate-memory %s: withdrew %d GPU memory figures on the board, "
              "corrected %d opponent-store records" % (args.invalidate_memory, nb, ns), flush=True)

    if args.invalidate_arm:
        if not (args.out and args.invalidate_arm_reason):
            raise SystemExit("--invalidate-arm needs --out and --invalidate-arm-reason")
        iout = os.path.abspath(os.path.expanduser(args.out))
        store = os.path.abspath(os.path.expanduser(
            args.opponent_store or os.path.join(os.path.dirname(iout), "opponent-store.jsonl")))
        for arm in args.invalidate_arm:
            nb, ns = invalidate_arm(iout, arm, args.invalidate_arm_reason, store,
                                    vendor=None if args.vendor == "auto" else args.vendor)
            print("bench_board: --invalidate-arm %s: refused %d cells on the board, %d opponent-store "
                  "records" % (arm, nb, ns), flush=True)

    if args.render_only:
        if not args.out:
            raise SystemExit("--render-only needs --out")
        result = load_result(os.path.join(args.out, "board.json"))
        if result is None:
            raise SystemExit("no board.json under %s" % args.out)
        write_board(args.out, result)
        print(os.path.join(args.out, "BOARD.md"))
        return 0

    vendor = detect_vendor() if args.vendor == "auto" else args.vendor
    if vendor is None:
        raise SystemExit("bench_board: no Metal, nvidia-smi or rocm-smi found; pass --vendor")
    if vendor == "nvidia" and args.system_site_packages:
        raise SystemExit("bench_board: REFUSING --system-site-packages on NVIDIA: an image's "
                         "dist-packages/nvidia/__init__.py shadows the venv's CUDA libraries and "
                         "cuML cannot load libcudf (2026-09-29). The board installs its own torch "
                         "(%s)." % DEFAULT_TORCH_SPEC["nvidia"])
    modes = modes_for(vendor, args.modes)
    families = _csv(args.families, FAMILIES, "family")
    lanes = (_csv(args.lanes, TREE_LANES + TREE_TASK_LANES + CLASSICAL_LANES + MORE_LANES
                  + NEURAL_LANES + ALGOS_LANES, "lane")
             if args.lanes else None)
    datasets = _csv(args.datasets, DATASETS, "dataset")
    rows = parse_rows(args.rows)
    races = plan_races(vendor, modes, families, lanes, datasets, rows, args.neural_shape)
    if args.opponents_only:
        races = [dict(r, our_arms={}, arms=list(r["opponents"])) for r in races]
    if args.shard:
        races = shard_races(races, args.shard)
    # taxi and Istella-S are read by trees and classical only; a neural-only
    # run needs no R2 data.
    needed = [ds for ds in datasets if any(r["dataset"] == ds for r in races
                                           if r["family"] != "neural")]
    needed += sorted({k for r in races if r["family"] == "trees"
                      for k in race_data_keys(r) if k not in needed})
    # the algos family's own taxi-derived data (taxi-hourly, taxi-zones)
    if "taxi" not in needed and any(r["family"] == "algos" and "gbm-bench/taxi/taxi_speed.npz"
                                    in ALGOS.r2_keys(r["lane"], r["dataset"]) for r in races):
        needed.append("taxi")
    data = data_status(os.path.abspath(os.path.expanduser(args.data_root)), needed)

    if args.dry_run:
        print_plan(vendor, modes, races, args, rows, data)
        return 0

    if not args.out:
        raise SystemExit("bench_board: --out is required (one result directory per run)")
    missing = [ds for ds, rec in data.items() if not rec["present"]]
    if missing:
        raise SystemExit(
            "bench_board: REFUSING: dataset(s) %s missing under %s. This script never downloads. "
            "Stage from R2 on the Mac that holds the credential:\n  sh tools/dataset_store.sh stage "
            "\"<ssh flags+target>\" %s\n(a remote Mac: prefix MOJOLEARN_STAGE_BOX_HOME=<its home>)"
            % (",".join(missing), args.data_root, " ".join(R2_KEYS[d] for d in missing)))
    corpus_missing = sorted({k for r in races if r["family"] == "algos"
                             for k in ALGOS.r2_keys(r["lane"], r["dataset"])
                             if k.startswith("corpus/") and not os.path.isfile(ALGOS.corpus_path(k))})
    if corpus_missing:
        raise SystemExit(
            "bench_board: REFUSING: corpus key(s) %s missing (looked in $MOJOLEARN_CORPUS_ROOT, "
            "<repo>/training, ~/r2-stage, ~/CascadeProjects/mojolearn/training and "
            "~/mojolearn-wt/*/training). This script never downloads. Stage from R2 on the Mac "
            "that holds the credential:\n  sh tools/dataset_store.sh stage \"<ssh flags+target>\" %s\n"
            "(a remote Mac: prefix MOJOLEARN_STAGE_BOX_HOME=<its home>)"
            % (",".join(corpus_missing), " ".join(corpus_missing)))
    if args.verify_data:
        data = data_status(os.path.abspath(os.path.expanduser(args.data_root)), needed, verify=True)
        bad = [d for d, r in data.items() if r.get("sha256_ok") is False]
        if bad:
            raise SystemExit("bench_board: sha256 mismatch against the manifest for %s" % ",".join(bad))

    out = os.path.abspath(os.path.expanduser(args.out))
    # THE SMOKE GATE: a full board (full rows, the full neural shape) starts only
    # after every planned race passed a smoke run of these board and driver files
    full = not args.smoke and rows is None and not (
        any(r["family"] == "neural" for r in races) and args.neural_shape != "full")
    try:
        artifact_identity = _load_tool("bench_board_provenance").identity(args.artifact_manifest)
    except (OSError, ValueError) as exc:
        raise SystemExit("bench_board: artifact guard: " + str(exc))
    guarded_smoke_override = validate_guarded_smoke_override(args, artifact_identity)
    if artifact_identity and (modes != ["identical"] or
            {"nvidia": "cuda", "amd": "hip"}.get(vendor) != artifact_identity.get("vendor", "cuda")):
        raise SystemExit("bench_board: guarded board vendor/mode must match its explicit IDENTICAL artifact identity")
    gate = None
    if full:
        gate = "overridden (--no-smoke-gate)" if args.no_smoke_gate else None
        if not args.no_smoke_gate:
            why = smoke_gate(out, vendor, races, args.smoke_json, artifact_identity)
            if why:
                raise SystemExit("bench_board: REFUSING to start a full board: " + why)
            gate = "passed (SMOKE PASS for every planned race, files %s)" % smoke_files_sha256()[:16]
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    rpath = os.path.join(out, "board.json")
    result = load_result(rpath)

    python, wheel = setup_python(args, vendor, out, os.path.join(out, "logs", "setup.log"))
    arm_python = setup_arm_venvs(args, vendor, out, os.path.join(out, "logs", "setup.log"))
    why = None if getattr(args, "opponents_only", False) else gpu_set_refusal(python, vendor)
    if why:
        raise SystemExit("bench_board: REFUSING: our IDENTICAL GPU set cannot load on this %s box, "
                         "so every race would refuse our arm:\n%s" % (vendor, why))
    if artifact_identity:
        # Validate installed bytes before spending GPU time, using the exact
        # interpreter handed to workers. This is not a worker-load receipt.
        verify = subprocess.run([python, os.path.join(HERE, "bench_board_provenance.py"),
                                 "--verify", os.path.abspath(args.artifact_manifest)],
                                capture_output=True, text=True)
        if verify.returncode:
            raise SystemExit("bench_board: installed artifact guard: " + verify.stderr[-1500:])
    if wheel is None and result:
        wheel = ((result.get("box") or {}).get("mojolearn") or {}).get("wheel")
    ptxas = None
    if vendor == "nvidia":
        drv = capture(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        try:
            if drv and int(drv.splitlines()[0].split(".")[0]) < 580 and os.access(
                    "/usr/local/cuda/bin/ptxas", os.X_OK):
                ptxas = "/usr/local/cuda/bin/ptxas"   # drivers below 580 (bench_all_ours.sh)
        except ValueError:
            pass
    ctx = {"vendor": vendor, "modes": modes, "python": python, "out": out,
           "rounds": args.rounds, "data_root": os.path.abspath(os.path.expanduser(args.data_root)),
           "ctd_data": os.path.abspath(args.ctd_data or os.path.join(cache_dir(args, out), "ctd-data")),
           "commit": repo_commit(), "mojolearn_version": args.mojolearn_version,
           "wheel": wheel, "arm_budget_s": args.arm_budget_s,
           "race_deadline_s": args.race_deadline_s, "round_seconds": args.round_seconds,
           "nice": args.nice, "ptxas": ptxas, "infer": not args.no_infer,
           "tree_driver": os.path.abspath(args.tree_driver),
           "classical_driver": os.path.abspath(args.classical_driver),
           "neural_driver": os.path.abspath(args.neural_driver),
           "more_driver": os.path.abspath(args.more_driver),
           "more_data": os.path.abspath(args.more_data or os.path.join(cache_dir(args, out), "more-data")),
           "algos_driver": os.path.abspath(args.algos_driver), "arm_python": arm_python,
           "algos_data": os.path.abspath(args.algos_data or os.path.join(cache_dir(args, out), "algos-data"))}
    ctx["artifact_manifest"] = os.path.abspath(args.artifact_manifest) if args.artifact_manifest else None
    ctx["artifact_identity"] = artifact_identity
    box = box_fingerprint(ctx)
    if artifact_identity:
        box["artifact_identity"] = artifact_identity
        box["artifact_hardware"] = _load_tool("bench_board_provenance").hardware_receipt(
            "cuda" if vendor == "nvidia" else "hip")
        ctx["artifact_hardware"] = box["artifact_hardware"]
    ctx["box"] = box
    ctx["retime"] = args.retime_opponents
    ctx["retime_cpu_opponents"] = args.retime_cpu_opponents
    ctx["opponents_only"] = args.opponents_only
    ctx["with_opponents"] = bool(args.with_opponents or args.retime_opponents or args.opponents_only)
    ctx["store_path"] = os.path.abspath(os.path.expanduser(
        args.opponent_store or os.path.join(os.path.dirname(out), "opponent-store.jsonl")))
    ctx["data_sha"] = {}
    if args.smoke:
        ctx["store_path"] = None          # a smoke run neither reads nor writes the store

    if result is None:
        result = {"schema": SCHEMA, "created": now_utc(), "box": box, "races": {}}
    else:
        if result.get("schema") != SCHEMA:
            raise SystemExit("bench_board: %s has schema %r, not %r" % (rpath, result.get("schema"), SCHEMA))
        old, new = box_key(result.get("box") or {}), box_key(box)
        if old != new:
            raise SystemExit(
                "bench_board: REFUSING to resume %s on a different box or wheel:\n  was %s\n  now %s\n"
                "One board is one box and one set of library bytes; use a new --out."
                % (out, json.dumps(old, sort_keys=True), json.dumps(new, sort_keys=True)))
        result.setdefault("box_history", []).append({"resumed": now_utc(), "packages": box["packages"]})
        result["box"] = box
    result["config"] = {"artifact_identity": artifact_identity,
                        "harness_sha256": smoke_files_sha256(), "vendor": vendor, "modes": modes, "families": families,
                        "lanes": lanes, "datasets": datasets, "rows": rows,
                        "rounds": args.rounds, "seed": SEED, "infer": not args.no_infer,
                        "neural_shape": args.neural_shape if "neural" in families else None,
                        "smoke": (bool(rows) and rows < TREE_ROW_FLOOR
                                  and any(r["family"] != "neural" for r in races))
                                 or ("neural" in families and args.neural_shape == "small"),
                        "data": data, "data_root": ctx["data_root"],
                        # sha256 of each data file a race read (the store's data key)
                        "data_sha256": ctx["data_sha"],
                        "opponent_store": ctx["store_path"],
                        "retime_opponents": ctx["retime"],
                        "opponents_only": args.opponents_only,
                        "retime_cpu_opponents": args.retime_cpu_opponents,
                        "smoke_check": bool(args.smoke), "shard": args.shard,
                        "smoke_gate": gate,
                        "guarded_smoke_override_reason": guarded_smoke_override}
    result["plan"] = [r["id"] for r in races]
    save_result(rpath, result)
    write_board(out, result)

    todo = []
    for r in races:
        prev = result["races"].get(r["id"])
        if prev and prev.get("status") == "done" and not prev.get("no_execution") and prev.get("params_check") != "MATCHED" \
                and _driver_has_params_check(ctx, r["family"]):
            print("bench_board: RERUN %s (done %s without a MATCHED parameter check: %s; its "
                  "driver now has the check)" % (r["id"], prev.get("finished"),
                                                 prev.get("params_check") or "NOT CHECKED"), flush=True)
            result.setdefault("superseded", []).append(prev)
        elif prev and prev.get("status") == "done" and not _rerun_wanted(args, r["id"], prev):
            print("bench_board: skip %s (done %s)" % (r["id"], prev.get("finished")), flush=True)
            continue
        if prev and prev.get("status") == "done":
            print("bench_board: RERUN %s (done %s, before --rerun-before %s)"
                  % (r["id"], prev.get("finished"), args.rerun_before), flush=True)
            result.setdefault("superseded", []).append(prev)
        if prev and prev.get("status") == "failed":
            if args.skip_failed and not _rerun_wanted(args, r["id"], prev):
                print("bench_board: skip %s (failed earlier; --skip-failed)" % r["id"], flush=True)
                continue
            # An explicitly selected repair retries only its failed race and
            # preserves the refusal/failure receipt beside the replacement.
            result.setdefault("superseded", []).append(prev)
        todo.append(r)
    if any(r["family"] == "classical2" for r in todo):
        ensure_more_prep(ctx, todo)
    if any(r["family"] == "algos" for r in todo):
        ensure_algos_prep(ctx, todo)
    if any(r["family"] == "classical" for r in todo):
        ensure_classical_prep(ctx, todo)
    for i, r in enumerate(todo):
        print("bench_board: [%d/%d] %s arms=%s" % (i + 1, len(todo), r["id"], ",".join(r["arms"])),
              flush=True)
        t0 = time.time()
        rec = run_race(ctx, r)
        rec["wall_s"] = round(time.time() - t0, 1)
        result["races"][r["id"]] = rec
        save_result(rpath, result)
        write_board(out, result)
        print("bench_board:   %s rc=%s %.0fs" % (rec["status"], rec["rc"], rec["wall_s"]), flush=True)
    print("bench_board: board %s" % os.path.join(out, "BOARD.md"), flush=True)
    if args.smoke:
        return 1 if write_smoke(out, races, result, vendor, args.shard) else 0
    failed = [rid for rid, rec in result["races"].items()
              if rid in result["plan"] and rec.get("status") not in ("done", "unsupported")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
