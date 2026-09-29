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
    interleaved in the same race, for trees and classical; NVIDIA and AMD run
    `identical` ONLY. The neural family is `identical` ONLY on EVERY vendor
    (the wheel builds its neural surface identical only), and `--modes fast`
    with the neural family is refused by name.
  * mojolearn comes from `pip install mojolearn==<V>` into a venv this script
    creates (or `--python-env` an interpreter that already has it). No source
    build. The wheel's sha256 is recorded.
  * Opponents are pinned from tools/opponent_wheels.sh; `--opponent-wheels DIR`
    installs them offline from a set staged out of R2.
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
             the board; `small` is a smoke.
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
import datetime
import hashlib
import importlib.util
import json
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
#: Fields added to schema /1 without breaking a resume: the `ours-cpu` arm and
#: per cell `peak_host_mb`, `peak_gpu_mb`, `memory` and `ratio_ours_cpu_over`.
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


def check_neural_modes(families, modes):
    """The neural surface is IDENTICAL only on every vendor
    (mojolearn._backend._IDENTICAL_ONLY). FAST is trees and classical."""
    if "neural" in families and "identical" not in modes:
        raise SystemExit(
            "bench_board: REFUSING --modes %s for the neural family: the wheel's neural surface "
            "(LanguageModelTrainer, linalg.matmul, the transformer and Mamba blocks) is IDENTICAL "
            "only on every vendor, and FAST is the trees and classical tier. Pass --families "
            "trees,classical for a FAST-only run, or add identical." % ",".join(modes))


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
#: there (tools/bench_board_leg.sh arranges it). NVIDIA: the image's own CUDA
#: torch through `--system-site-packages` unless --torch-spec names one.
AMD_TORCH_ROCM = (
    "https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/pytorch_triton_rocm-3.2.0%2Brocm6.4.1."
    "git6da9e660-cp312-cp312-linux_x86_64.whl"
    "#sha256=1d97c15798bf178299328032141a21d9777e7cdef59d5a7e3ac74e297c17198e "
    "https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/torch-2.6.0%2Brocm6.4.1.git1ded221d-"
    "cp312-cp312-linux_x86_64.whl"
    "#sha256=6b141e1a03148b007c6217519cd9947d760123ded5caebadffec22cba7358d2d")
DEFAULT_TORCH_SPEC = {"apple": "torch==2.13.0", "nvidia": "", "amd": AMD_TORCH_ROCM}
#: OUR CPU ARM. The wheel's public CPU switch is MOJOLEARN_VENDOR=cpu before
#: import (python/mojolearn/_backend.py): no GPU set loads and the host
#: bindings under mojolearn/host/ answer, IDENTICAL only. `ours-cpu` is our
#: IDENTICAL estimator in a worker started under that switch
#: (tools/bench_board_probe.py), raced beside the CPU opponents already on the
#: board, and its output is compared bit for bit with our GPU IDENTICAL arm.
#: Every estimator on the board routes to a shipped host family
#: (python/mojolearn/host_surface.py routed_modules); a configuration the host
#: side does not restate (the GBDT options in host_surface.NO_CPU_PATH)
#: refuses by name in its cell. `--no-cpu-arm` leaves it off.
CPU_ARM = "ours-cpu"
CPU_SWITCH = "MOJOLEARN_VENDOR=cpu before import (the wheel's public CPU switch), IDENTICAL"


def cpu_arm_reason(family, lane):
    """None when `ours-cpu` is planned for this lane, else why not."""
    if family == "neural" and NEURAL.DEVICE_OF.get(lane) == "cpu":
        return ("its `ours` arm already IS the CPU path (the public *Inference class runs on "
                "the host binding)")
    return None


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

#: Per-arm memory and the ours-cpu readback (standard library only at import).
PROBE = _load_tool("bench_board_probe")


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


def our_arms(family, modes, lane=None, cpu_arm=False):
    """driver arm name -> numeric mode, for our arms in one race. `cpu_arm`
    adds `ours-cpu` (IDENTICAL, the only tier the host bindings build) where
    identical is planned and the lane has a CPU arm (cpu_arm_reason)."""
    out = _our_gpu_arms(family, modes, lane)
    if cpu_arm and "identical" in modes and out and cpu_arm_reason(family, lane) is None:
        out[CPU_ARM] = "identical"
    return out


def _our_gpu_arms(family, modes, lane=None):
    if family == "neural":
        return {"ours": "identical"}          # identical only, every vendor
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


def plan_races(vendor, modes, families=FAMILIES, lanes=None, datasets=DATASETS, rows=None,
               neural_shape="full", cpu_arm=True):
    check_neural_modes(families, modes)
    races = []
    for fam in families:
        for lane in family_lanes(fam):
            if lanes and lane not in lanes:
                continue
            if fam == "neural":
                # one race per lane: its own data, not taxi/Istella; IDENTICAL only
                ours = our_arms(fam, modes, lane, cpu_arm)
                opp = NEURAL_OPPONENTS[vendor][lane]
                ds = NEURAL_DATA[lane]
                races.append({
                    "id": race_id(fam, lane, ds, None, neural_shape),
                    "family": fam, "lane": lane, "dataset": ds, "rows": None,
                    "shape": neural_shape, "modes": ["identical"], "our_arms": ours,
                    "opponents": list(opp), "arms": list(ours) + list(opp),
                })
                continue
            if fam == "algos":
                ours = our_arms(fam, modes, lane, cpu_arm)
                if not ours:
                    continue
                opp = ALGOS.opponents(vendor, lane)
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
                ours = our_arms(fam, modes, lane, cpu_arm)
                if not ours:
                    continue
                opp = MORE.OPPONENTS[vendor][lane]
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
                opp = (TREE_OPPONENTS if fam == "trees" else CLASSICAL_OPPONENTS)[vendor][lane]
                ours = our_arms(fam, modes, lane, cpu_arm)
                races.append({
                    "id": race_id(fam, lane, ds, rows),
                    "family": fam, "lane": lane, "dataset": ds, "rows": rows,
                    "modes": list(modes), "our_arms": ours,
                    "opponents": list(opp),
                    "arms": list(ours) + list(opp),
                })
    return races


def plan_summary(races):
    by_fam = {}
    for r in races:
        f = by_fam.setdefault(r["family"], {"races": 0, "cells": 0})
        f["races"] += 1
        f["cells"] += len(r["arms"])
    return {"races": len(races), "cells": sum(len(r["arms"]) for r in races),
            "cpu_cells": sum(1 for r in races if CPU_ARM in r["arms"]),
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
    if extra:
        env.update(extra)
    return env


# ---------------------------------------------------------------------------
# The box fingerprint
# ---------------------------------------------------------------------------

def repo_commit(repo=REPO):
    """The repo commit this script shipped in: git, else SHIPPED_COMMIT.txt
    (legs ship `git archive`, which has no .git), else the environment."""
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
        "repo": {"commit": ctx.get("commit"), "script": os.path.relpath(script, REPO),
                 "script_sha256": sha256_file(script)},
        "env": {k: os.environ.get(k) for k in THREAD_ENV + ("MODULAR_NVPTX_COMPILER_PATH",)},
    }


def box_key(box):
    """What must not change between a run and its resume: the same box and
    the same library bytes. A board mixing two boxes is the patchwork that
    produced a wrong CatBoost headline (bench_all_ours.sh)."""
    w = (box.get("mojolearn") or {}).get("wheel") or {}
    return {"vendor": (box.get("gpu") or {}).get("vendor"),
            "gpu": (box.get("gpu") or {}).get("name"),
            "hostname": (box.get("host") or {}).get("hostname"),
            "mojolearn": (box.get("mojolearn") or {}).get("version"),
            "wheel_sha256": w.get("sha256")}


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
    torch_spec = args.torch_spec if args.torch_spec is not None else DEFAULT_TORCH_SPEC[vendor]
    if torch_spec:
        cmd = list(pip)
        if args.torch_index_url:
            cmd += ["--index-url", args.torch_index_url]
        if run_logged(cmd + shlex.split(torch_spec), None, log, 3600) != 0:
            print("bench_board: torch install failed (%s); torch arms will refuse by name"
                  % torch_spec, flush=True)
    return python, wheel


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

def load_result(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


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
    bindings, warm, notes, verdict_line, mem = {}, {}, [], None, {}
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
            "mem": {a: [rs[k] for k in sorted(rs)] for a, rs in mem.items()}}


def tree_cmd(ctx, race):
    ours = race["our_arms"]
    primary = ours["ours"]
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
    if CPU_ARM in ours:
        cmd += ["--ours-cpu"]
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
                    fit=a["fit"], shape=parsed["shape"],
                    comparability={"fit_verdict": parsed["verdict"] or "UNKNOWN",
                                   "fit_verdict_line": parsed["verdict_line"]},
                    verdict=parsed["verdict"] or "UNKNOWN")
        cell.update(memory_fields(parsed.get("mem", {}).get(arm)))
        if mode:
            b = parsed["bindings"].get(arm) or {}
            cell["binding"] = b
            cell["mode_witness"] = b.get("compiled")
            cell["installed_wheel"] = _from_wheel(b.get("path"), ctx)
            if b and (b.get("compiled") != mode or b.get("resolved") != mode):
                cell["status"] = "MODE-MISMATCH(requested %s, compiled %s)" % (mode, b.get("compiled"))
            if arm == CPU_ARM:
                cell["vendor_witness"] = b.get("vendor")
                cell["cpu_switch"] = CPU_SWITCH
                if b and b.get("vendor") != "cpu":
                    cell["status"] = "VENDOR-MISMATCH(ours-cpu read back %s)" % b.get("vendor")
        cells.append(cell)
    add_cpu_bits(cells)
    return cells


def memory_fields(samples):
    """peak_host_mb / peak_gpu_mb and the `memory` record of one cell from
    its per-round samples (warm-up first; tools/bench_board_probe.py)."""
    if not samples:
        return {"peak_host_mb": None, "peak_gpu_mb": None,
                "memory": {"host_method": "not sampled", "gpu_method": "not sampled"}}
    m = PROBE.summarize(samples)
    return {"peak_host_mb": m["peak_host_mb"], "peak_gpu_mb": m["peak_gpu_mb"], "memory": m}


def add_cpu_bits(cells):
    """`bits_equal_vs_ours_identical` on the ours-cpu cell: the driver's
    array comparison when it made one (classical, classical2, neural), else
    the round hash of the same output against ours' (trees: the prediction
    vector's hash, FSPEED)."""
    cpu = next((c for c in cells if c["arm"] == CPU_ARM), None)
    ours = next((c for c in cells if c["arm"] == "ours"), None)
    if cpu is None:
        return cells
    q = cpu.setdefault("quality", {})
    if "bits_equal_vs_ours_identical" in q:
        cpu["bits_basis"] = "the saved outputs, array by array"
    elif ours is not None and cpu.get("hash") and ours.get("hash"):
        q["bits_equal_vs_ours_identical"] = cpu["hash"] == ours["hash"]
        cpu["bits_basis"] = "the last timed round's output hash"
    return cells


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
                    quality=q, hash=(a.get("digests") or [None])[-1],
                    hash_stable=a.get("digest_stable"), shape=shape,
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
            if arm == CPU_ARM:
                cell["vendor_witness"] = info.get("vendor_used")
                cell["cpu_switch"] = CPU_SWITCH
                if info and info.get("vendor_used") not in (None, "cpu"):
                    cell["status"] = "VENDOR-MISMATCH(ours-cpu read back %s)" % info.get("vendor_used")
        cells.append(cell)
    add_cpu_bits(cells)
    return cells


# ---------------------------------------------------------------------------
# Cells and ratios
# ---------------------------------------------------------------------------

def base_cell(ctx, race, arm, mode):
    lib = arm_library(arm)
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
            s["numeric_mode"] = "identical (the only tier the neural surface builds)"
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
    """Our cell of one kind: 'identical' (the GPU IDENTICAL arm), 'fast', or
    'cpu' (ours-cpu). By arm name, so ours-cpu (also IDENTICAL) never stands
    in for the GPU IDENTICAL arm."""
    for c in cells:
        if c["library"] != "mojolearn":
            continue
        if which == "cpu" and c["arm"] == CPU_ARM:
            return c
        if which == "fast" and c["arm"] != CPU_ARM and c["mode"] == "fast":
            return c
        if which == "identical" and c["arm"] != CPU_ARM and c["mode"] == "identical":
            return c
    return None


def add_ratios(cells):
    """ratio_ours_identical_over / ratio_ours_fast_over / ratio_ours_cpu_over:
    our median divided by this OPPONENT arm median, computed only when both
    sides completed every round."""
    ok = [c for c in cells if c["status"] == "ok" and c["median_ms"]]
    ours_id = ours_of(ok, "identical")
    ours_fast = ours_of(ok, "fast")
    ours_cpu = ours_of(ok, "cpu")
    done = {id(c) for c in ok}
    for c in cells:
        c["ratio_ours_identical_over"] = None
        c["ratio_ours_fast_over"] = None
        c["ratio_ours_cpu_over"] = None
        # Opponents only. FAST over IDENTICAL is the cost of identity, an
        # internal number that never reaches a board (ENGINEERING_RULES 0b-iii);
        # our CPU tier over our GPU tier is not a board number either.
        if id(c) not in done or c["library"] == "mojolearn":
            continue
        if ours_cpu:
            c["ratio_ours_cpu_over"] = ours_cpu["median_ms"] / c["median_ms"]
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


def _device_text(box, arm, vendor):
    dev = arm_device(arm, vendor)
    name = (box.get("gpu") or {}).get("name") if dev == "gpu" else (box.get("host") or {}).get("cpu_model")
    return "%s (%s)" % (dev, name) if name else None


def library_version(box, lib):
    """The installed version of an opponent library, from the box's package list
    (a distribution named `lib` or `lib-<suffix>`: cuml-cu12, faiss-cpu, cupy-cuda12x)."""
    pk = box.get("packages") or {}
    lib = (lib or "").lower()
    if lib in pk:
        return pk[lib]
    for name in sorted(pk):
        if name.startswith(lib + "-"):
            return pk[name]
    return None


_R2_TO_DATA = {v: k for k, v in R2_KEYS.items()}


def race_data_files(ctx, race):
    """[(name, path)] of the data files a race reads; [] for seeded data; None when
    the race reads a file the board does not know how to name."""
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
            cache[name] = sha256_file(path) if os.path.isfile(path) else None
        if cache[name] is None:
            return None
        shas.append(cache[name])
    return shas[0] if len(shas) == 1 else STORE.sha256_json(shas)


def opponent_key(box, race, arm, settings, data_sha, rounds):
    vendor = (box.get("gpu") or {}).get("vendor")
    lib = arm_library(arm)
    return {"box": (box.get("host") or {}).get("hostname"), "machine": _machine(box),
            "vendor": vendor, "device": _device_text(box, arm, vendor),
            "os": (box.get("os") or {}).get("platform"),
            "library": lib, "library_version": library_version(box, lib),
            "family": race["family"], "lane": race["lane"], "dataset": race["dataset"],
            "rows": race.get("rows"), "neural_shape": race.get("shape"), "arm": arm,
            "settings_sha256": STORE.sha256_json(settings) if settings else None,
            "data_sha256": data_sha, "rounds": rounds}


def stored_opponents(ctx, race):
    """{arm: stored record} for the race's opponents the store already holds
    (none with --retime-opponents or without a store)."""
    if not ctx.get("store_path") or ctx.get("retime") or not ctx.get("box"):
        return {}
    store = STORE.load(ctx["store_path"])
    if not store:
        return {}
    settings = race_settings(ctx, race)
    dsha = race_data_sha(ctx, race, ctx["box"])
    out = {}
    for arm in race.get("opponents") or []:
        hit = STORE.lookup(store, opponent_key(ctx["box"], race, arm, settings, dsha, ctx["rounds"]))
        if hit is not None:
            out[arm] = hit
    return out


def store_opponents(ctx, race, rec):
    """Append every opponent cell measured in this race to the store; returns
    how many were stored (a cell with a missing key field is not)."""
    if not ctx.get("store_path") or not ctx.get("box"):
        return 0
    settings = race_settings(ctx, race)
    dsha = race_data_sha(ctx, race, ctx["box"])
    params = ((rec.get("params") or {}).get("arms") or {})
    n = 0
    for c in rec.get("cells") or []:
        if c.get("library") == "mojolearn" or c.get("stored"):
            continue
        key = opponent_key(ctx["box"], race, c["arm"], settings, dsha, ctx["rounds"])
        if STORE.missing(key):
            c["store"] = "not stored (key fields missing: %s)" % ", ".join(STORE.missing(key))
            continue
        infer = [ic for ic in rec.get("infer_cells") or [] if ic.get("arm") == c["arm"]]
        STORE.append(ctx["store_path"], STORE.record(
            key, c, measured_at=rec.get("finished"), commit=ctx.get("commit"),
            params=params.get(c["arm"]), infer_cells=infer))
        n += 1
    return n


def backfill_store(board_json, store_path):
    """Import the opponent cells of an existing board.json into the store, the
    key and provenance from the board's box, config and each cell's settings.
    Returns (imported, skipped)."""
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
    ctx = {"data_root": cfg.get("data_root") or "", "data_sha": shas}
    imported = skipped = 0
    for rid, rr in sorted((res.get("races") or {}).items()):
        race = {"family": rr.get("family"), "lane": rr.get("lane"), "dataset": rr.get("dataset"),
                "rows": rr.get("rows"), "shape": rr.get("shape")}
        if not race["family"]:
            continue
        try:
            files = race_data_files(ctx, race)
            dsha = race_data_sha(ctx, race, box) if files is not None else None
        except Exception:               # noqa: BLE001  (a file the backfill cannot name)
            dsha = None
        if files and any(n not in shas for n, _ in files):
            dsha = None                 # never hash a file on a box it may not be on
        params = ((rr.get("params") or {}).get("arms") or {})
        for c in rr.get("cells") or []:
            if c.get("library") == "mojolearn" or c.get("stored"):
                continue
            key = opponent_key(box, race, c.get("arm"), c.get("settings"), dsha,
                               (c.get("settings") or {}).get("rounds") or cfg.get("rounds"))
            if STORE.missing(key):
                skipped += 1
                continue
            infer = [ic for ic in rr.get("infer_cells") or [] if ic.get("arm") == c.get("arm")]
            STORE.append(store_path, STORE.record(
                key, c, measured_at=rr.get("finished"), commit=(box.get("repo") or {}).get("commit"),
                params=params.get(c.get("arm")), infer_cells=infer))
            imported += 1
    return imported, skipped


def run_race(ctx, race):
    """Run one race and return its record (status, rc, log, cells). Opponents
    the store already holds for this key are not run; their stored cells join
    the race (tools/bench_board_store.py)."""
    stored = stored_opponents(ctx, race)
    full = race
    if stored:
        race = dict(race, opponents=[a for a in race["opponents"] if a not in stored],
                    arms=[a for a in race["arms"] if a not in stored])
        print("bench_board:   stored (not run): %s" % ", ".join(
            "%s [%s]" % (a, STORE.source_text(r)) for a, r in sorted(stored.items())), flush=True)
    rec = _run_race(ctx, race)
    rec["arms"] = full["arms"]
    rec["stored_arms"] = sorted(stored)
    for c in rec["cells"]:
        if c.get("library") != "mojolearn":
            c["source"] = "measured this run"
    for arm, r in sorted(stored.items()):
        rec["cells"].append(STORE.stored_cell(r))
        if r.get("infer_cells") and rec.get("infer_cells") is not None:
            for ic in r["infer_cells"]:
                ic = dict(ic, source=STORE.source_text(r))
                rec["infer_cells"].append(ic)
    rec["cells"] = add_ratios(rec["cells"])
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
    if c["arm"] == CPU_ARM:
        return "mojolearn CPU %s" % (c["mode"] or "?").upper()
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


def render_board(result):
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
             "far. The `*-infer` lanes are the CPU *Inference classes and race `torch-cpu-*` "
             "arms. An arm torch cannot run on this box is REFUSED by name in its cell. Every "
             "clock is host in, host out, synchronized. Every arm starts from the same "
             "parameters and reads the same inputs, so losses and outputs are comparable; "
             "`max_abs_diff_vs_ours` / `max_rel_diff_vs_ours` are the arm's output against ours.")
    L.append("- `installed_wheel` confirms our binding loaded from site-packages, not the repo tree.")
    L.append("- Our CPU tier (`mojolearn CPU IDENTICAL`, arm `ours-cpu`): the same public estimator "
             "in a worker started under MOJOLEARN_VENDOR=cpu, the wheel's CPU switch (no GPU set "
             "loads; the host bindings answer, IDENTICAL only), read back as vendor cpu or refused "
             "by name. It races in the same rounds as every arm; `ours CPU / arm` is its median "
             "over each opponent's. `bits_equal_vs_ours_identical` compares its output with our "
             "GPU IDENTICAL arm's, bit for bit. Our CPU and GPU times are never divided by each "
             "other here.")
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
    cells = all_cells(result)
    st = {}
    for c in cells:
        k = c["status"].split("(")[0]
        st[k] = st.get(k, 0) + 1
    L.append("## Coverage")
    L.append("")
    L.append("Races: %d planned, %d done, %d failed, %d pending. Cells: %d (%s)."
             % (len(planned), done, failed, max(0, len(planned) - done - failed), len(cells),
                ", ".join("%s %d" % kv for kv in sorted(st.items())) or "none"))
    icov = INFER.coverage(races)
    if icov:
        L.append("")
        L.append(icov)
    L.append("")

    # Quality at a glance
    L.append("## Quality at a glance")
    L.append("")
    L.append("Per lane and dataset: our FAST value, our IDENTICAL value, our CPU IDENTICAL value, "
             "and each opponent's.")
    L.append("")
    L.append("| family | lane | dataset | metric | ours FAST | ours IDENTICAL | ours CPU | opponents |")
    L.append("|---|---|---|---|---|---|---|---|")
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
            cpu = ours_of(rc, "cpu")
            opps = [c for c in rc if c["library"] != "mojolearn"]
            q = lambda c: _f((c.get("quality") or {}).get(m), 6) if c else "-"   # noqa: E731
            L.append("| %s | %s | %s | %s%s | %s | %s | %s | %s |" % (
                races[rid]["family"], races[rid]["lane"], races[rid]["dataset"], m,
                (" (%s)" % QUALITY_NOTE[m]) if m in QUALITY_NOTE else "",
                q(fast), q(ident), q(cpu),
                "; ".join("%s %s" % (c["arm"], q(c)) for c in opps) or "-"))
    L.append("")
    L.extend(render_cpu_glance(races))
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
                                                             rows_tag(rr["rows"]), _f(shape)))
            L.append("")
            L.append("race: %s, driver rc %s, log `%s`" % (rr.get("status"), rr.get("rc"), rr.get("log")))
            L.append("")
            L.append("| arm | library | device | mode | median ms | min..max ms | rounds | "
                     "ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | "
                     "peak GPU MB | quality | hash stable | comparability | installed_wheel | "
                     "status |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
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
                             _f(c.get("ratio_ours_cpu_over"), 3),
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


def render_cpu_glance(races):
    """'Our CPU tier at a glance': per race, ours-cpu's median, whether its
    output equals our GPU IDENTICAL output bit for bit, and each CPU
    opponent's median with ours CPU / arm."""
    rows = []
    for rid in sorted(races):
        rc = races[rid].get("cells") or []
        cpu = ours_of(rc, "cpu")
        if cpu is None:
            continue
        q = cpu.get("quality") or {}
        opps = [c for c in rc if c["library"] != "mojolearn" and c.get("device") == "cpu"]
        rows.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            races[rid]["family"], races[rid]["lane"], races[rid]["dataset"],
            _f(cpu["median_ms"]), _f(q.get("bits_equal_vs_ours_identical")),
            _f(cpu.get("peak_host_mb")),
            "; ".join("%s %s ms (ours CPU / arm %s)" % (
                c["arm"], _f(c["median_ms"]), _f(c.get("ratio_ours_cpu_over"), 3))
                for c in opps) or "- (no CPU opponent on this lane here)"))
    if not rows:
        return []
    return ["## Our CPU tier at a glance", "",
            "Arm `ours-cpu`: " + CPU_SWITCH + ". `CPU = GPU IDENTICAL bits` compares its output "
            "with our GPU IDENTICAL arm's in the same race. A REFUSED ours-cpu cell names why.", "",
            "| family | lane | dataset | ours CPU ms | CPU = GPU IDENTICAL bits | peak host MB | "
            "CPU opponents |", "|---|---|---|---|---|---|---|"] + rows + [""]


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
    if cfg.get("cpu_arm") is False:
        out.append("Our CPU tier: `--no-cpu-arm` was passed, so no `ours-cpu` arm ran.")
    lanes = sorted({l for l in NEURAL_LANES if cpu_arm_reason("neural", l)})
    if lanes:
        out.append("Our CPU tier, no ours-cpu arm: neural %s: %s." % (
            ", ".join(lanes), cpu_arm_reason("neural", lanes[0])))
    out.append("Our CPU tier: a GBDT configuration the host side does not restate refuses by name "
               "in its ours-cpu cell (python/mojolearn/host_surface.py NO_CPU_PATH lists them), and "
               "a FAST-only run (`--modes fast`) has no ours-cpu arm: the host bindings build "
               "IDENTICAL only.")
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
    p.add_argument("--no-cpu-arm", action="store_true",
                   help="skip our CPU tier: no `ours-cpu` arm (by default it races on every lane "
                        "whose estimator has a CPU path, on every vendor)")
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
                   help="measure every opponent again (and store the new measurement)")
    p.add_argument("--backfill-store", default=None, metavar="BOARD_JSON",
                   help="import the opponent cells of an existing board.json into the store, "
                        "print how many were imported and skipped, and exit")
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
    race on the board ends up checked (Andrew, 2026-09-29: same seed, same
    tuning parameters, enforced). Only the driver this run uses is read: the
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
        print("neural: IDENTICAL only; opponents torch %s on the GPU, %s on the CPU for the "
              "*-infer lanes; shape %s (LM %s; GEMM %s)" % (
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
    print("torch: %s" % (ts or "the image's own (use --system-site-packages)"))
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
    print("TOTAL races=%d cells=%d" % (s["races"], s["cells"]))
    if args.no_cpu_arm:
        print("ours-cpu: off (--no-cpu-arm)")
    else:
        print("ours-cpu cells=%d (%s; bit-compared with our GPU IDENTICAL arm)"
              % (s["cpu_cells"], CPU_SWITCH))
        skipped = sorted({(r["family"], r["lane"]) for r in races
                          if CPU_ARM not in r["arms"] and cpu_arm_reason(r["family"], r["lane"])})
        for fam, lane in skipped:
            print("ours-cpu NOT PLANNED: %s %s: %s" % (fam, lane, cpu_arm_reason(fam, lane)))
        if "identical" not in modes:
            print("ours-cpu NOT PLANNED: --modes %s has no identical; the host bindings build "
                  "IDENTICAL only" % ",".join(modes))
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


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")

    if args.backfill_store:
        board_json = os.path.abspath(os.path.expanduser(args.backfill_store))
        store = args.opponent_store or os.path.join(os.path.dirname(os.path.dirname(board_json)),
                                                    "opponent-store.jsonl")
        imported, skipped = backfill_store(board_json, os.path.abspath(os.path.expanduser(store)))
        print("bench_board: backfill %s -> %s: imported %d opponent cells, skipped %d (a key "
              "field missing)" % (board_json, store, imported, skipped), flush=True)
        return 0

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
    modes = modes_for(vendor, args.modes)
    families = _csv(args.families, FAMILIES, "family")
    lanes = (_csv(args.lanes, TREE_LANES + TREE_TASK_LANES + CLASSICAL_LANES + MORE_LANES
                  + NEURAL_LANES + ALGOS_LANES, "lane")
             if args.lanes else None)
    datasets = _csv(args.datasets, DATASETS, "dataset")
    rows = parse_rows(args.rows)
    races = plan_races(vendor, modes, families, lanes, datasets, rows, args.neural_shape,
                       cpu_arm=not args.no_cpu_arm)
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
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    rpath = os.path.join(out, "board.json")
    result = load_result(rpath)

    python, wheel = setup_python(args, vendor, out, os.path.join(out, "logs", "setup.log"))
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
           "algos_driver": os.path.abspath(args.algos_driver),
           "algos_data": os.path.abspath(args.algos_data or os.path.join(cache_dir(args, out), "algos-data"))}
    box = box_fingerprint(ctx)
    ctx["box"] = box
    ctx["retime"] = args.retime_opponents
    ctx["store_path"] = os.path.abspath(os.path.expanduser(
        args.opponent_store or os.path.join(os.path.dirname(out), "opponent-store.jsonl")))
    ctx["data_sha"] = {}

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
    result["config"] = {"vendor": vendor, "modes": modes, "families": families,
                        "lanes": lanes, "datasets": datasets, "rows": rows,
                        "rounds": args.rounds, "seed": SEED, "infer": not args.no_infer,
                        "cpu_arm": not args.no_cpu_arm,
                        "neural_shape": args.neural_shape if "neural" in families else None,
                        "smoke": (bool(rows) and rows < TREE_ROW_FLOOR
                                  and any(r["family"] != "neural" for r in races))
                                 or ("neural" in families and args.neural_shape == "small"),
                        "data": data, "data_root": ctx["data_root"],
                        # sha256 of each data file a race read (the store's data key)
                        "data_sha256": ctx["data_sha"],
                        "opponent_store": ctx["store_path"],
                        "retime_opponents": ctx["retime"]}
    result["plan"] = [r["id"] for r in races]
    save_result(rpath, result)
    write_board(out, result)

    todo = []
    for r in races:
        prev = result["races"].get(r["id"])
        if prev and prev.get("status") == "done" and prev.get("params_check") != "MATCHED" \
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
        if prev and prev.get("status") == "failed" and args.skip_failed:
            print("bench_board: skip %s (failed earlier; --skip-failed)" % r["id"], flush=True)
            continue
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
    failed = [rid for rid, rec in result["races"].items()
              if rid in result["plan"] and rec.get("status") != "done"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
