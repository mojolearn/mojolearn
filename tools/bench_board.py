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
  neural     tools/bench_board_neural.py race (the classical racer's worker
             protocol and JSON shape): the wheel's public
             LanguageModelTrainer.train_step, LanguageModelTrainer.logits and
             mojolearn.linalg.matmul against torch eager fp32 (TF32 off) on the
             box's GPU (MPS, CUDA, ROCm); the LM model is
             tools/torch_lm_step_opponent.py's twin. `--neural-shape full` is
             the 20.45 M-parameter control shape and a 4096^3 GEMM; `small` is
             a smoke.
  parsing    tools/bench_all_summarize.py's FSPEED parser.
  rosters    tools/bench_all_ours.sh's per-lane NVIDIA rosters.

NEURAL, NOT COVERED YET: the Mamba blocks, TransformerBlock on its own,
SambaStack and SmallMLPTrainer (public, not raced); torch's compile, TF32 and
bf16 columns (another precision, or labeled nondeterministic by
tools/torch_lm_step_opponent.py); the GPT-3-small target shape (the board uses
the smaller control shape so one shape runs on every box, a 16 GB Mac included). tools/speed_gemm_arm.py, tools/speed_torch_seq.py
and bench/model/harness.py time source-built Mojo binaries, not the wheel, and
are not used here.

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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

SCHEMA = "mojolearn-bench-board/1"
SEED = 7                 # the drivers' own seed (lane_config seed=7); one seed only
DEFAULT_ROUNDS = 5
TREE_ROW_FLOOR = 1_000_000

TREE_LANES = ("gbdt-symmetric", "gbdt-depthwise", "gbdt-lossguide", "rf", "et", "iforest")
CLASSICAL_LANES = ("kmeans", "pca", "ols", "knn", "kde", "svc", "dbscan", "hdbscan")
NEURAL_LANES = ("lm-train-step", "lm-forward", "gemm")
FAMILIES = ("trees", "classical", "neural")
#: The data a neural lane reads (tools/bench_board_neural.py DATA_OF). No R2
#: dataset: the driver builds its inputs from seed 7.
NEURAL_DATA = {"lm-train-step": "bytes", "lm-forward": "bytes", "gemm": "gaussian"}
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
        "gbdt-depthwise": ("catboost-cpu", "xgboost-cpu"),
        "gbdt-lossguide": ("catboost-cpu", "xgboost-cpu", "lightgbm-cpu"),
        "rf": ("sklearn-rf-cpu", "lightgbm-cpu"),
        "et": ("sklearn-et-cpu", "lightgbm-cpu"),
        "iforest": ("sklearn-iforest-cpu",),
    },
    # NVIDIA: the vendor GPU path only; bench_all_ours.sh's tree_arms_for.
    "nvidia": {
        "gbdt-symmetric": ("catboost-gpu",),
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
        "gbdt-depthwise": ("catboost-cpu", "xgboost-gpu", "xgboost-cpu"),
        "gbdt-lossguide": ("catboost-cpu", "xgboost-gpu", "xgboost-cpu", "lightgbm-cpu"),
        "rf": ("sklearn-rf-cpu", "lightgbm-cpu"),
        "et": ("sklearn-et-cpu", "lightgbm-cpu"),
        "iforest": ("sklearn-iforest-cpu",),
    },
}


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

#: Neural opponents: torch eager fp32 (TF32 off) on the box's GPU, every
#: vendor and lane (tools/bench_board_neural.py; MPS, CUDA, ROCm).
NEURAL_OPPONENTS = {v: {lane: ("torch-eager-fp32",) for lane in NEURAL_LANES} for v in VENDORS}


def family_lanes(fam):
    return {"trees": TREE_LANES, "classical": CLASSICAL_LANES, "neural": NEURAL_LANES}[fam]


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


#: DBSCAN eps,min_samples per dataset, the values the published rows used
#: (bench_all_ours.sh; the racer refuses with no default).
DBSCAN_DEFAULTS = {"MOJOLEARN_CTD_DBSCAN_TAXI": "0.177,10",
                   "MOJOLEARN_CTD_DBSCAN_ISTELLA": "4.17,10"}

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


def our_arms(family, modes):
    """driver arm name -> numeric mode, for our arms in one race."""
    if family == "neural":
        return {"ours": "identical"}          # identical only, every vendor
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
               neural_shape="full"):
    check_neural_modes(families, modes)
    races = []
    for fam in families:
        for lane in family_lanes(fam):
            if lanes and lane not in lanes:
                continue
            if fam == "neural":
                # one race per lane: its own data, not taxi/Istella; IDENTICAL only
                ours = our_arms(fam, modes)
                opp = NEURAL_OPPONENTS[vendor][lane]
                ds = NEURAL_DATA[lane]
                races.append({
                    "id": race_id(fam, lane, ds, None, neural_shape),
                    "family": fam, "lane": lane, "dataset": ds, "rows": None,
                    "shape": neural_shape, "modes": ["identical"], "our_arms": ours,
                    "opponents": list(opp), "arms": list(ours) + list(opp),
                })
                continue
            for ds in datasets:
                opp = (TREE_OPPONENTS if fam == "trees" else CLASSICAL_OPPONENTS)[vendor][lane]
                ours = our_arms(fam, modes)
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
            "by_family": by_fam}


# ---------------------------------------------------------------------------
# Library / device labels for an arm name
# ---------------------------------------------------------------------------

def arm_library(arm):
    if arm in ("ours", "ours-ab", "ours-fast", "ours-base"):
        return "mojolearn"
    head = arm.split("-", 1)[0]
    return {"sklearn": "scikit-learn"}.get(head, head)


def arm_device(arm, vendor):
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
    bindings, warm, notes, verdict_line = {}, {}, [], None
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
    return {"arms": arms, "verdict": verdict, "verdict_line": verdict_line,
            "shape": shape, "notes": notes, "bindings": bindings, "warmup": warm}


def tree_cmd(ctx, race):
    ours = race["our_arms"]
    primary = ours["ours"]
    cmd = [ctx["python"], "-u", ctx["tree_driver"], "--lane", race["lane"],
           "--dataset", race["dataset"]]
    if race["rows"]:
        cmd += ["--rows", str(int(race["rows"]))]
    if race["opponents"]:
        cmd += ["--devices", tree_devices(ctx["vendor"], race["lane"]),
                "--arms", ",".join(race["opponents"])]
    else:
        cmd += ["--ours-only"]
    if "ours-ab" in ours:
        cmd += ["--ours-ab", "numeric_mode='%s'" % ours["ours-ab"]]
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
        if mode:
            b = parsed["bindings"].get(arm) or {}
            cell["binding"] = b
            cell["mode_witness"] = b.get("compiled")
            cell["installed_wheel"] = _from_wheel(b.get("path"), ctx)
            if b and (b.get("compiled") != mode or b.get("resolved") != mode):
                cell["status"] = "MODE-MISMATCH(requested %s, compiled %s)" % (mode, b.get("compiled"))
        cells.append(cell)
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
    env = {k: os.environ.get(k, v) for k, v in DBSCAN_DEFAULTS.items()}
    ceiling = 600 + max(rsec, 600) * len(race["arms"]) + rsec * ctx["rounds"] * len(race["arms"]) + 900
    return cmd, env, ceiling


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
        if mode:
            cell["mode_witness"] = info.get("numeric_mode_used")
            cell["installed_wheel"] = _from_wheel(info.get("module_path"), ctx)
            if info and info.get("numeric_mode_used") not in (None, mode):
                cell["status"] = "MODE-MISMATCH(requested %s, read back %s)" % (
                    mode, info.get("numeric_mode_used"))
        cells.append(cell)
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
    }


_SETTINGS_CACHE = {}


NEURAL_SETTINGS = {
    "lm-train-step": {
        "ours_call": "mojolearn.LanguageModelTrainer(resident=True, step_result='lean').train_step(ids)",
        "torch_call": "tools/torch_lm_step_opponent.py build_model; zero_grad; forward + mean CE; "
                      "backward; torch.optim.AdamW step; loss.item()",
        "optimizer": "AdamW lr 1e-3, betas (0.9, 0.999), eps 1e-8, weight decay 0.01 on both",
        "clock": "ids host to device, one training step, loss back on the host, synchronized; "
                 "parameters and AdamW state device-resident on both sides; round r is step r+1",
        "quality": "loss_first_step, loss_last_step (same init, same batches), "
                   "loss_last_abs_diff_vs_ours"},
    "lm-forward": {
        "ours_call": "mojolearn.LanguageModelTrainer(resident=True).logits(ids)",
        "torch_call": "no_grad forward of the same twin to logits; logits.cpu()",
        "clock": "ids host to device, forward, float32 logits [B, L, V] back on the host",
        "quality": "mean_nll of the logits (float64), max_abs_diff_vs_ours"},
    "gemm": {
        "ours_call": "mojolearn.linalg.matmul(a, b)",
        "torch_call": "a.to(dev) @ b.to(dev), .cpu()",
        "clock": "A and B host to device, the product, C back on the host",
        "quality": "max_rel_err_vs_fp64, max_abs_diff_vs_ours"},
}
NEURAL_SHAPE_TEXT = {
    "full": {"lm": "B1 L2048 DM384 H6 KV6 HD64 FF1024 8 layers V8192, 20,453,376 parameters "
                   "(tools/torch_lm_step_opponent.py control shape)", "gemm": "m=n=k=4096"},
    "small": {"lm": "B2 L64 DM64 H4 KV2 HD16 FF128 2 layers V256 (smoke)", "gemm": "m=n=k=256 (smoke)"},
}


def race_settings(ctx, race):
    key = (race["family"], race["lane"], race.get("shape"))
    if key not in _SETTINGS_CACHE:
        s = {"seed": SEED, "rounds": ctx["rounds"], "warmup_rounds": 1,
             "interleaved": True, "rows_cap": race["rows"]}
        if race["family"] == "neural":
            s["driver"] = "tools/bench_board_neural.py"
            s["numeric_mode"] = "identical (the only tier the neural surface builds)"
            s["opponent_mode"] = ("torch eager float32, TF32 off: tools/torch_lm_step_opponent.py's "
                                  "eager_fp32, the opponent's fast setting at our precision")
            s["shape"] = race.get("shape")
            s["shape_dims"] = NEURAL_SHAPE_TEXT[race.get("shape") or "full"][
                "gemm" if race["lane"] == "gemm" else "lm"]
            s.update(NEURAL_SETTINGS[race["lane"]])
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
                               "Istella train split; knn 400,000 x 4,000 queries, k=10; kde "
                               "100,000 x 2,000; svc 10,000 + 10,000; dbscan 1,000,000)")
            if race["lane"] == "dbscan":
                s["dbscan_eps_min_samples"] = {k: os.environ.get(k, v) for k, v in DBSCAN_DEFAULTS.items()}
        _SETTINGS_CACHE[key] = s
    return dict(_SETTINGS_CACHE[key])


def add_ratios(cells):
    """ratio_ours_identical_over / ratio_ours_fast_over: our median divided by
    this OPPONENT arm median, computed only when both sides completed every round."""
    ok = {c["arm"]: c for c in cells if c["status"] == "ok" and c["median_ms"]}
    ours_id = next((c for c in ok.values() if c["library"] == "mojolearn" and c["mode"] == "identical"), None)
    ours_fast = next((c for c in ok.values() if c["library"] == "mojolearn" and c["mode"] == "fast"), None)
    for c in cells:
        c["ratio_ours_identical_over"] = None
        c["ratio_ours_fast_over"] = None
        # Opponents only. FAST over IDENTICAL is the cost of identity, an
        # internal number that never reaches a board (ENGINEERING_RULES 0b-iii).
        if c["arm"] not in ok or c["library"] == "mojolearn":
            continue
        if ours_id and c is not ours_id:
            c["ratio_ours_identical_over"] = ours_id["median_ms"] / c["median_ms"]
        if ours_fast and c is not ours_fast:
            c["ratio_ours_fast_over"] = ours_fast["median_ms"] / c["median_ms"]
    return cells


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_race(ctx, race):
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
    return rec


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
    "explained_variance_ratio_sum": "higher is better", "recall_at_10": "higher is better",
    "mean_log_likelihood": "higher is better",
    "loss_first_step": "same init and batches on every arm",
    "loss_last_step": "same init and batches on every arm",
    "loss_last_abs_diff_vs_ours": "0 is our value exactly",
    "mean_nll": "lower is better", "max_abs_diff_vs_ours": "0 is our output exactly",
    "max_rel_err_vs_fp64": "lower is better",
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
    opp = ["catboost", "xgboost", "lightgbm", "scikit-learn", "cuml-cu12", "torch", "numpy"]
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
    L.append("- Neural: our IDENTICAL arm only (the neural surface builds no other tier, on any "
             "vendor) against torch eager float32 with TF32 off on this box's GPU (MPS, CUDA or "
             "ROCm). Every clock is host in, host out: ids or operands to the device, the call, "
             "the result back on the host, synchronized. The LM lanes start from the same "
             "parameters and read the same batches on every arm, so their losses and logits are "
             "comparable; `max_abs_diff_vs_ours` is the opponent's output against ours.")
    L.append("- `installed_wheel` confirms our binding loaded from site-packages, not the repo tree.")
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
            fast = next((c for c in rc if c["library"] == "mojolearn" and c["mode"] == "fast"), None)
            ident = next((c for c in rc if c["library"] == "mojolearn" and c["mode"] == "identical"), None)
            opps = [c for c in rc if c["library"] != "mojolearn"]
            q = lambda c: _f((c.get("quality") or {}).get(m), 6) if c else "-"   # noqa: E731
            L.append("| %s | %s | %s | %s%s | %s | %s | %s |" % (
                races[rid]["family"], races[rid]["lane"], races[rid]["dataset"], m,
                (" (%s)" % QUALITY_NOTE[m]) if m in QUALITY_NOTE else "",
                q(fast), q(ident),
                "; ".join("%s %s" % (c["arm"], q(c)) for c in opps) or "-"))
    L.append("")

    for fam in FAMILIES:
        fam_races = [races[r] for r in sorted(races) if races[r]["family"] == fam]
        if not fam_races:
            continue
        L.append("## %s" % {"trees": "Trees", "classical": "Classical", "neural": "Neural"}[fam])
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
                     "ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | "
                     "comparability | installed_wheel | status |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for c in rc:
                L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    _arm_label(c), c["library"], c["device"], c["mode"],
                    _f(c["median_ms"]),
                    "%s..%s" % (_f(c["min_ms"]), _f(c["max_ms"])) if c["min_ms"] is not None else "-",
                    c["rounds"],
                    _f(c.get("ratio_ours_identical_over"), 3),
                    _f(c.get("ratio_ours_fast_over"), 3),
                    _q(c.get("quality")), _f(c.get("hash_stable")),
                    clean(c.get("verdict")), clean(c.get("installed_wheel", "-")),
                    clean(c["status"])))
            if rr.get("fit_verdict_line"):
                L.append("")
                L.append("FSPEED-FIT-VERDICT: `%s`" % clean(rr["fit_verdict_line"]))
            L.append("")
    L.append("## Not covered by this board")
    L.append("")
    L.append("- Neural: the Mamba blocks, TransformerBlock on its own, SambaStack and "
             "SmallMLPTrainer are public and not raced yet; torch's compile, TF32 and bf16 "
             "columns are another precision or labeled nondeterministic and are not raced; the "
             "GPT-3-small target shape is not on the board (it uses the smaller control shape "
             "so one shape runs on every box, a 16 GB Mac included).")
    L.append("")
    return "\n".join(L) + "\n"


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
                   help="neural family shape: full (the 20.45 M-parameter LM control shape, "
                        "4096^3 GEMM) or small (a smoke; the board says SMOKE). --rows does not "
                        "apply to neural lanes")
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                   help="timed rounds after one warm-up (default 5)")
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
    p.add_argument("--dry-run", action="store_true", help="print the plan and run nothing")
    p.add_argument("--render-only", action="store_true", help="re-render BOARD.md from board.json")
    p.add_argument("--tree-driver", default=os.path.join(REPO, "bench", "speed", "forest_speed_arm.py"),
                   help=argparse.SUPPRESS)
    p.add_argument("--classical-driver", default=os.path.join(HERE, "classical_two_datasets.py"),
                   help=argparse.SUPPRESS)
    p.add_argument("--neural-driver", default=os.path.join(HERE, "bench_board_neural.py"),
                   help=argparse.SUPPRESS)
    return p


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
        t = NEURAL_SHAPE_TEXT[args.neural_shape]
        print("neural: IDENTICAL only; opponent torch eager fp32 (TF32 off) on the GPU; "
              "shape %s (LM %s; GEMM %s)" % (args.neural_shape, t["lm"], t["gemm"]))
    print("mojolearn=%s (from the installed wheel; %s)" % (
        args.mojolearn_version or "<--mojolearn-version>",
        "python %s" % args.python_env if args.python_env else "venv %s" % (args.venv or "<cache>/venv")))
    for idx, reqs in opponent_requirements(vendor, opponent_pins()):
        print("opponents pinned: %s%s" % (" ".join(reqs), (" (index %s)" % idx) if idx else ""))
    ts = args.torch_spec if args.torch_spec is not None else DEFAULT_TORCH_SPEC[vendor]
    print("torch: %s" % (ts or "the image's own (use --system-site-packages)"))
    for ds, rec in data.items():
        print("data %-8s %s %s (R2 key %s)" % (ds, "present" if rec["present"] else "MISSING",
                                              rec["path"], rec["r2_key"]))
    print("")
    for r in races:
        print("RACE %-40s arms=%s" % (r["id"], ",".join(
            "%s[%s]" % (a, r["our_arms"][a]) if a in r["our_arms"] else a for a in r["arms"])))
    print("")
    for fam, f in sorted(s["by_family"].items()):
        print("family %-10s races=%d cells=%d" % (fam, f["races"], f["cells"]))
    print("TOTAL races=%d cells=%d" % (s["races"], s["cells"]))


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")

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
    lanes = _csv(args.lanes, TREE_LANES + CLASSICAL_LANES + NEURAL_LANES, "lane") if args.lanes else None
    datasets = _csv(args.datasets, DATASETS, "dataset")
    rows = parse_rows(args.rows)
    races = plan_races(vendor, modes, families, lanes, datasets, rows, args.neural_shape)
    # taxi and Istella-S are read by trees and classical only; a neural-only
    # run needs no R2 data.
    needed = [ds for ds in datasets if any(r["dataset"] == ds for r in races
                                           if r["family"] != "neural")]
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
           "nice": args.nice, "ptxas": ptxas,
           "tree_driver": os.path.abspath(args.tree_driver),
           "classical_driver": os.path.abspath(args.classical_driver),
           "neural_driver": os.path.abspath(args.neural_driver)}
    box = box_fingerprint(ctx)

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
                        "rounds": args.rounds, "seed": SEED,
                        "neural_shape": args.neural_shape if "neural" in families else None,
                        "smoke": (bool(rows) and rows < TREE_ROW_FLOOR
                                  and any(r["family"] != "neural" for r in races))
                                 or ("neural" in families and args.neural_shape == "small"),
                        "data": data}
    result["plan"] = [r["id"] for r in races]
    save_result(rpath, result)
    write_board(out, result)

    todo = []
    for r in races:
        prev = result["races"].get(r["id"])
        if prev and prev.get("status") == "done":
            print("bench_board: skip %s (done %s)" % (r["id"], prev.get("finished")), flush=True)
            continue
        if prev and prev.get("status") == "failed" and args.skip_failed:
            print("bench_board: skip %s (failed earlier; --skip-failed)" % r["id"], flush=True)
            continue
        todo.append(r)
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
