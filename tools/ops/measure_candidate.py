#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/ops/measure_candidate.py -- put a CANDIDATE build's wheels on a measurement box and print the one submit
line of tools/ops/paper_compare_job.sh (lane measure-prep, 2026-11-09; docs/plans/PRE_0867_MEASUREMENT.md).

    python3 tools/ops/measure_candidate.py stage nvidia|amd --version V --source-commit SHA [--wheel-dir DIR]
        [--lane LANE] [--tree DIR] [--b-arm NAME=DIR@SHA ...] [--tag TAG] [--parts PARTS] [--dry-run]
        [--rows FILE [--prior-wheel NAME=DIR@SHA ...]]

The wheels are the ones tools/release.py packs (its linux-pack step, Release.split_final): for a candidate build
(`tools/release.py candidate`, a local version X.Y.Z.devN+<sha12>[.label]) ~/mojolearn-evidence/release-candidates/
<version>/<commit12>/linux/final/, for a frozen release ~/mojolearn-evidence/release/<version>/<commit12>/linux/final/:
the core and the vendor wheel. They are validated here first with tools/candidate_wheel_install.py build_manifest (wheel RECORD, metadata,
tags, the frozen payload's source commit), then copied through the vendor wrapper's own channel and sha256-checked
on the box by tools/ops/release_gate.py Runner.stage, the same code path the release gate uses for the same wheels
(no second downloader; nothing is fetched from GitHub or R2 here). A B set (--b-arm, an A/B arm of
bench/results/paper_compare/owed-pre0867.json built from another commit the same way) goes beside them and into
the PC_B_ARMS file.

NVIDIA: `nvidia_central.sh sync <lane> <tree>` first puts the lane tree /root/mojolearn-<lane> at --tree (the
worktree of the branch whose tools/ the job runs). AMD: the droplet's /root/mojolearn must already be at that
branch's pushed head (`tools/do_amd_steward.sh update <sha>`); this tool checks it and refuses otherwise.
Nothing here rents, releases or extends a box. It prints the submit line; the orchestrator runs it.

Before any copy it also (fix lane tooling-measure-stage, 2026-11-13, after the measurement of 4c2340386abc):
  * refuses a --b-arm NAME that is not one of the owed file's b_arms (six B sets were staged under invented names and
    the job reported every one "not planned");
  * reads each staged arm's run variables from its owed entry (lane fx-tools-round3, 2026-11-20): a built B set's
    `run_env` ("VAR=VALUE[,..]") goes on that arm's PC_B_ARMS line as the 4th field env:..., so only that arm's board
    units and driver rows run with it; an env arm's `env` as before. It refuses a list the job would not admit, a box
    value other than @cgroup_cores, and an entry whose own text names a MOJOLEARN_EXPERIMENT_* variable its env does
    not set (gemm-leaf512 was staged twice without MOJOLEARN_EXPERIMENT_GEMM_PROFILE and lost its gemm row);
  * checks the box's free disk on /root: at least --min-free-gb (40), or the largest results dir already under the
    tree's box-out/ plus 20 GB, and prints what can be pruned (old results dirs, staging tarballs of
    box_persist_run.sh, old /root/pc-in sets, smoke caches); it refuses below that unless --allow-low-disk (the
    80 GB L40S pod filled twice);
  * stages every dataset the job reads through `tools/dataset_store.sh stage` (the first AMD submit died in 1 s on a
    missing taxi; half the smoke shards refused on a missing higgs). The key list is not written here: it is
    tools/measure_owed.py data_keys over the paper recipes and the owed rows, through the board's own data_needs.
    The pins are the candidate's full source tree's bench/results/dataset_store/manifest.tsv
    (<work dir>/source/<commit12>/, written by tools/release.py candidate; --source-tree overrides), since the lean
    tooling worktree has no bench/results/. A file already on the box at its pins is kept, not refetched.
    The keys follow --parts (PC_PARTS of the job, carried on the submit line; harness-small-row, 2026-11-16, after
    two board-all submits died on corpus/enwik8 and corpus/pile_github): the paper parts read the paper recipes'
    and owed rows' datasets; board-all (or any all-* part; that block exits before the paper parts) reads the full
    board's, from the same planner call the job's all-plan makes (tools/paper_compare.py plan_all) through the same
    data_keys, so the algos corpus keys (staged under <tree>/training/corpus/ on the box) are in the list.

PARALLEL PLAN (HARD RULE, 2026-11-07): this tool plans and copies; it runs no workload. The planner call is CPU,
seconds, one process. The dataset staging is one tools/dataset_store.sh stage call over every key (its downloads run
on the box). The wheel copies are a few files through the wrapper. Nothing here is split further because nothing here
is a unit of work worth a core: the measurement's parallel plan is tools/ops/paper_compare_job.sh's.

A restage never moves a tree a job is reading (fix lane fix-board-quality-scorers, 2026-11-13: a restage during job
0001 synced /root/mojolearn-<lane> under it and bench_board stopped with "source changed during measurement").
`stage nvidia` reads `nvidia_central.sh queue` BEFORE the sync and refuses while any job of --lane is queued,
starting or running on a pod, unless --allow-running. AMD stages no tree (it checks the droplet's head), so it has
no such check.

NVIDIA is measured on an L40S only (fix lane tooling-measure-pod-l40s-only, 2026-11-13: `nvidia_central.sh up`
rented a 2x RTX 6000 Ada when the L40S was out of stock). The paper's NVIDIA device is the L40S and the rule is
"compare on the same device" (AMD: MI300X). `stage nvidia` reads the pod's `nvidia-smi` names after the sync and
refuses any other GPU unless --allow-hw-diff (or PC_ALLOW_HW_DIFF=1); then the submit line carries
PC_ALLOW_HW_DIFF=1, tools/ops/paper_compare_job.sh goes on, and tools/paper_compare.py flags every row HW-DIFF.

Times are compared only on the same box (AGENTS.md, Andrew, 2026-11-14): the submit line carries PC_PRIOR_VERSIONS
(--prior-versions, default 0.8.55: a measurement job times ONE prior, the paper baseline; Andrew, 2026-11-16) and
PC_PRIOR_VERSION (its last entry); the job installs those releases from PyPI and times every paper row for each on
the same box in the same job, and the compare's columns are those on-box boards. An explicit extra prior is
accepted: a last entry other than 0.8.55 is the last-release arm, which also times the owed board rows. The prior
board doubles the board time (accepted). Its evidence counts in the disk need too.

OUR OWN EARLIER BUILT WHEELS beside the candidate (--prior-wheel NAME=DIR@SHA, repeatable, the targeted job only;
lane fx-same-pod-prior, Andrew 2026-11-19: a speed verdict compares two wheels timed on the SAME host in the SAME job;
the stored 0.8.55 board is a screen only, and 0.8.55 is never installed or timed). DIR is a measurement build on disk
(tools/release.py candidate's .../<commit12>/linux/final) of commit SHA; it is validated like a B set (check_built,
pick_wheels, build_manifest), copied beside the candidate under p-<NAME>/ and listed in the PC_PRIOR_WHEELS file
(`<name> <dir> <sha>`). The job installs it as a B set's venv, times every listed row on it right after the
candidate chunk by chunk (our GPU arm only, one scored run, the same small-row median rule), and the compare takes the
first one as the verdict baseline (tools/paper_compare.py --verdict-prior): REGRESSION / FASTER only from that pair,
with both box probes. No owed-file entry and no define. The estimate and the disk need count each one as an arm.
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "ops"))
import candidate_wheel_install as C  # noqa: E402
import measure_owed as MO  # noqa: E402
import paper_compare as PC  # noqa: E402
import release_gate as RG  # noqa: E402
import targeted_rows as TR  # noqa: E402

RELEASE_ROOT = Path.home() / "mojolearn-evidence" / "release"
#: the paper baseline only (Andrew, 2026-11-16: one prior, 0.8.55; 0.8.66 is never timed): re-timed on the box
#: beside the candidate (paper_compare_job.sh)
DEFAULT_PRIOR_VERSIONS = PC.PAPER_VERSION
#: tools/release.py candidate's work root: a local version (X.Y.Z.devN+<sha12>[.label]) is a candidate build there
CANDIDATE_ROOT = Path.home() / "mojolearn-evidence" / "release-candidates"


def default_wheel_dir(version, commit):
    root = CANDIDATE_ROOT if "+" in version else RELEASE_ROOT
    return root / version / commit[:12] / "linux" / "final"


def dir_version(wheel_dir, fallback):
    """The version of a wheel dir tools/release.py candidate wrote (its candidate.json), else FALLBACK."""
    try:
        import json
        return json.loads((Path(wheel_dir) / "candidate.json").read_text())["version"]
    except (OSError, ValueError, KeyError):
        return fallback
IN_BASE = "/root/pc-in"
#: the STORED baseline of the targeted job (Andrew, 2026-11-17, AGENTS.md part 8 "0.8.55 is not timed again"): the
#: 0.8.55 board of the 2026-11-17 L40S full board (host 26ee0a3b0f86, driver 595.112.07, kernel 6.8.0-149). 44 MB, too
#: large for bench/results/, so stage uploads it from the evidence path beside the wheels
STORED_BASELINE = (Path.home() / "mojolearn-evidence" / "measure-candidate" / "build-d95f805cce" / "full"
                   / "board-all-prior-0.8.55" / "board.json")
#: the candidate board of that full board: the stored wall seconds the A/B plan's minutes are estimated from
STORED_TIMES = STORED_BASELINE.parents[1] / "board-all" / "board.json"
#: the boards of the first targeted A/B job (ab1 of d652a4b0a1, L40S pod 37f71a6102af, a 14-core cgroup; lane
#: fb15-ab-verdicts, 2026-11-18): its host-bound races took up to 4.4x the 28-core pod's stored seconds
#: (elliptic-envelope istella 1748 s against 510, hdbscan istella 726 against 164), so each race is estimated from
#: the largest stored wall_s over every board given
STORED_TIMES_AB1 = sorted((Path.home() / "mojolearn-evidence" / "measure-candidate" / "build-d652a4b0a1" / "ab1")
                          .glob("board-tg*/board.json"))
AMD_TREE = "/root/mojolearn"
#: the job's --data-root (tools/ops/paper_compare_job.sh PC_DATA_ROOT default); dataset_store.sh stages there
DATA_ROOT = "/root/datasets/gbm-bench"
OWED_NAME = "owed-pre0867.json"
#: free disk the job needs on /root at least, and on top of the largest earlier results dir (GB): the 2026-11-13
#: NVIDIA job wrote 14 GB of board evidence and 12 GB of smoke scratch, and its persist staging held 36 GB more
MIN_FREE_GB = 40
HEADROOM_GB = 20
#: THE TARGETED JOB'S DISK FOLLOWS ITS ARMS (lane fx-pod-job, 2026-11-18; it was MIN_FREE_GB whatever was staged):
#: an ours-only venv with its install cache (1.3 GB measured on the L40S pod, ideas-cycle-time-RESULT.md 1e); the
#: smoke shards' scratch (12 GB on 2026-11-13; pruned after the smoke, before any board); a timed race's evidence
#: (paper_compare_job.sh PC_EVIDENCE_MB_PER_RACE; ab1 measured 8.5 MB a race on average, 31 at most, when every race
#: archived its arrays; since 2026-11-20 an own-arms race keeps no archive or array, tools/bench_board_evidence.py
#: scores_opponent, so its row, logs and JSON: 4 MB); a board dir's
#: fixed part (ab1: 98-166 MB a unit board with its races); the job's own reserve (PC_DISK_RESERVE_GB). The B arms'
#: venvs are made by window (paper_compare_job.sh arm-window), so the need counts one, not one per arm.
VENV_GB = 1.5
SMOKE_SCRATCH_GB = 12
EVIDENCE_MB_PER_RACE = 4
BOARD_MB = 200
RESERVE_GB = 5
#: the paper's NVIDIA device (tools/ops/paper_compare_job.sh PC_GPU_TOKEN default); `stage nvidia` refuses others
PAPER_NVIDIA_GPU = "L40S"
GPU_SCRIPT = "nvidia-smi --query-gpu=name --format=csv,noheader"


def gpu_verdict(text, token=PAPER_NVIDIA_GPU):
    """(names, ok) from GPU_SCRIPT's output: ok only when at least one GPU is listed and every one is TOKEN."""
    names = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith(("nvidia-smi", "NVIDIA-SMI"))]
    return names, bool(names) and all(token in n for n in names)


def source_tree(wheel_dir, commit, override=None):
    """The candidate's full source tree tools/release.py candidate checked out: <work dir>/source/<commit12>, where
    the wheel dir is <work dir>/<commit12>/linux/final."""
    if override:
        return Path(override).expanduser().resolve()
    return Path(wheel_dir).expanduser().resolve().parents[2] / "source" / commit[:12]


def results_file(name, *roots):
    """bench/results/<name> from the first root that has it (the job tree, else the candidate's source tree)."""
    for r in roots:
        f = Path(r) / "bench" / "results" / name
        if f.is_file():
            return f
    raise SystemExit(f"measure-candidate: no bench/results/{name} under " + " or ".join(str(r) for r in roots))


def check_b_arms(names, owed_file, envs=None):
    """Refuse a B set name the owed file does not list (the job would report its rows "not planned"), an arm the
    file classifies other than RUN (lane fb8-owed-arms, 2026-11-17: `owed` DECIDED already has its L40S A/B, PAUSED
    belongs to a paused programme, NO-ROW has no board race; lane fb15-ab-verdicts, 2026-11-18: HELD has its A/B and
    waits on a decision, INVALID measured nothing and a RUN arm replaces it; nothing is timed twice), and, with ENVS ({name: the
    'VAR=VALUE' of an env arm, or None for a built set}), an arm staged in the other form than the file's (`env`).
    Returns {name: the run variables of that arm or None} (check_run_env: a built set's `run_env` goes on its PC_B_ARMS
    line as the 4th field env:VAR=VALUE, so only that arm's venv runs with it)."""
    arms = PC.load_json(str(owed_file)).get("b_arms") or {}
    known = sorted(arms)
    bad = [n for n in names if n not in known]
    if bad:
        raise SystemExit(f"measure-candidate: --b-arm {', '.join(bad)} not in {owed_file} b_arms; "
                         f"the names are: {', '.join(known)}")
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f"measure-candidate: --b-arm {', '.join(dup)} given twice")
    done = [f"{n} ({arms[n]['owed']}: {str(arms[n].get('evidence', ''))[:120]})" for n in names
            if arms[n].get("owed", "RUN") != "RUN"]
    if done:
        raise SystemExit("measure-candidate: --b-arm not owed: " + "; ".join(done))
    for n, env in (envs or {}).items():
        want = arms[n].get("env")
        if (env or None) != (want or None):
            raise SystemExit(f"measure-candidate: --b-arm {n} is " + (f"the env arm {n}=env:{want}" if want else
                             "a built B set NAME=DIR@SHA (tools/release.py candidate --owed-arm), not an env arm"))
    doc = PC.load_json(str(owed_file))
    return {n: check_run_env(n, arms[n], doc.get("rows") or [],
                             built=((envs or {})[n] is None) if n in (envs or {}) else not arms[n].get("env"))
            for n in names}


#: one VAR=VALUE[,VAR=VALUE] list, the form tools/ops/paper_compare_job.sh's b-arms block admits (an env arm's second
#: field, a wheel arm's 4th field); the only box value the job resolves is @cgroup_cores
ENV_LIST_RE = re.compile(r"[A-Z_][A-Z0-9_]*=[^,\s]*(?:,[A-Z_][A-Z0-9_]*=[^,\s]*)*")
BOX_VALUES = ("@cgroup_cores",)
#: a variable an arm's text says its runs need (the admission switches of an experimental build; lane fx-tools-round3)
NAMED_ENV_RE = re.compile(r"\b(MOJOLEARN_EXPERIMENT_[A-Z0-9_]+)=")


def check_run_env(name, entry, rows, built=True):
    """The run variables ('VAR=VALUE[,..]' or None) a staged arm's runs need, from its owed entry: `run_env` of a
    built B set (a wheel arm that runs with variables, the job's 4th PC_B_ARMS field) or `env` of an env arm.
    Refused (lane fx-tools-round3, 2026-11-20; gemm-leaf512 was staged twice without the
    MOJOLEARN_EXPERIMENT_GEMM_PROFILE its status named, and its arm lost neural/gemm and its driver cases to the
    profile admission with no line saying why): a list the job would not admit, a box value other than @cgroup_cores,
    `run_env` on an env arm, and an entry whose own text (status, evidence, or a row naming the arm) names a
    MOJOLEARN_EXPERIMENT_* variable that its env / run_env does not set."""
    if entry.get("run_env") and not built:
        raise SystemExit(f"measure-candidate: --b-arm {name}: run_env belongs to a built B set; an env arm sets `env`")
    env = (entry.get("run_env") if built else entry.get("env")) or None
    if env is not None:
        if not ENV_LIST_RE.fullmatch(env):
            raise SystemExit(f"measure-candidate: --b-arm {name}: the owed entry's "
                             f"{'run_env' if built else 'env'} {env!r} is not VAR=VALUE[,VAR=VALUE] (the job refuses it)")
        odd = sorted({t for t in re.findall(r"@[A-Za-z_][A-Za-z0-9_]*", env) if t not in BOX_VALUES})
        if odd:
            raise SystemExit(f"measure-candidate: --b-arm {name}: {', '.join(odd)} in {env!r} is no value the job "
                             f"supplies (only {', '.join(BOX_VALUES)})")
    text = " ".join(str(entry.get(k) or "") for k in ("status", "evidence"))
    text += " " + " ".join(str(r.get("decides") or "") + " " + str(r.get("note") or "")
                           for r in rows if name in (r.get("b_arms") or []))
    have = {kv.split("=", 1)[0] for kv in (env or "").split(",") if kv}
    need = sorted(set(NAMED_ENV_RE.findall(text)) - have)
    if need:
        raise SystemExit(f"measure-candidate: --b-arm {name}: its owed entry names {', '.join(need)} for its runs, and "
                         f"its {'run_env' if built else 'env'} does not set it; the arm would lose those rows. Add it "
                         f"to b_arms.{name}.{'run_env' if built else 'env'} in the owed file")
    return env


#: tools/ops/paper_compare_job.sh's parts: its default (the paper job) and the full-board screen's (board-all expands
#: to the all-* parts; that block exits before the paper parts run)
PAPER_PARTS = ("plan", "install", "smoke", "board", "prior", "owed", "drivers", "compare")
ALL_PARTS = ("board-all", "all-plan", "all-install", "all-smoke", "all-board", "all-prior", "all-compare")
#: the targeted job's parts (PC_PARTS=targeted with PC_ROWS: tools/ops/targeted_rows.py; that block exits before the
#: paper parts too)
TG_PARTS = ("targeted", "tg-install", "tg-prep", "tg-smoke", "tg-board", "tg-compare", "tg-drivers")


def parse_parts(text):
    """[part] of a PC_PARTS value ('' = the job's default, the paper parts); refuses a name the job does not know."""
    parts = [p for p in (text or "").split(",") if p]
    known = PAPER_PARTS + ALL_PARTS + TG_PARTS
    bad = [p for p in parts if p not in known]
    if bad:
        raise SystemExit(f"measure-candidate: --parts {','.join(bad)} is not a part of tools/ops/paper_compare_job.sh "
                         f"({','.join(known)})")
    return parts


def runs_targeted(parts):
    return any(p in TG_PARTS for p in parts)


def runs_board_all(parts):
    return any(p in ALL_PARTS for p in parts)


def dataset_keys(vendor, owed_file, recipes_file, planner_file, b_names=(), last_prior=False, parts=(), rows_file=None):
    """The R2 keys the job's races read (tools/measure_owed.py data_keys: the board's own data_needs over the races
    the job plans). The paper parts: the paper recipes and the owed rows the job plans for these B sets and this prior
    choice. board-all (any all-* part): every race the board plans (tools/paper_compare.py plan_all, the job's
    all-plan), with the algos corpus keys. targeted (any tg-* part): the listed rows (ROWS_FILE) the board plans and
    the owed rows of the staged B sets, as the job's targeted plan and data-keys read them."""
    planner = MO.load_planner(str(planner_file))
    if runs_targeted(parts):
        want = {r["id"] for r in TR.read_rows(rows_file)}
        races = [r for r in PC.plan_all(planner, vendor)[4] if r["race_id"] in want]
        doc = PC.load_json(owed_file) if b_names else {"rows": []}
        keys = MO.data_keys(doc, {"recipes": races}, planner, vendor, DATA_ROOT, list(b_names), False)
    elif runs_board_all(parts):
        races = PC.plan_all(planner, vendor)[4]
        keys = MO.data_keys({"rows": []}, {"recipes": races}, planner, vendor, DATA_ROOT)
    else:
        keys = MO.data_keys(PC.load_json(owed_file), PC.load_json(recipes_file), planner, vendor, DATA_ROOT,
                            list(b_names), last_prior)
    return [k for k, _ in keys]


def disk_script(results_base):
    """The box-side report the disk check parses: FREE_KB, then RES / STAGING / CACHE lines (<kb> <path>)."""
    b = shlex.quote(results_base)
    return ("df -Pk /root | awk 'NR==2 {print \"FREE_KB \" $4}'; "
            f"for d in {b}/*/; do [ -d \"$d\" ] && du -sk \"$d\" | sed 's/^/RES /'; done; "
            "for f in /root/box-persist/state/*/upload.tgz; do [ -f \"$f\" ] && du -sk \"$f\" | sed 's/^/STAGING /'; done; "
            "for d in /root/pc-in/*/ /root/pc-cache/c*/; do [ -d \"$d\" ] && du -sk \"$d\" | sed 's/^/CACHE /'; done; true")


def targeted_disk_gb(venvs, window, upload_bytes, timed_races, boards):
    """The targeted job's disk need in GB: the reserve and the staged files throughout; the VENVS made in tg-install
    (the candidate and the priors); then the larger of the smoke's scratch (pruned before the boards) and the board
    phase (one B-arm install WINDOW of venvs, the TIMED_RACES' evidence and the BOARDS' fixed parts)."""
    board_phase = VENV_GB * window + (timed_races * EVIDENCE_MB_PER_RACE + boards * BOARD_MB) / 1024.0
    return RESERVE_GB + upload_bytes / 2.0 ** 30 + VENV_GB * venvs + max(SMOKE_SCRATCH_GB, board_phase)


def disk_verdict(text, tag, commit12, min_free_gb=MIN_FREE_GB, headroom_gb=HEADROOM_GB, need_gb=None):
    """(free GB, needed GB, [(GB, path, why)] prunable) from disk_script's output. Needed: NEED_GB when given (the
    targeted job's own, targeted_disk_gb), else min_free_gb, or the largest results dir already on the box +
    headroom_gb. Prunable: every other results dir, every staging tarball, other commits' /root/pc-in sets and the
    smoke caches c2.. (the timed boards use c1)."""
    free_kb, prune, largest = None, [], 0
    for line in text.splitlines():
        f = line.split(None, 2)
        if len(f) == 2 and f[0] == "FREE_KB" and f[1].isdigit():
            free_kb = int(f[1])
        elif len(f) == 3 and f[1].isdigit():
            kind, kb, path = f[0], int(f[1]), f[2].rstrip("/")
            gb = kb / 1048576
            if kind == "RES":
                largest = max(largest, gb)
                if os.path.basename(path) != tag:
                    prune.append((gb, path, "an earlier results dir (fetch it home first: tools/box_persist.py fetch)"))
            elif kind == "STAGING":
                prune.append((gb, path, "a box_persist_run.sh staging tarball (a failed or killed upload)"))
            elif kind == "CACHE":
                base = os.path.basename(path)
                if path.startswith(IN_BASE + "/") and base != commit12:
                    prune.append((gb, path, "another candidate's staged wheels"))
                elif re.fullmatch(r"c([2-9]|[1-9][0-9]+)", base):
                    prune.append((gb, path, "a smoke shard's cache (the timed boards use c1)"))
    if free_kb is None:
        raise SystemExit("measure-candidate: could not read the box's free disk (df -Pk /root)")
    need = float(need_gb) if need_gb is not None else max(float(min_free_gb), largest + headroom_gb)
    return free_kb / 1048576, need, sorted(prune, reverse=True)


def check_built(wheel_dir, vendor):
    """Refuse to stage VENDOR from a measurement build whose --targets did not build its sets (candidate.json
    `targets` / `wheels_built`, tools/release_candidate.py). A dir without them (a release, or a candidate made
    before --targets) built every set."""
    try:
        import json
        doc = json.loads((Path(wheel_dir) / "candidate.json").read_text())
    except (OSError, ValueError):
        return
    built = doc.get("wheels_built")
    if built is not None and vendor not in built:
        raise SystemExit(f"measure-candidate: {wheel_dir} is a measurement build of the targets "
                         f"{','.join(doc.get('targets') or [])}, which packed no {vendor} wheel (it built "
                         f"{','.join(built) or 'none'}); rebuild with tools/release.py candidate --targets naming "
                         f"the {vendor} sets, or stage {' or '.join(built) or 'another build'}")


def pick_wheels(wheel_dir, vendor, version):
    """[core wheel, vendor wheel] of VERSION in WHEEL_DIR (exactly one each; the meta wheel is left out)."""
    d = Path(wheel_dir)
    core = sorted(d.glob(f"mojolearn_core-{version}-*.whl"))
    plug = sorted(d.glob(f"mojolearn_{vendor}-{version}-*.whl"))
    if len(core) != 1 or len(plug) != 1:
        raise SystemExit(f"measure-candidate: {d} must hold one mojolearn_core-{version} and one "
                         f"mojolearn_{vendor}-{version} wheel (found {len(core)} and {len(plug)})")
    return [core[0], plug[0]]


class StagePlan:
    """The duck-typed plan tools/ops/release_gate.py Runner.stage reads: sync, prepare, uploads, upload, verify."""

    def __init__(self, vendor, lane, tree, indir, uploads):
        self.vendor, self.lane, self.tree, self.indir = vendor, lane, tree, indir
        self.wrapper = RG.VENDORS[vendor]["wrapper"]
        self._uploads = uploads
        self.dirs = sorted({os.path.dirname(r) for _, r, _ in uploads})

    def box_sh(self, command):
        if self.vendor == "nvidia":
            return ["bash", self.wrapper, "sh", self.lane, command]
        return ["bash", self.wrapper, "ssh", command]

    synced = False      # cmd_stage syncs first (the lane's pod is assigned at its first sync), then checks the disk

    def sync(self):
        if self.synced or self.vendor != "nvidia":
            return None
        return ["bash", self.wrapper, "sync", self.lane, str(self.tree)]

    def queue(self):
        """Every pod's queued/running jobs (nvidia_central.sh queue); NVIDIA only."""
        return ["bash", self.wrapper, "queue"]

    def ssh_target(self):
        """The wrapper's "<ssh flags> <target>" of the box, for tools/dataset_store.sh stage."""
        if self.vendor == "nvidia":
            return ["bash", self.wrapper, "ssh-target", self.lane]
        return ["bash", self.wrapper, "ssh-target"]

    def results_base(self):
        """The box dir the submit line's box-out/<tag> lands in."""
        return (f"/root/mojolearn-{self.lane}" if self.vendor == "nvidia" else AMD_TREE) + "/box-out"

    def prepare(self):
        return self.box_sh("set -e; mkdir -p " + " ".join(shlex.quote(d) for d in self.dirs) + "; echo READY")

    def uploads(self):
        return self._uploads

    def upload(self, remote):
        return self.box_sh("cat > " + shlex.quote(remote))

    def verify(self, remotes, executables):
        return self.box_sh("sha256sum " + " ".join(shlex.quote(r) for r in remotes))


ENV_ARM_RE = re.compile(r"([a-z0-9][a-z0-9.-]*)=env:([A-Z_][A-Z0-9_]*=[^,\s]*(?:,[A-Z_][A-Z0-9_]*=[^,\s]*)*)")


def parse_b_arm(text):
    """(name, wheel dir, sha) of a built B set NAME=DIR@SHA, or (name, 'env:VAR=VALUE[,..]', None) of an ENV-ONLY arm
    NAME=env:VAR=VALUE (no rebuild: the job runs the candidate's venv with those variables; the targeted job only)."""
    m = ENV_ARM_RE.fullmatch(text)
    if m:
        return m.group(1), "env:" + m.group(2), None
    m = re.fullmatch(r"([a-z0-9][a-z0-9.-]*)=(.+)@([0-9a-f]{40})", text)
    if not m:
        raise SystemExit(f"measure-candidate: --b-arm wants NAME=DIR@SHA (40 hex) or NAME=env:VAR=VALUE, not {text!r}")
    return m.group(1), Path(m.group(2)).expanduser(), m.group(3)


PRIOR_WHEEL_RE = re.compile(r"([a-z0-9][a-z0-9.-]*)=(.+)@([0-9a-f]{40})")


def parse_prior_wheel(text):
    """(name, wheel dir, sha) of a --prior-wheel NAME=DIR@SHA: one of our own earlier measurement builds on disk."""
    m = PRIOR_WHEEL_RE.fullmatch(text)
    if not m:
        raise SystemExit(f"measure-candidate: --prior-wheel wants NAME=DIR@SHA (a built wheel set, 40-hex sha), "
                         f"not {text!r}")
    return m.group(1), Path(m.group(2)).expanduser(), m.group(3)


def check_prior_wheels(names, b_names, prior_versions, rows):
    """Refuse a --prior-wheel outside the targeted job, a name given twice, or one that is also a --b-arm or a
    --prior-versions entry (the job's arm is prior-<name>, beside prior-<version>)."""
    if names and not rows:
        raise SystemExit("measure-candidate: --prior-wheel times the listed rows of the targeted job only (--rows)")
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f"measure-candidate: --prior-wheel {', '.join(dup)} given twice")
    clash = sorted(set(names) & (set(b_names) | set(prior_versions)))
    if clash:
        raise SystemExit(f"measure-candidate: --prior-wheel {', '.join(clash)} is also a --b-arm or a "
                         "--prior-versions entry; name it apart")


def stored_times(boards):
    """{race id: the LARGEST stored wall_s over BOARDS} (a path or a list of paths; missing files are skipped): a
    stored time from another pod is a floor, and the slowest box seen is the safer forecast (STORED_TIMES_AB1)."""
    secs = {}
    for b in ([boards] if isinstance(boards, (str, Path)) else (boards or [])):
        if b and Path(b).is_file():
            for rid, t in TR.stored_seconds(PC.load_json(str(b))).items():
                secs[rid] = max(t, secs.get(rid, 0.0))
    return secs


def ab_plan(owed_file, planner_file, vendor, b_names, times_board, chunk_s=TR.DEFAULT_CHUNK_S):
    """[(arm, rows tag, races, minutes)] of the owed A/B units the job times for these B arms (tools/measure_owed.py
    plan, the same call the job's targeted plan makes, with no paper race left to a paper board): each unit is its
    base (the candidate) and the arm back to back, so its minutes are twice the races' stored candidate seconds (the
    largest over TIMES_BOARD, one path or a list; the family median for a race or a neighbouring shape with no stored
    time) plus one process start per chunk and arm (TR.PROC_S; the races' wall_s holds the rest), and the base races'
    smoke (TR.SMOKE_S_PER_RACE; lane fb15-ab-verdicts, 2026-11-18: ab1 was estimated at 152 minutes and passed the
    240-minute cap with 17 of 25 sqrt-comptime-step cells left). Every unit times its base in the same job (the
    2026-11-17 STORED BASE, an arm alone against another host's base time, is gone: IN-JOB BASE CAP, measure_owed.py)."""
    if not b_names:
        return []
    planner = MO.load_planner(str(planner_file))
    doc = PC.load_json(owed_file)
    units = MO.plan(doc, {"recipes": [], "race_ids": []}, planner, vendor, list(b_names))[0]
    secs = stored_times(times_board)
    med = TR.family_medians(secs)
    out = []
    for (arm, rows), ids in sorted(units.items(), key=lambda kv: MO.unit_order(kv[0])):
        if arm == "candidate":
            continue
        each = {i: ((secs.get(i) if rows == "full" else None) or med.get(i.split("/")[0], chunk_s / 4.0)) for i in ids}
        s = sum(2 * t for t in each.values())
        procs = 2 * (len(TR.chunk_ids(ids, {}, chunk_s)) if ids else 0)
        smoke = TR.SMOKE_S_PER_RACE * len(ids) if rows == "full" else 0.0
        out.append((arm, rows, len(ids), (s + TR.PROC_S * procs + smoke) / 60.0))
    return out


def split_arms(abp, first_min, cap_min):
    """[[arm, ...], ...]: the B arms of AB_PLAN packed in plan order into submits under CAP_MIN, the first one also
    carrying FIRST_MIN (the rows, smoke and fixed cost); an arm whose units alone pass the cap is a submit of its own.
    Each later submit runs PC_PARTS=tg-board,tg-compare with PC_OWED_ARMS on the same results dir (tg-board resumes
    the boards from board.json)."""
    per = {}
    for arm, _, _, m in abp:
        per[arm] = per.get(arm, 0.0) + m
    groups, cur, acc = [], [], first_min
    for arm, m in per.items():
        if cur and acc + m > cap_min:
            groups.append(cur)
            cur, acc = [], TR.FIXED_S / 60.0
        cur.append(arm)
        acc += m
    if cur:
        groups.append(cur)
    return groups


def submit_line(vendor, lane, tag, env):
    q = " ".join
    res = f"box-out/{tag}"
    if vendor == "nvidia":
        envs = q(f"--env {k}={v}" for k, v in env.items())
        return (f"bash tools/nvidia_central.sh submit {lane} --cap 240 --note {tag} --results {res} {envs} "
                f"tools/ops/paper_compare_job.sh")
    envs = q(f"{k}={v}" for k, v in env.items())
    return f"bash tools/do_amd_steward.sh run {tag} {res} '{envs} bash tools/ops/paper_compare_job.sh'"


def cmd_stage(a):
    commit, version, vendor = a.source_commit, a.version, a.vendor
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit("measure-candidate: --source-commit must be the full frozen sha")
    wdir = Path(a.wheel_dir) if a.wheel_dir else default_wheel_dir(version, commit)
    check_built(wdir, vendor)
    wheels = pick_wheels(wdir, vendor, version)
    manifest = C.build_manifest(version, vendor, wheels, commit)     # refuses a wheel of another freeze
    print(f"candidate {version} {commit[:12]} {vendor}: " + ", ".join(
        f"{f['file']} {f['sha256'][:16]}" for f in manifest["files"]))
    tree = Path(a.tree).resolve() if a.tree else ROOT
    src = source_tree(wdir, commit, a.source_tree)
    owed_file = Path(a.owed_file).expanduser() if a.owed_file else results_file(f"paper_compare/{OWED_NAME}", tree, src)
    specs = [parse_b_arm(spec) for spec in a.b_arm]
    run_envs = check_b_arms([n for n, _, _ in specs], owed_file,
                            {n: (str(d)[4:] if sha is None else None) for n, d, sha in specs}) or {}
    if any(sha is None for _, _, sha in specs) and not a.rows:
        raise SystemExit("measure-candidate: an env arm (NAME=env:VAR=VALUE) runs in the targeted job only (--rows)")
    recipes = results_file(f"paper_compare/recipes-{vendor}.json", tree, src)
    planner = tree / "tools" / "bench_board.py"
    prior_versions = [v for v in a.prior_versions.split(",") if v]
    if not prior_versions or any(not re.fullmatch(r"[0-9]+(\.[0-9]+)+[.a-z0-9]*", v) for v in prior_versions):
        raise SystemExit(f"measure-candidate: --prior-versions wants a comma list of releases, not {a.prior_versions!r}")
    pw_specs = [parse_prior_wheel(x) for x in a.prior_wheel]
    check_prior_wheels([n for n, _, _ in pw_specs], [n for n, _, _ in specs], prior_versions, a.rows)
    parts = parse_parts(a.parts)
    if a.rows and not parts:
        parts = ["targeted"]
    if runs_targeted(parts) != bool(a.rows):
        raise SystemExit("measure-candidate: --rows and --parts targeted (or a tg-* part) go together")
    stored = bool(a.rows) and a.baseline_board != "none"
    if stored and not Path(a.baseline_board).expanduser().is_file():
        raise SystemExit(f"measure-candidate: --baseline-board {a.baseline_board}: no such file")
    if a.rows:
        # the targeted job's cap check, before anything is copied: an estimate over the cap is refused with its split.
        # Against the stored baseline the job times no 0.8.55 arm; each --prior-wheel is one more arm on every row.
        est, split = TR.check_cap(a.rows, a.cap_min, a.chunk_s, one_arm=stored, prior_wheels=len(pw_specs))
        print(TR.estimate_line(est))
        owed_arms = [x for x in (a.owed_arms or "").split(",") if x]
        bad = [x for x in owed_arms if x not in [n for n, _, _ in specs]]
        if bad:
            raise SystemExit(f"measure-candidate: --owed-arms {','.join(bad)} is not a staged --b-arm")
        abp = ab_plan(owed_file, planner if planner.is_file() else ROOT / "tools" / "bench_board.py", vendor,
                      owed_arms or [n for n, _, _ in specs], a.times_board or [STORED_TIMES, *STORED_TIMES_AB1],
                      a.chunk_s)
        for arm, rows, n, m in abp:
            print(f"TG-AB arm={arm} rows={rows} races={n} minutes={m:.0f} (base and arm back to back)")
        if abp:
            # a later split submit (no `targeted` part) resumes the rows already timed: only the fixed cost is new
            first = est["total_min"] if "targeted" in parts else TR.FIXED_S / 60.0
            tot = first + sum(m for *_, m in abp)
            print(f"TG-ESTIMATE-ALL minutes={tot:.0f} (rows {first:.0f} + A/B {tot - first:.0f})")
            if tot > a.cap_min:
                # lane fb15-ab-verdicts (2026-11-18): ab1 printed this as a warning, was submitted at 152 estimated
                # minutes and hit the cap; an over-cap job is refused with the split it needs, before anything moves
                groups = split_arms(abp, first, a.cap_min)
                raise SystemExit(
                    f"measure-candidate: the job is estimated at {tot:.0f} min, over the {a.cap_min:.0f}-minute cap; "
                    "submit it split, the same stage with:\n  "
                    + "\n  ".join(f"submit {k}: --owed-arms {','.join(g)}"
                                   + ("" if k == 1 else " --parts tg-board,tg-compare (same --tag: the boards resume)")
                                   for k, g in enumerate(groups, 1)))
        if split:
            raise SystemExit(f"measure-candidate: {a.rows} is estimated at {est['total_min']:.0f} min, over the "
                             f"{a.cap_min:.0f}-minute cap; submit the split, one job each (each times both arms):\n  "
                             + "\n  ".join(f"{f}  {TR.estimate_line(e)}" for f, e in split))
    keys = dataset_keys(vendor, owed_file, recipes, planner if planner.is_file() else ROOT / "tools" / "bench_board.py",
                        [n for n, _, _ in specs], prior_versions[-1] != PC.PAPER_VERSION and not stored, parts, a.rows)
    ds_manifest = results_file("dataset_store/manifest.tsv", src, tree)
    print(f"datasets the job reads ({vendor}, parts {','.join(parts) or 'default'}): {' '.join(keys)} "
          f"(pins {ds_manifest})")
    indir = f"{IN_BASE}/{commit[:12]}"
    uploads = [(w, f"{indir}/{vendor}/{w.name}", False) for w in wheels]
    b_lines = []
    for name, bdir, bsha in specs:
        if bsha is None:                                     # an env arm: the candidate's own wheels, no upload
            b_lines.append(f"{name} {bdir} {commit}\n")
            continue
        bver = dir_version(bdir, a.b_version or version)     # a B candidate has its own local version
        check_built(bdir, vendor)
        bw = pick_wheels(bdir, vendor, bver)
        C.build_manifest(bver, vendor, bw, bsha)
        uploads += [(w, f"{indir}/b-{name}/{vendor}/{w.name}", False) for w in bw]
        # the owed entry's run_env rides on this arm's line only (the job's 4th field; its board units and driver rows
        # run with it, the base and the other arms without it)
        xenv = run_envs.get(name)
        if xenv:
            print(f"B arm {name}: runs with {xenv} (owed run_env)")
        b_lines.append(f"{name} {indir}/b-{name}/{vendor} {bsha}" + (f" env:{xenv}" if xenv else "") + "\n")
    p_lines = []
    for name, pdir, psha in pw_specs:
        # our own earlier built wheel: validated and staged exactly as a B set (no owed entry), its own dir p-<name>
        pver = dir_version(pdir, a.b_version or version)
        check_built(pdir, vendor)
        pw = pick_wheels(pdir, vendor, pver)
        pm = C.build_manifest(pver, vendor, pw, psha)
        print(f"prior wheel {name}: {pver} {psha[:12]} " + ", ".join(
            f"{f['file']} {f['sha256'][:16]}" for f in pm["files"]))
        uploads += [(w, f"{indir}/p-{name}/{vendor}/{w.name}", False) for w in pw]
        p_lines.append(f"{name} {indir}/p-{name}/{vendor} {psha}\n")
    # OUR prior wheels re-timed on the same box in the same job (AGENTS.md "Times are compared only on the same box",
    # 2026-11-14): the compare's columns; the job installs them from PyPI. The last one is PC_PRIOR_VERSION (a last
    # release when it is not the paper baseline).
    env = {"PC_WHEEL_DIR": f"{indir}/{vendor}", "PC_SOURCE_COMMIT": commit,
           "PC_PRIOR_VERSIONS": ",".join(prior_versions), "PC_PRIOR_VERSION": prior_versions[-1]}
    if parts:
        env["PC_PARTS"] = ",".join(parts)     # the job runs the parts whose datasets were staged above
    if a.rows and a.owed_arms:
        env["PC_OWED_ARMS"] = a.owed_arms     # a split submit times only these arms' owed units
    if a.rows:
        rows_name = Path(a.rows).name
        uploads.append((Path(a.rows).expanduser(), f"{indir}/rows/{rows_name}", False))
        env["PC_ROWS"] = f"{indir}/rows/{rows_name}"
        if a.chunk_s != TR.DEFAULT_CHUNK_S:
            env["PC_TG_CHUNK_S"] = str(a.chunk_s)
        if stored:
            # the stored 0.8.55 board goes beside the wheels; the job times no prior arm (paper_compare_job.sh)
            base = Path(a.baseline_board).expanduser()
            uploads.append((base, f"{indir}/baseline/{base.parent.name}.board.json", False))
            env["PC_BASELINE_BOARD"] = f"{indir}/baseline/{base.parent.name}.board.json"
            for k in ("PC_PRIOR_VERSIONS", "PC_PRIOR_VERSION"):
                env.pop(k, None)
    allow_hw_diff = a.allow_hw_diff or os.environ.get("PC_ALLOW_HW_DIFF") == "1"
    if vendor == "nvidia" and allow_hw_diff:
        env["PC_ALLOW_HW_DIFF"] = "1"     # the job then runs on another GPU and every row is flagged HW-DIFF
    if p_lines:
        local = Path(a.state_dir).expanduser() / f"prior-wheels-{commit[:12]}-{vendor}.txt"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text("".join(p_lines))
        uploads.append((local, f"{indir}/prior-wheels-{vendor}.txt", False))
        env["PC_PRIOR_WHEELS"] = f"{indir}/prior-wheels-{vendor}.txt"
    if b_lines:
        local = Path(a.state_dir).expanduser() / f"b-arms-{commit[:12]}-{vendor}.txt"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text("".join(b_lines))
        uploads.append((local, f"{indir}/b-arms-{vendor}.txt", False))
        env["PC_B_ARMS"] = f"{indir}/b-arms-{vendor}.txt"
    plan = StagePlan(vendor, a.lane, tree, indir, uploads)
    need_gb = None
    if a.rows:
        # the disk follows the arms actually staged (targeted_disk_gb): the candidate's venv (and the priors' when
        # 0.8.55 is timed on the box), one B-arm window, every listed race and A/B unit race with its evidence
        nrows = len(TR.read_rows(a.rows))
        built = sum(1 for _, _, sha in specs if sha is not None)
        arms = (1 if stored else 1 + len(prior_versions)) + len(pw_specs)     # each prior wheel is one more venv
        ab_races = sum(2 * n for _, _, n, _ in abp)
        need_gb = targeted_disk_gb(arms, 1 if built else 0, sum(Path(x).stat().st_size for x, _, _ in uploads),
                                   nrows * arms + ab_races, arms + 2 * len(abp))
        print(f"TG-DISK need={need_gb:.1f} GB ({arms} venv(s) in tg-install, B-arm window {1 if built else 0}, "
              f"{built} built arm(s), {nrows * arms + ab_races} timed races, {arms + 2 * len(abp)} boards, "
              f"{sum(Path(x).stat().st_size for x, _, _ in uploads) / 2 ** 30:.2f} GB staged)")
        if a.min_free_gb is not None:
            print(f"  --min-free-gb {a.min_free_gb:g} given: it is the need instead")
    tag = a.tag or f"pc-{commit[:12]}-{vendor}"
    line = submit_line(vendor, a.lane, tag, env)
    stage_ds = ["sh", "tools/dataset_store.sh", "stage", "<" + " ".join(plan.ssh_target()) + ">"] + keys
    if a.dry_run:
        q = lambda argv: " ".join(shlex.quote(str(x)) for x in argv)  # noqa: E731
        if vendor == "nvidia":
            print(q(plan.queue()) + f"  # no queued/starting/running job of lane {a.lane}"
                  + (" (else allowed: --allow-running)" if a.allow_running else ", else refused"))
        if plan.sync():
            print(q(plan.sync()))
        if vendor == "nvidia":
            print(q(plan.box_sh(GPU_SCRIPT)) + f"  # every name {PAPER_NVIDIA_GPU}"
                  + (" (else HW-DIFF, allowed)" if allow_hw_diff else ", else refused"))
        print(q(plan.box_sh(disk_script(plan.results_base()))) + (
            f"  # free >= {need_gb:.1f} GB (TG-DISK)" if need_gb is not None and a.min_free_gb is None else
            f"  # free >= max({a.min_free_gb if a.min_free_gb is not None else MIN_FREE_GB:g}, largest results + "
            f"{HEADROOM_GB}) GB"))
        print(f"MOJOLEARN_DATASET_MANIFEST={shlex.quote(str(ds_manifest))} " + q(stage_ds))
        print(q(plan.prepare()))
        for local, remote, _ in uploads:
            print(q(plan.upload(remote)) + " < " + shlex.quote(str(local)))
        print(q(plan.verify([r for _, r, _ in uploads], [])) + "  # == local sha256")
        print("SUBMIT " + line)
        return 0
    if vendor == "amd":
        head = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        got = subprocess.run(["bash", plan.wrapper, "ssh", f"git -C {AMD_TREE} rev-parse HEAD"], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=120)
        if got.stdout.strip().splitlines()[-1:] != [head]:
            raise SystemExit(f"measure-candidate: the droplet's {AMD_TREE} is not at {head[:12]}; run "
                             f"`bash tools/do_amd_steward.sh update {head}` first (the branch must be pushed)")
    runner = RG.Runner(a)
    if vendor == "nvidia":
        check_queue(runner, plan, a.allow_running)       # before the sync: never move a tree a job is reading
    if plan.sync():
        print("$ " + " ".join(plan.sync()), flush=True)
        runner.must(plan.sync(), "the lane sync", timeout=3600)
    plan.synced = True
    if vendor == "nvidia":
        check_gpu(runner, plan, allow_hw_diff)
    check_disk(a, runner, plan, tag, commit[:12], need_gb)
    stage_datasets(a, runner, plan, keys, ds_manifest, commit[:12])
    runner.stage(plan)
    print("SUBMIT " + line)
    return 0


#: the pod queue's live states (tools/gpu_queue_box.sh queue): a job in one of them reads its lane's tree
LIVE_JOB_STATES = ("queued", "starting", "running")


def lane_jobs(text, lane):
    """[(pod, id, state, script)] of LANE's live jobs in `nvidia_central.sh queue` output: per pod a "== nvcN: ..."
    header, then `ID LANE GPUS STATE SUBMITTED STARTED SCRIPT` rows until "-- last finished:"."""
    jobs, pod, live = [], "?", False
    for line in (text or "").splitlines():
        if line.startswith("== "):
            pod, live = line[3:].split(":", 1)[0].strip(), True
            continue
        if line.startswith("-- last finished"):
            live = False
            continue
        parts = line.split()
        if not live or len(parts) < 4 or not parts[0].isdigit():
            continue
        state = parts[3].split("@", 1)[0]
        if parts[1] == lane and state in LIVE_JOB_STATES:
            jobs.append((pod, parts[0], state, parts[-1]))
    return jobs


def check_queue(runner, plan, allow_running):
    """Refuse to sync the lane's tree while a job of that lane is queued, starting or running (--allow-running)."""
    got = runner.must(plan.queue(), "the pod queue", timeout=600)
    jobs = lane_jobs(got.stdout or "", plan.lane)
    if not jobs:
        print(f"queue: no live job of lane {plan.lane}")
        return
    named = ", ".join(f"{pod}-{jid} {state} {script}" for pod, jid, state, script in jobs)
    msg = (f"measure-candidate: lane {plan.lane} has {len(jobs)} live job(s) on its tree /root/mojolearn-{plan.lane}: "
           f"{named}; a sync now changes the source under it (bench_board stops: \"source changed during "
           f"measurement\"). Wait for it, stage under another --lane, or pass --allow-running")
    if not allow_running:
        raise SystemExit(msg)
    print("WARNING " + msg, flush=True)


def check_gpu(runner, plan, allow_hw_diff):
    got = runner.must(plan.box_sh(GPU_SCRIPT), "the GPU check on the box", timeout=300)
    names, ok = gpu_verdict(got.stdout or "")
    print(f"gpu: {', '.join(names) or '(none listed)'}")
    if ok:
        return
    msg = (f"measure-candidate: the pod's GPU is {', '.join(names) or 'unreadable'}, not the paper's "
           f"{PAPER_NVIDIA_GPU}; timings would not be comparable. Bring up an L40S (tools/nvidia_central.sh up asks "
           f"for it only) or pass --allow-hw-diff (every row is then flagged HW-DIFF)")
    if not allow_hw_diff:
        raise SystemExit(msg)
    print("WARNING HW-DIFF " + msg, flush=True)


def check_disk(a, runner, plan, tag, commit12, need_gb=None):
    got = runner.must(plan.box_sh(disk_script(plan.results_base())), "the disk check on the box", timeout=1800)
    free, need, prune = disk_verdict(got.stdout or "", tag, commit12, a.min_free_gb if a.min_free_gb is not None
                                     else MIN_FREE_GB, need_gb=need_gb if a.min_free_gb is None else None)
    print(f"disk: {free:.1f} GB free on /root; the job needs {need:.1f} GB")
    for gb, path, why in prune:
        print(f"  prunable {gb:6.1f} GB  {path}  ({why})")
    if free < need:
        msg = (f"measure-candidate: {free:.1f} GB free on /root, under the {need:.1f} GB the job needs; prune the "
               f"paths above (results dirs only after their bytes are home) or pass --allow-low-disk")
        if not a.allow_low_disk:
            raise SystemExit(msg)
        print("WARNING " + msg, flush=True)


def stage_datasets(a, runner, plan, keys, ds_manifest, commit12):
    """Every dataset the job reads, through tools/dataset_store.sh stage (presigned on the Mac, verified on the box;
    a file already at its pins is kept). Its full output goes to the state dir."""
    got = runner.must(plan.ssh_target(), "the box's ssh target")
    target = ([ln for ln in (got.stdout or "").splitlines() if ln.strip()] or [""])[-1].strip()
    if not target:
        raise SystemExit("measure-candidate: the wrapper printed no ssh target")
    log = Path(a.state_dir).expanduser() / f"datasets-{commit12}-{plan.vendor}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"staging {len(keys)} dataset(s) through tools/dataset_store.sh stage (log {log})", flush=True)
    env = dict(os.environ, MOJOLEARN_DATASET_MANIFEST=str(ds_manifest))
    with open(log, "w") as fh:
        rc = subprocess.run(["sh", str(ROOT / "tools" / "dataset_store.sh"), "stage", target] + list(keys),
                            cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=7200).returncode
    tail = log.read_text(errors="replace").strip().splitlines()[-1:] if log.exists() else []
    if rc != 0:
        raise SystemExit(f"measure-candidate: dataset staging failed (rc {rc}); see {log}: {' '.join(tail)}")
    print("  " + " ".join(tail))


def main(argv=None):
    p = argparse.ArgumentParser(prog="measure_candidate", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage", help="validate + copy the candidate wheels to the box; print the submit line")
    s.add_argument("vendor", choices=("nvidia", "amd"))
    s.add_argument("--version", required=True, help="the candidate's version (the freeze, e.g. 0.8.67)")
    s.add_argument("--source-commit", required=True, help="the frozen source commit (40 hex)")
    s.add_argument("--wheel-dir", default=None,
                   help="default ~/mojolearn-evidence/release/<version>/<commit12>/linux/final (release.py linux-pack), "
                        "or release-candidates/<version>/... for a local version (tools/release.py candidate)")
    s.add_argument("--lane", default="measure", help="NVIDIA lane name: the tree /root/mojolearn-<lane> (default measure)")
    s.add_argument("--tree", default=None, help="the worktree whose tools/ the job runs (default this checkout)")
    s.add_argument("--b-arm", action="append", default=[], help="NAME=DIR@SHA: a B wheel set of the owed A/Bs")
    s.add_argument("--prior-wheel", action="append", default=[], metavar="NAME=DIR@SHA",
                   help="one of OUR OWN earlier built wheel sets (a measurement build dir on disk and its commit) timed "
                        "on every listed row in the same job right after the candidate (--rows only; repeat it): the "
                        "compare's verdict baseline is the first one; no owed-file entry, no define")
    s.add_argument("--b-version", default=None, help="the B sets' version (default: each dir's candidate.json, "
                   "else --version)")
    s.add_argument("--state-dir", default="~/mojolearn-evidence/measure-candidate")
    s.add_argument("--tag", default=None, help="the results tag (default pc-<commit12>-<vendor>)")
    s.add_argument("--source-tree", default=None, help="the candidate's full source tree (default <work dir>/source/"
                   "<commit12> beside the wheel dir): its bench/results/dataset_store/manifest.tsv pins the datasets")
    s.add_argument("--owed-file", default=None, help="the owed file whose b_arms name the B sets (default "
                   f"bench/results/paper_compare/{OWED_NAME} of --tree, else of the source tree)")
    s.add_argument("--min-free-gb", type=float, default=None,
                   help=f"free disk on /root the job needs at least (default: a targeted job's own need from the arms "
                   f"staged, TG-DISK; any other job {MIN_FREE_GB} or the largest earlier results dir + {HEADROOM_GB})")
    s.add_argument("--allow-low-disk", action="store_true", help="warn instead of refusing below the free-disk need")
    s.add_argument("--allow-hw-diff", action="store_true",
                   help=f"NVIDIA: go on when the pod's GPU is not an {PAPER_NVIDIA_GPU} (also PC_ALLOW_HW_DIFF=1); "
                   "the job then flags every row HW-DIFF")
    s.add_argument("--allow-running", action="store_true",
                   help="NVIDIA: sync the lane's tree even while a job of that lane is queued or running (it then "
                   "reads changed source)")
    s.add_argument("--prior-versions", default=DEFAULT_PRIOR_VERSIONS,
                   help="OUR prior releases the job installs from PyPI and times on the same box beside the candidate "
                        "(the compare's columns; a last entry other than the paper baseline is the last-release arm, "
                        f"which also times the owed board rows); default {DEFAULT_PRIOR_VERSIONS}")
    s.add_argument("--parts", default="",
                   help="the job's PC_PARTS (default: the job's own, the paper parts); board-all (or any all-* part) "
                        "stages the full board's datasets instead of the paper's, and the submit line carries PC_PARTS")
    s.add_argument("--rows", default=None,
                   help="a targeted rows file (tools/ops/targeted_rows.py): the job times only those rows "
                        "(PC_PARTS=targeted, PC_ROWS); refused over --cap-min, naming the split")
    s.add_argument("--cap-min", type=float, default=TR.DEFAULT_CAP_MIN,
                   help="the targeted job's estimate cap in minutes (default %(default)s)")
    s.add_argument("--chunk-s", type=int, default=TR.DEFAULT_CHUNK_S,
                   help="the targeted job's interleave chunk in seconds of candidate time (PC_TG_CHUNK_S)")
    s.add_argument("--baseline-board", default=str(STORED_BASELINE),
                   help="the targeted job's STORED 0.8.55 board (default the 2026-11-17 L40S full board's): uploaded "
                        "beside the wheels (PC_BASELINE_BOARD); the job then times our new wheel only. 'none' times "
                        "0.8.55 on the box beside the candidate instead (the two-arm targeted job)")
    s.add_argument("--times-board", action="append", default=None,
                   help="a candidate board.json whose stored wall seconds estimate the A/B arms' minutes (repeat it; "
                        "each race takes its largest; default the 2026-11-17 full board and the ab1 boards of "
                        "build-d652a4b0a1)")
    s.add_argument("--owed-arms", default="",
                   help="a split submit: the comma list of staged --b-arm names whose owed units this job times "
                        "(PC_OWED_ARMS); the stage refuses a job over --cap-min and names the split")
    s.add_argument("--dry-run", action="store_true", help="validate locally and print every command; copy nothing")
    a = p.parse_args(argv)
    return cmd_stage(a)


if __name__ == "__main__":
    sys.exit(main())
