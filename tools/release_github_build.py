#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE LINUX RELEASE SETS ON GITHUB'S FREE RUNNERS (2026-09-27): the `github`
build backend of `pixi run release` (tools/release.py --build-backend github,
opt-in) and the planner its workflow calls.

    python3 tools/release_github_build.py run --commit <40-hex> --arch sm_89 --out DIR
    python3 tools/release_github_build.py run ... --run-id N      attach to a dispatched run
    python3 tools/release_github_build.py plan --src SRC --sets 'sm_89 gfx942' --shards 6   (the workflow)
    python3 tools/release_github_build.py job-map ...                                       (the workflow)
    python3 tools/release_github_build.py layout --commit C [--planner]    merged | per-binding
    python3 tools/release_github_build.py run --commit C --host --host-donor <leg>/release-build/build --out DIR
    python3 tools/release_github_build.py estimate [--sets 'sm_89 sm_80 gfx942 host'] [--modes identical]

WHAT RUNS WHERE. .github/workflows/release-linux-build.yml (workflow_dispatch
only) builds one or more sets on ubuntu-24.04 runners (4 vCPU, 16 GB): a plan
job splits each set's ~76 builds into SHARDS matrix jobs balanced by measured
compile time, each shard compiles its slice inside the CPU build box's pinned
image (rocm/dev-ubuntu-22.04 by digest, the same image, path and route as
tools/release_linux_build.sh, so every binding's cache key is that route's),
and one assemble job per set runs the unchanged release route over the shards'
archives (tools/gha_release_box.sh). Its artifact `release-<vendor>-<arch>` is
the tree a build leg writes, <vendor>-<arch>/release-build/ with
build/build-provenance.json, plus GHA/ (every job's timings, memory, cache
outcomes and uploads).

`run` (the Mac, detached by release.py as a build leg): mint the binding-cache
URL map, dispatch the workflow for the frozen commit, poll the run, download
the artifact into the leg layout release.py reads (leg_layout "github", the
cpu-box layout), verify the tree against its proof, promote the cache uploads,
and exit 0 only for a verified tree. A failure names the job.

THE MERGED LAYOUT, ONE COMPILE PER JOB (2026-10-26; Andrew: compiling never
uses paid rentals, GitHub's free runners build programmatically). A source
whose packaging/linux/merged_layout.py says merged builds ONE compile per
merged group of bindings/merged_groups.json (each directory set -- the
identical tier, the host set, the fast tier when the release modes name it --
holds several groups since 0.8.59, so each compile fits a free runner). A
merged compile is one `mojo build` of one root:
Mojo has no separate compilation of the members into objects that link into
the same bytes, so a merged file cannot be split across jobs without becoming
a different artifact. `plan` therefore gives every merged compile of a set its
OWN shard job (each job holds exactly one compile, built by the unchanged
cpu-box route in the same image, path, flags and compiler worker count, so the
bytes and cache keys are the cpu-box route's), and sizes each job's swap from
UNIT_PEAK_GIB: a compile whose recorded peak exceeds the runner's 15 GiB gets a
swapfile on the runner's scratch disk (the workflow's Swap step), up to
SWAP_MAX_GIB; beyond that the plan refuses with unit_refusal(). The release
modes (MOJOLEARN_RELEASE_MODES, the workflow's `modes`) choose the tiers, as
on the cpu box (tools/release_linux_build.sh exports them to the box).

THE HOST PRODUCER. `host` in the workflow's `sets` adds one job that runs the
SOURCE's tools/release_host_build.py build (the retained-composition host
producer, docs/plans/RETAINED_DEVICE_HOST_COMPOSITION_20261025.md) in the same
image at /root/mojolearn from a git checkout of the commit, its donor runtime
build either this run's first assembled device set or a retained device leg
the Mac uploads to R2 and passes as ONE presigned GET (`host_donor_url`, read
from the event file and masked, never interpolated; its sha256 in
`host_donor_sha256`). Its artifact `host-build` is the producer's output
directory, which tools/retained_release_composition.py --host-build takes.

THE CACHE. With a map_url the binding cache is R2 (below). Without one, each
shard restores its hot directory from actions/cache (key rlb1-<arch>-<unit>-
<commit>, falling back to the newest entry of the same <arch>-<unit>): an
archive is named by its bincache content key and placed only after its key,
fields and every file's sha256 verify, so a stale archive is never used, only
skipped. The job saves back only the archives its own builds used or made.
gfx942 bytes are frozen by R2, not by a GitHub cache: with a map_url the
GitHub cache is neither read nor written.

THE BINDING CACHE WITHOUT A SECRET. R2 credentials never leave this Mac. The
Mac lists the cache partition and presigns a GET for every object and a PUT
for each job's disjoint inbox slots (tools/bincache.py plan_lines, the same map
tools/bincache_leg.sh stages over ssh), uploads that map to R2 as one object
and passes ONE presigned GET of it, valid MAP_HOURS, as the workflow input
map_url. The workflow never interpolates it: each job reads it from the event
file, masks it, fetches the map into a 0600 file outside every artifact, and
keeps only its own PUT slots. After the run the Mac deletes the map object and
promotes an inbox upload only when the object in R2 hashes to the archive the
job itself recorded (GHA/bincache/hot_sha256.tsv), so a leaked slot URL cannot
plant bytes under a key. gfx942 codegen is not reproducible from a cold cache,
so the AMD set refuses to build without the cache unless --allow-cold-amd.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

WORKFLOW = "release-linux-build.yml"
#: The CPU build box's image (tools/release_linux_build.sh IMAGE; a test holds them equal).
IMAGE = "rocm/dev-ubuntu-22.04@sha256:a3850e6638c6c390436ef1aacd72fd1359af36083ac823d5136818206998c484"
#: The image the cache keys name. The GitHub jobs run this very image by
#: digest, CPU only, at /root/mojolearn, through the same route, so their keys
#: are the RunPod CPU box's and they read (and freeze) the same partition; the
#: runner that built an archive is its manifest's `builder`, never a key field.
IMAGE_DECL = "runpod-cpu:" + IMAGE
#: sm_80 is the PTX slot of mojolearn-nvidia (gpu_plugins.PTX_ARCH), built by the
#: same route; release061_remote_build.sh gives it the ptx code format
#: (Andrew 2026-10-10: PTX is a normal target; no flag).
ARCHS = {"sm_90a": "cuda", "sm_89": "cuda", "sm_80": "cuda", "gfx942": "hip"}
#: Upload slots each job (a shard, or the assemble job) may PUT into.
SLOTS_PER_JOB = 80
MAP_HOURS = 8
MAP_PREFIX = "bincache/gha-maps"
#: Where `run --host-donor` stages a retained donor build for one run (deleted after).
DONOR_PREFIX = "bincache/gha-donors"
#: tools/release_linux_build.sh's route overlay (a test holds them equal).
OVERLAY = ("tools/release061_remote_build.sh", "tools/cpu_build_guard.py", "tools/release_linux_cpu_box.sh",
           "tools/bincache.py", "tools/release_modes.py")
#: Files whose committed bytes the runners use; a dirty one refuses the dispatch.
ROUTE_FILES = OVERLAY + ("tools/gha_release_box.sh", "tools/release_github_build.py",
                         ".github/workflows/" + WORKFLOW)
#: runpod_cpu_leg.sh make_source's exclusions: the shipped tree is the same file list.
SOURCE_EXCLUDES = (":!bench/results", ":!mamba/corpus", ":!bench/oracle_*", ":!bench/minentropy_oracle.txt")
#: Measured seconds per build: 0.8.24's cold cuda/sm_89 builds on GitHub's 4 vCPU
#: runners, two at a time (run 36318515173); only the shard balance reads them.
WEIGHTS = {
    "fast:build_gbdt.sh": 206, "identical:build_byte_lm.sh": 202, "identical:build_gbdt.sh": 182,
    "identical:build_metrics.sh": 132, "fast:build_metrics.sh": 130,
    "fast:build.sh": 95, "fast:build_estimators.sh": 87, "identical:build_transformer.sh": 87,
    "identical:build_mixture.sh": 85, "identical:build_mamba.sh": 82, "identical:build_hdbscan.sh": 80,
    "identical:build_gp.sh": 79, "identical:build.sh": 76, "fast:build_hdbscan.sh": 75,
    "identical:build_estimators.sh": 74, "fast:build_kernel_methods.sh": 73,
    "identical:build_training.sh": 73, "identical:build_kernel_methods.sh": 69,
    "identical:build_gbdt_host.sh": 65, "identical:build_solver.sh": 59, "fast:build_gp.sh": 55,
    "fast:build_mixture.sh": 55, "identical:build_linalg.sh": 55, "identical:build_ivf.sh": 53,
    "identical:build_svm.sh": 51, "fast:build_ivf.sh": 46, "fast:build_linalg.sh": 46,
    "fast:build_svm.sh": 42, "identical:build_rf.sh": 41,
    "identical:build_trees.sh": 41, "fast:build_trees.sh": 38,
    "fast:build_rf.sh": 36, "fast:build_solver.sh": 34, "identical:build_mamba_host.sh": 34,
    "identical:build_metrics_host.sh": 32, "identical:build_core_host.sh": 30,
    "identical:build_embedding.sh": 30, "fast:build_arima.sh": 27, "fast:build_resample.sh": 26,
    "identical:build_arima.sh": 26, "identical:build_neural_host.sh": 26, "identical:build_resample.sh": 25,
    "identical:build_byte_lm_host.sh": 24, "identical:build_estimators_host.sh": 24,
    "identical:build_tsa.sh": 24, "fast:build_tsa.sh": 21, "identical:build_trees_host.sh": 20,
    "identical:build_gp_host.sh": 19, "identical:build_rf_host.sh": 19,
    "identical:build_tokenizer_host.sh": 18, "identical:build_training_host.sh": 17,
    "identical:build_svm_host.sh": 16, "identical:build_kernel_methods_host.sh": 15,
    "identical:build_forecast_host.sh": 14, "identical:build_hdbscan_host.sh": 14,
    "identical:build_hdbscan_infer_host.sh": 14, "identical:build_transformer_host.sh": 14,
    "identical:build_forest_host.sh": 13, "identical:build_gp_infer_host.sh": 13,
    "identical:build_ivf_search_host.sh": 13, "identical:build_linalg_host.sh": 13,
    "identical:build_preprocessing.sh": 13, "fast:build_preprocessing.sh": 12,
    "identical:build_arima_host.sh": 12, "identical:build_ivf_host.sh": 12,
    "identical:build_mixture_host.sh": 12, "identical:build_mixture_infer_host.sh": 11,
    "identical:build_preprocessing_host.sh": 10, "identical:build_resample_host.sh": 10,
    "identical:build_tsa_host.sh": 10, "identical:build_embedding_infer_host.sh": 9,
    "identical:build_solver_host.sh": 9, "identical:build_embedding_host.sh": 7,
}
DEFAULT_GPU_WEIGHT, DEFAULT_HOST_WEIGHT = 100, 20
GH = os.environ.get("MOJOLEARN_GH", "gh")


def say(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')} gha-build] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def partition():
    """bincache_leg.sh's partition for the CPU box: <device arch>/<image slug>."""
    return "none/" + re.sub(r"[^A-Za-z0-9._-]", "-", IMAGE_DECL)[:100]


# ---------------------------------------------------------------- the plan (on a runner)
#: GitHub's free ubuntu-24.04 runner for a public repository: 4 vCPU, 16 GB
#: (15 GiB usable); the org has no larger runners.
FREE_RUNNER = "4 vCPU, 15 GB"
RUNNER_RAM_GIB = 15
#: The largest swapfile a job may add. The hosted image keeps its scratch disk
#: at /mnt (~70 GB free; the image's own 4 GB swapfile lives there); the Swap
#: step takes the larger of /mnt and / and caps the file at that disk's free
#: space less SWAP_DISK_RESERVE_GIB, recording what it made (GHA/swap.txt).
SWAP_MAX_GIB = 56
SWAP_DISK_RESERVE_GIB = 4
#: Headroom over a recorded peak before swap is sized: 20%, as
#: tools/development_build.py PEAK_HEADROOM["passed"].
PEAK_HEADROOM = 1.2
#: THE PLANNING PEAK OF ONE MERGED COMPILE, GiB: the LARGEST RSS recorded for
#: it on any box, so swap is never sized from the smallest. Unused swap costs
#: nothing; a compile killed for memory costs the run.
#: WHOLE_DIR_PEAK_GIB below is what ONE compile of a whole directory set
#: recorded before the 0.8.59 split (one group per directory):
#:   identical, cuda  0.8.57 cpu box (c00c4d800, 2 compiler workers): sm_89
#:                    48.59, sm_80 48.12 (the set's build-group peak, guard);
#:                    2026-10-26 one sm_89 identical set ~17 GB
#:   identical, hip   0.8.57 cpu box gfx942 47.87 (8 workers); 2026-10-26 ~39 GB
#:   fast             DO MI325X 2026-10-20, 36.6 GB (2 workers)
#:   host             GitHub run 40272710760 (2026-10-26, source 78af017e2, the
#:                    host producer, 2 workers): SIGKILLed by the kernel at the
#:                    free runner's 16.0 GB cgroup peak after 1123 s, so its peak
#:                    is ABOVE 15 GiB (a lower bound only); planned at the
#:                    identical tier's 48 until a run with swap records it.
#:                    Run 40275379578 (same source, 43 GiB swap on /mnt):
#:                    built in 5275 s, RAM cgroup at the runner's 15.98 GB
#:                    the whole time (the swap use was not sampled then; the
#:                    sampler records memory.swap.current since).
#:                    Older: DO MI325X 2026-10-20, 6.9 GB.
#:                    The 46-member host compile then passed 40 GB (0.8.59
#:                    run 40319501896, sm_89 host shard, > 5700 s on swap).
WHOLE_DIR_PEAK_GIB = {
    ("identical", "cuda"): 49, ("identical", "hip"): 48,
    ("fast", "cuda"): 37, ("fast", "hip"): 37,
    ("host", "cuda"): 48, ("host", "hip"): 48,
}
#: THE PLANNING PEAK PER MERGED GROUP (0.8.59 split), GiB, keyed
#: (tier, group stem, vendor). Initial estimates came from the
#: directory's whole-set peak above times the group's share of its members'
#: own peaks (per-member development builds, one group per member,
#: ~/mojolearn-evidence/dev-cache-mac/peaks 2026-10-26; the whole host set's
#: members sum to 40.5 GB there, matching the >40 GB merged host compile),
#: rounded up:
#:   device  _mojolearn_native 15.27 of 58.45 GB (identical; fast 56.02, no
#:           byte LM), _trees 14.52, _sequence 15.20 with the byte LM (12.77
#:           without), _cluster 13.46
#:   host    _mojolearn_native_host 8.77 of 40.51, _tokenizer_host 7.06,
#:           _neural_host 7.95, _forest_host 8.66, _x_host 8.07
#: Each shard records its own peak (GHA/memory.txt): raise a row when a run
#: records more, never lower one below a recorded value. A group not named
#: here plans at its directory's whole-set peak.
UNIT_PEAK_GIB = {
    ("identical", "_mojolearn_native", "cuda"): 13, ("identical", "_mojolearn_native", "hip"): 13,
    ("identical", "_mojolearn_native_trees", "cuda"): 13, ("identical", "_mojolearn_native_trees", "hip"): 12,
    ("identical", "_mojolearn_native_sequence", "cuda"): 13, ("identical", "_mojolearn_native_sequence", "hip"): 13,
    ("identical", "_mojolearn_native_cluster", "cuda"): 12, ("identical", "_mojolearn_native_cluster", "hip"): 12,
    ("fast", "_mojolearn_native", "cuda"): 11, ("fast", "_mojolearn_native", "hip"): 11,
    ("fast", "_mojolearn_native_trees", "cuda"): 10, ("fast", "_mojolearn_native_trees", "hip"): 10,
    ("fast", "_mojolearn_native_sequence", "cuda"): 9, ("fast", "_mojolearn_native_sequence", "hip"): 9,
    ("fast", "_mojolearn_native_cluster", "cuda"): 9, ("fast", "_mojolearn_native_cluster", "hip"): 9,
    # Run 40340886439 at 216d04c7c (2026-10-26): cgroup peaks 13.24 GiB
    # and 9.28 GiB respectively (process RSS 12.90 / 9.02). Rounded up;
    # AMD run 40340885959: host / tokenizer / x_host cgroup peaks
    # 12.88 / 9.25 / 10.37 GiB. Retained in codex-takeover-0859/
    # gha-40340886439.log, gha-40340885959.log and their memory summaries.
    ("host", "_mojolearn_native_host", "cuda"): 14, ("host", "_mojolearn_native_host", "hip"): 13,
    ("host", "_mojolearn_native_tokenizer_host", "cuda"): 10, ("host", "_mojolearn_native_tokenizer_host", "hip"): 10,
    ("host", "_mojolearn_native_neural_host", "cuda"): 10, ("host", "_mojolearn_native_neural_host", "hip"): 10,
    ("host", "_mojolearn_native_forest_host", "cuda"): 11, ("host", "_mojolearn_native_forest_host", "hip"): 11,
    ("host", "_mojolearn_native_x_host", "cuda"): 10, ("host", "_mojolearn_native_x_host", "hip"): 11,
}
#: A per-binding build: two at a time fit a free runner (0.8.24..0.8.52 runs).
PER_BINDING_PEAK_GIB = 6
#: The host producer (tools/release_host_build.py) compiles the host groups one
#: after another (bindings/merged_build.sh host), so its peak is the largest
#: host group's (the single whole-set compile was killed at 16 GB without swap,
#: run 40272710760).
HOST_PRODUCER_PEAK_GIB = max(v for (t, _g, _v), v in UNIT_PEAK_GIB.items() if t == "host")
#: Measured seconds of one merged compile on the 16-vCPU RunPod cpu box (0.8.57,
#: c00c4d800, full46-build.log start/built lines), for `estimate` only.
UNIT_SECONDS_CPU_BOX = {
    ("identical", "cuda"): 1262, ("identical", "hip"): 1295, ("fast", "cuda"): 960, ("fast", "hip"): 960,
    ("host", "cuda"): 2046, ("host", "hip"): 1735, ("adapters", "cuda"): 61, ("adapters", "hip"): 256,
}
#: Per-job fixed cost on a runner: checkout, image pull, pixi env (0.8.52 run
#: jobs.tsv less the route's phases.tsv: ~260 s), and the assemble route over
#: verified hot hits (~240 s).
JOB_OVERHEAD_SECONDS = 260
ASSEMBLE_ROUTE_SECONDS = 240
CPU_BOX_VCPU, RUNNER_VCPU = 16, 4
#: Seconds measured ON a free runner, which replace the scaled cpu-box figure
#: for a whole-set compile; a group's figure is its directory's times its share
#: of the directory's members (unit_seconds). Empty since the 0.8.59 split: the
#: one row it held, the whole merged host compile paging to swap (5275 s, run
#: 40275379578, 78af017e2), measured paging that a group that fits the runner
#: does not do.
UNIT_SECONDS_RUNNER = {}


def unit_tier(entry):
    """The tier a pool entry compiles: identical, fast, host or adapters."""
    return entry.split(":")[0]


def is_merged(entry):
    """Whether a pool entry is a merged compile: "<tier>:merged_build.sh[:<group>]"."""
    return entry.split(":")[1:2] == ["merged_build.sh"]


def unit_group(entry):
    """The merged group stem a pool entry compiles, or None."""
    parts = entry.split(":")
    return parts[2] if is_merged(entry) and len(parts) > 2 else None


def unit_peak_gib(entry, vendor, merged):
    if not merged:
        return PER_BINDING_PEAK_GIB
    tier = unit_tier(entry)
    row = UNIT_PEAK_GIB.get((tier, unit_group(entry), vendor))
    if row is not None:
        return row
    return WHOLE_DIR_PEAK_GIB.get((tier, vendor), max(WHOLE_DIR_PEAK_GIB.values()))


def swap_gib(peak_gib):
    """Swap a job adds for a compile of `peak_gib`: 0 when it fits the runner."""
    need = int(-(-peak_gib * PEAK_HEADROOM // 1))
    return max(0, need - RUNNER_RAM_GIB)


def unit_refusal(entry, vendor, need):
    """The one sentence a plan refuses a compile no free runner can hold with."""
    return (f"the merged compile {entry} for {vendor} needs {need} GiB with headroom (UNIT_PEAK_GIB), more "
            f"than a free runner's {RUNNER_RAM_GIB} GiB plus SWAP_MAX_GIB {SWAP_MAX_GIB} GiB of swap; build it "
            f"on the cpu-box route (`bash tools/release_linux_build.sh <commit> --rent --archs <arch>`)")


def unit_slug(entry):
    """A cache-key-safe name for a pool entry ("identical:merged_build.sh:_mojolearn_native" ->
    "identical-merged_build-_mojolearn_native")."""
    return re.sub(r"[^A-Za-z0-9_.-]", "-", entry.replace(".sh", ""))


def layout_merged(text, environ):
    """Whether a source's packaging/linux/merged_layout.py (its TEXT, None when
    the source predates it) says the merged layout under `environ`: its own
    enabled(), so the default is the source's, never retyped here."""
    if text is None:
        return False
    ns = {"__name__": "source_merged_layout", "__file__": "/source/packaging/linux/merged_layout.py"}
    exec(compile(text, "packaging/linux/merged_layout.py", "exec"), ns)
    try:
        return bool(ns["enabled"](environ))
    except ValueError as exc:
        raise SystemExit(f"merged_layout: {exc}")


def planner_env():
    """The environment the plan evaluates the source with: the box's, which
    the workflow gives no MOJOLEARN_* variable."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("MOJOLEARN_")}
    env["MOJOLEARN_PACKAGE_BYTE_LM"] = "1"
    return env


def source_merged(src):
    """Whether build_sets.sh in SRC builds the merged layout under the planner's env."""
    p = Path(src) / "packaging/linux/merged_layout.py"
    return layout_merged(p.read_text() if p.is_file() else None, planner_env())


def commit_merged(commit, environ=None, root=ROOT):
    """Whether COMMIT's tree builds the merged layout under `environ` (default:
    the planner's), read from its packaging/linux/merged_layout.py by git
    without a checkout; a commit without that file is per-binding."""
    r = subprocess.run(["git", "-C", str(root), "show", f"{commit}:packaging/linux/merged_layout.py"],
                       capture_output=True, text=True)
    return layout_merged(r.stdout if r.returncode == 0 else None, planner_env() if environ is None else environ)


def plan_set(src, modes=None):
    """One set's builds from the SOURCE's own build_sets.sh: its list variables,
    tier_scripts() and pool_jobs() are evaluated (not retyped) with the
    release's MOJOLEARN_PACKAGE_BYTE_LM=1, over its tiers (MODES, the release
    modes the box is given, else build_sets.sh's own default). Returns
    dict(builds=[(entry, is_host)], per_binding=[(entry, is_host)],
    merged=bool, weights={entry: w}, members={entry: n}, shares={entry: f}).
    `builds` is exactly what pool_jobs() runs: the per-binding scripts, or
    (merged) one "<tier>:merged_build.sh:<group>" per group of the source's
    bindings/merged_groups.json, the host groups riding the identical pass; a
    merged build's weight is its members' per-binding weights and its share
    the fraction of its directory's members it compiles."""
    text = (Path(src) / "packaging/linux/build_sets.sh").read_text()
    m = re.search(r"^SCRIPTS=.*?^host_so\(\)", text, re.M | re.S)
    if not m:
        raise SystemExit("plan: packaging/linux/build_sets.sh has no SCRIPTS= ... host_so() block")
    block = text[m.start():m.end() - len("host_so()")]
    tiers = "fast identical"
    t = re.search(r'^TIERS="\$\{MOJOLEARN_BUILD_TIERS:-([a-z ]+)\}"', text, re.M)
    if t:
        tiers = t.group(1)
    if modes:
        tiers = " ".join(modes)
    merged = source_merged(src)
    pj = re.search(r"^pool_jobs\(\) \{.*?^\}\n", text, re.M | re.S)
    if merged and not pj:
        raise SystemExit("plan: the source builds the merged layout but its build_sets.sh has no pool_jobs()")
    script = block + "\nfor t in " + tiers + "; do for s in $(tier_scripts $t); do echo \"P $t:$s\"; done; done\n"
    if pj:
        # pool_jobs() reads MERGED, which the block above set from the source's merged_layout.py
        script += pj.group(0) + "for t in " + tiers + "; do for j in $(pool_jobs $t); do echo \"J $j\"; done; done\n"
    r = subprocess.run(["bash", "-c", "set -u\n" + script], cwd=src, env=planner_env(), capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("plan: evaluating build_sets.sh's lists failed: " + r.stderr.strip()[-400:])
    lines = [line.strip() for line in r.stdout.splitlines() if line.strip()]
    per = [x[2:] for x in lines if x.startswith("P ")]
    jobs = [x[2:] for x in lines if x.startswith("J ")] if pj else per
    for rows in (per, jobs):
        if len(rows) != len(set(rows)):
            raise SystemExit("plan: a build appears twice in one set")
    is_host = lambda e: e.endswith("_host.sh") or e.startswith("host:")
    if merged != any(is_merged(e) for e in jobs):
        raise SystemExit(f"plan: merged_layout.py says {'merged' if merged else 'per-binding'} but pool_jobs() "
                         f"runs {' '.join(jobs)[:200]}")
    pw = lambda e: WEIGHTS.get(e, DEFAULT_HOST_WEIGHT if is_host(e) else DEFAULT_GPU_WEIGHT)
    weights, members, shares = {}, {}, {}
    groups = {}
    for e in jobs:
        if is_merged(e):
            tier, group = unit_tier(e), unit_group(e)
            if group is None:
                raise SystemExit(f"plan: the merged pool job {e} names no group (<tier>:merged_build.sh:<group>)")
            if tier not in groups:
                groups[tier] = source_groups(src, tier)
            names = groups[tier].get(group)
            if not names:
                raise SystemExit(f"plan: the merged {tier} group {group} is not in the source's plan")
            mine = [p for p in per if (is_host(p) if tier == "host" else (not is_host(p) and p.split(":")[0] == tier))]
            if not mine:
                raise SystemExit(f"plan: the merged {tier} build has no member in the per-binding lists")
            scripts = {member_entry(tier, n) for n in names}
            weights[e] = sum(pw(p) for p in mine if p in scripts) or pw(e)
            members[e] = len(names)
            shares[e] = len(names) / sum(len(v) for v in groups[tier].values())
        else:
            weights[e], members[e], shares[e] = pw(e), 1, 1.0
    return dict(builds=[(e, is_host(e)) for e in jobs], per_binding=[(e, is_host(e)) for e in per],
                merged=merged, weights=weights, members=members, shares=shares)


def source_groups(src, tier):
    """{group stem: [members]} the SOURCE's tools/merged_bindings.py plans for
    one directory set on Linux with the byte LM (the planner's environment)."""
    r = subprocess.run([sys.executable, "tools/merged_bindings.py", "plan", "--dir", tier, "--platform", "linux",
                        "--byte-lm", "1"], cwd=src, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"plan: the source's merged plan for {tier} failed: " + r.stderr.strip()[-300:])
    out = {}
    for line in r.stdout.splitlines():
        cells = line.split()
        if len(cells) == 3:
            out[cells[0]] = [n for n in cells[2].split(",") if n]
    return out


def member_entry(tier, name):
    """The per-binding pool entry of a merged member ("identical:build_gbdt.sh",
    "identical:build_core_host.sh" for a host member: the host scripts build
    under the identical numeric mode)."""
    script = "build.sh" if name == "_mojolearn" else "build_" + name[len("_mojolearn_"):] + ".sh"
    return ("identical" if tier == "host" else tier) + ":" + script


def set_builds(src):
    """[(entry, is_host)] of one set, "<tier>:<script>": what build_sets.sh's pool runs (plan_set)."""
    return plan_set(src)["builds"]


def check_counts(src, builds, modes=None):
    """The GPU builds per planned tier must be the verifier's expected bindings."""
    sys.path.insert(0, str(Path(src) / "tools"))
    try:
        from verify_linux_surface_qualification import MODES, expected_bindings
    finally:
        sys.path.pop(0)
    for mode in MODES:
        if modes and mode not in modes:
            continue
        got = sum(1 for e, host in builds if not host and e.split(":")[0] == mode)
        want = len(expected_bindings(mode, True))
        if got != want:
            raise SystemExit(f"plan: {got} {mode} builds, the verifier expects {want}")


def split(builds, shards, weights=None):
    """Longest first onto the least loaded shard (deterministic); `weights`
    (plan_set's) overrides WEIGHTS per entry."""
    weights = weights or {}
    w = lambda e, host: weights.get(e) or WEIGHTS.get(e, DEFAULT_HOST_WEIGHT if host else DEFAULT_GPU_WEIGHT)
    order = sorted(builds, key=lambda b: (-w(*b), b[0]))
    load = [[0, i, []] for i in range(shards)]
    for e, host in order:
        target = min(load, key=lambda x: (x[0], x[1]))
        target[0] += w(e, host)
        target[2].append(e)
    return [dict(shard=i, only=" ".join(sorted(es)), n=len(es), weight=wt) for wt, i, es in sorted(load, key=lambda x: x[1])]


#: Upload-slot jobs per set the Mac mints at least (a merged set plans one
#: shard per compile: the identical, fast and host groups, the adapters, plus
#: the assemble job; 4 + 4 + 5 + 1 + 1 at the 0.8.59 split, with room to add
#: a group). The plan refuses a source that needs more (build_plan).
MERGED_JOBS_PER_SET = 18


def unit_seconds(entry, vendor, merged, share=1.0):
    """Estimated compile seconds of one shard entry on a free runner: the cpu
    box's measured seconds of the whole directory set, times the group's
    `share` of its members, scaled by the compiler workers the runner's 4 vCPU
    cannot run at once (AMD GPU tiers: 8 workers; everything else 2, which a
    runner runs at full width). No swap penalty is modelled: a lower bound."""
    if not merged:
        return WEIGHTS.get(entry, DEFAULT_GPU_WEIGHT)
    tier = unit_tier(entry)
    if (tier, vendor) in UNIT_SECONDS_RUNNER:
        return int(UNIT_SECONDS_RUNNER[(tier, vendor)] * share)
    base = UNIT_SECONDS_CPU_BOX.get((tier, vendor), max(UNIT_SECONDS_CPU_BOX.values()))
    workers = 8 if vendor == "hip" and tier in ("identical", "fast") else 2
    return int(base * share * min(workers, CPU_BOX_VCPU) / min(workers, RUNNER_VCPU))


def estimate(include, merged, sets, host, host_donor=False, shares=None):
    """{arch: dict(jobs, wall_s, runner_s)} and the run's totals, from the plan's rows."""
    out, adapters = {}, lambda v: UNIT_SECONDS_CPU_BOX[("adapters", v)]
    shares = shares or {}
    for arch in sets:
        vendor = ARCHS[arch]
        rows = [r for r in include if r["arch"] == arch]
        shard_s = [JOB_OVERHEAD_SECONDS + sum(unit_seconds(e, vendor, merged, shares.get(e, 1.0))
                                              for e in r["only"].split())
                   // (1 if merged else 2) + (adapters(vendor) if merged and arch != "sm_80" else 0) for r in rows]
        asm = JOB_OVERHEAD_SECONDS + ASSEMBLE_ROUTE_SECONDS + (adapters(vendor) if arch != "sm_80" else 0)
        out[arch] = dict(jobs=len(rows) + 1, wall_s=max(shard_s or [0]) + asm, runner_s=sum(shard_s) + asm,
                         swap_gib=max([r["swap_gib"] for r in rows] or [0]))
    if host:
        # the producer compiles the host groups one after another: the whole set's seconds
        h = JOB_OVERHEAD_SECONDS + unit_seconds("host:merged_build.sh", "cuda", True) + ASSEMBLE_ROUTE_SECONDS
        # the donor is an uploaded build, else this run's first device set (its assemble job)
        donor_wall = out[sets[0]]["wall_s"] if sets and not host_donor else 0
        out["host"] = dict(jobs=1, wall_s=donor_wall + h, runner_s=h, swap_gib=swap_gib(HOST_PRODUCER_PEAK_GIB))
    total = dict(jobs=1 + sum(v["jobs"] for v in out.values()), wall_s=max([v["wall_s"] for v in out.values()] or [0]),
                 runner_s=sum(v["runner_s"] for v in out.values()))
    return out, total


def build_plan(src, tokens, shards, modes=None, host_donor=False):
    """The run's plan document: one shard row per compile slice (merged: one
    compile per row), its swap, the device sets, the host producer, and the
    estimate. Refuses what a free runner cannot build."""
    if not tokens or len(set(tokens)) != len(tokens) or any(t not in ARCHS and t != "host" for t in tokens):
        raise SystemExit(f"plan: sets must be distinct names from {' '.join(ARCHS)} host")
    if not 1 <= shards <= 16:
        raise SystemExit("plan: shards must be 1..16")
    host = "host" in tokens
    sets = [t for t in tokens if t != "host"]
    if host and not sets and not host_donor:
        raise SystemExit("plan: the host producer needs a donor runtime build: a device set in the same run "
                         "or host_donor_url (tools/release_github_build.py run --host --host-donor DIR)")
    include, merged, nbuilds, members, shares = [], None, 0, {}, {}
    jobs_per_set = shards + 1
    if sets:
        ps = plan_set(src, modes)
        builds, merged, members, shares = ps["builds"], ps["merged"], ps["members"], ps["shares"]
        nbuilds = len(builds)
        # the verifier's bindings are the per-binding members, in either layout
        check_counts(src, ps["per_binding"], modes)
        if merged:
            # ONE merged compile per shard: a merged file is one `mojo build` and cannot be split
            shards = len(builds)
            if shards + 1 > MERGED_JOBS_PER_SET:
                raise SystemExit(f"plan: {shards} merged compiles per set need {shards + 1} upload-slot jobs, more "
                                 f"than the {MERGED_JOBS_PER_SET} the Mac mints (MERGED_JOBS_PER_SET)")
            jobs_per_set = MERGED_JOBS_PER_SET
        for si, arch in enumerate(sets):
            vendor = ARCHS[arch]
            for sh in split(builds, shards, ps["weights"]):
                if not sh["n"]:
                    continue
                entries = sh["only"].split()
                peak = max(unit_peak_gib(e, vendor, merged) for e in entries)
                sw = swap_gib(peak)
                if sw > SWAP_MAX_GIB:
                    raise SystemExit("plan: " + unit_refusal(sh["only"], vendor, sw + RUNNER_RAM_GIB))
                include.append(dict(arch=arch, vendor=vendor, set_index=si, swap_gib=sw, peak_gib=peak,
                                    unit=unit_slug(sh["only"]) if merged else f"shard{sh['shard']}", **sh))
    per_set, total = estimate(include, bool(merged), sets, host, host_donor, shares)
    return dict(matrix=dict(include=include), sets=sets, host=host, builds=nbuilds, shards=shards,
                jobs_per_set=jobs_per_set, modes=list(modes or []),
                layout=None if merged is None else ("merged" if merged else "per-binding"), members=members,
                assemble_swap_gib=max([r["swap_gib"] for r in include] or [0]),
                host_swap_gib=swap_gib(HOST_PRODUCER_PEAK_GIB) if host else 0,
                estimate=dict(per_set=per_set, total=total))


def cmd_plan(a):
    import release_modes
    modes = release_modes.parse_modes(a.modes) if a.modes else None
    doc = build_plan(a.src, a.sets.split(), a.shards, modes, a.host_donor)
    text = json.dumps(doc, indent=1)
    if a.json:
        Path(a.json).write_text(text + "\n")
    if a.github_output:
        with open(a.github_output, "a") as fh:
            fh.write("matrix=" + json.dumps(doc["matrix"], separators=(",", ":")) + "\n")
            fh.write("sets=" + json.dumps(doc["sets"], separators=(",", ":")) + "\n")
            fh.write("has_devices=" + ("true" if doc["sets"] else "false") + "\n")
            fh.write("host=" + ("true" if doc["host"] else "false") + "\n")
            fh.write("host_donor=" + ("true" if doc["host"] and a.host_donor else "false") + "\n")
            fh.write("jobs_per_set=%d\n" % doc["jobs_per_set"])
            fh.write("assemble_swap_gib=%d\n" % doc["assemble_swap_gib"])
            fh.write("host_swap_gib=%d\n" % doc["host_swap_gib"])
    print(f"layout {doc['layout']}, modes {' '.join(doc['modes']) or 'build_sets.sh default'}, "
          f"{doc['builds']} compiles per set")
    for row in doc["matrix"]["include"]:
        print(f"{row['vendor']}-{row['arch']} shard {row['shard']} [{row['unit']}]: {row['n']} builds, "
              f"weight {row['weight']}, planning peak {row['peak_gib']} GiB, swap {row['swap_gib']} GiB")
    if doc["host"]:
        print("host producer: tools/release_host_build.py build, donor "
              + ("host_donor_url" if a.host_donor else f"this run's {doc['sets'][0]} set"))
    est = doc["estimate"]
    for k, v in est["per_set"].items():
        print(f"estimate {k}: {v['jobs']} jobs, wall >= {v['wall_s'] // 60} min, "
              f"{v['runner_s'] // 60} runner-min, swap {v['swap_gib']} GiB")
    t = est["total"]
    print(f"estimate run: {t['jobs']} jobs (free runners run 20 at once), wall >= {t['wall_s'] // 60} min, "
          f"{t['runner_s'] // 60} runner-min (cpu-box seconds scaled to 4 vCPU; swap paging not modelled)")
    return 0


def cmd_estimate(a):
    """The plan of a source (default this checkout) without a runner: jobs, swap and wall time."""
    import release_modes
    modes = release_modes.parse_modes(a.modes) if a.modes else None
    doc = build_plan(a.src, a.sets.split(), a.shards, modes, True)
    print(json.dumps(dict(layout=doc["layout"], modes=doc["modes"], jobs_per_set=doc["jobs_per_set"],
                          shards=[dict((k, r[k]) for k in ("arch", "unit", "only", "peak_gib", "swap_gib"))
                                  for r in doc["matrix"]["include"]],
                          estimate=doc["estimate"]), indent=1))
    return 0


def cmd_layout(a):
    """The binding layout a commit builds: `merged` or `per-binding`. Default:
    under this process's MOJOLEARN_MERGED_BINDINGS (the cpu-box route passes
    it on to the box when set); --planner: under the GitHub box's environment."""
    env = None if a.planner else {k: v for k, v in os.environ.items() if k == "MOJOLEARN_MERGED_BINDINGS"}
    print("merged" if commit_merged(a.commit, env) else "per-binding")
    return 0


def cmd_job_map(a):
    """This job's URL map: the Mac's (read from the event, masked, fetched)
    with only this job's PUT slots, or a header-only map (no R2: every build
    compiles, none uploads) when the dispatch passed none."""
    event = json.loads(Path(a.event).read_text()) if a.event else {}
    url = ((event.get("inputs") or {}).get("map_url") or "").strip()
    dest = Path(a.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    first = (a.set_index * a.jobs_per_set + a.job) * SLOTS_PER_JOB
    mine = {"%03d" % i for i in range(first, first + SLOTS_PER_JOB)} if a.job >= 0 else set()
    if url:
        print("::add-mask::" + url, flush=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            text = r.read().decode()
        keep, puts, gets = [], 0, 0
        for line in text.splitlines():
            parts = line.split("\t")
            if parts[0] == "put":
                if len(parts) == 3 and parts[1] in mine:
                    keep.append(line)
                    puts += 1
                continue
            gets += parts[0] == "get"
            keep.append(line)
        body = "\n".join(keep) + "\n"
    else:
        body = "#partition\t%s\n#image\t%s\n#leg\tgha-%s\n" % (partition(), IMAGE_DECL, a.leg or "local")
        gets = puts = 0
    fd = os.open(str(dest), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(body)
    print(f"map: {'R2' if url else 'none (no R2: the hot directory and actions/cache only, nothing uploads)'}, "
          f"{gets} gets, {puts} upload slots for this job")
    if a.github_output:
        # whether R2 is the cache: without it the job reads and writes actions/cache
        with open(a.github_output, "a") as fh:
            fh.write("r2=%s\n" % ("true" if url else "false"))
    return 0


def cmd_donor(a):
    """The host job's donor build from the event's presigned host_donor_url:
    fetched (the URL masked, never printed), its sha256 checked against the
    event's host_donor_sha256, extracted under DEST (DEST/build)."""
    import tarfile
    inputs = (json.loads(Path(a.event).read_text()).get("inputs") or {}) if a.event else {}
    url = (inputs.get("host_donor_url") or "").strip()
    want = (inputs.get("host_donor_sha256") or "").strip()
    if not url:
        raise SystemExit("donor: the dispatch carries no host_donor_url")
    if not re.fullmatch(r"[0-9a-f]{64}", want):
        raise SystemExit("donor: host_donor_sha256 must be 64 hex")
    print("::add-mask::" + url, flush=True)
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    arc = dest / "donor.tar.gz"
    with urllib.request.urlopen(url, timeout=600) as r, open(arc, "wb") as fh:
        shutil.copyfileobj(r, fh, 1 << 20)
    got = sha256(arc)
    if got != want:
        raise SystemExit(f"donor: the fetched archive is {got}, the dispatch named {want}")
    with tarfile.open(arc) as tf:
        names = tf.getnames()
        if not names or any(n != "build" and not n.startswith("build/") for n in names):
            raise SystemExit("donor: the archive holds more than build/")
        tf.extractall(dest, filter="data")
    arc.unlink()
    proof = json.loads((dest / "build" / "build-provenance.json").read_text())
    rev = proof.get("source_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", rev):
        raise SystemExit("donor: the donor proof names no source commit")
    print(f"donor: build of {rev[:12]}, sha256 {got[:16]}.. verified")
    if a.github_output:
        with open(a.github_output, "a") as fh:
            fh.write("revision=%s\n" % rev)
    return 0


# ---------------------------------------------------------------- the Mac
def read_creds(path=None):
    path = Path(os.path.expanduser(path or os.environ.get("MOJOLEARN_R2_CREDS", "~/.mojolearn_r2")))
    creds = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            creds[k.strip()] = v.strip().strip("'\"")
    need = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
    if not all(creds.get(k) for k in need):
        raise SystemExit(f"{path} lacks one of {', '.join(need)}")
    return {k: creds[k] for k in need}


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def gh(*args, check=True):
    r = subprocess.run([GH, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:3])} exited {r.returncode}: {r.stderr.strip()[-300:]}")
    return r.stdout


def workflow_id():
    """The workflow's numeric id. A workflow that is not on the default branch
    yet is found by its path among the registered ones (GitHub resolves a
    file name only on the default branch)."""
    out = gh("api", "--paginate", "repos/{owner}/{repo}/actions/workflows",
             "--jq", '.workflows[] | "\\(.id) \\(.path)"')
    for line in out.splitlines():
        wid, _, path = line.strip().partition(" ")
        if path == ".github/workflows/" + WORKFLOW and wid.isdigit():
            return wid
    raise SystemExit(f".github/workflows/{WORKFLOW} is not registered with GitHub (docs/RELEASE_CHECKLIST.md 2d)")


def tooling_ref(ref):
    """(branch, sha): the branch the workflow runs from must hold exactly this
    checkout's HEAD, and the route files must be committed."""
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", *ROUTE_FILES)
    if dirty:
        raise SystemExit("the GitHub route's files are not committed here:\n" + dirty)
    branch = ref or git("rev-parse", "--abbrev-ref", "HEAD")
    remote = subprocess.run(["git", "-C", str(ROOT), "ls-remote", "origin", "refs/heads/" + branch],
                            capture_output=True, text=True).stdout.split()
    if not remote or remote[0] != head:
        raise SystemExit(f"origin/{branch} is {remote[0][:12] if remote else 'absent'}, this checkout is "
                         f"{head[:12]}: push it (the runners use the pushed tooling)")
    return branch, head


def mint_map(creds, jobs_per_set, nsets, leg):
    import bincache
    lines, n_get, _ = bincache.plan_lines(creds, partition(), IMAGE_DECL, leg, jobs_per_set * nsets * SLOTS_PER_JOB,
                                          expires=MAP_HOURS * 3600)
    key = f"{MAP_PREFIX}/{leg}.tsv"
    status, _ = bincache.r2_request(creds, "PUT", key, data=("\n".join(lines) + "\n").encode())
    if status != 200:
        raise SystemExit(f"could not store the URL map in R2 (HTTP {status})")
    return bincache.presign("GET", key, MAP_HOURS * 3600, creds), key, n_get


def find_run(tag, branch, wid, tries=30, pause=5):
    for _ in range(tries):
        rows = json.loads(gh("run", "list", "--workflow", wid, "--branch", branch, "--event", "workflow_dispatch",
                             "--limit", "30", "--json", "databaseId,displayTitle,headSha,url") or "[]")
        for r in rows:
            if tag in (r.get("displayTitle") or ""):
                return r
        time.sleep(pause)
    raise SystemExit(f"no {WORKFLOW} run titled with {tag} appeared")


def seconds(a, b):
    try:
        f = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        return int((f(b) - f(a)).total_seconds())
    except (TypeError, ValueError, AttributeError):
        return None


def wait_run(run_id, timeout_min, pause):
    deadline = time.time() + timeout_min * 60
    last = None
    while True:
        # A failed status read must not strand an otherwise healthy build's
        # artifacts. Retry only this read, never workflow dispatch. Both
        # 216d04c7c collectors exited on a transient api.github.com connection
        # error while their remote jobs continued (2026-10-26).
        for attempt in range(3):
            try:
                view = json.loads(gh("run", "view", str(run_id), "--json",
                                     "status,conclusion,jobs,headSha,url,createdAt,updatedAt"))
                break
            except RuntimeError as exc:
                transient = any(text in str(exc).lower() for text in (
                    "error connecting", "connection reset", "connection refused",
                    "timed out", "timeout", "http 502", "http 503", "http 504"))
                if not transient or attempt == 2 or time.time() >= deadline:
                    raise
                say(f"run {run_id} status read failed; retry {attempt + 1}/2: {exc}")
                time.sleep(min(5, max(0, deadline - time.time())))
        jobs = view.get("jobs") or []
        line = ", ".join(f"{j['name']}={j.get('conclusion') or j.get('status')}" for j in jobs)
        if line != last:
            say(f"run {run_id} {view.get('status')}: {line}")
            last = line
        if view.get("status") == "completed":
            return view
        if time.time() > deadline:
            raise SystemExit(f"run {run_id} did not finish in {timeout_min} min ({view.get('url')})")
        time.sleep(pause)


def verify_tree(release_build, vendor, arch, commit):
    """release.py's verify_leg_tree, read-only: a complete proof of COMMIT and
    every binary it names on disk with its sha256."""
    rb = Path(release_build)
    try:
        proof = json.loads((rb / "build" / "build-provenance.json").read_text())
    except (OSError, ValueError):
        return False, f"no build proof under {rb}"
    if proof.get("complete") is not True or proof.get("build_exit") != 0 or proof.get("source_commit") != commit:
        return False, "the build proof is not complete, not exit 0 or not of " + commit[:12]
    ext = dict(proof.get("extensions") or {}, **(proof.get("host_extension") or {}))
    if not ext:
        return False, "the proof names no binary"
    for name, digest in ext.items():
        parts = Path(name).parts
        if parts[:3] != ("mojolearn", vendor, arch):
            return False, f"the proof names {name}, outside {vendor}/{arch}"
        p = rb / "build" / "sets" / Path(*parts[1:])
        if not p.is_file() or sha256(p) != digest:
            return False, f"{name} is missing or not the proof's bytes"
    # The runtime closure (.libs/) is not in the proof; the set's manifest
    # names every staged library and its sha256 (an artifact upload that
    # drops dot-directories loses it silently).
    set_dir = rb / "build" / "sets" / vendor / arch
    try:
        staged = json.loads((set_dir / "manifest.json").read_text()).get("staged_libs") or []
    except (OSError, ValueError):
        return False, "no manifest.json in the set"
    if not staged:
        return False, "the manifest names no staged runtime library"
    for lib in staged:
        p = set_dir / ".libs" / lib["name"]
        if not p.is_file() or sha256(p) != lib["sha256"]:
            return False, f".libs/{lib['name']} is missing or not the manifest's bytes"
    return True, (f"{len(proof['extensions'])} set binaries + {len(proof.get('host_extension') or {})} host"
                  f" + {len(staged)} runtime libraries, all as proved")


def promote(out, creds):
    """Promote the run's inbox uploads whose R2 object is the archive a job recorded."""
    import bincache
    gha = Path(out) / "GHA" / "bincache"
    up = gha / "uploads.tsv"
    if not up.is_file() or not up.read_text().strip():
        return "no uploads"
    want = {}
    for line in (gha / "hot_sha256.tsv").read_text().splitlines() if (gha / "hot_sha256.tsv").is_file() else []:
        k, _, h = line.partition("\t")
        want[k] = h
    good, refused = [], []
    for row in up.read_text().splitlines():
        parts = row.split("\t")
        if len(parts) != 5:
            continue
        leg, slot, _, key = parts[:4]
        with tempfile.NamedTemporaryFile() as tmp:
            code = bincache.http_get(bincache.presign("GET", f"{bincache.INBOX_PREFIX}/{leg}/{slot}.tar.gz", 600, creds),
                                     tmp.name)
            ok = code == 200 and want.get(key) and sha256(tmp.name) == want[key]
        (good if ok else refused).append(row)
    stage = Path(out) / "GHA" / "promote"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "keys").mkdir(parents=True)
    (stage / "uploads.tsv").write_text("".join(r + "\n" for r in good))
    for k in (gha / "keys").glob("*.json"):
        shutil.copy2(k, stage / "keys" / k.name)
    env = dict(os.environ, **creds)
    r = subprocess.run([sys.executable, str(ROOT / "tools/bincache.py"), "promote", str(stage)],
                       capture_output=True, text=True, env=env)
    (Path(out) / "GHA" / "promote.log").write_text(r.stdout + r.stderr
                                                    + "".join(f"  refused (not the job's archive): {x}\n" for x in refused))
    return (r.stdout.strip().splitlines() or ["?"])[-1] + (f"; {len(refused)} refused (bytes not the job's)" if refused else "")


def verify_host(out, commit):
    """The host producer's output (tools/release_host_build.py build), read-only:
    a complete proof of COMMIT whose staged host bytes are on disk as proved."""
    try:
        proof = json.loads((Path(out) / "host-build-proof.json").read_text())
    except (OSError, ValueError):
        return False, f"no host-build-proof.json under {out}"
    if (proof.get("schema") != "mojolearn.linux.host-build.v1" or proof.get("complete") is not True
            or proof.get("build_exit") != 0 or proof.get("source_commit") != commit):
        return False, "the host proof is not complete, not exit 0 or not of " + commit[:12]
    ext = proof.get("host_extension") or {}
    if not ext:
        return False, "the host proof names no binary"
    for name, digest in ext.items():
        if not name.startswith("mojolearn/cpu/"):
            return False, f"the host proof names {name}, outside mojolearn/cpu/"
        p = Path(out) / "set" / name[len("mojolearn/cpu/"):]
        if not p.is_file() or sha256(p) != digest:
            return False, f"{name} is missing or not the proof's bytes"
    return True, f"host {', '.join(sorted(ext))} as proved"


def upload_donor(creds, donor, tag):
    """Pack a retained donor build (<leg>/release-build/build) into R2 for the
    host job: (presigned GET, sha256, object key). The URL is passed once as a
    workflow input, read from the event file by the job and masked there."""
    import bincache
    import tarfile
    donor = Path(donor).resolve()
    if not (donor / "build-provenance.json").is_file() or not (donor / "compile-inputs.json").is_file():
        raise SystemExit(f"--host-donor {donor} is not a release build directory (build-provenance.json, "
                         "compile-inputs.json)")
    key = f"{DONOR_PREFIX}/{tag}.tar.gz"
    with tempfile.TemporaryDirectory() as td:
        arc = Path(td) / "donor.tar.gz"
        # the leg's pinned CUDA tools archive (cuda-tools-*) is not part of the donor contract
        # (release_host_composition.inspect_build): left out, it would be most of the upload
        skip = lambda ti: None if re.fullmatch(r"build/cuda-tools-[^/]*", ti.name) else ti
        with tarfile.open(arc, "w:gz", compresslevel=1) as tf:
            tf.add(str(donor), arcname="build", filter=skip)
        digest = sha256(arc)
        code = bincache.http_put(bincache.presign("PUT", key, 3600, creds), str(arc))
        if code != 200:
            raise SystemExit(f"could not store the donor build in R2 (HTTP {code})")
    return bincache.presign("GET", key, MAP_HOURS * 3600, creds), digest, key


def download(run_id, name, out, failed):
    """Artifact NAME of the run, moved into OUT (merged into what is there)."""
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([GH, "run", "download", str(run_id), "-n", name, "-D", td], capture_output=True, text=True)
        if r.returncode != 0:
            names = "; ".join(f"{x[0]} ({x[1]}) {x[5]}" for x in failed) or "none failed"
            raise SystemExit(f"no artifact {name} from run {run_id} ({r.stderr.strip()[-200:]}); failed jobs: {names}")
        src = Path(td) / ("leg" if (Path(td) / "leg").is_dir() else "")
        Path(out).mkdir(parents=True, exist_ok=True)
        for p in sorted(src.iterdir()):
            dest = Path(out) / p.name
            if dest.exists():
                if p.is_dir():
                    shutil.copytree(p, dest, dirs_exist_ok=True, symlinks=True)
                    continue
                dest.unlink()
            shutil.move(str(p), str(dest))


def run_modes(text):
    """The release modes the box builds: --modes, else MOJOLEARN_RELEASE_MODES
    (release.py sets it for every leg), else the release's default, identical."""
    import release_modes
    try:
        return release_modes.parse_modes(text or os.environ.get(release_modes.ENV) or "identical")
    except ValueError as exc:
        raise SystemExit(str(exc))


def cmd_run(a):
    if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
        raise SystemExit("--commit must be a full 40-hex SHA")
    if not a.arch and not a.host:
        raise SystemExit("--arch and/or --host is required")
    if a.arch and a.arch not in ARCHS:
        raise SystemExit(f"--arch must be one of {' '.join(ARCHS)}")
    if a.host and not a.arch and not a.host_donor and not a.run_id:
        raise SystemExit("--host without --arch needs --host-donor DIR (a retained <leg>/release-build/build)")
    modes = run_modes(a.modes)
    vendor = ARCHS.get(a.arch)
    name = f"{vendor}-{a.arch}" if a.arch else None
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()) and not a.run_id:
        raise SystemExit(f"{out} exists and is not empty")
    gha = out / "GHA"
    gha.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    creds = None
    if not a.no_bincache or a.host_donor:
        try:
            creds = read_creds()
        except (OSError, SystemExit) as exc:
            if a.host_donor:
                raise SystemExit(f"--host-donor goes to the run through R2 and needs credentials ({exc})")
            raise SystemExit(f"no R2 credentials for the binding cache ({exc}); --no-bincache builds without it")
    if a.no_bincache and a.arch == "gfx942" and not a.allow_cold_amd:
        raise SystemExit("gfx942 without the binding cache would ship bytes no later build reproduces "
                         "(gfx942 codegen varies run to run); pass --allow-cold-amd to build it anyway")
    cleanup = []
    if a.run_id:
        run = json.loads(gh("run", "view", str(a.run_id), "--json", "databaseId,displayTitle,headSha,url"))
        say(f"attached to run {a.run_id} ({run.get('url')})")
    else:
        branch, head = tooling_ref(a.ref)
        tag = f"gha-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.urandom(3).hex()}"
        map_url = donor_url = donor_sha = ""
        jobs_per_set = max(a.shards + 1, MERGED_JOBS_PER_SET)
        if creds and not a.no_bincache and a.arch:
            map_url, map_key, n_get = mint_map(creds, jobs_per_set, 1, tag)
            cleanup.append(map_key)
            say(f"binding cache map: partition {partition()}, {n_get} objects, "
                f"{jobs_per_set * SLOTS_PER_JOB} upload slots, valid {MAP_HOURS} h")
        if a.host_donor:
            donor_url, donor_sha, donor_key = upload_donor(creds, a.host_donor, tag)
            cleanup.append(donor_key)
            say(f"host donor {a.host_donor}: sha256 {donor_sha[:16]}.., valid {MAP_HOURS} h")
        sets = " ".join(x for x in (a.arch, "host" if a.host else "") if x)
        wid = workflow_id()
        gh("workflow", "run", wid, "--ref", branch, "-f", f"commit={a.commit}", "-f", f"sets={sets}",
           "-f", f"shards={a.shards}", "-f", f"jobs={a.jobs}", "-f", f"tag={tag}", "-f", f"map_url={map_url}",
           "-f", f"modes={' '.join(modes)}", "-f", f"host_donor_url={donor_url}",
           "-f", f"host_donor_sha256={donor_sha}")
        (gha / "dispatch.json").write_text(json.dumps(dict(
            tag=tag, branch=branch, tooling_commit=head, source_commit=a.commit, arch=a.arch, host=a.host,
            modes=list(modes), shards=a.shards, jobs=a.jobs, bincache=bool(map_url), host_donor=a.host_donor or None,
            host_donor_sha256=donor_sha or None, r2_objects=cleanup,
            dispatched_at=dt.datetime.now(dt.timezone.utc).isoformat()), indent=1) + "\n")
        run = find_run(tag, branch, wid)
        say(f"dispatched {run['url']}")
        if run.get("headSha") != head:
            raise SystemExit(f"the run is at {run.get('headSha')}, this checkout at {head}")
        a.run_id = run["databaseId"]
    try:
        view = wait_run(a.run_id, a.timeout_min, a.poll)
    finally:
        if cleanup and creds:
            import bincache
            for key in cleanup:
                try:
                    bincache.r2_request(creds, "DELETE", key)
                except Exception:
                    pass
    rows = []
    for j in view.get("jobs") or []:
        rows.append((j.get("name", ""), j.get("conclusion") or j.get("status"), j.get("startedAt"),
                     j.get("completedAt"), seconds(j.get("startedAt"), j.get("completedAt")), j.get("url", "")))
    (gha / "jobs.tsv").write_text("".join("\t".join(str(x) for x in r) + "\n" for r in rows))
    (gha / "run.json").write_text(json.dumps(view, indent=1) + "\n")
    failed = [r for r in rows if r[1] not in ("success", "skipped")]
    lines = [f"run {view.get('url')} {view.get('conclusion')}"]
    ok = True
    if a.arch:
        download(a.run_id, f"release-{name}", out, failed)
        good, why = verify_tree(out / name / "release-build", vendor, a.arch, a.commit)
        ok &= good
        lines.append(("TREE OK: " if good else "TREE REFUSED: ") + why)
    if a.host:
        download(a.run_id, "host-build", out / "host-build", failed)
        good, why = verify_host(out / "host-build", a.commit)
        ok &= good
        lines.append(("HOST OK: " if good else "HOST REFUSED: ") + why)
    wall = int(time.time() - t0)
    lines[0] += f" wall_seconds={wall}"
    lines[1:1] = [f"  {r[0]}: {r[1]} {r[4]}s" for r in rows]
    if failed:
        lines.append("FAILED JOBS: " + "; ".join(f"{x[0]} ({x[1]}) {x[5]}" for x in failed))
    if ok and creds and a.arch and not a.no_bincache:
        lines.append("binding cache: " + promote(out, creds))
    if ok:
        lines += next_steps(out, name, a)
    (out / "wall.txt").write_text(f"wall_seconds={wall} run={a.run_id} conclusion={view.get('conclusion')}\n")
    (gha / "summary.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        say(line)
    return 0 if ok else 1


def next_steps(out, name, a):
    """The verified artifacts' hand-off: the Linux packer for a fresh set, the
    retained composition for a host built against a donor."""
    rows = ["NEXT (docs/RELEASE_CHECKLIST.md section 3):"]
    if name:
        b = out / name / "release-build" / "build"
        rows.append(f"  pack: --set {b / 'sets' / ARCHS[a.arch]} --build-proof {b / 'build-provenance.json'}")
    if a.host:
        donor = a.host_donor or (str(out / name / "release-build" / "build") if name else "<donor build>")
        rows.append(f"  compose: python3 tools/retained_release_composition.py --device-build <retained leg>/release-build/build"
                    f" --host-build {out / 'host-build'} --source-root <frozen checkout> --compiler-sha256 "
                    f"<compile-inputs.json compiler_sha256> --out <assembled leg>   (host donor: {donor})")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--src", required=True)
    p.add_argument("--sets", required=True, help="device sets (sm_89 sm_80 gfx942) and/or host")
    p.add_argument("--shards", type=int, default=6, help="per-binding layout: shard jobs per set "
                   "(merged: one shard per merged compile, whatever this says)")
    p.add_argument("--modes", default="", help="the release modes (MOJOLEARN_RELEASE_MODES); empty: "
                   "build_sets.sh's default tiers")
    p.add_argument("--host-donor", action="store_true", help="the dispatch carries host_donor_url")
    p.add_argument("--json")
    p.add_argument("--github-output")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("estimate")
    p.add_argument("--src", default=str(ROOT))
    p.add_argument("--sets", default="sm_89 sm_80 gfx942 host")
    p.add_argument("--shards", type=int, default=6)
    p.add_argument("--modes", default="identical")
    p.set_defaults(fn=cmd_estimate)
    p = sub.add_parser("layout")
    p.add_argument("--commit", required=True)
    p.add_argument("--planner", action="store_true")
    p.set_defaults(fn=cmd_layout)
    p = sub.add_parser("job-map")
    p.add_argument("--event")
    p.add_argument("--dest", required=True)
    p.add_argument("--set-index", type=int, required=True)
    p.add_argument("--job", type=int, required=True, help="the shard index, or JOBS_PER_SET - 1 for the assemble job")
    p.add_argument("--jobs-per-set", type=int, required=True)
    p.add_argument("--leg", default="")
    p.add_argument("--github-output")
    p.set_defaults(fn=cmd_job_map)
    p = sub.add_parser("donor")
    p.add_argument("--event", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--github-output")
    p.set_defaults(fn=cmd_donor)
    p = sub.add_parser("run")
    p.add_argument("--commit", required=True)
    p.add_argument("--arch", default="", help="one device set: " + " ".join(ARCHS))
    p.add_argument("--host", action="store_true", help="also run the host producer (tools/release_host_build.py)")
    p.add_argument("--host-donor", default="", help="a retained <leg>/release-build/build as the host's runtime "
                   "donor (default: this run's --arch set)")
    p.add_argument("--modes", default="", help="release modes (default: MOJOLEARN_RELEASE_MODES, else identical)")
    p.add_argument("--out", required=True)
    p.add_argument("--ref", default="", help="the pushed branch holding this checkout's HEAD (default: the current branch)")
    p.add_argument("--shards", type=int, default=6)
    p.add_argument("--jobs", type=int, default=2, help="builds at once per runner (MOJOLEARN_BUILD_JOBS)")
    p.add_argument("--no-bincache", action="store_true",
                   help="no R2: the runners reuse their actions/cache hot archives instead")
    p.add_argument("--allow-cold-amd", action="store_true")
    p.add_argument("--run-id", type=int, default=0, help="attach to this run instead of dispatching")
    p.add_argument("--timeout-min", type=int, default=420)
    p.add_argument("--poll", type=int, default=60)
    p.set_defaults(fn=cmd_run)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
