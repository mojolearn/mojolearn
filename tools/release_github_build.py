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

THE MERGED LAYOUT IS NOT BUILT HERE (2026-10-21). A source whose
packaging/linux/merged_layout.py says merged (MOJOLEARN_MERGED_BINDINGS=1,
the default since 2026-10-20) builds ONE compile per tier, measured at up to
46.4 GB; a free runner has 15. `plan` plans exactly build_sets.sh's merged
pool (fast, identical and host merged_build.sh, one per shard) and then
refuses it, and `run` refuses before dispatching, both with merged_refusal(),
unless --runner-mem-gib says the runners have NEEDED_GIB. Such a source
builds on the cpu-box route (tools/release.py's default for it).

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
#: tools/release_linux_build.sh's route overlay (a test holds them equal).
OVERLAY = ("tools/release061_remote_build.sh", "tools/cpu_build_guard.py", "tools/release_linux_cpu_box.sh",
           "tools/bincache.py")
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
#: THE MERGED LAYOUT DOES NOT FIT A FREE RUNNER (2026-10-21). With
#: MOJOLEARN_MERGED_BINDINGS=1 (packaging/linux/merged_layout.py, the default
#: since lane/merged-bindings, 2026-10-20) build_sets.sh's pool runs ONE merged
#: compile per directory set (tools/merged_bindings.py plan --platform linux
#: --byte-lm 1: one group `_mojolearn_native` per tier, `_mojolearn_native_host`
#: for host/). Measured on the DO MI325X 2026-10-20, two compiler workers each:
#: host 6m23s peak 6.9 GB, fast 16m peak 36.6 GB, identical (+ byte LM)
#: 18m45s peak 46.4 GB. GiB a runner must have for each, the peak rounded up
#: with headroom for the toolchain and the read-backs:
NEEDED_GIB = {"fast": 40, "identical": 50, "host": 8}
MEASURED_PEAK = "identical (+ byte LM) 46.4 GB, fast 36.6 GB, host 6.9 GB"
#: GitHub's free ubuntu-24.04 runner (4 vCPU); the org has no larger runners.
FREE_RUNNER = "4 vCPU, 15 GB"


def merged_refusal(needed_gib):
    """The one sentence every GitHub entry point refuses the merged layout with."""
    return (f"the merged binding layout (MOJOLEARN_MERGED_BINDINGS=1) cannot build on GitHub's free runners "
            f"({FREE_RUNNER}): one merged compile per tier peaked at {MEASURED_PEAK} on the DO MI325X "
            f"2026-10-20, so build the Linux sets on the cpu-box route (`pixi run release <v>` defaults to "
            f"it; one set: `bash tools/release_linux_build.sh <commit> --rent --archs <arch>`), or pass "
            f"--runner-mem-gib N when the runners have N >= {needed_gib} GiB")


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


def plan_set(src):
    """One set's builds from the SOURCE's own build_sets.sh: its list variables,
    tier_scripts() and pool_jobs() are evaluated (not retyped) with the
    release's MOJOLEARN_PACKAGE_BYTE_LM=1, over its tiers. Returns
    dict(builds=[(entry, is_host)], per_binding=[(entry, is_host)],
    merged=bool, weights={entry: w}, members={entry: n}). `builds` is exactly
    what pool_jobs() runs: the per-binding scripts, or (merged) one
    "<tier>:merged_build.sh" per tier plus "host:merged_build.sh" riding the
    identical pass; a merged build's weight is its members' per-binding weights."""
    text = (Path(src) / "packaging/linux/build_sets.sh").read_text()
    m = re.search(r"^SCRIPTS=.*?^host_so\(\)", text, re.M | re.S)
    if not m:
        raise SystemExit("plan: packaging/linux/build_sets.sh has no SCRIPTS= ... host_so() block")
    block = text[m.start():m.end() - len("host_so()")]
    tiers = "fast identical"
    t = re.search(r'^TIERS="\$\{MOJOLEARN_BUILD_TIERS:-([a-z ]+)\}"', text, re.M)
    if t:
        tiers = t.group(1)
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
    if merged != any(e.endswith(":merged_build.sh") for e in jobs):
        raise SystemExit(f"plan: merged_layout.py says {'merged' if merged else 'per-binding'} but pool_jobs() "
                         f"runs {' '.join(jobs)[:200]}")
    pw = lambda e: WEIGHTS.get(e, DEFAULT_HOST_WEIGHT if is_host(e) else DEFAULT_GPU_WEIGHT)
    weights, members = {}, {}
    for e in jobs:
        if e.endswith(":merged_build.sh"):
            tier = e.split(":")[0]
            mine = [p for p in per if (is_host(p) if tier == "host" else (not is_host(p) and p.split(":")[0] == tier))]
            if not mine:
                raise SystemExit(f"plan: the merged {tier} build has no member in the per-binding lists")
            weights[e], members[e] = sum(pw(p) for p in mine), len(mine)
        else:
            weights[e], members[e] = pw(e), 1
    return dict(builds=[(e, is_host(e)) for e in jobs], per_binding=[(e, is_host(e)) for e in per],
                merged=merged, weights=weights, members=members)


def set_builds(src):
    """[(entry, is_host)] of one set, "<tier>:<script>": what build_sets.sh's pool runs (plan_set)."""
    return plan_set(src)["builds"]


def check_counts(src, builds):
    """The GPU builds per tier must be the verifier's expected bindings."""
    sys.path.insert(0, str(Path(src) / "tools"))
    try:
        from verify_linux_surface_qualification import MODES, expected_bindings
    finally:
        sys.path.pop(0)
    for mode in MODES:
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


def cmd_plan(a):
    sets = a.sets.split()
    if not sets or any(s not in ARCHS for s in sets) or len(set(sets)) != len(sets):
        raise SystemExit(f"plan: sets must be distinct names from {' '.join(ARCHS)}")
    if not 1 <= a.shards <= 16:
        raise SystemExit("plan: shards must be 1..16")
    ps = plan_set(a.src)
    builds = ps["builds"]
    # the verifier's bindings are the per-binding members, in either layout
    check_counts(a.src, ps["per_binding"])
    shards = a.shards
    if ps["merged"]:
        needed = max(NEEDED_GIB[e.split(":")[0]] for e, _ in builds)
        if a.runner_mem_gib < needed:
            raise SystemExit("plan: " + merged_refusal(needed))
        # one merged compile per shard at most: a merged build cannot be split
        shards = min(shards, len(builds))
    include = []
    for si, arch in enumerate(sets):
        for sh in split(builds, shards, ps["weights"]):
            include.append(dict(arch=arch, vendor=ARCHS[arch], set_index=si, **sh))
    doc = dict(matrix=dict(include=include), sets=sets, builds=len(builds), shards=shards,
               layout="merged" if ps["merged"] else "per-binding", members=ps["members"])
    text = json.dumps(doc, indent=1)
    if a.json:
        Path(a.json).write_text(text + "\n")
    if a.github_output:
        with open(a.github_output, "a") as fh:
            fh.write("matrix=" + json.dumps(doc["matrix"], separators=(",", ":")) + "\n")
            fh.write("sets=" + json.dumps(sets, separators=(",", ":")) + "\n")
    for row in include:
        print(f"{row['vendor']}-{row['arch']} shard {row['shard']}: {row['n']} builds, weight {row['weight']}")
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
    print(f"map: {'R2' if url else 'none (no R2: every build compiles here, nothing uploads)'}, "
          f"{gets} gets, {puts} upload slots for this job")
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
        view = json.loads(gh("run", "view", str(run_id), "--json",
                             "status,conclusion,jobs,headSha,url,createdAt,updatedAt"))
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


def cmd_run(a):
    if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
        raise SystemExit("--commit must be a full 40-hex SHA")
    if a.arch not in ARCHS:
        raise SystemExit(f"--arch must be one of {' '.join(ARCHS)}")
    vendor, name = ARCHS[a.arch], f"{ARCHS[a.arch]}-{a.arch}"
    if not a.run_id and commit_merged(a.commit) and a.runner_mem_gib < max(NEEDED_GIB.values()):
        # the workflow's plan job would refuse it; refuse before dispatching
        raise SystemExit(merged_refusal(max(NEEDED_GIB.values())))
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()) and not a.run_id:
        raise SystemExit(f"{out} exists and is not empty")
    gha = out / "GHA"
    gha.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    creds = None
    if not a.no_bincache:
        try:
            creds = read_creds()
        except (OSError, SystemExit) as exc:
            raise SystemExit(f"no R2 credentials for the binding cache ({exc}); --no-bincache builds without it")
    elif a.arch == "gfx942" and not a.allow_cold_amd:
        raise SystemExit("gfx942 without the binding cache would ship bytes no later build reproduces "
                         "(gfx942 codegen varies run to run); pass --allow-cold-amd to build it anyway")
    map_key = None
    if a.run_id:
        run = json.loads(gh("run", "view", str(a.run_id), "--json", "databaseId,displayTitle,headSha,url"))
        say(f"attached to run {a.run_id} ({run.get('url')})")
    else:
        branch, head = tooling_ref(a.ref)
        tag = f"gha-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.urandom(3).hex()}"
        map_url = ""
        if creds:
            map_url, map_key, n_get = mint_map(creds, a.shards + 1, 1, tag)
            say(f"binding cache map: partition {partition()}, {n_get} objects, "
                f"{(a.shards + 1) * SLOTS_PER_JOB} upload slots, valid {MAP_HOURS} h")
        wid = workflow_id()
        gh("workflow", "run", wid, "--ref", branch, "-f", f"commit={a.commit}", "-f", f"sets={a.arch}",
           "-f", f"shards={a.shards}", "-f", f"jobs={a.jobs}", "-f", f"tag={tag}", "-f", f"map_url={map_url}")
        (gha / "dispatch.json").write_text(json.dumps(dict(
            tag=tag, branch=branch, tooling_commit=head, source_commit=a.commit, arch=a.arch, shards=a.shards,
            jobs=a.jobs, bincache=bool(creds), map_object=map_key, dispatched_at=dt.datetime.now(dt.timezone.utc).isoformat()),
            indent=1) + "\n")
        run = find_run(tag, branch, wid)
        say(f"dispatched {run['url']}")
        if run.get("headSha") != head:
            raise SystemExit(f"the run is at {run.get('headSha')}, this checkout at {head}")
        a.run_id = run["databaseId"]
    try:
        view = wait_run(a.run_id, a.timeout_min, a.poll)
    finally:
        if map_key and creds:
            import bincache
            try:
                bincache.r2_request(creds, "DELETE", map_key)
            except Exception:
                pass
    rows = []
    for j in view.get("jobs") or []:
        rows.append((j.get("name", ""), j.get("conclusion") or j.get("status"), j.get("startedAt"),
                     j.get("completedAt"), seconds(j.get("startedAt"), j.get("completedAt")), j.get("url", "")))
    (gha / "jobs.tsv").write_text("".join("\t".join(str(x) for x in r) + "\n" for r in rows))
    (gha / "run.json").write_text(json.dumps(view, indent=1) + "\n")
    failed = [r for r in rows if r[1] not in ("success", "skipped")]
    art = f"release-{name}"
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([GH, "run", "download", str(a.run_id), "-n", art, "-D", td], capture_output=True, text=True)
        if r.returncode != 0:
            names = "; ".join(f"{x[0]} ({x[1]}) {x[5]}" for x in failed) or "none failed"
            raise SystemExit(f"no artifact {art} from run {a.run_id} ({r.stderr.strip()[-200:]}); failed jobs: {names}")
        src = Path(td) / "leg"
        for p in sorted(src.iterdir()):
            dest = out / p.name
            if dest.exists():
                if p.is_dir():
                    shutil.copytree(p, dest, dirs_exist_ok=True, symlinks=True)
                    continue
                dest.unlink()
            shutil.move(str(p), str(dest))
    ok, why = verify_tree(out / name / "release-build", vendor, a.arch, a.commit)
    wall = int(time.time() - t0)
    lines = [f"run {view.get('url')} {view.get('conclusion')} wall_seconds={wall}"]
    lines += [f"  {r[0]}: {r[1]} {r[4]}s" for r in rows]
    if failed:
        lines.append("FAILED JOBS: " + "; ".join(f"{x[0]} ({x[1]}) {x[5]}" for x in failed))
    lines.append(("TREE OK: " if ok else "TREE REFUSED: ") + why)
    if ok and creds:
        lines.append("binding cache: " + promote(out, creds))
    (out / "wall.txt").write_text(f"wall_seconds={wall} run={a.run_id} conclusion={view.get('conclusion')}\n")
    (gha / "summary.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        say(line)
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--src", required=True)
    p.add_argument("--sets", required=True)
    p.add_argument("--shards", type=int, default=6)
    p.add_argument("--json")
    p.add_argument("--github-output")
    p.add_argument("--runner-mem-gib", type=int, default=0,
                   help="the runners' memory; the merged layout plans only when it covers NEEDED_GIB "
                        "(the workflow passes none: free runners)")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("layout")
    p.add_argument("--commit", required=True)
    p.add_argument("--planner", action="store_true")
    p.set_defaults(fn=cmd_layout)
    p = sub.add_parser("job-map")
    p.add_argument("--event")
    p.add_argument("--dest", required=True)
    p.add_argument("--set-index", type=int, required=True)
    p.add_argument("--job", type=int, required=True, help="the shard index, or SHARDS for the assemble job")
    p.add_argument("--jobs-per-set", type=int, required=True)
    p.add_argument("--leg", default="")
    p.set_defaults(fn=cmd_job_map)
    p = sub.add_parser("run")
    p.add_argument("--commit", required=True)
    p.add_argument("--arch", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ref", default="", help="the pushed branch holding this checkout's HEAD (default: the current branch)")
    p.add_argument("--shards", type=int, default=6)
    p.add_argument("--jobs", type=int, default=2, help="builds at once per runner (MOJOLEARN_BUILD_JOBS)")
    p.add_argument("--no-bincache", action="store_true")
    p.add_argument("--allow-cold-amd", action="store_true")
    p.add_argument("--run-id", type=int, default=0, help="attach to this run instead of dispatching")
    p.add_argument("--runner-mem-gib", type=int, default=0,
                   help="the runners' memory; a merged-layout commit dispatches only when it covers NEEDED_GIB")
    p.add_argument("--timeout-min", type=int, default=420)
    p.add_argument("--poll", type=int, default=60)
    p.set_defaults(fn=cmd_run)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
