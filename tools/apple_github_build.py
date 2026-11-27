#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE APPLE (macOS) WHEEL ON GITHUB'S FREE STANDARD macOS RUNNERS (lane mac-speed-1, 2026-11-27): the `github`
macOS build backend of tools/release.py (macos-build, macos-smoke) and of `tools/release.py candidate`, and the
steps its workflow runs.

    python3 tools/apple_github_build.py run --commit <sha40> --out DIR [--modes identical] [--ref BRANCH]
                                                 [--run-id N] [--lanes L,...] [--timeout-min 360]   (the Mac)
    python3 tools/apple_github_build.py plan --modes identical [--changed-since REV] [--groups D:S,...]  (workflow)
    python3 tools/apple_github_build.py job-map --event FILE --index N --dest FILE                     (workflow)
    python3 tools/apple_github_build.py group --dir D --stem S --out OUT --cache DIR [--map FILE]      (workflow)
    python3 tools/apple_github_build.py pack --archives DIR --out OUT                                  (workflow)
    python3 tools/apple_github_build.py lanes --wheel W --out OUT --lanes L,...                        (workflow)

PERMISSION (Andrew, 2026-11-27): GitHub-hosted STANDARD macOS runners, `macos-15` (or `macos-14`), arm64, 3 cores,
7 GB, free for this public repository. NEVER a larger/xlarge label (billed), a self-hosted runner or a rented Mac
(AGENTS.md, HARD RULE on Apple machines). RUNNER_LABELS is the only list the workflow and this tool accept.

WHY. The 0.9.0 Apple build ran on the laptop for 4571 s (about 900 s of it waiting for load in
tools/compile_slot.sh), and the run at f6a2ad489a compiled all 25 merged files again (19,501 s of compile, 5615 s
wall) after a one-line change in autograd/nn_conv_ops.mojo. The closure keys were right (13 of the 25 closures read
that file; tools/compile_closures.py plan); the other 12 missed because the laptop's key carries the checkout path
(bincache repo_path: the Mojo rpath into .pixi is baked into the bytes) and every freeze checks the source out at a
new path, <version>/source/<commit12>/. On a runner the path is always the workflow's checkout, so an unchanged
group is a hit in R2 from one build to the next. Its Metal-only failure (float64 in a device kernel) showed only at
the release; on this route it shows in the integration round (tools/release.py candidate builds it too).

WHAT RUNS WHERE (.github/workflows/apple-release-build.yml, workflow_dispatch only):
  plan    (ubuntu) the Darwin merged plan of the release modes and the host set (tools/merged_bindings.py plan
          --platform darwin --byte-lm 1): one row per merged file;
  group   (macOS, one job per merged file, all at once: the account's macOS concurrency, 5 on the free plan, is the
          only bound) packaging/macos/build_release_wheel.sh itself with MOJOLEARN_RELEASE_MACOS_GROUPS="<dir>:<stem>"
          and the release build's cache route (MOJOLEARN_BINCACHE_DIR), so the group's environment, key and bytes
          are the full build's; MOJOLEARN_COMPILE_JOBS = the runner's cores. Before it compiles, the group's key
          (tools/bincache.py explain with the build's own environment, bincache.macos_merged_group) is looked up in
          R2 (partition PARTITION) and a verified archive is placed instead. A GPU file whose bytes hold no Metal
          kernels (packaging/macos/check_gpu_embedded.py: a hosted runner once emitted none, TestPyPI 0.1.0a2) fails
          the job and is never uploaded;
  pack    (macOS) every group's archive in one cache, every group checked to be a HIT under the pack's environment
          (a miss compiles here, serially, and is named: APPLE PACK MISS), then build_release_wheel.sh --build-only
          over the placed bytes: the release's own staging, signing, ISA, floor, GPU-kernel and AIR-contract gates and
          the wheel. Its cache directory (archives + provenance) and log are the release's macos-bincache and
          macos-build.log, which the ledger adapter (tools/release_requirements_macos_build.py) admits exactly as
          the laptop build's ("placed" groups name their original compile, a runner);
  smoke   (macOS) tools/wheel_self_test.py on the wheel (fresh venv, `verify --tamper-test`) and the Apple lanes
          (`verify --all --lanes ...` from the installed wheel). Whether the runner's Metal device is usable is read
          from the self-test's gpu side (sides.gpu.ran and its clean state), recorded in apple-lanes.json as
          metal_usable, and said in the summary. The release takes the runner's self-test receipt only when it is
          PASSED for these wheel bytes; otherwise macos-smoke runs on this Mac as before. The lanes FAIL the release
          only when the runner's Metal was usable (a lane that crashed on a working Metal device is a product bug);
          with no usable Metal they are reported, never a pass.

THE BINDING CACHE WITHOUT A SECRET is tools/release_github_build.py's: R2 credentials never leave the Mac. `run`
lists the partition, presigns a GET per object and a PUT per job slot (bincache.plan_lines), stores the map in R2 and
passes ONE presigned GET as the input map_url, which each job reads from the event file and masks. After the run the
Mac promotes an inbox upload only when the R2 object hashes to the archive the job recorded
(release_github_build.promote). The keys are the LOCAL cache's (bincache local_fields: the runner image's Xcode,
Metal compiler, SDK and macOS build are key fields), so a runner archive is never placed on the laptop or back.

ACCELERATOR. Default-class files compile with no --target-accelerator, exactly the release Mac's line. A hosted VM
once failed that ("Unknown GPU architecture detected", 2026-08-22, an older Mojo); the first dispatch shows whether
Mojo 1.2 detects the runner's paravirtual GPU. If it does not, the fix is a keyed MOJOLEARN_APPLE_ACCELERATOR in
bindings/merged_build.sh (a one-time full rebuild of every group, so it is not made blind; RESULT lists it).

Stdlib only on the runner's system python3, plus tools/bincache.py and tools/release_github_build.py on the Mac.
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

WORKFLOW = "apple-release-build.yml"
#: The ONLY runner labels (Andrew, 2026-11-27): GitHub's free standard Apple-silicon runners. Never -xlarge/-large.
RUNNER_LABELS = ("macos-15", "macos-14")
DEFAULT_RUNNER = "macos-15"
#: R2 partition of the runner archives (bincache OBJECT_PREFIX/<partition>/<key>.tar.gz). The runner image is a key
#: field already (host_toolchain); the label keeps the listing small.
PARTITION_PREFIX = "apple/gha-"
#: Upload slots each group job may PUT into (one archive per job; spare for a retried upload).
SLOTS_PER_JOB = 4
MAP_HOURS = 10
MAP_PREFIX = "bincache/gha-maps"
#: The lanes the Apple smoke runs from the installed wheel by default: the two that crashed the 0.9.0 Apple record
#: on the laptop (EXC_BAD_ACCESS in AGX useResource from a worker thread in _mojolearn_native_m1.so), as lane names
#: of `verify --lanes` (their fixtures run with them).
DEFAULT_LANES = ("fn-msel-cross-val-predict", "x-metrics-permutation-test-score")
#: Files whose committed bytes the runners use; a dirty one refuses the dispatch.
ROUTE_FILES = ("tools/apple_github_build.py", ".github/workflows/" + WORKFLOW,
               "packaging/macos/build_release_wheel.sh", "tools/bincache.py", "tools/wheel_self_test.py")
SCRIPT = "./bindings/merged_build.sh"


def say(msg):
    print(f"[apple-gha {dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def partition(runner):
    if runner not in RUNNER_LABELS:
        raise SystemExit(f"runner {runner!r} is not a free standard macOS label {RUNNER_LABELS} (Andrew, 2026-11-27)")
    return PARTITION_PREFIX + runner.replace("macos-", "macos")


def cores():
    """The runner's cores (a hosted VM: its whole allocation)."""
    try:
        return int(subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True).stdout.strip())
    except ValueError:
        return os.cpu_count() or 1


# ---------------------------------------------------------------- the plan (on a runner)
def merged_dirs(modes):
    """The directory sets of the macOS wheel: the release modes, then host (the byte LM and host bindings ship)."""
    return [m for m in modes] + ["host"]


def plan_rows(repo, modes, groups=()):
    """[{key, dir, stem, cls, members}] of the Darwin release plan, in build order (identical first)."""
    import merged_bindings
    rows = []
    for d in merged_dirs(modes):
        for stem, cls, members in merged_bindings.plan(d, "darwin", True, "release", repo=repo):
            if groups and f"{d}:{stem}" not in groups:
                continue
            rows.append(dict(key=f"{d}-{stem}", dir=d, stem=stem, cls=cls, members=",".join(members)))
    return rows


def changed_rows(repo, rows, since):
    """The ROWS whose closure-v2 source closure or executed inputs hold a file changed since SINCE (git diff
    --name-only SINCE HEAD in REPO): the merged files a change can rebuild, for the cross-compile check."""
    import bincache
    diff = subprocess.run(["git", "-C", str(repo), "diff", "--name-only", since, "HEAD"], capture_output=True,
                          text=True, check=True).stdout.split()
    changed = set(diff)
    out = []
    for r in rows:
        kind = "host" if r["dir"] == "host" else "gpu"
        _, _, env = bincache.macos_merged_group({"MOJOLEARN_PACKAGE_BYTE_LM": "1"}, r["dir"], r["stem"])
        info = bincache.source_closure(repo, Path(repo) / "bindings" / "merged_build.sh",
                                       [kind, "--group", r["stem"]], env, plat="darwin")
        reads = set(info["closure"]) | set(info["inputs"]["files"]) | {"bindings/merged_groups.json",
                                                                         "packaging/macos/build_release_wheel.sh"}
        hit = sorted(changed & reads)
        if hit:
            out.append(dict(r, why=", ".join(hit[:3]) + (f" (+{len(hit) - 3})" if len(hit) > 3 else "")))
    return out


def cmd_plan(a):
    repo = Path(a.repo).resolve()
    modes = [m for m in a.modes.replace(",", " ").split() if m]
    groups = {g.strip() for g in a.groups.split(",") if g.strip()}
    rows = plan_rows(repo, modes, groups)
    if a.changed_since:
        rows = changed_rows(repo, rows, a.changed_since)
    for i, r in enumerate(rows):
        r["index"] = i
    print(json.dumps(rows))
    return 0


# ---------------------------------------------------------------- one job's URL map (on a runner)
def cmd_job_map(a):
    """This job's URL map from the event's map_url (masked), keeping every GET and only this job's PUT slots; a
    header-only map when the dispatch passed none (no R2: the job compiles and uploads nothing)."""
    event = json.loads(Path(a.event).read_text()) if a.event and Path(a.event).is_file() else {}
    url = ((event.get("inputs") or {}).get("map_url") or "").strip()
    dest = Path(a.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    mine = {"%03d" % i for i in range(a.index * SLOTS_PER_JOB, (a.index + 1) * SLOTS_PER_JOB)} if a.index >= 0 else set()
    gets = puts = 0
    if url:
        print("::add-mask::" + url, flush=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            text = r.read().decode()
        keep = []
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
        body = "#partition\tnone/none\n#image\tlocal\n#leg\tgha-apple-local\n"
    fd = os.open(str(dest), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(body)
    print(f"map: {'R2' if url else 'none (nothing is fetched or uploaded)'}, {gets} gets, {puts} upload slots")
    return 0


# ---------------------------------------------------------------- one merged file (on a runner)
def build_env(cache, modes, repo):
    """The environment of the macOS release build on a runner: the release build's own variables
    (bincache.MACOS_RELEASE_COMPILE_ENV) with every core as compiler workers (a hosted runner is the job's whole
    allocation; keyed, so these archives never mix with the laptop's one-worker ones)."""
    import bincache
    env = dict(os.environ, **bincache.MACOS_RELEASE_COMPILE_ENV)
    env.update(MOJOLEARN_BINCACHE_DIR=str(cache), MOJOLEARN_COMPILE_JOBS=str(cores()), MOJOLEARN_BUILD_JOBS="1",
               MOJOLEARN_RELEASE_MODES=" ".join(modes), MOJOLEARN_MERGED_BINDINGS="1",
               # what build_release_wheel.sh exports before its build loop, so `explain` keys a group as the loop does
               MOJOLEARN_REUSE_PLAN="", MOJOLEARN_REUSE_DIR="", MOJOLEARN_MERGED_KEY=merged_key(repo))
    env.pop("MOJOLEARN_RELEASE_MACOS_GROUPS", None)
    return env


def merged_key(repo):
    """build_release_wheel.sh's MOJOLEARN_MERGED_KEY: sha256 of merged_groups.json then merged_bindings.py."""
    h = hashlib.sha256()
    for rel in ("bindings/merged_groups.json", "tools/merged_bindings.py"):
        h.update((Path(repo) / rel).read_bytes())
    return h.hexdigest()


def explain_key(repo, env, d, stem):
    """(key, line) of one merged file under ENV (tools/bincache.py explain, the build's own per-group environment)."""
    import bincache
    kind, _output, genv = bincache.macos_merged_group(env, d, stem)
    r = subprocess.run([sys.executable, "tools/bincache.py", "explain", SCRIPT, kind, "--group", stem], cwd=repo,
                       env=genv, capture_output=True, text=True)
    line = (r.stdout.strip().splitlines() or [r.stderr.strip()[-300:]])[-1]
    m = re.search(r"\bkey=([0-9a-f]{64})\b", line)
    return (m.group(1) if m else ""), line


def provenance_rows(cache):
    """[(script, outcome, key)] of the cache's provenance.tsv (bincache Record.row: time, script, outcome, key, ...)."""
    p = Path(cache) / "provenance" / "provenance.tsv"
    rows = []
    for line in p.read_text().splitlines() if p.is_file() else []:
        parts = line.split("\t")
        if len(parts) >= 4:
            rows.append((parts[1], parts[2], "" if parts[3] == "-" else parts[3]))
    return rows


def output_of(d, stem):
    return f"python/mojolearn/host/{stem}.so" if d == "host" else (
        f"python/mojolearn/{stem}.so" if d == "fast" else f"python/mojolearn/{d}/{stem}.so")


def gpu_embedded(repo, so):
    """(ok, line) of packaging/macos/check_gpu_embedded.py on one GPU file (the release gate, here per file)."""
    r = subprocess.run(["pixi", "run", "-e", "pkg", "python", "packaging/macos/check_gpu_embedded.py", so], cwd=repo,
                       capture_output=True, text=True)
    text = (r.stdout + r.stderr).strip().splitlines()
    return r.returncode == 0, (text[-1] if text else f"rc={r.returncode}")


def upload(map_path, arc, key, out):
    """PUT ARC into this job's first free slot; record the row the Mac promotes (release_github_build.promote)."""
    import bincache
    urls = bincache.read_map(map_path)
    if not urls["put"]:
        return "no upload slot (no R2 map)"
    slot, url = bincache.claim_slot(map_path, urls["put"])
    if slot is None:
        return "no free upload slot"
    code = bincache.http_put(url, arc)
    if code != 200:
        return f"upload failed HTTP {code}"
    gha = Path(out) / "GHA" / "bincache"
    (gha / "keys").mkdir(parents=True, exist_ok=True)
    leg = urls["header"].get("leg", "")
    dest = "%s/%s/%s.tar.gz" % (bincache.OBJECT_PREFIX, urls["header"].get("partition", ""), key)
    with open(gha / "uploads.tsv", "a") as fh:
        fh.write("%s\t%s\t%s\t%s\tsabotage=0\n" % (leg, slot, dest, key))
    with open(gha / "hot_sha256.tsv", "a") as fh:
        fh.write("%s\t%s\n" % (key, sha256(arc)))
    return f"uploaded to slot {slot}"


def cmd_group(a):
    import bincache
    repo = Path(a.repo).resolve()
    out = Path(a.out).resolve()
    cache = Path(a.cache).resolve()
    for p in (out, cache):
        p.mkdir(parents=True, exist_ok=True)
    modes = [m for m in a.modes.replace(",", " ").split() if m]
    env = build_env(cache, modes, repo)
    t0 = time.time()
    result = dict(dir=a.dir, stem=a.stem, runner_cores=cores(), started=dt.datetime.now(dt.timezone.utc).isoformat())
    key, line = explain_key(repo, env, a.dir, a.stem)
    result.update(key=key, explain=line)
    say(f"{a.dir}/{a.stem}: {line}")
    # THE R2 LOOKUP, before anything compiles: a verified archive of this key is placed by the build's own cache
    # route (cmd_build_local verifies key, fields and every file before a byte is placed).
    fetched = ""
    if key and a.map and Path(a.map).is_file():
        urls = bincache.read_map(a.map)
        if key in urls["get"]:
            tmp = cache / (".fetch-%s.tar.gz" % key)
            code = bincache.http_get(urls["get"][key], str(tmp))
            if code == 200:
                os.replace(tmp, cache / (key + ".tar.gz"))
                fetched = "r2"
            else:
                tmp.unlink(missing_ok=True)
                fetched = f"r2-get-{code}"
    result["fetched"] = fetched
    log = out / f"build-{a.dir}-{a.stem}.log"
    genv = dict(env, MOJOLEARN_RELEASE_MACOS_GROUPS=f"{a.dir}:{a.stem}")
    with open(log, "w") as fh:
        rc = subprocess.run(["sh", "packaging/macos/build_release_wheel.sh", "--build-only"], cwd=repo, env=genv,
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    result.update(rc=rc, seconds=round(time.time() - t0, 1))
    rows = [r for r in provenance_rows(cache) if r[0].endswith("merged_build.sh")]
    outcome = rows[-1][1] if rows else "none"
    built_key = rows[-1][2] if rows else ""
    result.update(outcome=outcome, built_key=built_key)
    if key and built_key and built_key != key:
        say(f"PREFETCH KEY MISMATCH {a.dir}/{a.stem}: explain {key[:16]}, build {built_key[:16]}")
        result["key_mismatch"] = True
    so = repo / output_of(a.dir, a.stem)
    if rc == 0 and a.dir != "host":
        ok, why = gpu_embedded(repo, str(so))
        result.update(gpu_embedded=ok, gpu_embedded_line=why)
        if not ok:
            # NO KERNELS, NO ARCHIVE: a hosted runner once emitted a Metal-less file that passed every other gate.
            say(f"{a.dir}/{a.stem}: NO METAL KERNELS ({why}); nothing is uploaded or packed")
            rc = result["rc"] = 3
    arc = cache / ((built_key or key) + ".tar.gz")
    if rc == 0 and arc.is_file():
        (out / "archives").mkdir(exist_ok=True)
        shutil.copy2(arc, out / "archives" / arc.name)
        shutil.copytree(cache / "provenance" / "keys", out / "GHA" / "bincache" / "keys", dirs_exist_ok=True)
        result["upload"] = (upload(a.map, str(arc), built_key, out)
                            if outcome.endswith("built-cached") and a.map and Path(a.map).is_file() else "hit, nothing uploaded")
    (out / f"result-{a.dir}-{a.stem}.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"APPLE GROUP {a.dir}/{a.stem} rc={rc} outcome={outcome} seconds={result['seconds']} "
          f"cores={result['runner_cores']} fetched={fetched or '-'} gpu_embedded={result.get('gpu_embedded', 'n/a')}")
    if rc != 0:
        subprocess.run(["tail", "-n", "40", str(log)])
    return rc


# ---------------------------------------------------------------- the wheel (on a runner)
def cmd_pack(a):
    repo = Path(a.repo).resolve()
    out = Path(a.out).resolve()
    cache = out / "macos-bincache"
    cache.mkdir(parents=True, exist_ok=True)
    for arc in sorted(Path(a.archives).rglob("*.tar.gz")):  # driver-loop: over the groups' archives
        shutil.copy2(arc, cache / arc.name)
    modes = [m for m in a.modes.replace(",", " ").split() if m]
    env = build_env(cache, modes, repo)
    # EVERY GROUP A HIT, before the build: a miss would compile here, one runner, serially.
    misses = []
    for r in plan_rows(repo, modes):
        key, line = explain_key(repo, env, r["dir"], r["stem"])
        if not line.startswith("hit"):
            misses.append(f"{r['dir']}/{r['stem']}: {line}")
    for m in misses:
        # not refused: the build below compiles it (the same bytes, only slower); said loudly, it is a key mismatch
        # between a group job and the pack (or a group job that failed), which the next round fixes
        print("APPLE PACK MISS (compiled here) " + m)
    log = out / "macos-build.log"
    with open(log, "w") as fh:
        rc = subprocess.run(["sh", "packaging/macos/build_release_wheel.sh", "--build-only"], cwd=repo, env=env,
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    compiled = [x for x in log.read_text(errors="replace").splitlines() if x.startswith("BINCACHE miss")]
    (out / "pack-compiled.txt").write_text("".join(x + "\n" for x in compiled))
    if compiled:
        print(f"APPLE PACK COMPILED {len(compiled)} group(s) itself (no group-job archive): {compiled[0]}")
    wheels = sorted((repo / "python" / "dist").glob("mojolearn-*-macosx_*.whl"))
    if rc == 0 and len(wheels) == 1:
        (out / "wheel").mkdir(exist_ok=True)
        shutil.copy2(wheels[0], out / "wheel" / wheels[0].name)
        print(f"APPLE PACK PASSED {wheels[0].name} sha256 {sha256(wheels[0])}")
        return 0
    subprocess.run(["grep", "-n", "-m", "12", "-i", "-E", "error|fail|refus", str(log)])
    print(f"APPLE PACK FAILED rc={rc} wheels={[w.name for w in wheels]}")
    return rc or 6


# ---------------------------------------------------------------- the Apple lanes (on a runner)
def metal_usable(self_test):
    """True when the runner's self-test ran its gpu side and read a state (the Metal device ran a fit)."""
    try:
        d = json.loads(Path(self_test).read_text())
    except (OSError, ValueError):
        return False
    gpu = (d.get("sides") or {}).get("gpu") or {}
    return bool(gpu.get("ran")) and (gpu.get("clean") or {}).get("state") in ("IDENTICAL", "DIVERGENT")


def cmd_lanes(a):
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lanes = [x.strip() for x in a.lanes.split(",") if x.strip()]
    usable = metal_usable(a.self_test) if a.self_test else False
    rec = dict(lanes=lanes, metal_usable=usable, wheel=Path(a.wheel).name, wheel_sha256=sha256(a.wheel))
    venv = Path(tempfile.mkdtemp(prefix="apple-lanes-")) / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    py = venv / "bin" / "python"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MOJOLEARN_", "PYTHON")) and k != "VIRTUAL_ENV"}
    env["MOJOLEARN_NUMERIC_MODE"] = "identical"
    log = out / "apple-lanes.log"
    with open(log, "w") as fh:
        rc = subprocess.run([str(py), "-m", "pip", "install", "--no-cache-dir", a.wheel + "[verify]"], env=env,
                            stdout=fh, stderr=subprocess.STDOUT).returncode
        rec["install_rc"] = rc
        if rc == 0:
            # the lanes' cells in parallel over every core (verify's own --cell-workers), one report
            rc = subprocess.run([str(py), "-m", "mojolearn", "verify", "--all", "--lanes", ",".join(lanes),
                                 "--cell-workers", str(cores()), "--json", str(out / "apple-lanes-report.json")],
                                env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=out).returncode
    rec["rc"] = rc
    rec["status"] = ("PASSED" if rc == 0 else "FAILED") if usable else ("NO-METAL" if rc != 0 else "PASSED-NO-GPU")
    (out / "apple-lanes.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(f"APPLE LANES {rec['status']} metal_usable={usable} rc={rc} lanes={','.join(lanes)}")
    if rc != 0:
        subprocess.run(["grep", "-n", "-m", "20", "-i", "-E", "DIVERGENT|crash|signal|segfault|EXC_|error|FAIL", str(log)])
    # reported, never hidden: the job fails when a usable Metal device failed a lane; with no usable Metal the
    # status says NO-METAL and the job passes (the Mac's macos-smoke decides then)
    return 1 if (usable and rc != 0) else 0


# ---------------------------------------------------------------- the Mac: dispatch, poll, collect
def tooling_ref(ref):
    """(branch, sha): the branch the workflow runs from holds this checkout's HEAD; the route files are committed."""
    git = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True, check=True).stdout.strip()
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", *ROUTE_FILES)
    if dirty:
        raise SystemExit("the Apple GitHub route's files are not committed here:\n" + dirty)
    branch = ref or git("rev-parse", "--abbrev-ref", "HEAD")
    remote = subprocess.run(["git", "-C", str(ROOT), "ls-remote", "origin", "refs/heads/" + branch],
                            capture_output=True, text=True).stdout.split()
    if not remote or remote[0] != head:
        raise SystemExit(f"origin/{branch} is {remote[0][:12] if remote else 'absent'}, this checkout is {head[:12]}:"
                         " push it (the runners use the pushed tooling)")
    return branch, head


def mint_map(creds, part, leg, jobs):
    import bincache
    lines, n_get, _ = bincache.plan_lines(creds, part, "github-" + part.split("/")[-1], leg, jobs * SLOTS_PER_JOB,
                                          expires=MAP_HOURS * 3600)
    key = f"{MAP_PREFIX}/{leg}.tsv"
    status, _ = bincache.r2_request(creds, "PUT", key, data=("\n".join(lines) + "\n").encode())
    if status != 200:
        raise SystemExit(f"could not store the URL map in R2 (HTTP {status})")
    return bincache.presign("GET", key, MAP_HOURS * 3600, creds), key, n_get


def workflow_id():
    import release_github_build as rgb
    out = rgb.gh_read("api", "--paginate", "repos/{owner}/{repo}/actions/workflows", "--jq", '.workflows[] | "\\(.id) \\(.path)"')
    for line in out.splitlines():
        wid, _, path = line.strip().partition(" ")
        if path == ".github/workflows/" + WORKFLOW and wid.isdigit():
            return wid
    raise SystemExit(f".github/workflows/{WORKFLOW} is not registered with GitHub (it must be on main once)")


def find_run(tag, branch, wid, tries=30, pause=5):
    import release_github_build as rgb
    for _ in range(tries):
        rows = json.loads(rgb.gh_read("run", "list", "--workflow", wid, "--branch", branch, "--event",
                                      "workflow_dispatch", "--limit", "30", "--json", "databaseId,displayTitle,url") or "[]")
        for r in rows:
            if tag in (r.get("displayTitle") or ""):
                return r
        time.sleep(pause)
    raise SystemExit(f"no {WORKFLOW} run titled with {tag} appeared")


def collect(run_id, out, conclusion):
    """Every artifact of the run that `run` needs, into OUT (the wheel, the pack's cache and log, the smoke, the
    group records)."""
    import release_github_build as rgb
    names = rgb.run_artifacts(run_id)
    got = []
    for name in names:
        if name == "apple-wheel" or name == "apple-smoke" or name.startswith("apple-records-"):
            # one directory per group's records: their GHA/bincache files share names
            rgb.download(run_id, name, out / "records" / name if name.startswith("apple-records-") else out / name,
                         [], conclusion)
            got.append(name)
    return got


def cmd_run(a):
    import release_github_build as rgb
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
        raise SystemExit("--commit takes the full 40-hex commit")
    part = partition(a.runner)
    branch, tooling = tooling_ref(a.ref)
    creds = rgb.read_creds()
    import merged_bindings
    njobs = sum(len(merged_bindings.plan(d, "darwin", True, "release", repo=ROOT))
                for d in merged_dirs(a.modes.replace(",", " ").split()))
    tag = f"apple-{a.commit[:12]}-{int(time.time())}"
    map_key = None
    if a.run_id:
        run = dict(databaseId=int(a.run_id))
    else:
        url, map_key, n_get = mint_map(creds, part, tag, njobs)
        say(f"map: {n_get} archives in {part}, {njobs * SLOTS_PER_JOB} upload slots; dispatching {WORKFLOW} on "
            f"{branch} ({tooling[:12]}) for {a.commit[:12]}, runner {a.runner}")
        rgb.gh("workflow", "run", WORKFLOW, "--ref", branch, "-f", f"commit={a.commit}", "-f", f"modes={a.modes}",
               "-f", f"runner={a.runner}", "-f", f"lanes={a.lanes}", "-f", f"token={tag}", "-f", f"map_url={url}")
        run = find_run(tag, branch, workflow_id())
    (out / "run.json").write_text(json.dumps(dict(run=run, commit=a.commit, tooling=tooling, branch=branch,
                                                  runner=a.runner, partition=part, tag=tag), indent=1) + "\n")
    say(f"run {run['databaseId']} {run.get('url', '')}")
    try:
        view = rgb.wait_run(run["databaseId"], a.timeout_min, 60)
        got = collect(run["databaseId"], out, view.get("conclusion"))
    finally:
        if map_key:
            import bincache
            bincache.r2_request(creds, "DELETE", map_key)
    # promote the group jobs' uploads (merged records of every group job)
    rec = out / "records"
    if rec.is_dir():
        gha = out / "GHA" / "bincache"
        (gha / "keys").mkdir(parents=True, exist_ok=True)
        for name in ("uploads.tsv", "hot_sha256.tsv"):
            with open(gha / name, "w") as fh:
                for f in sorted(rec.rglob(name)):
                    fh.write(f.read_text())
        for k in rec.rglob("keys/*.json"):
            shutil.copy2(k, gha / "keys" / k.name)
        say("promote: " + rgb.promote(out, creds))
    wheels = sorted((out / "apple-wheel" / "wheel").glob("mojolearn-*-macosx_*.whl"))
    failed = [f"{j['name']}={j.get('conclusion')}" for j in view.get("jobs") or []
              if j.get("conclusion") not in ("success", "skipped")]
    print(f"APPLE GITHUB BUILD {view.get('conclusion')} run {run['databaseId']}: artifacts {', '.join(got) or 'none'}"
          + (f"; failed jobs {'; '.join(failed)}" if failed else ""))
    if len(wheels) != 1:
        raise SystemExit(f"no macOS wheel from run {run['databaseId']} ({'; '.join(failed) or view.get('conclusion')})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--repo", default=os.getcwd(), help="the source checkout (default: the current directory)")
    p.add_argument("--modes", default="identical")
    p.add_argument("--groups", default="")
    p.add_argument("--changed-since", default="")
    p = sub.add_parser("job-map")
    p.add_argument("--event", default=os.environ.get("GITHUB_EVENT_PATH", ""))
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--dest", required=True)
    p = sub.add_parser("group")
    p.add_argument("--repo", default=os.getcwd(), help="the source checkout (default: the current directory)")
    p.add_argument("--dir", required=True)
    p.add_argument("--stem", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--map", default="")
    p.add_argument("--modes", default="identical")
    p = sub.add_parser("pack")
    p.add_argument("--repo", default=os.getcwd(), help="the source checkout (default: the current directory)")
    p.add_argument("--archives", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--modes", default="identical")
    p = sub.add_parser("lanes")
    p.add_argument("--wheel", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lanes", default=",".join(DEFAULT_LANES))
    p.add_argument("--self-test", default="")
    p = sub.add_parser("run")
    p.add_argument("--commit", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--modes", default="identical")
    p.add_argument("--ref", default="")
    p.add_argument("--run-id", default="")
    p.add_argument("--runner", default=DEFAULT_RUNNER, choices=RUNNER_LABELS)
    p.add_argument("--lanes", default=",".join(DEFAULT_LANES))
    p.add_argument("--timeout-min", type=int, default=420)
    a = ap.parse_args(argv)
    if getattr(a, "repo", None):
        # on a runner the step bodies come from the workflow's commit (.wf/) and the SOURCE is the checkout: its
        # tools/bincache.py and tools/merged_bindings.py key and plan the build
        sys.path.insert(0, str(Path(a.repo).resolve() / "tools"))
    return dict(plan=cmd_plan, group=cmd_group, pack=cmd_pack, lanes=cmd_lanes, run=cmd_run,
                **{"job-map": cmd_job_map})[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
