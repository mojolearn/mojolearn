#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""PRE-BUILD CHECKS: the minutes-long checks a measurement build runs before it dispatches its 45-minute GitHub build.

    python3 tools/prebuild_check.py --commit <sha> [--ref <pushed tooling branch>] [--only audits,host-parse,probe]
        [--probe-wheel <mojolearn_core-*.whl>] [--work DIR]

`tools/release.py candidate` runs it first and refuses to dispatch the build when it fails (`--no-prebuild-check`
skips it). Three builds were lost on 2026-11-15 to defects these catch:
  audits      (efe971044a: every shard compiled, then the pack refused mojolearn/_verify_model.py, a NumPy import not
              in NUMPY_ORACLES) the pack's source-only refusals over the commit's Python exactly as the wheel lays it
              out: the entries pack_wheel.py takes (python_package_entries, mojolearn_diagnostics.py and
              tools/verification_repository.py validate_payload, which also refuses verification inputs that are
              missing or differ from the commit), the platform-math/NumPy/dependency audit of
              packaging/portable_math/wheel.py (audit_tree python_only) and the source/API payload audit
              (tools/wheel_api_audit.py audit, `source_payload_complete`, the refusal pack_wheel raises). Every rule
              is the COMMIT's own file, loaded from an export of the commit; nothing is copied here. Seconds.
              Then the commit's tools/mojo_lastuse_audit.py and tools/identity_math_lint.py (raw stdlib math where
              IDENTICAL must call the exact seams) over every .mojo blob of the commit (a local freed at its
              last named use while an untracked address still reads it; tools/mojo_lastuse_allow.txt). Seconds.
              Then the commit's tools/python_free_dirs.sh over the commit (no Mojo kernel, RUN module or non-boundary
              binding imports Python), as the pre-push hook runs it, so a commit pushed past the hook cannot build.
              Then the commit's tools/native_caller_audit.py over the commit's blobs (a registered native export or
              dispatch name with no caller, constructed family or keep reason; fb12-dead-names). Seconds.
              Then the commit's tools/block_tables_check.py over the commit's blobs (every table that lists
              Sequential layer codes or names agrees for every code; lane layer-tables). Seconds.
  host-parse  (with device-parse since 2026-11-18: the identical device groups through the same parse with the
              NVIDIA sm_89 column's flags and no GPU; it cannot see what only optimised lowering, ptxas or a GPU finds)
              (f29f76bf57: two host units did not parse; native-checks passed 166/166 because no check imports those
              doors) .github/workflows/native-checks.yml with suite=host-parse on a free runner: every release host
              group the Linux CPU leg builds (tools/merged_bindings.py plan --dir host --layout release) through
              `mojo build --emit llvm --no-optimization` (parse, type check and elaboration; no LLVM optimisation and no machine
              code), one runner per group, all in parallel. A completed run of the same sha whose host-parse jobs all
              passed is reused; one in progress is waited for; otherwise it is dispatched on --ref and waited for.
  probe       (26709df246: the verify smoke failed on Python-only defects) only with --probe-wheel: a COPY of an
              earlier packed core wheel with every shipped Python file and verification input replaced by the
              commit's (the same entry mapping as `audits`; the wheel's generated _version.py is kept, .py files the
              commit no longer ships are dropped, RECORD rewritten), dispatched to verify-smoke.yml through
              tools/ops/verify_smoke_dispatch.py and waited for; its SUMMARY.txt lines are printed. The probe
              wheel is written under <work>/probe-wheel-NOT-MEASURED/ and is never measured or published. It is only
              meaningful while the commit changes no native code the Python calls into.

The parts run in parallel, each in its own process with its own log under the work dir (default
<candidate work dir>/prebuild when release.py runs it, else ~/mojolearn-evidence/release-candidates/prebuild/<sha12>),
and each prints one line: `PREBUILD <part> PASS|FAIL (<s>s) log <path>`. The exit status is non-zero when any part
failed. A pass is recorded in <work>/PASS.json, which a resumed candidate reuses for the same sha and parts.
"""
import argparse
import base64
import concurrent.futures
import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = Path(os.environ.get("MOJOLEARN_EVIDENCE_ROOT", os.path.expanduser("~/mojolearn-evidence")))
PARTS = ("audits", "host-parse", "probe")
#: the parts of the source tree the audits read (the whole commit is ~650 MB, mostly bench/results)
EXPORT_PATHS = ("python", "tools", "packaging", "CITATION.cff", "pyproject.toml")
NATIVE_WORKFLOW = "native-checks.yml"
SMOKE_WORKFLOW = "verify-smoke.yml"
PROBE_DIR = "probe-wheel-NOT-MEASURED"
#: wheel members a probe keeps from the packed wheel: generated at pack time, not a source file
PROBE_KEEP = ("mojolearn/_version.py",)
POLL_SECONDS = 30


def sh(*args, cwd=ROOT, check=True, **kw):
    p = subprocess.run(list(args), cwd=str(cwd), capture_output=True, text=True, **kw)
    if check and p.returncode != 0:
        raise SystemExit(f"prebuild: {' '.join(map(str, args[:4]))} failed: {(p.stderr or p.stdout).strip()[-400:]}")
    return p.stdout.strip()


# ------------------------------------------------------------------ the commit's tree and its wheel entries
def export_tree(commit, dest, repo=ROOT):
    """The paths the audits read, exported from COMMIT into DEST (git archive; no worktree, no checkout): EXPORT_PATHS
    plus the verification inputs host_surface.py names under bench/ (the training columns and the CTR models)."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    def extract(paths):
        listed = [p for p in paths if sh("git", "ls-tree", "--name-only", commit, "--", p, cwd=repo, check=False)]
        if not listed:
            return
        data = subprocess.run(["git", "archive", "--format=tar", commit, "--", *listed], cwd=str(repo),
                              capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            tar.extractall(dest, filter="data")

    extract(EXPORT_PATHS)
    manifest = runpy.run_path(str(dest / "python/mojolearn/host_surface.py"))
    extract(list(manifest.get("TRAINING_GPU_COLUMNS", ())) + [manifest.get("GBDT_CTR_MODELS_DIR", "")])
    return dest


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wheel_entries(tree, commit, git_root=ROOT):
    """{wheel member: source file} of every non-binary member the core wheel takes from the commit, by the commit's
    own packer functions: mojolearn_diagnostics.py, pack_wheel.python_package_entries (the pyproject package list)
    and verification_repository.validate_payload (verify's inputs: _verify*/_verification*/_identity* modules, the
    tools/ copies, reference cards, the reference table, the training columns, the CTR models), which refuses an
    input missing or differing from the commit. Run in a fresh process: it loads the commit's modules by name."""
    tree = Path(tree)
    pack = load_module(tree / "packaging/linux/pack_wheel.py", "prebuild_pack_wheel")   # puts tree/tools on sys.path
    entries = {"mojolearn_diagnostics.py": tree / "python" / "mojolearn_diagnostics.py"}
    entries.update(pack.python_package_entries(tree / "python"))
    sys.path.insert(0, str(tree / "tools"))
    from verification_repository import validate_payload
    entries.update(validate_payload(tree, commit, git_root=git_root))
    return entries


def stage_entries(entries, stage):
    """Lay ENTRIES out under STAGE as the wheel does (the wheel's root layout)."""
    stage = Path(stage)
    if stage.exists():
        shutil.rmtree(stage)
    for member, source in sorted(entries.items()):
        target = stage / member
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return stage


def audit_stage(tree, stage, workdir):
    """[error lines] of the commit's source-only pack audits over STAGE."""
    return math_errors(tree, stage) + payload_errors(tree, stage, workdir)


def math_errors(tree, stage):
    """[error lines] of TREE's packaging/portable_math/wheel.py audit_tree(STAGE, python_only=True): the NumPy,
    dependency and Python-math rules the pack's finalize step applies to every shipped file."""
    tree = Path(tree)
    sys.path.insert(0, str(tree / "packaging/portable_math"))
    math_audit = load_module(tree / "packaging/portable_math/wheel.py", "prebuild_portable_math_wheel")
    try:
        math_audit.audit_tree(Path(stage), python_only=True)
    except ValueError as exc:
        return [f"portable_math/wheel.py: {line}" for line in str(exc).splitlines()[1:] or [str(exc)]]
    return []


def payload_errors(tree, stage, workdir):
    """[error lines] of TREE's tools/wheel_api_audit.py audit over STAGE: the source_payload_complete refusal
    pack_wheel raises (a package module the packer's list misses, a public name absent from the payload)."""
    tree, stage = Path(tree), Path(stage)
    errors = []
    sys.path.insert(0, str(tree / "tools"))
    api = load_module(tree / "tools/wheel_api_audit.py", "prebuild_wheel_api_audit")
    # audit() reads members out of a zip; this is the staged payload zipped, not a wheel
    payload = Path(workdir) / "staged-payload.zip"
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(stage).as_posix())
    row = api.audit([payload], source_root=tree)["wheels"][0]
    if not row["source_payload_complete"]:
        for key in ("source_names_absent_from_wheel", "missing_python_payload", "changed_python_payload",
                    "missing_reference_payload", "changed_reference_payload"):
            errors += [f"wheel_api_audit.py: {key}: {item}" for item in row.get(key) or ()]
    return errors


def run_audits(tree, commit, workdir, git_root=ROOT):
    """The `audits` part (in its own process). Returns the exit code; prints AUDIT lines."""
    try:
        entries = wheel_entries(tree, commit, git_root)
    except (ValueError, SystemExit, OSError) as exc:
        print(f"AUDIT FAIL pack entries: {exc}")
        return 1
    stage = stage_entries(entries, Path(workdir) / "staged-wheel-tree")
    errors = audit_stage(tree, stage, workdir)
    py = sum(1 for m in entries if m.endswith(".py"))
    for line in errors:
        print(f"AUDIT FAIL {line}")
    print(f"AUDIT {'FAIL' if errors else 'PASS'} {len(entries)} wheel entries ({py} .py) of {commit[:12]}, "
          f"{len(errors)} refusal(s)")
    lastuse = lastuse_audit(tree, commit, git_root)
    pyfree = python_free_audit(tree, commit, git_root)
    nativecall = native_caller_audit(tree, commit, git_root)
    mathlint = math_lint_audit(tree, commit, git_root)
    tables = block_tables_audit(tree, commit, git_root)
    return 1 if errors or lastuse or pyfree or nativecall or mathlint or tables else 0


def block_tables_audit(tree, commit, git_root=ROOT):
    """The COMMIT's tools/block_tables_check.py over the commit's blobs (lane layer-tables, 2026-11-23: a literal code
    list in load_state_dict left out layer codes 42 to 46): every table that lists Sequential layer codes or names
    agrees for every code. Prints its DISAGREE lines and one BLOCKTABLES line; returns its exit code (0 when the
    commit predates the check)."""
    tool = Path(tree) / "tools/block_tables_check.py"
    if not tool.is_file():
        print(f"BLOCKTABLES SKIP {commit[:12]} has no tools/block_tables_check.py")
        return 0
    p = subprocess.run([sys.executable, str(tool), "--tree", commit, "--root", str(git_root)],
                       capture_output=True, text=True)
    lines = (p.stdout + p.stderr).splitlines()
    for line in [l for l in lines if l.startswith("DISAGREE")][:60]:
        print(f"BLOCKTABLES {line}")
    print(f"BLOCKTABLES {'PASS' if p.returncode == 0 else 'FAIL'} tools/block_tables_check.py {commit[:12]} "
          f"rc={p.returncode} {lines[-1] if lines else ''}")
    return p.returncode


def math_lint_audit(tree, commit, git_root=ROOT):
    """The COMMIT's tools/identity_math_lint.py over the commit's .mojo blobs (fb13-identity-audit, 2026-11-17): raw
    stdlib math where IDENTICAL must call the checks/numerics.mojo seams, before any compile. Prints its MATHLINT
    lines; returns its exit code (0 when the commit predates the lint)."""
    tool = Path(tree) / "tools/identity_math_lint.py"
    if not tool.is_file():
        print(f"MATHLINT SKIP {commit[:12]} has no tools/identity_math_lint.py")
        return 0
    # a child process: the lint splits its files across a process pool
    return subprocess.call([sys.executable, str(tool), "--commit", commit, "--root", str(git_root)])


def python_free_audit(tree, commit, git_root=ROOT):
    """The COMMIT's tools/python_free_dirs.sh over the commit's tree (fb4-census, 2026-11-17): the no-Python-import
    check for Mojo files the pre-push hook runs, here too, so a commit that bypassed the hook cannot build. Prints one
    PYFREE line (the refused files when it fails); returns its exit code (0 when the commit predates the script)."""
    tool = Path(tree) / "tools/python_free_dirs.sh"
    if not tool.is_file():
        print(f"PYFREE SKIP {commit[:12]} has no tools/python_free_dirs.sh")
        return 0
    p = subprocess.run(["bash", str(tool), commit], cwd=git_root, capture_output=True, text=True)
    for line in (p.stdout + p.stderr).splitlines()[:60]:
        print(f"PYFREE {line}")
    print(f"PYFREE {'PASS' if p.returncode == 0 else 'FAIL'} tools/python_free_dirs.sh {commit[:12]} rc={p.returncode}")
    return p.returncode


def native_caller_audit(tree, commit, git_root=ROOT):
    """The COMMIT's tools/native_caller_audit.py over the commit's blobs (fb12-dead-names, 2026-11-17): a registered
    native export or dispatch name that no literal caller, constructed family or KEEP reason reaches fails the build,
    so dead entries cannot regrow. Prints its NATIVECALL lines; returns its exit code (0 when the commit predates it)."""
    tool = Path(tree) / "tools/native_caller_audit.py"
    if not tool.is_file():
        print(f"NATIVECALL SKIP {commit[:12]} has no tools/native_caller_audit.py")
        return 0
    # a child process: the audit splits its files across a process pool, whose workers import it by path
    return subprocess.call([sys.executable, str(tool), "--commit", commit, "--root", str(git_root)])


def lastuse_audit(tree, commit, git_root=ROOT):
    """The COMMIT's tools/mojo_lastuse_audit.py over the commit's .mojo blobs (2026-11-16: a list freed at its last
    named use while an untracked address still read it crashed four test files and five native check suites).
    Prints its LASTUSE lines; returns its exit code (0 when the commit predates the audit)."""
    tool = Path(tree) / "tools/mojo_lastuse_audit.py"
    if not tool.is_file():
        print(f"LASTUSE SKIP {commit[:12]} has no tools/mojo_lastuse_audit.py")
        return 0
    # a child process: the audit splits its files across a process pool, whose workers import it by path
    return subprocess.call([sys.executable, str(tool), "--commit", commit, "--root", str(git_root)])


# ------------------------------------------------------------------ the probe wheel
def probe_plan(names, entries, keep=PROBE_KEEP):
    """(replace, add, drop) for a packed wheel's member NAMES against the commit's ENTRIES: members the commit
    ships are replaced or added; a Python module the wheel has and the commit does not ship (outside .dist-info and
    KEEP) is dropped; every other member (binaries, metadata, generated files) is kept as packed."""
    names = set(names)
    wanted = {m for m in entries if m not in keep}
    replace = sorted(wanted & names)
    add = sorted(wanted - names)
    drop = sorted(n for n in names - wanted
                  if n.endswith(".py") and n not in keep and ".dist-info/" not in n
                  and (n.startswith("mojolearn/") or "/" not in n))
    return replace, add, drop


def build_probe_wheel(wheel, entries, out_dir):
    """A copy of WHEEL under OUT_DIR (same file name, so pip and verify_smoke_dispatch take it) with probe_plan
    applied and RECORD rewritten. Returns (path, plan)."""
    wheel, out_dir = Path(wheel), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / wheel.name
    with zipfile.ZipFile(wheel) as src:
        infos = [i for i in src.infolist() if not i.is_dir()]
        names = [i.filename for i in infos]
        replace, add, drop = probe_plan(names, entries)
        record = [n for n in names if n.endswith(".dist-info/RECORD")]
        if len(record) != 1:
            raise SystemExit(f"prebuild: {wheel.name} has {len(record)} RECORD files")
        members = {}
        for info in infos:
            if info.filename in drop or info.filename == record[0]:
                continue
            data = Path(entries[info.filename]).read_bytes() if info.filename in replace else src.read(info)
            members[info.filename] = (info, data)
        for name in add:
            members[name] = (zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)), Path(entries[name]).read_bytes())
    rows = io.StringIO(newline="")
    writer = csv.writer(rows, lineterminator="\n")
    for name, (_, data) in members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        writer.writerow([name, "sha256=" + digest, len(data)])
    writer.writerow([record[0], "", ""])
    tmp = target.with_suffix(".whl.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
        for name, (info, data) in members.items():
            info.compress_type = zipfile.ZIP_DEFLATED
            dst.writestr(info, data)
        dst.writestr(record[0], rows.getvalue())
    tmp.replace(target)
    (out_dir / "PROBE.json").write_text(json.dumps(dict(
        note="probe wheel: never measured or published", source_wheel=str(wheel), replaced=replace, added=add,
        dropped=drop), indent=2) + "\n")
    return target, (replace, add, drop)


# ------------------------------------------------------------------ GitHub runs
def gh_json(*args):
    return json.loads(sh("gh", *args) or "null")


def runs_titled(workflow, title_prefix, limit=100):
    rows = gh_json("run", "list", "--workflow", workflow, "-L", str(limit),
                   "--json", "databaseId,displayTitle,status,conclusion,createdAt")
    return [r for r in rows or () if (r.get("displayTitle") or "").startswith(title_prefix)]


def wait_run(run_id, timeout, label):
    deadline = time.monotonic() + timeout
    while True:
        view = gh_json("run", "view", str(run_id), "--json", "status,conclusion,jobs,url")
        if view["status"] == "completed":
            return view
        if time.monotonic() > deadline:
            raise SystemExit(f"prebuild: {label} run {run_id} not completed after {timeout // 60} min ({view['url']})")
        time.sleep(POLL_SECONDS)


def find_new_run(workflow, title, since, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for r in runs_titled(workflow, title, limit=20):
            if r["createdAt"] >= since:
                return r["databaseId"]
        time.sleep(10)
    raise SystemExit(f"prebuild: no {workflow} run titled '{title}' appeared after the dispatch")


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 60))


def host_parse_jobs(view):
    """The parse jobs of a native-checks run: host-parse (the host groups) and device-parse (the identical device
    groups with the sm_89 column's flags; lane fx-build-tools, 2026-11-18)."""
    return [j for j in view.get("jobs") or () if j["name"].startswith(("host-parse", "device-parse"))]


def parse_group_jobs(jobs):
    """The last attempt of each parse group (lane fx-trees-device-parse, 2026-11-19): native-checks retries a device
    group once, as `device-parse retry <key> sm_89`, when its first runner was lost or its parse was killed at the
    limit, so a group passes when its retry passed. A retry that has not finished yet is its last attempt."""
    last = {}
    for j in jobs:
        retry = j["name"].startswith("device-parse retry ")
        # no group needed a retry: GitHub shows the unexpanded matrix job (`device-parse retry ${{ matrix.key }} sm_89`)
        # as skipped; a skipped retry is not an attempt (2026-11-19: it failed the pre-build check of 5b58d98aed)
        if retry and j.get("conclusion") == "skipped":
            continue
        key = j["name"].replace("device-parse retry ", "device-parse ", 1)
        if retry or key not in last:
            last[key] = j
    return list(last.values())


def run_host_parse(commit, ref, workdir, timeout):
    """The `host-parse` part. Reuse a passed run of COMMIT, wait for one in progress, or dispatch one on REF."""
    title = f"native-checks {commit} "
    chosen = None
    for r in runs_titled(NATIVE_WORKFLOW, title):
        if (r["displayTitle"].split() + ["", "", ""])[2] not in ("host-parse", "all"):
            continue
        if r["status"] != "completed":
            chosen = chosen or r["databaseId"]
            continue
        jobs = parse_group_jobs(host_parse_jobs(gh_json("run", "view", str(r["databaseId"]), "--json", "jobs")))
        # a run from before device-parse parsed no GPU unit: it is not reused
        if (jobs and all(j["conclusion"] == "success" for j in jobs)
                and any(j["name"].startswith("device-parse") for j in jobs)):
            print(f"HOST-PARSE PASS reused run {r['databaseId']}: {len(jobs)} host and device groups parsed at "
                  f"{commit[:12]}")
            return 0
    if chosen is None:
        if not ref:
            print("HOST-PARSE FAIL no --ref: the pushed tooling branch whose native-checks.yml has suite=host-parse")
            return 1
        since = utc_now()
        sh("gh", "workflow", "run", NATIVE_WORKFLOW, "--ref", ref, "-f", f"ref={commit}", "-f", "suite=host-parse")
        chosen = find_new_run(NATIVE_WORKFLOW, title + "host-parse", since)
        print(f"HOST-PARSE dispatched run {chosen} on --ref {ref}")
    else:
        print(f"HOST-PARSE waiting for run {chosen} already in progress for {commit[:12]}")
    view = wait_run(chosen, timeout, "host-parse")
    jobs = parse_group_jobs(host_parse_jobs(view))
    bad = [j for j in jobs if j["conclusion"] != "success"]
    for j in jobs:
        print(f"HOST-PARSE {j['conclusion'] or j['status']} {j['name']}")
    if not jobs or bad:
        log = Path(workdir) / "host-parse-failed.log"
        log.write_text(sh("gh", "run", "view", str(chosen), "--log-failed", check=False))
        errors = [ln for ln in log.read_text(errors="replace").splitlines() if re.search(r"\berror:", ln)][:20]
        for ln in errors:
            print(f"HOST-PARSE error {ln[-300:]}")
        print(f"HOST-PARSE FAIL {len(bad)} of {len(jobs)} host groups ({view['url']}); failed log {log}")
        return 1
    print(f"HOST-PARSE PASS {len(jobs)} host groups parsed at {commit[:12]} ({view['url']})")
    return 0


def run_probe(tree, commit, wheel, ref, workdir, timeout, git_root=ROOT):
    """The `probe` part: build the probe wheel, smoke it on verify-smoke.yml, print its SUMMARY.txt lines."""
    entries = wheel_entries(tree, commit, git_root)
    out = Path(workdir) / PROBE_DIR
    if out.exists():
        shutil.rmtree(out)
    probe, (replace, add, drop) = build_probe_wheel(wheel, entries, out)
    print(f"PROBE wheel {probe} (never measured or published): {len(replace)} replaced, {len(add)} added, "
          f"{len(drop)} dropped from {Path(wheel).name}")
    sys.path.insert(0, str(ROOT / "tools" / "ops"))
    import verify_smoke_dispatch
    since = utc_now()
    token = verify_smoke_dispatch.dispatch(out, ref, commit, census=True, record=False)
    run_id = find_new_run(SMOKE_WORKFLOW, f"verify-smoke {commit} {token}", since)
    view = wait_run(run_id, timeout, "verify smoke")
    smoke = Path(workdir) / "probe-smoke"
    shutil.rmtree(smoke, ignore_errors=True)
    sh("gh", "run", "download", str(run_id), "-n", f"verify-smoke-{commit}", "-D", str(smoke), check=False)
    summary = next(iter(sorted(smoke.rglob("SUMMARY.txt"))), None)
    lines = summary.read_text(errors="replace").splitlines() if summary else []
    for ln in lines:
        print(f"PROBE SUMMARY {ln}")
    ok = view["conclusion"] == "success" and any(ln.startswith("overall rc=0") for ln in lines)
    print(f"PROBE {'PASS' if ok else 'FAIL'} verify smoke run {run_id} {view['conclusion']} ({view.get('url', '')})"
          + ("" if summary else "; no SUMMARY.txt downloaded"))
    return 0 if ok else 1


# ------------------------------------------------------------------ driver
def resolve_commit(commit):
    full = sh("git", "rev-parse", "--verify", f"{commit}^{{commit}}", check=False)
    if not re.fullmatch(r"[0-9a-f]{40}", full):
        raise SystemExit(f"prebuild: {commit} is not a commit here; git fetch origin first")
    return full


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", required=True, help="the commit to check (any spelling git resolves)")
    ap.add_argument("--ref", default="", help="pushed tooling branch the GitHub workflows are dispatched from "
                    "(default: this checkout's branch)")
    ap.add_argument("--only", default="", help=f"comma list of parts ({', '.join(PARTS)}); default audits,host-parse "
                    "and probe when --probe-wheel is given")
    ap.add_argument("--probe-wheel", default="", help="a packed mojolearn_core-*.whl of an earlier commit to probe")
    ap.add_argument("--work", default="", help="logs and staging (default "
                    "~/mojolearn-evidence/release-candidates/prebuild/<sha12>)")
    ap.add_argument("--host-parse-timeout", type=int, default=120, help="minutes (default %(default)s)")
    ap.add_argument("--probe-timeout", type=int, default=360, help="minutes (default %(default)s)")
    ap.add_argument("--reuse-pass", action="store_true", help="skip when <work>/PASS.json records these parts for "
                    "this sha (a resumed candidate)")
    # the child processes, one per part
    ap.add_argument("--part", default="", help=argparse.SUPPRESS)
    ap.add_argument("--tree", default="", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def child(a):
    work = Path(a.work)
    if a.part == "audits":
        return run_audits(a.tree, a.commit, work)
    if a.part == "host-parse":
        return run_host_parse(a.commit, a.ref, work, a.host_parse_timeout * 60)
    if a.part == "probe":
        return run_probe(a.tree, a.commit, a.probe_wheel, a.ref, work, a.probe_timeout * 60)
    raise SystemExit(f"prebuild: unknown part {a.part}")


def main(argv=None):
    a = parse(sys.argv[1:] if argv is None else list(argv))
    if a.part:
        return child(a)
    commit = resolve_commit(a.commit)
    parts = [p for p in (a.only or "audits,host-parse" + (",probe" if a.probe_wheel else "")).split(",") if p]
    unknown = sorted(set(parts) - set(PARTS))
    if unknown:
        raise SystemExit(f"prebuild: unknown part(s) {unknown}; the parts are {', '.join(PARTS)}")
    if "probe" in parts and not a.probe_wheel:
        raise SystemExit("prebuild: the probe part needs --probe-wheel <mojolearn_core-*.whl>")
    ref = a.ref or sh("git", "rev-parse", "--abbrev-ref", "HEAD", check=False)
    ref = "" if ref == "HEAD" else ref
    work = Path(a.work).expanduser() if a.work else EVIDENCE / "release-candidates" / "prebuild" / commit[:12]
    work.mkdir(parents=True, exist_ok=True)
    passed = work / "PASS.json"
    if a.reuse_pass and passed.is_file():
        rec = json.loads(passed.read_text())
        if rec.get("commit") == commit and set(parts) <= set(rec.get("parts") or ()):
            print(f"PREBUILD reused {passed} ({', '.join(rec['parts'])} passed at {commit[:12]})")
            return 0
    passed.unlink(missing_ok=True)
    tree = work / "tree"
    if {"audits", "probe"} & set(parts):
        export_tree(commit, tree)
    print(f"# prebuild checks of {commit[:12]}: {', '.join(parts)}; logs in {work}", flush=True)

    def run(part):
        argv = [sys.executable, str(Path(__file__).resolve()), "--part", part, "--commit", commit, "--ref", ref,
                "--work", str(work), "--tree", str(tree), "--probe-wheel", a.probe_wheel,
                "--host-parse-timeout", str(a.host_parse_timeout), "--probe-timeout", str(a.probe_timeout)]
        log = work / f"{part}.log"
        t0 = time.monotonic()
        with open(log, "w") as fh:
            rc = subprocess.call(argv, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
        return part, rc, time.monotonic() - t0, log

    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(parts)) as pool:
        for done in concurrent.futures.as_completed([pool.submit(run, p) for p in parts]):
            part, rc, seconds, log = done.result()
            if rc == 0:
                print(f"PREBUILD {part} PASS ({seconds:.0f}s) log {log}", flush=True)
                continue
            failed.append(part)
            print(f"PREBUILD {part} FAIL ({seconds:.0f}s, exit {rc}) log {log}", flush=True)
            for line in log.read_text(errors="replace").splitlines()[-12:]:
                print(f"    {line[:300]}", flush=True)
    if not failed:
        # the export and the stage are kept for a failure's diagnosis only
        for scratch in (tree, work / "staged-wheel-tree", work / "staged-payload.zip"):
            shutil.rmtree(scratch, ignore_errors=True) if scratch.is_dir() else scratch.unlink(missing_ok=True)
        passed.write_text(json.dumps(dict(commit=commit, parts=parts, made=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                          time.gmtime())), indent=2))
    print(f"# PREBUILD {'PASSED' if not failed else 'FAILED: ' + ', '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
