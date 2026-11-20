#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""A MEASUREMENT BUILD'S PACK ON GITHUB (lane fx-pack-offload, 2026-11-19; Andrew: offload the laptop, 16 GB shared
with other projects). The end of `tools/release.py candidate` (linux-wait's tree checks, linux-assemble and linux-pack:
packaging/linux/pack_wheel.py --local-version, audit.sh, the directory-entry strip, split_audit) ran on the Mac for the
base build and every B arm: each set pulled 3.9 GB of leg trees to the laptop, and on 2026-11-19 Docker Desktop died
under audit.sh ("audit.sh mojolearn-core exited 125", a 502 from the Docker socket). It now runs on a free
ubuntu-24.04 runner (.github/workflows/candidate-pack.yml), and the Mac downloads only linux/final (130-170 MB a set).

NO SECOND PACKING PATH. The runner runs the same command, `tools/release.py candidate`, with `--pack-here`: the same
Release, the same linux-wait (proof, overlay, host-byte comparison, host composition, leg records), linux-assemble
(release_reuse.assemble_linux) and linux-pack code, the same pack_wheel.py, audit.sh and split_audit. What differs is
only where the inputs come from:

  the build's records   the Mac uploads its work dir's records (state.json with the three tree steps unmarked, the
                        local ledger, reuse/plan.json, the legs' collector records) to R2 and passes a presigned GET
                        (masked, never printed; the release-linux-build.yml map_url pattern: no repository secret)
  the legs' trees       downloaded on the runner from each leg's own release-linux-build.yml run (artifact
                        release-<vendor>-<arch>), every leg at once, and verified (release_github_build.verify_tree);
                        the Mac's collectors took only release-<vendor>-<arch>-gha (`run --records-only`)
  a B arm's base        the base build's records ride in the bundle; its wheels come from the base's own pack run
                        (artifact candidate-final, recorded in its candidate.json `pack`), or ride in the bundle when
                        the base was packed on the Mac or its artifact may have expired (BASE_ARTIFACT_DAYS)
  absolute paths        the runner recreates the Mac's paths (the work root, the evidence root, and the tooling
                        checkout as a symlink to the runner's), so every path state.json, plan.json and the leg
                        records name resolves unchanged; nothing is rewritten

audit.sh needs no change and no Docker-in-Docker: its containers are the pinned manylinux_2_28 image (the auditwheel
that measures and repairs the tag) and python:3.12-slim (twine), and Docker is native on the runner. The plugin and meta
audits run at once after the core's there (tools/release.py step_linux_pack); legs download at once.

THE RELEASE IS UNTOUCHED (`pixi run release` packs on the Mac as before): a release's state dir has no --state-dir, so
its Release opens the shared R2 ledger (a runner has no credentials for it), and its linux-pack feeds the gate, column
and publish steps of the same frozen release dir on the Mac. `release.py candidate --pack local` keeps the Mac pack.

The artifact the Mac downloads, candidate-final: final/ (linux/final: the wheels, the strip receipts, split-audit.json,
candidate.json of the runner's run), records.tar.gz (every record of the runner's work dir, KEEP_SUFFIXES: the legs'
proofs and logs, the assembly, the pack and audit logs) and runner.log. The Mac installs final/ as its linux/final,
the records under its work dir (the runner's state.json as remote-pack/<tag>/runner-state.json), and then writes its
own candidate.json and prints the STAGE / B-ARM / VERIFY-SMOKE lines exactly as before.
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

SPEC_SCHEMA = "mojolearn.candidate-pack.v1"
WORKFLOW = "candidate-pack.yml"
#: the pack job's artifact: final/, records.tar.gz, runner.log
ARTIFACT = "candidate-final"
#: where the Mac stages a build's records for its pack job (deleted after the download)
BUNDLE_PREFIX = "bincache/candidate-pack"
BUNDLE_HOURS = 8
#: the tree steps the runner runs again on the downloaded trees (unmarked in the bundled state.json)
RUNNER_STEPS = ("linux-wait", "linux-assemble", "linux-pack")
#: a base's candidate-final artifact is taken from its pack run while younger than this (retention 14 days)
BASE_ARTIFACT_DAYS = 12
#: the pack job's bound (the job itself has timeout-minutes 150)
WAIT_MINUTES = 170
#: the records a work dir keeps (release_candidate.KEEP_SUFFIXES; a test holds them equal)
KEEP_SUFFIXES = (".json", ".log", ".txt", ".md", ".exit", ".pid", ".sha256", ".tsv", ".csv")
SPEC_ARC = "spec/remote-pack.json"
ABS = "abs/"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def say(msg):
    print(msg, flush=True)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ the bundle (the Mac)
def records(top, suffixes=KEEP_SUFFIXES, skip=()):
    """Every file under TOP whose name ends with SUFFIXES, sorted, not under a moved-aside attempt (`*.failed-*`),
    the source worktree, or a path of SKIP. Symlinks are not followed."""
    top = Path(top)
    if not top.is_dir():
        return []
    skip = [Path(s) for s in skip]
    out = []
    for root, dirs, files in os.walk(top):
        r = Path(root)
        dirs[:] = sorted(d for d in dirs if ".failed-" not in d and not (r == top and d == "source")
                         and not any((r / d) == s or s in (r / d).parents for s in skip))
        for name in sorted(files):
            p = r / name
            if name.endswith(suffixes) and ".failed-" not in name and not p.is_symlink():
                out.append(p)
    return out


def runner_state(state):
    """The state.json the runner starts from: the Mac's, with the tree steps (RUNNER_STEPS) and their failures
    unmarked (so the runner's linux-wait runs its tree checks instead of taking the Mac's record) and no
    remote_pack record."""
    doc = json.loads(json.dumps(state))
    for step in RUNNER_STEPS:
        (doc.get("steps") or {}).pop(step, None)
        (doc.get("failures") or {}).pop(step, None)
    doc.pop("remote_pack", None)
    return doc


def base_wheel_source(base, today=None):
    """(run id or None, why): where the runner takes a B arm's base wheels. The base's own pack run (its
    candidate.json `pack.run_id`) while its artifact is younger than BASE_ARTIFACT_DAYS, else None: the wheels
    ride in the bundle."""
    if not base:
        return None, "no base"
    doc = read_json(Path(base["build_dir"]) / "linux" / "final" / "candidate.json") or {}
    pack = doc.get("pack") or {}
    run = str(pack.get("run_id") or "")
    if not run.isdigit():
        return None, "the base was packed on the Mac"
    try:
        made = dt.datetime.fromisoformat(str(pack.get("packed_at") or doc.get("made") or "").replace("Z", "+00:00"))
    except ValueError:
        return None, "the base's pack time is unreadable"
    today = today or dt.datetime.now(dt.timezone.utc)
    if made.tzinfo is None:
        made = made.replace(tzinfo=dt.timezone.utc)
    if (today - made).days >= BASE_ARTIFACT_DAYS:
        return None, f"the base's pack run {run} is {(today - made).days} days old (artifacts expire after 14)"
    return run, f"the base's pack run {run}"


def release_module(r):
    """The module of the running Release R (tools/release.py runs as __main__: importing `release` again would load
    a second copy whose StepFailed the scheduler does not catch)."""
    return sys.modules[type(r).__module__]


def make_spec(r, legs, tag, base_run):
    """The runner's instructions: where every path lives, the candidate's arguments, the legs' runs."""
    release = release_module(r)
    cand = r.state.get("candidate") or {}
    base = r.candidate_base()
    prev = (r.plan() or {}).get("previous") or {}
    work = Path(r.base)
    mirror = [str(work.parent), str(r.evidence)]
    if base:
        mirror.append(str(Path(base["build_dir"]).parent.parent))
    extra = [d for d in (prev.get("record_dir"), None if prev.get("base") else prev.get("build_dir")) if d]
    for d in extra:
        if not inside(d, ROOT):
            mirror.append(str(Path(d)))
    return dict(schema=SPEC_SCHEMA, tag=tag, version=r.version, commit=r.commit, work=str(work),
                root=str(ROOT), evidence=str(r.evidence), mirror=sorted(set(mirror)),
                candidate=dict(defines=cand.get("defines") or "", label=cand.get("label") or "",
                               targets=cand.get("targets") or [], timed_on=cand.get("timed_on") or [],
                               base_version=cand.get("base_version") or "", owed_arm=cand.get("owed_arm")),
                all_targets=[a for _, a in release.release_reuse.ALL_LINUX_SETS],
                legs=legs, base_run=base_run,
                base=dict(build_dir=base["build_dir"], version=base.get("version"),
                          wheels={k: base[k] for k in ("linux", "nvidia", "amd") if base.get(k)}) if base else None,
                record_dirs=[d for d in extra if not inside(d, ROOT)], made=now())


def inside(path, root):
    path, root = Path(path).absolute(), Path(root).absolute()
    return path == root or root in path.parents


def bundle(r, spec, dest):
    """Write the bundle tar.gz at DEST: the spec, the work dir's records (state.json for the runner), the base's
    records (its wheels when spec has no base_run), the previous release's record dirs outside the tooling
    checkout (.json). Returns DEST."""
    work = Path(spec["work"])
    rel = Path(r.rel)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    members = []
    members += records(work / "ledger")
    members += records(rel / "reuse")
    members += records(rel / "legs")
    base = spec.get("base")
    if base:
        bd = Path(base["build_dir"])
        members += [p for p in (bd.parent / "state.json", bd / "reuse" / "plan.json") if p.is_file()]
        members += records(bd / "linux" / "final", (".json",))
        members += records(bd / "legs", (".json",))
        if not spec.get("base_run"):
            members += [bd / "linux" / "final" / w["wheel"] for w in base["wheels"].values()]
    for d in spec.get("record_dirs") or []:
        members += records(d, (".json",))
    with tarfile.open(dest, "w:gz", compresslevel=6) as tf:
        def add_bytes(name, data):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), int(time.time()), 0o644
            tf.addfile(info, io.BytesIO(data))
        add_bytes(SPEC_ARC, (json.dumps(spec, indent=1, sort_keys=True) + "\n").encode())
        add_bytes(ABS + str(work / "state.json").lstrip("/"),
                  (json.dumps(runner_state(r.state), indent=2, sort_keys=True) + "\n").encode())
        for p in dict.fromkeys(members):
            if p.resolve() == (work / "state.json").resolve() or not p.is_file():
                continue
            tf.add(str(p), arcname=ABS + str(p.absolute()).lstrip("/"), recursive=False)
    return dest


def safe_members(tf, allowed):
    """(member, target path) of every ABS member of TF: a regular file whose path stays under one of ALLOWED."""
    out = []
    for m in tf.getmembers():
        if m.name == SPEC_ARC:
            continue
        if not m.name.startswith(ABS) or not m.isfile() or ".." in Path(m.name).parts:
            raise SystemExit(f"remote pack: the bundle holds {m.name!r}, not a regular file under abs/")
        target = Path("/" + m.name[len(ABS):])
        if not any(inside(target, a) for a in allowed):
            raise SystemExit(f"remote pack: the bundle's {target} is outside the mirrored roots {allowed}")
        out.append((m, target))
    return out


# ------------------------------------------------------------------ the Mac: dispatch, wait, install
def workflow_id(name=WORKFLOW):
    """The numeric id of .github/workflows/NAME (a workflow registered on the default branch)."""
    import release_github_build as G
    out = G.gh_read("api", "--paginate", "repos/{owner}/{repo}/actions/workflows",
                    "--jq", '.workflows[] | "\\(.id) \\(.path)"')
    for line in out.splitlines():
        wid, _, path = line.strip().partition(" ")
        if path == ".github/workflows/" + name and wid.isdigit():
            return wid
    raise SystemExit(f".github/workflows/{name} is not registered with GitHub (it must be on main to be dispatched)")


def remote_legs(r):
    """{leg name: run id} of the legs the pack job downloads; refuses a leg whose run did not succeed and a leg
    taken from an earlier freeze (its tree lives on this Mac: pack it with --pack local)."""
    release = release_module(r)
    import release_github_build as G
    admitted = r.admitted_legs()
    taken = sorted(n for n, d in admitted.items() if d.get("kind") != "published-ptx")
    if taken:
        raise release.StepFailed("leg(s) " + ", ".join(taken) + " are taken from an earlier freeze whose trees are on "
                                 "this Mac; the GitHub pack cannot read them: rerun with --pack local")
    legs = {}
    for leg in release.linux_legs(r):
        if leg.name in admitted:
            continue   # the published PTX set taken whole: the runner admits it again from the index
        ok, why = G.records_only_ok(leg.out_dir, r.commit, release.candidate_defines(r))
        if not ok:
            raise release.StepFailed(f"{leg.name}: {why}")
        legs[leg.name] = str((read_json(leg.out_dir / "GHA" / "records-only.json") or {})["run"])
    return legs


def install(r, got, tag):
    """Put the downloaded artifact GOT in place: final/ as <rel>/linux/final (an earlier linux/ moved aside, never
    deleted), records.tar.gz under the work dir (state.json as remote-pack/<tag>/runner-state.json)."""
    rel, work = Path(r.rel), Path(r.base)
    final = Path(got) / "final"
    if not final.is_dir():
        raise SystemExit(f"the pack job's artifact has no final/ ({got})")
    dist = rel / "linux"
    if dist.exists():
        dist.rename(dist.with_name("linux.failed-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")))
    dist.mkdir(parents=True)
    shutil.copytree(final, dist / "final")
    arc = Path(got) / "records.tar.gz"
    if arc.is_file():
        with tarfile.open(arc) as tf:
            for m in tf.getmembers():
                parts = Path(m.name).parts
                if not m.isfile() or ".." in parts or Path(m.name).is_absolute():
                    continue
                if m.name == "state.json":
                    target = rel / "remote-pack" / tag / "runner-state.json"
                elif parts[:3] == (r.commit[:12], "linux", "final"):
                    continue   # final/ above is the shipped set
                else:
                    target = work / m.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as src, open(target, "wb") as fh:
                    shutil.copyfileobj(src, fh)


def failure_lines(got, limit=12):
    """The first failure lines of the runner's log in GOT (the job's output when the pack failed)."""
    log = Path(got) / "runner.log"
    if not log.is_file():
        return ""
    keys = ("STOPPED at", "CRASHED at", "FAILED", "REFUSED", "error", "Error", "exited")
    lines = [x.strip() for x in log.read_text(errors="replace").splitlines() if any(k in x for k in keys)]
    return "; ".join(lines[:limit])[:1200]


def pack_on_github(r):
    """tools/release.py linux-pack of a measurement build packed on GitHub (candidate_pack "github"): upload the
    records, dispatch candidate-pack.yml (or attach to the run a resume recorded), wait, download candidate-final and
    install it. Returns the step's result line; raises StepFailed when the job fails."""
    release = release_module(r)
    import release_github_build as G
    if r.dry:
        return (f"would upload the build's records to R2, dispatch .github/workflows/{WORKFLOW} (linux-wait's tree "
                "checks, linux-assemble and linux-pack on a free runner beside the trees) and download linux/final")
    legs = remote_legs(r)
    rec = dict(r.state.get("remote_pack") or {})
    run_id = None
    if rec.get("run_id") and rec.get("commit") == r.commit and rec.get("legs") == legs:
        view = json.loads(G.gh_read("run", "view", str(rec["run_id"]), "--json", "status,conclusion,url"))
        if view.get("status") != "completed" or view.get("conclusion") == "success":
            run_id = rec["run_id"]
            r.say(f"  attached to the pack run {run_id} ({view.get('url')}; {view.get('status')})")
    creds = None
    if run_id is None:
        creds = G.read_creds()
        tag = f"cpk-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.urandom(3).hex()}"
        base_run, why = base_wheel_source(r.candidate_base())
        if r.candidate_base():
            r.say(f"  the base's wheels: {'artifact of run ' + base_run if base_run else 'in the bundle'} ({why})")
        spec = make_spec(r, legs, tag, base_run)
        arc = bundle(r, spec, Path(r.rel) / "remote-pack" / tag / "bundle.tar.gz")
        digest = sha256(arc)
        key = f"{BUNDLE_PREFIX}/{tag}.tar.gz"
        import bincache
        code = bincache.http_put(bincache.presign("PUT", key, 3600, creds), str(arc))
        if code != 200:
            raise release.StepFailed(f"could not store the pack bundle in R2 (HTTP {code})")
        url = bincache.presign("GET", key, BUNDLE_HOURS * 3600, creds)
        cand = r.state.get("candidate") or {}
        branch, head = G.tooling_ref(getattr(r.args, "candidate_ref", "") or cand.get("tooling_ref") or "",
                                     cand.get("tooling_sha") or "")
        wid = workflow_id()
        G.gh("workflow", "run", wid, "--ref", branch, "-f", f"commit={r.commit}", "-f", f"tooling_sha={head}",
             "-f", f"tag={tag}", "-f", f"bundle_url={url}", "-f", f"bundle_sha256={digest}",
             "-f", f"base_run={base_run or ''}", "-f", f"version={r.version}")
        run = G.find_run(tag, branch, wid)
        run_id = run["databaseId"]
        rec = dict(tag=tag, run_id=run_id, url=run.get("url"), commit=r.commit, legs=legs, tooling_sha=head,
                   branch=branch, bundle_sha256=digest, bundle_bytes=arc.stat().st_size, r2_key=key,
                   base_run=base_run, dispatched_at=now())
        with r._lock:
            r.state["remote_pack"] = rec
            r.save()
        r.say(f"  pack job dispatched: {run.get('url')} (bundle {arc.stat().st_size / 1e6:.1f} MB, legs "
              + ", ".join(f"{n}={i}" for n, i in sorted(legs.items())) + ")")
    view = G.wait_run(run_id, WAIT_MINUTES, 60)
    tag = rec["tag"]
    got = Path(r.rel) / "remote-pack" / tag / "artifact"
    if got.exists():
        shutil.rmtree(got)
    failed = [(j.get("name", ""), j.get("conclusion")) for j in view.get("jobs") or []
              if j.get("conclusion") not in ("success", "skipped")]
    try:
        G.download(run_id, ARTIFACT, got, [(n, c, "", "", "", "") for n, c in failed], view.get("conclusion"))
    except SystemExit as exc:
        if view.get("conclusion") == "success":
            raise release.StepFailed(f"the pack run {run_id} succeeded and its {ARTIFACT} did not download: {exc}")
        raise release.StepFailed(f"the pack run {run_id} failed ({view.get('url')}) and left no {ARTIFACT}: {exc}")
    if view.get("conclusion") != "success":
        raise release.StepFailed(f"the pack run {run_id} failed ({view.get('url')}): {failure_lines(got)}; "
                                 f"its log and records are in {got}")
    install(r, got, tag)
    with r._lock:
        r.state["remote_pack"] = dict(rec, packed_at=now(), conclusion="success")
        r.save()
    if creds is None:
        try:
            creds = G.read_creds()
        except (OSError, SystemExit):
            creds = None
    G.drop_r2(creds, [rec.get("r2_key")] if rec.get("r2_key") else [])
    finals = r.split_finals()
    if not all(finals.values()) or release.wheel_commit(finals["linux"]) != r.commit:
        raise release.StepFailed(f"the pack run {run_id} left no complete wheel set of {r.commit[:12]} in "
                                 f"{Path(r.rel) / 'linux' / 'final'}")
    audit = read_json(Path(r.rel) / "linux" / "final" / "split-audit.json") or {}
    if audit.get("problems") or "problems" not in audit:
        raise release.StepFailed("the final split wheels fail their audit: "
                                 + "; ".join(audit.get("problems") or ["no split-audit.json"]))
    r.state["linux_layout"] = "split"
    return (f"packed on GitHub ({rec.get('url')}): "
            + ", ".join(f"{w.name} sha256 {release.sha256(w)}" for w in finals.values()))


# ------------------------------------------------------------------ the runner
def sudo(*args):
    subprocess.run(["sudo", *args], check=True)


def mirror(spec):
    """Recreate the Mac's directories on the runner (owned by the runner user) and the tooling checkout's path as a
    symlink to this checkout."""
    uid = f"{os.getuid()}:{os.getgid()}"
    for d in spec["mirror"]:
        if not Path(d).is_dir():
            sudo("mkdir", "-p", d)
        sudo("chown", "-R", uid, d)
    root = Path(spec["root"])
    if not root.exists() and root != ROOT:
        sudo("mkdir", "-p", str(root.parent))
        sudo("ln", "-s", str(ROOT), str(root))


def fetch_bundle(url, want, dest):
    print("::add-mask::" + url, flush=True)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as fh:
        shutil.copyfileobj(r, fh, 1 << 20)
    got = sha256(dest)
    if got != want:
        raise SystemExit(f"remote pack: the bundle is {got}, the dispatch named {want}")
    return dest


def unpack(arc):
    """(spec, number of files) after the spec's directories are mirrored and every record is in place."""
    with tarfile.open(arc) as tf:
        spec = json.loads(tf.extractfile(SPEC_ARC).read())
        if spec.get("schema") != SPEC_SCHEMA:
            raise SystemExit(f"remote pack: the bundle's spec is {spec.get('schema')}, not {SPEC_SCHEMA}")
        mirror(spec)
        members = safe_members(tf, spec["mirror"])
        for m, target in members:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as src, open(target, "wb") as fh:
                shutil.copyfileobj(src, fh)
    return spec, len(members)


def fetch_legs(spec):
    """Every leg's tree from its own run, all at once (network-bound; one thread per leg), each verified against its
    proof. Returns [(name, ok, why)]."""
    import release_github_build as G
    rel = Path(spec["work"]) / spec["commit"][:12]

    def one(item):
        name, run = item
        vendor, arch = name.split("-", 1)
        out = rel / "legs" / name
        try:
            G.download(int(run), f"release-{name}", out, [], "success")
        except SystemExit as exc:
            return name, False, f"run {run}: {exc}"
        ok, why = G.verify_tree(out / name / "release-build", vendor, arch, spec["commit"])
        return name, ok, f"run {run}: {why}"

    with cf.ThreadPoolExecutor(max_workers=max(1, len(spec["legs"]))) as ex:
        return list(ex.map(one, sorted(spec["legs"].items())))


def fetch_base(spec, tmp):
    """A B arm's base wheels from the base's pack run (spec base_run) into its build dir's linux/final, each sha256
    checked against the base record."""
    import release_github_build as G
    base, run = spec.get("base"), spec.get("base_run")
    if not base or not run:
        return []
    G.download(int(run), ARTIFACT, tmp, [], "success")
    final = Path(base["build_dir"]) / "linux" / "final"
    final.mkdir(parents=True, exist_ok=True)
    placed = []
    for platform, info in sorted(base["wheels"].items()):
        src = Path(tmp) / "final" / info["wheel"]
        if not src.is_file() or sha256(src) != info["sha256"]:
            raise SystemExit(f"remote pack: the base run {run} has no {info['wheel']} with sha256 {info['sha256'][:12]}")
        shutil.copy2(src, final / info["wheel"])
        placed.append(info["wheel"])
    return placed


def candidate_command(spec):
    """The runner's `tools/release.py candidate --pack-here` of the same build: the same version, work dir, targets,
    defines, label and timed sets the Mac's run recorded."""
    c = spec["candidate"]
    cmd = [sys.executable, str(ROOT / "tools" / "release.py"), "candidate", "--pack-here", "--commit", spec["commit"],
           "--work-root", str(Path(spec["work"]).parent), "--base-version", c["base_version"],
           "--no-prebuild-check", "--no-verify-smoke", "--keep-intermediates"]
    if c.get("targets") and sorted(c["targets"]) != sorted(spec.get("all_targets") or []):
        cmd += ["--targets", ",".join(c["targets"])]
    if c.get("defines"):
        cmd += ["--defines", c["defines"]]
    if c.get("label"):
        cmd += ["--label", c["label"]]
    if c.get("timed_on"):
        cmd += ["--timed-on", ",".join(c["timed_on"])]
    return cmd


def collect(spec, out):
    """The artifact: final/ (linux/final), records.tar.gz (every record of the work dir relative to it, not the
    source worktree, not linux/final, not the bundle)."""
    work = Path(spec["work"])
    rel = work / spec["commit"][:12]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (rel / "linux" / "final").is_dir():
        shutil.copytree(rel / "linux" / "final", out / "final", dirs_exist_ok=True)
    with tarfile.open(out / "records.tar.gz", "w:gz", compresslevel=6) as tf:
        for p in records(work, skip=(rel / "linux" / "final", rel / "remote-pack")):
            tf.add(str(p), arcname=str(p.relative_to(work)), recursive=False)


def cmd_runner(a):
    """The pack job (.github/workflows/candidate-pack.yml): bundle, mirror, legs, base, the candidate's own tree
    steps, the artifact. Exits with the candidate's exit code; the artifact is written either way."""
    inputs = (json.loads(Path(a.event).read_text()).get("inputs") or {})
    url, want = (inputs.get("bundle_url") or "").strip(), (inputs.get("bundle_sha256") or "").strip()
    if not url or len(want) != 64:
        raise SystemExit("remote pack: the dispatch carries no bundle_url / bundle_sha256")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "runner.log", "w")

    def note(msg):
        say(msg)
        log.write(msg + "\n")
        log.flush()

    rc, spec = 1, None
    try:
        spec, n = unpack(fetch_bundle(url, want, Path(a.tmp) / "bundle.tar.gz"))
        note(f"remote pack {spec['tag']}: {spec['version']} of {spec['commit'][:12]}, {n} records in place under "
             + ", ".join(spec["mirror"]))
        if inputs.get("commit") and inputs["commit"] != spec["commit"]:
            raise SystemExit(f"remote pack: the dispatch names {inputs['commit']}, the bundle {spec['commit']}")
        rows = fetch_legs(spec)
        for name, ok, why in rows:
            note(f"  leg {name}: {'TREE OK' if ok else 'TREE REFUSED'} ({why})")
        if not all(ok for _, ok, _ in rows):
            raise SystemExit("remote pack: a leg's tree did not download or verify")
        placed = fetch_base(spec, Path(a.tmp) / "base")
        if placed:
            note(f"  base wheels from run {spec['base_run']}: {', '.join(placed)}")
        cmd = candidate_command(spec)
        note("  " + " ".join(cmd))
        env = dict(os.environ, MOJOLEARN_EVIDENCE_ROOT=spec["evidence"])
        p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = p.wait()
        note(f"remote pack: release.py candidate --pack-here exited {rc}")
        collect(spec, out)
    except (Exception, SystemExit) as exc:
        note(f"remote pack FAILED: {exc!r}")
        if spec is not None:
            try:
                collect(spec, out)
            except OSError as why:
                note(f"remote pack: the records were not collected: {why}")
        rc = rc or 1
    finally:
        log.close()
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("runner", help="the pack job's body (.github/workflows/candidate-pack.yml)")
    p.add_argument("--event", required=True, help="$GITHUB_EVENT_PATH (the bundle URL is read from it, masked)")
    p.add_argument("--out", required=True, help="the artifact directory")
    p.add_argument("--tmp", default=tempfile.gettempdir())
    p.set_defaults(fn=cmd_runner)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
