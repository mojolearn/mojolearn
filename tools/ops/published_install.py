#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""tools/ops/published_install.py -- the published-smoke-nvidia / published-smoke-amd job as ONE persisted box
job, through the supported job wrappers (lane release-no-hands, 2026-11-04; RUNBOOK 0.8.65 section 18).

Since lane published-no-reverify (2026-11-04) this runs ONLY for an explicit `tools/release.py
--published-smoke <vendor> --published-smoke-reason TEXT`: published-<vendor> is a bytes check on the Mac
(tools/release_published_install.py check; Andrew 2026-11-02, nothing reinstalls the gated bytes from PyPI
to verify again), so no published row needs a box.

    python3 tools/ops/published_install.py nvidia|amd VERSION --source-commit SHA --source-tree DIR \\
        --out DIR --state FILE --candidate-wheel W [--candidate-wheel W ...] [--dry-run]

tools/release.py step_published_smoke_<vendor> calls it after published-<vendor>; --out is
<commit12>/published-smoke-<vendor>/ (never the published-<vendor> ledger evidence). Nothing here rents,
releases or extends a box (gate-boxes brought the box up). In order, as tools/ops/release_gate.py stages its
gate job (its Runner does the staging, the R2 wait and the fetch):

1. NVIDIA only: `tools/nvidia_central.sh sync release-gate <source tree>` (the lane tree at the commit).
2. The candidate wheels, the committed job script tools/ops/published_install_job.sh (verbatim) and
   published.env go to <box>/root/release-published/<version>/<vendor>/<tag>/in/, sha256 checked there.
3. The job runs persisted to R2 from the box: `nvidia_central.sh submit release-gate --cap 120 --results <res>
   <in>/published_install_job.sh` or `do_amd_steward.sh run <tag> <res> bash <in>/published_install_job.sh`.
   On the box it runs tools/release_published_install.py run (PyPI install, published-artifact hashes,
   installed smoke, installed adapters) from the source commit's tree, its report in <res>/out.
4. `tools/box_persist.py fetch --tag <tag> --extract` brings <res>/out home into --out. The persisted results
   carry no *.whl (box_persist_run excludes wheels), so each downloaded published file the manifest names is
   restored from the local candidate wheel of the same name and sha256 (the bytes the box proved equal), and
   `tools/release_published_install.py judge` re-derives the report from the home copy.

Exit: 0 when the report that came home (and its re-judgement) says passed, else the job's rc (or 1); 3 when
no final result landed within --max-wait-hours (a rerun with the same --state resumes the same job).
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for p in (str(TOOLS), str(TOOLS / "ops")):
    if p not in sys.path:
        sys.path.insert(0, p)
import release_gate as rg  # noqa: E402

JOB_SCRIPT = "tools/ops/published_install_job.sh"
LANE = rg.LANE
IN_BASE = "/root/release-published"
RESULTS_BASE = "/root/results-release-published"
#: nvidia_central.sh submit --cap: a fresh venv from PyPI, the eleven installed correctness jobs and three
#: adapter phases took ~90 s on the Mac at 0.8.64 (published-apple.log); the box adds the torch download
NV_CAP_MIN = 120
#: release_published_install.py --timeout for its installed correctness run (its default)
TIMEOUT_S = 1800
VENDORS = ("nvidia", "amd")
REPORT = "published-report.json"


def say(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')} published-install] {msg}", flush=True)


class Plan:
    """Every path, file and wrapper command of one published-<vendor> job; pure (reads no file, runs nothing).
    The interface tools/ops/release_gate.Runner stages, waits for and fetches."""

    def __init__(self, vendor, version, source_commit, source_tree, out, wheels, tag, staged=None):
        if vendor not in VENDORS:
            raise SystemExit(f"published-install: vendor must be nvidia or amd, not {vendor!r}")
        if not re.fullmatch(r"[0-9A-Za-z.+-]+", version or ""):
            raise SystemExit(f"published-install: bad version {version!r}")
        if not re.fullmatch(r"[0-9a-f]{40}", source_commit or ""):
            raise SystemExit("published-install: --source-commit must be a full sha")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
            raise SystemExit("published-install: the tag must be [A-Za-z0-9_.-]")
        self.vendor, self.version, self.commit, self.tag = vendor, version, source_commit, tag
        self.wrapper = rg.VENDORS[vendor]["wrapper"]
        self.source_tree, self.out = str(source_tree), Path(out)
        self.indir = f"{IN_BASE}/{version}/{vendor}/{tag}/in"
        self.res = f"{RESULTS_BASE}/{version}/{vendor}/{tag}/results"
        # the trees the gate job uses: NVIDIA the lane tree the sync places, AMD one fetched at the commit
        self.tree = f"/root/mojolearn-{LANE}" if vendor == "nvidia" else f"{rg.IN_BASE}/src/{source_commit[:12]}"
        self.fetch_url = "" if vendor == "nvidia" else rg.REPO_URL
        self.wheels = [Path(w) for w in wheels]
        if not self.wheels:
            raise SystemExit("published-install: at least one --candidate-wheel")
        self.staged = Path(staged) if staged else self.out.parent / f"inputs-{tag}"

    def env_text(self):
        lines = ["# tools/ops/published_install.py: one published-<vendor> job (sourced by "
                 "tools/ops/published_install_job.sh)",
                 f"PUB_VENDOR={self.vendor}", f"PUB_VERSION={self.version}", f"PUB_SOURCE_COMMIT={self.commit}",
                 f"PUB_TAG={self.tag}", f"PUB_TREE={self.tree}", f"PUB_FETCH_URL={self.fetch_url}",
                 f"PUB_TIMEOUT={TIMEOUT_S}",
                 rg.bash_array("PUB_WHEELS", [f"{self.indir}/{w.name}" for w in self.wheels])]
        return "\n".join(lines) + "\n"

    def uploads(self):
        out = [(w, f"{self.indir}/{w.name}", False) for w in self.wheels]
        out.append((ROOT / JOB_SCRIPT, f"{self.indir}/published_install_job.sh", True))
        out.append((self.staged / "published.env", f"{self.indir}/published.env", False))
        names = [r for _, r, _ in out]
        if len(set(names)) != len(names):
            raise SystemExit(f"published-install: two inputs share a box name: {names}")
        return out

    def box_sh(self, command):
        if self.vendor == "nvidia":
            return ["bash", self.wrapper, "sh", LANE, command]
        return ["bash", self.wrapper, "ssh", command]

    def sync(self):
        return ["bash", self.wrapper, "sync", LANE, self.source_tree] if self.vendor == "nvidia" else None

    def prepare(self):
        return self.box_sh("set -e; mkdir -p " + shlex.quote(self.indir) + "; echo READY")

    def upload(self, remote):
        return self.box_sh("cat > " + shlex.quote(remote))

    def verify(self, remotes, executables):
        cmd = "sha256sum " + " ".join(shlex.quote(r) for r in remotes)
        if executables:
            cmd = "chmod +x " + " ".join(shlex.quote(r) for r in executables) + " && " + cmd
        return self.box_sh(cmd)

    def launch(self):
        script = f"{self.indir}/published_install_job.sh"
        if self.vendor == "nvidia":
            return ["bash", self.wrapper, "submit", LANE, "--cap", str(NV_CAP_MIN), "--note", self.tag,
                    "--results", self.res, script]
        return ["bash", self.wrapper, "run", self.tag, self.res, "bash", script]

    def fetch(self, tag, dest, listing=False):
        return [sys.executable, "tools/box_persist.py", "fetch", "--tag", tag, "--dest", str(dest)] + (
            ["--list"] if listing else ["--extract"])

    def status(self, job):
        return ["bash", self.wrapper, "status", job]

    def describe(self):
        q = lambda argv: " ".join(shlex.quote(str(a)) for a in argv)  # noqa: E731
        lines = [q(self.sync())] if self.sync() else []
        lines.append(q(self.prepare()))
        ups = self.uploads()
        lines += [q(self.upload(remote)) + " < " + shlex.quote(str(local)) for local, remote, _ in ups]
        lines.append(q(self.verify([r for _, r, _ in ups], [r for _, r, x in ups if x])) + "  # == local sha256")
        lines.append(q(self.launch()))
        lines.append(f"  on the box, in {self.tree} at {self.commit[:12]}: python3 tools/release_published_install.py "
                     f"run --vendor {self.vendor} --version {self.version} "
                     + " ".join(f"--candidate-wheel {w.name}" for w in self.wheels) + " --out <res>/out")
        lines.append(q(self.fetch("<the R2 tag the wrapper prints>", self.staged.parent / "fetch-<tag>"))
                     + f"  # then <fetch>/final/out -> {self.out}")
        return lines


def job_script_committed():
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", JOB_SCRIPT], capture_output=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", JOB_SCRIPT], capture_output=True)
    if tracked.returncode or dirty.returncode:
        raise SystemExit(f"published-install: {JOB_SCRIPT} is not committed and clean in {ROOT}; commit it first")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def restore_wheels(out, candidates, log=say):
    """Each published file wheel-manifest.json names that the persisted results lack (box_persist_run
    excludes *.whl), copied from the candidate of the same name and sha256. Returns the names restored;
    a file with no equal candidate stays missing (the box's report already failed it)."""
    try:
        files = json.loads((Path(out) / "wheel-manifest.json").read_text()).get("files", [])
    except (OSError, ValueError):
        return []
    by_name = {Path(w).name: Path(w) for w in candidates}
    restored = []
    for row in files:
        name, digest = row.get("file"), row.get("sha256")
        target = Path(out) / "wheels" / str(name)
        if not name or "/" in name or target.is_file():
            continue
        source = by_name.get(name)
        if source is None or not source.is_file() or sha256(source) != digest:
            log(f"{name}: no local candidate with its sha256; left missing")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        restored.append(name)
    if restored:
        log(f"restored {len(restored)} published file(s) from the equal candidate wheels: " + ", ".join(restored))
    return restored


def report_home(out, vendor, version):
    """(ok, why) of the published-report.json that came home."""
    try:
        doc = json.loads((Path(out) / REPORT).read_text())
    except (OSError, ValueError):
        return False, f"no {REPORT} came home"
    if doc.get("vendor") != vendor or doc.get("version") != version:
        return False, f"{REPORT} is for {doc.get('vendor')} {doc.get('version')}"
    if doc.get("status") != "passed":
        return False, f"status {doc.get('status')}: {str(doc.get('error') or '')[:200]}"
    return True, "passed: " + " ".join("%s=%s/%s" % (r.get("id"), r.get("completed"), r.get("expected"))
                                       for r in doc.get("coverage", []))


class Runner(rg.Runner):
    """tools/ops/release_gate.Runner's staging, R2 wait and fetch, for one published-<vendor> job."""

    def make_plan(self):
        a = self.a
        tag = f"rel-{a.version}-published-{a.vendor}-" + dt.datetime.fromtimestamp(
            self.clock(), dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        state_path = Path(a.state) if a.state else None
        st = {}
        if state_path and state_path.is_file():
            try:
                st = json.loads(state_path.read_text())
            except ValueError:
                st = {}
        key = hashlib.sha256(json.dumps([a.vendor, a.version, a.source_commit,
                                         [sha256(w) if Path(w).is_file() else str(w) for w in a.candidate_wheel],
                                         sha256(ROOT / JOB_SCRIPT) if (ROOT / JOB_SCRIPT).is_file() else None]
                                        ).encode()).hexdigest()
        if st.get("key") == key and st.get("plan_tag"):
            tag = st["plan_tag"]
        else:
            st = {}
        staged = (state_path.parent if state_path else Path(a.out).parent) / f"inputs-{tag}"
        plan = Plan(a.vendor, a.version, a.source_commit, a.source_tree, a.out, a.candidate_wheel, tag, staged)
        return plan, key, state_path, st

    def dry_lines(self):
        return self.make_plan()[0].describe()

    def main(self):
        a = self.a
        plan, key, state_path, st = self.make_plan()
        if a.dry_run:
            for line in plan.describe():
                print("  would: " + line)
            return 0
        ok, why = report_home(a.out, a.vendor, a.version)
        if ok:
            say(f"published-{a.vendor} {a.version}: already home, {why}")
            return 0
        job_script_committed()
        if a.vendor == "nvidia" and not rg.on_origin_main(a.source_commit):
            raise SystemExit("published-install: the source commit is not on origin/main (the lane sync needs it)")
        rc = None
        if st and not st.get("tag"):
            st["tag"] = rg.minted_tag(st["plan_tag"])
        if st.get("tag"):
            say(f"resuming {st['tag']} (launched {st.get('launched_at')}); not launched again")
        else:
            plan.staged.mkdir(parents=True, exist_ok=True)
            (plan.staged / "published.env").write_text(plan.env_text())
            self.stage(plan)
            record = dict(key=key, plan_tag=plan.tag, vendor=a.vendor, launch=plan.launch(),
                          launched_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))

            def keep(**fields):
                record.update(fields)
                if state_path:
                    state_path.parent.mkdir(parents=True, exist_ok=True)
                    state_path.write_text(json.dumps(record, indent=2) + "\n")

            say("$ " + " ".join(shlex.quote(c) for c in plan.launch()))
            if a.vendor == "nvidia":
                got = self.call(plan.launch(), timeout=900)
                text = (got.stdout or "") + "\n" + (got.stderr or "")
                job = next((l.strip() for l in reversed((got.stdout or "").splitlines())
                            if re.fullmatch(r"nvc[0-9]-[0-9]{4,}", l.strip())), None)
                ptag = rg.persist_tag(text)
                if got.returncode or not job or not ptag:
                    raise SystemExit(f"published-install: submit refused (rc {got.returncode}): {text[-600:]}")
                keep(tag=ptag, job=job)
                say(f"queued {job}, persisted as {ptag}")
            else:
                keep()
                rc = self.stream(plan.launch(), lambda line: rg.persist_tag(line) and keep(tag=rg.persist_tag(line)),
                                 timeout=int(a.max_wait_hours * 3600))
                say(f"the AMD job exited {rc}")
                if not record.get("tag"):
                    record["tag"] = rg.minted_tag(plan.tag)
                if not record.get("tag"):
                    raise SystemExit(f"published-install: the AMD run (rc {rc}) named no R2 tag; nothing to fetch")
                say(f"persisted as {record['tag']}")
            st = record
        final = self.wait_final(plan, st["tag"], st.get("job"))
        if final is None:
            say(f"no final result for {st['tag']} within {a.max_wait_hours} h; a rerun with this --state resumes it")
            return 3
        if final[0] == "lost":
            if state_path and state_path.is_file():
                state_path.unlink()
            return 1
        try:
            job_rc = int(final[1])
        except (TypeError, ValueError):
            job_rc = rc if rc is not None else 1
        home = self.bring_home(plan, st["tag"])
        if state_path and state_path.is_file():
            state_path.unlink()
        if not home:
            return job_rc or 1
        restore_wheels(a.out, a.candidate_wheel)
        judged = self.call([sys.executable, "tools/release_published_install.py", "judge", "--out", str(a.out)],
                           timeout=1800)
        ok, why = report_home(a.out, a.vendor, a.version)
        say(f"published-{a.vendor} {a.version}: {why}; re-judged at home: exit {judged.returncode}")
        return 0 if ok and judged.returncode == 0 else (job_rc or 1)


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("vendor", choices=VENDORS)
    ap.add_argument("version")
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--source-tree", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--state", default="")
    ap.add_argument("--candidate-wheel", action="append", default=[], required=True)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--max-wait-hours", type=float, default=6.0)
    ap.add_argument("--dry-run", action="store_true", help="print the wrapper commands; run nothing")
    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.exit(Runner(parse(sys.argv[1:])).main())
