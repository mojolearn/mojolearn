#!/usr/bin/env python3
"""tools/reference_refresh.py -- THE REFERENCE REFRESH after a merge to main (2026-11-04). Entry point:
tools/reference_refresh.sh, which execs this file. Docs: docs/release/REFERENCE_REFRESH.md.

  reference_refresh.sh <sha> [columns]          record the changed identity lanes of <sha> on each column, admit
                                                every fetched record at once into table.json and commit the
                                                admission on branch ref/<sha12>, pushed for the orchestrator
  reference_refresh.sh --plan <sha> [columns]   per column, REUSED / RECORD counts; submits nothing
  reference_refresh.sh --status <sha>           the saved state of a refresh
  reference_refresh.sh --resubmit COL <sha> [columns]
                                                forget COL's job (a box that died mid-job), then continue
  reference_refresh.sh --no-wait <sha> [columns]
                                                submit what is owed and exit; rerun the same command to
                                                fetch, admit and push

  columns: comma list of nvidia, amd, cpu, apple (default all four)

Each column runs exactly what its own tool runs (the same lane selection the plan prints):
  nvidia  tools/nvidia_central.sh sync + submit tools/identity_box_job.sh (box_merged_verify.sh: merged
          bindings, `python -m mojolearn verify`, ONLY_CHANGED=table record), persisted to R2 from the pod
  amd     tools/do_amd_steward.sh update <sha> + run tools/identity_box_job.sh amd, persisted to R2
  cpu     tools/identity_box_job_cpu_ptx.sh on the NVIDIA pod, in the tree the nvidia job built (auto reuse);
          when cpu owes lanes and nvidia owes none, the nvidia job still runs: it builds that tree
  apple   tools/record_identity_column.sh apple-<chip> (auto reuse) under tools/mac_slot.py metal, in
          REFRESH_APPLE_TREE: a checkout at <sha> with its Apple bindings already built (never built here)

IDEMPOTENT. Every decision lands in <REFRESH_HOME>/<sha12>/state.json before the next step. A rerun after a
crash reads each submitted job's R2 manifest (tools/box_persist.py fetch --list): final -> fetch, running or
queued -> wait, flushed (the box died) -> stop and say so (--resubmit COL starts it again). A NOTHING-CHANGED
column submits no job. An admitted commit is pushed, never made twice.

This tool rents, extends and releases nothing: it uses the NVIDIA pod and the AMD droplet the orchestrator has
up and stops with a message when there is none. --extend-amd MIN is the one opt-in exception: while the AMD job
runs it moves the droplet's lease to now + MIN at each poll (tools/do_amd_steward.sh extend).

Environment: REFRESH_HOME (~/mojolearn-evidence/reference-refresh), REFRESH_PYTHON (the interpreter for the
selection, box_persist and admission; default this tree's pixi python, else the main checkout's, else this
one), REFRESH_SRC_TREE (~/mojolearn-wt/refresh-src: the clean detached tree at <sha> that is planned and synced),
REFRESH_ADMIT_TREE (~/mojolearn-wt/reference-admit: the admission checkout, switched to ref/<sha12> from
origin/main), REFRESH_NV_LANE (identity), REFRESH_APPLE_TREE, REFRESH_APPLE_LABEL (apple-<chip of this Mac>),
REFRESH_POLL_SEC (300), REFRESH_WAIT_MIN (600), REFRESH_SCOPE (routine).
Tests: tools/test_reference_refresh.py (no box, no network).
"""
import argparse
import datetime
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLUMNS = ("nvidia", "amd", "cpu", "apple")
#: What each column's own job runs, so the plan prints the selection the job will make.
#:   table  ONLY_CHANGED=table (box_merged_verify.sh), backend-filtered
#:   class  record_identity_column.sh's default auto reuse from the class's newest good evidence
SELECTION = dict(nvidia=("table", "cuda"), amd=("table", "hip"), cpu=("class", "cpu"), apple=("class", "metal"))
#: Paths a record run may leave untracked in a box tree; they are removed before a sync (0.8.65: a resubmit
#: refused "untracked source bench/results/identity_break/retained/...").
RECORD_OUTPUT_PATHS = ("bench/results/identity_break", "bench/results/reference_admissions")
#: Files that travel with a record into admission (tools/admit_identity_columns.sh's NEXT line names them).
RECORD_COMPANIONS = (".provenance.txt", ".source-report.json.gz", ".verifier-run.json")
NOT_A_RECORD = (".admission-preview-", ".other-commit-")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def say(msg):
    print("reference_refresh: %s" % msg, flush=True)


class Refused(Exception):
    """A stop the operator has to act on; the state is saved first."""


# ------------------------------------------------------------------ parsing (pure; unit tested)
def parse_selection(text):
    """REUSED / RECORD counts from tools/identity_columns.py changed-lanes stderr.

    --class mode prints `# REUSE-SUMMARY class=c REUSED a RECORD b KNOWN-BAD k of N (scope)`; table mode prints
    `# CHANGED-LANES n of N (scope)`. Returns dict(record, reused, known_bad, total) or None."""
    m = re.search(r"REUSE-SUMMARY class=\S+ REUSED (\d+) RECORD (\d+) KNOWN-BAD (\d+) of (\d+)", text)
    if m:
        reused, record, bad, total = map(int, m.groups())
        return dict(record=record, reused=reused, known_bad=bad, total=total)
    m = re.search(r"CHANGED-LANES (\d+) of (\d+)", text)
    if m:
        record, total = map(int, m.groups())
        return dict(record=record, reused=total - record, known_bad=0, total=total)
    return None


def parse_submit(text):
    """(tag, job id) from tools/nvidia_central.sh submit output: the box_persist line names the tag; the job id
    (<pod>-<n>) is the last line of stdout."""
    tag = None
    m = re.search(r"box_persist: job (\S+) persists", text)
    if m:
        tag = m.group(1)
    job = None
    for line in text.splitlines():
        line = line.strip()
        if re.fullmatch(r"nvc\d+-\d+", line):
            job = line
    return tag, job


def parse_persist_tag(text):
    m = re.search(r"box_persist: job (\S+) persists", text)
    return m.group(1) if m else None


def parse_manifest(text):
    """The `tools/box_persist.py fetch --list` report: dict(state, exit, landed). state is None when no manifest
    landed yet (a queued job), else running|flushed|final."""
    out = dict(state=None, exit=None, landed=[])
    m = re.search(r"^MANIFEST state=(\S+) exit=(\S*) ", text, re.M)
    if m:
        out["state"], out["exit"] = m.group(1), m.group(2)
    out["landed"] = re.findall(r"^LANDED (\S+) ", text, re.M)
    return out


def job_phase(manifest):
    """queued | running | done | failed | dead from a parsed manifest."""
    st = manifest.get("state")
    if st is None:
        # a final upload whose manifest PUT failed still counts (box_persist.py fetch does the same)
        return "done-unknown" if "final.tgz" in manifest.get("landed", []) else "queued"
    if st == "running":
        return "running"
    if st == "flushed":
        return "dead"
    if st == "final":
        return "done" if str(manifest.get("exit")) == "0" else "failed"
    return "running"


def parse_admit_next(text):
    """The paths of admit_identity_columns.sh's `NEXT ... git add -- <paths> && git commit ...` line."""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("git add -- "):
            body = s[len("git add -- "):]
            body = body.split("&&", 1)[0]
            return shlex.split(body)
    return None


def record_log_outcome(text):
    """What a record.log says: nothing (NOTHING-CHANGED), recorded (RECORDED ... admissible=yes, every one),
    inadmissible, or None when it never finished."""
    if re.search(r"^NOTHING-CHANGED ", text, re.M):
        return "nothing"
    rec = re.findall(r"^RECORDED \S+ rc=\S+ admissible=(\S+)", text, re.M)
    if rec:
        return "recorded" if all(r == "yes" for r in rec) else "inadmissible"
    return None


def record_commit(path):
    try:
        return json.loads(Path(path).read_text()).get("commit")
    except (OSError, ValueError, AttributeError):
        return None


def find_records(tree, sha=None):
    """Every *.identical.json under an extracted job dir that was recorded at `sha` (records of another commit,
    e.g. an older run's leftovers in a reused results dir, are never admitted from here), with every
    record.log outcome. Returns (records, outcomes, other_commit)."""
    tree = Path(tree)
    found = sorted(p for p in tree.rglob("*.identical.json")
                   if not any(t in p.name for t in NOT_A_RECORD) and p.is_file())
    recs, other = [], []
    for p in found:
        (recs if sha is None or record_commit(p) == sha else other).append(p)
    outcomes = []
    for log in sorted(tree.rglob("record.log")):
        o = record_log_outcome(log.read_text(errors="replace"))
        if o:
            outcomes.append((str(log), o))
    return recs, outcomes, other


def collect_records(recs, dest):
    """Copy records and their companions into one admission input dir. Returns the copied record paths."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for r in recs:
        stem = r.name[: -len(".identical.json")]
        target = dest / r.name
        if target.exists() and target.read_bytes() != r.read_bytes():
            raise Refused("two different records named %s; refusing to pick one" % r.name)
        shutil.copy2(r, target)
        out.append(target)
        for suf in RECORD_COMPANIONS:
            c = r.with_name(stem + suf)
            if c.is_file():
                shutil.copy2(c, dest / c.name)
        log = r.with_name(r.name[: -len(".json")] + ".log")
        if log.is_file():
            shutil.copy2(log, dest / log.name)
    return out


# ------------------------------------------------------------------ state
class State:
    def __init__(self, path, sha):
        self.path = Path(path)
        if self.path.exists():
            self.d = json.loads(self.path.read_text())
            if self.d.get("sha") != sha:
                raise Refused("%s belongs to %s, not %s" % (self.path, self.d.get("sha"), sha))
        else:
            self.d = dict(schema="mojolearn.reference-refresh.v1", sha=sha, created=utc(), columns={}, admit={})

    def col(self, c):
        return self.d["columns"].setdefault(c, dict(phase="new"))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        self.d["updated"] = utc()
        tmp.write_text(json.dumps(self.d, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, self.path)


# ------------------------------------------------------------------ the runner (replaced in tests)
class Shell:
    def run(self, cmd, cwd=None, log=None, env=None, timeout=None):
        """(rc, combined output). The full output also goes to `log` when given."""
        e = dict(os.environ, **(env or {}))
        try:
            p = subprocess.run(cmd, cwd=cwd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=timeout)
            out, rc = p.stdout, p.returncode
        except subprocess.TimeoutExpired as exc:
            out, rc = (exc.stdout or "") if isinstance(exc.stdout, str) else "", 124
        if log:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            with open(log, "a") as fh:
                fh.write("\n# %s %s (cwd %s)\n%s" % (utc(), " ".join(shlex.quote(str(c)) for c in cmd), cwd, out))
        return rc, out

    def spawn(self, cmd, cwd, log, rcfile, env=None):
        """A detached local job that writes its exit code to rcfile; returns its pid."""
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        wrapped = ["bash", "-c", '"$@"; echo $? > %s' % shlex.quote(str(rcfile)), "reference-refresh-job", *cmd]
        with open(log, "a") as fh:
            p = subprocess.Popen(wrapped, cwd=cwd, env=dict(os.environ, **(env or {})), stdout=fh,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        return p.pid

    def alive(self, pid):
        try:
            os.kill(int(pid), 0)
            return True
        except (ProcessLookupError, ValueError, TypeError):
            return False
        except PermissionError:
            return True

    def sleep(self, s):
        time.sleep(s)


def default_python():
    for p in (ROOT / ".pixi/envs/default/bin/python",
              Path.home() / "CascadeProjects/mojolearn/.pixi/envs/default/bin/python"):
        if p.is_file():
            return str(p)
    return sys.executable


def apple_label():
    try:
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], stdout=subprocess.PIPE,
                               text=True).stdout.split()
        chip = brand[1].lower() if len(brand) > 1 and brand[0] == "Apple" else ""
    except OSError:
        chip = ""
    return "apple-" + (chip or "m")


# ------------------------------------------------------------------ the refresh
class Refresh:
    def __init__(self, sha, columns, env=None, sh=None, out=say):
        env = dict(os.environ if env is None else env)
        self.env, self.sh, self.say = env, sh or Shell(), out
        self.sha = sha
        self.sha12 = sha[:12]
        self.columns = list(columns)
        self.home = Path(env.get("REFRESH_HOME") or Path.home() / "mojolearn-evidence/reference-refresh") / self.sha12
        self.py = env.get("REFRESH_PYTHON") or default_python()
        self.src = Path(env.get("REFRESH_SRC_TREE") or Path.home() / "mojolearn-wt/refresh-src")
        self.admit_tree = Path(env.get("REFRESH_ADMIT_TREE") or Path.home() / "mojolearn-wt/reference-admit")
        self.nv_lane = env.get("REFRESH_NV_LANE") or "identity"
        self.apple_tree = env.get("REFRESH_APPLE_TREE") or ""
        self.apple_label = env.get("REFRESH_APPLE_LABEL") or ""
        self.poll = float(env.get("REFRESH_POLL_SEC") or 300)
        self.wait_min = float(env.get("REFRESH_WAIT_MIN") or 600)
        self.scope = env.get("REFRESH_SCOPE") or "routine"
        self.tools = ROOT / "tools"
        self.state = State(self.home / "state.json", sha)
        self.extend_amd = 0

    # -------------------------------------------------------------- helpers
    def git(self, *args, cwd=None):
        return self.sh.run(["git", *args], cwd=str(cwd or ROOT))

    def must(self, cmd, what, cwd=None, log=None, env=None):
        rc, out = self.sh.run(cmd, cwd=str(cwd) if cwd else None, log=log, env=env)
        if rc != 0:
            tail = "\n".join(out.splitlines()[-8:])[-1200:]
            raise Refused("%s failed (rc=%s)%s\n%s" % (what, rc, " (log %s)" % log if log else "", tail))
        return out

    def log(self, name):
        return str(self.home / "logs" / name)

    # -------------------------------------------------------------- the tree at <sha>
    def source_tree(self):
        """The clean detached checkout at <sha> that is planned and synced. Refuses tracked edits and untracked
        source files, so nothing but <sha> reaches a box."""
        src = self.src
        if not (src / ".git").exists():
            self.must(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(src), self.sha],
                      "creating the source tree %s" % src, log=self.log("src-tree.log"))
        rc, head = self.git("rev-parse", "HEAD", cwd=src)
        if rc != 0:
            raise Refused("%s is not a git checkout" % src)
        if head.strip() != self.sha:
            rc, st = self.git("status", "--porcelain", "--untracked-files=no", cwd=src)
            if st.strip():
                raise Refused("%s has tracked edits; it must be a clean tree (no reset is done here)" % src)
            self.must(["git", "-C", str(src), "checkout", "-q", "--detach", self.sha],
                      "checking out %s in %s" % (self.sha12, src), log=self.log("src-tree.log"))
        self.check_clean(src)
        return src

    def check_clean(self, tree):
        rc, st = self.git("status", "--porcelain", "--untracked-files=all", cwd=tree)
        if rc != 0:
            raise Refused("cannot read the status of %s" % tree)
        bad = [ln for ln in st.splitlines() if ln.strip()]
        if bad:
            raise Refused("%s is not clean at %s (the box would refuse it, or run code that is not %s): %s"
                          % (tree, self.sha12, self.sha12, "; ".join(bad[:5])))

    def on_main(self):
        rc, _ = self.git("merge-base", "--is-ancestor", self.sha, "origin/main")
        if rc != 0:
            raise Refused("%s is not on origin/main (git fetch first); a refresh records merged commits only"
                          % self.sha12)

    # -------------------------------------------------------------- plan
    def selection_cmd(self, c, tree):
        mode, backend = SELECTION[c]
        cmd = [self.py, "tools/identity_columns.py", "changed-lanes", "--scope", self.scope, "--backend", backend]
        if mode == "class":
            cls = "cpu" if c == "cpu" else "apple"
            cmd += ["--class", cls, "--records", str(Path(tree) / "bench/results/identity_break"),
                    "--summary-json", str(self.home / "plan" / ("%s.reuse.json" % c))]
        return cmd

    def select(self, c, tree):
        (self.home / "plan").mkdir(parents=True, exist_ok=True)
        rc, out = self.sh.run(self.selection_cmd(c, tree), cwd=str(tree), log=self.log("plan-%s.log" % c))
        sel = parse_selection(out)
        if rc != 0 or sel is None:
            raise Refused("the %s lane selection failed (rc=%s); see %s" % (c, rc, self.log("plan-%s.log" % c)))
        return sel

    def plan(self, dry=False):
        """REUSED / RECORD per column, from the same selection each column's job makes. A refresh keeps the
        plan it started with (it decides which jobs run); --plan (dry) recomputes it and saves nothing."""
        tree = self.source_tree()
        plans = {}
        for c in self.columns:
            col = self.state.col(c)
            if col.get("plan") and not dry:
                plans[c] = col["plan"]
                continue
            plans[c] = self.select(c, tree)
            if not dry:
                col["plan"] = plans[c]
                if col.get("phase") == "new":
                    col["phase"] = "nothing" if plans[c]["record"] == 0 else "planned"
                self.state.save()
        for c in self.columns:
            p = plans[c]
            if p["record"] == 0:
                self.say("PLAN %-6s NOTHING-CHANGED (REUSED %d of %d)" % (c, p["reused"], p["total"]))
            else:
                self.say("PLAN %-6s REUSED %d / RECORD %d of %d%s" % (
                    c, p["reused"], p["record"], p["total"],
                    " (known-bad %d)" % p["known_bad"] if p.get("known_bad") else ""))

    def nvidia_needed(self):
        """The nvidia job also builds the tree the cpu column records in."""
        nv = self.state.col("nvidia") if "nvidia" in self.columns else None
        cpu = self.state.col("cpu") if "cpu" in self.columns else None
        own = bool(nv and nv["plan"]["record"])
        for_cpu = bool(cpu and cpu["plan"]["record"])
        return own or for_cpu, for_cpu and not own

    # -------------------------------------------------------------- submit
    def nvc(self, *args):
        return ["bash", str(self.tools / "nvidia_central.sh"), *args]

    def amd(self, *args):
        return ["bash", str(self.tools / "do_amd_steward.sh"), *args]

    def clean_box_cmd(self, tree):
        """Remove untracked record outputs (never tracked files, never ignored build outputs) from a box tree."""
        paths = " ".join(RECORD_OUTPUT_PATHS)
        return ("[ ! -d %s/.git ] || { cd %s && git clean -fdq -- %s && git status --porcelain --untracked-files=all"
                " -- %s | head -n 5; }" % (tree, tree, paths, paths))

    def submit_nvidia(self):
        col = self.state.col("nvidia")
        tree = "/root/mojolearn-%s" % self.nv_lane
        self.must(self.nvc("sh", self.nv_lane, self.clean_box_cmd(tree)), "cleaning record outputs from the pod tree",
                  log=self.log("nvidia-sync.log"))
        self.must(self.nvc("sync", self.nv_lane, str(self.src)), "syncing %s to the NVIDIA pod" % self.sha12,
                  log=self.log("nvidia-sync.log"))
        out = self.must(self.nvc("submit", self.nv_lane, "--cap", "240", "--note", "reference_refresh %s" % self.sha12,
                                 "tools/identity_box_job.sh"),
                        "submitting the NVIDIA job", log=self.log("nvidia-submit.log"))
        tag, job = parse_submit(out)
        if not tag:
            raise Refused("the NVIDIA submit printed no persisted tag; see %s" % self.log("nvidia-submit.log"))
        col.update(phase="submitted", tag=tag, job=job, submitted=utc())
        self.state.save()
        self.say("SUBMITTED nvidia %s tag %s" % (job, tag))

    def submit_cpu(self):
        """In the tree the finished nvidia job built (HEAD == <sha>); no sync, so nothing moves under it."""
        col = self.state.col("cpu")
        out = self.must(self.nvc("submit", self.nv_lane, "--cap", "240", "--note", "reference_refresh cpu %s" % self.sha12,
                                 "tools/identity_box_job_cpu_ptx.sh"),
                        "submitting the cpu job", log=self.log("cpu-submit.log"))
        tag, job = parse_submit(out)
        if not tag:
            raise Refused("the cpu submit printed no persisted tag; see %s" % self.log("cpu-submit.log"))
        col.update(phase="submitted", tag=tag, job=job, submitted=utc())
        self.state.save()
        self.say("SUBMITTED cpu %s tag %s" % (job, tag))

    def launch_amd(self):
        """`do_amd_steward.sh run` holds its ssh session for the whole job, so it runs detached on this Mac; the
        job itself ignores HUP on the droplet and persists to R2, so a Mac crash loses nothing."""
        col = self.state.col("amd")
        self.must(self.amd("ssh", self.clean_box_cmd("/root/mojolearn")), "cleaning record outputs from the droplet tree",
                  log=self.log("amd-update.log"))
        self.must(self.amd("update", self.sha), "moving the droplet's tree to %s" % self.sha12,
                  log=self.log("amd-update.log"))
        log = self.log("amd-run.log")
        rcfile = self.home / "logs" / "amd-run.rc"
        if rcfile.exists():
            rcfile.unlink()
        res = "box-out/refresh-%s-amd-%s" % (self.sha12, datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        pid = self.sh.spawn(self.amd("run", "identity-amd", res, "bash tools/identity_box_job.sh amd"),
                            str(ROOT), log, str(rcfile))
        col.update(phase="launching", launcher_pid=pid, launch_log=log, submitted=utc())
        self.state.save()

    def amd_tag(self):
        col = self.state.col("amd")
        log = Path(col.get("launch_log") or "")
        tag = parse_persist_tag(log.read_text(errors="replace")) if log.is_file() else None
        if tag:
            col.update(phase="submitted", tag=tag)
            self.state.save()
            self.say("SUBMITTED amd tag %s" % tag)
            return True
        rcfile = self.home / "logs" / "amd-run.rc"
        if rcfile.exists() or not self.sh.alive(col.get("launcher_pid")):
            col.update(phase="failed", why="the AMD launcher ended before the job was persisted; see %s" % log)
            self.state.save()
        return False

    def run_apple(self):
        col = self.state.col("apple")
        tree = self.apple_tree
        if not tree:
            col.update(phase="blocked", why="apple owes %d lane(s): set REFRESH_APPLE_TREE to a checkout at %s with its "
                       "Apple bindings built" % (col["plan"]["record"], self.sha12))
            self.state.save()
            return
        rc, head = self.git("rev-parse", "HEAD", cwd=tree)
        if rc != 0 or head.strip() != self.sha:
            raise Refused("REFRESH_APPLE_TREE %s is not at %s" % (tree, self.sha12))
        rc, st = self.git("status", "--porcelain", "--untracked-files=no", cwd=tree)
        if st.strip():
            raise Refused("REFRESH_APPLE_TREE %s has tracked edits" % tree)
        label = self.apple_label or apple_label()
        out = self.home / "apple" / "column"
        out.mkdir(parents=True, exist_ok=True)
        rcfile = self.home / "apple" / "record.rc"
        if rcfile.exists():
            rcfile.unlink()
        cmd = [self.py, str(Path(tree) / "tools/mac_slot.py"), "--wait-timeout", "3600", "metal", "--",
               "env", "REPEATS=1", "SCOPE=%s" % self.scope, "PYTHON=%s" % self.py,
               "bash", "tools/record_identity_column.sh", label, str(out)]
        pid = self.sh.spawn(cmd, tree, str(self.home / "apple" / "record.log"), str(rcfile))
        col.update(phase="running-local", pid=pid, label=label, out=str(out), submitted=utc())
        self.state.save()
        self.say("STARTED apple record %s (pid %s) in %s" % (label, pid, tree))

    # -------------------------------------------------------------- wait + fetch
    def persist(self, *args):
        return [self.py, str(self.tools / "box_persist.py"), *args]

    def check_job(self, c):
        """Advance one submitted column by its R2 manifest. Returns True when the column is settled."""
        col = self.state.col(c)
        rc, out = self.sh.run(self.persist("fetch", "--tag", col["tag"], "--list"), log=self.log("%s-poll.log" % c))
        phase = job_phase(parse_manifest(out))
        col["last_poll"] = dict(utc=utc(), phase=phase)
        if phase in ("queued", "running"):
            self.state.save()
            return False
        if phase == "dead":
            col.update(phase="dead", why="the box died mid-job (manifest state=flushed); its partial results are "
                       "in R2 under %s. Rerun with --resubmit %s" % (col["tag"], c))
            self.state.save()
            return True
        dest = self.home / "fetch" / c
        rc, out = self.sh.run(self.persist("fetch", "--tag", col["tag"], "--dest", str(dest), "--extract"),
                              log=self.log("%s-fetch.log" % c))
        if rc != 0:
            col.update(phase="failed", why="fetching %s failed; see %s" % (col["tag"], self.log("%s-fetch.log" % c)))
            self.state.save()
            return True
        self.settle(c, dest, job_ok=(phase != "failed"))
        return True

    def settle(self, c, tree, job_ok=True):
        """Collect the records of a finished column (box or local) into <home>/records/<c>."""
        col = self.state.col(c)
        recs, outcomes, other = find_records(tree, self.sha)
        kinds = {o for _, o in outcomes}
        if other:
            col["other_commit_records"] = [str(p) for p in other[:20]]
        if not job_ok:
            col.update(phase="failed", why="the %s job exited non-zero; see the status.txt/job.log under %s" % (c, tree))
        elif "inadmissible" in kinds:
            col.update(phase="failed", why="a %s record is not admissible; see the record.log under %s" % (c, tree))
        elif recs:
            got = collect_records(recs, self.home / "records" / c)
            col.update(phase="done", records=[str(p) for p in got], fetched=utc())
        elif "nothing" in kinds:
            col.update(phase="done", records=[], fetched=utc(), note="NOTHING-CHANGED on the box")
        elif c == "nvidia" and col.get("for_cpu_only"):
            col.update(phase="done", records=[], fetched=utc(), note="built the tree for the cpu column")
        else:
            col.update(phase="failed", why="the %s job finished with no record and no NOTHING-CHANGED; see %s" % (c, tree))
        self.state.save()
        self.say("%s %s %s" % (col["phase"].upper(), c, col.get("why") or "%d record(s)" % len(col.get("records", []))))

    def check_apple(self):
        col = self.state.col("apple")
        rcfile = self.home / "apple" / "record.rc"
        if not rcfile.exists():
            if self.sh.alive(col.get("pid")):
                return False
            col.update(phase="failed", why="the apple record ended without an exit code; see %s"
                       % (self.home / "apple" / "record.log"))
            self.state.save()
            return True
        rc = rcfile.read_text().strip()
        apple_dir = self.home / "apple"
        if rc != "0":
            col.update(phase="failed", why="the apple record exited rc=%s; see %s" % (rc, apple_dir / "record.log"))
            self.state.save()
            return True
        self.settle("apple", apple_dir)
        return True

    # -------------------------------------------------------------- admission
    def admit(self):
        a = self.state.d["admit"]
        branch = "ref/%s" % self.sha12
        if a.get("pushed"):
            self.say("ADMITTED %s on %s (pushed); nothing more to do" % (a["commit"][:12], branch))
            return
        dirs = sorted({str(Path(r).parent) for c in self.columns for r in self.state.col(c).get("records", [])})
        if not dirs and not a.get("commit"):
            a.update(phase="nothing", note="no column recorded a lane; table.json is already current for %s" % self.sha12)
            self.state.save()
            self.say("NOTHING-CHANGED: every column reuses its rows; no admission")
            return
        tree = self.admit_tree
        if not a.get("commit"):
            self.prepare_admit_tree(tree, branch)
            env = dict(PYTHON=self.py, EVIDENCE=str(self.home / "admit"), REF_TAG="ref-%s" % self.sha12)
            log = self.log("admit.log")
            rc, out = self.sh.run(["bash", "tools/admit_identity_columns.sh", "--build-host", *dirs], cwd=str(tree),
                                  log=log, env=env)
            paths = parse_admit_next(out)
            if rc != 0 or not paths:
                a.update(phase="refused", why="admission refused (rc=%s); see %s" % (rc, log))
                self.state.save()
                raise Refused(a["why"])
            self.must(["git", "add", "--", *paths], "staging the admission", cwd=tree)
            cols = ",".join(c for c in self.columns if self.state.col(c).get("records"))
            msg = ("Admit identity columns at %s (%s) via tools/reference_refresh.sh\n\nRecords: %s.\n\n"
                   "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % (self.sha12, cols, ", ".join(dirs)))
            self.must(["git", "commit", "-q", "-m", msg], "committing the admission", cwd=tree)
            rc, head = self.git("rev-parse", "HEAD", cwd=tree)
            a.update(phase="committed", commit=head.strip(), branch=branch)
            self.state.save()
        self.must(["git", "push", "-q", "-u", "origin", "%s:%s" % (branch, branch)], "pushing %s" % branch, cwd=tree,
                  log=self.log("push.log"))
        a.update(phase="pushed", pushed=utc())
        self.state.save()
        self.say("ADMITTED %s; pushed %s: merge it into main (the orchestrator)" % (a["commit"][:12], branch))

    def prepare_admit_tree(self, tree, branch):
        """The admission checkout on ref/<sha12>, started from origin/main so earlier admissions are its base."""
        if not (Path(tree) / ".git").exists():
            self.must(["git", "-C", str(ROOT), "worktree", "add", "-B", branch, str(tree), "origin/main"],
                      "creating the admission tree %s" % tree, log=self.log("admit-tree.log"))
            return
        rc, st = self.git("status", "--porcelain", "--untracked-files=no", cwd=tree)
        if st.strip():
            raise Refused("the admission tree %s has tracked edits (a crashed admission?). Inspect it; nothing "
                          "is reset here" % tree)
        rc, cur = self.git("rev-parse", "--abbrev-ref", "HEAD", cwd=tree)
        if cur.strip() == branch:
            return
        self.must(["git", "-C", str(tree), "switch", "-q", "-C", branch, "origin/main"],
                  "switching %s to %s" % (tree, branch), log=self.log("admit-tree.log"))

    # -------------------------------------------------------------- the driver
    def step(self):
        """One pass: start what is owed, poll what runs. Returns the columns still in flight."""
        need_nv, nv_for_cpu = self.nvidia_needed()
        flight = []
        for c in self.columns:
            col = self.state.col(c)
            ph = col["phase"]
            if c == "nvidia" and ph == "nothing" and need_nv:
                col.update(phase="planned", for_cpu_only=nv_for_cpu)
                ph = "planned"
            if ph in ("nothing", "done", "failed", "dead", "blocked"):
                continue
            if ph == "planned":
                if c == "nvidia":
                    self.submit_nvidia()
                elif c == "amd":
                    self.launch_amd()
                elif c == "apple":
                    self.run_apple()
                    if self.state.col(c)["phase"] == "blocked":
                        continue
                elif c == "cpu":
                    nv = self.state.col("nvidia") if "nvidia" in self.columns else None
                    if nv is None:
                        raise Refused("the cpu column records in the tree the nvidia job builds; add nvidia")
                    if nv["phase"] in ("failed", "dead", "blocked"):
                        col.update(phase="failed", why="the nvidia job it needs did not finish (%s)" % nv["phase"])
                        self.state.save()
                        continue
                    if nv["phase"] != "done":
                        flight.append(c)
                        continue
                    self.submit_cpu()
                flight.append(c)
                continue
            if ph == "launching":
                if not self.amd_tag() and self.state.col(c)["phase"] == "launching":
                    flight.append(c)
                continue
            if ph == "submitted":
                if c == "amd" and self.extend_amd:
                    self.sh.run(self.amd("extend", str(self.extend_amd)), log=self.log("amd-extend.log"))
                if not self.check_job(c):
                    flight.append(c)
                continue
            if ph == "running-local":
                if not self.check_apple():
                    flight.append(c)
                continue
            raise Refused("unknown phase %s for %s" % (ph, c))
        return flight

    def run(self, wait=True):
        self.on_main()
        self.plan()
        deadline = time.monotonic() + self.wait_min * 60
        while True:
            flight = self.step()
            if not flight:
                break
            if not wait:
                self.say("IN FLIGHT %s; rerun the same command to fetch and admit" % ",".join(flight))
                return 0
            if time.monotonic() > deadline:
                self.say("STILL IN FLIGHT after %d min: %s; rerun the same command later (nothing is lost)"
                         % (self.wait_min, ",".join(flight)))
                return 3
            self.sh.sleep(self.poll)
        bad = [(c, self.state.col(c)) for c in self.columns
               if self.state.col(c)["phase"] in ("failed", "dead", "blocked")]
        for c, col in bad:
            self.say("%s %s: %s" % (col["phase"].upper(), c, col.get("why", "")))
        if bad:
            self.say("NOT ADMITTED: every requested column must finish first (fix it, or drop the column)")
            return 1
        self.admit()
        return 0

    def status(self):
        for c, col in sorted(self.state.d["columns"].items()):
            p = col.get("plan") or {}
            self.say("%-6s %-13s record=%s reused=%s tag=%s %s" % (
                c, col.get("phase"), p.get("record"), p.get("reused"), col.get("tag") or "-",
                (col.get("why") or col.get("note") or "")[:160]))
        a = self.state.d.get("admit") or {}
        self.say("admit  %s %s" % (a.get("phase") or "-", (a.get("commit") or "")[:12]))

    def resubmit(self, c):
        col = self.state.col(c)
        old = {k: col.get(k) for k in ("phase", "tag", "job", "why")}
        col.setdefault("history", []).append(dict(old, forgotten=utc()))
        for k in ("tag", "job", "why", "records", "launcher_pid", "launch_log", "pid", "last_poll"):
            col.pop(k, None)
        # an nvidia job owed only to build the cpu column's tree is planned again by step()
        col["phase"] = "planned" if (col.get("plan") or {}).get("record") else "nothing"
        self.state.save()
        self.say("FORGOT %s's job %s (kept in history); it is submitted again" % (c, old.get("tag")))


def parse_columns(text):
    cols = [c.strip() for c in (text or ",".join(COLUMNS)).split(",") if c.strip()]
    bad = [c for c in cols if c not in COLUMNS]
    if bad or not cols:
        raise SystemExit("reference_refresh: columns are a comma list of %s (got %s)" % (",".join(COLUMNS), text))
    return [c for c in COLUMNS if c in cols]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="reference_refresh.sh", description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", action="store_true", help="print REUSED / RECORD per column; submit nothing")
    ap.add_argument("--status", action="store_true", help="print the saved state")
    ap.add_argument("--no-wait", action="store_true", help="submit what is owed and exit")
    ap.add_argument("--resubmit", default="", help="forget this column's job (a dead box), then continue")
    ap.add_argument("--extend-amd", type=int, default=0, metavar="MIN",
                    help="while the AMD job runs, move the droplet's lease to now + MIN at each poll")
    ap.add_argument("sha")
    ap.add_argument("columns", nargs="?", default="")
    a = ap.parse_args(argv)
    cols = parse_columns(a.columns)
    rc, full = Shell().run(["git", "-C", str(ROOT), "rev-parse", "--verify", a.sha + "^{commit}"])
    if rc != 0:
        raise SystemExit("reference_refresh: %s is not a commit here (git fetch first)" % a.sha)
    try:
        r = Refresh(full.strip(), cols)
        r.extend_amd = a.extend_amd
        if a.status:
            r.status()
            return 0
        if a.plan:
            r.plan(dry=True)
            return 0
        if a.resubmit:
            if a.resubmit not in cols:
                raise Refused("--resubmit %s is not among the columns %s" % (a.resubmit, ",".join(cols)))
            r.resubmit(a.resubmit)
        return r.run(wait=not a.no_wait)
    except Refused as exc:
        say("REFUSED %s" % exc)
        return 2


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: sys.exit(130))
    sys.exit(main())
