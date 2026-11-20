#!/usr/bin/env python3
"""tools/reference_refresh.py -- THE REFERENCE REFRESH after a merge to main (2026-11-04). Entry point:
tools/reference_refresh.sh, which execs this file. Docs: docs/release/REFERENCE_REFRESH.md.

  reference_refresh.sh [<sha>] [columns]        record the changed identity lanes of <sha> (default: the tip of
                                                origin/main) on each column, admit every fetched record at once
                                                into table.json and commit the admission on branch ref/<sha12>,
                                                pushed for the orchestrator
  reference_refresh.sh --plan [<sha>] [columns] per column, lanes OWED / REUSED and the GAPS; submits nothing
  reference_refresh.sh --status <sha>           the saved state of a refresh
  reference_refresh.sh --resubmit COL <sha> [columns]
                                                forget COL's job (a box that died mid-job), then continue
  reference_refresh.sh --no-wait <sha> [columns]
                                                submit what is owed and exit; rerun the same command to
                                                fetch, admit and push

  columns: comma list of nvidia, amd, cpu, ptx, apple. The default (and the only scope that needs no reason) is
  the release's SUPPORTED SURFACE: every column of the table at <sha> x every lane the harness ships
  (identity_columns.py --scope all: the routine AND the neural-training lanes). Naming fewer columns, or
  --scope routine|neural-training, only NARROWS it and needs --reason TEXT, which the state and the
  admission README keep. A recorder default never decides the scope (2026-11-04: the routine default left
  85 neural-training lanes per column and the whole PTX column unrecorded after a merge).

COALESCED. A refresh always targets the newest selected commit: the requested one or a newer pending refresh.
Every pending refresh of an older commit is SUPERSEDED (kept in its state, never admitted), and the target
records only the lanes whose closure differs from their admitted record, once. A pushed admission that is not
merged into the target stops the refresh (merge ref/<sha12> first), so no lane is recorded twice.

GAPS. A column that owes lanes and has no box or column runner (no NVIDIA pod, no AMD droplet, no built Apple
tree, no sm_80 PTX set, a table column with no runner) is a GAP: --plan and the run print it per column, the
other columns are still admitted, the admission README records it, and the run exits 4.

Each column runs exactly what its own tool runs (the same lane selection the plan prints):
  nvidia  tools/nvidia_central.sh sync + submit tools/identity_box_job.sh (box_merged_verify.sh: merged
          bindings, `python -m mojolearn verify`, ONLY_CHANGED=table record), persisted to R2 from the pod
  amd     tools/do_amd_steward.sh update <sha> + run tools/identity_box_job.sh amd, persisted to R2
  cpu     tools/identity_box_job_cpu_ptx.sh on the NVIDIA pod, in the tree the nvidia job built (auto reuse);
          when cpu owes lanes and nvidia owes none, the nvidia job still runs: it builds that tree
  ptx     the same cpu job with the release leg's sm_80 PTX set (REFRESH_PTX_SET_TGZ, a path on the pod, with
          REFRESH_PTX_SET_SHA256 and REFRESH_PTX_SET_COMMIT); without a set an owed PTX column is a GAP
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
REFRESH_POLL_SEC (300), REFRESH_WAIT_MIN (600), REFRESH_PTX_SET_TGZ / _SHA256 / _COMMIT. REFRESH_SCOPE is
refused: the scope is the supported surface, narrowed only by --scope with --reason.
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
#: The columns a refresh can record, keyed like the table's columns (table.json cells[...][part]["cols"]).
COLUMNS = ("nvidia", "amd", "cpu", "ptx", "apple")
#: What each column's own job runs, so the plan prints the selection the job will make.
#:   table  ONLY_CHANGED=table (box_merged_verify.sh), backend-filtered
#:   class  record_identity_column.sh's default auto reuse from the class's newest good evidence
#: (ptx: identity_columns.CLASS_BACKEND ptx=cuda, the class record_identity_column.sh gives nvidia-ptx-* labels)
SELECTION = dict(nvidia=("table", "cuda"), amd=("table", "hip"), cpu=("class", "cpu"), ptx=("class", "cuda"),
                 apple=("class", "metal"))
#: The release's supported surface is every lane the harness ships (identity_columns.py --scope all = the
#: routine lanes + the neural-training lanes). These narrower scopes need a --reason.
SURFACE_SCOPE = "all"
NARROW_SCOPES = ("routine", "neural-training")
TABLE = "python/mojolearn/verify_reference/table.json"
ADMISSIONS = "bench/results/reference_admissions/"
#: The env a PTX record needs (tools/identity_box_job_cpu_ptx.sh): REFRESH_<name> on the Mac -> <name> on the pod.
PTX_ENV = ("PTX_SET_TGZ", "PTX_SET_SHA256", "PTX_SET_COMMIT")
#: Paths a record run may leave untracked in a box tree; they are removed before a sync (0.8.65: a resubmit
#: refused "untracked source bench/results/identity_break/retained/...").
RECORD_OUTPUT_PATHS = ("bench/results/identity_break", "bench/results/reference_admissions")
#: Files that travel with a record into admission (tools/admit_identity_columns.sh's NEXT line names them).
RECORD_COMPANIONS = (".provenance.txt", ".source-report.json.gz", ".verifier-run.json")
NOT_A_RECORD = (".admission-preview-", ".other-commit-")
#: Tracked files tools/admit_identity_columns.sh rewrites BEFORE its emit-reference can refuse: the
#: identity_admission_guard.py `inputs` step retains each record and repoints these per-vendor summaries
#: (tools/retain_run_summary.py retain_identity). A refused admission restores exactly these, and only when
#: the run itself changed them (0.8.66: the admission tree kept them after a refused emit, and the next run
#: refused on "tracked edits"). Untracked retained records stay: the next run reuses them.
ADMIT_GENERATED = re.compile(r"^docs/verification/latest/auto/identity-[^/]+\.json$")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def say(msg):
    print("reference_refresh: %s" % msg, flush=True)


class Refused(Exception):
    """A stop the operator has to act on; the state is saved first."""


# ------------------------------------------------------------------ parsing (pure; unit tested)
def parse_selection(text):
    """REUSED / RECORD counts from tools/identity_columns.py changed-lanes stderr.

    --class mode prints `# REUSE-SUMMARY class=c REUSED a RECORD b KNOWN-BAD k of N (scope) N/A n`; table mode prints
    `# CHANGED-LANES n of N (scope)`. Returns dict(record, reused, known_bad, total) or None."""
    m = re.search(r"REUSE-SUMMARY class=\S+ REUSED (\d+) RECORD (\d+) KNOWN-BAD (\d+) of (\d+)(?: \(\S+\))?"
                  r"(?: N/A (\d+))?", text)
    if m:
        reused, record, bad, total = map(int, m.groups()[:4])
        out = dict(record=record, reused=reused, known_bad=bad, total=total)
        if m.group(5) is not None:
            out["not_applicable"] = int(m.group(5))   # declared no-route lanes (GPU_ONLY_LANES on cpu)
        return out
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


def table_columns(table):
    """The column keys a reference table holds (cells[lane/fixture][part]["cols"]), sorted. `table` is a path
    or the parsed table."""
    if not isinstance(table, dict):
        table = json.loads(Path(table).read_text())
    cols = set()
    for cell in (table.get("cells") or {}).values():
        for ent in (cell or {}).values():
            if isinstance(ent, dict):
                cols.update((ent.get("cols") or {}).keys())
    return sorted(cols)


def is_ptx_output(path, tree):
    """The cpu job records the cpu column and the PTX column side by side (identity_box_job_cpu_ptx.sh: <out>/cpu,
    <out>/ptx, cpu-record.log, ptx-record.log); a path belongs to the PTX column when it sits under a `ptx`
    directory or is the ptx record log."""
    rel = Path(path).relative_to(tree)
    return "ptx" in rel.parts[:-1] or rel.name == "ptx-record.log"


def find_records(tree, sha=None, part=None):
    """Every *.identical.json under an extracted job dir that was recorded at `sha` (records of another commit,
    e.g. an older run's leftovers in a reused results dir, are never admitted from here), with every
    record log outcome (record.log, cpu-record.log, ptx-record.log). part="cpu" or "ptx" keeps only that
    column's half of a cpu job's results. Returns (records, outcomes, other_commit)."""
    tree = Path(tree)

    def mine(p):
        return part is None or (is_ptx_output(p, tree) == (part == "ptx"))
    found = sorted(p for p in tree.rglob("*.identical.json")
                   if not any(t in p.name for t in NOT_A_RECORD) and p.is_file() and mine(p))
    recs, other = [], []
    for p in found:
        (recs if sha is None or record_commit(p) == sha else other).append(p)
    outcomes = []
    logs = [p for p in tree.rglob("*record.log") if p.name in ("record.log", "cpu-record.log", "ptx-record.log")]
    for log in sorted(logs):
        if not mine(log):
            continue
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
    def __init__(self, sha, columns=None, env=None, sh=None, out=say, lane_scope=None, reason=""):
        """columns None (the default) and lane_scope None are the supported surface; a column list or a narrower
        lane scope narrows it and needs `reason`."""
        env = dict(os.environ if env is None else env)
        self.env, self.sh, self.say = env, sh or Shell(), out
        self.sha = sha
        self.sha12 = sha[:12]
        self.asked_columns = list(columns) if columns else None
        self.columns = list(self.asked_columns or COLUMNS)
        if lane_scope not in (None, SURFACE_SCOPE) + NARROW_SCOPES:
            raise Refused("--scope %s: the lane scope is %s (the supported surface) or narrows to one of %s"
                          % (lane_scope, SURFACE_SCOPE, ", ".join(NARROW_SCOPES)))
        self.lane_scope = None if lane_scope == SURFACE_SCOPE else lane_scope
        self.reason = (reason or "").strip()
        if env.get("REFRESH_SCOPE"):
            raise Refused("REFRESH_SCOPE is not read: the scope is the release's supported surface (every table "
                          "column x every lane the harness ships); narrow it with --scope %s --reason TEXT"
                          % "|".join(NARROW_SCOPES))
        self.home = Path(env.get("REFRESH_HOME") or Path.home() / "mojolearn-evidence/reference-refresh") / self.sha12
        self.py = env.get("REFRESH_PYTHON") or default_python()
        self.src = Path(env.get("REFRESH_SRC_TREE") or Path.home() / "mojolearn-wt/refresh-src")
        self.admit_tree = Path(env.get("REFRESH_ADMIT_TREE") or Path.home() / "mojolearn-wt/reference-admit")
        self.nv_lane = env.get("REFRESH_NV_LANE") or "identity"
        self.apple_tree = env.get("REFRESH_APPLE_TREE") or ""
        self.apple_label = env.get("REFRESH_APPLE_LABEL") or ""
        self.poll = float(env.get("REFRESH_POLL_SEC") or 300)
        self.wait_min = float(env.get("REFRESH_WAIT_MIN") or 600)
        self.ptx_set = {k: env.get("REFRESH_" + k) or "" for k in PTX_ENV}
        self.devpods = Path(env.get("MOJOLEARN_DEVPOD_STATE") or Path.home() / "mojolearn-evidence/devpods")
        self.do_state = Path(env.get("MOJOLEARN_STEWARD_DO_STATE") or Path.home() / "mojolearn-evidence/do-amd-steward")
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

    # -------------------------------------------------------------- scope
    def resolve_scope(self, tree):
        """The release's supported surface: the columns of the table at <sha> x every lane the harness ships.
        Returns the scope record kept in the state; a narrower request without a reason is refused, and a
        request the surface does not hold (a column the table has no rows for) is refused: --scope only
        narrows."""
        table = Path(tree) / TABLE
        try:
            surface_cols = table_columns(table)
        except (OSError, ValueError) as exc:
            raise Refused("cannot read the reference table %s (%s): the scope is derived from it" % (table, exc))
        runnable = [c for c in COLUMNS if c in surface_cols]
        no_runner = sorted(set(surface_cols) - set(COLUMNS))
        cols = self.asked_columns or runnable
        outside = [c for c in cols if c not in surface_cols]
        if outside:
            raise Refused("column(s) %s are not in the table at %s (it holds %s): --scope and the column list "
                          "only narrow the supported surface" % (",".join(outside), self.sha12, ",".join(surface_cols)))
        if "ptx" in cols and "cpu" not in cols:
            raise Refused("the PTX column records in the cpu job (tools/identity_box_job_cpu_ptx.sh); add cpu")
        dropped = [c for c in runnable if c not in cols]
        narrowed = bool(dropped or self.lane_scope)
        if narrowed and not self.reason:
            raise Refused("narrowing the supported surface (%s) needs --reason TEXT; the reason is kept in the "
                          "state and the admission README"
                          % "; ".join(x for x in (dropped and "columns without %s" % ",".join(dropped),
                                                  self.lane_scope and "lanes %s" % self.lane_scope) if x))
        return dict(surface_columns=surface_cols, columns=[c for c in COLUMNS if c in cols], dropped_columns=dropped,
                    no_runner_columns=no_runner, lanes=self.lane_scope or SURFACE_SCOPE, narrowed=narrowed,
                    reason=self.reason if narrowed else "")

    def fix_scope(self, scope, dry):
        """A refresh keeps the scope it was planned with; a rerun asking for another is refused."""
        saved = self.state.d.get("scope")
        if saved and not dry:
            keys = ("columns", "lanes")
            if any(saved.get(k) != scope.get(k) for k in keys):
                raise Refused("this refresh of %s was planned with columns %s, lanes %s; it keeps that scope "
                              "(asked: columns %s, lanes %s)" % (self.sha12, ",".join(saved["columns"]),
                                                                  saved["lanes"], ",".join(scope["columns"]),
                                                                  scope["lanes"]))
            scope = saved
        elif not dry:
            self.state.d["scope"] = scope
            self.state.save()
        self.columns = list(scope["columns"])
        return scope

    def box_scope(self):
        """The SCOPE every column's own record runs with: the surface, or the narrowed lane scope."""
        return (self.state.d.get("scope") or {}).get("lanes") or self.lane_scope or SURFACE_SCOPE

    def available(self, c):
        """(True, "") when column c has a box or a runner; else (False, why). Read from the local state files the
        box tools keep (no network): tools/nvidia_central.sh's pods, tools/do_amd_steward.sh's droplet."""
        if c not in COLUMNS:
            return False, "the refresh has no runner for the table column %s" % c
        if c in ("nvidia", "cpu", "ptx"):
            if not any((self.devpods / ("nvc%d" % i) / "state.env").is_file() for i in range(1, 4)):
                return False, ("no shared NVIDIA pod is up (%s/nvc*/state.env); only the orchestrator brings one "
                               "up (tools/nvidia_central.sh up)" % self.devpods)
            if c == "ptx" and not all(self.ptx_set.values()):
                return False, ("no sm_80 PTX set: the PTX column records only with the release leg's set; set "
                               "REFRESH_PTX_SET_TGZ (its path on the pod), REFRESH_PTX_SET_SHA256 and "
                               "REFRESH_PTX_SET_COMMIT")
        if c == "amd" and not (self.do_state / "state.env").is_file():
            return False, "no AMD droplet is up (%s/state.env); only the orchestrator brings one up" % self.do_state
        if c == "apple" and not self.apple_tree:
            return False, ("no built Apple tree: set REFRESH_APPLE_TREE to a checkout at %s with its Apple bindings "
                           "built (the refresh never compiles)" % self.sha12)
        return True, ""

    # -------------------------------------------------------------- plan
    def selection_cmd(self, c, tree):
        mode, backend = SELECTION[c]
        cmd = [self.py, "tools/identity_columns.py", "changed-lanes", "--scope", SURFACE_SCOPE, "--backend", backend]
        if mode == "class":
            cmd += ["--class", c, "--records", str(Path(tree) / "bench/results/identity_break")]
        return cmd + ["--summary-json", str(self.home / "plan" / ("%s.reuse.json" % c))]

    def select(self, c, tree):
        """The column's selection over the supported surface: counts plus the owed lane list (the summary json)."""
        (self.home / "plan").mkdir(parents=True, exist_ok=True)
        summary = self.home / "plan" / ("%s.reuse.json" % c)
        if summary.exists():
            summary.unlink()
        rc, out = self.sh.run(self.selection_cmd(c, tree), cwd=str(tree), log=self.log("plan-%s.log" % c))
        sel = parse_selection(out)
        if rc != 0 or sel is None:
            raise Refused("the %s lane selection failed (rc=%s); see %s" % (c, rc, self.log("plan-%s.log" % c)))
        try:
            owed = sorted((json.loads(summary.read_text()).get("record") or {}).keys())
        except (OSError, ValueError, AttributeError):
            owed = None
        if owed is not None and len(owed) != sel["record"]:
            raise Refused("the %s selection printed RECORD %d but its summary lists %d lane(s); see %s"
                          % (c, sel["record"], len(owed), self.log("plan-%s.log" % c)))
        sel["owed"] = sel["record"]
        if owed is not None:
            sel["owed_lanes"] = owed
        return sel

    def scope_lanes(self, tree):
        """The lanes of the narrowed lane scope (identity_columns.py lanes --scope X), read once."""
        if getattr(self, "_scope_lanes", None) is None:
            rc, out = self.sh.run([self.py, "tools/identity_columns.py", "lanes", "--scope", self.box_scope()],
                                  cwd=str(tree), log=self.log("plan-scope.log"))
            lanes = [x for x in (out.strip().splitlines() or [""])[-1].split(",") if x]
            if rc != 0 or not lanes:
                raise Refused("listing the %s lanes failed (rc=%s); see %s"
                              % (self.box_scope(), rc, self.log("plan-scope.log")))
            self._scope_lanes = set(lanes)
        return self._scope_lanes

    def plan_column(self, c, tree):
        sel = self.select(c, tree)
        if self.box_scope() != SURFACE_SCOPE and sel["owed"]:
            if "owed_lanes" not in sel:
                raise Refused("the %s selection wrote no lane list; a narrowed scope needs it" % c)
            keep = self.scope_lanes(tree)
            out = [x for x in sel["owed_lanes"] if x not in keep]
            sel["narrowed_lanes"] = out
            sel["narrowed"] = len(out)
            sel["record"] = sel["owed"] - len(out)
        ok, why = self.available(c)
        if sel["record"] and not ok:
            sel["gap"] = why
        return sel

    def plan(self, dry=False):
        """Per column: lanes OWED (closure differs from the admitted record), REUSED, and the GAPS (owed with no
        box or runner), from the same selection each column's job makes, over the supported surface. A refresh
        keeps the plan it started with (it decides which jobs run); --plan (dry) recomputes it and saves
        nothing."""
        tree = self.source_tree()
        scope = self.fix_scope(self.resolve_scope(tree), dry)
        plans = {}
        for c in self.columns:
            col = self.state.col(c)
            if col.get("plan") and not dry:
                plans[c] = col["plan"]
                continue
            plans[c] = self.plan_column(c, tree)
            if not dry:
                col["plan"] = plans[c]
                if col.get("phase") == "new":
                    if plans[c].get("gap"):
                        col.update(phase="gap", why=plans[c]["gap"])
                    else:
                        col["phase"] = "nothing" if plans[c]["record"] == 0 else "planned"
                self.state.save()
        self.say("SCOPE columns %s x lanes %s%s" % (
            ",".join(scope["columns"]), scope["lanes"],
            " (NARROWED from the supported surface %s x %s: %s)" % (",".join(scope["surface_columns"]), SURFACE_SCOPE,
                                                                    scope["reason"]) if scope["narrowed"] else
            " (the supported surface)"))
        gaps = []
        for c in self.columns:
            p = plans[c]
            extra = "".join((" (known-bad %d)" % p["known_bad"] if p.get("known_bad") else "",
                             " (n/a %d)" % p["not_applicable"] if p.get("not_applicable") else "",
                             " NARROWED %d owed lane(s) out of scope" % p["narrowed"] if p.get("narrowed") else ""))
            if p.get("gap"):
                gaps.append("%s=%d" % (c, p["record"]))
                self.say("PLAN %-6s GAP %d owed lane(s) not recorded / REUSED %d of %d: %s%s"
                         % (c, p["record"], p["reused"], p["total"], p["gap"], extra))
            elif p["record"] == 0:
                self.say("PLAN %-6s NOTHING-CHANGED (REUSED %d of %d)%s" % (c, p["reused"], p["total"], extra))
            else:
                self.say("PLAN %-6s REUSED %d / RECORD %d of %d%s" % (c, p["reused"], p["record"], p["total"], extra))
        for c in scope["no_runner_columns"]:
            gaps.append("%s=?" % c)
            self.say("PLAN %-6s GAP (owed lanes unknown): %s" % (c, self.available(c)[1]))
        for c in scope["dropped_columns"]:
            self.say("PLAN %-6s NARROWED out of this refresh: %s" % (c, scope["reason"]))
        self.say("GAPS %s" % (" ".join(gaps) if gaps else "none"))
        return plans

    def gaps(self):
        """[(column, owed, why)] of the saved state: owed lanes no box or runner records."""
        out = []
        for c in self.columns:
            col = self.state.col(c)
            if col.get("phase") == "gap":
                out.append((c, (col.get("plan") or {}).get("record"), col.get("why", "")))
        for c in (self.state.d.get("scope") or {}).get("no_runner_columns", []):
            out.append((c, None, self.available(c)[1]))
        return out

    def owes(self, c):
        """Column c is requested, owes lanes and has a runner."""
        if c not in self.columns:
            return False
        col = self.state.col(c)
        return bool((col.get("plan") or {}).get("record")) and col.get("phase") != "gap"

    def cpu_job_needed(self):
        """The cpu job records the cpu column and, with a PTX set, the PTX column."""
        return self.owes("cpu") or self.owes("ptx")

    def nvidia_needed(self):
        """The nvidia job also builds the tree the cpu job records in."""
        own = self.owes("nvidia")
        for_cpu = self.cpu_job_needed()
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
                                 "--env", "SCOPE=%s" % self.box_scope(), "tools/identity_box_job.sh"),
                        "submitting the NVIDIA job", log=self.log("nvidia-submit.log"))
        tag, job = parse_submit(out)
        if not tag:
            raise Refused("the NVIDIA submit printed no persisted tag; see %s" % self.log("nvidia-submit.log"))
        col.update(phase="submitted", tag=tag, job=job, submitted=utc())
        self.state.save()
        self.say("SUBMITTED nvidia %s tag %s" % (job, tag))

    def submit_cpu(self):
        """In the tree the finished nvidia job built (HEAD == <sha>); no sync, so nothing moves under it. With an
        owed PTX column and a PTX set the same job records the PTX column after the cpu column."""
        col = self.state.col("cpu")
        env = ["--env", "ID_SCOPE=%s" % self.box_scope(), "--env", "EXPECT_COMMIT=%s" % self.sha]
        with_ptx = self.owes("ptx")
        if with_ptx:
            for k in PTX_ENV:
                env += ["--env", "%s=%s" % (k, self.ptx_set[k])]
        out = self.must(self.nvc("submit", self.nv_lane, "--cap", "240", "--note", "reference_refresh cpu %s" % self.sha12,
                                 *env, "tools/identity_box_job_cpu_ptx.sh"),
                        "submitting the cpu job", log=self.log("cpu-submit.log"))
        tag, job = parse_submit(out)
        if not tag:
            raise Refused("the cpu submit printed no persisted tag; see %s" % self.log("cpu-submit.log"))
        col.update(phase="submitted", tag=tag, job=job, submitted=utc(), with_ptx=with_ptx)
        if with_ptx:
            self.state.col("ptx").update(phase="with-cpu", tag=tag, job=job, submitted=utc())
        self.state.save()
        self.say("SUBMITTED cpu%s %s tag %s" % ("+ptx" if with_ptx else "", job, tag))

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
        pid = self.sh.spawn(self.amd("run", "identity-amd", res,
                                     "env SCOPE=%s bash tools/identity_box_job.sh amd" % self.box_scope()),
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
               "env", "REPEATS=1", "SCOPE=%s" % self.box_scope(), "PYTHON=%s" % self.py,
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
        riders = [c] + (["ptx"] if c == "cpu" and col.get("with_ptx") else [])
        if phase == "dead":
            for r in riders:
                self.state.col(r).update(phase="dead", why="the box died mid-job (manifest state=flushed); its partial "
                                         "results are in R2 under %s. Rerun with --resubmit %s" % (col["tag"], c))
            self.state.save()
            return True
        dest = self.home / "fetch" / c
        rc, out = self.sh.run(self.persist("fetch", "--tag", col["tag"], "--dest", str(dest), "--extract"),
                              log=self.log("%s-fetch.log" % c))
        if rc != 0:
            for r in riders:
                self.state.col(r).update(phase="failed", why="fetching %s failed; see %s"
                                         % (col["tag"], self.log("%s-fetch.log" % c)))
            self.state.save()
            return True
        for r in riders:
            self.settle(r, dest, job_ok=(phase != "failed"), part=r if c == "cpu" else None)
        return True

    def settle(self, c, tree, job_ok=True, part=None):
        """Collect the records of a finished column (box or local) into <home>/records/<c>. part splits a cpu
        job's results into its cpu and PTX halves."""
        col = self.state.col(c)
        recs, outcomes, other = find_records(tree, self.sha, part=part)
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
        elif c == "cpu" and col.get("for_ptx_only") and job_ok:
            col.update(phase="done", records=[], fetched=utc(), note="ran for the PTX column")
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
            before = self.tracked_edits(tree)
            env = dict(PYTHON=self.py, EVIDENCE=str(self.home / "admit"), REF_TAG="ref-%s" % self.sha12)
            log = self.log("admit.log")
            rc, out = self.sh.run(["bash", "tools/admit_identity_columns.sh", "--build-host", *dirs], cwd=str(tree),
                                  log=log, env=env)
            paths = parse_admit_next(out)
            if rc != 0 or not paths:
                restored, left = self.restore_generated(tree, before)
                a.update(phase="refused", why="admission refused (rc=%s); see %s" % (rc, log),
                         restored=restored, tracked_edits_left=left)
                self.state.save()
                raise Refused(a["why"] + ("; restored %d generated pointer file(s)" % len(restored) if restored else "")
                              + ("; tracked edits LEFT in %s: %s" % (tree, ", ".join(left[:6])) if left else ""))
            dest = next((x for x in paths if x.startswith(ADMISSIONS)), None)
            if dest is None:
                a.update(phase="refused", why="the admission named no %s directory; see %s" % (ADMISSIONS, log))
                self.state.save()
                raise Refused(a["why"])
            self.write_readme(Path(tree) / dest / "README.md")
            self.must(["git", "add", "--", *paths], "staging the admission", cwd=tree)
            cols = ",".join(c for c in self.columns if self.state.col(c).get("records"))
            gaps = self.gaps()
            msg = ("Admit identity columns at %s (%s) via tools/reference_refresh.sh\n\nRecords: %s.\n%s"
                   % (self.sha12, cols, ", ".join(dirs),
                      "GAPS (owed, not recorded): %s.\n" % ", ".join("%s %s" % (c, "?" if o is None else o)
                                                                       for c, o, _ in gaps) if gaps else ""))
            self.must(["git", "commit", "-q", "-m", msg], "committing the admission", cwd=tree)
            rc, head = self.git("rev-parse", "HEAD", cwd=tree)
            a.update(phase="committed", commit=head.strip(), branch=branch)
            self.state.save()
        self.must(["git", "push", "-q", "-u", "origin", "%s:%s" % (branch, branch)], "pushing %s" % branch, cwd=tree,
                  log=self.log("push.log"))
        a.update(phase="pushed", pushed=utc())
        self.state.save()
        self.say("ADMITTED %s; pushed %s: merge it into main (the orchestrator)" % (a["commit"][:12], branch))

    def readme_section(self):
        """The refresh's scope, per-column counts, gaps and superseded commits, for the admission README."""
        sc = self.state.d.get("scope") or {}
        L = ["", "## Reference refresh scope", "",
             "Recorded by `tools/reference_refresh.sh` at `%s`." % self.sha]
        sup = self.state.d.get("supersedes") or []
        L.append("Superseded pending refreshes (never admitted): %s."
                 % (", ".join("`%s`" % x[:12] for x in sup) if sup else "none"))
        L.append("Scope: columns `%s` x lanes `%s`%s." % (
            ",".join(sc.get("columns", self.columns)), sc.get("lanes", SURFACE_SCOPE),
            "" if not sc.get("narrowed") else ", NARROWED from the supported surface (`%s` x `%s`); reason: %s"
            % (",".join(sc.get("surface_columns", [])), SURFACE_SCOPE, sc.get("reason"))))
        L += ["", "| column | owed | reused | recorded | narrowed out | gap |", "|---|---|---|---|---|---|"]
        for c in self.columns:
            col = self.state.col(c)
            p = col.get("plan") or {}
            gap = col.get("phase") == "gap"
            recs = col.get("records") or []
            recorded = "%d record file(s)" % len(recs) if recs else ("-" if gap else "0")
            L.append("| %s | %s | %s | %s | %s | %s |" % (c, p.get("owed", p.get("record")), p.get("reused"),
                                                         recorded, p.get("narrowed", 0), col.get("why", "") if gap else ""))
        for c in sc.get("dropped_columns", []):
            L.append("| %s | - | - | - | column narrowed out | |" % c)
        gaps = self.gaps()
        L += ["", "GAPS (owed lanes no box or column recorded; the table is not current for them at `%s`): %s."
              % (self.sha12, "; ".join("%s %s: %s" % (c, "?" if o is None else o, w) for c, o, w in gaps)
                 if gaps else "none")]
        return "\n".join(L) + "\n"

    def write_readme(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(self.readme_section())

    def tracked_edits(self, tree):
        """Tracked paths of TREE that differ from its HEAD (staged or not); None when git cannot say."""
        rc, out = self.git("diff", "--name-only", "-z", "HEAD", cwd=tree)
        if rc != 0:
            return None
        return sorted(p for p in out.split("\0") if p.strip())

    def restore_generated(self, tree, before):
        """After a refused admission: put back the ADMIT_GENERATED files this run changed, nothing else (no
        reset). A file already edited before the run, or any path git could not account for, is left and
        returned in `left`. Returns (restored, left)."""
        after = self.tracked_edits(tree)
        if after is None:
            return [], ["(git diff failed: nothing restored)"]
        if before is None:
            return [], list(after)   # the run's own changes cannot be told apart: restore nothing
        new = [p for p in after if p not in set(before)]
        ours = [p for p in new if ADMIT_GENERATED.match(p)]
        left = [p for p in after if p not in ours]
        if ours:
            rc, out = self.git("checkout", "-q", "HEAD", "--", *ours, cwd=tree)
            if rc != 0:
                return [], after
            self.say("admission refused: restored %d generated pointer file(s) in %s: %s"
                     % (len(ours), tree, ", ".join(ours)))
        return ours, left

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
        for c in self.columns:
            col = self.state.col(c)
            if col["phase"] == "gap" and not self.state.d["admit"].get("phase") and self.available(c)[0]:
                # the box or runner came up before admission: the owed lanes are recorded after all
                col.setdefault("history", []).append(dict(phase="gap", why=col.pop("why", ""), cleared=utc()))
                col["plan"].pop("gap", None)
                col["phase"] = "planned"
                self.state.save()
                self.say("GAP CLEARED %s: its box or runner is up; recording its %d owed lane(s)"
                         % (c, col["plan"]["record"]))
        need_nv, nv_for_cpu = self.nvidia_needed()
        flight = []
        for c in self.columns:
            col = self.state.col(c)
            ph = col["phase"]
            if c == "nvidia" and ph == "nothing" and need_nv:
                col.update(phase="planned", for_cpu_only=nv_for_cpu)
                ph = "planned"
            if c == "cpu" and ph == "nothing" and self.owes("ptx"):
                col.update(phase="planned", for_ptx_only=True)
                ph = "planned"
            if ph in ("nothing", "done", "failed", "dead", "blocked", "gap"):
                continue
            if c == "ptx":
                # the PTX column rides on the cpu job (submit_cpu moves it to with-cpu; check_job settles it)
                cpu = self.state.col("cpu")
                if ph == "planned" and cpu["phase"] in ("failed", "dead", "blocked", "gap"):
                    col.update(phase="failed", why="the cpu job it records in did not finish (%s)" % cpu["phase"])
                    self.state.save()
                elif ph == "planned" and cpu["phase"] == "done":
                    col.update(phase="failed", why="the cpu job finished without the PTX column; --resubmit cpu")
                    self.state.save()
                elif ph in ("planned", "with-cpu"):
                    flight.append(c)
                else:
                    raise Refused("unknown phase %s for %s" % (ph, c))
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
                self.amd_tag()
                ph = self.state.col(c)["phase"]
                if ph == "launching":
                    flight.append(c)
                if ph != "submitted":
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
        if self.state.d.get("superseded_by"):
            raise Refused("the refresh of %s was superseded by %s (%s): run that one"
                          % (self.sha12, self.state.d["superseded_by"][:12], self.state.d.get("superseded")))
        t_plan = time.monotonic()
        self.plan()
        t_wait = time.monotonic()
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
        t_admit = time.monotonic()
        self.admit()
        self.write_waste(plan_s=t_wait - t_plan, wait_s=t_admit - t_wait, admit_s=time.monotonic() - t_admit)
        gaps = self.gaps()
        for c, owed, why in gaps:
            self.say("GAP %s: %s owed lane(s) not recorded: %s" % (c, "?" if owed is None else owed, why))
        if gaps:
            self.say("GAPS %s: the table is not current for these columns at %s; bring the box or column up and "
                     "refresh again" % (" ".join("%s=%s" % (c, "?" if o is None else o) for c, o, _ in gaps),
                                         self.sha12))
            return 4
        return 0

    def write_waste(self, plan_s, wait_s, admit_s):
        """waste.json beside the state (docs/release/RELEASE_PLAN.md): lanes reused vs recorded per column,
        and this invocation's seconds planning, waiting on boxes and admitting. Minimal by design."""
        cols = {}
        for c in self.columns:
            p = self.state.col(c).get("plan") or {}
            cols[c] = dict(reused=p.get("reused"), recorded=p.get("record"))
        waste = dict(schema="mojolearn.refresh-waste.v1", sha=self.sha, columns=cols,
                     lanes_reused=sum(v["reused"] or 0 for v in cols.values()),
                     lanes_recorded=sum(v["recorded"] or 0 for v in cols.values()),
                     box_wait_seconds=round(wait_s, 1), admission_seconds=round(admit_s, 1),
                     steps=dict(plan=round(plan_s, 1), wait=round(wait_s, 1), admit=round(admit_s, 1)))
        try:
            path = self.state.path.parent / "waste.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(waste, indent=1, sort_keys=True) + "\n")
            os.replace(tmp, path)
            self.say("WASTE lanes reused %d / recorded %d; box wait %.0f s, admission %.0f s (%s)"
                     % (waste["lanes_reused"], waste["lanes_recorded"], wait_s, admit_s, path))
        except OSError as exc:
            self.say("waste.json not written: %s" % exc)
        return waste

    def status(self):
        sc = self.state.d.get("scope") or {}
        if sc:
            self.say("scope  columns %s x lanes %s%s" % (",".join(sc.get("columns", [])), sc.get("lanes"),
                                                         " NARROWED: %s" % sc.get("reason") if sc.get("narrowed") else ""))
        if self.state.d.get("superseded_by"):
            self.say("SUPERSEDED by %s at %s" % (self.state.d["superseded_by"][:12], self.state.d.get("superseded")))
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
        col.pop("with_ptx", None)
        if c == "cpu" and "ptx" in self.state.d["columns"] \
                and self.state.col("ptx").get("phase") in ("with-cpu", "dead", "failed"):
            p = self.state.col("ptx")
            p.setdefault("history", []).append(dict(phase=p.get("phase"), tag=p.get("tag"), forgotten=utc()))
            for k in ("tag", "job", "why", "records"):
                p.pop(k, None)
            p["phase"] = "planned"
        self.state.save()
        self.say("FORGOT %s's job %s (kept in history); it is submitted again" % (c, old.get("tag")))


def parse_columns(text):
    """The asked columns in COLUMNS order, or None for the default (the table's columns)."""
    if not (text or "").strip():
        return None
    cols = [c.strip() for c in text.split(",") if c.strip()]
    bad = [c for c in cols if c not in COLUMNS]
    if bad or not cols:
        raise SystemExit("reference_refresh: columns are a comma list of %s (got %s)" % (",".join(COLUMNS), text))
    return [c for c in COLUMNS if c in cols]


# ------------------------------------------------------------------ coalescing
def home_root(env):
    return Path(env.get("REFRESH_HOME") or Path.home() / "mojolearn-evidence/reference-refresh")


def is_ancestor(sh, a, b):
    """a == b or a is an ancestor of b (git merge-base --is-ancestor)."""
    if a == b:
        return True
    rc, _ = sh.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", a, b])
    return rc == 0


def finished(d):
    return (d.get("admit") or {}).get("phase") in ("pushed", "nothing")


def coalesce(sha, env=None, sh=None, out=say, dry=False):
    """The commit a refresh of `sha` targets, and the older pending refreshes it supersedes.

    A refresh always targets the newest selected commit: `sha`, or a pending (not yet admitted) refresh of a
    commit that contains it. Every pending refresh of an ancestor of the target is SUPERSEDED: its state keeps
    superseded_by, its jobs are not waited for and its records are never admitted (they stay under its home
    dir). The target then records each lane whose closure differs from its admitted record once, at the
    target. Refused: a request older than a refresh that already finished (nothing to do), and a pushed
    admission that is not merged into the target (merge ref/<sha12> first, or its lanes would be recorded
    again). dry (--plan) prints the decision and saves nothing. Returns (target, [superseded shas])."""
    env = dict(os.environ if env is None else env)
    sh = sh or Shell()
    states = []
    for p in sorted(home_root(env).glob("*/state.json")):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and d.get("sha"):
            states.append((p, d))
    own = next((d for _, d in states if d["sha"] == sha), None)
    if own and own.get("superseded_by"):
        raise Refused("the refresh of %s was superseded by %s (%s): run that one, or no sha (the tip of origin/main)"
                      % (sha[:12], own["superseded_by"][:12], own.get("superseded")))
    for _, d in states:
        if d["sha"] != sha and finished(d) and is_ancestor(sh, sha, d["sha"]):
            raise Refused("the refresh of %s, which contains %s, already finished (admit %s): nothing to do for the "
                          "older commit" % (d["sha"][:12], sha[:12], d["admit"]["phase"]))
    pending = [(p, d) for p, d in states if not d.get("superseded_by") and not finished(d)]
    target = sha
    for _, d in pending:
        if is_ancestor(sh, target, d["sha"]):
            target = d["sha"]
    if target != sha:
        out("COALESCED %s into the pending refresh of the newer %s" % (sha[:12], target[:12]))
    for _, d in states:
        a = d.get("admit") or {}
        if a.get("phase") == "pushed" and a.get("commit") and is_ancestor(sh, d["sha"], target) \
                and not is_ancestor(sh, a["commit"], target):
            raise Refused("the admission %s of %s (branch %s) is pushed but not merged into %s: merge it first, or "
                          "its lanes are recorded again" % (a["commit"][:12], d["sha"][:12],
                                                            a.get("branch") or "ref/" + d["sha"][:12], target[:12]))
    superseded = []
    for p, d in pending:
        if d["sha"] == target:
            continue
        if not is_ancestor(sh, d["sha"], target):
            out("PENDING %s stays pending: it is not an ancestor of %s (refresh a commit that contains both)"
                % (d["sha"][:12], target[:12]))
            continue
        superseded.append(d["sha"])
        jobs = ["%s %s %s" % (c, col.get("phase"), col.get("tag") or col.get("pid") or "-")
                for c, col in sorted((d.get("columns") or {}).items())
                if col.get("phase") in ("submitted", "launching", "running-local", "with-cpu")]
        out("%s %s by %s%s" % ("WOULD SUPERSEDE" if dry else "SUPERSEDED", d["sha"][:12], target[:12],
                                       "; not waited for: %s" % ", ".join(jobs) if jobs else ""))
        if not dry:
            d.update(superseded_by=target, superseded=utc(), superseded_jobs=jobs)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
            os.replace(tmp, p)
    return target, superseded


def main(argv=None, env=None, sh=None):
    ap = argparse.ArgumentParser(prog="reference_refresh.sh", description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", action="store_true", help="print lanes owed / reused and the GAPS per column; submit nothing")
    ap.add_argument("--status", action="store_true", help="print the saved state")
    ap.add_argument("--no-wait", action="store_true", help="submit what is owed and exit")
    ap.add_argument("--resubmit", default="", help="forget this column's job (a dead box), then continue")
    ap.add_argument("--extend-amd", type=int, default=0, metavar="MIN",
                    help="while the AMD job runs, move the droplet's lease to now + MIN at each poll")
    ap.add_argument("--scope", default=None, choices=(SURFACE_SCOPE,) + NARROW_SCOPES,
                    help="narrow the lanes (default: all, the supported surface); needs --reason")
    ap.add_argument("--reason", default="", help="why the scope is narrowed (kept in the state and the README)")
    ap.add_argument("sha", nargs="?", default="origin/main", help="the merged commit (default: the tip of origin/main)")
    ap.add_argument("columns", nargs="?", default="")
    a = ap.parse_args(argv)
    cols = parse_columns(a.columns)
    sh = sh or Shell()
    rc, full = sh.run(["git", "-C", str(ROOT), "rev-parse", "--verify", a.sha + "^{commit}"])
    if rc != 0:
        raise SystemExit("reference_refresh: %s is not a commit here (git fetch first)" % a.sha)
    try:
        sha = full.strip()
        supersedes = []
        if not a.status:
            sha, supersedes = coalesce(sha, env=env, sh=sh, dry=a.plan)
        r = Refresh(sha, cols, env=env, sh=sh, lane_scope=a.scope, reason=a.reason)
        r.extend_amd = a.extend_amd
        if a.status:
            r.status()
            return 0
        if a.plan:
            r.plan(dry=True)
            return 0
        if supersedes:
            r.state.d["supersedes"] = sorted(set(r.state.d.get("supersedes", [])) | set(supersedes))
            r.state.save()
        if a.resubmit:
            if a.resubmit not in r.columns:
                raise Refused("--resubmit %s is not among the columns %s" % (a.resubmit, ",".join(r.columns)))
            r.resubmit(a.resubmit)
        return r.run(wait=not a.no_wait)
    except Refused as exc:
        say("REFUSED %s" % exc)
        return 2


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: sys.exit(130))
    sys.exit(main())
