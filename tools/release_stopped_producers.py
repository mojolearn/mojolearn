"""Close the begun compile runs of a GitHub Linux build leg whose producer stopped.

0.8.64: the sm_89 leg at 40c11f7 failed on GitHub (run 41364664603, memory
guard). Its 124 begun compile runs stayed open in the readiness ledger, so
every later `release.py linux-builds` stopped with "resume or record the
existing run ...; another launch would duplicate work" until they were
interrupted by hand. The same happens when release.py is killed after
`begin` but before the leg dispatches.

For each open begun compile run of one leg this module finds its producer and
reads its state from GitHub:

  * the leg's launch provenance (<leg>.provenance.json) names the runs it
    began; the GitHub run is read from GHA/run-id.json, else GHA/run.json, else
    the collector log's "dispatched <url>" line, else the dispatch tag
    (GHA/dispatch.json or GHA/dispatch-intent.json) in the run listing, else
    the workflow's run title for this set and commit since the launch;
  * runs that no launch provenance names were begun by a release.py that died
    before it spawned the leg (after a grace period, with no matching run in
    the listing since they were begun).

States: "running" (queued or in progress: real duplicate work, refuse),
"succeeded" (record its output, never interrupt), "stopped" (concluded
failure, cancelled, timed out or startup failure), "never-dispatched" (no run
exists for the leg since the begin, the listing covering that window), and
"unknown" (anything this cannot prove: refuse). Only "stopped" and
"never-dispatched" runs are interrupted, with the gh run JSON, the failed-log
excerpt or the listing retained as evidence (producer_stopped, qualification
claim false). This module never dispatches, cancels or kills anything.
"""
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

#: Run statuses that can still produce the leg's output.
ACTIVE = frozenset({"queued", "in_progress", "waiting", "pending", "requested"})
#: Conclusions of a run that stopped without producing a complete set.
STOPPED = frozenset({"failure", "cancelled", "timed_out", "startup_failure"})
#: Runs that no launch names are judged only after this long: begin -> spawn
#: takes seconds, so a younger run may belong to a launch in progress.
GRACE_SECONDS = 600
#: How many dispatch runs one listing reads; the window must be covered.
LISTING_LIMIT = 200
#: Clock skew allowed between the ledger stamp and GitHub's createdAt.
SKEW_SECONDS = 120
#: Lines of `gh run view --log-failed` retained as the failure excerpt.
EXCERPT_LINES = 200
RUN_FIELDS = "databaseId,displayTitle,status,conclusion,headSha,url,createdAt,updatedAt,jobs"
LIST_FIELDS = "databaseId,displayTitle,status,conclusion,headSha,url,createdAt"
RUN_URL = re.compile(r"/actions/runs/(\d+)")


def is_github_leg(leg):
    return any("release_github_build.py" in str(part) for part in (leg.command or []))


def parse_time(value):
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def open_runs(document, names):
    """{run_id: run} of the active begun run (the latest, with no attempt or
    interruption) of each requirement in `names`: what begin refuses on."""
    closed = ({a["run_id"] for a in document.get("attempts", [])}
              | {i["run_id"] for i in document.get("interruptions", [])})
    latest = {}
    for run in document.get("runs", []):
        if run.get("requirement") in names:
            latest[run["requirement"]] = run
    return {r["id"]: r for r in latest.values() if r["id"] not in closed}


def recorded_run(leg):
    """(GitHub run id or None, dispatch tag or None, where the id came from)."""
    gha = Path(leg.out_dir) / "GHA"
    tag = None
    for name in ("dispatch.json", "dispatch-intent.json"):
        tag = tag or (read_json(gha / name) or {}).get("tag")
    found = (read_json(gha / "run-id.json") or {}).get("databaseId")
    if isinstance(found, int) or (isinstance(found, str) and found.isdigit()):
        return int(found), tag, str(gha / "run-id.json")
    match = RUN_URL.search(str((read_json(gha / "run.json") or {}).get("url") or ""))
    if match:
        return int(match.group(1)), tag, str(gha / "run.json")
    try:
        lines = Path(leg.log).read_text(errors="replace").splitlines()
    except OSError:
        lines = []
    for line in reversed(lines):
        if "gha-build] dispatched " in line or "gha-build] attached to run " in line:
            match = RUN_URL.search(line)
            if match:
                return int(match.group(1)), tag, str(leg.log)
    return None, tag, None


def title_matches(title, arch, commit, tag):
    title = title or ""
    if tag:
        return f"[{tag}]" in title
    return re.match(rf"release-linux-build {re.escape(arch)}(?: host)? @ {re.escape(commit)} \[", title) is not None


def run_state(view):
    status, conclusion = view.get("status"), view.get("conclusion")
    if status in ACTIVE:
        return "running"
    if status == "completed" and conclusion == "success":
        return "succeeded"
    if status == "completed" and conclusion in STOPPED:
        return "stopped"
    return "unknown"


class Producers:
    """Classify and close one leg's open begun compile runs. `gh(*args)` returns
    (exit code, stdout); `now` is a timezone-aware datetime."""

    def __init__(self, gh, workflow, now=None):
        self.gh, self.workflow = gh, workflow
        self.now = now or dt.datetime.now(dt.timezone.utc)
        self._listing = None

    def view(self, run_id):
        rc, out = self.gh("run", "view", str(run_id), "--json", RUN_FIELDS)
        if rc != 0:
            raise LookupError(f"gh run view {run_id} exited {rc}")
        return json.loads(out)

    def failed_log(self, run_id):
        rc, out = self.gh("run", "view", str(run_id), "--log-failed")
        lines = [line[:400] for line in (out or "").splitlines()][-EXCERPT_LINES:]
        head = f"# gh run view {run_id} --log-failed (exit {rc}); last {len(lines)} lines\n"
        return head + "\n".join(lines) + "\n"

    def listing(self):
        if self._listing is None:
            rc, out = self.gh("run", "list", "--workflow", self.workflow, "--event", "workflow_dispatch",
                              "--limit", str(LISTING_LIMIT), "--json", LIST_FIELDS)
            if rc != 0:
                raise LookupError(f"gh run list exited {rc}")
            self._listing = json.loads(out or "[]")
        return self._listing

    def search(self, arch, commit, tag, since):
        """(matching runs, whether the listing covers every run since `since`)."""
        rows = self.listing()
        floor = since - dt.timedelta(seconds=SKEW_SECONDS)
        matches = [r for r in rows if title_matches(r.get("displayTitle"), arch, commit, tag)
                   and (tag or (parse_time(r.get("createdAt")) or floor) >= floor)]
        created = [t for t in (parse_time(r.get("createdAt")) for r in rows) if t]
        covered = len(rows) < LISTING_LIMIT or bool(created and min(created) <= floor)
        return matches, covered

    def from_run(self, group, run_id, arch, commit, tag):
        view = self.view(run_id)
        if not title_matches(view.get("displayTitle"), arch, commit, tag):
            return dict(group, state="unknown", reason=f"GitHub run {run_id} ({view.get('displayTitle')}) "
                        f"is not this leg's {arch} @ {commit[:12]} build")
        state = run_state(view)
        group = dict(group, state=state, github_run=run_id, url=view.get("url"),
                     reason=f"GitHub run {run_id} {view.get('status')}/{view.get('conclusion')} ({view.get('url')})")
        group["evidence"] = {"gh-run.json": json.dumps(view, indent=1, sort_keys=True) + "\n"}
        if state == "stopped":
            group["evidence"]["failed-log.txt"] = self.failed_log(run_id)
        return group

    def classify(self, leg, document, names, commit):
        """Groups {state, run_ids, reason, evidence, github_run, url} over the
        leg's open begun runs; an empty list when none is open."""
        runs = open_runs(document, names)
        if not runs:
            return []
        prov = read_json(leg.provenance_file) or {}
        launch = (prov.get("compile_requirements") or {}).get("runs") or {}
        launched = {row.get("run_id") for row in launch.values()}
        commit = prov.get("source_commit") or commit
        own = sorted(i for i in runs if i in launched)
        orphans = sorted(i for i in runs if i not in launched)
        provenance = {"launch-provenance.json": json.dumps(prov, indent=1, sort_keys=True) + "\n"}
        groups = []
        if leg.running():
            return [dict(state="running", run_ids=sorted(runs), evidence={},
                         reason=f"the leg's collector is running (pid {leg.pid()})")]
        if own:
            group = dict(run_ids=own, evidence={}, launched=True)
            run_id, tag, where = recorded_run(leg)
            try:
                if run_id is not None:
                    group = self.from_run(group, run_id, leg.arch, commit, tag)
                    group["evidence"].update(provenance)
                    group["found_in"] = where
                else:
                    since = parse_time(prov.get("launched_at")) or min(
                        parse_time(runs[i].get("started_at")) or self.now for i in own)
                    matches, covered = self.search(leg.arch, commit, tag, since)
                    if len(matches) == 1:
                        group = self.from_run(group, matches[0]["databaseId"], leg.arch, commit, tag)
                        group["evidence"].update(provenance)
                    elif matches:
                        group.update(state="unknown", reason=f"{len(matches)} GitHub runs match the leg's launch")
                    elif not covered:
                        group.update(state="unknown", reason="the GitHub run listing does not reach back to the launch")
                    else:
                        group.update(state="never-dispatched", reason=(
                            f"no {self.workflow} run for {leg.arch} @ {commit[:12]}"
                            + (f" tagged {tag}" if tag else "") + f" since the launch at {since.isoformat()}"))
                        group["evidence"] = dict(provenance, **{"gh-run-list.json": json.dumps(
                            self.listing(), indent=1, sort_keys=True) + "\n"})
            except (LookupError, ValueError, KeyError, TypeError) as error:
                group.update(state="unknown", reason=f"GitHub state unreadable ({error})")
            groups.append(group)
        if orphans:
            group = dict(run_ids=orphans, evidence={}, launched=False)
            started = [parse_time(runs[i].get("started_at")) for i in orphans]
            if not all(started):
                group.update(state="unknown", reason="a begun run carries no start time")
            elif (self.now - max(started)).total_seconds() < GRACE_SECONDS:
                group.update(state="unknown", reason="begun moments ago; its launch may still be in progress")
            else:
                try:
                    matches, covered = self.search(leg.arch, commit, None, min(started))
                except (LookupError, ValueError) as error:
                    matches, covered = None, False
                    group.update(state="unknown", reason=f"GitHub run listing unreadable ({error})")
                if matches:
                    group.update(state="unknown", reason=(
                        f"no launch provenance names them, yet {len(matches)} GitHub run(s) match since the begin"))
                elif matches is not None and not covered:
                    group.update(state="unknown", reason="the GitHub run listing does not reach back to the begin")
                elif matches is not None:
                    group.update(state="never-dispatched", reason=(
                        f"begun at {min(started).isoformat()} by a launch that never spawned the leg (no launch "
                        f"provenance names them) and no {self.workflow} run for {leg.arch} @ {commit[:12]} since"))
                    group["evidence"] = dict(provenance, **{"gh-run-list.json": json.dumps(
                        self.listing(), indent=1, sort_keys=True) + "\n"})
            groups.append(group)
        return groups


def retain(group, root, leg_name):
    """Write a group's evidence under root/<identity>/ and return the paths."""
    identity = hashlib.sha256(json.dumps([leg_name, group["run_ids"], group["state"], group.get("github_run"),
                                          sorted(group["evidence"].items())]).encode()).hexdigest()[:24]
    dest = Path(root) / identity
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, text in sorted(group["evidence"].items()):
        path = dest / name
        path.write_text(text)
        paths.append(path)
    summary = dict(leg=leg_name, state=group["state"], reason=group["reason"], run_ids=group["run_ids"],
                   github_run=group.get("github_run"), url=group.get("url"), qualification_claim=False,
                   files=[p.name for p in paths])
    path = dest / "producer.json"
    path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return [path, *paths]


def close_stopped(ledger, groups, root, leg_name):
    """Interrupt every stopped or never-dispatched group; return the closed run ids."""
    closed = []
    for group in groups:
        if group["state"] not in ("stopped", "never-dispatched"):
            continue
        paths = retain(group, root, leg_name)
        reason = f"{leg_name}: producer {group['state']}: {group['reason']}"
        ledger.interrupt_many(group["run_ids"], paths, reason, producer_stopped=True)
        group["closed"] = True
        closed.extend(group["run_ids"])
    return closed
