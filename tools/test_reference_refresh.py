#!/usr/bin/env python3
"""Tests for tools/reference_refresh.py: the Mac-side planning, idempotency bookkeeping and fetch-state parsing.
No box, no network, no git writes: every external command goes through a fake shell."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reference_refresh as rr  # noqa: E402

SHA = "0814836ffb8f4ca6e6e9d0d58833f16d84195d25"
OTHER = "ba50061e5252b41adef5ae65dc8017cb073f887d"

# The NEXT line tools/admit_identity_columns.sh printed for the 0.8.65 NVIDIA admission (rel0865/admit-nv-fix.log).
ADMIT_NEXT = """PTX rc=0 ptx_records=13 divergent_parts=0
NEXT (after SELFTEST rc=0):
  git add -- bench/results/reference_admissions/2026-11-04-nv-fix0865 python/mojolearn/verify_reference/table.json \
bench/results/identity_break/retained/04eb/nvidia-rtx-sm_89.lanes-eaeec399.identical.json \
bench/results/identity_break/retained/04eb/nvidia-rtx-sm_89.lanes-eaeec399.provenance.txt \
&& git commit -m 'verify_reference: table regenerated from 0814836ff columns' && git push
  Re-record portable model files only if their admitted model/batch hashes changed.
"""


def table_sel(n, total=428):
    return "# diffing 12 path(s)\n# CHANGED-LANES %d of %d (routine); every other row is kept\n%s\n" % (
        n, total, ",".join("l%d" % i for i in range(n)))


def class_sel(reused, record, total=428, bad=0):
    return "# REUSE-SUMMARY class=x REUSED %d RECORD %d KNOWN-BAD %d of %d (routine)\n# CHANGED-LANES %d of %d\n" % (
        reused, record, bad, total, record, total)


def write_record(d, name, commit=SHA, log="RECORDED %s rc=0 admissible=yes\n"):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    (d / (name + ".identical.json")).write_text(json.dumps(dict(commit=commit, cells={})))
    (d / (name + ".provenance.txt")).write_text("commit=%s\n" % commit)
    (d / "record.log").write_text(log % name if "%s" in log else log)


def selection_column(cmd):
    """The column a changed-lanes command plans: its --class, else its backend (table mode: nvidia, amd)."""
    if "--class" in cmd:
        return cmd[cmd.index("--class") + 1]
    return dict(cuda="nvidia", hip="amd")[cmd[cmd.index("--backend") + 1]]


def write_table(path, cols=rr.COLUMNS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(cells={"lane-a/base": {"fit": {"cols": {c: 0 for c in cols}, "ref": "x"}}},
                                    records=[])))


class FakeShell:
    """Scripted stand-in for reference_refresh.Shell. `sel` maps column -> changed-lanes output; `manifest` maps
    a tag -> fetch --list output; `extract` maps a tag -> a function(dest) that lays out the fetched job dir."""

    def __init__(self, sel=None):
        self.cmds, self.spawned = [], []
        self.sel = sel or {}
        self.scope_lanes = {}      # scope -> the lane list `identity_columns.py lanes --scope X` prints
        self.history = []          # commits oldest first: merge-base --is-ancestor follows this order
        self.resolve = {}          # rev-parse --verify <name> -> sha
        self.manifest, self.extract = {}, {}
        self.next_tag = 0
        self.admit_out = ADMIT_NEXT
        self.admit_rc = 0
        self.alive_pids = set()

    def _new_tag(self, prefix):
        self.next_tag += 1
        return "%s-20261104T0000%02dZ-abc%03d" % (prefix, self.next_tag, self.next_tag)

    def run(self, cmd, cwd=None, log=None, env=None, timeout=None):
        cmd = [str(c) for c in cmd]
        self.cmds.append(cmd)
        j = " ".join(cmd)
        if cmd[0] == "git" or (len(cmd) > 2 and cmd[0] == "git"):
            if "--is-ancestor" in cmd:
                a, b = cmd[-2], cmd[-1]
                if a in self.history and b in self.history:
                    return (0 if self.history.index(a) <= self.history.index(b) else 1), ""
                return 0, ""
            if "rev-parse" in cmd and "--verify" in cmd:
                name = cmd[-1].replace("^{commit}", "")
                return 0, self.resolve.get(name, SHA) + "\n"
            if "rev-parse" in cmd and "HEAD" in cmd and "--abbrev-ref" in cmd:
                return 0, "ref/%s\n" % SHA[:12]
            if "rev-parse" in cmd and "HEAD" in cmd:
                return 0, SHA + "\n"
            if "status" in cmd:
                return 0, ""
            return 0, ""
        if "changed-lanes" in cmd:
            col = selection_column(cmd)
            text = self.sel[col]
            sel = rr.parse_selection(text)
            if sel and "--summary-json" in cmd:
                Path(cmd[cmd.index("--summary-json") + 1]).write_text(json.dumps(dict(
                    record={"l%d" % i: "closure changed" for i in range(sel["record"])})))
            return 0, text
        if "identity_columns.py" in j and " lanes " in j:
            scope = cmd[cmd.index("--scope") + 1]
            return 0, ",".join(self.scope_lanes[scope]) + "\n"
        if "nvidia_central.sh" in j and " submit " in j:
            script = cmd[-1]
            tag = self._new_tag("nvc-identity")
            return 0, "box_persist: job %s persists to R2 measurements/box-jobs/2026-11-04/%s/ (fetch: x)\nnvc1-%04d\n" % (
                tag, tag, self.next_tag) + ("#%s" % script)
        if "nvidia_central.sh" in j or "do_amd_steward.sh" in j:
            return 0, ""
        if "box_persist.py" in j and "--list" in cmd:
            tag = cmd[cmd.index("--tag") + 1]
            return (0 if tag in self.manifest else 1), self.manifest.get(tag, "MANIFEST missing (http 404)\nNOTHING LANDED\n")
        if "box_persist.py" in j and "--extract" in cmd:
            tag = cmd[cmd.index("--tag") + 1]
            dest = cmd[cmd.index("--dest") + 1]
            self.extract[tag](Path(dest) / "final")
            return 0, "VERIFIED final.tgz\nEXTRACTED final.tgz\n"
        if "admit_identity_columns.sh" in j:
            return self.admit_rc, self.admit_out
        return 0, ""

    def spawn(self, cmd, cwd, log, rcfile, env=None):
        self.spawned.append(([str(c) for c in cmd], log, rcfile))
        pid = 40000 + len(self.spawned)
        self.alive_pids.add(pid)
        return pid

    def alive(self, pid):
        return pid in self.alive_pids

    def sleep(self, s):
        pass

    def count(self, needle):
        return sum(1 for c in self.cmds if needle in " ".join(c))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        for d in ("src", "admit"):
            (t / d / ".git").mkdir(parents=True)
        write_table(t / "src" / rr.TABLE)
        # an NVIDIA pod and the AMD droplet are up (the box tools' local state files)
        for d in ("devpods/nvc1", "do"):
            (t / d).mkdir(parents=True)
            (t / d / "state.env").write_text("X=1\n")
        self.env = dict(REFRESH_HOME=str(t / "home"), REFRESH_PYTHON="python3", REFRESH_SRC_TREE=str(t / "src"),
                        REFRESH_ADMIT_TREE=str(t / "admit"), REFRESH_POLL_SEC="0", REFRESH_WAIT_MIN="0",
                        MOJOLEARN_DEVPOD_STATE=str(t / "devpods"), MOJOLEARN_STEWARD_DO_STATE=str(t / "do"))
        self.t = t
        self.lines = []

    def tearDown(self):
        self.tmp.cleanup()

    def refresh(self, sh, columns=None, sha=SHA, lane_scope=None, reason=None, **env):
        e = dict(self.env, **env)
        narrow = (columns and list(columns) != list(rr.COLUMNS)) or lane_scope
        if reason is None:
            reason = "test: one column" if narrow else ""
        return rr.Refresh(sha, columns, env=e, sh=sh, out=self.lines.append, lane_scope=lane_scope, reason=reason)

    def state(self, sha=SHA):
        return json.loads((self.t / "home" / sha[:12] / "state.json").read_text())


class TestParsing(unittest.TestCase):
    def test_selection_both_modes(self):
        self.assertEqual(rr.parse_selection(table_sel(9)), dict(record=9, reused=419, known_bad=0, total=428))
        self.assertEqual(rr.parse_selection(class_sel(416, 9, bad=3)), dict(record=9, reused=416, known_bad=3, total=428))
        self.assertIsNone(rr.parse_selection("Traceback ...\n"))

    def test_submit(self):
        tag, job = rr.parse_submit("box_persist: job nvc-identity-20261104T043823Z-74d7ff persists to R2 x\nnvc1-0007\n")
        self.assertEqual((tag, job), ("nvc-identity-20261104T043823Z-74d7ff", "nvc1-0007"))
        self.assertEqual(rr.parse_submit("nvidia_central: no pod\n"), (None, None))

    def test_manifest_phases(self):
        P = lambda t: rr.job_phase(rr.parse_manifest(t))  # noqa: E731
        self.assertEqual(P("MANIFEST missing (http 404): probing every sidecar\nNOTHING LANDED under x/\n"), "queued")
        self.assertEqual(P("MANIFEST state=running exit= updated=2026 uploads=2\nLANDED 00.tgz sha256=a\n"), "running")
        self.assertEqual(P("MANIFEST state=final exit=0 updated=2026 uploads=6\nLANDED final.tgz sha256=a\n"), "done")
        self.assertEqual(P("MANIFEST state=final exit=3 updated=2026 uploads=6\nLANDED final.tgz sha256=a\n"), "failed")
        self.assertEqual(P("MANIFEST state=flushed exit= updated=2026 uploads=3\n"), "dead")
        self.assertEqual(P("MANIFEST missing (http 0): probing\nLANDED final.tgz sha256=a bytes=?\n"), "done-unknown")

    def test_admit_next(self):
        paths = rr.parse_admit_next(ADMIT_NEXT)
        self.assertEqual(paths[1], "python/mojolearn/verify_reference/table.json")
        self.assertEqual(len(paths), 4)
        self.assertIsNone(rr.parse_admit_next("admit_identity_columns: refused\n"))

    def test_record_log_outcome(self):
        self.assertEqual(rr.record_log_outcome("NOTHING-CHANGED nvidia-x: no lane\n"), "nothing")
        self.assertEqual(rr.record_log_outcome("RECORDED a rc=0 admissible=yes\n"), "recorded")
        self.assertEqual(rr.record_log_outcome("RECORDED a rc=0 admissible=yes\nRECORDED b rc=1 admissible=NO\n"),
                         "inadmissible")
        self.assertIsNone(rr.record_log_outcome("still running\n"))

    def test_find_records_keeps_only_this_commit(self):
        with tempfile.TemporaryDirectory() as d:
            write_record(Path(d) / "merged-nvidia" / "column", "nvidia-rtx-sm_89.lanes-aaaa")
            write_record(Path(d) / "old" / "column", "nvidia-rtx-sm_89.lanes-bbbb", commit=OTHER)
            (Path(d) / "x.admission-preview-1.identical.json").write_text("{}")
            recs, outcomes, other = rr.find_records(d, SHA)
            self.assertEqual([p.name for p in recs], ["nvidia-rtx-sm_89.lanes-aaaa.identical.json"])
            self.assertEqual([p.name for p in other], ["nvidia-rtx-sm_89.lanes-bbbb.identical.json"])
            got = rr.collect_records(recs, Path(d) / "in")
            self.assertTrue((Path(d) / "in" / "nvidia-rtx-sm_89.lanes-aaaa.provenance.txt").is_file())
            self.assertEqual(len(got), 1)

    def test_columns(self):
        self.assertEqual(rr.parse_columns("amd,nvidia"), ["nvidia", "amd"])
        self.assertEqual(rr.parse_columns("ptx,cpu"), ["cpu", "ptx"])
        self.assertIsNone(rr.parse_columns(""))          # the default: the table's columns
        with self.assertRaises(SystemExit):
            rr.parse_columns("gpu")

    def test_table_columns(self):
        with tempfile.TemporaryDirectory() as d:
            write_table(Path(d) / "t.json", ("ptx", "nvidia", "cpu"))
            self.assertEqual(rr.table_columns(Path(d) / "t.json"), ["cpu", "nvidia", "ptx"])

    def test_find_records_splits_a_cpu_job_into_cpu_and_ptx(self):
        with tempfile.TemporaryDirectory() as d:
            write_record(Path(d) / "final" / "cpu-ptx" / "cpu", "cpu.lanes-1")
            write_record(Path(d) / "final" / "cpu-ptx" / "ptx", "nvidia-ptx-rtx-sm80.lanes-2")
            (Path(d) / "final" / "cpu-ptx" / "cpu-record.log").write_text("NOTHING-CHANGED cpu: x\n")
            recs, outs, _ = rr.find_records(d, SHA, part="cpu")
            self.assertEqual([p.name for p in recs], ["cpu.lanes-1.identical.json"])
            self.assertEqual(sorted(o for _, o in outs), ["nothing", "recorded"])
            recs, outs, _ = rr.find_records(d, SHA, part="ptx")
            self.assertEqual([p.name for p in recs], ["nvidia-ptx-rtx-sm80.lanes-2.identical.json"])
            self.assertEqual([o for _, o in outs], ["recorded"])


class TestPlan(Base):
    def test_plan_prints_counts_and_saves_nothing(self):
        sh = FakeShell(dict(nvidia=table_sel(9), amd=table_sel(0), cpu=class_sel(416, 12), ptx=class_sel(428, 0),
                            apple=class_sel(428, 0)))
        self.refresh(sh, REFRESH_APPLE_TREE=str(self.t)).plan(dry=True)
        self.assertIn("PLAN nvidia REUSED 419 / RECORD 9 of 428", self.lines)
        self.assertIn("PLAN amd    NOTHING-CHANGED (REUSED 428 of 428)", self.lines)
        self.assertIn("GAPS none", self.lines)
        self.assertFalse((self.t / "home" / SHA[:12] / "state.json").exists())
        self.assertEqual(sh.count(" submit "), 0)

    def test_waste_json_counts_lanes_per_column_and_the_seconds(self):
        sh = FakeShell(dict(nvidia=table_sel(9), amd=table_sel(0)))
        r = self.refresh(sh, ["nvidia", "amd"])
        r.plan()
        waste = r.write_waste(plan_s=1.0, wait_s=30.0, admit_s=4.0)
        saved = json.loads((self.t / "home" / SHA[:12] / "waste.json").read_text())
        self.assertEqual(saved, waste)
        self.assertEqual(saved["columns"]["nvidia"]["recorded"], 9)
        self.assertEqual(saved["lanes_recorded"], 9)
        self.assertEqual((saved["box_wait_seconds"], saved["admission_seconds"]), (30.0, 4.0))
        self.assertTrue(any(l.startswith("WASTE lanes reused") for l in self.lines))

    def test_plan_uses_each_jobs_selection(self):
        sh = FakeShell(dict(nvidia=table_sel(1), amd=table_sel(1), cpu=class_sel(1, 1), ptx=class_sel(1, 1),
                            apple=class_sel(1, 1)))
        self.refresh(sh).plan(dry=True)
        sel = {selection_column(c): c for c in sh.cmds if "changed-lanes" in c}
        self.assertEqual(sorted(sel), sorted(rr.COLUMNS))
        be = lambda c: sel[c][sel[c].index("--backend") + 1]  # noqa: E731
        self.assertNotIn("--class", sel["nvidia"])          # box_merged_verify: ONLY_CHANGED=table
        self.assertNotIn("--class", sel["amd"])
        self.assertEqual((be("nvidia"), be("amd"), be("cpu"), be("ptx"), be("apple")), ("cuda", "hip", "cpu", "cuda", "metal"))
        # every column is planned over the supported surface, never a recorder default
        self.assertEqual({c[c.index("--scope") + 1] for c in sel.values()}, {"all"})

    def test_failed_selection_refuses(self):
        sh = FakeShell(dict(nvidia="Traceback\n"))
        with self.assertRaises(rr.Refused):
            self.refresh(sh, ["nvidia"]).plan(dry=True)


class TestIdempotency(Base):
    def test_nothing_changed_submits_nothing(self):
        sh = FakeShell(dict(nvidia=table_sel(0), amd=table_sel(0), cpu=class_sel(428, 0), ptx=class_sel(428, 0),
                            apple=class_sel(428, 0)))
        self.assertEqual(self.refresh(sh).run(wait=False), 0)
        self.assertEqual(sh.count(" submit "), 0)
        self.assertEqual(sh.spawned, [])
        self.assertEqual(sh.count("admit_identity_columns"), 0)
        self.assertEqual(self.state()["admit"]["phase"], "nothing")

    def test_cpu_only_change_still_builds_the_nvidia_tree(self):
        sh = FakeShell(dict(nvidia=table_sel(0), cpu=class_sel(420, 8)))
        r = self.refresh(sh, ["nvidia", "cpu"])
        r.run(wait=False)
        st = self.state()["columns"]
        self.assertEqual(st["nvidia"]["phase"], "submitted")
        self.assertTrue(st["nvidia"]["for_cpu_only"])
        self.assertEqual(st["cpu"]["phase"], "planned")      # waits for the built tree
        self.assertEqual(sh.count("identity_box_job_cpu_ptx.sh"), 0)

    def test_crash_and_rerun_fetches_instead_of_resubmitting(self):
        sel = dict(nvidia=table_sel(3), amd=table_sel(3), cpu=class_sel(420, 8), ptx=class_sel(428, 0),
                   apple=class_sel(425, 3))
        sh = FakeShell(sel)
        apple_tree = self.t / "apple-tree"
        (apple_tree / ".git").mkdir(parents=True)
        env = dict(REFRESH_APPLE_TREE=str(apple_tree), REFRESH_APPLE_LABEL="apple-m4")
        self.assertEqual(self.refresh(sh, **env).run(wait=False), 0)
        st = self.state()["columns"]
        nv_tag = st["nvidia"]["tag"]
        self.assertEqual(st["nvidia"]["phase"], "submitted")
        self.assertEqual(st["amd"]["phase"], "launching")
        self.assertEqual(st["apple"]["phase"], "running-local")
        self.assertEqual(sh.count(" submit "), 1)
        # the sync cleaned record outputs from the pod tree first
        i_clean = next(i for i, c in enumerate(sh.cmds) if "git clean -fdq -- bench/results/identity_break" in " ".join(c))
        i_sync = next(i for i, c in enumerate(sh.cmds) if "nvidia_central.sh" in " ".join(c) and " sync " in " ".join(c))
        self.assertLess(i_clean, i_sync)

        # "crash": a new process, a new shell. The AMD launcher wrote its persisted tag; nothing landed yet.
        sh2 = FakeShell(sel)
        sh2.alive_pids = set(sh.alive_pids)
        amd_log = Path(st["amd"]["launch_log"])
        amd_log.parent.mkdir(parents=True, exist_ok=True)
        amd_log.write_text("box_persist: job identity-amd-20261104T1Z-ffffff persists to R2 x\n")
        self.assertEqual(self.refresh(sh2, **env).run(wait=False), 0)
        self.assertEqual(sh2.count(" submit "), 0)
        self.assertEqual(sh2.count("changed-lanes"), 0)      # the plan is kept, not recomputed
        st = self.state()["columns"]
        self.assertEqual(st["amd"]["tag"], "identity-amd-20261104T1Z-ffffff")
        self.assertEqual(st["nvidia"]["last_poll"]["phase"], "queued")

        # the NVIDIA job finished: fetched, its records collected, then the cpu job is submitted
        sh3 = FakeShell(sel)
        sh3.alive_pids = set(sh.alive_pids)
        sh3.manifest[nv_tag] = "MANIFEST state=final exit=0 updated=x uploads=4\nLANDED final.tgz sha256=a\n"
        sh3.extract[nv_tag] = lambda d: write_record(d / "merged-nvidia" / "column", "nvidia-rtx-sm_89.lanes-1111")
        sh3.manifest["identity-amd-20261104T1Z-ffffff"] = "MANIFEST state=running exit= updated=x uploads=1\n"
        self.assertEqual(self.refresh(sh3, **env).run(wait=False), 0)
        st = self.state()["columns"]
        self.assertEqual(st["nvidia"]["phase"], "done")
        self.assertEqual(len(st["nvidia"]["records"]), 1)
        self.assertEqual(sh3.count("identity_box_job_cpu_ptx.sh"), 1)   # in the tree the nvidia job built
        self.assertEqual(st["cpu"]["phase"], "submitted")
        self.assertEqual(sh3.count("--extract"), 1)

    def test_dead_box_stops_and_resubmit_starts_again(self):
        sh = FakeShell(dict(nvidia=table_sel(2)))
        self.refresh(sh, ["nvidia"]).run(wait=False)
        tag = self.state()["columns"]["nvidia"]["tag"]
        sh.manifest[tag] = "MANIFEST state=flushed exit= updated=x uploads=3\n"
        self.assertEqual(self.refresh(sh, ["nvidia"]).run(wait=False), 1)
        self.assertEqual(self.state()["columns"]["nvidia"]["phase"], "dead")
        self.assertEqual(sh.count(" submit "), 1)          # never resubmitted by itself
        r = self.refresh(sh, ["nvidia"])
        r.resubmit("nvidia")
        r.run(wait=False)
        st = self.state()["columns"]["nvidia"]
        self.assertEqual(sh.count(" submit "), 2)
        self.assertNotEqual(st["tag"], tag)
        self.assertEqual(st["history"][0]["tag"], tag)

    def test_apple_without_a_built_tree_is_a_gap_not_admitted(self):
        sh = FakeShell(dict(apple=class_sel(420, 8)))
        self.assertEqual(self.refresh(sh, ["apple"]).run(wait=False), 4)
        self.assertEqual(self.state()["columns"]["apple"]["phase"], "gap")
        self.assertIn("REFRESH_APPLE_TREE", self.state()["columns"]["apple"]["why"])
        self.assertEqual(sh.spawned, [])
        self.assertEqual(sh.count("admit_identity_columns"), 0)
        self.assertTrue(any(x.startswith("GAPS apple=8") for x in self.lines))


class TestAdmit(Base):
    def finished(self, sh, cols=("nvidia",)):
        sel = dict(nvidia=table_sel(2), amd=table_sel(2))
        sh.sel.update(sel)
        self.refresh(sh, list(cols)).run(wait=False)
        for c in cols:
            if c == "amd":
                continue                     # launched detached; its tag comes from the launcher log
            tag = self.state()["columns"][c]["tag"]
            sh.manifest[tag] = "MANIFEST state=final exit=0 updated=x uploads=4\nLANDED final.tgz sha256=a\n"
            sh.extract[tag] = lambda d, c=c: write_record(d / ("merged-" + c) / "column", "%s-x.lanes-2222" % c)

    def test_admit_commits_and_pushes_once(self):
        sh = FakeShell()
        self.finished(sh)
        self.assertEqual(self.refresh(sh, ["nvidia"]).run(wait=False), 0)
        a = self.state()["admit"]
        self.assertEqual(a["phase"], "pushed")
        self.assertEqual(a["branch"], "ref/" + SHA[:12])
        admit = [c for c in sh.cmds if "admit_identity_columns.sh" in " ".join(c)]
        self.assertEqual(len(admit), 1)
        self.assertIn("--build-host", admit[0])
        self.assertTrue(admit[0][-1].endswith("records/nvidia"))
        add = next(c for c in sh.cmds if c[:3] == ["git", "add", "--"])
        self.assertIn("python/mojolearn/verify_reference/table.json", add)
        self.assertEqual(sh.count("git commit"), 1)
        self.assertEqual(sh.count("git push"), 1)
        # rerun: nothing more
        self.refresh(sh, ["nvidia"]).run(wait=False)
        self.assertEqual(sh.count("admit_identity_columns"), 1)
        self.assertEqual(sh.count("git commit"), 1)
        self.assertEqual(sh.count("git push"), 1)

    def test_all_columns_admitted_in_one_call(self):
        sh = FakeShell()
        self.finished(sh, ("nvidia", "amd"))
        # amd's launcher log names its tag
        st = self.state()["columns"]
        Path(st["amd"]["launch_log"]).parent.mkdir(parents=True, exist_ok=True)
        Path(st["amd"]["launch_log"]).write_text("box_persist: job identity-amd-T-1 persists to R2\n")
        sh.manifest["identity-amd-T-1"] = "MANIFEST state=final exit=0 updated=x uploads=4\n"
        sh.extract["identity-amd-T-1"] = lambda d: write_record(d / "merged-amd" / "column", "amd-mi300x.lanes-3")
        self.refresh(sh, ["nvidia", "amd"]).run(wait=False)    # amd: tag learned, then fetched
        self.refresh(sh, ["nvidia", "amd"]).run(wait=False)
        admit = [c for c in sh.cmds if "admit_identity_columns.sh" in " ".join(c)]
        self.assertEqual(len(admit), 1)
        self.assertEqual(sorted(Path(x).name for x in admit[0][3:]), ["amd", "nvidia"])

    def test_push_retried_after_a_crash_without_recommitting(self):
        sh = FakeShell()
        self.finished(sh)
        r = self.refresh(sh, ["nvidia"])
        real = sh.run

        def failing_push(cmd, **k):
            if cmd[:2] == ["git", "push"]:
                sh.cmds.append([str(c) for c in cmd])
                return 1, "remote: hook refused"
            return real(cmd, **k)
        sh.run = failing_push
        with self.assertRaises(rr.Refused):
            r.run(wait=False)
        self.assertEqual(self.state()["admit"]["phase"], "committed")
        sh.run = real
        self.refresh(sh, ["nvidia"]).run(wait=False)
        self.assertEqual(sh.count("git commit"), 1)
        self.assertEqual(self.state()["admit"]["phase"], "pushed")

    def test_refused_admission_is_saved(self):
        sh = FakeShell()
        sh.admit_rc, sh.admit_out = 2, "admit_identity_columns: GPU and cpu reference agreement refused\n"
        self.finished(sh)
        with self.assertRaises(rr.Refused):
            self.refresh(sh, ["nvidia"]).run(wait=False)
        self.assertEqual(self.state()["admit"]["phase"], "refused")
        self.assertEqual(sh.count("git commit"), 0)


OLD = "1111111111111111111111111111111111111111"
NEW = "3333333333333333333333333333333333333333"
SIDE = "5555555555555555555555555555555555555555"


def env_of(cmd):
    """The --env K=V pairs of an nvidia_central submit command."""
    return dict(cmd[i + 1].split("=", 1) for i, x in enumerate(cmd) if x == "--env")


class TestScope(Base):
    def test_box_jobs_run_the_supported_surface(self):
        sel = dict(nvidia=table_sel(3), amd=table_sel(3), cpu=class_sel(420, 8), ptx=class_sel(428, 0),
                   apple=class_sel(425, 3))
        sh = FakeShell(sel)
        tree = self.t / "apple-tree"
        (tree / ".git").mkdir(parents=True)
        self.refresh(sh, REFRESH_APPLE_TREE=str(tree)).run(wait=False)
        self.assertEqual(self.state()["scope"]["lanes"], "all")
        self.assertFalse(self.state()["scope"]["narrowed"])
        nv = next(c for c in sh.cmds if "nvidia_central.sh" in " ".join(c) and " submit " in " ".join(c))
        self.assertEqual(env_of(nv), dict(SCOPE="all"))
        amd = sh.spawned[0][0]
        self.assertIn("env SCOPE=all bash tools/identity_box_job.sh amd", amd)
        apple = sh.spawned[1][0]
        self.assertIn("SCOPE=all", apple)

    def test_narrowing_needs_a_reason_and_keeps_it(self):
        sh = FakeShell(dict(nvidia=table_sel(2)))
        with self.assertRaises(rr.Refused) as cm:
            self.refresh(sh, ["nvidia"], reason="").plan(dry=True)
        self.assertIn("--reason", str(cm.exception))
        r = self.refresh(sh, ["nvidia"], reason="AMD droplet out of stock")
        r.run(wait=False)
        sc = self.state()["scope"]
        self.assertTrue(sc["narrowed"])
        self.assertEqual(sc["reason"], "AMD droplet out of stock")
        self.assertEqual(sc["dropped_columns"], ["amd", "cpu", "ptx", "apple"])
        self.assertTrue(any("NARROWED from the supported surface" in x for x in self.lines))

    def test_lane_scope_narrows_and_counts_what_it_leaves_out(self):
        sh = FakeShell(dict(nvidia=table_sel(5)))
        sh.scope_lanes["routine"] = ["l0", "l1", "x9"]            # l2..l4 are neural-training lanes
        self.refresh(sh, ["nvidia"], lane_scope="routine", reason="hold neural for the next pod").run(wait=False)
        p = self.state()["columns"]["nvidia"]["plan"]
        self.assertEqual((p["owed"], p["record"], p["narrowed"]), (5, 2, 3))
        self.assertEqual(p["narrowed_lanes"], ["l2", "l3", "l4"])
        nv = next(c for c in sh.cmds if " submit " in " ".join(c))
        self.assertEqual(env_of(nv)["SCOPE"], "routine")
        self.assertTrue(any("NARROWED 3 owed lane(s) out of scope" in x for x in self.lines))

    def test_scope_never_widens(self):
        write_table(self.t / "src" / rr.TABLE, ("nvidia", "amd", "cpu", "apple"))
        with self.assertRaises(rr.Refused) as cm:
            self.refresh(FakeShell(), ["cpu", "ptx"]).plan(dry=True)
        self.assertIn("only narrow", str(cm.exception))
        with self.assertRaises(rr.Refused):
            self.refresh(FakeShell(), lane_scope="everything")
        with self.assertRaises(rr.Refused):
            self.refresh(FakeShell(), REFRESH_SCOPE="routine")

    def test_ptx_needs_the_cpu_job(self):
        with self.assertRaises(rr.Refused) as cm:
            self.refresh(FakeShell(), ["nvidia", "ptx"]).plan(dry=True)
        self.assertIn("add cpu", str(cm.exception))

    def test_a_rerun_keeps_the_planned_scope(self):
        sh = FakeShell(dict(nvidia=table_sel(2), amd=table_sel(1)))
        self.refresh(sh, ["nvidia"]).run(wait=False)
        with self.assertRaises(rr.Refused) as cm:
            self.refresh(sh, ["nvidia", "amd"]).run(wait=False)
        self.assertIn("keeps that scope", str(cm.exception))


class TestGaps(Base):
    def all_sel(self, **kw):
        sel = dict(nvidia=table_sel(0), amd=table_sel(0), cpu=class_sel(428, 0), ptx=class_sel(428, 0),
                   apple=class_sel(428, 0))
        sel.update(kw)
        return sel

    def test_plan_prints_gaps_per_column(self):
        (self.t / "devpods" / "nvc1" / "state.env").unlink()            # no NVIDIA pod
        sh = FakeShell(self.all_sel(nvidia=table_sel(4), amd=table_sel(2), ptx=class_sel(343, 85),
                                    apple=class_sel(420, 8)))
        self.refresh(sh).plan(dry=True)
        line = lambda c: next(x for x in self.lines if x.startswith("PLAN %-6s" % c))  # noqa: E731
        self.assertIn("GAP 4 owed lane(s) not recorded", line("nvidia"))
        self.assertIn("no shared NVIDIA pod", line("nvidia"))
        self.assertIn("REUSED 426 / RECORD 2 of 428", line("amd"))
        self.assertIn("GAP 85 owed lane(s)", line("ptx"))
        self.assertIn("GAP 8 owed lane(s)", line("apple"))
        self.assertIn("NOTHING-CHANGED", line("cpu"))                    # owes nothing: no gap
        self.assertIn("GAPS nvidia=4 ptx=85 apple=8", self.lines)
        self.assertFalse((self.t / "home" / SHA[:12] / "state.json").exists())

    def test_ptx_without_a_set_is_a_gap_with_a_pod(self):
        sh = FakeShell(self.all_sel(ptx=class_sel(343, 85)))
        self.refresh(sh, REFRESH_APPLE_TREE=str(self.t)).plan(dry=True)
        self.assertIn("GAPS ptx=85", self.lines)
        self.assertTrue(any("REFRESH_PTX_SET_TGZ" in x for x in self.lines if x.startswith("PLAN ptx")))

    def test_a_table_column_without_a_runner_is_a_gap(self):
        write_table(self.t / "src" / rr.TABLE, rr.COLUMNS + ("intel",))
        sh = FakeShell(self.all_sel())
        self.refresh(sh, REFRESH_APPLE_TREE=str(self.t)).plan(dry=True)
        self.assertIn("GAPS intel=?", self.lines)

    def test_gaps_do_not_block_the_other_columns_and_reach_the_readme(self):
        sh = FakeShell(self.all_sel(nvidia=table_sel(2), ptx=class_sel(343, 85)))
        tree = self.t / "apple-tree"
        (tree / ".git").mkdir(parents=True)
        env = dict(REFRESH_APPLE_TREE=str(tree))
        self.refresh(sh, **env).run(wait=False)
        st = self.state()["columns"]
        self.assertEqual(st["ptx"]["phase"], "gap")
        self.assertEqual(st["cpu"]["phase"], "nothing")                # no PTX set: the cpu job is not needed
        tag = st["nvidia"]["tag"]
        sh.manifest[tag] = "MANIFEST state=final exit=0 updated=x uploads=4\nLANDED final.tgz sha256=a\n"
        sh.extract[tag] = lambda d: write_record(d / "merged-nvidia" / "column", "nvidia-rtx-sm_89.lanes-1")
        self.assertEqual(self.refresh(sh, **env).run(wait=False), 4)     # admitted, with gaps
        self.assertEqual(self.state()["admit"]["phase"], "pushed")
        readme = (self.t / "admit" / "bench/results/reference_admissions/2026-11-04-nv-fix0865/README.md").read_text()
        self.assertIn("## Reference refresh scope", readme)
        self.assertIn("GAPS (owed lanes no box or column recorded", readme)
        self.assertIn("ptx 85: no sm_80 PTX set", readme)
        self.assertIn("| nvidia | 2 | 426 | 1 record file(s) | 0 |  |", readme)
        msg = next(c for c in sh.cmds if c[:2] == ["git", "commit"])
        self.assertIn("GAPS (owed, not recorded): ptx 85", msg[-1])
        self.assertTrue(any(x.startswith("GAPS ptx=85") for x in self.lines))

    def test_a_gap_clears_when_its_box_comes_up_before_admission(self):
        (self.t / "do" / "state.env").unlink()                          # no AMD droplet yet
        sh = FakeShell(dict(nvidia=table_sel(2), amd=table_sel(3)))
        self.refresh(sh, ["nvidia", "amd"]).run(wait=False)
        self.assertEqual(self.state()["columns"]["amd"]["phase"], "gap")
        self.assertEqual(sh.spawned, [])
        (self.t / "do" / "state.env").write_text("X=1\n")              # the orchestrator brought it up
        self.refresh(sh, ["nvidia", "amd"]).run(wait=False)
        st = self.state()["columns"]["amd"]
        self.assertEqual(st["phase"], "launching")
        self.assertEqual(st["history"][0]["phase"], "gap")
        self.assertIn("GAP CLEARED amd: its box or runner is up; recording its 3 owed lane(s)", self.lines)


class TestPtx(Base):
    PTX = dict(REFRESH_PTX_SET_TGZ="/root/sets/cuda-sm_80.tar.gz", REFRESH_PTX_SET_SHA256="ab" * 32,
               REFRESH_PTX_SET_COMMIT=SHA)

    def test_ptx_rides_on_the_cpu_job_and_is_admitted_with_it(self):
        sel = dict(nvidia=table_sel(0), cpu=class_sel(420, 8), ptx=class_sel(343, 85))
        sh = FakeShell(sel)
        cols = ["nvidia", "cpu", "ptx"]
        self.refresh(sh, cols, **self.PTX).run(wait=False)
        st = self.state()["columns"]
        self.assertTrue(st["nvidia"]["for_cpu_only"])
        nv_tag = st["nvidia"]["tag"]
        sh.manifest[nv_tag] = "MANIFEST state=final exit=0 updated=x uploads=4\n"
        sh.extract[nv_tag] = lambda d: (d / "merged-nvidia").mkdir(parents=True)
        self.refresh(sh, cols, **self.PTX).run(wait=False)
        st = self.state()["columns"]
        self.assertEqual(st["nvidia"]["phase"], "done")
        cpu_cmd = next(c for c in sh.cmds if "identity_box_job_cpu_ptx.sh" in " ".join(c))
        e = env_of(cpu_cmd)
        self.assertEqual(e["ID_SCOPE"], "all")
        self.assertEqual(e["EXPECT_COMMIT"], SHA)
        self.assertEqual(e["PTX_SET_TGZ"], "/root/sets/cuda-sm_80.tar.gz")
        self.assertEqual(st["ptx"]["phase"], "with-cpu")
        self.assertEqual(st["ptx"]["tag"], st["cpu"]["tag"])
        tag = st["cpu"]["tag"]

        def lay(d):
            write_record(d / "cpu-ptx" / "cpu", "cpu.lanes-1")
            write_record(d / "cpu-ptx" / "ptx", "nvidia-ptx-rtx-sm80.lanes-2")
        sh.manifest[tag] = "MANIFEST state=final exit=0 updated=x uploads=4\n"
        sh.extract[tag] = lay
        self.assertEqual(self.refresh(sh, cols, **self.PTX).run(wait=False), 0)
        st = self.state()["columns"]
        self.assertEqual([Path(x).name for x in st["cpu"]["records"]], ["cpu.lanes-1.identical.json"])
        self.assertEqual([Path(x).name for x in st["ptx"]["records"]], ["nvidia-ptx-rtx-sm80.lanes-2.identical.json"])
        admit = next(c for c in sh.cmds if "admit_identity_columns.sh" in " ".join(c))
        self.assertEqual(sorted(Path(x).name for x in admit[3:]), ["cpu", "ptx"])
        self.assertEqual(sh.count("--extract"), 2)                    # one fetch per job, not per column

    def test_ptx_alone_still_runs_the_cpu_job(self):
        sh = FakeShell(dict(nvidia=table_sel(0), cpu=class_sel(428, 0), ptx=class_sel(343, 85)))
        cols = ["nvidia", "cpu", "ptx"]
        self.refresh(sh, cols, **self.PTX).run(wait=False)
        nv_tag = self.state()["columns"]["nvidia"]["tag"]
        sh.manifest[nv_tag] = "MANIFEST state=final exit=0 updated=x uploads=4\n"
        sh.extract[nv_tag] = lambda d: (d / "merged-nvidia").mkdir(parents=True)
        self.refresh(sh, cols, **self.PTX).run(wait=False)
        st = self.state()["columns"]
        self.assertTrue(st["cpu"]["for_ptx_only"])
        self.assertEqual(st["cpu"]["phase"], "submitted")
        self.assertEqual(st["ptx"]["phase"], "with-cpu")


class TestCoalesce(Base):
    def make_state(self, sha, **d):
        doc = dict(schema="mojolearn.reference-refresh.v1", sha=sha, columns={}, admit={})
        doc.update(d)
        p = self.t / "home" / sha[:12] / "state.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc))
        return p

    def sh(self):
        sh = FakeShell(dict(nvidia=table_sel(2)))
        sh.history = [OLD, SHA, NEW]
        return sh

    def test_a_newer_refresh_supersedes_pending_older_ones(self):
        self.make_state(OLD, columns=dict(nvidia=dict(phase="submitted", tag="nvc-identity-old")))
        target, sup = rr.coalesce(SHA, env=self.env, sh=self.sh(), out=self.lines.append)
        self.assertEqual((target, sup), (SHA, [OLD]))
        old = self.state(OLD)
        self.assertEqual(old["superseded_by"], SHA)
        self.assertEqual(old["superseded_jobs"], ["nvidia submitted nvc-identity-old"])
        self.assertTrue(any(x.startswith("SUPERSEDED 111111111111 by 0814836ffb8f") for x in self.lines))
        with self.assertRaises(rr.Refused) as cm:          # the superseded refresh never runs again
            self.refresh(self.sh(), ["nvidia"], sha=OLD).run(wait=False)
        self.assertIn("superseded by", str(cm.exception))

    def test_an_older_request_targets_the_newest_pending_commit(self):
        self.make_state(NEW)
        target, sup = rr.coalesce(SHA, env=self.env, sh=self.sh(), out=self.lines.append)
        self.assertEqual((target, sup), (NEW, []))
        self.assertIn("COALESCED 0814836ffb8f into the pending refresh of the newer 333333333333", self.lines)

    def test_dry_saves_nothing(self):
        self.make_state(OLD)
        target, sup = rr.coalesce(SHA, env=self.env, sh=self.sh(), out=self.lines.append, dry=True)
        self.assertEqual(sup, [OLD])
        self.assertNotIn("superseded_by", self.state(OLD))

    def test_a_finished_newer_refresh_leaves_nothing_to_do(self):
        self.make_state(NEW, admit=dict(phase="pushed", commit=NEW))
        with self.assertRaises(rr.Refused) as cm:
            rr.coalesce(SHA, env=self.env, sh=self.sh(), out=self.lines.append)
        self.assertIn("nothing to do", str(cm.exception))

    def test_an_unmerged_admission_stops_the_refresh(self):
        sh = self.sh()
        admit_commit = "7777777777777777777777777777777777777777"
        sh.history = [OLD, SHA, NEW, admit_commit]          # the ref/ branch commit is not in SHA
        self.make_state(OLD, admit=dict(phase="pushed", commit=admit_commit, branch="ref/" + OLD[:12]))
        with self.assertRaises(rr.Refused) as cm:
            rr.coalesce(SHA, env=self.env, sh=sh, out=self.lines.append)
        self.assertIn("not merged into", str(cm.exception))
        sh.history = [OLD, admit_commit, SHA, NEW]          # merged: fine
        self.assertEqual(rr.coalesce(SHA, env=self.env, sh=sh, out=self.lines.append), (SHA, []))

    def test_an_unrelated_pending_refresh_stays_pending(self):
        self.make_state(SIDE)
        sh = self.sh()
        real = sh.run

        def run(cmd, **k):
            if "--is-ancestor" in cmd and SIDE in cmd:
                sh.cmds.append([str(c) for c in cmd])
                return 1, ""
            return real(cmd, **k)
        sh.run = run
        self.assertEqual(rr.coalesce(SHA, env=self.env, sh=sh, out=self.lines.append), (SHA, []))
        self.assertNotIn("superseded_by", self.state(SIDE))
        self.assertTrue(any(x.startswith("PENDING 555555555555 stays pending") for x in self.lines))

    def test_main_targets_origin_main_and_records_the_superseded(self):
        self.make_state(OLD, columns=dict(nvidia=dict(phase="planned")))
        sh = self.sh()
        sh.resolve["origin/main"] = SHA
        rc = rr.main(["--no-wait", "--reason", "test", "origin/main", "nvidia"], env=self.env, sh=sh)
        self.assertEqual(rc, 0)
        self.assertEqual(self.state(SHA)["supersedes"], [OLD])
        self.assertEqual(self.state(OLD)["superseded_by"], SHA)
        self.assertEqual(sh.count(" submit "), 1)                    # one build, at the newest commit


if __name__ == "__main__":
    unittest.main(verbosity=2)
