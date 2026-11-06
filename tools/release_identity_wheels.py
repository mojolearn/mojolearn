#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ONE RELEASE RUN ON THE BUILT WHEELS (Andrew 2026-11-05): tools/release.py --identity-on-wheels.

Default OFF until the orchestrator has run it once; without the option nothing here changes a step. With it:

  macos-build, linux-pack  the wheels, as before;
  gate-nvidia / gate-amd   ONE box job per vendor (tools/ops/release_gate.py --identity-columns): the exact wheels
                           installed (sha256 checked), the owed identity lanes recorded from them
                           (tools/identity_wheel_record.py; the NVIDIA columns are --nvidia-columns, each on its
                           own GPU when the pod has several), then the device checks (install/load and kernel-set
                           selection, the compiled autograd checks and entries, the adapters, the installed-runtime
                           units) with the gate's identity-lane jobs dropped (--no-identity-lanes): nothing twice;
  gate-apple               the Apple column from the installed macOS wheel, then the adapters and runtime-apple,
                           in one step under the Metal lock (macos-smoke runs nothing: its self-test lanes are in
                           the Apple column);
  identity-cpu             the cpu column and every CPU check on GitHub's free runners
                           (.github/workflows/cpu-identity-gate.yml mode wheel), waited on and fetched like a leg;
  identity-admit           every fetched column (NVIDIA, PTX, AMD, Apple, CPU) admitted at once
                           (tools/admit_identity_columns.sh compares them; NVIDIA == AMD) on ref/<commit12> from the
                           frozen source, the admitted table repacked into the core and macOS wheels data-only
                           (tools/wheel_table_repack.py, every other member checked by digest), and the ledger told
                           to judge identity at the admission commit (source-equal to the freeze:
                           MOJOLEARN_IDENTITY_REFERENCE_AT, tools/release_requirements.py);
  publish-*                the repacked core and macOS wheels, the plugins as built.

Reuse is the existing rule everywhere: a lane runs only when its closure changed since its class's newest good
evidence (the recorder's selection, computed here before any rental); the runtime units by their own receipts;
a vendor whose lanes are all reused AND whose gate already passed for these exact wheel bytes rents nothing
(REUSED n / RUN 0, no rental). No step after the first publish runs a lane (release.DEVICE_LANE_STEPS).
Tests: tools/test_release_identity_wheels.py (no box, no network, no GitHub).
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = "identity-wheels.json"
#: the default NVIDIA columns: what main records today (the native sm_89 column and the PTX column)
DEFAULT_NVIDIA_COLUMNS = ("nvidia", "ptx")
#: the device slug of each vendor's column labels (MOJOLEARN_IDENTITY_LABEL_<VENDOR> overrides); only the label
#: prefix decides the device class (identity_columns.label_class), so the slug never changes reuse
DEFAULT_LABEL_DEVICE = {"nvidia": "l40s", "amd": "mi325x"}
CPU_WORKFLOW = "cpu-identity-gate.yml"
#: GitHub jobs in the cpu column matrix (each records its share of the owed lanes with nproc/3 workers)
CPU_SHARDS = 4
REFERENCE_ENV = "MOJOLEARN_IDENTITY_REFERENCE_AT"


class StepFailed(RuntimeError):
    """Replaced by tools/release.py with its own StepFailed (release.py binds it at import), so a refusal here
    is the release's ordinary failed step."""


def write_json(path, doc):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
TABLE = "python/mojolearn/verify_reference/table.json"


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


def summary_errors(doc, columns):
    """Why an identity-wheels.json does not prove its columns recorded, or []."""
    if not doc or doc.get("schema") != "mojolearn.identity-wheels.v1":
        return ["no identity-wheels summary"]
    errors = []
    got = set((doc.get("columns") or {}).keys())
    if got != set(columns):
        errors.append("columns %s, not %s" % (sorted(got), sorted(columns)))
    for col, row in sorted((doc.get("columns") or {}).items()):
        if row.get("rc") != 0:
            errors.append("%s exited %s" % (col, row.get("rc")))
        recorded = sum(c.get("lanes", 0) for c in row.get("chunks") or [])
        if recorded != row.get("owed"):
            errors.append("%s recorded %s of %s owed lane(s)" % (col, recorded, row.get("owed")))
        for chunk in row.get("chunks") or []:
            if chunk.get("lanes") and not all(r.get("admissible") for r in chunk.get("records") or [{}]):
                errors.append("%s worker %s left an inadmissible record" % (col, chunk.get("worker")))
    if doc.get("ok") is not True and not errors:
        errors.append("the summary is not ok")
    return errors


def record_dirs(root):
    """Every directory under ROOT that holds a column record (<label>/w<i>/ of identity_wheel_record.py)."""
    return sorted({str(p.parent) for p in Path(root).rglob("*.identical.json")})


class IdentityOnWheels:
    """The --identity-on-wheels steps and hooks of tools/release.py's Release (a mixin)."""

    # ------------------------------------------------------------------ the option
    def single(self):
        return bool(getattr(self.args, "identity_on_wheels", False))

    def nvidia_columns(self):
        text = getattr(self.args, "nvidia_columns", "") or ",".join(DEFAULT_NVIDIA_COLUMNS)
        cols = [c for c in text.split(",") if c]
        if not cols or any(c not in ("nvidia", "ptx") for c in cols) or len(set(cols)) != len(cols):
            raise SystemExit("release: --nvidia-columns is a comma list of nvidia, ptx (got %r)" % text)
        return cols

    def vendor_columns(self, vendor):
        return {"nvidia": self.nvidia_columns(), "amd": ["amd"], "apple": ["apple"], "cpu": ["cpu"]}[vendor]

    def label_device(self, vendor):
        env = os.environ.get("MOJOLEARN_IDENTITY_LABEL_" + vendor.upper(), "")
        if env:
            return env
        if vendor == "apple":
            try:
                brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                       text=True, timeout=10).stdout
            except (OSError, subprocess.SubprocessError):
                brand = ""
            return re.sub(r"[^a-z0-9]", "", brand.lower().replace("apple", "")) or "apple"
        return DEFAULT_LABEL_DEVICE[vendor]

    def identity_dir(self):
        return self.rel / "identity-wheels"

    # ------------------------------------------------------------------ what is owed, before any rental
    def lanes_owed(self, column):
        """[lanes] the recorder would record for COLUMN at the frozen source: identity_columns.py changed-lanes
        with the recorder's own reuse arguments (record_identity_column.sh auto-reuse), static, on this Mac.
        An unreadable selection owes every lane of the column (None means: not decidable, so owed)."""
        cache = self.__dict__.setdefault("_lanes_owed", {})
        if column in cache:
            return cache[column]
        cls = {"nvidia": "nvidia", "ptx": "ptx", "amd": "amd", "apple": "apple", "cpu": "cpu"}[column]
        backend = {"nvidia": "cuda", "ptx": "cuda", "amd": "hip", "apple": "metal", "cpu": "cpu"}[column]
        work = self.identity_dir() / "plan"
        work.mkdir(parents=True, exist_ok=True)
        summary = work / ("%s.reuse.json" % column)
        cmd = [sys.executable, "tools/identity_columns.py", "changed-lanes", "--scope", "all", "--class", cls,
               "--records", str(Path(self.src) / "bench/results/identity_break"), "--backend", backend,
               "--summary-json", str(summary)]
        with open(work / ("%s.log" % column), "w") as log:
            p = subprocess.run(cmd, cwd=str(self.src), stdout=subprocess.PIPE, stderr=log, text=True)
        doc = read_json(summary)
        cache[column] = sorted((doc or {}).get("record") or {}) if p.returncode == 0 and doc else None
        if cache[column] is not None:
            cache[column + ":reused"] = len((doc or {}).get("reused") or {})
        return cache[column]

    def vendor_lanes(self, vendor):
        """(reused, owed) lane counts of a vendor's columns by the recorder's selection; owed None: undecided."""
        reused, run = 0, 0
        for col in self.vendor_columns(vendor):
            owed = self.lanes_owed(col)
            if owed is None:
                return reused, None
            run += len(owed)
            reused += self.__dict__.get("_lanes_owed", {}).get(col + ":reused", 0)
        return reused, run

    def single_box_owed(self, vendor):
        """Why the vendor's one box job still has work, or None (no rental). The box job is done exactly when its
        gate report for these wheel bytes PASSes with a complete identity summary (gate_ok); that is also the only
        reuse of the device checks (an earlier PASS for the same bytes). Prints REUSED n / RUN m."""
        ok, why = self.gate_ok(vendor)
        reused, run = self.vendor_lanes(vendor)
        self.say("  %s: REUSED %d / RUN %s lane(s); device checks %s%s" % (
            vendor, reused, "?" if run is None else run, "done for these wheel bytes" if ok else "owed",
            "; no rental" if ok else ""))
        return None if ok else why

    # ------------------------------------------------------------------ the box job
    def gate_identity_args(self, vendor):
        if not self.single():
            return []
        return ["--identity-columns", ",".join(self.vendor_columns(vendor)), "--label-device", self.label_device(vendor)]

    def gate_identity_errors(self, vendor, output, correctness):
        """Under --identity-on-wheels: the box job's column record must be complete (or nothing owed) and its gate
        must have dropped the identity-lane jobs; without it the gate must have run them."""
        once = (correctness or {}).get("identity_lanes") == "recorded-once"
        if not self.single():
            return ["the gate dropped its identity-lane jobs, but this release did not record identity on the wheels"] \
                if once else []
        if not once:
            return ["--identity-on-wheels: the gate ran its identity-lane jobs again (no --no-identity-lanes)"]
        doc = read_json(Path(output) / "identity" / SUMMARY)
        return ["identity: " + e for e in summary_errors(doc, self.vendor_columns(vendor))]

    # ------------------------------------------------------------------ Apple: the column, in gate-apple
    def apple_identity_out(self):
        return self.identity_dir() / "apple"

    def apple_identity_ok(self):
        errors = summary_errors(read_json(self.apple_identity_out() / SUMMARY), ["apple"])
        return (not errors), ("Apple column recorded from the macOS wheel" if not errors else "; ".join(errors))

    def apple_identity_gate(self):
        if not self.single():
            return None
        ok, why = self.apple_identity_ok()
        if ok:
            return why + " (summary exists)"
        owed = self.lanes_owed("apple")
        if owed == []:
            return "Apple column: REUSED, RUN 0 (no lane's closure changed)"
        w = self.macos_wheel()
        if not w and not self.dry:
            raise StepFailed("no macOS wheel; run macos-build")
        self.refuse_device_lanes_after_publish("gate-apple")
        out = self.apple_identity_out()
        if out.exists() and not self.dry:
            out.rename(out.with_name(out.name + ".failed-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")))
        name = Path(w).name if w else "<macos wheel>"
        digest = sha256(w) if w and not self.dry else "DRY-RUN"
        self.must([sys.executable, str(ROOT / "tools" / "mac_slot.py"), "--wait-timeout", "3600", "metal", "--",
                   sys.executable, str(ROOT / "tools" / "identity_wheel_record.py"), "--columns", "apple",
                   "--label-device", self.label_device("apple"), "--wheel", str(w or name), "--sha256", "%s=%s" % (name, digest),
                   "--out", str(out), "--python", self.smoke_python() if not self.dry else "python3.12"],
                  log=self.rel / "gate-apple-identity.log", what="the Apple column from the macOS wheel", cwd=self.src)
        if self.dry:
            return None
        ok, why = self.apple_identity_ok()
        if not ok:
            raise StepFailed("the Apple column did not record: " + why)
        return why

    # ------------------------------------------------------------------ CPU on GitHub
    def cpu_out(self):
        return self.identity_dir() / "cpu"

    def cpu_ok(self):
        out = self.cpu_out()
        doc = read_json(out / "run.json") or {}
        if doc.get("conclusion") != "success" or doc.get("source_commit") != self.commit:
            return False, "no successful %s run for %s" % (CPU_WORKFLOW, self.commit[:12])
        core = self.split_final("linux")
        if not core or doc.get("wheel_sha256") != sha256(core):
            return False, "the CPU run checked another core wheel"
        checks = read_json(out / "artifacts" / "cpu-wheel-checks" / "checks.json") or {}
        bad = [k for k, v in (checks.get("checks") or {}).items() if v != 0]
        if not checks.get("checks") or bad:
            return False, "CPU checks failed or missing: %s" % (", ".join(bad) or "no checks.json")
        for shard in range(doc.get("shards", 0)):
            errors = summary_errors(read_json(out / "artifacts" / ("cpu-wheel-identity-%d" % shard) / SUMMARY), ["cpu"])
            if errors:
                return False, "cpu column shard %d: %s" % (shard, "; ".join(errors))
        return True, "cpu column and CPU checks PASSED on GitHub (run %s)" % doc.get("run_id")

    def step_identity_cpu(self):
        """IDENTITY-CPU (--identity-on-wheels): the cpu column and every CPU check, once, on GitHub's free runners
        from the built core wheel; the release waits for the run and fetches its artifacts."""
        if not self.single():
            return "skipped (no --identity-on-wheels: the cpu column is recorded by the reference refresh)"
        self.refuse_device_lanes_after_publish("identity-cpu")
        ok, why = self.cpu_ok()
        if ok:
            return why + " (run exists)"
        core = self.split_final("linux")
        if not core and not self.dry:
            raise StepFailed("no core wheel; run linux-pack")
        out = self.cpu_out()
        token = hashlib.sha256(("%s %s %s" % (self.commit, time.time(), os.getpid())).encode()).hexdigest()[:12]
        if self.dry:
            self.say("  would upload %s to R2, dispatch %s mode=wheel (%d shard(s)) on main and wait" % (
                core or "<core wheel>", CPU_WORKFLOW, CPU_SHARDS))
            return None
        url = self.r2_presigned_upload(core, "release-identity/%s/%s/%s" % (self.version, self.commit[:12], core.name))
        prior_url = ""
        prior = self.cpu_runtime_prior()
        if prior:
            prior_url = self.r2_presigned_upload(prior, "release-identity/%s/%s/runtime-units-cpu.tgz" % (
                self.version, self.commit[:12]))
        fields = dict(confirm="RUN-HEAVY", mode="wheel", wheel_url=url, wheel_name=core.name,
                      wheel_sha256=sha256(core), source_commit=self.commit, shards=str(CPU_SHARDS),
                      runtime_prior_url=prior_url, token=token)
        argv = ["workflow", "run", CPU_WORKFLOW, "--ref", "main"]
        for k, v in fields.items():
            argv += ["-f", "%s=%s" % (k, v)]
        rc, _ = self.gh(*argv, log=self.rel / "identity-cpu.log")
        if rc != 0:
            raise StepFailed("could not dispatch %s (see %s)" % (CPU_WORKFLOW, self.rel / "identity-cpu.log"))
        run_id, link = self.cpu_find_run(token)
        self.say("  %s run %s: %s" % (CPU_WORKFLOW, run_id, link))
        conclusion = self.cpu_wait(run_id)
        art = out / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        self.gh("run", "download", run_id, "-D", str(art), log=self.rel / "identity-cpu.log")
        out.mkdir(parents=True, exist_ok=True)
        (out / "run.json").write_text(json.dumps(dict(run_id=run_id, url=link, conclusion=conclusion,
                                                      source_commit=self.commit, wheel_sha256=fields["wheel_sha256"],
                                                      shards=CPU_SHARDS, token=token), indent=1) + "\n")
        ok, why = self.cpu_ok()
        if not ok:
            raise StepFailed("the CPU run did not pass: %s (%s)" % (why, link))
        return why

    def cpu_find_run(self, token, tries=24):
        for _ in range(tries):
            rc, out = self.gh("run", "list", "--workflow", CPU_WORKFLOW, "--event", "workflow_dispatch", "--limit", "30",
                              "--json", "databaseId,displayTitle,url")
            for r in (json.loads(out or "[]") if rc == 0 else []):
                if token in r.get("displayTitle", ""):
                    return str(r["databaseId"]), r.get("url", "")
            self.sleep(5)
        raise StepFailed("the dispatched %s run (token %s) never appeared" % (CPU_WORKFLOW, token))

    def cpu_wait(self, run_id, poll=120, hours=8):
        deadline = time.time() + hours * 3600
        while time.time() < deadline:
            rc, out = self.gh("run", "view", run_id, "--json", "status,conclusion")
            doc = json.loads(out or "{}") if rc == 0 else {}
            if doc.get("status") == "completed":
                return doc.get("conclusion")
            self.sleep(poll)
        raise StepFailed("%s run %s did not finish within %d h" % (CPU_WORKFLOW, run_id, hours))

    def cpu_runtime_prior(self):
        """A tgz of the newest passing cpu runtime unit receipts this machine retains, or None."""
        import runtime_qualifier_units as units
        import tarfile
        sources = self.runtime_unit_sources("gate-nvidia") + sorted(self.base.parent.glob("*/*/identity-wheels/cpu/artifacts/cpu-wheel-checks/runtime/units"))
        dest = self.identity_dir() / "runtime-prior-cpu"
        staged = units.stage_prior(sources, dest, ("cpu",))
        if not staged:
            return None
        tgz = self.identity_dir() / "runtime-units-cpu.tgz"
        with tarfile.open(tgz, "w:gz") as tar:
            tar.add(str(dest), arcname=".")
        return tgz

    def r2_presigned_upload(self, path, key, days=2):
        """PUT PATH to R2 at KEY (credentials stay on this Mac) and return a presigned GET of it."""
        import box_persist
        import bincache
        creds = box_persist.load_creds()
        put = bincache.presign("PUT", key, 3600, creds)
        status = bincache.http_put(put, str(path))
        if status not in (200, 201):
            raise StepFailed("R2 upload of %s failed (HTTP %s)" % (Path(path).name, status))
        return bincache.presign("GET", key, int(days * 86400), creds)

    # ------------------------------------------------------------------ admission + data-only repack
    def admitted(self):
        return self.state.get("identity_admitted") if (self.state.get("identity_admitted") or {}).get(
            "source_commit") == self.commit else None

    def all_record_dirs(self):
        dirs = []
        for vendor in ("nvidia", "amd"):
            try:
                dirs += record_dirs(self.gate_out(vendor) / "identity")
            except Exception:   # no candidate wheels: nothing recorded there
                pass
        dirs += record_dirs(self.apple_identity_out()) + record_dirs(self.cpu_out() / "artifacts")
        return dirs

    def step_identity_admit(self):
        """IDENTITY-ADMIT (--identity-on-wheels): every column of this release's one run admitted at once on
        ref/<commit12> from the frozen source, then the admitted table repacked into the core and macOS wheels."""
        if not self.single():
            return "skipped (no --identity-on-wheels)"
        done = self.admitted()
        if done and self.repacked_ok():
            return "admitted at %s; wheels repacked (exists)" % done["commit"][:12]
        for step in ("gate-nvidia", "gate-amd", "gate-apple", "identity-cpu"):
            if not self.recorded(step):
                raise StepFailed("%s has not passed: the one run is incomplete; nothing is admitted" % step)
        dirs = self.all_record_dirs()
        if self.dry:
            self.say("  would admit %d record dir(s) on ref/%s and repack the core and macOS wheels" % (len(dirs), self.commit[:12]))
            return None
        if not dirs:
            # every column reused every lane: the table at the freeze already holds this release's bits
            doc = dict(source_commit=self.commit, commit=self.commit, recorded=0, at=dt.datetime.now(dt.timezone.utc).isoformat())
            self.state["identity_admitted"] = doc
            self.save()
            os.environ[REFERENCE_ENV] = "%s:%s" % (self.commit, self.commit)
            return "nothing recorded (every lane reused): the table at the freeze ships as built; no repack"
        commit = self.admit_records(dirs)
        import identity_at_freeze
        if not identity_at_freeze.source_equal(self.src, self.commit, commit):
            raise StepFailed("the admission commit %s is not source-equal to the freeze %s" % (commit[:12], self.commit[:12]))
        table = self.identity_dir() / "admitted-table.json"
        table.write_bytes(subprocess.run(["git", "-C", str(self.src), "show", "%s:%s" % (commit, TABLE)],
                                         capture_output=True, check=True).stdout)
        receipts = self.repack(table)
        doc = dict(source_commit=self.commit, commit=commit, branch="ref/" + self.commit[:12], recorded=len(dirs),
                   table_sha256=sha256(table), repacked=receipts, at=dt.datetime.now(dt.timezone.utc).isoformat())
        self.state["identity_admitted"] = doc
        self.save()
        write_json(self.identity_dir() / "admitted.json", doc)
        os.environ[REFERENCE_ENV] = "%s:%s" % (self.commit, commit)
        return "admitted %d record dir(s) at %s (ref/%s); repacked %s" % (
            len(dirs), commit[:12], self.commit[:12], ", ".join(sorted(receipts)))

    def admit_records(self, dirs):
        """tools/admit_identity_columns.sh --build-host over DIRS in an admission tree at the freeze; the commit."""
        import reference_refresh
        tree = self.identity_dir() / "admit-tree"
        branch = "ref/" + self.commit[:12]
        if not (tree / ".git").exists():
            self.must(["git", "-C", str(self.src), "worktree", "add", "-B", branch, str(tree), self.commit],
                      what="the admission tree", log=self.rel / "identity-admit-tree.log")
        log = self.rel / "identity-admit.log"
        env = dict(PYTHON=sys.executable, EVIDENCE=str(self.identity_dir() / "admit-evidence"),
                   REF_TAG="ref-" + self.commit[:12])
        p = subprocess.run(["bash", "tools/admit_identity_columns.sh", "--build-host", *dirs], cwd=str(tree),
                           env=dict(os.environ, **env), capture_output=True, text=True)
        log.write_text((p.stdout or "") + (p.stderr or ""))
        paths = reference_refresh.parse_admit_next(p.stdout or "")
        if p.returncode != 0 or not paths:
            raise StepFailed("the admission refused (rc %s; %s)" % (p.returncode, log))
        self.must(["git", "-C", str(tree), "add", "--", *paths], what="staging the admission")
        self.must(["git", "-C", str(tree), "commit", "-q", "-m",
                   "Admit identity columns of release %s at %s (the one run on the built wheels)\n\n"
                   "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" % (self.version, self.commit[:12])],
                  what="committing the admission")
        self.must(["git", "-C", str(tree), "push", "-q", "-u", "origin", "%s:%s" % (branch, branch)],
                  what="pushing " + branch, log=self.rel / "identity-admit-push.log")
        return subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()

    def repack_dir(self):
        return self.identity_dir() / "repacked"

    def repack(self, table):
        import wheel_table_repack as W
        receipts = {}
        for platform, wheel in (("linux", self.split_final("linux")), ("macos", self.macos_wheel())):
            if not wheel:
                continue
            doc = W.repack(wheel, table, self.repack_dir() / platform)
            path = self.repack_dir() / platform / (Path(wheel).name + ".repack.json")
            path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
            receipts[platform] = str(path)
        return receipts

    def repacked_ok(self):
        done = self.admitted() or {}
        for platform, receipt in (done.get("repacked") or {}).items():
            doc = read_json(receipt) or {}
            out = Path((doc.get("output") or {}).get("path") or "/nonexistent")
            if not out.is_file() or sha256(out) != doc["output"]["sha256"]:
                return False
        return bool(done)

    def shipped_wheel(self, platform, wheel):
        """The bytes a publish ships: under --identity-on-wheels the repacked core / macOS wheel (its receipt names
        the gated input's sha256), every other wheel as built."""
        if not self.single() or wheel is None or platform not in ("linux", "macos"):
            return wheel
        done = self.admitted()
        if not done:
            raise StepFailed("--identity-on-wheels: identity-admit has not run; nothing is published before the one run is admitted")
        receipt = (done.get("repacked") or {}).get(platform)
        if not receipt:
            return wheel   # nothing recorded: the wheel ships as built
        doc = read_json(receipt) or {}
        if doc.get("input", {}).get("sha256") != sha256(wheel):
            raise StepFailed("the %s repack receipt was made from other bytes than the gated wheel" % platform)
        return Path(doc["output"]["path"])

    def reference_env(self):
        """Re-export the admission commit for the ledger (a resumed release reads it from the state)."""
        done = self.admitted()
        if self.single() and done:
            os.environ[REFERENCE_ENV] = "%s:%s" % (self.commit, done["commit"])
