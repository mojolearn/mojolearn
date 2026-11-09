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
import shutil
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



#: where the measurement smoke's own box job lands under a gate's out (tools/ops/release_gate.py --job measuresmoke)
#: where the measurement smoke's own box job (gone 2026-11-08) landed its summary; read only when the one gate job's
#: <gate out>/measuresmoke/ has none (a gate of an earlier freeze)
MEASURE_JOB_DIR = "measuresmoke-job"


def measure_smoke_summary(output):
    """<gate out>/measuresmoke/measuresmoke.json: the one gate job runs the smoke last and its final upload brings
    it home there (tools/ops/release_gate_job.sh phase measuresmoke)."""
    import release_measurement_smoke as MS
    path = Path(output) / "measuresmoke" / MS.SUMMARY
    legacy = Path(output) / MEASURE_JOB_DIR / "measuresmoke" / MS.SUMMARY
    return legacy if not path.is_file() and legacy.is_file() else path


def measure_smoke_errors(output):
    """['measuresmoke: ...'] unless the measurement smoke's own box job's summary passes
    (tools/release_measurement_smoke.py)."""
    import release_measurement_smoke as MS
    return ["measuresmoke: " + e for e in MS.summary_errors(read_json(measure_smoke_summary(output)))]


def measure_smoke_note(output):
    """'<passed>/<plan> recipes' of a passing measurement smoke summary (for the PASS line)."""
    import release_measurement_smoke as MS
    doc = read_json(measure_smoke_summary(output)) or {}
    c = doc.get("counts") or {}
    return "%s/%s recipes (%s unsupported)" % (c.get(MS.PASS), doc.get("plan_count"), c.get(MS.UNSUPPORTED, 0))

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
        # read defensively: partial Releases (tests' fakes, a publisher called directly) carry no args; the new
        # behaviour applies only when the option is explicitly on
        return bool(getattr(getattr(self, "args", None), "identity_on_wheels", False))

    # ------------------------------------------------------------------ fail soft (tools/release_verdicts.py)
    def strict(self):
        """--strict: every SOFT check is HARD (the behavior before 2026-11-08). Read defensively like single()."""
        return bool(getattr(getattr(self, "args", None), "strict", False))

    def soft_fail(self, vendor, name, reason, log=None):
        """Record one SOFT failure (soft-failures.json, the SOFT FAIL line, the notice); nothing in a dry run,
        a --status read or a partial Release without a release dir."""
        import release_verdicts as RV
        if getattr(self, "dry", False) or getattr(self, "readonly", False):
            return False
        try:
            rel = self.rel
        except (AttributeError, KeyError, TypeError):
            return False
        # the notice goes to <evidence root>/rel<ver>/ when the state is in its default place
        # (<evidence root>/release/<version>, the evidence root being ~/mojolearn-evidence or
        # MOJOLEARN_EVIDENCE_ROOT); any other state dir (--state-dir, the tests' fakes) keeps it under that dir
        base = getattr(self, "base", None)
        root = None if base is None else (Path(base).parent.parent
                                          if Path(base).name == self.version and Path(base).parent.name == "release"
                                          else Path(base))
        return RV.record(rel, self.version, vendor, name, reason, log=log, say=self.say,
                         evidence_root=root or Path(rel))

    def soft_judge(self, vendor, hard, soft):
        """(errors, soft_names): HARD errors stop the gate; each SOFT (name, reason, log) is recorded and lets it
        pass, unless --strict, which makes it an error again."""
        import release_verdicts as RV
        errors = list(hard)
        names = []
        for name, reason, log in soft:
            if RV.is_soft(name, self.strict()):
                self.soft_fail(vendor, name, reason, log)
                names.append(name)
            else:
                errors.append(reason)
        return errors, names

    def soft_failures(self):
        """Every SOFT failure recorded for this freeze (soft-failures.json), [] when none."""
        import release_verdicts as RV
        try:
            return RV.load(self.rel)
        except (AttributeError, KeyError, TypeError):
            return []

    def reuse(self):
        """--reuse-unchanged: opt in to reusing unchanged lanes, check units and runtime units. OFF by default
        (Andrew 2026-11-05: the selector cannot be trusted to narrow; the one run runs EVERYTHING)."""
        return self.single() and bool(getattr(getattr(self, "args", None), "reuse_unchanged", False))

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
        with the recorder's own reuse arguments (record_identity_column.sh REUSE=1), static, on this Mac.
        An unreadable selection owes every lane of the column (None means: not decidable, so owed)."""
        if self.dry:
            # A DRY RUN HASHES NO CLOSURES (2026-11-08): changed-lanes over every retained record took 20+ min per
            # column on the Mac (test_release's --reuse-unchanged dry run hit its 600 s bound in gate-apple). The
            # dry plan treats the selection as undecided, so it prints the commands the real run would decide by.
            return None
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
        if not self.reuse():
            return 0, "all"   # the full run: no selector decides
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
        plan = self.units_plan(vendor) if not ok else None
        composed = ""
        if self.reuse() and not ok and run == 0 and plan is not None and not plan["owed"] and not self.dry:
            # NOTHING OWED: no lane, no check unit; the runtime report is composed from reused unit receipts on
            # this Mac (release_runtime_installed.py run --reused-only, which launches nothing). If every runtime
            # unit is reused too, the box job's evidence exists without a box: no rental.
            composed = self.compose_without_box(vendor)
            ok, why = self.gate_ok(vendor)
        units = "" if plan is None else ("; units REUSED %d / RUN %d" % (plan["reused"], len(plan["owed"]))
                                         if self.reuse() else "; units RUN all")
        self.say("  %s: REUSED %d / RUN %s lane(s)%s; %s%s" % (
            vendor, reused, "?" if run is None else run, units,
            ("nothing owed, composed on this Mac" if composed == "composed" else "done for these wheel bytes")
            if ok else (composed or "owed on the box"), "; no rental" if ok else ""))
        return None if ok else why

    # ------------------------------------------------------------------ composing a vendor's evidence with no box
    def runtime_prior_dir(self, vendor):
        """A staged directory of the newest passing runtime unit receipts of VENDOR's device class, or None."""
        import runtime_qualifier_units as units
        gate = "gate-" + vendor if vendor != "cpu" else "gate-nvidia"
        sources = self.runtime_unit_sources(gate)
        if vendor == "apple":
            sources += [self.apple_runtime_store()] if self.apple_runtime_store().is_dir() else []
        if vendor == "cpu":
            sources += sorted(self.base.parent.glob("*/*/identity-wheels/cpu/artifacts/cpu-wheel-checks/runtime/units"))
        dest = self.identity_dir() / ("runtime-prior-compose-%s-%d" % (vendor, os.getpid()))
        staged = units.stage_prior(sources, dest, (vendor,))
        return dest if staged else None

    def matching_python(self, prior):
        """A local interpreter whose cache tag equals the reused receipts' (the reuse rule checks it), or None."""
        tags = {}
        for p in Path(prior).rglob("*.json"):
            tag = (read_json(p) or {}).get("interpreter_cache_tag")
            if tag:
                tags[tag] = tags.get(tag, 0) + 1
        if not tags:
            return None
        want = max(tags, key=tags.get)
        m = re.fullmatch(r"cpython-(\d)(\d+)", want)
        candidates = [sys.executable, self.smoke_python()] + (["python%s.%s" % m.groups()] if m else [])
        for py in candidates:
            try:
                got = subprocess.run([py, "-c", "import sys; print(sys.implementation.cache_tag)"],
                                     capture_output=True, text=True, timeout=60).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                continue
            if got == want:
                return py
        return None

    def compose_runtime(self, vendor, wheels, device, out):
        """True when OUT/runtime-report.json was composed from reused receipts only (nothing launched)."""
        prior = self.runtime_prior_dir(vendor)
        if prior is None:
            return False
        py = self.matching_python(prior)
        if py is None:
            return False
        tmp = Path(str(out) + ".compose-%d" % os.getpid())
        cmd = [sys.executable, "tools/release_runtime_installed.py", "run", "--vendor", vendor, "--source-root",
               str(self.src), "--source-commit", self.commit, "--python", py, "--device-class", device,
               "--prior-units", str(Path(prior) / device), "--reused-only", "--out", str(tmp)]
        for w in wheels:
            cmd += ["--wheel", str(w)]
        with open(self.identity_dir() / ("runtime-compose-%s.log" % vendor), "w") as log:
            rc = subprocess.run(cmd, cwd=str(self.src), stdout=log, stderr=subprocess.STDOUT).returncode
        report = read_json(tmp / "runtime-report.json") or {}
        if rc != 0 or report.get("status") != "passed":
            return False
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        if Path(out).exists():
            Path(out).rename(Path(str(out) + ".replaced-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")))
        tmp.rename(out)
        return True

    def compose_without_box(self, vendor):
        """'composed' when the vendor's identity summary, unit summary and runtime report were all written from
        reused evidence on this Mac (nothing owed anywhere), else why not."""
        import release_check_units as U
        plan = self.units_plan(vendor)
        output = self.gate_out(vendor)
        if not self.compose_runtime(vendor, self.candidate_gate_wheels(vendor), self.device_class(vendor),
                                    output / ("runtime-" + vendor)):
            return "runtime units owed (or no reusable receipts on this Mac): the box job runs them"
        cols = {c: dict(label=c, owed=0, reused=self.__dict__.get("_lanes_owed", {}).get(c + ":reused"), chunks=[], rc=0)
                for c in self.vendor_columns(vendor)}
        write_json(output / "identity" / SUMMARY, dict(schema="mojolearn.identity-wheels.v1", ok=True, columns=cols,
                                                       composed="nothing owed: every lane reused"))
        U.compose_reused(plan, output / "units", self.commit)
        return "composed"


    # ------------------------------------------------------------------ the device checks as units
    def unit_store(self):
        return self.base.parent / "check-units"

    def device_class(self, vendor):
        return {"nvidia": "nvidia-sm_89", "amd": "amd-gfx942", "cpu": "cpu-linux-x86_64"}.get(
            vendor) or self.apple_device_class()

    def units_plan_path(self, vendor):
        return self.identity_dir() / ("units-%s.json" % vendor)

    def units_plan(self, vendor):
        """tools/release_check_units.py plan for VENDOR's wheels at the frozen source (static, on this Mac): which
        device-check units are owed and which reuse an earlier record of the same key. None: undecidable."""
        cache = self.__dict__.setdefault("_units_plan", {})
        if vendor in cache:
            return cache[vendor]
        # the CPU column installs the core wheel alone (no cpu split package: SPLIT_PACKAGES has none, and
        # candidate_gate_wheels("cpu") raised KeyError at identity-cpu in 0.8.66 run e, 2026-11-07)
        wheels = ([self.macos_wheel()] if vendor == "apple" else [Path(self.split_final("linux"))] if vendor == "cpu"
                  else self.candidate_gate_wheels(vendor))
        out = self.units_plan_path(vendor)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "tools/release_check_units.py", "plan", "--vendor", vendor,
               "--device-class", self.device_class(vendor), "--source-root", str(self.src),
               "--store", str(self.unit_store()), "--json", str(out)] + (["--reuse-unchanged"] if self.reuse() else [])
        for w in wheels:
            cmd += ["--wheel", str(w)]
        with open(self.identity_dir() / ("units-%s.log" % vendor), "w") as log:
            rc = subprocess.run(cmd, cwd=str(self.src), stdout=log, stderr=subprocess.STDOUT).returncode
        cache[vendor] = read_json(out) if rc == 0 else None
        return cache[vendor]

    def keep_unit_records(self, root):
        """Every unit record a run brought home (<root>/records/<class>/<unit>/<key12>.json) into the store."""
        import shutil
        kept = 0
        for p in sorted(Path(root).glob("records/*/*/*.json")):
            dest = self.unit_store() / p.relative_to(Path(root) / "records")
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
                kept += 1
        return kept

    # ------------------------------------------------------------------ the box job
    def gate_identity_args(self, vendor):
        if not self.single():
            return []
        args = ["--identity-columns", ",".join(self.vendor_columns(vendor)), "--label-device", self.label_device(vendor)]
        if self.reuse():
            args.append("--reuse-unchanged")
        if not self.dry and self.units_plan(vendor) is not None:
            args += ["--units-plan", str(self.units_plan_path(vendor))]
        return args

    def single_gate_ok(self, vendor, output, measure=None):
        """(ok, why) of the one box job under --identity-on-wheels: the identity summary of its columns, every owed
        device-check unit passed (the rest reused), and the installed-runtime qualifier's report. MEASURE: judge the
        measurement smoke's summary here too; None (default) = only under --strict (otherwise the smoke arrives after
        the verdict and tools/release.py judge_measure_smoke records it)."""
        if measure is None:
            measure = self.strict()
        import release_check_units as U
        output = Path(output)
        # HARD (identity, install-load, shardcheck) stop the gate; SOFT (autograd/*, adapters, the measurement
        # smoke, the runtime qualifier) are recorded and notified (tools/release_verdicts.py CHECK_CLASS)
        errors = ["identity: " + e for e in summary_errors(read_json(output / "identity" / SUMMARY),
                                                           self.vendor_columns(vendor))]
        soft = []
        plan = read_json(self.units_plan_path(vendor))
        for e in U.summary_errors(read_json(output / "units" / U.SUMMARY), plan):
            m = re.fullmatch(r"unit (\S+) did not pass", e)
            if m:
                soft.append((m.group(1), "units: " + e, output / "units"))
            else:
                errors.append("units: " + e)
        # THE ONE-GPU MULTI-DEVICE GATE, UNCONDITIONAL (Andrew 2026-11-06): every box job runs
        # tools/logical_shard_identity.py on these wheel bytes whatever changed; no reuse or selector waives it.
        import logical_shard_identity as LS
        errors += ["shardcheck: " + e for e in LS.summary_errors(read_json(Path(output) / "shardcheck" / LS.SUMMARY))]
        # THE MEASUREMENT SMOKE, UNCONDITIONAL (Andrew 2026-11-06): every box job runs every measurement recipe once at
        # its smoke shape, ours only, on these wheel bytes (tools/release_measurement_smoke.py); nothing waives it.
        # It runs LAST in the one gate job (2026-11-08) and its summary comes home after the HARD parts: judged here
        # only under --strict (measure); otherwise tools/release.py judge_measure_smoke records it when it lands.
        smoke = measure_smoke_errors(output) if measure else []
        if smoke:
            soft.append(("measuresmoke", "; ".join(smoke), measure_smoke_summary(output)))
        runtime, cases = self.gate_runtime_error(vendor, output)
        if runtime:
            soft.append(("runtime", runtime, output / ("runtime-" + vendor) / "runtime-report.json"))
        errors, softened = self.soft_judge(vendor, errors, soft)
        if errors:
            return False, "; ".join(errors)
        self.keep_unit_records(output / "units")
        return True, ("PASS: identity recorded once from these wheels; device-check units passed or reused; "
                      "logical-shard gate MATCH; measurement smoke %s; %s runtime cases%s"
                      % ("SOFT FAIL" if "measuresmoke" in softened else measure_smoke_note(output) if measure
                         else "judged when it lands", cases,
                         ("; SOFT FAIL (fix lanes owed): " + ", ".join(softened)) if softened else ""))

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
        if self.reuse() and self.lanes_owed("apple") == []:
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
                   "--out", str(out), "--python", self.smoke_python() if not self.dry else "python3.12"]
                  + (["--reuse-unchanged"] if self.reuse() else []),
                  log=self.rel / "gate-apple-identity.log", what="the Apple column from the macOS wheel", cwd=self.src)
        if self.dry:
            return None
        ok, why = self.apple_identity_ok()
        if not ok:
            raise StepFailed("the Apple column did not record: " + why)
        return why

    # ------------------------------------------------------------------ Apple: adapters and runtime-apple as units
    def apple_adapter_plan(self):
        """The units plan of the macOS wheel, restricted to the adapter unit (what gate-apple runs)."""
        plan = self.units_plan("apple")
        if plan is None:
            return None
        rows = [r for r in plan["units"] if r["unit"] == "adapters"]
        return dict(plan, units=rows, owed=[r["unit"] for r in rows if r["decision"] == "run"],
                    reused=sum(r["decision"] == "reuse" for r in rows))

    def apple_adapter_unit_ok(self):
        plan = self.apple_adapter_plan()
        return bool(plan) and not plan["owed"]

    def apple_adapter_unit_gate(self):
        """The Apple adapter smoke only when its unit is owed; a PASS writes the unit's record."""
        import release_check_units as U
        plan = self.apple_adapter_plan()
        if plan is not None and not plan["owed"]:
            return "adapters REUSED (unit record of the same key on %s)" % plan["device_class"]
        result = self.apple_adapter_gate()
        if plan is not None and not self.dry and self.apple_adapters_ok()[0]:
            U.record_passed(self.unit_store(), plan, "adapters", self.commit,
                            dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        return result

    def apple_runtime_reused(self):
        """runtime-apple composed from reused receipts without the Metal lock (nothing launched), or None.
        Only with --reuse-unchanged: the default full run runs every unit."""
        if self.dry or not self.reuse() or self.apple_runtime_ok()[0]:
            return None
        w = self.macos_wheel()
        if not w:
            return None
        if self.compose_runtime("apple", [w], self.apple_device_class(), self.apple_runtime_out()):
            ok, why = self.apple_runtime_ok()
            if ok:
                return "runtime-apple REUSED (every unit's receipt; nothing ran): " + why
        return None

    # ------------------------------------------------------------------ CPU on GitHub
    def cpu_out(self):
        return self.identity_dir() / "cpu"

    def cpu_ok(self):
        out = self.cpu_out()
        reused = read_json(out / "reused.json") or {}
        core = self.split_final("linux")
        if (self.reuse() and reused.get("source_commit") == self.commit and core and reused.get("wheel_sha256") == sha256(core)
                and (read_json(out / "runtime-cpu" / "runtime-report.json") or {}).get("status") == "passed"):
            return True, "cpu: REUSED %s lane(s) / RUN 0, units REUSED %s / RUN 0, runtime composed; no dispatch" % (
                reused.get("lanes_reused"), reused.get("units_reused"))
        import release_verdicts as RV
        doc = read_json(out / "run.json") or {}
        # a run that concluded "failure" is judged by its artifacts unless --strict: install, correctness and the
        # column shards are HARD; the runtime, autograd and adapter groups SOFT (tools/release_verdicts.py)
        if doc.get("source_commit") != self.commit or doc.get("conclusion") not in (
                ("success",) if self.strict() else ("success", "failure")):
            return False, "no successful %s run for %s" % (CPU_WORKFLOW, self.commit[:12])
        core = self.split_final("linux")
        if not core or doc.get("wheel_sha256") != sha256(core):
            return False, "the CPU run checked another core wheel"
        checks_path = out / "artifacts" / "cpu-wheel-checks" / "checks.json"
        checks = read_json(checks_path) or {}
        groups = checks.get("checks") or {}
        bad = [k for k, v in groups.items() if v != 0]
        if not groups or "install" not in groups:
            return False, "CPU checks failed or missing: %s" % (", ".join(bad) or "no checks.json")
        hard = ["cpu %s rc %s" % (k, groups[k]) for k in bad if not RV.is_soft("cpu/" + k, self.strict())]
        for shard in range(doc.get("shards", 0)):
            errors = summary_errors(read_json(out / "artifacts" / ("cpu-wheel-identity-%d" % shard) / SUMMARY), ["cpu"])
            if errors:
                hard.append("cpu column shard %d: %s" % (shard, "; ".join(errors)))
        if hard:
            return False, "CPU checks failed: " + "; ".join(hard)
        soft = [k for k in bad if RV.is_soft("cpu/" + k, self.strict())]
        for k in soft:
            self.soft_fail("cpu", "cpu/" + k, "CPU check group %s exited %s (run %s)" % (k, groups[k], doc.get("run_id")),
                           out / "artifacts" / "cpu-wheel-checks" / k / "log.txt")
        if doc.get("conclusion") != "success":
            self.soft_fail("cpu", "cpu/job-exit", "%s run %s concluded %s; its HARD parts passed" % (
                CPU_WORKFLOW, doc.get("run_id"), doc.get("conclusion")), checks_path)
        self._cpu_soft_groups = set(soft)
        return True, "cpu column and CPU checks PASSED on GitHub (run %s)%s" % (
            doc.get("run_id"), ("; SOFT FAIL (fix lanes owed): " + ", ".join(soft)) if soft else "")

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
        # THE UNIT RULE ON THE CPU COLUMN: nothing owed (no cpu lane, no cpu unit, every runtime-cpu unit reused)
        # -> no dispatch at all; otherwise the run carries only the owed check groups
        lanes = self.lanes_owed("cpu") if not self.dry and self.reuse() else None
        plan = self.units_plan("cpu") if core and not self.dry else None
        if (self.reuse() and not self.dry and lanes == [] and plan is not None and not plan["owed"]
                and self.compose_runtime("cpu", [core], self.device_class("cpu"), out / "runtime-cpu")):
            write_json(out / "reused.json", dict(source_commit=self.commit, wheel_sha256=sha256(core),
                                                 lanes_reused=self.__dict__.get("_lanes_owed", {}).get("cpu:reused"),
                                                 units_reused=plan["reused"]))
            ok, why = self.cpu_ok()
            if ok:
                return why
        owed = (plan or {}).get("owed") or []
        groups = (sorted({"correctness" if u == "install-load" else "adapters" if u == "adapters" else "autograd"
                          for u in owed} | {"runtime"}) if plan is not None
                  else ["adapters", "autograd", "correctness", "runtime"])   # no plan: every group runs
        entries = ",".join(u.split("/", 1)[1] for u in owed if u.startswith("autograd/"))
        if plan is not None and not self.reuse():
            self.say("  cpu: RUN all lanes, all units (%s)" % ",".join(groups))
        elif plan is not None:
            self.say("  cpu: REUSED %s / RUN %s lane(s); units REUSED %d / RUN %d (%s)" % (
                self.__dict__.get("_lanes_owed", {}).get("cpu:reused"), "?" if lanes is None else len(lanes),
                plan["reused"], len(plan["owed"]), ",".join(groups)))
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
                      runtime_prior_url=prior_url, token=token, checks=",".join(groups), autograd_entries=entries,
                      reuse_unchanged="true" if self.reuse() else "false")
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
        # a fresh dir: `gh run download` into the previous run's extraction fails on existing files and leaves the
        # stale checks.json in place (0.8.66 run g, 2026-11-07: the job PASSED, the release read run f's failures)
        if art.exists():
            shutil.rmtree(art)
        art.mkdir(parents=True, exist_ok=True)
        self.gh("run", "download", run_id, "-D", str(art), log=self.rel / "identity-cpu.log")
        out.mkdir(parents=True, exist_ok=True)
        (out / "run.json").write_text(json.dumps(dict(run_id=run_id, url=link, conclusion=conclusion,
                                                      source_commit=self.commit, wheel_sha256=fields["wheel_sha256"],
                                                      shards=CPU_SHARDS, token=token), indent=1) + "\n")
        ok, why = self.cpu_ok()
        if not ok:
            raise StepFailed("the CPU run did not pass: %s (%s)" % (why, link))
        if plan is not None:
            import release_check_units as U
            at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            failed = getattr(self, "_cpu_soft_groups", set())
            for unit in plan["owed"]:
                group = "correctness" if unit == "install-load" else "adapters" if unit == "adapters" else "autograd"
                if group not in failed:   # a SOFT-failed group's units stay owed for the next run
                    U.record_passed(self.unit_store(), plan, unit, self.commit, at)
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
            doc = dict(source_commit=self.commit, commit=self.commit, recorded=0, at=dt.datetime.now(dt.timezone.utc).isoformat(),
                       soft_failures=self.soft_failures())
            # the facts the steps after this one ship by (Release.publication_gate): never a ledger evaluation
            doc["wheels"] = self.admitted_wheels() if hasattr(self, "admitted_wheels") else {}
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
                   table_sha256=sha256(table), repacked=receipts, at=dt.datetime.now(dt.timezone.utc).isoformat(),
                   soft_failures=self.soft_failures())
        # the wheels as repacked: what the steps after this one ship and compare to (Release.publication_gate)
        doc["wheels"] = self.admitted_wheels() if hasattr(self, "admitted_wheels") else {}
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
        # the verifier runs from the installed macOS wheel (the Apple column's venv) when it exists: every binding,
        # the exact bytes; the source tree alone needs host bindings this script never built (0.8.66 run k)
        # (the installed package itself cannot emit: _identity._checkout_root() walks up from the package dir for
        # tools/identity_break.py, so the verifier runs from the frozen tree WITH the wheel's compiled artifacts placed in it)
        packages = sorted((self.identity_dir() / "apple" / "venv" / "lib").glob("python3.*/site-packages/mojolearn/__init__.py"))
        if packages:
            env["ADMIT_WHEEL_PACKAGE"] = str(packages[-1].parent)
        # the script is the tooling's (HEAD), copied into the frozen tree and run there (it cds to its own root):
        # the source is frozen, the tooling runs from HEAD; the copy is never committed (only the NEXT paths are)
        shutil.copy2(ROOT / "tools" / "admit_identity_columns.sh", tree / "tools" / "admit_identity_columns.sh")
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
