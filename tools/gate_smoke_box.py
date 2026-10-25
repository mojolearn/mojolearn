# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Bounded publication qualification, or an explicitly selected performance campaign.

Default --profile release-light-v1 installs exact wheels without opponents,
completes eleven installed correctness records (nine checks plus location and
pip dependency checks), 24 autograd entries and three framework adapter phases.
A 30-minute total ceiling includes separately timed installation; timeout,
missing artifact and incomplete test coverage fail. No compilation occurs here.

--profile performance-campaign-v1 preserves the historical whole-board startup
smoke and full-dataset opponent campaign. Its performance evidence remains
required for performance claims and default promotions, not publication.
Nothing here rents, deletes, publishes or replaces historical result directories.
"""

import glob
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import release_gate_profiles as P

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STAGES = ("setup", "smoke", "autograd", "adapters", "measure")
#: measure-stage wall budget (hours) before no new line starts; MOJOLEARN_GATE_MEASURE_HOURS overrides
MEASURE_HOURS = 30.0
#: the box's persistent opponent store (scored once per box key, reused by every later gate on that box);
#: MOJOLEARN_GATE_OPPONENT_STORE overrides
STORE_DIR = os.path.join(os.path.expanduser("~"), "mojolearn-gate-store")


def gate_verdict(state):
    """(verdict 'PASS' | 'FAIL' | 'INCOMPLETE', [reasons]) of a gate-smoke.json document."""
    try:
        selected = P.profile(state)
    except ValueError as error:
        return "FAIL", [str(error)]
    if selected == P.LIGHT:
        return light_verdict(state)
    why = []
    for stage in ("smoke", "autograd", "measure"):
        if state.get(stage + "_finished") is None:
            why.append("%s stage not finished" % stage)
    counts = state.get("counts") or {}
    if state.get("smoke_finished") is not None:
        if state.get("smoke_rc") not in (0, None) or counts.get("FAILED"):
            why.append("smoke: rc=%s FAILED=%s" % (state.get("smoke_rc"), counts.get("FAILED", 0)))
        if not state.get("smoke_cpu_fallback"):
            why.append("smoke did not cover the whole plan (no --cpu-fallback)")
    if state.get("autograd_finished") is not None and state.get("autograd_rc") != 0:
        why.append("autograd: rc=%s" % state.get("autograd_rc"))
    # 2026-10-24: the installed-wheel adapter smoke is a gate stage; only PASSED passes
    if state.get("adapters_enabled", True):
        adapters = state.get("adapters") or {}
        if state.get("adapters_finished") is None:
            why.append("adapters stage not finished")
        elif not adapters.get("ok") or adapters.get("status") != "PASSED":
            why.append("adapters: %s" % (adapters.get("why") or adapters.get("status")))
    m = state.get("measure") or {}
    if state.get("measure_finished") is not None:
        if state.get("measure_rc") != 0:
            why.append("measure: rc=%s" % state.get("measure_rc"))
        if m.get("pending"):
            why.append("measure: %d line(s) never started" % m["pending"])
        if m.get("failed"):
            why.append("measure: %d line(s) failed: %s" % (len(m["failed"]), ", ".join(m["failed"][:10])))
        if m.get("expected") is not None and m.get("expected") != m.get("planned_whole_plan"):
            why.append("measure: %s of %s plan lines selected" % (m.get("expected"), m.get("planned_whole_plan")))
        # 2026-10-22: every opponent pair MATCHED; a refusal is listed, never silently dropped
        pairs = m.get("pairs")
        if m.get("board") and pairs is None:
            why.append("measure: gate board has no pair check recorded")
        # measure-coverage (2026-10-24): a line whose own arm did not import or construct is a failure, not a skip
        for row in m.get("own_not_built") or []:
            why.append("measure: own arm not built %s %s: %s" % (row.get("race"), row.get("arm"), row.get("reason")))
        for state in ("refused", "unchecked"):
            for p in (pairs or {}).get(state) or []:
                why.append("measure: opponent pair %s %s %s: %s" % (state, p.get("race"), p.get("arm"), p.get("reason")))
    if any(w.endswith("not finished") for w in why):
        return "INCOMPLETE", why
    return ("FAIL" if why else "PASS"), why


def light_verdict(state):
    why = []
    for stage in P.LIGHT_STAGES:
        if state.get(stage + "_finished") is None:
            why.append(stage + " stage not finished")
    identity = state.get("candidate") or {}
    why += P.correctness_errors(state.get("correctness"), identity.get("source_commit") or (state.get("correctness") or {}).get("source_commit"),
                                state.get("version"), state.get("vendor"))
    if state.get("correctness_rc") != 0:
        why.append("installed correctness process failed")
    import autograd_smoke as AS
    if state.get("autograd_rc") != 0 or state.get("autograd_counts") != {"ok": len(AS.ENTRIES)}:
        why.append("autograd requires every registry entry to complete; timeout/STARTED is not a pass")
    adapters = state.get("adapters") or {}
    import adapter_gate as AG
    if state.get("adapters_enabled", True) and (adapters.get("ok") is not True
            or adapters.get("status") != "PASSED" or adapters.get("rc") != 0
            or adapters.get("phases") != {name: 0 for name in AG.PHASES}):
        why.append("installed adapters did not complete all three phases")
    budget = state.get("budget") or {}
    if (budget.get("scope") != "all" or budget.get("limit_seconds") != P.LIGHT_BUDGET_SECONDS or budget.get("timed_out") is not False
            or not P.finite_seconds(budget.get("elapsed_seconds"), P.LIGHT_BUDGET_SECONDS)
            or state.get("status") == "failed"):
        why.append("light profile exceeded its deadline or has no completed budget record")
    if state.get("status") != "failed" and any(reason.endswith("not finished") for reason in why):
        return "INCOMPLETE", why
    return ("FAIL" if why else "PASS"), why


def own_not_built(board):
    """[{race, arm, reason}] for every own cell of a gate board that did not run because our arm did not import or
    construct (bench_board.ALGOS_SKIPPED, "SKIPPED: not built yet", with the driver's skip reason)."""
    out = []
    for group in ("races", "extra_races"):
        for rid, rr in sorted((board.get(group) or {}).items()):
            for c in rr.get("cells") or []:
                if (c.get("library") == "mojolearn" or str(c.get("arm", "")).startswith("ours")) and \
                        str(c.get("status") or "").startswith("SKIPPED"):
                    out.append({"race": rid, "arm": c.get("arm"), "reason": c.get("skip_reason") or c.get("status")})
    return out


class Gate:
    def __init__(self, vendor, version, root, candidate_wheels=(), source_commit=None, autograd_manifest=None,
                 adapters=True, profile=P.LIGHT):
        import bench_board as B
        import bench_board_evidence as E
        self.B, self.E = B, E
        self.profile = P.profile({"profile": profile})
        self.deadline = None
        self.vendor, self.version, self.root = vendor, version, Path(root)
        self.candidate_wheels, self.source_commit = list(candidate_wheels), source_commit
        self.autograd_manifest = autograd_manifest
        self.adapters_on = adapters
        self.root.mkdir(parents=True, exist_ok=True)
        self.out, self.cache = self.root / "setup", str(self.root / "cache")
        (self.out / "logs").mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "gate-smoke.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        if self.state.get("version") not in (None, version) or self.state.get("vendor") not in (None, vendor):
            raise SystemExit("gate: %s holds another gate (%s %s); use a new out-root"
                             % (self.state_path, self.state.get("vendor"), self.state.get("version")))
        mode, candidate = 'published', None
        if self.candidate_wheels:
            import candidate_wheel_install as C
            mode = 'candidate'
            candidate = C.identity(C.build_manifest(version, vendor, self.candidate_wheels, source_commit))
            if not autograd_manifest:
                raise SystemExit('candidate gate requires an eight-binary autograd artifact manifest')
            from gate_autograd_artifacts import validate
            self.autograd_artifacts = validate(autograd_manifest, vendor, source_commit, self.root/'autograd-bin')
        if self.state and P.profile(self.state) != self.profile:
            raise SystemExit("gate output belongs to another profile; use a new out-root")
        if self.state and (self.state.get('input_mode', 'published') != mode or self.state.get('candidate') != candidate):
            raise SystemExit('gate output belongs to different published/candidate artifacts; use a new out-root')
        self.state.update(input_mode=mode, candidate=candidate, profile=self.profile)
        if mode == 'candidate':
            self.state['autograd_artifacts'] = self.autograd_artifacts
        self.state.update(vendor=vendor, version=version, source_commit=B.repo_commit())

    def mark(self, **fields):
        self.state.update(fields)
        self.E.write(self.state_path, self.state)

    def amd_env(self):
        if self.vendor == "amd":
            os.environ["LD_LIBRARY_PATH"] = ":".join(
                glob.glob(self.cache + "/venv/lib/python*/site-packages/*/lib64")
                + glob.glob(str(self.root / "venv/lib/python*/site-packages/*/lib64"))
                + ["/opt/rocm-7.2.3/lib", "/opt/rocm-6.4.0/lib"])

    def env(self):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        if self.deadline is not None:
            env["MOJOLEARN_GATE_DEADLINE"] = str(self.deadline - 2)  # let nested runners kill their groups before our ceiling
        return env

    # -------------------------------------------------------------- stages
    def setup(self):
        if self.profile == P.LIGHT:
            return self.setup_light()
        B, out = self.B, self.out
        self.mark(phase="setup", status="running", setup_started=time.time())
        argv = ["--vendor", self.vendor, "--out", str(out), "--cache", self.cache,
                                            "--families", "neural,algos,classical,classical2,trees",
                                            "--mojolearn-version", self.version]
        if self.candidate_wheels:
            argv += ['--candidate-source-commit', self.source_commit]
            for path in self.candidate_wheels:
                argv += ['--candidate-wheel', path]
        args = B.build_parser().parse_args(argv)
        python, wheel = B.setup_python(args, self.vendor, str(out), str(out / "logs/install.log"))
        manifest = getattr(args, "wheel_manifest", None) or str(out / "wheel-manifest.json")
        with (out / "logs/neural-pins.log").open("w") as f:
            pins = subprocess.run([python, "-m", "pip", "install", "--disable-pip-version-check"] + list(B.NEURAL_PINS),
                                  stdout=f, stderr=f)
        if pins.returncode:
            self.mark(status='failed', error='neural opponent dependency installation failed')
            raise SystemExit('gate: neural dependency installation failed; see setup/logs/neural-pins.log')
        self.amd_env()
        refusal = B.gpu_set_refusal(python, self.vendor)
        if refusal:
            self.mark(status="failed", error="gpu set: " + refusal)
            raise SystemExit(refusal)
        _, refusal = B.PYPI.verify(manifest, python)
        if refusal:
            self.mark(status="failed", error="pypi verify: " + refusal)
            raise SystemExit(refusal)
        with (out / "logs/pip-freeze.log").open("w") as f:
            subprocess.run([python, "-m", "pip", "freeze", "--all"], stdout=f, stderr=f)
        self.mark(phase="setup", status="ready", python=python, wheel=wheel, wheel_manifest=manifest,
                  setup_finished=time.time())

    def remaining(self):
        return max(0.0, self.deadline - time.time()) if self.deadline else P.LIGHT_BUDGET_SECONDS

    def launch_light(self, command, log):
        result = P.bounded_run(command, log, self.remaining(), env=self.env(), cwd=HERE.parent)
        self.state.setdefault("commands", []).append(dict(command=list(map(str, command)), log=str(log), **result))
        self.mark()
        return result["exit_code"]

    def setup_light(self):
        # Exact candidate installer, without bench_board's opponent environment.
        # Torch/ONNX here are test dependencies for the importer probes; they
        # are pinned by the shipped adapter manifest, never opponent versions.
        self.mark(phase="setup", status="running", setup_started=time.time())
        python = str(self.root / "venv/bin/python")
        manifest = str(self.out / "wheel-manifest.json")
        commands = [[sys.executable, "-m", "venv", str(self.root / "venv")]]
        if self.candidate_wheels:
            command = [sys.executable, str(HERE / "candidate_wheel_install.py"), "--python", python,
                       "--version", self.version, "--vendor", self.vendor, "--source-commit", self.source_commit,
                       "--manifest", manifest, "--log", str(self.out / "logs/install.log")]
            for wheel in self.candidate_wheels:
                command += ["--wheel", wheel]
        else:
            command = [sys.executable, str(HERE / "pypi_wheel_install.py"), "install", "--python", python,
                       "--version", self.version, "--vendor", self.vendor, "--manifest", manifest,
                       "--dir", str(self.out / "wheels")]
        commands += [command, [python, "-m", "pip", "install", "--disable-pip-version-check",
                               "numpy", "pytest", "onnx", "onnxscript"]]
        for command in commands:
            if self.launch_light(command, self.out / "logs/install.log"):
                raise RuntimeError("light gate setup failed; see setup/logs/install.log")
        import adapter_wheel_smoke as AW
        probe_log = self.out / "logs/framework-manifest.log"
        if self.launch_light([python, "-c", AW.PROBE], probe_log):
            raise RuntimeError("installed framework manifest probe failed")
        facts = json.loads(probe_log.read_text().splitlines()[-1])
        installs = AW.install_commands(python, facts.get("adapters") or {}, ["torch"])
        if "install-torch" not in installs:
            raise RuntimeError("installed wheel lacks the Torch test dependency pin")
        if self.launch_light(installs["install-torch"], self.out / "logs/install.log"):
            raise RuntimeError("pinned importer test dependency installation failed")
        self.mark(phase="setup", status="ready", python=python, wheel_manifest=manifest,
                  setup_finished=time.time(), setup_elapsed_seconds=time.time() - self.state["setup_started"])

    def correctness(self):
        self.amd_env()
        self.mark(phase="correctness", status="running", correctness_started=time.time(), correctness_finished=None)
        output = self.root / ("correctness-" + str(time.time_ns()))
        command = [sys.executable, str(HERE / "qualify_verifier_wheel.py"), "--installed-python", self.state["python"],
                   "--expected-version", self.version, "--scope", "expanded", "--output", str(output)]
        if self.source_commit:
            command += ["--expected-source-commit", self.source_commit]
        rc = self.launch_light(command, self.root / "correctness.log")
        report = output / "results.json"
        document = json.loads(report.read_text()) if report.is_file() else {}
        self.mark(correctness_rc=rc, correctness=document, correctness_report=str(report),
                  correctness_finished=time.time(), status="done")

    def run_light(self, step):
        started = time.time()
        self.deadline = started + P.LIGHT_BUDGET_SECONDS
        self.mark(budget=dict(scope=step, limit_seconds=P.LIGHT_BUDGET_SECONDS, started=started, timed_out=False))
        try:
            for stage in P.LIGHT_STAGES:
                if step in (stage, "all"):
                    if self.remaining() <= 0:
                        raise TimeoutError("release-light total deadline expired")
                    before = time.time()
                    getattr(self, stage)()
                    timing = self.state.setdefault("stage_timing", {})
                    timing[stage] = dict(elapsed_seconds=time.time() - before,
                                         category="installation" if stage == "setup" else "installation-and-execution" if stage == "adapters" else "execution")
                    self.mark()
                    # Do not spend the remaining budget repeatedly exercising
                    # a runtime whose installed correctness check has failed.
                    if stage == "correctness" and (self.state.get("correctness_rc") != 0
                            or (self.state.get("correctness") or {}).get("status") != "PASSED"):
                        raise RuntimeError("installed correctness failed; remaining stages not run")
                    if stage == "autograd":
                        import autograd_smoke as AS
                        if self.state.get("autograd_rc") != 0 or self.state.get("autograd_counts") != {"ok": len(AS.ENTRIES)}:
                            raise RuntimeError("autograd did not complete; remaining stages not run")
                    if stage == "adapters" and self.adapters_on and not (self.state.get("adapters") or {}).get("ok"):
                        raise RuntimeError("installed adapters failed")
        except (Exception, SystemExit) as error:
            self.mark(status="failed", error=str(error))
        finally:
            elapsed = time.time() - started
            self.mark(budget=dict(scope=step, limit_seconds=P.LIGHT_BUDGET_SECONDS, started=started,
                                  elapsed_seconds=elapsed, timed_out=elapsed > P.LIGHT_BUDGET_SECONDS))
        return self.finish() if step == "all" else int(self.state.get("status") == "failed")

    def smoke(self):
        self.amd_env()
        python, manifest = self.state["python"], self.state["wheel_manifest"]
        smoke_out = self.root / ("smoke-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
        self.mark(phase="smoke", status="running", smoke_started=time.time(), smoke_finished=None)
        # --cpu-fallback (2026-10-22): the WHOLE plan, CPU-opponent and ours-only lines included
        cmd = [python, str(HERE / "gpu_gap_campaign.py"), "--vendor", self.vendor, "--out", str(smoke_out),
               "--cache", self.cache, "--python-env", python, "--wheel-manifest", manifest,
               "--smoke", "--smoke-verify-data", "--cpu-fallback"]
        with (self.root / "smoke.log").open("wb") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=self.env(),
                                cwd=str(HERE.parent)).returncode
        report = smoke_out / "smoke" / "smoke.json"
        doc = json.loads(report.read_text()) if report.exists() else {}
        rows = doc.get("recipes") or doc.get("rows") or []
        if isinstance(rows, dict):
            rows = list(rows.values())
        counts = Counter(str(r.get("status")) for r in rows if isinstance(r, dict))
        self.mark(phase="smoke", status="done", smoke_rc=rc, smoke_report=str(report), counts=dict(counts),
                  smoke_cpu_fallback=True, smoke_lines=sum(counts.values()), smoke_finished=time.time())
        print("GATE_SMOKE vendor=%s rc=%s lines=%d %s" % (self.vendor, rc, sum(counts.values()), " ".join(
            "%s=%d" % kv for kv in sorted(counts.items()))))

    def autograd(self):
        self.amd_env()
        import autograd_smoke as AS
        report = self.root / "autograd-smoke.json"
        self.mark(phase="autograd", status="running", autograd_started=time.time(), autograd_finished=None)
        arc = AS.run(self.vendor, str(self.root / "autograd-bin"), self.state["python"], str(report),
                     timeout=P.AUTOGRAD_TIMEOUT_SECONDS if self.profile == P.LIGHT else AS.TIMEOUT_S,
                     deadline=self.deadline)
        doc = json.loads(report.read_text()) if report.exists() else {}
        self.mark(phase="autograd", status="done", autograd_rc=arc, autograd_report=str(report),
                  autograd_counts=doc.get("counts"), autograd_finished=time.time())
        print("GATE_SMOKE_AUTOGRAD vendor=%s rc=%s %s" % (self.vendor, arc, " ".join(
            "%s=%d" % kv for kv in sorted((doc.get("counts") or {}).items()))))

    def adapter_installs(self):
        """What the adapter venv installs: the candidate wheel files, or the published release from PyPI."""
        if self.candidate_wheels:
            return list(self.candidate_wheels)
        plugin = {"nvidia": "mojolearn-nvidia", "amd": "mojolearn-amd"}[self.vendor]
        return ["mojolearn-core==" + self.version, "%s==%s" % (plugin, self.version)]

    def adapters(self):
        import adapter_gate as AG
        if not self.adapters_on:
            self.mark(phase="adapters", status="done", adapters_enabled=False,
                      adapters={"enabled": False, "status": "DISABLED"}, adapters_finished=time.time())
            print("GATE_ADAPTERS vendor=%s status=DISABLED (MOJOLEARN_RELEASE_ADAPTERS=0)" % self.vendor)
            return
        self.mark(phase="adapters", status="running", adapters_enabled=True, adapters_started=time.time(),
                  adapters_finished=None)
        env = self.env()
        if self.vendor == "amd":
            env["LD_LIBRARY_PATH"] = ":".join(
                glob.glob(str(self.root / "adapter-smoke/venv/lib/python*/site-packages/*/lib64"))
                + ["/opt/rocm-7.2.3/lib", "/opt/rocm-6.4.0/lib"])
        options = {"launch": self.launch_light} if self.profile == P.LIGHT else {}
        rec = AG.run(self.vendor, sys.executable, self.root / "adapter-smoke", self.adapter_installs(), env=env, **options)
        self.mark(phase="adapters", status="done", adapters=rec, adapters_finished=time.time())
        print("GATE_ADAPTERS vendor=%s status=%s ok=%s %s" % (self.vendor, rec.get("status"), rec.get("ok"),
                                                             rec.get("why")))

    def whole_plan(self):
        """Every recipe id of the vendor's whole plan (the smoke's own selection), sorted."""
        import gpu_gap_campaign as G
        return [r["id"] for r in G.selected_plan(self.vendor, None, smoke=True, cpu_fallback=True)]

    def measure(self):
        self.amd_env()
        python, wheel_manifest = self.state["python"], self.state["wheel_manifest"]
        ids = self.whole_plan()
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        mout = self.root / ("measure-" + stamp)
        manifest = self.root / ("measure-manifest-%s.json" % stamp)
        self.E.write(manifest, {self.vendor: ids})
        store_dir = os.environ.get("MOJOLEARN_GATE_OPPONENT_STORE_DIR", STORE_DIR)
        os.makedirs(store_dir, exist_ok=True)
        store = os.environ.get("MOJOLEARN_GATE_OPPONENT_STORE") or os.path.join(
            store_dir, "%s-opponent-store.jsonl" % self.vendor)
        hours = float(os.environ.get("MOJOLEARN_GATE_MEASURE_HOURS", MEASURE_HOURS))
        stop_at = time.time() + hours * 3600
        self.mark(phase="measure", status="running", measure_started=time.time(), measure_finished=None,
                  measure_out=str(mout), measure_manifest=str(manifest), opponent_store=store,
                  measure_stop_at=stop_at)
        # ONE seed, ONE scored run per arm: the campaign's bench_board.py --rounds 1 (its fixed seed), the
        # board's own workload budgets (--round-seconds 0), opponents from the store when already scored
        cmd = [python, str(HERE / "gpu_gap_campaign.py"), "--vendor", self.vendor, "--manifest", str(manifest),
               "--out", str(mout), "--cache", self.cache, "--python-env", python,
               "--wheel-manifest", wheel_manifest, "--cpu-fallback", "--opponent-store", store,
               "--stop-at", "%.0f" % stop_at, "--keep-going"]
        with (self.root / ("measure-%s.log" % stamp)).open("wb") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=self.env(),
                                cwd=str(HERE.parent)).returncode
        camp = {}
        try:
            camp = json.loads((mout / "campaign.json").read_text())
        except (OSError, ValueError):
            pass
        recipes = camp.get("recipes") or {}
        statuses = Counter(str(r.get("status")) for r in recipes.values())
        failed = sorted(rid for rid, r in recipes.items() if r.get("controller_failure"))
        pending = sum(1 for r in recipes.values() if r.get("status") in ("pending", "running"))
        board = self.merge_boards(mout)
        pairs = self.pair_check(board)
        rec = dict(planned_whole_plan=len(ids), expected=camp.get("expected"), statuses=dict(statuses),
                   pending=pending, failed=failed, stop_reason=camp.get("stop_reason"), board=board,
                   pairs=pairs, own_not_built=own_not_built(self.B.load_result(board) or {} if board else {}),
                   retention=str(mout / "campaign-retention.json"))
        self.mark(phase="measure", status="done", measure_rc=rc, measure=rec, measure_finished=time.time())
        print("GATE_MEASURE vendor=%s rc=%s lines=%d finished=%d pending=%d failed=%d board=%s" % (
            self.vendor, rc, len(ids), statuses.get("process finished", 0), pending, len(failed), board))
        if pairs is not None:
            print("GATE_PAIRS vendor=%s matched=%d unchecked=%d refused=%d" % (
                self.vendor, pairs["matched"], len(pairs["unchecked"]), len(pairs["refused"])))

    def pair_check(self, board):
        """bench_board.pair_report of the merged gate board, also written beside it as pair-check.json.
        None when no line wrote a board."""
        if not board:
            return None
        pairs = self.B.pair_report(self.B.load_result(board) or {})
        self.E.write(Path(board).parent / "pair-check.json", pairs)
        return pairs

    def merge_boards(self, mout):
        """Every line's board.json (bench_board wrote it) merged into one board through bench_board:
        <mout>/board/board.json + BOARD.md. Returns its path or None when no line wrote a board."""
        B = self.B
        merged = None
        for path in sorted(glob.glob(str(mout / "measured" / "*" / "board.json"))):
            try:
                one = B.load_result(path)
            except (OSError, ValueError):
                continue
            if not one:
                continue
            if merged is None:
                merged = dict(one, races={})
                merged["gate"] = {"vendor": self.vendor, "version": self.version, "lines": []}
            for rid, rr in (one.get("races") or {}).items():
                rr = dict(rr, cells=B.add_ratios(list(rr.get("cells") or []), race=rr), gate_line=path)
                merged["races"][rid] = rr
            merged["gate"]["lines"].append(path)
        if merged is None:
            return None
        bdir = mout / "board"
        bdir.mkdir(parents=True, exist_ok=True)
        B.save_result(str(bdir / "board.json"), merged)
        B.write_board(str(bdir), merged)
        return str(bdir / "board.json")

    def finish(self):
        verdict, why = gate_verdict(self.state)
        self.mark(verdict=verdict, verdict_reasons=why, verdict_at=time.time())
        import adapter_gate as AG
        adapters = dict(self.state.get("adapters") or {}, enabled=self.state.get("adapters_enabled", True))
        print("GATE_VERDICT vendor=%s version=%s verdict=%s %s%s" % (
            self.vendor, self.version, verdict, AG.summary(adapters), (" reasons=" + "; ".join(why)) if why else ""))
        return 0 if verdict == "PASS" else 1


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vendor', choices=('nvidia', 'amd'))
    parser.add_argument('version')
    parser.add_argument('root')
    parser.add_argument('step', nargs='?', default='all', choices=STAGES+('correctness', 'all', 'verdict'))
    parser.add_argument('--profile', choices=P.PROFILES, default=P.LIGHT)
    parser.add_argument('--candidate-wheel', action='append', default=[])
    parser.add_argument('--source-commit')
    parser.add_argument('--autograd-manifest')
    parser.add_argument('--no-adapters', action='store_true',
                        help='record the adapter stage as disabled (MOJOLEARN_RELEASE_ADAPTERS=0: no adapters ship)')
    args = parser.parse_args(argv)
    vendor, version, root, step = args.vendor, args.version, args.root, args.step
    if vendor not in ("nvidia", "amd"):
        raise SystemExit("gate: nvidia or amd (Apple runs its own smoke tooling; Apple timing is the M3 Ultra's)")
    stages = P.LIGHT_STAGES if args.profile == P.LIGHT else STAGES
    if step not in stages + ("all", "verdict"):
        raise SystemExit("gate: stage is one of %s or all" % ", ".join(STAGES))
    if bool(args.candidate_wheel) != bool(args.source_commit):
        parser.error('candidate wheels and source-commit must be supplied together')
    if args.autograd_manifest and not args.candidate_wheel:
        parser.error('autograd-manifest is a candidate-mode input')
    gate = Gate(vendor, version, root, args.candidate_wheel, args.source_commit, args.autograd_manifest,
                adapters=not args.no_adapters, profile=args.profile)
    if args.profile == P.LIGHT and step != "verdict":
        return gate.run_light(step)
    for stage in STAGES:
        if step in (stage, "all"):
            getattr(gate, stage)()
    if step in ("all", "verdict"):
        return gate.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
