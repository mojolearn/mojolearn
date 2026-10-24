# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Release gate: the FULL measurement-board smoke AND measurement on one box (test tooling).

    python3 tools/gate_smoke_box.py <nvidia|amd> <version> <out-root> [setup|smoke|autograd|measure|all]

setup: by default, the bench venv from the published wheels (mojolearn==<version> and the
vendor plugin, verified against PyPI) plus every pinned opponent, exactly as
tools/bench_board.setup_python builds it for a measurement run.
smoke (the first stage after setup, 5 s per line): tools/gpu_gap_campaign.py --smoke --cpu-fallback over the
vendor's WHOLE plan (no --manifest, no --smoke-suspects: every line of bench_board.plan_races, GPU opponents
first, the CPU library where this vendor has no GPU library, and lines with no opponent at all),
--smoke-verify-data. Before 2026-10-22 the gate did not pass --cpu-fallback and selected_plan(smoke=True) kept
only lines with a GPU-library opponent: 132 of 273 NVIDIA lanes and 198 AMD lanes never started on the gate
box (~/mojolearn-evidence/audit-20261022/measurement_coverage.md section 4).
autograd (2026-10-19, lane ag-bench): tools/autograd_smoke.py's entries on this vendor, 5 s "does it
start" each: the gradient suite and the MLP identity check from the check binaries the orchestrator built
into <out-root>/autograd-bin (`tools/autograd_smoke.py build-commands`), and the tiny fit test, the public
examples and the Sequential and ops.Session probes (2026-10-19, lane ag-board-twins: the board's
sequential-fit and ops-session-step lanes in miniature) through the installed wheel. A missing binary reads
`not built`, never a pass. The importer probes (import-model-smoke; since 2026-10-20, lane ag-torch-twins,
import-torch-smoke, a torch module exported to ONNX and imported) run there too; import-torch-smoke reads `SKIPPED` when the
wheel's environment has no torch (set MOJOLEARN_REQUIRE_TORCH=1 in the box environment to fail instead).
The shared registry also includes captured training/replay, dropout, validation, optimizers, rank-N,
higher-order public checks and the Mamba/LM/small-MLP/Samba tape binaries. Python entries verify their
actual compiled binding; incomplete pytest coverage and importer refusals are reported explicitly.
measure (Andrew 2026-10-22: every exposed API is "measured if appropriate with apples-to-apples comparison
against peers, and the measurements should be in the smoke test"): tools/gpu_gap_campaign.py over EVERY line
of the same whole plan (--cpu-fallback, one manifest of all its recipe ids), IDENTICAL, ONE seed and ONE
scored run per arm (the campaign's bench_board.py --rounds 1: one excluded warm-up, one scored sample), our
arm from the verified PyPI wheel. Opponents are scored once and stored (CLAUDE.md): --opponent-store reuses
every stored opponent whose key (box, device, library version, read-back parameters, settings, data) matches
and scores only the missing ones. --keep-going: a line that fails is recorded and the next line starts (its
evidence is uploaded first; an upload failure still stops the run). Each line's log, preflight, board.json
and campaign row go to R2 as the line finishes (gpu_gap_campaign.retain_recipe). The board rows are written
by tools/bench_board.py inside each line; this stage then merges them through bench_board (load_result,
add_ratios, save_result, write_board) into <out-root>/measure-*/board/board.json + BOARD.md, which
tools/main_board_ingest.py --gate-board BOX=PATH takes onto the rolling main board (ours' clock span kept).
all: setup, smoke, autograd, measure, then the verdict.

THE VERDICT (gate_verdict; tools/release.py reads the same function): PASS only when the smoke ran with no
FAILED line, the autograd smoke exited 0, and the measure stage finished every planned line (none pending,
none with a controller failure, every line's evidence retained), and (Andrew 2026-10-22: measurement enforces
apples-to-apples comparisons, our identical mode being the only difference) every opponent pair on the gate
board reads MATCHED (bench_board.pair_report over board_pairing.params_state): an opponent whose explicit
parameters cannot be read back or differ is refused for that cell, with the reason, the cell shows ours alone,
and the gate FAILS listing it (gate-smoke.json measure.pairs, the verdict reasons and <board>/pair-check.json).
A REFUSED line (an arm refused by name) and
a STARTED smoke line (slow, not broken) do not fail the gate. An own arm that does not import or construct (its
binding not built, an import error, a constructor that raised) is never a skip (measure-coverage, 2026-10-24): the
smoke line reads FAILED (gpu_gap_campaign.own_arm_failures) and a measured own cell that read "SKIPPED: not built
yet" fails the verdict (own_not_built). gate-smoke.json carries every stage's record and
`verdict`; the exit status is 0 only on PASS (or for a single stage that ran).

Nothing here rents or deletes anything; it runs ON the box (an lq CMD job, tools/release_lq_smoke.py --gate),
never on the Mac.
"""

import glob
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STAGES = ("setup", "smoke", "autograd", "measure")
#: measure-stage wall budget (hours) before no new line starts; MOJOLEARN_GATE_MEASURE_HOURS overrides
MEASURE_HOURS = 30.0
#: the box's persistent opponent store (scored once per box key, reused by every later gate on that box);
#: MOJOLEARN_GATE_OPPONENT_STORE overrides
STORE_DIR = os.path.join(os.path.expanduser("~"), "mojolearn-gate-store")


def gate_verdict(state):
    """(verdict 'PASS' | 'FAIL' | 'INCOMPLETE', [reasons]) of a gate-smoke.json document."""
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
    def __init__(self, vendor, version, root, candidate_wheels=(), source_commit=None, autograd_manifest=None):
        import bench_board as B
        import bench_board_evidence as E
        self.B, self.E = B, E
        self.vendor, self.version, self.root = vendor, version, Path(root)
        self.candidate_wheels, self.source_commit = list(candidate_wheels), source_commit
        self.autograd_manifest = autograd_manifest
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
        if self.state and (self.state.get('input_mode', 'published') != mode or self.state.get('candidate') != candidate):
            raise SystemExit('gate output belongs to different published/candidate artifacts; use a new out-root')
        self.state.update(input_mode=mode, candidate=candidate)
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
                + ["/opt/rocm-7.2.3/lib", "/opt/rocm-6.4.0/lib"])

    def env(self):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        return env

    # -------------------------------------------------------------- stages
    def setup(self):
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
        arc = AS.run(self.vendor, str(self.root / "autograd-bin"), self.state["python"], str(report))
        doc = json.loads(report.read_text()) if report.exists() else {}
        self.mark(phase="autograd", status="done", autograd_rc=arc, autograd_report=str(report),
                  autograd_counts=doc.get("counts"), autograd_finished=time.time())
        print("GATE_SMOKE_AUTOGRAD vendor=%s rc=%s %s" % (self.vendor, arc, " ".join(
            "%s=%d" % kv for kv in sorted((doc.get("counts") or {}).items()))))

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
        print("GATE_VERDICT vendor=%s version=%s verdict=%s%s" % (
            self.vendor, self.version, verdict, (" reasons=" + "; ".join(why)) if why else ""))
        return 0 if verdict == "PASS" else 1


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vendor', choices=('nvidia', 'amd'))
    parser.add_argument('version')
    parser.add_argument('root')
    parser.add_argument('step', nargs='?', default='all', choices=STAGES+('all',))
    parser.add_argument('--candidate-wheel', action='append', default=[])
    parser.add_argument('--source-commit')
    parser.add_argument('--autograd-manifest')
    args = parser.parse_args(argv)
    vendor, version, root, step = args.vendor, args.version, args.root, args.step
    if vendor not in ("nvidia", "amd"):
        raise SystemExit("gate: nvidia or amd (Apple runs its own smoke tooling; Apple timing is the M3 Ultra's)")
    if step not in STAGES + ("all",):
        raise SystemExit("gate: stage is one of %s or all" % ", ".join(STAGES))
    if bool(args.candidate_wheel) != bool(args.source_commit):
        parser.error('candidate wheels and source-commit must be supplied together')
    if args.autograd_manifest and not args.candidate_wheel:
        parser.error('autograd-manifest is a candidate-mode input')
    gate = Gate(vendor, version, root, args.candidate_wheel, args.source_commit, args.autograd_manifest)
    for stage in STAGES:
        if step in (stage, "all"):
            getattr(gate, stage)()
    if step == "all":
        return gate.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
