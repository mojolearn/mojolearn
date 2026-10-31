# SPDX-License-Identifier: Apache-2.0
"""The NVIDIA/AMD release gate as one persisted box job (tools/ops/release_gate.py, plan item 4 of
docs/plans/POST_0863_RELEASE_TOOLING.md): the exact wrapper commands per vendor, the exact command
release.py run_gate builds, and a recorded gate-smoke.json judged by release.py exactly as before.
Offline: every wrapper call is faked; nothing is rented, ssh'd or compiled."""
import hashlib
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release  # noqa: E402
from test_release_requirement_receipts import CandidateReceiptTests  # noqa: E402
import release_gate_profiles as profiles  # noqa: E402

RG = release.release_gate_module()
COMMIT = "a" * 40
VERSION = "0.8.57"


def wheel_files(tmp_path, vendor):
    names = ["mojolearn_core", "mojolearn_" + vendor] + (["mojolearn_nvidia_ptx"] if vendor == "nvidia" else [])
    out = []
    for name in names:
        path = tmp_path / f"{name}-{VERSION}-py3-none-manylinux_2_35_x86_64.whl"
        path.write_bytes(b"inert wheel fixture " + name.encode())
        out.append(path)
    return out


def args(tmp_path, vendor, **extra):
    argv = [vendor, VERSION, "--source-commit", COMMIT, "--source-tree", str(tmp_path / "source"),
            "--out", str(tmp_path / "gates-out" / "stamp-release-light-v1"),
            "--state", str(tmp_path / "gates" / f"{vendor}.job.json"), "--poll-seconds", "0"]
    for w in wheel_files(tmp_path, vendor):
        argv += ["--candidate-wheel", str(w)]
    for k, v in extra.items():
        argv += [k] + ([v] if v is not None else [])
    return RG.parse(argv)


# ------------------------------------------------------------------ the exact wrapper commands
def test_nvidia_job_goes_through_nvidia_central_submit_persisted(tmp_path):
    a = args(tmp_path, "nvidia")
    plan = RG.Runner(a, clock=lambda: 1793491200.0).make_plan()[0]
    tag = "rel-0.8.57-gate-nvidia-20261101T000000Z"
    assert plan.tag == tag
    indir = "/root/release-gate/0.8.57/nvidia/stamp-release-light-v1/in"
    res = f"/root/results-release-gate/0.8.57/nvidia/stamp-release-light-v1/{tag}/results"
    assert plan.sync() == ["bash", "tools/nvidia_central.sh", "sync", "release-gate", str(tmp_path / "source")]
    assert plan.launch() == ["bash", "tools/nvidia_central.sh", "submit", "release-gate", "--cap", "180",
                             "--note", tag, "--results", res, f"{indir}/release_gate_job.sh"]
    assert plan.upload(f"{indir}/gate.env") == ["bash", "tools/nvidia_central.sh", "sh", "release-gate",
                                                f"cat > {indir}/gate.env"]
    assert plan.fetch("T", "/d") == [sys.executable, "tools/box_persist.py", "fetch", "--tag", "T", "--dest", "/d",
                                     "--extract"]
    work = f"/root/results-release-gate/0.8.57/nvidia/stamp-release-light-v1/{tag}"
    assert plan.build == ["tools/gate_autograd_artifacts.py", "build", "--vendor", "nvidia", "--source-commit", COMMIT,
                          "--target", "sm_89", "--mojo", "/root/mojolearn-release-gate/.pixi/envs/default/bin/mojo",
                          "--compile-jobs", "2", "--out", f"{work}/out/autograd-bin"]
    wheels = [f"{indir}/{w.name}" for w in plan.wheels]
    assert plan.gate == ["tools/gate_smoke_box.py", "nvidia", "0.8.57", f"{work}/out", "all",
                         "--profile", "release-light-v1",
                         "--candidate-wheel", wheels[0], "--candidate-wheel", wheels[1], "--candidate-wheel", wheels[2],
                         "--source-commit", COMMIT, "--autograd-manifest", f"{work}/out/autograd-bin/build-manifest.json"]
    # every script the job runs is committed: the job script is the repository's own file, uploaded verbatim
    assert (RG.ROOT / RG.JOB_SCRIPT, f"{indir}/release_gate_job.sh", True) in plan.uploads()
    lines = "\n".join(plan.describe())
    assert "mojolearn-ops" not in lines and " lq " not in lines and "release_lq_smoke" not in lines


def test_amd_job_goes_through_do_amd_steward_run_persisted(tmp_path):
    a = args(tmp_path, "amd", **{"--no-adapters": None})
    plan = RG.Runner(a, clock=lambda: 1793491200.0).make_plan()[0]
    tag = "rel-0.8.57-gate-amd-20261101T000000Z"
    indir = "/root/release-gate/0.8.57/amd/stamp-release-light-v1/in"
    work = f"/root/results-release-gate/0.8.57/amd/stamp-release-light-v1/{tag}"
    assert plan.sync() is None
    assert plan.launch() == ["bash", "tools/do_amd_steward.sh", "run", tag, f"{work}/results",
                             "bash", f"{indir}/release_gate_job.sh"]
    assert plan.prepare()[:3] == ["bash", "tools/do_amd_steward.sh", "ssh"]
    assert plan.build[plan.build.index("--target") + 1] == "gfx942"
    assert plan.build[plan.build.index("--compile-jobs") + 1] == "8"
    assert plan.tree == "/root/release-gate/src/" + COMMIT[:12] and plan.fetch_url.startswith("https://")
    assert plan.gate[:7] == ["tools/gate_smoke_box.py", "amd", "0.8.57", f"{work}/out", "all", "--profile",
                             "release-light-v1"]
    assert plan.gate[7] == "--no-adapters"


def test_gate_env_is_bash_that_carries_the_exact_argv(tmp_path):
    plan = RG.Runner(args(tmp_path, "nvidia")).make_plan()[0]
    env = plan.env_text()
    gate_line = next(l for l in env.splitlines() if l.startswith("GATE_ARGV=("))
    assert shlex.split(gate_line[len("GATE_ARGV=("):-1]) == plan.gate
    build_line = next(l for l in env.splitlines() if l.startswith("BUILD_ARGV=("))
    assert shlex.split(build_line[len("BUILD_ARGV=("):-1]) == plan.build
    assert "GATE_TARGET=sm_89" in env and f"GATE_SOURCE_COMMIT={COMMIT}" in env and "GATE_REUSE=0" in env


def test_the_job_script_takes_its_argv_from_gate_env_and_keeps_the_gate_unchanged():
    text = (RG.ROOT / RG.JOB_SCRIPT).read_text()
    assert '"${BUILD_ARGV[@]}"' in text and 'python3 "${GATE_ARGV[@]}"' in text
    assert "MOJOLEARN_RESULTS_DIR" in text and "mojolearn-ops" not in text and "lq" not in text.split()


# ------------------------------------------------------------------ release.py run_gate
class FakeRelease(release.Release):
    rel = commit = version = dry = base = None


def runner(tmp_path, adapters=True):
    r = object.__new__(FakeRelease)
    r.dry, r.version, r.commit = True, VERSION, COMMIT
    r.rel, r.base = tmp_path / "rel" / COMMIT[:12], tmp_path / "rel"
    r.args = SimpleNamespace(gate_autograd_root=None)
    r.gate_ok = lambda vendor: (False, "no gate report")
    r.gate_out = lambda vendor: r.rel / f"gate-{vendor}" / "stamp-release-light-v1"
    r.source_checkout = lambda create=True: tmp_path / "source"
    r.candidate_gate_wheels = lambda vendor: wheel_files(tmp_path, vendor)
    r.calls, r.said = [], []
    r.must = lambda cmd, **kw: r.calls.append([str(c) for c in cmd])
    r.say = r.said.append
    return r


@pytest.mark.parametrize("vendor", ["nvidia", "amd"])
def test_release_run_gate_builds_the_exact_job_command(tmp_path, monkeypatch, vendor):
    monkeypatch.setenv("MOJOLEARN_RELEASE_ADAPTERS", "1")
    r = runner(tmp_path)
    assert r.run_gate(vendor) is None
    out = r.rel / f"gate-{vendor}" / "stamp-release-light-v1"
    expected = [release.PY, "tools/ops/release_gate.py", vendor, VERSION, "--source-commit", COMMIT,
                "--source-tree", str(tmp_path / "source"), "--out", str(out),
                "--state", str(r.rel / "gates" / f"{vendor}-stamp-release-light-v1.job.json"),
                "--profile", "release-light-v1", "--max-wait-hours", "72"]
    for w in wheel_files(tmp_path, vendor):
        expected += ["--candidate-wheel", str(w)]
    assert r.calls == [expected]
    would = [line for line in r.said if line.startswith("    would: ")]
    launch = [line for line in would if (" submit release-gate " if vendor == "nvidia" else " run rel-") in line]
    assert len(launch) == 1
    wrapper = "tools/nvidia_central.sh" if vendor == "nvidia" else "tools/do_amd_steward.sh"
    assert launch[0].startswith("    would: bash " + wrapper)
    assert not any("mojolearn-ops" in line or "release_lq_smoke" in line for line in r.said)


def test_release_run_gate_offers_a_built_earlier_autograd_set_for_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_RELEASE_ADAPTERS", "0")
    r = runner(tmp_path)
    earlier = r.base / "0123456789ab" / "gate-amd" / "x-release-light-v1" / "autograd-bin"
    earlier.mkdir(parents=True)
    (earlier / "build-manifest.json").write_text(json.dumps(dict(state="built", vendor="amd")))
    r.run_gate("amd")
    cmd = r.calls[0]
    assert cmd[cmd.index("--autograd-prior") + 1] == str(earlier) and cmd[-1] == "--no-adapters"
    (earlier / "build-manifest.json").write_text(json.dumps(dict(state="failed", vendor="amd")))
    r.calls.clear()
    r.run_gate("amd")
    assert "--autograd-prior" not in r.calls[0]


# ------------------------------------------------------------------ a whole job, faked, judged by release.py
@pytest.fixture
def recorded():
    fixture = CandidateReceiptTests()
    fixture.setUp()
    report = fixture.gate
    report.update(profile=profiles.LIGHT, correctness_rc=0, correctness_finished=1.0, verdict="PASS",
                  budget=dict(scope="all", limit_seconds=1800, elapsed_seconds=100, timed_out=False),
                  correctness=dict(status="PASSED", scope="expanded", source_commit="a" * 40,
                      installed_from=dict(expected_version="0.8.57"), installed=dict(vendor="cuda"),
                      jobs=[dict(name=name, exit_code=5 if name == "extended" else 0,
                                 timeout_seconds=180, elapsed_seconds=1, timed_out=False)
                            for name in profiles.CORRECTNESS_JOBS]))
    for name in ("measure", "measure_rc", "measure_finished", "counts", "smoke_lines", "smoke_rc", "smoke_finished"):
        report.pop(name)
    yield fixture
    fixture.doCleanups()


class FakeBox:
    """The wrappers and R2 as one fake: uploads land in .files, the submit queues, fetch serves REPORT."""

    def __init__(self, report, vendor, polls_before_final=1, job_exit="0"):
        self.report, self.vendor, self.files, self.calls = report, vendor, {}, []
        self.lists, self.polls_before_final, self.job_exit = 0, polls_before_final, job_exit
        self.ptag = ("nvc-release-gate-20261101T000000Z-abc123" if vendor == "nvidia"
                     else "rel-0.8.57-gate-amd-20261101T000000Z-20261101T000001Z-def456")

    def run(self, argv, cwd=None, stdin=None, timeout=None, capture_output=True, text=True):
        self.calls.append(argv)
        out = SimpleNamespace(returncode=0, stdout="", stderr="")
        command = argv[-1]
        if argv[:2] == ["bash", "tools/nvidia_central.sh"] and argv[2] == "submit":
            out.stdout = "nvc1-0007\n"
            out.stderr = f"box_persist: job {self.ptag} persists to R2 measurements/box-jobs/x/ (fetch: ...)\n"
        elif argv[2] in ("sh", "ssh") and command.startswith("cat > "):
            self.files[command[len("cat > "):]] = stdin.read()
            out.stdout = b""
        elif argv[2] in ("sh", "ssh") and "sha256sum" in command:
            out.stdout = "".join(f"{hashlib.sha256(v).hexdigest()}  {k}\n" for k, v in self.files.items())
        elif argv[2] in ("sh", "ssh") and "READY" in command:
            out.stdout = "READY\n"
        elif argv[2] == "status":
            out.stdout = "job 0007: running\n"
        elif "box_persist.py" in argv[1] and "--list" in argv:
            self.lists += 1
            out.stdout = (f"MANIFEST state=final exit={self.job_exit} updated=x uploads=2\n"
                          if self.lists > self.polls_before_final else "MANIFEST state=running exit= updated=x\n")
        elif "box_persist.py" in argv[1] and "--extract" in argv:
            assert argv[argv.index("--tag") + 1] == self.ptag
            final = Path(argv[argv.index("--dest") + 1]) / "final"
            (final / "out").mkdir(parents=True)
            (final / "out" / "gate-smoke.json").write_text(json.dumps(self.report))
            (final / "job").mkdir()
            (final / "job" / "status.json").write_text(json.dumps(dict(phase="complete", exit=0)))
        return out

    def popen(self, argv, **kw):
        self.calls.append(argv)
        lines = [f"box_persist: job {self.ptag} persists to R2 measurements/box-jobs/x/ (fetch: ...)\n",
                 "release-gate-job: running\n"]
        return SimpleNamespace(stdout=iter(lines), wait=lambda: 0, kill=lambda: None)


@pytest.mark.parametrize("vendor", ["nvidia", "amd"])
def test_a_recorded_gate_report_comes_home_and_release_judges_it_exactly_as_before(tmp_path, monkeypatch,
                                                                                    recorded, vendor):
    monkeypatch.setattr(RG, "job_script_committed", lambda: None)
    monkeypatch.setattr(RG, "on_origin_main", lambda commit: True)
    monkeypatch.setenv("MOJOLEARN_RELEASE_ADAPTERS", "1")
    report = dict(recorded.gate, vendor=vendor, candidate=dict(recorded.identity, vendor=vendor))
    report["correctness"] = dict(report["correctness"], installed=dict(vendor={"nvidia": "cuda", "amd": "hip"}[vendor]))
    import gate_autograd_artifacts as A
    report["autograd_artifacts"] = dict(report["autograd_artifacts"], vendor=vendor, files=[
        dict(file=name, source=source, exit_code=0, sha256="e" * 64, source_sha256="f" * 64)
        for name, source in A.expected(vendor).items()])
    identity = report["candidate"]
    box = FakeBox(report, vendor)
    a = args(tmp_path, vendor)
    assert RG.Runner(a, run=box.run, sleep=lambda s: None, popen=box.popen).main() == 0
    launches = [c for c in box.calls if c[2] in ("submit", "run")]
    assert len(launches) == 1 and launches[0][1] == RG.VENDORS[vendor]["wrapper"]
    # the bytes the job wrote are the bytes release.py judges, and the job script on the box is the committed one
    home = a.out / "gate-smoke.json"
    assert json.loads(home.read_text()) == report
    script = next(v for k, v in box.files.items() if k.endswith("/release_gate_job.sh"))
    assert script == (RG.ROOT / RG.JOB_SCRIPT).read_bytes()
    r = object.__new__(FakeRelease)
    r.version, r.dry = VERSION, False
    r.candidate_gate_identity = lambda v: identity
    r.gate_out = lambda v: a.out
    ok, why = r.gate_ok(vendor)
    assert ok, why
    assert why.startswith("PASS: 11 installed correctness records, 24 autograd entries, three adapter phases")
    # the same report, one record short, fails exactly as before
    broken = json.loads(home.read_text())
    broken["correctness"]["jobs"].pop()
    home.write_text(json.dumps(broken))
    assert not r.gate_ok(vendor)[0]
    assert not (tmp_path / "gates" / f"{vendor}.job.json").exists()   # the finished job leaves no resume state


def test_a_rerun_resumes_the_queued_job_instead_of_submitting_again(tmp_path, monkeypatch, recorded):
    monkeypatch.setattr(RG, "job_script_committed", lambda: None)
    monkeypatch.setattr(RG, "on_origin_main", lambda commit: True)
    box = FakeBox(dict(recorded.gate, verdict="PASS"), "nvidia", polls_before_final=10 ** 6)
    a = args(tmp_path, "nvidia", **{"--max-wait-hours": "0.0001"})
    clock = iter(range(0, 10 ** 6, 1))
    assert RG.Runner(a, run=box.run, sleep=lambda s: None, clock=lambda: float(next(clock))).main() == 3
    state = json.loads(Path(a.state).read_text())
    assert state["job"] == "nvc1-0007" and state["tag"] == box.ptag
    box.polls_before_final = 0
    submits = sum(1 for c in box.calls if c[2] == "submit")
    RG.Runner(args(tmp_path, "nvidia"), run=box.run, sleep=lambda s: None).main()
    assert sum(1 for c in box.calls if c[2] == "submit") == submits == 1


def test_an_amd_job_whose_connection_dropped_is_found_by_its_minted_tag(tmp_path, monkeypatch):
    home = tmp_path / "persist"
    home.mkdir()
    monkeypatch.setenv("MOJOLEARN_PERSIST_HOME", str(home))
    (home / "index.tsv").write_text("utc\ttag\tdate\n"
                                    "x\trel-0.8.57-gate-amd-20261101T000000Z-20261101T000001Z-def456\t2026-11-01\n"
                                    "x\trel-0.8.57-gate-amd-20261101T000000Zz-20261101T000001Z-aaaaaa\t2026-11-01\n")
    assert RG.minted_tag("rel-0.8.57-gate-amd-20261101T000000Z") == \
        "rel-0.8.57-gate-amd-20261101T000000Z-20261101T000001Z-def456"
    assert RG.minted_tag("rel-0.8.57-gate-nvidia-20261101T000000Z") is None


def test_dry_run_prints_the_submissions_and_runs_nothing(tmp_path, capsys):
    calls = []
    assert RG.Runner(args(tmp_path, "amd", **{"--dry-run": None}), run=lambda *a, **k: calls.append(a)).main() == 0
    out = capsys.readouterr().out
    assert calls == [] and "would: bash tools/do_amd_steward.sh run rel-0.8.57-gate-amd-" in out
    assert "mojolearn-ops" not in out
