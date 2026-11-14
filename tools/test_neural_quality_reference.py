"""QUALITY-MISSING FIX (2026-11-13, candidate ba08dac006 L40S): the neural forward lanes' untimed eager FP32 torch
quality reference runs from the board's CPU-torch reference venv, and an arm it could not score says so."""
import sys
import types

import bench_board as bb
import bench_board_neural as N


def test_reference_lanes_agree_between_the_board_and_the_driver():
    assert bb.NEURAL_TORCH_REFERENCE_LANES == N.TORCH_REFERENCE_LANES


def test_reference_venv_carries_the_neural_pins():
    for vendor in ("nvidia", "amd", "apple"):
        reqs = bb.fixtures_requirements(vendor)
        assert reqs[1].startswith("torch==") and "+" not in reqs[1]
        for pin in bb.NEURAL_PINS:
            assert pin in reqs


def _ctx(**kw):
    return dict({"python": "py", "neural_driver": "drv", "vendor": "nvidia", "rounds": 1, "out": "/o",
                 "round_seconds": 0}, **kw)


def test_neural_cmd_passes_the_reference_python_only_to_reference_lanes():
    m1 = bb.plan_races("nvidia", ["identical"], ["neural"], ["mamba1-forward"])[0]
    lm = bb.plan_races("nvidia", ["identical"], ["neural"], ["lm-train-step"])[0]
    cmd = bb.neural_cmd(_ctx(reference_python="/v/nf/bin/python"), m1)[0]
    assert cmd[cmd.index("--reference-python") + 1] == "/v/nf/bin/python"
    assert "--reference-python" not in bb.neural_cmd(_ctx(reference_python="/v/nf/bin/python"), lm)[0]
    assert "--reference-python" not in bb.neural_cmd(_ctx(), m1)[0]


def test_ensure_reference_python_makes_the_fixtures_venv(monkeypatch, tmp_path):
    made = []
    monkeypatch.setattr(bb, "ensure_fixtures_venv", lambda vdir, vendor, log: made.append(vdir) or vdir + "/bin/python")
    m1 = bb.plan_races("nvidia", ["identical"], ["neural"], ["mamba1-forward"])
    lm = bb.plan_races("nvidia", ["identical"], ["neural"], ["lm-train-step"])
    ctx = _ctx(out=str(tmp_path), fixtures_venv=str(tmp_path / "nf"))
    assert bb.ensure_reference_python(ctx, lm) is None and not made
    assert bb.ensure_reference_python(ctx, m1) == str(tmp_path / "nf") + "/bin/python"
    assert ctx["reference_python"] == str(tmp_path / "nf") + "/bin/python"
    ctx = _ctx(out=str(tmp_path))
    assert bb.ensure_reference_python(ctx, m1) is None and "reference_python" not in ctx


def test_torch_reference_runs_in_the_reference_python(monkeypatch, tmp_path):
    seen = {}

    def start_worker(arm, cmd, env, log, repo):
        seen.update(cmd=cmd, env=env)
        raise RuntimeError("No module named 'torch'")

    fake = types.SimpleNamespace(start_worker=start_worker, THREAD_ENV=(), apply_cpu_quota=lambda env: None,
                                 torch_worker_environment=lambda env: None)
    monkeypatch.setattr(N, "_load", lambda name: fake)
    args = N.build_parser().parse_args(["race", "--lane", "mamba1-forward", "--out", str(tmp_path), "--work", "w",
                                        "--theirs-python", "/v/ours/bin/python",
                                        "--reference-python", "/v/nf/bin/python"])
    receipt, out = N.torch_reference_outputs(args, "mamba1-forward", "full", "d.npz", "tag")
    assert out is None and receipt["status"] == "failed" and receipt["python"] == "/v/nf/bin/python"
    assert seen["cmd"][0] == "/v/nf/bin/python" and seen["env"][N.QUALITY_REFERENCE_ENV] == "1"
    args = N.build_parser().parse_args(["race", "--lane", "mamba1-forward", "--out", str(tmp_path), "--work", "w",
                                        "--theirs-python", "/v/ours/bin/python"])
    N.torch_reference_outputs(args, "mamba1-forward", "full", "d.npz", "tag")
    assert seen["cmd"][0] == "/v/ours/bin/python"


def test_an_arm_the_reference_could_not_score_says_why():
    result = {"torch_quality_reference": {"status": "failed", "error": "RuntimeError(\"No module named 'torch'\")"},
              "quality": {"ours": {"max_abs_diff_vs_host": 0.0}}}
    N.mark_reference_unavailable(result, {"ours": {}, "mamba-ssm-fp32": {}})
    q = result["quality"]["ours"]
    assert q["max_abs_diff_vs_host"] == 0.0
    assert "rel_fro_vs_torch_eager_fp32" in q["quality_unavailable"] and "torch" in q["quality_unavailable"]
    assert "mamba-ssm-fp32" not in result["quality"]
    ok = {"torch_quality_reference": {"status": "ok"}, "quality": {"ours": {"rel_fro_vs_torch_eager_fp32": 1e-8}}}
    N.mark_reference_unavailable(ok, {"ours": {}})
    assert "quality_unavailable" not in ok["quality"]["ours"]
    none = {"quality": {"ours": {}}}
    N.mark_reference_unavailable(none, {"ours": {}})     # no reference was due: nothing to say
    assert none["quality"]["ours"] == {}


def test_reference_runs_on_the_cpu_only_in_the_reference_worker(monkeypatch):
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False),
                                       backends=types.SimpleNamespace(mps=None))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv(N.QUALITY_REFERENCE_ENV, raising=False)
    assert N._reference_on_cpu() is False                 # a timed arm never falls back to the CPU
    monkeypatch.setenv(N.QUALITY_REFERENCE_ENV, "1")
    assert N._reference_on_cpu() is True
    fake_torch.cuda.is_available = lambda: True
    assert N._reference_on_cpu() is False                 # a GPU torch keeps the reference on the GPU
