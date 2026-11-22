"""CPU-only BF16 capability, retention and import-time alias checks."""
import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bench_board_bf16_test", HERE / "bench_board.py")
bb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bb)


def race():
    return dict(family="neural", lane="gemm-bf16", our_arms={"ours-fast": "fast"},
                arms=["ours-fast"], opponents=[], shape="full")


def test_plan_defers_optional_bf16_but_keeps_int8_unsupported():
    r = race()
    assert bb.unsupported_our_arms(r) == {}
    assert "ours-fast" in bb.unsupported_our_arms(dict(r, lane="gemm-int8"))
    plans = bb.plan_races("apple", ["fast"], ["neural"], ["gemm-bf16"])
    assert plans and plans[0]["capability_checks_pending"]


@pytest.mark.parametrize("enabled,wired,mode,expected", [
    (True, True, 0, True), (False, True, 0, False),
    (True, False, 0, False), (True, True, 1, False),
])
def test_capability_requires_native_flag_and_real_aliases(monkeypatch, tmp_path, enabled, wired, mode, expected):
    native = types.SimpleNamespace(fast_bf16_available=lambda: enabled, linalg_numeric_mode=lambda: mode,
                                   fast_matmul_bf16=object(), public_fast_to_bf16=object(), public_fast_from_bf16=object())
    binary = tmp_path / "binding.so"
    binary.write_bytes(b"fixture only; never loaded")
    native.__file__ = str(binary)
    package = types.ModuleType("mojolearn")
    package.__path__ = []
    package.__version__ = "fixture"
    linalg = types.ModuleType("mojolearn.linalg")
    linalg.matmul_bf16 = native.fast_matmul_bf16 if wired else object()
    linalg.to_bf16, linalg.from_bf16 = native.public_fast_to_bf16, native.public_fast_from_bf16
    impl = types.ModuleType("mojolearn._linalg_impl")
    impl._load = lambda: native
    package.linalg, package._linalg_impl = linalg, impl
    for name, value in [("mojolearn", package), ("mojolearn.linalg", linalg), ("mojolearn._linalg_impl", impl)]:
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(bb.NEURAL, "_ours_module", lambda: package)
    result = bb.NEURAL.fast_bf16_capability()
    assert result["supported"] is expected
    assert result["gpu_work_submitted"] is False
    assert len(result["binary_sha256"]) == 64


def test_actual_probe_retained_and_unsupported_survives_review(tmp_path):
    driver = tmp_path / "driver.py"
    driver.write_text("import json, os\nassert os.environ['MOJOLEARN_NUMERIC_MODE']=='fast'\n"
                      "print(json.dumps(dict(schema='mojolearn.fast-bf16-capability.v1', supported=False, reason='candidate disabled')))\n")
    ctx = dict(out=str(tmp_path), python=sys.executable, neural_driver=str(driver))
    r = race()
    unsupported = bb.unsupported_our_arms(r, ctx)
    probe = r["capability_probes"]["fast_bf16"]
    assert probe["exit_code"] == 0 and probe["status"] == "complete"
    assert Path(probe["log"]).is_file() and Path(probe["receipt"]).is_file()
    rec = dict(status="unsupported", unsupported_arms=unsupported,
               capability_probes=r["capability_probes"],
               cells=[dict(arm="ours-fast", status=unsupported["ours-fast"], median_ms=None)])
    assert bb.smoke_verdict(race(), rec, "apple") == []
    assert bb.smoke_verdict(race(), dict(rec, capability_probes={}), "apple")


def test_crashed_probe_is_failure_not_expected_unsupported(tmp_path):
    driver = tmp_path / "driver.py"
    driver.write_text("print('fixture import failure')\nraise SystemExit(3)\n")
    ctx = dict(out=str(tmp_path), python=sys.executable, neural_driver=str(driver))
    with pytest.raises(RuntimeError, match="capability probe failed"):
        bb.unsupported_our_arms(race(), ctx)
    receipt = json.loads(next(tmp_path.rglob("probe.json")).read_text())
    assert receipt["exit_code"] == 3 and receipt["status"] == "failed"
    assert "supported" not in receipt


def test_disabled_candidate_starts_no_own_or_opponent_worker(monkeypatch, tmp_path):
    r = bb.plan_races("apple", ["fast"], ["neural"], ["gemm-bf16"])[0]
    driver = tmp_path / "driver.py"
    driver.write_text("import json\nprint(json.dumps(dict(schema='mojolearn.fast-bf16-capability.v1', supported=False, reason='candidate disabled')))\n")
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported BF16 started a worker or opponent lookup")
    monkeypatch.setattr(bb, "stored_opponents", forbidden)
    monkeypatch.setattr(bb, "_run_race", forbidden)
    monkeypatch.setattr(bb, "store_opponents", lambda *args: 0)
    monkeypatch.setattr(bb, "base_cell", lambda ctx, race, arm, mode: dict(arm=arm, mode=mode, library="mojolearn"))
    rec = bb.run_race(dict(out=str(tmp_path), python=sys.executable, neural_driver=str(driver),
                           with_opponents=True, vendor="apple"), r)
    assert rec["status"] == "unsupported"
    assert rec["capability_probes"]["fast_bf16"]["supported"] is False
    assert all(c["library"] == "mojolearn" and "median_ms" not in c for c in rec["cells"])


@pytest.mark.parametrize("mode", ["identical", "fast"])
def test_generic_aliases_preserve_identical_and_select_native_fast(monkeypatch, mode):
    source = (HERE.parent / "python/mojolearn/linalg.py").read_text()
    package = types.ModuleType("bf16_alias_fixture")
    package.__path__ = []
    impl = types.ModuleType("bf16_alias_fixture._linalg_impl")
    for statement in ast.parse(source).body:
        if isinstance(statement, ast.ImportFrom) and statement.module == "_linalg_impl":
            for item in statement.names:
                setattr(impl, item.name, object())
    native = types.SimpleNamespace(fast_bf16_available=object(), fast_matmul_bf16=object(),
                                   public_fast_to_bf16=object(), public_fast_from_bf16=object())
    loads = []
    def load():
        loads.append(mode)
        return native
    impl._load = load
    backend = types.ModuleType("bf16_alias_fixture._backend")
    backend.default_mode = lambda: mode
    cholesky = types.ModuleType("bf16_alias_fixture._cholesky_impl")
    cholesky.Cholesky = object()
    for name, module in [(package.__name__, package), (impl.__name__, impl),
                         (backend.__name__, backend), (cholesky.__name__, cholesky)]:
        monkeypatch.setitem(sys.modules, name, module)
    namespace = dict(__name__="bf16_alias_fixture.linalg", __package__="bf16_alias_fixture")
    exec(compile(source, "linalg_alias_fixture", "exec"), namespace)
    if mode == "identical":
        assert loads == []
        for name in ["matmul_bf16", "to_bf16", "from_bf16"]:
            assert namespace[name] is getattr(impl, name)
    else:
        assert loads == ["fast"]
        assert namespace["matmul_bf16"] is native.fast_matmul_bf16
        assert namespace["to_bf16"] is native.public_fast_to_bf16
        assert namespace["from_bf16"] is native.public_fast_from_bf16
    assert namespace["matmul_int8"] is impl.matmul_int8


def test_portable_full_selector_uses_declared_board_shape_without_allocating(monkeypatch, tmp_path):
    path = HERE.parent / "experiments/apple_fast_bf16_20261015/local_benchmark.py"
    spec = importlib.util.spec_from_file_location("bf16_portable_fixture", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    calls = []
    monkeypatch.setattr(runner, "prepare", lambda root, shapes, preset: calls.append((root, shapes, preset)))
    monkeypatch.setattr(sys, "argv", [str(path), "prepare", "--preset", "full", "--output", str(tmp_path / "full")])
    assert runner.main() == 0
    assert calls == [(tmp_path / "full", [bb.NEURAL.GEMM_SHAPES["full"]], "full")]
    assert not (tmp_path / "full").exists()
