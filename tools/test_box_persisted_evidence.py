"""--evidence-local-only for a box-persisted run: tools/box_persist_run.sh exports MOJOLEARN_BOX_PERSIST=1 and
MOJOLEARN_RESULTS_DIR; bench_board admits local evidence only when --out lies inside that dir, and
six_lane_grid_bb passes the flag only then. Every other full run still refuses it.

    .pixi/envs/test/bin/python -m pytest tools/test_box_persisted_evidence.py
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BB = _load("bench_board_box_persist", "bench_board.py")
G = _load("six_lane_grid_bb_box_persist", "six_lane_grid_bb.py")


def test_bench_board_admits_only_an_out_inside_the_persisted_dir(tmp_path):
    res = tmp_path / "lq"
    out = res / "out" / "n0001" / "bb"
    out.mkdir(parents=True)
    env = {"MOJOLEARN_BOX_PERSIST": "1", "MOJOLEARN_RESULTS_DIR": str(res)}
    assert BB.box_persisted_out(str(out), env)
    assert BB.box_persisted_out(str(res), env)
    assert not BB.box_persisted_out(str(tmp_path / "elsewhere"), env)
    assert not BB.box_persisted_out(str(tmp_path / "lq-other"), env)          # a sibling prefix is not inside
    assert not BB.box_persisted_out(str(out), {"MOJOLEARN_RESULTS_DIR": str(res)})   # no wrapper flag
    assert not BB.box_persisted_out(str(out), {"MOJOLEARN_BOX_PERSIST": "1"})       # no persisted dir
    assert not BB.box_persisted_out(str(out), {})


def test_grid_uses_the_same_rule(tmp_path):
    res = tmp_path / "lq"
    out = res / "out" / "a0001" / "bb"
    env = {"MOJOLEARN_BOX_PERSIST": "1", "MOJOLEARN_RESULTS_DIR": str(res)}
    assert G.box_persisted(out, env)
    assert not G.box_persisted(tmp_path / "bb", env)
    assert not G.box_persisted(out, {})


def test_grid_dry_run_passes_the_flag_only_when_persisted(tmp_path, monkeypatch, capsys):
    res = tmp_path / "lq"
    out = res / "out" / "n0001" / "bb"
    argv = ["--tag", "t", "--vendor", "nvidia", "--race", "neural:gemm:gaussian", "--out", str(out), "--dry-run"]
    monkeypatch.delenv("MOJOLEARN_BOX_PERSIST", raising=False)
    monkeypatch.delenv("MOJOLEARN_RESULTS_DIR", raising=False)
    assert G.main(argv) == 0
    assert "--evidence-local-only" not in capsys.readouterr().out
    monkeypatch.setenv("MOJOLEARN_BOX_PERSIST", "1")
    monkeypatch.setenv("MOJOLEARN_RESULTS_DIR", str(res))
    assert G.main(argv) == 0
    assert "--evidence-local-only" in capsys.readouterr().out
