"""tools/bench_board.py race_data_sha for neural races: wheel, version, git fallback, refusal.

    .pixi/envs/test/bin/python -m pytest tools/test_bench_board_race_data_sha.py
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("bench_board", os.path.join(HERE, "bench_board.py"))
bb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bb)

RACE = {"family": "neural", "dataset": "enwik8"}
SHA = "0123456789abcdef0123456789abcdef01234567"


def test_wheel_sha_wins():
    box = {"mojolearn": {"wheel": {"sha256": "w" * 64}, "version": "0.8.59"}, "repo": {"commit": SHA}}
    assert bb.race_data_sha({}, RACE, box) == "mojolearn-sources:" + "w" * 64


def test_version_when_no_wheel():
    box = {"mojolearn": {"wheel": None, "version": "0.8.59"}, "repo": {"commit": SHA}}
    assert bb.race_data_sha({}, RACE, box) == "mojolearn-sources:0.8.59"


def test_git_fallback_for_source_tree():
    box = {"mojolearn": {"wheel": None, "version": None}, "repo": {"commit": SHA}}
    assert bb.race_data_sha({}, RACE, box) == "mojolearn-sources:git " + SHA


def test_git_fallback_marks_dirty_tree():
    box = {"mojolearn": {"version": None}, "repo": {"commit": SHA + "-dirty"}}
    assert bb.race_data_sha({}, RACE, box) == "mojolearn-sources:git " + SHA + "+dirty"


def test_refuses_when_nothing_is_known(monkeypatch):
    monkeypatch.setattr(bb, "repo_commit", lambda *a, **k: None)
    box = {"mojolearn": {"version": None}, "repo": {"commit": None}}
    assert bb.race_data_sha({}, RACE, box) is None
    assert bb.race_data_sha({"commit": "unknown"}, RACE, {"mojolearn": {}, "repo": {}}) is None
