"""tools/bench_board_compile_guard.py: compiles inside a timed run are counted and flagged; the
warm-up repeats the exact call until nothing compiles (no torch needed)."""
import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location("t_" + name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load("bench_board_compile_guard")


def test_compile_arms_by_name():
    assert G.is_compile_arm("torch-compile-fp32") and G.is_compile_arm("torch-cpu-compile-bf16")
    assert not G.is_compile_arm("torch-eager-fp32") and not G.is_compile_arm("ours")


def test_counts_compiles_between_readings():
    a = {"frames_ok": 2, "unique_graphs": 2, "recompiles": 0}
    assert G.compiles(a, dict(a)) == 0
    assert G.compiles(a, dict(a, frames_ok=5)) == 3
    assert G.compiles(a, dict(a, unique_graphs=3)) == 1
    assert G.compiles(None, a) is None and G.compiles(a, None) is None


def test_read_is_none_without_torch_loaded(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert G.read() is None


def test_read_uses_dynamo_counters(monkeypatch):
    import collections
    counters = collections.defaultdict(collections.Counter)
    counters["frames"]["ok"] = 4
    counters["stats"]["unique_graphs"] = 3
    counters["recompiles"]["guard"] = 2
    utils = types.ModuleType("torch._dynamo.utils")
    utils.counters = counters
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "torch._dynamo", types.ModuleType("torch._dynamo"))
    monkeypatch.setitem(sys.modules, "torch._dynamo.utils", utils)
    assert G.read() == {"frames_ok": 4, "unique_graphs": 3, "recompiles": 2}


def test_settle_repeats_the_call_until_nothing_compiles():
    state = {"n": 0, "calls": 0}
    pending = [2, 1, 0, 0]          # compiles each call would trigger

    def call():
        state["n"] += pending[state["calls"]]
        state["calls"] += 1

    rec = G.settle(call, reader=lambda: {"frames_ok": state["n"], "unique_graphs": 0})
    assert rec == {"extra_warmup_calls": 3, "compiles_per_extra_call": [2, 1, 0], "settled": True}
    state.update(n=0, calls=0)
    pending[:] = [1, 1, 1, 1]
    rec = G.settle(call, reader=lambda: {"frames_ok": state["n"], "unique_graphs": 0}, max_extra=2)
    assert rec["settled"] is False and rec["extra_warmup_calls"] == 2
    assert G.settle(call, reader=lambda: None)["extra_warmup_calls"] == 0


def test_cell_with_a_compile_in_its_clock_is_shown_flagged():
    assert G.summarize([0, None, 2]) == 2 and G.summarize([None]) is None
    c = G.flag_cell({"status": "ok", "median_ms": 281.0}, 3)
    assert c["status"] == "recompiled-in-clock" and c["recompiles_in_timed_run"] == 3
    assert c["median_ms"] == 281.0
    c = G.flag_cell({"status": "ok"}, 0)
    assert c["status"] == "ok" and c["recompiles_in_timed_run"] == 0
    c = G.flag_cell({"status": "not-scored"}, 1)
    assert c["status"] == "recompiled-in-clock" and c["also_not_scored"]
    c = G.flag_cell({"status": "REFUSED(x)"}, 4)
    assert c["status"] == "REFUSED(x)"
    assert G.flag_cell({"status": "ok"}, None) == {"status": "ok"}


def test_board_keeps_a_recompiled_cell_in_its_ratios_and_names_it():
    bb = _load("bench_board")
    cells = [{"status": "ok", "recompiles_in_timed_run": 2}, {"status": "ok"}]
    bb.flag_recompiles(cells)
    assert cells[0]["status"] == bb.RECOMPILED and cells[1]["status"] == "ok"
    assert bb.RECOMPILED in bb.TIMED_STATUSES
    assert bb.status_text(cells[0]).startswith("recompiled-in-clock: 2")
    rf = _load("race_fairness")
    assert rf.completed({"status": "recompiled-in-clock", "median_ms": 5.0})
    audit = _load("board_clock_audit")
    assert "recompiled-in-clock" in audit.TIMED_STATUSES
    ingest = _load("main_board_ingest")
    assert ingest.board_status("recompiled-in-clock", 5.0) == "recompiled-in-clock"
    assert ingest.is_ok({"status": "recompiled-in-clock", "median_ms": 5.0})


def test_algos_and_neural_drivers_carry_the_counts_into_their_arms():
    src = open(os.path.join(HERE, "bench_board_algos.py")).read()
    assert "COMPILE.settle(lambda: (runner.fit(), runner.infer()))" in src
    assert '"compile": compile_rec' in src and 'a["recompiles_in_timed_run"] = COMPILE.summarize(' in src
    nsrc = open(os.path.join(HERE, "bench_board_neural.py")).read()
    assert '"compiles": compiles_now' in nsrc and '["recompiles_in_timed_run"] = _load("bench_board_compile_guard")' in nsrc
