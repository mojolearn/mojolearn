"""tools/smoke_suspects.py: suspects-only smoke (Andrew 2026-10-14)."""
import json

import smoke_suspects as S


def test_changed_files_map_to_families_or_everything():
    why, fams, _ = S.families_for_changes(["tools/bench_board_neural.py", "docs/x.md", "tools/test_bench_board.py"])
    # algos imports the neural driver (its layer lanes reuse it), so both families re-run
    assert why is None and "neural" in fams and "classical" not in fams
    why, fams, _ = S.families_for_changes(["bench/speed/forest_speed_arm.py"])
    assert why is None and "trees" in fams
    for shared in ("tools/bench_board_params.py", "tools/race_fairness.py", "tools/smoke_worker.py",
                   "tools/bench_board.py", "python/mojolearn/ensemble.py", "bindings/x.mojo"):
        assert S.families_for_changes([shared])[0], shared
    assert S.families_for_changes(["docs/a.md"]) == (None, {}, {"docs/a.md": "test, doc or result file"})


def _prev(tmp_path, vendor, rows, commit="abc"):
    p = tmp_path / ("%s.json" % vendor)
    p.write_text(json.dumps({"vendor": vendor, "source_commit": commit, "rows": rows}))
    return str(p)


def test_select_runs_only_suspects_and_records_the_known_good(tmp_path, monkeypatch):
    plans = {"identical": [{"id": "algos/adam/synthetic/rows=full", "family": "algos"},
                           {"id": "algos/sgd/synthetic/rows=full", "family": "algos"},
                           {"id": "neural/gemm/gaussian/shape=full", "family": "neural"},
                           {"id": "classical/ols/taxi/rows=full", "family": "classical"},
                           {"id": "classical/pca/taxi/rows=full", "family": "classical"}]}
    rows = {"identical|algos/adam/synthetic/rows=full": {"status": "MATCHED"},
            "identical|algos/sgd/synthetic/rows=full": {"status": "STARTED"},
            "identical|neural/gemm/gaussian/shape=full": {"status": "MATCHED"},
            "identical|classical/ols/taxi/rows=full": {"status": "MATCHED"}}
    prev = _prev(tmp_path, "nvidia", rows)
    other = _prev(tmp_path, "amd", {})
    monkeypatch.setattr(S, "changed_files", lambda since, root=S.ROOT: ["tools/bench_board_neural.py"])
    monkeypatch.setattr(S, "families_for_changes", lambda files, root=S.ROOT: (
        None, {"neural": ["tools/bench_board_neural.py"]}, {}))
    out, rec = S.select(plans, prev + "," + other, "nvidia")
    ran = [r["id"] for r in out["identical"]]
    assert ran == ["algos/sgd/synthetic/rows=full", "neural/gemm/gaussian/shape=full", "classical/pca/taxi/rows=full"]
    assert set(rec["skipped_known_good"]) == {"identical|algos/adam/synthetic/rows=full",
                                              "identical|classical/ols/taxi/rows=full"}
    assert rec["skipped_known_good"]["identical|algos/adam/synthetic/rows=full"]["source_commit"] == "abc"
    assert rec["ignored_other_vendor"][0]["vendor"] == "amd" and rec["everything"] is None


def test_select_runs_everything_on_shared_changes_or_no_diff(tmp_path, monkeypatch):
    plans = {"identical": [{"id": "algos/adam/synthetic/rows=full", "family": "algos"}]}
    prev = _prev(tmp_path, "nvidia", {"identical|algos/adam/synthetic/rows=full": {"status": "MATCHED"}})
    monkeypatch.setattr(S, "changed_files", lambda since, root=S.ROOT: ["tools/bench_board_params.py"])
    out, rec = S.select(plans, prev, "nvidia")
    assert out == plans and "shared" in rec["everything"]
    monkeypatch.setattr(S, "changed_files", lambda since, root=S.ROOT: None)
    out, rec = S.select(plans, prev, "nvidia")
    assert out == plans and "cannot diff" in rec["everything"]


def test_named_unsupported_is_known_good_and_unnamed_is_a_suspect():
    assert S.known_good_status({"status": "UNSUPPORTED", "problems": [
        "no supported own arm (ours-fast: UNSUPPORTED(FAST low-bit API requires IDENTICAL))"]})
    assert not S.known_good_status({"status": "UNSUPPORTED", "problems": []})
    assert not S.known_good_status({"status": "REFUSED"})
