# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""tools/identity_timing_compare.py over tiny synthetic records (no Mojolearn import, < 1 s)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import identity_timing_compare as itc  # noqa: E402


def _cell(train, infer=0.0, batch=0.0, repeat2=None, verdict="STABLE", error=None):
    stages = {"repeat=1/train": train, "repeat=1/infer-save-reload": infer, "repeat=1/batch": batch}
    if repeat2 is not None:
        stages["repeat=2/train"] = repeat2
    cell = {"verdict": verdict, "timing": {"total_seconds": train + infer + batch + (repeat2 or 0.0), "stages": stages}}
    if error:
        cell["error"] = error
    return cell


def _record(path, cells, vendor="amd-test", commit="a" * 40, revs=None, fixtures=None):
    path.write_text(json.dumps(dict(
        vendor=vendor, commit=commit, repeats=1, cells=cells, lane_revisions=revs or {},
        fixtures=fixtures or {"base": {"X": "x0"}, "ties": {"X": "x1"}})))
    return str(path)


def test_regression_flag_stage_and_host_factor(tmp_path):
    before = _record(tmp_path / "b.json", {
        "slow/base": _cell(0.2, batch=0.1), "slow/ties": _cell(0.2, batch=0.1),
        "steady/base": _cell(1.0), "steady2/base": _cell(2.0),
    })
    after = _record(tmp_path / "a.json", {
        "slow/base": _cell(2.0, batch=0.1), "slow/ties": _cell(2.0, batch=0.1),
        "steady/base": _cell(1.0), "steady2/base": _cell(2.0),
    })
    rep = itc.compare([before], [after])
    rows = {r["lane"]: r for r in rep["rows"]}
    assert rep["rows"][0]["lane"] == "slow"
    slow = rows["slow"]
    assert slow["fixtures"] == 2
    assert abs(slow["before_s"] - 0.6) < 1e-9 and abs(slow["after_s"] - 4.2) < 1e-9
    assert slow["grew"] == "train" and slow["regressed"] and slow["outlier"]
    assert not rows["steady"]["regressed"]
    assert rep["host_factor"] == 1.0
    assert rep["regressed"] == ["slow"] and rep["improved"] == []


def test_repeat2_refused_and_error_cells_are_not_timed(tmp_path):
    before = _record(tmp_path / "b.json", {
        "x/base": _cell(1.0, repeat2=5.0),
        "r/base": _cell(0.0, verdict="REFUSED"),
        "e/base": _cell(0.0, error="ImportError: not built"),
    })
    after = _record(tmp_path / "a.json", {
        "x/base": _cell(1.0), "r/base": _cell(9.0), "e/base": _cell(9.0),
    })
    rep = itc.compare([before], [after])
    assert [r["lane"] for r in rep["rows"]] == ["x"]
    assert rep["rows"][0]["before_s"] == 1.0  # the repeat=2 stage is not summed
    assert rep["regressed"] == []


def test_changed_lane_revision_or_fixture_skips_the_lane(tmp_path):
    before = _record(tmp_path / "b.json", {"a/base": _cell(1.0), "b/base": _cell(1.0), "c/ties": _cell(1.0)},
                     revs={"a": "rows-1"})
    after = _record(tmp_path / "a.json", {"a/base": _cell(5.0), "b/base": _cell(5.0), "c/ties": _cell(5.0)},
                    revs={"a": "rows-2"}, fixtures={"base": {"X": "x0"}, "ties": {"X": "changed"}})
    rep = itc.compare([before], [after])
    assert set(rep["skipped"]) == {"a", "c"}
    assert [r["lane"] for r in rep["rows"]] == ["b"]


def test_last_record_of_a_side_supplies_the_cell_and_vendors_are_listed(tmp_path):
    old = _record(tmp_path / "old.json", {"x/base": _cell(9.0)}, vendor="nv-a")
    new = _record(tmp_path / "new.json", {"x/base": _cell(1.0)}, vendor="nv-b")
    after = _record(tmp_path / "a.json", {"x/base": _cell(1.0)}, vendor="nv-b")
    rep = itc.compare([old, new], [after])
    assert rep["rows"][0]["before_s"] == 1.0
    assert rep["before_vendors"] == ["nv-a", "nv-b"]


def test_cli_text_and_json(tmp_path, capsys):
    before = _record(tmp_path / "b.json", {"x/base": _cell(1.0)})
    after = _record(tmp_path / "a.json", {"x/base": _cell(3.0)})
    assert itc.main(["--before", before, "--after", after]) == 0
    out = capsys.readouterr().out
    assert "REGRESSED" in out and "TOTAL before" in out
    assert itc.main(["--before", before, "--after", after, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["regressed"] == ["x"]
    assert itc.main(["--before", str(tmp_path / "missing.json"), "--after", after]) == 1


# ---- THE PAIRING RULE (2026-11-06, lane/tools-timing-pairs) -----------------
# A cross-column statement needs every column at the same commit; otherwise the
# commits are printed side by side and the statement is NOT COMPARABLE.

C_OCT25 = "b25df4e213" + "0" * 30
C_NOV05 = "84da9f145c" + "0" * 30
C_OCT03 = "4840cb1db4" + "0" * 30


def _seed_dates(monkeypatch):
    monkeypatch.setattr(itc, "_DATES", {C_OCT25: "2026-10-25", C_NOV05: "2026-11-05", C_OCT03: "2026-10-03"})


def _col(tmp_path, name, commit, seconds, vendor, host=None, timing_env=None):
    path = tmp_path / f"{name}.json"
    data = dict(vendor=vendor, commit=commit, repeats=1, lane_revisions={},
                fixtures={"base": {"X": "x0"}, "ties": {"X": "x1"}},
                cells={f"{lane}/base": _cell(0.1, batch=s) for lane, s in seconds.items()})
    if host:
        data["host"] = host
    if timing_env:
        data["timing_env"] = timing_env
    path.write_text(json.dumps(data))
    return str(path)


def test_pairing_same_commit_on_both_sides_is_comparable(tmp_path, monkeypatch):
    _seed_dates(monkeypatch)
    before = {"cpu": [_col(tmp_path, "cb", C_OCT03, {"a": 1.0, "b": 1.0}, "cpu-x")],
              "amd": [_col(tmp_path, "ab", C_OCT03, {"a": 1.0, "b": 1.0}, "amd-x")]}
    after = {"cpu": [_col(tmp_path, "ca", C_NOV05, {"a": 4.0, "b": 4.0}, "cpu-x")],
             "amd": [_col(tmp_path, "aa", C_NOV05, {"a": 2.0, "b": 2.0}, "amd-x")]}
    res = itc.cross(before, after, ref="amd")
    pair = res["pairing"]
    assert pair["comparable"] and pair["before_same"] and pair["after_same"]
    assert pair["verdict"] == "COMPARABLE"
    cpu = next(r for r in res["summary"] if r["column"] == "cpu")
    # CPU 2.2 s -> 8.2 s against AMD 2.2 s -> 4.2 s over the same lanes
    assert abs(cpu["vs_ref"] - (8.2 / 2.2) / (4.2 / 2.2)) < 1e-9
    assert itc.vs_text(res, cpu).startswith("1.95")


def test_pairing_differing_after_commits_is_not_comparable(tmp_path, monkeypatch, capsys):
    """The 2026-11-05 report's mistake: CPU after at Oct 25, GPU after at Nov 5."""
    _seed_dates(monkeypatch)
    before = {"cpu": [_col(tmp_path, "cb", C_OCT03, {"a": 1.0}, "cpu-x")],
              "amd": [_col(tmp_path, "ab", C_OCT03, {"a": 1.0}, "amd-x")]}
    after = {"cpu": [_col(tmp_path, "ca", C_OCT25, {"a": 9.0}, "cpu-x")],
             "amd": [_col(tmp_path, "aa", C_NOV05, {"a": 1.0}, "amd-x")]}
    res = itc.cross(before, after, ref="amd")
    pair = res["pairing"]
    assert not pair["comparable"] and pair["before_same"] and not pair["after_same"]
    assert "after commits differ" in pair["verdict"] and "NOT COMPARABLE" in pair["verdict"]
    assert all(r["vs_ref"] is None for r in res["summary"])
    assert pair["columns"]["cpu"]["after"] == [dict(commit="b25df4e213", date="2026-10-25", cells=1)]
    assert pair["columns"]["amd"]["after"] == [dict(commit="84da9f145c", date="2026-11-05", cells=1)]
    itc._print_cross(res, 0)
    out = capsys.readouterr().out
    assert "CROSS-COLUMN: NOT COMPARABLE" in out
    assert "b25df4e213 (2026-10-25" in out and "84da9f145c (2026-11-05" in out
    assert "| NOT COMPARABLE |" in out


def test_pairing_after_only_when_before_commits_differ(tmp_path, monkeypatch):
    _seed_dates(monkeypatch)
    before = {"cpu": [_col(tmp_path, "cb", C_OCT03, {"a": 1.0}, "cpu-x")],
              "amd": [_col(tmp_path, "ab", C_OCT25, {"a": 1.0}, "amd-x")]}
    after = {"cpu": [_col(tmp_path, "ca", C_NOV05, {"a": 2.0}, "cpu-x")],
             "amd": [_col(tmp_path, "aa", C_NOV05, {"a": 2.0}, "amd-x")]}
    pair = itc.cross(before, after)["pairing"]
    assert pair["after_same"] and not pair["before_same"] and pair["after_only"] and not pair["comparable"]


def test_pairing_mixed_commit_side_must_match_as_a_set(tmp_path, monkeypatch):
    """A reuse record layered on a full record reads two commits; the other
    column must read the same two."""
    _seed_dates(monkeypatch)
    full = _col(tmp_path, "full", C_OCT25, {"a": 1.0, "b": 1.0}, "cpu-x")
    reuse = _col(tmp_path, "reuse", C_NOV05, {"b": 1.0}, "cpu-x")
    other = _col(tmp_path, "other", C_NOV05, {"a": 1.0, "b": 1.0}, "amd-x")
    before = {"cpu": [_col(tmp_path, "cb", C_OCT03, {"a": 1.0, "b": 1.0}, "cpu-x")],
              "amd": [_col(tmp_path, "ab", C_OCT03, {"a": 1.0, "b": 1.0}, "amd-x")]}
    pair = itc.cross(before, {"cpu": [full, reuse], "amd": [other]})["pairing"]
    assert not pair["after_same"]
    assert [p["commit"] for p in pair["columns"]["cpu"]["after"]] == ["b25df4e213", "84da9f145c"]


def test_cross_refuses_a_column_without_both_sides(tmp_path):
    a = _col(tmp_path, "a", C_OCT03, {"a": 1.0}, "cpu-x")
    try:
        itc.cross({"cpu": [a]}, {"cpu": [a], "amd": [a]})
    except SystemExit as exc:
        assert "amd" in str(exc)
    else:
        raise AssertionError("cross accepted a column with no before side")


def test_header_names_commit_date_and_host_per_side(tmp_path, monkeypatch, capsys):
    _seed_dates(monkeypatch)
    b = _col(tmp_path, "b", C_OCT25, {"a": 1.0}, "cpu-epyc-9575f", host={"cpu_model": "AMD EPYC 9575F"})
    a = _col(tmp_path, "a", C_NOV05, {"a": 1.0}, "cpu-epyc-7542", host={"cpu_model": "AMD EPYC 7542"})
    rep = itc.compare([b], [a])
    assert rep["before_points"][0]["commit"] == "b25df4e213" and rep["before_points"][0]["date"] == "2026-10-25"
    assert rep["after_points"][0]["date"] == "2026-11-05"
    assert not rep["same_host"]
    itc._print_text(rep, 5)
    out = capsys.readouterr().out
    assert "before: cpu-epyc-9575f at b25df4e213 (2026-10-25" in out
    assert "after:  cpu-epyc-7542 at 84da9f145c (2026-11-05" in out
    assert "WARNING: hosts differ" in out and "AMD EPYC 7542" in out


def test_hosts_comparable_reads_timing_env():
    env = dict(cpu_model="AMD EPYC 7542", logical_cpus=64, affinity_cpus=64, cgroup_cpu_quota=None,
               threads={"MOJOLEARN_CPU_THREADS": "8"}, concurrency="0/4")
    a = itc.record_host(dict(timing_env=env))
    b = itc.record_host(dict(timing_env=dict(env, threads={"MOJOLEARN_CPU_THREADS": "64"})))
    assert a["cpu_threads"] == "8" and a["stated"]
    ok, diffs = itc.hosts_comparable(a, a)
    assert ok and diffs == []
    ok, diffs = itc.hosts_comparable(a, b)
    assert not ok and any(d.startswith("threads") for d in diffs)
    # An older record states only the CPU model: unknown fields are listed, not judged.
    old = itc.record_host(dict(host=dict(cpu_model="AMD EPYC 7542")))
    ok, diffs = itc.hosts_comparable(old, a)
    assert ok and any("one unrecorded" in d for d in diffs)


def test_column_of_vendor_labels():
    assert itc.column_of("nvidia-ptx-l4-sm80") == "ptx"
    assert itc.column_of("nvidia-rtx-sm_89") == "nvidia"
    assert itc.column_of("amd-mi300x-gfx942") == "amd"
    assert itc.column_of("cpu-amd-epyc-7542-32-core-processor") == "cpu"
    assert itc.column_of("apple-m4") == "apple"


def test_series_one_lane_per_commit_across_columns(tmp_path, monkeypatch, capsys):
    _seed_dates(monkeypatch)

    def rec(name, commit, vendor, cells, revs=None):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(dict(vendor=vendor, commit=commit, repeats=1, lane_revisions=revs or {},
                                        fixtures={"base": {}, "ties": {}}, cells=cells)))
        return str(path)

    paths = [
        rec("c1", C_OCT25, "cpu-x", {"nc/base": _cell(1.0, batch=7.0), "nc/ties": _cell(1.0, batch=1.0)}),
        rec("a1", C_OCT25, "amd-x", {"nc/base": _cell(1.0, batch=7.0), "nc/ties": _cell(1.0, batch=1.0)}),
        # Nov 5: the CPU record lacks `ties`, so the lane is summed over `base` only everywhere.
        rec("c2", C_NOV05, "cpu-x", {"nc/base": _cell(1.0, batch=0.8)}),
        rec("a2", C_NOV05, "amd-x", {"nc/base": _cell(1.0, batch=0.5), "nc/ties": _cell(1.0, batch=0.1)},
            revs={"nc": "rows-2"}),
        rec("n3", C_OCT03, "nvidia-x", {"nc/base": _cell(1.0, batch=0.1)}),
    ]
    res = itc.series(paths, ["nc"], "non-train")
    assert [c["commit"] for c in res["commits"]] == ["4840cb1db4", "b25df4e213", "84da9f145c"]
    lane = res["lanes"]["nc"]
    assert lane["fixtures"] == ["base"]
    pts = lane["points"]
    assert pts["cpu@b25df4e213"]["seconds"] == 7.0 and pts["amd@b25df4e213"]["seconds"] == 7.0
    assert pts["cpu@84da9f145c"]["seconds"] == 0.8 and pts["amd@84da9f145c"]["seconds"] == 0.5
    # AMD's newest point (Nov 5) carries rows-2; its older point is marked, CPU's are not.
    assert pts["amd@b25df4e213"]["revision_differs"] and not pts["amd@84da9f145c"]["revision_differs"]
    assert not pts["cpu@b25df4e213"]["revision_differs"]
    common = itc.series(paths, ["nc"], "all", common_only=True, columns=["cpu", "amd"])
    assert [c["commit"] for c in common["commits"]] == ["b25df4e213", "84da9f145c"]
    assert abs(common["lanes"]["nc"]["points"]["cpu@84da9f145c"]["seconds"] - 1.8) < 1e-9
    assert itc.main(["series", "--lane", "nc", "--columns", "cpu,amd"] + paths) == 0
    out = capsys.readouterr().out
    assert "| 2026-10-25 b25df4e213 | 2026-11-05 84da9f145c |" in out
    assert "| nc | amd | 7.00* | 0.50 |" in out


def test_cli_cross(tmp_path, monkeypatch, capsys):
    _seed_dates(monkeypatch)
    cb = _col(tmp_path, "cb", C_OCT03, {"a": 1.0}, "cpu-x")
    ab = _col(tmp_path, "ab", C_OCT03, {"a": 1.0}, "amd-x")
    ca = _col(tmp_path, "ca", C_OCT25, {"a": 3.0}, "cpu-x")
    aa = _col(tmp_path, "aa", C_NOV05, {"a": 1.0}, "amd-x")
    assert itc.main(["cross", "--before", f"cpu={cb}", "--before", f"amd={ab}",
                     "--after", f"cpu={ca}", "--after", f"amd={aa}", "--ref", "amd"]) == 0
    out = capsys.readouterr().out
    assert "CROSS-COLUMN: NOT COMPARABLE (after commits differ)" in out
    assert itc.main(["cross", "--before", f"cpu={cb}", "--after", f"cpu={ca}", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["pairing"]["comparable"] is True
