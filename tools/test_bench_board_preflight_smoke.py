"""The board's preflight-smoke hooks (2026-10-14): --race-ids, the scorer receipt in the parameter probe,
planned reference-twin exclusions, the shared data list, and the macOS core-only PyPI manifest."""
import json

import pytest

import bench_board as B
import pypi_wheel_install as P


def _plan(vendor="amd"):
    return B.gpu_opponents_only(B.plan_races(vendor, ["identical"]), vendor)


def test_race_ids_keep_exactly_the_listed_races_in_their_order(tmp_path):
    races = _plan()
    ids = [races[3]["id"], races[0]["id"]]
    assert [r["id"] for r in B.select_race_ids(races, ",".join(ids))] == ids
    f = tmp_path / "ids.json"
    f.write_text(json.dumps(ids))
    assert [r["id"] for r in B.select_race_ids(races, "@" + str(f))] == ids
    f.write_text("\n".join(ids) + "\n")
    assert [r["id"] for r in B.select_race_ids(races, "@" + str(f))] == ids
    with pytest.raises(SystemExit):
        B.select_race_ids(races, ids[0] + ",no/such/race")
    with pytest.raises(SystemExit):
        B.select_race_ids(races, "")


def test_data_needs_matches_the_board_rules():
    races = _plan()
    needed, corpus_missing = B.data_needs(races)
    assert "taxi" in needed and isinstance(corpus_missing, list)
    neural = [r for r in races if r["family"] == "neural"]
    assert B.data_needs(neural)[0] == []


def _probe(record_impl="library.mod"):
    rec = {"__record__": True, "library": "library", "source": "declared", "params": {"alpha": 1},
           "implementation": {"module": record_impl, "name": "X"}}
    return dict(record=rec, version="1", hardware={"status": "verified"})


def test_scorer_refusal_in_the_probe_refuses_the_race(monkeypatch):
    race = dict(family="classical", lane="ridge", our_arms={"ours": "identical"})

    def probe(ctx, r, arms, scorer=None):
        scorer["receipt"] = {"status": "refused", "reason": "quality scorer unavailable: reference data: no yq"}
        return {a: _probe() for a in arms}
    monkeypatch.setattr(B, "params_probe", probe)
    monkeypatch.setattr(B, "arm_device", lambda arm, vendor: "cpu")
    rep = B.preflight_check(dict(vendor="amd", scorer_preflight=True), race, ["ours", "lib"])
    assert rep["verdict"] == "REFUSED"
    assert "quality scorer unavailable: reference data: no yq" in rep["problems"]
    assert rep["scorer_preflight"]["status"] == "refused"


def test_missing_scorer_receipt_is_a_problem_and_trees_are_not_checked(monkeypatch):
    monkeypatch.setattr(B, "params_probe", lambda ctx, r, arms, scorer=None: {a: _probe() for a in arms})
    monkeypatch.setattr(B, "arm_device", lambda arm, vendor: "cpu")
    rep = B.preflight_check(dict(vendor="amd", scorer_preflight=True),
                            dict(family="classical", lane="ridge", our_arms={"ours": "identical"}), ["ours"])
    assert any("missing scorer preflight receipt" in p for p in rep["problems"])
    rep = B.preflight_check(dict(vendor="amd", scorer_preflight=True),
                            dict(family="trees", lane="gbdt-depthwise", our_arms={"ours": "identical"}), ["ours"])
    assert rep["scorer_preflight"]["status"] == "not-checked"
    assert not any("scorer" in p for p in rep["problems"])


def test_without_the_flag_the_probe_signature_is_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr(B, "params_probe", lambda ctx, r, arms: calls.append(arms) or {})
    B.preflight_check(dict(vendor="amd"), dict(family="classical", lane="ridge", our_arms={"ours": "identical"}),
                      ["ours"])
    assert calls == [["ours"]]


def test_mamba_reference_torch_arms_are_excluded_by_name_not_missing_probes(monkeypatch):
    lane = "mamba1-forward"
    assert lane in B.NEURAL.REFERENCE_TWIN_LANES
    seen = []
    monkeypatch.setattr(B, "params_probe", lambda ctx, r, arms: seen.append(list(arms)) or
                        {a: _probe("mamba_ssm.modules.block" if a != "ours" else "mojolearn.x") for a in arms})
    monkeypatch.setattr(B, "arm_device", lambda arm, vendor: "cpu")
    rep = B.preflight_check(dict(vendor="nvidia"), dict(family="neural", lane=lane, our_arms={"ours": "identical"}),
                            ["ours", "torch-eager-fp32", "torch-compile-fp32", "mamba-ssm-fp32"])
    assert seen == [["ours", "mamba-ssm-fp32"]]
    assert set(rep["excluded_arms"]) == {"torch-eager-fp32", "torch-compile-fp32"}
    assert not any("missing parameter/version probe" in p for p in rep["problems"])


def test_macos_manifest_is_core_only():
    assert P.distributions("apple") == ["mojolearn"]
    assert P.distributions("nvidia") == ["mojolearn-core", "mojolearn-nvidia", "mojolearn-nvidia-ptx"]
    assert P.distributions("amd") == ["mojolearn-core", "mojolearn-amd"]
    assert P.distributions("nvidia", legacy=True) == ["mojolearn", "mojolearn-nvidia"]
    with pytest.raises(ValueError):
        P.distributions("intel")
