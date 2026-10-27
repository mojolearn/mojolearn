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
    assert B.data_needs(neural)[0] == ["higgs"]
    assert B.data_needs([r for r in neural if r['dataset'] != 'higgs'])[0] == []


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


@pytest.mark.parametrize("vendor,mode", [("apple", "fast"), ("nvidia", "identical"), ("amd", "identical")])
def test_cpu_tree_refusal_excludes_only_an_unbuilt_arm(monkeypatch, vendor, mode):
    race = dict(family="trees", lane="rf", dataset="taxi", our_arms={"ours": mode})
    probes = {"ours": _probe(), "sklearn-rf-cpu": _probe()}
    monkeypatch.setattr(B, "params_probe", lambda ctx, r, arms: probes)
    monkeypatch.setattr(B, "arm_device", lambda arm, vendor: "cpu")
    arms = ["ours", "sklearn-rf-cpu", "lightgbm-cpu"]
    rep = B.preflight_check(dict(vendor=vendor), race, arms)
    assert rep["verdict"] == "MATCHED"
    assert set(rep["excluded_arms"]) == {"lightgbm-cpu"}
    # If a future driver constructs it, its parameters must still match.
    probes["lightgbm-cpu"] = _probe()
    probes["lightgbm-cpu"]["record"]["params"]["alpha"] = 2
    rep = B.preflight_check(dict(vendor=vendor), race, arms)
    assert rep["verdict"] == "REFUSED"
    assert "lightgbm-cpu" not in rep.get("excluded_arms", {})
    assert any("alpha" in p for p in rep["problems"])


def test_checked_box_retains_split_wheel_identity_for_neural_smoke(tmp_path, monkeypatch):
    from types import SimpleNamespace
    manifest = dict(schema=P.SCHEMA, version='0.8.61', vendor='amd', files=[
        dict(distribution='mojolearn-core', file='core.whl', sha256='a'*64, bytes=123)])
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    def unexpected(*args):
        raise AssertionError('must not repeat the installed wheel check')
    monkeypatch.setattr(B, 'pypi_wheel_guard', unexpected)
    wheel, why = B.preflight_wheel_record(SimpleNamespace(wheel_manifest=str(path)), 'python', 'amd', True)
    assert why is None and wheel['sha256'] == 'a'*64
    assert B.race_data_sha({}, {'family': 'neural'}, {'mojolearn': {'wheel': wheel}}) == 'mojolearn-sources:' + 'a'*64


def test_neural_higgs_smoke_records_actual_data_and_refuses_missing_file(tmp_path, monkeypatch):
    monkeypatch.delenv('MOJOLEARN_BENCH_HIGGS', raising=False)
    race = dict(family='neural', lane='gru-fit', dataset='higgs')
    path = tmp_path / 'higgs/higgs_speed.npz'
    box = {'mojolearn': {'version': '0.8.61'}}
    assert B.race_data_files({'data_root': str(tmp_path)}, race) == [('higgs', str(path))]
    assert B.race_data_sha({'data_root': str(tmp_path)}, race, box) is None
    path.parent.mkdir()
    path.write_bytes(b'pinned input fixture')
    assert B.race_data_sha({'data_root': str(tmp_path)}, race, box) == B.sha256_file(path)
    monkeypatch.setenv('GBM_BENCH_DATA', str(tmp_path))
    assert B.NEURAL.higgs_path() == str(path)
    other = tmp_path / 'explicit.npz'
    monkeypatch.setenv('MOJOLEARN_BENCH_HIGGS', str(other))
    assert B.NEURAL.higgs_path() == str(other)
    assert B.race_data_files({'data_root': str(tmp_path)}, race) == [('higgs', str(other))]


def test_neural_higgs_accepts_pinned_unsplit_arrays_without_overlapping_rows(tmp_path, monkeypatch):
    import numpy as np
    path = tmp_path / 'higgs.npz'
    x = np.arange(18, dtype=np.float32).reshape(6, 3)
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32)
    np.savez(path, x=x, y=y)
    monkeypatch.setenv('MOJOLEARN_BENCH_HIGGS', str(path))
    monkeypatch.setitem(B.NEURAL.FIT_ROWS, 'fixture', {'train': 4, 'test': 2})
    record = {}
    arrays = B.NEURAL.fit_arrays('mlp-fit', 'fixture', record)
    np.testing.assert_array_equal(arrays['y_train'], y[:4])
    np.testing.assert_array_equal(arrays['y_test'], y[4:])
    expected = ((x[4:].astype(np.float64) - x[:4].astype(np.float64).mean(0)) /
                x[:4].astype(np.float64).std(0)).astype(np.float32)
    np.testing.assert_array_equal(arrays['X_test'], expected)
    assert record['higgs']['held_out'].startswith('x[4:6]')
