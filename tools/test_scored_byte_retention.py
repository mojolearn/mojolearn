"""CPU-only proof that scored receipts retain verifiable bytes, without refitting."""
import io
import importlib.util
import os
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_board_evidence as E
import bench_board_state as S


class PublicState:
    __module__ = "mojolearn.testing"

    def __init__(self):
        self.reads = 0

    def state_dict(self):
        self.reads += 1
        return {"weights": np.array([0.0, -0.0, np.nan], np.float32),
                "optimizer": {"step": 1, "scale": np.float32(0.5), "momentum": [None, True]}}


class PublicSave:
    __module__ = "mojolearn.testing"

    def save(self, path):
        np.savez(path, weights=np.arange(3, dtype=np.float32), device="amd")


def capture(tmp_path, model=None, family="neural"):
    directory = tmp_path / "evidence/runs/r1"
    workers = directory / "workers"
    workers.mkdir(parents=True)
    E.write(directory / "recipe.json", {})
    outputs = {"pred": np.array([[0.0, -0.0], [1.0, 2.0]], dtype=np.float32)}
    with patch.dict(os.environ, {E.RECEIPTS: str(workers)}):
        receipt = S.scored_receipt(outputs, model=model)
    cell = dict(arm="ours", mode="identical", status="ok", state_receipts=[receipt])
    rec = dict(family=family, lane="fixture", dataset="fixture", cells=[cell])
    members = {str(p.relative_to(directory)): p.read_bytes() for p in (directory / "scored").rglob("*.npz")}
    return directory, rec, members


@pytest.mark.parametrize("family", ["trees", "classical", "classical2", "neural", "algos"])
def test_each_family_retains_scored_arrays_without_model_export(tmp_path, family):
    directory, rec, members = capture(tmp_path, family=family)
    refs, missing = E.scored_state_refs(rec, directory, members)
    assert not missing
    output = refs[("fit", "ours")][0]["output"]
    with np.load(io.BytesIO(members[output["member"]]), allow_pickle=False) as saved:
        assert saved["pred"].tobytes() == np.array([[0., -0.], [1., 2.]], np.float32).tobytes()
    assert refs[("fit", "ours")][0]["model"]["status"] == "unavailable"
    assert len(output["canonical_sha256"]) == 64


@pytest.mark.parametrize("model_type", [PublicState, PublicSave])
def test_public_snapshot_bytes_match_hash_and_export_is_read_once(tmp_path, model_type):
    model = model_type()
    directory, rec, members = capture(tmp_path, model=model)
    refs, missing = E.scored_state_refs(rec, directory, members)
    assert not missing
    snapshot = refs[("fit", "ours")][0]["model"]
    assert snapshot["retained"]["member"] in members
    assert "not certified" in snapshot["completeness"]
    if model_type is PublicState:
        assert model.reads == 1
        restored = S.load_typed_state(io.BytesIO(members[snapshot["retained"]["member"]]))
        assert restored["weights"].dtype == np.float32
        assert restored["optimizer"]["scale"].shape == ()


@pytest.mark.parametrize("kind", ["missing", "corrupt", "forged-file-hash", "changed-scored-hash"])
def test_bad_output_bytes_never_establish_identity(tmp_path, kind):
    directory, rec, members = capture(tmp_path)
    receipt = rec["cells"][0]["state_receipts"][0]
    output = receipt["retained_output"]
    member = str(Path(output["path"]).relative_to(directory))
    if kind == "missing":
        del members[member]
    elif kind == "corrupt":
        members[member] = b"broken npz"
    elif kind == "forged-file-hash":
        payload = io.BytesIO()
        np.savez(payload, pred=np.full((2, 2), 99, np.float32))
        members[member] = payload.getvalue()
        output.update(bytes=len(members[member]), sha256=E.digest(members[member]))
    else:
        receipt["output_sha256"] = "0" * 64
    refs, missing = E.scored_state_refs(rec, directory, members)
    assert missing[("fit", "ours")]
    assert "output" not in refs[("fit", "ours")][0]


def test_missing_model_bytes_block_retention_but_keep_valid_output(tmp_path):
    directory, rec, members = capture(tmp_path, model=PublicState())
    model = rec["cells"][0]["state_receipts"][0]["model"]
    del members[str(Path(model["retained_export"]["path"]).relative_to(directory))]
    refs, missing = E.scored_state_refs(rec, directory, members)
    assert missing[("fit", "ours")][0].startswith("scored model")
    assert refs[("fit", "ours")][0]["output"]


def test_model_export_unavailable_does_not_invalidate_output_identity(tmp_path):
    class Unsupported:
        __module__ = "mojolearn.testing"
        def save(self, path):
            raise NotImplementedError("public export unavailable")
    directory, rec, members = capture(tmp_path, model=Unsupported())
    refs, missing = E.scored_state_refs(rec, directory, members)
    assert not missing
    assert refs[("fit", "ours")][0]["output"]
    assert refs[("fit", "ours")][0]["model"]["status"] == "error"


def test_model_write_failure_blocks_retention(tmp_path):
    class DiskFailure:
        __module__ = "mojolearn.testing"
        def save(self, path):
            raise OSError("disk full")
    directory, rec, members = capture(tmp_path, model=DiskFailure())
    refs, missing = E.scored_state_refs(rec, directory, members)
    assert "disk full" in missing[("fit", "ours")][0]
    assert refs[("fit", "ours")][0]["output"]


def test_no_receipt_is_explicit_and_stored_cells_are_not_recaptured(tmp_path):
    directory, rec, members = capture(tmp_path)
    rec["cells"][0]["state_receipts"] = []
    assert E.scored_state_refs(rec, directory, members)[1][("fit", "ours")] == ["scored output receipt missing"]
    rec["cells"][0]["stored"] = True
    assert E.scored_state_refs(rec, directory, members) == ({}, {})


def test_non_array_outputs_keep_scalar_types_and_rounds_keep_distinct_bytes(tmp_path):
    directory, rec, members = capture(tmp_path)
    with patch.dict(os.environ, {E.RECEIPTS: str(directory / "workers")}):
        second = S.scored_receipt({"loss": 0.5, "step": np.int32(2)})
    rec["cells"][0]["state_receipts"].append(second)
    members.update({str(p.relative_to(directory)): p.read_bytes() for p in (directory / "scored").rglob("*.npz")})
    refs, missing = E.scored_state_refs(rec, directory, members)
    assert not missing
    rows = refs[("fit", "ours")]
    assert len(rows) == 2 and rows[0]["output"]["member"] != rows[1]["output"]["member"]
    saved = S.load_typed_state(io.BytesIO(members[rows[1]["output"]["member"]]))
    assert type(saved["loss"]) is float
    assert saved["step"].dtype == np.int32 and saved["step"].shape == ()


def test_seal_archives_output_and_model_and_preserves_failure_without_remeasurement(tmp_path):
    directory, rec, members = capture(tmp_path, model=PublicState())
    receipt = rec["cells"][0]["state_receipts"][0]
    # This fixture intentionally lacks other worker provenance: seal must
    # retain its bytes and exact missing-evidence reasons, never rerun a model.
    E.seal(dict(out=str(tmp_path), measurement_inventory={}, measurement_source={"status": "local"},
                evidence_r2=False), directory, dict(source={}, dataset={"identity": "input-sha"}, hardware={}), rec)
    with tarfile.open(directory / "run.tar.gz") as archive:
        for name, payload in members.items():
            assert archive.extractfile(name).read() == payload
    measured = rec["cells"][0]["measurement"]
    assert measured["scored_state"]["comparison"] == "not compared"
    assert measured["retained_outputs"]["output_sha256"] == receipt["output_sha256"]
    assert measured["retained_outputs"]["status"] == "local"
    assert measured["completeness"] == "incomplete"
    assert rec["evidence_failure"]


def test_upload_retry_updates_all_byte_refs_without_changing_timings(tmp_path):
    import json
    directory, rec, members = capture(tmp_path, model=PublicState())
    rec["cells"][0].update(times_ms=[1.25], median_ms=1.25)
    E.seal(dict(out=str(tmp_path), measurement_inventory={}, measurement_source={"status": "local"},
                evidence_r2=False), directory, dict(source={}, dataset={"identity": "input-sha"}, hardware={}), rec)
    # Only the upload is retried; an unrelated incomplete worker receipt must
    # remain incomplete and the original times and scored bytes cannot change.
    original_failure = rec["evidence_failure"]
    path = tmp_path / "board.json"
    E.write(path, {"races": {"fixture": rec}})
    with patch.object(E, "upload", side_effect=lambda ref: dict(ref, status="verified", r2_key="test-key")):
        E.sync(path)
    after = json.loads(path.read_text())["races"]["fixture"]
    cell = after["cells"][0]
    assert cell["times_ms"] == [1.25] and cell["median_ms"] == 1.25
    assert after["evidence_failure"] == original_failure
    measurement = cell["measurement"]
    assert measurement["retained_outputs"]["status"] == "verified"
    for row in measurement["scored_state"]["receipts"]:
        assert row["output"]["status"] == "verified"
        assert row["model"]["retained"]["status"] == "verified"
    with tarfile.open(directory / "run.tar.gz") as archive:
        for name, payload in members.items():
            assert archive.extractfile(name).read() == payload


def test_missing_scored_bytes_halt_next_race_and_fast_never_claims_identity(tmp_path):
    directory, rec, members = capture(tmp_path)
    cell = rec["cells"][0]
    cell.update(mode="fast", times_ms=[1.25], median_ms=1.25)
    output = cell["state_receipts"][0]["retained_output"]
    Path(output["path"]).unlink()
    E.seal(dict(out=str(tmp_path), measurement_inventory={}, measurement_source={"status": "local"},
                evidence_r2=False), directory, dict(source={}, dataset={"identity": "input-sha"}, hardware={}), rec)
    assert rec["evidence_failure"].startswith("incomplete scored-byte retention")
    assert cell["times_ms"] == [1.25]
    assert any("scored archive member missing" in item for item in cell["measurement"]["missing"])
    assert cell["measurement"]["scored_state"]["cross_vendor_promise"] is False


def tree_inference(tmp_path, monkeypatch):
    import bench_board_infer as I
    from contextlib import redirect_stdout
    spec = importlib.util.spec_from_file_location("forest_byte_retention",
        Path(__file__).resolve().parents[1] / "bench/speed/forest_speed_arm.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    directory, rec, _ = capture(tmp_path, family="trees")
    data = SimpleNamespace(X_train=np.arange(12, dtype=np.float32).reshape(6, 2),
        X_test=np.zeros((3, 2), np.float32), _ours_Xtest=np.zeros((3, 2), np.float32),
        cat_idx=None, task="binary")
    calls = []
    def predict(x):
        calls.append(len(x))
        return np.full(len(x), len(calls), np.float32)
    monkeypatch.setattr(driver, "infer_spec", lambda *args, **kwargs: (predict, lambda raw: raw, "fixture"))
    monkeypatch.setattr(driver, "infer_metrics", lambda *args: [])
    monkeypatch.setattr(driver.spec, "per_arm_budget_s", lambda: 100)
    # Exactly one timer start/end for each existing call. The receipt is after
    # the clock; it cannot add a timed call, prediction, or model export.
    ticks = iter(range(12))
    monkeypatch.setattr(driver.time, "perf_counter", lambda: next(ticks))
    text = io.StringIO()
    with patch.dict(os.environ, {E.RECEIPTS: str(directory / "workers")}), redirect_stdout(text):
        driver.run_inference("rf", [SimpleNamespace(name="ours")], {"ours": object()},
                             data, 2, 5, float("inf"))
    assert calls == [3, 3, 3, 5, 5, 5]
    log = tmp_path / "inference.log"
    log.write_text(text.getvalue())
    bb = SimpleNamespace(base_cell=lambda ctx, race, arm, mode: dict(arm=arm, mode=mode),
                         _status=lambda ms, refused, rounds: "ok" if len(ms) == rounds and not refused else "partial",
                         strip_our_cpu_cells=lambda cells: cells, add_ratios=lambda cells: cells)
    rec["infer_cells"] = I.tree_cells(bb, {"rounds": 2},
        dict(arms=["ours"], our_arms={"ours": "identical"}), str(log), rec["cells"])
    members = {str(p.relative_to(directory)): p.read_bytes() for p in (directory / "scored").rglob("*.npz")}
    return directory, rec, members


def test_tree_inference_retains_every_scored_round_and_separates_fit_batches(tmp_path, monkeypatch):
    directory, rec, members = tree_inference(tmp_path, monkeypatch)
    refs, missing = E.scored_state_refs(rec, directory, members)
    assert not missing
    assert set(refs) == {("fit", "ours"), ("infer", "ours", "test"), ("infer", "ours", "large")}
    for batch, values, rows in (("test", [2, 3], 3), ("large", [5, 6], 5)):
        records = refs[("infer", "ours", batch)]
        assert [r["round"] for r in records] == [1, 2]
        for row, expected in zip(records, values):
            with np.load(io.BytesIO(members[row["output"]["member"]]), allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["prediction"], np.full(rows, expected, np.float32))
            assert row["input"]["shape"] == [rows, 2]
    E.seal(dict(out=str(tmp_path), measurement_inventory={}, measurement_source={"status": "local"},
                evidence_r2=False), directory, dict(source={}, dataset={"identity": "input-sha"}, hardware={}), rec)
    identities = set()
    for cell in rec["infer_cells"]:
        assert cell["times_ms"] == [1000.0, 1000.0]
        measurement = cell["measurement"]
        assert measurement["scored_state"]["phase"] == "infer"
        assert measurement["scored_state"]["batch"] == cell["batch"]
        identities.add(measurement["retained_outputs"]["canonical_sha256"])
    assert len(identities) == 2


@pytest.mark.parametrize("failure", ["missing", "corrupt", "wrong-batch", "fit-in-infer", "missing-round", "missing-all"])
def test_tree_inference_bad_evidence_does_not_hide_or_replace_other_cells(tmp_path, monkeypatch, failure):
    directory, rec, members = tree_inference(tmp_path, monkeypatch)
    bad = next(c for c in rec["infer_cells"] if c["batch"] == "test")
    receipt = bad["state_receipts"][0]
    member = str(Path(receipt["retained_output"]["path"]).relative_to(directory))
    if failure == "missing":
        del members[member]
    elif failure == "corrupt":
        members[member] = b"truncated npz"
    elif failure == "wrong-batch":
        receipt["batch"] = "large"
    elif failure == "fit-in-infer":
        bad["state_receipts"] = rec["cells"][0]["state_receipts"]
    elif failure == "missing-all":
        bad["state_receipts"] = []
    else:
        bad["state_receipts"].pop()
    refs, missing = E.scored_state_refs(rec, directory, members)
    assert set(missing) == {("infer", "ours", "test")}
    assert refs[("fit", "ours")][0]["output"]
    assert len(refs[("infer", "ours", "large")]) == 2
