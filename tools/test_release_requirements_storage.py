import copy
import json

import pytest

import release_requirements as R
import release_requirements_storage as S


def document():
    sources = {f"core/source_{i:04}.mojo": f"blob-{i}" for i in range(1000)}
    rows = {}
    for i in range(20):
        selected = dict(sources, **{f"local/{i}.mojo": f"local-{i}"})
        inputs = dict(native_sources=selected, flags={"mode": "identical"})
        rows[str(i)] = dict(inputs=inputs, fingerprint=R.digest(inputs),
                           provenance_sources=list(selected))
    return dict(schema=R.SCHEMA, requirements=rows,
        attempts=[dict(status="failed", inputs=copy.deepcopy(rows["0"]["inputs"]), receipt=None)], runs=[])


def test_content_addressed_storage_preserves_inputs_fingerprints_and_failed_history():
    old = document()
    packed = S.encode(old)
    restored = S.decode(json.loads(json.dumps(packed)))
    assert restored == old
    assert len(json.dumps(packed)) < len(json.dumps(old)) / 2
    assert restored["attempts"][0]["status"] == "failed"
    for row in restored["requirements"].values():
        assert row["fingerprint"] == R.digest(row["inputs"])


def test_repeated_source_maps_share_immutable_memory_without_sharing_mutable_contracts():
    doc = document()
    doc["requirements"]["same"] = copy.deepcopy(doc["requirements"]["0"])
    restored = S.decode(S.encode(doc))
    a, b = (restored["requirements"][key]["inputs"] for key in ("0", "same"))
    assert a is not b
    assert a["native_sources"] is b["native_sources"]
    assert copy.deepcopy(a["native_sources"]) is a["native_sources"]
    with pytest.raises(TypeError, match="immutable"):
        a["native_sources"]["core/source_0000.mojo"] = "corrupt"
    a["flags"]["mode"] = "new"
    assert b["flags"]["mode"] == "identical"


def test_legacy_plain_json_remains_readable_and_migrates_atomically(tmp_path):
    old = document()
    path = tmp_path / "requirements.json"
    path.write_text(json.dumps(old))
    ledger = R.RequirementsLedger(path)
    assert ledger.read() == old
    with ledger.edit() as doc:
        assert doc == old
    disk = json.loads(path.read_text())
    assert disk[S.TABLE]["schema"] == S.SCHEMA
    assert ledger.read() == old
    assert not list(tmp_path.glob("requirements.json.*"))


@pytest.mark.parametrize("damage", ["digest", "missing", "schema", "node"])
def test_corrupt_storage_never_qualifies(damage):
    packed = S.encode(document())
    table = packed[S.TABLE]
    if damage == "schema":
        table["schema"] = "unknown"
    elif damage == "missing":
        table["objects"].clear()
    elif damage == "node":
        bad = dict(kind="unknown", value={})
        key = S.identity(bad)
        table["objects"][key] = bad
        packed["requirements"]["0"]["inputs"] = {S.REF: key}
    else:
        next(iter(table["objects"].values()))["value"] = "tampered"
    with pytest.raises(ValueError):
        S.decode(packed)


def test_reference_shaped_user_data_and_reserved_nested_keys_roundtrip():
    doc = dict(schema=R.SCHEMA, inputs={S.REF: "literal user value"},
               receipt={S.TABLE: {"literal": True}})
    assert S.decode(S.encode(doc)) == doc


def test_failed_edit_does_not_replace_plain_history(tmp_path):
    path = tmp_path / "requirements.json"
    raw = json.dumps(document())
    path.write_text(raw)
    ledger = R.RequirementsLedger(path)
    with pytest.raises(RuntimeError):
        with ledger.edit() as doc:
            doc["attempts"].clear()
            raise RuntimeError("abort migration")
    assert path.read_text() == raw


def test_large_single_bucket_map_is_partitioned_only_once():
    keys = []
    number = 0
    while len(keys) < 70:
        key = f"core/source-{number}.mojo"
        if S.bucket_name(key) == "00":
            keys.append(key)
        number += 1
    doc = dict(inputs={key: "blob" for key in keys})
    assert S.decode(S.encode(doc)) == doc
