import copy
import json
from pathlib import Path

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
    raw = json.dumps(old).encode()
    path.write_bytes(raw)
    ledger = R.RequirementsLedger(path)
    assert ledger.read() == old
    with ledger.edit() as doc:
        assert doc == old
    disk = json.loads(path.read_text())
    assert disk[S.STORAGE]["schema"] == S.INDEX_SCHEMA
    assert ledger.read() == old
    assert not list(tmp_path.glob("requirements.json.*"))
    # The original bytes are kept, by their own SHA-256, beside the objects.
    kept = S.objects_dir(path) / "legacy" / (S.hashlib.sha256(raw).hexdigest() + ".json")
    assert kept.read_bytes() == raw


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


# ---------------------------------------------------------------- v2: index + immutable objects

def receipts_document():
    """Many attempts sharing one large producer provenance map (the 0.8.65 shape)."""
    doc = document()
    provenance = dict(schema="p", requirements={f"identity-apple/lane-{i:04}": f"{i:064x}" for i in range(400)})
    doc["attempts"] = []
    for i in range(30):
        receipt = dict(producer_provenance=copy.deepcopy(provenance), unit=f"lane-{i}",
                       evidence=[dict(path=f"/e/{i % 3}.json", sha256="0" * 64)])
        doc["attempts"].append(dict(id=f"attempt-{i}", run_id=f"run-{i}", requirement=str(i % 20), status="passed",
                                    fingerprint=doc["requirements"][str(i % 20)]["fingerprint"],
                                    receipt=receipt, evidence=copy.deepcopy(receipt["evidence"]),
                                    inputs=copy.deepcopy(doc["requirements"][str(i % 20)]["inputs"])))
    doc["status"] = {str(i): dict(status="passed", reasons=["same inputs"], attempt=f"attempt-{i}",
                                  receipt=copy.deepcopy(doc["attempts"][i]["receipt"])) for i in range(20)}
    return doc


def object_count(path):
    return sum(1 for _ in S.objects_dir(path).glob("[0-9a-f][0-9a-f]/*.json"))


def test_v2_round_trip_is_the_same_logical_document_with_a_small_index(tmp_path):
    old = receipts_document()
    path = tmp_path / "requirements.json"
    S.save(path, copy.deepcopy(old))
    doc, state = S.load(path)
    assert doc == old
    assert state.records["attempts"] and len(state.records["attempts"]) == 30
    # Each repeated receipt map is stored once; the index holds pointers, not receipts.
    assert path.stat().st_size < len(json.dumps(old)) / 10
    files = list(S.objects_dir(path).glob("[0-9a-f][0-9a-f]/*.json"))
    assert all(S.hashlib.sha256(f.read_bytes()).hexdigest() == f.stem for f in files)
    # Shared leaves are one immutable object in memory too.
    a, b = (doc["attempts"][n]["receipt"]["producer_provenance"]["requirements"] for n in (0, 1))
    assert a is b
    with pytest.raises(TypeError, match="immutable"):
        a["x"] = "corrupt"
    for row in doc["requirements"].values():
        assert row["fingerprint"] == R.digest(row["inputs"])


def test_reference_shaped_user_data_round_trips_in_v2(tmp_path):
    doc = dict(schema=R.SCHEMA, inputs={S.REF: "literal user value"}, receipt={S.RECORD: "x", "kind": 1},
               note={S.REF: "not a reference"}, attempts=[dict(id="a", evidence={S.REF: "y"})])
    path = tmp_path / "requirements.json"
    S.save(path, copy.deepcopy(doc))
    assert S.load(path)[0] == doc


def test_an_edit_writes_only_the_index_and_the_new_attempt(tmp_path):
    path = tmp_path / "requirements.json"
    S.save(path, receipts_document())
    before = {f: (f.stat().st_ino, f.stat().st_mtime_ns) for f in S.objects_dir(path).rglob("*.json")}
    ledger = R.RequirementsLedger(path)
    with ledger.edit() as doc:
        new = copy.deepcopy(doc["attempts"][0])
        new.update(id="attempt-new", run_id="run-new", summary="a new attempt")
        doc["attempts"].append(new)
        doc["status"]["0"]["attempt"] = "attempt-new"
    after = {f: (f.stat().st_ino, f.stat().st_mtime_ns) for f in S.objects_dir(path).rglob("*.json")}
    # Every old object is untouched; the only new object is the new attempt record (its receipt is shared).
    assert all(after[f] == stamp for f, stamp in before.items())
    added = set(after) - set(before)
    assert len(added) == 1
    assert json.loads(next(iter(added)).read_text())["id"] == "attempt-new"
    assert ledger.read()["attempts"][-1]["summary"] == "a new attempt"


@pytest.mark.parametrize("change", ["rewrite", "drop", "reorder", "remove-list"])
def test_historical_records_are_append_only(tmp_path, change):
    path = tmp_path / "requirements.json"
    S.save(path, receipts_document())
    raw = path.read_bytes()
    ledger = R.RequirementsLedger(path)
    with pytest.raises(ValueError, match="immutable and append-only"):
        with ledger.edit() as doc:
            if change == "rewrite":
                doc["attempts"][3]["status"] = "failed"
            elif change == "drop":
                del doc["attempts"][3]
            elif change == "reorder":
                doc["attempts"].reverse()
            else:
                del doc["attempts"]
    assert path.read_bytes() == raw


@pytest.mark.parametrize("damage", ["corrupt", "missing", "malformed-ref"])
def test_damaged_objects_never_read_as_a_ledger(tmp_path, damage):
    path = tmp_path / "requirements.json"
    S.save(path, receipts_document())
    index = json.loads(path.read_text())
    sha = index["attempts"][0][S.RECORD]
    target = S.ObjectStore(S.objects_dir(path)).path(sha)
    if damage == "corrupt":
        target.chmod(0o644)
        target.write_text(target.read_text().replace("passed", "failed"))
    elif damage == "missing":
        target.unlink()
    else:
        index["attempts"][0] = {S.RECORD: "../../etc/passwd"}
        path.write_text(json.dumps(index))
    with pytest.raises(ValueError):
        R.RequirementsLedger(path).read()


def legacy_v1(tmp_path, doc):
    path = tmp_path / "requirements.json"
    path.write_text(json.dumps(S.encode(doc), separators=(",", ":"), sort_keys=True) + "\n")
    return path


@pytest.mark.parametrize("form", ["v1", "plain"])
def test_migrate_preserves_every_attempt_and_keeps_the_original(tmp_path, form):
    old = receipts_document()
    old["runs"] = [dict(id=f"run-{i}", requirement=str(i % 20), inputs=copy.deepcopy(old["requirements"]["0"]["inputs"]))
                   for i in range(30)]
    old["interruptions"] = [dict(id="int-0", run_id="run-29", evidence=[dict(path="/stop", sha256="1" * 64)])]
    path = legacy_v1(tmp_path, old) if form == "v1" else tmp_path / "requirements.json"
    if form == "plain":
        path.write_text(json.dumps(old, indent=1))
    raw = path.read_bytes()
    result = R.RequirementsLedger(path).migrate(say=lambda line: None)
    assert result["status"] == "migrated"
    assert result["counts"]["attempts"] == 30 and result["counts"]["runs"] == 30
    assert result["counts"]["interruptions"] == 1
    assert S.is_index(json.loads(path.read_text()))
    assert R.RequirementsLedger(path).read() == old
    assert [a["id"] for a in R.RequirementsLedger(path).read()["attempts"]] == [a["id"] for a in old["attempts"]]
    assert Path(result["legacy"]).read_bytes() == raw
    assert R.RequirementsLedger(path).migrate(say=lambda line: None)["status"] == "already-v2"
    # The migrated ledger is append-only from here on.
    with pytest.raises(ValueError, match="append-only"):
        with R.RequirementsLedger(path).edit() as doc:
            doc["attempts"][0]["status"] = "failed"


def test_migrate_refuses_to_switch_when_verification_fails(tmp_path, monkeypatch):
    path = legacy_v1(tmp_path, receipts_document())
    raw = path.read_bytes()
    real = S.Decoder.record

    def lossy(self, sha):
        value = real(self, sha)
        value.pop("receipt", None)
        return value
    monkeypatch.setattr(S.Decoder, "record", lossy)
    with pytest.raises(ValueError, match="verification failed"):
        R.RequirementsLedger(path).migrate(say=lambda line: None)
    assert path.read_bytes() == raw


def test_concurrent_writers_keep_every_attempt_and_readers_see_complete_indexes(tmp_path):
    import threading
    path = tmp_path / "requirements.json"
    S.save(path, receipts_document())
    errors, stop = [], threading.Event()

    def writer(n):
        try:
            for k in range(5):
                # A separate ledger object per write, as separate producer processes would have.
                with R.RequirementsLedger(path).edit() as doc:
                    item = copy.deepcopy(doc["attempts"][0])
                    item.update(id=f"w{n}-{k}", run_id=f"w{n}-{k}")
                    doc["attempts"].append(item)
        except Exception as error:  # noqa: BLE001 - surfaced below
            errors.append(error)

    def reader():
        while not stop.is_set():
            try:
                doc = R.RequirementsLedger(path).read()
                assert len(doc["attempts"]) >= 30
            except Exception as error:  # noqa: BLE001
                errors.append(error)
                return

    readers = [threading.Thread(target=reader) for _ in range(2)]
    writers = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for thread in readers + writers:
        thread.start()
    for thread in writers:
        thread.join()
    stop.set()
    for thread in readers:
        thread.join()
    assert not errors
    ids = [a["id"] for a in R.RequirementsLedger(path).read()["attempts"]]
    assert len(ids) == 50 and {f"w{n}-{k}" for n in range(4) for k in range(5)} <= set(ids)
    assert not list(S.objects_dir(path).rglob(".write-*"))


def test_object_files_name_exactly_the_reachable_objects(tmp_path):
    path = tmp_path / "requirements.json"
    S.save(path, receipts_document())
    with R.RequirementsLedger(path).edit() as doc:
        doc["status"]["0"]["receipt"] = None   # an older status object is no longer reachable
    reached = {sha for sha, _ in S.object_files(path)}
    on_disk = {f.stem for f in S.objects_dir(path).glob("[0-9a-f][0-9a-f]/*.json")}
    assert reached <= on_disk
    index = json.loads(path.read_text())
    assert {ref[S.RECORD] for ref in index["attempts"]} <= reached
