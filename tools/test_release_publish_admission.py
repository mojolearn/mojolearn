"""Actual ledger/receipt replay off-machine; no network, build or publication."""
import copy
import io
import json
from pathlib import Path
import tarfile
import zipfile
import base64
import csv

import pytest
import yaml

import release_publish_admission as A
import release_requirement_receipts as R
import release_requirements as L
from test_release_requirement_receipts import SourceReceiptTests


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    fixture = SourceReceiptTests()
    fixture.setUp()
    source, version = fixture.wanted["source_revision"], "0.0.review"
    wheel = tmp_path / f"mojolearn-{version}-py3-none-macosx_11_0_arm64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"mojolearn-{version}.dist-info/METADATA", f"Name: mojolearn\nVersion: {version}\n")
        archive.writestr("mojolearn/identity_columns/COMMIT", source)
    config = dict(release_modes=["identical"], build_flags={}, **{"artifacts:apple": {wheel.name: A.sha256(wheel)}})
    fixture.wanted["inputs"]["configuration"] = config
    fixture.wanted["fingerprint"] = L.digest(fixture.wanted["inputs"])
    fixture.wanted.update(required=True, stage="pre", scope="all public endpoints", missing_inputs=[])
    manifest = dict(schema=L.SCHEMA, source_revision=source, version=version,
                    requirements={"runtime-source": fixture.wanted}, configuration=config)
    # Only source expansion is a small fixture. Actual ledger state, receipts,
    # raw producer parsing and all portable evidence checks remain real.
    monkeypatch.setattr(A, "source_revision", lambda root, revision: revision)
    monkeypatch.setattr(L, "requirements", lambda *args, **kwargs: copy.deepcopy(manifest))
    ledger = L.RequirementsLedger(tmp_path / "requirements.json")
    ledger.sync(manifest)
    run = ledger.begin("runtime-source")
    receipt = fixture.admit()
    counts = R.coverage_summary(receipt)
    ledger.record(run, "passed", [fixture.path], "actual full source gate fixture", 0, receipt=receipt, **counts)
    bundle = tmp_path / "admission.tar.gz"
    result = A.create_bundle(ledger.path, A.ROOT, source, [wheel], bundle)
    yield dict(fixture=fixture, manifest=manifest, ledger=ledger, bundle=bundle, result=result, wheel=wheel, source=source)
    fixture.doCleanups()


def rewrite_archive(admitted, tmp_path, mutate):
    with tarfile.open(admitted["bundle"], "r:gz") as archive:
        rows = [(item.name, archive.extractfile(item).read(), item.type) for item in archive]
    rows = mutate(rows)
    target = tmp_path / "changed.tar.gz"
    with tarfile.open(target, "w:gz") as archive:
        for name, data, kind in rows:
            info = tarfile.TarInfo(name)
            info.size, info.type = len(data), kind
            archive.addfile(info, io.BytesIO(data))
    return target


def test_off_machine_replay_preserves_original_ledger_and_receipt_json(admitted):
    before = admitted["ledger"].path.read_bytes()
    with A.open_bundle(admitted["bundle"], admitted["result"]["bundle_sha256"]) as (root, manifest):
        assert (root / "ledger.json").read_bytes() == before
        originals = list(manifest["evidence"])
    for original in originals:
        Path(original).unlink()
    result = A.verify_bundle(admitted["bundle"], admitted["result"]["bundle_sha256"], A.ROOT, admitted["source"])
    assert result["source_revision"] == admitted["source"]
    assert A.verify_wheels(admitted["bundle"], admitted["result"]["bundle_sha256"], admitted["source"], [admitted["wheel"]]) == 1


@pytest.mark.parametrize("damage", ["traversal", "duplicate", "symlink", "missing", "corrupt", "extra"])
def test_unsafe_or_incomplete_evidence_archives_are_refused(admitted, tmp_path, damage):
    def mutate(rows):
        name, data, kind = next(row for row in rows if row[0].startswith("objects/"))
        if damage == "traversal": return rows + [("../escape", b"x", tarfile.REGTYPE)]
        if damage == "duplicate": return rows + [(name, data, kind)]
        if damage == "symlink": return [(n, d, tarfile.SYMTYPE if n == name else k) for n, d, k in rows]
        if damage == "missing": return [row for row in rows if row[0] != name]
        if damage == "corrupt": return [(n, b"corrupt" if n == name else d, k) for n, d, k in rows]
        return rows + [("objects/" + "f" * 64, b"unreferenced", tarfile.REGTYPE)]
    target = rewrite_archive(admitted, tmp_path, mutate)
    with pytest.raises(ValueError):
        with A.open_bundle(target, A.sha256(target)):
            pass


def test_bundle_hash_and_duplicate_json_keys_are_refused(admitted, tmp_path):
    with pytest.raises(ValueError, match="SHA256"):
        with A.open_bundle(admitted["bundle"], "0" * 64):
            pass
    target = rewrite_archive(admitted, tmp_path, lambda rows: [
        (n, b'{"schema":"x","schema":"y"}' if n == "admission.json" else d, k) for n, d, k in rows])
    with pytest.raises(ValueError, match="duplicate"):
        with A.open_bundle(target, A.sha256(target)):
            pass


def test_wrong_source_wheel_or_later_ledger_update_prevents_upload(admitted, monkeypatch):
    sha = admitted["result"]["bundle_sha256"]
    with pytest.raises(ValueError, match="source"):
        A.verify_bundle(admitted["bundle"], sha, A.ROOT, "f" * 40)
    with monkeypatch.context() as altered:
        altered.setattr(A, "policy", lambda root: {"changed-validator": "f" * 64})
        with pytest.raises(ValueError, match="policy"):
            A.verify_bundle(admitted["bundle"], sha, A.ROOT, admitted["source"])
    admitted["ledger"].path.write_bytes(admitted["ledger"].path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="ledger changed"):
        A.verify_wheels(admitted["bundle"], sha, admitted["source"], [admitted["wheel"]], ledger_path=admitted["ledger"].path)
    admitted["wheel"].write_bytes(b"different wheel bytes")
    with pytest.raises(ValueError, match="wheel bytes differ"):
        A.verify_wheels(admitted["bundle"], sha, admitted["source"], [admitted["wheel"]])


def test_false_pass_or_shortened_ledger_cannot_reach_private_upload(admitted, monkeypatch, tmp_path):
    wanted = copy.deepcopy(admitted["manifest"]["requirements"]["runtime-source"])
    wanted.update(required=True, kind="runtime")
    admitted["manifest"]["requirements"]["missing-runtime"] = wanted
    monkeypatch.setattr(A, "r2_client", lambda: pytest.fail("network upload preceded canonical gate"))
    with pytest.raises(ValueError, match="blocks publication"):
        A.main(["bundle", "--ledger", str(admitted["ledger"].path), "--root", str(A.ROOT),
                "--source", admitted["source"], "--wheel", str(admitted["wheel"]),
                "--output", str(tmp_path / "blocked.tar.gz"), "--upload"])


def test_unqualified_artifact_and_forged_metadata_scope_are_refused(admitted, tmp_path):
    with pytest.raises(ValueError, match="artifact inputs"):
        A.wheel_identity(admitted["wheel"], "0.0.review", admitted["source"], {})
    def mutate(rows):
        changed = []
        for name, data, kind in rows:
            if name == "admission.json":
                manifest = json.loads(data)
                manifest["artifacts"][0]["scope"] = "file-free pinned dependencies"
                data = json.dumps(manifest).encode()
            changed.append((name, data, kind))
        return changed
    target = rewrite_archive(admitted, tmp_path, mutate)
    with pytest.raises(ValueError, match="source/metadata"):
        A.verify_wheels(target, A.sha256(target), admitted["source"], [admitted["wheel"]])


def test_file_free_linux_wheel_requires_exact_pins_and_no_entry_points(tmp_path):
    version, source = "0.0.review", "a" * 40
    prefix = "mojolearn-" + version + ".dist-info"
    wheel = tmp_path / f"mojolearn-{version}-py3-none-manylinux_2_28_x86_64.whl"
    files = {
        prefix + "/METADATA": ("Name: mojolearn\nVersion: " + version + "\n" + "".join(
            "Requires-Dist: " + name + "==" + version + "\n" for name in ("mojolearn-core", "mojolearn-nvidia", "mojolearn-amd"))).encode(),
        prefix + "/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-manylinux_2_28_x86_64\n",
        prefix + "/LINUX_PAYLOAD.json": json.dumps(dict(schema="mojolearn.linux-payload.v1", version=version,
            source_commit=source, split=dict(distribution="mojolearn"), release_modes=["identical"])).encode(),
    }

    def write():
        record = io.StringIO()
        writer = csv.writer(record, lineterminator="\n")
        for name, data in files.items():
            writer.writerow([name, "sha256=" + base64.urlsafe_b64encode(A.hashlib.sha256(data).digest()).rstrip(b"=").decode(), len(data)])
        writer.writerow([prefix + "/RECORD", "", ""])
        with zipfile.ZipFile(wheel, "w") as archive:
            for name, data in files.items(): archive.writestr(name, data)
            archive.writestr(prefix + "/RECORD", record.getvalue())

    write()
    config = {"artifacts:nvidia": {"mojolearn-core": "b" * 64}, "artifacts:amd": {"mojolearn-core": "b" * 64}}
    assert A.wheel_identity(wheel, version, source, config)["scope"] == "file-free pinned dependencies"
    files[prefix + "/entry_points.txt"] = b"[console_scripts]\nextra=unknown:run\n"
    write()
    with pytest.raises(ValueError, match="entry points"):
        A.wheel_identity(wheel, version, source, config)


def test_private_r2_fetch_binds_content_hash_without_network(tmp_path, monkeypatch):
    class Storage:
        def get(self, key):
            assert key == A.PREFIX + expected + ".tar.gz"
            return payload
    payload = b"private content"
    expected = A.hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(A, "r2_client", lambda: Storage())
    A.fetch_bundle(expected, tmp_path / "bundle")
    assert (tmp_path / "bundle").read_bytes() == payload
    payload = b"different content"
    with pytest.raises(ValueError, match="digest"):
        A.fetch_bundle(expected, tmp_path / "bad")


def test_all_eight_pypi_publishers_require_real_ledger_guard_and_exact_bytes():
    workflow = yaml.safe_load((A.ROOT / ".github/workflows/release-provenance.yml").read_text())
    jobs = workflow["jobs"]
    guard = jobs["ledger_admission"]
    assert guard["needs"] == "validate_inputs"
    assert any("release_publish_admission.py verify" in step.get("run", "") for step in guard["steps"])
    count = 0
    for job in jobs.values():
        steps = job.get("steps", [])
        for i, step in enumerate(steps):
            if "pypa/gh-action-pypi-publish@" not in step.get("uses", ""):
                continue
            count += 1
            assert "ledger_admission" in job["needs"]
            assert "needs.ledger_admission.result == 'success'" in job["if"]
            assert "release_publish_admission.py wheels" in steps[i - 1]["run"]
            assert "--wheel-dir upload" in steps[i - 1]["run"]
    assert count == 8
