# SPDX-License-Identifier: Apache-2.0
"""tools/release_evidence_set.py: the one committed release evidence set (no network, no box)."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_evidence_set as E  # noqa: E402
import release_reuse  # noqa: E402

COMMIT = "a" * 40
PLATFORMS = ["linux", "nvidia", "amd", "linux-meta", "macos"]


def put(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc if isinstance(doc, str) else json.dumps(doc))
    return path


def make_release(tmp, version="0.9.1", commit=COMMIT, single=False):
    """A fake freeze directory holding every item a default-mode release produces."""
    rel = tmp / ("ev-" + version) / commit[:12]
    staged = rel / "record-staged"
    wheels = {}
    for p in PLATFORMS:
        put(staged / ("alpha-manifest-%s.json" % p), dict(version=version, files={"w-%s.whl" % p: "0" * 64}))
        put(staged / ("file-admission-%s.json" % p), dict(passed=True))
        wheels[p] = put(tmp / "wheels" / version / ("mojolearn_%s-%s-py3-none-any.whl" % (p.replace("-", "_"), version)),
                        "wheel bytes " + p + version)
    put(staged / "release-provenance.json", dict(schema="mojolearn.release-provenance.v1", version=version,
                                                 source_commit=commit))
    put(staged / "binding-identities.json", dict(bindings={}))
    put(rel / "identity-at-freeze" / "report.json", dict(schema="mojolearn.identity-at-freeze.v1", adopted=True))
    for v in ("nvidia", "amd"):
        put(rel / "gates" / v / "gate-smoke.json", dict(version=version, vendor=v, source_commit=commit, verdict="PASS"))
        put(rel / "gates" / v / ("runtime-%s" % v) / "runtime-report.json", dict(status="passed"))
        put(rel / "gates" / v / "job" / "box-facts.txt",
            "nproc=48\ncpu_model=Some CPU\nmem_total_kb=1000\nkernel=6.8\ngpu=Some GPU, 48 GB, 1.0\n")
        put(rel / "gates" / v / "job" / "busy-cores.tsv", "1\t40\t48\n2\t46\t48\n")
    put(rel / "gate-apple" / "adapters" / "adapter-smoke.json", dict(status="PASSED"))
    put(rel / "rehearsal.log", "PASS docs-facts (3s)\nFAIL x (2s, exit 1) log y\n# REHEARSAL PASSED\n")
    put(rel / "cross-compile.json", dict(schema="mojolearn.release-cross-compile.v1", commit=commit, verdict="PASS"))
    put(rel / "reuse" / "plan.json", dict(commit=commit, rows=[dict(key="k", decision="BUILD", reason="changed")],
                                          previous=dict(version="0.9.0")))
    put(rel / "testpypi-rehearsal" / "rehearsal.json", dict(version=version, commit=commit, problems=[]))
    for v in ("apple", "nvidia", "amd"):
        put(rel / ("published-%s" % v) / "published-report.json", dict(version=version, ok=True))
    table = put(rel / "table.json", dict(cells={"knn/base": {"fit": dict(ref="abc", cols={"nvidia": 1, "amd": 2}),
                                                             "x": dict(ref="n/a:none", cols={})}}))
    steps = {s: dict(done=True, commit=commit, at="t", result="PASS " + s) for s in
             ("rehearsal", "finish-line", "publish-macos")}
    return E.Context(version=version, commit=commit, tooling_commit=commit, date="2026-11-06", rel=rel, staged=staged,
                     published=list(PLATFORMS), wheels=wheels, single=single,
                     gate_dirs={v: rel / "gates" / v for v in ("nvidia", "amd")}, steps=steps, adapters=True,
                     table=table, table_ref=commit + ":table.json", box_tags=["rel-gate-nv-1"])


def files_of(root):
    d = Path(root) / E.EVIDENCE_DIR
    return sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())


def test_the_set_is_written_once_and_replaces_the_previous_one_completely(tmp_path):
    root = tmp_path / "repo"
    first = make_release(tmp_path, "0.9.1")
    put(first.rel / "gates" / "nvidia" / "autograd-smoke.json", dict(version="0.9.1"))   # optional, only in 0.9.1
    E.write_set(first, root)
    assert "gates/nvidia/autograd-smoke.json" in files_of(root)
    put(root / E.EVIDENCE_DIR / "planted-stale.json", "{}")            # anything left from before
    second = make_release(tmp_path, "0.9.2", commit="b" * 40)
    manifest = E.write_set(second, root)
    names = files_of(root)
    assert "planted-stale.json" not in names and "gates/nvidia/autograd-smoke.json" not in names
    assert manifest["release"]["version"] == "0.9.2"
    on_disk = json.loads((root / E.EVIDENCE_DIR / "manifest.json").read_text())
    assert on_disk["release"]["source_commit"] == "b" * 40
    assert json.loads((root / E.EVIDENCE_DIR / "release-provenance.json").read_text())["version"] == "0.9.2"
    assert not list((root / "bench" / "results").glob(".release_evidence.new-*"))


def test_git_add_of_the_directory_drops_the_old_sets_files(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True, text=True).stdout
    run("init", "-q")
    run("config", "user.email", "fixture@example.invalid")
    run("config", "user.name", "Fixture")
    first = make_release(tmp_path, "0.9.1")
    put(first.rel / "gates" / "amd" / "autograd-smoke.json", dict(version="0.9.1"))
    E.write_set(first, root)
    run("add", "-A", "--", E.EVIDENCE_DIR)
    run("commit", "-qm", "0.9.1")
    run("tag", "v0.9.1")
    E.write_set(make_release(tmp_path, "0.9.2", commit="b" * 40), root)
    run("add", "-A", "--", E.EVIDENCE_DIR)
    run("commit", "-qm", "0.9.2")
    tracked = run("ls-files", E.EVIDENCE_DIR).split()
    assert E.EVIDENCE_DIR + "/gates/amd/autograd-smoke.json" not in tracked
    old = json.loads(run("show", "v0.9.1:%s/manifest.json" % E.EVIDENCE_DIR))
    assert old["release"]["version"] == "0.9.1"
    assert [v for v, _ in E.provenance_history(root, "HEAD")] == ["0.9.2", "0.9.1"]


def test_a_missing_required_item_fails_by_name_and_leaves_the_previous_set(tmp_path):
    root = tmp_path / "repo"
    E.write_set(make_release(tmp_path, "0.9.1"), root)
    before = files_of(root)
    ctx = make_release(tmp_path, "0.9.2", commit="b" * 40)
    (ctx.rel / "gates" / "amd" / "gate-smoke.json").unlink()
    (ctx.rel / "testpypi-rehearsal" / "rehearsal.json").unlink()
    with pytest.raises(E.EvidenceError) as err:
        E.write_set(ctx, root)
    assert "gates/amd/gate-smoke.json" in str(err.value) and "publish/testpypi-rehearsal.json" in str(err.value)
    assert files_of(root) == before
    assert json.loads((root / E.EVIDENCE_DIR / "manifest.json").read_text())["release"]["version"] == "0.9.1"


def test_optional_items_are_listed_as_absent_not_required():
    rows = {d: req for d, _, req, _, _ in E.expand(E.Context(version="1", commit=COMMIT, tooling_commit=COMMIT,
                                                            date="d", rel=Path("/r"), staged=Path("/s"),
                                                            published=PLATFORMS))}
    assert rows["gates/nvidia/gate-smoke.json"] is True and rows["gates/nvidia/autograd-smoke.json"] is False
    assert rows["gates/nvidia/shardcheck.json"] if "gates/nvidia/shardcheck.json" in rows else True
    assert "gates/cpu/checks.json" not in rows                    # the CPU column is the one-round mode only
    single = {d: req for d, _, req, _, _ in E.expand(E.Context(version="1", commit=COMMIT, tooling_commit=COMMIT,
                                                              date="d", rel=Path("/r"), staged=Path("/s"),
                                                              published=PLATFORMS, single=True))}
    for name in ("gates/nvidia/shardcheck.json", "gates/amd/measuresmoke.json", "gates/cpu/checks.json",
                 "identity/admitted.json", "gates/amd/check-units.json"):
        assert single[name] is True, name
    assert single["gates/nvidia/gate-smoke.json"] is False


def test_a_stale_report_from_another_release_fails(tmp_path):
    ctx = make_release(tmp_path, "0.9.2", commit="b" * 40)
    put(ctx.rel / "gates" / "nvidia" / "gate-smoke.json", dict(version="0.9.1", verdict="PASS"))
    with pytest.raises(E.EvidenceError, match="gates/nvidia/gate-smoke.json.*stale"):
        E.write_set(ctx, tmp_path / "repo")


def test_nothing_is_taken_from_the_previous_set(tmp_path):
    root = tmp_path / "repo"
    E.write_set(make_release(tmp_path, "0.9.1"), root)
    ctx = make_release(tmp_path, "0.9.2", commit="b" * 40)
    ctx.gate_dirs["nvidia"] = root / E.EVIDENCE_DIR / "gates" / "nvidia"
    with pytest.raises(E.EvidenceError, match="nothing is carried over|stale"):
        E.write_set(ctx, root)


def test_size_caps(tmp_path, monkeypatch):
    ctx = make_release(tmp_path, "0.9.1")
    put(ctx.rel / "gates" / "nvidia" / "gate-smoke.json", dict(version="0.9.1", pad="x" * (E.FILE_CAP + 10)))
    with pytest.raises(E.EvidenceError, match="gates/nvidia/gate-smoke.json is .* over the .*cap per file"):
        E.write_set(ctx, tmp_path / "repo")
    monkeypatch.setattr(E, "TOTAL_CAP", 500)
    with pytest.raises(E.EvidenceError, match="over the 500-byte cap"):
        E.write_set(make_release(tmp_path, "0.9.1"), tmp_path / "repo2")


SECRETS = dict(token="ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4", key="AKIAABCDEFGHIJKLMNOP",
               url="https://acct.r2.cloudflarestorage.com/b/k.tgz?X-Amz-Algorithm=AWS4&X-Amz-Signature=deadbeef01",
               ip="10.1.2.3", ip2="47.47.180.32", email="someone.private@example.org",
               pw='"api_key": "s3cr3tvalue123"', bearer="Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345")


def test_scrubbing_keeps_every_secret_out_of_the_set(tmp_path):
    root = tmp_path / "repo"
    ctx = make_release(tmp_path, "0.9.1")
    put(ctx.rel / "gates" / "nvidia" / "gate-smoke.json", dict(version="0.9.1", verdict="PASS", notes=SECRETS,
                                                               loopback="127.0.0.1:8080", version_like="0.8.65"))
    ctx.steps["finish-line"]["result"] = "PASS see " + SECRETS["url"] + " by " + SECRETS["email"]
    manifest = E.write_set(ctx, root)
    blob = "".join((root / E.EVIDENCE_DIR / f).read_text(errors="replace") for f in files_of(root))
    for value in ("A1b2C3d4E5f6G7h8", "AKIAABCDEFGHIJKLMNOP", "X-Amz-Signature=deadbeef01", "10.1.2.3",
                  "47.47.180.32", "someone.private@example.org", "s3cr3tvalue123", "abcdefghijklmnopqrstuvwxyz012345"):
        assert value not in blob, value
    assert "127.0.0.1:8080" in blob and "0.8.65" in blob
    item = next(i for i in manifest["items"] if i["path"] == "gates/nvidia/gate-smoke.json")
    assert item["redactions"] >= 7 and item["source_sha256"] != item["sha256"]
    assert E.audit(root / E.EVIDENCE_DIR) == []


def test_a_secret_in_a_verbatim_item_fails_instead_of_editing_it(tmp_path):
    ctx = make_release(tmp_path, "0.9.1")
    put(ctx.staged / "binding-identities.json", dict(bindings={}, note="pushed by " + SECRETS["email"]))
    with pytest.raises(E.EvidenceError, match="binding-identities.json holds email; a verbatim item is never edited"):
        E.write_set(ctx, tmp_path / "repo")


def test_manifest_hashes_match_and_name_producer_and_wheel_bytes(tmp_path):
    root = tmp_path / "repo"
    ctx = make_release(tmp_path, "0.9.1")
    manifest = E.write_set(ctx, root)
    d = root / E.EVIDENCE_DIR
    assert E.audit(d) == []
    for item in manifest["items"]:
        assert E.sha256_file(d / item["path"]) == item["sha256"]
        assert item["producer"]
    gate = next(i for i in manifest["items"] if i["path"] == "gates/amd/gate-smoke.json")
    assert sorted(a["platform"] for a in gate["against"]) == ["amd", "linux"]
    assert gate["against"][0]["sha256"] == E.sha256_file(ctx.wheels[gate["against"][0]["platform"]])
    assert {w["platform"] for w in manifest["wheels"]} == set(PLATFORMS)
    facts = json.loads((d / "gates/nvidia/box-facts.json").read_text())
    assert facts["cores"] == 48 and facts["busy_cores"] == dict(samples=2, median=46, max=46, of=48)
    summary = json.loads((d / "identity/table-summary.json").read_text())
    assert summary["parts"] == 1 and summary["parts_equal_on_every_column"] == 1
    assert any(r.get("r2_tag") == "rel-gate-nv-1" for r in manifest["references"])
    assert any(r["what"] == "reuse plan" for r in manifest["references"])
    assert not any(str(r.get("path", "")).startswith(str(Path.home())) for r in manifest["references"])
    # a tampered file is caught by the audit
    (d / "steps.json").write_text("{}")
    assert any("steps.json does not match" in p for p in E.audit(d))


def test_previous_release_reads_the_one_set(tmp_path):
    root = tmp_path / "repo"
    assert release_reuse.previous_release(root) is None
    put(root / E.LEGACY_DIR / "2026-11-04_pypi_0865" / "alpha-manifest-linux.json", dict(version="0.8.65"))
    assert release_reuse.previous_release(root) is None           # the dated records are not read any more
    put(root / E.EVIDENCE_DIR / "release-provenance.json", dict(version="0.8.66"))
    put(root / E.EVIDENCE_DIR / "alpha-manifest-linux.json",
        dict(version="0.8.66", files={"mojolearn_core-0.8.66-x.whl": "1" * 64},
             qualification_reuse=dict(source_commit=COMMIT)))
    put(root / E.EVIDENCE_DIR / "binding-identities.json", {})
    prev = release_reuse.previous_release(root)
    assert prev["version"] == "0.8.66" and prev["source_commit"] == COMMIT
    assert prev["record_dir"] == str(root / E.EVIDENCE_DIR)
    assert prev["identities"] == str(root / E.EVIDENCE_DIR / "binding-identities.json")
    assert prev["linux"] == dict(wheel="mojolearn_core-0.8.66-x.whl", sha256="1" * 64)


def test_history_reads_removed_dated_records_through_git(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True, text=True).stdout
    run("init", "-q")
    run("config", "user.email", "fixture@example.invalid")
    run("config", "user.name", "Fixture")
    prov = lambda v: dict(schema="mojolearn.release-provenance.v1", version=v, source_commit=COMMIT)
    put(root / E.LEGACY_DIR / "2026-11-01_pypi_0864" / "release-provenance.json", prov("0.8.64"))
    put(root / E.LEGACY_DIR / "2026-11-04_pypi_0865" / "release-provenance.json", prov("0.8.65"))
    run("add", "-A")
    run("commit", "-qm", "dated records")
    run("mv", E.LEGACY_DIR + "/2026-11-04_pypi_0865", E.EVIDENCE_DIR)
    run("rm", "-rq", E.LEGACY_DIR)
    run("commit", "-qm", "one set")
    assert not (root / E.LEGACY_DIR).exists()
    assert sorted(v for v, _ in E.provenance_history(root, "HEAD")) == ["0.8.64", "0.8.65"]


def test_the_coverage_evidence_reads_the_set_through_git(tmp_path):
    import release_coverage_source_evidence as C
    root = tmp_path / "repo"
    root.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True, text=True).stdout
    run("init", "-q")
    run("config", "user.email", "fixture@example.invalid")
    run("config", "user.name", "Fixture")
    put(root / E.EVIDENCE_DIR / "release-provenance.json",
        dict(schema="mojolearn.release-provenance.v1", version="0.9.0", source_commit="1" * 40))
    run("add", "-A")
    run("commit", "-qm", "0.9.0")
    put(root / E.EVIDENCE_DIR / "release-provenance.json",
        dict(schema="mojolearn.release-provenance.v1", version="0.9.1", source_commit="2" * 40))
    run("commit", "-qam", "0.9.1")
    assert C.published_base(root, "HEAD", "0.9.2") == ("0.9.1", "2" * 40)
    assert C.published_base(root, "HEAD", "0.9.1") == ("0.9.0", "1" * 40)      # from the set's git history
