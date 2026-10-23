"""Tier-selection integration with inert files and command spies only."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "packaging/linux"))
import release_modes as modes
import release_reuse as reuse
import release
import verify_linux_surface_qualification as surface
from test_release import args
from test_release_061_inventory import ReleaseInventory
from test_split_wheels import make_sets


def test_selection_is_canonical_and_legacy_scope_is_explicit(monkeypatch):
    assert modes.release_modes({}) == ("fast", "identical")
    assert modes.release_modes({}, default=("identical",)) == ("identical",)
    assert modes.parse_modes("identical fast") == ("fast", "identical")
    monkeypatch.setenv(modes.ENV, "identical")
    assert modes.release_modes() == ("identical",)
    assert modes.record_modes({}) == ("fast", "identical")
    assert modes.record_modes({"release_modes": ["identical"]}) == ("identical",)


@pytest.mark.parametrize("value", ["", "identical identical", "deterministic", "identical,fast", [None]])
def test_invalid_selection_is_refused(value):
    with pytest.raises(ValueError, match="distinct supported modes"):
        modes.parse_modes(value)


def test_reuse_rows_and_set_identity_follow_selection():
    for target in (reuse.LINUX, reuse.MACOS):
        selected = reuse.bindings(target, ("identical",))
        legacy = reuse.bindings(target)
        assert {b.tier for b in selected} == {"identical", "host"}
        assert {b.key for b in selected} == {b.key for b in legacy if b.tier != "fast"}
    runtime = {"identity_digest": "runtime", "identity": {
        "toolchain": {"packages": {"mojo": "pinned"}}, "builders": {}, "image": {"os": "pinned"},
    }}
    def plan(selected):
        return {"runtime": runtime, "rows": [dict(b.row(), identity_digest=b.key,
                    identity={"source": {"digest": "source"}}) for b in selected]}
    # The real identity checker also requires compiler/source fields; use the
    # serializer's explicit row digest boundary to isolate tier-set selection.
    old = reuse.unreadable
    try:
        reuse.unreadable = lambda record: None
        a = reuse.set_identity(plan(reuse.bindings(reuse.LINUX)), "cuda", "sm_89")
        b = reuse.set_identity(plan(reuse.bindings(reuse.LINUX, ("identical",))), "cuda", "sm_89")
    finally:
        reuse.unreadable = old
    assert a["digest"] != b["digest"]


def controller(tmp_path, monkeypatch):
    monkeypatch.setenv(modes.ENV, "identical")
    obj = release.Release(args(state_dir=str(tmp_path / "state"), dry_run=True, build_backend="cpu-box"))
    obj.state["commit"] = "a" * 40
    obj.say = lambda message: None
    return obj


def test_cross_compile_and_mac_commands_select_only_identical(tmp_path, monkeypatch):
    obj = controller(tmp_path, monkeypatch)
    calls = []
    obj.must = lambda command, **kwargs: calls.append((command, kwargs))
    monkeypatch.setattr(reuse, "previous_release", lambda: {"source_commit": "b" * 40, "version": "0.0.1"})
    obj.step_cross_compile()
    assert "tiers=identical" in calls[0][0]
    monkeypatch.setattr(release.Release, "src", property(lambda self: tmp_path))
    obj.plan = lambda: {"rows": []}
    obj.macos_wheel = lambda: None
    obj.gpu_smoke = lambda: False
    import mac_slot
    monkeypatch.setattr(mac_slot, "slot_count", lambda: 4)
    obj.step_macos_build()
    assert calls[1][1]["env"][modes.ENV] == "identical"
    assert "--build-only" in calls[1][0]


def test_new_tier_selection_cannot_reuse_legacy_completed_state(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    (state / "state.json").write_text(json.dumps({"commit": "a" * 40, "steps": {
        "linux-builds": {"done": True, "commit": "a" * 40},
    }}))
    monkeypatch.setenv(modes.ENV, "identical")
    with pytest.raises(SystemExit, match="tier selection differs"):
        release.Release(args(state_dir=str(state), build_backend="cpu-box"))


@pytest.mark.parametrize("selected,expected_count", [(None, 10), (["identical"], 9)])
def test_release_admission_summary_counts_the_qualified_wheel_modes(tmp_path, monkeypatch, selected, expected_count):
    import check_linux_release_qualification as admission
    # File-only receipt fixture: isolate the summary from the independently
    # tested native inventory/installed-record checks; never load a product.
    audit = dict(assembly_profile=surface.RELEASE_PROFILE, sha256="wheel",
                 extension_hashes={}, sets={})
    if selected is not None:
        audit["release_modes"] = selected
    directory = tmp_path / "hip/gfx942"
    directory.mkdir(parents=True)
    (directory / "wheel-audit.json").write_text(json.dumps(audit))
    (directory / "qualification.json").write_text("{}")
    monkeypatch.setattr(admission, "tracked_native_inventory", lambda root: {})
    monkeypatch.setattr(admission, "inspect_wheel", lambda *args, **kwargs: ({}, {"hip/gfx942/identical": {}}))
    monkeypatch.setattr(admission, "release_audit", lambda *args: audit)
    monkeypatch.setattr(admission, "check_vendor", lambda *args: None)
    monkeypatch.setattr(admission, "check_corpora", lambda *args: None)
    monkeypatch.setattr(surface, "release_version", lambda root: "0.8.56")
    monkeypatch.setattr(surface, "read_declared_failures", lambda *args: (frozenset(), {}))
    result = admission.check_release061(tmp_path / "inert.whl", tmp_path, tmp_path,
                                        wheel_sha="wheel", vendors=("hip",))
    assert result["jobs_per_runtime_architecture"] == expected_count
    assert f"FULL {expected_count} installed jobs" in result["scope"]


@pytest.mark.parametrize("merged", [False, True])
def test_packer_loads_identical_only_set_and_refuses_unselected_artifacts(tmp_path, monkeypatch, merged):
    import pack_wheel as pack
    paths = make_sets(tmp_path, sets=(("hip", "gfx942"),), include_byte_lm=True, merged=merged)
    adir = Path(paths[0]) / "gfx942"
    monkeypatch.setenv(modes.ENV, "identical")
    with pytest.raises(SystemExit, match="unselected fast artifacts"):
        pack.load_set(paths[0], include_byte_lm=True)
    for path in adir.glob("_mojolearn*.so"):
        path.unlink()
    (adir / pack.MERGED_INDEX).unlink(missing_ok=True)
    for name in ("readback.txt", "arch_readback.txt"):
        p = adir / name
        p.write_text("\n".join(line for line in p.read_text().splitlines() if not line.startswith("fast ")) + "\n")
    loaded = pack.load_set(paths[0], include_byte_lm=True)
    assert len(loaded) == 1
    assert loaded[0].hosts
    assert all("/identical/" in rel for rel in loaded[0].files)
    victim = next(path for path in loaded[0].files.values() if path.suffix == ".so")
    victim.unlink()
    with pytest.raises(SystemExit, match="missing|differs"):
        pack.load_set(paths[0], include_byte_lm=True)


def selected_inventory(root, merged):
    fixture = ReleaseInventory()
    sets, proofs = fixture.fixture(root, merged=merged)
    selected = []
    for entry, path in zip(sets, proofs):
        files = {rel: file for rel, file in entry.files.items() if "/identical/" in rel}
        indexes = {rel: file for rel, file in entry.indexes.items() if rel.startswith(("identical/", "host/"))}
        selected.append(entry._replace(files=files, indexes=indexes))
        proof = json.loads(path.read_text())
        proof["release_modes"] = ["identical"]
        proof["extensions"] = {rel: digest for rel, digest in proof["extensions"].items() if "/identical/" in rel}
        if merged:
            proof["merged_indexes"] = {rel: row for rel, row in proof["merged_indexes"].items()
                                       if "/identical/" in rel or "/host/" in rel}
        path.write_text(json.dumps(proof))
    return selected, proofs


@pytest.mark.parametrize("merged", [False, True])
def test_release_payload_binds_selected_modes_and_refuses_untagged_subset(tmp_path, monkeypatch, merged):
    # Reuse the existing inert inventory fixture, dropping only its fast rows.
    import pack_wheel as pack
    selected, proofs = selected_inventory(tmp_path, merged)
    monkeypatch.setenv(modes.ENV, "identical")
    payload = pack.release_inventory(selected, proofs, "9.9.9", source_root=tmp_path)
    assert payload["release_modes"] == ["identical"]
    assert payload["optional_native"]["_mojolearn_estimators"]["supported_modes"] == ["identical"]
    proof = json.loads(proofs[0].read_text())
    proof.pop("release_modes")
    proofs[0].write_text(json.dumps(proof))
    with pytest.raises(SystemExit, match="tier selection differs"):
        pack.release_inventory(selected, proofs, "9.9.9", source_root=tmp_path)


def test_installed_jobs_use_artifact_metadata_not_ambient_selection(monkeypatch):
    monkeypatch.setenv(modes.ENV, "identical")
    assert ("smoke", "fast") in surface.expected_jobs({})
    selected = surface.expected_jobs({"release_modes": ["identical"], "assembly_profile": surface.RELEASE_PROFILE})
    assert ("smoke", "fast") not in selected
    assert ("smoke", "identical") in selected
    assert ("byte-lm", "identical") in selected


def test_remote_postflight_accepts_selected_proof_and_rejects_missing_index(tmp_path, monkeypatch):
    selected, proofs = selected_inventory(tmp_path / "fixture", True)
    entry, proof_path = selected[0], proofs[0]
    proof = json.loads(proof_path.read_text())
    out = tmp_path / "out"
    (out / "build").mkdir(parents=True)
    target = out / "build/build-provenance.json"
    target.write_text(json.dumps(proof))
    before = {key: proof[key] for key in ("source_commit", "source_inventory", "source_sha256")}
    before.update(vendor=entry.vendor, device_architecture=entry.arch)
    (out / "preflight.json").write_text(json.dumps(before))
    shell = (ROOT / "tools/release061_remote_build.sh").read_text()
    script = shell.split("<<'PYPOST'\n", 1)[1].split("\nPYPOST", 1)[0]
    helper = tmp_path / "postflight.py"
    helper.write_text(script)
    env = dict(os.environ, MOJOLEARN_RELEASE_MODES="identical")
    def run():
        return subprocess.run([sys.executable, str(helper), str(out)], cwd=ROOT, env=env,
                              capture_output=True, text=True)
    result = run()
    assert result.returncode == 0, result.stderr
    proof["merged_indexes"].pop(next(iter(proof["merged_indexes"])))
    target.write_text(json.dumps(proof))
    result = run()
    assert result.returncode != 0
    assert "lacks an index" in result.stderr
