"""A Linux leg of an earlier freeze is taken by its compile requirements, and
only then (tools/release.py find_requirement_leg; tools/release_reuse.py
assemble_linux `host_from`). Inert bytes; no build, no ledger on disk."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import release
import release_reuse

HOST_KEY = "linux-64/host/-/host/_mojolearn_core_host"


def identity(vendor, arch, device="d" * 64, host="a" * 64, runtime="e" * 64):
    doc = dict(set=f"{vendor}/{arch}", bindings={f"linux-64/{vendor}/{arch}/identical/_mojolearn": device,
                                                 HOST_KEY: host}, runtime=runtime)
    return dict(doc, digest=release_reuse.digest_of(doc))


def controller(tmp_path, vendor="hip", arch="gfx942", actions=None, run_ids=None, prior=None, marker=True):
    names = [f"compile-amd/{arch}/a", f"compile-amd/{arch}/b"]
    wanted = {n: dict(fingerprint="f-" + n) for n in names}
    rows = {n: dict(action=(actions or {}).get(n, "reuse"), attempt="att-" + n) for n in names}
    attempts = {"att-" + n: dict(id="att-" + n, status="passed", fingerprint="f-" + n, run_id="run-" + n)
                for n in names}
    runs = {n: dict(run_id=(run_ids or {}).get(n, "run-" + n), fingerprint="f-" + n) for n in names}
    old = dict(source_commit="b" * 40, compile_requirements=dict(runs=runs), tooling_digest="t" * 64,
               set_identity=prior or identity(vendor, arch, host="b" * 64, runtime="f" * 64),
               release_build=str(tmp_path / "old/release-build"), exit_file=str(tmp_path / "old.exit"),
               out_dir=str(tmp_path / "old"))
    ctx = SimpleNamespace(commit="c" * 40, dry=False, readonly=False, rel=tmp_path / "rel",
                          set_identity=lambda v, a: identity(v, a),
                          packer_takes_legs=lambda: True,
                          requirement_units=lambda v, a: (wanted, rows, attempts),
                          leg_candidates=lambda *a: [("old-freeze", old)],
                          tooling=lambda: dict(commit="c" * 40, digest="u" * 64),
                          admission_path=lambda n: tmp_path / f"{n}.reused.json",
                          PTX_HOST_OWNER_MARKER=release.Release.PTX_HOST_OWNER_MARKER)
    return ctx, marker


@pytest.fixture
def verified(monkeypatch):
    state = dict(tree=(True, "", "p" * 64), overlay=(True, ""), ptx="")
    monkeypatch.setattr(release, "verify_leg_tree", lambda *a, **k: state["tree"])
    monkeypatch.setattr(release, "overlay_verified", lambda *a, **k: state["overlay"])
    monkeypatch.setattr(release, "file_at", lambda commit, rel: state["ptx"])
    return state


def find(ctx, name):
    vendor, arch = name.split("-", 1)
    return release.Release.find_requirement_leg(ctx, name, vendor, arch, write=False)


def test_a_leg_whose_own_units_all_passed_is_taken_with_the_owner_host(tmp_path, verified):
    ctx, _ = controller(tmp_path)
    doc, why, retained = find(ctx, "hip-gfx942")
    assert doc and why is None and retained
    assert doc["kind"] == "compile-requirements" and doc["host_from"] == "cuda-sm_89"
    assert doc["built_from"] == "b" * 40 and doc["admitted_for"] == "c" * 40
    assert set(doc["requirements"]) == {"compile-amd/gfx942/a", "compile-amd/gfx942/b"}


def test_the_host_owner_is_never_taken_by_requirements(tmp_path, verified):
    ctx, _ = controller(tmp_path, vendor="cuda", arch="sm_89")
    doc, why, retained = find(ctx, "cuda-sm_89")
    assert doc is None and not retained and "host" in why


def test_an_open_unit_launches_the_leg(tmp_path, verified):
    ctx, _ = controller(tmp_path, actions={"compile-amd/gfx942/b": "run"})
    doc, why, retained = find(ctx, "hip-gfx942")
    assert doc is None and not retained and "no retained pass" in why


def test_passes_of_other_runs_or_changed_devices_or_bad_trees_are_refused(tmp_path, verified):
    ctx, _ = controller(tmp_path, run_ids={"compile-amd/gfx942/a": "someone-else"})
    doc, why, retained = find(ctx, "hip-gfx942")
    assert doc is None and retained and "not this leg's runs" in why
    ctx, _ = controller(tmp_path, prior=identity("hip", "gfx942", device="9" * 64))
    doc, why, retained = find(ctx, "hip-gfx942")
    assert doc is None and retained and "device binding identity differs" in why
    ctx, _ = controller(tmp_path)
    verified["tree"] = (False, "a binary differs from its proof", None)
    doc, why, retained = find(ctx, "hip-gfx942")
    assert doc is None and retained and "differs from its proof" in why
    verified["tree"] = (True, "", "p" * 64)
    verified["overlay"] = (False, "the route overlay did not verify")
    doc, why, retained = find(ctx, "hip-gfx942")
    assert doc is None and "overlay" in why


def test_the_ptx_set_needs_the_frozen_packer_to_carry_owner_host_bytes(tmp_path, verified):
    ctx, _ = controller(tmp_path, vendor="cuda", arch="sm_80")
    doc, why, retained = find(ctx, "cuda-sm_80")
    assert doc is None and retained and "ptx_reuse.py" in why
    verified["ptx"] = "# " + release.Release.PTX_HOST_OWNER_MARKER
    doc, why, retained = find(ctx, "cuda-sm_80")
    assert doc and doc["host_from"] == "cuda-sm_89"


# ------------------------------------------------------------ assembly
def _set(root, vendor, arch, device, host, lib):
    d = root / "sets" / vendor / arch
    (d / "identical").mkdir(parents=True)
    (d / "host").mkdir()
    (d / ".libs").mkdir()
    (d / "identical" / "_mojolearn.so").write_bytes(device)
    (d / "host" / "_mojolearn_core_host.so").write_bytes(host)
    (d / ".libs" / "libMojolearnMath.so").write_bytes(lib)
    (d / "manifest.json").write_text("{}\n")
    return d


def _plan():
    rows = []
    for vendor, arch in release_reuse.LINUX_SETS:
        rows.append(dict(target=release_reuse.LINUX, tier="identical", vendor=vendor, arch=arch, name="_mojolearn",
                         decision="BUILD", archive_path=f"mojolearn/{vendor}/{arch}/identical/_mojolearn.so",
                         set_rel="identical/_mojolearn.so", identity_digest="1" * 64))
    rows.append(dict(target=release_reuse.LINUX, tier="host", vendor="host", arch="", name="_mojolearn_core_host",
                     decision="BUILD", archive_path="mojolearn/host/_mojolearn_core_host.so",
                     set_rel="host/_mojolearn_core_host.so", identity_digest="2" * 64))
    return dict(previous={}, runtime=dict(decision="BUILD"), rows=rows, commit="c" * 40)


def _origin(leg, commit):
    return dict(source_commit=commit, leg=leg, proof_sha256="p" * 64, admitted_for="c" * 40,
                set_identity_digest="s" * 64, tooling_digest="t" * 64)


def test_assembly_ships_the_owner_host_bytes_in_a_requirement_taken_set(tmp_path):
    owner = _set(tmp_path / "owner", "cuda", "sm_89", b"dev89", b"NEW host", b"lib")
    ptx = _set(tmp_path / "ptx", "cuda", "sm_80", b"dev80", b"NEW host", b"lib")
    old = _set(tmp_path / "old", "hip", "gfx942", b"devamd", b"OLD host", b"lib")
    leg_reuse = {"hip-gfx942": dict(dir=old, origin=_origin("hip-gfx942", "b" * 40),
                                    host_from=dict(name="cuda-sm_89", dir=owner,
                                                   origin=_origin("cuda-sm_89", "c" * 40)))}
    out = release_reuse.assemble_linux(_plan(), None, {"cuda-sm_89": owner, "cuda-sm_80": ptx},
                                       tmp_path / "dest", say=lambda *_: None, leg_reuse=leg_reuse)
    amd = Path(out["hip/gfx942"]) if isinstance(out, dict) and "hip/gfx942" in out else tmp_path / "dest/hip/gfx942"
    assert (amd / "host/_mojolearn_core_host.so").read_bytes() == b"NEW host"
    assert (amd / "identical/_mojolearn.so").read_bytes() == b"devamd"
    reuse = json.loads((amd / "reuse.json").read_text())
    assert reuse["files"]["host/_mojolearn_core_host.so"]["leg"] == "cuda-sm_89"
    assert reuse["files"]["identical/_mojolearn.so"]["leg"] == "hip-gfx942"
    assert set(reuse["from_legs"]) == {"hip-gfx942", "cuda-sm_89"}


def test_assembly_refuses_a_requirement_taken_set_with_another_runtime(tmp_path):
    owner = _set(tmp_path / "owner", "cuda", "sm_89", b"dev89", b"NEW host", b"lib")
    ptx = _set(tmp_path / "ptx", "cuda", "sm_80", b"dev80", b"NEW host", b"lib")
    old = _set(tmp_path / "old", "hip", "gfx942", b"devamd", b"OLD host", b"OTHER lib")
    leg_reuse = {"hip-gfx942": dict(dir=old, origin=_origin("hip-gfx942", "b" * 40),
                                    host_from=dict(name="cuda-sm_89", dir=owner,
                                                   origin=_origin("cuda-sm_89", "c" * 40)))}
    with pytest.raises(SystemExit, match="runtime closure differs"):
        release_reuse.assemble_linux(_plan(), None, {"cuda-sm_89": owner, "cuda-sm_80": ptx},
                                     tmp_path / "dest", say=lambda *_: None, leg_reuse=leg_reuse)


def test_a_taken_host_owner_ships_its_host_in_the_legs_built_here(tmp_path):
    owner = _set(tmp_path / "owner", "cuda", "sm_89", b"dev89", b"OWNER host", b"lib")
    ptx = _set(tmp_path / "ptx", "cuda", "sm_80", b"dev80", b"PTX host", b"lib")
    amd = _set(tmp_path / "amd", "hip", "gfx942", b"devamd", b"AMD-built host", b"lib")
    owner_from = dict(name="cuda-sm_89", dir=owner, origin=_origin("cuda-sm_89", "b" * 40))
    leg_reuse = {"cuda-sm_89": dict(dir=owner, origin=_origin("cuda-sm_89", "b" * 40))}
    release_reuse.assemble_linux(_plan(), None, {"hip-gfx942": amd, "cuda-sm_80": ptx}, tmp_path / "dest",
                                 say=lambda *_: None, leg_reuse=leg_reuse, host_owner=owner_from)
    for vendor, arch in (("hip", "gfx942"), ("cuda", "sm_80"), ("cuda", "sm_89")):
        sdir = tmp_path / "dest" / vendor / arch
        assert (sdir / "host/_mojolearn_core_host.so").read_bytes() == b"OWNER host", (vendor, arch)
    reuse = json.loads((tmp_path / "dest/hip/gfx942/reuse.json").read_text())
    assert reuse["files"]["host/_mojolearn_core_host.so"]["leg"] == "cuda-sm_89"
    assert "identical/_mojolearn.so" not in reuse["files"]        # the built leg's own proof covers it
    amd2 = _set(tmp_path / "amd2", "hip", "gfx942", b"devamd", b"AMD-built host", b"OTHER lib")
    with pytest.raises(SystemExit, match="runtime closure differs"):
        release_reuse.assemble_linux(_plan(), None, {"hip-gfx942": amd2, "cuda-sm_80": ptx}, tmp_path / "dest2",
                                     say=lambda *_: None, leg_reuse=leg_reuse, host_owner=owner_from)


def test_host_owner_origin_takes_a_whole_taken_owner_and_refuses_a_requirement_taken_one(tmp_path):
    whole = dict(set_dir=str(tmp_path / "s"), built_from="b" * 40, leg="cuda-sm_89", proof_sha256="p" * 64,
                 admitted_for="c" * 40, set_identity_digest="s" * 64, tooling_digest="t" * 64,
                 release_build=str(tmp_path / "rb"))
    ctx = SimpleNamespace(admitted_leg=lambda n: whole if n == "cuda-sm_89" else None)
    ctx.leg_origin = lambda d: release.Release.leg_origin(ctx, d)
    got = release.Release.host_owner_origin(ctx)
    assert got["name"] == "cuda-sm_89" and got["origin"]["source_commit"] == "b" * 40
    ctx.admitted_leg = lambda n: dict(whole, kind="compile-requirements")
    with pytest.raises(release.StepFailed, match="compile requirements"):
        release.Release.host_owner_origin(ctx)
