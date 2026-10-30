#!/usr/bin/env python3
"""Package a frozen, retained per-binding Mac install without rebuilding it.

This is a candidate layout/ABI producer, not compile or identity admission.
Its input receipt binds every Python, verification and native file to an
immutable installed snapshot. Compile receipts and release reference admission
remain separate gates. Each index truthfully names one existing image/member;
no development cache, merged compilation or runtime qualification is invented.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

import release_macos_assembly as common

SCHEMA = "mojolearn.macos.installed-candidate.v1"
# These packaging/audit changes do not enter a native compilation. An arbitrary
# later commit is not accepted as closure-equivalent to the captured install.
PACKAGING_REFRESH_FILES = frozenset({
    "tools/release_macos_installed_candidate.py", "tools/test_release_macos_installed_candidate.py",
    "packaging/portable_math/wheel.py", "packaging/portable_math/test_wheel.py",
    "packaging/macos/shrink_wheel.py", "tools/test_macos_recompression.py",
})


def validate_refresh(root, original, current):
    if original == current:
        return
    changed = set(common.git(root, "diff", "--name-only", original, current).splitlines())
    common.require(changed <= PACKAGING_REFRESH_FILES,
                   "source refresh exceeds packaging-only inputs: " + ", ".join(sorted(changed - PACKAGING_REFRESH_FILES)))


def validate_source_root(root, commit):
    common.require(common.revision(root) == commit, "physical source checkout must match selected frozen source")


def member_path(base, relative):
    rel = PurePosixPath(relative)
    common.require(not rel.is_absolute() and ".." not in rel.parts and str(rel) == relative,
                   "unsafe receipt path: " + relative)
    path = base / relative
    common.require(path.resolve().is_relative_to(base.resolve()), "receipt symlink escapes package")
    return path


def check_files(base, records):
    for relative, value in records.items():
        path = member_path(base, relative)
        digest = value["sha256"] if isinstance(value, dict) else value
        common.require(path.is_file() and common.sha(path) == digest, "retained file differs: " + relative)


def stage_smoke_corpora(root, commit, stage):
    """Restore the committed qualification operands excluded by git archive.

    Match release-provenance.yml's bounded three-case restoration. These are
    external test inputs, not wheel members. Read the frozen Git blobs, never
    working-tree bytes or regenerated expectations, and retain their hashes.
    """
    from verify_linux_surface_qualification import CORPUS_CASES
    records = {}
    for case in CORPUS_CASES:
        prefix = "mamba/corpus/" + case
        listing = subprocess.check_output(
            ["git", "-C", str(root), "ls-tree", "-r", "-z", commit, "--", prefix])
        entries = [entry for entry in listing.split(b"\0") if entry]
        common.require(bool(entries), "missing frozen smoke corpus: " + prefix)
        names = []
        for entry in entries:
            header, raw_name = entry.split(b"\t", 1)
            mode, kind, blob = header.decode().split()
            name = raw_name.decode()
            common.require(kind == "blob" and mode in ("100644", "100755")
                           and name.startswith(prefix + "/"), "invalid frozen corpus member: " + name)
            target = member_path(stage, name)
            content = subprocess.check_output(["git", "-C", str(root), "cat-file", "blob", blob])
            common.require(not target.exists() or target.read_bytes() == content,
                           "staged smoke corpus differs from frozen blob: " + name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            records[name] = {"git_blob": blob, "sha256": hashlib.sha256(content).hexdigest(),
                             "bytes": len(content)}
            names.append(name)
        common.require(prefix + "/x.f32" in names, "missing frozen corpus input: " + prefix)
    return {"source_commit": commit, "files": records}


def native_indexes(root, records):
    manifest = json.loads((root / "bindings/development_groups.json").read_text())["groups"]
    indexes = {}
    expected = set()
    for mode in common.MODES:
        files = {}
        for members in manifest[mode].values():
            common.require(len(members) == 1, "per-binding input requires single-member groups")
            member = members[0]
            name = member + ".so"
            common.require(name not in files, "duplicate binding member")
            files[name] = [member]
            expected.add(mode + "/" + name)
        indexes[mode] = {"format": 1, "files": files}
    actual = {name for name in records if name.endswith(".so")}
    common.require(actual == expected, "retained native coverage differs from current binding manifest")
    return indexes


def validate_snapshot(root, commit, package, receipt):
    common.require(receipt["source_commit"] == receipt["payload_commit"], "installed source/payload commit differs")
    validate_refresh(root, receipt["source_commit"], commit)
    check_files(package, receipt["python_source_files"])
    check_files(package, receipt["native_and_runtime_files"])
    check_files(package.parent, receipt["verifier_payload"])
    common.require((package / "identity_columns/COMMIT").read_text().strip() == receipt["payload_commit"],
                   "installed payload commit witness differs")
    for name, digest in receipt["python_source_files"].items():
        content = subprocess.check_output(["git", "-C", str(root), "show", commit + ":python/mojolearn/" + name])
        common.require(common.hashlib.sha256(content).hexdigest() == digest, "snapshot Python differs from source: " + name)
    validate_native_inventory(package, receipt)
    return native_indexes(root, receipt["native_and_runtime_files"])


def validate_native_inventory(package, receipt):
    # Installed snapshots retain adapters separately from the 79 numerical
    # images. Keep that ownership split while checking every actual native
    # file; the raw adapter build is still independently validated below.
    adapters = receipt.get("adapter_files", {})
    common.require(all(name.startswith("_adapters/") for name in adapters),
                   "adapter receipt contains a non-adapter path")
    check_files(package, adapters)
    expected = set(receipt["native_and_runtime_files"]) | {
        name for name in adapters if Path(name).suffix in (".so", ".dylib")}
    actual = {str(p.relative_to(package)) for p in package.rglob("*")
              if p.suffix in (".so", ".dylib")}
    common.require(actual == expected, "snapshot has unrecorded native/runtime files")


def validate_adapter_toolchain(original, current):
    # A retained worktree may disappear after its outputs are archived. The
    # invocation path is location metadata: the executable bytes, version,
    # locked/installed packages, precompiled libraries and OS facts still
    # must match exactly. Never discard any other toolchain field.
    old, new = dict(original), dict(current)
    old.pop("compiler", None)
    new.pop("compiler", None)
    digest = old.get("compiler_sha256", "")
    common.require(original.get("available") is True and len(digest) == 64
                   and all(c in "0123456789abcdef" for c in digest)
                   and old == new, "adapter toolchain changed")


def validate_adapters(root, directory, before, result, commit):
    common.require(result["exit_code"] == 0 and result["source_commit"] == before["source_commit"],
                   "adapter build did not complete for this source")
    common.require(before.get("source_dirty") in (False, "", []), "adapter build used dirty source")
    common.require(before["native_mode"] == "identical", "adapter mode differs")
    # Adapter reuse is based on the complete captured closure below, not an
    # unrelated package commit: a Python owner-lifetime fix need not rebuild it.
    check_files(root, before["captured_inputs"])
    # Keep the original role closure and compiler identity, rather than accepting
    # the manifest's smaller list of direct source files as closure evidence.
    import development_build as dev
    validate_adapter_toolchain(before["toolchain"], dev.toolchain_facts(root))
    for name, digest in result["artifacts"].items():
        path = directory / Path(name).name
        common.require(path.is_file() and common.sha(path) == digest, "adapter artifact differs: " + name)
    api = common.module(root / "packaging/adapters/manifest.py", "installed_adapter_manifest")
    record = json.loads((directory / api.BUILD_RECORD).read_text())
    common.require(record["source_commit"] == before["source_commit"] and record["numeric_mode"] == "identical",
                   "adapter build manifest source/mode differs")
    common.require(record["sources"] == api.source_digests(root), "adapter direct inputs changed")
    return api



def stage_frozen_source(root, commit, archive, stage):
    """Retain every frozen source byte without duplicating non-build archives.

    Documentation, recorded results and experiment data remain in the complete
    compressed Git archive. Packaging consumes the other directories; the tiny
    required smoke corpora are restored independently from committed blobs.
    """
    excluded = {"bench", "docs", "docs-site", "experiments", "release"}
    with archive.open("wb") as stream:
        subprocess.run(["git", "-C", str(root), "archive", "--format=tar.gz", commit],
                       stdout=stream, check=True)
    omitted = []
    with tarfile.open(archive) as stream:
        members = []
        for member in stream.getmembers():
            if member.name.split("/", 1)[0] in excluded:
                omitted.append(member.name)
            else:
                members.append(member)
        stream.extractall(stage, members=members, filter="data")
    return {"complete_archive": common.reference(archive),
            "unexpanded_nonbuild_members": omitted,
            "excluded_top_level_directories": sorted(excluded)}


def assemble(args):
    root, package, output, raw = (Path(p).resolve() for p in
                                 (args.source_root, args.package, args.output, args.adapters))
    commit = common.git(root, "rev-parse", args.source_ref)
    validate_source_root(root, commit)
    common.require(not output.exists() and not output.is_relative_to(root)
                   and not output.is_relative_to(package), "output must be new and outside source/install")
    receipt = json.loads(Path(args.receipt).read_text())
    indexes = validate_snapshot(root, commit, package, receipt)
    before_ref, result_ref = common.reference(args.adapter_before), common.reference(args.adapter_result)
    before, result = (json.loads(Path(p).read_text()) for p in (args.adapter_before, args.adapter_result))
    common.require(result["before_build_sha256"] == before_ref["sha256"], "adapter before-build receipt differs")
    api = validate_adapters(root, raw, before, result, commit)
    output.mkdir(parents=True)
    document = dict(schema=SCHEMA, state="preparing", source_root=str(root), source_commit=commit,
                    input_receipt=common.reference(args.receipt), package=str(package),
                    adapter_before=before_ref, adapter_result=result_ref,
                    compiler_invoked=False, runtime_qualified=False, identity_qualified=False,
                    producer=common.reference(__file__), commands=[])
    runner = common.Runner(output, document)
    shutil.copy2(__file__, output / "controllers" / (common.sha(__file__) + ".py"))
    try:
        stage = output / "source"
        stage.mkdir()
        archive = output / "source.tar.gz"
        document["source_staging"] = stage_frozen_source(root, commit, archive, stage)
        document["source_archive"] = common.reference(archive)
        document["smoke_corpora"] = stage_smoke_corpora(root, commit, stage)
        pkg = stage / "python/mojolearn"
        common.require(not list(pkg.rglob("*.so")), "source contains unadmitted native files")
        rows = []
        for name, value in receipt["native_and_runtime_files"].items():
            dest = member_path(pkg, name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(package / name, dest)
            if name.endswith(".so"):
                rows.append(dict(mode=name.split("/")[0], staged=name,
                                 source=common.reference(package / name), copied_sha256=value["sha256"]))
        document["groups"], document["indexes"] = rows, {}
        for mode, index in indexes.items():
            path = pkg / mode / common.INDEX
            common.write(path, index)
            document["indexes"][mode] = dict(staged=common.reference(path), staged_document=index)
        document["counts"] = {mode: sum(row["mode"] == mode for row in rows) for mode in common.MODES}
        adapter = pkg / "_adapters"
        shutil.copytree(raw, adapter)
        api.finalize(adapter)
        api.check(adapter, vendor="metal", arch="apple")
        helper = package / ".dylibs/libMojolearnMath.dylib"
        python = str(Path(args.python).resolve())
        env = dict(os.environ, MOJOLEARN_RELEASE_MODES="identical", MOJOLEARN_RELEASE_ADAPTERS="1",
                   MOJOLEARN_PORTABLE_MATH_HELPER=str(helper), MOJOLEARN_STAGE_SKIP="")
        native = [pkg / row["staged"] for row in rows] + [adapter / "libmojolearn_adapter.so"]
        gpu = [pkg / row["staged"] for row in rows if row["mode"] == "identical"] + [native[-1]]
        runner.run("stage-dylibs", [python, stage / "packaging/macos/stage_dylibs.py", *native,
                   root / ".pixi/envs/default/lib"], env=env)
        document["dependency_inputs"] = {p.name: common.reference(root / ".pixi/envs/default/lib" / p.name)
                                          for p in (pkg / ".dylibs").iterdir() if p.is_file()}
        runner.run("stage-portable-math", [python, stage / "packaging/portable_math/stage.py", pkg / ".dylibs",
                   "--helper", helper, "--receipt", output / "portable-math-stage.json"], env=env)
        manifest = adapter / "adapters.json"
        manifest.write_bytes(api.refresh(manifest.read_bytes(), lambda name: (adapter / name).read_bytes(),
                                        "retained candidate: normal dependency staging and signing"))
        api.check(adapter, vendor="metal", arch="apple")
        document["staged_files"] = {str(p.relative_to(pkg)): common.sha(p) for p in pkg.rglob("*")
                                    if p.is_file() and p.suffix in (".so", ".dylib", ".mojoc")}
        runner.run("compatibility", [python, stage / "packaging/macos/compatibility.py", "--package", pkg], env=env)
        runner.run("isa-baseline", [python, stage / "packaging/isa_baseline.py", *native, adapter / "mojolearn_torch.so"], env=env)
        runner.run("metal-embedded", [python, stage / "packaging/macos/check_gpu_embedded.py", *gpu], env=env)
        # Frozen installed verification payload is copied exactly; no reference
        # hashes are regenerated and no unfinished release column is admitted.
        for name in receipt["verifier_payload"]:
            dest = member_path(stage / "python", name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(package.parent / name, dest)
        (pkg / "identity_columns/COMMIT").write_text(commit + "\n")
        for name in ("LICENSE", "NOTICE", "README.md"):
            shutil.copy2(stage / name, stage / "python" / name)
        shutil.copy2(stage / "CITATION.cff", pkg / "CITATION.cff")
        wheel_env = dict(env, MOJOLEARN_VERIFICATION_GIT_ROOT=str(root),
                         MOJOLEARN_VERIFICATION_SOURCE_COMMIT=commit)
        runner.run("wheel-build", [python, "-m", "build", "--wheel", "--no-isolation"], cwd=stage / "python", env=wheel_env)
        wheels = list((stage / "python/dist").glob("*.whl"))
        common.require(len(wheels) == 1, "expected exactly one wheel")
        wheel = wheels[0]
        runner.run("wheel-shrink", [python, stage / "packaging/macos/shrink_wheel.py", wheel], env=env)
        document["wheel_native"] = common.verify_wheel(wheel, pkg, document)
        runner.run("wheel-compatibility", [python, stage / "packaging/macos/compatibility.py", "--wheel", wheel], env=env)
        runner.run("wheel-api", [python, stage / "tools/wheel_api_audit.py", "--require-complete",
                   "--output", output / "wheel-api.json", wheel], env=env)
        check_files(package, receipt["native_and_runtime_files"])
        validate_adapters(root, raw, before, result, commit)
        document.update(state="packed", wheel=common.reference(wheel),
                        required_next_checks=["five-Python installed-wheel smoke", "installed Apple adapter smoke",
                                              "release reference admission and remaining release gates"])
        runner.save()
        return document
    except Exception as exc:
        document.update(state="failed", error=str(exc))
        runner.save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "source-ref", "package", "receipt", "adapters", "adapter-before", "adapter-result", "output", "python"):
        parser.add_argument("--" + name, required=True)
    result = assemble(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("state", "counts", "wheel", "runtime_qualified", "identity_qualified")}, indent=2))


if __name__ == "__main__":
    main()
