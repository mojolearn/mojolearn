#!/usr/bin/env python3
"""Assemble a Mac wheel from admitted development groups, without compilation.

Admission runs against the original source/cache paths. Only copies enter the
isolated staging tree: dependency rewriting, signing and stripping change those
copies. The receipt preserves original inputs and each packaging transformation;
it does not claim runtime or identity qualification.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile


SCHEMA = "mojolearn.macos.development-assembly.v1"
INDEX = "_mojolearn_merged.json"
MODES = ("host", "identical")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def write(path, document):
    Path(path).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def verified(ref):
    path = Path(ref["path"])
    require(path.is_file() and sha(path) == ref["sha256"], f"retained bytes changed: {path}")
    return path


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def revision(root):
    require(not git(root, "status", "--porcelain", "--untracked-files=no"), "source has tracked edits")
    return git(root, "rev-parse", "HEAD")


def flatten_index(index):
    """Change physical paths only, preserving every logical group and key."""
    require(index.get("format") == 1 and index.get("layout") == "development",
            "expected an admitted development index")
    require(set(index) == {"format", "layout", "files", "provenance"}, "unknown index schema")
    out = copy.deepcopy(index)
    # Preserve all producer metadata. Only physical file paths change;
    # original-root admission and the relocation live in the assembly proof.
    out["files"] = {}
    pairs, members, stems = [], set(), set()
    for source, names in index["files"].items():
        path = PurePosixPath(source)
        require(len(path.parts) == 3 and path.parts[0] == ".development"
                and re.fullmatch(r"[0-9a-f]{64}", path.parts[1])
                and re.fullmatch(r"_mojolearn_dev_[a-zA-Z0-9_]+\.so", path.name)
                and str(path) == source, f"invalid development path: {source}")
        stem = path.stem
        require(index["provenance"].get(stem) == path.parts[1], f"index key differs: {source}")
        require(isinstance(names, list) and names and all(isinstance(n, str) and
                re.fullmatch(r"_mojolearn(?:_[a-zA-Z0-9_]+)?", n) for n in names), "invalid members")
        require(len(set(names)) == len(names) and not members.intersection(names), "duplicate logical member")
        require(path.name not in out["files"], "flattening would overwrite a group")
        members.update(names)
        stems.add(stem)
        out["files"][path.name] = list(names)
        pairs.append((source, path.name, list(names), path.parts[1]))
    require(stems and stems == set(index["provenance"]), "index provenance coverage differs")
    return out, pairs


# This child imports the ORIGINAL checkout's tools. Loading binding_stamps
# from the assembler's checkout would incorrectly validate a different root.
ADMIT = r'''
import json, os, pathlib, sys
root, cache, output = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(root / 'tools'))
import binding_stamps as stamps
import development_build as dev
pkg = root / 'python/mojolearn'
active, ignored = stamps.development_bindings(pkg, root, cache)
if not active:
    raise ValueError('no admitted development groups')
indexes = {mode: json.loads((pkg/mode/'_mojolearn_merged.json').read_text())
           for mode in ('host', 'identical')}
keys = set(key for index in indexes.values() for key in index['provenance'].values())
receipts = {key: str(cache/'artifacts'/key/'receipt.json') for key in keys}
first = json.loads(pathlib.Path(next(iter(receipts.values()))).read_text())['fields']
env = {'PATH': os.environ.get('PATH', ''), **first['environment'], **first['link_environment']}
support = dev.support_rows(root, cache, 'darwin', dev.c_toolchain_facts(env, 'darwin'))
for row in support:
    if dev.cached_artifact(cache, row) is None:
        raise ValueError('portable-math support receipt does not admit current inputs')
    row['receipt'] = str(cache/'artifacts'/row['key']/'receipt.json')
    row['installed'] = str(pkg/row['install'])
output.write_text(json.dumps(dict(active=sorted(map(str, active)),
    indexes=indexes, receipts=receipts, support=support), indent=2)+'\n')
'''


class Runner:
    def __init__(self, output, document):
        self.output, self.document = Path(output), document

    def save(self):
        write(self.output / "assembly.json", self.document)

    def run(self, name, argv, cwd=None, env=None):
        log = self.output / (name + ".log")
        row = {"phase": name, "argv": list(map(str, argv)), "cwd": str(cwd) if cwd else None,
               "log": str(log), "exit": None}
        self.document.setdefault("commands", []).append(row)
        self.save()
        with log.open("wb") as stream:
            result = subprocess.run(row["argv"], cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        row.update(exit=result.returncode, log_sha256=sha(log))
        self.save()
        require(result.returncode == 0, f"{name} failed ({result.returncode}); see {log}")
        return log


def retain(ref, directory):
    path = verified(ref)
    destination = Path(directory) / path.name
    require(not destination.exists(), f"retained filename collision: {destination}")
    shutil.copy2(path, destination)
    require(sha(destination) == ref["sha256"], "retention copy differs")
    return reference(destination)


def prepare(root, cache, output, python):
    root, cache, output = (Path(p).resolve() for p in (root, cache, output))
    require(sys.platform == "darwin", "Mac assembly requires macOS")
    require(not output.exists() and root not in output.parents and output not in root.parents,
            "output must be a new directory outside the source tree")
    commit = revision(root)
    output.mkdir(parents=True)
    document = dict(schema=SCHEMA, state="preparing", source_root=str(root), source_commit=commit,
                    cache=str(cache), python=str(Path(python).resolve()), runtime_qualified=False,
                    compiler_invoked=False, producer=reference(__file__), commands=[])
    runner = Runner(output, document)
    try:
        runner.run("admission", [python, "-c", ADMIT, root, cache, output / "admission.json"])
        admission = json.loads((output / "admission.json").read_text())
        document["admission"] = reference(output / "admission.json")
        stage = output / "source"
        stage.mkdir()
        archive = output / "source.tar"
        with archive.open("wb") as stream:
            subprocess.run(["git", "-C", str(root), "archive", "--format=tar", commit], stdout=stream, check=True)
        document["source_archive"] = reference(archive)
        with tarfile.open(archive) as source:
            source.extractall(stage, filter="data")
        pkg = stage / "python/mojolearn"
        require(not list(pkg.rglob("*.so")), "source export contains unadmitted native binaries")
        rows, indexes = [], {}
        for mode in MODES:
            original = root / "python/mojolearn" / mode / INDEX
            index = json.loads(original.read_text())
            require(index == admission["indexes"][mode], "active index changed during admission")
            flat, pairs = flatten_index(index)
            destination = pkg / mode
            destination.mkdir(parents=True, exist_ok=True)
            for source, name, members, key in pairs:
                binary = original.parent / source
                require(str(binary) in admission["active"], "binary was not admitted")
                before = reference(binary)
                receipt = reference(admission["receipts"][key])
                data = json.loads(verified(receipt).read_text())
                require(before["sha256"] == data["sha256"], "admitted binary changed before copy")
                shutil.copy2(binary, destination / name)
                require(sha(destination / name) == before["sha256"], "group copy differs")
                retained = output / "receipts" / key
                retained.mkdir(parents=True)
                rows.append(dict(mode=mode, source=before, staged=f"{mode}/{name}", members=members,
                                 key=key, receipt=retain(receipt, retained), copied_sha256=before["sha256"]))
            write(destination / INDEX, flat)
            indexes[mode] = dict(original=reference(original), source_document=index,
                                 staged=reference(destination / INDEX), staged_document=flat)
        require(len(rows) == len(admission["active"]), "active group coverage differs")
        support = admission["support"]
        require(len(support) == 1 and support[0]["name"] == "portable_math", "unknown runtime support set")
        helper = reference(support[0]["installed"])
        receipt = reference(support[0]["receipt"])
        require(json.loads(verified(receipt).read_text())["sha256"] == helper["sha256"], "support bytes changed")
        retained = output / "support"
        retained.mkdir()
        document.update(groups=rows, indexes=indexes, portable_math=dict(
            source=helper, retained=retain(helper, retained), receipt=retain(receipt, retained)))
        document["counts"] = {mode: sum(row["mode"] == mode for row in rows) for mode in MODES}
        require(revision(root) == commit, "source changed during staging")
        for row in rows:
            verified(row["source"])
        for index in indexes.values():
            verified(index["original"])
        document["state"] = "prepared"
        runner.save()
        return document
    except Exception as exc:
        document.update(state="failed", error=str(exc))
        runner.save()
        raise


def adapter_role_inputs(root, role):
    """The precise frontend source closure, plus its recorded build recipe."""
    import bincache
    root = Path(root).resolve()
    if role == "torch":
        todo = [root / "adapters/torch/registration.cpp"]
    elif role == "max":
        todo = list((root / "adapters/max_ops").rglob("*.mojo"))
    else:
        todo = [root / "adapters/native/c_api.mojo"]
    seen = set()
    while todo:
        path = todo.pop().resolve()
        if path in seen:
            continue
        require(path.is_relative_to(root) and path.is_file(), "missing/escaping adapter input")
        seen.add(path)
        if path.suffix == ".mojo":
            for name, names in bincache.parse_imports(path.read_text()):
                resolved = bincache.resolve(name, names, path, [root, root / "bindings"])
                require(resolved is not None, "unresolved adapter import: " + name)
                todo.extend(resolved)
        else:
            for name in re.findall(r'^\s*#\s*include\s*"([^"\n]+)"', path.read_text(), re.M):
                found = next((p for p in (path.parent / name, root / "adapters/native" / name) if p.is_file()), None)
                require(found is not None, "unresolved adapter C++ include")
                todo.append(found)
    seen.update(root / name for name in ("packaging/adapters/build.sh", "packaging/adapters/manifest.py",
                                        "packaging/adapters/pins.json", "packaging/adapters/fetch_torch.py", "pixi.lock"))
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(seen)}


def compose_adapters(root, native_result, published_wheel, output):
    """Combine a captured current native build with unchanged published frontends."""
    import retained_release_composition as retained
    import development_build as dev
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists() and root not in output.parents, "adapter output must be new and outside source")
    api = module(root / "packaging/adapters/manifest.py", "assembly_adapter_manifest")
    result_ref, wheel_ref = reference(native_result), reference(published_wheel)
    result = json.loads(verified(result_ref).read_text())
    require(result.get("schema") == "mojolearn.macos.native-adapter-build.v1" and result.get("complete") is True
            and result.get("state") == "built" and result.get("build_exit") == 0, "native adapter build is incomplete")
    before_ref = result["before_build"]
    before = json.loads(verified(before_ref).read_text())
    require(before["adapter_inputs"] == retained.adapter_inputs(root), "native adapter source closure changed")
    require(before["toolchain"] == dev.toolchain_facts(root), "native compiler/dependency inputs changed")
    for name, value in before["captured_inputs"].items():
        require(sha(root / name) == value, "captured native input changed: " + name)
    require(any(row["phase"] == "native-build" and row["argv"] == before["command"] for row in result["commands"]),
            "native build command differs from captured recipe")
    for row in result["commands"]:
        require(row["exit_code"] == 0 and sha(row["log"]) == row["log_sha256"], "native phase/log proof differs")
    native = verified(result["artifact"])
    with zipfile.ZipFile(verified(wheel_ref)) as wheel:
        prefix = "mojolearn/_adapters/"
        original = json.loads(wheel.read(prefix + api.MANIFEST))
        files = {name: wheel.read(prefix + name) for name in api.FILES.values()}
        api.validate(original, {name: (hashlib.sha256(data).hexdigest(), len(data)) for name, data in files.items()},
                     vendor="metal", arch="apple")
    origin = original["source_commit"]
    inputs = {}
    for role in ("torch", "max"):
        inputs[role] = adapter_role_inputs(root, role)
        for name, value in inputs[role].items():
            old = subprocess.check_output(["git", "-C", str(root), "show", origin + ":" + name])
            require(hashlib.sha256(old).hexdigest() == value, f"published {role} input changed: {name}")
    require(original["sources"] == api.source_digests(root), "published adapter local sources changed")
    facts = before["toolchain"]
    require(original["mojo"]["version"] == facts["version"], "MAX compiler version changed")
    require(original["mojo"]["conda"] == api.conda_version(root / ".pixi/envs/default", "mojo")
            and original["max"]["version"] == api.conda_version(root / ".pixi/envs/default", "max"),
            "MAX/Mojo package dependency changed")
    pins = api.pins(root / "packaging/adapters/pins.json")
    require(original["torch"]["series"] == pins["torch"]["series"]
            and original["torch_version_pin"] == pins["torch"]["build"]["metal"]["version"]
            and original["max"]["version"] == pins["max"]["version"], "frontend dependency pins changed")
    output.mkdir(parents=True)
    for name, data in files.items():
        if name != api.FILES["native"]:
            (output / name).write_bytes(data)
    shutil.copy2(native, output / api.FILES["native"])
    origins = {role: {"action": "retained-published-bytes", "source_commit": origin,
                     "wheel": wheel_ref, "source_inputs": inputs[role],
                     "sha256": original["files"][role]["sha256"]} for role in ("torch", "max")}
    origins["native"] = dict(action="current-native-build", source_commit=result["source_commit"],
                             result=result_ref, before_build=before_ref, sha256=sha(native))
    doc = copy.deepcopy(original)
    doc.update(source_commit=revision(root), source_commit_scope="admitted package source; compilation origins are per artifact",
               artifact_origins=origins, compilation_action="mixed-current-native-and-retained-frontends")
    doc["commands"]["native"] = before["command"]
    doc["files"] = api.file_rows(output)
    doc["unstaged"]["native"] = doc["files"]["native"]
    doc.setdefault("post_build", []).append("Current captured native library; unchanged Torch/MAX bytes retained from the published wheel")
    write(output / api.MANIFEST, doc)
    api.check(output, vendor="metal", arch="apple")
    proof = dict(schema="mojolearn.macos.adapter-composition.v1", complete=True,
                 source_root=str(root), admitted_commit=revision(root), manifest=reference(output / api.MANIFEST),
                 origins=origins, published_manifest=original,
                 limitations=["The published manifest did not record its original C++ compiler version or direct Mojo executable hash; these historical fields remain unrecorded.",
                              "Retained frontend admission uses the immutable published bytes, exact source/recipe/pixi.lock inputs, recorded Mojo/MAX versions and Torch wheel identity.",
                              "Installed Torch/MAX adapter smokes remain required; this is not runtime qualification."])
    proof_path = output.with_name(output.name + ".assembly.json")
    write(proof_path, proof)
    return dict(state="adapters-composed", counts={"built": 1, "retained": 2}, proof=str(proof_path), runtime_qualified=False)


def adapter_copy(root, directory, proof_path, destination, document):
    api = module(root / "packaging/adapters/manifest.py", "assembly_adapter_manifest")
    directory = Path(directory).resolve()
    doc = api.check(directory, vendor="metal", arch="apple")
    proof = json.loads(Path(proof_path).read_text())
    require(proof.get("schema") == "mojolearn.macos.adapter-composition.v1" and proof.get("complete") is True,
            "a per-role adapter composition proof is required")
    require(verified(proof["manifest"]) == directory / api.MANIFEST, "adapter proof names another manifest")
    require(doc.get("artifact_origins") == proof.get("origins"), "per-role adapter origins differ")
    require(all(doc["files"][role]["sha256"] == origin["sha256"] for role, origin in proof["origins"].items()),
            "adapter bytes differ from their individual origins")
    # Replay every source dependency at the actual packaging boundary.
    for role in ("torch", "max"):
        require(proof["origins"][role]["source_inputs"] == adapter_role_inputs(root, role),
                "retained frontend inputs changed: " + role)
    import retained_release_composition as retained
    import development_build as dev
    native_before = json.loads(verified(proof["origins"]["native"]["before_build"]).read_text())
    require(native_before["adapter_inputs"] == retained.adapter_inputs(root), "native adapter transitive source changed")
    require(native_before["toolchain"] == dev.toolchain_facts(root), "native compiler/dependency inputs changed")
    require(doc.get("numeric_mode") == "identical", "adapters are not identical mode")
    require(doc.get("sources") == api.source_digests(root), "adapter source inputs changed; a root-owned build is required")
    require(re.fullmatch(r"[0-9a-f]{40}", doc.get("source_commit", "")), "adapter source commit missing")
    require(doc.get("max", {}).get("version") == api.pins(root / "packaging/adapters/pins.json")["max"]["version"],
            "adapter MAX pin changed")
    require(doc.get("mojo", {}).get("conda") == api.conda_version(root / ".pixi/envs/default", "mojo"),
            "adapter Mojo compiler version changed")
    require(doc.get("max", {}).get("version") == api.conda_version(root / ".pixi/envs/default", "max"),
            "adapter MAX dependency changed")
    require(not destination.exists(), "staging already contains adapters")
    shutil.copytree(directory, destination, symlinks=False)
    document["adapters"] = dict(source=reference(directory / api.MANIFEST), proof=reference(proof_path),
        files={name: reference(directory / name) for name in api.FILES.values()})
    api.check(destination, vendor="metal", arch="apple")
    return api


def payload(stage, commit, runner, python, env):
    pkg = stage / "python/mojolearn"
    for name in ("LICENSE", "NOTICE", "README.md"):
        shutil.copy2(stage / name, stage / "python" / name)
    shutil.copy2(stage / "CITATION.cff", pkg / "CITATION.cff")
    for name in ("identity_trace_diff.py", "identity_break.py"):
        shutil.copy2(stage / "tools" / name, pkg / ("_" + name))
    for path in (stage / "tools/identity_lanes").glob("*.py"):
        shutil.copy2(path, pkg / ("_identity_lane_" + path.name))
    runner.run("verification-payload", [python, stage / "tools/verification_ctr_payload.py",
               "--output", pkg / "verify_reference/ctr_models"], cwd=stage, env=env)
    record = runner.run("identity-record", [python, pkg / "host_surface.py", "--training-gpu-column-record"], env=env).read_text().strip()
    require(record and len(PurePosixPath(record).parts) == 1 and record not in (".", ".."), "invalid identity record")
    columns = runner.run("identity-columns", [python, pkg / "host_surface.py", "--training-gpu-columns"], env=env).read_text().split()
    target = pkg / "identity_columns" / record
    target.mkdir(parents=True)
    for rel in columns:
        path = stage / rel
        require(path.resolve().is_relative_to(stage) and path.is_file(), "invalid identity column path")
        shutil.copy2(path, target / path.name)
    (target.parent / "COMMIT").write_text(commit + "\n")


def verify_wheel(wheel, pkg, document):
    """Prove flat-index membership and record final native bytes after stripping."""
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        require(len(set(names)) == len(names), "duplicate wheel members")
        expected = {"mojolearn/" + row["staged"] for row in document["groups"]}
        expected |= {"mojolearn/_adapters/libmojolearn_adapter.so", "mojolearn/_adapters/mojolearn_torch.so"}
        actual = {name for name in names if name.endswith(".so")}
        require(actual == expected, "wheel has missing or unexpected extensions")
        require({"mojolearn/_adapters/adapters.json", "mojolearn/_adapters/mojolearn_ops.mojoc"} <= set(names),
                "wheel is missing adapter metadata or MAX package")
        for mode, row in document["indexes"].items():
            member = f"mojolearn/{mode}/{INDEX}"
            require(archive.read(member) == (pkg / mode / INDEX).read_bytes(), "wheel index differs from checked staging index")
            require(json.loads(archive.read(member)) == row["staged_document"], "staged index was modified")
            for name in row["staged_document"]["files"]:
                require(f"mojolearn/{mode}/{name}" in actual, "wheel index names missing binary")
        require(not any("/.development/" in name for name in names), "wheel contains development cache paths")
        return {name: hashlib.sha256(archive.read(name)).hexdigest() for name in sorted(names)
                if name.endswith((".so", ".dylib", ".mojoc"))}


def pack(output, adapters, adapters_proof):
    output = Path(output).resolve()
    document = json.loads((output / "assembly.json").read_text())
    require(document.get("schema") == SCHEMA and document.get("state") == "prepared", "assembly is not prepared")
    runner = Runner(output, document)
    root, stage = Path(document["source_root"]), output / "source"
    pkg, python = stage / "python/mojolearn", document["python"]
    try:
        require(revision(root) == document["source_commit"], "source changed since preparation; prepare a new assembly")
        for row in document["groups"]:
            require(sha(pkg / row["staged"]) == row["copied_sha256"], "prepared group was modified")
        for row in document["indexes"].values():
            verified(row["staged"])
        helper = verified(document["portable_math"]["retained"])
        api = adapter_copy(root, adapters, adapters_proof, pkg / "_adapters", document)
        native = [pkg / row["staged"] for row in document["groups"]]
        adapter_native = pkg / "_adapters/libmojolearn_adapter.so"
        torch = pkg / "_adapters/mojolearn_torch.so"
        all_native = native + [adapter_native]
        gpu = [pkg / row["staged"] for row in document["groups"] if row["mode"] == "identical"] + [adapter_native]
        env = dict(os.environ, MOJOLEARN_RELEASE_MODES="identical", MOJOLEARN_RELEASE_ADAPTERS="1",
                   MOJOLEARN_PORTABLE_MATH_HELPER=str(helper), MOJOLEARN_STAGE_SKIP="")
        document["state"] = "packing"
        runner.run("stage-dylibs", [python, stage / "packaging/macos/stage_dylibs.py", *all_native,
                   root / ".pixi/envs/default/lib"], cwd=stage, env=env)
        runner.run("stage-portable-math", [python, stage / "packaging/portable_math/stage.py", pkg / ".dylibs",
                   "--helper", helper, "--receipt", output / "portable-math-stage.json"], cwd=stage, env=env)
        manifest = pkg / "_adapters/adapters.json"
        manifest.write_bytes(api.refresh(manifest.read_bytes(), lambda n: (manifest.parent / n).read_bytes(),
                             "release_macos_assembly.py: dependency staging and ad-hoc signing"))
        api.check(manifest.parent, vendor="metal", arch="apple")
        document["staged_native"] = {str(p.relative_to(pkg)): sha(p) for p in all_native + [torch]}
        runner.run("compatibility", [python, stage / "packaging/macos/compatibility.py", "--package", pkg], env=env)
        runner.run("isa-baseline", [python, stage / "packaging/isa_baseline.py", *all_native, torch], env=env)
        runner.run("metal-embedded", [python, stage / "packaging/macos/check_gpu_embedded.py", *gpu], env=env)
        payload(stage, document["source_commit"], runner, python, env)
        runner.run("wheel-build", [python, "-m", "build", "--wheel", "--no-isolation"], cwd=stage / "python", env=env)
        wheels = list((stage / "python/dist").glob("*.whl"))
        require(len(wheels) == 1, "expected exactly one assembled wheel")
        wheel = wheels[0]
        runner.run("wheel-shrink", [python, stage / "packaging/macos/shrink_wheel.py", wheel], env=env)
        document["wheel_native"] = verify_wheel(wheel, pkg, document)
        runner.run("wheel-compatibility", [python, stage / "packaging/macos/compatibility.py", "--wheel", wheel], env=env)
        runner.run("wheel-api", [python, stage / "tools/wheel_api_audit.py", "--require-complete",
                   "--output", output / "wheel-api.json", wheel], env=env)
        # Original inputs must survive signing/stripping unchanged.
        for row in document["groups"]:
            verified(row["source"])
        verified(document["portable_math"]["source"])
        for row in document["indexes"].values():
            verified(row["original"])
        for ref in document["adapters"]["files"].values():
            verified(ref)
        document.update(state="packed", wheel=reference(wheel), runtime_qualified=False,
                        required_next_checks=["installed loader/dependency/API smoke", "installed Apple Torch/MAX adapter smoke"])
        runner.save()
        return document
    except Exception as exc:
        document.update(state="failed", error=str(exc))
        runner.save()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source-root", type=Path, required=True)
    prep.add_argument("--cache", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--python", type=Path, required=True, help="packaging Python with the existing audit dependencies")
    wheel = commands.add_parser("pack")
    wheel.add_argument("--output", type=Path, required=True)
    wheel.add_argument("--adapters", type=Path, required=True)
    wheel.add_argument("--adapters-proof", type=Path, required=True)
    compose = commands.add_parser("compose-adapters")
    compose.add_argument("--source-root", type=Path, required=True)
    compose.add_argument("--native-result", type=Path, required=True)
    compose.add_argument("--published-wheel", type=Path, required=True)
    compose.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.source_root, args.cache, args.output, args.python)
        elif args.command == "compose-adapters":
            result = compose_adapters(args.source_root, args.native_result, args.published_wheel, args.output)
        else:
            result = pack(args.output, args.adapters, args.adapters_proof)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Mac assembly refused: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"state": result["state"], "groups": result["counts"], "runtime_qualified": False}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
