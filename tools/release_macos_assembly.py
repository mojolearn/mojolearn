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
STAGING_TOOLS = ("packaging/macos/stage_dylibs.py", "packaging/macos/compatibility.py",
                 "packaging/macos/check_gpu_embedded.py", "packaging/isa_baseline.py",
                 "packaging/portable_math/stage.py", "packaging/portable_math/portable_math.c",
                 "packaging/portable_math/powers_of_ten.h", "packaging/adapters/manifest.py")


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


FROZEN_ADMIT = r'''
import hashlib, json, os, pathlib, shutil, subprocess, sys
snapshot, original, cache, assembly, admission, output = (pathlib.Path(p).resolve() for p in sys.argv[1:])
sys.path.insert(0, str(snapshot/'tools'))
import development_build as dev
import bincache
document = json.loads(assembly.read_text())
current = json.loads(admission.read_text())
generated = {}
prepared_commit = document.get('prepared_source_commit', document['source_commit'])
for row in document['groups']:
    receipt = json.loads(pathlib.Path(row['receipt']['path']).read_text())
    for name, value in receipt['fields']['source']['files'].items():
        target = snapshot/name
        if target.is_file():
            continue
        if not target.resolve().is_relative_to(snapshot):
            raise ValueError('escaping captured source path')
        tracked = subprocess.check_output(['git', '-C', str(original), 'ls-tree', '-r', '--name-only', prepared_commit, '--', name], text=True).strip()
        if tracked:
            raise ValueError('frozen source deleted a tracked compiler input: '+name)
        source = original/name
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != value:
            raise ValueError('retained generated input is missing or changed: '+name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        generated[name] = dict(source=str(source), sha256=value,
            basis='untracked compiler input retained byte-for-byte from the admitted receipt')
support = current['support'][0]
support_receipt = json.loads(pathlib.Path(support['receipt']).read_text())
rows = []
for mode in ('host', 'identical'):
    originals = [row for row in document['groups'] if row['mode'] == mode]
    by_stem = {pathlib.Path(row['staged']).stem: row for row in originals}
    first = json.loads(pathlib.Path(originals[0]['receipt']['path']).read_text())['fields']
    if bincache.toolchain(snapshot) != first['toolchain']['locked']:
        raise ValueError('frozen compiler/dependency lock changed')
    configuration = hashlib.sha256(bincache.pixi_toml_build_part((snapshot/'pixi.toml').read_text()).encode()).hexdigest()
    if configuration != first['toolchain']['pixi_configuration']:
        raise ValueError('frozen compiler/dependency configuration changed')
    env = {'PATH': os.environ.get('PATH', ''), **first['environment'], **first['link_environment']}
    plan = dev.make_plan(snapshot, cache, [mode], env, facts=first['toolchain'],
                         c_facts=support_receipt['fields']['c_toolchain'], plat='darwin')
    if {r['stem'] for r in plan['groups']} != set(by_stem):
        raise ValueError('frozen group coverage differs')
    for row in plan['groups']:
        old = json.loads(pathlib.Path(by_stem[row['stem']]['receipt']['path']).read_text())
        fields = row['fields']
        # Relocation is not compilation: the snapshot has no toolchain and
        # is never used to build. Preserve the original absolute build root,
        # compiler/cache facts and generator interpreter identity explicitly.
        fields['repo'] = str(original)
        fields['environment']['MOJOLEARN_MERGED_PYTHON'] = old['fields']['environment']['MOJOLEARN_MERGED_PYTHON']
        if fields != old['fields'] or dev.digest(fields) != old['key']:
            raise ValueError(dev.input_change_reason(mode+':'+row['stem'], old['fields'], fields))
        rows.append({'mode': mode, 'stem': row['stem'], 'original_key': old['key'], 'inputs_equal': True})
    if len(plan['support']) != 1:
        raise ValueError('unexpected frozen support coverage')
    fields = plan['support'][0]['fields']
    fields['repo'] = str(original)
    if fields != support_receipt['fields'] or dev.digest(fields) != support_receipt['key']:
        raise ValueError('frozen portable-math compiler/source inputs changed')
# Frozen package metadata may differ; native adapters and retained frontend
# source closures may not differ from their individually admitted origins.
import retained_release_composition as retained
proof = json.loads(pathlib.Path(document['adapters']['proof']['path']).read_text())
native = json.loads(pathlib.Path(proof['origins']['native']['before_build']['path']).read_text())
if retained.adapter_inputs(snapshot) != native['adapter_inputs']:
    raise ValueError('frozen native adapter source closure changed')
for role in ('torch', 'max'):
    for name, value in proof['origins'][role]['source_inputs'].items():
        import hashlib
        if hashlib.sha256((snapshot/name).read_bytes()).hexdigest() != value:
            raise ValueError('frozen '+role+' input changed: '+name)
output.write_text(json.dumps(dict(schema='mojolearn.macos.frozen-input-equality.v1',
    complete=True, groups=rows, support_inputs_equal=True, adapter_inputs_equal=True,
    retained_generated_inputs=generated,
    normalizations=['snapshot directory -> original build root',
                    'verification Python -> recorded root-generator interpreter']), indent=2)+'\n')
'''


class Runner:
    def __init__(self, output, document):
        self.output, self.document = Path(output), document
        origin = reference(__file__)
        directory = self.output / "controllers"
        directory.mkdir(exist_ok=True)
        retained = directory / (origin["sha256"] + ".py")
        if not retained.exists():
            shutil.copy2(__file__, retained)
        controllers = self.document.setdefault("controllers", {})
        controllers[origin["sha256"]] = dict(source=origin, retained=reference(retained))

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


def native_source_refresh(original, current, native_inputs, frontend_inputs):
    """Permit a captured native repair without attributing it to retained frontends."""
    changed = {name for name in set(original) | set(current) if original.get(name) != current.get(name)}
    retained = set().union(*(set(values) for values in frontend_inputs.values()))
    require(changed <= set(native_inputs) - retained,
            "published adapter non-native sources changed: " + ", ".join(sorted(changed - (set(native_inputs) - retained))))
    return current


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
    sources = native_source_refresh(original["sources"], api.source_digests(root), before["adapter_inputs"], inputs)
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
               artifact_origins=origins, compilation_action="mixed-current-native-and-retained-frontends", sources=sources)
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


def adapter_copy(root, directory, proof_path, destination, document, copy_files=True):
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
    if not copy_files:
        return api
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


def refresh_source(output, document, runner, adapters, adapters_proof, source_ref=None):
    """Fresh frozen Python/package sources beside unchanged, admitted native bytes."""
    root, stage = Path(document["source_root"]), output / "source"
    commit = git(root, "rev-parse", source_ref or "HEAD")
    # Reconstruct the current original-root compiler/source/flags/dependency
    # admission. A commit/version name alone neither validates nor invalidates it.
    runner.run("frozen-admission", [document["python"], "-c", ADMIT, root, document["cache"],
                                   output / "frozen-admission.json"])
    current = json.loads((output / "frozen-admission.json").read_text())
    require(set(current["active"]) == {row["source"]["path"] for row in document["groups"]},
            "frozen source selected different native groups")
    require(current["indexes"] == {mode: row["source_document"] for mode, row in document["indexes"].items()},
            "frozen source group membership differs")
    for row in document["groups"]:
        verified(row["source"])
    for name, value in document["staging_tools"].items():
        require(sha(root / name) == value, "staging/check implementation changed; its check must be rerun: " + name)
    for ref in document["dependency_inputs"].values():
        verified(ref)
    verified(document["portable_math"]["source"])
    adapter_copy(root, adapters, adapters_proof, None, document, copy_files=False)
    pkg = stage / "python/mojolearn"
    for name, value in document["staged_files"].items():
        require(sha(pkg / name) == value, "staged bytes changed: " + name)
    fresh = output / "frozen-source"
    require(not fresh.exists(), "frozen-source staging already exists")
    fresh.mkdir()
    archive = output / "frozen-source.tar"
    with archive.open("wb") as stream:
        subprocess.run(["git", "-C", str(root), "archive", "--format=tar", commit], stdout=stream, check=True)
    with tarfile.open(archive) as source:
        source.extractall(fresh, filter="data")
    for name, value in document["staging_tools"].items():
        require(sha(fresh / name) == value, "frozen staging/check code changed: " + name)
    runner.run("frozen-input-equality", [document["python"], "-c", FROZEN_ADMIT, fresh, root,
        document["cache"], output / "assembly.json", output / "frozen-admission.json", output / "frozen-input-equality.json"])
    equality = json.loads((output / "frozen-input-equality.json").read_text())
    generated_archive = output / "frozen-generated-inputs.tar"
    with tarfile.open(generated_archive, "w") as retained:
        for name in equality.get("retained_generated_inputs", {}):
            retained.add(fresh / name, arcname=name)
    fresh_pkg = fresh / "python/mojolearn"
    require(not list(fresh_pkg.rglob("*.so")), "frozen source contains unadmitted extensions")
    # Copy just admitted staged artifacts/indexes, never the old package's
    # Python files. Deleted or renamed modules cannot survive the refresh.
    for name in document["staged_files"]:
        destination = fresh_pkg / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pkg / name, destination)
        require(sha(destination) == document["staged_files"][name], "staged artifact refresh copy differs")
    stage.rename(output / "prepared-source")
    fresh.rename(stage)
    document.update(prepared_source_commit=document["source_commit"], source_commit=commit,
                    frozen_source_archive=reference(archive), frozen_admission=reference(output / "frozen-admission.json"),
                    frozen_input_equality=reference(output / "frozen-input-equality.json"),
                    frozen_generated_inputs=reference(generated_archive),
                    reused_staging_checks=True)
    runner.save()


def pack(output, adapters, adapters_proof, stage_only=False, source_ref=None):
    output = Path(output).resolve()
    document = json.loads((output / "assembly.json").read_text())
    require(document.get("schema") == SCHEMA and document.get("state") in ("prepared", "staged"), "assembly is not prepared/staged")
    require(not stage_only or document["state"] == "prepared", "assembly has already been staged")
    runner = Runner(output, document)
    root, stage = Path(document["source_root"]), output / "source"
    pkg, python = stage / "python/mojolearn", document["python"]
    try:
        helper = verified(document["portable_math"]["retained"])
        env = dict(os.environ, MOJOLEARN_RELEASE_MODES="identical", MOJOLEARN_RELEASE_ADAPTERS="1",
                   MOJOLEARN_PORTABLE_MATH_HELPER=str(helper), MOJOLEARN_STAGE_SKIP="")
        if document["state"] == "staged":
            refresh_source(output, document, runner, adapters, adapters_proof, source_ref)
        else:
            for row in document["groups"]:
                require(sha(pkg / row["staged"]) == row["copied_sha256"], "prepared group was modified")
            for row in document["indexes"].values():
                verified(row["staged"])
            api = adapter_copy(root, adapters, adapters_proof, pkg / "_adapters", document)
            native = [pkg / row["staged"] for row in document["groups"]]
            adapter_native = pkg / "_adapters/libmojolearn_adapter.so"
            torch = pkg / "_adapters/mojolearn_torch.so"
            all_native = native + [adapter_native]
            gpu = [pkg / row["staged"] for row in document["groups"] if row["mode"] == "identical"] + [adapter_native]
            document["state"] = "staging"
            document["staging_tools"] = {name: sha(stage / name) for name in STAGING_TOOLS}
            runner.run("stage-dylibs", [python, stage / "packaging/macos/stage_dylibs.py", *all_native,
                       root / ".pixi/envs/default/lib"], cwd=stage, env=env)
            document["dependency_inputs"] = {p.name: reference(root / ".pixi/envs/default/lib" / p.name)
                                             for p in (pkg / ".dylibs").iterdir() if p.is_file()}
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
            retained = all_native + [torch, manifest, pkg / "_adapters/mojolearn_ops.mojoc"]
            retained += list((pkg / ".dylibs").iterdir()) + [pkg / mode / INDEX for mode in MODES]
            document["staged_files"] = {str(p.relative_to(pkg)): sha(p) for p in retained if p.is_file()}
            for row in document["groups"]:
                verified(row["source"])
            verified(document["portable_math"]["source"])
            for ref in document["adapters"]["files"].values():
                verified(ref)
            document["state"] = "staged"
            runner.save()
            if stage_only:
                return document
            if source_ref and git(root, "rev-parse", source_ref) != document["source_commit"]:
                refresh_source(output, document, runner, adapters, adapters_proof, source_ref)
        document["state"] = "packing"
        payload(stage, document["source_commit"], runner, python, env)
        wheel_env = dict(env, MOJOLEARN_VERIFICATION_GIT_ROOT=str(root),
                         MOJOLEARN_VERIFICATION_SOURCE_COMMIT=document["source_commit"])
        runner.run("wheel-build", [python, "-m", "build", "--wheel", "--no-isolation"], cwd=stage / "python", env=wheel_env)
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
    wheel.add_argument("--source-ref", help="frozen package commit; original build checkout remains unchanged")
    stage = commands.add_parser("stage")
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--adapters", type=Path, required=True)
    stage.add_argument("--adapters-proof", type=Path, required=True)
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
            result = pack(args.output, args.adapters, args.adapters_proof, stage_only=args.command == "stage",
                          source_ref=getattr(args, "source_ref", None))
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Mac assembly refused: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"state": result["state"], "groups": result["counts"], "runtime_qualified": False}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
