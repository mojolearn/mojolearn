"""Typed installed-slot evidence; classification only, never runtime qualification.

The collector is a test controller. This reader imports no product modules and
accepts only the registered public-attribute owner on a receipted development
activation. Wheel/release layouts need their own producer; they are not guessed.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

SCHEMA = "mojolearn.installed-attribute-slots.v2"
OBSERVATIONS = "mojolearn.installed-attribute-observations.v2"
OWNER = "mojolearn._native_public_attributes"
ENTRY = "public_attribute_native_call"
FACTORY = "classical_attribute_descriptor"
MEMBER = "_mojolearn_tokenizer_host"
POLICY = ("tools/installed_attribute_slots.py", "tools/collect_installed_attribute_slots.py",
          "tools/public_runtime_inventory.py", "tools/development_build.py",
          "tools/bincache.py", "tools/merged_bindings.py")
CHECKS = ("own-slot", "registered-context", "native-bound-entry", "native-factory-type",
          "registered-plan", "captured-mro", "captured-bindings", "captured-properties",
          "captured-availability", "no-getattr")


class WitnessError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise WitnessError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def source_snapshot(root):
    """Pin all package Python bytes and the admission/producer implementation."""
    root = Path(root).resolve()
    tracked = set(git(root, "ls-files", "python/mojolearn", *POLICY).splitlines())
    paths = {p for p in tracked if p.endswith(".py")} | set(POLICY)
    require(set(POLICY) <= tracked, "installed-slot producer policy is not committed")
    require(not git(root, "diff", "HEAD", "--name-only", "--", *sorted(paths)),
            "installed-slot source has uncommitted byte changes")
    # An untracked importable module can shadow the committed package too.
    actual = {str(p.relative_to(root)) for p in (root / "python/mojolearn").rglob("*.py")
              if "__pycache__" not in p.parts}
    require(actual <= tracked, "untracked package Python source cannot be witnessed")
    files = {p: sha256(root / p) for p in sorted(paths)}
    return {"revision": git(root, "rev-parse", "HEAD"), "tree": git(root, "rev-parse", "HEAD^{tree}"),
            "files": files, "files_sha256": digest(files)}


def validate_source(source, root):
    """Reuse an observation only while every package and policy byte is equal.

    Revision/tree identify where the observation was collected. An unrelated
    tooling or documentation commit does not change its import-time connections.
    Keep those original labels in the evidence, while independently checking the
    complete current source membership and byte map (including this policy).
    """
    current = source_snapshot(root)
    require(type(source) is dict and set(source) == set(current)
            and all(type(source.get(key)) is str and source[key] for key in ("revision", "tree")),
            "installed-slot source snapshot is malformed")
    require(source["files"] == current["files"] and source["files_sha256"] == current["files_sha256"],
            "installed-slot source bytes/policy are stale")
    return current


def reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def read_reference(value, label, *, json_data=True):
    require(type(value) is dict and set(value) == {"path", "sha256"}, label + " reference is malformed")
    path = Path(value["path"])
    require(path.is_absolute() and path.is_file(), label + " evidence is missing; restore retained bytes")
    require(sha256(path) == value["sha256"], label + " evidence digest mismatch")
    return json.loads(path.read_text()) if json_data else path


def owner_members(root):
    """The fixed import-time aliases decide each native entry's logical owner.

    Read the frozen registry without importing product code. Unsupported dynamic
    registry construction refuses instead of guessing where a pointer belongs.
    """
    tree = ast.parse((Path(root) / "python/mojolearn/host_surface.py").read_text())
    values = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "HOST_IMPORT_ALIASES"
                      for target in node.targets)]
    require(len(values) == 1, "native alias registry is missing or ambiguous")
    aliases = ast.literal_eval(values[0])
    members = dict.fromkeys((ENTRY, FACTORY), MEMBER)
    seen = set()
    for row in aliases.get(MEMBER, ()):
        for name in (ENTRY, FACTORY):
            if name in row["functions"]:
                require(name not in seen, "native entry has multiple alias owners")
                require(type(row["binding"]) is str and row["binding"].startswith("_mojolearn_"),
                        "native alias owner is malformed")
                members[name] = row["binding"]
                seen.add(name)
    return members


def validate_build(root, proof):
    """Reconcile actual loaded bytes with the existing successful cache receipt.

    No compiler is invoked. A valid older receipt may qualify unchanged native
    inputs, while the Python connection witness is exact-source evidence.
    """
    import development_build as build
    import merged_bindings as merged
    root = Path(root).resolve()
    policy_root = Path(__file__).resolve().parents[1]
    require(all(sha256(policy_root / path) == sha256(root / path) for path in POLICY),
            "executing installed-slot policy differs from frozen source")
    require(all(sha256(policy_root / path) == sha256(root / path) for path in
                ("bindings/development_groups.json", "bindings/merged_groups.json")),
            "executing group manifest differs from frozen source")
    members = owner_members(root)
    require(type(proof.get("owners")) is dict and set(proof["owners"]) == set(members),
            "native entry/factory owner proofs missing")
    index = read_reference(proof["index"], "activation index")
    source_graph = build.SourceGraph(root)
    return {name: _validate_owner(root, item, proof["index"], index, name, members[name], source_graph)
            for name, item in proof["owners"].items()}


def _validate_owner(root, proof, index_reference, index, name, member, source_graph):
    """Validate every loaded owner independently, including an aliased factory."""
    import development_build as build
    import merged_bindings as merged
    require(proof.get("member") == member, "native proof names the wrong declared owner")
    binary = read_reference(proof["binary"], "loaded binary", json_data=False)
    receipt = read_reference(proof["receipt"], "build receipt")
    fields = receipt.get("fields", {})
    require(receipt.get("exit_code") == 0 and receipt.get("state") == "built",
            "native build receipt is not a successful producer result")
    require(receipt.get("sha256") == proof["binary"]["sha256"], "loaded binary differs from build receipt")
    require(receipt.get("key") == build.digest(fields), "native build input key mismatch")
    require(fields.get("schema") == build.KEY_SCHEMA and fields.get("mode") == "host",
            "unsupported attribute-owner build receipt")
    require(receipt.get("stem") == fields.get("stem") and receipt.get("mode") == fields.get("mode"),
            "native receipt result differs from its input contract")
    require(fields.get("platform") in ("darwin", "linux"), "unsupported native build platform")
    require(re.fullmatch("[0-9a-f]{64}", str(fields.get("toolchain", {}).get("compiler_sha256", "")))
            and fields["toolchain"].get("available") is True,
            "native compiler byte identity missing")
    require(fields.get("environment", {}).get("MOJOLEARN_NUMERIC_MODE") == "identical",
            "attribute witness requires the identical build contract")
    # This also rejects sabotage/unsupported defines instead of trusting the label.
    build.build_environment("host", fields["environment"], fields["platform"])
    plans = list(merged.plan("host", fields["platform"],
                            fields["environment"].get("MOJOLEARN_PACKAGE_BYTE_LM", "1") == "1", "development"))
    expected = [r for r in plans if r[0] == fields.get("stem")]
    require(len(expected) == 1 and expected[0][1] == fields.get("flags_class")
            and expected[0][2] == fields.get("members") and member in fields["members"],
            "loaded group does not contain the declared attribute owner")
    closure = source_graph.closure(fields["members"])
    require(not closure["unresolved"] and fields.get("source") == closure,
            "native source closure differs or is unresolved")
    scripts = {p: hashlib.sha256(build.recipe_text((root / p).read_text()).encode()).hexdigest()
               for p in build.BUILD_INPUTS}
    require(fields.get("builders") == build.compile_recipe(scripts, "host", fields["stem"], fields["flags_class"]),
            "native build recipe differs")
    require(fields.get("root_sha256") == hashlib.sha256(merged.root_text(fields["stem"], fields["members"]).encode()).hexdigest(),
            "generated native root differs")
    require(index.get("format") == merged.INDEX_FORMAT and index.get("layout") == "development",
            "unsupported native activation layout")
    require(index.get("provenance", {}).get(fields["stem"]) == receipt["key"], "activation input key differs")
    index_dir = Path(index_reference["path"]).parent.resolve()
    matches = [name for name, members in index.get("files", {}).items()
               if member in members and (index_dir / name).resolve() == binary.resolve()]
    require(len(matches) == 1 and index["files"][matches[0]] == fields["members"],
            "loaded binary is not the activated attribute-owner group")
    require(binary.name == fields["stem"] + ".so", "native artifact name differs")
    origin = proof.get("origin", {})
    require(origin.get("name") == name and Path(origin.get("path", "")).resolve() == binary.resolve()
            and type(origin.get("offset")) is int and origin["offset"] >= 0,
            "native entry/factory pointer is not in the receipted binary")
    return receipt["key"]


def validate(path, root):
    envelope = json.loads(Path(path).read_text())
    require(envelope.get("schema") == SCHEMA, "unsupported installed-slot witness schema")
    data = read_reference(envelope.get("observations"), "slot observations")
    require(data.get("schema") == OBSERVATIONS and data.get("owner") == OWNER,
            "unsupported installed-slot observation producer")
    require(data.get("runtime_qualified") is False, "slot witness cannot qualify runtime safety")
    current_source = validate_source(data.get("source"), root)
    build_keys = validate_build(root, data["provenance"])
    admitted, seen = {}, set()
    require(type(data.get("rows")) is list, "installed-slot rows missing")
    for row in data["rows"]:
        key = row.get("class")
        require(type(key) is str and key.startswith("mojolearn.") and key not in seen,
                "duplicate/invalid installed-slot class")
        seen.add(key)
        require(row.get("status") in ("witnessed", "pending", "unmodeled"), "invalid installed-slot status")
        if row["status"] != "witnessed":
            continue
        require(row.get("method") == "__getattribute__" and row.get("checks") == list(CHECKS),
                "installed-slot identity checks are incomplete")
        exports = row.get("exports")
        require(type(exports) is list and exports and all(type(v) is str and v.startswith("mojolearn.")
                and v.count(".") == 1 for v in exports), "installed-slot root exports missing")
        admitted[key] = {"owner": OWNER, "entry": ENTRY, "exports": exports,
                         "witness_sha256": sha256(path), "observations_sha256": envelope["observations"]["sha256"],
                         "build_keys": build_keys,
                         "binary_sha256s": {name: item["binary"]["sha256"]
                                            for name, item in data["provenance"]["owners"].items()},
                         "source_revision": data["source"]["revision"],
                         "validated_source_revision": current_source["revision"],
                         "source_files_sha256": data["source"]["files_sha256"]}
    return admitted


def apply(row, class_key, witnesses):
    """Only the observed getter changes; every evidence axis remains unreviewed."""
    witness = witnesses.get(class_key)
    if row["method"] != "__getattribute__" or not witness:
        return
    require(set(row["aliases"]) & set(witness["exports"]), "installed class does not match public export resolution")
    row["superseded_source"] = {key: row.get(key) for key in
                                ("static_verdict", "runtime_status", "definition", "source_ast_sha256",
                                 "qualification_debt", "native_entries")}
    row.update(static_verdict="installed_native", runtime_status="native_boundary",
               native_entries=[OWNER + ":" + ENTRY], installed_slot_witness=witness,
               qualification_debt=["Installed slot identity only; public runtime and transitive callees remain unqualified."])
