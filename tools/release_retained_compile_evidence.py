"""Retained compile producers for the requirements ledger.

Used by tools/release_ledger_admission.py; never invokes a compiler. Three
actual producers are recognized, each bound to its own retained bytes:

  mac-development-build   tools/development_build.py result.json (schema
                          mojolearn.development-build.v1, state activated).
                          Owner policy (2026-10-25): an incremental development
                          build counts for a binding when its source, compiler,
                          flags, target and binary provenance are exact. The
                          development layout and builder differ from the release
                          packer; this admission covers compilation only, never
                          packaging, loading or installed-wheel behavior.
  apple-retained-device   the original production Metal subset run that
                          tools/apple_retained_groups.py admits; its
                          before-build capture holds the exact binding identities.
  linux-host-build        tools/release_host_build.py output: one merged Linux
                          host build with native readback.

Every unit must match the requirement's binding identity, toolchain, flags and
source closure at the producer's own tested revision. Anything unverifiable is
refused; the ledger reuses an admitted unit only for unchanged inputs.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from release_ledger_admission import require, load, reference, checked

TOOL = "tools/release_retained_compile_evidence.py"
BASE_COVERAGE = "all selected native bindings and required enabled/disabled switches"
MAC, LINUX = "osx-arm64", "linux-64"
DEV_SCHEMA = "mojolearn.development-build.v1"
DEV_KEY_SCHEMA = "mojolearn.development-group.v1"
# Generated from tracked generator inputs that are themselves in the closure.
from generated_sources import GENERATED as GENERATED_SOURCES  # noqa: E402
GENERATED = set(GENERATED_SOURCES)
# Binding identity flag -> the environment variable each builder exports.
FLAG_ENV = {"numeric_mode": "MOJOLEARN_NUMERIC_MODE", "target_column": "MOJOLEARN_TARGET_COLUMN",
            "gpu_archs": "MOJOLEARN_GPU_ARCHS", "linux_cpu": "MOJOLEARN_LINUX_CPU",
            "compile_jobs": "MOJOLEARN_COMPILE_JOBS", "package_byte_lm": "MOJOLEARN_PACKAGE_BYTE_LM"}
# Placement of the selected toolchain/interpreter, checked against the build
# checkout; not numerical/build switches.
DEV_PLACEMENT = {"MODULAR_HOME": ".pixi/envs/default/share/max"}
# MOJOLEARN_MERGED_PYTHON only runs the merged-index glue (bindings/merged_build.sh ->
# tools/merged_bindings.py); it never compiles. Any interpreter is admissible if it is
# recorded as an absolute path in the group's environment (and therefore in its key).
GLUE_INTERPRETERS = ("MOJOLEARN_MERGED_PYTHON",)
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")


def binding_key(identity):
    return "/".join(str(identity.get(k) or "-") for k in ("target", "vendor", "arch", "tier", "name"))


def supports_requirement(wanted):
    inputs = wanted.get("inputs", {})
    unit, binding = inputs.get("unit", {}), inputs.get("binding")
    if (wanted.get("kind") != "compile" or unit.get("kind") != "compile" or not isinstance(binding, dict)
            or wanted.get("missing_inputs") or unit.get("release_modes") != ["identical"]
            or unit.get("compile_variants") or inputs.get("configuration", {}).get("compile_variants")
            or not isinstance(unit.get("build_flags"), dict)
            or wanted.get("coverage") != [BASE_COVERAGE, binding_key(binding)]):
        return False
    if binding.get("target") == MAC:
        return ((unit.get("vendor") == "apple" and binding.get("tier") in ("host", "identical"))
                or (unit.get("vendor") == "cpu" and binding.get("tier") == "host"))
    if binding.get("target") == LINUX:
        return (binding.get("tier") == "host" and unit.get("vendor") in ("cpu", "nvidia", "amd")
                and isinstance(unit.get("compiler_sha256"), str) and HEX64.fullmatch(unit["compiler_sha256"]) is not None)
    return False


class Tree:
    """Exact content hashes of one tested revision's tracked files."""
    _open = {}

    def __init__(self, root, revision):
        from release_requirements import source_files
        from release_public_runtime_evidence import BlobHashes
        self.root, self.revision = str(Path(root).resolve()), revision
        self.files = source_files(self.root, revision)
        self.blobs = BlobHashes(self.root)

    @classmethod
    def at(cls, root, revision):
        key = (str(Path(root).resolve()), revision)
        if key not in cls._open:
            cls._open[key] = cls(root, revision)
        return cls._open[key]

    def sha256(self, rel):
        entry = self.files.get(rel)
        return None if entry is None else self.blobs.sha(entry)

    def text(self, rel):
        import subprocess
        return subprocess.check_output(["git", "-C", self.root, "show", self.revision + ":" + rel], text=True)


def generated_sha(tree, rel, checkout):
    """The sha256 the tested revision's tracked generator gives build-time
    generated source REL (generated_sources.expected_sha: regenerated here, or
    the checkout's copy verified byte for byte against the tracked generator)."""
    import generated_sources as gs
    require(all(name in tree.files for name in gs.GENERATED[rel]),
            "generated source has no tracked generator at the tested revision: " + rel)
    return gs.expected_sha(rel, {name: tree.text(name).encode() for name in gs.GENERATED[rel]},
                           [Path(checkout) / rel])


def closure_digest(tree, rels):
    """bincache.source_digest over exact tracked bytes at the tested revision."""
    import bincache
    value = hashlib.sha256()
    for rel in sorted(rels, key=lambda name: PurePosixPath(name).parts):
        if rel == "pixi.toml":
            found = bincache.sha256_bytes(bincache.pixi_toml_build_part(tree.text(rel)).encode())
        else:
            found = tree.sha256(rel)
        require(found is not None, "binding closure file is not tracked at the tested revision: " + rel)
        value.update(rel.encode() + b"\0" + found.encode() + b"\n")
    return value.hexdigest()


def check_flags(flags, environment):
    """Every identity flag equals the variable the producer exported; no other routing."""
    for key, value in flags.items():
        name = FLAG_ENV.get(key)
        require(name is not None, "unrecognized binding flag: " + key)
        require(environment.get(name) == (None if value is None else str(value)), "compiled flag differs: " + name)
    for name in set(FLAG_ENV.values()) - {FLAG_ENV[k] for k in flags}:
        require(name not in environment, "undeclared routing flag was exported: " + name)


def derived_build_flags(environment, routing):
    """The declared numerical/build switches: exported flags minus fixed routing."""
    from release_requirements import build_flags
    return {k: v for k, v in build_flags(environment).items() if k not in routing}


def coverage_rows(wanted):
    return [dict(id=name, status="passed", exit_code=0, expected=1, completed=1, failed=0, skipped=0)
            for name in wanted["coverage"]]


def closure_rels(wanted):
    binding = wanted["inputs"]["binding"]
    return [p for p in wanted.get("provenance_sources", []) if p not in (binding.get("builders") or {})]


def require_closure(tree, wanted):
    binding = wanted["inputs"]["binding"]
    require(binding.get("closure", {}).get("scope") == "closure", "binding has no exact native import closure")
    rels = closure_rels(wanted)
    require(rels and closure_digest(tree, rels) == binding["closure"]["digest"],
            "binding closure digest does not reproduce from tested source bytes")
    return rels


# ---------------------------------------------------------------- development build
DEV_ROUTING = set(FLAG_ENV.values()) | set(DEV_PLACEMENT) | set(GLUE_INTERPRETERS)


def _dev_group(row, repo, cache, tree, *, binary_path=None, cpu_contract=False):
    import development_build as devb
    fields = row["fields"]
    require(row["key"] == devb.digest(fields) and fields.get("schema") == DEV_KEY_SCHEMA
            and fields.get("platform") == "darwin" and fields.get("mode") == row["mode"]
            and fields.get("stem") == row["stem"] and fields.get("members") == row["members"]
            and Path(fields.get("repo", "")).resolve() == repo and Path(fields.get("cache", "")).resolve() == cache,
            "development group key/fields differ: " + row["stem"])
    source = fields.get("source") or {}
    require(source.get("scope") == "closure" and not source.get("unresolved"),
            "development group closure is unresolved: " + row["stem"])
    toolchain = fields.get("toolchain") or {}
    require(toolchain.get("available") is True and HEX64.fullmatch(str(toolchain.get("compiler_sha256")))
            and isinstance(toolchain.get("locked"), dict) and isinstance(toolchain.get("apple"), dict),
            "development compiler identity is missing: " + row["stem"])
    environment = fields.get("environment") or {}
    for name, rel in DEV_PLACEMENT.items():
        require(name not in environment or Path(environment[name]) == repo / rel,
                "development toolchain placement differs: " + name)
    for name in GLUE_INTERPRETERS:
        require(name not in environment or Path(environment[name]).is_absolute(),
                "development glue interpreter is not recorded as an absolute path: " + name)
    receipt_path = cache / "artifacts" / row["key"] / "receipt.json"
    receipt = load(receipt_path)
    require(receipt.get("key") == row["key"] and receipt.get("fields") == fields
            and type(receipt.get("exit_code")) is int and receipt["exit_code"] == 0
            and HEX64.fullmatch(str(receipt.get("sha256"))), "development compile receipt differs: " + row["stem"])
    binary = binary_path or repo / "python/mojolearn" / row["mode"] / ".development" / row["key"] / (row["stem"] + ".so")
    binary_ref = reference(binary)
    require(binary_ref["sha256"] == receipt["sha256"], "compiled binary differs from its receipt: " + row["stem"])
    for path, sha in source["files"].items():
        require(generated_sha(tree, path, repo) == sha if path in GENERATED else tree.sha256(path) == sha,
                "compiled source bytes differ from the tested revision: " + path)
    from development_build import recipe_text
    for path, sha in fields["builders"]["scripts"].items():
        require(path in tree.files and hashlib.sha256(recipe_text(tree.text(path)).encode()).hexdigest() == sha,
                "development builder recipe differs from the tested revision: " + path)
    host_toolchain = toolchain["apple"]
    if cpu_contract:
        import macos_cpu_compile_contract as cpu
        require(row["mode"] == "host", "CPU link contract cannot qualify a device image")
        lock_hash = toolchain["locked"].get("pixi_lock_sha256")
        require(lock_hash == tree.sha256("pixi.lock"), "compiled CPU lock differs from frozen source")
        host_toolchain = cpu.contract(toolchain["compiler_sha256"], lock_hash, cpu.read_image(binary))
    return dict(mode=row["mode"], key=row["key"], members=list(row["members"]), environment=dict(environment),
                toolchain_locked=toolchain["locked"], host_toolchain=host_toolchain,
                compiler_sha256=toolchain["compiler_sha256"], source_files=dict(source["files"]),
                receipt=reference(receipt_path), binary=binary_ref)


def _dev_facts(root, producer):
    result_path = Path(producer["result"]).resolve()
    result = load(result_path)
    plan = result.get("plan") or {}
    revision = result.get("source_commit")
    require(result.get("schema") == DEV_SCHEMA and result.get("state") == "activated"
            and result.get("missing") == [] and plan.get("platform") == "darwin"
            and plan.get("layout") == "development" and isinstance(revision, str) and HEX40.fullmatch(revision),
            "not a complete activated macOS development build")
    require(all(type(r.get("exit_code")) is int and r["exit_code"] == 0 for r in result.get("results", [])),
            "development build contains a failed compile")
    activated = set(result.get("activated", []))
    require(activated and activated <= {"identical", "host"}, "development build activated no supported set")
    repo, cache = Path(plan["repo"]).resolve(), Path(plan["cache"]).resolve()
    tree = Tree.at(root, revision)
    groups = {row["stem"]: _dev_group(row, repo, cache, tree) for row in plan["groups"] if row["mode"] in activated}
    require(groups, "development build has no activated groups")
    return dict(producer="mac-development-build", source_root=str(Path(root).resolve()), source_revision=revision,
                result=reference(result_path), activated=sorted(activated), groups=groups)


def _dev_collect(root, document, producer):
    facts = _dev_facts(root, producer)
    # compile-cpu names one target; a macOS host build cannot qualify Linux CPU.
    cpu_target = (document or {}).get("configuration", {}).get("dependency_context", {}).get("cpu_target")
    units, flags, apple, workers = {}, set(), set(), {}
    for stem, group in facts["groups"].items():
        # The builder records the actual worker setting in its content key. Keep
        # it per binding; another job count has not been proven byte-neutral.
        jobs = group["environment"].get("MOJOLEARN_COMPILE_JOBS")
        require(isinstance(jobs, str) and re.fullmatch(r"[1-9][0-9]*", jobs),
                "development compile worker count is missing or malformed: " + stem)
        flags.add(json.dumps(derived_build_flags(group["environment"], DEV_ROUTING), sort_keys=True))
        apple.add(json.dumps(group["host_toolchain"], sort_keys=True))
        for name in group["members"]:
            if group["mode"] == "host":
                key = f"{MAC}/host/-/host/{name}"
                specs = ("compile-apple", "compile-cpu") if cpu_target == MAC else ("compile-apple",)
            else:
                key = f"{MAC}/metal/apple/{group['mode']}/{name}"
                specs = ("compile-apple",)
            require(key not in workers or workers[key] == jobs,
                    "development compile worker counts conflict for: " + key)
            workers[key] = jobs
            for spec in specs:
                rid = spec + "/" + key
                require(rid not in units or units[rid]["group"] == stem,
                        "development groups conflict for binding: " + key)
                units[rid] = dict(group=stem, binding_key=key)
    require(len(flags) == 1 and len(apple) == 1, "development groups used different build flags or Apple toolchains")
    configuration = dict(release_modes=["identical"], build_flags=json.loads(flags.pop()),
                         dependency_context=dict(apple_toolchain=json.loads(apple.pop()), cpu_target=MAC,
                                                 selected_binding_compile_jobs=workers))
    return dict(source_revision=facts["source_revision"], facts=facts, units=units, configuration=configuration,
                spec_ids=sorted({rid.split("/", 1)[0] for rid in units}))


def _dev_check(requirement_id, wanted, facts, unit):
    require(supports_requirement(wanted), "unsupported compile target/configuration")
    binding, declared = wanted["inputs"]["binding"], wanted["inputs"]["unit"]
    require(binding_key(binding) == unit["binding_key"] and requirement_id.endswith("/" + unit["binding_key"])
            and binding.get("target") == MAC, "admitted unit names another binding")
    group = facts["groups"][unit["group"]]
    require(binding["name"] in group["members"]
            and group["mode"] == ("host" if binding["tier"] == "host" else binding["tier"]),
            "development group does not contain this binding/tier")
    require(binding.get("toolchain") == group["toolchain_locked"], "compiler packages differ from the binding inputs")
    require(binding.get("host_toolchain") == group["host_toolchain"], "Apple host toolchain differs")
    check_flags(binding.get("flags") or {}, group["environment"])
    require(declared["build_flags"] == derived_build_flags(group["environment"], DEV_ROUTING),
            "declared build flags differ from the compiled environment")
    tree = Tree.at(facts["source_root"], facts["source_revision"])
    entries = None
    for rel in require_closure(tree, wanted):
        if rel.endswith(".mojo"):
            actual = group["source_files"].get(rel)
            if actual is None:
                # Merged builds compile a generated root instead of the standalone
                # entries. Prove its recorded bytes and identical canonical public
                # exports; substantive sources and changed wrappers remain required.
                if entries is None:
                    from release_development_entry_proof import entry_proof
                    entries = entry_proof(tree, checked(group["receipt"])["fields"])
                actual = entries.get(rel)
            require(actual is not None and actual == tree.sha256(rel),
                    "binding Mojo source was not compiled at its tested bytes: " + rel)
    return coverage_rows(wanted), [facts["result"], group["receipt"], group["binary"]]


# ---------------------------------------------------------------- retained Apple production device run
DEVICE_ROUTING = {"MOJOLEARN_COMPILE_JOBS", "MOJOLEARN_MERGED_BINDINGS", "MOJOLEARN_MERGED_KEY",
                  "MOJOLEARN_NUMERIC_MODE", "MOJOLEARN_PACKAGE_BYTE_LM"}


def _device_facts(root, producer, full=False):
    run = Path(producer["run"]).resolve()
    if full:
        # Re-extracts the original source archive and reproduces all identities.
        import apple_retained_groups
        apple_retained_groups.original(run)
    before = reference(run / "before-build.json")
    capture = checked(before)
    summary_ref = reference(run / "summary.json")
    summary = checked(summary_ref)
    revision = capture.get("source_commit")
    require(capture.get("schema") == "mojolearn.apple.device-subset-inputs.v1" and capture.get("phase") == "before-build"
            and capture.get("layout") == "release" and capture.get("numeric_modes") == ["identical"]
            and isinstance(revision, str) and HEX40.fullmatch(revision)
            and summary.get("complete") is True and summary.get("source_commit") == revision
            and summary.get("expected_groups") == summary.get("completed_groups") == capture.get("groups")
            and (run / "controller.rc").read_text().strip() == "0", "original device run is incomplete")
    groups = {}
    for stem in capture["groups"]:
        base = run / stem
        receipt_ref, stamp_ref, index_ref = (reference(base / n) for n in
                                             ("receipt.json", "binding-stamp.json", "_mojolearn_merged.json"))
        artifact = reference(base / (stem + ".so"))
        receipt, stamp_doc, index = checked(receipt_ref), checked(stamp_ref), checked(index_ref)
        require(receipt.get("complete") is True and receipt.get("source_commit") == revision
                and receipt.get("group") == stem and type(receipt.get("build_exit")) is int and receipt["build_exit"] == 0
                and type(receipt.get("stamp_exit")) is int and receipt["stamp_exit"] == 0
                and receipt.get("compiler_sha256") == capture.get("compiler_sha256")
                and receipt.get("output_sha256") == artifact["sha256"]
                and (base / "build.rc").read_text().strip() == "0", "original device group did not build: " + stem)
        members = index.get("files", {}).get(stem + ".so")
        require(members and set(stamp_doc.get("members", {})) == set(members) and stamp_doc.get("commit") == revision,
                "original device group membership differs: " + stem)
        groups[stem] = dict(members=sorted(members), receipt=receipt_ref, stamp=stamp_ref, index=index_ref,
                            artifact=artifact, key=receipt.get("key"))
    identities = capture.get("binding_identities") or {}
    require(len(identities) == sum(len(g["members"]) for g in groups.values())
            and {i.get("name") for i in identities.values()} == {m for g in groups.values() for m in g["members"]},
            "captured identities and compiled members differ")
    return dict(producer="apple-retained-device", source_root=str(Path(root).resolve()), source_revision=revision,
                before_build=before, summary=summary_ref, compiler_sha256=capture.get("compiler_sha256"),
                host_toolchain=capture.get("host_toolchain"), build_environment=capture.get("build_environment"),
                binding_identities=identities, groups=groups)


def _device_collect(root, document, producer):
    facts = _device_facts(root, producer, full=True)
    environment = facts["build_environment"] or {}
    require(set(environment) <= DEVICE_ROUTING, "original device build exported undeclared switches")
    units = {"compile-apple/" + key: dict(binding_key=key) for key in facts["binding_identities"]}
    configuration = dict(release_modes=["identical"], build_flags={},
                         dependency_context=dict(apple_toolchain=facts["host_toolchain"], cpu_target=MAC))
    return dict(source_revision=facts["source_revision"], facts=facts, units=units,
                configuration=configuration, spec_ids=["compile-apple"])


def _device_check(requirement_id, wanted, facts, unit):
    require(supports_requirement(wanted), "unsupported compile target/configuration")
    binding, declared = wanted["inputs"]["binding"], wanted["inputs"]["unit"]
    key = unit["binding_key"]
    require(requirement_id == "compile-apple/" + key and binding_key(binding) == key
            and binding.get("tier") == "identical", "admitted unit names another binding")
    require(facts["binding_identities"].get(key) == binding, "captured binding identity differs from requirement inputs")
    require(declared["build_flags"] == {} and set(facts["build_environment"] or {}) <= DEVICE_ROUTING,
            "declared build flags differ from the original device build")
    owners = [stem for stem, group in facts["groups"].items() if binding["name"] in group["members"]]
    require(len(owners) == 1, "binding has missing or duplicate compiled owners")
    group = facts["groups"][owners[0]]
    return coverage_rows(wanted), [facts["before_build"], facts["summary"], group["receipt"],
                                   group["stamp"], group["index"], group["artifact"]]


# ---------------------------------------------------------------- Linux independent host build
def _host_facts(root, producer):
    import release_host_build as rhb
    import release_host_composition as hc
    directory = Path(producer["build"]).resolve()
    source, set_root = rhb.inspect(directory)
    capture = hc.decoded(source["capture"])
    compiler = capture.get("compiler_sha256")
    contract, _, proof, capture = rhb.record(source, compiler)
    rhb.validate_bytes(source, set_root, compiler)
    refs = [reference(directory / name) for name in ("host-build-proof.json", "compile-inputs.json",
            "source-inventory.json", "binding-stamp.json", "stage.json", "readback.json", "runtime-source.json")]
    files = sorted(rhb.host_paths().items())  # every merged host group file
    refs += [reference(set_root / path) for _f, path in files] + [reference(set_root / rhb.HOST_INDEX)]
    refs += [reference(directory / "raw" / f) for f, _path in files]
    import generated_sources as gs
    for rel in sorted(contract["generated_sources"]):  # retained, verified in rhb.record
        refs += [reference(directory / "generated" / rel)]
        refs += [reference(directory / "generated-inputs" / name) for name in gs.GENERATED[rel]]
    return dict(producer="linux-host-build", source_root=str(Path(root).resolve()),
                source_revision=contract["source_commit"], compiler_sha256=compiler,
                binding_identities=contract["host_identities"], build_environment=capture["build_environment"],
                group_origins=contract.get("group_origins", {}),
                member_groups={name:filename for filename,names in contract["index"]["files"].items() for name in names},
                members=sorted(n for names in contract["index"]["files"].values() for n in names), evidence=refs)


def _host_collect(root, document, producer):
    facts = _host_facts(root, producer)
    units = {}
    for key in facts["binding_identities"]:
        for spec in ("compile-cpu", "compile-nvidia", "compile-amd"):
            units[spec + "/" + key] = dict(binding_key=key,
                compilation_origin=facts["group_origins"].get(facts["member_groups"][key.rsplit("/", 1)[-1]]))
    configuration = dict(release_modes=["identical"], build_flags={},
                         dependency_context=dict(cpu_target=LINUX, compiler_sha256={LINUX: facts["compiler_sha256"]}))
    return dict(source_revision=facts["source_revision"], facts=facts, units=units,
                configuration=configuration, spec_ids=["compile-amd", "compile-cpu", "compile-nvidia"])


def _host_check(requirement_id, wanted, facts, unit):
    import release_host_build as rhb
    require(supports_requirement(wanted), "unsupported compile target/configuration")
    binding, declared = wanted["inputs"]["binding"], wanted["inputs"]["unit"]
    key = unit["binding_key"]
    require(requirement_id.endswith("/" + key) and binding_key(binding) == key and binding.get("target") == LINUX,
            "admitted unit names another binding")
    require(facts["binding_identities"].get(key) == binding, "captured host identity differs from requirement inputs")
    require(declared.get("compiler_sha256") == facts["compiler_sha256"], "compiler executable bytes differ")
    require(declared["build_flags"] == {} and facts["build_environment"] == rhb.FLAGS,
            "declared build flags differ from the host invocation")
    require(binding["name"] in facts["members"], "merged host index lacks this binding")
    origin = facts.get("group_origins", {}).get(facts.get("member_groups", {}).get(binding["name"]))
    require(unit.get("compilation_origin") == origin, "host compilation origin differs")
    return coverage_rows(wanted), list(facts["evidence"])


class _Adapter:
    def __init__(self, collect, facts, check):
        self.collect, self._facts, self.check_unit = collect, facts, check

    def verify(self, report):
        return self._facts(report["facts"]["source_root"], report["producer"])


ADAPTERS = {
    "mac-development-build": _Adapter(_dev_collect, _dev_facts, _dev_check),
    "apple-retained-device": _Adapter(_device_collect, _device_facts, _device_check),
    "linux-host-build": _Adapter(_host_collect, _host_facts, _host_check),
}
