"""Admission of retained Linux release builds, never development-cache passes.

The build producer captures actual inputs before invoking build_sets. The
controller's separate launch record binds begun ledger runs before execution.
Both records, the original complete build proof and actual output bytes are
required. Missing historical provenance is reconciliation, not a rerun request.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath

import release_evidence as evidence

CAPTURE_SCHEMA = "mojolearn.linux.release-compile-inputs.v1"
REPORT_SCHEMA = "mojolearn.linux.release-compile-report.v1"
LAUNCH_SCHEMA = "mojolearn.release-compile-launch.v1"
TOOL = "tools/linux_surface_qualification.sh"
BUILD_SCHEMA = "mojolearn.linux.build-provenance.v1"
BASE_COVERAGE = "all selected native bindings and required enabled/disabled switches"
CAPTURE_FLAG_NAMES = frozenset({
    "CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS",
    "MOJOLEARN_GPU_ARCHS", "MOJOLEARN_BUILD_TIERS", "MOJOLEARN_NUMERIC_MODE",
    "MOJOLEARN_PACKAGE_BYTE_LM", "MOJOLEARN_LINUX_CPU", "MOJOLEARN_TARGET_COLUMN",
    "MOJOLEARN_CUDA_CODE_FORMAT", "MOJOLEARN_MERGED_BINDINGS", "MOJOLEARN_BUILD_PIXI_ENV",
    "MOJOLEARN_COMPILE_JOBS", "MOJOLEARN_AMD_COMPILE_JOBS", "MOJOLEARN_BUILD_DEFINES",
    "MOJOLEARN_MOJO_BUILD_FLAGS", "MOJOLEARN_EXTRA_DEFINES", "MOJOLEARN_BUILD_EXTRA_DEFINES",
    "MOJOLEARN_MAMBA_DEFINES", "MOJOLEARN_ATTENTION_MEMORY_PROFILE", "MOJOLEARN_ALLOW_NO_GPU",
})


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(evidence.read_text(path))


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=evidence.sha256(path))


def checked(ref):
    require(isinstance(ref, dict) and evidence.sha256(ref["path"]) == ref["sha256"],
            "retained compile evidence hash differs")
    return load(ref["path"])


def binding_key(identity):
    return "/".join(str(identity.get(k) or "-") for k in ("target", "vendor", "arch", "tier", "name"))


def safe_build_flags(environ):
    from release_requirements import build_flags
    flags = build_flags(environ)
    unknown = set(flags) - CAPTURE_FLAG_NAMES
    require(not unknown, "unrecognized build input names need an explicit capture policy: " + ", ".join(sorted(unknown)))
    return {key: flags[key] for key in sorted(CAPTURE_FLAG_NAMES & flags.keys())}


def supports_requirement(wanted):
    inputs = wanted.get("inputs", {})
    unit = inputs.get("unit", {})
    identity = inputs.get("binding") or inputs.get("runtime") or inputs.get("environment") or {}
    if "binding" in inputs:
        expected = [BASE_COVERAGE, binding_key(identity)]
    else:
        # Portable-math runtime libraries and adapter registration also compile
        # C/C++. Their executable identities are not yet captured or bound to
        # begun inputs; a Mojo executable hash cannot qualify those units.
        return False
    return (wanted.get("kind") == "compile" and unit.get("kind") == "compile"
            and not wanted.get("missing_inputs")
            and identity.get("target") == "linux-64"
            and unit.get("release_modes") == ["identical"]
            and isinstance(unit.get("compiler_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", unit["compiler_sha256"]) is not None
            and isinstance(unit.get("build_flags"), dict)
            and set(unit["build_flags"]) <= CAPTURE_FLAG_NAMES
            and wanted.get("coverage") == expected
            and not unit.get("compile_variants")
            and not inputs.get("configuration", {}).get("compile_variants"))


def belongs_to_leg(wanted, vendor, arch):
    if not supports_requirement(wanted):
        return False
    inputs = wanted["inputs"]
    unit_vendor = inputs["unit"]["vendor"]
    # All legs contain host copies. One real release leg owns their ledger
    # execution, avoiding three simultaneous begin records for the same unit.
    if unit_vendor == "cpu":
        return (vendor, arch) == ("cuda", "sm_89")
    if unit_vendor != {"cuda": "nvidia", "hip": "amd"}.get(vendor):
        return False
    if "binding" in inputs:
        if inputs["binding"].get("tier") == "host":
            return arch != "sm_80"  # One native leg owns its vendor's host copies.
        return inputs["binding"].get("arch") == arch
    if "environment" in inputs:
        return inputs["environment"].get("arch") == arch
    return arch != "sm_80"  # The PTX leg does not own shared runtime qualification.


def capture(root, output, source, vendor, arch):
    """Called on the build machine before work; no compiler is invoked here."""
    import release_reuse as reuse
    root, output = Path(root).resolve(), Path(output)
    require(root == Path(reuse.ROOT).resolve(), "compile capture tools belong to another source tree")
    require(vendor in ("cuda", "hip"), "unsupported Linux release vendor")
    flags = safe_build_flags(os.environ)
    facts = reuse.tree_facts(root)
    identities = {}
    for binding in reuse.bindings(reuse.LINUX, ["identical"]):
        if binding.host or (binding.vendor, binding.arch) == (vendor, arch):
            identity = reuse.identity(binding, facts)
            require(not reuse.unreadable(identity), "unresolved compile identity: " + binding.key)
            identities[binding.key] = identity
    # Match build_sets.sh's actual environment selection, including its default.
    # Reading a different installed compiler would be a false byte witness.
    pixi_env = os.environ.get("MOJOLEARN_BUILD_PIXI_ENV", "gbmbench")
    require(re.fullmatch(r"[A-Za-z0-9_-]+", pixi_env) is not None, "unsupported build environment name")
    prefix = root / ".pixi/envs" / pixi_env
    packages = sorted(p.stem for p in (prefix / "conda-meta").glob("*.json"))
    expected = facts["toolchain"][reuse.LINUX]
    require(set(expected["packages"]) <= set(packages), "installed compiler packages differ from frozen pins")
    compiler = prefix / "bin/mojo"
    require(compiler.is_file(), "installed compiler executable is missing")
    inventory = load(output.parent / "source-inventory.json")
    for rel, wanted in inventory:
        require(evidence.sha256(root / rel) == wanted, "source changed before compilation: " + rel)
    record = dict(schema=CAPTURE_SCHEMA, source_commit=source, vendor=vendor, arch=arch,
        phase="before-build", source_inventory_sha256=evidence.sha256(output.parent / "source-inventory.json"),
        source_inventory_digest=digest(inventory), binding_identities=identities,
        toolchain=expected, installed_packages=packages, compiler_sha256=evidence.sha256(compiler),
        build_flags=flags, command=["bash", "packaging/linux/build_sets.sh", str(output.parent)])
    output.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    return record


def _member(root, relative):
    name = PurePosixPath(relative)
    require(not name.is_absolute() and ".." not in name.parts and str(name) == relative,
            "unsafe compile artifact member")
    return Path(root).joinpath(*name.parts)


def _artifact(refs, root, name, expected):
    ref = reference(_member(root, name))
    require(ref["sha256"] == expected, "compiled artifact bytes differ: " + name)
    refs.append(ref)
    return ref


def _flags(captured, wanted, binding):
    """Separate fixed release routing from declared numerical/build switches."""
    actual = dict(captured)
    routing = {
        "MOJOLEARN_GPU_ARCHS": binding.get("arch"),
        "MOJOLEARN_BUILD_TIERS": "identical",
        "MOJOLEARN_NUMERIC_MODE": "identical",
        "MOJOLEARN_PACKAGE_BYTE_LM": "1",
        "MOJOLEARN_LINUX_CPU": "x86-64-v3",
        "MOJOLEARN_TARGET_COLUMN": "nvidia" if binding.get("vendor") == "cuda" else "amd",
        "MOJOLEARN_CUDA_CODE_FORMAT": "ptx" if binding.get("arch") == "sm_80" else "native",
        "MOJOLEARN_MERGED_BINDINGS": "1",
        "MOJOLEARN_BUILD_PIXI_ENV": "default",
        "MOJOLEARN_COMPILE_JOBS": "2",
    }
    for key, default in routing.items():
        if key in actual and key not in wanted:
            require(actual[key] == default, "release routing flag differs: " + key)
            actual.pop(key)
    require(actual == wanted, "actual build flags/switch configuration differs from begun inputs")


def producer_payload(requirement_id, wanted, report_path, provenance):
    import release_requirement_receipts as receipts
    require(supports_requirement(wanted), "unsupported compile target/configuration; reconcile provenance")
    report_path = Path(report_path).resolve()
    report = load(report_path)
    require(report.get("schema") == REPORT_SCHEMA and report.get("producer") == TOOL,
            "not a supported Linux release compile report")
    refs = [reference(report_path)]
    records = {}
    for key in ("build_report", "capture_report", "launch_report", "set_manifest"):
        records[key] = checked(report[key]); refs.append(report[key])
    proof, capture_record, launch, manifest = (records[key] for key in
        ("build_report", "capture_report", "launch_report", "set_manifest"))
    exit_ref = report["exit_report"]
    require(evidence.sha256(exit_ref["path"]) == exit_ref["sha256"]
            and evidence.read_text(exit_ref["path"]).strip() == "0", "release build did not exit zero")
    refs.append(exit_ref)
    require(proof.get("schema") == BUILD_SCHEMA and proof.get("action") == "build"
            and proof.get("complete") is True and type(proof.get("build_exit")) is int
            and proof["build_exit"] == 0 and not proof.get("layout_errors"),
            "release build proof is incomplete or not an actual complete build")
    source = provenance.get("source_revision")
    require(proof.get("source_commit") == capture_record.get("source_commit") == launch.get("source_commit") == source,
            "compile source provenance differs")
    require(capture_record.get("schema") == CAPTURE_SCHEMA and capture_record.get("phase") == "before-build"
            and proof.get("compile_inputs_sha256") == report["capture_report"]["sha256"],
            "build lacks its original before-build compiler/flag capture")
    require(launch.get("compile_requirements", {}).get("schema") == LAUNCH_SCHEMA and launch.get("launched_at"),
            "build lacks original ledger launch provenance")
    begun = launch["compile_requirements"].get("runs", {}).get(requirement_id, {})
    require(begun.get("run_id") == provenance.get("run_id")
            and begun.get("fingerprint") == wanted.get("fingerprint")
            and begun.get("source_revision") == source, "build was not launched for these begun inputs")
    require(proof.get("release_modes") == ["identical"], "release compile modes differ")
    inventory = proof.get("source_inventory")
    require(isinstance(inventory, list) and inventory and len(dict(inventory)) == len(inventory),
            "missing or duplicate compiled source inventory")
    require(proof.get("source_sha256") == hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest()
            and capture_record.get("source_inventory_digest") == digest(inventory), "compiled source inventory differs")
    toolchain = capture_record.get("toolchain", {})
    require(set(toolchain.get("packages", [])) <= set(capture_record.get("installed_packages", []))
            and toolchain.get("packages") and re.fullmatch("[0-9a-f]{64}", capture_record.get("compiler_sha256", "")),
            "actual compiler/package identity is missing")
    inputs = wanted["inputs"]
    expected_identity = inputs.get("binding") or inputs.get("runtime") or inputs.get("environment")
    require(expected_identity.get("toolchain") == toolchain, "actual compiler toolchain differs from begun inputs")
    require(capture_record["compiler_sha256"] == inputs["unit"]["compiler_sha256"],
            "actual compiler executable bytes differ from begun inputs")
    vendor, arch = capture_record.get("vendor"), capture_record.get("arch")
    require((launch.get("vendor"), launch.get("arch")) == (vendor, arch), "build target differs from launch")
    _flags(capture_record.get("build_flags", {}), inputs["unit"]["build_flags"], dict(vendor=vendor, arch=arch))
    root = report["set_root"]
    require(Path(report["set_manifest"]["path"]).resolve() == (Path(root) / "manifest.json").resolve(),
            "set manifest does not belong to compiled artifacts")
    prefix = f"mojolearn/{vendor}/{arch}/"
    require(proof.get("set_manifests", {}).get(prefix + "manifest.json") == report["set_manifest"]["sha256"],
            "build proof does not bind the retained set manifest")
    counts = dict.fromkeys(wanted["coverage"], 1)
    binding = inputs["binding"]
    key = binding_key(binding)
    require(capture_record.get("binding_identities", {}).get(key) == binding, "compiled binding inputs differ")
    require(binding.get("toolchain") == toolchain, "compiled binding toolchain differs")
    tier = binding["tier"]
    outputs = proof.get("host_extension" if tier == "host" else "extensions", {})
    if proof.get("binding_layout") == "merged":
        index_name = prefix + tier + "/_mojolearn_merged.json"
        index = proof.get("merged_indexes", {}).get(index_name)
        require(isinstance(index, dict), "compiled merged index is missing")
        ref = _artifact(refs, root, tier + "/_mojolearn_merged.json", index["sha256"])
        actual_index = load(ref["path"])
        require(actual_index.get("files") == index.get("files"), "merged membership differs from build proof")
        owners = [name for name, members in index["files"].items() if binding["name"] in members]
        require(len(owners) == 1, "compiled binding has missing or duplicate merged owners")
        relative = tier + "/" + owners[0]
    else:
        require(proof.get("binding_layout") == "per-binding", "unsupported release binding layout")
        relative = tier + "/" + binding["name"] + ".so"
    require(prefix + relative in outputs, "build proof does not cover the selected native binding")
    _artifact(refs, root, relative, outputs[prefix + relative])
    proof_path, proof_hash = receipts._producer_stamp(requirement_id, wanted, report_path, provenance)
    proof_ref = dict(path=str(proof_path), sha256=proof_hash); refs.append(proof_ref)
    return dict(schema=receipts.SCHEMA, requirement_id=requirement_id, kind="compile",
        source_revision=source, input_fingerprint=wanted["fingerprint"],
        producer=dict(tool=TOOL, report_type="build"), producer_provenance=dict(provenance),
        producer_report=refs[0], producer_provenance_report=proof_ref, status="passed", exit_code=0,
        evidence=refs, coverage=[dict(id=name, status="passed", exit_code=0, expected=counts[name], completed=counts[name],
                                    failed=0, skipped=0) for name in wanted["coverage"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["capture"])
    parser.add_argument("--root", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--source", required=True); parser.add_argument("--vendor", required=True)
    parser.add_argument("--arch", required=True)
    args = parser.parse_args()
    capture(args.root, args.output, args.source, args.vendor, args.arch)


if __name__ == "__main__":
    main()
