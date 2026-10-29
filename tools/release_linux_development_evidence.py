"""Admit retained Linux development groups, including an incomplete whole set.

Compilation only: no claim about loading, runtime, identity, or wheel assembly.
The actual runpod_cpu_leg leg.txt supplies the source revision because its source
archive has no .git; every compiled closure is checked independently at that
revision. No producer documents are rewritten. Relocated cache/staged bytes,
selected-group executions and their original logs must all still be present.
"""
import hashlib
import json
import re
from pathlib import Path

import development_build as dev
from release_development_entry_proof import entry_proof
import release_retained_compile_evidence as common
from release_ledger_admission import require, load, reference

NAME = "linux-development-build"
ROUTING = common.DEV_ROUTING | set(common.GLUE_INTERPRETERS) | {"MOJOLEARN_CUDA_CODE_FORMAT"}


def supports_requirement(wanted):
    unit = wanted.get("inputs", {}).get("unit", {})
    binding = wanted.get("inputs", {}).get("binding", {})
    return (wanted.get("kind") == unit.get("kind") == "compile"
            and binding.get("target") == common.LINUX
            and binding.get("tier") in ("host", "identical")
            and unit.get("vendor") in ("cpu", "nvidia", "amd")
            and unit.get("release_modes") == ["identical"]
            and not unit.get("compile_variants")
            and not wanted.get("inputs", {}).get("configuration", {}).get("compile_variants")
            and not wanted.get("missing_inputs")
            and isinstance(unit.get("build_flags"), dict)
            and common.HEX64.fullmatch(str(unit.get("compiler_sha256"))) is not None
            and wanted.get("coverage") == [common.BASE_COVERAGE, common.binding_key(binding)])


def _relative(base, value):
    path = (base / value).resolve()
    require(path.is_relative_to(base.resolve()), "artifact path escapes retained producer directory")
    return path


def _group(row, result, result_path, leg, tree, stage):
    fields = row["fields"]
    stem, mode, key = row["stem"], row["mode"], row["key"]
    repo, cache = Path(result["plan"]["repo"]), Path(result["plan"]["cache"])
    require(key == dev.digest(fields) and fields.get("schema") == common.DEV_KEY_SCHEMA
            and fields.get("platform") == "linux" and mode in ("host", "identical")
            and fields.get("mode") == mode and fields.get("stem") == stem
            and fields.get("members") == row["members"] and row["members"]
            and fields.get("repo") == str(repo) and fields.get("cache") == str(cache),
            "development group key/fields differ: " + stem)
    entries = entry_proof(tree, fields)
    source, toolchain = fields.get("source", {}), fields.get("toolchain", {})
    require(source.get("scope") == "closure" and not source.get("unresolved") and source.get("files"),
            "development source closure is missing or unresolved")
    require(toolchain.get("available") is True
            and common.HEX64.fullmatch(str(toolchain.get("compiler_sha256")))
            and toolchain.get("locked", {}).get("platform") == common.LINUX
            and set(toolchain["locked"].get("packages", [])) <= set(toolchain.get("installed", [])),
            "Linux compiler executable/package witness is missing")
    environment = fields.get("environment", {})
    workers = environment.get("MOJOLEARN_COMPILE_JOBS", "")
    require(isinstance(workers, str) and re.fullmatch(r"[1-9][0-9]*", workers) is not None, "compiler worker count is missing")
    for name, rel in common.DEV_PLACEMENT.items():
        require(name not in environment or Path(environment[name]) == repo / rel,
                "toolchain placement differs: " + name)
    for name in common.GLUE_INTERPRETERS:
        require(name not in environment or Path(environment[name]).is_absolute(), "glue interpreter is not absolute")
    flags_class = "host" if mode == "host" else "default"
    require(row.get("flags_class") == fields.get("flags_class") == flags_class,
            "unexpected optimization flags class")
    scripts = fields.get("builders", {}).get("scripts", {})
    require(set(scripts) == {"bindings/merged_build.sh", "bindings/build_defines.sh"}, "builder scripts differ")
    for path, digest in scripts.items():
        require(hashlib.sha256(dev.recipe_text(tree.text(path)).encode()).hexdigest() == digest,
                "compiled builder recipe differs: " + path)
    require(fields["builders"] == dev.compile_recipe(scripts, mode, stem, flags_class), "builder command differs")
    for path, digest in source["files"].items():
        actual = common.generated_sha(tree, path, tree.root) if path in common.GENERATED else tree.sha256(path)
        require(actual == digest, "compiled source bytes differ: " + path)
    retained = leg / "remote/leg_out"
    receipt_path = retained / "development-cache/artifacts" / key / "receipt.json"
    receipt = load(receipt_path)
    require(receipt.get("fields") == fields and receipt.get("key") == key
            and receipt.get("stem") == stem and receipt.get("mode") == mode
            and type(receipt.get("exit_code")) is int and receipt["exit_code"] == 0,
            "selected group compile receipt differs or failed")
    execution = receipt.get("execution", {})
    command = dev.group_command(mode, stem, cache / "work" / key / "output", cache / "work" / key / "roots")
    require(receipt.get("command") == command == execution.get("command")
            and execution.get("complete") is True and execution.get("status") == "passed"
            and type(execution.get("exit_code")) is int and execution["exit_code"] == 0
            and type(execution.get("process_returncode")) is int and execution["process_returncode"] == 0
            and execution.get("cwd") == str(repo), "selected group has no successful exact execution")
    results = [r for r in result.get("results", []) if r.get("key") == key]
    require(len(results) == 1 and results[0].get("exit_code") == 0
            and results[0].get("execution") == execution and results[0].get("sha256") == receipt.get("sha256"),
            "result and selected group execution differ")
    execution_path = result_path.parent / Path(receipt["execution_receipt"]).name
    require(load(execution_path) == execution, "retained execution receipt differs")
    log = result_path.parent / Path(receipt["log"]).name
    binary = reference(receipt_path.parent / (stem + ".so"))
    require(binary["sha256"] == receipt.get("sha256"), "raw compiled binary differs")
    staged = [r for r in stage if r.get("source_key") == key and r.get("group") == stem]
    require(len(staged) == 1, "selected group has no unique retained stage witness")
    staged_binary = reference(_relative(retained, staged[0]["staged_artifact"]))
    require(staged_binary["sha256"] == staged[0].get("sha256") == binary["sha256"],
            "staged binary differs from original compile bytes")
    require(not fields.get("link_environment"), "unrecognized linker environment")
    return dict(mode=mode, members=row["members"], environment=environment, source_files=source["files"],
                canonical_entries=entries, root_sha256=fields["root_sha256"],
                compiler_sha256=toolchain["compiler_sha256"], toolchain_locked=toolchain["locked"],
                toolchain=toolchain, command=command, compiler_workers=int(workers),
                receipt=reference(receipt_path), execution=reference(execution_path), log=reference(log),
                binary=binary, staged_binary=staged_binary)


def facts(root, producer):
    leg, result_path = Path(producer["leg"]).resolve(), Path(producer["result"]).resolve()
    require(result_path.is_relative_to(leg / "remote/leg_out"), "result is outside retained CPU leg")
    leg_path = leg / "leg.txt"
    lines = dict(line.split("=", 1) for line in leg_path.read_text().splitlines() if "=" in line)
    revision = lines.get("commit", "")
    require(common.HEX40.fullmatch(revision) and lines.get("dirty_tracked") == "0", "CPU leg source witness is missing/dirty")
    result = load(result_path)
    plan = result.get("plan", {})
    require(result.get("schema") == common.DEV_SCHEMA and result.get("state") in ("incomplete", "activated")
            and result.get("source_commit") in ("", revision) and plan.get("platform") == "linux"
            and plan.get("layout") == "development", "not a retained Linux development result")
    stage_path = leg / "remote/leg_out/native-stage-manifest.json"
    stage = load(stage_path)
    tree = common.Tree.at(root, revision)
    rows = [r for r in plan.get("groups", []) if r.get("selected") is True]
    require(rows and len({r["stem"] for r in rows}) == len(rows), "no unique selected groups")
    groups = {r["stem"]: _group(r, result, result_path, leg, tree, stage) for r in rows}
    return dict(producer=NAME, source_root=str(Path(root).resolve()), source_revision=revision,
                leg=reference(leg_path), result=reference(result_path), stage=reference(stage_path), groups=groups)


def collect(root, document, producer):
    found = facts(root, producer)
    units, flags, compilers, workers = {}, set(), set(), {}
    for stem, group in found["groups"].items():
        env = group["environment"]
        flags.add(json.dumps(common.derived_build_flags(env, ROUTING), sort_keys=True))
        compilers.add(group["compiler_sha256"])
        if group["mode"] == "host":
            prefix, specs = "linux-64/host/-/host/", ("compile-cpu", "compile-nvidia", "compile-amd")
        else:
            arch = env.get("MOJOLEARN_GPU_ARCHS", "")
            require(arch in ("sm_89", "sm_80", "gfx942"), "unsupported retained GPU target")
            vendor = "cuda" if arch.startswith("sm_") else "hip"
            prefix, specs = f"linux-64/{vendor}/{arch}/identical/", ("compile-nvidia" if vendor == "cuda" else "compile-amd",)
        for name in group["members"]:
            for spec in specs:
                key = prefix + name
                units[spec + "/" + key] = dict(group=stem, binding_key=key)
                workers[key] = str(group["compiler_workers"])
    require(len(flags) == len(compilers) == 1, "selected groups used different compiler/semantic flags")
    config = dict(release_modes=["identical"], build_flags=json.loads(flags.pop()),
                  dependency_context=dict(cpu_target=common.LINUX, compiler_sha256={common.LINUX: compilers.pop()},
                                          selected_binding_compile_jobs=workers))
    return dict(source_revision=found["source_revision"], facts=found, units=units, configuration=config,
                spec_ids=sorted({rid.split("/", 1)[0] for rid in units}))


def check_unit(requirement_id, wanted, found, unit):
    require(supports_requirement(wanted), "unsupported Linux compile requirement")
    binding, declared = wanted["inputs"]["binding"], wanted["inputs"]["unit"]
    require(common.binding_key(binding) == unit["binding_key"] and requirement_id.endswith("/" + unit["binding_key"]),
            "admitted unit names another binding")
    group = found["groups"][unit["group"]]
    require(binding["name"] in group["members"] and binding["tier"] == group["mode"], "binding not compiled in this group")
    require(binding.get("toolchain") == group["toolchain_locked"]
            and declared["compiler_sha256"] == group["compiler_sha256"], "compiler identity differs")
    # Keep the actual worker setting in the selected binding's fingerprint.
    # Compiler codegen partitioning is not assumed byte-neutral across -j values.
    flags = dict(binding.get("flags") or {})
    environment = dict(group["environment"])
    common.check_flags(flags, environment)
    arch = binding.get("arch")
    expected_format = "ptx" if arch == "sm_80" else "native"
    require(environment.get("MOJOLEARN_CUDA_CODE_FORMAT", "native") == expected_format,
            "CUDA code format differs (native does not qualify PTX)")
    require(declared["build_flags"] == common.derived_build_flags(environment, ROUTING), "semantic build flags differ")
    tree = common.Tree.at(found["source_root"], found["source_revision"])
    for rel in common.require_closure(tree, wanted):
        if rel.endswith(".mojo"):
            actual = group["source_files"].get(rel)
            if actual is None:
                actual = group["canonical_entries"].get(rel)
            require(actual is not None and actual == tree.sha256(rel), "required source was not compiled: " + rel)
    return common.coverage_rows(wanted), [found["leg"], found["result"], found["stage"],
        *[group[k] for k in ("receipt", "execution", "log", "binary", "staged_binary")]]


ADAPTERS = {NAME: common._Adapter(collect, facts, check_unit)}
