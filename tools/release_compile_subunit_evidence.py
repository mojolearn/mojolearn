"""Retained compile sub-units (runtime libraries, framework adapters) for the ledger.

Used by tools/release_ledger_admission.py; never invokes a compiler. Two actual
producers are recognized, each bound to its own retained bytes:

  linux-release-leg   one retained Linux release leg (tools/release_linux_build.sh,
                      packaging/linux/build_sets.sh): the leg directory
                      ``legs/<vendor>-<arch>`` beside its ``.provenance.json``
                      launch record and ``.exit``. It admits
                        compile-<vendor>/runtime        the staged .libs (MAX runtime
                                                        and the portable math helper)
                        compile-cpu/runtime             from the host leg (cuda/sm_89)
                        compile-<vendor>/adapters/<arch> the native adapter library,
                                                        Torch registration and MAX
                                                        operation package
                      The PTX leg (cuda/sm_80) owns neither.
  apple-pack          a tools/apple_retained_groups.py pack directory (pack-proof.json,
                      launch.json, cache-provenance/): the unchanged Mac packer at
                      the admitted, clean source revision. It admits
                        compile-apple/osx-arm64/<tier>/...  every binding of a group
                                                        the packer compiled in this run
                                                        (host, and identical groups whose
                                                        retained placement was refused);
                                                        retained placements and cache hits
                                                        are not compilations and are left
                                                        to apple-retained-device
                        compile-apple/adapters/apple    the shipped adapters
                      Each compiled group's bincache key must reproduce from the
                      tested revision's tracked bytes (whole-tree scope), name the
                      admitted checkout and Apple toolchain, and carry the binding's
                      flags; its membership and bytes are read from the exact wheel.

Each unit must match the requirement's toolchain, compiler executable (Linux),
flags, target and source bytes at the producer's own tested revision:

  * the leg compiled the tested revision: launch, capture and proof name the
    same commit, the tooling checkout was that commit and clean with no overlay,
    and every compiled-source inventory entry equals the tracked bytes there;
  * runtime: the launch record's runtime identity digest equals the
    reconstructed runtime identity; the portable math helper was built from the
    tracked source/constants; every staged library equals the set manifest,
    which the complete build proof binds;
  * adapters: the box-captured identity of every device binding of the set
    agrees with the requirement's invocation environment; the adapter manifest
    names the set, the revision and the source bytes; the recorded compiler
    invocations carry the requirement's target/flags; the Torch build is the
    pinned one; every shipped file equals the proof/manifest bytes.

Limitation (shared with the admitted device bindings): an identity's ``image``
is the box-image pin read from the tested tree, not a capture of the image the
box ran. The image the CPU build box actually ran (its create request) is
retained as evidence, not compared. The C++ compiler executable is not captured
by the producers; its only pin is that image.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from release_ledger_admission import require, load, reference

TOOL = "tools/release_compile_subunit_evidence.py"
BASE_COVERAGE = "all selected native bindings and required enabled/disabled switches"
RUNTIME_COVERAGE = [BASE_COVERAGE, "native-runtime-libraries"]
ADAPTER_COVERAGE = [BASE_COVERAGE, "native-adapter-library", "torch-registration", "max-operation-package"]
LINUX, MAC = "linux-64", "osx-arm64"
SPEC_VENDOR = {"compile-nvidia": "cuda", "compile-amd": "hip"}
HOST_LEG = ("cuda", "sm_89")  # owns the CPU column's shared runtime, as for host bindings
PTX_ARCH = "sm_80"
ADAPTER_FILES = {"native": "libmojolearn_adapter.so", "torch": "mojolearn_torch.so", "max": "mojolearn_ops.mojoc"}
ADAPTER_SCHEMA = "mojolearn.adapters.v1"
PORTABLE_MATH = "packaging/portable_math"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _member(root, relative):
    name = PurePosixPath(relative)
    require(not name.is_absolute() and ".." not in name.parts and str(name) == relative and relative,
            "unsafe artifact member: " + str(relative))
    return Path(root).joinpath(*name.parts)


def _what(wanted):
    inputs = wanted.get("inputs", {})
    if "runtime" in inputs and "binding" not in inputs and "environment" not in inputs:
        return "runtime"
    if "environment" in inputs and "adapter_sources" in inputs and "binding" not in inputs:
        return "adapters"
    return None


def supports_requirement(wanted):
    inputs = wanted.get("inputs", {})
    unit = inputs.get("unit", {})
    what = _what(wanted)
    if (wanted.get("kind") != "compile" or unit.get("kind") != "compile" or what is None
            or wanted.get("missing_inputs") or unit.get("release_modes") != ["identical"]
            or unit.get("compile_variants") or inputs.get("configuration", {}).get("compile_variants")
            or unit.get("build_flags") != {}):
        return False
    vendor = unit.get("vendor")
    if what == "runtime":
        runtime = inputs["runtime"]
        return (wanted.get("coverage") == RUNTIME_COVERAGE and runtime.get("target") == LINUX
                and runtime.get("what") == "runtime" and vendor in ("cpu", "nvidia", "amd")
                and isinstance(unit.get("compiler_sha256"), str) and HEX64.fullmatch(unit["compiler_sha256"]) is not None)
    environment = inputs["environment"]
    if wanted.get("coverage") != ADAPTER_COVERAGE or not isinstance(inputs.get("adapter_sources"), dict):
        return False
    if environment.get("target") == LINUX:
        return (vendor in ("nvidia", "amd") and environment.get("vendor") == {"nvidia": "cuda", "amd": "hip"}[vendor]
                and environment.get("arch") not in (None, PTX_ARCH)
                and isinstance(unit.get("compiler_sha256"), str) and HEX64.fullmatch(unit["compiler_sha256"]) is not None)
    return (environment.get("target") == MAC and vendor == "apple" and environment.get("vendor") == "metal"
            and environment.get("arch") == "apple" and isinstance(environment.get("host_toolchain"), dict))


def _tree(root, revision):
    from release_retained_compile_evidence import Tree
    return Tree.at(root, revision)


def binding_key(identity):
    return "/".join(str(identity.get(k) or "-") for k in ("target", "vendor", "arch", "tier", "name"))


def coverage_rows(wanted):
    return [dict(id=name, status="passed", exit_code=0, expected=1, completed=1, failed=0, skipped=0)
            for name in wanted["coverage"]]


# ---------------------------------------------------------------- adapter manifest checks (both platforms)
def _manifest_files(manifest, actual):
    """{role: sha256} from the manifest, equal to the actual shipped {name: (sha256, bytes)}."""
    files = manifest.get("files")
    require(isinstance(files, dict) and set(files) == set(ADAPTER_FILES), "adapter manifest names other files")
    require(set(actual) == set(ADAPTER_FILES.values()), "shipped adapter directory holds other files")
    for role, name in ADAPTER_FILES.items():
        row = files[role]
        require(row.get("name") == name and (row.get("sha256"), row.get("bytes")) == tuple(actual[name]),
                "shipped adapter bytes differ from the manifest: " + name)
    return {name: actual[name][0] for name in ADAPTER_FILES.values()}


def _check_manifest(manifest, revision, vendor, arch, platform):
    require(isinstance(manifest, dict) and manifest.get("schema") == ADAPTER_SCHEMA and manifest.get("abi") == 1
            and manifest.get("vendor") == vendor and manifest.get("arch") == arch
            and manifest.get("platform") == platform and manifest.get("source_commit") == revision
            and manifest.get("numeric_mode") == "identical", "adapter manifest names another set or revision")
    commands = manifest.get("commands") or {}
    require(set(commands) == set(ADAPTER_FILES) and all(isinstance(v, list) for v in commands.values()),
            "adapter manifest lacks its compiler invocations")
    require(isinstance(manifest.get("sources"), dict) and manifest["sources"], "adapter manifest names no sources")


def _option(argv, name):
    values = [argv[i + 1] for i, word in enumerate(argv[:-1]) if word == name]
    require(len(values) <= 1, "compiler option repeated: " + name)
    return values[0] if values else None


def _check_invocations(manifest, flags, wanted_arch):
    """The recorded compiler invocations carry the requirement's target and flags."""
    commands = manifest["commands"]
    native = commands["native"]
    defines = [native[i + 1] for i, word in enumerate(native[:-1]) if word == "-D"]
    require("mojo" in native and "build" in native and "adapters/native/c_api.mojo" in native,
            "native adapter invocation differs")
    require("precompile" in commands["max"] and "adapters/max_ops" in commands["max"],
            "MAX operation package invocation differs")
    require("adapters/torch/registration.cpp" in commands["torch"], "Torch registration invocation differs")
    for key in flags:
        require(key in ("numeric_mode", "compile_jobs", "gpu_archs", "linux_cpu", "target_column", "package_byte_lm"),
                "unrecognized adapter environment flag: " + key)
    require(flags.get("numeric_mode") == "identical" and "MOJOLEARN_NUMERIC_IDENTICAL=1" in defines,
            "adapter numeric mode differs")
    require(_option(native, "-j") == flags.get("compile_jobs"), "adapter compile jobs differ")
    require(_option(native, "--target-accelerator") == flags.get("gpu_archs")
            and flags.get("gpu_archs") in (None, wanted_arch), "adapter target accelerator differs")
    if "linux_cpu" in flags:
        require(_option(native, "--target-cpu") == flags["linux_cpu"], "adapter target CPU differs")
    columns = sorted(d for d in defines if d.startswith("MOJOLEARN_COLUMN_"))
    expected = ["MOJOLEARN_COLUMN_" + flags["target_column"].upper()] if flags.get("target_column") else []
    require(columns == expected, "adapter target column differs")
    # package_byte_lm selects a packaged binding, not an adapter compile input.


def _check_versions(manifest, toolchain):
    packages = list(toolchain.get("packages") or [])
    conda = (manifest.get("mojo") or {}).get("conda")
    max_version = (manifest.get("max") or {}).get("version")
    require(conda and any(p.startswith("mojo-" + conda + "-") for p in packages)
            and max_version and any(p.startswith("max-" + max_version + "-") for p in packages),
            "adapter Mojo/MAX versions differ from the requirement toolchain")


def _check_torch(manifest, pins, vendor):
    pin = ((pins.get("torch") or {}).get("build") or {}).get(vendor) or {}
    torch = manifest.get("torch") or {}
    require(pin and manifest.get("torch_version_pin") == pin.get("version")
            and torch.get("series") == pins["torch"].get("series") and torch.get("backend") == pin.get("backend")
            and HEX64.fullmatch(str(torch.get("wheel_sha256")))
            and (pin.get("wheel_sha256") is None or torch["wheel_sha256"] == pin["wheel_sha256"]),
            "Torch build differs from the pinned build")
    require(manifest.get("requirements") == dict(torch=pins["torch"].get("requirement"),
                                                  max=(pins.get("max") or {}).get("requirement")),
            "adapter framework requirements differ from the pins")


def _check_sources(tree, wanted, manifest_sources, compiled):
    """Every requirement adapter source is bound to the bytes this producer compiled."""
    for rel, sha in manifest_sources.items():
        require(tree.sha256(rel) == sha, "adapter manifest source differs from the tested revision: " + rel)
    for rel, sha in wanted["inputs"]["adapter_sources"].items():
        require(sha is not None and tree.sha256(rel) == sha, "adapter source differs from the tested revision: " + rel)
        found = manifest_sources.get(rel, compiled.get(rel))
        require(found == sha, "adapter source is not bound to the compiled bytes: " + rel)


# ---------------------------------------------------------------- Linux retained release leg
def _leg_facts(root, producer):
    leg = Path(producer["leg"]).resolve()
    name = leg.name
    launch_ref = reference(leg.parent / (name + ".provenance.json"))
    exit_ref = reference(leg.parent / (name + ".exit"))
    launch = load(launch_ref["path"])
    require((leg.parent / (name + ".exit")).read_text().strip() == "0", "release leg did not exit zero")
    require(launch.get("schema") == "mojolearn.release-leg-provenance.v1" and launch.get("leg") == name
            and Path(launch.get("release_build", "")).resolve() == (leg / name / "release-build").resolve(),
            "release leg launch record names another leg")
    build = leg / name / "release-build" / "build"
    proof_ref, capture_ref = reference(build / "build-provenance.json"), reference(build / "compile-inputs.json")
    inventory_ref = reference(build / "source-inventory.json")
    proof, capture = load(proof_ref["path"]), load(capture_ref["path"])
    revision = proof.get("source_commit")
    require(isinstance(revision, str) and HEX40.fullmatch(revision)
            and capture.get("source_commit") == launch.get("source_commit") == launch.get("tooling_commit") == revision,
            "release leg source provenance differs")
    require(launch.get("tooling_dirty") == [] and launch.get("overlay") is None
            and not launch.get("route_overlay_env"), "release leg ran a dirty or overlaid tree")
    require(proof.get("schema") == "mojolearn.linux.build-provenance.v1" and proof.get("action") == "build"
            and proof.get("complete") is True and type(proof.get("build_exit")) is int and proof["build_exit"] == 0
            and not proof.get("layout_errors") and proof.get("release_modes") == ["identical"],
            "release build proof is incomplete or not an actual complete build")
    require(capture.get("schema") == "mojolearn.linux.release-compile-inputs.v1" and capture.get("phase") == "before-build"
            and proof.get("compile_inputs_sha256") == capture_ref["sha256"]
            and capture.get("source_inventory_sha256") == inventory_ref["sha256"],
            "build lacks its original before-build compiler/flag capture")
    vendor, arch = capture.get("vendor"), capture.get("arch")
    require((launch.get("vendor"), launch.get("arch")) == (vendor, arch) and vendor in ("cuda", "hip"),
            "build target differs from launch")
    inventory = proof.get("source_inventory")
    require(isinstance(inventory, list) and inventory and len(dict(inventory)) == len(inventory)
            and proof.get("source_sha256") == sha_bytes(json.dumps(inventory, separators=(",", ":")).encode())
            and capture.get("source_inventory_digest") == digest(inventory), "compiled source inventory differs")
    tree = _tree(root, revision)
    for rel, sha in inventory:
        require(tree.sha256(rel) == sha, "compiled source bytes differ from the tested revision: " + rel)
    toolchain = capture.get("toolchain") or {}
    require(toolchain.get("packages") and set(toolchain["packages"]) <= set(capture.get("installed_packages", []))
            and HEX64.fullmatch(str(capture.get("compiler_sha256"))), "actual compiler/package identity is missing")
    prefix = f"mojolearn/{vendor}/{arch}/"
    set_root = build / "sets" / vendor / arch
    manifest_ref = reference(set_root / "manifest.json")
    require(proof.get("set_manifests", {}).get(prefix + "manifest.json") == manifest_ref["sha256"],
            "build proof does not bind the retained set manifest")
    manifest = load(manifest_ref["path"])
    refs = [launch_ref, exit_ref, proof_ref, capture_ref, inventory_ref, manifest_ref]
    # runtime: every staged library, exactly the manifest's list
    staged = manifest.get("staged_libs")
    require(isinstance(staged, list) and staged, "set manifest stages no runtime libraries")
    libs = {}
    for row in staged:
        ref = reference(_member(set_root / ".libs", row.get("name", "")))
        require(ref["sha256"] == row.get("sha256") and Path(ref["path"]).stat().st_size == row.get("bytes"),
                "staged runtime library differs from the set manifest: " + str(row.get("name")))
        libs[row["name"]] = ref["sha256"]; refs.append(ref)
    require(sorted(p.name for p in (set_root / ".libs").iterdir()) == sorted(libs),
            "staged runtime directory holds other files")
    math = (manifest.get("portable_math") or {})
    math_build = math.get("build") or {}
    require(math.get("helper") in libs and math_build.get("output_sha256") == libs[math["helper"]],
            "portable math helper differs from its build record")
    runtime = dict(libs=libs, portable_math=dict(helper=math["helper"], source_sha256=math_build.get("source_sha256"),
                   constants_sha256=math_build.get("constants_sha256"), command=math_build.get("command")),
                   set_identity_runtime=(launch.get("set_identity") or {}).get("runtime"))
    # adapters: the proof's record, the shipped files and their manifest
    adapters = None
    record = (proof.get("adapters") or {}).get(prefix + "_adapters")
    if record is not None:
        directory = set_root / "_adapters"
        actual = {}
        for path in sorted(directory.iterdir()):
            ref = reference(path); refs.append(ref)
            actual[path.name] = (ref["sha256"], path.stat().st_size)
        manifest_sha = actual.pop("adapters.json", (None,))[0]
        require(record.get("manifest_sha256") == manifest_sha
                and record.get("files") == dict({n: s for n, (s, _) in actual.items()}, **{"adapters.json": manifest_sha}),
                "build proof does not bind the shipped adapter bytes")
        adapter_manifest = load(directory / "adapters.json")
        files = _manifest_files(adapter_manifest, actual)
        rows = {Path(r.get("path", "")).name: r.get("sha256") for r in manifest.get("adapters") or []}
        require(all(rows.get(n) == files[n] for n in rows) and rows,
                "set manifest adapter rows differ from the shipped bytes")
        adapters = dict(manifest=adapter_manifest, files=files)
    host_image = None
    request = leg / "LEG" / "create_request.json"
    if request.is_file():
        refs.append(reference(request))
        host_image = load(request).get("imageName")
    return dict(producer="linux-release-leg", source_root=str(Path(root).resolve()), source_revision=revision,
                leg=name, vendor=vendor, arch=arch, compiler_sha256=capture["compiler_sha256"], toolchain=toolchain,
                build_flags=capture.get("build_flags", {}), binding_identities=capture.get("binding_identities", {}),
                compiled_sources=dict(inventory), runtime=runtime, adapters=adapters,
                build_host_image=host_image, evidence=refs)


def _leg_units(facts):
    vendor, arch = facts["vendor"], facts["arch"]
    require(arch != PTX_ARCH, "the PTX leg owns no runtime or adapter qualification")
    spec = {"cuda": "compile-nvidia", "hip": "compile-amd"}[vendor]
    units = {spec + "/runtime": dict(what="runtime")}
    if (vendor, arch) == HOST_LEG:
        units["compile-cpu/runtime"] = dict(what="runtime")
    if facts["adapters"] is not None:
        units[spec + "/adapters/" + arch] = dict(what="adapters")
    return units


def _leg_collect(root, document, producer):
    facts = _leg_facts(root, producer)
    units = _leg_units(facts)
    configuration = dict(release_modes=["identical"], build_flags={},
                         dependency_context=dict(cpu_target=LINUX, compiler_sha256={LINUX: facts["compiler_sha256"]}))
    return dict(source_revision=facts["source_revision"], facts=facts, units=units, configuration=configuration,
                spec_ids=sorted({rid.split("/", 1)[0] for rid in units}))


def _leg_common(requirement_id, wanted, facts):
    from release_compile_evidence import _flags
    require(supports_requirement(wanted), "unsupported compile target/configuration")
    spec = requirement_id.split("/", 1)[0]
    unit = wanted["inputs"]["unit"]
    if spec == "compile-cpu":
        require((facts["vendor"], facts["arch"]) == HOST_LEG and unit["vendor"] == "cpu",
                "only the host leg qualifies the CPU column")
    else:
        require(SPEC_VENDOR.get(spec) == facts["vendor"] and facts["arch"] != PTX_ARCH,
                "release leg belongs to another vendor")
    require(unit["compiler_sha256"] == facts["compiler_sha256"], "actual compiler executable bytes differ")
    _flags(facts["build_flags"], unit["build_flags"], dict(vendor=facts["vendor"], arch=facts["arch"]))


def _leg_check(requirement_id, wanted, facts, unit):
    import release_reuse
    _leg_common(requirement_id, wanted, facts)
    what = unit["what"]
    require(_what(wanted) == what, "admitted unit names another sub-unit")
    tree = _tree(facts["source_root"], facts["source_revision"])
    inputs = wanted["inputs"]
    if what == "runtime":
        require(requirement_id.endswith("/runtime"), "admitted unit names another sub-unit")
        runtime = inputs["runtime"]
        require(runtime.get("toolchain") == facts["toolchain"], "actual compiler toolchain differs from the runtime inputs")
        require(facts["runtime"]["set_identity_runtime"] == release_reuse.digest_of(runtime),
                "release leg was launched for another runtime identity")
        math = facts["runtime"]["portable_math"]
        require(math["source_sha256"] == tree.sha256(PORTABLE_MATH + "/portable_math.c")
                and math["constants_sha256"] == tree.sha256(PORTABLE_MATH + "/powers_of_ten.h"),
                "portable math helper was not built from the tested source bytes")
        return coverage_rows(wanted), list(facts["evidence"])
    environment = inputs["environment"]
    arch = environment["arch"]
    require(requirement_id.endswith("/adapters/" + arch) and arch == facts["arch"]
            and environment["vendor"] == facts["vendor"] and facts["adapters"] is not None,
            "admitted unit names another adapter set")
    fields = ("target", "vendor", "arch", "toolchain", "flags", "image")
    wanted_env = {k: environment.get(k) for k in fields}
    devices = [{k: ident.get(k) for k in fields} for ident in facts["binding_identities"].values()
               if ident.get("tier") != "host" and (ident.get("vendor"), ident.get("arch")) == (facts["vendor"], arch)]
    require(devices and all(d == wanted_env for d in devices),
            "box-captured device identities differ from the adapter invocation environment")
    require(environment["toolchain"] == facts["toolchain"], "actual compiler toolchain differs from the adapter inputs")
    manifest = facts["adapters"]["manifest"]
    _check_manifest(manifest, facts["source_revision"], facts["vendor"], arch, "linux")
    _check_invocations(manifest, environment["flags"], arch)
    _check_versions(manifest, environment["toolchain"])
    _check_torch(manifest, json.loads(tree.text("packaging/adapters/pins.json")), facts["vendor"])
    _check_sources(tree, wanted, manifest["sources"], facts["compiled_sources"])
    return coverage_rows(wanted), list(facts["evidence"])


# ---------------------------------------------------------------- Apple pack (bindings + adapters/apple)
PACK_ROUTING = {"MOJOLEARN_MERGED_BINDINGS": "1", "MOJOLEARN_REUSE_DIR": "", "MOJOLEARN_REUSE_PLAN": ""}
TIER_DIRS = {"host": "host", "identical": "identical"}


def _tree_digest(tree, script, generated=None):
    """bincache.source_digest's whole-tree scope over the tracked bytes of the tested revision.

    GENERATED ({path: sha256}) adds gitignored build-time sources the walk saw in
    the checkout (the tokenizer unicode table, written by the first tokenizer
    host compile and walked by every later tree-scope key); each sha256 must be
    the exact output of its tracked generator (_generated_shas)."""
    import bincache
    skip = {".pixi", ".git", "__pycache__"}
    rels = {r for r in tree.files if r.endswith(".mojo") and not skip & set(PurePosixPath(r).parts[:-1])}
    rels |= {r for r in (script, "bindings/build_host_family.sh", "pixi.toml", "pixi.lock") if r in tree.files}
    generated = dict(generated or {})
    require(not set(generated) & rels, "a generated source is tracked at the tested revision")
    value = hashlib.sha256()
    for rel in sorted(rels | set(generated), key=lambda name: PurePosixPath(name).parts):
        found = generated.get(rel) or (bincache.sha256_bytes(bincache.pixi_toml_build_part(tree.text(rel)).encode())
                                       if rel == "pixi.toml" else tree.sha256(rel))
        value.update(rel.encode() + b"\0" + found.encode() + b"\n")
    return dict(scope="tree", digest=value.hexdigest(), files=len(rels) + len(generated))


def _generated_shas(tree, checkout):
    """{path: sha256} of every build-time generated source, as the tested
    revision's tracked generator writes it (generated_sources.expected_sha:
    regenerated here, or the checkout's copy verified byte for byte). A
    revision without the generator generates nothing."""
    import generated_sources as gs
    return {rel: gs.expected_sha(rel, {name: tree.text(name).encode() for name in gs.GENERATED[rel]},
                                 [Path(checkout) / rel])
            for rel in sorted(gs.GENERATED) if all(name in tree.files for name in gs.GENERATED[rel])}


def _pack_groups(directory, admission, tree, archive):
    """Groups the production packer actually compiled in this run (never cache hits or retained placements)."""
    import bincache
    provenance = directory / "cache-provenance"
    rows_ref = reference(provenance / "provenance.tsv")
    refs, groups, generated = [rows_ref], {}, None
    for line in Path(rows_ref["path"]).read_text().splitlines():
        cells = line.split("\t")
        require(len(cells) == 6, "malformed packer cache provenance row")
        _, script, outcome, key, _, files = cells
        require("build-failed" not in outcome and "declared-output-not-written" not in outcome,
                "Apple pack has a failed or unwritten build: " + outcome)
        if not ("+built-cached" in outcome or "+built-not-cached:" in outcome):
            continue  # hits and retained placements are not compilations in this run
        key_ref = reference(provenance / "keys" / (key + ".json")); refs.append(key_ref)
        fields = load(key_ref["path"])
        local = fields.get("local") or {}
        outputs = local.get("outputs") or []
        require(bincache.key_of(fields) == key and fields.get("schema") == "mojolearn-bincache-v1"
                and fields.get("script") == script and fields.get("numeric_mode") == "identical"
                and (fields.get("os") or {}).get("id") == "darwin"
                and Path(fields.get("repo_path", "")).resolve() == Path(admission["source_root"]).resolve()
                and local.get("host_toolchain") == admission["host_toolchain"] and len(outputs) == 1,
                "packer compile key differs from the admitted source/toolchain: " + key)
        # Keys taken before the first tokenizer host compile walked the tracked
        # tree alone; later ones also walked the generated unicode table.
        if fields.get("source") != _tree_digest(tree, script):
            if generated is None:
                generated = _generated_shas(tree, admission["source_root"])
            require(fields.get("source") == _tree_digest(tree, script, generated),
                    "packer compiled source does not reproduce from the tested revision: " + key)
        output = PurePosixPath(outputs[0])
        require(output.parts[:2] == ("python", "mojolearn") and len(output.parts) == 4
                and output.parts[2] in TIER_DIRS.values() and output.suffix == ".so",
                "packer compile output is not a release binding group: " + outputs[0])
        tier, stem = output.parts[2], output.stem
        raw = dict(item.split("=", 1) for item in files.split(" ") if "=" in item)
        require(set(raw) == {outputs[0]} and HEX64.fullmatch(raw[outputs[0]]), "packer output digest is missing")
        index = json.loads(archive.read(f"mojolearn/{tier}/_mojolearn_merged.json"))
        members = (index.get("files") or {}).get(stem + ".so")
        member = f"mojolearn/{tier}/{stem}.so"
        require(members and member in archive.namelist() and stem not in groups,
                "Mac wheel lacks the compiled group or its membership: " + stem)
        groups[stem] = dict(tier=tier, members=sorted(members), key=key, toolchain=fields.get("toolchain"),
                            host_toolchain=local["host_toolchain"], build_env=fields.get("build_env") or {},
                            raw_sha256=raw[outputs[0]], wheel_sha256=sha_bytes(archive.read(member)), key_ref=key_ref)
    return groups, refs


def _pack_facts(root, producer):
    import apple_retained_groups as arg
    import release_host_composition as witness
    directory = Path(producer["pack"]).resolve()
    proof_ref = reference(directory / "pack-proof.json")
    proof = load(proof_ref["path"])
    require(proof.get("schema") == arg.PACK_SCHEMA and proof.get("complete") is True
            and type(proof.get("pack_exit")) is int and proof["pack_exit"] == 0
            and (directory / "pack.rc").read_text().strip() == "0", "Apple pack did not complete")
    launch_ref = reference(directory / "launch.json")
    require((proof.get("launch") or {}).get("sha256") == launch_ref["sha256"], "Apple pack launch record differs")
    launch = witness.decoded(proof["launch"])
    admission = witness.decoded(proof.get("admission"))
    revision = admission.get("admitted_source_commit")
    require(isinstance(revision, str) and HEX40.fullmatch(revision)
            and launch.get("source_commit") == proof.get("source_commit") == revision
            and launch.get("admission") == proof.get("admission")
            and admission.get("schema") == arg.SCHEMA and admission.get("complete") is True
            and launch.get("compiler_sha256") == admission.get("compiler_sha256")
            and HEX64.fullmatch(str(admission.get("compiler_sha256")))
            and isinstance(admission.get("host_toolchain"), dict), "Apple pack admission/launch differ")
    wheel_ref = proof.get("wheel") or {}
    require(reference(wheel_ref.get("path", directory / "missing")) == wheel_ref, "retained Mac wheel bytes differ")
    tree = _tree(root, revision)
    dependencies = dict(admission.get("admitted_dependency_files") or {})
    for rel, sha in dependencies.items():
        require(tree.sha256(rel) == sha, "admitted dependency differs from the tested revision: " + rel)
    actual = {}
    with zipfile.ZipFile(wheel_ref["path"]) as archive:
        names = [n for n in archive.namelist() if n.startswith("mojolearn/_adapters/") and not n.endswith("/")]
        for name in names:
            data = archive.read(name)
            actual[name.rsplit("/", 1)[1]] = (sha_bytes(data), len(data))
        require("mojolearn/_adapters/adapters.json" in names, "Mac wheel ships no adapters")
        manifest = json.loads(archive.read("mojolearn/_adapters/adapters.json"))
        groups, refs = _pack_groups(directory, admission, tree, archive)
    actual.pop("adapters.json")
    files = _manifest_files(manifest, actual)
    return dict(producer="apple-pack", source_root=str(Path(root).resolve()), source_revision=revision,
                compiler_sha256=admission["compiler_sha256"], host_toolchain=admission["host_toolchain"],
                compiled_sources=dependencies, manifest=manifest, files=files, groups=groups,
                evidence=[proof_ref, launch_ref, dict(path=str(Path(wheel_ref["path"]).resolve()),
                                                      sha256=wheel_ref["sha256"])] + refs)


def _pack_collect(root, document, producer):
    facts = _pack_facts(root, producer)
    cpu_target = (document or {}).get("configuration", {}).get("dependency_context", {}).get("cpu_target")
    units = {"compile-apple/adapters/apple": dict(what="adapters")}
    for stem, group in facts["groups"].items():
        for name in group["members"]:
            if group["tier"] == "host":
                key = f"{MAC}/host/-/host/{name}"
                specs = ("compile-apple", "compile-cpu") if cpu_target == MAC else ("compile-apple",)
            else:
                key = f"{MAC}/metal/apple/{group['tier']}/{name}"
                specs = ("compile-apple",)
            for spec in specs:
                units[spec + "/" + key] = dict(what="binding", group=stem, binding_key=key)
    configuration = dict(release_modes=["identical"], build_flags={},
                         dependency_context=dict(apple_toolchain=facts["host_toolchain"], cpu_target=MAC))
    return dict(source_revision=facts["source_revision"], facts=facts, units=units,
                configuration=configuration, spec_ids=sorted({rid.split("/", 1)[0] for rid in units}))


def _pack_binding_check(requirement_id, wanted, facts, unit):
    import release_retained_compile_evidence as compiled
    require(compiled.supports_requirement(wanted), "unsupported compile target/configuration")
    binding, declared = wanted["inputs"]["binding"], wanted["inputs"]["unit"]
    key = unit["binding_key"]
    require(binding_key(binding) == key and requirement_id.endswith("/" + key) and binding.get("target") == MAC,
            "admitted unit names another binding")
    group = facts["groups"][unit["group"]]
    require(binding["name"] in group["members"] and binding["tier"] == group["tier"],
            "compiled group does not contain this binding/tier")
    require(binding.get("toolchain") == group["toolchain"], "compiler packages differ from the binding inputs")
    require(binding.get("host_toolchain") == group["host_toolchain"] == facts["host_toolchain"],
            "Apple host toolchain differs")
    environment = dict(group["build_env"])
    environment.pop("MOJOLEARN_MERGED_KEY", None)  # digest of the merged inputs, all in the reproduced tree
    for name, value in PACK_ROUTING.items():
        require(environment.pop(name, value) == value, "packer routing flag differs: " + name)
    compiled.check_flags(binding.get("flags") or {}, environment)
    require(set(environment) <= set(compiled.FLAG_ENV.values()) and declared["build_flags"] == {},
            "declared build flags differ from the packer invocation")
    tree = _tree(facts["source_root"], facts["source_revision"])
    compiled.require_closure(tree, wanted)
    return coverage_rows(wanted), list(facts["evidence"][:3]) + [group["key_ref"]]


def _pack_check(requirement_id, wanted, facts, unit):
    if unit["what"] == "binding":
        return _pack_binding_check(requirement_id, wanted, facts, unit)
    require(supports_requirement(wanted) and _what(wanted) == "adapters"
            and requirement_id == "compile-apple/adapters/apple", "unsupported compile target/configuration")
    environment = wanted["inputs"]["environment"]
    require(environment["host_toolchain"] == facts["host_toolchain"], "Apple host toolchain differs")
    tree = _tree(facts["source_root"], facts["source_revision"])
    manifest = facts["manifest"]
    _check_manifest(manifest, facts["source_revision"], "metal", "apple", "macos")
    _check_invocations(manifest, environment["flags"], "apple")
    _check_versions(manifest, environment["toolchain"])
    _check_torch(manifest, json.loads(tree.text("packaging/adapters/pins.json")), "metal")
    _check_sources(tree, wanted, manifest["sources"], facts["compiled_sources"])
    return coverage_rows(wanted), list(facts["evidence"][:3])


class _Adapter:
    def __init__(self, collect, facts, check):
        self.collect, self._facts, self.check_unit = collect, facts, check

    def verify(self, report):
        return self._facts(report["facts"]["source_root"], report["producer"])


ADAPTERS = {
    "linux-release-leg": _Adapter(_leg_collect, _leg_facts, _leg_check),
    "apple-pack": _Adapter(_pack_collect, _pack_facts, _pack_check),
}
