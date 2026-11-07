"""The macOS release build (packaging/macos/build_release_wheel.sh) as a retained compile producer.

Used by tools/release_ledger_admission.py (adapter ``macos-release-build``); never
invokes a compiler. tools/release.py step_macos_build runs the release build with
MOJOLEARN_BINCACHE_DIR set, so every merged group compile goes through
``tools/bincache.py build``: its full key fields land in <cache>/provenance/keys,
its raw compiled bytes in <cache>/<key>.tar.gz and one row per compile in
<cache>/provenance/provenance.tsv. The build itself never writes the ledger; this
adapter admits its compile-apple rows afterwards from those retained bytes:

    --producer wheel=REL/<commit>/macos/mojolearn-<v>-...-macosx_*.whl   (the shipped wheel)
    --producer log=REL/<commit>/macos-build.log                         (the build's own log)
    --producer cache=<MOJOLEARN_BINCACHE_DIR of that build>

The provenance chain, per merged group of the build log. A group is either
compiled in this build (``BINCACHE miss+built-cached``, "built") or placed from
an archive an earlier compile of the same key wrote (``BINCACHE hit``, "placed";
with lane post-one-mac-cache the development build of the frozen commit compiles
every macOS group into the one Mac cache and the release places them). One
compile per freeze, and a verdict is reused for the same bytes: a placed archive
whose key reproduces from the frozen revision is that group's compile evidence,
equal to a compile in this build. Any other outcome refuses.

  source   closure-v2 keys (tools/bincache.py KEY_RULE): the key's source closure,
           executed inputs and merged membership (bincache.revision_closure)
           reproduce from the wheel's own commit, and its build_env is exactly
           what those inputs read. Older keys (no key_rule, built only): the
           whole-tree source digest reproduces from the tracked bytes of the
           wheel's commit (plus the gitignored unicode table, as its tracked
           generator writes it); the builder inputs digest and
           MOJOLEARN_MERGED_KEY (merged_groups.json + merged_bindings.py)
           reproduce from the same commit;
  placed   additionally: the release cache's provenance.tsv has the hit row
           naming the archive's bytes; the archive's manifest names its original
           compile (source_commit, built, build_seconds, builder) and the
           cache's placements.jsonl line for the hit names the same compile; the
           key's closure also reproduces at that source_commit (so the revision
           it names is the revision its bytes were compiled from). A hit of an
           older key, or one missing any of these, refuses;
  compiler the key names the requirement's Mojo/MAX packages, pixi.lock and Apple
           toolchain, and carries exactly the binding's flags;
  bytes    the archive verifies against its key (bincache.verify_archive), and the
           shipped wheel member equals the raw compiled bytes after the tested
           revision's own packaging transforms, replayed here on a scratch copy
           (stage_dylibs.py: add the .dylibs rpath, delete absolute build rpaths,
           ad-hoc sign; shrink_wheel.py: strip -x, ad-hoc sign). install_name_tool,
           strip and codesign are not compilers; any byte difference refuses.

compile-apple/adapters/apple is admitted on the apple-pack contract
(release_compile_subunit_evidence): the shipped adapters.json names the commit,
sources, compiler invocations and shipped bytes; Mojo sources outside adapters/
are bound by this build's whole-tree compile keys. Limitation: the adapter's raw
(pre-strip) bytes are not retained, so its unstaged -> shipped step is the
manifest's own record, not a replay.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

from release_ledger_admission import require, load, reference

TOOL = "tools/release_requirements_macos_build.py"
NAME = "macos-release-build"
MAC = "osx-arm64"
SCRIPT = "bindings/merged_build.sh"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
LOG_LINE = re.compile(r"^BINCACHE (\S+) (\S+) key=([0-9a-f]{16}|-) seconds=[0-9.]+$")
#: bincache.seed_archive's note before a lookup; the outcome line follows it.
SEEDED_LINE = re.compile(r"^BINCACHE seeded \S+ key=[0-9a-f]{16} from .+$")
#: Build log outcome -> how the group's compile evidence came to be.
OUTCOMES = {"miss+built-cached": "built", "hit": "placed"}
BUILDING = re.compile(r"^== building ([0-9]+) merged files")
KINDS = {"identical": "gpu", "host": "host"}
# Merged-build routing, fixed by build_release_wheel.sh: merged layout on, no reuse plan.
ROUTING = {"MOJOLEARN_MERGED_BINDINGS": "1", "MOJOLEARN_REUSE_DIR": "", "MOJOLEARN_REUSE_PLAN": ""}
SHRINK_STEPS = ('pathlib.Path(tmp) / "b.so"', '["strip", "-x", str(path)]', '["codesign", "-f", "-s", "-", str(path)]')


def supports_requirement(wanted):
    import release_retained_compile_evidence as compiled
    import release_compile_subunit_evidence as sub
    binding = wanted.get("inputs", {}).get("binding")
    if isinstance(binding, dict):
        return binding.get("target") == MAC and compiled.supports_requirement(wanted)
    return sub.supports_requirement(wanted) and sub._what(wanted) == "adapters" \
        and wanted["inputs"]["environment"].get("target") == MAC


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _tree(root, revision):
    from release_retained_compile_evidence import Tree
    return Tree.at(root, revision)


def _merged_key(tree):
    """build_release_wheel.sh: cat bindings/merged_groups.json tools/merged_bindings.py | shasum -a 256."""
    return sha_bytes(tree.text("bindings/merged_groups.json").encode() + tree.text("tools/merged_bindings.py").encode())


def _inputs_digest(tree, rels):
    """bincache.local_inputs' digest over the tracked bytes of the tested revision."""
    value = hashlib.sha256()
    for rel in rels:
        found = tree.sha256(rel)
        require(found is not None, "build input is not tracked at the tested revision: " + rel)
        value.update(rel.encode() + b"\0" + found.encode() + b"\n")
    return value.hexdigest()


def _log_compiles(log_path):
    """[(outcome, key prefix)] of every bincache-routed compile, and the planned merged-file count."""
    compiles, planned = [], None
    with open(log_path, errors="replace") as stream:
        for line in stream:
            line = line.rstrip("\n")
            if SEEDED_LINE.match(line):
                continue
            if line.startswith("BINCACHE "):
                m = LOG_LINE.match(line)
                require(m is not None and m.group(2) == SCRIPT, "unrecognized bincache line in the build log")
                compiles.append((m.group(1), m.group(3)))
            elif planned is None and BUILDING.match(line):
                planned = int(BUILDING.match(line).group(1))
    require(planned is not None and planned == len(compiles),
            "the build log does not record one bincache-routed compile per planned merged file")
    return compiles


def _placements(path):
    """{key: [placement line]} of the cache's placements.jsonl (one line per hit), {} without one."""
    rows = {}
    if not Path(path).is_file():
        return rows
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                row = json.loads(line)
            except ValueError:
                raise ValueError("malformed bincache placement record")
            require(isinstance(row, dict), "malformed bincache placement record")
            rows.setdefault(row.get("key"), []).append(row)
    return rows


def _closure_reproduces(root, revision, fields):
    """True when a closure-v2 key's tree-derived fields (source closure, executed
    inputs, merged membership and the build_env those inputs read) are what
    tools/bincache.py computes at REVISION's committed bytes."""
    import bincache
    env = {"MOJOLEARN_NUMERIC_MODE": fields.get("numeric_mode") or "identical",
           "MOJOLEARN_PACKAGE_BYTE_LM": (fields.get("build_env") or {}).get("MOJOLEARN_PACKAGE_BYTE_LM", "1")}
    try:
        want = bincache.revision_closure(root, revision, fields["script"], fields.get("args") or [], env,
                                         plat="darwin")
    except bincache.UnkeyableBuild as error:
        raise ValueError("the compile key cannot be recomputed at %s: %s" % (revision, error))
    return (all(fields.get(k) == want[k] for k in ("source", "inputs", "merged"))
            and fields.get("build_env") == bincache.keyed_env(fields.get("build_env") or {}, want["readers"]))


def _provenance_rows(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        cells = line.split("\t")
        require(len(cells) == 6, "malformed bincache provenance row")
        rows.setdefault(cells[3], []).append(cells)
    return rows


def _stage_module(tree, scratch):
    """The tested revision's own packaging/macos/stage_dylibs.py."""
    path = Path(scratch) / "stage_dylibs_tested.py"
    path.write_text(tree.text("packaging/macos/stage_dylibs.py"))
    spec = importlib.util.spec_from_file_location("stage_dylibs_tested", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay(raw, tier, stem, stage, scratch):
    """(staged sha256, shipped sha256): the release build's packaging transforms on a copy of RAW.

    stage_dylibs.main for a fresh extension one directory below the package root,
    then shrink_wheel.shrink's per-binding step (the file is named b.so there,
    which is the ad-hoc signature's identifier)."""
    work = Path(tempfile.mkdtemp(dir=scratch))
    ext = work / tier / (stem + ".so")
    ext.parent.mkdir(parents=True)
    ext.write_bytes(raw)
    stage.add_rpath(ext, "@loader_path/../.dylibs")
    stage.strip_build_rpaths(ext)
    stage.sign(ext)
    staged = ext.read_bytes()
    shrunk = work / "shrink" / "b.so"
    shrunk.parent.mkdir()
    shrunk.write_bytes(staged)
    subprocess.run(["strip", "-x", str(shrunk)], check=True, capture_output=True)
    subprocess.run(["codesign", "-f", "-s", "-", str(shrunk)], check=True, capture_output=True)
    return sha_bytes(staged), sha_bytes(shrunk.read_bytes())


def _group(key, outcome, rows, placements, cache, tree, archive, generated, merged_key, stage, scratch):
    import bincache
    import release_compile_subunit_evidence as sub
    key_ref = reference(cache / "provenance" / "keys" / (key + ".json"))
    fields = load(key_ref["path"])
    local = fields.get("local") or {}
    outputs = local.get("outputs") or []
    require(bincache.key_of(fields) == key and fields.get("schema") == bincache.SCHEMA
            and fields.get("script") == SCRIPT and fields.get("numeric_mode") == "identical"
            and fields.get("image") == "local" and (fields.get("os") or {}).get("id") == "darwin"
            and isinstance(local.get("host_toolchain"), dict) and len(outputs) == 1,
            "release compile key is not a local macOS merged build: " + key)
    output = PurePosixPath(outputs[0])
    require(output.parts[:2] == ("python", "mojolearn") and len(output.parts) == 4
            and output.parts[2] in KINDS and output.suffix == ".so",
            "release compile output is not a merged binding group: " + outputs[0])
    tier, stem = output.parts[2], output.stem
    require(fields.get("args") == [KINDS[tier], "--group", stem], "release compile arguments differ: " + key)
    how = OUTCOMES[outcome]
    environment = fields.get("build_env") or {}
    rule = fields.get("key_rule")
    if rule == bincache.KEY_RULE:
        # closure-v2: the group's own closure, inputs and membership at the wheel's commit.
        require(_closure_reproduces(tree.root, tree.revision, fields),
                "compiled source does not reproduce from the tested revision: " + key)
        require("MOJOLEARN_MERGED_KEY" not in environment,
                "merged group plan differs from the tested revision: " + key)
    else:
        require(rule is None and how == "built",
                "a placed group's compile key is not closure-v2, so its compile cannot be named: " + key)
        # Keys taken before the first tokenizer host compile walk the tracked tree alone.
        source = fields.get("source")
        require(source == sub._tree_digest(tree, SCRIPT) or source == sub._tree_digest(tree, SCRIPT, generated),
                "compiled source does not reproduce from the tested revision: " + key)
        inputs = local.get("inputs") or {}
        require(inputs.get("digest") == _inputs_digest(tree, inputs.get("files") or []),
                "build scripts/generators differ from the tested revision: " + key)
        require(environment.get("MOJOLEARN_MERGED_KEY") == merged_key,
                "merged group plan differs from the tested revision: " + key)
    toolchain = fields.get("toolchain") or {}
    require(toolchain.get("platform") == MAC and toolchain.get("pixi_lock_sha256") == tree.sha256("pixi.lock"),
            "compiled lock differs from the tested revision: " + key)
    # The log line says what this build did with the key; the cache rows must agree.
    built = [r for r in rows.get(key, []) if r[1] == SCRIPT and r[2] == outcome]
    require(len(built) >= 1, ("bincache provenance has no compile row for: " if how == "built"
                              else "bincache provenance has no placement row for: ") + key)
    arc = cache / (key + ".tar.gz")
    try:
        manifest, blobs = bincache.verify_archive(str(arc), key, fields)
    except bincache.Reject as error:
        raise ValueError("retained compile archive does not verify: %s: %s" % (key, error))
    require(sorted(blobs) == [outputs[0]], "archive files differ from the declared output: " + key)
    raw = blobs[outputs[0]]
    raw_sha = sha_bytes(raw)
    require(all(r[5] == f"{outputs[0]}={raw_sha}" for r in built),
            "bincache provenance row names other bytes: " + key)
    origin = manifest.get("source_commit") or ""
    compile_ = dict(outcome=how, source_commit=origin or tree.revision,
                    compiled_at=manifest.get("built") or built[0][0],
                    build_seconds=manifest.get("build_seconds", float(built[0][4])),
                    builder=manifest.get("builder") or {})
    if how == "built":
        require(origin in ("", tree.revision), "the archive compiled in this build names another revision: " + key)
    else:
        # The original compile: named inside the verified archive, echoed by the
        # hit's placement record, and reproducing the key at the revision it names.
        require(HEX40.fullmatch(origin) is not None and bool(manifest.get("built"))
                and isinstance(manifest.get("build_seconds"), (int, float))
                and isinstance(manifest.get("builder"), dict) and bool(manifest["builder"]),
                "the placed archive does not name its original compile (revision, date, seconds, builder): " + key)
        lines = [p for p in placements.get(key, []) if p.get("script") == SCRIPT and p.get("outcome") == "hit"]
        require(len(lines) >= 1, "bincache placement record is missing for: " + key)
        require(all(p.get("archive_source_commit") == origin and p.get("archive_built") == manifest["built"]
                    and p.get("archive_build_seconds") == manifest["build_seconds"]
                    and p.get("archive_builder") == manifest["builder"]
                    and p.get("files") == [dict(path=outputs[0], sha256=raw_sha)] for p in lines),
                "bincache placement record names another compile or other bytes: " + key)
        require(origin == tree.revision or _closure_reproduces(tree.root, origin, fields),
                "the placed archive's key does not reproduce at the revision it names: " + key)
    member = f"mojolearn/{tier}/{stem}.so"
    require(member in archive.namelist(), "the wheel does not ship the compiled group: " + member)
    shipped = sha_bytes(archive.read(member))
    staged, final = replay(raw, tier, stem, stage, scratch)
    require(final == shipped, "shipped bytes are not the compiled bytes after packaging: " + member)
    index = json.loads(archive.read(f"mojolearn/{tier}/_mojolearn_merged.json"))
    members = (index.get("files") or {}).get(stem + ".so")
    require(isinstance(members, list) and members, "the wheel's merged index lacks the group: " + member)
    return stem, dict(tier=tier, members=sorted(members), key=key, key_rule=rule, compile=compile_,
                      toolchain=toolchain,
                      host_toolchain=local["host_toolchain"], build_env=environment,
                      installed_toolchain=manifest.get("installed_toolchain"), raw_sha256=raw_sha,
                      staged_sha256=staged, wheel_sha256=shipped, key_ref=key_ref,
                      archive=dict(path=str(arc.resolve()), sha256=sha_bytes(arc.read_bytes())))


def replay_tools():
    require(sys.platform == "darwin" and all(shutil.which(t) for t in ("strip", "codesign", "install_name_tool", "otool")),
            "the packaging replay needs the macOS strip/codesign/install_name_tool/otool")


def _facts(root, producer):
    import release_compile_subunit_evidence as sub
    replay_tools()
    wheel_ref, log_ref = reference(producer["wheel"]), reference(producer["log"])
    cache = Path(producer["cache"]).resolve()
    rows_ref = reference(cache / "provenance" / "provenance.tsv")
    compiles = _log_compiles(log_ref["path"])
    require(all(outcome in OUTCOMES for outcome, _ in compiles),
            "a merged group was neither compiled and cached in this build nor placed from a verified compile: "
            + ", ".join(sorted({o for o, _ in compiles if o not in OUTCOMES})))
    keys, outcomes = [], {}
    for outcome, prefix in compiles:
        found = sorted(p.name[:-len(".json")] for p in (cache / "provenance" / "keys").glob(prefix + "*.json"))
        require(len(found) == 1 and HEX64.fullmatch(found[0]), "the build log's key is not unique in the cache: " + prefix)
        keys.append(found[0])
        outcomes[found[0]] = outcome
    require(len(set(keys)) == len(keys), "the build log names one key twice")
    rows = _provenance_rows(rows_ref["path"])
    placements_path = cache / "provenance" / "placements.jsonl"
    placed_any = any(OUTCOMES[o] == "placed" for o in outcomes.values())
    placements = _placements(placements_path) if placed_any else {}
    evidence = [wheel_ref, log_ref, rows_ref] + ([reference(placements_path)] if placed_any else [])
    with zipfile.ZipFile(wheel_ref["path"]) as archive:
        revision = archive.read("mojolearn/identity_columns/COMMIT").decode().strip()
        require(HEX40.fullmatch(revision) is not None, "the wheel names no source commit")
        tree = _tree(root, revision)
        shrink = tree.text("packaging/macos/shrink_wheel.py")
        require(all(step in shrink for step in SHRINK_STEPS), "the tested revision's shrink step is not the one replayed")
        repos = {Path(load(cache / "provenance" / "keys" / (k + ".json")).get("repo_path", "")).resolve() for k in keys}
        require(len(repos) == 1, "the compile keys name different checkouts")
        checkout = repos.pop()
        generated = sub._generated_shas(tree, checkout)
        merged_key = _merged_key(tree)
        scratch = tempfile.mkdtemp(prefix="macos-release-replay-")
        try:
            stage = _stage_module(tree, scratch)
            groups = dict(_group(k, outcomes[k], rows, placements, cache, tree, archive, generated, merged_key,
                                 stage, scratch) for k in keys)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        require(len(groups) == len(keys), "two compiles wrote one group")
        for tier in KINDS:
            index = json.loads(archive.read(f"mojolearn/{tier}/_mojolearn_merged.json"))
            shipped = {n[:-3] for n in (index.get("files") or {})}
            require(shipped == {s for s, g in groups.items() if g["tier"] == tier},
                    "the wheel ships a merged group this build did not compile: " + tier)
        members = [m for g in groups.values() for m in g["members"]]
        require(len(members) == len(set(members)), "a binding is a member of two groups")
        actual = {}
        names = [n for n in archive.namelist() if n.startswith("mojolearn/_adapters/") and not n.endswith("/")]
        for name in names:
            data = archive.read(name)
            actual[name.rsplit("/", 1)[1]] = (sha_bytes(data), len(data))
        require("mojolearn/_adapters/adapters.json" in names, "the macOS wheel ships no adapters")
        manifest = json.loads(archive.read("mojolearn/_adapters/adapters.json"))
    actual.pop("adapters.json")
    files = sub._manifest_files(manifest, actual)
    host = {json.dumps(g["host_toolchain"], sort_keys=True) for g in groups.values()}
    require(len(host) == 1, "the merged groups name different Apple toolchains")
    return dict(producer=NAME, source_root=str(Path(root).resolve()), source_revision=revision,
                checkout=str(checkout), merged_key=merged_key, generated=generated,
                host_toolchain=json.loads(host.pop()), compiled_sources={}, manifest=manifest, files=files,
                groups=groups, evidence=evidence,
                # Which groups this build compiled and which it placed from an earlier
                # compile of the same key (each group's `compile` names when and where).
                compiles={how: sorted(s for s, g in groups.items() if g["compile"]["outcome"] == how)
                          for how in ("built", "placed")})


def _collect(root, document, producer):
    facts = _facts(root, producer)
    configuration = copy.deepcopy((document or {}).get("configuration") or {})
    wheel = Path(producer["wheel"])
    shipped = configuration.get("artifacts:apple")
    if isinstance(shipped, dict) and wheel.name in shipped:
        require(shipped[wheel.name] == facts["evidence"][0]["sha256"], "the ledger names other macOS wheel bytes")
    context = dict(configuration.get("dependency_context") or {})
    require(context.get("apple_toolchain") in (None, facts["host_toolchain"]),
            "the ledger names another Apple toolchain")
    context["apple_toolchain"] = facts["host_toolchain"]
    context.setdefault("cpu_target", MAC)
    units = {"compile-apple/adapters/apple": dict(what="adapters")}
    for stem, group in facts["groups"].items():
        for name in group["members"]:
            if group["tier"] == "host":
                key = f"{MAC}/host/-/host/{name}"
                specs = ("compile-apple", "compile-cpu") if context["cpu_target"] == MAC else ("compile-apple",)
            else:
                key = f"{MAC}/metal/apple/{group['tier']}/{name}"
                specs = ("compile-apple",)
            for spec in specs:
                units[spec + "/" + key] = dict(what="binding", group=stem, binding_key=key)
    return dict(source_revision=facts["source_revision"], facts=facts, units=units,
                configuration=dict(release_modes=["identical"], build_flags={}, dependency_context=context),
                spec_ids=sorted({rid.split("/", 1)[0] for rid in units}))


def _binding_check(requirement_id, wanted, facts, unit):
    import release_retained_compile_evidence as compiled
    from release_requirements import build_flags
    require(supports_requirement(wanted), "unsupported compile target/configuration")
    binding, declared = wanted["inputs"]["binding"], wanted["inputs"]["unit"]
    key = unit["binding_key"]
    require(compiled.binding_key(binding) == key and requirement_id.endswith("/" + key) and binding.get("target") == MAC,
            "admitted unit names another binding")
    group = facts["groups"][unit["group"]]
    require(binding["name"] in group["members"] and binding["tier"] == group["tier"],
            "compiled group does not contain this binding/tier")
    require(binding.get("toolchain") == group["toolchain"], "compiler packages differ from the binding inputs")
    require(binding.get("host_toolchain") == group["host_toolchain"] == facts["host_toolchain"],
            "Apple host toolchain differs")
    # Build switches as the requirement manifest defines them (credentials and the
    # scheduler's slot token are not compiler inputs); the merged key reproduced in facts.
    import bincache
    environment = build_flags(group["build_env"])
    # closure-v2 keys carry the group's exact membership (`merged`, reproduced in
    # facts) instead of MOJOLEARN_MERGED_KEY; older keys carry the whole-plan digest.
    merged_key = environment.pop("MOJOLEARN_MERGED_KEY", None)
    require(merged_key is None if group.get("key_rule") == bincache.KEY_RULE else merged_key == facts["merged_key"],
            "merged group plan differs")
    for name, value in ROUTING.items():
        require(environment.pop(name, value) == value, "release build routing flag differs: " + name)
    compiled.check_flags(binding.get("flags") or {}, environment)
    require(set(environment) <= set(compiled.FLAG_ENV.values()) and declared["build_flags"] == {},
            "declared build flags differ from the release build invocation")
    compiled.require_closure(_tree(facts["source_root"], facts["source_revision"]), wanted)
    return compiled.coverage_rows(wanted), list(facts["evidence"]) + [group["key_ref"], group["archive"]]


def _check(requirement_id, wanted, facts, unit):
    import release_compile_subunit_evidence as sub
    if unit["what"] == "binding":
        return _binding_check(requirement_id, wanted, facts, unit)
    require(wanted["inputs"]["environment"].get("target") == MAC, "unsupported compile target/configuration")
    rows, _ = sub._pack_check(requirement_id, wanted, facts, unit)
    return rows, list(facts["evidence"]) + [g["key_ref"] for g in facts["groups"].values()]


class _Adapter:
    collect = staticmethod(_collect)
    check_unit = staticmethod(_check)

    @staticmethod
    def verify(report):
        return _facts(report["facts"]["source_root"], report["producer"])


ADAPTERS = {NAME: _Adapter()}
