#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Build-time generated Mojo sources, and the exact proof that one is what its
tracked generator produces.

WHY (2026-10-25). The first real `tools/release_host_build.py build` of 0.8.57
compiled and read back all 46 host members, then refused itself: the merged
build stamp of `_mojolearn_tokenizer_host` (tools/binding_stamps.py, written
from the working tree after tokenizer/tools/gen_unicode_table.sh wrote the
gitignored tokenizer/impl/unicode_table_generated.mojo) is the closure digest
WITH that file, while the binding identity (tools/release_reuse.py, from
`git archive` of the commit) is the closure digest WITHOUT it plus the
generator inputs. Both are right about what they read; a release proof has to
join them exactly, never by dropping the comparison.

The generated table is fully determined by tracked bytes:
tokenizer/impl/unicode_class.mojo pins the unicodedata version and the sha256
of the table text, and tokenizer/tools/gen_unicode_categories.py renders the
file from (version, sha256, text). `verify` runs that generator's own
rendering, loaded from the given tracked bytes, on the text carried in the
file, and requires the result to be byte-identical to the file and the text to
hash to the pinned sha256.
"""
import contextlib
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import fcntl
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path, PurePosixPath

#: The gitignored file the tokenizer build writes before it compiles.
UNICODE_TABLE = "tokenizer/impl/unicode_table_generated.mojo"
#: The tracked inputs that fully determine it (with the pinned table sha256).
UNICODE_GENERATOR = "tokenizer/tools/gen_unicode_categories.py"
UNICODE_PIN = "tokenizer/impl/unicode_class.mojo"
ARCHIVE_TABLE = "core/archive_unicode_names_generated.mojo"
ARCHIVE_GENERATOR = "tools/gen_archive_unicode.py"
ARCHIVE_PIN = "core/native_archive_unicode_names.mojo"
GENERATED = {UNICODE_TABLE: (UNICODE_GENERATOR, UNICODE_PIN),
             ARCHIVE_TABLE: (ARCHIVE_GENERATOR, ARCHIVE_PIN)}
#: The directory release_reuse.identity records as `generators` for it.
GENERATOR_DIRS = {UNICODE_TABLE: "tokenizer/tools", ARCHIVE_TABLE: ARCHIVE_GENERATOR}
COMMANDS = {UNICODE_TABLE: ["sh", "tokenizer/tools/gen_unicode_table.sh"],
            ARCHIVE_TABLE: ["python3", ARCHIVE_GENERATOR]}
LOG_NAMES = {UNICODE_TABLE: "generate-unicode.log", ARCHIVE_TABLE: "generate-archive-unicode.log"}


class GeneratedSourceError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise GeneratedSourceError("generated source: " + message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _pins(pin_bytes):
    src = pin_bytes.decode("utf-8")
    version = re.search(r'^comptime UNICODE_VERSION_PINNED = "([^"]+)"$', src, re.M)
    digest = re.search(r'^comptime UNICODE_TABLE_SHA256_PINNED = "([0-9a-f]{64})"$', src, re.M)
    require(version and digest, UNICODE_PIN + " carries no version/sha256 pin")
    return version.group(1), digest.group(1)


def _literal_text(body):
    """The table text inside `comptime UNICODE_TABLE_TEXT = "..."`."""
    m = re.search(r'^comptime UNICODE_TABLE_TEXT = "((?:[^"\\\n]|\\.)*)"$', body, re.M)
    require(m is not None, UNICODE_TABLE + " carries no table literal")
    out, chars, i = [], m.group(1), 0
    escapes = {"\\": "\\", '"': '"', "t": "\t", "n": "\n"}
    while i < len(chars):
        c = chars[i]
        if c == "\\":
            require(i + 1 < len(chars) and chars[i + 1] in escapes, "unknown escape in the table literal")
            out.append(escapes[chars[i + 1]])
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def render_unicode_table(generator_bytes, pin_bytes, text, version):
    """What the tracked generator writes for this table text: its own main()
    with unicodedata's two outputs (the version string and table_text) given."""
    module = types.ModuleType("mojolearn_pinned_unicode_generator")
    exec(compile(generator_bytes, UNICODE_GENERATOR, "exec"), module.__dict__)  # tracked generator source
    pinned = _pins(pin_bytes)
    module.unicodedata = types.SimpleNamespace(unidata_version=version)
    module.table_text = lambda: text
    module.pins = lambda: pinned
    scratch = Path.home() / ".cache/mojolearn/generated-source-render"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="render-", dir=scratch) as directory:
        module.OUT = str(Path(directory) / "unicode_table_generated.mojo")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = module.main([])
        require(rc == 0, "the tracked generator refuses this table (version or sha256 pin differs)")
        return Path(module.OUT).read_bytes()


def _archive_payload(data, pin):
    body = data.decode("ascii")
    pinned = re.search(r'^comptime ARCHIVE_UNICODE_TABLE_SHA256_PINNED = "([0-9a-f]{64})"$', pin.decode(), re.M)
    require(pinned is not None, ARCHIVE_PIN + " carries no data digest pin")
    names = re.search(r'^comptime ARCHIVE_UNICODE_NAMES = """(.*?)"""$', body, re.M | re.S)
    count = re.search(r'^comptime ARCHIVE_UNICODE_NAME_COUNT = ([0-9]+)$', body, re.M)
    require(names is not None and count is not None, ARCHIVE_TABLE + " carries no name records")
    fields = []
    for field in ("NAME_OFFSETS", "DIGITS", "CJK_RANGES", "TANGUT_RANGES"):
        match = re.search(r'^comptime ARCHIVE_UNICODE_' + field + r' = "([0-9A-F]*)"$', body, re.M)
        require(match is not None, ARCHIVE_TABLE + " lacks " + field)
        fields.append(match.group(1))
    names = names.group(1)
    count = int(count.group(1))
    require(count == names.count("\n") and len(fields[0]) == (count + 1) * 8,
            ARCHIVE_TABLE + " name count/offset count differs")
    require(sha256((names + "".join(fields)).encode("ascii")) == pinned.group(1),
            ARCHIVE_TABLE + " payload differs from pinned sha256")
    return count, names, *fields


def render_archive_table(generator_bytes, payload):
    module = types.ModuleType("mojolearn_pinned_archive_unicode_generator")
    exec(compile(generator_bytes, ARCHIVE_GENERATOR, "exec"), module.__dict__)
    return module.render_payload(*payload).encode("ascii")


def verify(rel, data, inputs):
    """Require DATA (the bytes of generated file REL) to be exactly what its
    tracked generator produces. INPUTS maps each generator input path to its
    tracked bytes. Returns the sha256 of DATA."""
    require(rel in GENERATED, "unknown generated file " + rel)
    require(set(inputs) == set(GENERATED[rel]), "generator inputs differ for " + rel)
    generator, pin = (inputs[name] for name in GENERATED[rel])
    if rel == ARCHIVE_TABLE:
        try:
            payload = _archive_payload(data, pin)
        except UnicodeDecodeError as problem:
            raise GeneratedSourceError(rel + " is not ASCII") from problem
        require(render_archive_table(generator, payload) == data,
                rel + " differs from what the tracked generator writes")
        return sha256(data)
    version, pinned_sha = _pins(pin)
    try:
        body = data.decode("ascii")
    except UnicodeDecodeError:
        raise GeneratedSourceError("generated source: " + rel + " is not ASCII")
    text = _literal_text(body)
    require(text.startswith("# unicodedata " + version + "\n"), rel + " is not unicodedata " + version)
    require(sha256(text.encode("ascii")) == pinned_sha, rel + " table text is not the pinned sha256")
    require(render_unicode_table(generator, pin, text, version) == data,
            rel + " differs from what the tracked generator writes")
    return sha256(data)


def regenerate(rel, inputs):
    """The file the tracked generator writes, computed here when this
    interpreter's unicodedata is the pinned version; None otherwise."""
    import unicodedata
    require(rel in GENERATED and set(inputs) == set(GENERATED[rel]), "generator inputs differ for " + rel)
    # The archive table spans five pinned UCD versions. Evidence verification
    # uses retained bytes and the tracked digest/renderer, never a network
    # request or the evidence reader's own unicodedata version.
    if rel == ARCHIVE_TABLE:
        return None
    generator, pin = (inputs[name] for name in GENERATED[rel])
    version, _ = _pins(pin)
    if unicodedata.unidata_version != version:
        return None
    module = types.ModuleType("mojolearn_pinned_unicode_generator")
    exec(compile(generator, UNICODE_GENERATOR, "exec"), module.__dict__)  # tracked generator source
    text = module.table_text()
    return render_unicode_table(generator, pin, text, version)


_EXPECTED = {}


def expected_sha(rel, inputs, candidates=()):
    """sha256 of the exact bytes the tracked generator writes for REL from
    INPUTS ({input path: tracked bytes}). Regenerated in process when this
    interpreter carries the pinned unicodedata, else taken from the first
    existing CANDIDATE file and verified (verify). Refuses when neither can
    establish the bytes."""
    key = (rel, tuple(sorted((k, sha256(v)) for k, v in inputs.items())))
    if key not in _EXPECTED:
        data = regenerate(rel, inputs)
        if data is not None:
            _EXPECTED[key] = sha256(data)
        else:
            found = next((Path(c) for c in candidates if Path(c).is_file()), None)
            require(found is not None, rel + ": no generated copy exists to verify; run " +
                    " ".join(COMMANDS[rel]) + " in the source checkout")
            _EXPECTED[key] = verify(rel, found.read_bytes(), inputs)
    return _EXPECTED[key]


def closure_hash(rows):
    """bincache.source_digest's digest over recorded (path, file digest) rows,
    in their recorded (the walk's) order."""
    h = hashlib.sha256()
    for rel, digest in rows:
        h.update(rel.encode() + b"\0" + digest.encode() + b"\n")
    return h.hexdigest()


def closure_rows(repo, script):
    """(rows, scope) of bincache.source_digest for one build script in REPO,
    with each file's own digest, so a proof can recompute both the digest with
    and without a generated file."""
    import bincache
    info, rels = bincache.source_digest(repo, str(Path(repo) / "bindings" / script), [])
    rows = []
    for rel in rels:
        f = Path(repo) / rel
        rows.append([rel, bincache.sha256_bytes(bincache.pixi_toml_build_part(f.read_text(errors="replace")).encode())
                     if rel == "pixi.toml" else bincache.sha256_file(f)])
    require(closure_hash(rows) == info["digest"], "closure rows do not reproduce bincache's digest for " + script)
    return rows, info["scope"]


def sorted_rels(rels):
    """bincache's whole-tree order: sorted by path parts."""
    return sorted(rels, key=lambda name: PurePosixPath(name).parts)


def available_sources(repo):
    """Known outputs present or enabled by their tracked native pin module."""
    root = Path(repo)
    return [rel for rel, (_, pin) in GENERATED.items()
            if (root / pin).is_file() or (root / rel).is_file()]


def generation_workers(count):
    if count <= 1:
        return 1
    if sys.platform == "linux":
        from build_sizing import box_resources
        cores, _memory = box_resources()
    else:
        cores = os.cpu_count() or 1
    # At most two bounded metadata outputs exist. Both run independently; each
    # generator manages its own independent version-input units.
    return max(1, min(count, cores))


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".part-{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def retain_sources(repo, evidence, outputs=None):
    """Retain actual ignored output/input bytes, including missing/failing runs.

    Retention alone is not a validation verdict. The manifest reports whether
    bytes existed; prepare_sources adds the generator and verification results.
    """
    root, evidence = Path(repo), Path(evidence)
    selected = available_sources(root) if outputs is None else list(outputs)
    def retain(rel):
        require(rel in GENERATED, "unknown generated file " + rel)
        row = {"state": "missing", "inputs": {}}
        for name in GENERATED[rel]:
            if (root / name).is_file():
                target = evidence / "generated-inputs" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / name, target)
                row["inputs"][name] = sha256(target.read_bytes())
        if (root / rel).is_file():
            target = evidence / "generated" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / rel, target)
            row.update(state="retained", sha256=sha256(target.read_bytes()), path=str(target))
        return rel, row
    with ThreadPoolExecutor(max_workers=generation_workers(len(selected))) as pool:
        rows = dict(pool.map(retain, selected))
    _atomic_json(evidence / "generated-sources.json", {"sources": rows})
    return rows


def prepare_sources(repo, evidence=None, outputs=None, env=None, runner=None):
    """Prepare/verify independent outputs before cache keys or compilation.

    runner(command, repo, env, log) preserves the development controller's
    guarded process runner. Valid existing output is reused only after exact
    tracked-generator verification. Replaced bytes and failures are retained.
    """
    root = Path(repo).resolve()
    evidence = Path(evidence) if evidence else root / "build/generated-sources" / f"{time.time_ns()}-{os.getpid()}"
    evidence.mkdir(parents=True, exist_ok=True)
    selected = available_sources(root) if outputs is None else list(dict.fromkeys(outputs))
    environment = dict(os.environ if env is None else env)
    if runner is None:
        def runner(command, repo, env, log):
            with open(log, "wb") as stream:
                return subprocess.run(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                      stdout=stream, stderr=subprocess.STDOUT).returncode
    def prepare_locked(rel):
        row = {"state": "failed", "command": COMMANDS.get(rel), "verified": False}
        try:
            require(rel in GENERATED, "unknown generated file " + rel)
            inputs = {}
            for name in GENERATED[rel]:
                require((root / name).is_file(), "missing tracked generator input " + name)
                inputs[name] = (root / name).read_bytes()
            if (root / rel).is_file():
                previous = (root / rel).read_bytes()
                try:
                    row.update(sha256=verify(rel, previous, inputs), state="reused", verified=True)
                    return rel, row
                except GeneratedSourceError as problem:
                    old = evidence / "replaced-generated" / rel
                    old.parent.mkdir(parents=True, exist_ok=True)
                    old.write_bytes(previous)
                    row.update(replaced_sha256=sha256(previous), replaced_reason=str(problem))
            log = evidence / LOG_NAMES[rel]
            row.update(log=str(log), exit_code=runner(list(COMMANDS[rel]), root, environment, log))
            require(row["exit_code"] == 0, "generator exited " + str(row["exit_code"]) + "; see " + str(log))
            require((root / rel).is_file(), "generator did not create " + rel)
            require(all((root / name).read_bytes() == raw for name, raw in inputs.items()),
                    "generator inputs changed during preparation of " + rel)
            row.update(sha256=verify(rel, (root / rel).read_bytes(), inputs), state="generated", verified=True)
        except Exception as problem:
            row["error"] = str(problem)
        return rel, row
    def prepare(rel):
        # Separate compile groups may request the same ignored output at once.
        # One lock per output permits the two independent generators to run
        # concurrently while preventing partial reads or competing writers.
        locks = root / "build/generated-sources/.locks"
        locks.mkdir(parents=True, exist_ok=True)
        with open(locks / (sha256(rel.encode()) + ".lock"), "a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return prepare_locked(rel)
    with ThreadPoolExecutor(max_workers=generation_workers(len(selected))) as pool:
        rows = dict(pool.map(prepare, selected))
    retained = retain_sources(root, evidence, selected)
    for rel, row in rows.items():
        row["retained"] = retained[rel]
    _atomic_json(evidence / "generated-sources.json", {"sources": rows})
    failed = [rel for rel, row in rows.items() if not row["verified"]]
    require(not failed, "preparation failed for " + ", ".join(failed) + "; see " + str(evidence / "generated-sources.json"))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "retain"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", action="append", choices=tuple(GENERATED))
    args = parser.parse_args(argv)
    if args.command == "retain":
        require(args.evidence is not None, "retain requires --evidence")
        rows = retain_sources(args.repo, args.evidence, args.output)
    else:
        rows = prepare_sources(args.repo, args.evidence, args.output)
    print(json.dumps({rel: {key: row[key] for key in ("state", "sha256", "verified") if key in row}
                      for rel, row in rows.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
