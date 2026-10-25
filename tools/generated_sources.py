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
import hashlib
import io
import re
import tempfile
import types
from pathlib import Path, PurePosixPath

#: The gitignored file the tokenizer build writes before it compiles.
UNICODE_TABLE = "tokenizer/impl/unicode_table_generated.mojo"
#: The tracked inputs that fully determine it (with the pinned table sha256).
UNICODE_GENERATOR = "tokenizer/tools/gen_unicode_categories.py"
UNICODE_PIN = "tokenizer/impl/unicode_class.mojo"
GENERATED = {UNICODE_TABLE: (UNICODE_GENERATOR, UNICODE_PIN)}
#: The directory release_reuse.identity records as `generators` for it.
GENERATOR_DIRS = {UNICODE_TABLE: "tokenizer/tools"}


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
    with tempfile.TemporaryDirectory(prefix="mojolearn-generated-") as directory:
        module.OUT = str(Path(directory) / "unicode_table_generated.mojo")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = module.main([])
        require(rc == 0, "the tracked generator refuses this table (version or sha256 pin differs)")
        return Path(module.OUT).read_bytes()


def verify(rel, data, inputs):
    """Require DATA (the bytes of generated file REL) to be exactly what its
    tracked generator produces. INPUTS maps each generator input path to its
    tracked bytes. Returns the sha256 of DATA."""
    require(rel in GENERATED, "unknown generated file " + rel)
    require(set(inputs) == set(GENERATED[rel]), "generator inputs differ for " + rel)
    generator, pin = (inputs[name] for name in GENERATED[rel])
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
            require(found is not None, rel + ": this Python lacks the pinned unicodedata and no generated copy "
                    "exists to verify; run sh tokenizer/tools/gen_unicode_table.sh in the source checkout")
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
