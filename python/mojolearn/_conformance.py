# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`python -m mojolearn conformance` -- the identity claim as a portable artifact.

WHAT THIS IS

`verify` (`_verify.py`) checks THIS build against a reference card. This
module goes one step further: it packages the pinned fixture -- frozen input
bytes, expected stage bytes, the identity-trace card, and a manifest that
hashes all of it -- into a BUNDLE DIRECTORY that an external implementation
(a simulator, an accelerator bring-up, another library -- anything that
consumes the bundle without running Mojo or Python) can check itself against,
and it grades the result file such an implementation writes back.

Three subcommands:

    export     produce a bundle directory (format v1, docs/CONFORMANCE.md)
    validate   structural check of a bundle, or of an external
               implementation-report.json against a bundle
    diff       first-divergence localization, through the repository's ONE
               card comparator (tools/identity_trace_diff.py), same verdict
               vocabulary: IDENTICAL / DIVERGENT, FIRST DIVERGENCE names a
               stage

THE TWO HASHES, AND WHY THERE ARE TWO (DEVIATION 928)

    SHA-256    the AUDIT-INTEGRITY layer. Computed host-side (hashlib) over
               every file in the bundle and recorded in manifest.json, and
               computed by the external side over its raw stage bytes. This
               is the layer that answers "is this the artifact that was
               shipped" and "did anything get corrupted or swapped".
    FNV-1a64   the IN-KERNEL LOCALIZATION checksum. It is what
               `core/identity_trace.mojo` can afford to compute on-device at
               every checkpoint, it is what the cards carry, and it is what
               the differ aligns and walks. It is NOT collision-resistant
               and it is NEVER the audit layer. It stays because the first
               diverging stage is found by walking it, and because one hash
               function per purpose beats one hash function for both.

An external implementation reports BOTH per stage: SHA-256 so `validate` can
grade against the audit layer, FNV-1a64 so `diff` can hand the comparison to
the same differ every other card in this repository goes through
(DEVIATION 924: one comparator, never two).

WHAT A PASS MEANS -- docs/CONFORMANCE.md says it in full, but the short form
belongs here too: a validated bundle round-trip on ONE machine claims nothing
across vendors; the cross-vendor claim lives in the confirmed reference card
(docs/VERIFY.md); a conformant external implementation claims agreement on
THESE fixtures under THIS profile, and nothing wider.

ARTIFACT PROVENANCE (DEVIATION 929). A bundle certifies the runs made
through the compiled extension artifacts (`.so`) that were ON DISK at export
time -- and today NOTHING ties a `.so` to the commit that built it. The
manifest therefore records, for every loaded binding, its file SHA-256, size
and mtime, plus `artifact_source_commit: "unverified"`, stated rather than
implied. The closing mechanism (a compile-time commit+mode stamp in each
binding, read back at import) is specced in docs/CONFORMANCE.md and owned by
the bindings lane; it is not this module's to implement.

EXIT CODES (aligned with tools/identity_trace_diff.py, documented in
docs/CONFORMANCE.md):

    0   PASS: the bundle is structurally valid / the report is conformant /
        the traces are identical
    1   FAIL: a named structural defect, a named non-conformant stage, or a
        divergence with its first stage named
    2   COULD NOT JUDGE: usage, unparseable input, a refused report (profile
        mismatch, missing provenance), or no comparator. Never a verdict.
"""

import hashlib
import json
import os
import re
import sys
import tempfile
import time

from . import _verify

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2

FORMAT_NAME = "mojolearn-conformance-bundle"
FORMAT_VERSION = "1"
REPORT_FORMAT_NAME = "mojolearn-implementation-report"

#: Stage buffers at or below this many bytes ship raw (`expected/.../*.bin`);
#: above it, digest-only (`*.sha256`). Documented in docs/CONFORMANCE.md;
#: changing it is a format change and needs a version bump there.
RAW_THRESHOLD_BYTES = 1 << 20

#: The one fixture format v1 ships. Its parameters, input generation and the
#: reason it was chosen are `_verify.py`'s (DEVIATIONS 921/922/926); this
#: module reuses them rather than restating them.
FIXTURE_ID = "kmeans"

_ITEMSIZE = {"f32": 4, "f64": 8, "u32": 4, "i32": 4, "u8": 1, "i64": 8}
"""dtype -> bytes per element. This is format data (the card format's own
table, docs/CONFORMANCE.md section 3), not comparison logic, so keeping a
copy here does not violate the one-comparator rule."""

_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_FNV_RE = re.compile(r"\A[0-9a-f]{16}\Z")


def _utc(ts=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ",
                         time.gmtime(ts if ts is not None else time.time()))


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _sanitize_tag(tag):
    """Filename-safe tag, the SAME rule as `core/identity_trace.mojo` and
    the differ: anything outside [A-Za-z0-9._-] becomes '_'. Part of format
    v1, restated in docs/CONFORMANCE.md so an external team needs no code."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", tag)


def _stage_basename(rec):
    """`<seq>.<sanitized-tag>` -- unpadded seq, matching the trace writer's
    own dump naming, so a dump and its bundle file are recognizably the same
    stage."""
    return "%d.%s" % (rec.seq, _sanitize_tag(rec.tag))


def _rel(*parts):
    """Bundle-internal paths are ALWAYS forward-slash, on every host, so a
    manifest written on one OS validates on another byte for byte."""
    return "/".join(parts)


def _emit(lines):
    sys.stdout.write("\n".join(lines) + "\n")


def _load_differ_or_none(out):
    try:
        return _verify.load_differ()
    except RuntimeError as exc:
        out.append("COULD NOT JUDGE: no card comparator.")
        out.append("")
        for ln in str(exc).split("\n"):
            out.append("  " + ln)
        return None


# --------------------------------------------------------------------------
# EXPORT
# --------------------------------------------------------------------------


def cmd_export(args):
    """Produce a format v1 bundle at args.out.

    Two modes.

    LIVE (default): runs the pinned fixture on THIS machine through
    `_verify.run_fixture`, with per-stage raw dumps enabled, then assembles.
    Needs an identical-mode build and a GPU, and refuses on FAST for exactly
    the reason `verify` does: a bundle exported from the FAST arm would
    package numbers that make no identity claim, under a directory name that
    says they do.

    --from-card CARD: host-side assembly from an already-produced card whose
    per-stage `.bin` dumps sit beside it (the trace writer's own
    `<card>.<seq>.<tag>.bin` naming). No GPU, no extension import. The
    card must carry `# profile:` and `# numeric-mode: identical` comment
    lines; a card that cannot state its own arm is refused rather than
    trusted.
    """
    out_dir = args.out
    if os.path.exists(out_dir) and os.listdir(out_dir) and not args.force:
        _emit(["USAGE: %s exists and is not empty. A bundle directory is "
               "written whole or not at all; pass --force to overwrite its "
               "contents in place." % out_dir])
        return EXIT_USAGE

    lines = []
    resolved = _load_differ_or_none(lines)
    if resolved is None:
        _emit(lines)
        return EXIT_USAGE
    differ, differ_path, differ_how = resolved

    if args.from_card:
        return _export_from_card(args, differ, differ_path, differ_how)

    # ---- live mode -------------------------------------------------------
    mode = _verify._mode_report()
    env = _verify.environment(mode)
    if mode.problem is not None:
        _emit(["COULD NOT JUDGE: the numeric mode of this process cannot be "
               "established, so nothing can be exported under an identity "
               "profile.", "", "  " + mode.problem])
        return EXIT_USAGE
    if mode.loaded != "identical":
        _emit([
            "REFUSED. This process loaded the FAST binaries, which make no",
            "identity claim, and a conformance bundle IS the identity claim",
            "in portable form. Set the mode before import:",
            "",
            "    MOJOLEARN_NUMERIC_MODE=identical python -m mojolearn "
            "conformance export " + out_dir,
        ])
        return EXIT_USAGE

    tmpdir = tempfile.mkdtemp(prefix="mojolearn-conformance-export-")
    card = os.path.join(tmpdir, FIXTURE_ID + ".card")
    try:
        # EVERY tag this fixture emits contains a '.', so '.' as the dump
        # substring dumps every stage. That is a property of THIS fixture's
        # tag vocabulary (fit.*, restartNN.iterMM.*), not of traces in
        # general, and it is asserted after the fact: a stage whose dump is
        # missing fails the export by name below.
        _est, elapsed, _xh, _ch = _verify.run_fixture(card, dump=".")
    except _verify.FixtureError as exc:
        _emit(["COULD NOT JUDGE: the fixture did not run, so there is "
               "nothing to export.", "", "  %s" % exc])
        return EXIT_USAGE

    provenance = {
        "produced_by": "python -m mojolearn conformance export",
        "date": _utc(),
        "mojolearn_version": env["mojolearn_version"],
        "commit": env["commit"],
        "numeric_mode": mode.loaded,
        "vendor": env.get("vendor", "not established"),
        "device": env["device"],
        "host": env["platform"],
        "python": env["python"],
        "fixture_binary": mode.binary or "unknown",
        "fit_seconds": round(elapsed, 3),
        "comparator": {"path": differ_path, "how": differ_how,
                       "sha256": _verify._sha256(differ_path)},
        # DEVIATION 929, stated where the reader will meet it.
        "artifacts": _verify.binding_artifacts(),
        "artifacts_recorded": True,
        "artifact_caveat": (
            "this bundle certifies the compiled artifacts it hashes, not "
            "the source tree; nothing currently ties a .so to the commit "
            "that built it (artifact_source_commit is 'unverified' until "
            "the bindings carry a build stamp -- docs/CONFORMANCE.md, "
            "'Artifact provenance')"),
    }
    rc = _assemble(out_dir, card, card, provenance, differ, lines)
    if rc == EXIT_OK:
        lines.append("")
        lines.append("Next: python -m mojolearn conformance validate %s"
                     % out_dir)
        lines.append("")
        lines.append("HONESTY: this bundle is one machine's output. Exported")
        lines.append("and validated on the same box, it proves the format")
        lines.append("round-trips -- nothing about a second vendor. See")
        lines.append("docs/CONFORMANCE.md, 'What a PASS means'.")
    _emit(lines)
    return rc


def _export_from_card(args, differ, differ_path, differ_how):
    lines = []
    src_card = args.from_card
    if not os.path.isfile(src_card):
        _emit(["USAGE: --from-card %s: not a file" % src_card])
        return EXIT_USAGE
    kv, _comments, _n = _verify.parse_card_comments(src_card)
    profile = kv.get("profile")
    if profile != _verify.PROFILE:
        _emit(["REFUSED before any byte was compared: the card's profile "
               "line is %r, this build's is %r. A bundle assembled across "
               "that mismatch would label one computation with another's "
               "name." % (profile, _verify.PROFILE)])
        return EXIT_USAGE
    if kv.get("numeric-mode") != "identical":
        _emit(["REFUSED: the card at %s does not state '# numeric-mode: "
               "identical' (it says %r). A card that cannot state its own "
               "arm is not assembled into an identity bundle."
               % (src_card, kv.get("numeric-mode"))])
        return EXIT_USAGE
    provenance = {
        "produced_by": "python -m mojolearn conformance export --from-card",
        "date": _utc(),
        "assembled_from_card": os.path.abspath(src_card),
        "card_provenance": _comments,
        "comparator": {"path": differ_path, "how": differ_how,
                       "sha256": _verify._sha256(differ_path)},
        # The assembly host did not run the fit and cannot know which .so
        # files the producing run loaded. Saying so beats guessing.
        "artifacts": [],
        "artifacts_recorded": False,
        "artifact_caveat": (
            "assembled from a pre-existing card; the artifact record "
            "belongs to the run that produced the card, whose provenance "
            "comment lines are carried under card_provenance"),
    }
    rc = _assemble(args.out, src_card, src_card, provenance, differ, lines)
    _emit(lines)
    return rc


def _assemble(out_dir, card_path, dumps_beside, provenance, differ, out):
    """Build the bundle directory from a card plus its per-stage dumps.

    `dumps_beside` is the path whose `<path>.<seq>.<tag>.bin` siblings hold
    the raw stage bytes (for a live export that is the card itself). ALL
    stages must have dumps: a bundle with holes in `expected/` would grade
    an external implementation against silence, so a missing dump fails the
    export naming the stage rather than shipping less.
    """
    try:
        recs, _c, _b = differ.parse_trace(card_path)
    except differ.TraceError as exc:
        out.append("COULD NOT JUDGE: the card does not parse: %s" % exc)
        return EXIT_USAGE
    if not recs:
        out.append("COULD NOT JUDGE: the card at %s holds no records; an "
                   "empty bundle certifies nothing." % card_path)
        return EXIT_USAGE

    # Inputs are rebuilt host-side and GATED against the pinned three-vendor
    # hashes (DEVIATION 922) before anything is written. A bundle whose
    # inputs drifted would grade every consumer wrong at once.
    x, c = _verify.build_fixture()
    x_bytes = _verify._le_bytes(x)
    c_bytes = _verify._le_bytes(c)
    xh = differ.fnv1a64(x_bytes)
    ch = differ.fnv1a64(c_bytes)
    if xh != _verify.FIXTURE_X_FNV1A64 or ch != _verify.FIXTURE_C_FNV1A64:
        out.append("COULD NOT JUDGE: the rebuilt fixture inputs do not match "
                   "the pinned E1U hashes (DEVIATION 922); "
                   "`python -m mojolearn check-fixture` is the diagnostic.")
        return EXIT_USAGE

    fx_inputs = os.path.join(out_dir, "inputs", FIXTURE_ID)
    fx_expected = os.path.join(out_dir, "expected", FIXTURE_ID)
    traces_dir = os.path.join(out_dir, "traces")
    for d in (fx_inputs, fx_expected, traces_dir):
        os.makedirs(d, exist_ok=True)

    files = {}

    def _write(rel, data):
        full = os.path.join(out_dir, rel.replace("/", os.sep))
        with open(full, "wb") as fh:
            fh.write(data)
        files[rel] = _sha256_bytes(data)

    _write(_rel("inputs", FIXTURE_ID, "x.bin"), x_bytes)
    _write(_rel("inputs", FIXTURE_ID, "centroids.bin"), c_bytes)

    with open(card_path, "rb") as fh:
        card_bytes = fh.read()
    _write(_rel("traces", FIXTURE_ID + ".card"), card_bytes)

    stage_schema = []
    n_raw = 0
    n_digest = 0
    for rec in recs:
        dump = "%s.%d.%s.bin" % (dumps_beside, rec.seq, _sanitize_tag(rec.tag))
        if not os.path.isfile(dump):
            out.append("EXPORT FAILED. MISSING STAGE BYTES: %s (seq %d)"
                       % (rec.tag, rec.seq))
            out.append("  no dump at %s" % dump)
            out.append("  Every stage needs its raw bytes at export time; a "
                       "bundle with holes")
            out.append("  in expected/ would grade a consumer against "
                       "silence. For a live")
            out.append("  export this means the dump env did not reach the "
                       "extension; for")
            out.append("  --from-card, re-run the producing fit with "
                       "MOJOLEARN_IDENTITY_TRACE_DUMP=. ")
            return EXIT_FAIL
        with open(dump, "rb") as fh:
            data = fh.read()
        want_len = rec.count * _ITEMSIZE[rec.dtype]
        if len(data) != want_len:
            out.append("EXPORT FAILED. STAGE %s: dump is %d bytes, the card "
                       "says %s x %d = %d. The writer and this reader "
                       "disagree; nothing downstream is trustworthy."
                       % (rec.tag, len(data), rec.dtype, rec.count, want_len))
            return EXIT_FAIL
        got_fnv = "%016x" % differ.fnv1a64(data)
        if got_fnv != rec.hash:
            out.append("EXPORT FAILED. STAGE %s: dump hashes to %s, the card "
                       "records %s. The two layers disagree at export time; "
                       "refusing to package either." % (rec.tag, got_fnv,
                                                        rec.hash))
            return EXIT_FAIL
        base = _stage_basename(rec)
        if len(data) <= RAW_THRESHOLD_BYTES:
            expected_rel = _rel("expected", FIXTURE_ID, base + ".bin")
            _write(expected_rel, data)
            kind = "raw"
            n_raw += 1
        else:
            expected_rel = _rel("expected", FIXTURE_ID, base + ".sha256")
            _write(expected_rel, (_sha256_bytes(data) + "\n").encode("ascii"))
            kind = "sha256"
            n_digest += 1
        stage_schema.append({
            "seq": rec.seq,
            "tag": rec.tag,
            "dtype": rec.dtype,
            "count": rec.count,
            "fnv1a64": rec.hash,
            "sha256": _sha256_bytes(data),
            "expected": expected_rel,
            "kind": kind,
        })

    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "profile": _verify.PROFILE,
        "numeric_profile": {
            "mode": "identical",
            "dtype": "fp32",
            "version": "v1",
        },
        "raw_threshold_bytes": RAW_THRESHOLD_BYTES,
        "fixtures": [{
            "id": FIXTURE_ID,
            "profile": _verify.PROFILE,
            "params": {
                "algorithm": "kmeans",
                "n": _verify.KM_N, "d": _verify.KM_D, "k": _verify.KM_K,
                "init": "array", "n_init": 1,
                "max_iter": _verify.KM_ITERS, "tol": _verify.KM_TOL,
                "seed": _verify.KM_SEED, "sample_weight": None,
            },
            "inputs": [
                {"name": "x",
                 "file": _rel("inputs", FIXTURE_ID, "x.bin"),
                 "dtype": "f32", "shape": [_verify.KM_N, _verify.KM_D],
                 "fnv1a64": "%016x" % xh,
                 "sha256": files[_rel("inputs", FIXTURE_ID, "x.bin")]},
                {"name": "centroids",
                 "file": _rel("inputs", FIXTURE_ID, "centroids.bin"),
                 "dtype": "f32", "shape": [_verify.KM_K, _verify.KM_D],
                 "fnv1a64": "%016x" % ch,
                 "sha256": files[_rel("inputs", FIXTURE_ID,
                                      "centroids.bin")]},
            ],
            "trace": _rel("traces", FIXTURE_ID + ".card"),
            "stage_schema": stage_schema,
        }],
        "provenance": provenance,
        "files": files,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                      ).encode("utf-8")
    with open(os.path.join(out_dir, "manifest.json"), "wb") as fh:
        fh.write(manifest_bytes)

    out.append("=" * 74)
    out.append("mojolearn conformance export")
    out.append("=" * 74)
    out.append("  bundle       %s" % os.path.abspath(out_dir))
    out.append("  profile      %s" % _verify.PROFILE)
    out.append("  fixture      %s (%d stages: %d raw, %d digest-only)"
               % (FIXTURE_ID, len(stage_schema), n_raw, n_digest))
    out.append("  manifest     sha256 %s" % _sha256_bytes(manifest_bytes))
    out.append("  files hashed %d (SHA-256, the audit layer; the cards keep "
               "FNV-1a64" % len(files))
    out.append("               for localization -- docs/CONFORMANCE.md, "
               "'The two hashes')")
    return EXIT_OK


# --------------------------------------------------------------------------
# VALIDATE
# --------------------------------------------------------------------------


def _read_manifest(bundle, out):
    mpath = os.path.join(bundle, "manifest.json")
    if not os.path.isfile(mpath):
        out.append("COULD NOT JUDGE: no manifest.json in %s -- not a bundle."
                   % bundle)
        return None, None
    try:
        with open(mpath, "rb") as fh:
            raw = fh.read()
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as exc:
        out.append("COULD NOT JUDGE: manifest.json does not parse: %s" % exc)
        return None, None
    if manifest.get("format") != FORMAT_NAME or \
            manifest.get("format_version") != FORMAT_VERSION:
        out.append("COULD NOT JUDGE: manifest declares format %r version %r; "
                   "this tool reads %r version %r. A silent reinterpretation "
                   "would be worse than a refusal."
                   % (manifest.get("format"), manifest.get("format_version"),
                      FORMAT_NAME, FORMAT_VERSION))
        return None, None
    return manifest, _sha256_bytes(raw)


def _validate_bundle(bundle, manifest, differ, out):
    """Structural check. Returns a list of failure strings (empty = valid).

    Failures are SPECIFIC by construction: a wrong hash names the file, a
    missing stage names the stage, a schema drift names the record. A
    validator that says only 'invalid' teaches nobody anything.
    """
    failures = []
    warnings = []

    files = manifest.get("files", {})
    for rel in sorted(files):
        full = os.path.join(bundle, rel.replace("/", os.sep))
        if not os.path.isfile(full):
            failures.append("MISSING FILE: %s (listed in manifest, absent "
                            "on disk)" % rel)
            continue
        got = _sha256_file(full)
        if got != files[rel]:
            failures.append("WRONG HASH: %s\n    manifest %s\n    on disk  %s"
                            % (rel, files[rel], got))

    # Files on disk the manifest does not vouch for. The report file and
    # anything under reports/ are the external side's to write; everything
    # else unlisted is named, as a warning, because an audit that ignores
    # unexplained files is not an audit.
    for root, _dirs, names in os.walk(bundle):
        for name in names:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, bundle).replace(os.sep, "/")
            if rel == "manifest.json" or rel == "implementation-report.json":
                continue
            if rel.startswith("reports/"):
                continue
            if rel not in files:
                warnings.append("UNLISTED FILE: %s (present on disk, not in "
                                "the manifest)" % rel)

    for fx in manifest.get("fixtures", []):
        fxid = fx.get("id", "?")

        # Inputs: size and both hash layers.
        for inp in fx.get("inputs", []):
            rel = inp["file"]
            full = os.path.join(bundle, rel.replace("/", os.sep))
            if not os.path.isfile(full):
                continue        # already a MISSING FILE failure above
            with open(full, "rb") as fh:
                data = fh.read()
            want = 1
            for dim in inp["shape"]:
                want *= dim
            want *= _ITEMSIZE[inp["dtype"]]
            if len(data) != want:
                failures.append("INPUT SIZE: %s is %d bytes, shape %r %s "
                                "needs %d" % (rel, len(data), inp["shape"],
                                              inp["dtype"], want))
                continue
            got_fnv = "%016x" % differ.fnv1a64(data)
            if got_fnv != inp["fnv1a64"]:
                failures.append("INPUT FNV MISMATCH: %s hashes to %s, "
                                "manifest says %s" % (rel, got_fnv,
                                                      inp["fnv1a64"]))

        # The kmeans fixture's inputs are additionally held to the pinned
        # three-vendor constants, defense in depth against a manifest and
        # its inputs drifting TOGETHER (DEVIATION 922).
        if fxid == FIXTURE_ID:
            pinned = {"x": "%016x" % _verify.FIXTURE_X_FNV1A64,
                      "centroids": "%016x" % _verify.FIXTURE_C_FNV1A64}
            for inp in fx.get("inputs", []):
                want = pinned.get(inp["name"])
                if want and inp["fnv1a64"] != want:
                    failures.append(
                        "INPUT NOT THE PINNED FIXTURE: %s.%s manifest fnv "
                        "%s, the E1U three-vendor value is %s"
                        % (fxid, inp["name"], inp["fnv1a64"], want))

        # The card, through the one comparator's parser.
        card_rel = fx.get("trace", "")
        card_full = os.path.join(bundle, card_rel.replace("/", os.sep))
        recs = []
        if os.path.isfile(card_full):
            try:
                recs, _c, _b = differ.parse_trace(card_full)
            except differ.TraceError as exc:
                failures.append("CARD DOES NOT PARSE: %s: %s"
                                % (card_rel, exc))
        schema = fx.get("stage_schema", [])
        if recs:
            if len(recs) != len(schema):
                failures.append("SCHEMA DRIFT: %s holds %d records, "
                                "stage_schema lists %d"
                                % (card_rel, len(recs), len(schema)))
            for rec, st in zip(recs, schema):
                for field, got in (("seq", rec.seq), ("tag", rec.tag),
                                   ("dtype", rec.dtype),
                                   ("count", rec.count),
                                   ("fnv1a64", rec.hash)):
                    if st.get(field) != got:
                        failures.append(
                            "SCHEMA DRIFT AT STAGE %s (seq %d): card %s=%r, "
                            "manifest %r" % (rec.tag, rec.seq, field, got,
                                             st.get(field)))
                        break

        # Every stage's expected bytes or digest.
        for st in schema:
            rel = st["expected"]
            full = os.path.join(bundle, rel.replace("/", os.sep))
            if not os.path.isfile(full):
                failures.append("MISSING STAGE: %s (seq %d) -- no expected "
                                "file at %s" % (st["tag"], st["seq"], rel))
                continue
            with open(full, "rb") as fh:
                data = fh.read()
            if st["kind"] == "raw":
                want = st["count"] * _ITEMSIZE[st["dtype"]]
                if len(data) != want:
                    failures.append("STAGE SIZE: %s is %d bytes, %s x %d "
                                    "needs %d" % (rel, len(data), st["dtype"],
                                                  st["count"], want))
                    continue
                got_fnv = "%016x" % differ.fnv1a64(data)
                if got_fnv != st["fnv1a64"]:
                    failures.append(
                        "STAGE BYTES DISAGREE WITH THE CARD: %s (seq %d)\n"
                        "    expected/ bytes hash (FNV-1a64) to %s\n"
                        "    the card records                  %s\n"
                        "    `conformance diff %s --self` localizes this in "
                        "the differ's own vocabulary."
                        % (st["tag"], st["seq"], got_fnv, st["fnv1a64"],
                           bundle))
                if _sha256_bytes(data) != st["sha256"]:
                    failures.append("STAGE SHA-256 MISMATCH: %s (seq %d) -- "
                                    "expected bytes do not match the "
                                    "manifest's stage digest"
                                    % (st["tag"], st["seq"]))
            else:
                text = data.decode("ascii", "replace").strip()
                if not _SHA256_RE.match(text):
                    failures.append("MALFORMED DIGEST FILE: %s does not hold "
                                    "64 lowercase hex characters" % rel)
                elif text != st["sha256"]:
                    failures.append("DIGEST FILE DISAGREES WITH MANIFEST: %s"
                                    % rel)

    for w in warnings:
        out.append("  WARNING: %s" % w)
    return failures


def cmd_validate(args):
    lines = []
    resolved = _load_differ_or_none(lines)
    if resolved is None:
        _emit(lines)
        return EXIT_USAGE
    differ, _dp, _dh = resolved

    manifest, manifest_sha = _read_manifest(args.bundle, lines)
    if manifest is None:
        _emit(lines)
        return EXIT_USAGE

    lines.append("=" * 74)
    lines.append("mojolearn conformance validate")
    lines.append("=" * 74)
    lines.append("  bundle    %s" % os.path.abspath(args.bundle))
    lines.append("  profile   %s" % manifest.get("profile"))
    lines.append("  manifest  sha256 %s" % manifest_sha)
    prov = manifest.get("provenance", {})
    if not prov.get("artifacts_recorded", False):
        lines.append("  NOTE: artifact hashes were not recorded at export "
                     "(assembled from a card);")
        lines.append("        the producing run's card provenance is the "
                     "record.")
    lines.append("-" * 74)

    failures = _validate_bundle(args.bundle, manifest, differ, lines)
    if failures:
        lines.append("BUNDLE INVALID: %d failure(s)." % len(failures))
        for f in failures:
            lines.append("")
            for ln in f.split("\n"):
                lines.append("  " + ln)
        lines.append("=" * 74)
        lines.append("RESULT: FAIL. exit %d" % EXIT_FAIL)
        lines.append("=" * 74)
        _emit(lines)
        return EXIT_FAIL

    if not args.report:
        lines.append("BUNDLE STRUCTURALLY VALID.")
        lines.append("  Every listed file present and SHA-256-clean, every "
                     "stage's expected")
        lines.append("  bytes/digest present and consistent with the card "
                     "and the manifest.")
        lines.append("")
        lines.append("  WHAT THIS DOES NOT MEAN: nothing was computed and no "
                     "second")
        lines.append("  implementation was compared. A valid bundle on one "
                     "machine claims")
        lines.append("  nothing across vendors (docs/CONFORMANCE.md, 'What a "
                     "PASS means').")
        lines.append("=" * 74)
        lines.append("RESULT: PASS. exit %d" % EXIT_OK)
        lines.append("=" * 74)
        _emit(lines)
        return EXIT_OK

    rc = _validate_report(args, manifest, manifest_sha, differ, lines)
    _emit(lines)
    return rc


def _read_report(path, out):
    if not os.path.isfile(path):
        out.append("COULD NOT JUDGE: no report at %s" % path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
    except (OSError, ValueError) as exc:
        out.append("COULD NOT JUDGE: the report does not parse: %s" % exc)
        return None
    if report.get("format") != REPORT_FORMAT_NAME or \
            report.get("format_version") != FORMAT_VERSION:
        out.append("COULD NOT JUDGE: report declares format %r version %r; "
                   "this tool reads %r version %r."
                   % (report.get("format"), report.get("format_version"),
                      REPORT_FORMAT_NAME, FORMAT_VERSION))
        return None
    return report


_REPORT_PROVENANCE_REQUIRED = ("implementation", "device", "os", "date")


def _refuse_report(report, manifest, manifest_sha, out):
    """Every reason to refuse a report BEFORE comparing any bytes.

    Returns an error string or None. The profile gate is first and absolute:
    grading a report produced under one profile against another profile's
    expected bytes would manufacture a red (or worse, a green) about the
    wrong computation.
    """
    if report.get("profile") != manifest.get("profile"):
        return ("REFUSED BEFORE COMPARING ANY BYTES: the report's profile is "
                "%r, the bundle's is %r. These name different computations."
                % (report.get("profile"), manifest.get("profile")))
    want_ms = report.get("bundle_manifest_sha256")
    if want_ms != manifest_sha:
        return ("REFUSED BEFORE COMPARING ANY BYTES: the report was produced "
                "against a manifest with sha256 %s; this bundle's manifest "
                "is %s. The consumer validated a different bundle."
                % (want_ms, manifest_sha))
    prov = report.get("provenance") or {}
    missing = [k for k in _REPORT_PROVENANCE_REQUIRED if not prov.get(k)]
    if missing:
        return ("REFUSED: the report carries no provenance for: %s. A result "
                "with no provenance is a number, not evidence "
                "(docs/VERIFY.md), and that rule does not soften for other "
                "people's runs." % ", ".join(missing))
    return None


def _validate_report(args, manifest, manifest_sha, differ, out):
    report = _read_report(args.report, out)
    if report is None:
        return EXIT_USAGE
    refusal = _refuse_report(report, manifest, manifest_sha, out)
    if refusal is not None:
        for ln in refusal.split("\n"):
            out.append(ln)
        out.append("=" * 74)
        out.append("RESULT: COULD NOT JUDGE. exit %d" % EXIT_USAGE)
        out.append("=" * 74)
        return EXIT_USAGE

    fx = None
    for cand in manifest.get("fixtures", []):
        if cand.get("id") == report.get("fixture"):
            fx = cand
            break
    if fx is None:
        out.append("COULD NOT JUDGE: the report names fixture %r; the bundle "
                   "carries %s."
                   % (report.get("fixture"),
                      ", ".join(repr(f.get("id"))
                                for f in manifest.get("fixtures", []))))
        return EXIT_USAGE

    out.append("  report    %s" % os.path.abspath(args.report))
    prov = report.get("provenance", {})
    out.append("  reported by %s on %s, %s, %s"
               % (prov.get("implementation"), prov.get("device"),
                  prov.get("os"), prov.get("date")))
    out.append("-" * 74)

    by_tag = {}
    failures = []
    for st in report.get("stages", []):
        tag = st.get("tag")
        if tag in by_tag:
            out.append("COULD NOT JUDGE: the report holds tag %r twice. Tags "
                       "are unique within a trace (the writer raises on a "
                       "duplicate); a report that repeats one cannot be "
                       "aligned." % tag)
            return EXIT_USAGE
        by_tag[tag] = st

    schema = fx.get("stage_schema", [])
    schema_tags = set(st["tag"] for st in schema)
    n_ok = 0
    for st in schema:
        rst = by_tag.get(st["tag"])
        if rst is None:
            failures.append("MISSING STAGE: %s (seq %d) -- the report does "
                            "not answer for it" % (st["tag"], st["seq"]))
            continue
        if rst.get("dtype") != st["dtype"] or rst.get("count") != st["count"]:
            failures.append(
                "STRUCTURAL MISMATCH AT %s: bundle %s x %d, report %s x %s. "
                "The two implementations built different amounts of work "
                "here; chasing bytes before resolving that wastes the "
                "investigation." % (st["tag"], st["dtype"], st["count"],
                                    rst.get("dtype"), rst.get("count")))
            continue
        sha = rst.get("sha256")
        fnv = rst.get("fnv1a64")
        if not (isinstance(sha, str) and _SHA256_RE.match(sha)):
            failures.append("MALFORMED REPORT STAGE %s: sha256 %r is not 64 "
                            "lowercase hex" % (st["tag"], sha))
            continue
        if not (isinstance(fnv, str) and _FNV_RE.match(fnv)):
            failures.append("MALFORMED REPORT STAGE %s: fnv1a64 %r is not 16 "
                            "lowercase hex (required so `conformance diff` "
                            "can localize through the one comparator)"
                            % (st["tag"], fnv))
            continue
        if sha != st["sha256"]:
            failures.append(
                "NOT CONFORMANT AT %s (seq %d):\n"
                "    expected sha256 %s\n"
                "    reported sha256 %s\n"
                "    `conformance diff %s --report %s` names the FIRST "
                "diverging stage."
                % (st["tag"], st["seq"], st["sha256"], sha,
                   args.bundle, args.report))
            continue
        n_ok += 1
    for tag in sorted(t for t in by_tag if t not in schema_tags):
        failures.append("EXTRA STAGE IN REPORT: %r -- the bundle's schema "
                        "does not contain it; the implementation took a "
                        "different path or invented a checkpoint" % tag)

    if failures:
        out.append("NOT CONFORMANT: %d of %d stages agree; %d failure(s)."
                   % (n_ok, len(schema), len(failures)))
        for f in failures:
            out.append("")
            for ln in f.split("\n"):
                out.append("  " + ln)
        out.append("=" * 74)
        out.append("RESULT: FAIL. exit %d" % EXIT_FAIL)
        out.append("=" * 74)
        return EXIT_FAIL

    out.append("CONFORMANT: all %d stages agree (SHA-256 over raw "
               "little-endian bytes)." % len(schema))
    out.append("")
    out.append("WHAT THIS MEANS, EXACTLY: the reporting implementation "
               "reproduced the")
    out.append("expected bytes of THESE fixtures under THIS profile (%s)."
               % manifest.get("profile"))
    out.append("It does NOT mean 'bit-identical AI', does not cover stages "
               "the trace")
    out.append("does not checkpoint, and says nothing about any other input, "
               "size, or")
    out.append("algorithm. docs/CONFORMANCE.md, 'What a PASS means'.")
    out.append("=" * 74)
    out.append("RESULT: PASS. exit %d" % EXIT_OK)
    out.append("=" * 74)
    return EXIT_OK


# --------------------------------------------------------------------------
# DIFF
# --------------------------------------------------------------------------


def cmd_diff(args):
    """First-divergence localization, through the one comparator.

    --report R.json   synthesize a card from the report's per-stage FNV-1a64
                      column and run tools/identity_trace_diff.py against
                      the bundle's card. Same verdict vocabulary: RESULT:
                      IDENTICAL / DIVERGENT, FIRST DIVERGENCE names a stage.
    --self            synthesize a card from the bundle's own expected/ raw
                      bytes and diff it against the bundle's card. This is
                      the corruption witness: flip one byte under expected/
                      and this command names the stage it lives in. Digest-
                      only stages carry no bytes to re-hash and are NAMED as
                      uncovered rather than silently skipped (DEVIATION 930).
    """
    lines = []
    resolved = _load_differ_or_none(lines)
    if resolved is None:
        _emit(lines)
        return EXIT_USAGE
    differ, _dp, _dh = resolved

    if bool(args.report) == bool(args.self):
        _emit(["USAGE: pass exactly one of --report REPORT.json or --self."])
        return EXIT_USAGE

    manifest, manifest_sha = _read_manifest(args.bundle, lines)
    if manifest is None:
        _emit(lines)
        return EXIT_USAGE

    fixtures = manifest.get("fixtures", [])
    if not fixtures:
        _emit(lines + ["COULD NOT JUDGE: the manifest lists no fixtures."])
        return EXIT_USAGE
    fx = fixtures[0]
    if args.report:
        report = _read_report(args.report, lines)
        if report is None:
            _emit(lines)
            return EXIT_USAGE
        refusal = _refuse_report(report, manifest, manifest_sha, lines)
        if refusal is not None:
            lines.append(refusal)
            _emit(lines)
            return EXIT_USAGE
        for cand in manifest.get("fixtures", []):
            if cand.get("id") == report.get("fixture"):
                fx = cand
                break
        else:
            _emit(lines + ["COULD NOT JUDGE: the report names fixture %r, "
                           "which this bundle does not carry."
                           % report.get("fixture")])
            return EXIT_USAGE
        synth_records, problem = _records_from_report(report)
        label_b = "IMPL"
    else:
        synth_records, problem = _records_from_expected(args.bundle, fx,
                                                        differ, lines)
        label_b = "EXPECTED"
    if problem is not None:
        _emit(lines + [problem])
        return EXIT_USAGE

    card_rel = fx.get("trace", "")
    card_full = os.path.join(args.bundle, card_rel.replace("/", os.sep))
    if not os.path.isfile(card_full):
        _emit(lines + ["COULD NOT JUDGE: the bundle's card %s is missing."
                       % card_rel])
        return EXIT_USAGE

    tmpdir = tempfile.mkdtemp(prefix="mojolearn-conformance-diff-")
    synth = os.path.join(tmpdir, "synthesized.card")
    with open(synth, "w", encoding="utf-8") as fh:
        fh.write("# synthesized by `python -m mojolearn conformance diff` "
                 "at %s\n" % _utc())
        fh.write("# source: %s\n"
                 % (os.path.abspath(args.report) if args.report
                    else "expected/ bytes of " + os.path.abspath(args.bundle)))
        for seq, tag, dtype, count, hsh in synth_records:
            fh.write("%d\t%s\t%s\t%d\t%s\n" % (seq, tag, dtype, count, hsh))

    if lines:
        _emit(lines)
    # THE COMPARISON IS THE DIFFER'S, verdict and all (DEVIATION 924). Its
    # exit codes are already this command's: 0 identical, 1 divergent,
    # 2 parse. Nothing is added on top of its report except the synth path.
    rc = differ.run([card_full, synth, "--labels", "CARD,%s" % label_b,
                     "--no-verify-dumps"] + (["--all"] if args.all else []))
    _emit(["synthesized card kept at %s" % synth])
    return rc


def _records_from_report(report):
    """(records, problem): records as (seq, tag, dtype, count, fnv-hex),
    renumbered 0..n-1 in report order so the differ's seq invariant holds
    (alignment is by TAG, so renumbering loses nothing)."""
    recs = []
    for i, st in enumerate(report.get("stages", [])):
        fnv = st.get("fnv1a64")
        if not (isinstance(fnv, str) and _FNV_RE.match(fnv)):
            return None, ("COULD NOT JUDGE: report stage %r carries no "
                          "well-formed fnv1a64 (%r); the localization diff "
                          "needs the same checksum the cards carry."
                          % (st.get("tag"), fnv))
        try:
            count = int(st["count"])
        except (KeyError, TypeError, ValueError):
            return None, ("COULD NOT JUDGE: report stage %r has no integer "
                          "count." % st.get("tag"))
        recs.append((i, st.get("tag", "?"), st.get("dtype", "?"), count, fnv))
    if not recs:
        return None, "COULD NOT JUDGE: the report holds no stages."
    return recs, None


def _records_from_expected(bundle, fx, differ, out):
    """Synthesize records by re-hashing the bundle's own expected/ raw bytes.

    Digest-only stages (kind == sha256) hold no bytes to FNV. They are
    COPIED THROUGH from the schema -- the same fnv the card records -- so
    the tag alignment stays whole, and they are named as UNCOVERED, because
    a copied-through hash cannot witness a corruption; that is `validate`'s
    SHA-256 check's job (DEVIATION 930)."""
    recs = []
    uncovered = []
    n_raw = 0
    for st in fx.get("stage_schema", []):
        if st["kind"] != "raw":
            uncovered.append(st["tag"])
            recs.append((len(recs), st["tag"], st["dtype"], st["count"],
                         st["fnv1a64"]))
            continue
        full = os.path.join(bundle, st["expected"].replace("/", os.sep))
        if not os.path.isfile(full):
            return None, ("COULD NOT JUDGE: expected file for stage %s is "
                          "missing (%s); run `conformance validate` first."
                          % (st["tag"], st["expected"]))
        with open(full, "rb") as fh:
            data = fh.read()
        recs.append((len(recs), st["tag"], st["dtype"], st["count"],
                     "%016x" % differ.fnv1a64(data)))
        n_raw += 1
    if uncovered:
        out.append("NOTE: %d digest-only stage(s) carry no raw bytes and are "
                   "COPIED THROUGH, not re-hashed -- --self cannot witness a "
                   "corruption in them (that is `conformance validate`'s "
                   "SHA-256 check): %s"
                   % (len(uncovered), ", ".join(uncovered)))
    if n_raw == 0:
        return None, ("COULD NOT JUDGE: every stage is digest-only; --self "
                      "has nothing to re-hash.")
    return recs, None
