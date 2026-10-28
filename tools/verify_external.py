#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Reproduce a retained identity column using PyPI, local wheels, or an installed package.

This is external verification tooling, not product execution. Local modes never
install from an index or clone/fetch a repository. Source identity is checked
before setting the harness commit witness; it is not inferred from the record.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/mojolearn/mojolearn.git"


def verdict(column_path, diff_path, diff_rc):
    """Require stable output and a clean comparison across every reported part."""
    #: verdict names a summary line may carry and still be a pass. IDENTICAL is
    #: the one that proves anything; N/A is a part the lane declares absent;
    #: NOT-COMPARED is a part no JSON here carries (an older record), which is a
    #: weaker result and not a failure. EVERYTHING ELSE GATES, including
    #: ONE-COLUMN, REFUSED, OWED and every *_MOVED.
    TOLERATED = ("IDENTICAL", "N/A", "NOT-COMPARED")

    col = json.loads(Path(column_path).read_text())
    cells = col["cells"]
    bad = {k: v for k, v in cells.items() if v.get("verdict") != "STABLE"}
    bad2 = {k: v for k, v in cells.items() if v.get("infer_verdict") in ("MOVED", "REFUSED") or v.get("model_verdict") in ("MOVED", "REFUSED", "RELOAD-MOVED")}
    text = Path(diff_path).read_text()
    groups = []
    for m in re.finditer(r"^summary(?: \((?P<group>[^)]*)\))?: (?P<body>.*)$", text, re.M):
        counts = {k: int(v) for k, v in re.findall(r"([A-Za-z][A-Za-z0-9/_-]*)=(\d+)", m.group("body"))}
        groups.append((m.group("group") or "train", counts, m.group("body").strip()))
    problems = []
    if not cells:
        problems.append("this column contains no cells")
    if not groups:
        problems.append("the diff printed no summary line at all")
    train = [c for g, c, _ in groups if g == "train"]
    if not train:
        problems.append("the diff printed no train summary line")
    elif not train[0].get("IDENTICAL", 0):
        problems.append("the diff compared no training cells successfully")
    for group, counts, body in groups:
        off = sorted(k for k, v in counts.items() if v and k not in TOLERATED)
        if off:
            problems.append(f"summary ({group}) carries {', '.join(k + '=' + str(counts[k]) for k in off)}")
        if not body:
            continue                       # nothing of this part was asked at all
        if not counts.get("IDENTICAL") and counts.get("NOT-COMPARED", 0) != sum(counts.values()):
            problems.append(f"summary ({group}) has no IDENTICAL cell: {body}")
    # CELLS THE DIFF ITSELF SAID IT WOULD NOT COMPARE. `identity_break.diff`
    # drops every cell of a lane a column hashed at an older LANE_REVISIONS and
    # announces the count; those cells are absent from the train table by the
    # diff's own decision, not by anything this box did, and the floor below is
    # over what remained. It is printed so that a record which drops most of the
    # column cannot pass quietly.
    label = col.get("vendor") or ""
    note = re.search(r"^NOTE: column " + re.escape(label) +
                     r" hashed .* its (\d+) cell\(s\) there are not compared", text, re.M)
    dropped = int(note.group(1)) if (label and note) else 0
    if dropped:
        print(f"NOTE: the diff dropped {dropped} of this column's {len(cells)} cells as hashed at an "
              "older lane revision; they are not compared and prove nothing")
    if train and train[0].get("IDENTICAL", 0) != len(cells) - dropped:
        problems.append(f"the diff read {train[0].get('IDENTICAL', 0)} cells IDENTICAL, but this column "
                        f"has {len(cells)} cells and the diff dropped {dropped} of them as stale; every "
                        "cell it did not drop has to be compared and read IDENTICAL")
    if not col.get("complete", True):
        problems.append("this column is INCOMPLETE: the run was killed and the lanes after the "
                        "last one written are absent, not clean")
    if col.get("partial_column"):
        problems.append("this column is PARTIAL: it left out "
                        f"{', '.join(col.get('parts_omitted') or ['(unnamed)'])}")
    if diff_rc != 0:
        problems.append(f"the diff itself exited {diff_rc}")
    summary = "; ".join(f"{g}: {b or '(empty)'}" for g, _, b in groups) or "no summary line"
    print(f"column: cells={len(cells)} not-stable={len(bad)} infer/model refused or moved={len(bad2)}; "
          f"diff: {summary}; diff exit {diff_rc}")
    for k, v in sorted(bad.items()):
        print(f"NOT STABLE {k}: {v.get('verdict')} {v.get('error', '')[:160]}")
    for k, v in sorted(bad2.items()):
        print(f"PROBE {k}: infer={v.get('infer_verdict')} model={v.get('model_verdict')} {v.get('probe_error', '')[:160]}")
    for p in problems:
        print("DIFF NOT CLEAN: " + p)
    ok = not bad and not bad2 and not problems
    print("VERDICT: " + ("PASS, every cell of this column is stable and equals the committed columns" if ok else "FAIL, see the lines above"))
    return 0 if ok else 1


def record_metadata(directory, requested_version=None):
    """Select column JSONs, ignoring summary/manifest JSONs beside them."""
    columns, commits, versions = [], set(), set()
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not isinstance(data.get("cells"), dict):
            continue
        if not data.get("vendor"):
            continue
        commit = data.get("commit", "")
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError(f"column has no full source commit: {path}")
        columns.append(path)
        commits.add(commit)
        version = data.get("package", {}).get("version")
        if version:
            versions.add(version)
    if not columns:
        raise ValueError(f"no identity column JSONs in {directory}")
    if len(commits) != 1 or len(versions) > 1:
        raise ValueError("record columns must name one source commit and package version")
    version = requested_version or next(iter(versions), None)
    if not version or not re.fullmatch(r"[0-9][A-Za-z0-9.!+_-]*", version):
        raise ValueError("record has no valid package version; supply the version explicitly")
    if versions and version not in versions:
        raise ValueError("requested version differs from the record's package version")
    return next(iter(commits)), version, columns


def source_witness(source, commit):
    """Check an existing checkout without changing it, or an exported witness."""
    if not (source / "tools/identity_break.py").is_file():
        raise ValueError(f"source has no tools/identity_break.py: {source}")
    if (source / ".git").exists():
        actual = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            text=True).strip()
        if dirty:
            raise ValueError("source checkout has tracked modifications; supply the frozen source")
        basis = "git HEAD (no tracked modifications)"
    else:
        witnesses = [p for p in (source / "COMMIT", source / "commit.txt") if p.is_file()]
        if not witnesses:
            raise ValueError("source archive needs a COMMIT or commit.txt witness")
        values = {p.read_text().strip() for p in witnesses}
        if len(values) != 1:
            raise ValueError("source archive commit witnesses disagree")
        actual = values.pop()
        basis = "archive commit witness (not an attestation of source bytes)"
    if actual != commit:
        raise ValueError(f"source commit {actual!r} differs from record commit {commit}")
    return {"commit": actual, "basis": basis,
            "harness_sha256": hashlib.sha256(
                (source / "tools/identity_break.py").read_bytes()).hexdigest()}


def run_logged(command, log, *, env=None, cwd=None):
    with log.open("w") as output:
        output.write("command: " + json.dumps([str(arg) for arg in command]) + "\n")
        output.flush()
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                env=env, cwd=cwd)
    print(f"{log.stem}: exit {result.returncode}; log={log}", flush=True)
    return result.returncode


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", help="machine label, e.g. apple-m4 or nvidia-rtx4090-sm_89")
    parser.add_argument("record", nargs="?", type=Path, help="retained identity-column directory")
    parser.add_argument("version", nargs="?", help="defaults to the version in the record")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--wheelhouse", type=Path,
                      help="install only from this local wheel directory (no package index)")
    mode.add_argument("--installed", action="store_true",
                      help="use the active Python environment without installing anything")
    parser.add_argument("--source", type=Path,
                        help="matching checkout or source archive; required for local modes")
    parser.add_argument("--output", type=Path, help="empty directory for retained logs and results")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", args.label):
        parser.error("label must contain only lowercase letters, digits, dots, underscores or hyphens")
    if (args.wheelhouse or args.installed) and not args.source:
        parser.error("--wheelhouse and --installed require --source; no public clone is used")
    if args.record is None:
        records = sorted((ROOT / "bench/results/identity_break").glob("*-lanes"))
        if not records:
            parser.error("no default identity record; supply a record directory")
        args.record = records[-1]
    args.record = args.record.resolve()
    for field in ("source", "wheelhouse"):
        path = getattr(args, field)
        if path is not None:
            path = path.resolve()
            if not path.is_dir():
                parser.error(f"--{field} is not a directory: {path}")
            setattr(args, field, path)
    return args


PACKAGE_PROBE = '''
import json, sys
from pathlib import Path
import numpy
import mojolearn as ml
package = Path(ml.__file__).resolve().parent
witness = package / "identity_columns" / "COMMIT"
data = dict(version=ml.__version__, package_dir=str(package), numpy=numpy.__version__,
            vendor=ml.vendor(), mode=ml.numeric_mode(),
            commit=witness.read_text().strip() if witness.is_file() else None)
Path(sys.argv[1]).write_text(json.dumps(data, indent=2) + "\\n")
if data["version"] != sys.argv[2]:
    raise SystemExit("installed package version differs from the selected record")
if data["commit"] and data["commit"] != sys.argv[3]:
    raise SystemExit("installed package commit witness differs from the selected record")
if data["mode"] != "identical":
    raise SystemExit("installed package did not select identical mode")
'''


def main(argv=None):
    args = arguments(argv)
    try:
        commit, version, columns = record_metadata(args.record, args.version)
        # Validate supplied source before installing anything. Never checkout or
        # rewrite a reviewer's existing tree, and never trust an env override.
        witness = source_witness(args.source, commit) if args.source else None
        from datetime import datetime, timezone
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        out = (args.output or ROOT / "verify_external_out" / f"{args.label}-{stamp}").resolve()
        if out.exists() and any(out.iterdir()):
            raise ValueError(f"output directory is not empty; preserve the prior run: {out}")
        out.mkdir(parents=True, exist_ok=True)
        source = args.source or out / "src"
        mode = "installed" if args.installed else "wheelhouse" if args.wheelhouse else "pypi"
        print(f"record={args.record} commit={commit} version={version} mode={mode}", flush=True)
        if not args.source:
            for command, log in [
                (["git", "clone", "--quiet", REPOSITORY, str(source)], "clone.log"),
                (["git", "-C", str(source), "checkout", "--quiet", commit], "checkout.log"),
            ]:
                rc = run_logged(command, out / log)
                if rc:
                    return rc
            witness = source_witness(source, commit)
        provenance = dict(mode=mode, version=version, source=str(source), source_witness=witness,
                          python=sys.executable, record=str(args.record),
                          record_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in columns})
        (out / "input-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        if not args.installed:
            command = [sys.executable, "-m", "pip", "--isolated", "install",
                       "--disable-pip-version-check"]
            if args.wheelhouse:
                command += ["--no-index", "--find-links", str(args.wheelhouse),
                            "--only-binary=:all:"]
            command += ["numpy", f"mojolearn=={version}"]
            rc = run_logged(command, out / "install.log")
            if rc:
                return rc
        env = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical", MOJOLEARN_COMMIT=commit)
        # Running from the output directory avoids importing a package from the
        # caller's working directory instead of the installed compiled package.
        rc = run_logged([sys.executable, "-c", PACKAGE_PROBE, str(out / "package.json"),
                         version, commit], out / "package.log", env=env, cwd=out)
        if rc:
            return rc
        harness = source / "tools/identity_break.py"
        column, run_log, diff_log = out / f"{args.label}.json", out / "run.log", out / "diff.log"
        rc = run_logged([sys.executable, str(harness), "--vendor", args.label, "--json", str(column)],
                        run_log, env=env, cwd=out)
        if rc:
            return rc
        diff_rc = run_logged([sys.executable, str(harness), "--diff",
                              *map(str, columns), str(column)], diff_log, env=env, cwd=out)
        rc = verdict(column, diff_log, diff_rc)
        print(f"column={column}; provenance={out / 'input-provenance.json'}")
        return rc
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
