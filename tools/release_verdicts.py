#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE RELEASE FAILS SOFT ON HARNESS CHECKS AND HARD ONLY ON VERIFICATION (Andrew 2026-11-08 01:40).

"can we fail soft: failure notifies us unless it is verification, otherwise we proceed and fix it in future;
no hard stop so we get all of the info." 0.8.66 needed five reruns of the rented gates, each stopped by ONE
harness defect (a venv without pytest, a 5 s timer, a torch install timeout, a stale artifact read) while every
verification passed.

CHECK_CLASS below is the ONE table of which check decides what ships (hard) and which is recorded, notified and
fixed in a later lane while the release continues (soft). tools/release.py and tools/release_identity_wheels.py
judge every gate with it: a gate passes when its HARD parts passed, even when its box job exited non-zero because
a SOFT part failed. Each SOFT failure is logged as "SOFT FAIL <vendor> <unit>: <reason>" the moment it is read,
appended once to <release dir>/soft-failures.json, carried into the identity admission record and into the
evidence set's SUMMARY.md ("Soft failures (fix lanes owed)"), and notified (tools/notify.sh when the repository
has one; always ~/mojolearn-evidence/rel<ver>/SOFT-FAILURES.md, which the orchestrator's watcher reads).

`tools/release.py --strict` makes every SOFT check HARD (the behavior before 2026-11-08). A name the table does
not know is HARD: a new check stops the release until someone classifies it.
Tests: tools/test_release_verdicts.py (no box, no network).
"""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
HARD, SOFT = "hard", "soft"
SOFT_FAILURES = "soft-failures.json"
SCHEMA = "mojolearn.release-soft-failures.v1"

#: unit or phase name -> "hard" | "soft". A name "<family>/<x>" not listed itself takes its family's class
#: ("autograd/linear-regression" -> "autograd"); a name not found at all is HARD.
CHECK_CLASS = {
    # ---- HARD: verification; these decide what ships
    "identity": HARD,             # the box job's identity columns recorded from the installed wheel bytes
    "identity-admit": HARD,       # all five columns agree (CPU included); the admitted table ships
    "install-load": HARD,         # check unit: qualify_verifier_wheel on the exact wheel bytes (install, load, verify)
    "shardcheck": HARD,           # the one-GPU multi-device same-bits gate (tools/logical_shard_identity.py)
    "cpu/install": HARD,          # the CPU checks' venv: the wheel's sha256, `import mojolearn`, the COMMIT stamp
    "cpu/correctness": HARD,      # qualify_verifier_wheel --scope cpu-only: it runs verify
    "cpu/column": HARD,           # the cpu identity column shards on GitHub
    "apple/identity": HARD,       # the Apple column from the installed macOS wheel
    "apple/self-test": HARD,      # the macOS wheel's self-test import (macos-smoke)
    "rehearsal/sha256": HARD,     # TestPyPI served exactly the gated bytes
    "publish": HARD,              # the sha256 of what is published = what was admitted
    "record": HARD,               # the release record names the published bytes
    "freeze": HARD,               # no frozen source, nothing to ship
    "build": HARD,                # no wheel, nothing to ship
    "pack": HARD,                 # no wheel, nothing to ship
    # ---- SOFT: harness and smoke checks; recorded, notified, the release continues
    "autograd": SOFT,             # check units autograd/<entry>: the autograd smoke entries
    "adapters": SOFT,             # check unit: the framework adapter smoke
    "runtime": SOFT,              # release_runtime_installed / qualify_native_runtime_installed reports
    "measuresmoke": SOFT,         # the measurement smoke (every recipe once at its smoke shape)
    "cpu/runtime": SOFT,          # the CPU checks' runtime group
    "cpu/autograd": SOFT,         # the CPU checks' autograd group
    "cpu/adapters": SOFT,         # the CPU checks' adapter refusal group
    "cpu/job-exit": SOFT,         # the CPU workflow concluded failure while its HARD parts are present and passed
    "apple/adapters": SOFT,       # gate-apple's adapter smoke
    "apple/runtime": SOFT,        # gate-apple's runtime-apple qualifier
    "box-exit": SOFT,             # a box job's non-zero exit when its HARD parts are present and passed
    "artifacts": SOFT,            # artifact uploads and sizes
    "rehearsal/secondary": SOFT,  # the TestPyPI rehearsal's resolutions and macOS install/import from the index
}


def check_class(name):
    """HARD or SOFT for NAME (CHECK_CLASS; a family prefix; unknown -> HARD)."""
    name = str(name)
    if name in CHECK_CLASS:
        return CHECK_CLASS[name]
    head = name.split("/", 1)[0]
    if head in CHECK_CLASS and "/" in name:
        return CHECK_CLASS[head]
    return HARD


def is_soft(name, strict=False):
    """True when NAME's failure lets the release continue (never under --strict)."""
    return not strict and check_class(name) == SOFT


def rel_name(version):
    """'rel0866' for 0.8.66: the evidence directory the orchestrator's watcher reads."""
    return "rel" + str(version).replace(".", "")


def load(rel):
    """The soft failures recorded under the release dir REL, [] when none."""
    try:
        doc = json.loads((Path(rel) / SOFT_FAILURES).read_text())
    except (OSError, ValueError):
        return []
    return list(doc.get("failures") or []) if isinstance(doc, dict) else []


def first_lines(log, n=12):
    try:
        with open(log, errors="replace") as fh:
            return [line.rstrip("\n")[:240] for _, line in zip(range(n), fh)]
    except (OSError, TypeError):
        return []


def record(rel, version, vendor, name, reason, log=None, say=print, evidence_root=None):
    """Append one SOFT failure to REL/soft-failures.json unless it is there already (same vendor, name and
    reason); log "SOFT FAIL <vendor> <name>: <reason>" and notify when it is new. True when new."""
    if rel is None:
        return False
    rel = Path(rel)
    failures = load(rel)
    reason = str(reason)[:600]
    if any(f.get("vendor") == vendor and f.get("name") == name and f.get("reason") == reason for f in failures):
        return False
    entry = dict(name=name, vendor=vendor, reason=reason, log=str(log) if log else None,
                 first_lines=first_lines(log) if log else [],
                 at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    failures.append(entry)
    rel.mkdir(parents=True, exist_ok=True)
    tmp = rel / (SOFT_FAILURES + ".tmp-%d" % os.getpid())
    tmp.write_text(json.dumps(dict(schema=SCHEMA, version=version, failures=failures), indent=1, sort_keys=True) + "\n")
    tmp.replace(rel / SOFT_FAILURES)
    say("SOFT FAIL %s %s: %s" % (vendor, name, reason))
    notify(version, failures, entry, say=say, evidence_root=evidence_root)
    return True


def markdown(version, failures):
    lines = ["# Soft failures of release %s (fix lanes owed)" % version, "",
             "The release continued past each of these (tools/release_verdicts.py CHECK_CLASS); "
             "`tools/release.py --strict` stops on them instead.", ""]
    if not failures:
        return "\n".join(lines + ["None.", ""])
    lines += ["| vendor | unit | reason | log |", "|---|---|---|---|"]
    for f in failures:
        lines.append("| %s | %s | %s | %s |" % (f.get("vendor"), f.get("name"),
                                              str(f.get("reason", "")).replace("|", "/").replace("\n", " ")[:300],
                                              f.get("log") or ""))
    return "\n".join(lines + [""])


def notify(version, failures, entry, say=print, evidence_root=None):
    """tools/notify.sh "<message>" when the repository has one; always the SOFT-FAILURES.md the watcher reads."""
    root = Path(evidence_root or os.environ.get("MOJOLEARN_EVIDENCE_ROOT") or os.path.expanduser("~/mojolearn-evidence"))
    md = root / rel_name(version) / "SOFT-FAILURES.md"
    try:
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(markdown(version, failures))
    except OSError as error:
        say("SOFT FAIL notice not written to %s: %s" % (md, error))
    hook = ROOT / "tools" / "notify.sh"
    if hook.is_file():
        message = "SOFT FAIL %s %s %s: %s" % (version, entry.get("vendor"), entry.get("name"), entry.get("reason"))
        try:
            subprocess.run(["bash", str(hook), message[:900]], timeout=60, capture_output=True)
        except (OSError, subprocess.SubprocessError) as error:
            say("SOFT FAIL notify hook failed: %s" % error)
    return md
