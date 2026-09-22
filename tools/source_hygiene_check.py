#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SOURCE PATTERNS THAT ARE ALWAYS A BUG, kept at zero.

    python3 tools/source_hygiene_check.py              # fail on any match
    python3 tools/source_hygiene_check.py --self-test  # prove each pattern FIRES

One pattern today, from the 2026-09-16 discarded-atomic audit. An atomic whose
result is thrown away.

    _ = Atomic.load(...)
    _ = Atomic.compare_exchange(...)

`Atomic.load` exists to READ a value and `compare_exchange` exists to say
whether it won. Discarding either emits nothing on any column, so the line is
dead on every vendor while looking like synchronization; the audit measured
that on Apple a discarded atomic leaves the same AIR as no atomic at all.
There is no correct use of these two spellings, which is what makes a grep the
right tool here rather than a judgement call.

A CHECK THAT CANNOT FAIL IS NOT A CHECK, and this one could not.

  * `git grep -E` is POSIX ERE and has NO `\\s`. The pattern the audit wrote
    down, `'^\\s*_ = Atomic\\.(load|compare_exchange)'`, matches NOTHING through
    `git grep -E`: it exits 1, prints nothing, and reads exactly like a clean
    tree. Measured 2026-09-16 against a file carrying both spellings. The
    patterns below use `[[:space:]]`.
  * The first invocation here was `git grep -nE -- REGEX -- GLOB`, where the
    first `--` ends the options and the pattern is read as a PATH. Same silent
    nothing.

Both were caught by running the check against a probe that carries the
pattern, so `--self-test` does that every time, through the REAL `git grep`
rather than through Python's regex engine, which accepts different syntax and
would have hidden the first of those two. It also requires the near miss
beside the probe NOT to match, so a pattern cannot pass by matching
everything. Matches are printed, never counted.
"""
import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: name, POSIX ERE, path globs, lines that MUST match, lines that must NOT.
PATTERNS = (
    (
        "discarded atomic",
        r"^[[:space:]]*_[[:space:]]*=[[:space:]]*Atomic\.(load|compare_exchange)",
        ("*.mojo",),
        ("    _ = Atomic.load(ptr)",
         "        _ = Atomic.compare_exchange(ptr, expected, desired)",
         "_=Atomic.load(ptr)"),
        ("    seen = Atomic.load(ptr)",
         "        won = Atomic.compare_exchange(ptr, expected, desired)",
         "    _ = Atomic.fetch_add(ptr, 1)",
         "    # _ = Atomic.load(ptr) in a comment is prose, not code"),
    ),
)


def _grep(regex, paths, no_index=False):
    """The matching lines. `git grep` exits 1 for no match and 2 or more for a
    broken invocation, and the two must never be confused."""
    cmd = ["git", "-C", ROOT, "grep", "-nE"]
    if no_index:
        cmd.append("--no-index")
    cmd += ["-e", regex, "--"] + list(paths)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode not in (0, 1):
        raise SystemExit(f"REFUSING: git grep failed ({res.returncode}): {res.stderr.strip()}")
    return [line for line in res.stdout.split("\n") if line]


def self_test():
    """Every pattern must FIRE on each probe and stay quiet on each near miss,
    through the real invocation."""
    bad = []
    for name, regex, globs, probes, near_misses in PATTERNS:
        suffix = globs[0].lstrip("*") if globs else ".mojo"
        fh = tempfile.NamedTemporaryFile("w", suffix=suffix, dir=ROOT, delete=False)
        with fh:
            for line in list(probes) + list(near_misses):
                fh.write(line + "\n")
        rel = os.path.relpath(fh.name, ROOT)
        try:
            hits = _grep(regex, [rel], no_index=True)
        finally:
            os.unlink(fh.name)
        found = {line.split(":", 2)[-1] for line in hits}
        for probe in probes:
            if probe not in found:
                bad.append(f"{name}: the probe {probe.strip()!r} did NOT match, so a clean answer "
                           "from this pattern would mean nothing")
        for miss in near_misses:
            if miss in found:
                bad.append(f"{name}: the near miss {miss.strip()!r} matched, so the pattern is "
                           "too wide to act on")
    for line in bad:
        print(f"SELF-TEST FAIL: {line}")
    print(f"self-test {'OK' if not bad else 'FAILED'}: {len(PATTERNS)} pattern(s)")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true",
                    help="prove each pattern fires on a line that carries it, and stop")
    args = ap.parse_args(argv)
    if self_test():
        return 1
    if args.self_test:
        return 0
    failed = False
    for name, regex, globs, _probes, _near_misses in PATTERNS:
        for hit in _grep(regex, globs):
            print(f"FAIL {name}: {hit}")
            failed = True
    print(f"source hygiene {'OK' if not failed else 'FAILED'}: "
          f"{len(PATTERNS)} pattern(s) over the tracked tree")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
