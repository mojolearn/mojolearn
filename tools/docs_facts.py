#!/usr/bin/env python3
"""Facts the docs state that the tree already knows. Check them, or rewrite them.

    python3 tools/docs_facts.py --check     # fail if a doc disagrees with the tree
    python3 tools/docs_facts.py --write     # rewrite marked spans from the tree
    python3 tools/docs_facts.py --print     # dump what the tree says

WHY. `tools/check_docs_reference_reality.py` exists because six statements in
this repository were wrong on 2026-09-03 and every one of them was TRUE WHEN
WRITTEN. That checker takes the two classes it can settle without guessing at
meaning: a `pixi run -e` naming an undefined environment, and a backticked
path that is not on disk. It deliberately stops there.

This file takes a third class, and it is the one that goes stale fastest. A
handful of facts have exactly ONE source of truth in the tree and are then
RESTATED in prose, in an install command, in a badge. The version is the worst
of them. On 2026-09-09 the README said "Version 0.6.0 is published on PyPI"
and pinned `pip install mojolearn==0.6.0` while `_version.py` said 0.7.0 and
the CHANGELOG had 0.7.0 published three days earlier. Nothing was lying; the
release moved and the prose did not. The same day the README described
`identical` as opt-in while `_backend.py` had been defaulting to it.

A restated fact needs either a generator or a gate, or it drifts. Prose is
worth writing by hand, so this does not template the README. It marks the few
spans that are pure restatement and can rewrite exactly those:

    Version **<!--fact:version-->0.7.0<!--/fact-->** is published on PyPI

The markers are HTML comments, invisible in every renderer that matters,
GitHub and PyPI included.

## The facts, and where each one actually lives

    version         python/mojolearn/_version.py   __version__
    version_date    CHANGELOG.md                   newest "## X (published D)"
    default_mode    python/mojolearn/_backend.py   requested_mode()'s fallback
    doi             CITATION.cff                   doi:

## What --check settles

  1. The four places that carry the version agree: `_version.py`,
     `python/pyproject.toml`, `CITATION.cff`, and the newest published
     CHANGELOG heading. `_version.py` is the one that wins; it says so itself.
  2. Every marked span in a doc matches the fact it names.
  3. Every `mojolearn==<version>` pin in a tracked doc is the current version.
     These are unmarked on purpose, because an install command should be
     copy-pasteable and a marker inside a fenced block is not.
  4. No doc calls a numeric mode "the default" except the one `_backend.py`
     actually falls back to.

It does not read prose for meaning. A claim about what a measurement showed
still needs a person, and a checker that guesses produces false alarms.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Docs whose marked spans are rewritten and whose pins are checked.
#: `python/README.md` is absent deliberately: it is gitignored and
#: `packaging/macos/build_release_wheel.sh` overwrites it with this README at
#: wheel-build time, so it is an artifact, not a source.
DOCS = ("README.md", "CONTRIBUTING.md", "docs/RELEASE_CHECKLIST.md")

MARKER = re.compile(
    r"<!--fact:(?P<name>[a-z_]+)-->(?P<value>.*?)<!--/fact-->", re.DOTALL
)
MODES = ("fast", "deterministic", "identical")


def _read(rel):
    p = ROOT / rel
    return p.read_text(encoding="utf-8") if p.exists() else ""


def facts():
    """Every fact, read from the one file that owns it."""
    out = {}

    m = re.search(r'^__version__\s*=\s*"([^"]+)"', _read("python/mojolearn/_version.py"), re.M)
    if not m:
        raise SystemExit("docs_facts: no __version__ in python/mojolearn/_version.py")
    out["version"] = m.group(1)

    m = re.search(r"^## (\d+\.\d+\.\d+) \((?:published|unreleased) (\d{4}-\d{2}-\d{2})\)", _read("CHANGELOG.md"), re.M)
    out["changelog_version"] = m.group(1) if m else ""
    out["version_date"] = m.group(2) if m else ""
    published = re.search(r"^## (\d+\.\d+\.\d+) \(published (\d{4}-\d{2}-\d{2})\)", _read("CHANGELOG.md"), re.M)
    out["published_version"] = published.group(1) if published else ""
    out["published_date"] = published.group(2) if published else ""

    m = re.search(
        r'os\.environ\.get\(\s*"MOJOLEARN_NUMERIC_MODE"\s*,\s*"([a-z]+)"',
        _read("python/mojolearn/_backend.py"),
    )
    if not m:
        raise SystemExit("docs_facts: cannot find the MOJOLEARN_NUMERIC_MODE fallback in _backend.py")
    out["default_mode"] = m.group(1)

    m = re.search(r"^doi:\s*(\S+)", _read("CITATION.cff"), re.M)
    out["doi"] = m.group(1) if m else ""

    m = re.search(r'^version:\s*"?([^"\s]+)"?', _read("CITATION.cff"), re.M)
    out["citation_version"] = m.group(1) if m else ""

    m = re.search(r'^version\s*=\s*"([^"]+)"', _read("python/pyproject.toml"), re.M)
    out["pyproject_version"] = m.group(1) if m else ""

    return out


def _sources_agree(f):
    """The version is restated in four files. They have to say one thing."""
    bad = []
    v = f["version"]
    for label, got in (
        ("python/pyproject.toml", f["pyproject_version"]),
        ("CITATION.cff", f["citation_version"]),
        ("CHANGELOG.md newest release heading", f["changelog_version"]),
    ):
        if got and got != v:
            bad.append(f"  {label} says {got}; python/mojolearn/_version.py says {v}")
    return bad


def _mode_claims(text, default_mode):
    """Any doc sentence naming a mode as the default, other than the real one."""
    bad = []
    for mode in MODES:
        if mode == default_mode:
            continue
        pat = re.compile(
            r"`?" + mode + r"`?[^.\n]{0,40}\bis the default\b"
            r"|\bdefaults? (?:to|is) `?" + mode + r"`?",
            re.I,
        )
        for m in pat.finditer(text):
            bad.append(m.group(0).strip())
    return bad


def check():
    f = facts()
    problems = list(_sources_agree(f))

    for rel in DOCS:
        text = _read(rel)
        if not text:
            continue

        for m in MARKER.finditer(text):
            name, got = m.group("name"), m.group("value")
            if name not in f:
                problems.append(f"  {rel}: <!--fact:{name}--> names no known fact")
            elif got != f[name]:
                problems.append(f"  {rel}: {name} marked {got!r}, tree says {f[name]!r}")

        for m in re.finditer(r"mojolearn==(\d+\.\d+\.\d+)", text):
            if m.group(1) != f["version"]:
                problems.append(
                    f"  {rel}: pins mojolearn=={m.group(1)}, current version is {f['version']}"
                )

        #: The concept DOI is restated in a badge URL, a link URL and its link
        #: text. A marker cannot live inside a URL, so match the bare token.
        want = re.search(r"zenodo\.(\d+)", f["doi"])
        if want:
            want = want.group(1)
            for m in re.finditer(r"zenodo\.(\d+)", text):
                if m.group(1) != want:
                    problems.append(
                        f"  {rel}: cites zenodo.{m.group(1)}, CITATION.cff says {f['doi']}"
                    )

        for claim in _mode_claims(text, f["default_mode"]):
            problems.append(
                f"  {rel}: says {claim!r}, but _backend.py falls back to {f['default_mode']!r}"
            )

    if problems:
        print("docs_facts: docs disagree with the tree\n", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        print("\nRun `python3 tools/docs_facts.py --write` for the marked spans;"
              "\nthe pins and the prose claims are edits a person makes.", file=sys.stderr)
        return 1

    marked = sum(len(MARKER.findall(_read(rel))) for rel in DOCS)
    print(f"docs_facts OK: {len(f)} facts, {marked} marked spans, version {f['version']}")
    return 0


def write():
    f = facts()
    changed = 0
    for rel in DOCS:
        p = ROOT / rel
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")

        def sub(m):
            name = m.group("name")
            if name not in f:
                return m.group(0)
            return f"<!--fact:{name}-->{f[name]}<!--/fact-->"

        new = MARKER.sub(sub, text)
        if new != text:
            p.write_text(new, encoding="utf-8")
            print(f"  rewrote {rel}")
            changed += 1
    print(f"docs_facts: {changed} file(s) rewritten")
    return 0


def main(argv):
    mode = argv[1] if len(argv) > 1 else "--check"
    if mode == "--check":
        return check()
    if mode == "--write":
        return write()
    if mode == "--print":
        for k, v in sorted(facts().items()):
            print(f"{k:20} {v}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
