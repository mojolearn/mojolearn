#!/usr/bin/env python3
"""FIVE LISTS NAME THE SHIPPED EXTENSIONS AND NOTHING MADE THEM AGREE.

    python3 packaging/check_ext_lists.py

WHY THIS EXISTS. The same defect shipped three times, and twice in one day:

  * `packaging/linux/pack_wheel.py`'s EXT_NAMES carried THIRTEEN of fifteen,
    so the 0.4.0 Linux wheel shipped `_mamba_impl.py` and
    `_transformer_impl.py` with NO `.so` behind either, on all six
    architecture sets. The legs had built them -- every readback.txt lists
    both -- and the pack loop only raises on a name IN the tuple that is
    missing on disk, so a name absent from the tuple is never looked for and
    never missed.
  * `packaging/linux/smoke.py`'s ALL_BINDINGS carried the same thirteen, so
    the release smoke never LOADED either extension.
  * `packaging/macos/smoke.py` reached thirteen too, so nothing ever
    LAUNCHED them.

Each list was correct when written and none was updated when the fourteenth
and fifteenth bindings landed. That is not a mistake anyone makes once: it is
what happens when five copies of one fact have no check between them.

`python/mojolearn/_backend.py`'s `_MODULES` IS THE SOURCE OF TRUTH, because
it is the list the RUNNING LIBRARY resolves imports through -- a name missing
there is broken for users immediately and loudly, which is what keeps it
honest. Every other list is checked against it.

Exit is non-zero on any disagreement, naming the file and the missing or
extra names. Wire it into the release path ahead of any build.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def truth():
    sys.path.insert(0, str(ROOT / "python"))
    from mojolearn import _backend
    return set(_backend._MODULES)


def from_python_tuple(path, varname):
    """Names inside `VARNAME = ( ... )`, quoted, up to the closing paren."""
    text = (ROOT / path).read_text()
    m = re.search(varname + r"\s*=\s*\((.*?)\n\)", text, re.S)
    if not m:
        return None
    return set(re.findall(r'"(_mojolearn[a-z_]*)"', m.group(1)))


def from_shell_string(path, varname):
    """Names inside `VARNAME="a b c"` on one line."""
    text = (ROOT / path).read_text()
    m = re.search(varname + r'="([^"]*)"', text)
    if not m:
        return None
    return set(m.group(1).split())


#: (path, how, varname). Each is a list that must name every shipped
#: extension; a short one is a wheel that ships without that extension, or a
#: gate that never looks at it.
SOURCES = [
    ("packaging/linux/pack_wheel.py", from_python_tuple, "EXT_NAMES"),
    ("packaging/linux/smoke.py", from_python_tuple, "ALL_BINDINGS"),
    ("packaging/linux/build_sets.sh", from_shell_string, "EXT_NAMES"),
    ("packaging/macos/build_release_wheel.sh", from_shell_string, "EXT_NAMES"),
]


def main():
    want = truth()
    print(f"source of truth: python/mojolearn/_backend.py _MODULES "
          f"({len(want)} extensions)")
    bad = 0
    for path, how, var in SOURCES:
        got = how(path, var)
        if got is None:
            print(f"  UNREADABLE {path}: no {var} found -- the parser and the "
                  f"file have diverged, which is its own defect")
            bad += 1
            continue
        missing = sorted(want - got)
        extra = sorted(got - want)
        if not missing and not extra:
            print(f"  OK        {path} {var} ({len(got)})")
            continue
        bad += 1
        print(f"  MISMATCH  {path} {var} ({len(got)})")
        if missing:
            print(f"              MISSING (ships without a gate): {', '.join(missing)}")
        if extra:
            print(f"              EXTRA (named but not a module): {', '.join(extra)}")
    if bad:
        print(f"\nFAILED: {bad} list(s) disagree with _backend._MODULES.")
        print("A short list does not raise on its own -- it simply never looks")
        print("for the name it is missing. That is how 0.4.0 shipped two")
        print("extensions with no .so behind them.")
        return 1
    print("\nAll extension lists agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
