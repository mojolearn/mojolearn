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


def _backend_tiered():
    sys.path.insert(0, str(ROOT / "python"))
    from mojolearn import _backend
    return set(_backend._TIERED)


def from_python_tuple(path, varname):
    """Names inside `VARNAME = ( ... )`, quoted, up to the closing paren."""
    text = (ROOT / path).read_text()
    m = re.search(varname + r"\s*=\s*\((.*?)\n\)", text, re.S)
    if not m:
        return None
    return set(re.findall(r'"(_mojolearn[a-z_]*)"', m.group(1)))


def from_python_frozenset(path, varname):
    """Names inside `VARNAME = frozenset({ ... })`, quoted, single line or not."""
    text = (ROOT / path).read_text()
    m = re.search(varname + r"\s*=\s*frozenset\(\{(.*?)\}\)", text, re.S)
    if not m:
        return None
    return set(re.findall(r"'(_mojolearn[a-z_]*)'", m.group(1)))


def from_shell_string(path, varname):
    """Names inside `VARNAME="a b c"` on one line."""
    text = (ROOT / path).read_text()
    m = re.search(varname + r'="([^"]*)"', text)
    if not m:
        return None
    return set(m.group(1).split())


#: (path, how, every-tier varname, identical-only varname or None). Since
#: DEVIATION 2490 (2026-09-10) the pack and build lists come in PAIRS: the
#: three TREE lanes build in every tier and every other binding in identical
#: only, and each file names both halves. The UNION of a pair must equal
#: `_MODULES`, and the every-tier half must equal `_backend._TIERED`, or a
#: binding has been quietly moved back into three tiers (or out of the wheel).
#: The smoke list is one flat list: it loads every binding under identical.
SOURCES = [
    ("packaging/linux/pack_wheel.py", from_python_tuple, "EXT_NAMES", "IDENTICAL_ONLY_NAMES"),
    ("packaging/linux/smoke.py", from_python_tuple, "ALL_BINDINGS", None),
    ("packaging/linux/build_sets.sh", from_shell_string, "EXT_NAMES", "IDENTICAL_ONLY_NAMES"),
    ("packaging/macos/build_release_wheel.sh", from_shell_string, "EXT_NAMES", "IDENTICAL_ONLY_NAMES"),
]
#: The Linux admission side (tools/verify_linux_surface_qualification.py)
#: spells the every-tier set on its own because it never imports the package;
#: it must equal `_backend._TIERED` too, or the release legs refuse a correct
#: build as "Incomplete build outputs" (2026-09-10, first 0.8.0 AMD leg).
TIERED_MIRRORS = [
    ("tools/verify_linux_surface_qualification.py", from_python_frozenset, "TIERED"),
]


def main():
    want = truth()
    # Byte LM is an explicit IDENTICAL-only profile addition, not a binding
    # required in FAST/DETERMINISTIC. Each pack/build/smoke must still name it.
    profile_only = {"_mojolearn_byte_lm"}
    if not profile_only <= want:
        print("FAIL profile-only binding disappeared from backend")
        return 1
    want -= profile_only
    print(f"source of truth: python/mojolearn/_backend.py _MODULES "
          f"({len(want)} extensions)")
    bad = 0
    tiered = set(_backend_tiered())
    for path, how, var in TIERED_MIRRORS:
        got = how(path, var)
        if got is None:
            print(f"  UNREADABLE {path}: no {var} found")
            bad += 1
        elif got != tiered:
            print(f"  MISMATCH  {path} {var} ({len(got)}) is not _backend._TIERED")
            bad += 1
        else:
            print(f"  OK        {path} {var} ({len(got)}) == _backend._TIERED")
    for path, how, var, ident_var in SOURCES:
        got = how(path, var)
        if got is None:
            print(f"  UNREADABLE {path}: no {var} found -- the parser and the "
                  f"file have diverged, which is its own defect")
            bad += 1
            continue
        label = var
        if ident_var is not None:
            ident = how(path, ident_var)
            if ident is None:
                print(f"  UNREADABLE {path}: no {ident_var} found")
                bad += 1
                continue
            if got != tiered:
                bad += 1
                print(f"  MISMATCH  {path} {var} ({len(got)}) is not _backend._TIERED")
                print(f"              every-tier list must be exactly: {', '.join(sorted(tiered))}")
            got = got | ident
            label = f"{var} + {ident_var}"
        text = (ROOT / path).read_text()
        for name in profile_only:
            if name not in text:
                print(f"  MISSING profile-specific binding {name}: {path}")
                bad += 1
        missing = sorted(want - got)
        extra = sorted(got - want)
        if not missing and not extra:
            print(f"  OK        {path} {label} ({len(got)})")
            continue
        bad += 1
        print(f"  MISMATCH  {path} {label} ({len(got)})")
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
