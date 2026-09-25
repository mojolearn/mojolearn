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

THE HOST (CPU) BINDINGS ARE A SIXTH LIST, AND IT IS NOT ALLOWED TO BE A LIST
(the packaging lane, 2026-09-14). Until 0.8.5 the byte LM's host binding was
the only one in a wheel and its name was spelled by hand in seven pack,
build, smoke and admission files. Since 0.8.6 every host family ships, the
one declaration is `wheel_bindings()` in python/mojolearn/host_surface.py
(the CPU surface manifest), and every one of those files READS it. This
checker holds that: the manifest must ship every binding
`_backend._HOST_MODULES` routes and the three loaded by path, each reader
must carry the manifest token it reads by, and no reader may carry a host
name list of its own. It prints what it matched, file by file, and runs
without a built binary (`--host` runs only this part; the manifest imports
nothing from the package).
"""
import ast
import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: (path, the token that proves the file reads the manifest). A shell script
#: reads it by running the manifest with a flag; a Python file through the
#: admission module's `wheel_host_bindings()` / `wheel_host_members()` or
#: the package's `host_surface.wheel_bindings()`.
HOST_READERS = [
    ("packaging/linux/pack_wheel.py", "wheel_host_bindings()"),
    ("packaging/linux/build_sets.sh", "host_surface.py --wheel-families"),
    ("packaging/linux/build_sets.sh", "host_surface.py --wheel-bindings"),
    ("packaging/macos/build_release_wheel.sh", "host_surface.py --wheel-families"),
    ("packaging/macos/build_release_wheel.sh", "host_surface.py --wheel-bindings"),
    ("packaging/linux/smoke.py", "host_surface.wheel_bindings()"),
    ("packaging/macos/verify_wheel.sh", "host_surface.wheel_bindings()"),
    ("tools/linux_surface_qualification.sh", "wheel_host_bindings()"),
    ("tools/linux_surface_qualification.sh", "wheel_host_members()"),
    ("tools/linux_surface_qualification.sh", "host_surface.wheel_bindings()"),
    ("tools/check_linux_release_qualification.py", "wheel_host_bindings()"),
    ("tools/verify_linux_surface_qualification.py", "def wheel_host_bindings()"),
]
#: A host name list spelled in a reader: `HOST_NAME(S) = "_mojolearn_..._host"`
#: in shell or Python, or a tuple/list/set literal of host basenames.
HOST_LITERAL = re.compile(
    r"^\s*HOST_NAMES?\s*=\s*[\"\x27(\[{]\s*[\"\x27]?_mojolearn_[a-z_]+_host"
    r"|[\"\x27]_mojolearn_[a-z_]+_host[\"\x27]\s*,\s*[\"\x27]_mojolearn_[a-z_]+_host[\"\x27]",
    re.M)
#: The four host bindings loaded by path rather than routed through
#: `_backend._HOST_MODULES`; the manifest must ship them too.
HOST_BY_PATH = ("_mojolearn_byte_lm_host", "_mojolearn_forest_host", "_mojolearn_tokenizer_host",
                "_mojolearn_neural_host")


def _manifest():
    """python/mojolearn/host_surface.py by path: it imports nothing from
    the package, so this needs no built binary."""
    path = ROOT / "python" / "mojolearn" / "host_surface.py"
    spec = importlib.util.spec_from_file_location("mojolearn_host_surface_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def host_problems(read, routed):
    """Every disagreement about the host bindings, as printable lines.
    `read(rel)` returns a file's text (or None when absent); `routed` is
    the set of basenames `_backend._HOST_MODULES` routes."""
    hs = _manifest()
    shipped = list(hs.wheel_bindings())
    declared = list(hs.bindings())
    out = []
    print(f"host manifest: {len(shipped)} of {len(declared)} declared host bindings ship "
          f"({', '.join(hs.wheel_families())})")
    for name in sorted(set(routed) - set(declared)):
        out.append(f"  UNDECLARED reference route {name}")
    public_routes = {f["binding"] for f in hs.FAMILIES if f["routes"] and f["ships_in_wheel"]}
    for name in sorted(public_routes | set(HOST_BY_PATH)):
        if name not in shipped:
            out.append(f"  MISSING host binding {name} is {'routed by _backend._HOST_MODULES' if name in routed else 'loaded by path'} but ships_in_wheel is False in the manifest")
        else:
            print(f"  OK        manifest ships {name}")
    for rel, token in HOST_READERS:
        text = read(rel)
        if text is None:
            out.append(f"  UNREADABLE {rel}: absent")
            continue
        if token in text:
            print(f"  OK        {rel} reads the manifest: {token!r}")
        else:
            out.append(f"  MISSING   {rel} does not read the manifest by {token!r}; it must, never a copy")
    for rel in sorted({r for r, _ in HOST_READERS}):
        text = read(rel)
        if text is None:
            continue
        for m in HOST_LITERAL.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            out.append(f"  LITERAL   {rel}:{line}: carries a host name list of its own: {m.group(0).strip()!r}")
    return out


def _read_tree(rel):
    p = ROOT / rel
    return p.read_text() if p.exists() else None


def _backend_names(name):
    """Read the declared inventory before native binaries have been built."""
    tree = ast.parse((ROOT / "python/mojolearn/_backend.py").read_text())
    values = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == name
                      for target in node.targets)]
    if len(values) != 1:
        raise ValueError(f"Expected one literal declaration of {name}")
    value = values[0]
    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            and value.func.id == "frozenset" and len(value.args) == 1
            and not value.keywords):
        value = value.args[0]
    names = ast.literal_eval(value)
    if not isinstance(names, (tuple, list, set)) or not all(isinstance(n, str) for n in names):
        raise ValueError(f"Expected a literal string inventory for {name}")
    return set(names)


def truth():
    return _backend_names("_MODULES")


def _backend_tiered():
    return _backend_names("_TIERED")


def _backend_classical_fast(linux=False):
    """`_backend._CLASSICAL_FAST`: fast + identical, never deterministic. The
    Linux lists mirror `_CLASSICAL_FAST_LINUX` (FAST svm is Apple only)."""
    full = set(_backend_names("_CLASSICAL_FAST"))
    return full - set(_backend_names("_APPLE_ONLY_FAST")) if linux else full


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
#: Since 2026-09-25 each pair above is a TRIPLE: FAST_CLASSICAL_NAMES (fast +
#: identical) must equal `_backend._CLASSICAL_FAST` in every file that has an
#: identical-only half.
CLASSICAL_VAR = "FAST_CLASSICAL_NAMES"
#: The Linux admission side (tools/verify_linux_surface_qualification.py)
#: spells the every-tier set on its own because it never imports the package;
#: it must equal `_backend._TIERED` too, or the release legs refuse a correct
#: build as "Incomplete build outputs" (2026-09-10, first 0.8.0 AMD leg).
TIERED_MIRRORS = [
    ("tools/verify_linux_surface_qualification.py", from_python_frozenset, "TIERED"),
]
CLASSICAL_MIRRORS = [
    ("tools/verify_linux_surface_qualification.py", "CLASSICAL_FAST"),
]


def main():
    if "--host" in sys.argv[1:]:
        # The host section alone, from the manifest and the routing table
        # read out of _backend.py's source, so it runs with nothing built.
        source = (ROOT / "python" / "mojolearn" / "_backend.py").read_text()
        routed = set(re.findall(r"_mojolearn_[a-z_]+_host", source))
        problems = host_problems(_read_tree, routed)
        if problems:
            print("\n".join(problems))
            print(f"\nFAILED: {len(problems)} host list problem(s); the manifest is the one list.")
            return 1
        print("\nThe host bindings are read from the manifest everywhere.")
        return 0
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
    classical = set(_backend_classical_fast(linux=True))   # every mirror is Linux admission
    for path, var in CLASSICAL_MIRRORS:
        vs = importlib.util.spec_from_file_location("check_ext_lists_classical", ROOT / path)
        vm = importlib.util.module_from_spec(vs)
        vs.loader.exec_module(vm)
        got = set(getattr(vm, var, set()))
        if got != classical:
            print(f"  MISMATCH  {path} {var} ({len(got)}) is not _backend._CLASSICAL_FAST")
            bad += 1
        else:
            print(f"  OK        {path} {var} ({len(got)}) == _backend._CLASSICAL_FAST")
    # The admission side's identical set, BINDINGS (without the byte LM, which
    # expected_bindings adds). A short BINDINGS refused every complete 0.8.6
    # release build as "Incomplete build outputs" (2026-09-15, six bindings
    # added since 0.8.5 were missing), so it is held to _MODULES here too.
    vspec = importlib.util.spec_from_file_location(
        "check_ext_lists_admission", ROOT / "tools" / "verify_linux_surface_qualification.py")
    vmod = importlib.util.module_from_spec(vspec)
    vspec.loader.exec_module(vmod)
    admitted = set(vmod.BINDINGS)
    if admitted != want:
        bad += 1
        print(f"  MISMATCH  tools/verify_linux_surface_qualification.py BINDINGS ({len(admitted)})")
        if want - admitted:
            print(f"              MISSING (a complete build is refused): {', '.join(sorted(want - admitted))}")
        if admitted - want:
            print(f"              EXTRA (named but not a module): {', '.join(sorted(admitted - want))}")
    else:
        print(f"  OK        tools/verify_linux_surface_qualification.py BINDINGS ({len(admitted)})")
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
            fastc = how(path, CLASSICAL_VAR)
            classical = set(_backend_classical_fast(linux=path.startswith("packaging/linux/")))
            if fastc != classical:
                bad += 1
                print(f"  MISMATCH  {path} {CLASSICAL_VAR} ({0 if fastc is None else len(fastc)}) is not _backend._CLASSICAL_FAST")
                fastc = fastc or set()
            got = got | ident | fastc
            label = f"{var} + {CLASSICAL_VAR} + {ident_var}"
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
    # _backend._HOST_MODULES delegates to this manifest. Keep the inventory
    # check usable before building binaries, as the release checklist requires.
    host_bad = host_problems(_read_tree, set(_manifest().routed_modules().values()))
    if host_bad:
        print("\n".join(host_bad))
        bad += len(host_bad)
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
