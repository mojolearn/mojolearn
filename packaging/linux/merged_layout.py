#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The binding layout of one Linux set directory, for the packers and audits.

A Linux set directory (``<vendor>/<arch>/``, ``<vendor>/<arch>/identical/``,
``host/``) ships its bindings in one of two layouts:

  per-binding   one ``<name>.so`` per binding (MOJOLEARN_MERGED_BINDINGS=0,
                the layout every release before the merged bindings shipped)
  merged        one ``<stem>.so`` per group of bindings/merged_groups.json,
                exporting every member's ``PyInit_<name>``, plus the index
                ``_mojolearn_merged.json`` (``{"format": 1, "files":
                {"<stem>.so": [members]}}``) that python/mojolearn/_merged.py
                reads to load each member under its own module name
                (MOJOLEARN_MERGED_BINDINGS=1, the default).

The layout is read from the directory itself (an index present means merged),
never from the switch, so a packer handles either set whatever its own
environment says. A directory never mixes the two: a member's own file would
shadow the merged file at load time (_merged.binding_path), so a set that
carries both is refused here rather than shipping two copies of one module.

The index is shipped and hashed like a binding: the loader trusts it to say
which file holds which module, so every inventory that lists a set's
binaries lists the index beside them (``files_of`` returns it).

Build and packaging glue only (stdlib); it never runs in the library.

  python3 packaging/linux/merged_layout.py enabled
  python3 packaging/linux/merged_layout.py check DIR --names "a b c" [--plan fast|identical|host --byte-lm 0|1]
  python3 packaging/linux/merged_layout.py files DIR        shipped binaries (+ index), one per line
  python3 packaging/linux/merged_layout.py member DIR NAME  the file NAME loads from
  python3 packaging/linux/merged_layout.py members DIR      every member, one per line
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

INDEX = "_mojolearn_merged.json"
INDEX_FORMAT = 1
SWITCH = "MOJOLEARN_MERGED_BINDINGS"
REPO = Path(__file__).resolve().parents[2]


class LayoutError(ValueError):
    """A set directory whose binaries do not match the bindings it must ship."""


def enabled(environ=None):
    """MOJOLEARN_MERGED_BINDINGS: '1' (default) builds and packs merged files,
    '0' the per-binding layout."""
    value = (os.environ if environ is None else environ).get(SWITCH, "1")
    if value not in ("0", "1"):
        raise LayoutError(f"{SWITCH} must be 0 or 1, got {value!r}")
    return value == "1"


def plan(dir_key, byte_lm, platform="linux"):
    """{'<stem>.so': [members]} bindings/merged_groups.json plans for one
    directory set ('fast', 'identical', 'host') on `platform`."""
    sys.path.insert(0, str(REPO / "tools"))
    try:
        import merged_bindings
    finally:
        sys.path.pop(0)
    return {stem + ".so": list(mem) for stem, _cls, mem in merged_bindings.plan(dir_key, platform, bool(byte_lm))}


def read_index(directory):
    """The directory's index {'<stem>.so': [members]}, or None without one."""
    p = Path(directory) / INDEX
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text())
    except ValueError as exc:
        raise LayoutError(f"{p}: not JSON ({exc})") from exc
    files = doc.get("files") if isinstance(doc, dict) else None
    if (doc.get("format") != INDEX_FORMAT or not isinstance(files, dict) or not files
            or not all(isinstance(f, str) and f.endswith(".so") and "/" not in f
                       and isinstance(m, list) and m and all(isinstance(n, str) and n for n in m)
                       for f, m in files.items())):
        raise LayoutError(f"{p}: not a format-{INDEX_FORMAT} merged index with files")
    return {f: list(m) for f, m in files.items()}


def layout(directory, names, glob="_mojolearn*.so"):
    """('merged' | 'per-binding', {'<file>.so': [members]}) for a directory that
    must serve exactly `names`; LayoutError otherwise. `glob` is the pattern
    of the binaries the directory holds (the host set: ``_mojolearn_*_host.so``)."""
    d = Path(directory)
    names = list(names)
    if len(set(names)) != len(names):
        raise LayoutError(f"{d}: duplicate binding names asked for")
    on_disk = {p.name for p in d.glob(glob)} if d.is_dir() else set()
    index = read_index(d)
    if index is None:
        want = {n + ".so" for n in names}
        if on_disk != want:
            raise LayoutError(f"{d}: per-binding set differs from its bindings "
                              f"(missing {sorted(want - on_disk) or '-'}, unexpected {sorted(on_disk - want) or '-'})")
        return "per-binding", {n + ".so": [n] for n in names}
    seen = [n for m in index.values() for n in m]
    if len(seen) != len(set(seen)):
        raise LayoutError(f"{d}/{INDEX}: a member is listed in more than one merged file")
    if set(seen) != set(names):
        raise LayoutError(f"{d}/{INDEX}: members differ from the set's bindings "
                          f"(only indexed {sorted(set(seen) - set(names)) or '-'}, "
                          f"not indexed {sorted(set(names) - set(seen)) or '-'})")
    own = sorted(n + ".so" for n in names if n + ".so" in on_disk and n + ".so" not in index)
    if own:
        raise LayoutError(f"{d}: own files {own} beside the merged index would shadow the merged file")
    if on_disk != set(index):
        raise LayoutError(f"{d}: merged files differ from the index "
                          f"(missing {sorted(set(index) - on_disk) or '-'}, unexpected {sorted(on_disk - set(index)) or '-'})")
    return "merged", index


def check_plan(directory, dir_key, byte_lm, platform="linux"):
    """A merged directory's index must be exactly the manifest's plan."""
    index = read_index(directory)
    if index is None:
        return
    want = {f: sorted(m) for f, m in plan(dir_key, byte_lm, platform).items()}
    got = {f: sorted(m) for f, m in index.items()}
    if want != got:
        raise LayoutError(f"{directory}/{INDEX} differs from the bindings/merged_groups.json plan "
                          f"for {dir_key} (byte LM {int(bool(byte_lm))}): index {got}, plan {want}")


def files_of(directory, names, glob="_mojolearn*.so"):
    """Every file the directory ships for `names`: its binaries, and the index
    when merged (sorted file names)."""
    kind, files = layout(directory, names, glob)
    return sorted(files) + ([INDEX] if kind == "merged" else [])


def member_file(directory, name):
    """The file `name` loads from: its own file, else the merged file the
    index names (the loader's rule, python/mojolearn/_merged.binding_path)."""
    d = Path(directory)
    own = d / (name + ".so")
    if own.exists():
        return own
    for f, m in (read_index(d) or {}).items():
        if name in m:
            return d / f
    return own


def members_of(directory):
    """{member: file name} for every member the directory serves."""
    d = Path(directory)
    index = read_index(d)
    if index is None:
        return {p.stem: p.name for p in sorted(d.glob("_mojolearn*.so"))}
    return {n: f for f, m in index.items() for n in m}


def is_index(path):
    return Path(path).name == INDEX


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("enabled")
    p = sub.add_parser("check")
    p.add_argument("directory")
    p.add_argument("--names", required=True)
    p.add_argument("--glob", default="_mojolearn*.so")
    p.add_argument("--plan", choices=("fast", "identical", "host"))
    p.add_argument("--byte-lm", choices=("0", "1"), default="0")
    p = sub.add_parser("files")
    p.add_argument("directory")
    p = sub.add_parser("member")
    p.add_argument("directory")
    p.add_argument("name")
    p = sub.add_parser("members")
    p.add_argument("directory")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "enabled":
            print("1" if enabled() else "0")
        elif a.cmd == "check":
            kind, files = layout(a.directory, a.names.split(), a.glob)
            if a.plan:
                check_plan(a.directory, a.plan, a.byte_lm == "1")
            print(f"{kind} {len(files)} file(s) serving {sum(len(m) for m in files.values())} binding(s)")
        elif a.cmd == "files":
            index = read_index(a.directory)
            names = sorted(members_of(a.directory)) if index is not None else None
            out = files_of(a.directory, names) if names else sorted(p.name for p in Path(a.directory).glob("_mojolearn*.so"))
            print("\n".join(out))
        elif a.cmd == "member":
            print(member_file(a.directory, a.name))
        elif a.cmd == "members":
            print("\n".join(sorted(members_of(a.directory))))
    except LayoutError as exc:
        print(f"merged_layout: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
