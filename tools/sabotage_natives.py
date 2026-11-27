#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inventory the same native files for sabotage wheel staging, scratch builds and restoration."""
import argparse
import os
from pathlib import Path, PurePosixPath
import re
import zipfile


PACKAGE_DIRS = ("mojolearn", "mojolearn.libs")
NATIVE_DIRS = tuple("python/" + name for name in PACKAGE_DIRS)
EXTENSION = re.compile(r"\.so(?:\.[0-9]+)*\Z|\.dylib\Z")


def native_name(name):
    return name == "_mojolearn_merged.json" or bool(EXTENSION.search(name))


def configured_dirs():
    return tuple(dict.fromkeys((*NATIVE_DIRS, *os.environ.get("NATIVE_DIRS", "").split())))


def _raise_walk_error(error):
    raise error


def natives(root, directories=NATIVE_DIRS):
    """Return sorted paths relative to root, preserving explicit absolute backup locations."""
    found = set()
    for directory in directories:
        base = Path(root) / directory
        if not base.exists():  # The sibling dependency directory is optional in a development build.
            continue
        for current, _, files in os.walk(base, onerror=_raise_walk_error):
            for name in files:
                path = Path(current) / name
                if native_name(name) and (path.is_file() or path.is_symlink()):
                    found.add(str(Path(directory) / path.relative_to(base)))
    return sorted(found)


def stage_wheels(root, wheels):
    """Replace staged natives, preserving source files and wheel member permissions."""
    old = natives(root)
    for rel in old:
        (root / rel).unlink()
    placed = 0
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                if (len(path.parts) < 2 or path.parts[0] not in PACKAGE_DIRS
                        or not native_name(path.name) or member.is_dir()):
                    continue
                if ".." in path.parts:
                    raise ValueError(f"invalid wheel path {member.filename!r}")
                extracted = archive.extract(member, root / "python")
                mode = (member.external_attr >> 16) & 0o777
                if mode:
                    os.chmod(extracted, mode)
                placed += 1
    if not placed:
        raise ValueError("no native file in the staged wheels")
    return len(old), placed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("root", type=Path)
    stage = commands.add_parser("stage")
    stage.add_argument("root", type=Path)
    stage.add_argument("wheels", nargs="+")
    args = parser.parse_args()
    if args.command == "list":
        for path in natives(args.root, configured_dirs()):
            print(path)
    else:
        removed, placed = stage_wheels(args.root, args.wheels)
        print(f"removed {removed} native files, placed {placed} from {len(args.wheels)} wheels")


if __name__ == "__main__":
    main()
