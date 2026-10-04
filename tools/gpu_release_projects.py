#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Classify GPU release projects from the shared registry before dispatch/upload.

This does not replace byte/provenance admission. It closes the workflow's
project allow-list and rejects experimental payloads even on subset dispatches.
"""
from __future__ import annotations

import argparse
import email.parser
import importlib.util
import json
import os
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("gpu_release_registry", ROOT / "python/mojolearn/gpu_plugins.py")
gpu_plugins = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gpu_plugins)


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_prefix(profile):
    if profile == gpu_plugins.CORE_PROFILE:
        return "mojolearn"
    try:
        row = gpu_plugins.package(profile)
    except (KeyError, StopIteration) as exc:
        raise ValueError(f"unknown GPU release profile: {profile}") from exc
    if not row["release_enabled"]:
        raise ValueError(f"experimental GPU payload cannot publish: {profile}")
    return row["wheel_name"]


def classify(wheels, *, version=None, require_complete=False):
    rows = {r["wheel_name"]: r for r in gpu_plugins.distribution_rows(include_experimental=True)}
    roles = {"vendor": set(), "payload": set()}
    profiles, versions, core_wheels = set(), set(), []
    paths = sorted(map(Path, wheels))
    if not paths:
        raise ValueError("no wheels to classify")
    for path in paths:
        parts = path.name.split("-")
        if len(parts) not in (5, 6) or not path.name.endswith(".whl"):
            raise ValueError(f"invalid wheel filename: {path.name}")
        prefix, wheel_version = parts[:2]
        if prefix == "mojolearn":
            distribution = "mojolearn"
            core_wheels.append(path.name)
        else:
            row = rows.get(prefix)
            if row is None:
                raise ValueError(f"unknown GPU release project: {path.name}")
            wheel_prefix(row["profile"])  # experimental projects are never admitted
            if row["profile"] in profiles:
                raise ValueError(f"duplicate GPU project in candidate: {row['profile']}")
            profiles.add(row["profile"])
            roles[row["role"]].add(row["profile"])
            distribution = row["distribution"]
        with zipfile.ZipFile(path) as archive:
            metadata = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            expected_metadata = f"{prefix}-{wheel_version}.dist-info/METADATA"
            if metadata != [expected_metadata]:
                raise ValueError(f"unexpected distribution metadata in {path.name}")
            message = email.parser.BytesParser().parsebytes(archive.read(metadata[0]))
            platform_tags = parts[-1][:-4].split(".")
            linux = any(tag.startswith(("linux_", "manylinux", "musllinux")) for tag in platform_tags)
            if prefix == "mojolearn" and linux:
                marker_name = f"{prefix}-{wheel_version}.dist-info/{gpu_plugins.CORE_MARKER}"
                try:
                    marker = json.loads(archive.read(marker_name))
                except (KeyError, ValueError):
                    marker = None
                if marker != gpu_plugins.core_marker(wheel_version):
                    raise ValueError(f"Linux core requires the current split-package marker: {path.name}")
                for member in archive.namelist():
                    gpu_member = gpu_plugins.member_vendor(member) is not None
                    flat_native = (member.startswith("mojolearn/") and member.endswith(".so")
                                   and not member.startswith(("mojolearn/host/", "mojolearn/.libs/")))
                    if gpu_member or flat_native:
                        raise ValueError(f"Linux core carries a GPU payload member: {member}")
        if (normalize(message.get("Name", "")) != distribution
                or message.get("Version") != wheel_version):
            raise ValueError(f"filename and metadata disagree: {path.name}")
        if version is not None and wheel_version != version:
            raise ValueError(f"expected version {version}, found {path.name}")
        versions.add(wheel_version)
    if len(versions) != 1:
        raise ValueError("candidate wheel versions disagree")
    if require_complete:
        expected = {r["profile"] for r in gpu_plugins.distribution_rows()}
        if profiles != expected or len(core_wheels) != 1:
            raise ValueError("complete Linux release requires one core and both native vendor packages")
    return dict(core=bool(core_wheels), plugins=sorted(roles["vendor"]),
                payloads=sorted(roles["payload"]), version=next(iter(versions)))


def github_outputs(result):
    return "".join(f"{key}={json.dumps(result[key], separators=(',', ':'))}\n"
                   for key in ("core", "plugins", "payloads"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?")
    parser.add_argument("--version")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--github-output", action="store_true")
    parser.add_argument("--wheel-prefix", metavar="PROFILE")
    args = parser.parse_args(argv)
    try:
        if args.wheel_prefix:
            print(wheel_prefix(args.wheel_prefix))
            return 0
        if args.directory is None:
            parser.error("directory required unless --wheel-prefix is used")
        result = classify(args.directory.glob("*.whl"), version=args.version,
                          require_complete=args.require_complete)
        if args.github_output:
            with open(os.environ["GITHUB_OUTPUT"], "a") as output:
                output.write(github_outputs(result))
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"gpu_release_projects: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
