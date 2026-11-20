# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""PyPI upload limits per Mojolearn project, from packaging/pypi_limits.json.

Build tooling only (the packers, tools/one_layout.py, the release plan); never
imported by the installed package. Standard library only, so the Linux packer,
the macOS shrink step and plain `python3` tools can all load it by path:

    import pypi_limits                      # with packaging/ on sys.path
    pypi_limits.file_limit("mojolearn-nvidia")      # 262144000
    pypi_limits.check_wheels({"mojolearn": [path, ...]})

Distribution names are normalized the PyPI way (case, `_`, `.` -> `-`), so a
wheel's file name prefix (`mojolearn_nvidia`) finds its project.
"""
import json
import os
import re
from pathlib import Path

CONFIG = Path(__file__).resolve().parent / "pypi_limits.json"
SCHEMA = "mojolearn.pypi-limits.v1"
MiB = 1024 ** 2


def normalize(name):
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def load(path=None):
    cfg = json.loads(Path(path or os.environ.get("MOJOLEARN_PYPI_LIMITS") or CONFIG).read_text())
    if cfg.get("schema") != SCHEMA:
        raise ValueError(f"{path or CONFIG}: schema {cfg.get('schema')!r}, want {SCHEMA}")
    return cfg


def project(distribution, cfg=None):
    """The limits row of DISTRIBUTION with the defaults filled in."""
    cfg = cfg or load()
    want = normalize(distribution)
    row = next((dict(v) for k, v in cfg["projects"].items() if normalize(k) == want), None)
    known = row is not None
    row = dict(cfg["defaults"], **(row or {}))
    row.update(distribution=want, configured=known)
    return row


def file_limit(distribution, cfg=None):
    return int(project(distribution, cfg)["file_bytes"])


def project_limit(distribution, cfg=None):
    return int(project(distribution, cfg)["project_bytes"])


def wheel_distribution(path):
    """`mojolearn_nvidia-0.8.65-...whl` -> `mojolearn-nvidia`."""
    return normalize(Path(path).name.split("-", 1)[0])


def check_sizes(sizes, cfg=None, used=None):
    """Rows for {distribution: [bytes of each new file]}; `used` is {distribution:
    bytes already on PyPI}. A row is over when any file exceeds the per-file
    limit or used + new exceeds the project limit."""
    cfg = cfg or load()
    rows = []
    for dist, files in sorted(sizes.items()):
        p = project(dist, cfg)
        biggest = max(files) if files else 0
        total = (used or {}).get(normalize(dist))
        row = dict(distribution=p["distribution"], files=len(files), biggest_bytes=biggest,
                   file_limit_bytes=int(p["file_bytes"]), project_limit_bytes=int(p["project_bytes"]),
                   file_over=biggest > int(p["file_bytes"]), configured=p["configured"],
                   pending=[x.get("issue") for x in p.get("pending", [])])
        if total is not None:
            row.update(project_used_bytes=total, project_after_bytes=total + sum(files),
                       project_over=total + sum(files) > int(p["project_bytes"]))
        row["over"] = row["file_over"] or bool(row.get("project_over"))
        rows.append(row)
    return rows


def check_wheels(paths, cfg=None, used=None):
    sizes = {}
    for p in paths:
        sizes.setdefault(wheel_distribution(p), []).append(Path(p).stat().st_size)
    return check_sizes(sizes, cfg, used)


def line(row):
    return (f"{'OVER' if row['over'] else 'ok  '} {row['distribution']}: biggest file "
            f"{row['biggest_bytes'] / MiB:.1f} MiB (limit {row['file_limit_bytes'] / MiB:.0f} MiB)"
            + (f", project {row['project_after_bytes'] / 2**30:.2f} GiB after upload (limit "
               f"{row['project_limit_bytes'] / 2**30:.0f} GiB)" if "project_after_bytes" in row else "")
            + (f"; pending {', '.join(row['pending'])}" if row["pending"] else ""))
