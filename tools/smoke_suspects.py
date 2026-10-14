#!/usr/bin/env python3
"""Suspects-only smoke (Andrew 2026-10-14): `gpu_gap_campaign.py --smoke --smoke-suspects <prev smoke.json>[,...]`
runs only the recipes that were not known good in a previous smoke of the same vendor, plus every recipe whose
code changed since that smoke's source_commit. Everything skipped is recorded with the smoke and commit it was
known good in. Without the flag the smoke runs every recipe.

Known good: MATCHED, or UNSUPPORTED with a named reason ("no supported own arm ..."). Everything else is a
suspect: REFUSED, STARTED, FAILED, an UNSUPPORTED without a named reason, or a recipe the previous smoke did
not run.

Changed code: `git diff --name-only <source_commit> HEAD`. A change to shared smoke/board/parameter/fairness
code, to the library (python/, bindings/, any .mojo), or to any Python file outside tools/ and bench/speed/
re-runs everything; a change to a family driver, or a tools module a family driver imports (statically, or
through the drivers' _load/_tool/_sibling/_load_tool/_module helpers), re-runs that family. Tests, docs and
results never count. When the diff cannot be computed (unknown commit), everything runs.
"""
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent.parent

FAMILY_ROOTS = {
    "classical": ("tools/classical_two_datasets.py",),
    "classical2": ("tools/bench_board_more.py",),
    "algos": ("tools/bench_board_algos.py", "tools/bench_board_extra.py"),
    "neural": ("tools/bench_board_neural.py",),
    "trees": ("bench/speed/forest_speed_arm.py", "tools/speed_gbdt_arm.py"),
}
#: a change here can change any recipe's check
SHARED = {"gpu_gap_campaign", "smoke_worker", "smoke_inproc", "smoke_suspects", "bench_board", "bench_board_params",
          "race_fairness", "bench_board_evidence", "race_scorer_preflight", "board_pairing", "bench_board_probe",
          "cpu_quota", "bench_board_store", "pypi_wheel_install", "install_mamba_ssm"}
_IMPORT = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", re.M)
_DYNAMIC = re.compile(r"(?:_load|_tool|_sibling|_load_tool|_module|_mod)\(\s*[\"']([A-Za-z_]\w*)[\"']")


def _module_paths(root=ROOT):
    """{module name: repo-relative path} for tools/*.py and bench/speed/*.py."""
    out = {}
    for d in ("tools", "bench/speed"):
        for p in sorted((root / d).glob("*.py")):
            out.setdefault(p.stem, "%s/%s" % (d, p.name))
    return out


def import_graph(root=ROOT):
    """{path: {paths it imports}} among tools/ and bench/speed/ modules."""
    mods = _module_paths(root)
    graph = {}
    for name, rel in mods.items():
        try:
            text = (root / rel).read_text(errors="replace")
        except OSError:
            continue
        names = set(_IMPORT.findall(text)) | set(_DYNAMIC.findall(text))
        graph[rel] = {mods[n] for n in names if n in mods and mods[n] != rel}
    return graph


def _reach(graph, starts):
    seen, todo = set(), list(starts)
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        todo.extend(graph.get(p, ()))
    return seen


def ignored(path):
    name = os.path.basename(path)
    return (name.startswith("test_") or path.startswith(("docs/", "docs-site/", "bench/results/", "experiments/"))
            or path.endswith((".md", ".txt", ".json", ".tsv", ".csv", ".patch")))


def families_for_changes(changed, root=ROOT):
    """(everything reason or None, {family: [changed paths]}, {path: why ignored})."""
    graph = import_graph(root)
    reach = {fam: _reach(graph, roots) for fam, roots in FAMILY_ROOTS.items()}
    shared_reach = _reach(graph, [p for p in _module_paths(root).values() if Path(p).stem in SHARED])
    fams, skipped = {}, {}
    for path in sorted(changed):
        if ignored(path):
            skipped[path] = "test, doc or result file"
            continue
        if path.startswith(("python/", "bindings/")) or path.endswith(".mojo") or os.path.basename(path) in (
                "pixi.toml", "pixi.lock", "pyproject.toml", "setup.py"):
            return "library or environment changed: %s" % path, {}, skipped
        if not path.endswith(".py"):
            skipped[path] = "not Python"
            continue
        if not path.startswith(("tools/", "bench/speed/")):
            return "Python outside tools/ and bench/speed/ changed (may be imported by a driver): %s" % path, {}, skipped
        if Path(path).stem in SHARED:
            return "shared smoke/board/parameter/fairness code changed: %s" % path, {}, skipped
        hit = [fam for fam, seen in reach.items() if path in seen]
        if hit:
            for fam in hit:
                fams.setdefault(fam, []).append(path)
        elif path in shared_reach:
            return "a module the shared board code imports changed: %s" % path, {}, skipped
        else:
            skipped[path] = "no smoke path imports it"
    return None, fams, skipped


def changed_files(since, root=ROOT):
    """Repo-relative paths changed between `since` and HEAD, or None when the diff cannot be computed."""
    if not since:
        return None
    try:
        p = subprocess.run(["git", "-C", str(root), "diff", "--name-only", since, "HEAD"], capture_output=True,
                           text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if p.returncode != 0:
        return None
    return [l.strip() for l in p.stdout.splitlines() if l.strip()]


def known_good_status(row):
    status = row.get("status")
    if status == "MATCHED":
        return True
    if status == "UNSUPPORTED":
        return any(str(p).startswith("no supported own arm") and "(" in str(p) for p in row.get("problems") or ())
    return False


def load_previous(paths, vendor):
    """[(path, doc)] of the previous smokes of `vendor` (others are listed as ignored)."""
    got, ignored_paths = [], []
    for path in [p for p in (paths or "").split(",") if p.strip()]:
        doc = json.loads(Path(path).read_text())
        (got if doc.get("vendor") == vendor else ignored_paths).append((path, doc))
    return got, ignored_paths


def select(plans, previous_paths, vendor, root=ROOT):
    """(plans of suspects only, record for smoke.json)."""
    prev, other = load_previous(previous_paths, vendor)
    record = dict(previous=[dict(path=p, source_commit=d.get("source_commit"), rows=len(d.get("rows") or {}))
                            for p, d in prev],
                  ignored_other_vendor=[dict(path=p, vendor=d.get("vendor")) for p, d in other],
                  skipped_known_good={}, ran={}, everything=None, families_changed={}, changed_files={})
    if not prev:
        record["everything"] = "no previous smoke of vendor %s given" % vendor
        return plans, record
    good = {}
    for path, doc in prev:              # later files win for a recipe they both ran
        for key, row in (doc.get("rows") or {}).items():
            good[key] = (known_good_status(row), row.get("status"), path, doc.get("source_commit"))
    commits = sorted({d.get("source_commit") for _, d in prev if d.get("source_commit")})
    fams = {}
    for c in commits:
        files = changed_files(c, root)
        if files is None:
            record["everything"] = "cannot diff from the previous smoke's commit %s" % c
            break
        record["changed_files"][c] = files
        why, f, _ = families_for_changes(files, root)
        if why:
            record["everything"] = why
            break
        for fam, paths in f.items():
            fams.setdefault(fam, set()).update(paths)
    if not commits and record["everything"] is None:
        record["everything"] = "the previous smoke records no source_commit"
    record["families_changed"] = {k: sorted(v) for k, v in sorted(fams.items())}
    if record["everything"]:
        return plans, record
    out = {}
    for mode, plan in plans.items():
        keep = []
        for race in plan:
            key = "%s|%s" % (mode, race["id"])
            ok, status, path, commit = good.get(key, (False, None, None, None))
            if race["family"] in fams:
                record["ran"][key] = "code changed: %s" % ", ".join(sorted(fams[race["family"]]))[:300]
                keep.append(race)
            elif not ok:
                record["ran"][key] = ("not run in the previous smoke" if status is None
                                      else "%s in %s" % (status, path))
                keep.append(race)
            else:
                record["skipped_known_good"][key] = dict(status=status, smoke=path, source_commit=commit)
        out[mode] = keep
    return out, record
