#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Move the identity-record HISTORY out of the tree (2026-11-07).

bench/results/identity_break/ holds one dated directory per identity record
round since 2026-09-13, plus `retained/` (the content-addressed live set the
admission keeps). Most dated directories are history nothing reads any more.
This tool lists them and, on order, removes the ones nothing needs with
`git rm -r`, writing one committed line per removed directory to
docs/identity_history_pruned.tsv so each stays retrievable from git.

  prune_identity_history.py --plan  [--root R] [--out TSV]
  prune_identity_history.py --apply [--root R]

A dated directory (a top-level entry of identity_break whose name starts with
YYYY-MM-DD) is KEPT when any of these holds; otherwise it is PRUNE:

  table     a record of the admitted table (the shipped
            python/mojolearn/verify_reference/table.json, and any table-shaped
            JSON under bench/results/release_evidence/) cites a file in it;
            the table regeneration (admit_identity_columns.sh) reads these;
  evidence  a file under bench/results/release_evidence/ names it;
  retained  a file under identity_break/retained/ names it;
  code      a tracked file outside bench/results/ and docs/ (and not *.md)
            names it: a tool, test or workflow still reads it;
  newest    it holds the newest record (by the record's commit time) of a
            column (its vendor label) or of a device class, with retained/
            competing: a byte-identical retained copy wins a tie;
  newest-lane
            it holds the newest complete identical record, by commit time, of
            some (device class, lane): that is what the reuse mechanism
            (identity_columns.reuse_points) adopts unchanged lanes from, so
            pruning it would make an unchanged lane re-record.

`retained/` and every non-dated entry are never touched. --apply re-plans on
the spot (it never acts on an old plan), refuses while tools/release.py runs,
refuses when anything under identity_break or the docs tsv is uncommitted, and
stages the removal and the docs tsv; it does not commit.

Reads only git metadata and the record files; imports no binding.
"""
import argparse
import datetime as _dt
import gzip
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent
REL_BREAK = "bench/results/identity_break"
REL_EVIDENCE = "bench/results/release_evidence"
REL_TABLE = "python/mojolearn/verify_reference/table.json"
REL_DOCS_TSV = "docs/identity_history_pruned.tsv"
DATED = re.compile(r"^\d{4}-\d{2}-\d{2}")
# A token that may be a dated directory name; trailing dots are trimmed.
TOKEN = re.compile(r"20\d{2}-\d{2}-\d{2}[A-Za-z0-9._-]*")
NOT_A_SOURCE = (".admission-preview-", ".other-commit-")  # identity_columns._NOT_A_SOURCE
PLAN_HEADER = ("path", "verdict", "reasons", "files", "bytes", "retrievable_at")
DOCS_HEADER = ("path", "files", "bytes", "retrievable_at", "pruned_on")


def git(root, *args, check=True):
    out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if check and out.returncode != 0:
        raise SystemExit(f"git {' '.join(args[:3])} failed: {out.stderr.strip()[:300]}")
    return out


# ---------------------------------------------------------------- device class
def _fallback_device_class(vendor, filename):
    """Mirror of _verify_reference.device_class, used only when the tree has none."""
    v = (vendor or "").lower()
    base = os.path.basename(filename).lower()
    if v.startswith("cpu") or base.startswith("cpu"):
        return "cpu"
    if v.startswith("nvidia-ptx") or base.startswith("nvidia-ptx"):
        return "ptx"
    for cls in ("apple", "nvidia", "amd"):
        if cls in v or cls in base or (cls == "apple" and v == "arm64" and ("apple" in base or "metal" in base)):
            return cls
    return None


def class_function(root):
    """The tree's own record_device_class (pure stdlib module), else the fallback."""
    path = Path(root) / "python" / "mojolearn" / "_verify_reference.py"
    if path.is_file():
        try:
            spec = importlib.util.spec_from_file_location("_prune_verify_reference", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fn = getattr(mod, "record_device_class", None)
            if fn is not None:
                return lambda rec, name: fn(rec, name)[0]
        except Exception:  # noqa: BLE001  an old tree: fall back, never fail the plan
            pass
    return lambda rec, name: _fallback_device_class(rec.get("vendor"), name)


# ---------------------------------------------------------------- inventory
def tracked_dirs(root):
    """{dated dir name: [files, bytes]} of tracked files at HEAD, plus the set of other entries."""
    out = git(root, "ls-tree", "-r", "-l", "-z", "HEAD", "--", REL_BREAK + "/").stdout
    dated, other = {}, set()
    prefix = REL_BREAK + "/"
    for entry in out.split("\0"):
        if not entry:
            continue
        meta, _, path = entry.partition("\t")
        if not path.startswith(prefix):
            continue
        rest = path[len(prefix):]
        top, sep, _ = rest.partition("/")
        if not sep:
            continue                     # a file at the top level: never ours
        if not DATED.match(top):
            other.add(top)
            continue
        size = meta.split()[3]
        acc = dated.setdefault(top, [0, 0])
        acc[0] += 1
        acc[1] += int(size) if size.isdigit() else 0
    return dated, other


def read_text(path):
    try:
        if path.name.endswith(".gz"):
            with gzip.open(path, "rt", errors="replace") as fh:
                return fh.read()
        return path.read_text(errors="replace")
    except OSError:
        return ""


def tokens(text, names):
    """Dated identity_break dir names that `text` names. A name right after a
    `/` counts only as `identity_break/<name>`: `identity_columns/2026-10-14_0844-pypi`
    is another directory that happens to share a dated name."""
    found = set()
    for m in TOKEN.finditer(text):
        start = m.start()
        if start and text[start - 1] == "/" and not text[max(0, start - 15):start] == "identity_break/":
            continue
        t = m.group(0).rstrip(".")
        if t in names:
            found.add(t)
            continue
        # `<dir>/<file>` and `<dir>.<ext>` style tails: the longest prefix that is a name.
        for cut in range(len(t) - 1, 9, -1):
            if t[cut] in "/._-" and t[:cut] in names:
                found.add(t[:cut])
                break
    return found


def table_files(root):
    """Table JSONs: the shipped table plus any table-shaped JSON in release_evidence."""
    out = []
    shipped = Path(root) / REL_TABLE
    if shipped.is_file():
        out.append(shipped)
    ev = Path(root) / REL_EVIDENCE
    if ev.is_dir():
        for p in sorted(ev.rglob("*.json")):
            out.append(p)
    return out


def table_refs(root, names):
    """{name: [table paths]} for tables whose `records[].dir` cites the name."""
    refs = {}
    seen_table = False
    for path in table_files(root):
        try:
            j = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        recs = j.get("records") if isinstance(j, dict) else None
        if not isinstance(recs, list):
            continue
        seen_table = seen_table or path.name == "table.json"
        for r in recs:
            d = str((r or {}).get("dir") or "")
            top = d.split("/", 1)[0]
            if top in names:
                refs.setdefault(top, set()).add(str(path.relative_to(root)))
    return refs, seen_table


def text_refs(base, names, skip_suffixes=()):
    found = set()
    if base.is_dir():
        for p in base.rglob("*"):
            if p.is_file() and not p.name.endswith(tuple(skip_suffixes)):
                found |= tokens(read_text(p), names)
    return found


# A retained `*.source-report.json.gz` is the frozen bytes of the verifier report a
# record was produced with; it lists every reference input that verifier read (the
# whole history at the time). identity_columns.retained_verifier_evidence rechecks
# only the report's own hash, never the paths inside it, so those mentions are not
# a dependency. The record, its provenance and its verifier-run receipt are read.
RETAINED_NOT_A_REFERENCE = (".source-report.json.gz",)


def code_refs(root, names):
    """Names a tracked non-doc file outside bench/results/ mentions."""
    out = git(root, "grep", "-h", "-o", "-I", "-E", r"20[0-9]{2}-[0-9]{2}-[0-9]{2}[A-Za-z0-9._-]*",
              "--", ".", ":!bench/results", ":!docs", ":!*.md", check=False)
    return tokens(out.stdout, names)


def records_in(base):
    """Every column-record JSON (an identity_break column) under `base`."""
    for p in sorted(base.rglob("*.json")):
        s = str(p)
        if any(tag in s for tag in NOT_A_SOURCE):
            continue
        try:
            j = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(j, dict) and isinstance(j.get("cells"), dict) and j.get("vendor"):
            yield p, j


def newest_refs(root, brk, names, commit_time, device_class):
    """(names that hold a newest column/class record, names that hold a newest (class, lane) record)."""
    best_col, best_lane = {}, {}

    def offer(table, key, rank, owner):
        cur = table.get(key)
        if cur is None or rank > cur[0]:
            table[key] = (rank, owner)

    sources = [(n, brk / n) for n in sorted(names)]
    if (brk / "retained").is_dir():
        sources.append((None, brk / "retained"))   # the live set competes; owner None keeps nothing
    for owner, base in sources:
        fallback = 0
        if owner:
            try:
                fallback = int(_dt.datetime.strptime(owner[:10], "%Y-%m-%d")
                               .replace(tzinfo=_dt.timezone.utc).timestamp())
            except ValueError:
                fallback = 0
        for path, j in records_in(base):
            t = commit_time(j.get("commit")) or fallback
            # A retained copy beats a dated copy at the same time (tie-break 1 > 0).
            rank = (t, 1 if owner is None else 0)
            cls = device_class(j, str(path)) or "unknown"
            offer(best_col, ("vendor", str(j.get("vendor"))), rank, owner)
            offer(best_col, ("class", cls), rank, owner)
            if j.get("mode") == "identical" and j.get("complete") is True:
                for key in j["cells"]:
                    offer(best_lane, (cls, key.partition("/")[0]), rank, owner)
    col = {o for _, o in best_col.values() if o}
    lane = {o for _, o in best_lane.values() if o}
    return col, lane


def make_commit_time(root):
    cache = {}

    def commit_time(commit):
        if not commit or not re.fullmatch(r"[0-9a-f]{7,40}", str(commit)):
            return None
        if commit not in cache:
            out = git(root, "log", "-1", "--format=%ct", commit + "^{commit}", check=False)
            s = out.stdout.strip()
            cache[commit] = int(s) if out.returncode == 0 and s.isdigit() else None
        return cache[commit]
    return commit_time


def plan(root, commit_time=None, device_class=None, head=None):
    """[(path, verdict, reasons, files, bytes, retrievable_at)] sorted by name, plus the other entries."""
    root = Path(root).resolve()
    brk = root / REL_BREAK
    if not brk.is_dir():
        raise SystemExit(f"REFUSING: {brk} is not on disk (a lean worktree? pass --root to a full checkout)")
    dated, other = tracked_dirs(root)
    names = set(dated)
    head = head or git(root, "rev-parse", "HEAD").stdout.strip()
    commit_time = commit_time or make_commit_time(root)
    device_class = device_class or class_function(root)
    tab, seen_table = table_refs(root, names)
    if not seen_table:
        raise SystemExit(f"REFUSING: no admitted table found ({REL_TABLE}); cannot tell what the table cites")
    reasons = {n: [] for n in names}
    for n in tab:
        reasons[n].append("table")
    for n in text_refs(root / REL_EVIDENCE, names):
        reasons[n].append("evidence")
    for n in text_refs(brk / "retained", names, RETAINED_NOT_A_REFERENCE):
        reasons[n].append("retained")
    for n in code_refs(root, names):
        reasons[n].append("code")
    col, lane = newest_refs(root, brk, names, commit_time, device_class)
    for n in col:
        reasons[n].append("newest")
    for n in lane:
        reasons[n].append("newest-lane")
    rows = []
    for n in sorted(names):
        files, size = dated[n]
        verdict = "KEEP" if reasons[n] else "PRUNE"
        rows.append((f"{REL_BREAK}/{n}", verdict, ",".join(reasons[n]) or "-", files, size, head))
    return rows, sorted(other)


def summarize(rows, other):
    keep = [r for r in rows if r[1] == "KEEP"]
    prune = [r for r in rows if r[1] == "PRUNE"]
    lines = [
        f"dated dirs {len(rows)}: KEEP {len(keep)} ({sum(r[3] for r in keep)} files, "
        f"{sum(r[4] for r in keep)} bytes), PRUNE {len(prune)} ({sum(r[3] for r in prune)} files, "
        f"{sum(r[4] for r in prune)} bytes)",
        f"untouched non-dated entries: {', '.join(other) or '-'}",
    ]
    by = {}
    for r in keep:
        for why in r[2].split(","):
            by[why] = by.get(why, 0) + 1
    lines.append("KEEP reasons (a dir may have several): " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())))
    return "\n".join(lines)


def write_tsv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")


def release_running():
    out = subprocess.run(["pgrep", "-f", "tools/release.py"], capture_output=True, text=True)
    me = str(os.getpid())
    return [p for p in out.stdout.split() if p != me]


def apply(root, today):
    root = Path(root).resolve()
    pids = release_running()
    if pids:
        raise SystemExit(f"REFUSING: tools/release.py is running (pid {', '.join(pids)}); prune after the release records")
    dirty = git(root, "status", "--porcelain", "--", REL_BREAK, REL_DOCS_TSV).stdout.strip()
    if dirty:
        raise SystemExit("REFUSING: uncommitted changes under identity_break or the docs tsv; commit them first:\n"
                         + "\n".join(dirty.splitlines()[:10]))
    rows, other = plan(root)
    print(summarize(rows, other))
    prune = [r for r in rows if r[1] == "PRUNE"]
    for r in prune:
        name = r[0][len(REL_BREAK) + 1:]
        # Belt and braces: never the live set, never a non-dated entry, never a nested path.
        if name == "retained" or "/" in name or not DATED.match(name):
            raise SystemExit(f"REFUSING: {r[0]} is not a dated record dir")
    if not prune:
        print("nothing to prune")
        return 0
    paths = [r[0] for r in prune]
    for i in range(0, len(paths), 50):
        git(root, "rm", "-r", "-q", "--", *paths[i:i + 50])
    docs = root / REL_DOCS_TSV
    existing = docs.read_text().splitlines() if docs.is_file() else []
    if not existing:
        existing = ["\t".join(DOCS_HEADER)]
    docs.parent.mkdir(parents=True, exist_ok=True)
    with docs.open("w") as fh:
        fh.write("\n".join(existing) + "\n")
        for r in prune:
            fh.write("\t".join(str(x) for x in (r[0], r[3], r[4], r[5], today)) + "\n")
    git(root, "add", "--", REL_DOCS_TSV)
    print(f"staged: git rm -r of {len(prune)} dirs, {REL_DOCS_TSV} (+{len(prune)} lines); commit them together")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="classify and write the plan tsv; changes nothing")
    mode.add_argument("--apply", action="store_true", help="git rm -r the PRUNE dirs and append the docs tsv")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="checkout holding bench/results/identity_break")
    ap.add_argument("--out", default=None, help="plan tsv (default ~/mojolearn-evidence/identity-history-prune-<date>.tsv)")
    a = ap.parse_args(argv)
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    if a.apply:
        return apply(a.root, today)
    rows, other = plan(a.root)
    out = Path(a.out) if a.out else Path.home() / "mojolearn-evidence" / f"identity-history-prune-{today}.tsv"
    write_tsv(out, PLAN_HEADER, rows)
    print(summarize(rows, other))
    print(f"plan: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
