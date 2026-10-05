#!/usr/bin/env python3
"""Trim old run evidence from the tracked tree.

Evidence roots (bench/results, bench/evidence) hold one directory or file per
run. Most of it is old and every worktree pays for it. This tool keeps, per
evidence group:

  * the newest KEEP (3) runs,
  * every run with a file committed in the last 24 hours,
  * every run a tracked doc or tool names by path,
  * the canonical boards linked from the README "Benchmark boards" section
    (and docs/apple-fast/BOARD_*.md, which live outside the roots),

and removes the rest with `git rm`. Before removing anything it appends the
removed paths, each with its last commit sha, date and size, to
bench/results/ARCHIVE_INDEX.md. Restore one with
`git checkout <sha> -- <path>`.

Grouping, worked out from the real layout:

  * A run is a child (dir or file) whose name carries a date token:
    2026-09-14, 20261001, 2026-09-15_172413, ...
  * A group is any directory under a root (the root included) with more than
    KEEP dated children. Its members are its dated children; undated
    children (lowbit_int15/, resume/, README.md) are not members.
  * The walk recurses into undated children and into kept members, so nested
    groups (identity_break/<date>/, runpod_cpu/<date>/, ...) are found.
    Removed members are not walked.
  * Newest = the run's date token, then its last commit time.

A reference protects a member when a source file contains the member's
`<parent>/<name>` (which matches full and relative paths), or its bare name
when that name has a distinctive non-date part. Sources are tracked text
files outside the roots, plus text files inside the roots that stay; the
check repeats until no new member is protected. A path that names only the
group directory protects nothing below it.

Usage:
  tools/evidence_trim.py --dry-run     # summary only: count, bytes
  tools/evidence_trim.py               # write ARCHIVE_INDEX.md, git rm
  tools/evidence_trim.py --list        # dry run, plus one line per removal
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

ROOTS = ("bench/results", "bench/evidence")
KEEP = 3
RECENT_SECONDS = 24 * 3600
INDEX = "bench/results/ARCHIVE_INDEX.md"
# Always kept even if the README section moves or cannot be parsed.
CANONICAL_FIXED = (
    "bench/results/bench_board/m3ultra-0834",
    "bench/results/board-archive",
    INDEX,
)
MAX_SOURCE_BYTES = 2 * 1024 * 1024
# Docs and tools are always read; other text files only when small, so data
# files (oracles, corpora) do not count as references.
SOURCE_EXTS = (".md", ".rst", ".py", ".sh", ".yml", ".yaml", ".toml", ".mojo", ".cfg")
SMALL_TEXT_BYTES = 64 * 1024

DATE_RE = re.compile(
    r"(?<!\d)(20\d\d)-?(0[1-9]|1[0-2])-?(0[1-9]|[12]\d|3[01])"
    r"(?:[_T-]?(\d{6})(?!\d))?"
)


def git(*args: str, input_text: str | None = None) -> str:
    return subprocess.run(
        ("git",) + args, check=True, capture_output=True, text=True,
        input=input_text,
    ).stdout


def date_key(name: str) -> str | None:
    m = DATE_RE.search(name)
    if not m:
        return None
    return m.group(1) + m.group(2) + m.group(3) + (m.group(4) or "000000")


def distinctive(name: str) -> bool:
    rest = re.sub(r"[^A-Za-z]", "", DATE_RE.sub("", name.rsplit(".", 1)[0]))
    return len(rest) >= 6


def canonical_paths() -> set[str]:
    out = set(CANONICAL_FIXED)
    try:
        readme = open("README.md", encoding="utf-8").read()
    except OSError:
        return out
    m = re.search(r"^## Benchmark boards\s*$(.*?)(?=^## |\Z)", readme, re.M | re.S)
    if not m:
        return out
    for link in re.findall(r"\]\(([^)#\s]+)", m.group(1)):
        link = link.lstrip("./")
        if any(link.startswith(r + "/") for r in ROOTS):
            # A linked board page protects its whole board directory.
            out.add(os.path.dirname(link))
    return out


class Tree:
    def __init__(self) -> None:
        self.size: dict[str, int] = {}
        for line in git("ls-tree", "-r", "-l", "-z", "HEAD", "--", *ROOTS).split("\0"):
            if not line:
                continue
            meta, path = line.split("\t", 1)
            parts = meta.split()
            if parts[1] != "blob":
                continue
            self.size[path] = int(parts[3]) if parts[3] != "-" else 0
        self.children: dict[str, set[str]] = defaultdict(set)
        self.is_dir: set[str] = set()
        for path in self.size:
            parts = path.split("/")
            for i in range(1, len(parts)):
                parent = "/".join(parts[:i])
                child = "/".join(parts[: i + 1])
                self.children[parent].add(child)
                if i + 1 < len(parts):
                    self.is_dir.add(child)
        self.files_under: dict[str, list[str]] = defaultdict(list)
        for path in self.size:
            parts = path.split("/")
            for i in range(1, len(parts) + 1):
                self.files_under["/".join(parts[:i])].append(path)

    def bytes(self, p: str) -> int:
        return sum(self.size[f] for f in self.files_under.get(p, ()))


def last_commits(files: set[str]) -> dict[str, tuple[int, str]]:
    """Newest (commit time, short sha) touching each tracked file."""
    out: dict[str, tuple[int, str]] = {}
    proc = subprocess.Popen(
        ["git", "log", "-m", "--first-parent", "--format=\x01%ct %h",
         "--name-only", "--no-renames",
         "-z", "HEAD", "--", *ROOTS],
        stdout=subprocess.PIPE, text=True,
    )
    cur = (0, "")
    assert proc.stdout is not None
    for tok in proc.stdout.read().split("\0"):
        for piece in tok.split("\n"):
            if not piece:
                continue
            if piece.startswith("\x01"):
                ct, sha = piece[1:].split()
                cur = (int(ct), sha)
            elif piece in files and piece not in out:
                out[piece] = cur
        if len(out) == len(files):
            break
    proc.stdout.close()
    proc.kill()
    proc.wait()
    return out


def read_text(path: str) -> str:
    try:
        limit = MAX_SOURCE_BYTES if path.endswith(SOURCE_EXTS) else SMALL_TEXT_BYTES
        if os.path.getsize(path) > limit:
            return ""
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return ""
    if b"\0" in data[:8192]:
        return ""
    return data.decode("utf-8", "replace")


TOKEN_RE = re.compile(r"[A-Za-z0-9_.\-/]+")


def add_tokens(text: str, pairs: set[str], singles: set[str]) -> None:
    """Index path segments: every single segment and adjacent pair."""
    for tok in TOKEN_RE.findall(text):
        if "/" not in tok and not DATE_RE.search(tok):
            continue
        segs = [s.strip(".,:;-") for s in tok.split("/")]
        segs = [s for s in segs if s and s not in (".", "..")]
        singles.update(segs)
        pairs.update(a + "/" + b for a, b in zip(segs, segs[1:]))


def outside_files() -> list[str]:
    return [
        n for n in git("ls-files", "-z").split("\0")
        if n and n != INDEX and not any(n == r or n.startswith(r + "/") for r in ROOTS)
    ]


def plan(now: float):
    tree = Tree()
    commits = last_commits(set(tree.size))
    canon = canonical_paths()

    def covers_canon(p: str) -> bool:
        return any(c == p or c.startswith(p + "/") or p.startswith(c + "/") for c in canon)

    def newest(p: str) -> tuple[int, str]:
        return max((commits.get(f, (0, "")) for f in tree.files_under[p]), default=(0, ""))

    pairs: set[str] = set()
    singles: set[str] = set()

    def is_referenced(p: str) -> bool:
        parent, name = p.rsplit("/", 1)
        if parent.rsplit("/", 1)[-1] + "/" + name in pairs:
            return True
        return distinctive(name) and name in singles

    for f in outside_files():
        add_tokens(read_text(f), pairs, singles)
    referenced: set[str] = set()
    reasons: dict[str, str] = {}

    def walk() -> tuple[list[str], set[str]]:
        removed: list[str] = []
        kept_files: set[str] = set()
        reasons.clear()
        stack = [r for r in ROOTS if r in tree.children]
        while stack:
            d = stack.pop()
            if any(d == c or d.startswith(c + "/") for c in canon):
                kept_files.update(tree.files_under[d])
                continue
            kids = sorted(tree.children[d])
            dated = [k for k in kids if date_key(k.rsplit("/", 1)[-1])]
            members = dated if len(dated) > KEEP else []
            mset = set(members)
            ranked = sorted(
                members,
                key=lambda k: (date_key(k.rsplit("/", 1)[-1]), newest(k)[0]),
                reverse=True,
            )
            keep = set(ranked[:KEEP])
            for k in members:
                why = ("newest" if k in keep else
                       "recent" if now - newest(k)[0] < RECENT_SECONDS else
                       "referenced" if k in referenced else
                       "canonical" if covers_canon(k) else None)
                if why:
                    keep.add(k)
                    reasons[k] = why
            for k in kids:
                if k in mset and k not in keep:
                    removed.append(k)
                elif k in tree.is_dir:
                    stack.append(k)
                else:
                    kept_files.add(k)
        return removed, kept_files

    # Fixed point: text inside kept evidence counts as a source too.
    seen_inside: set[str] = set()
    while True:
        removed, kept_files = walk()
        new_files = {f for f in kept_files if f not in seen_inside and f != INDEX}
        seen_inside |= new_files
        for f in sorted(new_files):
            add_tokens(read_text(f), pairs, singles)
        grew = False
        for p in removed:
            if p not in referenced and is_referenced(p):
                referenced.add(p)
                grew = True
        if not grew:
            return tree, sorted(removed), commits, reasons


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return str(n)


def write_index(tree: Tree, removed: list[str], commits, now: float) -> None:
    stamp = dt.datetime.fromtimestamp(now, dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    head = git("rev-parse", "--short", "HEAD").strip()
    lines = []
    if not os.path.exists(INDEX):
        lines += [
            "# Archived evidence",
            "",
            "`tools/evidence_trim.py` removed these paths from the tree. Each is in git",
            "history at its listed sha: `git checkout <sha> -- <path>` restores it.",
            "",
        ]
    lines += [
        f"## Trim of {stamp} (tree at {head})",
        "",
        "| path | last commit | date | size |",
        "|---|---|---|---|",
    ]
    for p in removed:
        ct, sha = max(commits.get(f, (0, "")) for f in tree.files_under[p])
        day = dt.datetime.fromtimestamp(ct, dt.timezone.utc).strftime("%Y-%m-%d")
        suffix = "/" if p in tree.is_dir else ""
        lines.append(f"| `{p}{suffix}` | {sha} | {day} | {tree.bytes(p)} |")
    lines.append("")
    with open(INDEX, "a", encoding="utf-8") as fh:
        if os.path.getsize(INDEX) if os.path.exists(INDEX) else 0:
            fh.write("\n")
        fh.write("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="print the summary only")
    ap.add_argument("--list", action="store_true", help="dry run, one line per removal")
    args = ap.parse_args()
    os.chdir(git("rev-parse", "--show-toplevel").strip())

    now = time.time()
    tree, removed, commits, reasons = plan(now)
    total = sum(tree.size.values())
    nbytes = sum(tree.bytes(p) for p in removed)
    nfiles = sum(len(tree.files_under[p]) for p in removed)
    if args.list:
        for p in removed:
            print(f"{tree.bytes(p):>11} remove {p}")
        for p, why in sorted(reasons.items()):
            print(f"{tree.bytes(p):>11} keep-{why} {p}")
    print(f"evidence-trim: remove {len(removed)} runs, {nfiles} files, "
          f"{nbytes} bytes ({human(nbytes)}) of {human(total)} under {', '.join(ROOTS)}; "
          f"keeps {human(total - nbytes)}")
    if args.dry_run or args.list or not removed:
        return 0

    write_index(tree, removed, commits, now)
    git("--literal-pathspecs", "rm", "-r", "-q", "--pathspec-from-file=-", "--pathspec-file-nul",
        input_text="\0".join(removed) + "\0")
    git("add", INDEX)
    return 0


if __name__ == "__main__":
    sys.exit(main())
