#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""IDENTITY MATH LINT: raw stdlib math in product Mojo, where IDENTICAL must call the exact seams.

    python3 tools/identity_math_lint.py [--commit <sha>] [--root DIR] [--list]

WHY (lane fb13-identity-audit, 2026-11-17). Each backend computes stdlib transcendentals its own way: NVIDIA lowers
Float32 sqrt/log2/exp2/cos/tanh to approximate PTX, AMD to V_SQRT/V_LOG/V_EXP, Apple to Metal's precise (not
correctly rounded) intrinsics, the host to LLVM/Mojo stdlib code. The house rule (IDENTITY_PATHS rows 9-10,
checks/numerics.mojo): IDENTICAL arithmetic goes through the seams of checks/numerics.mojo -- identical_sqrt,
identical_exp, identical_log, identical_log1p, identical_exp64, identical_log64, identical_log2_64, portable_* --
which keep the stdlib for FAST only. The compiled-kernel audits (packaging/linux/ptx_contract.py, hip_contract.py,
packaging/macos/air_contract.py) catch a raw call that reaches an IDENTICAL GPU kernel, after a build; this lint
catches it before one, and also on the host column, which no kernel audit reads.

WHAT IT FLAGS. A call of a std.math function in NAMES (imported from `std.math`/`math`, or `math.<name>(`) in a
tracked .mojo file outside bench/, experiments/, docs/, the verify checks and oracles (`checks/` directories,
`*oracle*` files: references, not product) and checks/numerics.mojo itself (the seam library; its raw calls are
the FAST arms). Comments, docstrings and string literals are ignored.

THE ALLOW LIST (tools/identity_math_lint_allow.txt): one line per (file, enclosing function, op) that is safe,
`<path>:<function>:<op>  # <CLASS>: <reason>`, CLASS one of SAFE-FAST (reachable only in FAST), SAFE-EXACT (exact
on every backend for the values it sees, or feeds only an integer/layout decision), SAFE-NOT-SHIPPED (in no
product binding). Unlisted sites fail; allow lines that match nothing are printed as STALE (and fail, so the list
never outlives the code). sqrt is in NAMES: it is correctly rounded on the host and Apple but not on NVIDIA/AMD.

THE MODE POLICY TABLE (checks/mode_policy.mojo, 2026-11-19). The seams of checks/numerics.mojo keep the stdlib under
the table's math row: `comptime if not ROW_VENDOR_MATH:` is the exact arm, the code after it the vendor's. The row is
False in every identical build (a relaxation define there is a compile error) and True in the legacy fast set, so
"FAST only" above now reads "math row relaxed". A raw call that is reachable only under a relaxed row is listed
SAFE-ROW with the row first in its reason (`# SAFE-ROW: vendor_math: <reason>`, a name from POLICY_ROWS); SAFE-FAST
stays valid for a site behind a test of the mode itself (the legacy fast arms). checks/mode_policy.mojo is the
table, not product arithmetic: POLICY_MODULE is out of scope like the seam library.

Files are split across os.cpu_count() worker processes. tools/prebuild_check.py runs it in its `audits` part.
"""
import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOW = "tools/identity_math_lint_allow.txt"
NAMES = ("sqrt", "rsqrt", "cbrt", "exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "pow", "tanh", "sinh",
         "cosh", "atanh", "asinh", "acosh", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "erf", "erfc",
         "erfinv", "lgamma", "tgamma", "hypot")
CLASSES = ("SAFE-FAST", "SAFE-ROW", "SAFE-EXACT", "SAFE-NOT-SHIPPED")
POLICY_MODULE = "checks/mode_policy.mojo"
#: Row names of the mode policy table, by bit of MODE_POLICY_MASK (checks/mode_policy.mojo).
POLICY_ROWS = ("fold_free", "contract_free", "flush_off", "vendor_math", "matrix_units", "waits_elided", "tile_free", "bf16_units")
_ROW_REASON = re.compile(r"#\s*SAFE-ROW:\s*(\w+):\s*\S")
_SKIP_PREFIX = ("bench/", "experiments/", "docs/")
_IMPORT = re.compile(r"^[ \t]*from[ \t]+(?:std\.)?math[ \t]+import[ \t]+(\([^)]*\)|[^\n]+)", re.M)
_QUALIFIED = re.compile(r"^[ \t]*import[ \t]+(?:std\.)?math\b", re.M)
_DEF = re.compile(r"^[ \t]*def[ \t]+(\w+)")
_ALLOW_LINE = re.compile(r"^([^:#\s]+):(\w+):(\w+)\s+#\s*(SAFE-[A-Z-]+):\s*\S")


def in_scope(path):
    p = "/" + path
    return (path.endswith(".mojo") and not path.startswith(_SKIP_PREFIX) and "/checks/" not in p
            and "oracle" not in Path(path).name and path != "checks/numerics.mojo" and path != POLICY_MODULE)


def _blank(m):
    return "\n" * m.group(0).count("\n")


def code_only(text):
    """`text` with docstrings, comments and string literals blanked (line numbers kept)."""
    text = re.sub(r'(?s)(""".*?"""|\'\'\'.*?\'\'\')', _blank, text)
    text = re.sub(r"#[^\n]*", "", text)
    return re.sub(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'', '""', text)


def sites(path, text):
    """[(path, line, function, op)] for every raw std.math call in one file."""
    imported = set()
    for m in _IMPORT.finditer(text):
        for item in re.split(r"[,\n()]", m.group(1)):
            name = item.split("#")[0].strip().split(" as ")[0].strip()
            if name:
                imported.add(name)
    qualified = bool(_QUALIFIED.search(text))
    names = [n for n in NAMES if n in imported]
    if not names and not qualified:
        return []
    code = code_only(text)
    lines = code.split("\n")
    pats = []
    if names:
        pats.append(re.compile(r"(?<![\w.])(" + "|".join(names) + r")\s*\("))
    if qualified:
        pats.append(re.compile(r"\bmath\.(" + "|".join(NAMES) + r")\s*\("))
    out = []
    for pat in pats:
        for m in pat.finditer(code):
            ln = code.count("\n", 0, m.start())
            if _DEF.match(lines[ln]):
                continue
            fn = "<module>"
            for k in range(ln, -1, -1):
                d = _DEF.match(lines[k])
                if d and (len(lines[k]) - len(lines[k].lstrip())) <= (len(lines[ln]) - len(lines[ln].lstrip())):
                    fn = d.group(1)
                    break
            out.append((path, ln + 1, fn, m.group(1)))
    return sorted(set(out), key=lambda s: (s[1], s[3]))


def read_allow(text):
    """({(path, function, op): line}, [malformed lines])."""
    allow, bad = {}, []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ALLOW_LINE.match(line)
        if not m or m.group(4) not in CLASSES:
            bad.append(raw)
            continue
        if m.group(4) == "SAFE-ROW":
            # a SAFE-ROW line names the row that guards it; an unknown row is malformed
            row = _ROW_REASON.search(line)
            if not row or row.group(1) not in POLICY_ROWS:
                bad.append(raw)
                continue
        allow[(m.group(1), m.group(2), m.group(3))] = raw
    return allow, bad


def judge(found, allow):
    """(unlisted sites, stale allow keys)."""
    keys = {(p, f, o) for p, _, f, o in found}
    return [s for s in found if (s[0], s[2], s[3]) not in allow], sorted(set(allow) - keys)


def _scan_one(job):
    path, text = job
    return sites(path, text)


def load(root, commit):
    if commit:
        names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit], cwd=root, text=True).split("\n")
        names = [n for n in names if in_scope(n)]
        # one `git cat-file --batch` for every blob (a process per file costs seconds on ~3,700 files)
        batch = subprocess.run(["git", "cat-file", "--batch"], cwd=root, check=True, capture_output=True,
                               input="".join(f"{commit}:{n}\n" for n in names).encode()).stdout
        texts, pos = [], 0
        for _ in names:
            header_end = batch.index(b"\n", pos)
            size = int(batch[pos:header_end].split()[2])
            texts.append(batch[header_end + 1:header_end + 1 + size].decode("utf8", "replace"))
            pos = header_end + 1 + size + 1
        allow = subprocess.run(["git", "show", f"{commit}:{ALLOW}"], cwd=root, capture_output=True).stdout.decode()
        return list(zip(names, texts)), allow
    names = [n for n in subprocess.check_output(["git", "ls-files", "*.mojo"], cwd=root, text=True).split("\n")
             if in_scope(n)]
    allow_path = Path(root) / ALLOW
    return ([(n, (Path(root) / n).read_text(errors="replace")) for n in names],
            allow_path.read_text() if allow_path.exists() else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", default="")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--list", action="store_true", help="print every site (file:line function op)")
    a = ap.parse_args()
    files, allow_text = load(a.root, a.commit)
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as ex:
        found = [s for chunk in ex.map(_scan_one, files, chunksize=16) for s in chunk]
    allow, bad = read_allow(allow_text)
    unlisted, stale = judge(found, allow)
    if a.list:
        for p, ln, fn, op in found:
            print(f"MATHLINT SITE {p}:{ln} {fn} {op}")
    for raw in bad:
        print(f"MATHLINT FAIL malformed allow line: {raw}")
    for p, ln, fn, op in unlisted:
        print(f"MATHLINT FAIL {p}:{ln} {fn}: raw {op}() -- use the checks/numerics.mojo seam, or allow-list it with a"
              f" reason in {ALLOW}")
    for p, fn, op in stale:
        print(f"MATHLINT FAIL stale allow line (matches nothing): {p}:{fn}:{op}")
    ok = not (unlisted or stale or bad)
    print(f"MATHLINT {'PASS' if ok else 'FAIL'} {len(files)} files, {len(found)} raw math sites, "
          f"{len(found) - len(unlisted)} allowed, {len(unlisted)} unlisted, {len(stale)} stale")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
