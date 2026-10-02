#!/usr/bin/env python3
"""Refuse a push whose added lines add or touch a CPU route on GPU installs.

On a GPU install every fit, transform and predict runs on the GPU. A slow
GPU kernel gets fixed, never routed around (Andrew, Oct 1 2026). The routes
already on main are debt. There is no allowlist: an added line that names one
of them fails too, so touching a route means removing it, and the debt only
shrinks.

Only added lines are read (`git diff -U0`), so the check costs well under a
second. Usage:
    no_host_routes.py <base> <tip>     check the lines <tip> adds over <base>
    no_host_routes.py --diff <file>    check a unified diff (tests)
Exit 1 with one line per finding, 0 when clean.
"""
import re
import subprocess
import sys

# Files that never run on a GPU install's fit path: the CPU-only host
# bindings, any file named *host* (host walks, host lanes), their Python
# loaders and the verifier's CPU column. Rules marked `scoped` skip them;
# the route-table rule does not, so the named routes cannot hide there.
_CPU_ONLY = re.compile(
    r"(^|/)(tests?|checks|bench|tools|docs)/"
    r"|\.(md|txt|json|tsv|toml|cfg|sh)$"
    r"|^bindings/_mojolearn_\w*_host\.mojo$"
    r"|^bindings/build_\w*\.sh$"
    r"|(^|/)host/"
    r"|(^|/)[^/]*host[^/]*$"
    r"|^python/mojolearn/(_backend|host_surface|_verify\w*|_\w*_host)\.py$"
)

# Lines that only define the host executor type itself.
_DEFINES_HOSTEXEC = re.compile(r"^\s*(struct|trait|comptime|alias)\s+HostExec\b")

# Env var names of the host binding's own plumbing, not routes.
_HOST_PLUMBING = re.compile(
    r"MOJOLEARN_(\w+_)?HOST_(DIR|ALLOW_SABOTAGE|BINARY|BLOCK_TIMING)\b"
)

# (name, scoped, pattern, why). An unscoped rule applies to every file
# outside tests, tools, benchmarks, checks and docs. Comment lines never count.
_RULES = [
    ("host-module", True,
     re.compile(r"\b(load_host_module|host_module_path)\s*\("),
     "loads a host binding outside the CPU-only modules"),
    ("host-exec", True,
     re.compile(r"\bHostExec\b"),
     "runs HostExec outside the CPU-only bindings"),
    ("host-import", True,
     re.compile(r"^\s*from\s+[\w.]*_host\b[\w.]*\s+import\b"
                r"|^\s*import\s+[\w.]*_host\b"
                r"|\b\w+(_on_host|_host_rows)\s*\("),
     "pulls a host walk into GPU code"),
    ("host-env", True,
     re.compile(r"MOJOLEARN_\w*HOST\w*"),
     "adds a MOJOLEARN_*HOST* switch (a host route knob)"),
    ("host-threshold", True,
     re.compile(r"\b\w*HOST\w*_(MIN|MAX)(_\w+)?\b|\bHOST_(MIN|MAX)\b"),
     "a size threshold that picks the host"),
    ("route-table", False,
     re.compile(r"\b(_HOST_ALGOS|_HOST_ROUTE_LANES|_HOST_ROUTE_MAX_CELLS"
                r"|_HOST_ONE_BORDER_\w+|_glm_host|_small_pool_host"
                r"|_fit_on_host|_sgd_on_host|HOST_RUN)\b"),
     "touches an existing host route (remove it, don't extend it)"),
]

_ALWAYS_SKIP = re.compile(r"(^|/)(tests?|checks|bench|tools|docs)/|\.md$")
_COMMENT = re.compile(r"^\s*(#|//)")


def findings(diff_text):
    out = []
    path = None
    line_no = 0
    for raw in diff_text.splitlines():
        if raw.startswith("+++ "):
            p = raw[4:]
            path = None if p == "/dev/null" else p[2:] if p.startswith("b/") else p
            continue
        if raw.startswith("@@"):
            m = re.search(r"\+(\d+)", raw)
            line_no = int(m.group(1)) if m else 0
            continue
        if not raw.startswith("+") or raw.startswith("+++") or path is None:
            continue
        text = raw[1:]
        here = line_no
        line_no += 1
        if _ALWAYS_SKIP.search(path) or _COMMENT.match(text):
            continue
        scoped_skip = bool(_CPU_ONLY.search(path))
        for name, scoped, pat, why in _RULES:
            if scoped and scoped_skip:
                continue
            hits = [m.group(0) for m in pat.finditer(text)]
            if name == "host-env":
                hits = [h for h in hits if not _HOST_PLUMBING.fullmatch(h)]
            if not hits:
                continue
            if name == "host-exec" and _DEFINES_HOSTEXEC.search(text):
                continue
            out.append(f"{path}:{here}: [{name}] {why}: {text.strip()[:120]}")
    return out


def main(argv):
    if len(argv) == 3 and argv[1] == "--diff":
        with open(argv[2], encoding="utf-8", errors="replace") as f:
            diff = f.read()
    elif len(argv) == 3:
        diff = subprocess.run(
            ["git", "diff", "-U0", "--no-color", "--no-ext-diff",
             "--diff-filter=AMR", argv[1], argv[2]],
            check=True, capture_output=True, text=True, errors="replace",
        ).stdout
    else:
        print(__doc__, file=sys.stderr)
        return 2
    found = findings(diff)
    if not found:
        return 0
    print("no-host-routes: REFUSED. These added lines add or touch a CPU route "
          "on GPU installs:", file=sys.stderr)
    for f in found:
        print("  " + f, file=sys.stderr)
    print("  Fix the GPU kernel instead. Touching an existing route means "
          "removing it (memory gpu-kernels-not-cpu-routes).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
