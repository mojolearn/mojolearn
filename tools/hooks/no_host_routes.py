#!/usr/bin/env python3
"""Refuse a push whose added lines add or touch a CPU route on GPU installs.

On a GPU install every fit, transform and predict runs on the GPU. A slow
GPU kernel gets fixed, never routed around (Andrew, Oct 1 2026). The routes
already on main are debt. There is no allowlist: an added line that names one
of them fails too, so touching a route means removing it, and the debt only
shrinks.

Only added lines are read (`git diff -U0`), so the check costs well under a
second. The same check runs on GitHub (.github/workflows/no-host-routes.yml)
for every PR and every push to main, and the main ruleset requires it before
a PR merges, so a merge done on GitHub is checked too. Usage:
    no_host_routes.py <base> <tip> [--main REF]   lines <tip> adds over <base>
    no_host_routes.py --diff <file>               check a unified diff (tests)
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

# Six PRs in flight when the check went in (Andrew, Oct 2: everything in
# flight lands). Exactly the lines each PR had added up to this pinned head
# pass; a commit added to the PR after the pin is checked like any other. An
# entry goes away when its PR lands or closes; never add one.
_IN_FLIGHT = {
    77: "e0918aa35bfbf6b81b095ee364ed8cc5bfb2a003",
    85: "6a483da91d76ae2560b5316910738e7b3328e375",
    86: "11912bf71d8e8d7faaadf8301088c517562c7758",
    106: "e2128048602c358e3a3c402fc70ea718c7a01d64",
    116: "28671ad0e1bdb7e1c59d7ee7925e1fccf1fb204a",
    126: "9901c6a91a8d8c8aa3e5162024c7b1acacd7bc37",
}

_ALWAYS_SKIP = re.compile(r"(^|/)(tests?|checks|bench|tools|docs)/|\.md$")
_COMMENT = re.compile(r"^\s*(#|//)")


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          errors="replace")


def _added(base, tip):
    return _git("diff", "-U0", "--no-color", "--no-ext-diff",
                "--diff-filter=AMR", base, tip).stdout


def in_flight_lines(main_ref):
    """(path, stripped text) of every line the pinned PR heads add over main."""
    lines = set()
    for sha in _IN_FLIGHT.values():
        if _git("cat-file", "-e", sha + "^{commit}").returncode != 0:
            _git("fetch", "-q", "origin", sha)
        mb = _git("merge-base", main_ref, sha)
        if mb.returncode != 0:
            continue
        for path, _, text in _walk(_added(mb.stdout.strip(), sha)):
            lines.add((path, text.strip()))
    return lines


def _walk(diff_text):
    """(path, line number, text) of every added line of a -U0 diff."""
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
        line_no += 1
        yield path, line_no - 1, raw[1:]


def findings(diff_text, exempt=frozenset()):
    out = []
    for path, here, text in _walk(diff_text):
        if _ALWAYS_SKIP.search(path) or _COMMENT.match(text):
            continue
        if (path, text.strip()) in exempt:
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
    main_ref = "origin/main"
    if "--main" in argv:
        i = argv.index("--main")
        main_ref = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if len(argv) == 3 and argv[1] == "--diff":
        with open(argv[2], encoding="utf-8", errors="replace") as f:
            diff = f.read()
        exempt = frozenset()
    elif len(argv) == 3:
        base = argv[1]
        if main_ref:
            # count what the tip adds over main (the CI check's range): from
            # merge-base(main, tip). A push that merges main into a lane then
            # is not charged with main's own lines, which were checked when
            # they landed; every line the lane itself adds is still read.
            mb = _git("merge-base", main_ref, argv[2])
            if mb.returncode == 0:
                base = mb.stdout.strip()
        r = _git("diff", "-U0", "--no-color", "--no-ext-diff",
                 "--diff-filter=AMR", base, argv[2])
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            return 2
        diff = r.stdout
        exempt = in_flight_lines(main_ref) if diff else frozenset()
    else:
        print(__doc__, file=sys.stderr)
        return 2
    found = findings(diff, exempt)
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
