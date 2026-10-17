#!/usr/bin/env python3
"""rebuild_fence.py <tip-sha> [--branch]: refuse a lane push whose change to a shared Mojo module forces a wide rebuild.

Why (Andrew 2026-10-17, 0.8.52): one refusal line a lane added to core/six_lane_experiment_guards.mojo made 76 bindings
"changed" against the last release instead of 7, so the box compile, the release's macOS build and the identity
re-record (449 lanes instead of the few the code touched) all ran over everything. The cost of a change is the number
of bindings that import it, transitively; a change to one area belongs in that area's own module.

For every changed .mojo file (the push's commits over its merge-base with origin/main), count the build scripts whose
binding reaches it through the import graph of tools/hooks/no_host_routes.py. A file reached by more than
WIDE_BUILDS builds is a shared module. With --branch (lane pushes) such a change is refused unless
MOJOLEARN_WIDE_REBUILD_OK=1 is set for a change that genuinely belongs in the shared module (a real core fix);
without --branch (a push to main) it is reported, never refused.

WIDE_BUILDS = 20: the release ships about 140 build scripts; one area (a model family with its host twin and its
check bindings) reaches well under 20 of them, while a module reached by more than 20 spans several families.
"""
import importlib.util, os, re, subprocess, sys

WIDE_BUILDS = 20


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def _graph_module():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("nhr_rebuild_fence", os.path.join(here, "no_host_routes.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reach_counts(tip, changed):
    """{changed .mojo path: number of build scripts whose binding imports it, transitively}."""
    tree = _graph_module().Tree(tip)
    files = tree.files
    scripts = sorted(p for p in files if re.fullmatch(r"bindings/build(_\w+)?\.sh", p)
                     and p != "bindings/build_host_family.sh")
    counts = {p: 0 for p in changed}
    for s in scripts:
        text = _git("show", f"{tip}:{s}")
        roots = {f"bindings/{m}.mojo" for m in re.findall(r"bindings/(_mojolearn\w*)\.mojo", text)}
        fam = re.search(r"build_host_family\.sh\"?\s+(\w+)", text)
        if fam:
            roots.add(f"bindings/_mojolearn_{fam.group(1)}_host.mojo")
        roots &= files
        seen, stack = set(), list(roots)
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            stack.extend(tree.edges.get(p, ()))
        for p in seen & changed:
            counts[p] += 1
    return counts


def main():
    tip = sys.argv[1]
    branch = "--branch" in sys.argv[2:]
    try:
        base = _git("merge-base", "origin/main", tip).strip()
    except subprocess.CalledProcessError:
        return 0
    changed = {p for p in _git("diff", "--name-only", f"{base}..{tip}").split() if p.endswith(".mojo")}
    if not changed:
        return 0
    wide = sorted(((n, p) for p, n in reach_counts(tip, changed).items() if n > WIDE_BUILDS), reverse=True)
    if not wide:
        return 0
    head = "rebuild-fence: shared Mojo modules changed (each is imported by more than %d bindings):" % WIDE_BUILDS
    lines = [f"  {p}: {n} bindings rebuild and their identity lanes re-record" for n, p in wide]
    if not branch or os.environ.get("MOJOLEARN_WIDE_REBUILD_OK") == "1":
        print("\n".join([head, *lines]), file=sys.stderr)
        return 0
    print("\n".join([head.replace("rebuild-fence:", "rebuild-fence: REFUSED."), *lines,
                     "  Put an area's change (a refusal, a constant, a helper) in that area's own module.",
                     "  A change that genuinely belongs in the shared module: push with MOJOLEARN_WIDE_REBUILD_OK=1."]),
          file=sys.stderr)
    return 5


if __name__ == "__main__":
    sys.exit(main())
