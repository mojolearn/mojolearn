#!/usr/bin/env python3
"""Do the things this tree TELLS you to run, and the paths it cites, exist?

    python3 tools/check_docs_reference_reality.py

WHY. Six statements in this repository were wrong on 2026-09-03, and every
one of them was TRUE WHEN WRITTEN and never re-checked:

  * README.md claimed three-vendor identity for k-NN metrics that have never
    run on any vendor.
  * tools/e1_bootstrap.sh said the E2U matrix is 67 cells and
    tools/e2u_README.md said 73; it is 111.
  * pixi.toml asserted an ABI wall between the vendor arms and our
    extensions that does not exist.
  * BOTH speed harnesses told the reader to run `pixi run -e speedbench`,
    an environment that has never been defined.
  * tools/e1_traced_fit.py told the reader to EXPECT a divergence that was
    fixed weeks earlier.

That is a process gap, not six accidents. Most of it is not mechanically
checkable -- a claim about what a measurement showed needs a person. TWO
CLASSES ARE, and they are the two that waste the most time, because they
send someone to run a command that cannot work or to open a file that is
not there:

  1. `pixi run -e <env>` naming an environment pixi.toml does not define.
  2. A repo-relative path cited in backticks that does not exist on disk.

This checks those two and nothing else. It deliberately does NOT try to
validate prose claims: a checker that guesses at meaning produces false
alarms, and a checker people ignore is worse than none.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".pixi", "bench/results", "__pycache__", ".venv",
             "upstream", "node_modules", "corpus"}
SCAN_EXT = {".md", ".py", ".sh", ".mojo", ".toml"}

#: A backticked token is treated as a path claim only if it looks like one:
#: it has a directory separator or a known source extension, and no spaces,
#: glob characters or shell metacharacters. Anything vaguer is prose.
PATHISH = re.compile(r"^[A-Za-z0-9_./-]+$")
#: Directory roots that belong to a VENDORED OR PORTED library rather than to
#: this tree. Citing them precisely is what a DEVIATION block is for.
UPSTREAM_PREFIXES = (
    "catboost/", "upstream/", "cuml/", "raft/", "cuvs/", "src/", "csrc/",
    "max/", "models/", "mamba_ssm/", "src_prims/", "torch/", "triton/",
    "sparse/", "solver/", "matrix/", "hierarchy/", "selection/",
    "sklearn/", "scipy/", "numpy/", "faiss/", "xgboost/", "lightgbm/",
)
REPO_FILES = []
SRC_EXT = (".py", ".sh", ".mojo", ".md", ".toml", ".json", ".txt", ".cu", ".cuh")


def environments():
    text = (ROOT / "pixi.toml").read_text()
    m = re.search(r"^\[environments\]\s*(.*?)(?=^\[|\Z)", text, re.S | re.M)
    if not m:
        return set()
    return set(re.findall(r"^([A-Za-z0-9_-]+)\s*=", m.group(1), re.M))


def scan_files():
    for p in ROOT.rglob("*"):
        if p.suffix not in SCAN_EXT or not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if any(rel.startswith(d) or f"/{d}/" in f"/{rel}" for d in SKIP_DIRS):
            continue
        yield p, rel


def main():
    global REPO_FILES
    REPO_FILES = [f.relative_to(ROOT) for f in ROOT.rglob("*")
                  if f.is_file() and ".git/" not in f.as_posix()
                  and "/.pixi/" not in f.as_posix()]
    envs = environments()
    print(f"pixi environments defined: {', '.join(sorted(envs)) or '(none)'}")
    bad_env, bad_path, checked = [], [], 0

    for p, rel in scan_files():
        # This file QUOTES the defects it looks for, so it would flag itself.
        if rel == "tools/check_docs_reference_reality.py":
            continue
        try:
            text = p.read_text(errors="replace")
        except Exception:
            continue
        checked += 1

        for m in re.finditer(r"pixi run\s+(?:--frozen\s+)?-e\s+([A-Za-z0-9_-]+)", text):
            env = m.group(1)
            if env not in envs:
                line = text[:m.start()].count("\n") + 1
                bad_env.append((rel, line, env))

        # Path claims in backticks. Only flag a MISSING path, never an
        # ambiguous one: a token with no separator and no source extension is
        # prose, and a token naming a directory that exists is fine.
        for m in re.finditer(r"`([^`\n]{3,120})`", text):
            tok = m.group(1).strip()
            if not PATHISH.match(tok):
                continue
            # AN ELIDED PATH IS PROSE. `mamba/impl/.../modeling_mamba.mojo`
            # is an author shortening a long path on purpose, not claiming
            # that literal file exists.
            if "..." in tok:
                continue
            # A BARE FILENAME IS PROSE, NOT A PATH CLAIM. `numerics.mojo`
            # in a sentence means "the numerics file" and the reader finds
            # it; requiring those to resolve produced 3,593 hits on the
            # first run, almost all of them legitimate shorthand, and a
            # checker with that signal-to-noise is one nobody runs. Only a
            # token carrying a DIRECTORY is claiming a location.
            if "/" not in tok:
                continue
            if tok.startswith(("http", "-", ".")) or tok.endswith("/"):
                continue
            # A trailing :NN line reference is a citation, not part of the path
            base = tok.split(":")[0]
            if not base or base.startswith("/"):
                continue
            if (ROOT / base).exists():
                continue
            # UPSTREAM IS CITED ON PURPOSE AND IS NOT CHECKED OUT HERE. This
            # package ports CatBoost, cuML, cuVS, RAFT, HuggingFace and
            # mamba_ssm, and naming the exact upstream file is the whole
            # point of a DEVIATION block. Those citations are evidence, not
            # broken links, and a checker that flags them is telling the
            # author to stop doing the right thing.
            if base.endswith((".cu", ".cuh", ".h", ".hpp", ".cpp")):
                continue
            if base.startswith(UPSTREAM_PREFIXES):
                continue
            if not base.endswith(SRC_EXT):
                continue
            # A PATH RELATIVE TO A SUBDIRECTORY STILL RESOLVES. Files under
            # gbdt/ cite `gpu_util/partitions_reduce.mojo` and mean the one
            # beside them. Accept any repo file whose path ENDS with the
            # citation; only a suffix that matches nothing is a dead link.
            if any(f.as_posix().endswith("/" + base) for f in REPO_FILES):
                continue
            line = text[:m.start()].count("\n") + 1
            bad_path.append((rel, line, base))

    print(f"scanned {checked} files\n")
    if bad_env:
        print(f"DEAD PIXI ENVIRONMENT ({len(bad_env)}) -- these tell a reader to")
        print("run a command that cannot work:")
        for rel, line, env in bad_env:
            print(f"  {rel}:{line}  pixi run -e {env}")
        print()
    if bad_path:
        print(f"MISSING PATH CITED ({len(bad_path)}):")
        for rel, line, base in sorted(bad_path)[:60]:
            print(f"  {rel}:{line}  {base}")
        if len(bad_path) > 60:
            print(f"  ... and {len(bad_path) - 60} more")
        print()
    if bad_env or bad_path:
        print("A citation that does not resolve costs the next person the time")
        print("it takes to discover that, which is exactly what this catches.")
        return 1
    print("Every pixi environment named exists; every cited path resolves.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
