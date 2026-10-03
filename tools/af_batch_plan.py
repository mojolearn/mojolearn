#!/usr/bin/env python3
"""af_batch_plan.py <pending-lines.txt> <new-lines.txt> <plan.tsv> [--branch B] [--skip-branch B ...] [--drop-tag T ...]

Lane apple-fast-batch: compile the Apple FAST A/B arms once on the M2, time on
the M3 only.

Reads the M3 queue lines "CMD <branch> <tag> <command...>" and
  * rewrites each line to run on the consolidated branch (default
    lane/apple-fast-batch) under tag "<tag>-x" (the tag inside the command is
    renamed too), wrapped as
      CMD <branch> <tag>-x bash tools/af_batch_run.sh <tag>-x <command>
  * writes a build plan TSV, one row per line:
      tag  tool  binding  defA  defB  outdir
    tool = aft (tools/aft_ab.sh), afcdef (tools/afc_ab_def.sh) or afcenv
    (tools/afc_ab.sh: env-form arms, nothing to build; binding/defs "-").
    outdir = the directory the tool reads A.so / B.so from on the M3
    ($HOME left literal).
Lines on a --skip-branch (default lane/apple-fast-opp, lane/apple-fast) and
--drop-tag tags are left out. Prints one summary line: lines, plan rows,
distinct (binding, defines) builds.
"""
import re
import shlex
import sys

TOOLS = {"tools/aft_ab.sh": "aft", "tools/afc_ab_def.sh": "afcdef", "tools/afc_ab.sh": "afcenv"}


def norm(d):
    return " ".join(d.split())


def parse(cmd):
    """(tool, binding, defA, defB, outdir) of one command (tag already renamed)."""
    toks = shlex.split(cmd, posix=True)
    for i, t in enumerate(toks):
        if t in TOOLS:
            tool, args = TOOLS[t], toks[i + 1:]
            env = dict(x.split("=", 1) for x in toks[:i] if re.match(r"^[A-Z_][A-Z0-9_]*=", x))
            break
    else:
        return None
    if tool == "aft":
        # aft_ab.sh <binding> <lane> <ds> <pairs> <defA> <defB>
        bind = args[0]
        out = env.get("AFT_OUT", "$HOME/aft-ab/" + bind)
        return tool, bind, norm(args[4]), norm(args[5]), out
    if tool == "afcdef":
        # afc_ab_def.sh <tag> <binding> <lane> <ds> <reps> <rounds> <defA> <defB>
        return tool, args[1], norm(args[6]), norm(args[7]), "$HOME/afc-def/" + args[0]
    return tool, "-", "-", "-", "-"


def main(argv):
    pos, opts = [], {"--branch": ["lane/apple-fast-batch"], "--skip-branch": [], "--drop-tag": []}
    it = iter(argv)
    for a in it:
        if a in opts:
            opts[a].append(next(it))
        else:
            pos.append(a)
    if len(pos) != 3:
        sys.exit(__doc__)
    src, new_path, plan_path = pos
    branch = opts["--branch"][-1]
    skip = set(opts["--skip-branch"] or ["lane/apple-fast-opp", "lane/apple-fast"])
    drop = set(opts["--drop-tag"])
    new, plan, builds = [], [], set()
    for line in open(src):
        line = line.rstrip("\n")
        m = re.match(r"^CMD (\S+) (\S+) (.*)$", line)
        if not m or m.group(1) in skip or m.group(2) in drop:
            continue
        tag, cmd = m.group(2), m.group(3)
        xt = tag + "-x"
        cmd = re.sub(r"(?<![\w-])" + re.escape(tag) + r"(?![\w-])", xt, cmd)
        p = parse(cmd)
        if p is None:
            print(f"SKIP no A/B tool: {tag}", file=sys.stderr)
            continue
        new.append(f"CMD {branch} {xt} bash tools/af_batch_run.sh {xt} {cmd}")
        plan.append((xt,) + p)
        if p[0] != "afcenv":
            builds.add((p[1], p[2]))
            builds.add((p[1], p[3]))
    with open(new_path, "w") as f:
        f.write("\n".join(new) + "\n")
    with open(plan_path, "w") as f:
        f.write("tag\ttool\tbinding\tdefA\tdefB\toutdir\n")
        for r in plan:
            f.write("\t".join(r) + "\n")
    n_ab = sum(1 for r in plan if r[1] != "afcenv")
    print(f"AF-BATCH-PLAN lines={len(new)} ab_lines={n_ab} env_lines={len(new) - n_ab} "
          f"builds={len(builds)} (vs {2 * n_ab} per-arm builds before)")


if __name__ == "__main__":
    main(sys.argv[1:])
