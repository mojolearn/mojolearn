# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane linear-apple3: the arms of FAST changes in ONE steward speed job
(same commit, same Mac).

    pixi run -e default python tools/linear_apple3/ab.py tools/linear_apple3/<job>.json

The job file:
  arms      a list; an arm is a set of build defines (so every arm is one
            commit's source): name, defines, builds (bindings/build_<b>.sh),
            cases and rows for bench/x_linear_speed.py, reps, and qual (the
            cases of bench/linear_apple3_quality.py; "" for none). An arm
            whose build fails prints the compiler's words and is skipped.
  base      digest comparisons against the base commit's copy of `files`:
            for each mode (identical, fast) the bindings are built at HEAD
            and with the base files, with NO defines, and every case's
            digest is compared (SAME / MOVED). IDENTICAL must be SAME on
            every line; FAST SAME shows the default path did not move.
Every binding an arm built is rebuilt with no defines at the end.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sh(cmd, env=None, quiet=False):
    e = dict(os.environ)
    e["PYTHONUNBUFFERED"] = "1"
    if env:
        e.update(env)
    r = subprocess.run(cmd, cwd=ROOT, env=e, shell=isinstance(cmd, str), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    if not quiet:
        sys.stdout.write(r.stdout)
        sys.stdout.flush()
    return r.returncode, r.stdout


def build(mode, defines, binding):
    rc, out = sh(["sh", f"bindings/build_{binding}.sh"],
                 {"MOJOLEARN_MOJO_BUILD_FLAGS": defines, "MOJOLEARN_NUMERIC_MODE": mode}, quiet=True)
    if rc:
        lines = [l for l in out.splitlines() if "ld: warning" not in l]
        print(f"BUILD FAIL mode={mode} defines=[{defines}] {binding}")
        print("\n".join(lines[-120:]), flush=True)
    return rc == 0


def speed(tag, mode, cases, rows, column="gpu"):
    rc, out = sh([sys.executable, "bench/x_linear_speed.py", "--rows", str(rows), "--column", column,
                  "--only", cases], {"MOJOLEARN_NUMERIC_MODE": mode}, quiet=True)
    got = {}
    for l in out.splitlines():
        print(f"[{tag} {mode} {rows}] {l}", flush=True)
        w = l.split()
        if len(w) >= 6 and w[0] == "XLSPEED" and w[2] == "gpu":
            got[w[1]] = w[-1]
    return got


def main():
    job = json.load(open(sys.argv[1]))
    rc, head = sh("git rev-parse --short HEAD", quiet=True)
    rc, cpu = sh("sysctl -n machdep.cpu.brand_string", quiet=True)
    print(f"L3AB commit={head.strip()} job={sys.argv[1]} {cpu.strip()}", flush=True)
    touched = set()
    host_built = False
    first_qual = True
    for arm in job.get("arms", []):
        name, defines = arm["name"], arm.get("defines", "")
        print(f"##### ARM {name} defines=[{defines}]", flush=True)
        ok = True
        for b in arm.get("builds", ["x_linear"]):
            touched.add(b)
            ok = build("fast", defines, b) and ok
        if not ok:
            continue
        cases = arm["cases"]
        # the first fit of a process builds the pipelines
        speed(name + " warm", "fast", cases, arm.get("warm_rows", 20000), arm.get("warm_column", "gpu"))
        for r in arm.get("rows", [100000]):
            for _ in range(arm.get("reps", 1)):
                speed(name, "fast", cases, r)
        if arm.get("qual"):
            if not host_built:
                # the builder refuses to overwrite: set the one in place aside
                so = os.path.join(ROOT, "python", "mojolearn", "host", "_mojolearn_x_linear_host.so")
                had = os.path.exists(so)
                if had:
                    os.replace(so, so + ".l3prev")
                if build("identical", "", "x_linear_host"):
                    print("HOST reference binding rebuilt at HEAD", flush=True)
                elif had:
                    os.replace(so + ".l3prev", so)
                    print("HOST reference binding: the one in place (not rebuilt)", flush=True)
                host_built = True
            cmd = [sys.executable, "bench/linear_apple3_quality.py", "--arm", name, "--cases", arm["qual"],
                   "--seeds", arm.get("seeds", "0,1,2,3,4")]
            if not (first_qual or arm.get("host_ref")):
                cmd.append("--no-host")
            first_qual = False
            sh(cmd, {"MOJOLEARN_NUMERIC_MODE": "fast"})
    base = job.get("base")
    if base:
        files = [f for f in base["files"]
                 if sh(["git", "cat-file", "-e", f"{base['commit']}:{f}"], quiet=True)[0] == 0]
        for mode in base.get("modes", ["identical", "fast"]):
            sides = {}
            for side in ("head", "base"):
                if side == "base":
                    sh(["git", "checkout", base["commit"], "--"] + files)
                ok = all([build(mode, "", b) for b in base["builds"]])
                got = {}
                if ok:
                    for cases, rows in base["runs"]:
                        got.update({f"{k}@{rows}": v for k, v in speed(f"{side}", mode, cases, rows).items()})
                    if side == "head" and mode == "identical" and base.get("qual"):
                        sh([sys.executable, "bench/linear_apple3_quality.py", "--arm", "identical", "--cases",
                            base["qual"], "--seeds", base.get("seeds", "0,1,2,3,4"), "--no-host"],
                           {"MOJOLEARN_NUMERIC_MODE": "identical"})
                if side == "base":
                    sh(["git", "checkout", "HEAD", "--"] + files)
                    sh("git status --short | grep -v '^??' | head")
                sides[side] = got
                touched.update(base["builds"])
            for k in sorted(set(sides["head"]) | set(sides["base"])):
                a, b = sides["head"].get(k), sides["base"].get(k)
                print(f"DIGEST {mode} {k} {'SAME' if a and a == b else 'MOVED'} head={a} base={b}", flush=True)
    for b in sorted(touched):
        build("fast", "", b)
        if base and "identical" in base.get("modes", ["identical", "fast"]):
            build("identical", "", b)
    print("JOBDONE", flush=True)


if __name__ == "__main__":
    main()
