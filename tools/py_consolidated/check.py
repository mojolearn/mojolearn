#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/py-consolidated (from lane/py-bugs' check.py): the before/after bit
record, LIGHT (build once per tree, one GPU arm and one CPU arm per lane, the
py-bugs probe once per column). The GPU arms run one at a time; the CPU arms
run beside them, CPU_ARMS (default 4) at once, each on one thread
(OMP_NUM_THREADS=1) with its own log. A `par-*` lane's CPU arm may end in the
declared by-design refusal (algos_lane_check.known_cpu_refusal).

    check.py arms  --tree T --out D --lanes a,b     build T's bindings for the lanes, run each
                                                    lane's GPU and CPU arm (GPU == CPU verdict
                                                    per lane), the probe on both columns
    check.py cross --base D0 --new D1 --lanes a,b   base vs new, per lane and column, part by
                                                    part; the probe's diff per column

`arms` imports the tools of the TREE it checks (tools/algos_lane_check.py), so
the base tree builds and runs with its own code. Set MOJOLEARN_LANE_CHECK_STORE
to one directory for both trees: a binding whose sources the lane did not
touch is built once and reused by source-closure digest.
"""
import argparse, concurrent.futures as cf, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent  # job.sh runs a copy outside the tree


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def arms(tree, out, lanes):
    tree = Path(tree).resolve()
    sys.path.insert(0, str(tree / "tools"))
    import algos_lane_check as alc
    assert Path(alc.ROOT).resolve() == tree, (alc.ROOT, tree)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "arms.log"
    alc.ensure_portable_math(out / "portable_math.log")
    needed, rows = {}, {}
    for lane in lanes:
        try:
            needed.update(alc.needed_bindings([lane]))
        except Exception as e:  # no CPU arm or no GPU arm: not exposed, recorded, skipped
            rows[lane] = {"gpu_vs_cpu": "NOT EXPOSED " + str(e).splitlines()[0][:160]}
    lanes = [l for l in lanes if l in needed]
    allb = sorted(set().union(*needed.values()))
    t0 = time.time()
    alc.ensure_built(set(allb), out / "build.log", publish=True)
    print(f"{now()} {tree.name}: {len(allb)} bindings ready in {time.time() - t0:.0f} s", flush=True)
    ib = alc.load_harness()
    backend = alc.gpu_backend()
    pool = cf.ThreadPoolExecutor(max_workers=int(os.environ.get("CPU_ARMS", "4")))

    def cpu_arm(lane):
        t = time.time()
        rc = alc.run_arm("cpu", lane, backend, "", out / f"{lane}.cpu.json", out / f"{lane}.cpu.log",
                         refusal_ok=lane.startswith("par-"))
        return rc, time.time() - t

    cpu = {lane: pool.submit(cpu_arm, lane) for lane in lanes}
    for lane in lanes:
        gj, cj = out / f"{lane}.gpu.json", out / f"{lane}.cpu.json"
        row = rows.setdefault(lane, {})
        try:
            t = time.time()
            alc.run_arm("gpu", lane, backend, "", gj, log)
            row["gpu_s"] = round(time.time() - t, 1)
        except Exception as e:
            row["gpu_vs_cpu"] = "ERROR gpu " + str(e).splitlines()[0][:200]
        print(f"{now()} {lane}: gpu {row}", flush=True)
    for lane in lanes:
        gj, cj = out / f"{lane}.gpu.json", out / f"{lane}.cpu.json"
        row = rows[lane]
        try:
            rc, tc = cpu[lane].result()
            row["cpu_s"] = round(tc, 1)
            if "gpu_vs_cpu" not in row:
                if rc == 1 and alc.known_cpu_refusal(ib, lane, gj, cj, "")[0]:
                    row["gpu_vs_cpu"] = "KNOWN REFUSAL (par-* CPU by design)"
                elif rc == 1:
                    row["gpu_vs_cpu"] = "ERROR cpu refused (not the declared by-design refusal)"
                else:
                    row["gpu_vs_cpu"] = alc.compare(ib, lane, gj, cj, "", log, backend)[0]
        except Exception as e:
            row.setdefault("gpu_vs_cpu", "ERROR " + str(e).splitlines()[0][:200])
        print(f"{now()} {lane}: {row}", flush=True)
    pool.shutdown()
    (out / "lanes.json").write_text(json.dumps(rows, indent=1))
    if os.environ.get("NO_PROBE"):
        return
    probe = HERE / "probe.py"
    for col in ("gpu", "cpu"):
        print(f"{now()} probe {col}", flush=True)
        with open(out / f"probe.{col}.log", "w") as fh:
            subprocess.run([sys.executable, "-u", str(probe), "--out", str(out / f"probe.{col}.json")],
                           cwd=tree, env=dict(alc.arm_env(col), PROBE_TREE=str(tree)), stdout=fh,
                           stderr=subprocess.STDOUT)
        print((out / f"probe.{col}.log").read_text()[-3000:], flush=True)


def compare_records(A, B, lane):
    """Saved numerical evidence only; missing/refused/unstable cells never pass."""
    import importlib.util
    helper = Path(__file__).resolve().parents[1] / "consolidated_check/compare.py"
    if not helper.is_file():  # job copies this driver outside the checkout
        helper = Path.cwd() / "tools/consolidated_check/compare.py"
    spec = importlib.util.spec_from_file_location("saved_column_compare", helper)
    strict = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strict)
    issues, numeric, na = [], 0, 0
    for name, record in (("base", A), ("new", B)):
        if record.get("complete") is not True or record.get("partial_column"):
            issues.append(name + ": incomplete/partial record")
    ac, bc = A.get("cells", {}), B.get("cells", {})
    keys = sorted(k for k in set(ac) | set(bc) if k.split("/")[0] == lane)
    for key in keys:
        if key not in ac or key not in bc:
            issues.append(key + ": missing cell")
            continue
        x, y = ac[key], bc[key]
        if any(c.get("error") or c.get("refusal") for c in (x, y)):
            issues.append(key + ": refusal/error")
            continue
        present = [p for p in strict.PARTS if ("hashes" if p == "train" else p) in x
                   or ("hashes" if p == "train" else p) in y]
        cell_numeric = 0
        for part in present:
            va = strict._value(x, part, A.get("repeats", 0))
            vb = strict._value(y, part, B.get("repeats", 0))
            if va[0] == vb[0] == "NUMERIC" and va[1] == vb[1]:
                numeric += 1
                cell_numeric += 1
            elif va[0] == vb[0] == "NA" and va[1] == vb[1]:
                na += 1
            else:
                issues.append(f"{key}:{part}: {va} -> {vb}")
        if not cell_numeric:
            issues.append(key + ": no successful numeric parts")
    if not keys:
        issues.append("no cells")
    return dict(status="SAME" if not issues and numeric else "INCOMPLETE/DIFFERENT",
                cells=len(keys), numeric=numeric, na=na, issues=issues)


def cross(base, new, lanes):
    bad = 0
    print(f"{'lane':34s} {'col':4s} {'cells':>5s}  verdict   (base s / new s)")
    lb = json.loads((base / "lanes.json").read_text())
    ln = json.loads((new / "lanes.json").read_text())
    for lane in lanes:
        for col in ("gpu", "cpu"):
            a, b = base / f"{lane}.{col}.json", new / f"{lane}.{col}.json"
            if not (a.is_file() and b.is_file()):
                print(f"{lane:34s} {col:4s}   -    MISSING")
                bad += 1
                continue
            result = compare_records(json.loads(a.read_text()), json.loads(b.read_text()), lane)
            keys = range(result["cells"])
            moved = result["issues"]
            v = result["status"]
            bad += v != "SAME"
            ts = f"({lb.get(lane, {}).get(col + '_s')} / {ln.get(lane, {}).get(col + '_s')})"
            print(f"{lane:34s} {col:4s} {len(keys):5d}  {v:8s}  {ts}" + (f"  {moved[:6]}" if moved else ""))
        print(f"{'':34s} gpu==cpu base {lb.get(lane, {}).get('gpu_vs_cpu')}, new {ln.get(lane, {}).get('gpu_vs_cpu')}")
    for col in () if os.environ.get("NO_PROBE") else ("gpu", "cpu"):
        print(f"\n== probe, {col} column, base -> new")
        result = subprocess.run([sys.executable, str(HERE / "probe.py"), "--diff", str(base / f"probe.{col}.json"),
                        str(new / f"probe.{col}.json")])
        bad += result.returncode != 0
    print(f"\n{now()} CROSS RESULT: {bad} lane column(s) not SAME")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("arms", "cross"))
    ap.add_argument("--tree")
    ap.add_argument("--out")
    ap.add_argument("--base")
    ap.add_argument("--new")
    ap.add_argument("--lanes", required=True)
    a = ap.parse_args()
    lanes = [x for x in a.lanes.split(",") if x]
    if a.cmd == "arms":
        arms(a.tree, Path(a.out), lanes)
    else:
        raise SystemExit(cross(Path(a.base), Path(a.new), lanes))
