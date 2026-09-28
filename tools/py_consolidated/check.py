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


def _cell_parts(cell):
    """Every compared field of a cell, first repeat only; timings dropped."""
    out = {}
    for k, v in cell.items():
        if k in ("verdict", "seconds", "times", "time") or k.endswith("_verdict") or k.endswith("_s"):
            continue
        out[k] = v[0] if isinstance(v, list) and v else v
    return out


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
            A, B = json.loads(a.read_text())["cells"], json.loads(b.read_text())["cells"]
            keys = sorted(k for k in set(A) & set(B) if k.split("/")[0] == lane)
            moved = []
            for k in keys:
                pa, pb = _cell_parts(A[k]), _cell_parts(B[k])
                for f in sorted(set(pa) | set(pb)):
                    if pa.get(f) != pb.get(f):
                        moved.append(f"{k}:{f}")
            v = "SAME" if keys and not moved else ("NOTHING" if not keys else "MOVED")
            bad += v != "SAME"
            ts = f"({lb.get(lane, {}).get(col + '_s')} / {ln.get(lane, {}).get(col + '_s')})"
            print(f"{lane:34s} {col:4s} {len(keys):5d}  {v:8s}  {ts}" + (f"  {moved[:6]}" if moved else ""))
        print(f"{'':34s} gpu==cpu base {lb.get(lane, {}).get('gpu_vs_cpu')}, new {ln.get(lane, {}).get('gpu_vs_cpu')}")
    for col in () if os.environ.get("NO_PROBE") else ("gpu", "cpu"):
        print(f"\n== probe, {col} column, base -> new")
        subprocess.run([sys.executable, str(HERE / "probe.py"), "--diff", str(base / f"probe.{col}.json"),
                        str(new / f"probe.{col}.json")])
    print(f"\n{now()} CROSS RESULT: {bad} lane column(s) not SAME")


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
        cross(Path(a.base), Path(a.new), lanes)
