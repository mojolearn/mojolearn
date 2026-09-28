#!/usr/bin/env python3
"""lane/merged global check driver (UNTRACKED; never committed).

Reuses tools/algos_lane_check.py's own pieces (needed_bindings, build + stamp,
run_arm, compare) so every verdict means what a lane gate's verdict means.

  merged_check.py plan   --out DIR                 exposed lanes + bindings -> DIR/plan.json
  merged_check.py build  --out DIR [--jobs N]      build every stale binding in parallel (stamped)
  merged_check.py clean  --out DIR --shard i/N [--cpu-threads 1,3,default]
                                                   GPU arm once, CPU arm per thread setting, compare
                                                   each CPU column to the GPU column cell for cell
  merged_check.py sabotage --out DIR --patch P --lanes a,b
                                                   apply P, rebuild what it makes stale, both arms,
                                                   every lane must DISAGREE; reverse, rebuild clean

clean never builds and never edits the tree, so shards run side by side;
sabotage edits the tree and must run alone.
"""
import argparse, json, os, subprocess, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import algos_lane_check as alc  # noqa: E402


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def plan(out):
    ib = alc.load_harness()
    lanes, needed, refused = [], {}, {}
    for lane in sorted(ib.LANES):
        try:
            needed[lane] = alc.needed_bindings([lane])[lane]
            lanes.append(lane)
        except Exception as e:  # a lane with no CPU arm or no GPU arm is not exposed
            refused[lane] = str(e).splitlines()[0][:300]
    p = {"lanes": lanes, "needed": needed, "not_exposed": refused,
         "bindings": sorted(set().union(*needed.values()))}
    (out / "plan.json").write_text(json.dumps(p, indent=1))
    print(f"{now()} plan: {len(lanes)} exposed lanes, {len(refused)} not exposed, "
          f"{len(p['bindings'])} bindings", flush=True)
    return p


def load_plan(out):
    f = out / "plan.json"
    return json.loads(f.read_text()) if f.is_file() else plan(out)


def build(out, jobs):
    p = load_plan(out)
    logs = out / "build_logs"
    logs.mkdir(exist_ok=True)
    alc.ensure_portable_math(out / "build_portable_math.log")
    todo = [b for b in p["bindings"] if alc.stale(b)]
    print(f"{now()} build: {len(todo)} of {len(p['bindings'])} stale; jobs {jobs}", flush=True)
    fails = {}

    def one(b):
        try:
            alc.build(b, logs / f"{b}.log")
        except Exception as e:
            fails[b] = str(e)[:300]
            print(f"{now()} BUILD FAIL {b}: {e}", flush=True)

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        list(ex.map(one, todo))
    still = [b for b in p["bindings"] if alc.stale(b)]
    (out / "build_result.json").write_text(json.dumps({"failed": fails, "still_stale": still}, indent=1))
    print(f"{now()} BUILD RESULT: {len(fails)} failed, {len(still)} still stale"
          + (": " + ", ".join(sorted(fails)) if fails else ""), flush=True)
    return 1 if fails or still else 0


def clean(out, shard, cpu_threads):
    p = load_plan(out)
    i, n = (int(x) for x in shard.split("/"))
    lanes = p["lanes"][i::n]
    ib = alc.load_harness()
    backend = alc.gpu_backend()
    stale = {b for b in sorted(set().union(*[p["needed"][l] for l in lanes])) if alc.stale(b)}
    if stale:  # a binding that did not build fails only the lanes that run it
        print(f"STALE (not built): {sorted(stale)}; their lanes are recorded as ERROR", flush=True)
    d = out / f"clean_{i}of{n}"
    d.mkdir(exist_ok=True)
    log = d / "clean.log"
    res = d / "verdicts.tsv"
    done = set()
    if res.is_file():  # resumable: skip lanes already verdicted
        done = {ln.split("\t")[0] for ln in res.read_text().splitlines() if ln}
    print(f"{now()} clean shard {i}/{n}: {len(lanes)} lanes ({len(done)} done), backend {backend}, "
          f"cpu threads {cpu_threads}", flush=True)
    base_env = dict(os.environ)
    for lane in lanes:
        if lane in done:
            continue
        row = [lane]
        try:
            sb = sorted(stale & set(p["needed"][lane]))
            if sb:
                raise RuntimeError(f"binding not built: {','.join(sb)}")
            gj = d / f"{lane}.gpu.json"
            alc.run_arm("gpu", lane, backend, "", gj, log)
            for t in cpu_threads:
                os.environ.clear()
                os.environ.update(base_env)
                if t == "default":
                    os.environ.pop("MOJOLEARN_CPU_THREADS", None)
                else:
                    os.environ["MOJOLEARN_CPU_THREADS"] = t
                cj = d / f"{lane}.cpu{t}.json"
                alc.run_arm("cpu", lane, backend, "", cj, log)
                verdict, detail = alc.compare(ib, lane, gj, cj, "", log, backend)
                row.append(f"cpu{t}={verdict}")
                row.append(str(detail)[:200].replace("\t", " "))
        except Exception as e:
            row.append("ERROR=" + str(e).splitlines()[0][:300].replace("\t", " "))
        finally:
            os.environ.clear()
            os.environ.update(base_env)
        with open(res, "a") as fh:
            fh.write("\t".join(row) + "\n")
        print(f"{now()} " + "  ".join(row[:1] + [c for c in row[1:] if "=" in c]), flush=True)
    rows = [ln.split("\t") for ln in res.read_text().splitlines() if ln]
    bad = [r[0] for r in rows if any(c.startswith("ERROR=") or ("=" in c and c.split("=", 1)[1] != "AGREE"
                                                                  and c.startswith("cpu")) for c in r[1:])]
    print(f"{now()} CLEAN RESULT shard {i}/{n}: {len(rows)} lanes, {len(bad)} not AGREE"
          + (": " + ",".join(bad) if bad else ""), flush=True)
    return 1 if bad else 0


def sabotage(out, patch, lanes):
    p = load_plan(out)
    ib = alc.load_harness()
    backend = alc.gpu_backend()
    lanes = [x for x in lanes.split(",") if x]
    miss = [x for x in lanes if x not in p["needed"]]
    if miss:
        print(f"REFUSED: not exposed lanes {miss}", flush=True)
        return 2
    patch = Path(patch)
    if not patch.is_absolute():
        patch = ROOT / patch
    tag = str(patch.relative_to(ROOT) if patch.is_relative_to(ROOT) else patch.name).replace("/", "_").removesuffix(".patch")
    d = out / f"sab_{tag}"
    d.mkdir(exist_ok=True)
    if (d / "verdicts.json").is_file():  # resumable: this patch already has its verdict
        v = json.loads((d / "verdicts.json").read_text())
        bad = [l for l, x in v.items() if x != "DISAGREE"]
        print(f"{now()} SABOTAGE RESULT {tag}: (done earlier) {len(v) - len(bad)}/{len(v)} DISAGREE"
              + (" NOT BITING: " + ",".join(bad) if bad else ""), flush=True)
        return 1 if bad else 0
    log = d / "sab.log"
    needed = set().union(*[p["needed"][l] for l in lanes])
    if subprocess.run(["git", "apply", "--check", str(patch)], cwd=ROOT).returncode:
        print(f"SABOTAGE RESULT {tag}: BROKEN (does not apply)", flush=True)
        return 1
    subprocess.run(["git", "apply", str(patch)], cwd=ROOT, check=True)
    verdicts = {}
    try:
        moved = [b for b in sorted(needed) if alc.stale(b)]
        print(f"{now()} {tag}: applied; makes stale: {moved}", flush=True)
        alc.ensure_built(needed, log)
        for lane in lanes:
            try:
                gj, cj = d / f"{lane}.gpu.json", d / f"{lane}.cpu.json"
                alc.run_arm("gpu", lane, backend, "", gj, log, moved_ok=True)
                alc.run_arm("cpu", lane, backend, "", cj, log, moved_ok=True)
                verdicts[lane] = alc.compare(ib, lane, gj, cj, "", log, backend)[0]
            except Exception as e:
                verdicts[lane] = "ERROR " + str(e).splitlines()[0][:200]
            print(f"{now()} SABOTAGED {tag} {lane}: {verdicts[lane]}", flush=True)
    finally:
        subprocess.run(["git", "apply", "-R", str(patch)], cwd=ROOT, check=True)
        print(f"{now()} {tag}: reversed; rebuilding clean", flush=True)
        alc.ensure_built(needed, log)
    bad = [l for l, v in verdicts.items() if v != "DISAGREE"]
    (d / "verdicts.json").write_text(json.dumps(verdicts, indent=1))
    print(f"{now()} SABOTAGE RESULT {tag}: {len(lanes) - len(bad)}/{len(lanes)} DISAGREE"
          + (" NOT BITING: " + ",".join(f"{l}={verdicts[l]}" for l in bad) if bad else ""), flush=True)
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "build", "clean", "sabotage"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--cpu-threads", default="default")
    ap.add_argument("--patch", default="")
    ap.add_argument("--lanes", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        if a.cmd == "plan":
            plan(out)
            return 0
        if a.cmd == "build":
            return build(out, a.jobs)
        if a.cmd == "clean":
            return clean(out, a.shard, a.cpu_threads.split(","))
        return sabotage(out, a.patch, a.lanes)
    except Exception:
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
