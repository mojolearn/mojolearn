#!/usr/bin/env python3
"""lane/merged global check driver (UNTRACKED; never committed).

Reuses tools/algos_lane_check.py's own pieces (needed_bindings, build + stamp,
run_arm, compare) so every verdict means what a lane gate's verdict means.

  merged_check.py plan   --out DIR                 exposed lanes + bindings -> DIR/plan.json
  merged_check.py build  --out DIR [--jobs N]      build every stale binding in parallel (stamped)
  merged_check.py clean  --out DIR --shard i/N [--cpu-threads 1,3,default]
                                                   GPU arm once, CPU arm per thread setting, compare
                                                   each CPU column to the GPU column cell for cell
  merged_check.py light  --out DIR [--batch 20] [--reference a.json,b.json] [--skip-par]
                                                   THE LIGHT CHECK (2026-09-28): both columns in
                                                   batches of lanes, one process per batch; a lane
                                                   that is not AGREE in its batch runs again alone
                                                   and that verdict is recorded; --reference reads
                                                   a CPU column of this commit instead of running
                                                   the CPU arm here
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


def batch_arm(kind, lanes, backend, out, log):
    """One column of MANY lanes in one harness process (the harness takes
    `--lanes a,b,c` and keys its cells `lane/fixture`), so the interpreter, the
    bindings and the device kernels load once per batch, not once per lane.
    Returns the exit status, 124 for a hang; never raises."""
    cmd = [sys.executable, "-u", str(alc.HARNESS), "--lanes", ",".join(lanes), "--repeats", "1",
           "--fail-on-refused", "--json", str(out)]
    cmd += ["--require-cpu", "--require-backend", "cpu"] if kind == "cpu" else ["--require-backend", backend]
    if out.exists():
        out.unlink()
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        try:
            return subprocess.run(cmd, cwd=ROOT, env=alc.arm_env(kind), stdout=fh, stderr=subprocess.STDOUT,
                                  timeout=alc.ARM_TIMEOUT).returncode
        except subprocess.TimeoutExpired:
            fh.write(f"\n[light] HUNG: the {kind} batch did not exit in {alc.ARM_TIMEOUT} s; killed\n")
            return 124


def load_references(paths, head):
    """CPU columns computed ONCE elsewhere, read instead of a CPU arm on this
    box. A reference is refused unless it is a complete IDENTICAL CPU column
    of this very commit. Returns {lane: path}."""
    by_lane = {}
    for p in [Path(x) for x in paths if x]:
        j = json.loads(p.read_text())
        why = ("not a CPU column" if not str(j.get("vendor", "")).startswith("cpu") else
               "not IDENTICAL mode" if j.get("mode") != "identical" else
               "not complete" if j.get("complete") is not True else
               f"commit {str(j.get('commit'))[:12]} is not HEAD {head[:12]}" if j.get("commit") != head else "")
        if why:
            print(f"{now()} REFERENCE REFUSED {p}: {why}", flush=True)
            continue
        for k in j["cells"]:
            by_lane.setdefault(k.split("/")[0], p)
    return by_lane


def light(out, batch, references, skip_par):
    """The light check: the GPU column of every planned lane against a CPU
    column, in batches. A lane whose batch verdict is AGREE is done. Every
    other lane, and every lane of a batch whose process did not exit 0, is
    run again alone through `alc.run_arm` + `alc.compare`, and THAT verdict
    is the one recorded, so a verdict that is not AGREE means exactly what
    `clean` means by it. `par-*` lanes need two GPUs and their CPU arm refuses
    by design; `skip_par` lists them as skipped instead of running them."""
    p = load_plan(out)
    ib = alc.load_harness()
    backend = alc.gpu_backend()
    os.environ.pop("MOJOLEARN_CPU_THREADS", None)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    ref = load_references(references, head)
    lanes = list(p["lanes"])
    skipped = [l for l in lanes if skip_par and l.startswith("par-")]
    lanes = [l for l in lanes if l not in skipped]
    stale = {b for b in sorted(set().union(*[p["needed"][l] for l in lanes] or [set()])) if alc.stale(b)}
    d = out / "light"
    d.mkdir(exist_ok=True)
    log, res = d / "light.log", d / "verdicts.tsv"
    (d / "skipped.txt").write_text("\n".join(skipped) + ("\n" if skipped else ""))
    done = {ln.split("\t")[0] for ln in res.read_text().splitlines() if ln} if res.is_file() else set()
    print(f"{now()} light: {len(lanes)} lanes ({len(done)} done), {len(skipped)} par-* skipped, backend {backend}, "
          f"batch {batch}, reference covers {sum(l in ref for l in lanes)} lanes, HEAD {head[:9]}", flush=True)

    def record(lane, how, verdict, detail, secs):
        with open(res, "a") as fh:
            fh.write("\t".join([lane, how, f"cpudefault={verdict}", str(detail)[:200].replace("\t", " "),
                                f"{secs:.1f}"]) + "\n")
        print(f"{now()} {lane}  {how}  cpudefault={verdict}", flush=True)

    def alone(lane):
        t0 = time.time()
        try:
            sb = sorted(stale & set(p["needed"][lane]))
            if sb:
                raise RuntimeError(f"binding not built: {','.join(sb)}")
            gj = d / f"{lane}.gpu.json"
            alc.run_arm("gpu", lane, backend, "", gj, log)
            cj = ref.get(lane)
            if cj is None:
                cj = d / f"{lane}.cpudefault.json"
                alc.run_arm("cpu", lane, backend, "", cj, log)
            verdict, detail = alc.compare(ib, lane, gj, cj, "", log, backend)
        except Exception as e:
            verdict, detail = "ERROR", str(e).splitlines()[0][:300]
        record(lane, "alone", verdict, detail, time.time() - t0)

    todo = [l for l in lanes if l not in done]
    t_all = time.time()
    for n, i in enumerate(range(0, len(todo), batch)):
        group = todo[i:i + batch]
        ready = [l for l in group if not (stale & set(p["needed"][l]))]
        t0 = time.time()
        agreed = []
        if len(ready) > 1:
            gj, cj = d / f"batch{n:03d}.gpu.json", d / f"batch{n:03d}.cpudefault.json"
            rc_g = batch_arm("gpu", ready, backend, gj, log)
            local = [l for l in ready if l not in ref]
            rc_c = batch_arm("cpu", local, backend, cj, log) if local else 0
            if rc_g == 0 and rc_c == 0 and gj.is_file() and (cj.is_file() or not local):
                for lane in ready:
                    try:
                        verdict, detail = alc.compare(ib, lane, gj, ref.get(lane, cj), "", log, backend)
                    except Exception as e:
                        verdict, detail = "ERROR", str(e).splitlines()[0][:300]
                    if verdict == "AGREE":
                        agreed.append((lane, detail))
            else:
                print(f"{now()} batch {n}: gpu exit {rc_g}, cpu exit {rc_c}; its {len(ready)} lanes run alone",
                      flush=True)
        per = (time.time() - t0) / max(len(ready), 1)
        for lane, detail in agreed:
            record(lane, "batch", "AGREE", detail, per)
        for lane in group:
            if lane not in {l for l, _ in agreed}:
                alone(lane)
    rows = [ln.split("\t") for ln in res.read_text().splitlines() if ln] if res.is_file() else []
    bad = [r[0] for r in rows if r[2] != "cpudefault=AGREE"]
    missing = [l for l in lanes if l not in {r[0] for r in rows}]
    print(f"{now()} LIGHT RESULT: {len(rows)} lanes, {len(bad)} not AGREE, {len(missing)} missing, "
          f"{sum(r[1] == 'batch' for r in rows)} by batch, {sum(r[1] == 'alone' for r in rows)} alone, "
          f"{len(skipped)} par-* skipped, {time.time() - t_all:.0f} s"
          + (": " + ",".join(bad + missing) if bad or missing else ""), flush=True)
    return 1 if bad or missing else 0


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
    ap.add_argument("cmd", choices=("plan", "build", "clean", "light", "sabotage"))
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--reference", default="", help="light: comma separated CPU column JSONs of this commit")
    ap.add_argument("--skip-par", action="store_true", help="light: list par-* lanes as skipped")
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
        if a.cmd == "light":
            return light(out, max(a.batch, 1), a.reference.split(","), a.skip_par)
        return sabotage(out, a.patch, a.lanes)
    except Exception:
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
