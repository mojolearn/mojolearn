#!/usr/bin/env python3
"""Consolidated source-tree GPU versus CPU check driver.

Reuses tools/algos_lane_check.py's own pieces (needed_bindings, build + stamp,
run_arm, compare) so every verdict means what a lane gate's verdict means.

  check.py plan   --out DIR                 exposed lanes + bindings -> DIR/plan.json
  check.py build  --out DIR [--jobs N]      build every stale binding in parallel (stamped)
  check.py clean  --out DIR --shard i/N [--cpu-threads 1,3,default]
                                                   GPU arm once, CPU arm per thread setting, compare
                                                   each CPU column to the GPU column cell for cell
Resume accepts only the same committed sources, native bytes and execution settings.
Failures, missing evidence and incomplete selections never pass.
"""
import argparse, fcntl, hashlib, importlib.util, json, os, subprocess, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import algos_lane_check as alc  # noqa: E402


def source_commit():
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
    if dirty:
        raise ValueError("commit tracked source changes before running consolidated checks")
    return sha


def fingerprint(p, lanes, cpu_threads, backend, fixtures="base", arm_timeout=120):
    bindings = sorted(set().union(*(set(p["needed"][l]) for l in lanes)))
    paths = {b: alc.output_for(b) for b in bindings}
    math = alc.PKG / (".dylibs/libMojolearnMath.dylib" if sys.platform == "darwin" else ".libs/libMojolearnMath.so")
    paths["portable_math"] = math
    hashes = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"missing binding/library: {path}")
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        hashes[name] = h.hexdigest()
    return dict(source_commit=source_commit(), bindings=hashes, lanes=lanes,
                cpu_threads=cpu_threads, backend=backend, arch=alc.gpu_arch(),
                fixtures=fixtures, arm_timeout=arm_timeout,
                settings={k: v for k, v in sorted(os.environ.items())
                          if k.startswith(("MOJOLEARN_", "OMP_", "MTL_"))})


def read_rows(path, lanes, threads):
    rows = [line.split("\t") for line in path.read_text().splitlines() if line]
    seen = set()
    for row in rows:
        if row[0] not in lanes or row[0] in seen:
            raise ValueError("unexpected or duplicate result lane")
        seen.add(row[0])
        if len(row) == 2 and row[1].startswith("ERROR="):
            continue
        if len(row) != 1 + 2 * len(threads):
            raise ValueError("incomplete CPU comparison columns")
        for idx, thread in enumerate(threads):
            if not row[1 + 2 * idx].startswith(f"cpu{thread}="):
                raise ValueError("missing or reordered CPU comparison column")
    return rows


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def structural_exclusions(lanes):
    # Read the source manifest without importing GPU bindings or fitting models.
    spec = importlib.util.spec_from_file_location("consolidated_host_surface", ROOT / "python/mojolearn/host_surface.py")
    surface = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(surface)
    return {lane: row["reason"] for lane, row in surface.lane_exposure(lanes, "cpu").items()
            if row["status"] == surface.LANE_NOT_APPLICABLE}


def plan(out, only=""):
    source = source_commit()
    ib = alc.load_harness()
    inventory = sorted(ib.LANES)
    excluded = structural_exclusions(inventory)
    candidates = [lane for lane in inventory if lane not in excluded]
    lanes, needed, refused = candidates, {}, {}
    try:
        needed = alc.needed_bindings(candidates)
        if set(needed) != set(candidates):
            raise ValueError("binding inventory is incomplete")
    except Exception:
        lanes = []
        for lane in candidates:
            try:
                needed[lane] = alc.needed_bindings([lane])[lane]
                lanes.append(lane)
            except Exception as exc:
                refused[lane] = str(exc).splitlines()[0][:300]
    if refused and not only:
        raise ValueError(f"unavailable lanes cannot be silently omitted: {refused}")
    if only:
        selected = only.split(",")
        if len(set(selected)) != len(selected) or any(l not in lanes for l in selected):
            raise ValueError("duplicate, unknown, or unavailable requested lanes")
        lanes = selected
        needed = {l: needed[l] for l in lanes}
    if not lanes:
        raise ValueError("empty lane selection")
    omitted = {l: (excluded[l] if l in excluded else "not selected for this run") for l in inventory if l not in lanes}
    p = {"inventory": inventory, "omitted": omitted, "structural_exclusions": excluded,
         "source_commit": source, "lanes": lanes, "needed": needed, "not_exposed": refused,
         "bindings": sorted(set().union(*needed.values()))}
    (out / "plan.json").write_text(json.dumps(p, indent=1))
    print(f"{now()} plan: {len(inventory)} inventory, {len(lanes)} selected, {len(omitted)} explicitly omitted, "
          f"{len(refused)} unavailable, "
          f"{len(p['bindings'])} bindings", flush=True)
    return p


def load_plan(out):
    f = out / "plan.json"
    p = json.loads(f.read_text()) if f.is_file() else plan(out)
    if p.get("source_commit") != source_commit():
        raise ValueError("plan source changed; choose a new output directory")
    return p


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


def clean(out, shard, cpu_threads, fixtures="base", arm_timeout=120):
    p = load_plan(out)
    i, n = (int(x) for x in shard.split("/"))
    if n <= 0 or i < 0 or i >= n:
        raise ValueError("invalid shard")
    lanes = p["lanes"][i::n]
    if not lanes or not cpu_threads or len(set(cpu_threads)) != len(cpu_threads):
        raise ValueError("empty selection or duplicate CPU thread columns")
    if any(t != "default" and (not t.isdigit() or int(t) < 1) for t in cpu_threads):
        raise ValueError("invalid CPU thread count")
    ib = alc.load_harness()
    if arm_timeout <= 0:
        raise ValueError("arm timeout must be positive")
    alc.ARM_TIMEOUT = arm_timeout
    if fixtures == "all":
        fixtures = ""
    selected_fixtures = fixtures.split(",") if fixtures else list(ib.FIXTURES)
    if not selected_fixtures or len(set(selected_fixtures)) != len(selected_fixtures) or any(f not in ib.FIXTURES for f in selected_fixtures):
        raise ValueError("empty, duplicate or unknown fixtures")
    backend = alc.gpu_backend()
    stale = {b for b in sorted(set().union(*[p["needed"][l] for l in lanes])) if alc.stale(b)}
    if stale:  # a binding that did not build fails only the lanes that run it
        print(f"STALE (not built): {sorted(stale)}; their lanes are recorded as ERROR", flush=True)
    d = out / f"clean_{i}of{n}"
    d.mkdir(exist_ok=True)
    log = d / "clean.log"
    res = d / "verdicts.tsv"
    identity = fingerprint(p, lanes, cpu_threads, backend, fixtures, arm_timeout)
    identity_file = d / "identity.json"
    if identity_file.exists():
        if json.loads(identity_file.read_text()) != identity:
            raise ValueError("resume source, bindings, settings, or selection changed; use a new output directory")
    elif res.exists():
        raise ValueError("unfingerprinted results cannot be resumed")
    else:
        identity_file.write_text(json.dumps(identity, indent=2) + "\n")
    done = set()
    if res.is_file():  # resumable: skip lanes already verdicted
        previous = read_rows(res, lanes, cpu_threads)
        for row in previous:
            if any(c.startswith("ERROR=") for c in row[1:]):
                continue
            for suffix in ["gpu"] + [f"cpu{t}" for t in cpu_threads]:
                evidence = d / f"{row[0]}.{suffix}.json"
                if not evidence.is_file() or not isinstance(json.loads(evidence.read_text()), dict):
                    raise ValueError("missing or invalid resumed comparison evidence")
        done = {r[0] for r in previous}
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
            alc.run_arm("gpu", lane, backend, fixtures, gj, log)
            for t in cpu_threads:
                os.environ.clear()
                os.environ.update(base_env)
                if t == "default":
                    os.environ.pop("MOJOLEARN_CPU_THREADS", None)
                else:
                    os.environ["MOJOLEARN_CPU_THREADS"] = t
                cj = d / f"{lane}.cpu{t}.json"
                alc.run_arm("cpu", lane, backend, fixtures, cj, log)
                verdict, detail = alc.compare(ib, lane, gj, cj, fixtures, log, backend)
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
    if fingerprint(p, lanes, cpu_threads, backend, fixtures, arm_timeout) != identity:
        raise ValueError("source or bindings changed during the run")
    rows = read_rows(res, lanes, cpu_threads)
    if {r[0] for r in rows} != set(lanes):
        raise ValueError("incomplete lane results")
    bad = [r[0] for r in rows if any(c.startswith("ERROR=") or ("=" in c and c.split("=", 1)[1] != "AGREE"
                                                                  and c.startswith("cpu")) for c in r[1:])]
    print(f"{now()} scope: {len(p['inventory'])} inventory, {len(p['omitted'])} omitted (reasons in plan.json); only this shard compared", flush=True)
    print(f"{now()} CLEAN RESULT shard {i}/{n}: {len(rows)} lanes, {len(bad)} not AGREE"
          + (": " + ",".join(bad) if bad else ""), flush=True)
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "build", "clean"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--cpu-threads", default="default")
    ap.add_argument("--lanes", default="")
    ap.add_argument("--fixtures", default="base", help="comma-separated fixtures; all explicitly enables every fixture")
    ap.add_argument("--arm-timeout", type=int, default=120, help="maximum seconds per GPU or CPU arm")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".check.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.cmd == "plan":
            plan(out, a.lanes)
            return 0
        if a.cmd == "build":
            return build(out, a.jobs)
        if a.cmd == "clean":
            return clean(out, a.shard, a.cpu_threads.split(","), a.fixtures, a.arm_timeout)
        raise ValueError("unknown command")
    except Exception:
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
