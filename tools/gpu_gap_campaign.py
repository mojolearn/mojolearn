#!/usr/bin/env python3
"""Run an explicit list of missing GPU recipes, serially on one frozen worker.

The caller provisions a bounded worker and an earlier process watchdog. This
controller never rents, extends or deletes machines. It does not select work
from timing ratios and never retries a completed recipe implicitly.

--smoke (2026-10-14) answers only "does every recipe start": no timed work, no
fits, no warm-ups. It checks the drivers import, the data, the neural fixtures,
the PyPI wheel and our GPU set once, then gives each recipe's preflight-only
board process (parameter probe plus scorer preflight) a hard --smoke-timeout
(default 5 s). A recipe still alive then is STARTED (no verdict yet); one that
exits earlier records MATCHED/REFUSED with every problem string. Apple runs
--smoke only (its timing belongs to the M3 Ultra's own tools).
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import bench_board as B
import bench_board_evidence as E

ROOT = Path(__file__).resolve().parent.parent


def selected_plan(vendor, manifest, mode="identical", smoke=False):
    """The manifest's recipes for `vendor` and `mode`. Apple is allowed for --smoke only (Andrew 2026-10-14:
    the M4 runs smoke and identity; Apple timing is the M3 Ultra's). In a smoke, a manifest without this
    vendor's list selects the vendor's whole GPU-opponent plan."""
    if vendor not in ("nvidia", "amd") and not (smoke and vendor == "apple"):
        raise ValueError("this campaign times NVIDIA and AMD only; Apple runs --smoke only")
    if mode != "identical" and vendor != "apple":
        raise ValueError("fast is Apple only")
    recipes = {r["id"]: r for r in B.gpu_opponents_only(B.plan_races(vendor, [mode]), vendor)
               if r["opponents"]}
    if smoke and vendor not in (manifest or {}):
        return [recipes[rid] for rid in sorted(recipes)]
    wanted = manifest[vendor]
    if not isinstance(wanted, list) or not wanted or len(wanted) != len(set(wanted)):
        raise ValueError("supply a nonempty list of unique recipe IDs")
    missing = set(wanted) - set(recipes)
    if missing:
        raise ValueError("unknown GPU recipes: " + ", ".join(sorted(missing)))
    return [recipes[rid] for rid in wanted]


def command(args, race, directory):
    cmd = [args.python_env, str(ROOT / "tools/bench_board.py"),
           "--vendor", args.vendor, "--python-env", args.python_env, "--skip-install",
           "--cache", args.cache, "--rows", "full", "--rounds", "1", "--modes", "identical",
           "--gpu-opponents-only", "--verify-data", "--no-smoke-gate",
           "--round-seconds", str(getattr(args, "round_seconds", 0)),
           "--families", race["family"], "--lanes", race["lane"], "--out", str(directory)]
    if getattr(args, "wheel_manifest", None):
        # Ours is the PyPI release installed from the manifest's verified files; the board refuses
        # to time unless it loads from them and copies the manifest into every cell's measurement.
        cmd += ["--wheel-manifest", str(Path(args.wheel_manifest).resolve())]
    if getattr(args, "opponent_store", None):
        # The board rechecks recorded parameters, library/device and input
        # identity before reusing a retained opponent from a failed pairing.
        cmd += ["--with-opponents", "--opponent-store", args.opponent_store]
    else:
        cmd += ["--retime-opponents"]
    if race["dataset"] in B.DATASETS:
        cmd += ["--datasets", race["dataset"]]
    return cmd


def logged(cmd, log):
    with Path(log).open("ab", buffering=0) as stream:
        return subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode


def retain(archive, members):
    """Archive `members` ({name: bytes}) at `archive` and upload it to R2 (bench_board_evidence: PUT, then
    GET and sha256 check). Returns the public ref; raises ValueError when retention failed (local bytes
    kept)."""
    ref = E.archive(str(archive), members)
    return E.public_ref(E.upload(ref))


def _read(path, limit=64 * 1024 * 1024):
    path = Path(path)
    return path.read_bytes() if path.is_file() and path.stat().st_size <= limit else None


def recipe_members(row, log, directory):
    """Everything one recipe left (log as we go, 2026-10-14): its log, every preflight.json and the
    board.json it wrote (if any), the parameter-probe logs and its campaign row."""
    members = {"campaign-row.json": E.encoded(row)}
    data = _read(log)
    if data is not None:
        members["recipe.log"] = data
    directory = Path(directory)
    if directory.is_dir():
        for path in sorted(directory.rglob("*")):
            if path.is_file() and (path.name in ("preflight.json", "board.json", "probe.log")
                                   or path.name.endswith(".params.json")):
                data = _read(path)
                if data is not None:
                    members["out/" + str(path.relative_to(directory))] = data
    return members


def retain_recipe(out, rid, row, log, directory, state, path):
    """Upload this recipe's evidence and then campaign.json, right after the recipe finished, refused or
    timed. Returns None or the retention failure."""
    name = rid.replace("/", ".")
    try:
        row["retention"] = retain(out / "retained" / (name + ".tar.gz"),
                                  recipe_members(row, log, directory))
        E.write(path, state)
        ref = retain(out / "retained" / "campaign.tar.gz", {"campaign.json": path.read_bytes()})
        E.write(out / "campaign-retention.json", ref)
        return None
    except (OSError, ValueError) as exc:
        row["retention_failure"] = str(exc)[:600]
        E.write(path, state)
        return str(exc)[:600]


def prepare_fixtures(args, plan, out):
    # FAIRNESS FIX (2026-10-13 audit): earlier full-plan audits staged tabular
    # data but no layer fixtures. Generate test inputs with the pinned worker
    # torch, verify every byte and retain them before running an own GPU arm.
    A = B.ALGOS
    lanes = sorted({r["lane"] for r in plan if r["family"] == "algos"
                    and A.LANES[r["lane"]]["kind"] == "layer"})
    if not lanes:
        return None
    data = Path(args.cache) / "algos-data/rows-full"
    fixtures = Path(args.cache) / "algos-data/neural-fixtures"
    commands = [
        [args.python_env, str(ROOT / "tools/bench_board_algos.py"), "prep",
         "--data", str(data), "--lanes", ",".join(lanes), "--datasets", "taxi,istella"],
        [args.python_env, str(ROOT / "tools/neural_fixtures.py"), "generate",
         "--out", str(fixtures), "--data", str(data), "--lanes", ",".join(lanes)],
        [args.python_env, str(ROOT / "tools/neural_fixtures.py"), "verify", "--dir", str(fixtures)],
    ]
    for cmd in commands:
        if logged(cmd, out / "fixture-preparation.log"):
            raise RuntimeError("fixture preparation failed; no measurements started")
    archive = out / "prepared-fixtures.tar.gz"
    if logged(["tar", "-czf", str(archive), "-C", str(fixtures.parent), fixtures.name], out / "fixture-retention.log"):
        raise RuntimeError("fixture archive failed")
    ref = E.upload(dict(local_path=str(archive), sha256=E.file_hash(archive), bytes=archive.stat().st_size))
    E.write(out / "fixture-retention.json", ref)
    os.environ["MOJOLEARN_NEURAL_FIXTURES"] = str(fixtures)
    return E.public_ref(ref)


# ---------------------------------------------------------------------------
# --smoke: does every recipe start (no timed work)
# ---------------------------------------------------------------------------

SMOKE_SCHEMA = "mojolearn-gpu-gap-smoke/1"
DRIVERS = ("bench_board", "bench_board_algos", "bench_board_neural", "bench_board_more",
           "classical_two_datasets", "bench_board_params", "race_scorer_preflight", "neural_fixtures")


def smoke_command(args, race, directory, mode, receipts):
    """One recipe's preflight-only board process: parameter probe plus scorer preflight, nothing timed."""
    cmd = [args.python_env, str(ROOT / "tools/bench_board.py"),
           "--vendor", args.vendor, "--python-env", args.python_env, "--skip-install",
           "--cache", args.cache, "--rows", "full", "--rounds", "1", "--modes", mode,
           "--gpu-opponents-only", "--preflight-only", "--preflight-scorer",
           "--families", race["family"], "--lanes", race["lane"], "--race-ids", race["id"],
           "--out", str(directory)]
    if race["dataset"] in B.DATASETS:
        cmd += ["--datasets", race["dataset"]]
    if getattr(args, "wheel_manifest", None):
        cmd += ["--wheel-manifest", str(Path(args.wheel_manifest).resolve())]
    if receipts.get("source"):
        cmd += ["--measurement-source", receipts["source"]]
    if receipts.get("box_checked"):
        cmd += ["--box-checked", receipts["box_checked"]]
    return cmd


def run_bounded(cmd, log, timeout, env=None):
    """(returncode or None when still alive at `timeout`, seconds). The whole process group is killed at the
    timeout: the board, its probe driver and every arm worker."""
    t0 = time.time()
    with Path(log).open("ab", buffering=0) as stream:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = None
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    continue
    return rc, round(time.time() - t0, 2)


def _tail(path, n=4, width=300):
    try:
        lines = [l for l in Path(path).read_text(errors="replace").splitlines() if l.strip()]
    except OSError:
        return []
    return [l[:width] for l in lines[-n:]]


def reached(directory, log):
    """What a recipe's process reached before it stopped: the last board stage line, whether its plan
    preflight checkpoint and its parameter probe exist, and the probe log's last lines."""
    directory = Path(directory)
    stages = [l for l in _tail(log, 200) if l.startswith("bench_board")]
    probe_logs = sorted(directory.glob("raw/params-only/*/probe.log"))
    return dict(last_board_line=stages[-1] if stages else None,
                plan_preflight_checkpoint=(directory / "evidence" / "preflight.json").is_file(),
                parameter_probe_started=bool(probe_logs),
                probe_tail=_tail(probe_logs[-1]) if probe_logs else [],
                log_tail=_tail(log))


def smoke_row(race, mode, rc, seconds, directory, log, timeout):
    """One smoke row: MATCHED / REFUSED / UNSUPPORTED from the recipe's own preflight, STARTED when it was
    still running at the timeout (no verdict yet), FAILED when it exited without a preflight verdict."""
    row = dict(id=race["id"], family=race["family"], lane=race["lane"], dataset=race["dataset"], mode=mode,
               returncode=rc, seconds=seconds, timeout_s=timeout, problems=[])
    entry = None
    try:
        board = json.loads((Path(directory) / "board.json").read_text())
        entry = (((board.get("preflight") or {}).get("races") or {}).get(race["id"]) or {}).get("audit")
    except (OSError, ValueError):
        board = None
    if entry is not None:
        row.update(status=entry.get("verdict") or "FAILED", problems=list(entry.get("problems") or []),
                   exceptions=entry.get("exceptions"), scorer_preflight=entry.get("scorer_preflight"))
    elif rc is None:
        row.update(status="STARTED", reached=reached(directory, log))
    else:
        row.update(status="FAILED", reached=reached(directory, log))
        tail = row["reached"]["log_tail"]
        row["problems"].append("exited rc %s before a preflight verdict: %s" % (rc, tail[-1] if tail else "no output"))
    return row


def problem_kind(text):
    """A problem's kind for grouping: the text before its first ': ' with arm names and numbers folded."""
    head = str(text).split(": ", 1)[0]
    head = re.sub(r"\b(torch|cuml|cupy|jax|sklearn|ours|mamba|hf|xgboost|lightgbm|catboost)[-\w.]*", "<arm>", head)
    return re.sub(r"\d+(\.\d+)?", "<n>", head)[:160]


def smoke_text(doc):
    """smoke.txt: the box checks, the counts, the problems grouped by kind, then STARTED and FAILED."""
    rows = list(doc["rows"].values())
    lines = ["SMOKE vendor=%s modes=%s recipes=%d timeout=%ss" % (doc["vendor"], ",".join(doc["modes"]),
                                                                  len(rows), doc["timeout_s"])]
    for name, chk in sorted(doc["checks"].items()):
        lines.append("CHECK %s %s%s" % (name, chk.get("status"), (": " + "; ".join(chk["problems"])[:600])
                                       if chk.get("problems") else ""))
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines.append("COUNTS " + " ".join("%s=%d" % kv for kv in sorted(counts.items())))
    kinds = {}
    for r in rows:
        for text in r.get("problems") or []:
            kinds.setdefault(problem_kind(text), []).append((r, text))
    lines.append("")
    lines.append("PROBLEMS BY KIND (%d kinds)" % len(kinds))
    for kind, hits in sorted(kinds.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        lines.append("== %s (%d)" % (kind, len(hits)))
        for r, text in hits:
            lines.append("   %s [%s] %s" % (r["id"], r["mode"], str(text)[:400]))
    started = [r for r in rows if r["status"] == "STARTED"]
    lines.append("")
    lines.append("STARTED, NO VERDICT WITHIN %ss (%d): these need a longer look" % (doc["timeout_s"], len(started)))
    for r in started:
        got = r.get("reached") or {}
        lines.append("   %s [%s] reached: %s%s" % (r["id"], r["mode"], got.get("last_board_line") or "no board line",
                                                   "; parameter probe started" if got.get("parameter_probe_started") else ""))
    failed = [r for r in rows if r["status"] == "FAILED"]
    lines.append("")
    lines.append("FAILED TO START (%d)" % len(failed))
    for r in failed:
        lines.append("   %s [%s] %s" % (r["id"], r["mode"], "; ".join(r["problems"])[:400]))
    return "\n".join(lines) + "\n"


def bounded_check(cmd, timeout=120):
    """(ok, detail) for one box-level check command."""
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, repr(exc)[:600]
    text = (p.stdout + p.stderr).strip().splitlines()
    return p.returncode == 0, (text[-1] if text else "rc %d" % p.returncode)[:600]


def smoke_checks(args, plans, out):
    """The box-level checks, once: driver imports, data, neural fixtures, the PyPI wheel, our GPU set and the
    retained source snapshot the recipe processes reuse. Returns ({name: check}, receipts for the commands)."""
    checks, receipts = {}, {}
    races = [r for plan in plans.values() for r in plan]
    probs = []
    for mod in DRIVERS:
        ok, detail = bounded_check([args.python_env, "-c", "import sys; sys.path.insert(0, %r); import %s"
                                    % (str(ROOT / "tools"), mod)])
        if not ok:
            probs.append("%s does not import: %s" % (mod, detail))
    checks["drivers-import"] = dict(status="ok" if not probs else "failed", problems=probs)
    needed, corpus_missing = B.data_needs(races)
    root = os.path.abspath(os.path.expanduser(args.data_root))
    data = B.data_status(root, needed, verify=args.smoke_verify_data)
    probs = ["dataset %s missing under %s" % (d, root) for d, r in data.items() if not r["present"]]
    probs += ["dataset %s size %s, pinned %s" % (d, r.get("bytes"), r.get("pinned_bytes"))
              for d, r in data.items() if r.get("size_ok") is False]
    probs += ["dataset %s sha256 differs from the manifest" % d for d, r in data.items() if r.get("sha256_ok") is False]
    probs += ["corpus key %s missing" % k for k in corpus_missing]
    checks["data"] = dict(status="ok" if not probs else "failed", problems=probs, datasets=data,
                          sha256_verified=bool(args.smoke_verify_data))
    layer = sorted({r["lane"] for r in races if r["family"] == "algos" and B.ALGOS.LANES[r["lane"]]["kind"] == "layer"})
    if layer:
        fixdir = os.environ.get("MOJOLEARN_NEURAL_FIXTURES") or str(Path(args.cache) / "algos-data/neural-fixtures")
        if not Path(fixdir).is_dir():
            checks["neural-fixtures"] = dict(status="failed", lanes=layer, problems=[
                "no neural fixtures at %s (the timed campaign generates them before its first recipe)" % fixdir])
        else:
            ok, detail = bounded_check([args.python_env, str(ROOT / "tools/neural_fixtures.py"), "verify",
                                        "--dir", fixdir], timeout=600)
            checks["neural-fixtures"] = dict(status="ok" if ok else "failed", lanes=layer, dir=fixdir,
                                             problems=[] if ok else ["fixture verify failed: " + detail])
    if not args.wheel_manifest:
        checks["wheel"] = dict(status="failed", problems=["no --wheel-manifest: ours has no PyPI wheel provenance"])
    else:
        try:
            manifest, why = B.PYPI.verify(str(Path(args.wheel_manifest).resolve()), args.python_env)
        except (OSError, ValueError) as exc:
            manifest, why = {}, "unreadable wheel manifest: %s" % exc
        if not why and manifest.get("vendor") != args.vendor:
            why = "wheel manifest is for %s" % manifest.get("vendor")
        checks["wheel"] = dict(status="ok" if not why else "failed", problems=[why] if why else [],
                               version=manifest.get("version"), files=[f.get("file") for f in manifest.get("files") or []])
    why = B.gpu_set_refusal(args.python_env, args.vendor)
    checks["gpu-set"] = dict(status="ok" if not why else "failed", problems=[str(why)[:1500]] if why else [])
    try:
        ref, inventory = E.source_snapshot(out / "source")
        ref = E.upload(ref)
        E.write(out / "source.json", dict(source=ref, inventory=inventory))
        receipts["source"] = str(out / "source.json")
        checks["source-snapshot"] = dict(status="ok", problems=[], source=E.public_ref(ref))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        checks["source-snapshot"] = dict(status="failed", problems=["source snapshot not retained: %s" % str(exc)[:600]])
    if checks["wheel"]["status"] == "ok" and checks["gpu-set"]["status"] == "ok":
        receipts["box_checked"] = str(out / "smoke.json")
    return checks, receipts


def write_smoke(out, doc):
    E.write(out / "smoke.json", doc)
    (out / "smoke.txt").write_text(smoke_text(doc))


def run_smoke(args, plans, out):
    """--smoke: every recipe of every mode's plan, one bounded preflight-only process each. Uploads the smoke
    archive (smoke.json, smoke.txt, every recipe's log and preflight) to R2 after each recipe."""
    out = out / "smoke"
    if (out / "smoke.json").exists():
        raise ValueError("smoke directory already exists; preserve it and use a new --out")
    out.mkdir(parents=True, exist_ok=True)
    doc = dict(schema=SMOKE_SCHEMA, vendor=args.vendor, modes=sorted(plans), source_commit=B.repo_commit(),
               started=time.time(), timeout_s=args.smoke_timeout, expected=sum(len(p) for p in plans.values()),
               scope="does every recipe start: no timed work, no fits, no warm-ups", checks={}, rows={})
    write_smoke(out, doc)
    doc["checks"], receipts = smoke_checks(args, plans, out)
    write_smoke(out, doc)
    members = {}
    for mode, plan in sorted(plans.items()):
        for race in plan:
            key = "%s|%s" % (mode, race["id"])
            name = "%s.%s" % (mode, race["id"].replace("/", "."))
            directory, log = out / "recipes" / name, out / "logs" / (name + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            rc, seconds = run_bounded(smoke_command(args, race, directory, mode, receipts), log,
                                      args.smoke_timeout)
            row = smoke_row(race, mode, rc, seconds, directory, log, args.smoke_timeout)
            doc["rows"][key] = row
            members.update({name + "/" + k: v for k, v in recipe_members(row, log, directory).items()})
            write_smoke(out, doc)
            members["smoke.json"] = (out / "smoke.json").read_bytes()
            members["smoke.txt"] = (out / "smoke.txt").read_bytes()
            try:
                doc["retention"] = retain(out / "smoke.tar.gz", members)
                doc.pop("retention_failure", None)
            except (OSError, ValueError) as exc:
                doc["retention_failure"] = str(exc)[:600]
            write_smoke(out, doc)
    doc["finished"] = time.time()
    write_smoke(out, doc)
    print((out / "smoke.txt").read_text().splitlines()[0])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vendor", choices=("nvidia", "amd", "apple"), required=True,
                    help="apple runs --smoke only")
    ap.add_argument("--manifest", help="{vendor: [recipe ids]}; required to time; a smoke without this "
                                       "vendor's list checks the vendor's whole GPU plan")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--python-env", default=sys.executable)
    ap.add_argument("--opponent-store", help="reuse verified matching opponents; measure only missing ones")
    # FAIRNESS FIX (2026-10-13 audit): a fixed 240-second override truncated
    # legitimate full clustering workloads. Reuse the board's established
    # workload budgets unless the caller explicitly supplies another ceiling.
    # These are wall-time safeguards; they never cap estimator iterations.
    ap.add_argument("--round-seconds", type=int, default=0,
                    help="per-round wall-time ceiling; 0 uses the board's workload defaults")
    ap.add_argument("--minimum-start-seconds", type=int, default=600,
                    help="required remaining worker time before starting a recipe; size this for the selected workload")
    ap.add_argument("--stop-at", type=float,
                    help="epoch after which no new recipe starts; leave retention time before provider deadline")
    ap.add_argument("--wheel-manifest",
                    help="PyPI wheel manifest from tools/pypi_wheel_install.py (bench_board.py setup writes "
                         "<out>/wheel-manifest.json); required to measure")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="does every recipe start: box checks once, then each recipe's preflight-only board "
                         "process under --smoke-timeout; writes <out>/smoke/smoke.json and smoke.txt; no timing")
    ap.add_argument("--smoke-timeout", type=float, default=5.0,
                    help="seconds per recipe before it is recorded STARTED (no verdict yet) and killed")
    ap.add_argument("--smoke-verify-data", action="store_true",
                    help="also sha256 the datasets in the smoke (default: presence and pinned size)")
    ap.add_argument("--data-root", default=os.environ.get(
        "GBM_BENCH_DATA", os.path.join(os.path.expanduser("~"), "datasets", "gbm-bench")),
        help="the board's --data-root (the smoke's data check)")
    args = ap.parse_args(argv)
    if args.smoke:
        if args.smoke_timeout <= 0:
            ap.error("smoke-timeout must be positive")
        manifest = json.loads(Path(args.manifest).read_text()) if args.manifest else {}
        # Apple: IDENTICAL, then FAST (Apple only) as a second pass
        modes = ("identical", "fast") if args.vendor == "apple" else ("identical",)
        plans = {m: selected_plan(args.vendor, manifest, m, smoke=True) for m in modes}
        os.environ["MOJOLEARN_REPO_COMMIT"] = B.repo_commit()
        return run_smoke(args, plans, Path(args.out).resolve())
    if args.vendor == "apple":
        ap.error("Apple runs --smoke only; Apple timing is the M3 Ultra's")
    if not args.manifest or args.stop_at is None:
        ap.error("--manifest and --stop-at are required to time")
    if args.round_seconds < 0 or args.minimum_start_seconds < 1:
        ap.error("round-seconds must be nonnegative and minimum-start-seconds positive")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = selected_plan(args.vendor, json.loads(Path(args.manifest).read_text()))
    os.environ["MOJOLEARN_REPO_COMMIT"] = B.repo_commit()
    path = out / "campaign.json"
    if path.exists():
        raise ValueError("campaign directory already exists; preserve it and select remaining work in a new manifest")
    state = dict(vendor=args.vendor, source_commit=B.repo_commit(), started=time.time(),
                 stop_at=args.stop_at, expected=len(plan),
                 scope="explicit missing-evidence recipes; one excluded warmup and one scored sample; no Apple",
                 recipes={r["id"]: dict(status="pending", recipe=r) for r in plan})
    E.write(path, state)
    if args.plan_only:
        return 0
    # PyPI provenance (Andrew 2026-10-14): no recipe starts unless ours loads from the verified
    # PyPI files at the requested version; the manifest goes into campaign.json and every cell.
    if not args.wheel_manifest:
        state["stop_reason"] = "no --wheel-manifest: ours has no PyPI wheel provenance"
        E.write(path, state)
        return 1
    manifest, why = B.PYPI.verify(str(Path(args.wheel_manifest).resolve()), args.python_env)
    state["wheel_manifest"] = manifest
    if not why and manifest.get("vendor") != args.vendor:
        why = "wheel manifest is for %s" % manifest.get("vendor")
    if why:
        state["stop_reason"] = "refused before timing: " + why
        E.write(path, state)
        return 1
    E.write(path, state)
    state["fixture_artifacts"] = prepare_fixtures(args, plan, out)
    E.write(path, state)
    controller_failed = False
    for race in plan:
        if time.time() + args.minimum_start_seconds >= args.stop_at:
            state["stop_reason"] = "insufficient time for another recipe and retention"
            break
        directory = out / "measured" / race["id"].replace("/", ".")
        row = state["recipes"][race["id"]]
        cmd = command(args, race, directory)
        row.update(status="running", started=time.time(), command=cmd)
        E.write(path, state)
        rc = logged(cmd, out / (race["id"].replace("/", ".") + ".log"))
        boardpath = directory / "board.json"
        board = json.loads(boardpath.read_text()) if boardpath.exists() else {}
        row.update(status="process finished", returncode=rc, finished=time.time(), board=str(boardpath),
                   preflight=(board.get("preflight") or {}).get("verdict"),
                   results={rid: {"status": r.get("status"), "audit": r.get("audit"),
                                  "evidence_failure": r.get("evidence_failure")}
                            for rid, r in board.get("races", {}).items()})
        E.write(path, state)
        # LOG AS WE GO (2026-10-14): every recipe, refused or timed, uploads its log, preflight.json,
        # board.json and campaign row, then campaign.json, before the next recipe starts.
        failure = retain_recipe(out, race["id"], row, out / (race["id"].replace("/", ".") + ".log"),
                                directory, state, path)
        if failure:
            state["stop_reason"] = "recipe evidence upload failed (local bytes kept): " + failure
            controller_failed = True
            break
        if any(r.get("evidence_failure") for r in board.get("races", {}).values()):
            state["stop_reason"] = "retention failed; retry upload without rerunning completed measurements"
            controller_failed = True
            break
        # FAIRNESS FIX (2026-10-13 audit): a crash before seal() may leave
        # scored arrays and logs but no final
        # race record. Preserve that work and stop before another measurement.
        # An explicit preflight refusal has not entered the timed driver.
        if row["preflight"] != "REFUSED" and (rc != 0 or not board.get("races")):
            state["stop_reason"] = "controller failed or result missing; inspect and retain existing work before continuing"
            controller_failed = True
            break
    state["finished"] = time.time()
    E.write(path, state)
    return 1 if controller_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
