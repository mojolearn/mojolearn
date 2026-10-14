#!/usr/bin/env python3
"""Run an explicit list of missing GPU recipes, serially on one frozen worker.

The caller provisions a bounded worker and an earlier process watchdog. This
controller never rents, extends or deletes machines. It does not select work
from timing ratios and never retries a completed recipe implicitly.

--smoke (2026-10-14) answers only "does every recipe start": no timed work, no
fits, no warm-ups. It loads every driver and library ONCE in a warm process
(tools/smoke_warm.py; its import cost is recorded separately in smoke.json),
checks the data, the neural fixtures, the PyPI wheel and our GPU set once, then
gives each recipe's preflight-only check (parameter probe plus scorer preflight)
its own hard --smoke-timeout (default 5 s), timed from when that recipe's check
starts with everything already loaded. A recipe still alive then is STARTED (no
verdict yet); one that exits earlier records MATCHED/REFUSED with every problem
string. A recipe that errors, refuses or times out is recorded and the loop goes
on; every failure is reported at the end. Apple runs --smoke only (its timing
belongs to the M3 Ultra's own tools) and keeps one cold process per recipe.
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
import smoke_warm as W

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


def smoke_command(args, race, directory, mode, receipts, python=None):
    """One recipe's preflight-only board process: parameter probe plus scorer preflight, nothing timed.
    `python` is the warm launcher when the warm process serves (every level of the recipe forks from it)."""
    python = python or args.python_env
    cmd = [python, str(ROOT / "tools/bench_board.py"),
           "--vendor", args.vendor, "--python-env", python, "--skip-install",
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
    """A problem's kind for grouping: arm names and numbers folded, the leading "<arm>: " dropped, the text up
    to its first ';' and at most two ': '-separated parts (e.g. "maximize missing from ours read-back",
    "quality scorer unavailable: reference data")."""
    t = re.sub(r"\b(torch|cuml|cupy|jax|sklearn|mamba-ssm|hf|xgboost|lightgbm|catboost|hipvs|cuvs)[-\w.]*",
               "<arm>", str(text))
    t = re.sub(r"^<arm>( \([^)]*\))?: ", "", t).split(";")[0]
    t = ": ".join(t.split(": ")[:2])
    return re.sub(r"\d+(\.\d+)?", "<n>", t).strip()[:160]


def smoke_text(doc):
    """smoke.txt: the box checks, the counts, the problems grouped by kind, then STARTED and FAILED."""
    rows = list(doc["rows"].values())
    lines = ["SMOKE vendor=%s modes=%s recipes=%d timeout=%ss" % (doc["vendor"], ",".join(doc["modes"]),
                                                                  len(rows), doc["timeout_s"])]
    warm = doc.get("warm") or {}
    if warm:
        cold = sum(len((r.get("launches") or {}).get("cold") or []) for r in rows)
        hot = sum((r.get("launches") or {}).get("warm") or 0 for r in rows)
        lines.append("WARM %s import_s=%s starts=%d launches warm=%d cold=%d%s" % (
            warm.get("status"), warm.get("import_seconds"), len(warm.get("starts") or []), hot, cold,
            (": " + str(warm.get("problem") or (warm.get("starts") or [{}])[-1].get("problem") or ""))[:400]
            if warm.get("status") not in ("serving", "off") else ""))
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


def smoke_checks(args, plans, out, warm=None):
    """The box-level checks, once: driver imports, data, neural fixtures, the PyPI wheel, our GPU set and the
    retained source snapshot the recipe processes reuse. Returns ({name: check}, receipts for the commands).
    With the warm process serving, its preload report is the drivers-and-libraries check."""
    checks, receipts = {}, {}
    races = [r for plan in plans.values() for r in plan]
    python = warm.python() if warm is not None else args.python_env
    if warm is not None and warm.serving():
        checks["drivers-import"] = warm_check(warm.report)
    else:
        probs = []
        for mod in DRIVERS:
            ok, detail = bounded_check([args.python_env, "-c", "import sys; sys.path.insert(0, %r); import %s"
                                        % (str(ROOT / "tools"), mod)])
            if not ok:
                probs.append("%s does not import: %s" % (mod, detail))
        if warm is not None:
            probs += warm_check(warm.report)["problems"]
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
            ok, detail = bounded_check([python, str(ROOT / "tools/neural_fixtures.py"), "verify",
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


def warm_wanted(args):
    """--smoke-warm: auto serves on Linux (NVIDIA, AMD); macOS forks are not safe once Metal is loaded."""
    mode = getattr(args, "smoke_warm", "auto")
    return mode == "on" or (mode == "auto" and sys.platform.startswith("linux"))


def _sock_dir(out):
    """A directory for the warm server's socket: under the smoke dir when its path fits a Unix socket
    name (108 bytes), else a fresh private directory under the system temp dir (the socket only)."""
    if len(str(out / "warm" / "s.sock")) < 100:
        return out / "warm"
    import tempfile
    return Path(tempfile.mkdtemp(prefix="mlsw"))


def warm_arch(args, env):
    """mojolearn's GPU arch from one cold process, so the warm server can import mojolearn without cuInit.
    (arch or None, detail)."""
    code = "import sys, mojolearn; sys.stdout.write(str(mojolearn.gpu_arch or ''))"
    try:
        p = subprocess.run([args.python_env, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True,
                           timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, repr(exc)[:400]
    arch = p.stdout.strip().splitlines()[-1].strip() if p.stdout.strip() else ""
    if p.returncode or not arch:
        return None, ("rc %d: " % p.returncode) + (p.stderr.strip().splitlines() or [""])[-1][:400]
    return arch, "reported by a cold `import mojolearn`"


class Warm:
    """The smoke's warm process (tools/smoke_warm.py serve) and its launcher."""

    def __init__(self, args, out):
        self.args, self.out = args, out
        self.dir = out / "warm"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sock = _sock_dir(out) / "s.sock"
        self.launcher = W.write_launcher(self.dir, args.python_env, self.sock)
        self.proc = None
        self.report = dict(status="starting", launcher=str(self.launcher), starts=[])
        ctx = dict(vendor=args.vendor, python=args.python_env, data_root=os.path.abspath(
            os.path.expanduser(args.data_root)), commit=os.environ.get("MOJOLEARN_REPO_COMMIT"))
        self.env = B.child_env(ctx)
        # ours' arm workers run identical (the NVIDIA and AMD smoke is identical only); a child whose numeric
        # mode differs re-imports mojolearn in its own environment
        self.env["MOJOLEARN_NUMERIC_MODE"] = "identical"
        self.arch = None

    def start(self):
        """Start the server and wait for its preload report. Returns the report (import cost, per-module
        status). A server that cannot serve leaves every launch cold, recorded."""
        if self.arch is None:
            self.arch, self.report["mojolearn_arch"] = warm_arch(self.args, self.env)
        ready = Path(str(self.sock) + ".ready")
        for p in (ready, self.sock):
            try:
                p.unlink()
            except OSError:
                pass
        cmd = [self.args.python_env, str(ROOT / "tools/smoke_warm.py"), "serve", "--sock", str(self.sock),
               "--launcher", str(self.launcher)] + (["--mojolearn-arch", self.arch] if self.arch else [])
        t0 = time.time()
        log = (self.dir / "server.log").open("ab")
        self.proc = subprocess.Popen(cmd, cwd=ROOT, env=self.env, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        start = dict(started=t0, pid=self.proc.pid)
        deadline = t0 + self.args.smoke_warm_timeout
        while time.time() < deadline and not ready.exists() and self.proc.poll() is None:
            time.sleep(0.2)
        start["wall_seconds"] = round(time.time() - t0, 2)
        if ready.exists():
            try:
                start.update(json.loads(ready.read_text()))
            except ValueError as exc:
                start.update(serving=False, problem="unreadable ready report: %s" % exc)
        else:
            start.update(serving=False, problem="no preload report within %ss (rc %s): %s" % (
                self.args.smoke_warm_timeout, self.proc.poll(), "; ".join(_tail(self.dir / "server.log", 2))))
            self.stop()
        if start.get("gpu_initialized_by"):
            start["problem"] = "a preload initialized a GPU runtime (%s): not fork safe, every launch runs cold" % (
                start["gpu_initialized_by"])
        self.report["starts"].append(start)
        self.report.update(status="serving" if self.serving() else "cold",
                           import_seconds=start.get("import_seconds"), wall_seconds=start["wall_seconds"])
        return self.report

    def serving(self):
        return self.proc is not None and self.proc.poll() is None and self.sock.exists()

    def python(self):
        """The interpreter recipes run under: the launcher while the server serves, else the real one."""
        return str(self.launcher) if self.serving() else self.args.python_env

    def end_recipe(self):
        """Kill whatever the recipe left alive in the server and return its launch counts."""
        if not self.serving():
            return dict(warm=0, cold=["warm process not serving"])
        reply = W.control(self.sock, "end") or {}
        return {k: reply.get(k) for k in ("warm", "cold", "killed") if k in reply}

    def stop(self):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            W.control(self.sock, "shutdown", timeout=20)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except OSError:
                    pass
                self.proc.wait()


def warm_check(report):
    """The drivers-and-libraries check from the warm server's preload report: a driver that does not
    import, or a library that is installed but fails to import, is a problem; an absent library is
    listed (the recipe that needs it fails on its own)."""
    start = (report.get("starts") or [{}])[-1]
    pre = start.get("preloads") or {}
    probs = ["%s does not import: %s" % (n, r.get("error")) for n, r in sorted(pre.items())
             if r.get("status") == "failed"]
    if start.get("problem"):
        probs.append(start["problem"])
    return dict(status="ok" if not probs else "failed", problems=probs,
                absent=sorted(n for n, r in pre.items() if r.get("status") == "absent"),
                import_seconds=start.get("import_seconds"))


def smoke_stage(args, plans, out, receipts, python=None):
    """Every recipe's untimed block prep (algos, classical, classical2 blocks under --cache), ONCE, before the
    per-recipe loop: one --preflight-only --prep-only board process per mode over that mode's whole plan. A
    recipe's own check then finds its blocks prepared. Returns the stage check."""
    probs, runs = [], {}
    for mode, plan in sorted(plans.items()):
        races = {r["id"]: r for r in plan if r["family"] in ("algos", "classical", "classical2")}
        if not races:
            continue
        ids = out / ("stage-race-ids.%s.txt" % mode)
        ids.write_text("\n".join(sorted(races)) + "\n")
        cmd = smoke_command(args, next(iter(races.values())), out / "stage" / mode, mode, receipts, python)
        cmd[cmd.index("--families") + 1] = ",".join(sorted({r["family"] for r in races.values()}))
        cmd[cmd.index("--lanes") + 1] = ",".join(sorted({r["lane"] for r in races.values()}))
        cmd[cmd.index("--race-ids") + 1] = "@" + str(ids)
        if "--datasets" in cmd:
            i = cmd.index("--datasets")
            del cmd[i:i + 2]
        cmd.append("--prep-only")
        log = out / "logs" / ("stage.%s.log" % mode)
        log.parent.mkdir(parents=True, exist_ok=True)
        rc, seconds = run_bounded(cmd, log, args.smoke_stage_timeout)
        runs[mode] = dict(returncode=rc, seconds=seconds, log=str(log), races=len(races))
        if rc != 0:
            probs.append("data stage (%s) %s: %s" % (mode, "timed out at %ss" % args.smoke_stage_timeout
                                                     if rc is None else "rc %s" % rc, "; ".join(_tail(log, 2))))
    if not runs:
        return dict(status="ok", problems=[], skipped="no recipe prepares blocks")
    return dict(status="ok" if not probs else "failed", problems=probs, runs=runs)


def write_smoke(out, doc):
    E.write(out / "smoke.json", doc)
    (out / "smoke.txt").write_text(smoke_text(doc))


def run_smoke(args, plans, out):
    """--smoke: load everything once (the warm process), then every recipe of every mode's plan, each with its
    own --smoke-timeout from when its check starts. Never stops early: an error, refusal or timeout is
    recorded and the loop goes on. Uploads the smoke archive (smoke.json, smoke.txt, every recipe's log and
    preflight) to R2 after each recipe."""
    out = out / "smoke"
    if (out / "smoke.json").exists():
        raise ValueError("smoke directory already exists; preserve it and use a new --out")
    out.mkdir(parents=True, exist_ok=True)
    doc = dict(schema=SMOKE_SCHEMA, vendor=args.vendor, modes=sorted(plans), source_commit=B.repo_commit(),
               started=time.time(), timeout_s=args.smoke_timeout, expected=sum(len(p) for p in plans.values()),
               scope="does every recipe start: no timed work, no fits, no warm-ups", checks={}, rows={},
               timer="each recipe's own --smoke-timeout, from when its check starts")
    write_smoke(out, doc)
    warm = None
    if warm_wanted(args):
        warm = Warm(args, out)
        try:
            doc["warm"] = warm.start()
        except Exception as exc:  # noqa: BLE001 - recorded; the smoke goes on cold
            doc["warm"] = dict(status="failed", problem=repr(exc)[:600])
            warm = None
    else:
        doc["warm"] = dict(status="off", reason="--smoke-warm off" if getattr(args, "smoke_warm", "auto") == "off"
                           else "macOS: forks are not safe once Metal is loaded; one cold process per recipe")
    write_smoke(out, doc)
    try:
        doc["checks"], receipts = smoke_checks(args, plans, out, warm)
    except Exception as exc:  # noqa: BLE001 - never stop early
        doc["checks"], receipts = dict(box=dict(status="failed", problems=["box checks raised: %r" % (exc,)])), {}
    write_smoke(out, doc)
    try:
        doc["checks"]["stage"] = smoke_stage(args, plans, out, receipts, warm.python() if warm is not None else None)
    except Exception as exc:  # noqa: BLE001 - never stop early
        doc["checks"]["stage"] = dict(status="failed", problems=["data stage raised: %r" % (exc,)])
    write_smoke(out, doc)
    members = {}
    try:
        for mode, plan in sorted(plans.items()):
            for race in plan:
                key = "%s|%s" % (mode, race["id"])
                name = "%s.%s" % (mode, race["id"].replace("/", "."))
                directory, log = out / "recipes" / name, out / "logs" / (name + ".log")
                try:
                    log.parent.mkdir(parents=True, exist_ok=True)
                    if warm is not None and not warm.serving() and len(warm.report["starts"]) < 3:
                        warm.report["restarted_before"] = key
                        doc["warm"] = warm.start()          # the server died: start it again (recorded)
                    python = warm.python() if warm is not None else None
                    rc, seconds = run_bounded(smoke_command(args, race, directory, mode, receipts, python),
                                              log, args.smoke_timeout)
                    row = smoke_row(race, mode, rc, seconds, directory, log, args.smoke_timeout)
                    if warm is not None:
                        row["launches"] = warm.end_recipe()
                except Exception as exc:  # noqa: BLE001 - recorded, the loop goes on
                    row = dict(id=race["id"], family=race["family"], lane=race["lane"], dataset=race["dataset"],
                               mode=mode, returncode=None, seconds=None, timeout_s=args.smoke_timeout,
                               status="FAILED", problems=["smoke harness raised: %r" % (exc,)])
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
    finally:
        if warm is not None:
            warm.stop()
    doc["finished"] = time.time()
    write_smoke(out, doc)
    members["smoke.json"] = (out / "smoke.json").read_bytes()
    members["smoke.txt"] = (out / "smoke.txt").read_bytes()
    for extra in ("warm/server.log",):
        data = _read(out / extra)
        if data is not None:
            members[extra] = data
    try:
        doc["retention"] = retain(out / "smoke.tar.gz", members)
        doc.pop("retention_failure", None)
    except (OSError, ValueError) as exc:
        doc["retention_failure"] = str(exc)[:600]
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
    ap.add_argument("--smoke-warm", choices=("auto", "on", "off"), default="auto",
                    help="load every driver and library once in a warm process and fork each recipe's "
                         "processes from it (auto: Linux on, macOS off)")
    ap.add_argument("--smoke-warm-timeout", type=float, default=1800.0,
                    help="seconds the warm process may take to import everything (once per box, recorded)")
    ap.add_argument("--smoke-stage-timeout", type=float, default=6 * 3600.0,
                    help="seconds for the one-time block prep of every recipe before the per-recipe loop")
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
