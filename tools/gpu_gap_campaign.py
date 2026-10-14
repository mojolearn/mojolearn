#!/usr/bin/env python3
"""Run an explicit list of missing GPU recipes, serially on one frozen worker.

The caller provisions a bounded worker and an earlier process watchdog. This
controller never rents, extends or deletes machines. It does not select work
from timing ratios and never retries a completed recipe implicitly.

--smoke (2026-10-14) answers only "does every recipe start": no timed work, no
fits, no warm-ups. Andrew's design: LOAD EVERYTHING ONCE. One long-lived worker
(tools/smoke_worker.py) imports every driver and library and initializes the
GPU once (both costs recorded in smoke.json), the box facts and the data
hashes are read once, then the worker runs the ENTIRE recipe list in process:
each recipe's check (ours and every opponent arm constructed and read back as
function calls, the parameter audit, the light scorer preflight) under its own
SIGALRM --smoke-timeout (default 5 s) from when that check starts. A recipe
still running then is STARTED (no verdict yet, with how far it got); one that
finishes records MATCHED/REFUSED with every problem string; an exception is
FAILED. Nothing stops the loop: a worker that stops answering (a C call that
never returns) is killed and restarted (recorded) and the next recipe goes on.
Every failure is reported at the end. Apple runs --smoke only (its timing
belongs to the M3 Ultra's own tools), identical then fast.
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
import smoke_inproc as SI

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


def prepare_arm_venvs(args, plan, out):
    """smoke-fixes (2026-10-14; algos/als/taxi-zones: "missing parameter/version probe: implicit-gpu"): every
    recipe runs with --python-env and --skip-install, so the board never created an arm's own venv
    (tools/bench_board_algos.py ARM_VENVS: implicit-gpu needs CUDA 13 libraries the board venv lacks) and the
    arm refused in the board venv. Prepare each planned arm venv once per box under --cache; each recipe's
    board process then finds it (bench_board.setup_arm_venvs, marker-checked). Returns a check record."""
    arms = {a for r in plan if r["family"] == "algos" for a in list(r.get("arms") or ()) + list(r.get("opponents") or ())}
    wanted = arms & set(B.ALGOS.ARM_VENVS.get(args.vendor, {}))
    if not wanted:
        return dict(status="ok", problems=[], arms={})
    got, problems = B.prepare_arm_venvs(os.path.abspath(os.path.expanduser(args.cache)), args.python_env,
                                        args.vendor, wanted, str(out / "arm-venvs.log"))
    return dict(status="ok" if not problems else "failed", problems=sorted(problems.values()), arms=got)


def prepare_mamba_ssm(args, plan, out):
    """smoke-fixes (2026-10-14; neural/mamba1/2/3-forward on AMD: "missing parameter/version probe: mamba-ssm-fp32"):
    every recipe runs with --skip-install, so bench_board's own mamba-ssm build (mamba_ssm_install_steps) never ran
    and the arms refused in every recipe. Build causal-conv1d + mamba-ssm into the board venv once per box
    (tools/install_mamba_ssm.py: idempotent, the same steps as bench_board setup; on AMD it needs the HIP SDK,
    tools/amd_box_runtime_setup.sh). Returns a check record; a failure is reported once here and the arms then
    refuse by name in their recipes."""
    wants = [r for r in plan if r["family"] == "neural" and (
        any(str(a).startswith("mamba-ssm") for a in list(r.get("arms") or ()) + list(r.get("opponents") or ()))
        or str(r["lane"]).startswith("samba"))]
    if not wants or args.vendor not in ("nvidia", "amd"):
        return dict(status="ok", problems=[], skipped="no planned arm needs mamba_ssm")
    M = __import__("install_mamba_ssm")
    rec = M.install(args.python_env, args.vendor, str(out / "mamba-ssm-install.log"))
    return dict(rec, problems=[rec["problem"]] if rec.get("problem") else [],
                recipes=sorted(r["id"] for r in wants))


def fixture_commands(args, lanes, fixtures):
    """The layer lanes' fixture preparation: the graph lanes' data blocks, generate, verify (the timed
    campaign's prepare_fixtures and the smoke's smoke_fixtures run the same three commands)."""
    data = Path(args.cache) / "algos-data/rows-full"
    return [
        [args.python_env, str(ROOT / "tools/bench_board_algos.py"), "prep",
         "--data", str(data), "--lanes", ",".join(lanes), "--datasets", "taxi,istella"],
        [args.python_env, str(ROOT / "tools/neural_fixtures.py"), "generate",
         "--out", str(fixtures), "--data", str(data), "--lanes", ",".join(lanes)],
        [args.python_env, str(ROOT / "tools/neural_fixtures.py"), "verify", "--dir", str(fixtures)],
    ]


def prepare_fixtures(args, plan, out):
    # FAIRNESS FIX (2026-10-13 audit): earlier full-plan audits staged tabular
    # data but no layer fixtures. Generate test inputs with the pinned worker
    # torch, verify every byte and retain them before running an own GPU arm.
    A = B.ALGOS
    lanes = sorted({r["lane"] for r in plan if r["family"] == "algos"
                    and A.LANES[r["lane"]]["kind"] == "layer"})
    if not lanes:
        return None
    fixtures = Path(args.cache) / "algos-data/neural-fixtures"
    for cmd in fixture_commands(args, lanes, fixtures):
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


#: Andrew, 2026-10-14: "keep the default 5 s, and give a longer timer ONLY to the recipes that need it". Per recipe id
#: (every mode), never global: {recipe id: (seconds, reason)}. A row records the timeout it ran under. Rule for the
#: value: the smallest that covers the recipe's measured check with margin; where it never finished within its timer
#: on the box that was slow (no measurement), 30 s. Evidence: apple-smoke-20261014/smoke-b5 and
#: gpu-rerun-20261014/final-nvidia/smoke-b4 (STARTED rows); the same recipes finish in 1-4.4 s on other boxes.
_SLOW_UNMEASURED = "STARTED at 5 s with no measurement of its real check time (%s); 30 s rule"
SMOKE_SLOW = {
    "neural/lm-train-step/bytes/shape=full": (30.0, _SLOW_UNMEASURED % "Apple fast; ours-fast starts its own-tier arm "
                                                                      "server inside the check"),
    "neural/lm-forward/bytes/shape=full": (30.0, _SLOW_UNMEASURED % "Apple fast; ours-fast starts its own-tier arm "
                                                                    "server inside the check"),
    "neural/gemm/gaussian/shape=full": (30.0, _SLOW_UNMEASURED % "Apple fast"),
    "algos/moe/synthetic/rows=full": (30.0, _SLOW_UNMEASURED % "Apple fast+identical, NVIDIA; 4.1-4.4 s where it "
                                                                "finished: the 69M-parameter twin built twice per arm"),
    "algos/gcn/taxi/rows=full": (30.0, _SLOW_UNMEASURED % "Apple identical, killed at 7.5 s"),
    "algos/gcn/istella/rows=full": (30.0, _SLOW_UNMEASURED % "Apple identical, killed at 7.0 s"),
    "classical/kmeans/istella/rows=full": (30.0, _SLOW_UNMEASURED % "NVIDIA; 1.6 s on AMD"),
    "classical/pca/istella/rows=full": (30.0, _SLOW_UNMEASURED % "NVIDIA"),
    "classical/ols/istella/rows=full": (30.0, _SLOW_UNMEASURED % "NVIDIA; 1.3 s on AMD"),
    "algos/conv1d/synthetic/rows=full": (30.0, _SLOW_UNMEASURED % "Apple fast, killed at 5 s; passed under 5 s "
                                                                    "on the run before"),
    "algos/maxpool2d/synthetic/rows=full": (30.0, _SLOW_UNMEASURED % "Apple fast, killed at 5 s; passed under 5 s "
                                                                       "on the run before"),
}


def recipe_timeout(args, race):
    """(seconds, reason) for one recipe's check: --smoke-timeout, or its SMOKE_SLOW entry when that is longer."""
    slow = SMOKE_SLOW.get(race["id"])
    if slow and slow[0] > args.smoke_timeout:
        return slow
    return args.smoke_timeout, "default --smoke-timeout"


def smoke_command(args, race, directory, mode, receipts):
    """One recipe's preflight-only board command: parameter probe plus scorer preflight, nothing timed. The
    smoke worker runs it in process (bench_board.main(cmd[2:])); the stage runs it as a process."""
    python = args.python_env
    cmd = [python, str(ROOT / "tools/bench_board.py"),
           "--vendor", args.vendor, "--python-env", python, "--skip-install",
           "--cache", args.cache, "--rows", "full", "--rounds", "1", "--modes", mode,
           "--gpu-opponents-only", "--preflight-only", "--preflight-scorer", "--evidence-local-only",
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
    # the board's result, else its plan-preflight checkpoint (written as soon as the audit's verdict exists,
    # before the evidence archive is uploaded)
    for path in (Path(directory) / "board.json", Path(directory) / "evidence" / "preflight.json"):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        doc = (doc.get("preflight") or {}) if path.name == "board.json" else doc
        entry = ((doc.get("races") or {}).get(race["id"]) or {}).get("audit")
        if entry is not None:
            break
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
    worker = doc.get("worker") or {}
    if worker:
        starts = worker.get("starts") or []
        lines.append("WORKER %s import_s=%s gpu_init_s=%s starts=%d restarts=%d%s" % (
            worker.get("status"), worker.get("import_seconds"), worker.get("gpu_init_seconds"), len(starts),
            len(worker.get("restarts") or []),
            (": " + str(worker.get("problem") or (starts or [{}])[-1].get("problem") or ""))[:400]
            if worker.get("status") != "serving" else ""))
        for r in worker.get("restarts") or []:
            lines.append("   restarted after %s: %s" % (r.get("after"), str(r.get("reason"))[:300]))
    for name, chk in sorted(doc["checks"].items()):
        lines.append("CHECK %s %s%s" % (name, chk.get("status"), (": " + "; ".join(chk["problems"])[:600])
                                       if chk.get("problems") else ""))
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines.append("COUNTS " + " ".join("%s=%d" % kv for kv in sorted(counts.items())))
    sus = doc.get("suspects")
    if sus is not None:
        prev = ", ".join("%s@%s" % (p.get("path"), str(p.get("source_commit"))[:9]) for p in sus.get("previous") or [])
        lines.append("SUSPECTS ran=%d skipped_known_good=%d from %s%s" % (
            len(sus.get("ran") or {}) if not sus.get("everything") else len(rows),
            len(sus.get("skipped_known_good") or {}), prev or "-",
            ("; EVERYTHING: " + str(sus["everything"])) if sus.get("everything") else
            ("; families changed: " + ",".join(sorted(sus.get("families_changed") or {})) or "")))
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
    lines.append("STARTED, NO VERDICT WITHIN ITS TIMER (default %ss; per-recipe SMOKE_SLOW shown) (%d): these need a "
                 "longer look" % (doc["timeout_s"], len(started)))
    for r in started:
        got = r.get("reached") or {}
        own = r.get("timeout_s")
        lines.append("   %s [%s]%s reached: %s%s%s" % (r["id"], r["mode"],
                                                     " (timer %ss)" % own if own not in (None, doc["timeout_s"]) else "",
                                                     got.get("last_board_line") or "no board line",
                                                     "; parameter probe started" if got.get("parameter_probe_started") else "",
                                                     "; stopped in %s" % got["stopped_at"][-1] if got.get("stopped_at") else ""))
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


def smoke_checks(args, plans, out, worker=None):
    """The box-level checks, once: driver imports, data, the PyPI wheel, our GPU set and the
    retained source snapshot every recipe's check reuses. Returns ({name: check}, receipts for the commands).
    With the smoke worker loaded, its load report is the drivers-and-libraries check."""
    checks, receipts = {}, {}
    races = [r for plan in plans.values() for r in plan]
    if worker is not None and worker.alive():
        checks["drivers-import"] = worker_check(worker.report)
    else:
        probs = []
        for mod in DRIVERS:
            ok, detail = bounded_check([args.python_env, "-c", "import sys; sys.path.insert(0, %r); import %s"
                                        % (str(ROOT / "tools"), mod)])
            if not ok:
                probs.append("%s does not import: %s" % (mod, detail))
        if worker is not None:
            probs += worker_check(worker.report)["problems"]
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
    # the neural fixtures are staged (generated when missing) and verified by smoke_fixtures, after the data
    # stage (smoke-fixes 2026-10-14)
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
    try:
        checks["arm-venvs"] = prepare_arm_venvs(args, races, out)
    except Exception as exc:  # noqa: BLE001 - recorded; the arm then refuses by name in its recipe
        checks["arm-venvs"] = dict(status="failed", problems=["arm venvs raised: %r" % (exc,)])
    try:
        checks["mamba-ssm"] = prepare_mamba_ssm(args, races, out)
    except Exception as exc:  # noqa: BLE001 - recorded; the arms then refuse by name in their recipes
        checks["mamba-ssm"] = dict(status="failed", problems=["mamba-ssm install raised: %r" % (exc,)])
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


class SmokeWorker:
    """The smoke's one long-lived worker process (tools/smoke_worker.py serve) and its outer watchdog."""

    def __init__(self, args, out, env, worker_cmd=None):
        self.args, self.out, self.env = args, Path(out), env
        self.dir = self.out / "worker"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cmd = worker_cmd or [args.python_env, str(ROOT / "tools/smoke_worker.py"), "serve",
                                  "--vendor", args.vendor, "--out", str(self.dir)]
        self.proc, self.buf = None, b""
        self.report = dict(status="starting", starts=[], restarts=[])
        self.warm_request, self.warm_timeout = None, None

    def start(self):
        """Start the worker and wait for its load report (imports and GPU init, recorded per start)."""
        t0 = time.time()
        log = (self.dir / "worker.log").open("ab")
        self.proc = subprocess.Popen(self.cmd, cwd=ROOT, env=self.env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=log, start_new_session=True)
        log.close()
        self.buf = b""
        start = dict(started=t0, pid=self.proc.pid)
        deadline = time.monotonic() + self.args.smoke_load_timeout
        msg = None
        while True:
            msg, self.buf = SI.read_json_line(self.proc, self.buf, deadline)
            if msg is None or msg.get("event") == "ready":
                break
        start["wall_seconds"] = round(time.time() - t0, 2)
        if msg is None:
            start["problem"] = "no load report within %ss (rc %s): %s" % (
                self.args.smoke_load_timeout, self.proc.poll(), "; ".join(_tail(self.dir / "worker.log", 2)))
            self.kill()
        else:
            rep = msg.get("report") or {}
            start.update(report=rep, import_seconds=rep.get("import_seconds"),
                         gpu_init_seconds=rep.get("gpu_init_seconds"))
            if self.warm_request is not None:
                # a restarted worker reads the box facts again, outside every recipe's timer (recorded)
                reply = self.ask(self.warm_request, self.warm_timeout)
                start["box_facts"] = (reply or {}).get("facts") or dict(status="failed", problem="no answer")
        self.report["starts"].append(start)
        self.report.update(status="serving" if self.alive() else "failed",
                           import_seconds=start.get("import_seconds"),
                           gpu_init_seconds=start.get("gpu_init_seconds"))
        return self.report

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def warm(self, request, timeout):
        """The box facts (and arm servers) every recipe reads, once per worker start; the reply's facts."""
        self.warm_request, self.warm_timeout = request, timeout
        reply = self.ask(request, timeout)
        if reply is None:
            self.kill()
            return None
        return reply.get("facts")

    def ask(self, request, timeout):
        """Send one request; its reply, or None when the worker did not answer within `timeout`."""
        if not self.alive():
            return None
        try:
            self.proc.stdin.write((json.dumps(request) + "\n").encode())
            self.proc.stdin.flush()
        except OSError:
            return None
        deadline = time.monotonic() + timeout
        while True:
            msg, self.buf = SI.read_json_line(self.proc, self.buf, deadline)
            if msg is None or msg.get("event") != "noise":
                return msg

    def recipe(self, key, argv, log, timeout):
        """One recipe's in-process check. The worker's own alarm stops it at `timeout`; the outer watchdog
        (timeout + --smoke-grace) kills and restarts a worker that does not answer. Returns (reply or None,
        restart record or None)."""
        reply = self.ask(dict(op="recipe", key=key, argv=list(argv), log=str(log), timeout=timeout),
                         timeout + self.args.smoke_grace)
        if reply is not None:
            return reply, None
        why = ("no answer within %ss (timer %ss + grace %ss): killed" % (
            timeout + self.args.smoke_grace, timeout, self.args.smoke_grace)) if self.alive() else (
            "worker exited (rc %s)" % self.proc.poll() if self.proc is not None else "worker not running")
        self.kill()
        restart = dict(after=key, reason=why, at=time.time())
        self.start()
        restart["started"] = self.alive()
        self.report["restarts"].append(restart)
        return None, restart

    def stage(self, request, timeout):
        """The next recipe's inputs staged in the worker (and its arm servers), outside every recipe's timer
        (smoke-fixes r12, 2026-10-14). Returns the worker's staging record; a worker that does not answer
        within `timeout` is killed and restarted (recorded), and the recipe then builds its inputs inside
        its own check."""
        reply = self.ask(request, timeout)
        if reply is not None:
            return reply.get("staged") or dict(status="failed", problem="stage-data reply without a record")
        why = "no stage-data answer within %ss: killed" % timeout
        self.kill()
        restart = dict(after="stage-data " + str(request.get("key")), reason=why, at=time.time())
        self.start()
        restart["started"] = self.alive()
        self.report["restarts"].append(restart)
        return dict(status="failed", problem=why)

    def kill(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass

    def stop(self):
        if self.alive():
            self.ask(dict(op="quit"), 30)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        self.kill()


def worker_check(report):
    """The drivers-and-libraries check from the worker's load report: a driver or harness module that does
    not import, an installed library that fails to import, or a GPU runtime that fails to initialize is a
    problem; an absent library is listed (the recipe that needs it fails on its own)."""
    start = (report.get("starts") or [{}])[-1]
    rep = start.get("report") or {}
    imports = rep.get("imports") or {}
    probs = ["%s does not import: %s" % (n, r.get("error")) for n, r in sorted(imports.items())
             if r.get("status") == "failed"]
    probs += ["GPU runtime %s did not initialize: %s" % (n, r.get("error"))
              for n, r in sorted((rep.get("gpu_init") or {}).items()) if r.get("status") == "failed"]
    if start.get("problem"):
        probs.append(start["problem"])
    return dict(status="ok" if not probs else "failed", problems=probs,
                absent=sorted(n for n, r in imports.items() if r.get("status") == "absent"),
                import_seconds=start.get("import_seconds"), gpu_init_seconds=start.get("gpu_init_seconds"))


def worker_env(args):
    """The worker's environment: the board's child environment (installed wheel, vendor, data root,
    hipVS lib64 on AMD), so the drivers' environments differ from it only in per-arm settings."""
    ctx = dict(vendor=args.vendor, python=args.python_env, data_root=os.path.abspath(
        os.path.expanduser(args.data_root)), commit=os.environ.get("MOJOLEARN_REPO_COMMIT"))
    env = B.child_env(ctx)
    env.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")
    return env


def plan_data_files(args, plans):
    """Every data file the plans' recipes read (their sha256 is read once by the worker)."""
    ctx = dict(data_root=os.path.abspath(os.path.expanduser(args.data_root)))
    paths = set()
    for plan in plans.values():
        for race in plan:
            try:
                paths.update(p for _, p in B.race_data_files(ctx, race) if p)
            except Exception:  # noqa: BLE001 - that recipe reads it inside its own check instead
                pass
    return sorted(paths)


def stage_request(args, race, arm_python=None):
    """The stage-data request for one recipe (smoke-fixes r12, 2026-10-14): an algos recipe's dataset block,
    lane arrays and generated synthetic inputs (with their sha256), staged in the smoke worker and in the arm
    servers of the race's arm interpreters before the recipe's timer starts.

    smoke-fixes r14 (2026-10-14; NVIDIA in-process smoke, 26 classical/classical2/trees recipes STARTED at 5 s
    in their drivers' own npz reads, which nothing staged): the classical, classical2 and trees recipes too,
    each through its own driver's stage_inputs (the block or decoded dataset, pinned for the whole smoke, and
    what the arms derive from it). None for the other families."""
    cache = os.path.abspath(os.path.expanduser(args.cache))
    family = race.get("family")
    if family == "algos":                                         # the smoke runs --rows full
        driver, data = ROOT / "tools/bench_board_algos.py", B.algos_data_dir(
            dict(algos_data=os.path.join(cache, "algos-data")), None)
        dataset = race["dataset"]
    elif family == "classical":
        driver, data = ROOT / "tools/classical_two_datasets.py", B.classical_data_dir(
            dict(ctd_data=os.path.join(cache, "ctd-data")), None)
        dataset = race["dataset"]
    elif family == "classical2":
        driver, data = ROOT / "tools/bench_board_more.py", B.more_data_dir(
            dict(more_data=os.path.join(cache, "more-data")), None)
        dataset = race["dataset"]
    elif family == "trees":
        # the forest driver's dataset under the board's tree command (bench_board.tree_cmd: --dataset
        # tree_driver_dataset, MOJOLEARN_SPEED_SIZE shipped, the full rows)
        driver = ROOT / "bench/speed/forest_speed_arm.py"
        dataset = B.tree_driver_dataset(race["lane"], race["dataset"])
        data = dict(size="shipped", rows=None, arms=sorted(race.get("opponents") or race.get("arms") or ()))
    else:
        return None
    arms = set(race.get("arms") or ())
    pythons = sorted({py for arm, py in (arm_python or {}).items() if arm in arms and py})
    return dict(op="stage-data", key=race["id"], arm_pythons=pythons,
                items=[dict(driver=str(driver), lane=race["lane"], dataset=dataset, data=data)])


def smoke_stage(args, plans, out, receipts):
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
        cmd = smoke_command(args, next(iter(races.values())), out / "stage" / mode, mode, receipts)
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


def smoke_fixtures(args, plans, out):
    """The layer lanes' neural fixtures, ONCE, before the per-recipe loop, like the data.

    smoke-fixes (2026-10-14; Apple smoke, 26 layer recipes x 2 modes: "quality scorer unavailable: neural
    fixtures: neural fixtures missing: run tools/neural_fixtures.py generate", and from the same refusal
    "missing parameter/version probe: ours|ours-fast" + "missing worker GPU/driver/OS evidence" on every
    layer lane, "no parameter records" on gcn). The smoke only CHECKED for the fixture set (the timed
    campaign's prepare_fixtures generates it before its first recipe), so on a box that had never run the
    timed campaign (the Apple M2 Pro) every ours/ours-fast layer arm refused at construction
    (tools/neural_fixtures.py load_for_lane -> FixturesMissing). Now the smoke stages them itself: when the
    manifest is missing, or lacks a (lane, dataset) a planned layer recipe needs, it runs the timed campaign's
    own commands (fixture_commands: the graph lanes' data blocks, generate, verify) under
    --smoke-stage-timeout each, outside every recipe's timer; then it verifies every file's sha256 against
    the manifest and records the result as the "neural-fixtures" check in smoke.json (dir, manifest sha256,
    entries, torch version, whether it generated, seconds). The recipes read the same directory
    ($MOJOLEARN_NEURAL_FIXTURES, else <cache>/algos-data/neural-fixtures, i.e. <dirname(--data)>/
    neural-fixtures: tools/neural_fixtures.py default_dir)."""
    races = [r for plan in plans.values() for r in plan
             if r["family"] == "algos" and B.ALGOS.LANES[r["lane"]]["kind"] == "layer"]
    if not races:
        return dict(status="ok", problems=[], skipped="no layer recipe")
    need = sorted({(r["lane"], r["dataset"]) for r in races})
    fixdir = Path(os.environ.get("MOJOLEARN_NEURAL_FIXTURES") or Path(args.cache) / "algos-data/neural-fixtures")
    manifest = fixdir / "manifest.json"

    def held():
        try:
            man = json.loads(manifest.read_text())
        except (OSError, ValueError):
            return None, set()
        return man, {(e.get("lane"), e.get("dataset")) for e in (man.get("entries") or {}).values()}

    man, have = held()
    missing = [x for x in need if x not in have]
    rec = dict(dir=str(fixdir), lanes=sorted({l for l, _ in need}), generated=False, problems=[])
    if man is None or missing:
        # generate covers every lane the set already held too: generate rewrites the manifest whole
        lanes = sorted({l for l, _ in need} | {l for l, _ in have})
        log = out / "logs" / "neural-fixtures.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        rec.update(generated=True, generated_lanes=lanes, log=str(log), runs=[],
                   reason="no manifest" if man is None else "missing %s" % ", ".join("/".join(x) for x in missing))
        for cmd in fixture_commands(args, lanes, fixdir)[:2]:          # prep, generate (verify: below)
            rc, seconds = run_bounded(cmd, log, args.smoke_stage_timeout)
            rec["runs"].append(dict(command=Path(cmd[1]).name + " " + cmd[2], returncode=rc, seconds=seconds))
            if rc != 0:
                rec["problems"].append("fixture %s %s: %s" % (cmd[2], "timed out at %ss" % args.smoke_stage_timeout
                                                              if rc is None else "rc %s" % rc,
                                                              "; ".join(_tail(log, 2))))
                break
        man, have = held()
    ok, detail = bounded_check(fixture_commands(args, [], fixdir)[2], timeout=max(600, args.smoke_stage_timeout))
    if not ok:
        rec["problems"].append("fixture verify failed: " + detail)
    missing = [x for x in need if x not in have]
    if missing:
        rec["problems"].append("no fixture entry for %s in %s" % (", ".join("/".join(x) for x in missing), manifest))
    if man is not None:
        rec.update(entries=len(man.get("entries") or {}), torch_version=man.get("torch_version"),
                   torch_geometric_version=man.get("torch_geometric_version"), manifest_sha256=E.file_hash(manifest))
    rec["status"] = "ok" if not rec["problems"] else "failed"
    return rec


def write_smoke(out, doc):
    E.write(out / "smoke.json", doc)
    (out / "smoke.txt").write_text(smoke_text(doc))


def failed_row(race, mode, timeout, problem):
    return dict(id=race["id"], family=race["family"], lane=race["lane"], dataset=race["dataset"], mode=mode,
                returncode=None, seconds=None, timeout_s=timeout, status="FAILED", problems=[problem])


def worker_row(race, mode, reply, restart, directory, log, timeout):
    """A smoke row from the worker's reply: done -> the recipe's own verdict (or FAILED without one),
    timeout -> STARTED, error -> FAILED with the exception, no answer -> STARTED with the restart recorded."""
    if reply is None:
        row = smoke_row(race, mode, None, None, directory, log, timeout)
        row["worker"] = dict(answered=False, restart=restart)
        return row
    status = reply.get("status")
    rc = reply.get("rc") if status == "done" else (None if status == "timeout" else 1)
    row = smoke_row(race, mode, rc, reply.get("seconds"), directory, log, timeout)
    row["worker"] = {k: reply.get(k) for k in ("status", "error", "children_killed", "arm_servers",
                                                "torch_switches_restored", "staging")}
    if reply.get("stopped_at") and isinstance(row.get("reached"), dict):
        row["reached"]["stopped_at"] = reply["stopped_at"]     # where the recipe's timer stopped its check
    if status == "error" and row["status"] == "FAILED":
        row["problems"].insert(0, "recipe check raised in the smoke worker: %s" % reply.get("error"))
    return row


def run_smoke(args, plans, out, suspects=None):
    """--smoke: load everything once (the smoke worker), stage the data once, read the box facts once, then
    every recipe of every mode's plan in the worker, each with its own --smoke-timeout from when its check
    starts. Never stops early: an error, refusal or timeout is recorded and the loop goes on. Uploads the smoke
    archive (smoke.json, smoke.txt, every recipe's log and preflight) to R2 after each recipe."""
    out = out / "smoke"
    if (out / "smoke.json").exists():
        raise ValueError("smoke directory already exists; preserve it and use a new --out")
    out.mkdir(parents=True, exist_ok=True)
    doc = dict(schema=SMOKE_SCHEMA, vendor=args.vendor, modes=sorted(plans), source_commit=B.repo_commit(),
               started=time.time(), timeout_s=args.smoke_timeout, expected=sum(len(p) for p in plans.values()),
               scope="does every recipe start: no timed work, no fits, no warm-ups", checks={}, rows={},
               timer="each recipe's own --smoke-timeout (SIGALRM in the worker), from when its check starts")
    if suspects is not None:
        doc["suspects"] = suspects
    write_smoke(out, doc)
    worker = SmokeWorker(args, out, worker_env(args))
    try:
        doc["worker"] = worker.start()
    except Exception as exc:  # noqa: BLE001 - recorded; every recipe then fails by name
        doc["worker"] = dict(status="failed", problem="smoke worker did not start: %r" % (exc,), starts=[],
                             restarts=[])
    write_smoke(out, doc)
    try:
        doc["checks"], receipts = smoke_checks(args, plans, out, worker)
    except Exception as exc:  # noqa: BLE001 - never stop early
        doc["checks"], receipts = dict(box=dict(status="failed", problems=["box checks raised: %r" % (exc,)])), {}
    write_smoke(out, doc)
    try:
        doc["checks"]["stage"] = smoke_stage(args, plans, out, receipts)
    except Exception as exc:  # noqa: BLE001 - never stop early
        doc["checks"]["stage"] = dict(status="failed", problems=["data stage raised: %r" % (exc,)])
    write_smoke(out, doc)
    try:
        doc["checks"]["neural-fixtures"] = smoke_fixtures(args, plans, out)
    except Exception as exc:  # noqa: BLE001 - never stop early; the layer recipes then refuse by name
        doc["checks"]["neural-fixtures"] = dict(status="failed", problems=["neural fixtures raised: %r" % (exc,)])
    write_smoke(out, doc)
    first = next((r for p in plans.values() for r in p), None)
    if first is not None and worker.alive():
        # the box facts and the data hashes every recipe's check reads, once, outside every recipe's timer
        facts = worker.warm(dict(op="warm-box", argv=smoke_command(args, first, out / "warm-box", "identical",
                                                                   receipts)[2:],
                                 paths=plan_data_files(args, plans),
                                 arm_pythons=sorted(set(((doc["checks"].get("arm-venvs") or {}).get("arms")
                                                         or {}).values()))), args.smoke_stage_timeout)
        doc["worker"]["box_facts"] = facts or dict(status="failed", problem="no answer (worker killed)")
        write_smoke(out, doc)
    members = {}
    try:
        for mode, plan in sorted(plans.items()):
            for race in plan:
                key = "%s|%s" % (mode, race["id"])
                name = "%s.%s" % (mode, race["id"].replace("/", "."))
                directory, log = out / "recipes" / name, out / "logs" / (name + ".log")
                timeout, timeout_reason = recipe_timeout(args, race)
                try:
                    log.parent.mkdir(parents=True, exist_ok=True)
                    if not worker.alive():
                        restart = dict(before=key, reason="worker not running", at=time.time())
                        worker.start()
                        restart["started"] = worker.alive()
                        worker.report["restarts"].append(restart)
                    if not worker.alive():
                        row = failed_row(race, mode, timeout, "smoke worker not running: %s" % (
                            (worker.report.get("starts") or [{}])[-1].get("problem")))
                    else:
                        # the recipe's inputs, loaded once and hashed once, before its own timer starts
                        request = stage_request(args, race, ((doc["checks"].get("arm-venvs") or {}).get("arms")
                                                             or {}))
                        staged = worker.stage(request, args.smoke_stage_timeout) if request else None
                        argv = smoke_command(args, race, directory, mode, receipts)[2:]
                        reply, restart = worker.recipe(key, argv, log, timeout)
                        row = worker_row(race, mode, reply, restart, directory, log, timeout)
                        if staged is not None:
                            row["stage_data"] = staged
                except Exception as exc:  # noqa: BLE001 - recorded, the loop goes on
                    row = failed_row(race, mode, timeout, "smoke harness raised: %r" % (exc,))
                row["timeout_s"], row["timeout_reason"] = timeout, timeout_reason
                doc["rows"][key] = row
                doc["worker"] = worker.report
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
        worker.stop()
    doc["finished"] = time.time()
    write_smoke(out, doc)
    members["smoke.json"] = (out / "smoke.json").read_bytes()
    members["smoke.txt"] = (out / "smoke.txt").read_bytes()
    for path in sorted((out / "worker").glob("*.log")):
        data = _read(path)
        if data is not None:
            members["worker/" + path.name] = data
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
                    help="does every recipe start: everything loaded once in one worker, box checks once, then "
                         "each recipe's preflight-only check in that worker under --smoke-timeout; writes "
                         "<out>/smoke/smoke.json and smoke.txt; no timing")
    ap.add_argument("--smoke-timeout", type=float, default=5.0,
                    help="seconds per recipe check before it is recorded STARTED (no verdict yet) and stopped")
    ap.add_argument("--smoke-grace", type=float, default=30.0,
                    help="seconds past --smoke-timeout before a worker that has not answered is killed and "
                         "restarted (a C call that never returns to Python)")
    ap.add_argument("--smoke-load-timeout", type=float, default=1800.0,
                    help="seconds the smoke worker may take to import everything and initialize the GPU "
                         "(once per worker start, recorded)")
    ap.add_argument("--smoke-stage-timeout", type=float, default=6 * 3600.0,
                    help="seconds for the one-time block prep of every recipe before the per-recipe loop")
    ap.add_argument("--smoke-suspects", metavar="SMOKE_JSON[,...]",
                    help="run only the recipes not known good in these previous smokes (one per vendor; others are "
                         "ignored) plus every recipe whose code changed since their source_commit; the skipped "
                         "recipes are recorded in smoke.json (tools/smoke_suspects.py)")
    ap.add_argument("--smoke-verify-data", action="store_true",
                    help="also sha256 the datasets in the smoke (default: presence and pinned size)")
    ap.add_argument("--data-root", default=os.environ.get(
        "GBM_BENCH_DATA", os.path.join(os.path.expanduser("~"), "datasets", "gbm-bench")),
        help="the board's --data-root (the smoke's data check)")
    args = ap.parse_args(argv)
    if args.smoke:
        if args.smoke_timeout <= 0 or args.smoke_grace < 0:
            ap.error("smoke-timeout must be positive and smoke-grace nonnegative")
        manifest = json.loads(Path(args.manifest).read_text()) if args.manifest else {}
        # Apple: IDENTICAL, then FAST (Apple only) as a second pass
        modes = ("identical", "fast") if args.vendor == "apple" else ("identical",)
        plans = {m: selected_plan(args.vendor, manifest, m, smoke=True) for m in modes}
        os.environ["MOJOLEARN_REPO_COMMIT"] = B.repo_commit()
        suspects = None
        if args.smoke_suspects:
            # Andrew 2026-10-14: only what was not known good last time, plus what the code changes touch
            plans, suspects = __import__("smoke_suspects").select(plans, args.smoke_suspects, args.vendor)
        return run_smoke(args, plans, Path(args.out).resolve(), suspects=suspects)
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
    state["arm_venvs"] = prepare_arm_venvs(args, plan, out)
    state["mamba_ssm"] = prepare_mamba_ssm(args, plan, out)
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
