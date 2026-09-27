#!/usr/bin/env python3
"""The Apple steward queue for the algorithm expansion
(docs/lanes/ALGORITHM_EXPANSION_PLAN.md, FINAL DECISIONS).

The nine lanes build and verify on rented NVIDIA pods. Metal only compiles on
a Mac, and the MacBook is off-limits to lanes, so the Apple column is TWO
cloud Macs, an M2 Pro (`m2pro`) and an M3 Ultra (`m3ultra`), reached with
tools/cloudmac.sh. Every request is queued for BOTH Macs.

MERGE GATING (Andrew, 2026-09-27): the M3 Ultra is running a multi-hour
GPT-3 training segment and must not be contacted, not even by ssh. So a lane
may merge once its pod (NVIDIA + CPU) and the GATING steward (`m2pro`,
MOJOLEARN_STEWARD_GATING) pass. A DEFERRED steward (`m3ultra`,
MOJOLEARN_STEWARD_DEFERRED; empty it to stop deferring) is never contacted:
`submit` spools its copy of the request on the laptop, and when the GPT-3
run ends the orchestrator runs `apple_steward.py flush-deferred --steward
m3ultra`, which ships the spooled backlog to that Mac's queue. An M3 FAIL
then comes back to the lane as a fix.

ON THE LAPTOP (a lane's agent):
  apple_steward.py submit --lane prep --commit <sha> --verify-lanes robust-scaler,max-abs-scaler \\
      --sabotage /abs/path/sabotage.patch
      copies the request and the patch into BOTH Macs' queues over ssh
  apple_steward.py status [--json]
      collects the verdicts over ssh from every Mac that is not deferred; one
      line per request: PASS (every gating Mac passed), FAIL (any Mac that
      answered failed, with the step) or PENDING; a deferred Mac reads DEFERRED
  apple_steward.py flush-deferred --steward m3ultra   (orchestrator, later)

ON EACH CLOUD MAC (started by the orchestrator, one per Mac):
  apple_steward.py work --steward m2pro [--once]
      works ITS OWN queue, one request at a time (one Metal job per Mac)

For each request the steward, in a private worktree at the commit
($HOME/mojolearn-wt/steward-<name>, from the clone at $MOJOLEARN_STEWARD_REPO,
default ~/mojolearn), runs the one lane check:

  tools/algos_lane_check.sh <verify lanes> --sabotage <patch>

which derives and builds every binding the lanes run (Metal and the CPU host
bindings), requires Metal == CPU (AGREE; NOTHING COMPARED or a missing host
binding is a failure), applies the sabotage (a SOURCE edit), rebuilds and
requires DISAGREE, reverses it with `git apply -R` (never git checkout),
rebuilds and requires AGREE again. The verdict is PASS or FAIL naming the
step. Queues and verdicts live outside the repository, in
$MOJOLEARN_STEWARD_ROOT (default ~/mojolearn-evidence/apple-steward) on each
Mac.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

STEWARDS = ("m2pro", "m3ultra")
GATING = tuple(x for x in os.environ.get("MOJOLEARN_STEWARD_GATING", "m2pro").split(",") if x)
DEFERRED = tuple(x for x in os.environ.get("MOJOLEARN_STEWARD_DEFERRED", "m3ultra").split(",") if x)
REMOTE_ROOT = "~/mojolearn-evidence/apple-steward"
ROOT = Path(os.environ.get("MOJOLEARN_STEWARD_ROOT",
                           Path.home() / "mojolearn-evidence" / "apple-steward"))
REPO = Path(os.environ.get("MOJOLEARN_STEWARD_REPO", Path.home() / "mojolearn"))
TOOLS = Path(__file__).resolve().parent
Q, WORK, DONE, PATCHES = ROOT / "queue", ROOT / "working", ROOT / "done", ROOT / "patches"
SPOOL = ROOT / "deferred"            # on the laptop: requests for a deferred Mac


def _dirs():
    for d in (Q, WORK, DONE, PATCHES):
        d.mkdir(parents=True, exist_ok=True)


def _cloudmac(name, command, stdin=None, timeout=120):
    """One command on cloud Mac `name` through tools/cloudmac.sh."""
    r = subprocess.run([str(TOOLS / "cloudmac.sh"), "ssh", name, command], input=stdin,
                       capture_output=True, timeout=timeout)
    if r.returncode:
        raise SystemExit(f"cloudmac {name}: `{command[:80]}` failed (exit {r.returncode}): "
                         f"{r.stderr.decode(errors='replace').strip()[:300]}")
    return r.stdout.decode(errors="replace")


# --------------------------------------------------------------- the laptop
def submit(a):
    patch = Path(a.sabotage).resolve()
    if not patch.is_file():
        sys.exit(f"sabotage patch {patch} does not exist")
    if not all(c in "0123456789abcdef" for c in a.commit) or len(a.commit) < 7:
        sys.exit(f"--commit must be a hex sha pushed to origin, not {a.commit!r}")
    name = f"{int(time.time() * 1000)}-{a.lane}-{a.commit[:10]}"
    req = {"name": name, "lane": a.lane, "commit": a.commit,
           "verify_lanes": [x for x in a.verify_lanes.split(",") if x],
           "sabotage": f"{REMOTE_ROOT}/patches/{name}.patch",
           "submitted": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    body = json.dumps(req, indent=2).encode()
    for mac in STEWARDS:
        if mac in DEFERRED:
            spool = SPOOL / mac
            spool.mkdir(parents=True, exist_ok=True)
            (spool / f"{name}.patch").write_bytes(patch.read_bytes())
            (spool / f"{name}.json").write_bytes(body)
            print(f"{mac} is deferred: spooled {name} in {spool} (flush-deferred ships it later)")
            continue
        # The patch lands first; the request appears last and atomically (mv),
        # so a steward never claims a request whose patch is not there yet.
        _cloudmac(mac, f"mkdir -p {REMOTE_ROOT}/queue {REMOTE_ROOT}/patches && "
                       f"cat > {REMOTE_ROOT}/patches/{name}.patch", stdin=patch.read_bytes())
        _cloudmac(mac, f"cat > {REMOTE_ROOT}/queue/.{name}.json && "
                       f"mv {REMOTE_ROOT}/queue/.{name}.json {REMOTE_ROOT}/queue/{name}.json", stdin=body)
    print(f"queued {name} on {', '.join(m for m in STEWARDS if m not in DEFERRED)}")


def flush_deferred(a):
    """Ship a deferred Mac's spooled requests to its queue (run once the Mac is
    free, with it no longer listed in MOJOLEARN_STEWARD_DEFERRED)."""
    if a.steward in DEFERRED:
        sys.exit(f"{a.steward} is still deferred (MOJOLEARN_STEWARD_DEFERRED); clear it first")
    spool = SPOOL / a.steward
    for req in sorted(spool.glob("[0-9]*.json")):
        name = req.stem
        _cloudmac(a.steward, f"mkdir -p {REMOTE_ROOT}/queue {REMOTE_ROOT}/patches && "
                             f"cat > {REMOTE_ROOT}/patches/{name}.patch", stdin=(spool / f"{name}.patch").read_bytes())
        _cloudmac(a.steward, f"cat > {REMOTE_ROOT}/queue/.{name}.json && "
                             f"mv {REMOTE_ROOT}/queue/.{name}.json {REMOTE_ROOT}/queue/{name}.json", stdin=req.read_bytes())
        (spool / f"{name}.patch").unlink()
        req.unlink()
        print(f"flushed {name} to {a.steward}")


_REMOTE_LIST = (f"cd {REMOTE_ROOT} 2>/dev/null || exit 0; "
                "for f in queue/[0-9]*.json working/[0-9]*.json; do [ -f \"$f\" ] && echo \"STATE $f\"; done; "
                "for f in done/*/verdict.json; do [ -f \"$f\" ] && { echo \"VERDICT $f\"; cat \"$f\"; echo; }; done; true")


def _collect(mac):
    """{request name: ('queued'|'working'|verdict dict)} from one Mac."""
    out, text = {}, _cloudmac(mac, _REMOTE_LIST)
    dec, pos = json.JSONDecoder(), 0
    while pos < len(text):
        nl = text.find("\n", pos)
        line = text[pos:nl if nl >= 0 else len(text)]
        if line.startswith("STATE "):
            path = line.split(" ", 1)[1]
            out[Path(path).name.split(".")[0]] = path.split("/")[0]
            pos = nl + 1 if nl >= 0 else len(text)
        elif line.startswith("VERDICT "):
            path = line.split(" ", 1)[1]
            obj, end = dec.raw_decode(text, nl + 1)
            out[path.split("/")[1]] = obj
            pos = end
        else:
            pos = nl + 1 if nl >= 0 else len(text)
    return out


def status(a):
    per = {mac: ({} if mac in DEFERRED else _collect(mac)) for mac in STEWARDS}
    for mac in DEFERRED:
        per[mac] = {p.stem: "deferred (spooled)" for p in (SPOOL / mac).glob("[0-9]*.json")}
    names = sorted(set().union(*[set(v) for v in per.values()]))
    rows = []
    for name in names:
        states = {mac: per[mac].get(name, "DEFERRED" if mac in DEFERRED else "missing") for mac in STEWARDS}
        verdicts = {m: s for m, s in states.items() if isinstance(s, dict)}
        if any(v.get("result") == "FAIL" for v in verdicts.values()):
            result = "FAIL"
        elif all(isinstance(states[m], dict) and states[m].get("result") == "PASS" for m in GATING):
            result = "PASS"          # mergeable: every gating Mac passed; a deferred one may still FAIL later
        else:
            result = "PENDING"
        detail = "; ".join(f"{m}: " + (s["result"] + (f" at {s['failed_step']}" if s.get("failed_step") else "")
                                       if isinstance(s, dict) else s) for m, s in states.items())
        rows.append(dict(request=name, result=result, stewards=states))
        if not a.json:
            print(f"{result:8} {name}  ({detail})")
    if a.json:
        print(json.dumps(rows, indent=1))


# --------------------------------------------------------------- a cloud Mac
def _run(cmd, cwd, log, timeout):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        try:
            return subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            f.write(f"TIMEOUT after {timeout}s\n")
            return 124


def _claim(steward):
    for p in sorted(Q.glob("[0-9]*.json")):
        dst = WORK / f"{p.stem}.{steward}.json"
        try:
            p.rename(dst)  # atomic: a request is worked once on this Mac
            return dst
        except FileNotFoundError:
            continue
    return None


def process(req_path, steward):
    req = json.loads(req_path.read_text())
    out = DONE / req["name"]
    out.mkdir(parents=True, exist_ok=True)
    log = out / "steward.log"
    wt = Path.home() / "mojolearn-wt" / f"steward-{steward}"
    verdict = {**req, "steward": steward, "host": os.uname().nodename}

    def finish(result, step=None):
        verdict["result"] = result
        if step:
            verdict["failed_step"] = step
        verdict["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
        req_path.unlink(missing_ok=True)
        print(f"{req['name']}: {result}{' at ' + step if step else ''}", flush=True)

    _run(["git", "fetch", "-q", "origin"], REPO, log, 600)
    if wt.exists():
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=wt,
                               capture_output=True, text=True).stdout.strip()
        if dirty:
            return finish("FAIL", f"the steward worktree {wt} is dirty (an earlier sabotage was not "
                                  f"reversed?); fix it by hand: {dirty[:200]}")
        rc = _run(["git", "checkout", "-q", "--detach", req["commit"]], wt, log, 300)
    else:
        rc = _run(["git", "worktree", "add", "-q", "--detach", str(wt), req["commit"]], REPO, log, 900)
    if rc:
        return finish("FAIL", f"checkout of {req['commit']} (is it pushed to origin?)")
    patch = Path(os.path.expanduser(req["sabotage"]))
    cmd = ["sh", "tools/algos_lane_check.sh", ",".join(req["verify_lanes"]), "--sabotage", str(patch),
           "--out", str(out / "check")]
    rc = _run(cmd, wt, log, 6 * 3600)
    last = [line for line in log.read_text(errors="replace").splitlines() if line.startswith("RESULT:")]
    verdict["check"] = last[-1] if last else "no RESULT line"
    if rc == 0 and last and last[-1].startswith("RESULT: PASS"):
        return finish("PASS")
    return finish("FAIL", last[-1][len("RESULT: "):] if last else f"the lane check exited {rc}")


def work(a):
    _dirs()
    if not (REPO / ".git").exists():
        sys.exit(f"no clone at {REPO} (set MOJOLEARN_STEWARD_REPO)")
    while True:
        req = _claim(a.steward)
        if req:
            process(req, a.steward)
        elif a.once:
            return
        else:
            time.sleep(30)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit", help="laptop: queue a request for both cloud Macs (a deferred one is spooled)")
    s.add_argument("--lane", required=True, help="the expansion lane (linear, ..., ann)")
    s.add_argument("--commit", required=True, help="a commit pushed to origin")
    s.add_argument("--verify-lanes", required=True, help="comma separated identity lanes")
    s.add_argument("--sabotage", required=True, help="a SOURCE patch that must make the check DISAGREE")
    s.set_defaults(fn=submit)
    st = sub.add_parser("status", help="laptop: verdicts; PASS when every gating Mac (m2pro) passed")
    st.add_argument("--json", action="store_true")
    st.set_defaults(fn=status)
    f = sub.add_parser("flush-deferred", help="laptop: ship a formerly deferred Mac's spooled requests")
    f.add_argument("--steward", required=True, choices=STEWARDS)
    f.set_defaults(fn=flush_deferred)
    w = sub.add_parser("work", help="cloud Mac: work this Mac's queue")
    w.add_argument("--steward", required=True, choices=STEWARDS)
    w.add_argument("--once", action="store_true", help="drain the queue, then exit")
    w.set_defaults(fn=work)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
