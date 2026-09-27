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
      pushes the commit to each non-deferred Mac's bare repo (cloudmac.sh push,
      then fetched into the steward clone), then copies the request and the
      patch into BOTH Macs' queues over ssh
  apple_steward.py submit --kind speed --lane linear --commit <sha> \\
      --builds bindings/build_glm.sh --cmd 'pixi run -e default python bench/x.py' [--mode fast] \
      [--target m3ultra|do-amd|both]
      a SPEED job, for the timing Mac (the M3 Ultra, MOJOLEARN_STEWARD_SPEED)
      and/or the AMD steward do-amd (--target; default both, or m3ultra alone
      while do-amd is down): the steward runs the builds, then the timing
      command with nothing else on its GPU (a busy box requeues the job;
      another job appearing during the timing fails it), and records stdout,
      stderr and every wall time in the verdict dir. On do-amd a speed job and
      the identity requests share the one queue, FIFO. While the M3 Ultra is
      deferred, its copy spools on the laptop exactly like identity requests;
      flush-deferred pushes and ships them.
  apple_steward.py status [--json]
      collects the verdicts over ssh from every Mac that is not deferred; one
      line per request: PASS (every gating Mac passed), FAIL (any Mac that
      answered failed, with the step) or PENDING; a deferred Mac reads DEFERRED
  apple_steward.py flush-deferred --steward m3ultra   (orchestrator, later;
      pushes each spooled commit to the Mac before shipping its requests)

THE AMD STEWARD (`do-amd`, 2026-09-27: "treat AMD like Apple"). One
DigitalOcean MI300X/MI325X droplet (tools/do_amd_steward.sh up|extend|down)
runs `work --steward do-amd` as a systemd service. While its state file
($MOJOLEARN_STEWARD_DO_STATE/state.env, default
~/mojolearn-evidence/do-amd-steward) exists on the laptop, `submit` ships every
IDENTITY request to it as well (ssh as root, no cloudmac.sh), and `status`
counts it as GATING for every request it received: a lane merges on m2pro
PASS and do-amd PASS. The droplet fetches the submitted sha from GitHub
(`git fetch --depth=1 origin <sha>`), so the commit must be pushed to origin
(the lane's branch) before `submit`. Speed jobs go to it with --target do-amd
or both (the AMD FAST and IDENTICAL speed phases); on the box a build that
needs one explicit GPU arch (bindings/build_byte_lm.sh) gets
MOJOLEARN_GPU_ARCHS from the device (bincache.device_arch: gfx942), and a
*_host.sh build never gets one. On AMD the
check compares NUMBERS (CPU == AMD); never .so digests.

ON EACH CLOUD MAC (started by the orchestrator, one per Mac):
  apple_steward.py work --steward m2pro [--once]
      works ITS OWN queue, one request at a time (one Metal job per Mac)
  on do-amd (the systemd service): identity requests run up to
      MOJOLEARN_STEWARD_AMD_PARALLEL (6) at a time, one worktree each
      (steward-do-amd, steward-do-amd-1 .. -5); a speed job at the head of the
      FIFO waits for them to finish and runs alone

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

MACS = ("m2pro", "m3ultra")
AMD_STEWARDS = ("do-amd",)
STEWARDS = MACS + AMD_STEWARDS
AMD_STATE = Path(os.environ.get("MOJOLEARN_STEWARD_DO_STATE",
                                Path.home() / "mojolearn-evidence" / "do-amd-steward")) / "state.env"
GATING = tuple(x for x in os.environ.get("MOJOLEARN_STEWARD_GATING", "m2pro").split(",") if x)
DEFERRED = tuple(x for x in os.environ.get("MOJOLEARN_STEWARD_DEFERRED", "m3ultra").split(",") if x)
#: SPEED JOBS go to the M3 Ultra ONLY (Andrew, 2026-09-27): one timing Mac, so
#: every before/after is on the same machine. While it is deferred they spool
#: on the laptop exactly like identity requests. The variable exists for the
#: end-to-end test of the tool on the M2 Pro; a lane never sets it.
SPEED = tuple(x for x in os.environ.get("MOJOLEARN_STEWARD_SPEED", "m3ultra").split(",") if x)
REMOTE_ROOT = "~/mojolearn-evidence/apple-steward"
STEWARD_CLONE = "~/mojolearn"        # MOJOLEARN_STEWARD_REPO's default on the cloud Macs
ROOT = Path(os.environ.get("MOJOLEARN_STEWARD_ROOT",
                           Path.home() / "mojolearn-evidence" / "apple-steward"))
REPO = Path(os.environ.get("MOJOLEARN_STEWARD_REPO", Path.home() / "mojolearn"))
TOOLS = Path(__file__).resolve().parent
Q, WORK, DONE, PATCHES = ROOT / "queue", ROOT / "working", ROOT / "done", ROOT / "patches"
SPOOL = ROOT / "deferred"            # on the laptop: requests for a deferred Mac


def _amd_live():
    """The AMD stewards the laptop can reach now (their state file exists)."""
    return tuple(s for s in AMD_STEWARDS if AMD_STATE.is_file())


def _targets():
    """Every steward an identity request goes to: both Macs, plus do-amd while it is up."""
    return MACS + _amd_live()


def _amd_ssh(command):
    env = {}
    for line in AMD_STATE.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v.strip("'\"")
    return ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", "-i", str(Path.home() / ".ssh" / "id_ed25519"),
            "-o", "IdentitiesOnly=yes", f"root@{env['IP']}", command]


def _dirs():
    for d in (Q, WORK, DONE, PATCHES):
        d.mkdir(parents=True, exist_ok=True)


def _cloudmac(name, command, stdin=None, timeout=120):
    """One command on cloud Mac `name` through tools/cloudmac.sh (the AMD
    steward: plain ssh as root to the droplet in its state file)."""
    argv = _amd_ssh(command) if name in AMD_STEWARDS else [str(TOOLS / "cloudmac.sh"), "ssh", name, command]
    r = subprocess.run(argv, input=stdin, capture_output=True, timeout=timeout)
    if r.returncode:
        raise SystemExit(f"cloudmac {name}: `{command[:80]}` failed (exit {r.returncode}): "
                         f"{r.stderr.decode(errors='replace').strip()[:300]}")
    return r.stdout.decode(errors="replace")


# --------------------------------------------------------------- the laptop
def _push_commit(mac, commit):
    """R1: a cloud Mac's `origin` is the bare repo the laptop pushes to, so a
    submitted commit reaches it only by a push. `cloudmac.sh push <mac> <sha>`
    lands it as refs/steward/<sha> in the bare repo; the steward clone then
    fetches that namespace, so the object is there before the request is
    queued (and `process` fetches it again, for a request that raced this).
    The AMD steward fetches the sha from GitHub instead: it must be pushed to origin."""
    full = subprocess.run(["git", "-C", str(TOOLS.parent), "rev-parse", "--verify", f"{commit}^{{commit}}"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if mac in AMD_STEWARDS:
        # lanes submit concurrently: a fetch that meets another's shallow.lock retries
        got = _cloudmac(mac, f"cd {STEWARD_CLONE} && for i in $(seq 1 40); do "
                             f"git fetch -q --depth=1 origin {full} 2>/dev/null && break; sleep 3; done; "
                             f"git cat-file -t {full}", timeout=900).strip()
        if got != "commit":
            raise SystemExit(f"{mac}: {full[:12]} could not be fetched from GitHub ({got!r}); push it to origin first")
        print(f"{mac}: fetched {full[:12]} from origin into {STEWARD_CLONE}")
        return
    r = subprocess.run([str(TOOLS / "cloudmac.sh"), "push", mac, commit], capture_output=True, timeout=900)
    if r.returncode:
        raise SystemExit(f"cloudmac push {mac} {commit} failed (exit {r.returncode}): "
                         f"{r.stderr.decode(errors='replace').strip()[:300]}; nothing was queued on {mac}")
    full = subprocess.run(["git", "-C", str(TOOLS.parent), "rev-parse", "--verify", f"{commit}^{{commit}}"],
                          capture_output=True, text=True, check=True).stdout.strip()
    got = _cloudmac(mac, f"cd {STEWARD_CLONE} && git fetch -q origin '+refs/steward/*:refs/steward/*' && "
                         f"git cat-file -t {full}", timeout=900).strip()
    if got != "commit":
        raise SystemExit(f"{mac}: {full[:12]} is not in {STEWARD_CLONE} after the push ({got!r}); nothing queued")
    print(f"{mac}: pushed {full[:12]} to its bare repo and fetched it into {STEWARD_CLONE}")


def submit(a):
    if not all(c in "0123456789abcdef" for c in a.commit) or len(a.commit) < 7:
        sys.exit(f"--commit must be a hex sha pushed to origin, not {a.commit!r}")
    name = f"{int(time.time() * 1000)}-{'speed-' if a.kind == 'speed' else ''}{a.lane}-{a.commit[:10]}"
    req = {"name": name, "kind": a.kind, "lane": a.lane, "commit": a.commit,
           "submitted": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    patch = None
    if a.kind == "identity":
        if not (a.verify_lanes and a.sabotage):
            sys.exit("an identity request needs --verify-lanes and --sabotage")
        patch = Path(a.sabotage).resolve()
        if not patch.is_file():
            sys.exit(f"sabotage patch {patch} does not exist")
        req.update(verify_lanes=[x for x in a.verify_lanes.split(",") if x], **{"pass": a.pass_no},
                   sabotage=f"{REMOTE_ROOT}/patches/{name}.patch")
        macs = _targets()
    else:
        if not a.cmd:
            sys.exit("a speed request needs --cmd '<timing command>'")
        builds = [x for x in (a.builds or "").split(",") if x]
        for b in builds:
            if not (b.startswith("bindings/") and b.endswith(".sh")) or ".." in b:
                sys.exit(f"--builds takes bindings/build_*.sh scripts, not {b!r}")
        req.update(builds=builds, cmd=a.cmd, mode=a.mode or "")
        amd = _amd_live()
        target = a.target or ("both" if amd else "m3ultra")
        if target != "m3ultra" and not amd:
            sys.exit(f"--target {target}: the do-amd steward is not up (no {AMD_STATE}); "
                     f"start it with tools/do_amd_steward.sh up, or use --target m3ultra")
        macs = {"m3ultra": SPEED, "do-amd": amd, "both": SPEED + amd}[target]
    req["stewards"] = list(macs)
    body = json.dumps(req, indent=2).encode()
    for mac in macs:
        if mac in DEFERRED:
            spool = SPOOL / mac
            spool.mkdir(parents=True, exist_ok=True)
            if patch:
                (spool / f"{name}.patch").write_bytes(patch.read_bytes())
            (spool / f"{name}.json").write_bytes(body)
            print(f"{mac} is deferred: spooled {name} in {spool} (flush-deferred ships it later)")
            continue
        _push_commit(mac, a.commit)
        _ship(mac, name, body, patch.read_bytes() if patch else None)
    print(f"queued {a.kind} request {name} on {', '.join(m for m in macs if m not in DEFERRED) or 'no Mac yet (deferred)'}")


def _ship(mac, name, body, patch_bytes):
    """The patch lands first; the request appears last and atomically (mv),
    so a steward never claims a request whose patch is not there yet."""
    if patch_bytes is not None:
        _cloudmac(mac, f"mkdir -p {REMOTE_ROOT}/queue {REMOTE_ROOT}/patches && "
                       f"cat > {REMOTE_ROOT}/patches/{name}.patch", stdin=patch_bytes)
    _cloudmac(mac, f"mkdir -p {REMOTE_ROOT}/queue && cat > {REMOTE_ROOT}/queue/.{name}.json && "
                   f"mv {REMOTE_ROOT}/queue/.{name}.json {REMOTE_ROOT}/queue/{name}.json", stdin=body)


def flush_deferred(a):
    """Ship a deferred Mac's spooled requests to its queue (run once the Mac is
    free, with it no longer listed in MOJOLEARN_STEWARD_DEFERRED). Each commit
    is pushed to the Mac before the first request that names it ships."""
    if a.steward in DEFERRED:
        sys.exit(f"{a.steward} is still deferred (MOJOLEARN_STEWARD_DEFERRED); clear it first")
    spool = SPOOL / a.steward
    pushed = set()
    for req in sorted(spool.glob("[0-9]*.json")):
        name = req.stem
        commit = json.loads(req.read_text())["commit"]
        if commit not in pushed:
            _push_commit(a.steward, commit)
            pushed.add(commit)
        pf = spool / f"{name}.patch"
        _ship(a.steward, name, req.read_bytes(), pf.read_bytes() if pf.is_file() else None)
        pf.unlink(missing_ok=True)
        req.unlink()
        print(f"flushed {name} to {a.steward}")


# the cloud Macs' login shell is zsh, where an unmatched glob is an error
_REMOTE_LIST = (f"setopt nullglob 2>/dev/null || true; cd {REMOTE_ROOT} 2>/dev/null || exit 0; "
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


def _is_speed(name):
    return name.split("-")[1:2] == ["speed"]


def status(a):
    targets = _targets()
    per = {mac: ({} if mac in DEFERRED else _collect(mac)) for mac in targets}
    for mac in DEFERRED:
        per[mac] = {p.stem: "deferred (spooled)" for p in (SPOOL / mac).glob("[0-9]*.json")}
    names = sorted(set().union(*[set(v) for v in per.values()]))
    rows = []
    for name in names:
        speed = _is_speed(name)
        # the AMD steward gates every identity request it received
        amd = tuple(m for m in _amd_live() if name in per.get(m, {}))
        if speed:   # the stewards it was sent to: the timing Mac and/or do-amd
            macs = tuple(m for m in SPEED + _amd_live() if name in per.get(m, {})) or SPEED
            gating = macs
        else:
            macs, gating = MACS + amd, GATING + amd
        states = {mac: per[mac].get(name, "DEFERRED" if mac in DEFERRED else "missing") for mac in macs}
        verdicts = {m: s for m, s in states.items() if isinstance(s, dict)}
        if any(v.get("result") == "FAIL" for v in verdicts.values()):
            result = "FAIL"
        elif all(isinstance(states[m], dict) and states[m].get("result") == "PASS" for m in gating):
            result = "PASS"          # identity: mergeable once every gating Mac passed; a deferred one may still FAIL later
        else:
            result = "PENDING"

        def one(m, s):
            if not isinstance(s, dict):
                return f"{m}: {s}"
            out = s["result"] + (f" at {s['failed_step']}" if s.get("failed_step") else "")
            if speed and s.get("timing"):
                t = s["timing"]
                out += f" (builds {t.get('builds_wall_s')}s, cmd {t.get('cmd_wall_s')}s, stdout {s.get('stdout')})"
            return f"{m}: {out}"
        detail = "; ".join(one(m, s) for m, s in states.items())
        rows.append(dict(request=name, kind="speed" if speed else "identity", result=result, stewards=states))
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


def _worktree(steward, slot=0):
    """Slot 0 is the steward's one worktree; do-amd's parallel identity slots
    1..AMD_PARALLEL-1 each have their own (steward-do-amd-<n>)."""
    return Path.home() / "mojolearn-wt" / (f"steward-{steward}" + (f"-{slot}" if slot else ""))


def process(req_path, steward, slot=0):
    req = json.loads(req_path.read_text())
    out = DONE / req["name"]
    out.mkdir(parents=True, exist_ok=True)
    log = out / "steward.log"
    wt = _worktree(steward, slot)
    verdict = {**req, "steward": steward, "host": os.uname().nodename}

    def finish(result, step=None):
        verdict["result"] = result
        if step:
            verdict["failed_step"] = step
        verdict["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
        req_path.unlink(missing_ok=True)
        print(f"{req['name']}: {result}{' at ' + step if step else ''}", flush=True)

    # the laptop pushes a submitted sha as refs/steward/<sha> (cloudmac.sh push);
    # the default refspec fetches only branches, so name that namespace too
    if steward in AMD_STEWARDS:   # a shallow tree: the sha itself, from GitHub
        for _ in range(40):   # a laptop `submit` may hold shallow.lock for a moment
            if not _run(["git", "fetch", "-q", "--depth=1", "origin", req["commit"]], REPO, log, 900):
                break
            time.sleep(3)
    else:
        _run(["git", "fetch", "-q", "origin", "+refs/heads/*:refs/remotes/origin/*", "+refs/steward/*:refs/steward/*"],
             REPO, log, 600)
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
    if req.get("kind") == "speed":
        return speed(req, wt, out, log, verdict, finish)
    patch = Path(os.path.expanduser(req["sabotage"]))
    cmd = ["sh", "tools/algos_lane_check.sh", ",".join(req["verify_lanes"]), "--sabotage", str(patch),
           "--out", str(out / "check")]
    if req.get("pass"):
        cmd += ["--pass", str(req["pass"])]
    rc = _run(cmd, wt, log, 6 * 3600)
    last = [line for line in log.read_text(errors="replace").splitlines() if line.startswith("RESULT:")]
    verdict["check"] = last[-1] if last else "no RESULT line"
    if rc == 0 and last and last[-1].startswith("RESULT: PASS"):
        return finish("PASS")
    return finish("FAIL", last[-1][len("RESULT: "):] if last else f"the lane check exited {rc}")


#: A process whose command line holds one of these is another Metal (or
#: heavy CPU) job; a speed job never times beside one.
FOREIGN = ("lm_segment", "algos_lane_check", "identity_break.py", "mac_slot.sh", "apple_steward.py work",
           "/mojo ", "mojo build", "mojo run", "verify_all", "bench_board")


def _metal_busy():
    """Other jobs on this Mac that would share the Metal queue or the CPU with
    a timing run: [(pid, command)], this steward's own ancestry excluded."""
    fmt = ["-axo", "pid=,ppid=,command="] if sys.platform == "darwin" else ["-eo", "pid=,ppid=,args="]
    ps = subprocess.run(["ps"] + fmt, capture_output=True, text=True).stdout
    procs = {}
    for line in ps.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and parts[0].isdigit():
            procs[int(parts[0])] = (int(parts[1]), parts[2])
    mine, p = set(), os.getpid()
    while p in procs and p not in mine and p > 1:
        mine.add(p)
        p = procs[p][0]
    return [(pid, c[:160]) for pid, (_, c) in sorted(procs.items())
            if pid not in mine and any(f in c for f in FOREIGN) and not c.startswith("ps ")]


def _pixi():
    import shutil
    return shutil.which("pixi") or str(Path.home() / ".pixi" / "bin" / "pixi")


#: Linux build scripts that refuse to build without ONE explicit GPU arch
NEEDS_ARCH = ("bindings/build_byte_lm.sh",)


def _build_env(script, env):
    """A *_host.sh build never gets MOJOLEARN_GPU_ARCHS (a CPU build takes
    none); on Linux a script in NEEDS_ARCH gets this box's own arch
    (bincache.device_arch, gfx942 on do-amd) unless the job's env names one."""
    env = dict(env)
    if script.endswith("_host.sh"):
        env.pop("MOJOLEARN_GPU_ARCHS", None)
    elif script in NEEDS_ARCH and sys.platform != "darwin" and not env.get("MOJOLEARN_GPU_ARCHS"):
        sys.path.insert(0, str(TOOLS))
        import bincache
        arch = bincache.device_arch()
        if arch != "none":
            env["MOJOLEARN_GPU_ARCHS"] = arch
    return env


class Busy(Exception):
    pass


def speed(req, wt, out, log, verdict, finish):
    """A SPEED job (the M3 Ultra or do-amd): the builds, then the timing
    command, with nothing else on this box's GPU. stdout and every wall time go to
    the verdict dir; the result is PASS when every step exited 0 and no other
    job appeared while the command was timed."""
    busy = _metal_busy()
    if busy:
        raise Busy(busy)
    env = dict(os.environ)
    env.pop("MOJOLEARN_NUMERIC_MODE", None)
    if req.get("mode"):
        env["MOJOLEARN_NUMERIC_MODE"] = req["mode"]
    timing = {"builds": [], "quiet_before": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    verdict["timing"] = timing
    t_all = time.time()
    for script in req.get("builds", []):
        t0 = time.time()
        benv = _build_env(script, env)
        with open(log, "a") as f:
            f.write(f"\n$ MOJOLEARN_GPU_ARCHS={benv.get('MOJOLEARN_GPU_ARCHS', '')} pixi run -e default sh {script}\n")
            f.flush()
            # in the pixi default environment, as tools/algos_lane_check.sh builds
            rc = subprocess.run([_pixi(), "run", "-e", "default", "sh", script], cwd=wt, env=benv, stdout=f,
                                stderr=subprocess.STDOUT, timeout=3 * 3600).returncode
        timing["builds"].append({"script": script, "rc": rc, "wall_s": round(time.time() - t0, 3)})
        if rc:
            timing["builds_wall_s"] = round(time.time() - t_all, 3)
            return finish("FAIL", f"build {script} exited {rc}")
    timing["builds_wall_s"] = round(time.time() - t_all, 3)
    busy = _metal_busy()
    if busy:
        return finish("FAIL", f"another job started during the builds, the timing was not run: {busy[:3]}")
    stdout, stderr = out / "speed.stdout", out / "speed.stderr"
    with open(stdout, "w") as fo, open(stderr, "w") as fe:
        t0 = time.time()
        try:
            rc = subprocess.run(["sh", "-c", req["cmd"]], cwd=wt, env=env, stdout=fo, stderr=fe,
                                timeout=3 * 3600).returncode
        except subprocess.TimeoutExpired:
            rc = 124
        timing["cmd_wall_s"] = round(time.time() - t0, 3)
    timing["cmd_rc"] = rc
    verdict["stdout"], verdict["stderr"] = str(stdout), str(stderr)
    verdict["stdout_tail"] = stdout.read_text(errors="replace").splitlines()[-40:]
    after = _metal_busy()
    timing["quiet_after"] = not after
    if after:
        return finish("FAIL", f"another job was running when the timing ended, so the times are not clean: {after[:3]}")
    if rc:
        return finish("FAIL", f"the timing command exited {rc}")
    return finish("PASS")


#: do-amd runs IDENTITY requests up to this many at a time (one worktree
#: each; the MI325X has 256 GB and the check is deterministic whatever else
#: runs). A SPEED job is exclusive: it waits for the running identity jobs to
#: finish, and nothing starts until it is done. The Macs stay at one Metal job.
AMD_PARALLEL = int(os.environ.get("MOJOLEARN_STEWARD_AMD_PARALLEL", "6"))


def _head_is_speed():
    heads = sorted(Q.glob("[0-9]*.json"))
    return bool(heads) and _is_speed(heads[0].stem)


def _work_parallel(a):
    """do-amd: FIFO; identity requests fill free slots, a speed job at the
    head of the queue drains the slots and then runs alone."""
    import threading
    running = {}                         # slot -> thread

    def reap():
        for n in [n for n, t in running.items() if not t.is_alive()]:
            del running[n]

    def one(req, n):
        try:
            process(req, a.steward, n)
        except Exception as exc:        # a crash must not strand the request in working/
            print(f"{req.name}: steward error {exc!r}", flush=True)
            if req.exists():
                req.rename(Q / f"{req.name.split('.')[0]}.json")

    while True:
        reap()
        if _head_is_speed():
            if running:                  # nothing new starts; the speed job waits for the slots
                time.sleep(10)
                continue
            req = _claim(a.steward)
            if req:
                try:
                    process(req, a.steward, 0)
                except Busy as exc:
                    req.rename(Q / f"{req.name.split('.')[0]}.json")
                    print(f"{req.name}: GPU not quiet ({exc.args[0][:2]}); requeued", flush=True)
                    time.sleep(60)
            continue
        free = [n for n in range(max(1, AMD_PARALLEL)) if n not in running]
        req = _claim(a.steward) if free else None
        if req and _is_speed(req.stem) and running:   # a speed job raced in at the head
            req.rename(Q / f"{req.name.split('.')[0]}.json")
            continue
        if req:
            t = threading.Thread(target=one, args=(req, free[0]), daemon=True)
            running[free[0]] = t
            t.start()
            print(f"{req.stem}: started in slot {free[0]} ({len(running)} running)", flush=True)
            continue
        if a.once and not running and not any(Q.glob("[0-9]*.json")):
            return
        time.sleep(10 if running else 30)


def work(a):
    _dirs()
    if not (REPO / ".git").exists():
        sys.exit(f"no clone at {REPO} (set MOJOLEARN_STEWARD_REPO)")
    if a.steward in AMD_STEWARDS and AMD_PARALLEL > 1:
        return _work_parallel(a)
    while True:
        req = _claim(a.steward)
        if req:
            try:
                process(req, a.steward)
            except Busy as exc:
                # a speed job waits for a quiet Mac: back to the queue, untouched
                req.rename(Q / f"{req.name.split('.')[0]}.json")
                print(f"{req.name}: Metal queue not quiet ({exc.args[0][:2]}); requeued", flush=True)
                if a.once:
                    return
                time.sleep(60)
        elif a.once:
            return
        else:
            time.sleep(30)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit", help="laptop: queue a request for both cloud Macs (a deferred one is spooled) "
                                       "and, for identity, the do-amd steward while it is up")
    s.add_argument("--lane", required=True, help="the expansion lane (linear, ..., ann)")
    s.add_argument("--commit", required=True, help="a commit pushed to origin")
    s.add_argument("--kind", choices=("identity", "speed"), default="identity",
                   help="identity (both Macs, m2pro gates) or speed (--target m3ultra|do-amd|both; m3ultra spools while deferred)")
    s.add_argument("--verify-lanes", help="identity: comma separated identity lanes")
    s.add_argument("--sabotage", help="identity: a SOURCE patch that must make the check DISAGREE")
    s.add_argument("--pass", dest="pass_no", type=int, choices=(1, 2), default=2,
                   help="identity: the lane check's --pass (default 2: the steward is a pass-2 step, so every "
                        "fragment needs its .checks with a sabotage patch per driver)")
    s.add_argument("--builds", help="speed: comma separated bindings/build_*.sh, run before the timing")
    s.add_argument("--cmd", help="speed: the timing command, run in the worktree at the commit (sh -c)")
    s.add_argument("--mode", choices=("identical", "fast"), help="speed: MOJOLEARN_NUMERIC_MODE for builds and cmd")
    s.add_argument("--target", choices=("m3ultra", "do-amd", "both"),
                   help="speed: the timing Mac, the AMD steward, or both (default: both while do-amd is up, "
                        "else m3ultra)")
    s.set_defaults(fn=submit)
    st = sub.add_parser("status", help="laptop: verdicts; PASS when every gating Mac (m2pro) passed")
    st.add_argument("--json", action="store_true")
    st.set_defaults(fn=status)
    f = sub.add_parser("flush-deferred", help="laptop: ship a formerly deferred Mac's spooled requests")
    f.add_argument("--steward", required=True, choices=MACS)
    f.set_defaults(fn=flush_deferred)
    w = sub.add_parser("work", help="cloud Mac: work this Mac's queue")
    w.add_argument("--steward", required=True, choices=STEWARDS)
    w.add_argument("--once", action="store_true", help="drain the queue, then exit")
    w.set_defaults(fn=work)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
