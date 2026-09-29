#!/usr/bin/env python3
"""The Apple steward queue for the algorithm expansion
(docs/lanes/ALGORITHM_EXPANSION_PLAN.md, FINAL DECISIONS).

The nine lanes build and verify on rented NVIDIA pods. Metal only compiles on
a Mac, and the MacBook is off-limits to lanes, so the Apple column is the
FLEET of cloud Macs in ~/mojolearn-evidence/cloudmacs.tsv (MOJOLEARN_CLOUDMAC_REG;
columns name, instance, host, ip), reached with tools/cloudmac.sh. A Mac's
MODEL is its name without a trailing -<suffix> (m4pro-a -> m4pro, m4-a -> m4,
m3ultra-b -> m3ultra) and its GENERATION the chip family (M2, M3, M4).

ROUTING (Andrew, 2026-09-27): an IDENTITY request goes to ONE Mac per
generation (the least busy one: fewest queued + working requests), not to
every Mac: M2 -> m2pro, M3 -> m3ultra or m3ultra-b, M4 -> m4pro-a, m4pro-b
or m4-a. MERGE GATE: a lane may merge once ANY one Apple steward PASSES and
do-amd PASSES. The other generations' verdicts follow; a later FAIL (M2 vs
M3 vs M4 codegen) comes back to the lane as a fix at the root. A SPEED job
goes to the least busy Mac of the requested model or generation (default
m4pro). Each Mac works one request at a time (one Metal job per Mac).

A DRAINING Mac (MOJOLEARN_STEWARD_DRAIN, comma separated; e.g. before its
host is released) gets no new request; `redistribute --apply` moves its
queued requests to the other Macs of its generation.
A DEFERRED Mac (MOJOLEARN_STEWARD_DEFERRED, comma separated; default none)
is never contacted, and a Mac that does not answer ssh is skipped: when every
Mac of a generation is deferred or down, `submit` spools that generation's
copy on the laptop (~/mojolearn-evidence/apple-steward/deferred/<mac>), and
`flush-deferred --steward <mac>` ships it later.

ON THE LAPTOP (a lane's agent):
  apple_steward.py submit --lane prep --commit <sha> --verify-lanes robust-scaler,max-abs-scaler \\
      --sabotage /abs/path/sabotage.patch
      picks one Mac per generation, pushes the commit to each (cloudmac.sh
      push, then fetched into the steward clone), then copies the request and
      the patch into its queue over ssh; do-amd too while it is up
  apple_steward.py submit --kind speed --lane linear --commit <sha> \\
      --builds bindings/build_glm.sh --cmd 'pixi run -e default python bench/x.py' [--mode fast] \\
      [--target m4pro|m4|m3ultra|m2pro|M4|M3|M2|<mac name>|apple|do-amd|both]
      a SPEED job for the least busy Mac of that model or generation (a Mac
      name pins one Mac, e.g. to time a before and an after on the same box;
      apple = m4pro) and/or the AMD steward do-amd (default both = m4pro +
      do-amd while do-amd is up, else m4pro). The steward runs the builds, then the timing
      command with nothing else on its GPU (a busy box requeues the job;
      another job appearing during the timing fails it), and records stdout,
      stderr and every wall time in the verdict dir. On do-amd a speed job and
      the identity requests share the one queue, FIFO.
  apple_steward.py status [--json]
      collects the verdicts over ssh from every Mac (and do-amd); one line per
      request: PASS (identity: one Apple PASS and every do-amd copy PASS;
      speed: every copy PASS), FAIL (any steward failed, with the step) or PENDING
  apple_steward.py redistribute [--apply]
      spreads the pending backlog over the fleet: every pending identity
      request ends with exactly one copy per generation (a missing one is
      copied to the least busy Mac of that generation, a duplicate queued
      copy is withdrawn), queued copies are rebalanced between the Macs of a
      generation, laptop spools are shipped. A queued request moves by an
      atomic mv out of the source queue (a request the steward already
      claimed is left where it is), so nothing is lost or run twice.
  apple_steward.py flush-deferred --steward <mac>   (orchestrator, later;
      pushes each spooled commit to the Mac before shipping its requests)

THE AMD STEWARD (`do-amd`, 2026-09-27: "treat AMD like Apple"). One
DigitalOcean MI300X/MI325X droplet (tools/do_amd_steward.sh up|extend|down)
runs `work --steward do-amd` as a systemd service. While its state file
($MOJOLEARN_STEWARD_DO_STATE/state.env, default
~/mojolearn-evidence/do-amd-steward) exists on the laptop, `submit` ships every
IDENTITY request to it as well (ssh as root, no cloudmac.sh), and `status`
counts it as GATING for every request it received: a lane merges on one
Apple PASS and do-amd PASS. The droplet fetches the submitted sha from GitHub
(`git fetch --depth=1 origin <sha>`), so the commit must be pushed to origin
(the lane's branch) before `submit`. Speed jobs go to it with --target do-amd
or both (the AMD FAST and IDENTICAL speed phases); on the box a build that
needs one explicit GPU arch (bindings/build_byte_lm.sh) gets
MOJOLEARN_GPU_ARCHS from the device (bincache.device_arch: gfx942), and a
*_host.sh build never gets one. On AMD the
check compares NUMBERS (CPU == AMD); never .so digests.

ON EACH CLOUD MAC (a launchd daemon, tools/cloudmac.sh steward <mac> install):
  apple_steward.py work --steward <mac> [--once]
      works ITS OWN queue, one request at a time (one Metal job per Mac)
  on do-amd (the systemd service): BUILD ONCE, RUN MANY. One builder thread
      builds each distinct (commit, lanes) once, in its own worktree
      (steward-do-amd-build), into the shared store $MOJOLEARN_STEWARD_ROOT/builds
      (tools/steward_build.py: bindings keyed by their source-closure digest,
      READY markers per commit and lane); identity requests whose builds are
      done run up to MOJOLEARN_STEWARD_AMD_PARALLEL (6) at a time, one worktree
      each (steward-do-amd, steward-do-amd-1 .. -5), oldest first, so a build
      never holds a slot and requests at one commit share one build. A speed
      job at the head of the FIFO waits for the slots and the builder and runs
      alone. Every steward (the Macs too, one Metal job at a time) seeds its
      worktree from the store before a check and publishes what it built
      clean, so a commit is built once per box.

SPEED DUPLICATES: a speed `submit` identical to one still queued or running
on the chosen steward (lane, commit, builds, command, mode) is not queued
again there (--no-coalesce overrides).

RESTARTS NEVER MOVE THE QUEUE: `cloudmac.sh steward <mac> restart` and
`do_amd_steward.sh update` touch $MOJOLEARN_STEWARD_ROOT/drain; the steward
then claims nothing new and writes `drained` once its running work is done;
the tree moves and the steward restarts. The queued requests never leave
queue/, so `status` and coalescing keep seeing them.

COALESCING (one queued request per lane per steward): an identity `submit`
from a lane that already has QUEUED (not yet started) identity requests takes
them out of every queue (an atomic mv; one a steward claimed first runs as
submitted) and merges them into the new request: the union of verify lanes,
the new commit (an earlier request whose commit is not an ancestor of it is
left alone), this submit's one end-to-end sabotage patch, and the oldest
request's place in the FIFO. Only a request carrying the SAME sabotage patch
(byte for byte) is taken: another patch proves other lanes (a lane's store
lanes under e2e_store_branch never DISAGREE under e2e_host_branch), so a
request with a different patch stays queued and runs as submitted. `status` shows a merged request as COALESCED
(into the new name) and ends with the queue depth per steward and per lane.
--no-coalesce queues beside it instead.

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

import re
import signal
import threading
import uuid

REG = Path(os.environ.get("MOJOLEARN_CLOUDMAC_REG", Path.home() / "mojolearn-evidence" / "cloudmacs.tsv"))


def _fleet():
    """The Apple stewards, in file order: every row of cloudmacs.tsv (empty on
    a cloud Mac itself, which only ever works its own queue)."""
    if not REG.is_file():
        return ()
    return tuple(f[0] for f in (line.split() for line in REG.read_text().splitlines())
                 if f and not f[0].startswith("#"))


def _model(mac):
    """m4pro-a -> m4pro, m4-a -> m4, m3ultra-b -> m3ultra, m2pro -> m2pro"""
    return mac.split("-", 1)[0]


def _gen(mac):
    """M2, M3, M4: the chip family (a model name starts m<digit>)."""
    m = re.match(r"m(\d+)", _model(mac))
    return f"M{m.group(1)}" if m else _model(mac)


MACS = _fleet()
AMD_STEWARDS = ("do-amd",)
STEWARDS = MACS + AMD_STEWARDS
AMD_STATE = Path(os.environ.get("MOJOLEARN_STEWARD_DO_STATE",
                                Path.home() / "mojolearn-evidence" / "do-amd-steward")) / "state.env"
DEFERRED = tuple(x for x in os.environ.get("MOJOLEARN_STEWARD_DEFERRED", "").split(",") if x)
#: DRAINING Macs (e.g. before their hosts are released) get no new request;
#: `status` still reads them and `redistribute --apply` moves their queued
#: requests to the other Macs of the generation.
DRAIN = tuple(x for x in os.environ.get("MOJOLEARN_STEWARD_DRAIN", "").split(",") if x)
#: SPEED JOBS go to the least busy Mac of the requested model or generation;
#: the default model is the M4 Pro (Andrew, 2026-09-27).
SPEED_DEFAULT = os.environ.get("MOJOLEARN_STEWARD_SPEED", "m4pro")
REMOTE_ROOT = "~/mojolearn-evidence/apple-steward"
STEWARD_CLONE = "~/mojolearn"        # MOJOLEARN_STEWARD_REPO's default on the cloud Macs
ROOT = Path(os.environ.get("MOJOLEARN_STEWARD_ROOT",
                           Path.home() / "mojolearn-evidence" / "apple-steward"))
REPO = Path(os.environ.get("MOJOLEARN_STEWARD_REPO", Path.home() / "mojolearn"))
TOOLS = Path(__file__).resolve().parent
Q, WORK, DONE, PATCHES = ROOT / "queue", ROOT / "working", ROOT / "done", ROOT / "patches"
#: BUILD ONCE, RUN MANY (tools/steward_build.py): built bindings keyed by
#: their source-closure digest, and per-commit READY markers per lane.
STORE = ROOT / "builds"
#: DRAIN (a restart that never holds the queue): while this file exists the
#: steward claims nothing new; once nothing of its own runs it writes DRAINED.
#: `cloudmac.sh steward <mac> restart` and `do_amd_steward.sh update` touch
#: DRAIN_FILE, wait for DRAINED, move the tree and restart. The queue stays
#: where it is and `status` keeps showing it (an update that moved queued
#: requests aside made them vanish, and lanes resubmitted them as duplicates).
DRAIN_FILE, DRAINED = ROOT / "drain", ROOT / "drained"
SPOOL = ROOT / "deferred"            # on the laptop: requests for a deferred Mac


def _amd_live():
    """The AMD stewards the laptop can reach now (their state file exists)."""
    return tuple(s for s in AMD_STEWARDS if AMD_STATE.is_file())


_LOAD = ("setopt nullglob 2>/dev/null || true; cd {root} 2>/dev/null || {{ echo 0; exit 0; }}; "
         "ls queue working 2>/dev/null | grep -c '^[0-9].*[.]json$' || true")


def _parallel(fn, items):
    """{item: fn(item)} run on one thread per item (one ssh each); an item
    whose call fails maps to the exception."""
    out = {}

    def one(x):
        try:
            out[x] = fn(x)
        except BaseException as exc:      # _cloudmac raises SystemExit
            out[x] = exc
    ts = [threading.Thread(target=one, args=(x,)) for x in items]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def _loads(macs):
    """{mac: queued + working requests}, None for a Mac that did not answer."""
    got = _parallel(lambda m: int(_cloudmac(m, _LOAD.format(root=REMOTE_ROOT), timeout=60).strip() or 0),
                    [m for m in macs if m not in DEFERRED and m not in DRAIN])
    return {m: (v if isinstance(v, int) else None) for m, v in got.items()}


def _select(sel):
    """The Macs a selector names: a Mac name, a model (m4pro) or a generation (M4)."""
    if sel in MACS:
        return (sel,)
    macs = tuple(m for m in MACS if (_model(m) == sel or _gen(m) == sel.upper()) and m not in DRAIN)
    if not macs:
        sys.exit(f"no cloud Mac in {REG} matches {sel!r} (Macs: {', '.join(MACS) or 'none'})")
    return macs


def _least_busy(macs, loads):
    """(mac, reachable): the reachable Mac with the fewest requests (file order
    breaks ties), or the first one when none answered (its copy spools)."""
    live = [m for m in macs if loads.get(m) is not None]
    if live:
        return min(live, key=lambda m: (loads[m], macs.index(m))), True
    return macs[0], False


def _generations():
    """{generation: (macs...)}, in file order."""
    gens = {}
    for m in MACS:
        if m in DRAIN:
            continue
        gens.setdefault(_gen(m), []).append(m)
    return {g: tuple(v) for g, v in gens.items()}


def _route_identity():
    """One Mac per generation, the least busy of each: [(mac, reachable)]."""
    loads = _loads(MACS)
    picks = [_least_busy(macs, loads) for macs in _generations().values()]
    for g, macs in _generations().items():
        down = [m for m in macs if loads.get(m) is None]
        if down:
            print(f"{g}: not routed to {', '.join(down)} (deferred or not answering)")
    if DRAIN:
        print(f"draining, not routed to: {', '.join(DRAIN)}")
    return picks


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


def _cloudmac(name, command, stdin=None, timeout=120, raw=False):
    """One command on cloud Mac `name` through tools/cloudmac.sh (the AMD
    steward: plain ssh as root to the droplet in its state file)."""
    argv = _amd_ssh(command) if name in AMD_STEWARDS else [str(TOOLS / "cloudmac.sh"), "ssh", name, command]
    r = subprocess.run(argv, input=stdin, capture_output=True, timeout=timeout)
    if r.returncode:
        raise SystemExit(f"cloudmac {name}: `{command[:80]}` failed (exit {r.returncode}): "
                         f"{r.stderr.decode(errors='replace').strip()[:300]}")
    return r.stdout if raw else r.stdout.decode(errors="replace")


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


def _name_lane(name):
    """(kind, lane) from a request name <ms>-[speed-]<lane>-<sha10>"""
    body = name.split("-", 1)[1].rsplit("-", 1)[0] if name.count("-") >= 2 else ""
    return ("speed", body[len("speed-"):]) if body.startswith("speed-") else ("identity", body)


_QUEUED = (f"setopt nullglob 2>/dev/null || true; cd {REMOTE_ROOT} 2>/dev/null || exit 0; "
           "for f in queue/[0-9]*.json; do [ -f \"$f\" ] && echo \"$f\"; done; true")


def _queued_names(mac):
    return [Path(x).stem for x in _cloudmac(mac, _QUEUED, timeout=60).split() if x.startswith("queue/")]


def _is_ancestor(old, new):
    r = subprocess.run(["git", "-C", str(TOOLS.parent), "merge-base", "--is-ancestor", old, new],
                       capture_output=True)
    return r.returncode == 0


def _coalesce_take(lane, commit, patch_bytes):
    """COALESCE (one queued identity request per lane per steward): take
    every QUEUED (not yet started) identity request of `lane`, on every
    steward and in the laptop spool, out of its queue by an atomic mv (one the
    steward claimed first stays and runs). Only a request whose commit is an
    ancestor of (or equal to) `commit` AND whose sabotage patch is
    byte-identical to `patch_bytes` is taken; another is left queued (a
    different patch proves different lanes: merging it under this patch made
    every lane it proved read "sabotage not seen", prep 1790537100517).
    Returns (taken: {old name: [(mac, state)]}, their verify lanes, their names' timestamps)."""
    stewards = [m for m in MACS if m not in DEFERRED] + list(_amd_live())
    listed = _parallel(_queued_names, stewards)
    where = {}
    for m, names in listed.items():
        if not isinstance(names, list):
            print(f"{m}: did not answer ({str(names)[:160]}); its queued requests are not coalesced", file=sys.stderr)
            continue
        for n in names:
            if _name_lane(n) == ("identity", lane):
                where.setdefault(n, []).append((m, "queue"))
    for m, spooled in _spooled().items():
        for n in spooled:
            if _name_lane(n) == ("identity", lane):
                where.setdefault(n, []).append((m, "deferred (spooled)"))
    taken, lanes, stamps = {}, [], []
    for n, holders in sorted(where.items()):
        old_commit = None
        for m, st in holders:
            got = _withdraw(m, n, st)
            if got is None:
                print(f"{n}: {m} claimed it before the coalesce; it runs as submitted")
                continue
            body, old_patch = got
            req = json.loads(body)
            old_commit = req["commit"]
            if (old_patch or None) != (patch_bytes or None):
                _restore(m, n, st)
                print(f"{n}: its sabotage patch differs from this submit's; left queued on {m}")
                continue
            if not _is_ancestor(old_commit, commit):
                _restore(m, n, st)
                print(f"{n}: its commit {old_commit[:10]} is not an ancestor of {commit[:10]}; left queued on {m}")
                continue
            taken.setdefault(n, []).append((m, st))
            for x in req.get("verify_lanes") or []:
                if x not in lanes:
                    lanes.append(x)
        if n in taken:
            stamps.append(int(n.split("-", 1)[0]))
    return taken, lanes, stamps


def submit(a):
    if not all(c in "0123456789abcdef" for c in a.commit) or len(a.commit) < 7:
        sys.exit(f"--commit must be a hex sha pushed to origin, not {a.commit!r}")
    # GitHub cannot fetch an abbreviated object ID. Push and request must
    # name the same immutable, full commit on both speed and identity jobs.
    full = subprocess.run(["git", "-C", str(TOOLS.parent), "rev-parse", "--verify", f"{a.commit}^{{commit}}"],
                          capture_output=True, text=True)
    if full.returncode:
        sys.exit(f"{a.commit} is not a commit in {TOOLS.parent} (fetch the lane's branch first)")
    a.commit = full.stdout.strip()
    name = f"{int(time.time() * 1000)}-{'speed-' if a.kind == 'speed' else ''}{a.lane}-{a.commit[:10]}"
    taken = {}
    if a.kind == "identity" and not a.no_coalesce:
        patch_bytes = Path(a.sabotage).resolve().read_bytes() if a.sabotage and Path(a.sabotage).is_file() else None
        taken, old_lanes, stamps = _coalesce_take(a.lane, a.commit, patch_bytes)
        if taken:
            new_lanes = [x for x in (a.verify_lanes or "").split(",") if x]
            added = [x for x in old_lanes if x not in new_lanes]
            a.verify_lanes = ",".join(new_lanes + added)
            # the merged request keeps the oldest one's place in every FIFO
            name = f"{min(stamps) + 1}-{a.lane}-{a.commit[:10]}"
            print(f"COALESCED {', '.join(sorted(taken))} into {name}: verify lanes {a.verify_lanes}"
                  + (f" ({', '.join(added)} from the earlier request, which carried this same sabotage patch)"
                     if added else "") + f", commit {a.commit[:10]}, this submit's sabotage patch")
    try:
        _submit(a, name, taken)
    except BaseException:
        for n, holders in taken.items():      # nothing shipped: the earlier requests go back where they were
            for m, st in holders:
                if st != "deferred (spooled)":
                    _restore(m, n, st)
        if taken:
            print(f"the submit failed; {', '.join(taken)} restored to their queues", file=sys.stderr)
        raise
    for n, holders in taken.items():
        for m, st in holders:
            if st == "deferred (spooled)":
                _settle(m, n, st, None)
            else:
                _cloudmac(m, f"cd {REMOTE_ROOT} && mv moved/{n}.json moved/{n}.json.coalesced-into-{name}")


def _submit(a, name, taken):
    req = {"name": name, "kind": a.kind, "lane": a.lane, "commit": a.commit,
           "submitted": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if taken:
        req["coalesced"] = sorted(taken)
    patch = None
    if a.kind == "identity":
        if not (a.verify_lanes and a.sabotage):
            sys.exit("an identity request needs --verify-lanes and --sabotage")
        patch = Path(a.sabotage).resolve()
        if not patch.is_file():
            sys.exit(f"sabotage patch {patch} does not exist")
        req.update(verify_lanes=[x for x in a.verify_lanes.split(",") if x], **{"pass": a.pass_no},
                   sabotage=f"{REMOTE_ROOT}/patches/{name}.patch")
        picks = _route_identity() + [(m, True) for m in _amd_live()]
    else:
        if not a.cmd:
            sys.exit("a speed request needs --cmd '<timing command>'")
        builds = [x for x in (a.builds or "").split(",") if x]
        for b in builds:
            if not (b.startswith("bindings/") and b.endswith(".sh")) or ".." in b:
                sys.exit(f"--builds takes bindings/build_*.sh scripts, not {b!r}")
        req.update(builds=builds, cmd=a.cmd, mode=a.mode or "")
        amd = _amd_live()
        target = a.target or ("both" if amd else "apple")
        picks = []
        for t in target.split(","):
            if t in ("do-amd", "both") and not amd:
                sys.exit(f"--target {target}: the do-amd steward is not up (no {AMD_STATE}); "
                         f"start it with tools/do_amd_steward.sh up, or use --target apple")
            if t in ("do-amd", "both"):
                picks += [(m, True) for m in amd]
            if t != "do-amd":
                macs = _select(SPEED_DEFAULT if t in ("apple", "both") else t)
                picks.append(_least_busy(macs, _loads(macs)))
    if a.kind == "speed" and not a.no_coalesce:
        picks = _drop_duplicate_speed(picks, req)
        if not picks:
            return
    macs = [m for m, _ in picks]
    spool_to = {m for m, up in picks if not up or m in DEFERRED}
    req["stewards"] = list(macs)
    body = json.dumps(req, indent=2).encode()
    def spool_it(mac):
        spool = SPOOL / mac
        spool.mkdir(parents=True, exist_ok=True)
        if patch:
            (spool / f"{name}.patch").write_bytes(patch.read_bytes())
        (spool / f"{name}.json").write_bytes(body)
        print(f"{mac} is deferred or not answering: spooled {name} in {spool} (flush-deferred ships it later)")
    for mac in macs:
        if mac in spool_to:
            spool_it(mac)
            continue
        try:
            _push_commit(mac, a.commit)
            _ship(mac, name, body, patch.read_bytes() if patch else None)
        except SystemExit as exc:
            if mac in AMD_STEWARDS:
                raise
            print(f"{mac}: {exc}", file=sys.stderr)   # a Mac that fails mid-submit keeps its copy on the laptop
            spool_to.add(mac)
            spool_it(mac)
    print(f"queued {a.kind} request {name} on {', '.join(m for m in macs if m not in spool_to) or 'no Mac yet (spooled)'}")


_SAME_SPEED = ("setopt nullglob 2>/dev/null || true; cd {root} 2>/dev/null || exit 0; "
               "for f in queue/[0-9]*-{tag}.json queue/held/[0-9]*-{tag}.json working/[0-9]*-{tag}.*.json; "
               "do [ -f \"$f\" ] && {{ echo \"REQ $f\"; cat \"$f\"; echo; }}; done; true")


def _drop_duplicate_speed(picks, req):
    """A speed job identical to one still queued or running on that steward
    (same lane, commit, builds, command and mode) is not queued again: its
    verdict answers both. Returns the picks that still need a copy."""
    tag = f"speed-{req['lane']}-{req['commit'][:10]}"
    keep = []
    for m, up in picks:
        if not up or m in DEFERRED:
            keep.append((m, up))
            continue
        try:
            text = _cloudmac(m, _SAME_SPEED.format(root=REMOTE_ROOT, tag=tag), timeout=60)
        except SystemExit as exc:
            print(f"{m}: {exc}; not checked for a duplicate", file=sys.stderr)
            keep.append((m, up))
            continue
        dec, same = json.JSONDecoder(), None
        for chunk in text.split("REQ ")[1:]:
            path, _, rest = chunk.partition("\n")
            try:
                old, _ = dec.raw_decode(rest.lstrip())
            except ValueError:
                continue
            oc, nc = str(old.get("commit", "")), req["commit"]
            if (oc.startswith(nc) or nc.startswith(oc)) and \
                    all(old.get(k) == req.get(k) for k in ("lane", "builds", "cmd", "mode")):
                same = path.strip()
                break
        if same:
            where = "held" if same.startswith("queue/held/") else {"queue": "queued", "working": "running"}[same.split("/")[0]]
            print(f"{m}: the same speed job is already {where} there ({Path(same).name.split('.')[0]}); "
                  f"not queued again (--no-coalesce queues it anyway)")
        else:
            keep.append((m, up))
    return keep


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


REQ_KEYS = ("name", "kind", "lane", "commit", "submitted", "verify_lanes", "pass", "sabotage", "builds", "cmd",
            "mode", "stewards", "coalesced")


def _req_bytes(mac, name, state):
    """(request json, patch or None) of a copy on `mac` (or in its laptop spool)."""
    if state == "deferred (spooled)":
        pf = SPOOL / mac / f"{name}.patch"
        return (SPOOL / mac / f"{name}.json").read_bytes(), (pf.read_bytes() if pf.is_file() else None)
    # the steward may claim (queue -> working) or finish it meanwhile: try each place
    body = _cloudmac(mac, f"setopt nullglob 2>/dev/null || true; cd {REMOTE_ROOT} && "
                          f"for f in queue/{name}.json working/{name}.{mac}.json moved/{name}.json "
                          f"moved/{name}.json.to-*; do [ -f \"$f\" ] && exec cat \"$f\"; done; "
                          f"cat done/{name}/verdict.json", raw=True)
    if b'"result"' in body:      # it finished there meanwhile: the request is the verdict's request fields
        v = json.loads(body)
        body = json.dumps({k: v[k] for k in REQ_KEYS if k in v}, indent=2).encode()
    patch = _cloudmac(mac, f"cat {REMOTE_ROOT}/patches/{name}.patch 2>/dev/null || true", raw=True)
    return body, (patch or None)


def _withdraw(mac, name, state):
    """Take a queued copy out of `mac`'s queue by an atomic mv (the steward
    may claim it first: then it stays, and False comes back). Returns the
    request and patch bytes."""
    if state == "deferred (spooled)":
        return _req_bytes(mac, name, state)
    got = _cloudmac(mac, f"cd {REMOTE_ROOT} && mkdir -p moved && mv queue/{name}.json moved/{name}.json "
                         f"2>/dev/null && echo MOVED || echo CLAIMED").strip()
    if got != "MOVED":
        return None
    body = _cloudmac(mac, f"cat {REMOTE_ROOT}/moved/{name}.json", raw=True)
    patch = _cloudmac(mac, f"cat {REMOTE_ROOT}/patches/{name}.patch 2>/dev/null || true", raw=True)
    return body, (patch or None)


def _settle(mac, name, state, dest):
    """After the copy reached `dest` (or was dropped), record the withdrawal."""
    if state == "deferred (spooled)":
        (SPOOL / mac / f"{name}.patch").unlink(missing_ok=True)
        (SPOOL / mac / f"{name}.json").unlink()
    else:
        _cloudmac(mac, f"cd {REMOTE_ROOT} && mv moved/{name}.json moved/{name}.json.to-{dest}")


def _restore(mac, name, state):
    if state != "deferred (spooled)":
        _cloudmac(mac, f"cd {REMOTE_ROOT} && mv moved/{name}.json queue/{name}.json")


def redistribute(a):
    """Every PENDING request (queued or working on some Mac, or spooled on the
    laptop) ends with exactly one copy per generation for identity, and a
    speed job stays in its generation; queued copies are spread over the Macs
    of each generation, oldest first, onto the least loaded Mac (a speed job
    follows the lane's other speed jobs, so a before/after pair stays on one
    box). Without --apply it only prints the plan."""
    live = [m for m in MACS if m not in DEFERRED]
    per = _collect_all(live)
    down = [m for m in live if not per[m] and _loads([m]).get(m) is None]
    if down:
        sys.exit(f"{', '.join(down)} did not answer; nothing moved (a Mac that is down cannot be counted)")
    for mac, spooled in _spooled().items():
        for name, st in spooled.items():
            per.setdefault(mac, {})[name] = st
    merged = {n for m in per for n, st in per[m].items() if isinstance(st, str) and st.startswith("coalesced")}
    pending = sorted({n for m in per for n, st in per[m].items() if not isinstance(st, dict)} - merged)
    plan = []          # (name, gen, action, source mac, source state, dest)
    for g, gmacs in _generations().items():
        gmacs = [m for m in gmacs if m not in DEFERRED]
        if not gmacs:
            continue
        load = {m: sum(1 for st in per.get(m, {}).values() if st == "working") for m in gmacs}
        speed_home = {}
        for name in pending:
            holders = {m: per[m][name] for m in per if name in per[m] and m not in AMD_STEWARDS
                       and _gen(m) == g}
            fixed = [m for m, st in holders.items() if st == "working" or isinstance(st, dict)]
            movable = [(m, st) for m, st in holders.items() if st in ("queue", "deferred (spooled)")]
            if fixed:                                  # already worked in this generation
                plan += [(name, g, "drop", m, st, None) for m, st in movable]
                continue
            if not movable:
                if _is_speed(name):
                    continue                           # a speed job lives in one generation only
                srcs = [(m, st) for m in per for st in [per[m].get(name)]
                        if st is not None and not isinstance(st, dict) and m not in AMD_STEWARDS]
                src = sorted(srcs, key=lambda x: x[1] == "deferred (spooled)")[0]   # a Mac's copy first
                action = "copy"
            else:
                src = next(((m, st) for m, st in movable if st == "queue"), movable[0])
                plan += [(name, g, "drop", m, st, None) for m, st in movable if (m, st) != src]
                action = "move"
            lane = name.split("-")[2] if _is_speed(name) else None
            if lane in speed_home:
                dest = speed_home[lane]
            else:   # the least loaded Mac; its current holder wins a tie
                dest = min(gmacs, key=lambda m: (load[m], m != src[0] or action == "copy", gmacs.index(m)))
                if action == "move" and src[1] == "queue" and src[0] in load and load[src[0]] <= load[dest] + 1:
                    dest = src[0]      # one request of imbalance is not worth a move
            if lane:
                speed_home[lane] = dest
            load[dest] += 1
            if action == "copy" or dest != src[0] or src[1] != "queue":
                plan.append((name, g, action, src[0], src[1], dest))
    plan.sort(key=lambda x: x[2] != "copy")   # copies read their source before any move takes it
    for name, g, act, m, st, dest in plan:
        print(f"{g} {act:4} {name}: {m} ({st})" + (f" -> {dest}" if dest else ""))
    print(f"{len(pending)} pending requests; {len(plan)} actions" + ("" if a.apply else " (dry run; --apply does them)"))
    if not a.apply:
        return
    pushed = set()
    for name, g, act, m, st, dest in plan:
        if act == "drop":
            got = _withdraw(m, name, st)
            if got is None:
                print(f"{name}: {m} claimed it before the drop; left there")
                continue
            _settle(m, name, st, "dropped-duplicate")
            print(f"dropped duplicate {name} from {m}")
            continue
        got = _req_bytes(m, name, st) if act == "copy" else _withdraw(m, name, st)
        if got is None:
            print(f"{name}: {m} claimed it before the move; left there")
            continue
        body, patch = got
        try:
            commit = json.loads(body)["commit"]
            if (dest, commit) not in pushed:
                _push_commit(dest, commit)
                pushed.add((dest, commit))
            _ship(dest, name, body, patch)
        except BaseException:
            if act == "move":
                _restore(m, name, st)
            raise
        if act == "move":
            _settle(m, name, st, dest)
        print(f"{act} {name}: {m} -> {dest}")


# the cloud Macs' login shell is zsh, where an unmatched glob is an error
_REMOTE_LIST = (f"setopt nullglob 2>/dev/null || true; cd {REMOTE_ROOT} 2>/dev/null || exit 0; "
                "for f in queue/[0-9]*.json working/[0-9]*.json queue/held/[0-9]*.json; do [ -f \"$f\" ] && echo \"STATE $f\"; done; "
                "for f in moved/*.json.coalesced-into-*; do [ -f \"$f\" ] && echo \"COALESCED $f\"; done; "
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
            # queue/held/ is an older `do_amd_steward.sh update`'s hold: shown, never hidden
            out[Path(path).name.split(".")[0]] = "held" if path.startswith("queue/held/") else path.split("/")[0]
            pos = nl + 1 if nl >= 0 else len(text)
        elif line.startswith("COALESCED "):
            fname = Path(line.split(" ", 1)[1]).name
            out.setdefault(fname.split(".")[0], "coalesced into " + fname.split(".coalesced-into-", 1)[1])
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


def _collect_all(macs):
    """{mac: _collect(mac)} over ssh in parallel; a Mac that did not answer
    maps to {} and is named on stderr."""
    got = _parallel(_collect, [m for m in macs if m not in DEFERRED])
    for m, v in got.items():
        if not isinstance(v, dict):
            print(f"{m}: did not answer ({str(v)[:160]})", file=sys.stderr)
    return {m: (v if isinstance(v, dict) else {}) for m, v in got.items()}


def _spooled():
    """{mac: {name: 'deferred (spooled)'}} from the laptop spool."""
    return {d.name: {p.stem: "deferred (spooled)" for p in d.glob("[0-9]*.json")}
            for d in SPOOL.glob("*") if d.is_dir()}


def status(a):
    amd_live = _amd_live()
    per = _collect_all(MACS + amd_live)
    for mac, spooled in _spooled().items():
        for name, st in spooled.items():
            per.setdefault(mac, {}).setdefault(name, st)
    names = sorted(set().union(*[set(v) for v in per.values()]))
    rows = []
    for name in names:
        speed = _is_speed(name)
        # the stewards that hold a copy (queued, working, done or spooled)
        states = {m: per[m][name] for m in per if name in per[m]}
        verdicts = {m: s for m, s in states.items() if isinstance(s, dict)}
        passed = {m for m, v in verdicts.items() if v.get("result") == "PASS"}
        into = sorted({s.split(" ", 2)[2] for s in states.values() if isinstance(s, str) and s.startswith("coalesced")})
        states_live = {m: s for m, s in states.items() if not (isinstance(s, str) and s.startswith("coalesced"))}
        amd = [m for m in states_live if m in AMD_STEWARDS]
        apple = [m for m in states_live if m not in AMD_STEWARDS]
        if into:
            result = "COALESCED"     # its lanes run in the request it was merged into
        elif any(v.get("result") == "FAIL" for v in verdicts.values()):
            result = "FAIL"
        elif speed and states_live and set(states_live) <= passed:
            result = "PASS"
        elif not speed and any(m in passed for m in apple) and all(m in passed for m in amd):
            result = "PASS"          # the merge gate: one Apple PASS + do-amd PASS; the rest follow
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
        rows.append(dict(request=name, kind="speed" if speed else "identity", result=result, stewards=states,
                         **({"coalesced_into": into} if into else {})))
        if not a.json:
            print(f"{result:8} {name}  ({detail})")
    if a.json:
        print(json.dumps(rows, indent=1))
        return
    # QUEUE DEPTHS: per steward, and per lane (coalescing keeps one queued
    # identity request per lane per steward; a 2 here is a submit that raced)
    print("\nsteward     queued working  lanes queued (identity)")
    for m in sorted(per):
        q = [n for n, st in per[m].items() if st in ("queue", "held", "deferred (spooled)")]
        w = sum(1 for st in per[m].values() if st == "working")
        by = {}
        for n in q:
            k, lane = _name_lane(n)
            if k == "identity":
                by[lane] = by.get(lane, 0) + 1
        lanes = ", ".join(f"{k}{'' if v == 1 else ' x' + str(v)}" for k, v in sorted(by.items()))
        print(f"{m:11} {len(q):6} {w:7}  {lanes}")


# --------------------------------------------------------------- a cloud Mac
#: Every job process carries this variable (a tag unique to the job). A job's
#: processes are its process group plus every process whose environment holds
#: the tag, so a driver that left the group (setsid) or was reparented to init
#: when its parent died is still found. 2026-09-29: a timed-out lowbit speed
#: job on do-amd left its grandchildren spinning on the GPU for 7 h 46 min,
#: because subprocess.run's timeout killed only the `sh -c` child.
JOB_TAG_VAR = "MOJOLEARN_STEWARD_JOB"


def _job_pids(tag, pgid):
    """The live (not zombie) processes of the job tagged `tag` whose group is `pgid`."""
    pids = set()
    ps = subprocess.run(["ps", "-A", "-o", "pid=,pgid=,stat="], capture_output=True, text=True).stdout
    for line in ps.splitlines():
        f = line.split()
        if len(f) >= 3 and f[1] == str(pgid) and not f[2].startswith("Z"):
            pids.add(int(f[0]))
    proc = Path("/proc")
    if proc.is_dir():                     # Linux (do-amd): the environment tag
        needle = f"{JOB_TAG_VAR}={tag}".encode()
        for d in proc.iterdir():
            if not d.name.isdigit():
                continue
            try:
                if needle in (d / "environ").read_bytes().split(b"\0"):
                    pids.add(int(d.name))
            except OSError:
                continue
    elif sys.platform == "darwin":        # a Mac: `ps -E` shows the environment of our own processes
        ps = subprocess.run(["ps", "-A", "-E", "-ww", "-o", "pid=,stat=,command="],
                            capture_output=True, text=True).stdout
        needle = f"{JOB_TAG_VAR}={tag}"
        for line in ps.splitlines():
            f = line.split(None, 2)
            if len(f) == 3 and not f[1].startswith("Z") and needle in f[2].split():
                pids.add(int(f[0]))
    pids.discard(os.getpid())
    return pids


def _kill_job(tag, pgid, proc=None):
    """TERM, then KILL, every process of the job; returns the pids that were
    alive when it started and the ones that survived (empty unless the kernel
    refused)."""
    found = sorted(_job_pids(tag, pgid))
    for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 10)):
        pids = _job_pids(tag, pgid)
        if not pids:
            break
        for pid in pids:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        end = time.time() + grace
        while time.time() < end:
            if proc is not None:
                proc.poll()               # reap the direct child so it is not counted as alive
            if not _job_pids(tag, pgid):
                break
            time.sleep(0.2)
    if proc is not None:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    return found, sorted(_job_pids(tag, pgid))


def _run_job(cmd, timeout, cwd=None, env=None, stdout=None, stderr=None):
    """Run a job command in its own session with a job tag. On a timeout, and
    after any exit, every process of the job still alive is killed (a speed job
    is exclusive: nothing of it may keep the GPU). Returns (rc, record): rc 124
    on a timeout; record names what was killed ({} when nothing was left)."""
    tag = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
    env = dict(os.environ if env is None else env)
    env[JOB_TAG_VAR] = tag
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    try:
        rc = p.wait(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        rc, timed_out = 124, True
    found, survivors = _kill_job(tag, p.pid, p)
    record = {}
    if found or timed_out:
        record = {"timed_out": timed_out, "killed": found, "survivors": survivors}
    return rc, record


def _run(cmd, cwd, log, timeout, env=None):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        rc, record = _run_job(cmd, timeout, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
        if record.get("timed_out"):
            f.write(f"TIMEOUT after {timeout}s\n")
        if record:
            f.write(f"steward killed the job's processes: {json.dumps(record)}\n")
        return rc


def _draining(idle):
    """True while DRAIN_FILE exists; DRAINED marks it once `idle`."""
    if not DRAIN_FILE.exists():
        DRAINED.unlink(missing_ok=True)
        return False
    if idle and not DRAINED.exists():
        DRAINED.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        print("drained: nothing running, nothing claimed until the drain file goes", flush=True)
    return True


def _claim(steward, skip_speed=False):
    for p in sorted(Q.glob("[0-9]*.json")):
        if skip_speed and _is_speed(p.stem):
            continue
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


def _fetch(steward, commit, log):
    """the laptop pushes a submitted sha as refs/steward/<sha> (cloudmac.sh push);
    the default refspec fetches only branches, so name that namespace too"""
    if steward in AMD_STEWARDS:   # a shallow tree: the sha itself, from GitHub
        for _ in range(40):   # a laptop `submit` may hold shallow.lock for a moment
            if not _run(["git", "fetch", "-q", "--depth=1", "origin", commit], REPO, log, 900):
                break
            time.sleep(3)
    else:
        _run(["git", "fetch", "-q", "origin", "+refs/heads/*:refs/remotes/origin/*", "+refs/steward/*:refs/steward/*"],
             REPO, log, 600)


def _store_helper(sub, wt, log, *extra, timeout=4 * 3600):
    """tools/steward_build.py (THIS steward's copy) against worktree `wt`, in
    the tree's pixi default environment; returns the exit status."""
    return _run([_pixi(), "run", "-e", "default", "python", "-u", str(TOOLS / "steward_build.py"), sub,
                 "--root", str(wt), "--store", str(STORE), *extra], wt, log, timeout)


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

    _fetch(steward, req["commit"], log)
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
    # BUILD ONCE, RUN MANY: the bindings this tree's sources already built on
    # this box (do-amd's prebuild, or an earlier request) are copied in, so the
    # check finds them fresh; a lane check that knows the store also takes the
    # restored stage's bindings from it and publishes what it builds clean.
    t0 = time.time()
    seeded = _store_helper("seed", wt, log, timeout=1800)
    verdict["store"] = {"seed_rc": seeded, "seed_s": round(time.time() - t0, 1),
                        "seeded": _last_line(log, "SEEDED ")}
    env = dict(os.environ, MOJOLEARN_LANE_CHECK_STORE=str(STORE))
    t0 = time.time()
    rc = _run(cmd, wt, log, 6 * 3600, env=env)
    verdict["check_wall_s"] = round(time.time() - t0, 1)
    clean = not subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=wt,
                               capture_output=True, text=True).stdout.strip()
    if clean:     # an older lane check does not publish itself: what it built clean goes to the store
        _store_helper("publish", wt, log, timeout=1800)
    last = [line for line in log.read_text(errors="replace").splitlines() if line.startswith("RESULT:")]
    verdict["check"] = last[-1] if last else "no RESULT line"
    if rc == 0 and last and last[-1].startswith("RESULT: PASS"):
        return finish("PASS")
    return finish("FAIL", last[-1][len("RESULT: "):] if last else f"the lane check exited {rc}")


def _last_line(log, prefix):
    lines = [x for x in Path(log).read_text(errors="replace").splitlines() if x.startswith(prefix)]
    return lines[-1][len(prefix):] if lines else None


#: A process whose command line holds one of these is another Metal (or
#: heavy CPU) job; a speed job never times beside one.
FOREIGN = ("lm_segment", "algos_lane_check", "steward_build.py", "identity_break.py", "mac_slot.sh",
           "apple_steward.py work",
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
    # caffeinate only holds a power assertion: what it wraps is its own
    # process and is judged by its own command line. (launchd starts this
    # steward as `caffeinate -i python3 ... work`; caffeinate execs python in
    # its own pid and leaves a CHILD caffeinate whose command line names
    # "apple_steward.py work": the steward took that for a foreign job and
    # requeued every speed job forever.)
    return [(pid, c[:160]) for pid, (_, c) in sorted(procs.items())
            if pid not in mine and any(f in c for f in FOREIGN) and not c.startswith("ps ")
            and Path(c.split(None, 1)[0]).name != "caffeinate"]


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
    """A SPEED job (a cloud Mac or do-amd): the builds, then the timing
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
            rc, killed = _run_job([_pixi(), "run", "-e", "default", "sh", script], 3 * 3600, cwd=wt, env=benv,
                                  stdout=f, stderr=subprocess.STDOUT)
        timing["builds"].append({"script": script, "rc": rc, "wall_s": round(time.time() - t0, 3),
                                 **({"killed": killed} if killed else {})})
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
        rc, killed = _run_job(["sh", "-c", req["cmd"]], 3 * 3600, cwd=wt, env=env, stdout=fo, stderr=fe)
        timing["cmd_wall_s"] = round(time.time() - t0, 3)
    timing["cmd_rc"] = rc
    if killed:
        timing["killed"] = killed
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


def _build_state(commit, lanes):
    """READY when every lane's bindings for this commit are in the store (or
    the prebuild could not build them: NOBUILD, and the run builds and reports
    itself), else None."""
    c = STORE / "commits" / commit
    if all((c / "lanes" / x).is_file() or (c / "nobuild" / x).is_file() for x in lanes):
        return "READY"
    return None


def _build_worktree(steward):
    return Path.home() / "mojolearn-wt" / f"steward-{steward}-build"


def _prebuild(steward, commit, lanes):
    """THE ONE BUILD for (commit, lanes) on this box, in its own worktree that
    no sabotage ever touches; marks each lane READY or NOBUILD in the store.
    Any failure here marks the lanes NOBUILD: their runs then build as before
    and report the real error, so a broken prebuild never decides a verdict."""
    cdir = STORE / "commits" / commit
    cdir.mkdir(parents=True, exist_ok=True)
    log = cdir / "steward-prebuild.log"
    wt = _build_worktree(steward)
    t0 = time.time()
    try:
        _fetch(steward, commit, log)
        if wt.exists():
            rc = _run(["git", "checkout", "-q", "--detach", "-f", commit], wt, log, 300)
        else:
            rc = _run(["git", "worktree", "add", "-q", "--detach", str(wt), commit], REPO, log, 900)
        if not rc:
            rc = _store_helper("build", wt, log, "--commit", commit, "--lanes", ",".join(sorted(lanes)))
    except Exception as exc:          # noqa: BLE001 - recorded, the runs build themselves
        rc = f"steward error {exc!r}"
    for lane in lanes:
        if not ((cdir / "lanes" / lane).is_file() or (cdir / "nobuild" / lane).is_file()):
            (cdir / "nobuild").mkdir(exist_ok=True)
            (cdir / "nobuild" / lane).write_text(f"the prebuild ended ({rc}) without building it; see {log}\n")
    print(f"prebuild {commit[:10]} {','.join(sorted(lanes))}: done in {time.time() - t0:.0f}s (rc {rc})", flush=True)


def _claim_one(p, steward):
    dst = WORK / f"{p.stem}.{steward}.json"
    try:
        p.rename(dst)
        return dst
    except FileNotFoundError:
        return None


def _work_parallel(a):
    """do-amd: BUILD ONCE, RUN MANY. One builder thread builds each distinct
    (commit, lanes) once into the store (tools/steward_build.py); up to
    AMD_PARALLEL identity slots run requests whose builds are done, oldest
    first, so a build never holds a slot or the FIFO: a request whose build
    is running waits while later requests with finished builds run. Requests
    at one commit share one build. A speed job at the head of the queue waits
    for every slot and the builder to finish and runs alone."""
    import threading
    running = {}                         # slot -> thread
    builder = {"t": None, "key": None}

    def reap():
        for n in [n for n, t in running.items() if not t.is_alive()]:
            del running[n]
        if builder["t"] is not None and not builder["t"].is_alive():
            builder["t"], builder["key"] = None, None

    def one(req, n):
        try:
            process(req, a.steward, n)
        except Exception as exc:        # a crash must not strand the request in working/
            print(f"{req.name}: steward error {exc!r}", flush=True)
            if req.exists():
                req.rename(Q / f"{req.name.split('.')[0]}.json")

    last_prune = 0.0
    while True:
        reap()
        if _draining(idle=not running and builder["t"] is None):
            time.sleep(5)
            continue
        if time.time() - last_prune > 3600 and builder["t"] is None:
            subprocess.run([sys.executable, str(TOOLS / "steward_build.py"), "prune", "--store", str(STORE)])
            last_prune = time.time()
        if _head_is_speed():
            if running or builder["t"] is not None:   # nothing new starts; the speed job waits for the box
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
        want = {}                        # commit -> lanes still to build, oldest request first
        started = False
        for p in sorted(Q.glob("[0-9]*.json")):
            if _is_speed(p.stem):
                break                    # FIFO: nothing passes a speed job
            try:
                req = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            lanes = req.get("verify_lanes") or []
            if _build_state(req["commit"], lanes) == "READY":
                if not free:
                    continue
                got = _claim_one(p, a.steward)
                if got:
                    n = free.pop(0)
                    t = threading.Thread(target=one, args=(got, n), daemon=True)
                    running[n] = t
                    t.start()
                    started = True
                    print(f"{got.stem}: started in slot {n} ({len(running)} running; build shared from the store)",
                          flush=True)
                continue
            key = builder["key"]
            if key and key[0] == req["commit"] and set(lanes) <= key[1]:
                continue                 # its build is running now
            want.setdefault(req["commit"], set()).update(lanes)
        if builder["t"] is None and want:
            commit, lanes = next(iter(want.items()))
            builder["key"] = (commit, set(lanes))
            builder["t"] = threading.Thread(target=_prebuild, args=(a.steward, commit, sorted(lanes)), daemon=True)
            builder["t"].start()
            print(f"prebuild {commit[:10]} {','.join(sorted(lanes))}: started ({len(running)} slots running)",
                  flush=True)
            started = True
        if started:
            continue
        if a.once and not running and builder["t"] is None and not any(Q.glob("[0-9]*.json")):
            return
        time.sleep(5 if (running or builder["t"] is not None) else 30)


def work(a):
    _dirs()
    if not (REPO / ".git").exists():
        sys.exit(f"no clone at {REPO} (set MOJOLEARN_STEWARD_REPO)")
    if a.steward in AMD_STEWARDS and AMD_PARALLEL > 1:
        return _work_parallel(a)
    last_prune = 0.0
    speed_wait = 0.0      # until then a busy Mac works the identity requests behind a speed job
    while True:
        if _draining(idle=True):     # one request at a time: between requests nothing runs
            time.sleep(5)
            continue
        if time.time() - last_prune > 3600:
            subprocess.run([sys.executable, str(TOOLS / "steward_build.py"), "prune", "--store", str(STORE)])
            last_prune = time.time()
        req = _claim(a.steward, skip_speed=time.time() < speed_wait)
        if req:
            try:
                process(req, a.steward)
            except Busy as exc:
                # a speed job waits for a quiet Mac: back to the queue, untouched,
                # and the identity requests behind it run meanwhile (one at a time)
                req.rename(Q / f"{req.name.split('.')[0]}.json")
                print(f"{req.name}: Metal queue not quiet ({exc.args[0][:2]}); requeued", flush=True)
                if a.once:
                    return
                speed_wait = time.time() + 60
                if not any(not _is_speed(p.stem) for p in Q.glob("[0-9]*.json")):
                    time.sleep(60)
        elif a.once:
            return
        else:
            time.sleep(30)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit", help="laptop: queue a request on one cloud Mac per generation (speed: one Mac) "
                                       "and, for identity, the do-amd steward while it is up")
    s.add_argument("--lane", required=True, help="the expansion lane (linear, ..., ann)")
    s.add_argument("--commit", required=True, help="a commit pushed to origin")
    s.add_argument("--kind", choices=("identity", "speed"), default="identity",
                   help="identity (one Mac per generation + do-amd; one Apple PASS + do-amd PASS gates) "
                        "or speed (--target)")
    s.add_argument("--verify-lanes", help="identity: comma separated identity lanes")
    s.add_argument("--sabotage", help="identity: a SOURCE patch that must make the check DISAGREE")
    s.add_argument("--no-coalesce", action="store_true",
                   help="identity: queue beside this lane's queued request instead of merging into it; "
                        "speed: queue even when the same speed job is already queued or running there")
    s.add_argument("--pass", dest="pass_no", type=int, choices=(1, 2), default=2,
                   help="identity: the lane check's --pass (default 2: the steward is a pass-2 step, so every "
                        "fragment needs its .checks with a sabotage patch per driver)")
    s.add_argument("--builds", help="speed: comma separated bindings/build_*.sh, run before the timing")
    s.add_argument("--cmd", help="speed: the timing command, run in the worktree at the commit (sh -c)")
    s.add_argument("--mode", choices=("identical", "fast"), help="speed: MOJOLEARN_NUMERIC_MODE for builds and cmd")
    s.add_argument("--target",
                   help="speed, comma separated: a Mac name (pins that box), a model (m4pro, m4, m3ultra, m2pro) "
                        "or generation (M2, M3, M4) whose least busy Mac times it, apple (= m4pro), do-amd, or "
                        "both (= m4pro + do-amd; the default while do-amd is up, else m4pro)")
    s.set_defaults(fn=submit)
    st = sub.add_parser("status", help="laptop: verdicts; identity PASS = one Apple PASS + do-amd PASS")
    st.add_argument("--json", action="store_true")
    st.set_defaults(fn=status)
    f = sub.add_parser("flush-deferred", help="laptop: ship a formerly deferred Mac's spooled requests")
    f.add_argument("--steward", required=True, choices=MACS)
    f.set_defaults(fn=flush_deferred)
    r = sub.add_parser("redistribute", help="laptop: spread the pending backlog, one copy per generation")
    r.add_argument("--apply", action="store_true", help="do it (default: print the plan)")
    r.set_defaults(fn=redistribute)
    w = sub.add_parser("work", help="cloud Mac: work this Mac's queue")
    w.add_argument("--steward", required=True, help="this Mac's name in cloudmacs.tsv, or do-amd")
    w.add_argument("--once", action="store_true", help="drain the queue, then exit")
    w.set_defaults(fn=work)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
