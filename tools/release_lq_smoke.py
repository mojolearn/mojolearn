#!/usr/bin/env python3
"""tools/release_lq_smoke.py -- one release GPU column on a box we already hold,
through the lane queue (2026-10-08, Andrew: "let's deprecate the
per-architecture rented smoke"). Releases rent nothing.

    python3 tools/release_lq_smoke.py --box nv|amd --tag rel-0.8.37-nvidia \\
        --remote-dir /root/release-smoke/0.8.37/nvidia --out <local out> [--state FILE] \\
        -- <core wheel> --expected-source-commit <sha> --vendor cuda --column <selection> \\
           --ref-column <ref> ... --plugin <wheel> --plugin <wheel>

tools/release.py --smoke-via lq launches it detached, as it launched the rented
smoke, and judges the out directory exactly as before (results.json,
column-<vendor>.json, diff-ref-<vendor>.txt). In order:

1. every local file the smoke names (the core wheel, each --plugin, --column,
   each --ref-column) is copied to <remote-dir>/in/ on the box (ssh, sha256
   checked there); the box's earlier out and run directories of this column
   are removed;
2. `lq add --front <box> CMD <branch> <tag> bash tools/release_wheel_smoke.sh
   <box paths> --local --out <remote-dir>/out BUILDS=none`: next in the box's
   queue, after the running job and ahead of the grid lines; the box tree of
   <branch> runs the smoke ON the box (no build, no pixi env);
3. `lq results <box> <tag>` is polled until the job's `CMD <tag> rc=` line;
4. <remote-dir>/out comes home into --out (tar over ssh).

The exit code is the box job's rc (the smoke's own verdict), 1 when no
results.json came home, 3 when the job never reported within --max-wait-hours.
--state keeps the queued tag: a relaunch while the job is queued or running
polls it again instead of queueing a second one; it is removed once the result
is home. The ssh targets are lq's own (NV= and AMD= in the lq script).
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

LQ_DEFAULT = Path(os.environ.get("MOJOLEARN_LQ", str(Path.home() / "mojolearn-evidence" / "lq" / "lq")))
SSH_OPTS = ["-o", "ConnectTimeout=20", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
#: smoke options whose value is a local file the box needs
FILE_OPTS = ("--plugin", "--column", "--ref-column", "--cpu-column")
#: smoke options release.py's rented route passed that mean nothing on a held box
DROP_FLAGS = ("--rent",)
DROP_OPTS = ("--gpu", "--provider", "--out", "--ssh")
BOXES = {"nv": "NV", "nv2": "NV2", "amd": "AMD"}  # nv2: the second RunPod L40S (lq NV2=)


def say(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')} lq-smoke] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lq_target(lq, box):
    """The ssh target lq itself uses for BOX (its NV=/AMD= line), as argv."""
    var = BOXES[box]
    m = re.search(rf'^\s*{var}="([^"]+)"|[;\s]{var}="([^"]+)"', Path(lq).read_text(), re.M)
    if not m:
        raise SystemExit(f"lq-smoke: no {var}=\"...\" ssh target in {lq}")
    return shlex.split(m.group(1) or m.group(2))


def box_plan(smoke_args, remote_dir):
    """(uploads [(local, remote)], box argv) for the smoke's arguments: each
    local file renamed into <remote_dir>/in (reference columns all share the
    name column.json, so each gets its index), the rented-route options
    dropped, --local and the box out directory added."""
    rin = f"{remote_dir}/in"
    uploads, out, i, refs = [], [], 0, 0
    wheel_seen = False
    while i < len(smoke_args):
        a = smoke_args[i]
        if a in DROP_FLAGS:
            i += 1
            continue
        if a in DROP_OPTS:
            i += 2
            continue
        if a in FILE_OPTS:
            local = Path(smoke_args[i + 1])
            if a in ("--ref-column", "--cpu-column"):
                name = f"ref-{refs}-{local.parent.name or 'column'}-{local.name}"
                refs += 1
            else:
                name = local.name
            uploads.append((local, f"{rin}/{name}"))
            out += [a, f"{rin}/{name}"]
            i += 2
            continue
        if a.startswith("-"):
            out.append(a)
            if a == "--expected-source-commit" or a == "--vendor" or a == "--smoke-seconds" or a == "--lease":
                out.append(smoke_args[i + 1])
                i += 2
                continue
            i += 1
            continue
        if wheel_seen:
            raise SystemExit(f"lq-smoke: a second positional argument {a!r}")
        wheel_seen = True
        local = Path(a)
        uploads.append((local, f"{rin}/{local.name}"))
        out.append(f"{rin}/{local.name}")
        i += 1
    if not wheel_seen:
        raise SystemExit("lq-smoke: no wheel among the smoke arguments")
    names = [r for _, r in uploads]
    if len(set(names)) != len(names):
        raise SystemExit(f"lq-smoke: two files share a box name: {names}")
    return uploads, ["bash", "tools/release_wheel_smoke.sh", *out, "--local", "--out", f"{remote_dir}/out"]


def lq_line(lq, box, branch, tag, box_argv, timeout_s):
    """The lq command that queues the smoke next on BOX."""
    return [str(lq), "add", "--front", box, "CMD", branch, tag, *box_argv, "BUILDS=none",
            f"LQ_CMD_TIMEOUT={int(timeout_s)}"]


def result_line(text, tag):
    """(rc, line) of the job's `CMD <tag> rc=N` result line in `lq results` output, or None."""
    for line in text.splitlines():
        m = re.search(rf"\bCMD {re.escape(tag)} rc=(\d+)", line)
        if m:
            return int(m.group(1)), line
    return None


class Runner:
    def __init__(self, args, run=subprocess.run, sleep=time.sleep):
        self.a, self.run, self.sleep = args, run, sleep
        self.lq = Path(args.lq)
        self.target = lq_target(self.lq, args.box) if not args.ssh else shlex.split(args.ssh)

    def ssh(self, command, stdin=None, timeout=600, binary=False):
        return self.run(["ssh", *SSH_OPTS, *self.target, command], stdin=stdin, timeout=timeout,
                        capture_output=True, text=not binary and stdin is None)

    def upload(self, uploads, remote_dir):
        rdir = shlex.quote(remote_dir)
        got = self.ssh(f"set -e; rm -rf {rdir}/in {rdir}/out {rdir}/out.run; mkdir -p {rdir}/in; echo READY")
        if got.returncode != 0 or "READY" not in (got.stdout or ""):
            raise SystemExit(f"lq-smoke: could not prepare {remote_dir} on {self.a.box}: {(got.stderr or '')[-400:]}")
        want = {}
        for local, remote in uploads:
            if not local.is_file():
                raise SystemExit(f"lq-smoke: no file {local}")
            size = local.stat().st_size
            say(f"copying {local.name} ({size} bytes) to {self.a.box}:{remote}")
            with open(local, "rb") as fh:
                got = self.ssh(f"cat > {shlex.quote(remote)}", stdin=fh, timeout=120 + size // 200000)
            if got.returncode != 0:
                raise SystemExit(f"lq-smoke: copy of {local} failed (rc {got.returncode})")
            want[remote] = sha256(local)
        got = self.ssh("sha256sum " + " ".join(shlex.quote(r) for r in want))
        have = {}
        for line in (got.stdout or "").splitlines():
            parts = line.split()
            if len(parts) == 2:
                have[parts[1]] = parts[0]
        bad = [r for r, s in want.items() if have.get(r) != s]
        if bad:
            raise SystemExit(f"lq-smoke: sha256 differs on the box for {bad}")
        say(f"{len(want)} file(s) on the box, sha256 verified")

    def queue(self, box_argv, tag):
        cmd = lq_line(self.lq, self.a.box, self.a.branch, tag, box_argv, self.a.max_job_hours * 3600)
        say("$ " + " ".join(shlex.quote(c) for c in cmd))
        got = self.run(cmd, capture_output=True, text=True, timeout=300)
        say(f"lq: rc={got.returncode} {(got.stdout or '').strip()[-300:]} {(got.stderr or '').strip()[-300:]}")
        if got.returncode != 0:
            raise SystemExit(f"lq-smoke: lq add refused (rc {got.returncode})")

    def wait(self, tag):
        deadline = time.time() + self.a.max_wait_hours * 3600
        polls = 0
        while time.time() < deadline:
            got = self.run([str(self.lq), "results", self.a.box, tag], capture_output=True, text=True, timeout=300)
            hit = result_line(got.stdout or "", tag)
            if hit:
                say("result: " + hit[1][:400])
                return hit[0]
            polls += 1
            if polls % 10 == 1:
                say(f"waiting for {tag} on {self.a.box} (lq results rc={got.returncode})")
            self.sleep(self.a.poll_seconds)
        return None

    def fetch(self, remote_dir, out):
        out.mkdir(parents=True, exist_ok=True)
        tar = self.ssh(f"cd {shlex.quote(remote_dir)}/out && tar czf - .", timeout=1800, binary=True)
        if tar.returncode != 0:
            say(f"fetch: tar on the box exited {tar.returncode}")
            return False
        got = self.run(["tar", "xzf", "-", "-C", str(out)], input=tar.stdout, capture_output=True)
        if got.returncode != 0:
            say(f"fetch: local untar exited {got.returncode}")
            return False
        say(f"fetched {remote_dir}/out into {out}")
        return True

    def main(self):
        a = self.a
        uploads, box_argv = box_plan(a.smoke, a.remote_dir)
        key = hashlib.sha256(json.dumps([a.box, a.branch, a.remote_dir, box_argv,
                                         [sha256(l) if l.is_file() else str(l) for l, _ in uploads]]).encode()).hexdigest()
        state = Path(a.state) if a.state else None
        st = {}
        if state and state.is_file():
            try:
                st = json.loads(state.read_text())
            except ValueError:
                st = {}
        if st.get("key") == key and st.get("tag"):
            tag = st["tag"]
            say(f"resuming {tag} on {a.box} (queued {st.get('queued_at')}); not queued again")
        else:
            tag = f"{a.tag}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            self.upload(uploads, a.remote_dir)
            self.queue(box_argv, tag)
            if state:
                state.parent.mkdir(parents=True, exist_ok=True)
                state.write_text(json.dumps(dict(key=key, tag=tag, box=a.box, queued_at=dt.datetime.now(
                    dt.timezone.utc).isoformat(timespec="seconds"), box_argv=box_argv), indent=2) + "\n")
        rc = self.wait(tag)
        if rc is None:
            say(f"no result for {tag} within {a.max_wait_hours} h; the job stays queued (rerun resumes it)")
            return 3
        fetched = self.fetch(a.remote_dir, Path(a.out))
        if state and state.is_file():
            state.unlink()
        if not fetched:
            return rc or 1
        if rc == 0 and not (Path(a.out) / "results.json").is_file():
            say("the job said rc=0 but no results.json came home")
            return 1
        return rc


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--box", required=True, choices=sorted(BOXES))
    ap.add_argument("--tag", required=True, help="tag stem; a UTC stamp is appended")
    ap.add_argument("--remote-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--branch", default="main", help="the box tree the smoke script runs from (default main)")
    ap.add_argument("--state", default="")
    ap.add_argument("--lq", default=str(LQ_DEFAULT))
    ap.add_argument("--ssh", default="", help="ssh target override (default: lq's own for --box)")
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--max-wait-hours", type=float, default=14.0)
    ap.add_argument("--max-job-hours", type=float, default=2.0)
    ap.add_argument("smoke", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.smoke[:1] == ["--"]:
        a.smoke = a.smoke[1:]
    if not re.fullmatch(r"/[A-Za-z0-9/_.-]+", a.remote_dir):
        ap.error("--remote-dir must be an absolute path of [A-Za-z0-9/_.-]")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", a.tag):
        ap.error("--tag must be [A-Za-z0-9_.-]")
    return a


if __name__ == "__main__":
    sys.exit(Runner(parse(sys.argv[1:])).main())
