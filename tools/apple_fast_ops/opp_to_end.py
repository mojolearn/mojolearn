#!/usr/bin/env python3
"""M3: keep opponent-only jobs (tag opp*) behind every other job.
- pending opp* lines move behind all other pending lines (pass a JSON list to append new lines first);
- if the RUNNING job is opp* while non-opp jobs are pending, stop it and requeue it at the end under a new tag,
  dropping races that already have an `OPP ... status=ok` line (no opponent reruns)."""
import json, os, re, shutil, signal, subprocess, sys, time
from pathlib import Path
q = Path.home() / "mq"; queue = q / "queue.txt"
is_opp = lambda l: len(l.split()) > 2 and l.split()[2].startswith("opp")
def write(new, orig, pos, why):
    assert queue.read_text() == orig and int((q / "pos").read_text()) == pos, "queue changed: retry"
    shutil.copy2(queue, q / "queue.before-opp-to-end.txt")
    tmp = q / "queue.opp.tmp"; tmp.write_text("\n".join(new) + "\n"); os.replace(tmp, queue); print("OPP_TO_END", why)
orig = queue.read_text(); pos = int((q / "pos").read_text()); lines = orig.splitlines()
extra = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else []
known = {tuple(l.split()[1:3]) for l in lines if l.startswith("CMD ")}
extra = [l for l in extra if tuple(l.split()[1:3]) not in known]
pend = lines[pos:] + extra
new = lines[:pos] + [l for l in pend if not is_opp(l)] + [l for l in pend if is_opp(l)]
if new != lines:
    write(new, orig, pos, "reordered/added; opp pending at end: %d" % sum(map(is_opp, pend)))
orig = queue.read_text(); lines = orig.splitlines(); pos = int((q / "pos").read_text())
cur = lines[pos - 1] if pos >= 1 else ""
if is_opp(cur) and any(not is_opp(l) for l in lines[pos:]):
    tag = cur.split()[2]
    done = set()
    log = q / "out" / (tag + ".log")
    if log.exists():
        for ln in log.read_text(errors="replace").splitlines():
            m = re.match(r"OPP family=\S+ lane=(\S+) ds=(\S+) .*status=ok", ln)
            if m: done.add("%s:%s" % m.groups())
    m = re.search(r"--races (\S+)", cur)
    left = [r for r in (m.group(1).split(",") if m else []) if r not in done]
    pids = subprocess.run(["pgrep", "-f", "opp_only_board.py .*--tag %s( |$)" % tag], capture_output=True, text=True).stdout.split()
    for p in pids:
        for c in subprocess.run(["pgrep", "-P", p], capture_output=True, text=True).stdout.split():
            os.kill(int(c), signal.SIGTERM)
        os.kill(int(p), signal.SIGTERM)
    time.sleep(10)
    if left:
        n = re.sub(r"-r\d+$", "", tag) + "-r%d" % int(time.time() % 100000)
        line = cur.replace(" %s " % tag, " %s " % n, 1).replace("--tag %s" % tag, "--tag %s" % n)
        if m: line = line.replace("--races " + m.group(1), "--races " + ",".join(left))
        orig = queue.read_text(); pos = int((q / "pos").read_text())
        write(orig.splitlines() + [line], orig, pos, "stopped running %s, requeued %s at end (%s)" % (tag, n, ",".join(left)))
    else:
        print("OPP_TO_END stopped running", tag, "(all races done)")
