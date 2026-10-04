#!/usr/bin/env python3
"""M3 queue health check, scheduled every 20 minutes; never runs benchmarks.

Restart only a missing runner with pending work and no surviving benchmark or
compiler. Live stalls are sampled and recorded for the manager to diagnose;
they are never mistaken for successful measurements or killed blindly.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def process_table(raw):
    result = {}
    for line in raw.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) == 4:
            result[int(parts[0])] = {"ppid": int(parts[1]), "cpu_time": parts[2], "command": parts[3]}
    return result


def classify(processes, home, pos, lines):
    runner_command = "bash " + str(home / "mq.sh")
    runners = {p for p, v in processes.items() if v["command"] == runner_command}
    roots = sorted(p for p in runners if processes[p]["ppid"] not in runners)
    workers = sorted(p for p, v in processes.items() if any(
        token in v["command"] for token in ("bench_board_algos.py", "bench_board_more.py", "afc_ab.sh ",
                                             "aft_ab.sh ", "mojo build ", "sgdoc_check.py")))
    stopped = 0 < pos <= len(lines) and lines[pos - 1].strip() == "STOP"
    restart = not roots and not workers and pos < len(lines) and not stopped
    return roots, workers, restart


def tail(path, count=20):
    if not path.exists():
        return []
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 16384))
        return stream.read().decode(errors="replace").splitlines()[-count:]


def check(home, recover=False):
    q = home / "mq"
    now = time.time()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    pos = int((q / "pos").read_text())
    lines = (q / "queue.txt").read_text().splitlines()
    results = q / "results.txt"
    recent = tail(results)
    current = lines[pos - 1] if 0 < pos <= len(lines) else ""
    fields = current.split()
    tag = fields[2] if len(fields) > 2 and fields[0] == "CMD" else ""
    logs = [q / "out" / (tag + ".log")] if tag else []
    if len(fields) > 1:
        safe = fields[1].replace("/", "_")
        logs.extend((q / "out").glob(safe + "-*build*.log"))
        logs.extend((q / "out").glob(safe + "-ibase.log"))
    progress = max([results.stat().st_mtime] + [p.stat().st_mtime for p in logs if p.exists()])
    processes = process_table(subprocess.check_output(["ps", "-axo", "pid=,ppid=,time=,command="], text=True))
    roots, workers, restart = classify(processes, home, pos, lines)
    state_path = q / "watchdog-state.json"
    old = json.loads(state_path.read_text()) if state_path.exists() else {}
    same = old.get("pos") == pos and old.get("tag") == tag
    started = old.get("job_observed_since", now) if same else now
    state = {"checked_at": timestamp, "pos": pos, "total": len(lines), "pending": len(lines) - pos,
             "tag": tag, "runner_pids": roots, "worker_pids": workers,
             "job_observed_since": started, "silent_seconds": round(now - progress),
             "free_gib": round(shutil.disk_usage(home).free / 2**30, 1), "alerts": [], "action": "observe"}
    if len(roots) > 1:
        state["alerts"].append("multiple queue runners: inspect before further timing")
    if state["free_gib"] < 10:
        state["alerts"].append("disk below 10 GiB: reclaim only verified disposable build outputs")
    if same and now - started >= 2400 and now - progress >= 2400 and workers:
        state["alerts"].append("live job has no log/result progress for 40 minutes; inspect sample")
        sample_pid = next((p for p in workers if "python" in processes[p]["command"]), workers[0])
        sample_file = q / "out" / ("watchdog-%d-%s.sample.txt" % (pos, tag or "job"))
        if not sample_file.exists():
            subprocess.run(["sample", str(sample_pid), "2", "-file", str(sample_file)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        state["sample"] = str(sample_file)
    if not roots and workers:
        state["alerts"].append("runner absent but surviving worker exists; restart refused")
    if restart:
        state["alerts"].append("queue runner missing with pending work")
        if recover:
            # Check a second time immediately before launch to avoid a restart race.
            again = process_table(subprocess.check_output(["ps", "-axo", "pid=,ppid=,time=,command="], text=True))
            if classify(again, home, int((q / "pos").read_text()), (q / "queue.txt").read_text().splitlines())[2]:
                subprocess.run(["/usr/bin/perl", str(home / "daemon.pl"), str(home / "mq.sh")], check=True)
                state["action"] = "restarted missing runner; interrupted job is not replayed"
    state["recent_failures"] = [x[:500] for x in recent if re.search(r"(?:rc=[1-9]|FETCHFAIL|FAIL|INCOMPLETE)", x)]
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    os.replace(tmp, state_path)
    with (q / "watchdog.log").open("a") as stream:
        stream.write(json.dumps(state) + "\n")
    print(json.dumps(state))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path.home())
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    with (args.home / "mq/watchdog.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        check(args.home, args.recover)


if __name__ == "__main__":
    main()
