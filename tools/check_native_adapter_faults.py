#!/usr/bin/env python3
"""Process-group-bounded runner shared by the adapter fault controls (check_training_adapter_faults.py,
check_max_training_fault.py). The per-operation fault controls that lived here (wrong-operation, stall on
torch.ops.mojolearn.operation) were removed with the per-op surfaces on 2026-10-24 (docs/TOMBSTONES.md)."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, log, budget):
    start = time.monotonic()
    expired = False
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            expired = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=5)
    record = {"command": command, "exit_code": code, "timed_out": expired,
              "budget_seconds": budget, "elapsed_seconds": time.monotonic() - start, "log": str(log)}
    log.with_suffix(".status.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
