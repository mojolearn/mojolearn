"""A launcher exit must not remove an already armed bounded lease guard."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


LAUNCHER = Path(__file__).resolve().parents[1] / "deadman_session.py"


def test_deadman_survives_launcher_exit_in_own_session(tmp_path):
    script = tmp_path / "finite-deadman.sh"
    script.write_text("#!/bin/sh\nsleep 20\n")
    launched = subprocess.run([sys.executable, str(LAUNCHER), str(script)],
                              capture_output=True, text=True, check=True, timeout=5)
    pid = int(launched.stdout.strip())
    try:
        assert os.getsid(pid) == pid
        assert os.getpgid(pid) == pid
        os.kill(pid, 0)
        # The former invoking Python process has exited, yet this guard is live.
        time.sleep(0.05)
        os.kill(pid, 0)
    finally:
        # Only this test's new session is terminated; no real rental is touched.
        os.killpg(pid, signal.SIGTERM)


def test_missing_script_never_creates_a_guard(tmp_path):
    launched = subprocess.run([sys.executable, str(LAUNCHER), str(tmp_path / "absent")],
                              capture_output=True, text=True, timeout=5)
    assert launched.returncode != 0
    assert not launched.stdout.strip()
