#!/usr/bin/env python3
"""Bound a benchmark and clean up its entire child process group."""
import argparse
import os
import signal
import subprocess
import sys


def cleanup(proc, grace=2):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # The leader may have exited while workers ignore TERM.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run(command, seconds):
    if seconds <= 0 or not command:
        raise ValueError('positive timeout and a command are required')
    proc = subprocess.Popen(command, start_new_session=True)
    try:
        return proc.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        print(f'TIMING INCOMPLETE: exceeded {seconds:g}s: {command}', file=sys.stderr, flush=True)
        return 124
    finally:
        cleanup(proc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    raise SystemExit(run(command, args.seconds))
