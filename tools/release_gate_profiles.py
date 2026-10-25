"""Versioned publication checks, separate from the full performance campaign."""
import math
import os
import signal
import subprocess
import time

LIGHT = "release-light-v1"
CAMPAIGN = "performance-campaign-v1"
PROFILES = (LIGHT, CAMPAIGN)
LIGHT_STAGES = ("setup", "correctness", "autograd", "adapters")
LIGHT_COVERAGE = ("install", "installed-correctness", "autograd-artifacts", "adapters")
CAMPAIGN_COVERAGE = ("install", "whole-plan-smoke", "autograd-artifacts", "adapters", "measured-plan-pairs")
# A hard ceiling, not an expected duration. Downloads and framework installation
# are timed separately; a slow installation never becomes a test PASS.
LIGHT_BUDGET_SECONDS = 1800
COMMAND_TIMEOUT_SECONDS = 180
AUTOGRAD_TIMEOUT_SECONDS = 30
CORRECTNESS_JOBS = ("locate-installed", "dependencies", "installed", "expanded-api",
                    "loaded-lm-cpu", "loaded-lm-gpu", "loaded-lm-cpu-gpu-compare",
                    "coverage", "models", "extended", "self-test")


def profile(document):
    # Absence means the historical full-campaign schema, never the light gate.
    value = document.get("profile", CAMPAIGN)
    if value not in PROFILES:
        raise ValueError("unknown release gate profile: " + str(value))
    return value


def finite_seconds(value, maximum):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= maximum


def correctness_errors(document, source_commit, version, vendor=None):
    if not isinstance(document, dict):
        return ["missing installed correctness report"]
    errors = []
    if (document.get("status") != "PASSED" or document.get("scope") != "expanded"
            or document.get("source_commit") != source_commit
            or document.get("installed_from", {}).get("expected_version") != version):
        errors.append("installed correctness scope/status/source/version differs")
    if vendor and document.get("installed", {}).get("vendor") != {"nvidia": "cuda", "amd": "hip"}.get(vendor):
        errors.append("installed correctness ran another vendor backend")
    jobs = document.get("jobs", [])
    if not isinstance(jobs, list) or [row.get("name") for row in jobs] != list(CORRECTNESS_JOBS):
        return errors + ["installed correctness must complete all 11 named jobs"]
    for row in jobs:
        # The extended verifier explicitly reports OWED properties with exit5.
        # The producer admits its checked cells and retains those unqualified
        # properties; this boundary smoke does not qualify them.
        allowed = (0, 5) if row["name"] == "extended" else (0,)
        if (type(row.get("exit_code")) is not int or row["exit_code"] not in allowed
                or row.get("timed_out") is not False
                or row.get("timeout_seconds") != COMMAND_TIMEOUT_SECONDS
                or not finite_seconds(row.get("elapsed_seconds"), COMMAND_TIMEOUT_SECONDS + 5)):
            errors.append("installed correctness failed/incomplete/timed out: " + row["name"])
    return errors


def bounded_run(command, log, timeout, *, env=None, cwd=None):
    """Kill the entire process group at the deadline; retain actual elapsed time."""
    start = time.monotonic()
    timed_out = False
    if timeout <= 0:
        return dict(exit_code=124, elapsed_seconds=0.0, timed_out=True)
    with open(log, "ab") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   env=env, cwd=cwd, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            code = 124
    return dict(exit_code=code, elapsed_seconds=round(time.monotonic() - start, 6), timed_out=timed_out)
