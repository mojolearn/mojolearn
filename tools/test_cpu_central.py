"""tools/cpu_central.sh, the CPU check box (lane fx-cpu-box-checks, 2026-11-19): argument handling and every refusal
path, with NO rental and no RunPod call (MOJOLEARN_NVC_NO_API=1, the cap and balance from test variables). Also the CPU
request builder and instance pick in tools/runpod_gpu_request.py, the dev_pod.sh CPU refusals, and the box job's
argument and preflight handling."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import runpod_gpu_request as rgr  # noqa: E402

CPC = ROOT / "tools/cpu_central.sh"
DEVPOD = ROOT / "tools/dev_pod.sh"
JOB = ROOT / "tools/ops/cpu_box_checks_job.sh"
STEPS = ROOT / "tools/ops/native_checks_steps.sh"


def env_for(tmp_path, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MOJOLEARN_", "CPC_", "RUNPOD_"))}
    env.update(MOJOLEARN_CPC_HOME=str(tmp_path / "cpc"), MOJOLEARN_DEVPOD_STATE=str(tmp_path / "devpods"),
               MOJOLEARN_NVC_NO_API="1", MOJOLEARN_RUNPOD_KEY_FILE=str(tmp_path / "no-key"),
               MOJOLEARN_PERSIST_HOME=str(tmp_path / "persist"), MOJOLEARN_DEVPOD_DISK_GB="100",
               TMPDIR=str(tmp_path), CPC_ALLOW_UNPUSHED="1")
    env.update(extra)
    return env


def run(tmp_path, *args, script=CPC, **extra):
    return subprocess.run(["bash", str(script), *args], env=env_for(tmp_path, **extra), capture_output=True,
                          text=True, timeout=120, cwd=ROOT)


def no_box_rented(tmp_path):
    return not (tmp_path / "devpods" / "cpc1").exists()


def test_usage_and_unknown_command(tmp_path):
    got = run(tmp_path)
    assert got.returncode == 2 and "cpu_central.sh up [minutes]" in got.stdout
    got = run(tmp_path, "nonsense")
    assert got.returncode == 2


@pytest.mark.parametrize("cmd", [["up"], ["down"], ["tools", "HEAD"]])
def test_provisioning_is_orchestrator_only(tmp_path, cmd):
    got = run(tmp_path, *cmd)
    assert got.returncode != 0 and "only the orchestrator" in got.stderr
    assert no_box_rented(tmp_path)


@pytest.mark.parametrize("cmd", [["check", "HEAD"], ["status"], ["log", "cpc1-0001"], ["summary", "0001"],
                                 ["cancel", "cpc1-0001"], ["sh", "true"]])
def test_no_box_refuses_every_box_command(tmp_path, cmd):
    got = run(tmp_path, *cmd)
    assert got.returncode != 0 and "the CPU check box is not up" in got.stderr, got.stderr


def test_check_argument_handling(tmp_path):
    got = run(tmp_path, "check")
    assert got.returncode != 0 and "check <sha>" in got.stderr
    got = run(tmp_path, "check", "HEAD", "native", "--bogus")
    assert got.returncode != 0 and "unexpected '--bogus'" in got.stderr
    got = run(tmp_path, "check", "HEAD", "--cap", "x")
    assert got.returncode != 0 and "--cap MIN" in got.stderr
    got = run(tmp_path, "check", "HEAD", "--timeout", "0")
    assert got.returncode != 0 and "--timeout S" in got.stderr
    got = run(tmp_path, "check", "no-such-commit-anywhere")
    assert got.returncode != 0 and "no commit" in got.stderr
    for bad in ("log", "cancel"):
        got = run(tmp_path, bad, "nvc1-x")
        assert got.returncode != 0 and "a job id is cpc1-<n>" in got.stderr


def test_up_refuses_at_the_account_cap(tmp_path):
    got = run(tmp_path, "up", MOJOLEARN_ORCHESTRATOR="1", MOJOLEARN_RENT_GUARD_TEST_PODS="12 3",
              MOJOLEARN_NVC_TEST_BALANCE="100")
    assert got.returncode != 0 and "rent_guard.sh refused" in got.stderr and "REFUSED" in got.stderr
    assert no_box_rented(tmp_path)


def test_up_refuses_a_low_balance(tmp_path):
    got = run(tmp_path, "up", MOJOLEARN_ORCHESTRATOR="1", MOJOLEARN_RENT_GUARD_TEST_PODS="1 0",
              MOJOLEARN_NVC_TEST_BALANCE="3")
    assert got.returncode != 0 and "under $10" in got.stderr
    assert no_box_rented(tmp_path)


def test_up_minutes_bounds(tmp_path):
    for bad in ("0", "241", "x"):
        got = run(tmp_path, "up", bad, MOJOLEARN_ORCHESTRATOR="1")
        assert got.returncode != 0 and "1..240" in got.stderr
    got = run(tmp_path, "up", "60", "70", MOJOLEARN_ORCHESTRATOR="1")
    assert got.returncode != 0 and "unexpected" in got.stderr


@pytest.mark.parametrize("var,value", [("MOJOLEARN_DEVPOD_CPU_IMAGE", "apple/silicon-runner:1"),
                                       ("MOJOLEARN_CPC_FLAVORS", "macos-m2"),
                                       ("CPC_PROVIDER", "darwin-cloud")])
def test_anything_apple_is_refused_before_anything_else(tmp_path, var, value):
    for cmd in (["up"], ["check", "HEAD"], ["status"]):
        got = run(tmp_path, *cmd, MOJOLEARN_ORCHESTRATOR="1", MOJOLEARN_RENT_GUARD_TEST_PODS="0 0", **{var: value})
        assert got.returncode != 0 and "Apple machine" in got.stderr, (cmd, got.stderr)
    assert no_box_rented(tmp_path)


def test_other_provider_is_refused(tmp_path):
    got = run(tmp_path, "status", CPC_PROVIDER="elsewhere")
    assert got.returncode != 0 and "CPC_PROVIDER must be runpod" in got.stderr


def test_up_prints_the_picked_request_without_renting(tmp_path):
    cat = tmp_path / "catalog.json"
    cat.write_text(json.dumps({"data": {"cpuFlavors": [
        {"id": "cpu3c", "maxVcpu": 32, "ramMultiplier": 2}, {"id": "cpu5c", "maxVcpu": 32, "ramMultiplier": 2},
        {"id": "cpu3g", "maxVcpu": 64, "ramMultiplier": 4}, {"id": "cpu3m", "maxVcpu": 16, "ramMultiplier": 8}]}}))
    got = run(tmp_path, "up", MOJOLEARN_ORCHESTRATOR="1", MOJOLEARN_RENT_GUARD_TEST_PODS="1 0",
              MOJOLEARN_CPC_CATALOG=str(cat), MOJOLEARN_CPC_PRINT_REQUEST="1")
    assert got.returncode == 0, got.stderr
    assert "64 vCPU of flavors cpu3g" in got.stdout and "RunPod's CPU catalog" in got.stdout
    req = json.loads(got.stdout.split("request: ", 1)[1].strip())
    # the RunPod API v2 CreatePodRequest (runpod/docs api-reference-v2/openapi.json): one flavor, a vCPU count
    assert req["cpu"] == {"id": "cpu3g", "vcpuCount": 64} and req["image"] and req["startSsh"] is True
    assert req["ports"] == ["22/tcp"] and req["cloud"] == "SECURE" and isinstance(req["disk"], int)
    assert not {"gpu", "gpuTypeIds", "imageName", "computeType", "cpuFlavorIds", "vcpuCount", "queue"} & set(req)
    assert no_box_rented(tmp_path)
    # no catalog: the fallback order at the minimum vCPUs
    got = run(tmp_path, "up", MOJOLEARN_ORCHESTRATOR="1", MOJOLEARN_RENT_GUARD_TEST_PODS="1 0",
              MOJOLEARN_CPC_PRINT_REQUEST="1")
    assert got.returncode == 0 and "fallback order" in got.stdout and "32 vCPU of flavors cpu5c,cpu3c" in got.stdout


def test_cpu_pick_rules():
    cat = [{"id": "cpu3c", "maxVcpu": 32, "ramMultiplier": 2, "pricePerVcpu": 0.03},
           {"id": "cpu5c", "maxVcpu": 32, "ramMultiplier": 2, "pricePerVcpu": 0.025},
           {"id": "cpu3g", "maxVcpu": 32, "ramMultiplier": 4, "pricePerVcpu": 0.04},
           {"id": "cpu5g", "maxVcpu": 8, "ramMultiplier": 4, "pricePerVcpu": 0.01}]
    assert rgr.pick_cpu_instance(cat, 32, 2) == (["cpu5c", "cpu3c", "cpu3g"], 32)     # cheapest per vCPU first
    assert rgr.pick_cpu_instance(cat, 32, 4) == (["cpu3g"], 32)                       # >= 4 GB a vCPU
    assert rgr.pick_cpu_instance(None, 32, 2)[0][:2] == ["cpu5c", "cpu3c"]
    assert all(f[-1] in "gm" for f in rgr.pick_cpu_instance([], 48, 4)[0])
    with pytest.raises(ValueError):
        rgr.pick_cpu_instance(None, 32, 16)
    # v2 takes only a power-of-two vCPU count: a 48-vCPU flavor offers 32; the fallback rounds the minimum up
    assert rgr.pick_cpu_instance([{"id": "cpu5c", "maxVcpu": 48, "ramMultiplier": 2}], 32, 2) == (["cpu5c"], 32)
    assert rgr.pick_cpu_instance(None, 48, 2)[1] == 64
    with pytest.raises(ValueError):
        rgr.pick_cpu_instance([{"id": "cpu5c", "maxVcpu": 48, "ramMultiplier": 2}], 48, 2)


def test_cpu_request_refusals():
    good = rgr.build_cpu_request("n", "runpod/base:1.3.1-ubuntu2204", "cpu5c,cpu3c", 32, 100, env={})
    assert good == {"name": "n", "image": "runpod/base:1.3.1-ubuntu2204", "cloud": "SECURE", "disk": 100,
                    "startSsh": True, "ports": ["22/tcp"], "cpu": {"id": "cpu5c", "vcpuCount": 32}}
    # one flavor per create attempt, in our order, wrapping
    assert [rgr.build_cpu_request("n", "runpod/base:1", "cpu5c,cpu3c", 32, 100, a, env={})["cpu"]["id"]
            for a in range(3)] == ["cpu5c", "cpu3c", "cpu5c"]
    assert rgr.build_cpu_request("n", "runpod/base:1", "cpu5c", 32, 100,
                                 env={"MOJOLEARN_DEVPOD_CLOUD_TYPE": "COMMUNITY"})["cloud"] == "COMMUNITY"
    for vcpu in (24, 48, 1024):
        with pytest.raises(ValueError):
            rgr.build_cpu_request("n", "runpod/base:1", "cpu5c", vcpu, 100, env={})
    with pytest.raises(ValueError):
        rgr.build_cpu_request("n", "runpod/base:1", "cpu5c", 32, 100, -1, env={})
    for flavors in ("m2pro", "applem2", "cpu4c", ""):
        with pytest.raises(ValueError):
            rgr.build_cpu_request("n", "runpod/base:1", flavors, 32, 100, env={})
    with pytest.raises(ValueError, match="Apple"):
        rgr.build_cpu_request("n", "ghcr.io/x/macos-runner:1", "cpu5c", 32, 100, env={})
    with pytest.raises(ValueError):
        rgr.build_cpu_request("n", "runpod/base:1", "cpu5c", 1, 100, env={})


def test_cpu_request_cli_attempt(tmp_path):
    out = tmp_path / "req.json"
    for args, want in (([], "cpu5c"), (["1"], "cpu3c")):
        got = subprocess.run([sys.executable, str(ROOT / "tools/runpod_gpu_request.py"), "cpu", str(out), "n",
                              "runpod/base:1", "cpu5c,cpu3c", "32", "100", *args], capture_output=True, text=True)
        assert got.returncode == 0, got.stderr
        assert json.loads(out.read_text())["cpu"]["id"] == want
    got = subprocess.run([sys.executable, str(ROOT / "tools/runpod_gpu_request.py"), "cpu", str(out), "n",
                          "runpod/base:1", "cpu5c", "48", "100"], capture_output=True, text=True)
    assert got.returncode != 0 and "power of two" in got.stderr


def test_dev_pod_cpu_create_is_v2():
    """The CPU create, readiness and delete use v2: the v1 host rest.runpod.io stopped resolving (2026-11-20)."""
    src = DEVPOD.read_text()
    assert 'if [ "$4" = 1 ]; then rp_select_api 1; else rp_select_api 2; fi' in src
    assert '"$CPU_VCPU" "$DISK_GB" "$((attempt - 1))"' in src
    assert "create HTTP 202 (queued)" in src


def test_dev_pod_cpu_refusals(tmp_path):
    got = run(tmp_path, "up", "cpc1", "60", "--vendor", "cpu", script=DEVPOD)
    assert got.returncode != 0 and "only tools/cpu_central.sh rents the CPU check box" in got.stderr
    assert no_box_rented(tmp_path)
    for cmd, word in ((["extend", "cpc1"], "nothing extended"), (["down", "cpc1"], "tools/cpu_central.sh down"),
                      (["run", "cpc1", "true"], "tools/cpu_central.sh check/sh"),
                      (["sync", "cpc1", str(ROOT)], "tools/cpu_central.sh check/sh")):
        got = run(tmp_path, *cmd, script=DEVPOD)
        assert got.returncode != 0 and word in got.stderr, (cmd, got.stderr)
    got = run(tmp_path, "up", "x", "60", "--vendor", "sparc", script=DEVPOD)
    assert got.returncode != 0 and "nvidia, amd or cpu" in got.stderr
    got = run(tmp_path, "seed", "cpc1", "HEAD", "/root/x", script=DEVPOD)
    assert got.returncode != 0 and "no box for cpc1" in got.stderr


def test_box_job_arguments_and_preflight(tmp_path):
    for args in ([], ["abc"], ["0" * 40, "everything"]):
        got = run(tmp_path, *args, script=JOB)
        assert got.returncode == 2 and "usage" in got.stderr
    got = run(tmp_path, "--preflight", "0" * 40, "parse", script=JOB, CPU_CHECKS_HOME=str(tmp_path / "box"))
    assert got.returncode != 0 and "PREFLIGHT FAIL" in got.stdout and "is not in" in got.stdout
    assert "PREFLIGHT ok steps script" in got.stdout
    # the sha and suite also come from the submit's --env values
    got = run(tmp_path, "--preflight", script=JOB, CPU_CHECKS_HOME=str(tmp_path / "box"), CPU_CHECK_SHA="1" * 40,
              CPU_CHECK_SUITE="tools")
    assert "PREFLIGHT FAIL" in got.stdout and "usage" not in got.stderr


def test_steps_script_names_its_steps(tmp_path):
    got = run(tmp_path, script=STEPS)
    assert got.returncode == 2 and "parse-group" in got.stdout
    text = STEPS.read_text()
    for step in ("plan", "parse_group", "build", "shard", "summary", "verdict"):
        assert f"step_{step}() {{" in text
    job = JOB.read_text()
    # the box job calls the workflow's steps and the ref's tool test runner, never its own copy of them
    for step in ("plan", "parse-group", "build", "shard", "summary", "verdict"):
        assert f'bash "$STEPS" {step}' in job, step
    assert "bash tools/run_tool_tests.sh" in job and "mojo build" not in job
    assert "busy_sampler" in job and "BUSY" in job and "busy cores of" in job
