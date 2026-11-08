# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""identity_break's run block (lane post-record-device-time): the device, drivers, toolchain, times, per-lane
seconds, wheels, box and host every identity record states, the run_* provenance.txt lines written from it, and
the retained copy that keeps it byte for byte. Fake commands and environment only: no Mojolearn import, no
device, < 2 s."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import identity_break as ib  # noqa: E402
import retain_run_summary as R  # noqa: E402

TOOLS = Path(__file__).resolve().parent
SHA_CORE, SHA_NV = "a" * 64, "b" * 64
NVSMI_QUERY = ("0, NVIDIA L40S, 550.54.15, GPU-1111\n"
               "1, NVIDIA GeForce RTX 4090, 550.54.15, GPU-2222\n")
NVSMI_PLAIN = "| NVIDIA-SMI 550.54.15   Driver Version: 550.54.15   CUDA Version: 12.4 |\n"
BOX_VARS = ("IDENTITY_RECORD_BOX", "GITHUB_ACTIONS", "GITHUB_RUN_ID", "GITHUB_JOB", "RUNPOD_POD_ID")


def fake_commands(table):
    def run(argv, timeout=20):
        key = " ".join(argv)
        for prefix, value in table.items():
            if key.startswith(prefix):
                return value
        return None, f"{argv[0]} not on PATH"
    return run


@pytest.fixture
def clean_env(monkeypatch):
    for v in BOX_VARS + ("IDENTITY_RECORD_WHEELS", "MOJOLEARN_MOJO_VERSION", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(ib, "MOJO_DISTRIBUTIONS", ("no-such-mojo-distribution",))
    return monkeypatch


def nvidia_static(monkeypatch, tmp_path):
    wheels = tmp_path / "installed-wheels.txt"
    wheels.write_text("mojolearn-0.8.67-py3-none-manylinux_2_34_x86_64.whl %s\n"
                      "mojolearn_nvidia-0.8.67-py3-none-manylinux_2_34_x86_64.whl %s\n" % (SHA_CORE, SHA_NV))
    monkeypatch.setenv("IDENTITY_RECORD_WHEELS", str(wheels))
    monkeypatch.setenv("IDENTITY_RECORD_BOX", "runpod:pod123")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("MOJOLEARN_MOJO_VERSION", "Mojo 26.1.0")
    monkeypatch.setattr(ib, "_run_command", fake_commands({
        "nvidia-smi --query-gpu": (NVSMI_QUERY, None), "nvidia-smi": (NVSMI_PLAIN, None)}))
    return ib.run_facts_static("cuda", str(tmp_path / "venv" / "lib" / "site-packages" / "mojolearn"))


def test_record_round_trip_states_device_driver_times_wheels_box_and_lanes(clean_env, tmp_path):
    static = nvidia_static(clean_env, tmp_path)
    cells = {"knn/base": dict(timing=dict(total_seconds=1.5)), "knn/wide": dict(timing=dict(total_seconds=2.0)),
             "pca/base": dict(timing=dict(total_seconds=0.25)), "lr/base": dict(verdict="REFUSED")}
    run = ib.run_block(static, cells, started_wall=1_800_000_000.0, now=1_800_000_090.5)
    path = tmp_path / "nvidia-l40s.lanes-1.identical.json"
    ib.atomic_json(path, dict(commit="c" * 40, vendor="nvidia-l40s", cells=cells, run=run))
    back = json.loads(path.read_text())["run"]
    assert back == json.loads(json.dumps(run))
    assert back["schema"] == ib.RUN_FACTS_SCHEMA
    # the device this process saw (CUDA_VISIBLE_DEVICES=1), never simply the first GPU
    assert back["device"] == dict(kind="gpu", backend="cuda", model="NVIDIA GeForce RTX 4090", index="1",
                                  count_used=1, unit="devices")
    assert back["driver"]["nvidia_driver"] == "550.54.15" and back["driver"]["cuda"] == "12.4"
    assert back["mojo"] == "Mojo 26.1.0"
    assert back["python"]["version"]
    assert back["started_utc"] == "2027-01-15T08:00:00Z" and back["finished_utc"] == "2027-01-15T08:01:30Z"
    assert back["seconds_total"] == 90.5 and back["attempts"] == 1
    assert back["seconds_per_lane"] == {"knn": 3.5, "pca": 0.25}
    assert back["wheels"] == [
        dict(name="mojolearn-0.8.67-py3-none-manylinux_2_34_x86_64.whl", sha256=SHA_CORE, version="0.8.67"),
        dict(name="mojolearn_nvidia-0.8.67-py3-none-manylinux_2_34_x86_64.whl", sha256=SHA_NV, version="0.8.67")]
    assert back["box"] == dict(provider="runpod", id="pod123", source="IDENTITY_RECORD_BOX")
    assert back["host"]["kernel"] and len(back["host"]["hostname_sha256"]) == 16
    assert "device.model" not in back["unavailable"] and "wheels" not in back["unavailable"]


def test_provenance_lines_parse_back_to_the_run_block(clean_env, tmp_path):
    static = nvidia_static(clean_env, tmp_path)
    run = ib.run_block(static, {"knn/base": dict(timing=dict(total_seconds=4.0))}, 1_800_000_000.0,
                       now=1_800_000_010.0)
    lines = ib.run_provenance_lines(dict(run=run))
    assert [line.partition("=")[0] for line in lines] == list(ib.RUN_PROVENANCE_KEYS)
    assert all("\n" not in line for line in lines)
    p = ib.parse_provenance("commit=%s\nlabel=nvidia-l40s\n%s\nrecord_sha256=%s\n" % ("c" * 40, "\n".join(lines),
                                                                                     "d" * 64))
    assert p["run_device"] == "NVIDIA GeForce RTX 4090" and p["run_device_count"] == "1 devices"
    assert "cuda=12.4" in p["run_driver"] and "nvidia_driver=550.54.15" in p["run_driver"]
    assert p["run_mojo"] == "Mojo 26.1.0"
    assert p["run_started_utc"] == "2027-01-15T08:00:00Z" and p["run_finished_utc"] == "2027-01-15T08:00:10Z"
    assert p["run_seconds_total"] == "10.0" and p["run_seconds_per_lane"] == "knn:4.0"
    assert p["run_wheels"].split(",")[0] == "mojolearn-0.8.67-py3-none-manylinux_2_34_x86_64.whl:%s:0.8.67" % SHA_CORE
    assert p["run_box"] == "runpod:pod123"
    assert p["commit"] == "c" * 40 and p["record_sha256"] == "d" * 64
    # a record from a harness before the run block says so instead of inventing facts
    assert ib.run_provenance_lines(dict(commit="c" * 40)) == [
        "run_facts=absent (a record from a harness before the run block)"]


def test_the_cli_prints_the_provenance_lines(clean_env, tmp_path):
    static = nvidia_static(clean_env, tmp_path)
    path = tmp_path / "x.identical.json"
    path.write_text(json.dumps(dict(run=ib.run_block(static, {}, 1_800_000_000.0, now=1_800_000_001.0))))
    out = subprocess.run([sys.executable, str(TOOLS / "identity_break.py"), "--provenance-lines", str(path)],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.splitlines() == ib.run_provenance_lines(json.loads(path.read_text()))


def test_unavailable_facts_write_null_and_a_reason_and_never_refuse(clean_env, tmp_path):
    clean_env.setattr(ib, "_run_command", fake_commands({}))      # no nvidia-smi, no mojo, nothing
    clean_env.setattr(ib.platform, "system", lambda: "Linux")
    clean_env.setattr(ib, "box_identity", lambda: dict(provider=None, id=None, source="no provider variable"))
    static = ib.run_facts_static("cuda", str(tmp_path / "python" / "mojolearn"))
    run = ib.run_block(static, {}, 1_800_000_000.0, now=1_800_000_001.0)
    assert run["device"]["model"] is None and "not on PATH" in run["unavailable"]["device.model"]
    assert run["driver"]["nvidia_driver"] is None and run["unavailable"]["driver.nvidia_driver"]
    assert run["driver"]["cuda"] is None and run["unavailable"]["driver.cuda"]
    assert run["mojo"] is None and "compiled bindings" in run["unavailable"]["mojo"]
    assert run["wheels"] is None and run["unavailable"]["wheels"] == \
        "recorded from the source tree, not an installed wheel"
    assert run["box"]["id"] is None and run["unavailable"]["box.id"] == "no provider variable"
    lines = ib.parse_provenance("\n".join(ib.run_provenance_lines(dict(run=run))))
    assert lines["run_device"] == "null" and lines["run_wheels"] == "null" and lines["run_box"] == "null:null"
    assert "device.model: nvidia-smi not on PATH" in lines["run_unavailable"]
    # a command that raises inside a probe is a reason too, never a lost record
    def boom(argv, timeout=20):
        raise RuntimeError("probe exploded")
    clean_env.setattr(ib, "_run_command", boom)
    static = ib.run_facts_static("hip", str(tmp_path))
    assert static["device"]["model"] is None and "probe exploded" in static["unavailable"]["device"]


def test_box_is_derived_from_the_environment_when_the_launcher_names_none(clean_env):
    clean_env.setattr(ib.platform, "system", lambda: "Linux")
    clean_env.setenv("GITHUB_ACTIONS", "true")
    clean_env.setenv("GITHUB_RUN_ID", "987")
    clean_env.setenv("GITHUB_JOB", "wheel-column")
    assert ib.box_identity() == dict(provider="github", id="987/wheel-column", source="GITHUB_RUN_ID/GITHUB_JOB")
    clean_env.delenv("GITHUB_ACTIONS")
    clean_env.setenv("RUNPOD_POD_ID", "abc")
    assert ib.box_identity()["provider"] == "runpod" and ib.box_identity()["id"] == "abc"
    clean_env.setenv("IDENTITY_RECORD_BOX", "digitalocean:55501")
    assert ib.box_identity() == dict(provider="digitalocean", id="55501", source="IDENTITY_RECORD_BOX")
    clean_env.setenv("IDENTITY_RECORD_BOX", "nonsense")
    assert ib.box_identity()["provider"] is None and "not <provider>:<id>" in ib.box_identity()["source"]
    clean_env.delenv("IDENTITY_RECORD_BOX")
    clean_env.delenv("RUNPOD_POD_ID")
    clean_env.setattr(ib.platform, "system", lambda: "Darwin")
    mac = ib.box_identity()
    assert mac["provider"] == "mac" and len(mac["id"]) == 16


def test_amd_and_apple_and_cpu_devices(clean_env, tmp_path):
    rocminfo = ("Agent 1\n  Name:                    AMD EPYC 9534\n  Marketing Name:          AMD EPYC 9534\n"
                "Agent 2\n  Name:                    gfx942\n  Marketing Name:          AMD Instinct MI300X\n")
    clean_env.setattr(ib, "_run_command", fake_commands({"rocminfo": (rocminfo, None)}))
    dev, drv = ib._amd_device({})
    assert dev["model"] == "AMD Instinct MI300X" and dev["arch"] == "gfx942" and dev["index"] == "0"
    clean_env.setattr(ib, "_run_command", fake_commands({"system_profiler": (
        "      Chipset Model: Apple M4\n      Total Number of Cores: 10\n      Metal Support: Metal 3\n", None)}))
    clean_env.setattr(ib, "cpu_model", lambda: "Apple M4")
    dev, drv = ib._apple_device({})
    assert dev == dict(model="Apple M4", gpu_cores=10) and drv["metal"] == "Metal 3"
    clean_env.setenv("MOJOLEARN_CPU_THREADS", "3")
    clean_env.setattr(ib, "cpu_model", lambda: "Intel Xeon 6973P-C")
    cpu = ib.run_facts_static("cpu", str(tmp_path))["device"]
    assert cpu == dict(kind="cpu", backend="cpu", model="Intel Xeon 6973P-C", count_used=3, unit="cpu threads")


def test_a_resume_keeps_the_first_start_and_adds_the_seconds(clean_env, tmp_path):
    static = nvidia_static(clean_env, tmp_path)
    first = ib.run_block(static, {}, 1_800_000_000.0, now=1_800_000_100.0)
    second = ib.run_block(static, {"knn/base": dict(timing=dict(total_seconds=1.0))}, 1_800_001_000.0,
                          previous=first, now=1_800_001_050.0)
    assert second["started_utc"] == first["started_utc"] and second["seconds_total"] == 150.0
    assert second["attempts"] == 2 and second["finished_utc"] == "2027-01-15T08:17:30Z"


def test_merge_joins_the_parts_run_blocks(clean_env, tmp_path):
    static = nvidia_static(clean_env, tmp_path)
    a = ib.run_block(static, {}, 1_800_000_000.0, now=1_800_000_060.0)
    b = ib.run_block(static, {}, 1_800_000_030.0, now=1_800_000_200.0)
    cells = {"knn/base": dict(timing=dict(total_seconds=2.0)), "pca/base": dict(timing=dict(total_seconds=3.0))}
    m = ib.merged_run_block([a, b, None], cells)
    assert m["started_utc"] == a["started_utc"] and m["finished_utc"] == b["finished_utc"]
    assert m["seconds_total"] == 230.0 and m["parts"] == 2 and m["seconds_per_lane"] == {"knn": 2.0, "pca": 3.0}
    assert m["device"] == a["device"]
    assert ib.merged_run_block([None], cells) is None


def test_the_run_block_is_outside_merge_equality_and_the_resume_signature(clean_env):
    assert "run" not in ib.MERGE_SAME
    # the box and wheel variables never reach the resume signature's environment digests
    for name in (ib.RUN_BOX_ENV, ib.RUN_WHEELS_ENV):
        assert not name.startswith(("MOJO", "MODULAR", "OMP_", "MKL_", "OPENBLAS_", "VECLIB_", "NUMEXPR_", "BLIS_"))


def test_retention_keeps_the_run_block_and_its_provenance_lines_byte_for_byte(clean_env, tmp_path):
    root = tmp_path / "mojolearn"
    (root / ".git").mkdir(parents=True)
    (root / "tools").mkdir()
    clean_env.setenv("MOJOLEARN_SUMMARY_REPO", str(root))
    static = nvidia_static(clean_env, tmp_path)
    record = dict(commit="c" * 40, vendor="nvidia-l40s", cells={},
                  run=ib.run_block(static, {}, 1_800_000_000.0, now=1_800_000_005.0))
    column = tmp_path / "out" / "nvidia-l40s.lanes-1.identical.json"
    column.parent.mkdir()
    column.write_text(json.dumps(record, indent=1))
    prov = column.with_name("nvidia-l40s.lanes-1.provenance.txt")
    prov.write_text("commit=%s\n%s\n" % ("c" * 40, "\n".join(ib.run_provenance_lines(record))))
    target = R.retain_identity(column)
    assert target.read_bytes() == column.read_bytes()
    assert json.loads(target.read_text())["run"] == record["run"]
    assert target.with_name(prov.name).read_bytes() == prov.read_bytes()
