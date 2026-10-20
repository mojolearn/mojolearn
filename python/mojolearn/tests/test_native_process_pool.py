"""Independent wire/process observations for the native local worker owner.

Test controllers may prepare NumPy data and inspect process state. The audited
pool operation itself must contain no Python callbacks, orchestration or codecs.
"""
import os
import sys
import json

import numpy as np
import pytest

from mojolearn import _backend
from mojolearn._native_process import _native_process
from mojolearn._verify_native_runtime import audit_native_call


def call(operation, args):
    return _native_process(operation, args)


def checked(operation, args):
    audit = audit_native_call(call, operation, args)
    audit.assert_native()
    return audit.result


@pytest.fixture(autouse=True)
def reset_vendor():
    yield
    _native_process("initialize", (sys.executable, os.environ._data, _backend.vendor()))


@pytest.mark.parametrize("vendor,key,other", [
    ("cuda", "CUDA_VISIBLE_DEVICES", None),
    ("hip", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"),
])
def test_visibility_mapping_and_cooperative_count(monkeypatch, vendor, key, other):
    monkeypatch.setenv("MOJOLEARN_VENDOR", "cpu")
    monkeypatch.setenv(key, "GPU-a,GPU-b,GPU-c")
    if other:
        monkeypatch.setenv(other, "discard-this-second-mask")
    _native_process("initialize", (sys.executable, os.environ._data, vendor))
    env = dict(row.split("=", 1) for row in checked("environment", ((2, 0), (0,))))
    assert env[key] == "GPU-a"
    assert env["MOJOLEARN_OPTIMIZER_DEVICE_COUNT"] == "1"
    assert env["MOJOLEARN_NUMERIC_MODE"] == "identical"
    assert env["MOJOLEARN_VENDOR"] == vendor
    if other:
        assert other not in env
    env = dict(row.split("=", 1) for row in checked("environment", ((2, 0), (2, 0))))
    assert env[key] == "GPU-c,GPU-a"
    assert env["MOJOLEARN_OPTIMIZER_DEVICE_COUNT"] == "2"
    monkeypatch.setenv(key, "GPU-a,GPU-a")
    with pytest.raises(ValueError, match="repeat"):
        call("environment", ((0, 1), (0,)))


@pytest.mark.parametrize("vendor", ["cpu", "metal"])
def test_environment_keeps_connected_cpu_or_metal_route(monkeypatch, vendor):
    _native_process("initialize", (sys.executable, os.environ._data, vendor))
    monkeypatch.setenv("MOJOLEARN_VENDOR", "metal" if vendor == "cpu" else "cpu")
    monkeypatch.setenv("MOJOLEARN_NUMERIC_MODE", "fast")
    env = dict(row.split("=", 1) for row in checked("environment", ((0,), (0,))))
    assert env["MOJOLEARN_VENDOR"] == vendor
    assert env["MOJOLEARN_NUMERIC_MODE"] == "identical"


def test_real_worker_keeps_imported_vendor_after_environment_changes(monkeypatch):
    vendor = _backend.vendor()
    _native_process("initialize", (sys.executable, os.environ._data, vendor))
    # A CPU run exercises the reverse change without needing a GPU install;
    # each GPU run proves that its parent cannot silently spawn CPU workers.
    monkeypatch.setenv("MOJOLEARN_VENDOR", "metal" if vendor == "cpu" else "cpu")
    pool = checked("create", ((0,), False, "mojolearn._native_process"))
    try:
        assert checked("map", (pool, ["vendor-pinned"])) == ["vendor-pinned"]
        assert pool["workers"][0]["identity"]["vendor"] == vendor
    finally:
        checked("close", (pool,))
    assert pool["workers"] == []


@pytest.mark.parametrize("missing_vendor", [False, True])
def test_authenticated_wrong_or_missing_vendor_is_refused_and_reaped(tmp_path, monkeypatch, missing_vendor):
    # Independent protocol fixture: authenticate with the real PID/token but
    # supply contradictory or absent backend metadata. No model work runs.
    (tmp_path / "native_pool_wrong_vendor.py").write_text('''
import json, os, socket, struct, sys
from pathlib import Path
record = Path(os.environ["NATIVE_POOL_VENDOR_RECORD"])
hello = {"kind": "hello", "token": sys.argv[2], "pid": os.getpid(), "nbytes": 0}
if os.environ["NATIVE_POOL_OMIT_VENDOR"] != "1":
    hello["vendor"] = os.environ["NATIVE_POOL_WRONG_VENDOR"]
record.write_text(json.dumps({"pid": os.getpid(), "hello": hello}))
with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=10) as sock:
    raw = json.dumps(hello).encode("ascii")
    sock.sendall(struct.pack("!I", len(raw)) + raw)
    with sock.makefile("rb") as reader:
        prefix = reader.read(4)
        received = json.loads(reader.read(struct.unpack("!I", prefix)[0])) if len(prefix) == 4 else {}
    record.write_text(json.dumps({"pid": os.getpid(), "hello": hello, "received_kind": received.get("kind")}))
''')
    record = tmp_path / "child.json"
    monkeypatch.setenv("NATIVE_POOL_VENDOR_RECORD", str(record))
    monkeypatch.setenv("NATIVE_POOL_OMIT_VENDOR", "1" if missing_vendor else "0")
    monkeypatch.setenv("NATIVE_POOL_WRONG_VENDOR", "metal" if _backend.vendor() == "cpu" else "cpu")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    pool = checked("create", ((0,), False, "native_pool_wrong_vendor"))
    try:
        audit = audit_native_call(call, "map", (pool, ["must-not-run"]))
        assert isinstance(audit.exception, ValueError), audit.report()
        assert "child vendor differs" in str(audit.exception)
        for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
            assert not audit.report()[key], audit.report()
        assert pool["workers"] == [] and pool["closed"] is False
        child = json.loads(record.read_text())
        assert child["received_kind"] in (None, "close"), child
        with pytest.raises(ChildProcessError):
            os.waitpid(child["pid"], os.WNOHANG)
        # A rejected handshake cannot leave the pool poisoned or old replies
        # queued; its ordinary installed child can start and serve fresh work.
        pool["module"] = "mojolearn._native_process"
        assert checked("map", (pool, ["recovered"])) == ["recovered"]
        assert pool["workers"][0]["identity"]["vendor"] == _backend.vendor()
    finally:
        checked("close", (pool,))


def test_real_children_preserve_order_bits_and_are_reaped(tmp_path, monkeypatch):
    # The independent child controller audits its COMPLETE native worker
    # entry. Parent tracer silence cannot establish child compliance.
    (tmp_path / 'native_pool_audit_child.py').write_text('''
import json, os, sys
from pathlib import Path
from mojolearn._native_process import _native_process
from mojolearn._verify_native_runtime import audit_native_call
audit = audit_native_call(_native_process, 'worker_echo', sys.argv)
report = audit.report()
report['exception'] = None if audit.exception is None else repr(audit.exception)
Path(os.environ['NATIVE_POOL_AUDIT_DIRECTORY'], str(os.getpid()) + '.json').write_text(json.dumps(report, indent=2))
audit.assert_native()
''')
    monkeypatch.setenv('NATIVE_POOL_AUDIT_DIRECTORY', str(tmp_path))
    monkeypatch.setenv('PYTHONPATH', str(tmp_path) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    devices = (0, 1) if _backend.vendor() == "cpu" else (0,)
    pool = checked("create", (devices, False, "native_pool_audit_child"))
    arrays = [np.arange(12, dtype=np.float32).reshape(3, 4) + i for i in range(5)]
    arrays[2].view(np.uint32)[0, :3] = [0x80000000, 0x7FC01234, 0x3F800001]
    requests = [(i, np.asfortranarray(a), b"\x00\xff" + bytes([i])) for i, a in enumerate(arrays)]
    pids = []
    try:
        results = checked("map", (pool, requests))
        assert len(results) == len(requests)
        for i, (ordinal, result, payload) in enumerate(results):
            assert ordinal == i
            assert result.shape == (3, 4)
            np.testing.assert_array_equal(np.asarray(result).view(np.uint32), arrays[i].view(np.uint32))
            assert payload == b"\x00\xff" + bytes([i])
        pids = [worker["pid"] for worker in pool["workers"]]
        assert len(set(pids)) == len(devices)
        assert all(pid != os.getpid() for pid in pids)
        assert [worker["identity"]["pid"] for worker in pool["workers"]] == pids
        # A later wave reuses the same physical processes and continues the
        # response sequence; this checks lifecycle, not just one encode/decode.
        assert checked("map", (pool, ["last", 17])) == ["last", 17]
        assert [worker["pid"] for worker in pool["workers"]] == pids
    finally:
        checked("close", (pool,))
    assert pool["workers"] == []
    for pid in pids:
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)
        child_report = tmp_path / (str(pid) + '.json')
        assert child_report.exists(), 'child did not finish its whole-operation audit: ' + str(child_report)
        report = json.loads(child_report.read_text())
        assert report['exception'] is None, report
        for key in ('python_frames', 'forbidden_native_calls', 'wrapper_work',
                    'unobserved_existing_threads'):
            assert not report[key], report
    with pytest.raises(ValueError, match="closed"):
        call("map", (pool, [1]))


def test_invalid_devices_refused_before_spawn():
    for devices in [(), (0, 0), (-1,), (True,), (0.0,)]:
        with pytest.raises(ValueError):
            call("create", (devices, False, "mojolearn._native_process"))


def test_cpu_cooperative_multidevice_refusal_has_no_children():
    if _backend.vendor() != "cpu":
        pytest.skip("host-only cooperative refusal")
    pool = checked("create", ((0, 1), True, "mojolearn._native_process"))
    try:
        with pytest.raises(NotImplementedError, match="multiple CPU devices"):
            call("map", (pool, [1]))
        assert pool["workers"] == []
    finally:
        checked("close", (pool,))
