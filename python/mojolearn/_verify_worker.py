# SPDX-License-Identifier: Apache-2.0
"""Bounded, disposable workers for the installed identity verifier.

The controller never reuses a worker's native/GPU context. Requests name the
same harness and reference files and preserve every fixture and probe setting.
The private transport is a file: native-library stdout cannot corrupt it.
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time


class WorkerFailure(RuntimeError):
    pass


def unhealthy(value):
    text = json.dumps(value, default=str).lower()
    return any(marker in text for marker in (
        "dead/saturated", "device lost", "device_lost", "device is unhealthy",
        "unhealthy device", "gpu context is unhealthy", "command buffer failed",
        "failed to create metal command queue", "deviation 2002",
        "gpu context is not trustworthy",
    ))


def atomic_json(path, value):
    """Replace one checkpoint without exposing a partially written document."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(prefix=".verify-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _kill(process):
    # A lane may itself spawn driver children. Killing only its Python parent
    # would leave those children occupying the device after the timeout.
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    process.wait()


def run(request, timeout=120, log=None):
    if not 0 < timeout < float("inf"):
        raise ValueError("worker timeout must be finite and positive")
    label = "/".join(str(request[k]) for k in ("action", "lane", "fixture") if k in request)
    with tempfile.TemporaryDirectory(prefix="mojolearn-verify-") as directory:
        request_path = os.path.join(directory, "request.json")
        result_path = os.path.join(directory, "result.json")
        atomic_json(request_path, request)
        env = dict(os.environ)
        # Match the actual package imported by the controller, including an
        # editable checkout launched without an inherited PYTHONPATH.
        package_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = package_parent + os.pathsep + env.get("PYTHONPATH", "")
        env["MOJOLEARN_NUMERIC_MODE"] = "identical"
        started = time.monotonic()
        with open(os.path.join(directory, "worker.log"), "w+b") as output:
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", "mojolearn._verify_worker", request_path, result_path],
                    env=env, stdout=output, stderr=output, start_new_session=os.name == "posix")
            except OSError as exc:
                raise WorkerFailure(f"{label}: could not launch worker: {exc}") from exc
            try:
                while True:
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        raise WorkerFailure(f"{label}: exceeded {timeout:g}s worker limit")
                    try:
                        process.wait(timeout=min(10, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        if log:
                            log(f"  {label}: running {time.monotonic() - started:.0f}s / {timeout:g}s limit")
                output.seek(0, os.SEEK_END)
                output.seek(max(0, output.tell() - 8192))
                tail = output.read().decode("utf-8", "replace")
                if process.returncode:
                    raise WorkerFailure(f"{label}: worker exited {process.returncode}; {tail[-2000:]}")
                if unhealthy(tail):
                    raise WorkerFailure(f"{label}: unhealthy-device signal; {tail[-2000:]}")
                try:
                    with open(result_path, encoding="utf-8") as stream:
                        result = json.load(stream)
                except (OSError, ValueError) as exc:
                    raise WorkerFailure(f"{label}: missing or invalid worker result: {exc}; {tail[-1000:]}") from exc
                if not isinstance(result, dict):
                    raise WorkerFailure(f"{label}: invalid worker result: expected object")
                if result.get("error"):
                    raise WorkerFailure(f"{label}: {result['error']}")
                if "value" not in result or not isinstance(result.get("bindings"), list):
                    raise WorkerFailure(f"{label}: invalid worker result: missing value/bindings")
                if unhealthy(result):
                    raise WorkerFailure(f"{label}: unhealthy-device signal in result: {str(result)[:2000]}")
                return result
            finally:
                # Also reap a driver's descendants after normal parent exit.
                _kill(process)


def execute(request):
    from . import _verify_all as verifier
    from . import _verify_reference as reference
    from . import _backend
    from ._cpu_reference import reference_training
    import mojolearn as ml
    if _backend.numeric_mode() != "identical":
        raise WorkerFailure("worker did not load identical numeric mode")
    harness = verifier.load_harness(request["harness"])
    action = request["action"]
    with reference_training():
        if action == "cell":
            fixture = request["fixture"]
            data, held = harness.fixture(fixture), harness.heldout(fixture)
            actual = dict(X=harness._h(data[0]), y_clf=harness._h(data[1]), y_reg=harness._h(data[2]),
                          held=harness._h(held))
            if actual != request["fixture_hashes"]:
                raise WorkerFailure("worker fixture bytes differ from the controller")
            value = verifier.run_cell(harness, ml, request["lane"], fixture, data, held,
                                      request["repeats"], extra_parts=request["extra_parts"])
        elif action == "self-test":
            value = verifier.self_test(harness, ml, reference.load_table(request["table"]))
        elif action == "models":
            value = verifier.run_models(harness, ml, reference.load_table(request["table"]),
                                        repeats=request["repeats"], host_only=request["host_only"])
        elif action == "smoke":
            fixture = request["fixture"]
            value = verifier.run_smoke(harness, ml, [request["lane"]], fixture,
                                       harness.fixture(fixture), harness.heldout(fixture))
        else:
            raise WorkerFailure(f"unknown worker action {action!r}")
    from . import _verify
    bindings = [dict(module=b["module"], sha256=b["sha256"], size=b["size"])
                for b in _verify.binding_artifacts()]
    # Do not hash every installed host family in every disposable process.
    # Only bindings actually loaded by this worker contribute provenance.
    import hashlib
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", "") or ""
        if name.startswith("mojolearn._host.") and path.endswith((".so", ".pyd")):
            with open(path, "rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()
            bindings.append(dict(module=name, sha256=digest, size=os.path.getsize(path), path=path, kind="host"))
    return dict(value=value, bindings=bindings)


def main():
    try:
        with open(sys.argv[1], encoding="utf-8") as stream:
            result = execute(json.load(stream))
    except Exception as exc:
        result = dict(error=f"{type(exc).__name__}: {exc}")
    atomic_json(sys.argv[2], result)


if __name__ == "__main__":
    main()
