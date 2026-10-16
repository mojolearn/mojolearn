# SPDX-License-Identifier: Apache-2.0
"""Bounded workers for the installed identity verifier.

`run` executes one request in a fresh process. `run_batch` executes a batch of
cell requests in ONE process, one after another, and hands each result back as
it lands, so a sweep pays interpreter, import and GPU-runtime startup once per
batch instead of once per cell (on ROCm that startup is ~5 s per process). On a
crash, timeout or unhealthy-device signal inside a batch, `run_batch` reports
the first cell without an accepted result; the controller then runs that cell
and the rest of the batch each in a fresh process, so one crash cannot taint
another cell. Requests name the same harness and reference files and preserve
every fixture and probe setting. The private transport is a file per result:
native-library stdout cannot corrupt it.
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


class WorkerCrash(WorkerFailure):
    """The worker process died (non-zero exit or signal) without a timeout
    or an unhealthy-device signal. Its cell did not run, but the next cell's
    fresh process starts from a clean native context, so the controller can
    record this cell as refused and continue."""


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
    label = _label(request)
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
                tail = _tail(output)
                if unhealthy(tail):
                    raise WorkerFailure(f"{label}: unhealthy-device signal; {tail[-2000:]}")
                if process.returncode:
                    raise WorkerCrash(f"{label}: worker exited {process.returncode}; {tail[-2000:]}")
                return _accept(label, result_path, tail)
            finally:
                # Also reap a driver's descendants after normal parent exit.
                _kill(process)


def _tail(stream, size=8192):
    stream.flush()
    stream.seek(0, os.SEEK_END)
    stream.seek(max(0, stream.tell() - size))
    return stream.read().decode("utf-8", "replace")


def _file_tail(path, size=8192):
    try:
        with open(path, "rb") as stream:
            return _tail(stream, size)
    except OSError:
        return ""


def _accept(label, result_path, tail):
    """One worker result, or WorkerFailure: the same checks for a fresh
    process and for one cell of a batch."""
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


def _label(request):
    return "/".join(str(request[k]) for k in ("action", "lane", "fixture") if k in request)


class BatchStop:
    """Why a batch stopped before its last cell: `index` is the first request
    without an accepted result; `kind` is crash, timeout, unhealthy or failure."""

    def __init__(self, index, kind, message):
        self.index, self.kind, self.message = index, kind, message

    def as_dict(self):
        return dict(index=self.index, kind=self.kind, error=self.message[:500])


def run_batch(requests, timeout=120, log=None, on_result=None, poll=0.02):
    """Run `requests` in order in ONE worker process.

    `on_result(index, result)` is called as each cell's result is accepted
    (the same checks as `run`). `timeout` bounds each cell: the clock starts
    at process launch for the first cell, then at the previous cell's result.
    Returns None when every request finished, else a BatchStop naming the
    first request without a result; the worker is killed before returning.
    """
    if not 0 < timeout < float("inf"):
        raise ValueError("worker timeout must be finite and positive")
    requests = list(requests)
    if not requests:
        return None
    with tempfile.TemporaryDirectory(prefix="mojolearn-verify-batch-") as directory:
        request_path = os.path.join(directory, "requests.json")
        atomic_json(request_path, dict(batch=requests))
        results = os.path.join(directory, "results")
        os.mkdir(results)
        env = dict(os.environ)
        package_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = package_parent + os.pathsep + env.get("PYTHONPATH", "")
        env["MOJOLEARN_NUMERIC_MODE"] = "identical"
        index, started = 0, time.monotonic()
        heartbeat = started
        with open(os.path.join(directory, "worker.log"), "w+b") as output:
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", "mojolearn._verify_worker", "--batch", request_path, results],
                    env=env, stdout=output, stderr=output, start_new_session=os.name == "posix")
            except OSError as exc:
                return BatchStop(0, "crash", f"{_label(requests[0])}: could not launch batch worker: {exc}")
            try:
                while index < len(requests):
                    label = _label(requests[index])
                    result_path = os.path.join(results, f"result-{index}.json")
                    cell_log = os.path.join(results, f"cell-{index}.log")
                    # Read the exit status BEFORE looking for the file: a
                    # worker that wrote its last result and exited is done,
                    # not crashed.
                    code = process.poll()
                    if os.path.exists(result_path):
                        tail = _file_tail(cell_log)
                        if unhealthy(tail):
                            return BatchStop(index, "unhealthy", f"{label}: unhealthy-device signal; {tail[-2000:]}")
                        try:
                            result = _accept(label, result_path, tail)
                        except WorkerFailure as exc:
                            kind = "unhealthy" if "unhealthy-device" in str(exc) else "failure"
                            return BatchStop(index, kind, str(exc))
                        if on_result:
                            on_result(index, result)
                        index += 1
                        started = heartbeat = time.monotonic()
                        continue
                    if code is not None:
                        tail = _file_tail(cell_log) + _tail(output)
                        if unhealthy(tail):
                            return BatchStop(index, "unhealthy", f"{label}: unhealthy-device signal; {tail[-2000:]}")
                        if code:
                            return BatchStop(index, "crash", f"{label}: batch worker exited {code}; {tail[-2000:]}")
                        return BatchStop(index, "failure", f"{label}: batch worker exited without a result; {tail[-1000:]}")
                    now = time.monotonic()
                    if now - started > timeout:
                        return BatchStop(index, "timeout", f"{label}: exceeded {timeout:g}s worker limit")
                    if log and now - heartbeat >= 10:
                        heartbeat = now
                        log(f"  {label}: running {now - started:.0f}s / {timeout:g}s limit")
                    time.sleep(poll)
                return None
            finally:
                _kill(process)


# One harness module per process: a batched worker loads it once.
_HARNESSES = {}


def execute(request):
    from . import _verify_all as verifier
    from . import _verify_reference as reference
    from . import _backend
    from ._cpu_reference import reference_training
    import mojolearn as ml
    if _backend.numeric_mode() != "identical":
        raise WorkerFailure("worker did not load identical numeric mode")
    path = request["harness"]
    if path not in _HARNESSES:
        _HARNESSES[path] = verifier.load_harness(path)
    harness = _HARNESSES[path]
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


def _redirect(path):
    """Send this process's stdout/stderr (native writes included) to `path`,
    so each batched cell's diagnostics are judged on their own."""
    sys.stdout.flush()
    sys.stderr.flush()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    finally:
        os.close(fd)


def batch_main(request_path, results):
    with open(request_path, encoding="utf-8") as stream:
        requests = json.load(stream)["batch"]
    for index, request in enumerate(requests):
        _redirect(os.path.join(results, f"cell-{index}.log"))
        try:
            result = execute(request)
        except Exception as exc:
            result = dict(error=f"{type(exc).__name__}: {exc}")
        sys.stdout.flush()
        sys.stderr.flush()
        atomic_json(os.path.join(results, f"result-{index}.json"), result)
        if result.get("error"):
            # The controller runs this cell and the rest in fresh processes.
            break


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--batch":
        batch_main(sys.argv[2], sys.argv[3])
        return
    try:
        with open(sys.argv[1], encoding="utf-8") as stream:
            result = execute(json.load(stream))
    except Exception as exc:
        result = dict(error=f"{type(exc).__name__}: {exc}")
    atomic_json(sys.argv[2], result)


if __name__ == "__main__":
    main()
