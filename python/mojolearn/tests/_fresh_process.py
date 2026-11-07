"""Child interpreters for tests: one spawn API and one shared child per test process.

A child interpreter costs a fresh `import mojolearn` and binding load (about a
second on the development build), so the suite spawns as few as it can:

* `run_python(source, *argv, ...)`: one fresh interpreter for a test whose
  check IS the fresh process (cold first call, import order, a patched
  builtin, signal handlers, a killed worker). The source tree is put on
  PYTHONPATH unless the caller passes its own `env`.
* `plain_import_report()`: ONE child per test process that runs only
  `import mojolearn` and reports what that import loaded. Tests that check
  "a plain `import mojolearn` loads no X" read it instead of spawning.
* `SharedChild(prelude)` / `shared_child(prelude)`: ONE child per prelude per
  test process, fed many self-contained snippets in turn, each in a fresh
  namespace. Use it only for snippets that need neither a cold process nor
  leave process-wide state changed (no patched builtins or modules, no
  `sys.meta_path`, environment or signal changes beyond the prelude): their
  results do not depend on which snippets ran before them in the same child.

A failure in a shared child is reported as that case's failure with the
child's traceback; a child that dies is restarted for the next case.
"""
import atexit
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

#: The directory holding the `mojolearn` package of this checkout or install.
PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])


def child_env(**overrides):
    """os.environ with the package parent first on PYTHONPATH, plus overrides
    (a value of None removes that variable)."""
    env = dict(os.environ)
    path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = PACKAGE_PARENT + (os.pathsep + path if path else "")
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def run_python(source, *argv, env=None, cwd=None, timeout=300, input=None):
    """Run `source` in a fresh interpreter; returns the CompletedProcess (text)."""
    return subprocess.run([sys.executable, "-c", source, *map(str, argv)],
                          env=child_env() if env is None else env, cwd=cwd, input=input,
                          capture_output=True, text=True, timeout=timeout)


_PLAIN_IMPORT = r"""
import json, sys
before = set(sys.modules)
import mojolearn
print(json.dumps({"modules": sorted(sys.modules), "new": sorted(set(sys.modules) - before),
                  "names": sorted(dir(mojolearn)), "file": mojolearn.__file__}))
"""
_REPORT = None
_REPORT_LOCK = threading.Lock()


def plain_import_report():
    """What a plain `import mojolearn` loads in a fresh interpreter: dict with
    `modules` (all of sys.modules after the import), `new` (loaded by it),
    `names` (dir(mojolearn)) and `file`. Spawned once per test process."""
    global _REPORT
    with _REPORT_LOCK:
        if _REPORT is None:
            done = run_python(_PLAIN_IMPORT)
            assert done.returncode == 0, done.stderr[-4000:]
            _REPORT = json.loads(done.stdout.strip().splitlines()[-1])
        return _REPORT


# The shared child: reads one JSON request per line ({"code": ...}), runs it
# in a fresh namespace with its stdout captured, answers one JSON line.
_SERVER = r"""
import contextlib, io, json, sys, traceback
_out = sys.stdout
exec(compile(sys.argv[1], "<prelude>", "exec"), {"__name__": "__prelude__"})
for _line in sys.stdin:
    _request = json.loads(_line)
    _buffer = io.StringIO()
    _answer = {"ok": True, "stdout": "", "error": ""}
    try:
        with contextlib.redirect_stdout(_buffer):
            exec(compile(_request["code"], "<case>", "exec"), {"__name__": "__case__"})
    except BaseException:
        _answer["ok"] = False
        _answer["error"] = traceback.format_exc()
    _answer["stdout"] = _buffer.getvalue()
    _out.write(json.dumps(_answer) + "\n")
    _out.flush()
"""


class CaseFailed(AssertionError):
    pass


class SharedChild:
    """One long-lived interpreter that ran `prelude` once and then runs cases."""

    def __init__(self, prelude="", env=None, timeout=300):
        self.prelude, self.env, self.timeout = prelude, env, timeout
        self._process = None
        self._lock = threading.Lock()

    def _start(self):
        self._process = subprocess.Popen([sys.executable, "-c", _SERVER, self.prelude],
                                         env=child_env() if self.env is None else self.env,
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         text=True, bufsize=1)

    def run(self, code):
        """Run `code` in the child; returns its stdout. Raises CaseFailed with
        the child's traceback when the case raised or the child died."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._start()
            process = self._process
            reply = {}

            def exchange():
                process.stdin.write(json.dumps({"code": code}) + "\n")
                process.stdin.flush()
                reply["line"] = process.stdout.readline()

            worker = threading.Thread(target=exchange, daemon=True)
            worker.start()
            worker.join(self.timeout)
            if worker.is_alive() or not reply.get("line"):
                timed_out = worker.is_alive()
                process.kill()
                _, err = process.communicate()
                self._process = None
                raise CaseFailed(("shared child timed out" if timed_out else "shared child died")
                                 + f" (rc={process.returncode}):\n{(err or '')[-4000:]}")
            answer = json.loads(reply["line"])
        if not answer["ok"]:
            raise CaseFailed(answer["error"][-4000:])
        return answer["stdout"]

    def close(self):
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.stdin.close()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


_CHILDREN = {}
_CHILDREN_LOCK = threading.Lock()


def shared_child(prelude=""):
    """The one SharedChild for `prelude` in this test process."""
    with _CHILDREN_LOCK:
        child = _CHILDREN.get(prelude)
        if child is None:
            child = _CHILDREN[prelude] = SharedChild(prelude)
        return child


@atexit.register
def _close_children():
    for child in list(_CHILDREN.values()):
        child.close()


#: Prelude for NumPy-free checks: NumPy cannot be imported in the child even
#: when the test environment has it.
BLOCK_NUMPY = r"""
import sys
class _BlockNumpy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ModuleNotFoundError("No module named 'numpy'", name="numpy")
sys.meta_path.insert(0, _BlockNumpy())
"""

#: BLOCK_NUMPY, then `import mojolearn` once: a package that cannot import
#: without NumPy fails every case of this child, with the traceback.
BLOCK_NUMPY_IMPORTED = BLOCK_NUMPY + "import mojolearn\n"
