"""Resolve retained evidence without rewriting historical receipt JSON."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
from pathlib import Path

_STATE = ContextVar("release_evidence", default=None)


@contextmanager
def evidence_context(*, objects=None, mapping=None, collect=None):
    token = _STATE.set((Path(objects) if objects is not None else None, mapping, collect))
    try:
        yield
    finally:
        _STATE.reset(token)


def _resolved(path):
    original = str(Path(path).resolve())
    state = _STATE.get()
    actual, expected = Path(path), None
    if state is not None and state[1] is not None:
        expected = state[1].get(original)
        if expected is None:
            raise FileNotFoundError("retained evidence is absent from admission bundle: " + original)
        actual = state[0] / expected
    return original, actual, expected, state


def _checked(original, expected, state, found):
    if expected is not None and found != expected:
        raise ValueError("retained evidence object digest differs: " + original)
    if state is not None and state[2] is not None:
        previous = state[2].setdefault(original, found)
        if previous != found:
            raise ValueError("retained evidence changed during admission: " + original)


def read_bytes(path):
    original, actual, expected, state = _resolved(path)
    data = actual.read_bytes()
    _checked(original, expected, state, hashlib.sha256(data).hexdigest())
    return data


def read_text(path):
    return read_bytes(path).decode("utf-8")


#: (actual path, size, mtime_ns) -> sha256. A release step hashes the same retained evidence files tens of
#: thousands of times (0.8.64 --status: 106,791 file hashes, 120 s); the bytes are keyed by their stat, so a
#: rewritten file is hashed again. The admission-bundle bookkeeping (_checked) still runs on every call.
_DIGESTS = {}


def sha256(path):
    original, actual, expected, state = _resolved(path)
    try:
        st = actual.stat()
        key = (str(actual), st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    except OSError:
        key = None
    found = _DIGESTS.get(key) if key is not None else None
    if found is None:
        with actual.open("rb") as stream:
            found = hashlib.file_digest(stream, "sha256").hexdigest()
        if key is not None:
            _DIGESTS[key] = found
    _checked(original, expected, state, found)
    return found
