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


def sha256(path):
    original, actual, expected, state = _resolved(path)
    with actual.open("rb") as stream:
        found = hashlib.file_digest(stream, "sha256").hexdigest()
    _checked(original, expected, state, found)
    return found
