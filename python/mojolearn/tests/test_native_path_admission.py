"""Public checkpoint path admission; run against the rebuilt installed runtime."""
import pathlib
import numpy as np
import pytest
from mojolearn import Sequential
from mojolearn._verify_native_runtime import audit_native_call


@pytest.fixture
def fitted():
    x = np.array([[0, 1], [1, 0], [1, 1]], dtype='f4')
    y = np.array([0, 1, 1], dtype='f4')
    return Sequential().linear(1).compile(loss='mse', optimizer='adamw', lr=0.01).fit(x, y, epochs=1, seed=1)


@pytest.mark.parametrize('kind', ['str', 'bytes', 'PosixPath', 'PurePosixPath'])
def test_checkpoint_exact_standard_path_matches_string_bytes(fitted, tmp_path, kind):
    reference = tmp_path / 'reference.ckpt'
    fitted.save_checkpoint(str(reference))
    target = tmp_path / f'{kind}.ckpt'
    value = {'str': str(target), 'bytes': bytes(target), 'PosixPath': target,
             'PurePosixPath': pathlib.PurePosixPath(target)}[kind]
    audit_native_call(fitted.save_checkpoint, value).assert_native()
    assert target.read_bytes() == reference.read_bytes()


@pytest.mark.parametrize('kind', ['subclass', 'protocol', 'descriptor'])
def test_checkpoint_refuses_path_callbacks_without_invocation(fitted, tmp_path, monkeypatch, kind):
    called = []
    class ForeignPath(pathlib.PosixPath):
        def __fspath__(self):
            called.append('fspath')
            return str(tmp_path / 'bad.ckpt')
    class Protocol:
        def __fspath__(self):
            called.append('protocol')
            return str(tmp_path / 'bad.ckpt')
    value = ForeignPath(tmp_path / 'bad.ckpt') if kind == 'subclass' else Protocol()
    if kind == 'descriptor':
        value = tmp_path / 'bad.ckpt'
        field = next(name for name in ('_raw_paths', '_parts') if any(name in vars(base) for base in type(value).__mro__))
        owner = next(base for base in type(value).__mro__ if field in vars(base))
        def read(self):
            called.append('descriptor')
            raise AssertionError('path descriptor invoked')
        monkeypatch.setattr(owner, field, property(read))
    if kind == "descriptor":
        try:
            with pytest.raises(TypeError):
                fitted.save_checkpoint(value)
        finally:
            monkeypatch.undo()  # restore pathlib before pytest renders diagnostics
    else:
        audit = audit_native_call(fitted.save_checkpoint, value)
        assert isinstance(audit.exception, TypeError), audit.report()
    assert not called
