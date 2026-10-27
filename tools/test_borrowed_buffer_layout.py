"""Native buffer lifetime/copy source contracts, no runtime qualification."""
from pathlib import Path
from check_generation_native_boundary import audit
ROOT=Path(__file__).resolve().parents[1]


def test_native_buffer_layout_has_no_python_dependency_or_numeric_conversion():
    _, failures=audit(ROOT,('core.borrowed_buffer',))
    assert not failures, failures
    source=(ROOT/'core/borrowed_buffer.mojo').read_text()
    assert 'Float32' not in source and 'Float64' not in source
    assert 'self.suboffsets[axis] >= 0' in source
    assert 'self.readonly' in source and 'self.count == 0' in source
    assert 'self.strides[axis]' in source
    assert 'memcpy(' in source


def test_export_is_retained_and_released_on_error_and_destruction():
    source=(ROOT/'providers/python/borrowed_buffer_pyapi.mojo').read_text()
    assert source.count('"PyObject_GetBuffer"') == 1
    assert 'self.owner = admitted_exporter' in source
    assert 'self.acquired = True' in source
    assert 'except problem:\n            self.release()' in source
    assert 'def __deinit__' in source and source.count('"PyBuffer_Release"') == 2
    assert 'already-admitted' in source.lower()
