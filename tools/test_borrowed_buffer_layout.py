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


def test_shared_exporter_admission_precedes_buffer_protocols():
    source=(ROOT/'providers/python/borrowed_buffer_pyapi.mojo').read_text()
    admission=source.split('def admit_native_buffer_exporter(')[1].split('struct AdmittedBorrowedBuffer')[0]
    assert 'admit_native_buffer_exporter(owner, depth + 1)' in admission
    assert 'PyType_GetFlags' in admission
    assert 'flags & UInt64(512)' in admission
    assert '"__buffer__"' in admission and '"__release_buffer__"' in admission
    assert 'PyObject_GetBuffer' not in admission
    assert 'py_call(bi.memoryview' not in admission
