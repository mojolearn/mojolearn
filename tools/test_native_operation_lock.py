"""Shared lock dependency/lifetime contracts, not a runtime concurrency pass."""
import ast
from pathlib import Path
from check_generation_native_boundary import audit
ROOT = Path(__file__).resolve().parents[1]
CORE = (ROOT/'core/operation_lock.mojo').read_text()
BOUNDARY = (ROOT/'providers/python/operation_lock_pyapi.mojo').read_text()


def test_native_lock_has_no_python_dependency_or_object_layout_cast():
    _, failures = audit(ROOT, ('core.operation_lock',))
    assert not failures, failures
    for name in ('PythonObject', 'PyThread_', 'PyObject_', '_thread.RLock'):
        assert name not in CORE
    assert 'Ordering.ACQUIRE' in CORE and 'Ordering.RELEASE' in CORE
    assert 'words[1] += 1' in CORE and 'words[1] -= 1' in CORE
    assert 'operation_lock_thread()' in CORE


def test_exact_owner_is_pinned_once_for_native_acquire_and_release():
    lease = BOUNDARY.split('struct OperationLockLease')[1].split('def operation_lock_c')[0]
    assert lease.count('operation_lock_owner(value)') == 1
    assert 'self.owner = operation_lock_owner(value)' in lease
    assert 'self.handle = self.owner.downcast_value_ptr[ProviderOperationLock]()' in lease
    methods = lease.split('def acquire(')[1]
    assert 'operation_lock_owner(' not in methods
    assert 'with GILReleased(Python()):' in methods
    assert 'operation_lock_release(self.handle)' in methods
    assert 'py_type(value) is not state[].legacy' in lease


def test_python_lock_frontend_only_installs_compiled_descriptors():
    tree = ast.parse((ROOT/'python/mojolearn/_native_operation_lock.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in cls.body)
    source = ast.unparse(tree)
    assert 'classical_attribute_descriptor' in source
    assert 'MethodType' in source
    for name in ('operation_lock_init','operation_lock_acquire','operation_lock_release','operation_lock_enter','operation_lock_exit'):
        assert name in source


def test_neural_layer_boundaries_use_lease_without_preacquiring_models():
    for name in ('bindings/neural_block_public.mojo','bindings/neural_prefill_native.mojo'):
        source = (ROOT/name).read_text()
        assert 'OperationLockLease(lock)' in source
        assert 'lock.acquire()' not in source and 'lock.release()' not in source
        assert 'lock_lease.acquire()' in source and 'lock_lease.release()' in source
    assert 'InterruptedError: native operation lock interrupted' in CORE
