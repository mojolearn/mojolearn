"""Run the real import-time enumeration without loading product binaries."""
import ast
import ctypes
from pathlib import Path
from types import ModuleType, SimpleNamespace
import warnings

import pytest


ROOT = Path(__file__).resolve().parents[1]


def missing_module_native_type():
    # CPython's public type-construction API reproduces the extension type
    # observed in the retained package-import failure, without a compiler.
    class Slot(ctypes.Structure):
        _fields_ = [('slot', ctypes.c_int), ('pfunc', ctypes.c_void_p)]

    class Spec(ctypes.Structure):
        _fields_ = [('name', ctypes.c_char_p), ('basicsize', ctypes.c_int),
                   ('itemsize', ctypes.c_int), ('flags', ctypes.c_uint),
                   ('slots', ctypes.POINTER(Slot))]

    slots = (Slot * 1)()
    spec = Spec(b'MissingModuleClass', object.__basicsize__, 0, 0, slots)
    make = ctypes.pythonapi.PyType_FromSpec
    make.argtypes = [ctypes.POINTER(Spec)]
    make.restype = ctypes.py_object
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        return make(ctypes.byref(spec))


def scan(source, package, modules):
    # Execute the exact declaration-enumeration loop from the real connector;
    # later descriptor installation needs compiled binaries and belongs to the
    # orchestrator's public import/runtime probe, not this isolated regression.
    tree = ast.parse((ROOT / 'python/mojolearn' / source).read_text())
    connector = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == 'connect_classes')
    loop = next(node for node in connector.body if isinstance(node, ast.For)
                and ast.unparse(node.iter) == 'tuple(sys.modules.items())')
    namespace = dict(package=package, sys=SimpleNamespace(modules=modules),
                     ModuleType=ModuleType, candidates=set(), classes=set())
    exec(compile(ast.Module(body=[loop], type_ignores=[]), source, 'exec'), namespace)
    return namespace['candidates'] | namespace['classes']


@pytest.mark.parametrize('source,include_root', [('_native_public_attributes.py', False),
                                               ('_native_params_metadata.py', True)])
def test_extension_metadata_is_skipped_and_owned_declarations_remain(source, include_root):
    package = ModuleType('mojolearn')
    declarations = ModuleType('mojolearn.fixture')
    native = missing_module_native_type()
    assert type(native) is type and '__module__' not in vars(native)
    with pytest.raises(AttributeError):
        _ = native.__module__

    def owner(name, module):
        return type(name, (), {'__module__': module})

    class Bomb:
        def __get__(self, *args):
            raise AssertionError('module descriptor invoked')
        def startswith(self, *args):
            raise AssertionError('module protocol invoked')

    class StringProtocol(str):
        def startswith(self, *args):
            raise AssertionError('string subclass protocol invoked')

    class Meta(type):
        def __getattribute__(self, name):
            raise AssertionError('metaclass callback invoked')

    owned = owner('Owned', 'mojolearn.fixture')
    root = owner('Root', 'mojolearn')
    declarations.Owned = owned
    declarations.Native = native
    declarations.NoneModule = owner('NoneModule', None)
    declarations.IntegerModule = owner('IntegerModule', 9)
    declarations.DescriptorModule = owner('DescriptorModule', Bomb())
    declarations.StringSubclass = owner('StringSubclass', StringProtocol('mojolearn.fixture'))
    declarations.External = owner('External', 'other.fixture')
    declarations.Test = owner('Test', 'mojolearn.tests.fixture')
    declarations.CustomMeta = Meta('CustomMeta', (), {'__module__': 'mojolearn.fixture'})
    package.Root = root
    package.Alias = owned
    tests = ModuleType('mojolearn.tests.fixture')
    tests.Hidden = owner('Hidden', 'mojolearn.fixture')
    modules = {module.__name__: module for module in (package, declarations, tests)}
    assert scan(source, package, modules) == {owned} | ({root} if include_root else set())
