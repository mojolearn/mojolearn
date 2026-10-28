# SPDX-License-Identifier: Apache-2.0
"""Import-time class storage connections for the shared native family door."""
import sys
from types import GetSetDescriptorType, ModuleType

LAYOUTS = {}


def _descriptor(value):
    # Raw native class metadata; do not resolve a descriptor's own __get__.
    return any('__get__' in type.__dict__['__dict__'].__get__(base)
               for base in type.__dict__['__mro__'].__get__(type(value)))  # glue: import-time descriptor metadata


def connect(package):
    """Retain final built-in layouts once; later imports only add new classes."""
    classes = set()
    prefix = package.__name__ + '.'
    for name, module in tuple(sys.modules.items()):  # glue: loaded package declarations only
        if type(module) is not ModuleType or not (name == package.__name__ or name.startswith(prefix)):
            continue
        if '.tests' in name:
            continue
        for value in vars(module).values():  # glue: exact class identities, never instances
            if type(value) is type:
                owner = vars(value).get('__module__')
                if type(owner) is str and (owner == package.__name__ or owner.startswith(prefix)):
                    classes.add(value)
    for cls in classes:  # glue: final class/storage identities after native installers
        if cls in LAYOUTS:
            continue
        mro = type.__dict__['__mro__'].__get__(cls)
        if any(type(base) is not type for base in mro):  # glue: import-time native class ancestry
            continue
        rows = tuple((base, dict(type.__dict__['__dict__'].__get__(base))) for base in mro)  # glue: import-time class metadata
        fields = next((row['__dict__'] for _, row in rows if '__dict__' in row), None)  # glue: native instance storage
        if type(fields) is not GetSetDescriptorType or fields.__name__ != '__dict__':
            continue
        names = {}
        for _, row in reversed(rows):  # glue: effective inherited member identities
            names.update(row)
        LAYOUTS[cls] = dict(cls=cls, mro=mro, fields=fields, fields_get=fields.__get__, names=names,
                           descriptors=tuple((base, {name: value for name, value in row.items() if _descriptor(value)})  # glue: import-time descriptor identities
                                             for base, row in rows))  # glue: immutable descriptor identities


from ._native_loaded_modules import connect_module  # noqa: E402

# glue: import-time native module connection; LAYOUTS is filled in place, never reassigned.
connect_module(__name__, sys.modules[__name__], ("LAYOUTS",))
