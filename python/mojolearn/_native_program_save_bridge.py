# SPDX-License-Identifier: Apache-2.0
"""Import-time owner layouts and provenance for native Program serialization."""


def _layout(cls):
    # glue: fixed storage and descriptor identities at package import.
    mro = type.__dict__["__mro__"].__get__(cls)
    rows = tuple(type.__dict__["__dict__"].__get__(base) for base in mro)  # glue: connected class ancestry
    descriptor = next(row["__dict__"] for row in rows if "__dict__" in row)  # glue: native dictionary storage
    names = ("__getattribute__", "__getattr__", "__setattr__", "__delattr__", "__del__", "__dict__", "_s", "_id", "inputs", "outputs")
    hooks = tuple((base, name, row.get(name)) for base, row in zip(mro, rows) for name in names)  # glue: fixed native storage identities
    return dict(cls=cls, mro=mro, fields_get=descriptor.__get__, hooks=hooks)


def connect_program(cls):
    # glue: install metadata belongs to setup; product save reads exact strings.
    import os
    from ._version import __version__
    from ._native_family_bridge import shared_context
    system = os.uname()
    shared_context()['program_save_context'] = dict(
        layouts=(_layout(cls),), version=__version__,
        platform=system.sysname + ' ' + system.machine)


def connect_imported(cls):
    # glue: importing the optional artifact frontend adds its exact owner class.
    from ._native_family_bridge import shared_context
    context = shared_context()['program_save_context']
    context['layouts'] += (_layout(cls),)
