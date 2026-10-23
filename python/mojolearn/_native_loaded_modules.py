# SPDX-License-Identifier: Apache-2.0
"""Import-time canonical module and native-export connections."""
import array
import builtins
import itertools
import math
import operator
import os
import sys
import types
import warnings

CONNECTIONS = ()


def connect_module(name, module, export_names):
    # glue: fixed modules register during package setup, before any operation;
    # native lookups consume the immutable records and never call this helper.
    global CONNECTIONS
    if type(name) is not str or type(module) is not types.ModuleType:
        raise TypeError('native module setup requires a name and an exact module')
    if type(export_names) is not tuple or any(type(key) is not str for key in export_names):  # glue: validate fixed export metadata at import
        raise TypeError('native module exports require an exact string tuple')
    if any(record[0] == name for record in CONNECTIONS):  # glue: reject duplicate import-time ownership
        raise TypeError('native module is already connected: ' + name)
    fields = types.ModuleType.__dict__['__dict__'].__get__(module)
    if '__getattr__' in fields:
        raise TypeError('native module setup cannot retain module attribute hooks')
    exports = tuple((key, fields[key]) for key in export_names)  # glue: retain exact export identities at import
    CONNECTIONS += ((name, module, exports),)


# glue: snapshot only exports consumed by native helpers; mutable process data
# such as sys.stdout and the application's __import__ hook are not called or
# restricted by these connections. Changes to a relevant export refuse.
connect_module('builtins', builtins, tuple(name for name in vars(builtins)  # glue: fixed builtin native APIs and exception classes
                                          if not name.startswith('__')))
connect_module('sys', sys, ('modules',))
connect_module('array', array, ('array',))
connect_module('operator', operator, tuple(name for name in vars(operator)  # glue: fixed scalar/index operator exports
                                          if not name.startswith('_')))
connect_module('math', math, tuple(name for name in vars(math) if not name.startswith('_')))  # glue: fixed native scalar math exports
connect_module('itertools', itertools, tuple(name for name in vars(itertools) if not name.startswith('_')))  # glue: fixed native iterator types
connect_module('os', os, ('stat', 'urandom', 'getcwd'))
connect_module('types', types, ('ModuleType', 'MemberDescriptorType', 'MethodDescriptorType'))
connect_module('warnings', warnings, ())  # warning filters are mutable native configuration, validated by their owner
