# SPDX-License-Identifier: Apache-2.0
# The typed absent value at the UNPACK/PACK edge. A Python result of None
# becomes an absent Optional, never a PythonObject(None) sentinel carried
# through a call (Mojo 1.2.0 mis-lowers a loop-carried PythonObject sentinel,
# modular/modular#7546; the Optional form stays after the fix: it is the typed
# form of an absent value, not a workaround).
from std.python import PythonObject


@always_inline
def py_present(value: PythonObject) -> Optional[PythonObject]:
    """UNPACK: `value` when it is not Python's None, otherwise absent."""
    if value is None:
        return Optional[PythonObject]()
    return Optional[PythonObject](value)


@always_inline
def py_or_none(value: Optional[PythonObject]) -> PythonObject:
    """PACK: the held object, or Python's None when absent."""
    if value:
        return value.value()
    return PythonObject(None)
