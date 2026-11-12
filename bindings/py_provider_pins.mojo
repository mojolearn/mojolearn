# SPDX-License-Identifier: Apache-2.0
"""Door-side pins for connected native provider images (UNPACK only).

A typed connection record (driver, compiled mode, vendor, capabilities) lives
in a RUN module and names its provider by address word. The provider object
itself is pinned here, in the door, for the process lifetime, at the same
index as its record: the door appends the pin and the record together, under
the GIL, so the two lists stay aligned. A pinned provider stays alive, so its
address word cannot be reused by another provider while it is listed.
"""
from std.python import PythonObject


struct ProviderPins(Defaultable, Movable):
    var providers: List[PythonObject]

    def __init__(out self):
        self.providers = List[PythonObject]()


def provider_word(provider: PythonObject) -> Int:
    # id(provider), natively: the key of the provider's typed RUN record.
    return Int(provider._obj_ptr)
