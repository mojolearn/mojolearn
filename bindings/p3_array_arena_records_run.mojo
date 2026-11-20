# SPDX-License-Identifier: Apache-2.0
"""RUN of the Array / arena process records (bindings/p3_array_arena_records.mojo):
the typed record structs held in the shared slots (class addresses and dictionary
epochs as plain words). No Python import (tools/python_free_dirs.sh). Lane
rs-shared-infra-b (2026-11-12): moved word for word."""


struct P3NumpyClasses(Defaultable, Movable):
    var factory: Int
    # Exact NumPy real scalar types an object-dtype element may have.
    var scalars: List[Int]
    # dtype classes `_prepare` accepts before any dtype access.
    var dtypes: List[Int]

    def __init__(out self):
        self.factory = 0
        self.scalars = List[Int]()
        self.dtypes = List[Int]()


struct P3DictRecords(Defaultable, Movable):
    var epochs: Dict[Int, Int]

    def __init__(out self):
        self.epochs = Dict[Int, Int]()
