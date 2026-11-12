# SPDX-License-Identifier: Apache-2.0
"""Optimizer container UNPACK/PACK; native vectors pin admitted Python owners."""
from std.python import Python, PythonObject
from bindings.cpython_table import cpython_table, _PyBuffer_Release
from bindings.optimizer_admission import opt_value
from bindings.py_balanced import py_type



@fieldwise_init
struct OptimizerBufferLease(Movable):
    # The existing Py_buffer allocation owns its exporter reference until RUN
    # returns. Keeping only the array object would not prevent buffer resize.
    var words: List[Int]
    var release: _PyBuffer_Release.type

    def __del__(deinit self):
        self.release(self.words.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]())
        _ = self.words^


struct OptimizerInputOwners(Defaultable, Movable):
    var objects: List[PythonObject]
    var buffers: List[OptimizerBufferLease]

    def __init__(out self):
        self.objects = List[PythonObject]()
        self.buffers = List[OptimizerBufferLease]()

    def append(mut self, value: PythonObject):
        self.objects.append(value)
