# SPDX-License-Identifier: Apache-2.0
# Typed program inputs and host results for the autograd kernel entries.
# The Python doors build a ProgramInput from an array or a nested sequence
# (program_pyapi.mojo program_input) and turn a HostArray into an ndarray
# (program_arrays_pyapi.mojo host_array_py); the entries themselves take
# and return only these types.
from core.cast_in import CastIn
from autograd.rankn import ShapeN


struct ProgramInput(Movable):
    """One program input: its checked descriptor `src`, its shape, and the
    owners of a sequence input's converted values (`src` addresses them;
    empty for an array input)."""
    var src: CastIn
    var shape: ShapeN
    var values: List[Float64]
    var ids: List[Int64]

    def __init__(out self, src: CastIn, shape: ShapeN, var values: List[Float64], var ids: List[Int64]):
        self.src = src
        self.shape = shape
        self.values = values^
        self.ids = ids^


struct HostArray(Movable):
    """One float32 result downloaded to the host, row-major, with its shape."""
    var data: List[Float32]
    var shape: ShapeN

    def __init__(out self, shape: ShapeN):
        self.shape = shape
        self.data = List[Float32](length=shape.count(), fill=Float32(0))
