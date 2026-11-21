# SPDX-License-Identifier: Apache-2.0
"""RUN of the parameter grid (device wave, lane dw-msel-parallel,
2026-11-21): the shape of a prepared grid (how many choices each parameter
of each dictionary has) and the arithmetic on it: the candidate count, the
choice positions of one candidate by mixed-radix digits, and the odometer
step that enumerates them, rightmost parameter fastest (itertools.product's
order).

The parameter names and the choice values are the caller's Python objects
and stay with the door (bindings/msel_parameter_grid_door.mojo), which
builds each candidate dictionary from the positions computed here.

No Python object is read, built or called here.
"""


struct ParameterGridShape(Movable):
    """widths[i][j]: the number of choices of parameter j of dictionary i."""

    var widths: List[List[Int]]

    def __init__(out self):
        self.widths = List[List[Int]]()

    def add(mut self, var widths: List[Int]):
        self.widths.append(widths^)

    def grids(self) -> Int:
        return len(self.widths)

    def rank(self, i: Int) -> Int:
        return len(self.widths[i])

    def size(self, i: Int) -> Int:
        """The number of candidates of dictionary i."""
        var count = 1
        for j in range(len(self.widths[i])):  # driver-loop: dimensions of one grid
            count *= self.widths[i][j]
        return count

    def count(self) -> Int:
        var total = 0
        for i in range(len(self.widths)):  # driver-loop: native cardinalities; no value rematerialization
            total += self.size(i)
        return total

    def locate(self, index_in: Int, mut positions: List[Int]) raises -> Int:
        """The dictionary of the candidate at a (negative-normalized) index
        of the whole enumeration, and in `positions` its choice of each
        parameter (mixed-radix digits, rightmost parameter fastest)."""
        var total = self.count()
        var index = index_in + total if index_in < 0 else index_in
        if index < 0 or index >= total:
            raise Error("IndexError: list index out of range")
        for i in range(len(self.widths)):  # driver-loop: outer grids, in emission order
            var size = self.size(i)
            if index >= size:
                index -= size
                continue
            var rank = len(self.widths[i])
            positions = List[Int](length=rank, fill=0)
            var axis = rank - 1
            while axis >= 0:  # driver-loop: mixed-radix digits of one candidate
                positions[axis] = index % self.widths[i][axis]
                index //= self.widths[i][axis]
                axis -= 1
            return i
        raise Error("IndexError: list index out of range")


@always_inline
def parameter_grid_advance(widths: List[Int], mut positions: List[Int]) -> Bool:
    """The odometer step to the next candidate of one dictionary: the
    rightmost parameter changes fastest. False once the last candidate has
    been passed (every position is back at 0). No multiplied product size
    can overflow an enumeration bound."""
    var axis = len(widths) - 1
    while axis >= 0:  # driver-loop: carry across native mixed-radix counters
        positions[axis] += 1
        if positions[axis] < widths[axis]:
            return True
        positions[axis] = 0
        axis -= 1
    return False
