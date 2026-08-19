"""A half-open range of objects inside a buffer.

PORT OF `catboost/cuda/cuda_lib/slice.h` at CatBoost `54a8143a`.
Transliterated. Do not improve.

`TSlice` is how every read in their level loop names the part it wants.
`ParallelStripeView(subsets->PartitionsCpu, TSlice(newId, newId + 1))`
(`split_properties_helper.cpp:821`) is a one-leaf slice, and it is the reason
`FastUpdateLeavesSizes` reads one partition instead of all of them.
"""


struct TSlice(Copyable, ImplicitlyCopyable, Movable):
    """Their `TSlice` (`slice.h`). Half open, `[Left, Right)`."""

    var left: Int
    var right: Int

    def __init__(out self):
        self.left = 0
        self.right = 0

    def __init__(out self, left: Int, right: Int):
        self.left = left
        self.right = right

    def size(self) -> Int:
        """Their `Size()`."""
        return self.right - self.left

    def is_empty(self) -> Bool:
        """Their `IsEmpty()`."""
        return self.size() == 0

    def __eq__(self, other: Self) -> Bool:
        return self.left == other.left and self.right == other.right

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def __add__(self, shift: Int) -> Self:
        """Their `operator+=`, which is how they walk stripes."""
        return Self(self.left + shift, self.right + shift)

    def to_string(self) -> String:
        return String("[") + String(self.left) + ", " + String(self.right) + ")"
