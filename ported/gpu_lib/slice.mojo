"""A half-open range of objects inside a buffer.

PORT OF `catboost/cuda/cuda_lib/slice.h` at CatBoost `54a8143a`.
Transliterated. Do not improve.

`TSlice` is how every read in their level loop names the part it wants.
`ParallelStripeView(subsets->PartitionsCpu, TSlice(newId, newId + 1))`
(`split_properties_helper.cpp:821`) is a one-leaf slice, and it is the reason
`FastUpdateLeavesSizes` reads one partition instead of all of them.

`TSlice` is also the unit `mapping.mojo` is written in: every `TMapping`
answers questions about device ownership by returning slices of the object
range (`mapping.h:31-61`).

================================ DEVIATION BLOCK ======================
Theirs stores `Left`/`Right` as `ui64` (`slice.h:10-11`). Ours stores `Int`,
which is signed and 64-bit. This matters in exactly one place: their
`IsEmpty()` is `Left >= Right` (`slice.h:52`), which under `ui64` plus their
`Y_ASSERT(left <= right)` can only ever be `Left == Right`. Ours keeps the
`>=` spelling so an inverted slice built without going through the asserting
constructor still reads as empty rather than reporting a negative size.

`Y_ASSERT` (`slice.h:26`, `slice.h:48`) is a debug-build assert in their tree
and compiles out of a release build, so it is transcribed as a comment rather
than as a `raise`. `CB_ENSURE` (`slice.h:74`) is live in every build and IS a
`raise`, which is why `remove` is the only raising method here.
======================================================================
"""


struct TSlice(Copyable, ImplicitlyCopyable, Movable):
    """Their `TSlice` (`slice.h:9`). Half open, `[Left, Right)`."""

    var left: Int
    var right: Int

    def __init__(out self):
        """Their `TSlice()` (`slice.h:19-23`)."""
        self.left = 0
        self.right = 0

    def __init__(out self, left: Int):
        """Their one-argument `TSlice(ui64 left)` (`slice.h:13-17`).

        A single object, `[left, left + 1)`. This is the spelling
        `ParallelStripeView(..., TSlice(newId, newId + 1))` avoids by hand at
        `split_properties_helper.cpp:821`, and the one `TStripeMapping`
        reaches for when it wants one leaf.
        """
        self.left = left
        self.right = left + 1

    def __init__(out self, left: Int, right: Int):
        """Their two-argument constructor (`slice.h:43-49`).

        Y_ASSERT(left <= right), debug only in their build.
        """
        self.left = left
        self.right = right

    def size(self) -> Int:
        """Their `Size()` (`slice.h:25-28`). Y_ASSERT(Left <= Right)."""
        return self.right - self.left

    def is_empty(self) -> Bool:
        """Their `IsEmpty()` (`slice.h:51-53`). `Left >= Right`, not `== 0`.
        """
        return self.left >= self.right

    def not_empty(self) -> Bool:
        """Their `NotEmpty()` (`slice.h:55-57`)."""
        return not self.is_empty()

    def __eq__(self, other: Self) -> Bool:
        """Their free `operator==` (`slice.h:103-105`).

        Two EMPTY slices compare equal whatever their bounds are. `[7, 7)` and
        `[0, 0)` are the same slice to them, and a mapping that returns
        `{0, 0}` for a device it does not own (`mapping.h:136`, `mapping.h:229`)
        relies on it.
        """
        if self.is_empty() and other.is_empty():
            return True
        return self.left == other.left and self.right == other.right

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def __lt__(self, other: Self) -> Bool:
        """Their `operator<` (`slice.h:92-94`), lexicographic on (Left, Right).
        """
        if self.left != other.left:
            return self.left < other.left
        return self.right < other.right

    def __add__(self, step: Int) -> Self:
        """Their `operator+=` (`slice.h:30-34`), which is how they walk stripes.

        Their `slice.h:99` also declares a free `operator==(const TSlice&,
        ui64)` that RETURNS a shifted `TSlice`. That is a typo for `operator+`
        in their tree, not a comparison; the shift is spelled `__add__` here
        and the typo is not reproduced.
        """
        return Self(self.left + step, self.right + step)

    def __mul__(self, repeat: Int) -> Self:
        """Their `operator*=` (`slice.h:36-41`).

        Shifts by `repeat` WHOLE slices, so `slice * 2` is the third block of
        `size()` objects. It does not scale the size.
        """
        var size = self.size()
        return Self(self.left + repeat * size, self.right + repeat * size)

    @staticmethod
    def intersection(lhs: Self, rhs: Self) -> Self:
        """Their `Intersection` (`slice.h:59-67`).

        A non-overlapping pair collapses to `[0, 0)`, not to an inverted
        slice. `TStripeMapping::ToLocalSlice` (`mapping.h:248`) depends on
        that, because it feeds the result straight into `Size()`.
        """
        var left = lhs.left if lhs.left > rhs.left else rhs.left
        var right = lhs.right if lhs.right < rhs.right else rhs.right
        if left >= right:
            return Self(0, 0)
        return Self(left, right)

    def contains(self, other: Self) -> Bool:
        """Their `Contains` (`slice.h:69-71`). An empty slice is contained by
        anything."""
        if other.is_empty():
            return True
        return self.left <= other.left and other.right <= self.right

    @staticmethod
    def remove(from_slice: Self, part: Self) raises -> List[Self]:
        """Their `Remove` (`slice.h:73-90`).

        Returns what is left of `from_slice` after `part` is taken out: zero,
        one or two pieces, in left-to-right order. CB_ENSURE, so it raises.
        """
        if not from_slice.contains(part):
            raise Error(
                String("TSlice::Remove: ")
                + from_slice.to_string()
                + " does not contain "
                + part.to_string()
            )

        var result = List[Self]()
        if part.is_empty():
            result.append(from_slice)
            return result^

        if part.left > from_slice.left:
            result.append(Self(from_slice.left, part.left))
        if part.right < from_slice.right:
            result.append(Self(part.right, from_slice.right))

        return result^

    def to_string(self) -> String:
        return String("[") + String(self.left) + ", " + String(self.right) + ")"
