"""How a buffer's objects are spread over the devices.

PORT OF `catboost/cuda/cuda_lib/mapping.h` at CatBoost `54a8143a`.
Transliterated where it transliterates. See the DEVIATION BLOCK.

A mapping answers, in `TSlice`, three questions about a buffer:

    GetObjectsSlice()      which objects exist at all
    DeviceSlice(dev)       which of them live on this device
    ToLocalSlice(slice)    what that sub-range looks like renumbered from 0

and two about bytes, `MemorySize` and `MemoryOffset` (`mapping.h:31-61`).
Their three mappings differ only in the answers:

    TSingleMapping    all objects on one device, everywhere else empty
    TMirrorMapping    every device holds a copy of all of them
    TStripeMapping    consecutive disjoint stripes, one per device

The reason to port this at one device is that the questions are asked
regardless. `ParallelStripeView(subsets->PartitionsCpu, TSlice(newId, newId +
1))` (`split_properties_helper.cpp:821`) is a stripe view of one leaf, and its
`MemoryOffset` is what makes the read touch one partition instead of all of
them.

================================ DEVIATION BLOCK ======================
**1. No CRTP base.** Their `TMappingBase<TImpl>` (`mapping.h:13-81`) calls
down into the derived type for `GetObjectsSlice`, `CountAt` and
`DeviceSlice`. Mojo 1.0 has no CRTP and no dynamic trait objects, so the five
base methods that do not need the derived type -- `single_object_size`,
`memory_size`, `memory_offset`, `memory_usage_at`, `device_memory_offset` --
are repeated verbatim in each of the three mappings. Their bodies are
identical to theirs and to each other; if one changes, all three change.

**2. One device.** Every `GetCudaManager().GetDeviceCount()` in their file
(`mapping.h:65`, `:101`, `:225`, `:257`, `:273`, `:284`, `:315`, `:346`,
`:369`) reads 1 here, so `TStripeMapping::SplitBetweenDevices` gives one
stripe holding everything and `TMirrorMapping` and `TSingleMapping` collapse
onto it. The degenerate answers are still computed by their arithmetic rather
than short-circuited, so the file stops being degenerate the day a second
device exists.

**3. `Transform`, `Apply`, `At` and `NonEmptyDevices` are not ported.**
`Transform` and `Apply` (`mapping.h:74-80`, `:145-149`, `:195-199`,
`:293-305`) take a callback over slices; `At` (`mapping.h:43-46`) returns
`NKernelHost::TObjectsMeta` from `cuda_kernel_buffer.h` and `NonEmptyDevices`
(`mapping.h:63-72`) returns a `TDevicesList`. All four have exactly one class
of caller, `TCudaBuffer` (`cuda_buffer.h`), which is not ported. See
NOT_PORTED.md.

**4. `TSingleMapping`'s slices constructor is not ported.** Theirs
(`mapping.h:98-112`) takes one `TSlice` per device, CB_ENSUREs that at most
one is non-empty, and adopts that device. At one device it is
`TSingleMapping(0, slices[0].Size(), size)` and adds nothing that the
two-argument constructor does not already do; `TSingleMappingBuilder` below is
their same rule spelled the way their callers actually reach it
(`mapping.h:396-403`, `CreateMapping`). It comes back the day a second device
does.
======================================================================
"""

from .slice import TSlice


comptime DEVICE_COUNT = 1
"""Their `GetCudaManager().GetDeviceCount()` (`mapping.h:65` and eight other
places). See DEVIATION 2."""


struct TSingleMapping(Copyable, ImplicitlyCopyable, Movable):
    """Their `TSingleMapping` (`mapping.h:83-154`).

    Every object on one device. `CountAt` is 0 for every other device, which
    is what makes a `TSingleMapping` buffer allocate nowhere else.
    """

    var count: Int
    var device_id: Int
    var object_size: Int

    def __init__(out self, device_id: Int = 0, count: Int = 0, size: Int = 1):
        """Their constructor (`mapping.h:91-96`)."""
        self.device_id = device_id
        self.count = count
        self.object_size = size

    def single_object_size(self) -> Int:
        """Their `SingleObjectSize()` (`mapping.h:26-28`)."""
        return self.object_size

    def get_objects_slice(self) -> TSlice:
        """Their `GetObjectsSlice()` (`mapping.h:114-116`)."""
        return TSlice(0, self.count)

    def count_at(self, dev: Int) -> Int:
        """Their `CountAt` (`mapping.h:118-123`)."""
        if dev != self.device_id:
            return 0
        return self.count

    def device_slice(self, dev: Int) -> TSlice:
        """Their `DeviceSlice` (`mapping.h:134-139`)."""
        if dev != self.device_id:
            return TSlice(0, 0)
        return TSlice(0, self.count)

    def get_device_id(self) -> Int:
        """Their `GetDeviceId()` (`mapping.h:125-127`)."""
        return self.device_id

    def memory_size(self, slice: TSlice) -> Int:
        """Their `MemorySize(slice)` (`mapping.h:31-33`)."""
        return slice.size() * self.object_size

    def memory_size(self) -> Int:
        """Their `MemorySize()` (`mapping.h:35-37`)."""
        return self.memory_size(self.get_objects_slice())

    def memory_offset(self, slice: TSlice) -> Int:
        """Their `MemoryOffset` (`mapping.h:39-41`)."""
        return slice.left * self.single_object_size()

    def memory_usage_at(self, dev: Int) -> Int:
        """Their `MemoryUsageAt` (`mapping.h:49-51`)."""
        return self.count_at(dev) * self.object_size

    def device_memory_offset(self, dev: Int, slice: TSlice) raises -> Int:
        """Their `DeviceMemoryOffset` (`mapping.h:53-61`).

        CB_ENSURE that the slice asked for is entirely on that device, then
        the offset is measured from the device's own left edge.
        """
        var device_slice = self.device_slice(dev)
        if TSlice.intersection(slice, device_slice) != slice:
            raise Error(
                String("DeviceMemoryOffset: ")
                + slice.to_string()
                + " is not inside "
                + device_slice.to_string()
            )
        var dev_size = (
            (slice.left - device_slice.left) if slice.size() != 0 else 0
        )
        if dev_size != 0 and slice.left < device_slice.left:
            raise Error(
                slice.to_string() + " " + device_slice.to_string()
            )
        return dev_size * self.single_object_size()

    def to_local_slice(self, slice: TSlice) raises -> Self:
        """Their `ToLocalSlice` (`mapping.h:129-132`)."""
        if not self.get_objects_slice().contains(slice):
            raise Error(
                String("Slice ")
                + slice.to_string()
                + " should be subset of "
                + self.get_objects_slice().to_string()
            )
        return Self(self.device_id, slice.size(), self.single_object_size())

    def repeat_on_all_devices(
        self, object_count: Int, object_size: Int = 1
    ) -> Self:
        """Their `RepeatOnAllDevices` (`mapping.h:141-143`)."""
        return Self(self.device_id, object_count, object_size)

    def change_device(self, new_device_id: Int) -> Self:
        """Their `ChangeDevice` (`mapping.h:151-153`)."""
        return Self(new_device_id, self.count, self.single_object_size())


struct TMirrorMapping(Copyable, ImplicitlyCopyable, Movable):
    """Their `TMirrorMapping` (`mapping.h:156-200`).

    Every device holds every object, so `CountAt` ignores its argument.
    """

    var count: Int
    var object_size: Int

    def __init__(out self, count: Int = 0, object_size: Int = 1):
        """Their constructor (`mapping.h:163-168`)."""
        self.count = count
        self.object_size = object_size

    def single_object_size(self) -> Int:
        """Their `SingleObjectSize()` (`mapping.h:26-28`)."""
        return self.object_size

    def count_at(self, dev: Int) -> Int:
        """Their `CountAt` (`mapping.h:170-173`), which ignores `dev`."""
        return self.count

    def device_slice(self, dev: Int) -> TSlice:
        """Their `DeviceSlice` (`mapping.h:175-178`), which ignores `dev`."""
        return TSlice(0, self.count)

    def get_objects_slice(self) -> TSlice:
        """Their `GetObjectsSlice()` (`mapping.h:180-182`)."""
        return TSlice(0, self.count)

    def memory_size(self, slice: TSlice) -> Int:
        """Their `MemorySize(slice)` (`mapping.h:31-33`)."""
        return slice.size() * self.object_size

    def memory_size(self) -> Int:
        """Their `MemorySize()` (`mapping.h:35-37`)."""
        return self.memory_size(self.get_objects_slice())

    def memory_offset(self, slice: TSlice) -> Int:
        """Their `MemoryOffset` (`mapping.h:39-41`)."""
        return slice.left * self.single_object_size()

    def memory_usage_at(self, dev: Int) -> Int:
        """Their `MemoryUsageAt` (`mapping.h:49-51`)."""
        return self.count_at(dev) * self.object_size

    def device_memory_offset(self, dev: Int, slice: TSlice) raises -> Int:
        """Their `DeviceMemoryOffset` (`mapping.h:53-61`)."""
        var device_slice = self.device_slice(dev)
        if TSlice.intersection(slice, device_slice) != slice:
            raise Error(
                String("DeviceMemoryOffset: ")
                + slice.to_string()
                + " is not inside "
                + device_slice.to_string()
            )
        var dev_size = (
            (slice.left - device_slice.left) if slice.size() != 0 else 0
        )
        if dev_size != 0 and slice.left < device_slice.left:
            raise Error(
                slice.to_string() + " " + device_slice.to_string()
            )
        return dev_size * self.single_object_size()

    def to_local_slice(self, slice: TSlice) raises -> Self:
        """Their `ToLocalSlice` (`mapping.h:184-189`)."""
        if not self.get_objects_slice().contains(slice):
            raise Error(
                String("Slice ")
                + slice.to_string()
                + " should be subset of "
                + self.get_objects_slice().to_string()
            )
        return Self(slice.size(), self.single_object_size())

    def repeat_on_all_devices(
        self, object_count: Int, object_size: Int = 1
    ) -> Self:
        """Their `RepeatOnAllDevices` (`mapping.h:191-193`)."""
        return Self(object_count, object_size)


struct TStripeMapping(Copyable, Movable):
    """Their `TStripeMapping` (`mapping.h:202-306`).

    One consecutive stripe per device, and their constructor CB_ENSUREs that
    the stripes touch: `Slices[i].Left == Slices[i - 1].Right`
    (`mapping.h:213-215`). That invariant is why `GetObjectsSlice` can be a
    min and a max.
    """

    var slices: List[TSlice]
    var object_size: Int

    def __init__(out self):
        """Their default constructor (`mapping.h:222-226`), one empty stripe
        per device."""
        self.slices = List[TSlice]()
        for _ in range(DEVICE_COUNT):
            self.slices.append(TSlice(0, 0))
        self.object_size = 0

    def __init__(
        out self, var slices: List[TSlice], single_object_size: Int = 1
    ) raises:
        """Their constructor (`mapping.h:209-216`), with the touching check.
        """
        for i in range(1, len(slices)):
            if slices[i].left != slices[i - 1].right:
                raise Error(
                    String("TStripeMapping: stripe ")
                    + String(i)
                    + " starts at "
                    + String(slices[i].left)
                    + " but the previous one ends at "
                    + String(slices[i - 1].right)
                )
        self.slices = slices^
        self.object_size = single_object_size

    def single_object_size(self) -> Int:
        """Their `SingleObjectSize()` (`mapping.h:26-28`)."""
        return self.object_size

    def count_at(self, dev: Int) -> Int:
        """Their `CountAt` (`mapping.h:218-220`)."""
        return self.slices[dev].size()

    def device_slice(self, dev: Int) -> TSlice:
        """Their `DeviceSlice` (`mapping.h:228-230`)."""
        if dev < len(self.slices):
            return self.slices[dev]
        return TSlice(0, 0)

    def get_objects_slice(self) -> TSlice:
        """Their `GetObjectsSlice()` (`mapping.h:232-241`), the min left and
        the max right over every stripe."""
        var min = self.slices[0].left
        var max = self.slices[0].right
        for dev in range(len(self.slices)):
            if self.slices[dev].left < min:
                min = self.slices[dev].left
            if self.slices[dev].right > max:
                max = self.slices[dev].right
        return TSlice(min, max)

    def memory_size(self, slice: TSlice) -> Int:
        """Their `MemorySize(slice)` (`mapping.h:31-33`)."""
        return slice.size() * self.object_size

    def memory_size(self) -> Int:
        """Their `MemorySize()` (`mapping.h:35-37`)."""
        return self.memory_size(self.get_objects_slice())

    def memory_offset(self, slice: TSlice) -> Int:
        """Their `MemoryOffset` (`mapping.h:39-41`)."""
        return slice.left * self.single_object_size()

    def memory_usage_at(self, dev: Int) -> Int:
        """Their `MemoryUsageAt` (`mapping.h:49-51`)."""
        return self.count_at(dev) * self.object_size

    def device_memory_offset(self, dev: Int, slice: TSlice) raises -> Int:
        """Their `DeviceMemoryOffset` (`mapping.h:53-61`)."""
        var device_slice = self.device_slice(dev)
        if TSlice.intersection(slice, device_slice) != slice:
            raise Error(
                String("DeviceMemoryOffset: ")
                + slice.to_string()
                + " is not inside "
                + device_slice.to_string()
            )
        var dev_size = (
            (slice.left - device_slice.left) if slice.size() != 0 else 0
        )
        if dev_size != 0 and slice.left < device_slice.left:
            raise Error(
                slice.to_string() + " " + device_slice.to_string()
            )
        return dev_size * self.single_object_size()

    def to_local_slice(self, slice: TSlice) raises -> Self:
        """Their `ToLocalSlice` (`mapping.h:243-254`).

        Each stripe is intersected with the requested range and then the
        stripes are re-laid end to end from 0, which is what makes the result
        a valid stripe mapping again.
        """
        if not self.get_objects_slice().contains(slice):
            raise Error(
                String("Slice ")
                + slice.to_string()
                + " should be subset of "
                + self.get_objects_slice().to_string()
            )
        var slices = List[TSlice]()
        for i in range(len(self.slices)):
            var part = TSlice.intersection(self.slices[i], slice)
            var left = slices[i - 1].right if i > 0 else 0
            slices.append(TSlice(left, left + part.size()))
        return Self(slices^, self.single_object_size())

    @staticmethod
    def split_between_devices(
        object_count: Int, object_size: Int = 1
    ) raises -> Self:
        """Their `SplitBetweenDevices` (`mapping.h:256-269`).

        The LAST device takes the remainder, not the first. At one device it
        takes everything.
        """
        var slices = List[TSlice]()
        var object_per_device = object_count // DEVICE_COUNT
        var total = 0
        for i in range(DEVICE_COUNT):
            var dev_size = (
                object_per_device if (i + 1) != DEVICE_COUNT
                else (object_count - total)
            )
            slices.append(TSlice(total, total + dev_size))
            total += dev_size
        return Self(slices^, object_size)

    @staticmethod
    def repeat_on_all_devices(
        object_count: Int, object_size: Int = 1
    ) raises -> Self:
        """Their static `RepeatOnAllDevices` (`mapping.h:271-280`).

        Note it does NOT mirror: device `i` gets `[i * count, (i + 1) * count)`,
        so the stripes are distinct ranges of a buffer `devCount` times as
        long.
        """
        var slices = List[TSlice]()
        for i in range(DEVICE_COUNT):
            slices.append(TSlice(i * object_count, (i + 1) * object_count))
        return Self(slices^, object_size)

    @staticmethod
    def create_from_sizes(sizes: List[Int], object_size: Int = 1) raises -> Self:
        """Their `CreateFromSizes` (`mapping.h:281-291`)."""
        var slices = List[TSlice]()
        for i in range(DEVICE_COUNT):
            var left = slices[i - 1].right if i > 0 else 0
            slices.append(TSlice(left, left + sizes[i]))
        return Self(slices^, object_size)


struct TStripeMappingBuilder(Movable):
    """Their `TMappingBuilder<TStripeMapping>` (`mapping.h:311-340`).

    Mojo has no template specialisation, so their one builder name becomes
    three; the bodies are theirs.
    """

    var device_sizes: List[Int]

    def __init__(out self):
        self.device_sizes = List[Int]()
        for _ in range(DEVICE_COUNT):
            self.device_sizes.append(0)

    def set_size_at(mut self, device: Int, size: Int):
        """Their `SetSizeAt` (`mapping.h:318-321`)."""
        self.device_sizes[device] = size

    def update_max_size_at(mut self, device: Int, size: Int):
        """Their `UpdateMaxSizeAt` (`mapping.h:323-326`)."""
        if size > self.device_sizes[device]:
            self.device_sizes[device] = size

    def build(self, object_size: Int = 1) raises -> TStripeMapping:
        """Their `Build` (`mapping.h:328-336`)."""
        var slices = List[TSlice]()
        var offset = 0
        for dev in range(len(self.device_sizes)):
            slices.append(TSlice(offset, offset + self.device_sizes[dev]))
            offset += self.device_sizes[dev]
        return TStripeMapping(slices^, object_size)


struct TMirrorMappingBuilder(Movable):
    """Their `TMappingBuilder<TMirrorMapping>` (`mapping.h:342-363`)."""

    var device_sizes: List[Int]

    def __init__(out self):
        self.device_sizes = List[Int]()
        for _ in range(DEVICE_COUNT):
            self.device_sizes.append(0)

    def set_size_at(mut self, device: Int, size: Int):
        """Their `SetSizeAt` (`mapping.h:349-352`)."""
        self.device_sizes[device] = size

    def build(self, object_size: Int = 1) raises -> TMirrorMapping:
        """Their `Build` (`mapping.h:354-359`), which CB_ENSUREs every device
        was given the same size."""
        for dev in range(len(self.device_sizes)):
            if self.device_sizes[dev] != self.device_sizes[0]:
                raise Error(
                    "TMirrorMapping: every device must hold the same count"
                )
        return TMirrorMapping(self.device_sizes[0], object_size)


struct TSingleMappingBuilder(Movable):
    """Their `TMappingBuilder<TSingleMapping>` (`mapping.h:365-392`)."""

    var device_sizes: List[Int]

    def __init__(out self):
        self.device_sizes = List[Int]()
        for _ in range(DEVICE_COUNT):
            self.device_sizes.append(0)

    def set_size_at(mut self, device: Int, size: Int):
        """Their `SetSizeAt` (`mapping.h:372-375`)."""
        self.device_sizes[device] = size

    def build(self) raises -> TSingleMapping:
        """Their `Build` (`mapping.h:377-388`), which CB_ENSUREs at most one
        device was given a non-zero size."""
        var non_zero_devices = 0
        var non_zero_id = 0
        for i in range(len(self.device_sizes)):
            if self.device_sizes[i] != 0:
                non_zero_devices += 1
                non_zero_id = i
        if non_zero_devices > 1:
            raise Error(
                "TSingleMapping: more than one device was given a size"
            )
        return TSingleMapping(non_zero_id, self.device_sizes[non_zero_id])
