# SPDX-License-Identifier: Apache-2.0
"""Owned GP tables and scratch; public arrays and provider pins stay outside.

Views borrow these owners for synchronous descriptor calls. Metadata copies
use the same native bulk-copy primitive as the existing GP binding. No data
arithmetic or numerical kernel is implemented here.
"""
from std.memory import memcpy
from core.gp_native_api import GPKernelView


def gp_native_ints(address: Int, count: Int) raises -> List[Int]:
    if count < 0 or (count > 0 and address == 0):
        raise Error("GP native metadata: invalid integer span")
    var result = List[Int](unsafe_uninit_length=count)
    if count > 0:
        var source = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=address)
        memcpy(dest=result.unsafe_ptr(), src=source, count=count)
    return result^


def gp_native_f64(address: Int, count: Int) raises -> List[Float64]:
    if count < 0 or (count > 0 and address == 0):
        raise Error("GP native metadata: invalid float64 span")
    var result = List[Float64](unsafe_uninit_length=count)
    if count > 0:
        var source = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=address)
        memcpy(dest=result.unsafe_ptr(), src=source, count=count)
    return result^


struct GPKernelStorage(Copyable, Movable):
    var kinds: List[Int32]
    var parameters: List[Float32]
    var lengths: List[Int32]
    var scales: List[Float32]
    var scale_count: Int

    def __init__(out self):
        self.kinds = List[Int32]()
        self.parameters = List[Float32]()
        self.lengths = List[Int32]()
        self.scales = List[Float32]()
        self.scale_count = 0

    def finish(mut self):
        # A no-length-scale kernel still addresses one unused 1.0 word.
        # finish is called once after the public kernel has been unpacked.
        self.scale_count = len(self.scales)
        if self.scale_count == 0:
            self.scales.append(Float32(1.0))

    def view(self) -> GPKernelView:
        return GPKernelView(Int(self.kinds.unsafe_ptr()), Int(self.parameters.unsafe_ptr()),
            Int(self.lengths.unsafe_ptr()), Int(self.scales.unsafe_ptr()),
            len(self.kinds), self.scale_count)


@fieldwise_init
struct GPRFitResult(Copyable, Movable):
    var info: Int
    var panel_width: Int
    var logdet: Float64
    var y_dot_alpha: Float64
    var likelihood: Float64


struct GPRFitScratch(Movable):
    var scalars: List[Float64]

    def __init__(out self):
        self.scalars = List[Float64](length=5, fill=0.0)

    def address(mut self) -> Int:
        return Int(self.scalars.unsafe_ptr())

    def result(self) -> GPRFitResult:
        return GPRFitResult(Int(self.scalars[0]), Int(self.scalars[1]),
            self.scalars[2], self.scalars[3], self.scalars[4])


struct GPRGradientScratch(Movable):
    var gradient: List[Float64]
    var scalars: List[Float64]

    def __init__(out self, count: Int):
        self.gradient = List[Float64](length=max(count, 1), fill=0.0)
        self.scalars = List[Float64](length=2, fill=0.0)


@fieldwise_init
struct GPRGradientResult(Movable):
    var info: Int
    var likelihood: Float64
    var gradient: List[Float64]


struct GPFreeParameters(Movable):
    var flags: List[Int32]
    var values: List[Float64]
    var bounds: List[Float32]

    def __init__(out self):
        self.flags = List[Int32]()
        self.values = List[Float64]()
        self.bounds = List[Float32]()


struct GPOptimizerScratch(Movable):
    var theta: List[Float64]
    var values: List[Float64]
    var runs: List[Float64]

    def __init__(out self, count: Int, starts: Int):
        self.theta = List[Float64](length=max(count, 1), fill=0.0)
        self.values = List[Float64](length=max(count, 1), fill=0.0)
        self.runs = List[Float64](length=max(4 * starts, 1), fill=0.0)


struct GPCMemberTables(Movable):
    """Native lists of member addresses, not copies of training/model arrays."""
    var columns: List[Int]
    var targets: List[Int]
    var factors: List[Int]
    var pis: List[Int]
    var wsrs: List[Int]
    var scalars: List[Int]
    var scalar_storage: List[Float64]

    def __init__(out self, members: Int):
        self.columns = List[Int](length=members, fill=0)
        self.targets = List[Int](length=members, fill=0)
        self.factors = List[Int](length=members, fill=0)
        self.pis = List[Int](length=members, fill=0)
        self.wsrs = List[Int](length=members, fill=0)
        self.scalars = List[Int](length=members, fill=0)
        self.scalar_storage = List[Float64](length=3 * members, fill=0.0)

    def prepare_scalars(mut self):
        for member in range(len(self.scalars)):  # driver-loop: one metadata entry per fitted OVR member
            self.scalars[member] = Int(self.scalar_storage.unsafe_ptr() + 3 * member)


struct GPCQueryTables(Movable):
    var targets: List[Int]
    var pis: List[Int]
    var wsrs: List[Int]
    var factors: List[Int]

    def __init__(out self, members: Int):
        self.targets = List[Int](length=members, fill=0)
        self.pis = List[Int](length=members, fill=0)
        self.wsrs = List[Int](length=members, fill=0)
        self.factors = List[Int](length=members, fill=0)


struct GPCQueryScratch(Movable):
    """Only unexposed query outputs; selected public/transport spans are external."""
    var mean: List[Float32]
    var variance: List[Float32]
    var probability: List[Float64]

    def __init__(out self, rows: Int, external: Int) raises:
        if rows < 0 or external < 0 or external > 7:
            raise Error("GP query scratch: invalid row count or external span mask")
        self.mean = List[Float32](length=0 if (external & 1) != 0 else rows, fill=0)
        self.variance = List[Float32](length=0 if (external & 2) != 0 else rows, fill=0)
        self.probability = List[Float64](length=0 if (external & 4) != 0 else rows, fill=0)
