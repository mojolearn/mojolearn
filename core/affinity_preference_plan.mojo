# SPDX-License-Identifier: Apache-2.0
"""Native storage for the existing inferred-dtype then Float32 preference cast."""
from core.sequential_input_plan import SequenceInputPlan, sequence_write, sequence_array_typestr, sequence_kind_code
from core.admitted_input import admitted_cast_f32
from core.host_finite_scan import host_all_finite_f32
from core.kmeans_variants_api import VariantsSpan
from core.native_array_layout import NativeArrayLayout


struct AffinityPreferencePlan(Movable):
    var inferred: List[UInt8]
    var values: List[Float32]
    var count: Int
    var direct: Bool

    def __init__(out self):
        self.inferred = List[UInt8]()
        self.values = List[Float32]()
        self.count = 0
        self.direct = False

    def prepare(mut self, plan: SequenceInputPlan, numpy_connected: Bool) raises:
        # Materialize the inferred dtype before narrowing, exactly as the old
        # sequence Array did. In particular mixed signed/unsigned 64-bit words
        # first round to Float64; direct integer-to-Float32 would change bits.
        var layout = NativeArrayLayout(plan.shape.copy(), plan.itemsize)
        self.inferred = List[UInt8](unsafe_uninit_length=layout.nbytes)
        sequence_write(plan, Int(self.inferred.unsafe_ptr()))
        if not numpy_connected:
            _ = sequence_array_typestr(plan.kind, plan.itemsize, "preference")
        if len(plan.shape) != 2:
            var shape = String("(")
            for axis in range(len(plan.shape)):  # metadata: admitted rank is at most eight
                if axis:
                    shape += ", "
                shape += String(plan.shape[axis])
            if len(plan.shape) == 1:
                shape += ","
            shape += ")"
            raise Error("mojolearn: preference must be 2-D, got " + String(len(plan.shape)) + "-D shape " + shape)
        self.count = plan.count
        self.direct = plan.kind == "f" and plan.itemsize == 4
        var finite = True
        if self.direct:
            finite = host_all_finite_f32(rebind[MutPointer[Float32, MutUntrackedOrigin]](self.inferred.unsafe_ptr().bitcast[Float32]()), self.count)
        else:
            self.values = List[Float32](unsafe_uninit_length=self.count)
            # Flatten the single preference row into independent conversion
            # units, using the same CastIn implementation and fold-free words.
            finite = admitted_cast_f32(Int(self.inferred.unsafe_ptr()), plan.itemsize, plan.itemsize,
                self.count, 1, plan.itemsize, sequence_kind_code(plan.kind),
                rebind[MutPointer[Float32, MutUntrackedOrigin]](self.values.unsafe_ptr()))
            self.inferred = List[UInt8]()
        if not finite:
            raise Error("mojolearn: preference contains NaN or infinity")

    def span(self) -> VariantsSpan:
        if self.count == 0:
            return VariantsSpan(0, 0)
        return VariantsSpan(Int(self.inferred.unsafe_ptr()) if self.direct else Int(self.values.unsafe_ptr()), self.count)
