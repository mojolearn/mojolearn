# SPDX-License-Identifier: Apache-2.0
"""Native optimizer state spans and host-mirror preparation decisions."""
from bindings.tensor_view import TensorView, base_format
from training.optimizer_public_common import PublicStep


@fieldwise_init
struct OptimizerStateSnapshot(Copyable, ImplicitlyCopyable, Movable):
    var handle: Int
    var resident_wanted: Int
    var pristine: Bool
    var first: Int
    var second: Int

    def apply(self, mut request: PublicStep):
        request.handle = self.handle
        request.resident_wanted = self.resident_wanted
        request.pristine = self.pristine
        request.m_addr = self.first
        request.v_addr = self.second


def optimizer_state_count(view: TensorView) raises -> Int:
    var count = 1
    for axis in range(len(view.shape)):  # driver-loop: admitted state shape words
        var extent = view.shape[axis]
        if extent < 0 or (extent > 0 and count > 9223372036854775807 // extent):
            raise Error("ValueError: optimizer state shape is out of range")
        count *= extent
    return count


def optimizer_state_span(view: TensorView, count: Int, name: String, integer: Bool = False) raises -> Int:
    if count < 0 or count > 9223372036854775807 // 4:
        raise Error("ValueError: optimizer state size is out of range")
    var format = base_format(view.fmt)
    if view.itemsize != 4 or format != (String("i") if integer else String("f")):
        raise Error("TypeError: optimizer " + name + " requires a native " + (String("int32") if integer else String("float32")) + " buffer")
    if optimizer_state_count(view) != count or view.nbytes != count * 4:
        raise Error("ValueError: optimizer " + name + " does not match the optimizer registry")
    if not view.c_contig or view.readonly or (count > 0 and view.addr == 0):
        raise Error("ValueError: optimizer " + name + " requires contiguous writable native storage")
    return view.addr


def optimizer_checkpoint_spans(first: TensorView, second: TensorView, initialized: TensorView,
    count: Int, parameters: Int, owner: String) raises:
    var first_count = optimizer_state_count(first)
    var second_count = optimizer_state_count(second)
    var initialized_count = optimizer_state_count(initialized)
    if first_count != count:
        raise Error("ValueError: mojolearn." + owner + ".load_state_dict: exp_avg holds " + String(first_count)
            + " floats, this optimizer's registry is " + String(count))
    if second_count != count:
        raise Error("ValueError: mojolearn." + owner + ".load_state_dict: exp_avg_sq holds " + String(second_count)
            + " floats, this optimizer's registry is " + String(count))
    if initialized_count != parameters:
        raise Error("ValueError: buf_initialized does not match the optimizer parameter registry")
    _ = optimizer_state_span(first, count, "exp_avg")
    _ = optimizer_state_span(second, count, "exp_avg_sq")
    _ = optimizer_state_span(initialized, parameters, "buf_initialized", True)


def optimizer_state_download_needed(handle: Int, host_fresh: Bool) -> Bool:
    return handle >= 0 and not host_fresh


def optimizer_host_moment_plan(count: Int, first_present: Bool, second_present: Bool) raises -> Tuple[Bool, Bool]:
    if count < 0 or count > 9223372036854775807 // 4:
        raise Error("ValueError: optimizer state size is out of range")
    return (not first_present, not second_present)
