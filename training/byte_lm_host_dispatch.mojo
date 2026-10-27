# SPDX-License-Identifier: Apache-2.0
# Host ByteLM uses the same native descriptor transport and ownership interface
# as GPU models. This table owns four existing model operations and profile readback.
from std.ffi import _Global
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from training.byte_lm_config import ByteConfig, byte_lm_arithmetic_suffix
from training.byte_lm_host_execution import byte_host_logits_native, byte_host_next_native, byte_host_loss_native, byte_host_train_native

comptime BYTE_HOST_NATIVE_ABI = "mojolearn.training.byte_lm_host.v1"
comptime BYTE_HOST_NATIVE_OPERATION_COUNT = 5


def byte_host_operation(op: Int, addresses: List[Int], words: List[Int], scalars: List[Float64]) raises -> Int:
    if len(words) != 15 or (words[9] != 0 and words[9] != 1):
        raise Error("byte LM host: invalid model configuration descriptor")
    var cfg = ByteConfig(words[0], words[1], words[2], words[3], words[4], words[5],
        words[6], words[7], words[8], words[9] == 1)
    cfg.validate()
    var use_threads = words[10]
    var thread_count = words[11]
    if use_threads != 0 and use_threads != 1:
        raise Error("byte LM host: threaded must be 0 or 1")
    if thread_count < 0 or thread_count > 1024:
        raise Error("byte LM host: threads must be in [0, 1024] (0: one per physical core)")
    if op == 4:
        if len(addresses) != 1 or addresses[0] == 0 or len(scalars) != 0:
            raise Error("byte LM host: invalid native profile output")
        MutPointer[String, MutUntrackedOrigin](unsafe_from_address=addresses[0])[0] = cfg.profile()
        MutPointer[String, MutUntrackedOrigin](unsafe_from_address=addresses[0])[1] = byte_lm_arithmetic_suffix()
        return 0
    if op == 3:
        if len(addresses) != 8 or len(scalars) != 5:
            raise Error("byte LM host: train step expects eight buffers and five optimizer scalars")
        if words[14] < 0:
            raise Error("byte LM host: completed_steps must not be negative")
        return byte_host_train_native(addresses, cfg, scalars, words[14])
    if len(scalars) != 0:
        raise Error("byte LM host: inference takes no optimizer scalar descriptors")
    if op == 2:
        if len(addresses) != 2:
            raise Error("byte LM host: expected 2 addresses")
        return byte_host_loss_native(addresses, cfg, use_threads, thread_count)
    if len(addresses) != 3:
        raise Error("byte LM host: expected 3 addresses")
    var dims: List[Int] = [words[12], words[13]]
    if op == 0:
        return byte_host_logits_native(addresses, dims, cfg, use_threads, thread_count)
    if op == 1:
        return byte_host_next_native(addresses, dims, cfg, thread_count)
    raise Error("byte LM host: unknown native operation")


def byte_host_native_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                        result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if op < 0 or op >= BYTE_HOST_NATIVE_OPERATION_COUNT or result == 0 or nf < 0 or (nf > 0 and ff == 0):
            raise Error("byte LM host: invalid native operation descriptor")
        var scalars = List[Float64]()
        var raw = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        for i in range(nf):  # driver-loop: over admitted optimizer scalar metadata
            scalars.append(raw[i])
        var value = byte_host_operation(op, native_descriptor_words(aa, na), native_descriptor_words(pp, np), scalars)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct ByteHostNativeTable(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(BYTE_HOST_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(byte_host_native_v1)


comptime BYTE_HOST_NATIVE_TABLE = _Global[StorageType=ByteHostNativeTable,
    name="MojoByteHostNativeTableV1", init_fn=ByteHostNativeTable.__init__]
