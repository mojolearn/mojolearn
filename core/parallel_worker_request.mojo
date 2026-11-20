# SPDX-License-Identifier: Apache-2.0
"""The typed request of a parallel worker RPC (lane parallel-workers-typed-rpc,
2026-11-12).

A request to a pool worker (a separate process, `bindings/native_process_pool_door.mojo`)
is a plain Mojo struct: an operation code, a handle id that is valid only in
the addressed worker (a layer index or a state token in the worker's own
registry) and a few Int words. It travels as its bytes, never as a Python
object: the parent UNPACKs the public call once into this struct, the worker
decodes the bytes once and dispatches on the code. Tensors that ride with a
request (activations, token ids) travel beside it as the transport's existing
buffer payload.

Wire layout, little-endian Int64 words:
  [MAGIC, LAYOUT, op, handle, n_words, words...]
Changing the layout requires a new LAYOUT value; a worker refuses any other.

Encoding and decoding are a handful of words per request: no independent work
to split across cores.
"""
from std.memory import unsafe_memcpy

comptime WORKER_REQUEST_MAGIC = 0x4D4C5752  # "MLWR"
comptime WORKER_REQUEST_LAYOUT = 1
comptime WORKER_REQUEST_HEADER_WORDS = 5
# Bounds a decoded request: the longest request is a "run_layers" over every
# layer of one worker (3 flags + 1 count + 2 words per layer).
comptime WORKER_REQUEST_MAX_WORDS = 1 << 16
comptime WORKER_NO_HANDLE = -1

# ParallelCausalLM worker operations (bindings/parallel_causal_native.mojo).
comptime PC_CONFIGURE = 1
comptime PC_BLOCK = 2
comptime PC_TENSORS = 3
comptime PC_EMBEDDING = 4
comptime PC_FINAL = 5
comptime PC_HEAD = 6
comptime PC_HEAD_INT15 = 7
comptime PC_NORM = 8
comptime PC_ROUTE = 9
comptime PC_ALLOCATE = 10
comptime PC_RELEASE = 11
comptime PC_STATE = 12
comptime PC_PUBLISH_STATE = 13
comptime PC_RUN_LAYERS = 14
comptime PC_PARAMETERS = 15
comptime PC_RUN = 16
comptime PC_BLOCK_FIELDS = 17


struct WorkerRequest(Copyable, Movable):
    var op: Int
    var handle: Int
    var words: List[Int]

    def __init__(out self, op: Int, handle: Int = WORKER_NO_HANDLE):
        self.op = op
        self.handle = handle
        self.words = List[Int]()

    def encode(self) -> List[Int64]:
        """The wire words; the caller hands `len * 8` bytes from
        `unsafe_ptr()` to the transport once."""
        var count = len(self.words)
        var wire = List[Int64](length=WORKER_REQUEST_HEADER_WORDS + count, fill=0)
        wire[0] = Int64(WORKER_REQUEST_MAGIC)
        wire[1] = Int64(WORKER_REQUEST_LAYOUT)
        wire[2] = Int64(self.op)
        wire[3] = Int64(self.handle)
        wire[4] = Int64(count)
        for i in range(count):  # small-loop(request words: a few per request, one pair per layer of one worker)
            wire[WORKER_REQUEST_HEADER_WORDS + i] = Int64(self.words[i])
        return wire^


def worker_request_decode(address: Int, byte_count: Int) raises -> WorkerRequest:
    """The request whose wire bytes start at `address` (any alignment)."""
    if address == 0 or byte_count < WORKER_REQUEST_HEADER_WORDS * 8 or byte_count % 8 != 0:
        raise Error("parallel worker: malformed typed request")
    var total = byte_count // 8
    if total - WORKER_REQUEST_HEADER_WORDS > WORKER_REQUEST_MAX_WORDS:
        raise Error("parallel worker: typed request exceeds its word bound")
    var wire = List[Int64](length=total, fill=0)
    unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(wire.unsafe_ptr())),
                  src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=address), count=byte_count)
    if Int(wire[0]) != WORKER_REQUEST_MAGIC or Int(wire[1]) != WORKER_REQUEST_LAYOUT:
        raise Error("parallel worker: typed request layout differs from this worker")
    var count = Int(wire[4])
    if count != total - WORKER_REQUEST_HEADER_WORDS:
        raise Error("parallel worker: typed request word count differs from its bytes")
    var request = WorkerRequest(Int(wire[2]), Int(wire[3]))
    request.words = List[Int](length=count, fill=0)
    for i in range(count):  # small-loop(request words: a few per request, one pair per layer of one worker)
        request.words[i] = Int(wire[WORKER_REQUEST_HEADER_WORDS + i])
    _ = wire^
    return request^


def parallel_causal_op_code(name: String) raises -> Int:
    """UNPACK of the private by-name entry points (`ParallelCausalLM._rpc`,
    `_RemoteBlock.call`, the primitives): the operation name to its code."""
    if name == "configure":
        return PC_CONFIGURE
    if name == "block":
        return PC_BLOCK
    if name == "tensors":
        return PC_TENSORS
    if name == "embedding":
        return PC_EMBEDDING
    if name == "final":
        return PC_FINAL
    if name == "head":
        return PC_HEAD
    if name == "head_int15":
        return PC_HEAD_INT15
    if name == "norm":
        return PC_NORM
    if name == "route":
        return PC_ROUTE
    if name == "allocate":
        return PC_ALLOCATE
    if name == "release":
        return PC_RELEASE
    if name == "state":
        return PC_STATE
    if name == "publish_state":
        return PC_PUBLISH_STATE
    if name == "run_layers":
        return PC_RUN_LAYERS
    if name == "parameters":
        return PC_PARAMETERS
    if name == "run":
        return PC_RUN
    if name == "block_fields":
        return PC_BLOCK_FIELDS
    raise Error("ParallelCausalLM: unknown worker operation " + name)
