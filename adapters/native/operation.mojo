# SPDX-License-Identifier: Apache-2.0
# Native adapter boundary over the existing OpsSession dispatcher. No new math.
from autograd.ops_api import OpsSession, OPS_COPY
from autograd.rankn import ShapeN, RN_MAX_RANK
from autograd.program import PROGRAM_MAX_CODE, same_shape
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_COPY


comptime WORDS = 3 + RN_MAX_RANK  # address, kind (0=f32, 1=i32 bits), rank, dimensions
comptime IP = MutPointer[Int, MutUntrackedOrigin]


def descriptor_shape(words: IP, offset: Int) raises -> ShapeN:
    var rank = words[offset + 2]
    if rank < 0 or rank > RN_MAX_RANK:
        raise Error("adapter: rank must be between zero and eight")
    var dims = List[Int]()
    for j in range(rank):  # small-loop(rank<=8): descriptor metadata
        dims.append(words[offset + 3 + j])
    var shape = ShapeN.of(dims)
    if words[offset] == 0 or words[offset] % 4 != 0:
        raise Error("adapter: non-null four-byte-aligned storage required")
    if words[offset + 1] != 0 and words[offset + 1] != 1:
        raise Error("adapter: storage must be float32 or int32")
    return shape


def copy_words[E: Exec](mut ex: E, src: FP, dst: FP, count: Int) raises:
    var args = Args()
    args.p0 = src
    args.p1 = dst
    ex.launch[OP_COPY](args, count)


def run_operation[E: Exec & Movable & Deinitable](
    var ex: E, code: Int, descriptors: IP, count: Int, output: IP,
    int_args: IP, n_ints: Int, float_args: FP, n_floats: Int,
) raises:
    if code < 1 or code > PROGRAM_MAX_CODE or code == 47:
        raise Error("adapter: unknown operation code")
    if count < 0 or n_ints < 0 or n_floats < 0:
        raise Error("adapter: negative argument count")
    # These legacy one-input forms encode host addresses in integer options.
    # A framework call must carry indices as a real second tensor instead.
    if (code == 32 or code == 41) and count != 2:
        raise Error("adapter: embedding and take require tensor indices")
    if code == 22:
        raise Error("adapter: gather_rows requires session permutation provenance; use take with int32 indices")
    if (count > 0 and Int(descriptors) == 0) or Int(output) == 0:
        raise Error("adapter: missing tensor descriptors")
    if (n_ints > 0 and Int(int_args) == 0) or (n_floats > 0 and Int(float_args) == 0):
        raise Error("adapter: missing operation arguments")
    # Validate all descriptors before allocating or enqueueing work.
    var expected = descriptor_shape(output, 0)
    var shapes = List[ShapeN]()
    for j in range(count):  # driver-loop: declared input descriptors
        shapes.append(descriptor_shape(descriptors, j * WORDS))
        var integer = descriptors[j * WORDS + 1] == 1
        # Integer storage is for indices and exact copies. In particular the
        # existing embedding/slab kernels apply floating-point normalization
        # to values: integer value tables or concatenations would lose bits.
        # Do not advertise general integer tensor computation through this ABI.
        var ids = (code == 32 or code == 41) and j == 1
        if ids and not integer:
            raise Error("adapter: embedding and take indices must be int32")
        if integer and not (code == OPS_COPY or ids):
            raise Error("adapter: this operation requires float32 inputs")
    var ints = List[Int]()
    var floats = List[Float32]()
    for j in range(n_ints):  # driver-loop: operation metadata
        ints.append(int_args[j])
    for j in range(n_floats):  # driver-loop: operation scalar configuration
        floats.append(float_args[j])
    var s = OpsSession(ex^, False)
    try:
        var handles = List[Int]()
        for j in range(count):  # driver-loop: native storage ownership
            var offset = j * WORDS
            # A device-to-device copy gives the existing executor its ordinary
            # ownership contract, including operations which inspect integer
            # indices. This first adapter does not claim zero-copy execution.
            var buf = s.ex.alloc(shapes[j].count())
            copy_words(s.ex, FP(unsafe_from_address=descriptors[offset]), buf, shapes[j].count())
            var handle = s._leaf(buf, shapes[j], descriptors[offset + 1] == 1)
            s.owned[s._slot(handle)] = Int(buf)
            handles.append(handle)
        var result = s._op_run(code, handles, ints, floats)
        var slot = s._slot(result)
        if not same_shape(s.shapes[slot], expected):
            raise Error("adapter: declared output shape differs from the native result")
        if Int(s.int_words[slot]) != output[1]:
            raise Error("adapter: declared output dtype differs from the native result")
        # Layout operations can return a deferred view. Use the same native
        # materialization entry as ordinary public result consumption.
        var tensor = s.tensor(result)
        var source = s.tape.value_of(s.ex, tensor)
        copy_words(s.ex, source, FP(unsafe_from_address=output[0]), expected.count())
        s.ex.sync()
    except error:
        s.close()
        raise error
    s.close()
