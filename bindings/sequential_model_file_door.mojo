# SPDX-License-Identifier: Apache-2.0
# Sequential's model-file operations (checkpoint save / info / load, the
# verify sidecar's numbers, the parameter count, the weight import) as calls
# of the whole-model native descriptor of the model's route
# (sequence/model_file_dispatch.mojo; lane dw-autograd-seq, 2026-11-21).
# Each helper takes typed words, makes ONE descriptor call and returns typed
# values: no CPython call, no Python list of words. Before, each was the
# x_sequence binding's Python entry of the same name called through
# `py_call_typed` (a CPython call that re-read every word from a Python list
# and built its result as a Python list). The same typed functions run.
from std.python import PythonObject
from bindings.py_balanced import py_item
from bindings.sequential_keys_unpack import sequential_keys
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_execute
from sequence.neural_model_dispatch import (
    MODEL_SAVE_CHECKPOINT, MODEL_CHECKPOINT_INFO, MODEL_LOAD_CHECKPOINT, MODEL_VERIFY_DIGEST,
    MODEL_VERIFY_FILL, MODEL_VERIFY_FILE_SHA256, MODEL_VERIFY_MODEL_INFO, MODEL_N_PARAMS,
    MODEL_IMPORT_PARAMS, MODEL_EXPORT_PARAMS,
)

# Text results are short and fixed: an FNV-1a64 digest (16 hex digits), a
# SHA-256 (64), a kind name plus a description hash.
comptime SEQUENTIAL_FILE_TEXT = 256
# The first checkpoint-info call holds this many layers; a longer model asks
# once more with the exact size the first call reported.
comptime SEQUENTIAL_FILE_LAYERS = 256
comptime SEQUENTIAL_FILE_INFO_INTS = 64
comptime SEQUENTIAL_FILE_INFO_FLOATS = 16
#: autograd/model.mojo SPEC2_ARGS (each layer's argument words) and the sample record's words (its rank, then at
#: most RN_MAX_RANK axes)
comptime SEQUENTIAL_FILE_ARGS = 6
comptime SEQUENTIAL_FILE_SAMPLE_WORDS = 9


struct SequentialCheckpointInfo(Movable):
    """`model_checkpoint_info`'s answer: per layer three words in
    `layer_ints` and one float in `layer_floats`; `ints` and `floats` the
    checkpoint's option words and floats."""
    var is_export: Bool
    var layer_ints: List[Int]
    var layer_floats: List[Float64]
    var ints: List[Int]
    var floats: List[Float64]
    # lane nn-integrate: SEQUENTIAL_FILE_ARGS argument words per layer and the sample shape (empty: 2-D)
    var layer_args: List[Int]
    var sample: List[Int]

    def __init__(out self):
        self.is_export = False
        self.layer_ints = List[Int]()
        self.layer_floats = List[Float64]()
        self.layer_args = List[Int]()
        self.sample = List[Int]()
        self.ints = List[Int]()
        self.floats = List[Float64]()


def _file_call(entries: PythonObject, operation: Int, addresses: List[Int], ints: List[Int],
               floats: List[Float64]) raises -> Int:
    # The provider of the model's route (bindings/sequential_route_door.mojo
    # stores it beside the entries at the first resolution).
    var nk = sequential_keys()
    var provider = py_item(entries, nk[].k_native_model_provider)
    return neural_descriptor_execute(provider, operation, addresses, ints, floats)


def _file_text(buffer: List[UInt8], start: Int, count: Int) raises -> String:
    if start < 0 or count < 0 or start + count > len(buffer):
        raise Error("RuntimeError: Sequential native text result runs past its buffer")
    var bytes = List[UInt8](capacity=count)
    for i in range(count):  # driver-loop: over the bytes of one digest or kind name
        bytes.append(buffer[start + i])
    return String(StringSlice(unsafe_from_utf8=Span(bytes)))


def _file_extend(mut dst: List[Int], src: List[Int]):
    for i in range(len(src)):  # driver-loop: over description / option words
        dst.append(src[i])


def sequential_file_digest(entries: PythonObject, address: Int, nbytes: Int) raises -> String:
    var text = List[UInt8](length=SEQUENTIAL_FILE_TEXT, fill=0)
    var addresses: List[Int] = [address, Int(text.unsafe_ptr())]
    var ints: List[Int] = [nbytes, SEQUENTIAL_FILE_TEXT]
    var floats = List[Float64]()
    var count = _file_call(entries, MODEL_VERIFY_DIGEST, addresses, ints, floats)
    return _file_text(text, 0, count)


def sequential_file_fill(entries: PythonObject, address: Int, count: Int, seed: Int, stream: Int) raises -> Int:
    var addresses: List[Int] = [address]
    var ints: List[Int] = [count, seed, stream]
    var floats = List[Float64]()
    return _file_call(entries, MODEL_VERIFY_FILL, addresses, ints, floats)


def sequential_file_sha256(entries: PythonObject, file: String) raises -> Tuple[String, Int]:
    """(64 hex digits, the file's byte count)."""
    var path = file.as_bytes()
    var text = List[UInt8](length=SEQUENTIAL_FILE_TEXT, fill=0)
    var addresses: List[Int] = [Int(path.unsafe_ptr()), Int(text.unsafe_ptr())]
    var ints: List[Int] = [len(path), SEQUENTIAL_FILE_TEXT]
    var floats = List[Float64]()
    var size = _file_call(entries, MODEL_VERIFY_FILE_SHA256, addresses, ints, floats)
    _ = path
    _ = file
    return (_file_text(text, 0, 64), size)


def sequential_file_model_info(entries: PythonObject, file: String) raises -> Tuple[String, String]:
    """(kind, description hash) of a saved model file."""
    var path = file.as_bytes()
    var text = List[UInt8](length=SEQUENTIAL_FILE_TEXT, fill=0)
    var lengths = List[Int](length=2, fill=0)
    var addresses: List[Int] = [Int(path.unsafe_ptr()), Int(text.unsafe_ptr()), Int(lengths.unsafe_ptr())]
    var ints: List[Int] = [len(path), SEQUENTIAL_FILE_TEXT]
    var floats = List[Float64]()
    _ = _file_call(entries, MODEL_VERIFY_MODEL_INFO, addresses, ints, floats)
    _ = path
    _ = file
    var kind = _file_text(text, 0, lengths[0])
    var description_hash = _file_text(text, lengths[0], lengths[1])
    return (kind^, description_hash^)


def sequential_file_save(entries: PythonObject, file: String, buffers: List[Int], program: List[Int],
                         ip: List[Int], fp: List[Float64], classes: List[Int]) raises -> Int:
    """Write the checkpoint; `buffers` = [params, m, v, state, curve].
    Returns the file's byte count."""
    if len(buffers) != 5:
        raise Error("RuntimeError: Sequential checkpoint save takes five buffers")
    var path = file.as_bytes()
    var addresses: List[Int] = [Int(path.unsafe_ptr())]
    _file_extend(addresses, buffers)
    var ints: List[Int] = [len(path), len(program), len(ip), len(classes)]
    _file_extend(ints, program)
    _file_extend(ints, ip)
    _file_extend(ints, classes)
    var size = _file_call(entries, MODEL_SAVE_CHECKPOINT, addresses, ints, fp)
    _ = path
    _ = file
    return size


def _file_info_call(entries: PythonObject, file: String, layers: Int, mut words: List[Int],
                    mut scalars: List[Float64]) raises -> Int:
    var path = file.as_bytes()
    words = List[Int](length=4 + (3 + SEQUENTIAL_FILE_ARGS) * layers + SEQUENTIAL_FILE_INFO_INTS
                      + SEQUENTIAL_FILE_SAMPLE_WORDS, fill=0)
    scalars = List[Float64](length=layers + SEQUENTIAL_FILE_INFO_FLOATS, fill=0.0)
    var addresses: List[Int] = [Int(path.unsafe_ptr()), Int(words.unsafe_ptr()), Int(scalars.unsafe_ptr())]
    var ints: List[Int] = [len(path), len(words), len(scalars)]
    var floats = List[Float64]()
    var count = _file_call(entries, MODEL_CHECKPOINT_INFO, addresses, ints, floats)
    _ = path
    _ = file
    _ = len(words)
    _ = len(scalars)
    return count


def sequential_file_info(entries: PythonObject, file: String) raises -> SequentialCheckpointInfo:
    var words = List[Int]()
    var scalars = List[Float64]()
    var count = _file_info_call(entries, file, SEQUENTIAL_FILE_LAYERS, words, scalars)
    if count < 0:
        count = _file_info_call(entries, file, -count - 1, words, scalars)
    if count < 0 or words[1] != count:
        raise Error("RuntimeError: Sequential checkpoint description changed while it was read")
    var n_ints = words[2]
    var n_floats = words[3]
    if 4 + 3 * count + n_ints > len(words) or count + n_floats > len(scalars):
        raise Error("RuntimeError: Sequential checkpoint description runs past its buffer")
    var info = SequentialCheckpointInfo()
    info.is_export = words[0] != 0
    for i in range(3 * count):  # driver-loop: over layer description words
        info.layer_ints.append(words[4 + i])
    for i in range(n_ints):  # driver-loop: over checkpoint option words
        info.ints.append(words[4 + 3 * count + i])
    var at = 4 + 3 * count + n_ints
    if at + SEQUENTIAL_FILE_ARGS * count + 1 > len(words):
        raise Error("RuntimeError: Sequential checkpoint description runs past its buffer")
    for i in range(SEQUENTIAL_FILE_ARGS * count):  # driver-loop: over layer argument words
        info.layer_args.append(words[at + i])
    var rank = words[at + SEQUENTIAL_FILE_ARGS * count]
    if rank < 0 or rank >= SEQUENTIAL_FILE_SAMPLE_WORDS or at + SEQUENTIAL_FILE_ARGS * count + 1 + rank > len(words):
        raise Error("RuntimeError: Sequential checkpoint sample shape is damaged")
    for k in range(rank):  # small-loop(at most eight sample axes)
        info.sample.append(words[at + SEQUENTIAL_FILE_ARGS * count + 1 + k])
    for i in range(count):  # driver-loop: over layer scalars
        info.layer_floats.append(scalars[i])
    for i in range(n_floats):  # driver-loop: over checkpoint option floats
        info.floats.append(scalars[count + i])
    return info^


def sequential_file_load(entries: PythonObject, file: String, buffers: List[Int]) raises -> Int:
    """Fill the caller's buffers; `buffers` = [params, m, v, state, curve,
    classes] (every one but params may be 0). Returns n_params."""
    if len(buffers) != 6:
        raise Error("RuntimeError: Sequential checkpoint load takes six buffers")
    var path = file.as_bytes()
    var addresses: List[Int] = [Int(path.unsafe_ptr())]
    _file_extend(addresses, buffers)
    var ints: List[Int] = [len(path)]
    var floats = List[Float64]()
    var count = _file_call(entries, MODEL_LOAD_CHECKPOINT, addresses, ints, floats)
    _ = path
    _ = file
    return count


def sequential_file_counts(entries: PythonObject, program: List[Int], table: List[Float64]) raises -> Tuple[Int, Int]:
    """(parameter count, output width) of a description (validates it)."""
    var counts = List[Int](length=2, fill=0)
    var addresses: List[Int] = [Int(counts.unsafe_ptr())]
    _ = _file_call(entries, MODEL_N_PARAMS, addresses, program, table)
    return (counts[0], counts[1])


def sequential_file_import(entries: PythonObject, program: List[Int], table: List[Float64], records: List[Int],
                           names: List[String], params: Int) raises -> Tuple[Int, Int, Int]:
    """Sequential.load_state_dict's transfer into the float32 buffer at
    `params`. Returns (parameter count, output width, input values per
    sample)."""
    var text = List[UInt8]()
    var ints: List[Int] = [len(program), len(records), len(names)]
    _file_extend(ints, program)
    _file_extend(ints, records)
    for k in range(len(names)):  # driver-loop: over source tensor names
        var bytes = names[k].as_bytes()
        for i in range(len(bytes)):  # driver-loop: over the bytes of one tensor name
            text.append(bytes[i])
        ints.append(len(bytes))
    if len(text) == 0:
        text.append(0)
    var counts = List[Int](length=3, fill=0)
    var addresses: List[Int] = [params, Int(text.unsafe_ptr()), Int(counts.unsafe_ptr())]
    _ = _file_call(entries, MODEL_IMPORT_PARAMS, addresses, ints, table)
    _ = len(text)
    return (counts[0], counts[1], counts[2])


def sequential_file_export(entries: PythonObject, program: List[Int], table: List[Float64], params: Int,
                           dst: Int, mut shapes: List[Int]) raises -> Int:
    """Sequential.state_dict's transfer: the float32 parameters at `params`
    into the float32 buffer at `dst` in PyTorch's layouts, and one shape
    record per tensor (autograd/weight_import.mojo: 4 + its rank words, laid
    end to end) into `shapes`. A `shapes` too short for the model's ranks is
    grown to the words the export reports and the export asked once more
    (nothing was written the first time). Returns the tensor count."""
    var count = _file_export_once(entries, program, table, params, dst, shapes)
    if count < 0:
        shapes = List[Int](length=-count, fill=0)
        count = _file_export_once(entries, program, table, params, dst, shapes)
        if count < 0:
            raise Error("RuntimeError: Sequential state_dict shape result is still too small")
    return count


def _file_export_once(entries: PythonObject, program: List[Int], table: List[Float64], params: Int,
                      dst: Int, mut shapes: List[Int]) raises -> Int:
    var ints: List[Int] = [len(program), len(shapes)]
    _file_extend(ints, program)
    var addresses: List[Int] = [params, dst, Int(shapes.unsafe_ptr())]
    var count = _file_call(entries, MODEL_EXPORT_PARAMS, addresses, ints, table)
    _ = len(shapes)
    return count
