# SPDX-License-Identifier: Apache-2.0
# The Sequential model-file operations of the whole-model descriptor
# (sequence/neural_model_dispatch.mojo MODEL_SAVE_CHECKPOINT ..
# MODEL_IMPORT_PARAMS), shared by the host and the device dispatch. RUN only:
# typed words in, typed words out, no Python object and no Python import.
#
# Conventions of these nine operations (the descriptor carries words only):
#   text in    (a file path, tensor names): the address of its UTF-8 bytes in
#              `addresses`, its byte count in `ints`.
#   text out   (a digest, a hash, a kind name): the caller's byte buffer in
#              `addresses`, its capacity in `ints`; the operation writes the
#              bytes (no terminator) and reports the count.
#   lists out  (checkpoint info, counts): the caller's word / float64 buffers
#              in `addresses`, their capacities in `ints`.
# Per operation (aa = addresses, pp = ints, ff = floats; `result` is the one
# word the descriptor returns):
#   MODEL_SAVE_CHECKPOINT   aa [path, params, m, v, state, curve]
#                           pp [path bytes, n_prog, n_ip, n_classes, prog...,
#                               ip..., classes descriptor...]   ff fp
#                           result = the file's byte count
#   MODEL_CHECKPOINT_INFO   aa [path, ints out, floats out]
#                           pp [path bytes, ints capacity, floats capacity]
#                           ints out = [export, L, n_ints, n_floats,
#                             3 words per layer..., the option words...];
#                           floats out = [one per layer..., the option floats...]
#                           result = L, or -(L + 1) with nothing written when
#                           a capacity is too small (the caller asks again)
#   MODEL_LOAD_CHECKPOINT   aa [path, params, m, v, state, curve, classes]
#                           pp [path bytes]     result = n_params
#   MODEL_VERIFY_DIGEST     aa [buffer, text out]  pp [byte count, capacity]
#                           result = text bytes written
#   MODEL_VERIFY_FILL       aa [buffer]  pp [count, seed, stream(, id range)]
#                           (an id range n > 0: ids in [0, n), model_verify.mojo)
#   MODEL_VERIFY_FILE_SHA256  aa [path, text out]  pp [path bytes, capacity]
#                           result = the file's byte count (text = 64 digits)
#   MODEL_VERIFY_MODEL_INFO aa [path, text out, lengths out (2 words)]
#                           pp [path bytes, capacity]; text = kind then hash
#                           result = text bytes written
#   MODEL_N_PARAMS          aa [counts out (2 words)]  pp prog  ff table
#                           counts = [parameter count, output width]
#   MODEL_IMPORT_PARAMS     aa [params, names bytes, counts out (3 words)]
#                           pp [n_prog, n_recs, n_names, prog..., recs...,
#                               one byte count per name...]   ff table
#                           counts = [parameter count, output width, inputs]
#   MODEL_EXPORT_PARAMS     aa [params, dst (n_params floats), shapes out]
#                           pp [n_prog, shapes capacity, prog...]  ff table
#                           result = tensor count (8 shape words each)
# Not split across cores: each operation is one file read or write, one
# digest fold in a fixed order, or one weight transfer; none loops over
# independent units.
from core.native_descriptor_call import native_descriptor_words
from autograd.arg_words import fptr_word
from autograd.model import parse_model
from autograd.model_execution import model_counts_native, model_float_words
from autograd.model_checkpoint import model_save_checkpoint, model_checkpoint_info, model_load_checkpoint
from autograd.model_verify import verify_digest, verify_fill, verify_file_sha256, verify_model_info
from autograd.weight_import import model_import_params, model_export_params
from sequence.exec_trait import Exec
from sequence.neural_model_dispatch import (
    MODEL_SAVE_CHECKPOINT, MODEL_CHECKPOINT_INFO, MODEL_LOAD_CHECKPOINT, MODEL_VERIFY_DIGEST,
    MODEL_VERIFY_FILL, MODEL_VERIFY_FILE_SHA256, MODEL_VERIFY_MODEL_INFO, MODEL_N_PARAMS,
    MODEL_IMPORT_PARAMS,
)


def model_file_operation(op: Int) -> Bool:
    """True for the eight operations `model_file_execute` serves (every
    model-file operation but the weight import, which takes an executor)."""
    return op >= MODEL_SAVE_CHECKPOINT and op < MODEL_IMPORT_PARAMS


def _file_floats(ff: Int, nf: Int) raises -> List[Float64]:
    if nf < 0 or (nf > 0 and ff == 0):
        raise Error("native model file: invalid scalar span")
    var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
    var result = List[Float64](capacity=nf)
    for i in range(nf):  # driver-loop: over model description scalars
        result.append(source[i])
    return result^


def _file_text(address: Int, count: Int) raises -> String:
    if count < 0 or (count > 0 and address == 0):
        raise Error("native model file: null or invalid text span")
    var source = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    var bytes = List[UInt8](capacity=count)
    for i in range(count):  # driver-loop: over the bytes of one path or name
        bytes.append(source[i])
    return String(StringSlice(unsafe_from_utf8=Span(bytes)))


def _file_put_text(text: String, address: Int, at: Int, capacity: Int) raises -> Int:
    """Write `text`'s bytes at `address + at`; returns the byte count."""
    var bytes = text.as_bytes()
    var count = len(bytes)
    if address == 0 or at < 0 or at + count > capacity:
        raise Error("native model file: the text result buffer is too small")
    var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(count):  # driver-loop: over the bytes of one digest or kind name
        dst[at + i] = bytes[i]
    return count


def _file_words(words: List[Int], start: Int, count: Int) raises -> List[Int]:
    if start < 0 or count < 0 or start + count > len(words):
        raise Error("native model file: a word list runs past the descriptor")
    var result = List[Int](capacity=count)
    for i in range(count):  # driver-loop: over description / option words
        result.append(words[start + i])
    return result^


def _file_word_out(address: Int) raises -> MutPointer[Int, MutUntrackedOrigin]:
    if address == 0:
        raise Error("native model file: null word result buffer")
    return MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=address)


def model_file_execute(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int) raises -> Int:
    var addresses = native_descriptor_words(aa, na)
    var ints = native_descriptor_words(pp, np)
    if op == MODEL_SAVE_CHECKPOINT:
        # lane stream-fits: a seventh address, centered RMSprop's third optimizer state buffer
        if (na != 6 and na != 7) or np < 4:
            raise Error("native model file: invalid checkpoint save descriptor")
        var n_prog = ints[1]
        var n_ip = ints[2]
        var n_classes = ints[3]
        if n_prog < 0 or n_ip < 0 or n_classes < 0 or np != 4 + n_prog + n_ip + n_classes:
            raise Error("native model file: invalid checkpoint save descriptor")
        var file = _file_text(addresses[0], ints[0])
        var buffers = _file_words(addresses, 1, na - 1)
        return model_save_checkpoint(file, _file_words(ints, 4, n_prog), buffers,
                                     _file_words(ints, 4 + n_prog, n_ip), _file_floats(ff, nf),
                                     _file_words(ints, 4 + n_prog + n_ip, n_classes))
    if op == MODEL_CHECKPOINT_INFO:
        if na != 3 or np != 3 or nf != 0:
            raise Error("native model file: invalid checkpoint info descriptor")
        var info = model_checkpoint_info(_file_text(addresses[0], ints[0]))
        var layer_count = len(info.layer_floats)
        var n_ints = len(info.ints)
        var n_floats = len(info.floats)
        # lane nn-integrate: after the option words, SPEC2_ARGS argument words per layer, then the sample rank
        # (0: a 2-D model) and its axes
        var n_sample = len(info.sample)
        var n_args = len(info.layer_args)
        if ints[1] < 4 + 3 * layer_count + n_ints + n_args + 1 + n_sample or ints[2] < layer_count + n_floats:
            return -(layer_count + 1)
        var words = _file_word_out(addresses[1])
        if addresses[2] == 0:
            raise Error("native model file: null scalar result buffer")
        var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=addresses[2])
        words[0] = 1 if info.is_export else 0
        words[1] = layer_count
        words[2] = n_ints
        words[3] = n_floats
        for i in range(3 * layer_count):  # driver-loop: over layer description words
            words[4 + i] = info.layer_ints[i]
        for i in range(n_ints):  # driver-loop: over checkpoint option words
            words[4 + 3 * layer_count + i] = info.ints[i]
        var at = 4 + 3 * layer_count + n_ints
        for i in range(n_args):  # driver-loop: over layer argument words
            words[at + i] = info.layer_args[i]
        words[at + n_args] = n_sample
        for k in range(n_sample):  # small-loop(at most RN_MAX_RANK - 1 sample axes)
            words[at + n_args + 1 + k] = info.sample[k]
        for i in range(layer_count):  # driver-loop: over layer scalars
            scalars[i] = info.layer_floats[i]
        for i in range(n_floats):  # driver-loop: over checkpoint option floats
            scalars[layer_count + i] = info.floats[i]
        return layer_count
    if op == MODEL_LOAD_CHECKPOINT:
        # lane stream-fits: an eighth address, centered RMSprop's third optimizer state buffer
        if (na != 7 and na != 8) or np != 1 or nf != 0:
            raise Error("native model file: invalid checkpoint load descriptor")
        var file = _file_text(addresses[0], ints[0])
        return model_load_checkpoint(file, _file_words(addresses, 1, na - 1))
    if op == MODEL_VERIFY_DIGEST:
        if na != 2 or np != 2 or nf != 0:
            raise Error("native model file: invalid digest descriptor")
        return _file_put_text(verify_digest(addresses[0], ints[0]), addresses[1], 0, ints[1])
    if op == MODEL_VERIFY_FILL:
        if na != 1 or (np != 3 and np != 4) or nf != 0:
            raise Error("native model file: invalid fill descriptor")
        return verify_fill(addresses[0], ints[0], ints[1], ints[2], ints[3] if np == 4 else 0)
    if op == MODEL_VERIFY_FILE_SHA256:
        if na != 2 or np != 2 or nf != 0:
            raise Error("native model file: invalid file hash descriptor")
        var hashed = verify_file_sha256(_file_text(addresses[0], ints[0]))
        if _file_put_text(hashed[0], addresses[1], 0, ints[1]) != 64:
            raise Error("native model file: a SHA-256 is 64 hex digits")
        return hashed[1]
    if op == MODEL_VERIFY_MODEL_INFO:
        if na != 3 or np != 2 or nf != 0:
            raise Error("native model file: invalid model info descriptor")
        var described = verify_model_info(_file_text(addresses[0], ints[0]))
        var lengths = _file_word_out(addresses[2])
        var kind_bytes = _file_put_text(described[0], addresses[1], 0, ints[1])
        var hash_bytes = _file_put_text(described[1], addresses[1], kind_bytes, ints[1])
        lengths[0] = kind_bytes
        lengths[1] = hash_bytes
        return kind_bytes + hash_bytes
    if op == MODEL_N_PARAMS:
        if na != 1:
            raise Error("native model file: invalid parameter count descriptor")
        var counts = model_counts_native(ints, _file_floats(ff, nf))
        var counted = _file_word_out(addresses[0])
        counted[0] = counts[0]
        counted[1] = counts[1]
        return counts[0]
    raise Error("native model: unknown operation")


def model_import_execute[E: Exec](mut ex: E, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int) raises -> Int:
    """MODEL_IMPORT_PARAMS: Sequential.load_state_dict's transfer
    (autograd/weight_import.mojo) on the image's executor."""
    var addresses = native_descriptor_words(aa, na)
    var ints = native_descriptor_words(pp, np)
    if na != 3 or np < 3:
        raise Error("native model file: invalid weight import descriptor")
    var n_prog = ints[0]
    var n_recs = ints[1]
    var n_names = ints[2]
    if n_prog < 0 or n_recs < 0 or n_names < 0 or np != 3 + n_prog + n_recs + n_names:
        raise Error("native model file: invalid weight import descriptor")
    var names = List[String]()
    var at = 0
    for k in range(n_names):  # driver-loop: over source tensor names
        var count = ints[3 + n_prog + n_recs + k]
        if count < 0:
            raise Error("native model file: invalid weight import descriptor")
        names.append(_file_text(addresses[1] + at, count))
        at += count
    var spec = parse_model(_file_words(ints, 3, n_prog), model_float_words(_file_floats(ff, nf)))
    model_import_params(ex, spec, _file_words(ints, 3 + n_prog, n_recs), names, fptr_word(addresses[0], "params"))
    var counted = _file_word_out(addresses[2])
    counted[0] = spec.n_params
    counted[1] = spec.out_features()
    counted[2] = spec.n_inputs
    return spec.n_params


def model_export_execute[E: Exec](mut ex: E, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int) raises -> Int:
    """MODEL_EXPORT_PARAMS: Sequential.state_dict's transfer out of params_
    (autograd/weight_import.mojo model_export_params) on the image's
    executor."""
    var addresses = native_descriptor_words(aa, na)
    var ints = native_descriptor_words(pp, np)
    if na != 3 or np < 2:
        raise Error("native model file: invalid weight export descriptor")
    var n_prog = ints[0]
    if n_prog < 0 or np != 2 + n_prog:
        raise Error("native model file: invalid weight export descriptor")
    var spec = parse_model(_file_words(ints, 2, n_prog), model_float_words(_file_floats(ff, nf)))
    return model_export_params(ex, spec, addresses[0], addresses[1], addresses[2], ints[1])
