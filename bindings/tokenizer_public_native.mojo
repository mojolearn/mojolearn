# SPDX-License-Identifier: Apache-2.0
# Whole-operation tokenizer admission, orchestration and result conversion.
# Public wrappers connect once at import. No user iterator, scalar conversion,
# descriptor, path or buffer callback is used to admit product inputs.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr, GILReleased
from std.python.bindings import ExceptionType, raise_python_exception
from std.os import getenv
from std.ffi import _Global
from std.memory import memcpy
from bindings.native_loaded_module import native_loaded_module
from bindings.native_instance_item import native_instance_item
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.native_json import native_json_read, native_json_parse, native_path, native_path_exists
from bindings.native_error_class import error_without_class
from bindings.optimizer_public_words import stash_python_error, optimizer_python_error, PYTHON_ERROR_PENDING
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from providers.python.borrowed_buffer_pyapi import admit_native_buffer_exporter, AdmittedBorrowedBuffer
from providers.python.autograd_program_arrays_pyapi import is_integer_input
from tokenizer.encoding import GPT2_ENDOFTEXT, GPT2_PAT_STR, BpeTokenizer, string_bytes
from tokenizer.impl.ranks import RankTable
from tokenizer.impl.byte_unicode import codepoint_to_byte
from tokenizer.impl.pretokenize import pretokenize_pattern
from tokenizer.impl.unicode_class import builtin_unicode_classes, UNICODE_VERSION_PINNED
from tokenizer.impl.vocab_build import MergeCheck, check_loose, table_from_tokens, ranks_text, unspell_error
from tokenizer.train.bpe_train import TrainedVocabulary
from tokenizer.train.emit import render_ranks, render_tokenizer_json
from tokenizer.vocab import VocabTokenizer
from core.sha256 import sha256_hex
from tokenizer.impl.tokenizer_host_shell_impl import (
    BpeHandle, BpeTrainedHandle, VocabHandle, _handle_from_table,
    bpe_handle_load, bpe_encode_into, bpe_encode_batch_into, bpe_check_ids,
    bpe_decode_seq_bytes, bpe_train_fill, trained_vocab_settle,
    tokenizer_append_bytes, vocab_check_pairs, vocab_encode_ids, vocab_decode_ids,
    vocab_fingerprint_hex, SpelledVocab, SpelledMerges, SPELLED_OK,
    SPELLED_BAD_ID, SPELLED_NOT_STR, SPELLED_RAISED, SPELLED_NOT_PAIR,
    SPELLED_NOT_LEVEL, spelled_vocab_tokens, spelled_merges_text, spelled_merges_list,
)


struct TokenizerPublicConnection(Defaultable, Movable):
    var vocab: PythonObject

    def __init__(out self):
        self.vocab = PythonObject(None)


comptime TOKENIZER_PUBLIC_CONNECTION = _Global[
    StorageType=TokenizerPublicConnection, name="MojoTokenizerPublicConnection",
    init_fn=TokenizerPublicConnection.__init__,
]


def _vocab_context() raises -> PythonObject:
    var context = TOKENIZER_PUBLIC_CONNECTION.get_or_create_ptr()[].vocab
    if context is None:
        raise Error("RuntimeError: tokenizer requires its import-time native connection")
    return context


def _int(value: PythonObject, name: String) raises -> Int:
    if py_type(value) is not native_loaded_module("builtins").int:
        raise Error("TypeError: tokenizer " + name + " requires an exact int; Python scalar callbacks are unsupported")
    return Int(py=value)


def _bool(value: PythonObject, name: String) raises -> Bool:
    if py_type(value) is not native_loaded_module("builtins").bool:
        raise Error("TypeError: tokenizer " + name + " must be a bool, got " + _type_name(value))
    return Bool(py=value)


def _str(value: PythonObject, name: String) raises -> String:
    if py_type(value) is not native_loaded_module("builtins").str:
        raise Error("TypeError: tokenizer " + name + " requires an exact str")
    return String(py=value)


def _type_name(value: PythonObject) raises -> String:
    var bi = native_loaded_module("builtins")
    return String(py_call(bi.type.__dict__["__name__"].__get__, py_type(value)))


def _sequence(value: PythonObject, name: String) raises:
    var bi = native_loaded_module("builtins")
    if py_type(value) is not bi.list and py_type(value) is not bi.tuple:
        raise Error("TypeError: tokenizer " + name + " requires a list or tuple; materialize Python iterators before the call")


def _documents(value: PythonObject, name: String) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var kind = py_type(value)
    if kind is bi.list or kind is bi.tuple:
        return value
    # Exact builtin iterators walk native container storage; a user iterator,
    # generator, list subclass or __length_hint__ override is never admitted.
    if (kind is py_type(py_call(bi.iter, Python.list()))
            or kind is py_type(py_call(bi.iter, Python.tuple()))
            or kind is py_type(py_call(bi.iter, py_call(bi.range, 0)))
            or kind is py_type(py_call(bi.reversed, Python.list()))):
        return py_call(bi.list, value)
    if name == "encode_batch":
        raise Error("TypeError: encode_batch takes a sequence of documents, got " + _type_name(value) + "; use native containers")
    if name == "decode_bytes_batch":
        raise Error("TypeError: decode_bytes_batch takes a sequence of id sequences, got " + _type_name(value) + "; use native containers")
    if name == "tokens":
        raise Error("TypeError: tokens must be a sequence of bytes; use native containers")
    raise Error("TypeError: " + name + " requires native containers; materialize Python iterators before the call")


def _dictionary(value: PythonObject) raises:
    var bi = native_loaded_module("builtins")
    if py_type(value) is not bi.dict:
        raise Error("TypeError: tokenizer metadata requires an exact dictionary")
    var keys = py_call(bi.list, value.keys())
    for i in range(len(keys)):  # driver-loop: over admitted metadata dictionary key names
        _ = _str(keys[i], "dictionary key")


def _plain(value: PythonObject, depth: Int = 0) raises:
    # Validate the complete JSON/configuration tree before any lookup,
    # formatting, truth conversion or equality can invoke caller code.
    if depth > 1000:
        raise Error("ValueError: tokenizer metadata nesting limit exceeded")
    var bi = native_loaded_module("builtins")
    var kind = py_type(value)
    if value is None or kind is bi.str or kind is bi.int or kind is bi.float or kind is bi.bool:
        return
    if kind is bi.dict:
        _dictionary(value)
        var values = py_call(bi.list, value.values())
        for i in range(len(values)):  # driver-loop: over native JSON configuration members before conversion
            _plain(values[i], depth + 1)
        return
    _sequence(value, "metadata")
    for i in range(len(value)):  # driver-loop: over native JSON configuration members before conversion
        _plain(value[i], depth + 1)


def _owner(value: PythonObject, cls: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    if py_type(cls) is not bi.type or py_type(value) is not cls:
        raise Error("TypeError: tokenizer requires its connected class; Python subclass protocols are unsupported")
    var bases = py_call(bi.type.__dict__["__bases__"].__get__, cls)
    if len(bases) != 1 or bases[0] is not bi.object:
        raise Error("TypeError: tokenizer class ancestry changed after connection")
    var namespace = _class_fields(cls)
    if py_contains(namespace, "__getattribute__") or py_contains(namespace, "__getattr__") or py_contains(namespace, "__setattr__"):
        raise Error("TypeError: tokenizer attribute hooks are unsupported")
    # _fields checks a replaced __dict__ descriptor BEFORE storage access.
    var fields = _fields(value)
    var names = py_call(bi.list, fields.keys())
    for name in [String("_handle"), String("_m"), String("_source"), String("_n_vocab"), String("_n_ranks"), String("_identity"), String("_max_token_bytes"), String("_fast"), String("tokens"), String("merges"), String("stats"), String("vocab_size"), String("min_frequency"), String("backend"), String("pattern"), String("normalizer"), String("source"), String("added"), String("_added_by_id"), String("bos_ids"), String("eos_ids"), String("bos_token_id"), String("eos_token_id")]:  # driver-loop: over native class and instance field metadata before access
        _ = py_call(names.append, PythonObject(name))
    for i in range(len(names)):  # driver-loop: over native class and instance field metadata before access
        if py_contains(namespace, names[i]):
            var descriptor = namespace[names[i]]
            var slots = _class_fields(py_type(descriptor))
            if py_contains(slots, "__get__") or py_contains(slots, "__set__"):
                raise Error("TypeError: tokenizer state field was replaced by a descriptor")
    return fields


def _new(cls: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    if py_type(cls) is not bi.type:
        raise Error("TypeError: tokenizer class requires an ordinary connected type")
    return py_call(bi.object.__new__, cls)


def _native(entry: PythonObject) raises:
    if py_type(entry) is not py_type(native_loaded_module("builtins").len):
        raise Error("TypeError: tokenizer provider entry must be native")


def _bytes(value: PythonObject, allow_text: Bool = True, name: String = "text") raises -> List[UInt8]:
    var raw = value
    if py_type(raw) is native_loaded_module("builtins").str:
        if not allow_text:
            raise Error("TypeError: encode_bytes takes bytes; use encode for str")
        # Exact str uses CPython's native UTF-8 conversion, with no overridable
        # encode attribute or arbitrary codec name.
        ref cpy = Python().cpython()
        var pointer = cpy.lib.get_function[PyObjectPtr]("PyUnicode_AsUTF8String")(raw._obj_ptr)
        if not pointer:
            raise stash_python_error()
        raw = PythonObject(from_owned=pointer)
    try:
        admit_native_buffer_exporter(raw)
    except:
        raise Error("TypeError: " + name + " must be str or bytes-like, got " + _type_name(raw) + "; use exact native buffers without Python hooks")
    var pin = AdmittedBorrowedBuffer(raw)
    var count = pin.layout.value().count * pin.layout.value().itemsize
    var result = List[UInt8](length=count, fill=0)
    try:
        pin.layout.value().copy_to(Int(result.unsafe_ptr()))
    except error:
        pin.release()
        raise error
    pin.release()
    _ = raw
    return result^


def _py_bytes(values: List[UInt8]) raises -> PythonObject:
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyBytes_FromStringAndSize")(values.unsafe_ptr(), len(values))
    _ = values
    if not raw:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=raw)


def _py_int_list(values: List[Int]) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(values)):  # cpu-route: publish completed native token ids as Python integers
        _ = py_call(out.append, values[i])
    return out


def _ids(var value: PythonObject) raises -> List[Int]:
    var bi = native_loaded_module("builtins")
    var result = List[Int]()
    var kind = py_type(value)
    if (kind is py_type(py_call(bi.iter, Python.list()))
            or kind is py_type(py_call(bi.iter, Python.tuple()))
            or kind is py_type(py_call(bi.iter, py_call(bi.range, 0)))):
        value = py_call(bi.list, value)
    if py_type(value) is bi.list or py_type(value) is bi.tuple:
        for i in range(len(value)):  # cpu-route: capture admitted native integer input storage without scalar callbacks
            var item = value[i]
            if py_type(item) is bi.bool:
                raise Error("TypeError: ids must be int, not bool, at position " + String(i))
            if not is_integer_input(item):
                raise Error("TypeError: ids must be int, got " + _type_name(item) + " at position " + String(i))
            result.append(Int(py=py_call(bi.int, item)))
        return result^
    # Ordinary integer array/memoryview/ndarray inputs retain their native
    # buffer semantics. Metadata and owner are captured once, before reading.
    if py_type(value) is bi.str or py_type(value) is bi.bytes or py_type(value) is bi.bytearray:
        raise Error("TypeError: ids must be a sequence of int, got " + _type_name(value))
    try:
        admit_native_buffer_exporter(value)
    except:
        raise Error("TypeError: ids must be a sequence of int, got " + _type_name(value) + "; use native integer buffers")
    var pin = AdmittedBorrowedBuffer(value)
    try:
        var layout = pin.layout.value()
        if len(layout.shape) != 1:
            raise Error("TypeError: tokenizer ids require a one-dimensional integer buffer")
        var format = pin.format
        var big_endian = format.startswith(">") or format.startswith("!")
        if format.startswith("@") or format.startswith("=") or format.startswith("<") or big_endian:
            format = String(format[byte=1:])
        if format != "b" and format != "B" and format != "h" and format != "H" and format != "i" and format != "I" and format != "l" and format != "L" and format != "q" and format != "Q" and format != "n" and format != "N":
            raise Error("TypeError: tokenizer ids require a native integer buffer")
        var packed = List[UInt8](length=layout.count * layout.itemsize, fill=0)
        layout.copy_to(Int(packed.unsafe_ptr()))
        for i in range(layout.count):  # cpu-route: capture admitted native integer input storage without scalar callbacks
            var bits = UInt64(0)
            if layout.itemsize != 1 and layout.itemsize != 2 and layout.itemsize != 4 and layout.itemsize != 8:
                raise Error("TypeError: tokenizer integer buffer is wider than 64 bits")
            for byte in range(layout.itemsize):  # cpu-route: capture admitted native integer input storage without scalar callbacks
                var source = layout.itemsize - 1 - byte if big_endian else byte
                bits |= UInt64(packed[i * layout.itemsize + source]) << (8 * byte)
            var signed = format == "b" or format == "h" or format == "i" or format == "l" or format == "q" or format == "n"
            if signed and layout.itemsize < 8 and (bits & (UInt64(1) << (layout.itemsize * 8 - 1))) != 0:
                bits |= UInt64(0xffffffffffffffff) << (layout.itemsize * 8)
            if not signed and bits > UInt64(9223372036854775807):
                raise Error("ValueError: tokenizer id exceeds the supported integer range")
            result.append(Int(bits))
    except error:
        pin.release()
        raise error
    pin.release()
    return result^


def _text_result(raw: PythonObject, errors: PythonObject) raises -> PythonObject:
    var name = _str(errors, "errors")
    # Arbitrary codec error handlers are callbacks. Admit only CPython's
    # builtin handlers, retained by the import-time connection.
    var codecs = native_loaded_module("_codecs")
    var entry = py_call(codecs.lookup_error, errors)
    _native(entry)
    if name != "strict" and name != "ignore" and name != "replace" and name != "backslashreplace" and name != "surrogateescape" and name != "surrogatepass":
        raise Error("ValueError: tokenizer decode requires a builtin UTF-8 error policy")
    ref cpy = Python().cpython()
    var args = Python.tuple("utf-8", errors)
    var decode = raw.decode
    var result = cpy.PyObject_Call(decode._obj_ptr, args._obj_ptr, PyObjectPtr())
    if not result:
        raise stash_python_error()
    return PythonObject(from_owned=result)


def _path(value: PythonObject) raises -> PythonObject:
    var path = native_path(value)
    if not String(path).startswith("/"):
        var getcwd = native_loaded_module("os").getcwd
        _native(getcwd)
        path = py_call(getcwd) + PythonObject("/") + path
    return path


def _identity(text: String, ranks: Int) raises -> PythonObject:
    var result = Python.dict()
    py_setitem(result, "schema", PythonObject("mojolearn.bpe-vocabulary.v1"))
    py_setitem(result, "sha256", PythonObject(sha256_hex(string_bytes(text))))
    py_setitem(result, "n_ranks", PythonObject(ranks))
    py_setitem(result, "n_vocab", PythonObject(ranks + 1))
    py_setitem(result, "endoftext_id", PythonObject(ranks))
    return result


def _pack_tokens(tokens: PythonObject, mut arena: List[UInt8], mut lengths: List[Int]) raises:
    var values = _documents(tokens, "tokens")
    for i in range(len(values)):  # cpu-route: marshal admitted native token buffers into rank storage
        var bytes = _bytes(values[i], False)
        lengths.append(len(bytes))
        tokenizer_append_bytes(arena, bytes)


def _fast_selected(context: PythonObject) raises -> Bool:
    # Import connects available providers; native execution captures the
    # mutable process default at the same boundary as the former Python
    # selector. A later set_numeric_mode therefore retains its semantics.
    var backend = context[6]
    _dictionary(backend)
    var selected = _str(native_instance_item(backend, PythonObject("_DEFAULT_MODE")), "default numeric mode")
    if selected != "identical" and selected != "fast":
        raise Error("ValueError: tokenizer numeric mode must be identical or fast")
    return selected == "fast"


def _bpe_adopt(tokenizer: PythonObject, handle: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _owner(tokenizer, context[0])
    _ = _str(source, "source")
    var owner = handle.downcast_value_ptr[BpeHandle]()
    if not owner[].tok:
        raise Error("ValueError: tokenizer handle is not loaded")
    var text = ranks_text(owner[].tok.value().ranks)
    var count = owner[].tok.value().n_vocab()
    var fast = PythonObject(None)
    if _fast_selected(context) and _bool(context[4][2], "connected fast encoder"):
        var entry = context[5][0]
        _native(entry)
        fast = Python.tuple(context[4][0], py_call(entry, PythonObject(text)))
    py_setitem(fields, "_source", source)
    py_setitem(fields, "_m", context[3])
    py_setitem(fields, "_handle", handle)
    py_setitem(fields, "_n_vocab", PythonObject(count))
    py_setitem(fields, "_identity", _identity(text, count - 1))
    py_setitem(fields, "_max_token_bytes", PythonObject(owner[].max_token_bytes))
    py_setitem(fields, "_fast", fast)
    _ = handle
    return PythonObject(None)


def _bpe_tokens(cls: PythonObject, tokens: PythonObject, context: PythonObject) raises -> PythonObject:
    if cls is not context[0]:
        raise Error("TypeError: tokenizer factory requires the connected BpeTokenizer class")
    var arena = List[UInt8]()
    var lengths = List[Int]()
    _pack_tokens(tokens, arena, lengths)
    var table = table_from_tokens(arena, lengths)
    var handle = PythonObject(alloc=_handle_from_table(table^))
    var result = _new(cls)
    _ = _bpe_adopt(result, handle, PythonObject("<token bytes>"), context)
    return result


def _bpe_init(tokenizer: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = _owner(tokenizer, context[0])
    if path is None:
        raise Error("ValueError: BpeTokenizer needs a vocabulary; mojolearn ships none. Use from_ranks_file, from_files or from_token_bytes")
    var source = _path(path)
    if not native_path_exists(source):
        raise Error("FileNotFoundError: BpeTokenizer rank file does not exist: " + String(source))
    var handle = BpeHandle()
    bpe_handle_load(handle, String(source))
    return _bpe_adopt(tokenizer, PythonObject(alloc=handle^), source, context)


def _bpe_encode(tokenizer: PythonObject, data: PythonObject, allow_value: PythonObject, bytes_only: Bool, batch: Bool, context: PythonObject) raises -> PythonObject:
    var fields = _owner(tokenizer, context[0])
    var handle = native_instance_item(fields, PythonObject("_handle"))
    var owner = handle.downcast_value_ptr[BpeHandle]()
    if not owner[].tok:
        raise Error("ValueError: tokenizer handle is not loaded")
    var allow = _bool(allow_value, "allow_endoftext")
    var documents = Python.tuple(data)
    if batch:
        documents = _documents(data, "encode_batch")
    var text = List[UInt8]()
    var lengths = List[Int64]()
    for i in range(len(documents)):  # cpu-route: prepare native document buffers and publish completed token lists
        var bytes = _bytes(documents[i], not bytes_only, "document " + String(i) if batch else "text")
        lengths.append(len(bytes))
        tokenizer_append_bytes(text, bytes)
    var ids = List[Int32](length=max(len(text), 1), fill=0)
    var counts = List[Int64](length=max(len(lengths), 1), fill=0)
    var total = 0
    if not batch:
        if len(text) > 0:
            total = bpe_encode_into(owner[], Int(text.unsafe_ptr()), len(text), Int(ids.unsafe_ptr()), len(text), allow)
            counts[0] = total
    elif len(lengths) > 0:
        # Retain the existing optional fast batch route. The imported module
        # and handle are read from the operation snapshot, never self._m.
        var fast = native_instance_item(fields, PythonObject("_fast"))
        if fast is not None and not allow:
            _sequence(fast, "fast provider")
            if len(fast) != 2 or fast[0] is not context[4][0]:
                raise Error("TypeError: tokenizer fast provider changed after connection")
            var entry = context[5][1]
            _native(entry)
            total = _int(py_call(entry, fast[1], Int(text.unsafe_ptr()), Int(lengths.unsafe_ptr()), Int(ids.unsafe_ptr()), Int(counts.unsafe_ptr()), Python.tuple(len(lengths), len(text), len(text), allow)), "encoded count")
        else:
            total = bpe_encode_batch_into(owner[], Int(text.unsafe_ptr()), Int(lengths.unsafe_ptr()), Int(ids.unsafe_ptr()), Int(counts.unsafe_ptr()), len(lengths), len(text), len(text), allow)
    if total < 0 or total > len(text):
        raise Error("RuntimeError: tokenizer returned an invalid encoded count")
    var result = Python.list()
    var at = 0
    for i in range(len(lengths)):  # cpu-route: prepare native document buffers and publish completed token lists
        var count = Int(counts[i])
        if count < 0 or at + count > total:
            raise Error("RuntimeError: tokenizer returned an invalid document count")
        var row = Python.list()
        for j in range(count):  # cpu-route: prepare native document buffers and publish completed token lists
            _ = py_call(row.append, Int(ids[at + j]))
        at += count
        _ = py_call(result.append, row)
    if at != total:
        raise Error("RuntimeError: tokenizer counts differ from its total")
    _ = text
    _ = lengths
    _ = handle
    _ = data
    return result if batch else result[0]


def _bpe_decode(tokenizer: PythonObject, data: PythonObject, bytes_only: Bool, batch: Bool, errors: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _owner(tokenizer, context[0])
    var handle = native_instance_item(fields, PythonObject("_handle"))
    var owner = handle.downcast_value_ptr[BpeHandle]()
    if not owner[].tok:
        raise Error("ValueError: tokenizer handle is not loaded")
    var sequences = Python.tuple(data)
    if batch:
        sequences = _documents(data, "decode_bytes_batch")
    var result = Python.list()
    for i in range(len(sequences)):  # cpu-route: marshal admitted id sequences and publish completed decoded bytes
        var ids = _ids(sequences[i])
        bpe_check_ids(ids, owner[].tok.value().n_vocab())
        var bytes = bpe_decode_seq_bytes(owner[], ids, len(ids) * owner[].max_token_bytes)
        var raw = _py_bytes(bytes)
        _ = py_call(result.append, raw if bytes_only else _text_result(raw, errors))
    _ = handle
    _ = data
    return result if batch else result[0]


def _trained_value(tokens: PythonObject, merges: PythonObject) raises -> TrainedVocabulary:
    var value = TrainedVocabulary()
    _pack_tokens(tokens, value.arena, value.length)
    _sequence(merges, "merges")
    for i in range(len(merges)):  # cpu-route: capture native trained-vocabulary fields and merge ids
        _sequence(merges[i], "merge ids")
        if len(merges[i]) != 3:
            raise Error("ValueError: trained merges require (left, right, new) ids")
        value.merge_left.append(_int(merges[i][0], "merge left id"))
        value.merge_right.append(_int(merges[i][1], "merge right id"))
        _ = _int(merges[i][2], "merge result id")
    trained_vocab_settle(value, len(value.length), len(merges))
    return value^


def _trained_init(value: PythonObject, tokens: PythonObject, merges: PythonObject, stats: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _owner(value, context[1])
    _ = _trained_value(tokens, merges)
    _dictionary(stats)
    _plain(stats)
    py_setitem(fields, "tokens", tokens)
    py_setitem(fields, "merges", merges)
    py_setitem(fields, "stats", py_call(native_loaded_module("builtins").dict, stats))
    return PythonObject(None)


def _trained_render(value: PythonObject, which: Int, context: PythonObject) raises -> String:
    var fields = _owner(value, context[1])
    var tokens = native_instance_item(fields, PythonObject("tokens"))
    var merges = native_instance_item(fields, PythonObject("merges"))
    var vocab = _trained_value(tokens, merges)
    return render_ranks(vocab) if which == 0 else render_tokenizer_json(vocab, String(GPT2_PAT_STR), String(GPT2_ENDOFTEXT))


def _trainer_init(value: PythonObject, size: PythonObject, frequency: PythonObject, backend: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _owner(value, context[2])
    var n = _int(size, "vocab_size")
    var f = _int(frequency, "min_frequency")
    var name = _str(backend, "backend")
    if n < 256:
        raise Error("ValueError: vocab_size is below the 256 single-byte tokens")
    if f < 1:
        raise Error("ValueError: min_frequency must be at least 1")
    if name != "auto" and name != "mojo":
        raise Error("ValueError: backend must be auto or mojo; Python training is a verification oracle only")
    py_setitem(fields, "vocab_size", size)
    py_setitem(fields, "min_frequency", frequency)
    py_setitem(fields, "backend", backend)
    return PythonObject(None)


def _trained_finish(value: TrainedVocabulary, size: Int, frequency: Int, context: PythonObject) raises -> PythonObject:
    var tokens = Python.list()
    for i in range(value.n_tokens()):  # cpu-route: publish completed native trained vocabulary and merge records
        var bytes = List[UInt8]()
        for j in range(value.length[i]):  # cpu-route: publish completed native trained vocabulary and merge records
            bytes.append(value.arena[value.offset[i] + j])
        _ = py_call(tokens.append, _py_bytes(bytes))
    var merges = Python.list()
    for i in range(value.n_merges()):  # cpu-route: publish completed native trained vocabulary and merge records
        _ = py_call(merges.append, Python.tuple(value.merge_left[i], value.merge_right[i], 256 + i))
    var stats = Python.dict()
    py_setitem(stats, "n_tokens", PythonObject(value.n_tokens()))
    py_setitem(stats, "n_merges", PythonObject(value.n_merges()))
    py_setitem(stats, "n_groups", PythonObject(value.n_groups))
    py_setitem(stats, "n_ties_broken", PythonObject(value.n_ties_broken))
    py_setitem(stats, "tie_break", PythonObject("highest count, then smallest (left_id, right_id)"))
    py_setitem(stats, "vocab_size", PythonObject(size))
    py_setitem(stats, "min_frequency", PythonObject(frequency))
    py_setitem(stats, "backend", PythonObject("mojo"))
    var result = _new(context[1])
    var fields = _owner(result, context[1])
    py_setitem(fields, "tokens", tokens)
    py_setitem(fields, "merges", merges)
    py_setitem(fields, "stats", stats)
    return result


def _trainer_train(trainer: PythonObject, input_documents: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _owner(trainer, context[2])
    var size = _int(native_instance_item(fields, PythonObject("vocab_size")), "vocab_size")
    var frequency = _int(native_instance_item(fields, PythonObject("min_frequency")), "min_frequency")
    var backend = _str(native_instance_item(fields, PythonObject("backend")), "backend")
    if size < 256 or frequency < 1 or (backend != "auto" and backend != "mojo"):
        raise Error("ValueError: tokenizer trainer configuration was changed to an invalid value")
    var documents = _documents(input_documents, "documents")
    if len(documents) == 0:
        raise Error("ValueError: train needs at least one document")
    var arena = List[UInt8]()
    var lengths = List[Int64]()
    for i in range(len(documents)):  # cpu-route: capture documents into retained native training buffers
        var bytes = _bytes(documents[i])
        lengths.append(len(bytes))
        tokenizer_append_bytes(arena, bytes)
    var sabotage = getenv("MOJOLEARN_BPE_TRAINER_SABOTAGE", "")
    var reverse = sabotage != "" and sabotage != "0"
    var fast = context[4]
    if _fast_selected(context) and _bool(fast[1], "connected fast trainer") and not reverse and size <= _int(fast[3], "connected fast vocabulary limit"):
        var entry = context[5][2]
        var sizes_entry = context[5][3]
        var copy_entry = context[5][4]
        _native(entry)
        _native(sizes_entry)
        _native(copy_entry)
        var handle = py_call(entry, Int(arena.unsafe_ptr()), Int(lengths.unsafe_ptr()), Python.tuple(len(lengths), len(arena), size, frequency, reverse))
        var sizes = py_call(sizes_entry, handle)
        _sequence(sizes, "trained sizes")
        if len(sizes) != 5:
            raise Error("RuntimeError: tokenizer trainer returned invalid sizes")
        var count = _int(sizes[0], "trained tokens")
        var bytes_count = _int(sizes[1], "trained bytes")
        var merges = _int(sizes[2], "trained merges")
        if count < 256 or bytes_count < 0 or merges < 0:
            raise Error("RuntimeError: tokenizer trainer returned negative sizes")
        var vocab = TrainedVocabulary()
        vocab.arena = List[UInt8](length=bytes_count, fill=0)
        vocab.length = List[Int](length=count, fill=0)
        vocab.merge_left = List[Int](length=merges, fill=0)
        vocab.merge_right = List[Int](length=merges, fill=0)
        _ = py_call(copy_entry, handle, Int(vocab.arena.unsafe_ptr()), Int(vocab.length.unsafe_ptr()), Int(vocab.merge_left.unsafe_ptr()), Int(vocab.merge_right.unsafe_ptr()))
        vocab.n_ties_broken = _int(sizes[3], "trained ties")
        vocab.n_groups = _int(sizes[4], "trained groups")
        trained_vocab_settle(vocab, count, merges)
        _ = arena
        _ = lengths
        _ = input_documents
        return _trained_finish(vocab, size, frequency, context)
    var handle = BpeTrainedHandle()
    bpe_train_fill(handle, Int(arena.unsafe_ptr()), Int(lengths.unsafe_ptr()), len(lengths), len(arena), size, frequency, reverse)
    _ = arena
    _ = lengths
    _ = input_documents
    return _trained_finish(handle.vocab.value(), size, frequency, context)


def _py_repr(obj: PythonObject) raises -> String:
    var bi = native_loaded_module("builtins")
    return String(py=py_call(bi.repr, obj))


def _py_str_bytes(text: PythonObject, mut out: List[UInt8], mut err: String) raises -> Int:
    """A Python object's spelling bytes: 0 a str (bytes in `out`), 1 not a
    str, 2 the conversion raised (its message in `err`)."""
    if py_type(text) is not native_loaded_module("builtins").str:
        return 1
    try:
        out = string_bytes(String(py=text))
    except error:
        err = String(error)
        return 2
    return 0


def _table_from_spelled(
    vocab: PythonObject,
    merges: PythonObject,
    version_line: Bool,
    ignore_merges: Bool,
) raises -> RankTable:
    """A rank table from `{byte-level spelling: id}` (ids 0..n-1, each
    once) and a merge list: either the text of a `vocab.bpe` file (one
    `a b` per line, an optional leading `#version` line, blank lines
    skipped) or a list whose items are `"a b"` strings or `[a, b]` pairs
    (a `tokenizer.json`'s `model.merges`). Every check in
    `impl/vocab_build.mojo` runs in `tokenizer_host_shell_impl.mojo`; this
    door reads the Python items into typed records and names a refusal by
    the item's Python repr."""
    var bi = native_loaded_module("builtins")
    _dictionary(vocab)
    _plain(vocab)
    _plain(merges)
    var items = py_call(bi.list, vocab.items())
    var n = Int(py=py_call(bi.len, items))
    var inv = codepoint_to_byte()
    var sv = SpelledVocab(n)
    var key_err = List[String](capacity=n)
    for k in range(n):  # cpu-route: native public operation marshaling of admitted configuration values
        var key = items[k][0]
        var val = items[k][1]
        var is_int = py_type(val) is bi.int
        sv.id_is_int.append(is_int)
        sv.ids.append(Int(py=val) if is_int else -1)
        var kb = List[UInt8]()
        var ke = String("")
        sv.key_kind.append(_py_str_bytes(key, kb, ke))
        sv.keys.append(kb^)
        key_err.append(ke^)
    var arena = List[UInt8]()
    var lengths = List[Int](capacity=n)
    var vf = spelled_vocab_tokens(sv, inv, arena, lengths)
    if vf.code != SPELLED_OK:
        var key = items[vf.k][0]
        if vf.code == SPELLED_BAD_ID:
            raise Error(
                "ValueError: id "
                + _py_repr(items[vf.k][1])
                + " of "
                + _py_repr(key)
                + " is not a unique id in [0, "
                + String(n)
                + ")"
            )
        if vf.code == SPELLED_NOT_STR:
            raise Error("ValueError: spelling " + _py_repr(key) + " is not a string")
        if vf.code == SPELLED_RAISED:
            raise Error(key_err[vf.k])
        raise Error(unspell_error("spelling " + _py_repr(key), vf.bad))
    var table = table_from_tokens(arena, lengths)
    var chk = MergeCheck(n)
    if py_type(merges) is bi.str:
        spelled_merges_text(table, string_bytes(String(py=merges)), inv, version_line, chk)
    else:
        var m = Int(py=py_call(bi.len, merges))
        var sm = SpelledMerges(m)
        var errs = List[String](capacity=3 * m)
        for k in range(m):  # cpu-route: native public operation marshaling of admitted configuration values
            var item = merges[k]
            var it = py_type(item)
            var kind = 2
            if it is bi.str:
                kind = 0
            elif (it is bi.list or it is bi.tuple) and Int(py=py_call(bi.len, item)) == 2:
                kind = 1
            sm.kind.append(kind)
            var tb = List[UInt8]()
            var te = String("")
            var tk = 0
            if kind == 0:
                tk = _py_str_bytes(item, tb, te)
            sm.str_kind.append(tk)
            sm.text.append(tb^)
            errs.append(te^)
            comptime for p in range(2):  # driver-loop: over admitted tokenizer configuration metadata
                var pb = List[UInt8]()
                var pe = String("")
                var pk = 0
                if kind == 1:
                    pk = _py_str_bytes(item[p], pb, pe)
                sm.part_kind.append(pk)
                sm.parts.append(pb^)
                errs.append(pe^)
        var mf = spelled_merges_list(table, sm, inv, chk)
        if mf.code != SPELLED_OK:
            var item = merges[mf.k]
            var label = "merge " + String(mf.k)
            if mf.code == SPELLED_NOT_PAIR:
                raise Error("ValueError: " + label + " " + _py_repr(item) + " is not a pair")
            if mf.code == SPELLED_NOT_LEVEL:
                raise Error(
                    "ValueError: " + label + " names " + _py_repr(item) + ", not a byte-level spelling"
                )
            if sm.kind[mf.k] == 0:
                raise Error(errs[3 * mf.k])
            var part = item[mf.part]
            if mf.code == SPELLED_NOT_STR:
                raise Error("ValueError: " + label + " names " + _py_repr(part) + " is not a string")
            if mf.code == SPELLED_RAISED:
                raise Error(errs[3 * mf.k + 1 + mf.part])
            raise Error(unspell_error(label + " names " + _py_repr(part), mf.bad))
    check_loose(table, chk, ignore_merges)
    return table^


def _bpe_files(cls: PythonObject, encoder: PythonObject, merges_path: PythonObject, context: PythonObject) raises -> PythonObject:
    if cls is not context[0]:
        raise Error("TypeError: tokenizer factory requires the connected BpeTokenizer class")
    var path = _path(encoder)
    var merge_file = _path(merges_path)
    var vocab = native_json_read(path)
    _dictionary(vocab)
    var eot = py_call(vocab.pop, PythonObject(String(GPT2_ENDOFTEXT)), PythonObject(None))
    if eot is not None and _int(eot, "endoftext id") != len(vocab):
        raise Error("ValueError: endoftext id must follow the vocabulary ranks")
    var text = String("")
    with open(String(merge_file), "r") as file:
        text = file.read()
    var table = _table_from_spelled(vocab, PythonObject(text), True, False)
    var handle = PythonObject(alloc=_handle_from_table(table^))
    var result = _new(cls)
    _ = _bpe_adopt(result, handle, path + PythonObject(" + ") + merge_file, context)
    return result


def _vocab_pattern(pattern: PythonObject) raises -> Int:
    var name = _str(pattern, "pattern")
    if name == "gpt2":
        return 0
    if name == "llama3":
        return 1
    if name == "qwen2":
        return 2
    raise Error("ValueError: tokenizer unknown pattern " + name)


def _vocab_settle(tokenizer: PythonObject, options: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _owner(tokenizer, context[0])
    _sequence(options, "options")
    if len(options) != 7:
        raise Error("TypeError: tokenizer requires seven options")
    _ = _vocab_pattern(options[0])
    if options[2] is not None and _str(options[2], "normalizer") != "NFC":
        raise Error("ValueError: tokenizer normalizer must be null or NFC")
    _ = _str(options[6], "source")
    _ = _bool(options[5], "ignore_merges")
    var added = Python.dict()
    if options[1] is not None:
        _dictionary(options[1])
        added = py_call(native_loaded_module("builtins").dict, options[1])
    var by_id = Python.dict()
    var keys = py_call(native_loaded_module("builtins").list, added.keys())
    for i in range(len(keys)):  # driver-loop: over admitted added-token configuration entries
        var id = _int(added[keys[i]], "added token id")
        if len(keys[i]) == 0 or id < 0:
            raise Error("ValueError: added token requires nonempty str and nonnegative int")
        if py_contains(by_id, PythonObject(id)):
            raise Error("ValueError: added tokens share an id")
        py_setitem(by_id, PythonObject(id), keys[i])
    var bos = _ids(options[3])
    var eos = _ids(options[4])
    py_setitem(fields, "pattern", options[0])
    py_setitem(fields, "normalizer", options[2])
    py_setitem(fields, "source", options[6])
    py_setitem(fields, "added", added)
    py_setitem(fields, "_added_by_id", by_id)
    py_setitem(fields, "bos_ids", py_call(native_loaded_module("builtins").tuple, _py_int_list(bos)))
    py_setitem(fields, "eos_ids", py_call(native_loaded_module("builtins").tuple, _py_int_list(eos)))
    py_setitem(fields, "_m", context[1])
    return fields


def _vocab_finish(fields: PythonObject, var tok: VocabTokenizer) raises:
    var added = native_instance_item(fields, PythonObject("added"))
    var keys = py_call(native_loaded_module("builtins").list, added.keys())
    var arena = List[UInt8]()
    var lengths = List[Int]()
    var ids = List[Int]()
    for i in range(len(keys)):  # cpu-route: marshal admitted special-token text into native table storage
        var bytes = _bytes(keys[i])
        lengths.append(len(bytes))
        ids.append(_int(added[keys[i]], "added token id"))
        tokenizer_append_bytes(arena, bytes)
    tok.set_specials(arena, lengths, ids)
    py_setitem(fields, "_n_ranks", PythonObject(tok.n_ranks()))
    py_setitem(fields, "_n_vocab", PythonObject(tok.n_vocab()))
    py_setitem(fields, "_max_token_bytes", PythonObject(tok.max_token_bytes()))
    var handle = VocabHandle()
    handle.tok = tok^
    py_setitem(fields, "_handle", PythonObject(alloc=handle^))


def _vocab_init(tokenizer: PythonObject, tokens: PythonObject, merges: PythonObject, options: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _vocab_settle(tokenizer, options, context)
    var arena = List[UInt8]()
    var lengths = List[Int]()
    _pack_tokens(tokens, arena, lengths)
    var table = table_from_tokens(arena, lengths)
    _sequence(merges, "merges")
    var left = List[List[UInt8]]()
    var right = List[List[UInt8]]()
    for i in range(len(merges)):  # cpu-route: capture admitted native byte merge pairs
        _sequence(merges[i], "merge pair")
        if len(merges[i]) != 2:
            raise Error("ValueError: tokenizer every merge must be a pair of bytes")
        left.append(_bytes(merges[i][0], False))
        right.append(_bytes(merges[i][1], False))
    var checks = MergeCheck(len(lengths))
    vocab_check_pairs(table, left, right, checks)
    check_loose(table, checks, _bool(options[5], "ignore_merges"))
    var tok = VocabTokenizer(table^, builtin_unicode_classes(), _vocab_pattern(options[0]))
    _vocab_finish(fields, tok^)
    return PythonObject(None)


def vocab_spelled_public_binding(cls: PythonObject, vocab: PythonObject, merges: PythonObject, options: PythonObject, context: PythonObject) raises -> PythonObject:
    if cls is not context[0]:
        raise Error("TypeError: tokenizer factory requires the connected Tokenizer class")
    var tokenizer = _new(cls)
    var fields = _vocab_settle(tokenizer, options, context)
    var table = _table_from_spelled(vocab, merges, False, _bool(options[5], "ignore_merges"))
    var tok = VocabTokenizer(table^, builtin_unicode_classes(), _vocab_pattern(options[0]))
    _vocab_finish(fields, tok^)
    return tokenizer


def _vocab_encode(tokenizer: PythonObject, data: PythonObject, add_special: PythonObject, allow_special: PythonObject, bytes_only: Bool, context: PythonObject) raises -> PythonObject:
    var fields = _owner(tokenizer, context[0])
    var handle = native_instance_item(fields, PythonObject("_handle"))
    var owner = handle.downcast_value_ptr[VocabHandle]()
    if not owner[].tok:
        raise Error("ValueError: tokenizer vocabulary is not loaded")
    var add = _bool(add_special, "add_special_tokens")
    var allow = _bool(allow_special, "allow_special")
    var normalizer = native_instance_item(fields, PythonObject("normalizer"))
    if normalizer is not None and _str(normalizer, "normalizer") != "NFC":
        raise Error("ValueError: tokenizer normalizer must be null or NFC")
    var text = data
    if py_type(text) is native_loaded_module("builtins").str and not bytes_only and normalizer is not None:
        var normalize = context[4]
        _native(normalize)
        text = py_call(normalize, "NFC", text)
    var bytes = _bytes(text, not bytes_only)
    var bos = List[Int]()
    var eos = List[Int]()
    if add:
        bos = _ids(native_instance_item(fields, PythonObject("bos_ids")))
        eos = _ids(native_instance_item(fields, PythonObject("eos_ids")))
    var ids = vocab_encode_ids(owner[].tok.value(), bytes, allow)
    var result = Python.list()
    for i in range(len(bos)):  # cpu-route: publish completed native token ids with admitted BOS and EOS metadata
        _ = py_call(result.append, bos[i])
    for i in range(len(ids)):  # cpu-route: publish completed native token ids with admitted BOS and EOS metadata
        _ = py_call(result.append, ids[i])
    for i in range(len(eos)):  # cpu-route: publish completed native token ids with admitted BOS and EOS metadata
        _ = py_call(result.append, eos[i])
    _ = handle
    _ = data
    _ = text
    return result


def _vocab_decode(tokenizer: PythonObject, data: PythonObject, bytes_only: Bool, errors: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _owner(tokenizer, context[0])
    var handle = native_instance_item(fields, PythonObject("_handle"))
    var owner = handle.downcast_value_ptr[VocabHandle]()
    if not owner[].tok:
        raise Error("ValueError: tokenizer vocabulary is not loaded")
    var ids = _ids(data)
    var bytes = vocab_decode_ids(owner[].tok.value(), ids)
    var raw = _py_bytes(bytes)
    _ = handle
    _ = data
    return raw if bytes_only else _text_result(raw, errors)


def _vocab_pretokenize(tokenizer: PythonObject, data: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _owner(tokenizer, context[0])
    var handle = native_instance_item(fields, PythonObject("_handle"))
    var owner = handle.downcast_value_ptr[VocabHandle]()
    if not owner[].tok:
        raise Error("ValueError: tokenizer vocabulary is not loaded")
    var bytes = _bytes(data)
    var offsets = owner[].tok.value().pretokenize(bytes)
    var result = Python.list()
    for i in range(len(offsets) - 1):  # cpu-route: publish completed native pre-token boundaries as bytes
        var part = List[UInt8]()
        for j in range(offsets[i], offsets[i + 1]):  # cpu-route: publish completed native pre-token boundaries as bytes
            part.append(bytes[j])
        _ = py_call(result.append, _py_bytes(part))
    _ = handle
    _ = data
    return result


def tokenizer_vocab_encode(tokenizer: PythonObject, text: PythonObject, add_special: PythonObject, allow_special: PythonObject, bytes_only: PythonObject) raises -> PythonObject:
    return _vocab_encode(tokenizer, text, add_special, allow_special, _bool(bytes_only, "bytes_only"), _vocab_context())


def tokenizer_vocab_decode(tokenizer: PythonObject, ids: PythonObject, bytes_only: PythonObject, errors: PythonObject) raises -> PythonObject:
    return _vocab_decode(tokenizer, ids, _bool(bytes_only, "bytes_only"), errors, _vocab_context())


def tokenizer_vocab_pretokenize(tokenizer: PythonObject, data: PythonObject) raises -> PythonObject:
    return _vocab_pretokenize(tokenizer, data, _vocab_context())


def tokenizer_vocab_fingerprint(tokenizer: PythonObject) raises -> PythonObject:
    var context = _vocab_context()
    var fields = _owner(tokenizer, context[0])
    var handle = native_instance_item(fields, PythonObject("_handle"))
    var owner = handle.downcast_value_ptr[VocabHandle]()
    if not owner[].tok:
        raise Error("ValueError: tokenizer vocabulary is not loaded")
    var pattern = native_instance_item(fields, PythonObject("pattern"))
    _ = _vocab_pattern(pattern)
    var normalizer = native_instance_item(fields, PythonObject("normalizer"))
    var unicode_version = PythonObject(UNICODE_VERSION_PINNED)
    if normalizer is not None:
        if _str(normalizer, "normalizer") != "NFC":
            raise Error("ValueError: tokenizer normalizer must be null or NFC")
        var namespace = _class_fields(context[0])
        var version = py_get(fields, "unicode_version", py_get(namespace, "unicode_version"))
        unicode_version = PythonObject(String(UNICODE_VERSION_PINNED) + "/normalizer:" + _str(version, "unicode_version"))
    var bos = native_instance_item(fields, PythonObject("bos_ids"))
    var eos = native_instance_item(fields, PythonObject("eos_ids"))
    _sequence(bos, "bos_ids")
    _sequence(eos, "eos_ids")
    _plain(bos)
    _plain(eos)
    _ = _ids(bos)
    _ = _ids(eos)
    # Preserve the exact historical settings serialization and hash. All
    # values are admitted native scalars/containers before CPython formats
    # them; no user repr, field descriptor or tokenizer._m lookup is used.
    var settings = Python.tuple(pattern, normalizer, unicode_version, bos, eos)
    var tail = List[UInt8]()
    for i in range(5):  # small-loop(the five historical tokenizer fingerprint settings)
        var text = py_call(native_loaded_module("builtins").str, settings[i])
        tokenizer_append_bytes(tail, _bytes(text))
        tail.append(10)
    var result = PythonObject(vocab_fingerprint_hex(owner[].tok.value(), tail))
    _ = handle
    return result




def _vocab_config_pattern(pre: PythonObject, patterns: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    if pre is None:
        raise Error("tokenizer: pre_tokenizer is null; byte-level BPE requires a cut")
    var kind = String(py_get(pre, "type"))
    if kind == "ByteLevel":
        if Bool(py=py_get(pre, "use_regex", PythonObject(True))):
            return PythonObject("gpt2")
        raise Error("tokenizer: ByteLevel use_regex false is not a known cut")
    if kind == "Metaspace":
        raise Error("tokenizer: pre_tokenizer Metaspace is a SentencePiece family")
    if kind != "Sequence":
        raise Error("tokenizer: pre_tokenizer type " + kind + " is not a known cut")
    var found = PythonObject(None)
    var steps = py_get(pre, "pretokenizers", bi.list())
    for i in range(len(steps)):  # small-loop(pre_tokenizer steps: configuration structure)
        var step = steps[i]
        var t = String(py_get(step, "type"))
        if t == "Split":
            var pat = py_get(step, "pattern", bi.dict())
            if py_type(pat) is not bi.dict or py_get(pat, "Regex") is None:
                raise Error("tokenizer: Split pattern requires a Regex")
            var behavior = py_get(step, "behavior")
            if (behavior is not None and String(behavior) != "Isolated") or Bool(py=py_get(step, "invert", PythonObject(False))):
                raise Error("tokenizer: Split requires Isolated behavior without invert")
            var name = py_get(patterns, pat["Regex"])
            if name is None:
                raise Error("tokenizer: pre_tokenizer regex " + String(pat["Regex"]) + " is not one of the known patterns; refused by name")
            if found is not None and not Bool(py=found == name):
                raise Error("tokenizer: two Split steps with different patterns")
            found = name
        elif t == "ByteLevel":
            if Bool(py=py_get(step, "use_regex", PythonObject(False))):
                if found is not None and String(found) != "gpt2":
                    raise Error("tokenizer: Split followed by ByteLevel use_regex true cuts twice")
                found = PythonObject("gpt2")
        elif t == "Metaspace":
            raise Error("tokenizer: pre_tokenizer Metaspace is a SentencePiece family")
        else:
            raise Error("tokenizer: pre_tokenizer step " + t + " is not a known cut")
    if found is None:
        raise Error("tokenizer: pre_tokenizer Sequence names no cut")
    return found


def vocab_json_public_binding(cls: PythonObject, data: PythonObject, path: PythonObject,
                             cfg: PythonObject, module: PythonObject, patterns: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    _plain(data)
    _plain(cfg)
    _plain(patterns)
    if py_type(data) is not bi.dict:
        raise Error("tokenizer: not a JSON object")
    var model = py_get(data, "model")
    if model is None:
        model = bi.dict()
    if String(py_get(model, "type")) != "BPE":
        raise Error("tokenizer: model.type must be BPE; SentencePiece Unigram is unsupported")
    if Bool(py=py_get(model, "byte_fallback", PythonObject(False))):
        raise Error("tokenizer: model.byte_fallback identifies SentencePiece-style BPE")
    if Bool(py=py_get(model, "continuing_subword_prefix")) or Bool(py=py_get(model, "end_of_word_suffix")):
        raise Error("tokenizer: continuing_subword_prefix / end_of_word_suffix is not byte-level BPE")
    var dec = py_get(data, "decoder")
    if dec is not None and String(py_get(dec, "type")) != "ByteLevel":
        raise Error("tokenizer: decoder must be ByteLevel")
    var norm = py_get(data, "normalizer")
    var normalizer = PythonObject(None)
    if norm is not None:
        var kind = String(py_get(norm, "type"))
        if kind == "NFC":
            normalizer = PythonObject("NFC")
        elif kind == "Sequence":
            var steps = py_get(norm, "normalizers", bi.list())
            if len(steps) == 0:
                raise Error("tokenizer: empty normalizer Sequence")
            for i in range(len(steps)):  # small-loop(normalizer steps: configuration structure)
                if String(steps[i].get("type")) != "NFC":
                    raise Error("tokenizer: normalizer must be null or NFC")
            normalizer = PythonObject("NFC")
        else:
            raise Error("tokenizer: normalizer '" + kind + "' must be null or NFC")
    var pattern = _vocab_config_pattern(py_get(data, "pre_tokenizer"), patterns)
    var added = bi.dict()
    var entries = py_get(data, "added_tokens")
    if entries is not None:
        for i in range(len(entries)):  # small-loop(added_tokens: configuration entries, not input data)
            var content = py_get(entries[i], "content")
            var id = py_get(entries[i], "id")
            if py_type(content) is not bi.str or py_type(id) is not bi.int:
                raise Error("tokenizer: added token lacks string content or int id")
            py_setitem(added, content, id)
    var vocab = py_get(model, "vocab")
    if py_type(vocab) is not bi.dict:
        raise Error("tokenizer: model.vocab is not an object")
    var base = py_call(bi.dict, vocab)
    var keys = py_call(bi.list, added)
    for i in range(len(keys)):  # small-loop(added tokens: configuration entries)
        _ = py_call(base.pop, keys[i], PythonObject(None))
    var merges = py_get(model, "merges")
    if merges is None:
        merges = bi.list()
    if py_type(merges) is not bi.list:
        raise Error("tokenizer: model.merges is not a list")
    var bos = bi.list()
    var eos = bi.list()
    var post = py_get(data, "post_processor")
    if post is not None:
        var kind = String(py_get(post, "type"))
        if kind == "TemplateProcessing":
            var single = py_get(post, "single", bi.list())
            var specials = py_get(post, "special_tokens", bi.dict())
            var seen = False
            for i in range(len(single)):  # small-loop(post_processor template items: configuration structure)
                var item = single[i]
                if py_contains(item, "Sequence"):
                    seen = True
                elif py_contains(item, "SpecialToken"):
                    var name = py_get(item["SpecialToken"], "id")
                    var ids = py_get(py_get(specials, name, bi.dict()), "ids")
                    if ids is None or len(ids) == 0:
                        raise Error("tokenizer: post_processor special token has no ids")
                    for j in range(len(ids)):  # small-loop(special token ids: configuration words)
                        if seen:
                            _ = py_call(eos.append, py_call(bi.int, ids[j]))
                        else:
                            _ = py_call(bos.append, py_call(bi.int, ids[j]))
                else:
                    raise Error("tokenizer: post_processor template must contain Sequence or SpecialToken")
        elif kind != "ByteLevel":
            raise Error("tokenizer: post_processor type " + kind + " is not honored")
    if cfg is not None and py_type(cfg) is bi.dict:
        var flag = py_get(cfg, "add_bos_token")
        if py_type(flag) is bi.bool and not Bool(py=flag):
            bos.clear()
        flag = py_get(cfg, "add_eos_token")
        if py_type(flag) is bi.bool and not Bool(py=flag):
            eos.clear()
    var options = bi.list()
    _ = py_call(options.append, pattern)
    _ = py_call(options.append, added)
    _ = py_call(options.append, normalizer)
    _ = py_call(options.append, bos)
    _ = py_call(options.append, eos)
    _ = py_call(options.append, py_get(model, "ignore_merges", PythonObject(False)))
    var source = _path(path)
    _ = py_call(options.append, source)
    var tokenizer = vocab_spelled_public_binding(cls, base, merges, options, module)
    var fields = _owner(tokenizer, module[0])
    py_setitem(fields, "bos_token_id", bos[0] if len(bos) != 0 else PythonObject(None))
    py_setitem(fields, "eos_token_id", eos[len(eos) - 1] if len(eos) != 0 else PythonObject(None))
    if cfg is not None and py_type(cfg) is bi.dict:
        for key in [String("bos_token"), String("eos_token")]:  # small-loop(the two special-token names)
            var value = py_get(cfg, key)
            if py_type(value) is bi.dict:
                value = py_get(value, "content")
            if py_type(value) is bi.str and py_contains(added, value):
                py_setitem(fields, PythonObject(key + "_id"), added[value])
    return tokenizer


def vocab_pretrained_public_binding(cls: PythonObject, path: PythonObject, module: PythonObject,
                                   patterns: PythonObject) raises -> PythonObject:
    var text = native_path(path)
    var filename = text
    var directory = PythonObject(".")
    if native_path_exists(text, True):
        directory = py_call(text.rstrip, "/")
        filename = directory + PythonObject("/tokenizer.json")
        if not native_path_exists(filename):
            if native_path_exists(directory + PythonObject("/tokenizer.model")):
                raise Error("tokenizer: tokenizer.model without tokenizer.json is a SentencePiece family")
            raise Error("FileNotFoundError: directory holds no tokenizer.json")
    else:
        var pieces = py_call(text.rsplit, "/", 1)
        if len(pieces) == 2:
            directory = pieces[0] if len(pieces[0]) != 0 else PythonObject("/")
    if not native_path_exists(filename):
        raise Error("FileNotFoundError: tokenizer.json does not exist")
    var data = native_json_read(filename)
    var cfg = PythonObject(None)
    var config_path = directory + PythonObject("/tokenizer_config.json")
    if native_path_exists(config_path):
        cfg = native_json_read(config_path)
    filename = _path(filename)
    return vocab_json_public_binding(cls, data, filename, cfg, module, patterns)


def tokenizer_public_call(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    _sequence(args, "operation arguments")
    _sequence(context, "connected context")
    if operation == "connect_vocab":
        if len(context) != 5 or py_type(context[0]) is not native_loaded_module("builtins").type:
            raise Error("TypeError: tokenizer connection requires its exact class and five metadata entries")
        _dictionary(context[2])
        _plain(context[2])
        _native(context[4])
        var state = TOKENIZER_PUBLIC_CONNECTION.get_or_create_ptr()
        if state[].vocab is not None and state[].vocab is not context:
            raise Error("TypeError: tokenizer native context was already connected")
        state[].vocab = context
        return PythonObject(None)
    if operation == "bpe_init":
        return _bpe_init(args[0], args[1], context)
    if operation == "bpe_ranks":
        if args[0] is not context[0]:
            raise Error("TypeError: tokenizer factory requires its connected class")
        var result = _new(args[0])
        _ = _bpe_init(result, args[1], context)
        return result
    if operation == "bpe_tokens":
        return _bpe_tokens(args[0], args[1], context)
    if operation == "bpe_files":
        return _bpe_files(args[0], args[1], args[2], context)
    if operation == "bpe_adopt":
        return _bpe_adopt(args[0], args[1], args[2], context)
    if operation == "bpe_handle":
        if args[0] is not context[0]:
            raise Error("TypeError: tokenizer factory requires its connected class")
        var result = _new(args[0])
        _ = _bpe_adopt(result, args[1], args[2], context)
        return result
    if operation == "bpe_encode":
        return _bpe_encode(args[0], args[1], args[2], _bool(args[3], "bytes_only"), _bool(args[4], "batch"), context)
    if operation == "bpe_decode":
        return _bpe_decode(args[0], args[1], _bool(args[2], "bytes_only"), _bool(args[3], "batch"), args[4], context)
    if operation == "bpe_fact":
        var fields = _owner(args[0], context[0])
        var name = _str(args[1], "fact")
        if name == "n_vocab" or name == "eot_token":
            var count = _int(native_instance_item(fields, PythonObject("_n_vocab")), "n_vocab")
            return PythonObject(count if name == "n_vocab" else count - 1)
        if name == "source":
            return PythonObject(_str(native_instance_item(fields, PythonObject("_source")), "source"))
        if name == "identity":
            var value = native_instance_item(fields, PythonObject("_identity"))
            _dictionary(value)
            _plain(value)
            return py_call(native_loaded_module("builtins").dict, value)
        var count = _int(native_instance_item(fields, PythonObject("_n_vocab")), "n_vocab")
        var source = _str(native_instance_item(fields, PythonObject("_source")), "source")
        return PythonObject("BpeTokenizer(n_vocab=" + String(count) + ", vocabulary=" + _py_repr(PythonObject(source)) + ")")
    if operation == "trained_init":
        return _trained_init(args[0], args[1], args[2], args[3], context)
    if operation == "trained_render" or operation == "trained_write":
        var text = _trained_render(args[0], _int(args[1], "format"), context)
        if operation == "trained_write":
            var path = native_path(args[2])
            with open(String(path), "w") as file:
                file.write(text)
            return args[2]
        return PythonObject(text)
    if operation == "trained_tokenizer":
        var fields = _owner(args[0], context[1])
        return _bpe_tokens(context[0], native_instance_item(fields, PythonObject("tokens")), context)
    if operation == "trained_fact":
        var fields = _owner(args[0], context[1])
        var name = _str(args[1], "fact")
        var tokens = native_instance_item(fields, PythonObject("tokens"))
        _sequence(tokens, "tokens")
        if name == "n_tokens":
            return PythonObject(len(tokens))
        if name == "identity":
            return _identity(_trained_render(args[0], 0, context), len(tokens))
        var stats = native_instance_item(fields, PythonObject("stats"))
        _dictionary(stats)
        if name == "tie_break":
            return PythonObject(_str(native_instance_item(stats, PythonObject("tie_break")), "tie_break"))
        var count = _int(native_instance_item(stats, PythonObject("n_ties_broken")), "n_ties_broken")
        if name == "n_ties_broken":
            return PythonObject(count)
        var merges = native_instance_item(fields, PythonObject("merges"))
        _sequence(merges, "merges")
        return PythonObject("TrainedBpeVocabulary(n_tokens=" + String(len(tokens)) + ", n_merges=" + String(len(merges)) + ", n_ties_broken=" + String(count) + ")")
    if operation == "trainer_init":
        return _trainer_init(args[0], args[1], args[2], args[3], context)
    if operation == "trainer_train":
        return _trainer_train(args[0], args[1], context)
    if operation == "trainer_repr":
        var fields = _owner(args[0], context[2])
        var size = _int(native_instance_item(fields, PythonObject("vocab_size")), "vocab_size")
        var frequency = _int(native_instance_item(fields, PythonObject("min_frequency")), "min_frequency")
        var backend = _str(native_instance_item(fields, PythonObject("backend")), "backend")
        return PythonObject("BpeVocabularyTrainer(vocab_size=" + String(size) + ", min_frequency=" + String(frequency) + ", backend=" + _py_repr(PythonObject(backend)) + ")")
    if operation == "vocab_init":
        return _vocab_init(args[0], args[1], args[2], args[3], context)
    if operation == "vocab_spelled":
        return vocab_spelled_public_binding(args[0], args[1], args[2], args[3], context)
    if operation == "vocab_pretrained":
        return vocab_pretrained_public_binding(args[0], args[1], context, context[2])
    if operation == "vocab_json":
        return vocab_json_public_binding(args[0], args[1], args[2], args[3], context, context[2])
    if operation == "vocab_encode":
        return _vocab_encode(args[0], args[1], args[2], args[3], _bool(args[4], "bytes_only"), context)
    if operation == "vocab_decode":
        return _vocab_decode(args[0], args[1], _bool(args[2], "bytes_only"), args[3], context)
    if operation == "vocab_pretokenize":
        return _vocab_pretokenize(args[0], args[1], context)
    if operation == "vocab_pattern":
        var name = _str(args[0], "regex")
        _dictionary(context[2])
        _plain(context[2])
        return py_get(context[2], PythonObject(name))
    if operation == "vocab_cut":
        var bytes = _bytes(args[0])
        var offsets = pretokenize_pattern(bytes, builtin_unicode_classes(), _vocab_pattern(args[1]))
        return _py_int_list(offsets)
    if operation == "vocab_fact":
        var fields = _owner(args[0], context[0])
        var name = _str(args[1], "fact")
        if name == "n_vocab" or name == "n_ranks":
            return PythonObject(_int(native_instance_item(fields, PythonObject("_" + name)), name))
        var pattern = _str(native_instance_item(fields, PythonObject("pattern")), "pattern")
        var ranks = _int(native_instance_item(fields, PythonObject("_n_ranks")), "n_ranks")
        var added = native_instance_item(fields, PythonObject("added"))
        _dictionary(added)
        var source = _str(native_instance_item(fields, PythonObject("source")), "source")
        return PythonObject("Tokenizer(pattern=" + _py_repr(PythonObject(pattern)) + ", n_ranks=" + String(ranks) + ", added=" + String(len(added)) + ", source=" + _py_repr(PythonObject(source)) + ")")
    raise Error("ValueError: unknown tokenizer public operation")


def tokenizer_public_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        if len(args) != 3:
            raise Error("TypeError: tokenizer public binding requires operation, values and context")
        var result = tokenizer_public_call(_str(args[0], "operation"), args[1], args[2])
        return result.steal_data()
    except error:
        var message = String(error)
        if message == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error)
        if message.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if message.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if message.startswith("FileNotFoundError:"):
            return raise_python_exception(error_without_class(error, "FileNotFoundError"), ExceptionType("PyExc_FileNotFoundError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
