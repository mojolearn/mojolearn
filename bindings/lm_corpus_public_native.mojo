# SPDX-License-Identifier: Apache-2.0
"""Whole public corpus preparation, cache validation and lazy batch operations."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_loaded_module import native_loaded_module
from bindings.native_error_class import error_without_class
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.lm_corpus_metadata import (corpus_context, corpus_connect, corpus_int, corpus_dict,
    corpus_plain, corpus_fields, corpus_new, corpus_empty, corpus_mapped, corpus_ids_snapshot, corpus_batch_base)
from bindings.lm_corpus_files import (corpus_path, corpus_path_object, corpus_read, corpus_write,
    corpus_hash_bytes, corpus_hash_file, corpus_read_json, corpus_write_json, corpus_exists,
    corpus_atomic_begin, corpus_atomic_commit, corpus_atomic_abort, corpus_seconds, corpus_file_size)
from bindings.tokenizer_public_native import (tokenizer_bpe_identity_file, tokenizer_bpe_identity_text,
    tokenizer_bpe_encode_corpus, tokenizer_bpe_train_corpus, tokenizer_bpe_export, tokenizer_bpe_for,
    tokenizer_bpe_identity, tokenizer_trained_info, tokenizer_gather_rows_native)
from std.os import getenv
from bindings.native_json import native_json_parse
from core.sha256 import sha256_hex_addr
from bindings.optimizer_public_words import optimizer_python_error, PYTHON_ERROR_PENDING

comptime TOKENS_SCHEMA = "mojolearn.byte-lm.tokens.v1"
comptime CORPUS_SCHEMA = "mojolearn.byte-lm.corpus.v1"
comptime VOCABULARY_SCHEMA = "mojolearn.bpe-vocabulary.v1"
comptime SCHEDULE = "train-range-modulo.v1"


def _join(path: PythonObject, leaf: String) raises -> PythonObject:
    return py_call(path.rstrip, "/") + PythonObject("/" + leaf)


def _prefix(value: PythonObject) raises -> String:
    if py_type(value) is not native_loaded_module("builtins").str:
        raise Error("TypeError: corpus fingerprints require ordinary strings")
    return String(value[py_call(native_loaded_module("builtins").slice, 0, 16)])


def _positive(value: PythonObject, name: String) raises -> Int:
    if py_type(value) is not native_loaded_module("builtins").int:
        raise Error("ValueError: mojolearn: " + name + " must be a positive int")
    var n = Int(py=value)
    if n < 1:
        raise Error("ValueError: mojolearn: " + name + " must be a positive int")
    return n


def _progress(value: PythonObject) raises:
    if value is not None:
        raise Error("NotImplementedError: corpus preparation cannot invoke Python progress callbacks; inspect the returned manifest's native timing and document counters")


def _default_cache() raises -> PythonObject:
    var configured = getenv("MOJOLEARN_LM_CACHE", "")
    return corpus_path(PythonObject(configured if configured != "" else "~/.cache/mojolearn/lm"), True)


def _source(path_value: PythonObject) raises -> PythonObject:
    var path = corpus_path(path_value, True, True)
    var data = corpus_read(path)
    var sha = corpus_hash_bytes(data)
    var source = Python.dict(path=path, sha256=sha, bytes=PythonObject(len(data)), schema=PythonObject(None))
    var ranges = Python.dict()
    var parts = py_call(path.rsplit, "/", 1)
    var manifest = _join(parts[0], "manifest.json")
    if corpus_exists(manifest):
        var raw = corpus_read(manifest)
        var record = native_json_parse(raw)
        if py_type(record) is native_loaded_module("builtins").dict and Bool(py=py_get(record, "schema") == PythonObject(CORPUS_SCHEMA)):
            if not Bool(py=py_get(record, "sha256") == sha) or not Bool(py=py_get(record, "bytes") == PythonObject(len(data))):
                raise Error("ValueError: mojolearn: pinned corpus length/SHA mismatch for " + String(path))
            py_setitem(source, "schema", PythonObject(CORPUS_SCHEMA))
            py_setitem(source, "manifest_sha256", corpus_hash_bytes(raw))
            py_setitem(source, "source_url", py_get(record, "source_url"))
            for key in [String("train_range"), String("validation_range"), String("test_range")]:  # small-loop(three named corpus ranges)
                var span = py_get(record, key)
                if span is not None and Bool(py=span):
                    if py_type(span) is not native_loaded_module("builtins").list or len(span) != 2:
                        raise Error("ValueError: corpus ranges require two offsets")
                    py_setitem(ranges, key, Python.list(corpus_int(span[0], "range start"), corpus_int(span[1], "range end")))
    if not py_contains(ranges, "train_range"):
        ranges = Python.dict(train_range=Python.list(0, len(data)))
    return Python.tuple(data, source, ranges)


def _split_points(ranges: PythonObject, count: Int) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var points = Python.list(0, count)
    var keys = py_call(bi.list, ranges.keys())
    for i in range(len(keys)):  # driver-loop: declared train/validation/test boundaries
        var span = ranges[keys[i]]
        if py_type(span) is not bi.list and py_type(span) is not bi.tuple:
            raise Error("TypeError: corpus ranges require ordinary offset pairs")
        if len(span) != 2:
            raise Error("ValueError: corpus ranges require two offsets")
        for j in range(2):  # small-loop(two range endpoints)
            var value = corpus_int(span[j], "range offset", True)
            if value >= 0 and value <= count:
                py_call(points.append, value)
    points = py_call(bi.sorted, points)
    var result = Python.list()
    var previous = -1
    for i in range(len(points)):  # small-loop(unique sorted manifest split points)
        var value = Int(py=points[i])
        if value != previous:
            py_call(result.append, value)
            previous = value
    return result


def _vocabulary_record(identity: PythonObject, recipe: PythonObject) raises -> PythonObject:
    return Python.dict(schema=PythonObject(VOCABULARY_SCHEMA), identity=identity, recipe=recipe,
        scope=PythonObject("trained on or loaded for the caller's corpus; never shipped with mojolearn"))


def _check_vocab(directory: PythonObject) raises -> PythonObject:
    var record = corpus_read_json(_join(directory, "vocabulary.json"))
    var identity = tokenizer_bpe_identity_file(_join(directory, "ranks.tsv"))
    if not Bool(py=identity["sha256"] == record["identity"]["sha256"]):
        raise Error("ValueError: mojolearn: cached vocabulary does not match its vocabulary.json; remove it to rebuild")
    return record


def _trained_vocabulary(data: PythonObject, ranges: PythonObject, sha: PythonObject,
    cache: PythonObject, size: Int, frequency: Int, sample: Int, document_bytes: Int) raises -> PythonObject:
    var lo = Int(py=ranges["train_range"][0])
    var hi = min(Int(py=ranges["train_range"][1]), lo + sample)
    var key = _prefix(sha) + "-v" + String(size) + "-f" + String(frequency) + "-s" + String(hi - lo)
    var final = _join(_join(cache, "vocab"), key)
    if corpus_exists(_join(final, "vocabulary.json")):
        return Python.tuple(final, _check_vocab(final))
    var temporary = corpus_atomic_begin(final)
    try:
        var began = corpus_seconds()
        var trained = tokenizer_bpe_train_corpus(data, _split_points(ranges, len(data)), PythonObject(lo), PythonObject(hi),
            PythonObject(document_bytes), PythonObject(size), PythonObject(frequency))
        var seconds = corpus_seconds() - began
        var exported = tokenizer_bpe_export(trained)
        var info = tokenizer_trained_info(trained)
        corpus_write(_join(temporary, "ranks.tsv"), py_call(exported[0].encode, "ascii"))
        var recipe = Python.dict(trainer=PythonObject("mojolearn.tokenizer.BpeVocabularyTrainer"), backend=info["backend"],
            format=info["format"], tie_break=info["tie_break"], vocab_size=PythonObject(size), min_frequency=PythonObject(frequency),
            corpus_sha256=sha, sample=Python.list(lo, hi), document_bytes=PythonObject(document_bytes),
            n_tokens=info["n_tokens"], n_merges=info["n_merges"], n_ties_broken=info["n_ties_broken"], train_seconds=PythonObject(seconds))
        corpus_write_json(_join(temporary, "vocabulary.json"), _vocabulary_record(exported[3], recipe))
        corpus_atomic_commit(temporary, final)
    except error:
        corpus_atomic_abort(temporary)
        raise error^
    return Python.tuple(final, _check_vocab(final))


def _user_vocabulary(value: PythonObject, cache: PythonObject) raises -> PythonObject:
    var exported = tokenizer_bpe_export(value)
    var identity = exported[3]
    var final = _join(_join(cache, "vocab"), "user-" + _prefix(identity["sha256"]))
    if not corpus_exists(_join(final, "vocabulary.json")):
        var temporary = corpus_atomic_begin(final)
        try:
            corpus_write(_join(temporary, "ranks.tsv"), py_call(exported[0].encode, "ascii"))
            corpus_write_json(_join(temporary, "vocabulary.json"), _vocabulary_record(identity,
                Python.dict(source=exported[2], trainer=PythonObject(None))))
            corpus_atomic_commit(temporary, final)
        except error:
            corpus_atomic_abort(temporary)
            raise error^
    return Python.tuple(final, _check_vocab(final))


def _tokenize(data: PythonObject, source: PythonObject, ranges: PythonObject, tokenizer: PythonObject,
    identity: PythonObject, output: PythonObject, document_bytes: Int) raises:
    var began = corpus_seconds()
    var encoded = tokenizer_bpe_encode_corpus(tokenizer, data, _split_points(ranges, len(data)), PythonObject(document_bytes))
    var seconds = corpus_seconds() - began
    var ids = encoded[0]
    var boundaries = encoded[1]
    var count = len(ids)
    # Supported targets are little endian; preserve the on-disk int32 stream.
    var payload = py_call(ids.tobytes)
    var token_ranges = Python.dict()
    var keys = py_call(native_loaded_module("builtins").list, ranges.keys())
    for i in range(len(keys)):  # small-loop(three named byte ranges mapped to token offsets)
        var span = ranges[keys[i]]
        py_setitem(token_ranges, keys[i], Python.list(boundaries[span[0]], boundaries[span[1]]))
    var manifest = Python.dict(schema=PythonObject(TOKENS_SCHEMA), source=source, vocabulary=identity,
        encoder=PythonObject("mojolearn.tokenizer.BpeTokenizer.encode_batch (host binding _mojolearn_tokenizer_host, tokenizer/encoding.mojo); integers and tables only"),
        endoftext=PythonObject("not inserted and not recognized (allow_endoftext=False)"),
        document_rule=PythonObject("no document spans a declared range boundary; at most " + String(document_bytes) + " bytes, ending at the last 0x0A at or before the limit (or at the limit when the window holds none); each encoded alone, ids concatenated in document order"),
        document_bytes=PythonObject(document_bytes), n_documents=encoded[2], dtype=PythonObject("int32"), byte_order=PythonObject("little"),
        sha256=corpus_hash_bytes(payload), bytes=PythonObject(len(payload)), tokens=PythonObject(count),
        bytes_per_token=PythonObject(Float64(len(data)) / Float64(count)) if count else PythonObject(None),
        max_id=encoded[3], ids_above_255=encoded[4])
    py_call(manifest.update, token_ranges)
    py_setitem(manifest, "schedule", PythonObject(SCHEDULE))
    py_setitem(manifest, "tokenize_seconds", PythonObject(seconds))
    py_setitem(manifest, "encode_seconds", PythonObject(seconds))
    py_setitem(manifest, "encode_bytes_per_second", PythonObject(Float64(len(data)) / seconds) if seconds != 0 else PythonObject(None))
    corpus_write(_join(output, "tokens.i32"), payload)
    corpus_write_json(_join(output, "manifest.json"), manifest)


def _corpus_fields(value: PythonObject, context: PythonObject) raises -> PythonObject:
    return corpus_fields(value, context["corpus_class"], Python.tuple("tokens_dir", "vocabulary_path", "manifest"))


def _corpus_init(value: PythonObject, directory: PythonObject, ranks: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _corpus_fields(value, context)
    var path = corpus_path(directory)
    var vocabulary = corpus_path(ranks)
    var manifest = corpus_read_json(_join(path, "manifest.json"))
    corpus_dict(manifest)
    if not Bool(py=py_get(manifest, "schema") == PythonObject(TOKENS_SCHEMA)):
        raise Error("ValueError: mojolearn: corpus manifest is not " + String(TOKENS_SCHEMA))
    py_setitem(fields, "tokens_dir", corpus_path_object(path))
    py_setitem(fields, "vocabulary_path", corpus_path_object(vocabulary))
    py_setitem(fields, "manifest", manifest)
    return PythonObject(None)


def _corpus_construct(directory: PythonObject, ranks: PythonObject, context: PythonObject) raises -> PythonObject:
    var result = corpus_new(context["corpus_class"])
    _corpus_init(result, directory, ranks, context)
    return result


def _prepare(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var size = _positive(args[3], "vocab_size")
    var frequency = _positive(args[4], "min_frequency")
    var sample = _positive(args[5], "vocab_sample_bytes")
    var document_bytes = _positive(args[6], "document_bytes")
    _ = _positive(args[7], "batch_documents")
    _progress(args[8])
    var vocab = args[1]
    var bi = native_loaded_module("builtins")
    if py_type(vocab) is bi.str and String(vocab) == "bytes":
        raise Error("ValueError: mojolearn: byte training has no vocabulary to prepare; use the byte schedule instead of lm_corpus.prepare")
    var cache = _default_cache() if args[2] is None else corpus_path(args[2], True)
    var source = _source(args[0])
    var data = source[0]
    var src = source[1]
    var ranges = source[2]
    var vocabulary = _trained_vocabulary(data, ranges, src["sha256"], cache, size, frequency, sample, document_bytes) if vocab is None else _user_vocabulary(vocab, cache)
    var identity = vocabulary[1]["identity"]
    var ranks = _join(vocabulary[0], "ranks.tsv")
    var key = _prefix(src["sha256"]) + "-" + _prefix(identity["sha256"]) + "-d" + String(document_bytes)
    var final = _join(_join(cache, "tokens"), key)
    if corpus_exists(_join(final, "manifest.json")):
        var result = _corpus_construct(final, ranks, context)
        var manifest = _fields(result)["manifest"]
        if not Bool(py=manifest["source"]["sha256"] == src["sha256"]) or not Bool(py=manifest["vocabulary"]["sha256"] == identity["sha256"]):
            raise Error("ValueError: mojolearn: cached tokens were made from another corpus or vocabulary; remove the directory to rebuild")
        if not Bool(py=corpus_hash_file(_join(final, "tokens.i32")) == manifest["sha256"]):
            raise Error("ValueError: mojolearn: cached token hash differs from its manifest; remove the directory to rebuild")
        return result
    var tokenizer = tokenizer_bpe_for(ranks)
    if not Bool(py=tokenizer_bpe_identity(tokenizer)["sha256"] == identity["sha256"]):
        raise Error("ValueError: mojolearn: cached rank file identity differs from its vocabulary manifest")
    var temporary = corpus_atomic_begin(final)
    try:
        _tokenize(data, src, ranges, tokenizer, identity, temporary, document_bytes)
        corpus_atomic_commit(temporary, final)
    except error:
        corpus_atomic_abort(temporary)
        raise error^
    return _corpus_construct(final, ranks, context)


def _batch_fields(value: PythonObject, context: PythonObject) raises -> PythonObject:
    return corpus_fields(value, context["batches_class"], Python.tuple("dir", "manifest", "manifest_sha256", "sha256",
        "ids_all", "batch", "length", "lo", "hi", "modulus"))


def _batch_method(owner: PythonObject, fields: PythonObject, context: PythonObject) raises:
    if py_contains(fields, "ids"):
        raise Error("TypeError: native prefetch cannot call an overridden Python ids method; request batches with the native ids operation")
    var bi = native_loaded_module("builtins")
    var ancestry = py_call(bi.type.__dict__["__mro__"].__get__, py_type(owner))
    for i in range(len(ancestry)):  # small-loop(pinned public batch operation ancestry)
        var namespace = _class_fields(ancestry[i])
        if py_contains(namespace, "ids"):
            if namespace["ids"] is not context["batch_ids_method"]:
                raise Error("TypeError: native prefetch cannot call an overridden Python ids method")
            return
    raise Error("TypeError: native batch operation connection changed")


def _batch_init(value: PythonObject, directory: PythonObject, batch_value: PythonObject,
                 length_value: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _batch_fields(value, context)
    var path = corpus_path(directory)
    var raw = corpus_read(_join(path, "manifest.json"))
    var manifest = native_json_parse(raw)
    corpus_dict(manifest)
    if not Bool(py=py_get(manifest, "schema") == PythonObject(TOKENS_SCHEMA)):
        raise Error("ValueError: mojolearn: token manifest is not " + String(TOKENS_SCHEMA))
    var token_file = _join(path, "tokens.i32")
    var size = corpus_file_size(token_file)
    if size % 4:
        raise Error("ValueError: mojolearn: token payload is not an int32 stream")
    var mapped = corpus_mapped(token_file, size, context)
    var sha = PythonObject(sha256_hex_addr(Int(py=mapped._addr), size))
    if not Bool(py=sha == manifest["sha256"]) or not Bool(py=PythonObject(size) == manifest["bytes"]):
        raise Error("ValueError: mojolearn: pinned tokens length/SHA mismatch for " + String(token_file))
    var batch = corpus_int(batch_value, "batch")
    var length = corpus_int(length_value, "length")
    var span = py_get(manifest, "train_range")
    if span is None or not Bool(py=span):
        span = Python.list(0, size // 4)
    if py_type(span) is not native_loaded_module("builtins").list or len(span) != 2:
        raise Error("ValueError: corpus train_range requires two token offsets")
    var lo = corpus_int(span[0], "train range start")
    var hi = corpus_int(span[1], "train range end")
    if length < 0 or length > 2147483645 or hi - lo < length + 2:
        raise Error("ValueError: mojolearn: the train range is shorter than one batch row")
    py_setitem(fields, "dir", corpus_path_object(path))
    py_setitem(fields, "manifest", manifest)
    py_setitem(fields, "manifest_sha256", corpus_hash_bytes(raw))
    py_setitem(fields, "sha256", sha)
    py_setitem(fields, "ids_all", mapped)
    py_setitem(fields, "batch", batch)
    py_setitem(fields, "length", length)
    py_setitem(fields, "lo", lo)
    py_setitem(fields, "hi", hi)
    py_setitem(fields, "modulus", hi - lo - length - 1)
    return PythonObject(None)


def _batch_construct(directory: PythonObject, batch: PythonObject, length: PythonObject, context: PythonObject) raises -> PythonObject:
    var result = corpus_new(context["batches_class"])
    _batch_init(result, directory, batch, length, context)
    return result


def _batch_ids(value: PythonObject, step_value: PythonObject, context: PythonObject,
               captured_fields: PythonObject) raises -> PythonObject:
    var fields = captured_fields
    var batch = corpus_int(fields["batch"], "batch", True)
    var length = corpus_int(fields["length"], "length", True)
    var lo = corpus_int(fields["lo"], "lo", True)
    var modulus = corpus_int(fields["modulus"], "modulus", True)
    if batch < 0 or length < 1:
        raise Error("ValueError: tokens_gather_rows: bad dims")
    var base = corpus_batch_base(step_value, batch, length)
    var source = corpus_ids_snapshot(fields["ids_all"], context)
    var result = corpus_empty(Python.tuple(batch, length + 1), context)
    _ = tokenizer_gather_rows_native(Int(py=source._addr), Int(py=source.size), Int(py=result._addr), Int(py=result.size),
        batch, length, lo, modulus, base)
    _ = source^
    return result


struct LmCorpusIteratorHandle(Movable, Writable):
    var owner: PythonObject
    var current: Int
    var stop: Int
    var closed: Bool

    def __init__(out self, owner: PythonObject, start: Int, stop: Int):
        self.owner = owner
        self.current = start
        self.stop = stop
        self.closed = False

    def write_to(self, mut writer: Some[Writer]):
        writer.write("<native corpus batch iterator>")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def _prefetch(owner: PythonObject, start_value: PythonObject, steps_value: PythonObject,
              depth_value: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    if py_type(start_value) is not bi.int or py_type(steps_value) is not bi.int or py_type(depth_value) is not bi.int:
        raise Error("ValueError: prefetch start_step/steps require nonnegative ints and depth a positive int")
    var start = Int(py=start_value)
    var steps = Int(py=steps_value)
    var depth = Int(py=depth_value)
    if start < 0 or steps < 0 or depth < 1 or steps > 9223372036854775807 - start:
        raise Error("ValueError: prefetch start_step/steps require nonnegative ints and depth a positive int")
    var fields = _batch_fields(owner, context)
    _batch_method(owner, fields, context)
    var iterator = corpus_new(context["iterator_class"])
    var slots = corpus_fields(iterator, context["iterator_class"], Python.tuple("_handle"))
    py_setitem(slots, "_handle", PythonObject(alloc=LmCorpusIteratorHandle(owner, start, start + steps)))
    return iterator


def _iterator_next(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = corpus_fields(value, context["iterator_class"], Python.tuple("_handle"))
    var owner = fields["_handle"]
    var handle = owner.downcast_value_ptr[LmCorpusIteratorHandle]()
    if handle[].closed or handle[].current >= handle[].stop:
        handle[].closed = True
        raise Error("StopIteration:")
    var step = handle[].current
    try:
        var batch_fields = _batch_fields(handle[].owner, context)
        _batch_method(handle[].owner, batch_fields, context)
        var output = _batch_ids(handle[].owner, PythonObject(step), context, batch_fields)
        handle[].current += 1
        return Python.tuple(step, output)
    except error:
        handle[].closed = True
        raise error^


def _batch_describe(owner: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _batch_fields(owner, context)
    var manifest = corpus_plain(fields["manifest"])
    var vocabulary = manifest["vocabulary"]
    var source = corpus_ids_snapshot(fields["ids_all"], context)
    var lo = corpus_int(fields["lo"], "lo", True)
    var hi = corpus_int(fields["hi"], "hi", True)
    return Python.dict(path=_join(corpus_path(fields["dir"]), "tokens.i32"), sha256=corpus_plain(fields["sha256"]),
        manifest_sha256=corpus_plain(fields["manifest_sha256"]), tokens=source.size, train_range=Python.list(lo, hi),
        corpus_sha256=manifest["source"]["sha256"], vocabulary_sha256=vocabulary["sha256"], n_vocab=vocabulary["n_vocab"],
        schedule=PythonObject("step k row b: ids[lo + (k*batch*length + b*length) % (hi - lo - length - 1) : +length+1] over the TRAIN range [lo, hi) in tokens; targets shifted one id"))


def _batch_schedule(owner: PythonObject, extra: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _batch_fields(owner, context)
    var manifest = corpus_plain(fields["manifest"])
    var vocabulary = manifest["vocabulary"]
    var result = Python.dict(schema=PythonObject(TOKENS_SCHEMA), schedule=PythonObject(SCHEDULE), tokens_sha256=corpus_plain(fields["sha256"]),
        corpus_sha256=manifest["source"]["sha256"], batch=PythonObject(corpus_int(fields["batch"], "batch", True)),
        length=PythonObject(corpus_int(fields["length"], "length", True)),
        train_range=Python.list(corpus_int(fields["lo"], "lo", True), corpus_int(fields["hi"], "hi", True)),
        vocabulary=Python.dict(schema=vocabulary["schema"], sha256=vocabulary["sha256"], n_vocab=vocabulary["n_vocab"]))
    corpus_dict(extra)
    py_call(result.update, extra)
    return result


def _expected_vocabulary(input: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var model = input
    if py_type(model) is not bi.dict:
        # A trainer is a passive source of stored schedule metadata. Never
        # call its getter, __getattr__, descriptor, or schedule method.
        var ancestry = py_call(bi.type.__dict__["__mro__"].__get__, py_type(model))
        for i in range(len(ancestry)):  # driver-loop: raw trainer schedule descriptor admission
            var namespace = _class_fields(ancestry[i])
            if py_contains(namespace, "data_schedule"):
                var member = namespace["data_schedule"]
                var bases = py_call(bi.type.__dict__["__mro__"].__get__, py_type(member))
                for j in range(len(bases)):  # small-loop(descriptor protocol ancestry)
                    if py_contains(_class_fields(bases[j]), "__get__"):
                        raise Error("TypeError: vocabulary checks require a stored data_schedule or an explicit checkpoint/schedule dictionary")
        var fields = _fields(model)
        model = py_get(fields, "data_schedule")
    if model is None or py_type(model) is not bi.dict:
        return PythonObject(None)
    corpus_dict(model)
    var nested = py_get(model, "data_schedule")
    if nested is not None and py_type(nested) is bi.dict:
        model = nested
        corpus_dict(model)
    var vocabulary = py_get(model, "vocabulary")
    if vocabulary is not None and py_type(vocabulary) is bi.dict:
        corpus_dict(vocabulary)
        if py_contains(vocabulary, "sha256"):
            return corpus_plain(vocabulary)
    return PythonObject(None)


def _require_vocabulary(model: PythonObject, tokenizer: PythonObject) raises -> PythonObject:
    var expected = _expected_vocabulary(model)
    if expected is None:
        raise Error("ValueError: mojolearn: this model carries no vocabulary identity (a byte model, or trained without lm_corpus); there is nothing to check a tokenizer against")
    var got = tokenizer_bpe_identity(tokenizer)
    if not Bool(py=got["sha256"] == expected["sha256"]) or not Bool(py=got["n_vocab"] == expected["n_vocab"]):
        raise Error("ValueError: mojolearn: vocabulary mismatch: the model and tokenizer have different vocabulary hashes or sizes; its ids would decode to the wrong text")
    return tokenizer


def _arity(args: PythonObject, count: Int) raises:
    if len(args) != count:
        raise Error("TypeError: corpus public operation received the wrong argument count")


def lm_corpus_public_call(operation: String, args: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    if py_type(args) is not bi.tuple:
        raise Error("TypeError: corpus operations require their native argument tuple")
    if operation == "connect":
        return corpus_connect(args)
    var context = corpus_context()
    if operation == "default_cache_dir":
        _arity(args, 0)
        return corpus_path_object(_default_cache())
    if operation == "prepare":
        _arity(args, 9)
        return _prepare(args, context)
    if operation == "tokenize":
        _arity(args, 9)
        _progress(args[8])
        var document_bytes = _positive(args[6], "document_bytes")
        _ = _positive(args[7], "batch_documents")
        if py_type(args[0]) is not bi.bytes:
            raise Error("TypeError: corpus tokenization requires an immutable bytes snapshot")
        var source = corpus_plain(args[1])
        var ranges = corpus_plain(args[2])
        var identity = corpus_plain(args[4])
        corpus_dict(source)
        corpus_dict(ranges)
        corpus_dict(identity)
        _tokenize(args[0], source, ranges, args[3], identity, corpus_path(args[5]), document_bytes)
        return PythonObject(None)
    if operation == "corpus_init":
        _arity(args, 3)
        return _corpus_init(args[0], args[1], args[2], context)
    if operation == "corpus_batches":
        _arity(args, 3)
        var fields = _corpus_fields(args[0], context)
        return _batch_construct(fields["tokens_dir"], args[1], args[2], context)
    if operation == "corpus_vocabulary" or operation == "corpus_n_vocab" or operation == "corpus_tokenizer" or operation == "corpus_repr":
        _arity(args, 1)
        var fields = _corpus_fields(args[0], context)
        var manifest = corpus_plain(fields["manifest"])
        corpus_dict(manifest)
        var vocabulary = manifest["vocabulary"]
        corpus_dict(vocabulary)
        if operation == "corpus_vocabulary":
            return py_call(bi.dict, vocabulary)
        if operation == "corpus_n_vocab":
            return PythonObject(corpus_int(vocabulary["n_vocab"], "n_vocab"))
        if operation == "corpus_tokenizer":
            var tokenizer = tokenizer_bpe_for(fields["vocabulary_path"])
            return _require_vocabulary(manifest, tokenizer)
        var count = corpus_int(manifest["tokens"], "tokens")
        var n_vocab = corpus_int(vocabulary["n_vocab"], "n_vocab")
        var path = corpus_path(fields["tokens_dir"])
        return PythonObject("TokenizedCorpus(tokens=" + String(count) + ", n_vocab=" + String(n_vocab)
            + ", sha256=" + _prefix(manifest["sha256"]) + ", dir=" + String(py_call(bi.repr, path)) + ")")
    if operation == "batches_init":
        _arity(args, 4)
        return _batch_init(args[0], args[1], args[2], args[3], context)
    if operation == "batches_ids":
        _arity(args, 2)
        return _batch_ids(args[0], args[1], context, _batch_fields(args[0], context))
    if operation == "batches_prefetch":
        _arity(args, 4)
        return _prefetch(args[0], args[1], args[2], args[3], context)
    if operation == "batches_vocabulary":
        _arity(args, 1)
        var fields = _batch_fields(args[0], context)
        var manifest = corpus_plain(fields["manifest"])
        corpus_dict(manifest)
        var vocabulary = manifest["vocabulary"]
        corpus_dict(vocabulary)
        return py_call(bi.dict, vocabulary)
    if operation == "batches_describe":
        _arity(args, 1)
        return _batch_describe(args[0], context)
    if operation == "batches_schedule":
        _arity(args, 2)
        return _batch_schedule(args[0], args[1], context)
    if operation == "iterator_iter":
        _arity(args, 1)
        var fields = corpus_fields(args[0], context["iterator_class"], Python.tuple("_handle"))
        _ = fields["_handle"].downcast_value_ptr[LmCorpusIteratorHandle]()
        return args[0]
    if operation == "iterator_next":
        _arity(args, 1)
        return _iterator_next(args[0], context)
    if operation == "expected_vocabulary":
        _arity(args, 1)
        return _expected_vocabulary(args[0])
    if operation == "require_vocabulary":
        _arity(args, 2)
        return _require_vocabulary(args[0], args[1])
    if operation == "tokenizer_for":
        _arity(args, 2)
        var tokenizer = tokenizer_bpe_for(args[1])
        return _require_vocabulary(args[0], tokenizer)
    raise Error("ValueError: unknown native corpus public operation")


def lm_corpus_public_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var bi = native_loaded_module("builtins")
        if py_type(args) is not bi.tuple or len(args) != 2 or py_type(args[0]) is not bi.str:
            raise Error("TypeError: corpus public binding requires an operation string and argument tuple")
        var result = lm_corpus_public_call(String(args[0]), args[1])
        return result.steal_data()
    except error:
        var message = String(error)
        if message == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error)
        if message.startswith("StopIteration:"):
            return raise_python_exception(error_without_class(error, "StopIteration"), ExceptionType("PyExc_StopIteration"))
        if message.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if message.startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if message.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if message.startswith("FileNotFoundError:"):
            return raise_python_exception(error_without_class(error, "FileNotFoundError"), ExceptionType("PyExc_FileNotFoundError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
