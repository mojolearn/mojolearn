# SPDX-License-Identifier: Apache-2.0
"""Whole public corpus preparation, cache validation and lazy batch operations.

Every operation is UNPACK (Python objects to strings, words and native JSON
nodes) / RUN (`bindings/lm_corpus_prepare_run.mojo`, `core/lm_batch_run.mojo`:
no Python) / PACK (the public records). The metadata read-backs
(`vocabulary`, `n_vocab`, `describe`, `data_schedule`, `repr`) compute
nothing: they are UNPACK of stored fields and PACK of a dictionary or text.
"""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_error_class import error_without_class
from bindings.mode_metadata_native_door import _fields, _class_fields
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.lm_corpus_metadata_door import (corpus_context, corpus_connect, corpus_int, corpus_dict,
    corpus_plain, corpus_fields, corpus_new, corpus_empty, corpus_mapped, corpus_ids_snapshot, corpus_batch_base,
    corpus_json_node)
from bindings.lm_corpus_files_door import corpus_path, corpus_path_object, corpus_bytes_object, corpus_os_error_pending
from bindings.lm_corpus_files_run import (CORPUS_OS_ERROR, corpus_env_lm_cache, corpus_run_expand, corpus_run_normalize,
    corpus_run_join, corpus_run_file_size)
from bindings.lm_corpus_prepare_run import (TOKENS_SCHEMA, SCHEDULE, CorpusRange, CorpusPrepareRequest, TokensManifest,
    corpus_prepare_run, corpus_tokenize_run, tokens_manifest_read, tokens_manifest_sha, tokens_manifest_digest,
    tokens_manifest_payload, tokens_manifest_range)
from bindings.tokenizer_public_native_door import (tokenizer_bpe_export, tokenizer_bpe_for, tokenizer_bpe_identity,
    tokenizer_bpe_handle)
from bindings.py_cpython_table import cpython_table
from core.native_json_value import JsonDoc
from tokenizer.impl.tokenizer_host_shell_impl import BpeHandle
from core.lm_batch_run import LmBatchRequest, lm_batch_dims, lm_batch_run, lm_batch_steps_run
from bindings.native_json_door import native_json_parse
from core.sha256 import sha256_hex_addr
from bindings.optimizer_public_words_unpack import optimizer_python_error, PYTHON_ERROR_PENDING
from bindings.parallel_byte_native_door import parallel_byte_shard_words, parallel_byte_step_shards
from bindings.py_keys import py_op_code
from bindings.lm_keys_unpack import (
    lm_keys,
    LM_OPN_BATCHES_DESCRIBE,
    LM_OPN_BATCHES_IDS,
    LM_OPN_BATCHES_INIT,
    LM_OPN_BATCHES_PARALLEL_STEP,
    LM_OPN_BATCHES_PREFETCH,
    LM_OPN_BATCHES_SCHEDULE,
    LM_OPN_BATCHES_VOCABULARY,
    LM_OPN_CONNECT,
    LM_OPN_CORPUS_BATCHES,
    LM_OPN_CORPUS_INIT,
    LM_OPN_CORPUS_N_VOCAB,
    LM_OPN_CORPUS_REPR,
    LM_OPN_CORPUS_TOKENIZER,
    LM_OPN_CORPUS_VOCABULARY,
    LM_OPN_EXPECTED_VOCABULARY,
    LM_OPN_ITERATOR_ITER,
    LM_OPN_ITERATOR_NEXT,
    LM_OPN_PREPARE,
    LM_OPN_REQUIRE_VOCABULARY,
    LM_OPN_TOKENIZE,
    LM_OPN_TOKENIZER_FOR,
)
from bindings.py_balanced import py_item

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


def _default_cache() raises -> String:
    var configured = corpus_env_lm_cache()
    return corpus_run_normalize(corpus_run_expand(configured if configured != "" else String("~/.cache/mojolearn/lm")))


def _ranges_unpack(value: PythonObject) raises -> List[CorpusRange]:
    """UNPACK: the named byte ranges of the private tokenize step, in the
    dictionary's order."""
    var bi = native_loaded_module("builtins")
    corpus_dict(value)
    var result = List[CorpusRange]()
    var keys = py_call(bi.list, value.keys())
    for i in range(len(keys)):  # driver-loop: declared train/validation/test boundaries
        var span = value[keys[i]]
        if py_type(span) is not bi.list and py_type(span) is not bi.tuple:
            raise Error("TypeError: corpus ranges require ordinary offset pairs")
        if len(span) != 2:
            raise Error("ValueError: corpus ranges require two offsets")
        result.append(CorpusRange(String(py=keys[i]), corpus_int(span[0], "range offset", True),
                                  corpus_int(span[1], "range offset", True)))
    return result^


def _corpus_fields(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = lm_keys()
    return corpus_fields(value, py_item(context, nk[].k_corpus_class), Python.tuple("tokens_dir", "vocabulary_path", "manifest"))


def _manifest_object(manifest: TokensManifest) raises -> PythonObject:
    """PACK: the public manifest dictionary, from the bytes RUN read once."""
    return native_json_parse(corpus_bytes_object(manifest.raw))


def _corpus_publish(value: PythonObject, directory: String, ranks: String, manifest: TokensManifest,
                    context: PythonObject) raises:
    """PACK: the fields of a TokenizedCorpus."""
    var nk = lm_keys()
    var fields = _corpus_fields(value, context)
    py_setitem(fields, nk[].k_tokens_dir, corpus_path_object(directory))
    py_setitem(fields, nk[].k_vocabulary_path, corpus_path_object(ranks))
    py_setitem(fields, nk[].k_manifest, _manifest_object(manifest))
    py_setitem(fields, nk[].k__tokens_verified, PythonObject(None))


def _corpus_init(value: PythonObject, directory: PythonObject, ranks: PythonObject, context: PythonObject) raises -> PythonObject:
    # UNPACK
    _ = _corpus_fields(value, context)
    var path = corpus_path(directory)
    var vocabulary = corpus_path(ranks)
    # RUN
    var manifest = tokens_manifest_read(path, "corpus")
    # PACK
    _corpus_publish(value, path, vocabulary, manifest, context)
    return PythonObject(None)


def _prepare(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = lm_keys()
    # UNPACK
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
    var corpus = corpus_path(args[0], True, True)
    var request = CorpusPrepareRequest(corpus, cache, size, frequency, sample, document_bytes)
    if vocab is not None:
        # The caller's vocabulary as its canonical rank text (the tokenizer
        # door's export): text, rank count, where it came from.
        var exported = tokenizer_bpe_export(vocab)
        request.user = True
        request.user_ranks = String(py=exported[0])
        request.user_count = Int(py=exported[1])
        request.user_source = String(py=exported[2])
    # RUN: files, hashing, training and encoding; no Python object.
    var prepared = corpus_prepare_run(request)
    # PACK
    var result = corpus_new(py_item(context, nk[].k_corpus_class))
    _corpus_publish(result, prepared.tokens_dir, prepared.ranks, prepared.manifest, context)
    if prepared.cached:
        # pass 3 (N174): the cached token file is mapped and its mapping
        # hashed once here; the verified mapping rides on the corpus, so
        # batches() reuses it instead of mapping and hashing the file again
        # (it re-checks the path, size and manifest hash before trusting it).
        var token_file = corpus_run_join(prepared.tokens_dir, "tokens.i32")
        var token_size = corpus_run_file_size(token_file)
        var mapped = corpus_mapped(PythonObject(token_file), token_size, context)
        var token_sha = sha256_hex_addr(Int(py=mapped._addr), token_size)
        if token_sha != tokens_manifest_sha(prepared.manifest):
            raise Error("ValueError: mojolearn: cached token hash differs from its manifest; remove the directory to rebuild")
        py_setitem(_fields(result), nk[].k__tokens_verified,
                   Python.tuple(PythonObject(token_file), PythonObject(token_size), PythonObject(token_sha), mapped))
    return result


def _batch_fields(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = lm_keys()
    return corpus_fields(value, py_item(context, nk[].k_batches_class), Python.tuple("dir", "manifest", "manifest_sha256", "sha256",
        "ids_all", "batch", "length", "lo", "hi", "modulus"))


def _batch_method(owner: PythonObject, fields: PythonObject, context: PythonObject) raises:
    var nk = lm_keys()
    if py_contains(fields, nk[].k_ids):
        raise Error("TypeError: native prefetch cannot call an overridden Python ids method; request batches with the native ids operation")
    var bi = native_loaded_module("builtins")
    var ancestry = py_call(py_item(bi.type.__dict__, nk[].k___mro__).__get__, py_type(owner))
    for i in range(len(ancestry)):  # small-loop(pinned public batch operation ancestry)
        var namespace = _class_fields(ancestry[i])
        if py_contains(namespace, nk[].k_ids):
            if py_item(namespace, nk[].k_ids) is not py_item(context, nk[].k_batch_ids_method):
                raise Error("TypeError: native prefetch cannot call an overridden Python ids method")
            return
    raise Error("TypeError: native batch operation connection changed")


def _batch_verified(verified: PythonObject, token_file: String, size: Int, pinned: String) raises -> Bool:
    # A verdict carried from prepare (N174) is used only for the same token
    # file, at the same size, against the manifest just read, and only in the
    # exact form prepare stored; anything else is verified afresh.
    var bi = native_loaded_module("builtins")
    if py_type(verified) is not bi.tuple or len(verified) != 4:
        return False
    if py_type(verified[0]) is not bi.str or py_type(verified[1]) is not bi.int or py_type(verified[2]) is not bi.str:
        return False
    if String(py=verified[0]) != token_file or Int(py=verified[1]) != size:
        return False
    return String(py=verified[2]) == pinned


def _batch_init(value: PythonObject, directory: PythonObject, batch_value: PythonObject,
                 length_value: PythonObject, context: PythonObject, verified: PythonObject) raises -> PythonObject:
    var nk = lm_keys()
    # UNPACK
    var fields = _batch_fields(value, context)
    var path = corpus_path(directory)
    # RUN: the manifest, read once and admitted.
    var manifest = tokens_manifest_read(path, "token")
    var token_file = corpus_run_join(path, "tokens.i32")
    var size = corpus_run_file_size(token_file)
    if size % 4:
        raise Error("ValueError: mojolearn: token payload is not an int32 stream")
    # The mapping is the public `ids_all` (PACK storage); RUN hashes it in place.
    var mapped = PythonObject(None)
    var sha = String("")
    if _batch_verified(verified, token_file, size, tokens_manifest_sha(manifest)):
        mapped = verified[3]
        sha = String(py=verified[2])
    else:
        mapped = corpus_mapped(PythonObject(token_file), size, context)
        sha = sha256_hex_addr(Int(py=mapped._addr), size)
    tokens_manifest_payload(manifest, token_file, size, sha)
    var batch = corpus_int(batch_value, "batch")
    var length = corpus_int(length_value, "length")
    var lo = 0
    var hi = 0
    tokens_manifest_range(manifest, size, length, lo, hi)
    # PACK
    py_setitem(fields, nk[].k_dir, corpus_path_object(path))
    py_setitem(fields, nk[].k_manifest, _manifest_object(manifest))
    py_setitem(fields, nk[].k_manifest_sha256, PythonObject(tokens_manifest_digest(manifest)))
    py_setitem(fields, nk[].k_sha256, PythonObject(sha))
    py_setitem(fields, nk[].k_ids_all, mapped)
    py_setitem(fields, nk[].k_batch, batch)
    py_setitem(fields, nk[].k_length, length)
    py_setitem(fields, nk[].k_lo, lo)
    py_setitem(fields, nk[].k_hi, hi)
    py_setitem(fields, nk[].k_modulus, hi - lo - length - 1)
    return PythonObject(None)


def _batch_construct(directory: PythonObject, batch: PythonObject, length: PythonObject, context: PythonObject,
                     verified: PythonObject) raises -> PythonObject:
    var nk = lm_keys()
    var result = corpus_new(py_item(context, nk[].k_batches_class))
    _batch_init(result, directory, batch, length, context, verified)
    return result


def _batch_words(fields: PythonObject) raises -> LmBatchRequest:
    """UNPACK: the four admitted words of a batches owner (no token source yet)."""
    var nk = lm_keys()
    var batch = corpus_int(py_item(fields, nk[].k_batch), "batch", True)
    var length = corpus_int(py_item(fields, nk[].k_length), "length", True)
    var lo = corpus_int(py_item(fields, nk[].k_lo), "lo", True)
    var modulus = corpus_int(py_item(fields, nk[].k_modulus), "modulus", True)
    return LmBatchRequest(batch, length, lo, modulus)


def _batch_source(fields: PythonObject, context: PythonObject, mut request: LmBatchRequest) raises -> PythonObject:
    """UNPACK: the mapped token ids of a batches owner into `request`. The
    returned snapshot owns the mapping: the caller keeps it past the RUN."""
    var nk = lm_keys()
    var source = corpus_ids_snapshot(py_item(fields, nk[].k_ids_all), context)
    request.source = Int(py=source._addr)
    request.source_count = Int(py=source.size)
    return source


def _batch_ids(value: PythonObject, step_value: PythonObject, context: PythonObject,
               captured_fields: PythonObject) raises -> PythonObject:
    # UNPACK
    var request = _batch_words(captured_fields)
    lm_batch_dims(request)
    var base = corpus_batch_base(step_value, request.batch, request.length)
    var source = _batch_source(captured_fields, context, request)
    # PACK storage, then RUN into it
    var result = corpus_empty(Python.tuple(request.batch, request.length + 1), context)
    _ = lm_batch_run(request, base, Int(py=result._addr), Int(py=result.size))
    _ = source
    return result


def lm_corpus_gather_steps(owner: PythonObject, steps: List[Int], batch: Int, length: Int,
                           target_address: Int, context: PythonObject) raises:
    """`ids(steps[k])` for every k, gathered straight into caller storage of
    len(steps) * batch * (length + 1) int32 cells at `target_address`, under
    ONE admission of the batches owner (fields, ids method, token snapshot).
    The same rows, in the same order, as `ids`: `ids(s)` reads base
    s * batch * length; here the base is computed natively for exact ints."""
    # UNPACK
    var fields = _batch_fields(owner, context)
    _batch_method(owner, fields, context)
    var request = _batch_words(fields)
    lm_batch_dims(request)
    if request.batch != batch or request.length != length:
        raise Error("ValueError: parallel input has wrong shape")
    var source = _batch_source(fields, context, request)
    # RUN
    lm_batch_steps_run(request, steps, target_address)
    _ = source


def _batch_parallel_step(owner: PythonObject, trainer: PythonObject, steps_value: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    # pass 3 (N105): one parallel byte-LM step fed from a TokenBatches owner.
    # Shard k is `ids(steps[k])`, gathered natively into one staging buffer
    # the session copies at its start: no per-shard Python crossing by
    # operation name, no per-shard Array, and no second admission of the
    # ids by the trainer (it was ids() per shard, then train_step).
    var bi = native_loaded_module("builtins")
    if py_type(steps_value) is not bi.list and py_type(steps_value) is not bi.tuple:
        raise Error("TypeError: steps must be a list or tuple of step indices")
    var words = parallel_byte_shard_words(trainer)
    var count = words[0]
    var batch = words[1]
    var length = words[2]
    if len(steps_value) != count:
        raise Error("ValueError: logical shard count mismatch")
    var steps = List[Int]()
    for k in range(count):  # driver-loop: over the step's logical shards
        steps.append(corpus_int(steps_value[k], "step_index", True))
    var cells = batch * (length + 1)
    var staging = List[Int32](length=count * cells, fill=0)
    var base = Int(staging.unsafe_ptr())
    lm_corpus_gather_steps(owner, steps, batch, length, base, context)
    var shard_addresses = List[Int]()
    for k in range(count):  # small-loop(count: one staged buffer per logical shard)
        shard_addresses.append(base + k * cells * 4)
    var result = parallel_byte_step_shards(trainer, shard_addresses)
    _ = staging^
    return result


struct LmCorpusIteratorHandle(Movable, Writable):
    # The batches owner is admitted once, at prefetch: the iterator keeps the
    # typed request (core/lm_batch_run.mojo) and the token source snapshot
    # (which owns the mapped ids), so each step only allocates its output
    # and gathers natively.
    var owner: PythonObject
    var current: Int
    var stop: Int
    var closed: Bool
    var request: LmBatchRequest
    var ready: Bool
    var source: PythonObject

    def __init__(out self, owner: PythonObject, start: Int, stop: Int, request: LmBatchRequest):
        self.owner = owner
        self.current = start
        self.stop = stop
        self.closed = False
        self.request = request.copy()
        self.ready = False
        self.source = PythonObject(None)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("<native corpus batch iterator>")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def _prefetch(owner: PythonObject, start_value: PythonObject, steps_value: PythonObject,
              depth_value: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = lm_keys()
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
    var request = _batch_words(fields)
    var handle = LmCorpusIteratorHandle(owner, start, start + steps, request)
    # Bad dimensions keep raising at the first step, as before (not ready).
    if request.batch >= 0 and request.length >= 1:
        handle.source = _batch_source(fields, context, handle.request)
        handle.ready = True
    var iterator = corpus_new(py_item(context, nk[].k_iterator_class))
    var slots = corpus_fields(iterator, py_item(context, nk[].k_iterator_class), Python.tuple("_handle"))
    py_setitem(slots, nk[].k__handle, PythonObject(alloc=handle^))
    return iterator


def _iterator_next(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = lm_keys()
    var fields = corpus_fields(value, py_item(context, nk[].k_iterator_class), Python.tuple("_handle"))
    var owner = py_item(fields, nk[].k__handle)
    var handle = owner.downcast_value_ptr[LmCorpusIteratorHandle]()
    if handle[].closed or handle[].current >= handle[].stop:
        handle[].closed = True
        raise Error("StopIteration:")
    var step = handle[].current
    try:
        if not handle[].ready:
            raise Error("ValueError: tokens_gather_rows: bad dims")
        var batch = handle[].request.batch
        var length = handle[].request.length
        var base = corpus_batch_base(PythonObject(step), batch, length)
        var output = corpus_empty(Python.tuple(batch, length + 1), context)
        _ = lm_batch_run(handle[].request, base, Int(py=output._addr), Int(py=output.size))
        handle[].current += 1
        return Python.tuple(step, output)
    except error:
        handle[].closed = True
        raise error^


def _batch_describe(owner: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = lm_keys()
    var fields = _batch_fields(owner, context)
    var manifest = corpus_plain(py_item(fields, nk[].k_manifest))
    var vocabulary = py_item(manifest, nk[].k_vocabulary)
    var source = corpus_ids_snapshot(py_item(fields, nk[].k_ids_all), context)
    var lo = corpus_int(py_item(fields, nk[].k_lo), "lo", True)
    var hi = corpus_int(py_item(fields, nk[].k_hi), "hi", True)
    return Python.dict(path=PythonObject(corpus_run_join(corpus_path(py_item(fields, nk[].k_dir)), "tokens.i32")), sha256=corpus_plain(py_item(fields, nk[].k_sha256)),
        manifest_sha256=corpus_plain(py_item(fields, nk[].k_manifest_sha256)), tokens=source.size, train_range=Python.list(lo, hi),
        corpus_sha256=py_item(py_item(manifest, nk[].k_source), nk[].k_sha256), vocabulary_sha256=py_item(vocabulary, nk[].k_sha256), n_vocab=py_item(vocabulary, nk[].k_n_vocab),
        schedule=PythonObject("step k row b: ids[lo + (k*batch*length + b*length) % (hi - lo - length - 1) : +length+1] over the TRAIN range [lo, hi) in tokens; targets shifted one id"))


def _batch_schedule(owner: PythonObject, extra: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = lm_keys()
    var fields = _batch_fields(owner, context)
    var manifest = corpus_plain(py_item(fields, nk[].k_manifest))
    var vocabulary = py_item(manifest, nk[].k_vocabulary)
    var result = Python.dict(schema=PythonObject(TOKENS_SCHEMA), schedule=PythonObject(SCHEDULE), tokens_sha256=corpus_plain(py_item(fields, nk[].k_sha256)),
        corpus_sha256=py_item(py_item(manifest, nk[].k_source), nk[].k_sha256), batch=PythonObject(corpus_int(py_item(fields, nk[].k_batch), "batch", True)),
        length=PythonObject(corpus_int(py_item(fields, nk[].k_length), "length", True)),
        train_range=Python.list(corpus_int(py_item(fields, nk[].k_lo), "lo", True), corpus_int(py_item(fields, nk[].k_hi), "hi", True)),
        vocabulary=Python.dict(schema=py_item(vocabulary, nk[].k_schema), sha256=py_item(vocabulary, nk[].k_sha256), n_vocab=py_item(vocabulary, nk[].k_n_vocab)))
    corpus_dict(extra)
    py_call(result.update, extra)
    return result


def _expected_vocabulary(input: PythonObject) raises -> PythonObject:
    var nk = lm_keys()
    var bi = native_loaded_module("builtins")
    var model = input
    if py_type(model) is not bi.dict:
        # A trainer is a passive source of stored schedule metadata. Never
        # call its getter, __getattr__, descriptor, or schedule method.
        var ancestry = py_call(py_item(bi.type.__dict__, nk[].k___mro__).__get__, py_type(model))
        for i in range(len(ancestry)):  # driver-loop: raw trainer schedule descriptor admission
            var namespace = _class_fields(ancestry[i])
            if py_contains(namespace, nk[].k_data_schedule):
                var member = py_item(namespace, nk[].k_data_schedule)
                var bases = py_call(py_item(bi.type.__dict__, nk[].k___mro__).__get__, py_type(member))
                for j in range(len(bases)):  # small-loop(descriptor protocol ancestry)
                    if py_contains(_class_fields(bases[j]), nk[].k___get__):
                        raise Error("TypeError: vocabulary checks require a stored data_schedule or an explicit checkpoint/schedule dictionary")
        var fields = _fields(model)
        model = py_get(fields, nk[].k_data_schedule)
    if model is None or py_type(model) is not bi.dict:
        return PythonObject(None)
    corpus_dict(model)
    var nested = py_get(model, nk[].k_data_schedule)
    if nested is not None and py_type(nested) is bi.dict:
        model = nested
        corpus_dict(model)
    var vocabulary = py_get(model, nk[].k_vocabulary)
    if vocabulary is not None and py_type(vocabulary) is bi.dict:
        corpus_dict(vocabulary)
        if py_contains(vocabulary, nk[].k_sha256):
            return corpus_plain(vocabulary)
    return PythonObject(None)


def _require_vocabulary(model: PythonObject, tokenizer: PythonObject) raises -> PythonObject:
    var nk = lm_keys()
    var expected = _expected_vocabulary(model)
    if expected is None:
        raise Error("ValueError: mojolearn: this model carries no vocabulary identity (a byte model, or trained without lm_corpus); there is nothing to check a tokenizer against")
    var got = tokenizer_bpe_identity(tokenizer)
    if not Bool(py=py_item(got, nk[].k_sha256) == py_item(expected, nk[].k_sha256)) or not Bool(py=py_item(got, nk[].k_n_vocab) == py_item(expected, nk[].k_n_vocab)):
        raise Error("ValueError: mojolearn: vocabulary mismatch: the model and tokenizer have different vocabulary hashes or sizes; its ids would decode to the wrong text")
    return tokenizer


def _arity(args: PythonObject, count: Int) raises:
    if len(args) != count:
        raise Error("TypeError: corpus public operation received the wrong argument count")


def lm_corpus_public_call(name: PythonObject, args: PythonObject) raises -> PythonObject:
    var nk = lm_keys()
    # One lookup with the caller's own str (bindings/py_keys.mojo): the
    # former door copied the name into a String and made up to 24 compares
    # (17 before iterator_next, the per-batch call).
    var operation = py_op_code(nk[].ops, name)
    var bi = native_loaded_module("builtins")
    if py_type(args) is not bi.tuple:
        raise Error("TypeError: corpus operations require their native argument tuple")
    if operation == LM_OPN_CONNECT:
        return corpus_connect(args)
    var context = corpus_context()
    if operation == LM_OPN_PREPARE:
        _arity(args, 9)
        return _prepare(args, context)
    if operation == LM_OPN_TOKENIZE:
        _arity(args, 9)
        _progress(args[8])
        var document_bytes = _positive(args[6], "document_bytes")
        _ = _positive(args[7], "batch_documents")
        if py_type(args[0]) is not bi.bytes:
            raise Error("TypeError: corpus tokenization requires an immutable bytes snapshot")
        # UNPACK: the caller's source and vocabulary records as native JSON
        # nodes, the ranges as words, the tokenizer's native handle, the
        # snapshot's address.
        var data = args[0]
        corpus_dict(args[1])
        corpus_dict(args[2])
        corpus_dict(args[4])
        var doc = JsonDoc()
        var source = corpus_json_node(doc, args[1])
        var ranges = _ranges_unpack(args[2])
        var identity = corpus_json_node(doc, args[4])
        var handle = tokenizer_bpe_handle(args[3])
        var output = corpus_path(args[5])
        var address = cpython_table()[].PyBytes_AsString(data._obj_ptr)
        # RUN
        corpus_tokenize_run(doc, address, len(data), source, ranges, handle.downcast_value_ptr[BpeHandle]()[],
                            identity, output, document_bytes)
        _ = handle
        _ = data
        return PythonObject(None)
    if operation == LM_OPN_CORPUS_INIT:
        _arity(args, 3)
        return _corpus_init(args[0], args[1], args[2], context)
    if operation == LM_OPN_CORPUS_BATCHES:
        _arity(args, 3)
        var fields = _corpus_fields(args[0], context)
        return _batch_construct(py_item(fields, nk[].k_tokens_dir), args[1], args[2], context, py_get(fields, nk[].k__tokens_verified))
    if operation == LM_OPN_CORPUS_VOCABULARY or operation == LM_OPN_CORPUS_N_VOCAB or operation == LM_OPN_CORPUS_TOKENIZER or operation == LM_OPN_CORPUS_REPR:
        _arity(args, 1)
        var fields = _corpus_fields(args[0], context)
        var manifest = corpus_plain(py_item(fields, nk[].k_manifest))
        corpus_dict(manifest)
        var vocabulary = py_item(manifest, nk[].k_vocabulary)
        corpus_dict(vocabulary)
        if operation == LM_OPN_CORPUS_VOCABULARY:
            return py_call(bi.dict, vocabulary)
        if operation == LM_OPN_CORPUS_N_VOCAB:
            return PythonObject(corpus_int(py_item(vocabulary, nk[].k_n_vocab), "n_vocab"))
        if operation == LM_OPN_CORPUS_TOKENIZER:
            var tokenizer = tokenizer_bpe_for(py_item(fields, nk[].k_vocabulary_path))
            return _require_vocabulary(manifest, tokenizer)
        var count = corpus_int(py_item(manifest, nk[].k_tokens), "tokens")
        var n_vocab = corpus_int(py_item(vocabulary, nk[].k_n_vocab), "n_vocab")
        var path = corpus_path(py_item(fields, nk[].k_tokens_dir))
        return PythonObject("TokenizedCorpus(tokens=" + String(count) + ", n_vocab=" + String(n_vocab)
            + ", sha256=" + _prefix(py_item(manifest, nk[].k_sha256)) + ", dir=" + String(py_call(bi.repr, PythonObject(path))) + ")")
    if operation == LM_OPN_BATCHES_INIT:
        _arity(args, 4)
        return _batch_init(args[0], args[1], args[2], args[3], context, PythonObject(None))
    if operation == LM_OPN_BATCHES_IDS:
        _arity(args, 2)
        return _batch_ids(args[0], args[1], context, _batch_fields(args[0], context))
    if operation == LM_OPN_BATCHES_PARALLEL_STEP:
        _arity(args, 3)
        return _batch_parallel_step(args[0], args[1], args[2], context)
    if operation == LM_OPN_BATCHES_PREFETCH:
        _arity(args, 4)
        return _prefetch(args[0], args[1], args[2], args[3], context)
    if operation == LM_OPN_BATCHES_VOCABULARY:
        _arity(args, 1)
        var fields = _batch_fields(args[0], context)
        var manifest = corpus_plain(py_item(fields, nk[].k_manifest))
        corpus_dict(manifest)
        var vocabulary = py_item(manifest, nk[].k_vocabulary)
        corpus_dict(vocabulary)
        return py_call(bi.dict, vocabulary)
    if operation == LM_OPN_BATCHES_DESCRIBE:
        _arity(args, 1)
        return _batch_describe(args[0], context)
    if operation == LM_OPN_BATCHES_SCHEDULE:
        _arity(args, 2)
        return _batch_schedule(args[0], args[1], context)
    if operation == LM_OPN_ITERATOR_ITER:
        _arity(args, 1)
        var fields = corpus_fields(args[0], py_item(context, nk[].k_iterator_class), Python.tuple("_handle"))
        _ = py_item(fields, nk[].k__handle).downcast_value_ptr[LmCorpusIteratorHandle]()
        return args[0]
    if operation == LM_OPN_ITERATOR_NEXT:
        _arity(args, 1)
        return _iterator_next(args[0], context)
    if operation == LM_OPN_EXPECTED_VOCABULARY:
        _arity(args, 1)
        return _expected_vocabulary(args[0])
    if operation == LM_OPN_REQUIRE_VOCABULARY:
        _arity(args, 2)
        return _require_vocabulary(args[0], args[1])
    if operation == LM_OPN_TOKENIZER_FOR:
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
        var result = lm_corpus_public_call(args[0], args[1])
        return result.steal_data()
    except error:
        var message = String(error)
        if message == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error)
        if message.startswith(CORPUS_OS_ERROR):
            # PACK of a native file failure: OSError(errno, strerror, path).
            try:
                return optimizer_python_error(corpus_os_error_pending(message))
            except:
                return raise_python_exception(Error(message), ExceptionType("PyExc_OSError"))
        if message.startswith("KeyError:"):
            return raise_python_exception(error_without_class(error, "KeyError"), ExceptionType("PyExc_KeyError"))
        if message.startswith("OverflowError:"):
            return raise_python_exception(error_without_class(error, "OverflowError"), ExceptionType("PyExc_OverflowError"))
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
