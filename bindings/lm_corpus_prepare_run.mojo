# SPDX-License-Identifier: Apache-2.0
"""RUN of corpus preparation (`lm_corpus.prepare`, the private tokenize
step, and the manifest facts the corpus and batches constructors need):
typed values only, no Python.

The door (`bindings/lm_corpus_public_native_door.mojo`) hands over strings
and words (UNPACK) and wraps what comes back (PACK). Here: the corpus file is
hashed and counted in one streamed pass; its bytes are read once, only when a
cache miss needs them; the sibling corpus manifest, the vocabulary record and
the token manifest are `core/native_json_value.mojo` documents; the cache
directories are made and published by `bindings/lm_corpus_files_run.mojo`.

The vocabulary's rank text is `render_ranks` of the trained table, the text
the public `TrainedBpeVocabulary.render_ranks` writes, and the token ids are
`bpe_encode_corpus_into`'s: the same functions over the same bytes as before
this file existed, so a cached vocabulary or token directory made earlier is
found under the same key and verified by the same hashes.

Parallel plan: BPE training and the per-document encode split their work
across the host's cores inside `tokenizer/` (unchanged). With `gpu` set
(`prepare(device="gpu")`, lane vdf2-tok) the document cut, the training and
the encode run on the device through the GPU tokenizer image's entry
(`tokenizer/gpu/corpus_gpu.mojo`, resolved once at connection: `gpu_entry`),
called here with the GIL already released; the same vocabulary text and the
same ids come back, so the cache keys and hashes are the host column's. What is serial here
is serial by nature: one SHA-256 chain per file, one file write, and a
handful of manifest members.
"""
from bindings.lm_corpus_files_run import (
    corpus_run_atomic_abort, corpus_run_atomic_begin, corpus_run_atomic_commit, corpus_run_hash_file,
    corpus_run_is_file, corpus_run_join, corpus_run_parent, corpus_run_read, corpus_run_seconds,
    corpus_run_write, corpus_run_write_bytes,
)
from core.native_json_value import (
    JSON_ARRAY, JSON_NUMBER, JSON_OBJECT, JsonDoc, json_add_array, json_add_float, json_add_int, json_add_null,
    json_add_object, json_add_string, json_count, json_item, json_kind, json_member, json_number_equals,
    json_number_int, json_parse, json_push, json_put, json_string, json_string_is, json_truthy, json_write,
)
from core.sha256 import sha256_hex, sha256_hex_addr, sha256_hex_text
from tokenizer.encoding import string_bytes
from tokenizer.impl.tokenizer_host_shell_impl import (
    BpeHandle, BpeTrainedHandle, bpe_encode_corpus_into, bpe_handle_load, bpe_train_corpus_fill,
)
from tokenizer.impl.vocab_build import ranks_text
from tokenizer.train.bpe_train import BPE_TRAINER_SABOTAGE, TrainedVocabulary, trained_vocab_from_pairs
from core.entry_table import EntryRef, entry_ref_call, entry_ref_unresolved
from core.tokenizer_gpu_entry_table import TOK_GPU_OP_CORPUS_ENCODE, TOK_GPU_OP_CORPUS_TRAIN
from tokenizer.train.emit import render_ranks

comptime TOKENS_SCHEMA = "mojolearn.byte-lm.tokens.v1"
comptime CORPUS_SCHEMA = "mojolearn.byte-lm.corpus.v1"
comptime VOCABULARY_SCHEMA = "mojolearn.bpe-vocabulary.v1"
comptime SCHEDULE = "train-range-modulo.v1"


struct CorpusRange(Copyable, Movable):
    """One named range of the corpus, in bytes: `[lo, hi)`."""

    var name: String
    var lo: Int
    var hi: Int

    def __init__(out self, name: String, lo: Int, hi: Int):
        self.name = name.copy()
        self.lo = lo
        self.hi = hi


struct CorpusPrepareRequest(Movable):
    """`prepare`'s admitted arguments. `user` says the caller supplied a
    vocabulary: then `user_ranks` is its canonical rank text, `user_count`
    its rank count and `user_source` where it came from."""

    var corpus: String
    var cache: String
    var vocab_size: Int
    var min_frequency: Int
    var sample: Int
    var document_bytes: Int
    var user: Bool
    var user_ranks: String
    var user_count: Int
    var user_source: String
    var gpu: Bool
    var gpu_entry: EntryRef

    def __init__(out self, corpus: String, cache: String, vocab_size: Int, min_frequency: Int, sample: Int,
                 document_bytes: Int):
        self.corpus = corpus.copy()
        self.cache = cache.copy()
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.sample = sample
        self.document_bytes = document_bytes
        self.user = False
        self.user_ranks = String("")
        self.user_count = 0
        self.user_source = String("")
        self.gpu = False
        self.gpu_entry = entry_ref_unresolved()


struct TokensManifest(Movable):
    """A token directory's `manifest.json`: its bytes as read and the parsed
    document (`root` is an object with the token schema)."""

    var doc: JsonDoc
    var root: Int
    var raw: List[UInt8]

    def __init__(out self, var doc: JsonDoc, root: Int, var raw: List[UInt8]):
        self.doc = doc^
        self.root = root
        self.raw = raw^


struct CorpusPrepared(Movable):
    """What `prepare` found or made: the token directory, the rank file, the
    token manifest, and whether the token directory was already there (then
    the caller verifies the token file against the manifest's hash)."""

    var tokens_dir: String
    var ranks: String
    var cached: Bool
    var manifest: TokensManifest

    def __init__(out self, tokens_dir: String, ranks: String, cached: Bool, var manifest: TokensManifest):
        self.tokens_dir = tokens_dir.copy()
        self.ranks = ranks.copy()
        self.cached = cached
        self.manifest = manifest^


struct CorpusSource(Movable):
    """The corpus file: resolved path, digest, byte count, its `source`
    record (a node of the working document) and its named ranges."""

    var path: String
    var sha256: String
    var bytes: Int
    var node: Int
    var ranges: List[CorpusRange]

    def __init__(out self, path: String, sha256: String, bytes: Int):
        self.path = path.copy()
        self.sha256 = sha256.copy()
        self.bytes = bytes
        self.node = -1
        self.ranges = List[CorpusRange]()


def _prefix16(text: String) -> String:
    """The first sixteen characters of a hex digest."""
    var view = text.as_bytes()
    var bytes = List[UInt8]()
    for i in range(min(len(view), 16)):  # small-loop(sixteen fingerprint digits)
        bytes.append(view[i])
    var result = String(StringSlice(unsafe_from_utf8=Span(bytes)))
    _ = len(bytes)
    return result


def _write_json(doc: JsonDoc, node: Int, path: String) raises:
    var bytes = List[UInt8]()
    json_write(doc, node, bytes)
    bytes.append(UInt8(10))
    corpus_run_write_bytes(path, bytes)


def _range_offset(doc: JsonDoc, node: Int, name: String) raises -> Int:
    if json_kind(doc, node) != JSON_NUMBER:
        raise Error("TypeError: corpus " + name + " requires a native integer; Python conversion hooks are unsupported")
    return json_number_int(doc, node)


def tokens_manifest_read(directory: String, label: String) raises -> TokensManifest:
    """Read and admit `directory/manifest.json` (`label` is `corpus` or
    `token`, the word the schema refusal has always used)."""
    var raw = corpus_run_read(corpus_run_join(directory, "manifest.json"))
    var doc = JsonDoc()
    var root = json_parse(doc, raw)
    if json_kind(doc, root) != JSON_OBJECT:
        raise Error("TypeError: corpus metadata requires an ordinary dictionary")
    if not json_string_is(doc, json_member(doc, root, "schema"), TOKENS_SCHEMA):
        raise Error("ValueError: mojolearn: " + label + " manifest is not " + String(TOKENS_SCHEMA))
    return TokensManifest(doc^, root, raw^)


def tokens_manifest_sha(manifest: TokensManifest) -> String:
    """The token file's digest the manifest pins (empty when it pins none)."""
    return json_string(manifest.doc, json_member(manifest.doc, manifest.root, "sha256"))


def tokens_manifest_digest(manifest: TokensManifest) -> String:
    """The digest of the manifest file itself."""
    return sha256_hex(manifest.raw)


def tokens_manifest_payload(manifest: TokensManifest, token_file: String, size: Int, sha: String) raises:
    """The mapped token file is the one the manifest pins: digest and length."""
    var pinned = json_member(manifest.doc, manifest.root, "sha256")
    if (not json_string_is(manifest.doc, pinned, sha)
            or not json_number_equals(manifest.doc, json_member(manifest.doc, manifest.root, "bytes"), size)):
        raise Error("ValueError: mojolearn: pinned tokens length/SHA mismatch for " + token_file)


def tokens_manifest_range(manifest: TokensManifest, size: Int, length: Int, mut lo: Int, mut hi: Int) raises:
    """The train range in tokens (the whole file when the manifest names
    none) and the check that it holds one batch row."""
    var span = json_member(manifest.doc, manifest.root, "train_range")
    if not json_truthy(manifest.doc, span):
        lo = 0
        hi = size // 4
    else:
        if json_kind(manifest.doc, span) != JSON_ARRAY or json_count(manifest.doc, span) != 2:
            raise Error("ValueError: corpus train_range requires two token offsets")
        lo = _range_offset(manifest.doc, json_item(manifest.doc, span, 0), "train range start")
        hi = _range_offset(manifest.doc, json_item(manifest.doc, span, 1), "train range end")
    if length < 0 or length > 2147483645 or hi - lo < length + 2:
        raise Error("ValueError: mojolearn: the train range is shorter than one batch row")


def corpus_split_points(ranges: List[CorpusRange], count: Int) -> List[Int]:
    """0, `count` and every range end inside the corpus: sorted, each once."""
    var points = List[Int]()
    points.append(0)
    points.append(count)
    for i in range(len(ranges)):  # small-loop(declared train/validation/test boundaries)
        if ranges[i].lo >= 0 and ranges[i].lo <= count:
            points.append(ranges[i].lo)
        if ranges[i].hi >= 0 and ranges[i].hi <= count:
            points.append(ranges[i].hi)
    for i in range(1, len(points)):  # small-loop(insertion sort of at most eight split points)
        var value = points[i]
        var at = i
        while at > 0 and points[at - 1] > value:
            points[at] = points[at - 1]
            at -= 1
        points[at] = value
    var result = List[Int]()
    for i in range(len(points)):  # small-loop(unique sorted manifest split points)
        if i == 0 or points[i] != points[i - 1]:
            result.append(points[i])
    return result^


def corpus_source_run(mut doc: JsonDoc, path: String) raises -> CorpusSource:
    """Hash and count the corpus; read the corpus manifest beside it when
    there is one with the corpus schema (it pins the length and digest and
    names the ranges)."""
    var size = 0
    var sha = corpus_run_hash_file(path, size)
    var source = CorpusSource(path, sha, size)
    var node = json_add_object(doc)
    json_put(doc, node, "path", json_add_string(doc, path))
    json_put(doc, node, "sha256", json_add_string(doc, sha))
    json_put(doc, node, "bytes", json_add_int(doc, size))
    json_put(doc, node, "schema", json_add_null(doc))
    var manifest = corpus_run_join(corpus_run_parent(path), "manifest.json")
    if corpus_run_is_file(manifest):
        var raw = corpus_run_read(manifest)
        var record = json_parse(doc, raw)
        if json_string_is(doc, json_member(doc, record, "schema"), CORPUS_SCHEMA):
            if (not json_string_is(doc, json_member(doc, record, "sha256"), sha)
                    or not json_number_equals(doc, json_member(doc, record, "bytes"), size)):
                raise Error("ValueError: mojolearn: pinned corpus length/SHA mismatch for " + path)
            json_put(doc, node, "schema", json_add_string(doc, CORPUS_SCHEMA))
            json_put(doc, node, "manifest_sha256", json_add_string(doc, sha256_hex(raw)))
            var url = json_member(doc, record, "source_url")
            json_put(doc, node, "source_url", url if url >= 0 else json_add_null(doc))
            for key in [String("train_range"), String("validation_range"), String("test_range")]:  # small-loop(three named corpus ranges)
                var span = json_member(doc, record, key)
                if json_truthy(doc, span):
                    if json_kind(doc, span) != JSON_ARRAY or json_count(doc, span) != 2:
                        raise Error("ValueError: corpus ranges require two offsets")
                    source.ranges.append(CorpusRange(key,
                        _range_offset(doc, json_item(doc, span, 0), "range start"),
                        _range_offset(doc, json_item(doc, span, 1), "range end")))
    var has_train = False
    for i in range(len(source.ranges)):  # small-loop(three named corpus ranges)
        has_train = has_train or source.ranges[i].name == "train_range"
    if not has_train:
        source.ranges = List[CorpusRange]()
        source.ranges.append(CorpusRange("train_range", 0, size))
    source.node = node
    return source^


def _identity_node(mut doc: JsonDoc, sha: String, ranks: Int) -> Int:
    var node = json_add_object(doc)
    json_put(doc, node, "schema", json_add_string(doc, VOCABULARY_SCHEMA))
    json_put(doc, node, "sha256", json_add_string(doc, sha))
    json_put(doc, node, "n_ranks", json_add_int(doc, ranks))
    json_put(doc, node, "n_vocab", json_add_int(doc, ranks + 1))
    json_put(doc, node, "endoftext_id", json_add_int(doc, ranks))
    return node


def _vocabulary_write(mut doc: JsonDoc, directory: String, text: String, identity: Int, recipe: Int) raises:
    """`ranks.tsv` (the canonical rank text) and `vocabulary.json`."""
    corpus_run_write_bytes(corpus_run_join(directory, "ranks.tsv"), string_bytes(text))
    var record = json_add_object(doc)
    json_put(doc, record, "schema", json_add_string(doc, VOCABULARY_SCHEMA))
    json_put(doc, record, "identity", identity)
    json_put(doc, record, "recipe", recipe)
    json_put(doc, record, "scope", json_add_string(doc, "trained on or loaded for the caller's corpus; never shipped with mojolearn"))
    _write_json(doc, record, corpus_run_join(directory, "vocabulary.json"))


def _vocabulary_check(mut doc: JsonDoc, directory: String, mut handle: BpeHandle, mut sha: String,
                      mut text: String) raises -> Int:
    """Load the cached rank file ONCE, hash its canonical text (kept in
    `text`: the GPU encoder parses the same text), and require the vocabulary
    record to pin that hash; returns the record's identity."""
    var raw = corpus_run_read(corpus_run_join(directory, "vocabulary.json"))
    var record = json_parse(doc, raw)
    bpe_handle_load(handle, corpus_run_join(directory, "ranks.tsv"))
    text = ranks_text(handle.tok.value().ranks)
    sha = sha256_hex_text(text)
    var identity = json_member(doc, record, "identity")
    if not json_string_is(doc, json_member(doc, identity, "sha256"), sha):
        raise Error("ValueError: mojolearn: cached vocabulary does not match its vocabulary.json; remove it to rebuild")
    return identity


def _train_range(ranges: List[CorpusRange], mut lo: Int, mut hi: Int):
    for i in range(len(ranges)):  # small-loop(three named corpus ranges)
        if ranges[i].name == "train_range":
            lo = ranges[i].lo
            hi = ranges[i].hi


def _source_data(mut data: List[UInt8], mut have: Bool, source: CorpusSource) raises:
    """The corpus bytes, read once, on the first cache miss that needs them.
    They must be the bytes the source pass hashed: a file changed in between
    is refused (the vocabulary and token files are keyed by that hash)."""
    if have:
        return
    data = corpus_run_read(source.path)
    if len(data) != source.bytes or sha256_hex(data) != source.sha256:
        raise Error("ValueError: mojolearn: the corpus changed while it was being prepared; run prepare again")
    have = True


def _gpu_train(request: CorpusPrepareRequest, data: Int, points: List[Int], lo: Int, hi: Int,
               width: Int) raises -> TrainedVocabulary:
    """The host `bpe_train_corpus_fill` on the device (TOK_GPU_OP_CORPUS_TRAIN): the span
    `[lo, hi)` cut at the split points strictly inside it, trained; the merges come back and the
    vocabulary is written from them as train_bpe writes it (`trained_vocab_from_pairs`)."""
    var cuts = List[Int64]()
    cuts.append(Int64(0))
    for k in range(len(points)):  # small-loop(at most ten split points)
        if points[k] > lo and points[k] < hi:
            cuts.append(Int64(points[k] - lo))
    cuts.append(Int64(hi - lo))
    var cap = max(2 * (request.vocab_size - 256), 2)
    var pairs = List[Int32](length=cap, fill=Int32(0))
    var words = List[Int64](length=3, fill=Int64(0))
    var addresses: List[Int] = [data + lo, Int(cuts.unsafe_ptr()), Int(pairs.unsafe_ptr()), Int(words.unsafe_ptr())]
    var ints: List[Int] = [hi - lo, len(cuts), width, request.vocab_size, request.min_frequency, cap]
    var floats = List[Float64]()
    var n_docs = entry_ref_call(request.gpu_entry, TOK_GPU_OP_CORPUS_TRAIN, addresses, ints, floats)
    if n_docs < 1:
        raise Error("ValueError: train needs at least one document")
    var n_merges = Int(words[0])
    if n_merges < 0 or 2 * n_merges > cap:
        raise Error("RuntimeError: lm_corpus.prepare (gpu): the trainer returned " + String(n_merges) + " merges")
    var vocab = trained_vocab_from_pairs(
        MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(pairs.unsafe_ptr())), n_merges,
        Int(words[1]), Int(words[2]))
    _ = len(cuts)
    _ = len(pairs)
    _ = len(words)
    return vocab^


def _trained_vocabulary(mut doc: JsonDoc, request: CorpusPrepareRequest, source: CorpusSource,
                        mut data: List[UInt8], mut have: Bool) raises -> String:
    """The cache directory of the vocabulary trained on the sample of the
    train range, trained and written when it is not there yet."""
    var lo = 0
    var hi = 0
    _train_range(source.ranges, lo, hi)
    hi = min(hi, lo + request.sample)
    var key = (_prefix16(source.sha256) + "-v" + String(request.vocab_size) + "-f" + String(request.min_frequency)
               + "-s" + String(hi - lo))
    var final = corpus_run_join(corpus_run_join(request.cache, "vocab"), key)
    if corpus_run_is_file(corpus_run_join(final, "vocabulary.json")):
        return final
    _source_data(data, have, source)
    var temporary = corpus_run_atomic_begin(final)
    try:
        var began = corpus_run_seconds()
        var points = corpus_split_points(source.ranges, len(data))
        if lo < 0 or hi > len(data) or lo > hi:
            raise Error("ValueError: bpe_train_corpus: [lo, hi) must lie inside the corpus")
        # The trainer sees only [lo, hi); a larger positive document limit has the
        # same cuts as this span length, without adding Int.MAX to a split offset.
        var width = min(request.document_bytes, max(hi - lo, 1))
        var trained = BpeTrainedHandle()
        if request.gpu:
            trained.vocab = _gpu_train(request, Int(data.unsafe_ptr()), points, lo, hi, width)
        else:
            # The verify sabotage arm is the build define -D MOJOLEARN_BPE_TRAINER_SABOTAGE.
            bpe_train_corpus_fill(trained, Int(data.unsafe_ptr()), points, lo, hi, width, request.vocab_size,
                                  request.min_frequency, BPE_TRAINER_SABOTAGE)
        var seconds = corpus_run_seconds() - began
        var text = render_ranks(trained.vocab.value())
        var n_tokens = trained.vocab.value().n_tokens()
        var recipe = json_add_object(doc)
        json_put(doc, recipe, "trainer", json_add_string(doc, "mojolearn.tokenizer.BpeVocabularyTrainer"))
        json_put(doc, recipe, "backend", json_add_string(doc, "mojo"))
        json_put(doc, recipe, "format", json_add_string(doc, "mojolearn.bpe-trainer.v1"))
        json_put(doc, recipe, "tie_break", json_add_string(doc, "highest count, then smallest (left_id, right_id)"))
        json_put(doc, recipe, "vocab_size", json_add_int(doc, request.vocab_size))
        json_put(doc, recipe, "min_frequency", json_add_int(doc, request.min_frequency))
        json_put(doc, recipe, "corpus_sha256", json_add_string(doc, source.sha256))
        var sample = json_add_array(doc)
        json_push(doc, sample, json_add_int(doc, lo))
        json_push(doc, sample, json_add_int(doc, hi))
        json_put(doc, recipe, "sample", sample)
        json_put(doc, recipe, "document_bytes", json_add_int(doc, request.document_bytes))
        json_put(doc, recipe, "n_tokens", json_add_int(doc, n_tokens))
        json_put(doc, recipe, "n_merges", json_add_int(doc, trained.vocab.value().n_merges()))
        json_put(doc, recipe, "n_ties_broken", json_add_int(doc, trained.vocab.value().n_ties_broken))
        json_put(doc, recipe, "train_seconds", json_add_float(doc, seconds))
        _vocabulary_write(doc, temporary, text, _identity_node(doc, sha256_hex_text(text), n_tokens), recipe)
        corpus_run_atomic_commit(temporary, final)
    except error:
        corpus_run_atomic_abort(temporary)
        raise error^
    _ = len(data)
    return final


def _user_vocabulary(mut doc: JsonDoc, request: CorpusPrepareRequest) raises -> String:
    """The cache directory of the caller's vocabulary, written when new."""
    var sha = sha256_hex_text(request.user_ranks)
    var final = corpus_run_join(corpus_run_join(request.cache, "vocab"), "user-" + _prefix16(sha))
    if corpus_run_is_file(corpus_run_join(final, "vocabulary.json")):
        return final
    var temporary = corpus_run_atomic_begin(final)
    try:
        var recipe = json_add_object(doc)
        json_put(doc, recipe, "source", json_add_string(doc, request.user_source))
        json_put(doc, recipe, "trainer", json_add_null(doc))
        _vocabulary_write(doc, temporary, request.user_ranks, _identity_node(doc, sha, request.user_count), recipe)
        corpus_run_atomic_commit(temporary, final)
    except error:
        corpus_run_atomic_abort(temporary)
        raise error^
    return final


def _point_index(points: List[Int], value: Int) raises -> Int:
    for i in range(len(points)):  # small-loop(at most eight split points)
        if points[i] == value:
            return i
    raise Error("ValueError: mojolearn: a corpus range offset lies outside the corpus")


def corpus_tokenize_run(mut doc: JsonDoc, data: Int, count: Int, source: Int, ranges: List[CorpusRange],
                        handle: BpeHandle, identity: Int, output: String, document_bytes: Int) raises:
    """Encode the `count` corpus bytes at `data` document by document and
    write `tokens.i32` and `manifest.json` into `output`. `source` and
    `identity` are nodes of `doc` (the manifest's `source` and `vocabulary`)."""
    corpus_tokenize_run_on(doc, data, count, source, ranges, handle, identity, output, document_bytes, False,
                           entry_ref_unresolved(), String(""))


def _gpu_encode(entry: EntryRef, data: Int, count: Int, points: List[Int], width: Int, ranks: String,
                ids: Int, bounds: Int, stats: Int) raises -> Int:
    """The host `bpe_encode_corpus_into` on the device (TOK_GPU_OP_CORPUS_ENCODE): the same ids,
    point bounds and statistics, written at the same addresses."""
    var marks = List[Int64](capacity=len(points))
    for k in range(len(points)):  # small-loop(at most ten split points)
        marks.append(Int64(points[k]))
    var addresses: List[Int] = [data, Int(marks.unsafe_ptr()), Int(ranks.unsafe_ptr()), ids, bounds, stats]
    var ints: List[Int] = [count, len(marks), width, ranks.byte_length()]
    var floats = List[Float64]()
    var total = entry_ref_call(entry, TOK_GPU_OP_CORPUS_ENCODE, addresses, ints, floats)
    _ = len(marks)
    _ = ranks.byte_length()
    return total


def corpus_tokenize_run_on(mut doc: JsonDoc, data: Int, count: Int, source: Int, ranges: List[CorpusRange],
                           handle: BpeHandle, identity: Int, output: String, document_bytes: Int, gpu: Bool,
                           entry: EntryRef, ranks: String) raises:
    """`corpus_tokenize_run` on the side `gpu` names: the device encoder through `entry` (over the
    canonical rank text `ranks` of `handle`'s tokenizer) or the host one."""
    if not handle.tok:
        raise Error("ValueError: tokenizer handle is not loaded")
    if document_bytes < 1:
        raise Error("ValueError: document_bytes must be >= 1")
    var began = corpus_run_seconds()
    var points = corpus_split_points(ranges, count)
    # No document can exceed its corpus. Normalize a larger positive limit
    # before the cutter adds it to a later split offset; an otherwise valid
    # Int.MAX limit must not wrap that addition.
    var width = min(document_bytes, max(count, 1))
    var ids = List[Int32](length=max(count, 1), fill=0)
    var bounds = List[Int64](length=max(len(points), 1), fill=0)
    var stats = List[Int64](length=3, fill=0)
    var total = 0
    if gpu:
        total = _gpu_encode(entry, data, count, points, width, ranks, Int(ids.unsafe_ptr()),
                            Int(bounds.unsafe_ptr()), Int(stats.unsafe_ptr()))
    else:
        total = bpe_encode_corpus_into(handle, data, points, count, len(points), width, count,
                                       Int(ids.unsafe_ptr()), Int(bounds.unsafe_ptr()), Int(stats.unsafe_ptr()))
    if total < 0 or total > count:
        raise Error("RuntimeError: corpus encoder returned an invalid token count")
    var seconds = corpus_run_seconds() - began
    # Supported targets are little endian; the id buffer is the on-disk int32 stream.
    var payload = total * 4
    var manifest = json_add_object(doc)
    json_put(doc, manifest, "schema", json_add_string(doc, TOKENS_SCHEMA))
    json_put(doc, manifest, "source", source)
    json_put(doc, manifest, "vocabulary", identity)
    if gpu:
        json_put(doc, manifest, "encoder", json_add_string(doc, "mojolearn.tokenizer.BpeTokenizer.encode_batch on the GPU (binding _mojolearn_x_tokenizer, tokenizer/gpu/corpus_gpu.mojo); integers and tables only, the host column's ids"))
    else:
        json_put(doc, manifest, "encoder", json_add_string(doc, "mojolearn.tokenizer.BpeTokenizer.encode_batch (host binding _mojolearn_tokenizer_host, tokenizer/encoding.mojo); integers and tables only"))
    json_put(doc, manifest, "endoftext", json_add_string(doc, "not inserted and not recognized (allow_endoftext=False)"))
    json_put(doc, manifest, "document_rule", json_add_string(doc, "no document spans a declared range boundary; at most " + String(document_bytes) + " bytes, ending at the last 0x0A at or before the limit (or at the limit when the window holds none); each encoded alone, ids concatenated in document order"))
    json_put(doc, manifest, "document_bytes", json_add_int(doc, document_bytes))
    json_put(doc, manifest, "n_documents", json_add_int(doc, Int(stats[0])))
    json_put(doc, manifest, "dtype", json_add_string(doc, "int32"))
    json_put(doc, manifest, "byte_order", json_add_string(doc, "little"))
    json_put(doc, manifest, "sha256", json_add_string(doc, sha256_hex_addr(Int(ids.unsafe_ptr()), payload)))
    json_put(doc, manifest, "bytes", json_add_int(doc, payload))
    json_put(doc, manifest, "tokens", json_add_int(doc, total))
    json_put(doc, manifest, "bytes_per_token",
             json_add_float(doc, Float64(count) / Float64(total)) if total != 0 else json_add_null(doc))
    json_put(doc, manifest, "max_id", json_add_int(doc, Int(stats[1])))
    json_put(doc, manifest, "ids_above_255", json_add_int(doc, Int(stats[2])))
    for i in range(len(ranges)):  # small-loop(three named byte ranges mapped to token offsets)
        var span = json_add_array(doc)
        json_push(doc, span, json_add_int(doc, Int(bounds[_point_index(points, ranges[i].lo)])))
        json_push(doc, span, json_add_int(doc, Int(bounds[_point_index(points, ranges[i].hi)])))
        json_put(doc, manifest, ranges[i].name, span)
    json_put(doc, manifest, "schedule", json_add_string(doc, SCHEDULE))
    json_put(doc, manifest, "tokenize_seconds", json_add_float(doc, seconds))
    json_put(doc, manifest, "encode_seconds", json_add_float(doc, seconds))
    json_put(doc, manifest, "encode_bytes_per_second",
             json_add_float(doc, Float64(count) / seconds) if seconds != 0.0 else json_add_null(doc))
    corpus_run_write(corpus_run_join(output, "tokens.i32"), Int(ids.unsafe_ptr()), payload)
    _write_json(doc, manifest, corpus_run_join(output, "manifest.json"))
    _ = len(ids)
    _ = len(bounds)
    _ = len(stats)


def corpus_prepare_run(request: CorpusPrepareRequest) raises -> CorpusPrepared:
    """`prepare`: find or make the vocabulary directory, then find or make
    the token directory keyed by corpus digest, vocabulary digest and
    document size."""
    var doc = JsonDoc()
    var source = corpus_source_run(doc, request.corpus)
    var data = List[UInt8]()
    var have = False
    var vocabulary = String("")
    if request.user:
        vocabulary = _user_vocabulary(doc, request)
    else:
        vocabulary = _trained_vocabulary(doc, request, source, data, have)
    var handle = BpeHandle()
    var vocabulary_sha = String("")
    var vocabulary_text = String("")
    var identity = _vocabulary_check(doc, vocabulary, handle, vocabulary_sha, vocabulary_text)
    var ranks = corpus_run_join(vocabulary, "ranks.tsv")
    var key = _prefix16(source.sha256) + "-" + _prefix16(vocabulary_sha) + "-d" + String(request.document_bytes)
    var final = corpus_run_join(corpus_run_join(request.cache, "tokens"), key)
    if corpus_run_is_file(corpus_run_join(final, "manifest.json")):
        var cached = tokens_manifest_read(final, "corpus")
        var made_from = json_member(cached.doc, cached.root, "source")
        var made_with = json_member(cached.doc, cached.root, "vocabulary")
        if (not json_string_is(cached.doc, json_member(cached.doc, made_from, "sha256"), source.sha256)
                or not json_string_is(cached.doc, json_member(cached.doc, made_with, "sha256"), vocabulary_sha)):
            raise Error("ValueError: mojolearn: cached tokens were made from another corpus or vocabulary; remove the directory to rebuild")
        return CorpusPrepared(final, ranks, True, cached^)
    _source_data(data, have, source)
    var temporary = corpus_run_atomic_begin(final)
    try:
        corpus_tokenize_run_on(doc, Int(data.unsafe_ptr()), len(data), source.node, source.ranges, handle, identity,
                               temporary, request.document_bytes, request.gpu, request.gpu_entry, vocabulary_text)
        corpus_run_atomic_commit(temporary, final)
    except error:
        corpus_run_atomic_abort(temporary)
        raise error^
    _ = len(data)
    var made = tokens_manifest_read(final, "corpus")
    return CorpusPrepared(final, ranks, False, made^)
