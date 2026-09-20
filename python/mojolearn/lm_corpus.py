# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Tokenize a training corpus ONCE, and train the language model on the ids.

    from mojolearn import lm_corpus, LanguageModelTrainer, LanguageModelConfig

    corpus = lm_corpus.prepare("corpus/input.txt")                   # trains OUR vocabulary
    corpus = lm_corpus.prepare("corpus/input.txt", vocab="ranks.tsv")  # YOUR token map
    corpus = lm_corpus.prepare("corpus/input.txt",
                               vocab=("encoder.json", "vocab.bpe"))    # YOUR token map
    batches = corpus.batches(batch=1, length=2048)
    shape = LanguageModelConfig(..., vocab_size=corpus.n_vocab)
    trainer = LanguageModelTrainer(weights, shape=shape, data_schedule=batches.data_schedule())
    trainer.train_step(batches.ids(0))
    tok = lm_corpus.tokenizer_for(trainer, corpus.vocabulary_path)     # refuses any other table

THE TOKENIZER IS PART OF THE MODEL. It is chosen once, before the first step,
and every id the model ever sees or emits means something only through it. So
tokenization is the DEFAULT here, not an add-on:

  * with no `vocab`, `prepare` trains a byte-level BPE vocabulary with OUR OWN
    `BpeVocabularyTrainer` on the corpus's TRAIN range (never validation or
    test), tokenizes the whole corpus once, and caches the pinned id array;
  * `vocab=` uses the caller's token map instead and trains nothing: a rank
    file, `(encoder.json, vocab.bpe)`, a `BpeTokenizer` or a
    `TrainedBpeVocabulary`;
  * byte training (ids = raw bytes, vocabulary 256) is still there by explicit
    choice: `tools/lm_train.py --bytes`, whose batches are
    `tools/lm_step_memory_probe.py::CorpusBatches`, unchanged.

THE CACHE. Under `cache_dir` (default `$MOJOLEARN_LM_CACHE`, else
`~/.cache/mojolearn/lm`):

    vocab/<corpus sha16>-v<vocab_size>-f<min_frequency>-s<sample bytes>/
        ranks.tsv, vocabulary.json            (a trained vocabulary)
    vocab/user-<vocabulary sha16>/ranks.tsv, vocabulary.json   (yours, canonical)
    tokens/<corpus sha16>-<vocabulary sha16>-d<document bytes>/
        tokens.i32, manifest.json             (schema TOKENS_SCHEMA)

A rerun with the same corpus and vocabulary reads the cache: the corpus and
the id array are re-hashed against the manifest, and a mismatch is refused by
name rather than silently rebuilt. Nothing here is ever written into the
package or the repository: a vocabulary trained on someone's text is derived
from that text, and mojolearn ships no vocabulary and no corpus.

THE VOCABULARY TRAVELS WITH THE MODEL. `TokenBatches.data_schedule()` carries
the vocabulary's identity (`BpeTokenizer.identity`: the sha256 of its
canonical rank file and its `n_vocab`). `LanguageModelTrainer` refuses a
schedule whose `n_vocab` is not its `vocab_size`, and keeps the schedule in
every checkpoint under the envelope's payload sha256. `tokenizer_for(model,
vocabulary)` loads a tokenizer only if it IS that vocabulary, by sha256, and
refuses by name otherwise, so ids are not decoded with the wrong table.

DOCUMENTS ARE PART OF THE ARTIFACT. BPE never merges across a pre-token
boundary, and where a long file is cut into pieces changes the ids at every
cut. So the cut is declared and recorded: a document never spans the source
manifest's train / validation / test boundaries (so each split's token range
is exact), is at most `document_bytes`, and ends at the last 0x0A at or before
that limit (or at the limit exactly when the window holds none). Each
document is encoded ALONE and the ids are concatenated in order.
`<|endoftext|>` is not inserted and not recognized in the text: the byte path
inserts no separator either.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from . import tokenizer as _tokenizer

__all__ = ["prepare", "TokenizedCorpus", "TokenBatches", "require_vocabulary",
           "tokenizer_for"]

TOKENS_SCHEMA = "mojolearn.byte-lm.tokens.v1"
CORPUS_SCHEMA = "mojolearn.byte-lm.corpus.v1"
VOCABULARY_SCHEMA = _tokenizer.VOCABULARY_SCHEMA
#: GPT-2 / GPT-3 Small's vocabulary size is 50,257 ids: 50,256 ranks and
#: `<|endoftext|>`. The default trains to that size.
DEFAULT_VOCAB_SIZE = 50256
DEFAULT_MIN_FREQUENCY = 2
#: the vocabulary is trained on at most this many bytes from the start of the
#: TRAIN range (see `prepare`)
DEFAULT_VOCAB_SAMPLE_BYTES = 10_000_000
DEFAULT_DOCUMENT_BYTES = 1 << 20
DEFAULT_BATCH_DOCUMENTS = 16
_SCHEDULE = "train-range-modulo.v1"


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                return h.hexdigest()
            h.update(block)


def default_cache_dir():
    return Path(os.environ.get("MOJOLEARN_LM_CACHE") or Path.home() / ".cache" / "mojolearn" / "lm")


# ---------------------------------------------------------------- the corpus

def _source(corpus_path):
    """`(bytes, source dict, ranges)` for a corpus file. A pinned corpus
    (`manifest.json` beside it, schema `mojolearn.byte-lm.corpus.v1`) is
    checked against its sha256 and length, and its declared ranges are used.
    Any other file is one train range covering the whole file."""
    path = Path(corpus_path).expanduser().resolve()
    data = path.read_bytes()
    sha = _sha_bytes(data)
    manifest_path = path.with_name("manifest.json")
    ranges = {}
    src = dict(path=str(path), sha256=sha, bytes=len(data), schema=None)
    if manifest_path.is_file():
        raw = manifest_path.read_bytes()
        m = json.loads(raw)
        if isinstance(m, dict) and m.get("schema") == CORPUS_SCHEMA:
            if m.get("sha256") != sha or m.get("bytes") != len(data):
                raise ValueError(f"mojolearn: pinned corpus length/SHA mismatch for {path} "
                                 f"(manifest {m.get('sha256')}, {m.get('bytes')} bytes; file {sha}, {len(data)})")
            src.update(schema=CORPUS_SCHEMA, manifest_sha256=_sha_bytes(raw), source_url=m.get("source_url"))
            for key in ("train_range", "validation_range", "test_range"):
                r = m.get(key)
                if r:
                    ranges[key] = [int(r[0]), int(r[1])]
    if "train_range" not in ranges:
        ranges = {"train_range": [0, len(data)]}
    return data, src, ranges


def _split_points(ranges, n):
    points = {0, n}
    for r in ranges.values():
        for x in r:
            if 0 <= x <= n:
                points.add(x)
    return sorted(points)


def documents(data, ranges, document_bytes, lo=None, hi=None):
    """`[(start, stop)]` by the declared rule, half-open and in order,
    covering `[lo, hi)` (default the whole file)."""
    n = len(data)
    lo = 0 if lo is None else lo
    hi = n if hi is None else hi
    out = []
    points = [p for p in _split_points(ranges, n) if lo < p < hi]
    for a, b in zip([lo] + points, points + [hi]):
        at = a
        while at < b:
            stop = min(at + document_bytes, b)
            if stop < b:
                cut = data.rfind(b"\n", at, stop)
                if cut != -1 and cut + 1 > at:
                    stop = cut + 1
            out.append((at, stop))
            at = stop
    return out


# ---------------------------------------------------------------- vocabulary

def _atomic_dir(final, fill):
    """Build `final` in a temporary sibling and rename it into place, so a
    killed run never leaves a half-written cache entry that a rerun trusts."""
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=final.name + ".partial-", dir=final.parent))
    try:
        fill(tmp)
        if final.exists():
            shutil.rmtree(tmp)
        else:
            os.rename(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def _vocabulary_record(identity, recipe):
    return dict(schema=VOCABULARY_SCHEMA, identity=identity, recipe=recipe,
                scope="trained on or loaded for the caller's corpus; never shipped with mojolearn")


def _check_vocab_dir(d):
    rec = json.loads((d / "vocabulary.json").read_text())
    ident = _tokenizer._identity_of_ranks_file(str(d / "ranks.tsv"))
    if ident["sha256"] != rec["identity"]["sha256"]:
        raise ValueError(f"mojolearn: cached vocabulary {d} does not match its vocabulary.json "
                         f"({ident['sha256']} against {rec['identity']['sha256']}); remove it to rebuild")
    return rec


def _trained_vocabulary(data, ranges, sha, cache, vocab_size, min_frequency, sample_bytes, document_bytes,
                        progress=None):
    lo, hi = ranges["train_range"]
    hi = min(hi, lo + sample_bytes)
    key = f"{sha[:16]}-v{vocab_size}-f{min_frequency}-s{hi - lo}"
    final = cache / "vocab" / key
    if (final / "vocabulary.json").is_file():
        return final, _check_vocab_dir(final)

    def fill(tmp):
        docs = [bytes(data[a:b]) for a, b in documents(data, ranges, document_bytes, lo, hi)]
        t0 = time.perf_counter()
        if progress:
            progress(f"training a {vocab_size}-rank vocabulary on {hi - lo} bytes ({len(docs)} documents)")
        v = _tokenizer.BpeVocabularyTrainer(vocab_size=vocab_size, min_frequency=min_frequency).train(docs)
        seconds = time.perf_counter() - t0
        v.write_ranks(str(tmp / "ranks.tsv"))
        recipe = dict(trainer="mojolearn.tokenizer.BpeVocabularyTrainer", backend=v.stats.get("backend"),
                      format=_tokenizer._bpe_trainer.FORMAT,
                      tie_break=v.tie_break, vocab_size=vocab_size, min_frequency=min_frequency,
                      corpus_sha256=sha, sample=[lo, hi], document_bytes=document_bytes,
                      n_tokens=v.n_tokens, n_merges=len(v.merges), n_ties_broken=v.n_ties_broken,
                      train_seconds=seconds)
        (tmp / "vocabulary.json").write_text(json.dumps(_vocabulary_record(v.identity, recipe), indent=2) + "\n")

    final = _atomic_dir(final, fill)
    return final, _check_vocab_dir(final)


def _user_vocabulary(vocab, cache):
    """A caller's token map, loaded and written canonically into the cache."""
    if isinstance(vocab, _tokenizer.TrainedBpeVocabulary):
        tokens = vocab.tokens
        source = "TrainedBpeVocabulary"
    else:
        if isinstance(vocab, _tokenizer.BpeTokenizer):
            tok = vocab
        elif isinstance(vocab, (tuple, list)) and len(vocab) == 2:
            tok = _tokenizer.BpeTokenizer.from_files(os.fspath(vocab[0]), os.fspath(vocab[1]))
        elif isinstance(vocab, (str, os.PathLike)):
            tok = _tokenizer.BpeTokenizer.from_ranks_file(os.fspath(vocab))
        else:
            raise TypeError("mojolearn: vocab must be a rank file path, (encoder_json, vocab_bpe), a BpeTokenizer "
                            f"or a TrainedBpeVocabulary, got {type(vocab).__name__}")
        tokens = _tokens_by_decoding(tok)
        source = tok.vocabulary_source
    text = _tokenizer._bpe_trainer.render_ranks(tokens)
    identity = _tokenizer._identity_from_tokens_text(text, len(tokens))
    final = cache / "vocab" / f"user-{identity['sha256'][:16]}"
    if not (final / "vocabulary.json").is_file():
        def fill(tmp):
            (tmp / "ranks.tsv").write_text(text, encoding="ascii")
            (tmp / "vocabulary.json").write_text(json.dumps(
                _vocabulary_record(identity, dict(source=str(source), trainer=None)), indent=2) + "\n")
        _atomic_dir(final, fill)
    return final, _check_vocab_dir(final)


def _tokens_by_decoding(tok):
    """Every rank's bytes, read back through `decode_bytes` one id at a time
    (the binding exposes no token listing; `NOT_IMPLEMENTED.tsv`)."""
    return [tok.decode_bytes([i]) for i in range(tok.n_vocab - 1)]


# ---------------------------------------------------------------- tokenizing

def _tokenize(data, src, ranges, tok, identity, out, document_bytes, batch_documents, progress=None):
    from ._array import Array
    docs = documents(data, ranges, document_bytes)
    boundaries = {p: None for p in _split_points(ranges, len(data))}
    payload, at_token, above_255, max_id = bytearray(), 0, 0, -1
    encode_seconds, t_start = 0.0, time.perf_counter()
    for k in range(0, len(docs), batch_documents):
        group = docs[k:k + batch_documents]
        t0 = time.perf_counter()
        encoded = tok.encode_batch([bytes(data[a:b]) for a, b in group])
        encode_seconds += time.perf_counter() - t0
        for (a, _b), ids in zip(group, encoded):
            if a in boundaries:
                boundaries[a] = at_token
            arr = Array.from_list(ids, "<i4")
            payload.extend(arr.tobytes())
            at_token += arr.size
            if arr.size:
                above_255 += sum(value > 255 for value in ids)
                max_id = max(max_id, int(arr.max()))
        if progress and (k // batch_documents) % 8 == 0:
            done = group[-1][1]
            secs = time.perf_counter() - t_start
            progress(f"tokenized {done}/{len(data)} bytes, {at_token} ids, {secs:.1f} s, "
                     f"{done / max(secs, 1e-9) / 1e6:.3f} MB/s")
    boundaries[len(data)] = at_token
    seconds = time.perf_counter() - t_start
    payload = bytes(payload)
    token_ranges = {key: [boundaries[a], boundaries[b]] for key, (a, b) in ranges.items()}
    manifest = dict(
        schema=TOKENS_SCHEMA,
        source=src,
        vocabulary=identity,
        encoder="mojolearn.tokenizer.BpeTokenizer.encode_batch (host binding _mojolearn_tokenizer_host, "
                "tokenizer/encoding.mojo); integers and tables only",
        endoftext="not inserted and not recognized (allow_endoftext=False)",
        document_rule=f"no document spans a declared range boundary; at most {document_bytes} bytes, ending at "
                      "the last 0x0A at or before the limit (or at the limit when the window holds none); each "
                      "encoded alone, ids concatenated in document order",
        document_bytes=document_bytes, n_documents=len(docs),
        dtype="int32", byte_order="little",
        sha256=_sha_bytes(payload), bytes=len(payload), tokens=at_token,
        bytes_per_token=(len(data) / at_token) if at_token else None,
        max_id=max_id, ids_above_255=above_255,
        **{k: v for k, v in token_ranges.items()},
        schedule=_SCHEDULE,
        tokenize_seconds=seconds, encode_seconds=encode_seconds,
        encode_bytes_per_second=(len(data) / encode_seconds) if encode_seconds else None,
    )
    (out / "tokens.i32").write_bytes(payload)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


class TokenizedCorpus:
    """A prepared corpus: `tokens_dir` (the pinned id array and its
    manifest) and `vocabulary_path` (its canonical rank file)."""

    def __init__(self, tokens_dir, vocabulary_path):
        self.tokens_dir = Path(tokens_dir)
        self.vocabulary_path = Path(vocabulary_path)
        self.manifest = json.loads((self.tokens_dir / "manifest.json").read_text())
        if self.manifest.get("schema") != TOKENS_SCHEMA:
            raise ValueError(f"mojolearn: {self.tokens_dir}/manifest.json is not {TOKENS_SCHEMA}")

    @property
    def vocabulary(self):
        return dict(self.manifest["vocabulary"])

    @property
    def n_vocab(self):
        return int(self.manifest["vocabulary"]["n_vocab"])

    def tokenizer(self):
        return tokenizer_for(self.manifest, self.vocabulary_path)

    def batches(self, batch, length):
        return TokenBatches(self.tokens_dir, batch, length)

    def __repr__(self):
        m = self.manifest
        return (f"TokenizedCorpus(tokens={m['tokens']}, n_vocab={self.n_vocab}, "
                f"sha256={m['sha256'][:16]}, dir={str(self.tokens_dir)!r})")


def prepare(corpus, *, vocab=None, cache_dir=None, vocab_size=DEFAULT_VOCAB_SIZE,
            min_frequency=DEFAULT_MIN_FREQUENCY, vocab_sample_bytes=DEFAULT_VOCAB_SAMPLE_BYTES,
            document_bytes=DEFAULT_DOCUMENT_BYTES, batch_documents=DEFAULT_BATCH_DOCUMENTS, progress=None):
    """Tokenize `corpus` (a file path) once and return a `TokenizedCorpus`.

    `vocab=None` trains a `vocab_size`-rank vocabulary (n_vocab = vocab_size
    + 1 with `<|endoftext|>`) with `BpeVocabularyTrainer` on the first
    `vocab_sample_bytes` of the corpus's TRAIN range. Anything else is the
    caller's own token map (see the module docstring) and nothing is
    trained. Both are cached; a rerun with the same corpus and vocabulary
    reuses the id array after re-hashing it."""
    for name, value in (("vocab_size", vocab_size), ("min_frequency", min_frequency),
                        ("vocab_sample_bytes", vocab_sample_bytes), ("document_bytes", document_bytes),
                        ("batch_documents", batch_documents)):
        if type(value) is not int or value < 1:
            raise ValueError(f"mojolearn: {name} must be a positive int, got {value!r}")
    if isinstance(vocab, str) and vocab == "bytes":
        raise ValueError("mojolearn: byte training has no vocabulary to prepare; use the byte schedule "
                         "(tools/lm_train.py --bytes) instead of lm_corpus.prepare")
    cache = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
    data, src, ranges = _source(corpus)
    if vocab is None:
        vdir, rec = _trained_vocabulary(data, ranges, src["sha256"], cache, vocab_size, min_frequency,
                                        vocab_sample_bytes, document_bytes, progress)
    else:
        vdir, rec = _user_vocabulary(vocab, cache)
    identity = rec["identity"]
    ranks = vdir / "ranks.tsv"
    key = f"{src['sha256'][:16]}-{identity['sha256'][:16]}-d{document_bytes}"
    final = cache / "tokens" / key
    if (final / "manifest.json").is_file():
        out = TokenizedCorpus(final, ranks)
        m = out.manifest
        if m["source"]["sha256"] != src["sha256"] or m["vocabulary"]["sha256"] != identity["sha256"]:
            raise ValueError(f"mojolearn: cached tokens {final} were made from another corpus or vocabulary; "
                             "remove the directory to rebuild")
        got = _sha_file(final / "tokens.i32")
        if got != m["sha256"]:
            raise ValueError(f"mojolearn: cached tokens {final}/tokens.i32 hash {got}, its manifest says "
                             f"{m['sha256']}; remove the directory to rebuild")
        return out
    tok = _tokenizer.BpeTokenizer.from_ranks_file(str(ranks))
    if tok.identity["sha256"] != identity["sha256"]:
        raise ValueError(f"mojolearn: {ranks} loads as {tok.identity['sha256']}, not {identity['sha256']}")
    _atomic_dir(final, lambda tmp: _tokenize(data, src, ranges, tok, identity, tmp, document_bytes,
                                             batch_documents, progress))
    return TokenizedCorpus(final, ranks)


# ---------------------------------------------------------------- batches

class TokenBatches:
    """Token batches from a pinned id array, with `CorpusBatches`' interface
    (`ids(step_index)`, `describe()`), plus `data_schedule()` for the
    trainer.

    Step `k` (zero-based), row `b` reads ids
    `[lo + (k*batch*length + b*length) % (hi - lo - length - 1) : + length + 1]`
    where `[lo, hi)` is the TRAIN range in tokens, so a run of any length
    reads no validation or test id. (`CorpusBatches` takes the same form over
    the WHOLE byte file, test range included; it is kept that way so byte
    runs hash as they always have.)"""

    def __init__(self, tokens_dir, batch, length):
        import mmap
        from ._array import Array
        self.dir = Path(tokens_dir)
        raw = (self.dir / "manifest.json").read_bytes()
        self.manifest = json.loads(raw)
        if self.manifest.get("schema") != TOKENS_SCHEMA:
            raise ValueError(f"mojolearn: {self.dir}/manifest.json is not {TOKENS_SCHEMA}")
        self.manifest_sha256 = _sha_bytes(raw)
        path = self.dir / "tokens.i32"
        self.sha256 = _sha_file(path)
        if self.sha256 != self.manifest["sha256"] or path.stat().st_size != self.manifest["bytes"]:
            raise ValueError(f"mojolearn: pinned tokens length/SHA mismatch for {path}")
        if path.stat().st_size % 4:
            raise ValueError("mojolearn: token payload is not an int32 stream")
        if path.stat().st_size:
            with path.open("rb") as stream:
                mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            # The borrowed Array pins the memoryview and its mmap owner.
            self.ids_all = Array.from_buffer(memoryview(mapping).cast("i"))
        else:
            self.ids_all = Array((0,), "<i4")
        self.batch, self.length = int(batch), int(length)
        lo, hi = self.manifest.get("train_range") or [0, int(self.ids_all.size)]
        self.lo, self.hi = int(lo), int(hi)
        if self.hi - self.lo < self.length + 2:
            raise ValueError("mojolearn: the train range is shorter than one batch row")
        self.modulus = self.hi - self.lo - self.length - 1

    @property
    def vocabulary(self):
        return dict(self.manifest["vocabulary"])

    def ids(self, step_index):
        from ._array import Array
        width = self.length + 1
        out = Array((self.batch, width), "<i4")
        for b in range(self.batch):
            start = self.lo + (step_index * self.batch * self.length + b * self.length) % self.modulus
            out._mv[b * width:(b + 1) * width] = self.ids_all._mv[start:start + width]
        return out

    def prefetch(self, start_step, steps, *, depth=2):
        """Yield ``(step, ids)`` in order while preparing later batches.

        This is the bounded loading stage for pretokenized corpora staged from
        the dataset store: the producer reads only the already verified mmap,
        never R2 credentials or mutable remote state.  At most ``depth``
        batches are live.  Producer failures are re-raised at their exact
        logical step, before any later batch is yielded.
        """
        import queue
        import threading

        if type(start_step) is not int or start_step < 0:
            raise ValueError(f"mojolearn: start_step must be a nonnegative int, got {start_step!r}")
        if type(steps) is not int or steps < 0:
            raise ValueError(f"mojolearn: steps must be a nonnegative int, got {steps!r}")
        if type(depth) is not int or depth < 1:
            raise ValueError(f"mojolearn: prefetch depth must be a positive int, got {depth!r}")

        ready = queue.Queue(maxsize=depth)
        stopped = threading.Event()
        done = object()

        def put(item):
            while not stopped.is_set():
                try:
                    ready.put(item, timeout=0.05)
                    return True
                except queue.Full:
                    pass
            return False

        def produce():
            for step in range(start_step, start_step + steps):
                try:
                    batch = self.ids(step)
                except BaseException as exc:
                    put((step, None, exc))
                    return
                if not put((step, batch, None)):
                    return
            put(done)

        worker = threading.Thread(target=produce, name="mojolearn-token-prefetch", daemon=True)
        worker.start()
        try:
            while True:
                item = ready.get()
                if item is done:
                    return
                step, batch, error = item
                if error is not None:
                    raise error
                yield step, batch
        finally:
            stopped.set()
            worker.join()

    def describe(self):
        v = self.manifest["vocabulary"]
        return dict(path=str(self.dir / "tokens.i32"), sha256=self.sha256, manifest_sha256=self.manifest_sha256,
                    tokens=int(self.ids_all.size), train_range=[self.lo, self.hi],
                    corpus_sha256=self.manifest["source"]["sha256"],
                    vocabulary_sha256=v["sha256"], n_vocab=v["n_vocab"],
                    schedule='step k row b: ids[lo + (k*batch*length + b*length) % (hi - lo - length - 1) '
                             ': +length+1] over the TRAIN range [lo, hi) in tokens; targets shifted one id')

    def data_schedule(self, **extra):
        """The trainer's `data_schedule`: which ids, in which order, from
        which vocabulary. Kept in every checkpoint."""
        v = self.manifest["vocabulary"]
        out = dict(schema=TOKENS_SCHEMA, schedule=_SCHEDULE, tokens_sha256=self.sha256,
                   corpus_sha256=self.manifest["source"]["sha256"], batch=self.batch, length=self.length,
                   train_range=[self.lo, self.hi],
                   vocabulary=dict(schema=v["schema"], sha256=v["sha256"], n_vocab=v["n_vocab"]))
        out.update(extra)
        return out


# ---------------------------------------------------------------- identity

def _expected_vocabulary(model):
    """The vocabulary identity a model, checkpoint state, schedule or
    manifest carries, or None when it carries none."""
    if hasattr(model, "data_schedule") and not isinstance(model, dict):
        model = model.data_schedule
    if isinstance(model, dict):
        if "data_schedule" in model and isinstance(model["data_schedule"], dict):
            model = model["data_schedule"]
        v = model.get("vocabulary")
        if isinstance(v, dict) and "sha256" in v:
            return v
    return None


def require_vocabulary(model, tokenizer):
    """Refuse by name unless `tokenizer` is the vocabulary `model` was
    trained on. `model` is a trainer, a checkpoint state, a data schedule or
    a tokens manifest; `tokenizer` is a `BpeTokenizer`."""
    expected = _expected_vocabulary(model)
    if expected is None:
        raise ValueError("mojolearn: this model carries no vocabulary identity (a byte model, or trained "
                         "without lm_corpus); there is nothing to check a tokenizer against")
    got = tokenizer.identity
    if got["sha256"] != expected["sha256"] or got["n_vocab"] != expected["n_vocab"]:
        raise ValueError(
            f"mojolearn: vocabulary mismatch: the model was trained on vocabulary {expected['sha256']} "
            f"(n_vocab {expected['n_vocab']}) and this tokenizer is {got['sha256']} (n_vocab {got['n_vocab']}); "
            "its ids would decode to the wrong text")
    return tokenizer


def tokenizer_for(model, vocabulary):
    """A `BpeTokenizer` over `vocabulary` (any form `prepare(vocab=...)`
    takes), refused by name unless it is the one `model` was trained on."""
    if isinstance(vocabulary, _tokenizer.BpeTokenizer):
        tok = vocabulary
    elif isinstance(vocabulary, _tokenizer.TrainedBpeVocabulary):
        tok = vocabulary.tokenizer()
    elif isinstance(vocabulary, (tuple, list)) and len(vocabulary) == 2:
        tok = _tokenizer.BpeTokenizer.from_files(os.fspath(vocabulary[0]), os.fspath(vocabulary[1]))
    else:
        tok = _tokenizer.BpeTokenizer.from_ranks_file(os.fspath(vocabulary))
    return require_vocabulary(model, tok)
