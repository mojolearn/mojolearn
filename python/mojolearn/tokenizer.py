# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A byte-level BPE tokenizer over a vocabulary the caller supplies or
trains with `BpeVocabularyTrainer`.

MOJOLEARN SHIPS NO VOCABULARY (2026-09-15). `BpeTokenizer` is byte-level BPE
by merge rank, cut into pre-tokens by the GPT-2 pre-tokenization pattern (the
only pattern the compiled binding cuts; `mojolearn.models.tokenizer` cuts
Llama 3 and Qwen 2 in Python, see `tokenizer/README.md`), with
`<|endoftext|>` as the id after the last rank. The vocabulary is yours:

    tok = BpeTokenizer.from_files("encoder.json", "vocab.bpe")
    tok = BpeTokenizer.from_ranks_file("ranks.tsv")      # rank<TAB>hex lines
    tok = BpeTokenizer.from_token_bytes(list_of_bytes)   # rank = list index
    tok = BpeVocabularyTrainer(vocab_size=32000).train(docs).tokenizer()

THE NAME (2026-09-18, lane/tokenized-corpus). This class was `GPT2Tokenizer`
until 0.8.7. It ships no GPT-2 vocabulary and `TrainedBpeVocabulary.
tokenizer()` returns one over OUR OWN trained table, so the old name described
a model it does not carry. `GPT2Tokenizer` stays importable as a deprecated
alias of this class (same object, a DeprecationWarning on access). The GPT-2
name is kept where it names a GPT-2 thing: the pre-tokenization pattern and
the `encoder.json` + `vocab.bpe` file format `from_files` reads.

OpenAI publishes the GPT-2 `encoder.json` and `vocab.bpe` with its GPT-2
release, for example under
https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/ ; download
them yourself and pass their paths. With those two files the ids are the
GPT-2 ids. `BpeTokenizer()` with no vocabulary is refused by name.

The arithmetic is `tokenizer/encoding.mojo` (integers and tables only, no
float, no device kernel), reached through the host binding
`_mojolearn_tokenizer_host` under `mojolearn/host/`. The Unicode letter,
number and White_Space classes the pattern needs are compiled into that
binding at build time (`tokenizer/tools/gen_unicode_categories.py`), so no
data file is read at run time except the vocabulary.

What is asserted, and where: `pixi run check-tokenizer` holds the Mojo encoder
to a synthetic vocabulary mojolearn trains itself
(`mojolearn/_tokenizer_synthetic.py`) and to a second, pure Python encoder of
the same algorithm; `python/mojolearn/tests/test_tokenizer_surface.py` holds
THIS door to the same cases through the binding, plus every refusal by name.
"""
import array
import json
import os
import tempfile

from . import _backend
from . import _bpe_trainer
from . import _tokenizer_synthetic
from ._buffer import addr, addr_ro

_EXTENSION = "_mojolearn_tokenizer_host"

__all__ = ["BpeTokenizer", "BpeVocabularyTrainer", "TrainedBpeVocabulary"]

_NO_VOCABULARY = (
    "mojolearn: BpeTokenizer needs a vocabulary, and mojolearn ships none. "
    "Load one you obtained yourself: BpeTokenizer.from_files(encoder_json, vocab_bpe) "
    "with the encoder.json and vocab.bpe that OpenAI publishes with its GPT-2 release, "
    "BpeTokenizer.from_ranks_file(path) with a rank<TAB>hex file, or "
    "BpeTokenizer.from_token_bytes(tokens)"
)


#: The binding's entries were `gpt2_*` until 2026-09-18 and are `bpe_*`
#: since. A binding built before the rename (an older host directory under
#: MOJOLEARN_HOST_DIR) is read through its old names so it keeps loading;
#: the ids are the same arithmetic under either name.
_ABI = ("load", "n_vocab", "max_token_bytes", "encode", "encode_batch", "decode")


class _Entries:
    """The binding module with its tokenizer entries under their `bpe_*`
    names; every other attribute is the module's own."""

    def __init__(self, module):
        self._module = module
        for name in _ABI:
            f = getattr(module, "bpe_" + name, None) or getattr(module, "gpt2_" + name, None)
            if f is None:
                raise ImportError(f"mojolearn: the tokenizer binding exports neither bpe_{name} nor gpt2_{name}")
            setattr(self, "bpe_" + name, f)

    def __getattr__(self, name):
        return getattr(self._module, name)


VOCABULARY_SCHEMA = "mojolearn.bpe-vocabulary.v1"


def _identity_from_tokens_text(text, n_ranks):
    import hashlib
    return dict(schema=VOCABULARY_SCHEMA, sha256=hashlib.sha256(text.encode("ascii")).hexdigest(),
                n_ranks=n_ranks, n_vocab=n_ranks + 1, endoftext_id=n_ranks)


def _identity_of_ranks_file(path):
    """`_identity_from_tokens_text` over the canonical rendering of a rank
    file. The binding is what refuses a malformed file; this reads only what
    it needs to canonicalize and leaves a line it cannot read to the loader."""
    with open(path, "rb") as fh:
        raw = fh.read()
    lines = raw.decode("ascii", errors="replace").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    out = []
    for k, line in enumerate(lines):
        _rank, _, hx = line.strip().partition("\t")
        try:
            out.append(f"{k}\t{bytes.fromhex(hx).hex()}\n")
        except ValueError:
            out.append(f"{k}\t{hx}\n")
    return _identity_from_tokens_text("".join(out), len(lines))


def _binding():
    return _Entries(_backend.load_host_module(_EXTENSION))


def _byte_to_char():
    """The GPT-2 format's byte-to-unicode spelling (`tokenizer/impl/
    byte_unicode.mojo` builds the same table): the printable Latin-1 runs
    are fixed points and the other 68 bytes take U+0100 upward in byte
    order."""
    fixed = set(range(0x21, 0x7F)) | set(range(0xA1, 0xAD)) | set(range(0xAE, 0x100))
    out, extra = {}, 256
    for b in range(256):
        if b in fixed:
            out[b] = chr(b)
        else:
            out[b] = chr(extra)
            extra += 1
    return out


class BpeTokenizer:
    """`encode` and `decode` for byte-level BPE in the GPT-2 format.

    Ids are in [0, n_vocab): the vocabulary's ranks, then `<|endoftext|>` at
    `n_vocab - 1`. The special token is recognized only when
    `allow_endoftext=True`; otherwise its thirteen characters are ordinary
    text. There is no "raise on a disallowed special token" mode and no
    other special token (`tokenizer/NOT_IMPLEMENTED.tsv`).

    Bytes are the interface: `encode_bytes` and `decode_bytes` are the real
    entries. `encode` accepts `str` (UTF-8 encoded first) or a bytes-like
    object; `decode` is `decode_bytes` decoded as UTF-8 with
    `errors="replace"`, because a token stream cut inside a character is
    still the right bytes. Invalid UTF-8 input encodes as one-byte
    pre-tokens and round trips.
    """

    ENDOFTEXT = "<|endoftext|>"

    def __init__(self, ranks_file=None):
        if ranks_file is None:
            raise ValueError(_NO_VOCABULARY)
        path = os.path.abspath(os.fspath(ranks_file))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"mojolearn: BpeTokenizer rank file {path} does not exist")
        self._source = path
        self._identity = _identity_of_ranks_file(path)
        self._m = _binding()
        self._handle = self._m.bpe_load(path)
        self._n_vocab = int(self._m.bpe_n_vocab(self._handle))
        self._max_token_bytes = int(self._m.bpe_max_token_bytes(self._handle))

    # ------------------------------------------------------------ loading

    @classmethod
    def from_ranks_file(cls, path):
        """A rank file: one `rank<TAB>hex_of_token_bytes` line per rank,
        ascending from 0, every one of the 256 single bytes present."""
        return cls(path)

    @classmethod
    def from_token_bytes(cls, tokens):
        """A vocabulary as a sequence of bytes-like tokens; a token's rank
        (and id) is its index. Unique, non-empty, and holding all 256 single
        bytes, or refused by name before anything is loaded."""
        if isinstance(tokens, (str, bytes, bytearray, memoryview)):
            raise TypeError(f"mojolearn: tokens must be a sequence of bytes, got {type(tokens).__name__}")
        toks = []
        for k, t in enumerate(tokens):
            if not isinstance(t, (bytes, bytearray, memoryview)):
                raise TypeError(f"mojolearn: token {k} must be bytes-like, got {type(t).__name__}")
            t = bytes(t)
            if not t:
                raise ValueError(f"mojolearn: token {k} is empty")
            toks.append(t)
        seen = {}
        for k, t in enumerate(toks):
            if t in seen:
                raise ValueError(f"mojolearn: token {k} repeats token {seen[t]} ({t.hex()})")
            seen[t] = k
        missing = [b for b in range(256) if bytes([b]) not in seen]
        if missing:
            raise ValueError(
                f"mojolearn: the vocabulary lacks {len(missing)} of the 256 single-byte tokens "
                f"(first 0x{missing[0]:02x}); byte-level BPE needs every byte")
        fd, path = tempfile.mkstemp(prefix="mojolearn-ranks-", suffix=".tsv")
        try:
            with os.fdopen(fd, "w", encoding="ascii") as fh:
                for k, t in enumerate(toks):
                    fh.write(f"{k}\t{t.hex()}\n")
            tok = cls(path)
        finally:
            os.unlink(path)
        tok._source = "<token bytes>"
        return tok

    @classmethod
    def from_files(cls, encoder_json, vocab_bpe):
        """The two files of the GPT-2 format: `encoder.json` (token spelling
        to id) and `vocab.bpe` (the merge list). mojolearn ships neither;
        OpenAI publishes GPT-2's with its GPT-2 release.

        Checked before loading, each refused by name: every spelling decodes
        through the byte-to-unicode table; ids are 0 to n-1 with
        `<|endoftext|>` (if present) at n; every merge's two parts and result
        are tokens; merge results take strictly increasing ids above both
        parts; and every token no merge makes is a single byte. Those are
        the conditions under which merging by rank gives the merge list's
        own result."""
        byte_of = {c: b for b, c in _byte_to_char().items()}
        with open(encoder_json, "r", encoding="utf-8") as fh:
            encoder = json.load(fh)
        if not isinstance(encoder, dict):
            raise ValueError(f"mojolearn: {encoder_json} is not a JSON object of spelling to id")
        eot_id = encoder.pop(cls.ENDOFTEXT, None)
        n = len(encoder)
        tokens = [None] * n
        for spelling, i in encoder.items():
            if type(i) is not int or not 0 <= i < n or tokens[i] is not None:
                raise ValueError(f"mojolearn: {encoder_json}: id {i!r} of {spelling!r} is not a unique id in [0, {n})")
            try:
                tokens[i] = bytes(byte_of[c] for c in spelling)
            except KeyError as exc:
                raise ValueError(
                    f"mojolearn: {encoder_json}: {spelling!r} holds {exc.args[0]!r}, which is not a "
                    "byte-level spelling") from None
        if eot_id is not None and eot_id != n:
            raise ValueError(f"mojolearn: {encoder_json}: {cls.ENDOFTEXT} is id {eot_id}, not {n} (after the ranks)")
        index = {t: i for i, t in enumerate(tokens)}
        spelled = {c: b for c, b in zip(encoder.keys(), (tokens[i] for i in encoder.values()))}
        made = set()
        last = -1
        with open(vocab_bpe, "r", encoding="utf-8") as fh:
            lines = fh.read().split("\n")
        for lineno, line in enumerate(lines, 1):
            if not line.strip() or (lineno == 1 and line.startswith("#version")):
                continue
            parts = line.split(" ")
            if len(parts) != 2 or parts[0] not in spelled or parts[1] not in spelled:
                raise ValueError(f"mojolearn: {vocab_bpe}:{lineno}: {line!r} is not a merge of two tokens")
            a, b = spelled[parts[0]], spelled[parts[1]]
            m = index.get(a + b)
            if m is None:
                raise ValueError(f"mojolearn: {vocab_bpe}:{lineno}: the merge result {parts[0] + parts[1]!r} is not a token")
            if m <= last or m <= index[a] or m <= index[b]:
                raise ValueError(
                    f"mojolearn: {vocab_bpe}:{lineno}: merge result id {m} does not rise above the previous "
                    f"merge ({last}) and both parts; merging by rank would not follow this merge list")
            last = m
            made.add(m)
        loose = [i for i in range(n) if i not in made and len(tokens[i]) != 1]
        if loose:
            raise ValueError(f"mojolearn: {encoder_json}: id {loose[0]} is neither a single byte nor made by a merge")
        tok = cls.from_token_bytes(tokens)
        tok._source = f"{os.path.abspath(encoder_json)} + {os.path.abspath(vocab_bpe)}"
        return tok

    @classmethod
    def _synthetic(cls):
        """The synthetic vocabulary mojolearn trains itself
        (`_tokenizer_synthetic.py`), for the gates and the identity lane."""
        tok = cls.from_token_bytes(_tokenizer_synthetic.vocabulary())
        tok._source = "<mojolearn synthetic vocabulary>"
        return tok

    @property
    def n_vocab(self):
        """The ranks plus `<|endoftext|>`."""
        return self._n_vocab

    @property
    def eot_token(self):
        """`<|endoftext|>`'s id, `n_vocab - 1`."""
        return self._n_vocab - 1

    @property
    def vocabulary_source(self):
        return self._source

    @property
    def identity(self):
        """What names this vocabulary, whatever file it came from:
        `dict(schema, sha256, n_ranks, n_vocab, endoftext_id)`, `sha256` being
        over the CANONICAL rank file (`rank<TAB>lowercase hex` per line, the
        text `TrainedBpeVocabulary.render_ranks` writes). The same table read
        from a rank file, from `encoder.json` + `vocab.bpe` or from a trained
        vocabulary has the same identity. A model trained on ids carries this
        (`mojolearn.lm_corpus`), so its ids are never decoded with another
        table."""
        return dict(self._identity)

    # ------------------------------------------------------------ encode

    @staticmethod
    def _text_bytes(text):
        if isinstance(text, str):
            return text.encode("utf-8")
        if isinstance(text, (bytes, bytearray, memoryview)):
            return bytes(text)
        raise TypeError(
            f"mojolearn: text must be str or bytes-like, got {type(text).__name__}"
        )

    @staticmethod
    def _flag(allow_endoftext):
        if type(allow_endoftext) is not bool:
            raise TypeError(
                f"mojolearn: allow_endoftext must be a bool, got {type(allow_endoftext).__name__}"
            )
        return allow_endoftext

    def encode_bytes(self, data, allow_endoftext=False):
        """The ids of a byte string, as a list of int. `data` is bytes-like."""
        if isinstance(data, str):
            raise TypeError("mojolearn: encode_bytes takes bytes; use encode for str")
        raw = self._text_bytes(data)
        allow = self._flag(allow_endoftext)
        n = len(raw)
        if n == 0:
            return []
        out = array.array("i", bytes(4 * n))
        count = int(self._m.bpe_encode(
            self._handle, addr_ro(raw, name="text"), n, addr(out, name="ids"), n, allow))
        if not 0 <= count <= n:
            raise RuntimeError(f"mojolearn: bpe_encode returned {count} ids for {n} bytes")
        return out[:count].tolist()

    def encode(self, text, allow_endoftext=False):
        """The ids of `text` (str, UTF-8 encoded first, or bytes-like)."""
        return self.encode_bytes(self._text_bytes(text), allow_endoftext)

    def encode_batch(self, documents, allow_endoftext=False):
        """The ids of each document, as a list of lists of int, in order.

        Each document is a str (UTF-8 encoded first) or bytes-like and is
        encoded ALONE: `encode_batch(docs)[k] == encode(docs[k])` id for id,
        whatever else is in the batch. The documents are encoded one after
        another inside ONE binding call (`bpe_encode_batch`), which saves
        the per-call crossing that dominates short documents."""
        if isinstance(documents, (str, bytes, bytearray, memoryview)):
            raise TypeError(
                f"mojolearn: encode_batch takes a sequence of documents, got {type(documents).__name__}"
            )
        try:
            docs = list(documents)
        except TypeError:
            raise TypeError(
                f"mojolearn: encode_batch takes a sequence of documents, got {type(documents).__name__}"
            ) from None
        allow = self._flag(allow_endoftext)
        raws = []
        for k, d in enumerate(docs):
            if not isinstance(d, (str, bytes, bytearray, memoryview)):
                raise TypeError(
                    f"mojolearn: document {k} must be str or bytes-like, got {type(d).__name__}"
                )
            raws.append(self._text_bytes(d))
        n_docs = len(raws)
        if n_docs == 0:
            return []
        text = b"".join(raws)
        n = len(text)
        offsets = array.array("q", bytes(8 * (n_docs + 1)))
        pos = 0
        for k, r in enumerate(raws):
            pos += len(r)
            offsets[k + 1] = pos
        counts = array.array("q", bytes(8 * n_docs))
        ids = array.array("i", bytes(4 * max(n, 1)))
        text_buf = text if n > 0 else b"\0"
        total = int(self._m.bpe_encode_batch(
            self._handle, addr_ro(text_buf, name="text"), addr_ro(offsets, name="offsets"),
            addr(ids, name="ids"), addr(counts, name="counts"), [n_docs, n, n, allow]))
        if not 0 <= total <= n or sum(counts) != total:
            raise RuntimeError(
                f"mojolearn: bpe_encode_batch returned {total} ids for {n} bytes (counts sum {sum(counts)})"
            )
        out, at = [], 0
        for c in counts:
            out.append(ids[at:at + c].tolist())
            at += c
        return out

    # ------------------------------------------------------------ decode

    def _ids_array(self, ids):
        if isinstance(ids, (str, bytes, bytearray)):
            raise TypeError(f"mojolearn: ids must be a sequence of int, got {type(ids).__name__}")
        try:
            seq = list(ids)
        except TypeError:
            raise TypeError(f"mojolearn: ids must be a sequence of int, got {type(ids).__name__}") from None
        out = array.array("i", bytes(4 * max(len(seq), 1)))
        for k, v in enumerate(seq):
            if isinstance(v, bool):
                raise TypeError(f"mojolearn: ids must be int, not bool, at position {k}")
            try:
                iv = v.__index__()
            except AttributeError:
                raise TypeError(
                    f"mojolearn: ids must be int, got {type(v).__name__} at position {k}"
                ) from None
            if not 0 <= iv < self._n_vocab:
                raise ValueError(
                    f"mojolearn: id {iv} at position {k} is outside [0, {self._n_vocab})"
                )
            out[k] = iv
        return out, len(seq)

    def decode_bytes(self, ids):
        """The bytes of an id sequence. An id outside [0, n_vocab) is refused
        by value and position before the binding is called."""
        arr, n = self._ids_array(ids)
        if n == 0:
            return b""
        cap = n * self._max_token_bytes
        out = bytearray(cap)
        count = int(self._m.bpe_decode(
            self._handle, addr_ro(arr, name="ids"), n, addr(out, name="text"), cap))
        if not 0 <= count <= cap:
            raise RuntimeError(f"mojolearn: bpe_decode returned {count} bytes into {cap}")
        return bytes(out[:count])

    def decode(self, ids, errors="replace"):
        """`decode_bytes(ids)` as text; `errors` is passed to `bytes.decode`."""
        return self.decode_bytes(ids).decode("utf-8", errors)

    def decode_bytes_batch(self, batch):
        """`[decode_bytes(ids) for ids in batch]`. A thin loop over the one
        `bpe_decode` call per sequence: decode is a table copy, and a
        sequence of ids already carries its own boundaries."""
        if isinstance(batch, (str, bytes, bytearray)):
            raise TypeError(f"mojolearn: decode_bytes_batch takes a sequence of id sequences, got {type(batch).__name__}")
        return [self.decode_bytes(ids) for ids in batch]

    def decode_batch(self, batch, errors="replace"):
        """`[decode(ids, errors) for ids in batch]`."""
        return [b.decode("utf-8", errors) for b in self.decode_bytes_batch(batch)]

    def __repr__(self):
        return f"BpeTokenizer(n_vocab={self._n_vocab}, vocabulary={self._source!r})"


class TrainedBpeVocabulary:
    """What `BpeVocabularyTrainer.train` returns: the vocabulary, the merges,
    and the two formats it can be written in.

    `tokens` are the token byte strings in rank order (rank = id, the 256
    single bytes first). `merges` are `(left_id, right_id, new_id)` in the
    order they were made. `n_ties_broken` is how many selections had two or
    more pairs at the top count, which is how you can see the tie-break rule
    was actually REACHED on your corpus rather than merely present.
    """

    def __init__(self, tokens, merges, stats):
        self.tokens = tokens
        self.merges = merges
        self.stats = dict(stats)

    @property
    def n_tokens(self):
        """The ranks. `<|endoftext|>` takes the id after the last one."""
        return len(self.tokens)

    @property
    def n_ties_broken(self):
        return self.stats["n_ties_broken"]

    @property
    def tie_break(self):
        """The total order that settles equal counts, spelled out."""
        return self.stats["tie_break"]

    def render_ranks(self):
        """OUR format as text: `rank<TAB>hex_of_token_bytes` per line."""
        return _bpe_trainer.render_ranks(self.tokens)

    def render_tokenizer_json(self):
        """A `tokenizer.json` Hugging Face `tokenizers` loads, as text."""
        return _bpe_trainer.render_tokenizer_json(self.tokens, self.merges)

    def write_ranks(self, path):
        _bpe_trainer.write_ranks(self.tokens, path)
        return path

    def write_tokenizer_json(self, path):
        _bpe_trainer.write_tokenizer_json(self.tokens, self.merges, path)
        return path

    @property
    def identity(self):
        """`BpeTokenizer.identity` of this vocabulary, without loading it."""
        return _identity_from_tokens_text(self.render_ranks(), self.n_tokens)

    def tokenizer(self):
        """A `BpeTokenizer` over this vocabulary, so a freshly trained table
        can be used without going through a file."""
        return BpeTokenizer.from_token_bytes(self.tokens)

    def __repr__(self):
        return (f"TrainedBpeVocabulary(n_tokens={self.n_tokens}, "
                f"n_merges={len(self.merges)}, n_ties_broken={self.n_ties_broken})")


class BpeVocabularyTrainer:
    """Train a byte-level BPE vocabulary, deterministically.

        tok = BpeVocabularyTrainer(vocab_size=32000).train(documents).tokenizer()

    mojolearn ships no vocabulary and no corpus; `documents` is yours, as a
    sequence of `bytes` (or `str`, encoded UTF-8 first). Each document is
    pre-tokenized ALONE, so no pre-token spans a document join and the ORDER
    the documents arrive in cannot reach the result.

    WHAT IS BEING CLAIMED. Vocabulary training is host-only everywhere --
    Hugging Face, SentencePiece and tiktoken all train on a CPU -- so this is
    not a cross-vendor GPU claim and there is no vendor column. The claim is
    that THE SAME CORPUS AND CONFIG PRODUCE THE SAME VOCABULARY BYTES ON ANY
    MACHINE AND ARCHITECTURE, and it rests on four things: a total order on
    the tie-break (highest count, then smallest `(left_id, right_id)`),
    single-threaded counting so there is no reduction order to get wrong,
    selection that never depends on an iteration order, and no float anywhere
    in the selection.

    THE BACKEND (lane/bpe-builder-native, 2026-09-18). `backend="auto"`
    (the default) trains with the Mojo trainer `tokenizer/train/
    bpe_train.mojo` through the tokenizer host binding (`bpe_train`), and
    falls back to the pure Python reference `_bpe_trainer.train` when the
    binding is not built or predates that entry. `"mojo"` requires the
    binding; `"python"` runs the reference. The two are held to the SAME
    BYTES by `pixi run check-bpe-trainer` (file byte for file byte, both
    formats) and by the identity lanes, so which one ran cannot reach the
    vocabulary; `stats["backend"]` records it. The Python reference recounts
    every pair per merge in pure Python and is impractical at tens of
    thousands of ranks; the Mojo one trained 50,256 ranks on 20 MB on one
    core.
    `MOJOLEARN_BPE_TRAINER_SABOTAGE=1` reverses the tie-break on EITHER
    backend (the negative control).
    """

    _BACKENDS = ("auto", "mojo", "python")

    def __init__(self, vocab_size=32000, min_frequency=2, backend="auto"):
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
            raise TypeError(f"mojolearn: vocab_size must be an int, got {type(vocab_size).__name__}")
        if vocab_size < 256:
            raise ValueError(
                f"mojolearn: vocab_size {vocab_size} is below the 256 single-byte tokens; "
                "byte-level BPE needs every byte")
        if not isinstance(min_frequency, int) or isinstance(min_frequency, bool):
            raise TypeError(f"mojolearn: min_frequency must be an int, got {type(min_frequency).__name__}")
        if min_frequency < 1:
            raise ValueError(f"mojolearn: min_frequency {min_frequency} must be at least 1")
        if backend not in self._BACKENDS:
            raise ValueError(f"mojolearn: backend must be one of {self._BACKENDS}, got {backend!r}")
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.backend = backend

    def train(self, documents):
        """Train on `documents`, a sequence of bytes-like or str."""
        if isinstance(documents, (str, bytes, bytearray, memoryview)):
            raise TypeError(
                f"mojolearn: train takes a sequence of documents, got {type(documents).__name__}; "
                "wrap a single document in a list")
        try:
            docs = list(documents)
        except TypeError:
            raise TypeError(
                f"mojolearn: train takes a sequence of documents, got {type(documents).__name__}"
            ) from None
        if not docs:
            raise ValueError("mojolearn: train needs at least one document")
        raws = []
        for k, d in enumerate(docs):
            if isinstance(d, str):
                raws.append(d.encode("utf-8"))
            elif isinstance(d, (bytes, bytearray, memoryview)):
                raws.append(bytes(d))
            else:
                raise TypeError(
                    f"mojolearn: document {k} must be str or bytes-like, got {type(d).__name__}")
        native = None
        if self.backend != "python":
            native = _native_trainer(required=self.backend == "mojo")
        if native is not None:
            tokens, merges, stats = _train_native(native, raws, self.vocab_size, self.min_frequency,
                                                  _bpe_trainer.sabotaged())
            stats["backend"] = "mojo"
        else:
            tokens, merges, stats = _bpe_trainer.train(raws, self.vocab_size, self.min_frequency)
            stats["backend"] = "python"
        return TrainedBpeVocabulary(tokens, merges, stats)

    def __repr__(self):
        return (f"BpeVocabularyTrainer(vocab_size={self.vocab_size}, "
                f"min_frequency={self.min_frequency}, backend={self.backend!r})")


def _native_trainer(required):
    """The binding when it exports `bpe_train`, else None (or the reason,
    raised, when `required`)."""
    try:
        module = _backend.load_host_module(_EXTENSION)
    except ImportError as exc:
        if required:
            raise ImportError(f"mojolearn: backend='mojo' needs the tokenizer host binding: {exc}") from exc
        return None
    if not all(hasattr(module, n) for n in ("bpe_train", "bpe_trained_sizes", "bpe_trained_copy")):
        if required:
            raise ImportError("mojolearn: backend='mojo': this tokenizer binding predates bpe_train; rebuild it")
        return None
    return module


def _train_native(module, raws, vocab_size, min_frequency, break_ties_high):
    """`_bpe_trainer.train`'s return shape, `(tokens, merges, stats)`, from
    the Mojo trainer: one crossing in (the documents back to back with
    int64 offsets), one out (the token bytes, lengths and merge ids)."""
    offsets = array.array("q", [0])
    for r in raws:
        offsets.append(offsets[-1] + len(r))
    text = bytearray(b"".join(raws))
    n = len(text)
    handle = module.bpe_train(addr_ro(text, name="documents") if n else 0, addr_ro(offsets, name="offsets"),
                              [len(raws), n, int(vocab_size), int(min_frequency), bool(break_ties_high)])
    n_tokens, arena_bytes, n_merges, n_ties, n_groups = (int(x) for x in module.bpe_trained_sizes(handle))
    arena = bytearray(max(arena_bytes, 1))
    lengths = array.array("q", [0]) * n_tokens
    left = array.array("q", [0]) * max(n_merges, 1)
    right = array.array("q", [0]) * max(n_merges, 1)
    module.bpe_trained_copy(handle, addr(arena, name="arena"), addr(lengths, name="lengths"),
                            addr(left, name="merge_left"), addr(right, name="merge_right"))
    tokens, at = [], 0
    for m in lengths:
        tokens.append(bytes(arena[at:at + m]))
        at += m
    merges = [(int(left[k]), int(right[k]), 256 + k) for k in range(n_merges)]
    stats = {
        "n_tokens": n_tokens,
        "n_merges": n_merges,
        "n_groups": n_groups,
        "n_ties_broken": n_ties,
        "tie_break": _bpe_trainer.TIE_BREAK,
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
    }
    return tokens, merges, stats


#: Renamed classes still importable under their old name, {old: new}.
#: tools/verification_matrix.py reads this literal as an alias, so the old
#: name is covered by the new one's lanes instead of reading as a laneless
#: algorithm.
_DEPRECATED_ALIASES = {"GPT2Tokenizer": "BpeTokenizer"}


def __getattr__(name):
    if name in _DEPRECATED_ALIASES:
        import warnings
        new = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"mojolearn.tokenizer.{name} is renamed {new} (it ships no GPT-2 vocabulary); "
            "the old name is a deprecated alias of the same class",
            DeprecationWarning, stacklevel=2)
        return globals()[new]
    raise AttributeError(f"module 'mojolearn.tokenizer' has no attribute {name!r}")
