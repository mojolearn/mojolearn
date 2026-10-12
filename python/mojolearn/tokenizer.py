# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Byte-level BPE tokenization with a vocabulary you supply or train.

mojolearn ships no vocabulary. :class:`BpeTokenizer` encodes and decodes
byte-level BPE by merge rank, splitting text into pre-tokens with the GPT-2
pre-tokenization pattern, with ``<|endoftext|>`` as the id after the last
rank. Load a vocabulary you have, or train one with
:class:`BpeVocabularyTrainer`::

    tok = BpeTokenizer.from_files("encoder.json", "vocab.bpe")
    tok = BpeTokenizer.from_ranks_file("ranks.tsv")      # rank<TAB>hex lines
    tok = BpeTokenizer.from_token_bytes(list_of_bytes)   # rank = list index
    tok = BpeVocabularyTrainer(vocab_size=32000).train(docs).tokenizer()

With the ``encoder.json`` and ``vocab.bpe`` files OpenAI publishes with
GPT-2 (download them yourself), the ids are GPT-2's. Tokenization runs in
compiled code on the CPU and reads no data file other than the vocabulary.
For the Llama 3 and Qwen 2 pre-tokenization patterns, see
:class:`mojolearn.models.Tokenizer`.

``GPT2Tokenizer`` is a deprecated alias of :class:`BpeTokenizer` (accessing it
issues a DeprecationWarning).
"""
# Maintainer notes moved from the docstring:
# - The class was `GPT2Tokenizer` until 0.8.7 (renamed 2026-09-18,
#   lane/tokenized-corpus): it ships no GPT-2 vocabulary. The GPT-2 name is kept
#   where it names a GPT-2 thing: the pre-tokenization pattern and the
#   `encoder.json` + `vocab.bpe` format `from_files` reads.
# - The GPT-2 pattern is the only one the compiled binding cuts;
#   `mojolearn.models.tokenizer` cuts Llama 3 and Qwen 2 in Python (see
#   `tokenizer/README.md`). GPT-2's files are published under
#   https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/
# - The arithmetic is `tokenizer/encoding.mojo` (integers and tables only),
#   reached through the host binding `_mojolearn_tokenizer_host` under
#   `mojolearn/host/`. The Unicode letter, number and White_Space classes are
#   compiled into that binding at build time
#   (`tokenizer/tools/gen_unicode_categories.py`).
# - `pixi run check-tokenizer` holds the Mojo encoder to a synthetic vocabulary
#   (`mojolearn/_tokenizer_synthetic.py`) and to a pure Python encoder of the
#   same algorithm; `python/mojolearn/tests/test_tokenizer_surface.py` holds
#   this door to the same cases plus every refusal by name.
import array
import json
import operator
import os

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
        for name in _ABI:  # glue: looks up the binding ABI entries
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
    file. The binding loads (and refuses) the file and renders the canonical
    text (`tokenizer/impl/vocab_build.mojo::ranks_text`)."""
    m = _binding()
    return _identity_of_handle(m, _native(m.bpe_load, os.fspath(path)))


def _identity_of_handle(m, handle):
    return _identity_from_tokens_text(m.bpe_ranks_text(handle), int(m.bpe_n_vocab(handle)) - 1)


def _native(fn, *args, prefix="mojolearn: "):
    """Call a binding entry; a refusal it raises as `ValueError: ...` or
    `TypeError: ...` (the Mojo side names the Python type) is re-raised as
    that type with `prefix`."""
    try:
        return fn(*args)
    except Exception as exc:
        msg = str(exc)
        for name, kind in (("ValueError: ", ValueError), ("TypeError: ", TypeError)):  # glue: maps two native error prefixes
            if msg.startswith(name):
                raise kind(prefix + msg[len(name):]) from None
        raise


def _blob(parts, what):
    """`(bytes, int64 lengths)` of a sequence of bytes-like parts, joined
    by the C `bytes.join` (no Python loop); a part that is not bytes-like
    is refused."""
    try:
        blob = b"".join(parts)
    except TypeError as exc:
        raise TypeError(f"mojolearn: {what} must be bytes-like ({exc})") from None
    return blob, array.array("q", map(len, parts))


def _ro(buf, name):
    return addr_ro(buf, name=name) if len(buf) else 0


def _binding():
    return _Entries(_backend.load_host_module(_EXTENSION))


def _byte_to_char():
    """The GPT-2 format's byte-to-unicode spelling (`tokenizer/impl/
    byte_unicode.mojo` builds the same table): the printable Latin-1 runs
    are fixed points and the other 68 bytes take U+0100 upward in byte
    order."""
    fixed = set(range(0x21, 0x7F)) | set(range(0xA1, 0xAD)) | set(range(0xAE, 0x100))
    out, extra = {}, 256
    for b in range(256):  # cpu-route: byte to char table of the text encoding
        if b in fixed:
            out[b] = chr(b)
        else:
            out[b] = chr(extra)
            extra += 1
    return out


# `encode` and `decode` for byte-level BPE in the GPT-2 format.
#
# Ids are in [0, n_vocab): the vocabulary's ranks, then `<|endoftext|>` at
# `n_vocab - 1`. The special token is recognized only when
# `allow_endoftext=True`; otherwise its thirteen characters are ordinary
# text. There is no "raise on a disallowed special token" mode and no
# other special token (`tokenizer/NOT_IMPLEMENTED.tsv`).
#
# Bytes are the interface: `encode_bytes` and `decode_bytes` are the real
# entries. `encode` accepts `str` (UTF-8 encoded first) or a bytes-like
# object; `decode` is `decode_bytes` decoded as UTF-8 with
# `errors="replace"`, because a token stream cut inside a character is
# still the right bytes. Invalid UTF-8 input encodes as one-byte
# pre-tokens and round trips.
class BpeTokenizer:
    """Byte-level BPE tokenizer in the GPT-2 format, over your vocabulary.

    Ids are in ``[0, n_vocab)``: the vocabulary's ranks, then
    ``<|endoftext|>`` at ``n_vocab - 1``. The special token is recognized only
    with ``allow_endoftext=True``; otherwise its characters are ordinary text.
    There are no other special tokens.

    Bytes are the interface: :meth:`encode_bytes` and :meth:`decode_bytes` do
    the work. :meth:`encode` also accepts ``str`` (encoded as UTF-8 first);
    :meth:`decode` decodes the bytes as UTF-8 with ``errors="replace"``, since a
    token sequence may end inside a character. Invalid UTF-8 input round trips.

    ``GPT2Tokenizer`` is a deprecated alias of this class.

    Parameters
    ----------
    ranks_file : str or path-like
        A rank file (see :meth:`from_ranks_file`). Required: there is no
        built-in vocabulary, so omitting it raises ValueError.

    Raises
    ------
    FileNotFoundError
        If ``ranks_file`` does not exist.

    Examples
    --------
    >>> from mojolearn.tokenizer import BpeTokenizer
    >>> tok = BpeTokenizer.from_files("encoder.json", "vocab.bpe")
    >>> ids = tok.encode("Hello, world")
    >>> tok.decode(ids)
    'Hello, world'
    """

    ENDOFTEXT = "<|endoftext|>"

    def __init__(self, ranks_file=None):
        if ranks_file is None:
            raise ValueError(_NO_VOCABULARY)
        path = os.path.abspath(os.fspath(ranks_file))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"mojolearn: BpeTokenizer rank file {path} does not exist")
        m = _binding()
        self._adopt(m, _native(m.bpe_load, path), path)

    def _adopt(self, m, handle, source):
        self._source = source
        self._m = m
        self._handle = handle
        self._identity = _identity_of_handle(m, handle)
        self._n_vocab = int(m.bpe_n_vocab(handle))
        self._max_token_bytes = int(m.bpe_max_token_bytes(handle))

    @classmethod
    def _from_handle(cls, m, handle, source):
        tok = cls.__new__(cls)
        tok._adopt(m, handle, source)
        return tok

    # ------------------------------------------------------------ loading

    @classmethod
    def from_ranks_file(cls, path):
        """Load a rank file.

        One ``rank<TAB>hex_of_token_bytes`` line per rank, ascending from 0, with
        all 256 single bytes present (the format :meth:`TrainedBpeVocabulary.write_ranks`
        writes).

        Returns
        -------
        BpeTokenizer
        """
        return cls(path)

    @classmethod
    def from_token_bytes(cls, tokens):
        """Build a tokenizer from a list of token byte strings.

        A token's rank (and id) is its index. Tokens must be unique and non-empty,
        and all 256 single bytes must be present; otherwise ValueError is raised
        before anything is loaded.

        Parameters
        ----------
        tokens : sequence of bytes-like

        Returns
        -------
        BpeTokenizer
        """
        if isinstance(tokens, (str, bytes, bytearray, memoryview)):
            raise TypeError(f"mojolearn: tokens must be a sequence of bytes, got {type(tokens).__name__}")
        try:
            toks = list(tokens)
        except TypeError:
            raise TypeError(f"mojolearn: tokens must be a sequence of bytes, got {type(tokens).__name__}") from None
        blob, lengths = _blob(toks, "every token")
        m = _binding()
        handle = _native(m.bpe_load_tokens, [_ro(blob, "tokens"), _ro(lengths, "lengths")], [len(toks)])
        return cls._from_handle(m, handle, "<token bytes>")

    @classmethod
    def from_files(cls, encoder_json, vocab_bpe):
        """Load the two files of the GPT-2 format.

        Parameters
        ----------
        encoder_json : str or path-like
            ``encoder.json``: token spelling to id.
        vocab_bpe : str or path-like
            ``vocab.bpe``: the merge list.

        Returns
        -------
        BpeTokenizer

        Raises
        ------
        ValueError
            If the files are inconsistent: a spelling that does not decode through
            the byte-to-unicode table, ids that are not 0 to n-1 (with
            ``<|endoftext|>``, if present, at n), a merge whose parts or result are
            not tokens, merge results whose ids do not increase above both parts, or
            a multi-byte token no merge makes. These are the conditions under which
            merging by rank reproduces the merge list.
        """
        with open(encoder_json, "r", encoding="utf-8") as fh:
            encoder = json.load(fh)
        if not isinstance(encoder, dict):
            raise ValueError(f"mojolearn: {encoder_json} is not a JSON object of spelling to id")
        eot_id = encoder.pop(cls.ENDOFTEXT, None)
        n = len(encoder)
        if eot_id is not None and eot_id != n:
            raise ValueError(f"mojolearn: {encoder_json}: {cls.ENDOFTEXT} is id {eot_id}, not {n} (after the ranks)")
        with open(vocab_bpe, "r", encoding="utf-8") as fh:
            merges = fh.read()
        m = _binding()
        handle = _native(m.bpe_load_spelled, encoder, merges,
                         prefix=f"mojolearn: {encoder_json} + {vocab_bpe}: ")
        return cls._from_handle(m, handle, f"{os.path.abspath(encoder_json)} + {os.path.abspath(vocab_bpe)}")

    @classmethod
    def _synthetic(cls):
        """The synthetic vocabulary mojolearn trains itself
        (`_tokenizer_synthetic.py`), for the gates and the identity lane."""
        tok = cls.from_token_bytes(_tokenizer_synthetic.vocabulary())
        tok._source = "<mojolearn synthetic vocabulary>"
        return tok

    @property
    def n_vocab(self):
        """Number of ids: the ranks plus ``<|endoftext|>``."""
        return self._n_vocab

    @property
    def eot_token(self):
        """The id of ``<|endoftext|>``, ``n_vocab - 1``."""
        return self._n_vocab - 1

    @property
    def vocabulary_source(self):
        """Where the vocabulary was loaded from."""
        return self._source

    @property
    def identity(self):
        """The vocabulary's identity, independent of the file it came from.

        ``dict(schema, sha256, n_ranks, n_vocab, endoftext_id)``, with ``sha256``
        over the canonical rank file (``rank<TAB>lowercase hex`` per line). The same
        table loaded from a rank file, from ``encoder.json`` + ``vocab.bpe`` or from a
        trained vocabulary has the same identity. Models trained through
        :mod:`mojolearn.lm_corpus` record it, so their ids are never decoded with
        another table.
        """
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
        """Encode a byte string.

        Parameters
        ----------
        data : bytes-like
            The input; ``str`` raises TypeError (use :meth:`encode`).
        allow_endoftext : bool, default False
            Recognize ``<|endoftext|>`` as the special token.

        Returns
        -------
        list of int
        """
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
        """Encode text.

        Parameters
        ----------
        text : str or bytes-like
            ``str`` is encoded as UTF-8 first.
        allow_endoftext : bool, default False
            Recognize ``<|endoftext|>`` as the special token.

        Returns
        -------
        list of int
        """
        return self.encode_bytes(self._text_bytes(text), allow_endoftext)

    # The ids of each document, as a list of lists of int, in order.
    #
    # Each document is a str (UTF-8 encoded first) or bytes-like and is
    # encoded ALONE: `encode_batch(docs)[k] == encode(docs[k])` id for id,
    # whatever else is in the batch. The documents are encoded one after
    # another inside ONE binding call (`bpe_encode_batch`), which saves
    # the per-call crossing that dominates short documents.
    def encode_batch(self, documents, allow_endoftext=False):
        """Encode several documents in one call.

        Each document is encoded alone: ``encode_batch(docs)[k] ==
        encode(docs[k])``, whatever else is in the batch. One call is faster than
        many for short documents.

        Parameters
        ----------
        documents : sequence of str or bytes-like
        allow_endoftext : bool, default False
            Recognize ``<|endoftext|>`` as the special token.

        Returns
        -------
        list of list of int
        """
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
        for k, d in enumerate(docs):  # cpu-route: packs the text documents for the native encoder (text input)
            if not isinstance(d, (str, bytes, bytearray, memoryview)):
                raise TypeError(
                    f"mojolearn: document {k} must be str or bytes-like, got {type(d).__name__}"
                )
            raws.append(self._text_bytes(d))
        n_docs = len(raws)
        if n_docs == 0:
            return []
        text, lengths = _blob(raws, "every document")
        n = len(text)
        counts = array.array("q", bytes(8 * n_docs))
        ids = array.array("i", bytes(4 * max(n, 1)))
        text_buf = text if n > 0 else b"\0"
        total = int(self._m.bpe_encode_batch(
            self._handle, addr_ro(text_buf, name="text"), addr_ro(lengths, name="lengths"),
            addr(ids, name="ids"), addr(counts, name="counts"), [n_docs, n, n, allow]))
        if not 0 <= total <= n or sum(counts) != total:  # glue: checks the native token counts (counts-sized: per-document token counts)
            raise RuntimeError(
                f"mojolearn: bpe_encode_batch returned {total} ids for {n} bytes (counts sum {sum(counts)})"
            )
        out, at = [], 0
        for c in counts:  # glue: splits the native token output per document (counts-sized: per-document token counts)
            out.append(ids[at:at + c].tolist())
            at += c
        return out

    def _encode_corpus(self, data, points, document_bytes):
        """`lm_corpus._tokenize`'s work in one call (`bpe_encode_corpus`):
        `data` cut into documents by the corpus rule, each encoded alone.
        Returns `(ids int32 array, {point: id offset}, n_documents, max_id,
        ids_above_255)`."""
        n = len(data)
        pts = array.array("q", points)
        ids = array.array("i", bytes(4 * max(n, 1)))
        bounds = array.array("q", bytes(8 * max(len(pts), 1)))
        stats = array.array("q", bytes(24))
        total = int(_native(self._m.bpe_encode_corpus, self._handle,
                            [_ro(data, "corpus"), addr_ro(pts, name="points"), addr(ids, name="ids"),
                             addr(bounds, name="bounds"), addr(stats, name="stats")],
                            [n, len(pts), int(document_bytes), n]))
        del ids[total:]
        return ids, dict(zip(points, bounds)), int(stats[0]), int(stats[1]), int(stats[2])

    # ------------------------------------------------------------ decode

    @staticmethod
    def _id_list(ids):
        if isinstance(ids, (str, bytes, bytearray)):
            raise TypeError(f"mojolearn: ids must be a sequence of int, got {type(ids).__name__}")
        try:
            return list(ids)
        except TypeError:
            raise TypeError(f"mojolearn: ids must be a sequence of int, got {type(ids).__name__}") from None

    def decode_bytes(self, ids):
        """Decode ids to bytes.

        Parameters
        ----------
        ids : sequence of int
            Each in ``[0, n_vocab)``; an invalid id raises an error naming its value
            and position before anything is decoded.

        Returns
        -------
        bytes
        """
        seq = self._id_list(ids)
        if not seq:
            return b""
        cap = len(seq) * self._max_token_bytes
        out = bytearray(cap)
        count = int(_native(self._m.bpe_decode_seq, self._handle, seq, addr(out, name="text"), cap))
        return bytes(out[:count])

    def decode(self, ids, errors="replace"):
        """Decode ids to text.

        Parameters
        ----------
        ids : sequence of int
        errors : str, default "replace"
            Passed to ``bytes.decode``.

        Returns
        -------
        str
        """
        return self.decode_bytes(ids).decode("utf-8", errors)

    def decode_bytes_batch(self, batch):
        """Decode several id sequences to bytes in one call.

        Equivalent to ``[decode_bytes(ids) for ids in batch]``; an invalid id is
        reported by its position within its own sequence.

        Returns
        -------
        list of bytes
        """
        if isinstance(batch, (str, bytes, bytearray)):
            raise TypeError(f"mojolearn: decode_bytes_batch takes a sequence of id sequences, got {type(batch).__name__}")
        seqs = self._id_list(batch)
        cap = sum(map(len, seqs)) * self._max_token_bytes  # glue: output capacity of the native decode (seqs-sized: token sequences)
        out = bytearray(max(cap, 1))
        lengths = array.array("q", bytes(8 * max(len(seqs), 1)))
        _native(self._m.bpe_decode_batch, self._handle, seqs, addr(out, name="text"), cap,
                addr(lengths, name="lengths"))
        result, at = [], 0
        for k in range(len(seqs)):  # glue: splits the native decode output per sequence (seqs-sized: token sequences)
            result.append(bytes(out[at:at + lengths[k]]))
            at += lengths[k]
        return result

    def decode_batch(self, batch, errors="replace"):
        """Decode several id sequences to text: ``[decode(ids, errors) for ids in batch]``."""
        return [b.decode("utf-8", errors) for b in self.decode_bytes_batch(batch)]  # cpu-route: decodes bytes to str per sequence (text output)

    def __repr__(self):
        return f"BpeTokenizer(n_vocab={self._n_vocab}, vocabulary={self._source!r})"


class TrainedBpeVocabulary:
    """A vocabulary trained by :meth:`BpeVocabularyTrainer.train`.

    Attributes
    ----------
    tokens : list of bytes
        Token byte strings in rank order (rank = id), the 256 single bytes
        first.
    merges : list of tuple of int
        ``(left_id, right_id, new_id)`` in the order the merges were made.
    stats : dict
        Training statistics, including ``n_ties_broken`` and ``tie_break``.
    """

    def __init__(self, tokens, merges, stats):
        self.tokens = tokens
        self.merges = merges
        self.stats = dict(stats)

    @property
    def n_tokens(self):
        """Number of ranks; ``<|endoftext|>`` takes the id after the last one."""
        return len(self.tokens)

    @property
    def n_ties_broken(self):
        """How many merge selections had two or more pairs at the top count."""
        return self.stats["n_ties_broken"]

    @property
    def tie_break(self):
        """The rule that settles equal counts, as text."""
        return self.stats["tie_break"]

    def _render(self, which):
        blob, lengths = _blob(self.tokens, "every token")
        left = array.array("q", map(operator.itemgetter(0), self.merges))
        right = array.array("q", map(operator.itemgetter(1), self.merges))
        m = _binding()
        return _native(m.bpe_render, [_ro(blob, "tokens"), _ro(lengths, "lengths"), _ro(left, "merge_left"),
                                      _ro(right, "merge_right")], [len(self.tokens), len(self.merges), which])

    # OUR format as text: `rank<TAB>hex_of_token_bytes` per line
    # (`tokenizer/train/emit.mojo::render_ranks`).
    def render_ranks(self):
        """The vocabulary as rank-file text: ``rank<TAB>hex_of_token_bytes`` per line."""
        return self._render(0)

    # A `tokenizer.json` Hugging Face `tokenizers` loads, as text
    # (`tokenizer/train/emit.mojo::render_tokenizer_json`).
    def render_tokenizer_json(self):
        """The vocabulary as Hugging Face ``tokenizer.json`` text, loadable by the ``tokenizers`` library."""
        return self._render(1)

    def write_ranks(self, path):
        """Write :meth:`render_ranks` to ``path`` and return ``path``."""
        with open(path, "w", encoding="ascii", newline="\n") as fh:
            fh.write(self.render_ranks())
        return path

    def write_tokenizer_json(self, path):
        """Write :meth:`render_tokenizer_json` to ``path`` and return ``path``."""
        with open(path, "w", encoding="ascii", newline="\n") as fh:
            fh.write(self.render_tokenizer_json())
        return path

    @property
    def identity(self):
        """The :attr:`BpeTokenizer.identity` of this vocabulary, without loading it."""
        return _identity_from_tokens_text(self.render_ranks(), self.n_tokens)

    def tokenizer(self):
        """A :class:`BpeTokenizer` over this vocabulary, without writing a file."""
        return BpeTokenizer.from_token_bytes(self.tokens)

    def __repr__(self):
        return (f"TrainedBpeVocabulary(n_tokens={self.n_tokens}, "
                f"n_merges={len(self.merges)}, n_ties_broken={self.n_ties_broken})")


# Train a byte-level BPE vocabulary, deterministically.
#
#     tok = BpeVocabularyTrainer(vocab_size=32000).train(documents).tokenizer()
#
# mojolearn ships no vocabulary and no corpus; `documents` is yours, as a
# sequence of `bytes` (or `str`, encoded UTF-8 first). Each document is
# pre-tokenized ALONE, so no pre-token spans a document join and the ORDER
# the documents arrive in cannot reach the result.
#
# WHAT IS BEING CLAIMED. Vocabulary training is host-only everywhere --
# Hugging Face, SentencePiece and tiktoken all train on a CPU -- so this is
# not a cross-vendor GPU claim and there is no vendor column. The claim is
# that THE SAME CORPUS AND CONFIG PRODUCE THE SAME VOCABULARY BYTES ON ANY
# MACHINE AND ARCHITECTURE, and it rests on four things: a total order on
# the tie-break (highest count, then smallest `(left_id, right_id)`),
# single-threaded counting so there is no reduction order to get wrong,
# selection that never depends on an iteration order, and no float anywhere
# in the selection.
#
# THE TRAINER is the Mojo one, `tokenizer/train/bpe_train.mojo`, through
# the tokenizer host binding (`bpe_train`). `backend` is kept for its
# old spellings: "auto" and "mojo" both name it. The pure Python
# reference `_bpe_trainer.train` is a VERIFICATION oracle only
# (lane/pyglue-text-io, 2026-10-03: Python is glue at run time), held to
# the Mojo trainer's bytes by `pixi run check-bpe-trainer` and the
# tokenizer surface test; `backend="python"` is refused by name.
# `stats["backend"]` records "mojo".
# `MOJOLEARN_BPE_TRAINER_SABOTAGE=1` reverses the tie-break (the negative
# control).
class BpeVocabularyTrainer:
    """Train a byte-level BPE vocabulary deterministically.

    The same corpus and settings give the same vocabulary bytes on any machine:
    ties are broken by a total order (highest count, then smallest
    ``(left_id, right_id)``), counting is exact and single-threaded, and no
    floating point is involved. Training runs on the CPU in compiled code.

    Each document is pre-tokenized alone, so no pre-token spans two documents
    and the order of the documents does not affect the result.

    Parameters
    ----------
    vocab_size : int, default 32000
        Number of ranks to train, including the 256 single bytes; at least 256.
    min_frequency : int, default 2
        Minimum pair count for a merge; at least 1.
    backend : {"auto", "mojo"}, default "auto"
        Both name the compiled trainer.

    Examples
    --------
    >>> from mojolearn.tokenizer import BpeVocabularyTrainer
    >>> docs = ["the cat sat on the mat", "the dog sat on the log"]
    >>> vocab = BpeVocabularyTrainer(vocab_size=300).train(docs)
    >>> tok = vocab.tokenizer()
    >>> tok.decode(tok.encode("the cat"))
    'the cat'
    """

    _BACKENDS = ("auto", "mojo")

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
        if backend == "python":
            raise ValueError("mojolearn: backend='python' is gone: the Python BPE trainer is a verification "
                             "oracle (mojolearn._bpe_trainer.train), not a run-time route; use backend='auto'")
        if backend not in self._BACKENDS:
            raise ValueError(f"mojolearn: backend must be one of {self._BACKENDS}, got {backend!r}")
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.backend = backend

    def train(self, documents):
        """Train on ``documents``.

        Parameters
        ----------
        documents : sequence of str or bytes-like
            At least one document; ``str`` is encoded as UTF-8.

        Returns
        -------
        TrainedBpeVocabulary
        """
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
        for k, d in enumerate(docs):  # cpu-route: packs the text documents for the native encoder (text input)
            if isinstance(d, str):
                raws.append(d.encode("utf-8"))
            elif isinstance(d, (bytes, bytearray, memoryview)):
                raws.append(bytes(d))
            else:
                raise TypeError(
                    f"mojolearn: document {k} must be str or bytes-like, got {type(d).__name__}")
        native = _native_trainer(required=True)
        tokens, merges, stats = _train_native(native, raws, self.vocab_size, self.min_frequency,
                                              _bpe_trainer.sabotaged())
        stats["backend"] = "mojo"
        return TrainedBpeVocabulary(tokens, merges, stats)

    def __repr__(self):
        return (f"BpeVocabularyTrainer(vocab_size={self.vocab_size}, "
                f"min_frequency={self.min_frequency}, backend={self.backend!r})")


def _native_trainer(required):
    """The binding when it exports `bpe_train`, else None (or the reason,
    raised, when `required`)."""
    try:
        module = _backend.load_host_module(_EXTENSION)
    except (ImportError, AttributeError) as exc:
        if required:
            raise ImportError(f"mojolearn: backend='mojo' needs the tokenizer host binding: {exc}") from exc
        return None
    if not all(hasattr(module, n) for n in ("bpe_train", "bpe_trained_sizes", "bpe_trained_copy")):  # glue: checks three trainer binding entries
        if required:
            raise ImportError("mojolearn: backend='mojo': this tokenizer binding predates bpe_train; rebuild it")
        return None
    return module


def _train_native(module, raws, vocab_size, min_frequency, break_ties_high):
    """`(tokens, merges, stats)` from the Mojo trainer: one crossing in (the
    documents back to back with int64 lengths), one out (the token bytes,
    lengths and merge ids)."""
    text, lengths = _blob(raws, "every document")
    n = len(text)
    handle = _native(module.bpe_train, addr_ro(text, name="documents") if n else 0,
                     addr_ro(lengths, name="lengths"),
                     [len(raws), n, int(vocab_size), int(min_frequency), bool(break_ties_high)])
    return _read_trained(module, handle, vocab_size, min_frequency)


def _read_trained(module, handle, vocab_size, min_frequency):
    n_tokens, arena_bytes, n_merges, n_ties, n_groups = (int(x) for x in module.bpe_trained_sizes(handle))  # glue: unpacks five trainer sizes
    arena = bytearray(max(arena_bytes, 1))
    lengths = array.array("q", [0]) * n_tokens
    left = array.array("q", [0]) * max(n_merges, 1)
    right = array.array("q", [0]) * max(n_merges, 1)
    module.bpe_trained_copy(handle, addr(arena, name="arena"), addr(lengths, name="lengths"),
                            addr(left, name="merge_left"), addr(right, name="merge_right"))
    # The result as the public shape: token bytes by rank, merges as
    # (left, right, new) ids.
    tokens, at = [], 0
    for m in lengths:  # glue: splits the native trained token arena (lengths-sized: trained token lengths)
        tokens.append(bytes(arena[at:at + m]))
        at += m
    merges = list(zip(left[:n_merges], right[:n_merges], range(256, 256 + n_merges)))
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


def _train_corpus(data, points, lo, hi, document_bytes, vocab_size, min_frequency):
    """`lm_corpus`'s vocabulary: the documents of `data[lo:hi]` (cut by the
    corpus rule in Mojo, `bpe_train_corpus`) trained in the same call.
    `points` are the corpus's sorted split points."""
    module = _native_trainer(required=True)
    pts = array.array("q", points)
    n = len(data)
    handle = _native(module.bpe_train_corpus, [_ro(data, "corpus"), _ro(pts, "points")],
                     [n, len(pts), int(lo), int(hi), int(document_bytes), int(vocab_size), int(min_frequency),
                      bool(_bpe_trainer.sabotaged())])
    tokens, merges, stats = _read_trained(module, handle, vocab_size, min_frequency)
    stats["backend"] = "mojo"
    return TrainedBpeVocabulary(tokens, merges, stats)


def _gather_rows(ids_all, out, batch, length, lo, modulus, base):
    """`out[b] = ids_all[lo + (base + b*length) % modulus : + length + 1]`
    for every row b, in one binding call (int32 buffers by address)."""
    m = _binding()
    return _native(m.tokens_gather_rows, [addr_ro(ids_all, name="ids"), addr(out, name="out")],
                   [int(ids_all.size), int(batch), int(length), int(lo), int(modulus), int(base)])


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
