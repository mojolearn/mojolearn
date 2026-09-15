# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GPT-2 byte-level BPE tokenizer (the expose-tokenizer lane, 2026-09-14).

`GPT2Tokenizer` computes the id sequence that tiktoken 0.14.0's `gpt2`
encoding assigns to a byte string, id for id, and the bytes back. The
arithmetic is `tokenizer/encoding.mojo` (integers and tables only, no float,
no device kernel), reached through the host binding
`_mojolearn_tokenizer_host` under `mojolearn/host/`, built from source with
`bindings/build_tokenizer_host.sh`. The same binary serves a GPU box and a
CPU-only install, because there is no GPU path to route from: the tokenizer
is host work by construction.

What is asserted, and where: `pixi run check-tokenizer` holds the Mojo
encoder to 43 cases recorded from tiktoken (exact id sequences, byte-exact
round trips, the pattern and table preconditions and the pre-tokenizer reach
arms); `python/mojolearn/tests/test_tokenizer_surface.py` holds THIS door to
the same 43 cases through the binding, plus the refusals below. Cross-vendor
bitwise identity is true by construction here and is not a claim this module
makes; agreement with the reference is the property that can be wrong.

The two tables (`gpt2_ranks.tsv`, `unicode_categories.tsv`) are read from
the first of: the directory `MOJOLEARN_TOKENIZER_DATA` names, the package's
own `mojolearn/data/tokenizer/` (absent until a wheel carries it), and the
checkout's `tokenizer/data/`. None found is refused by name.
"""
import array
import os

from . import _backend
from ._buffer import addr, addr_ro

_EXTENSION = "_mojolearn_tokenizer_host"
_DATA_ENV = "MOJOLEARN_TOKENIZER_DATA"
_RANKS_FILE = "gpt2_ranks.tsv"
_UNICODE_FILE = "unicode_categories.tsv"

__all__ = ["GPT2Tokenizer", "data_dir"]


def _candidates():
    here = os.path.dirname(os.path.abspath(__file__))
    out = []
    override = os.environ.get(_DATA_ENV, "").strip()
    if override:
        out.append((_DATA_ENV, os.path.abspath(override)))
    out.append(("the package's data directory", os.path.join(here, "data", "tokenizer")))
    out.append(("the checkout's tokenizer/data", os.path.normpath(os.path.join(here, "..", "..", "tokenizer", "data"))))
    return out


def data_dir():
    """The directory holding both tables, resolved in the order the module
    docstring gives; FileNotFoundError naming every place looked at."""
    looked = []
    for label, d in _candidates():
        if os.path.isfile(os.path.join(d, _RANKS_FILE)) and os.path.isfile(os.path.join(d, _UNICODE_FILE)):
            return d
        looked.append(f"{label}: {d}")
    raise FileNotFoundError(
        f"mojolearn: no directory holds both {_RANKS_FILE} and {_UNICODE_FILE}; "
        f"looked at {'; '.join(looked)}. Set {_DATA_ENV} to a directory that does"
    )


def _binding():
    return _backend.load_host_module(_EXTENSION)


class GPT2Tokenizer:
    """`encode` and `decode` for tiktoken 0.14.0's `gpt2` encoding.

    Ids are in [0, 50257): 50256 ranks from `gpt2_ranks.tsv` and the special
    token `<|endoftext|>` at 50256. The special token is recognized only
    when `allow_endoftext=True`; otherwise its thirteen characters are
    ordinary text and encode as seven ids, which is tiktoken's
    `encode_ordinary`. There is no "raise on a disallowed special token"
    mode and no other special token (`tokenizer/NOT_IMPLEMENTED.tsv`).

    Bytes are the interface: `encode_bytes` and `decode_bytes` are the real
    entries. `encode` accepts `str` (UTF-8 encoded first) or a bytes-like
    object; `decode` is `decode_bytes` decoded as UTF-8 with
    `errors="replace"`, tiktoken's own default, because a token stream cut
    inside a character is still the right bytes. Invalid UTF-8 input
    encodes as one-byte pre-tokens and round trips (tiktoken's `&str` input
    cannot hold it, so there is nothing to compare against there).
    """

    N_VOCAB = 50257
    ENDOFTEXT_ID = 50256
    ENDOFTEXT = "<|endoftext|>"

    def __init__(self, data_directory=None):
        d = os.path.abspath(data_directory) if data_directory is not None else data_dir()
        ranks = os.path.join(d, _RANKS_FILE)
        unicode = os.path.join(d, _UNICODE_FILE)
        for p in (ranks, unicode):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"mojolearn: GPT2Tokenizer table {p} does not exist")
        self._data_dir = d
        self._m = _binding()
        self._handle = self._m.gpt2_load(ranks, unicode)
        self._n_vocab = int(self._m.gpt2_n_vocab(self._handle))
        self._max_token_bytes = int(self._m.gpt2_max_token_bytes(self._handle))
        if self._n_vocab != self.N_VOCAB:
            raise RuntimeError(
                f"mojolearn: {ranks} loads {self._n_vocab} ids, not the gpt2 encoding's {self.N_VOCAB}"
            )

    @property
    def n_vocab(self):
        return self._n_vocab

    @property
    def eot_token(self):
        """`<|endoftext|>`'s id, 50256."""
        return self.ENDOFTEXT_ID

    @property
    def data_directory(self):
        return self._data_dir

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
        count = int(self._m.gpt2_encode(
            self._handle, addr_ro(raw, name="text"), n, addr(out, name="ids"), n, allow))
        if not 0 <= count <= n:
            raise RuntimeError(f"mojolearn: gpt2_encode returned {count} ids for {n} bytes")
        return out[:count].tolist()

    def encode(self, text, allow_endoftext=False):
        """The ids of `text` (str, UTF-8 encoded first, or bytes-like)."""
        return self.encode_bytes(self._text_bytes(text), allow_endoftext)

    def encode_batch(self, documents, allow_endoftext=False):
        """The ids of each document, as a list of lists of int, in order.

        Each document is a str (UTF-8 encoded first) or bytes-like and is
        encoded ALONE: `encode_batch(docs)[k] == encode(docs[k])` id for id,
        whatever else is in the batch. This is tiktoken's `encode_batch`
        with `allow_endoftext` in place of `allowed_special` and no
        `num_threads`: the documents are encoded one after another inside
        ONE binding call (`gpt2_encode_batch`), which saves the per-call
        crossing that dominates short documents."""
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
        total = int(self._m.gpt2_encode_batch(
            self._handle, addr_ro(text_buf, name="text"), addr_ro(offsets, name="offsets"),
            addr(ids, name="ids"), addr(counts, name="counts"), [n_docs, n, n, allow]))
        if not 0 <= total <= n or sum(counts) != total:
            raise RuntimeError(
                f"mojolearn: gpt2_encode_batch returned {total} ids for {n} bytes (counts sum {sum(counts)})"
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
        """The bytes of an id sequence. An id outside [0, 50257) is refused
        by value and position before the binding is called."""
        arr, n = self._ids_array(ids)
        if n == 0:
            return b""
        cap = n * self._max_token_bytes
        out = bytearray(cap)
        count = int(self._m.gpt2_decode(
            self._handle, addr_ro(arr, name="ids"), n, addr(out, name="text"), cap))
        if not 0 <= count <= cap:
            raise RuntimeError(f"mojolearn: gpt2_decode returned {count} bytes into {cap}")
        return bytes(out[:count])

    def decode(self, ids, errors="replace"):
        """`decode_bytes(ids)` as text; `errors` is passed to `bytes.decode`."""
        return self.decode_bytes(ids).decode("utf-8", errors)

    def decode_bytes_batch(self, batch):
        """`[decode_bytes(ids) for ids in batch]`. A thin loop over the one
        `gpt2_decode` call per sequence: decode is a table copy, and a
        sequence of ids already carries its own boundaries."""
        if isinstance(batch, (str, bytes, bytearray)):
            raise TypeError(f"mojolearn: decode_bytes_batch takes a sequence of id sequences, got {type(batch).__name__}")
        return [self.decode_bytes(ids) for ids in batch]

    def decode_batch(self, batch, errors="replace"):
        """`[decode(ids, errors) for ids in batch]`, tiktoken's `decode_batch`."""
        return [b.decode("utf-8", errors) for b in self.decode_bytes_batch(batch)]

    def __repr__(self):
        return f"GPT2Tokenizer(n_vocab={self._n_vocab}, data_directory={self._data_dir!r})"
