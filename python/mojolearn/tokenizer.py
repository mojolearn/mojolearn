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
#   `mojolearn.models.tokenizer` cuts Llama 3 and Qwen 2 in native code (see
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
import os as _os

from . import _backend

_EXTENSION = "_mojolearn_tokenizer_host"

__all__ = ["BpeTokenizer", "BpeVocabularyTrainer", "TrainedBpeVocabulary"]


#: The binding's entries were `gpt2_*` until 2026-09-18 and are `bpe_*`
#: since. A binding built before the rename (an older host directory under
#: MOJOLEARN_HOST_DIR) is read through its old names so it keeps loading;
#: the ids are the same arithmetic under either name.
_ABI = ("load", "n_vocab", "max_token_bytes", "encode", "encode_batch")


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


_VOCABULARY_SCHEMA = "mojolearn.bpe-vocabulary.v1"


def _binding():
    return _Entries(_backend.load_host_module(_EXTENSION))  # import-time: tokenizer host binding module loaded once by load_host_module (text input step)


#: The fast tier's Apple GPU tokenizer (`tokenizer/fast/`, built by
#: `bindings/build_tokenizer_fast.sh`). Used only in numeric mode fast, only
#: when built, and only for the device paths it compiled (`bpe_fast_flags()`:
#: each needs an Apple GPU target and its `-D MOJOLEARN_BPE_*` define, all
#: default off). Every other call runs on the host binding as before.
_FAST_EXTENSION = "_mojolearn_tokenizer_fast"
_NO_FAST = (None, False, False, 0)


def _fast_binding(*, connect_all=False):
    """`(module, train_device, encode_device, max_vocab)`, or `_NO_FAST`."""
    selected = _backend.default_mode()
    if not connect_all and selected != "fast":
        return _NO_FAST
    try:
        path = _os.path.join(_backend.tier_dir("fast"), _FAST_EXTENSION + ".so")
    except ImportError:
        return _NO_FAST
    if not _os.path.exists(path):
        return _NO_FAST
    module = _backend._load_member("fast", _FAST_EXTENSION, path)
    if int(module.tokenizer_fast_numeric_mode()) != 0:
        raise ImportError(f"mojolearn: {path} was not compiled fast; rebuild it with "
                          "MOJOLEARN_NUMERIC_MODE=fast bash bindings/build_tokenizer_fast.sh")
    train, encode, _batch, _filter, _livebuf, max_vocab = module.bpe_fast_flags()
    return module, bool(train), bool(encode), int(max_vocab)


#: The identical GPU tokenizer (`tokenizer/gpu/`, binding `_mojolearn_x_tokenizer`): what a
#: BpeTokenizer built with device="gpu" runs. Connected once at import:
#: `(module, bpe_gpu_load, bpe_gpu_encode_batch, bpe_gpu_decode_batch, bpe_gpu_train,
#: bpe_gpu_trained_sizes, bpe_gpu_trained_copy)`, or a one-item tuple
#: holding the words device="gpu" is refused with where it is not installed.
_GPU_EXTENSION = "_mojolearn_x_tokenizer"


def _gpu_binding():
    if _backend._CPU_ONLY is not None:
        return ("ValueError: device='gpu' needs a GPU build of mojolearn, and this installation is CPU only",)
    try:
        # import-time: the GPU tokenizer of this process's tier, connected once (integer work: both
        # tiers are the same code and give the same ids)
        module = _backend.binding(_GPU_EXTENSION, _backend.default_mode())
        return (module, module.bpe_gpu_load, module.bpe_gpu_encode_batch, module.bpe_gpu_decode_batch,
                module.bpe_gpu_train, module.bpe_gpu_trained_sizes, module.bpe_gpu_trained_copy)
    except (ImportError, AttributeError, ValueError, OSError) as error:
        return ("ImportError: device='gpu' needs the GPU tokenizer binding, which is not connected: " + str(error),)


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
    device : {None, 'gpu', 'cpu'}, default None
        Where encoding and decoding run. None follows the numeric mode. 'cpu'
        runs on the CPU and raises ValueError in numeric mode 'fast'. 'gpu'
        runs on the GPU, with the same ids as the CPU, and raises ValueError on
        a CPU-only installation. Other values raise ValueError.

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

    def __init__(self, ranks_file=None, *, device=None):
        return _TOKENIZER_NATIVE.tokenizer_public_call('bpe_init', (self, ranks_file, device), _TOKENIZER_CONTEXT)

    def _adopt(self, m, handle, source):
        return _TOKENIZER_NATIVE.tokenizer_public_call('bpe_adopt', (self, handle, source), _TOKENIZER_CONTEXT)

    # ------------------------------------------------------------ loading

    @classmethod
    def from_ranks_file(cls, path, *, device=None):
        """Load a rank file.

        One ``rank<TAB>hex_of_token_bytes`` line per rank, ascending from 0, with
        all 256 single bytes present (the format :meth:`TrainedBpeVocabulary.write_ranks`
        writes).

        Parameters
        ----------
        path : str or path-like
            The rank file.
        device : {None, 'gpu', 'cpu'}, default None
            As for the constructor.

        Returns
        -------
        BpeTokenizer
        """
        return _TOKENIZER_NATIVE.tokenizer_public_call('bpe_ranks', (cls, path, device), _TOKENIZER_CONTEXT)

    @classmethod
    def from_token_bytes(cls, tokens, *, device=None):
        """Build a tokenizer from a list of token byte strings.

        A token's rank (and id) is its index. Tokens must be unique and non-empty,
        and all 256 single bytes must be present; otherwise ValueError is raised
        before anything is loaded.

        Parameters
        ----------
        tokens : sequence of bytes-like
        device : {None, 'gpu', 'cpu'}, default None
            As for the constructor.

        Returns
        -------
        BpeTokenizer
        """
        return _TOKENIZER_NATIVE.tokenizer_public_call('bpe_tokens', (cls, tokens, device), _TOKENIZER_CONTEXT)

    @classmethod
    def from_files(cls, encoder_json, vocab_bpe, *, device=None):
        """Load the two files of the GPT-2 format.

        Parameters
        ----------
        encoder_json : str or path-like
            ``encoder.json``: token spelling to id.
        vocab_bpe : str or path-like
            ``vocab.bpe``: the merge list.
        device : {None, 'gpu', 'cpu'}, default None
            As for the constructor.

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
        return _TOKENIZER_NATIVE.tokenizer_public_call('bpe_files', (cls, encoder_json, vocab_bpe, device), _TOKENIZER_CONTEXT)

    @property
    def n_vocab(self):
        """Number of ids: the ranks plus ``<|endoftext|>``."""
        return _TOKENIZER_NATIVE.tokenizer_public_call('bpe_fact', (self, 'n_vocab'), _TOKENIZER_CONTEXT)

    @property
    def eot_token(self):
        """The id of ``<|endoftext|>``, ``n_vocab - 1``."""
        return _TOKENIZER_NATIVE.tokenizer_public_call('bpe_fact', (self, 'eot_token'), _TOKENIZER_CONTEXT)

    @property
    def vocabulary_source(self):
        """Where the vocabulary was loaded from."""
        return _TOKENIZER_NATIVE.tokenizer_public_call('bpe_fact', (self, 'source'), _TOKENIZER_CONTEXT)

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
        return _TOKENIZER_NATIVE.tokenizer_public_call('bpe_fact', (self, 'identity'), _TOKENIZER_CONTEXT)

    # ------------------------------------------------------------ encode

    # The uncalled `_text_bytes` / `_flag` refusals are gone (lane py-tail,
    # 2026-11-15): the tokenizer binding refuses text and `allow_endoftext`
    # itself (bindings/_mojolearn_tokenizer_host.mojo `_flag`).

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
        return _TOKENIZER_NATIVE.tokenizer_public_call('bpe_encode', (self, data, allow_endoftext, True, False), _TOKENIZER_CONTEXT)

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
        return _TOKENIZER_NATIVE.tokenizer_public_call('bpe_encode', (self, text, allow_endoftext, False, False), _TOKENIZER_CONTEXT)

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
        return _TOKENIZER_NATIVE.tokenizer_public_call('bpe_encode', (self, documents, allow_endoftext, False, True), _TOKENIZER_CONTEXT)


    # ------------------------------------------------------------ decode

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
        return _TOKENIZER_NATIVE.tokenizer_public_call('bpe_decode', (self, ids, True, False, 'replace'), _TOKENIZER_CONTEXT)

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
        return _TOKENIZER_NATIVE.tokenizer_public_call('bpe_decode', (self, ids, False, False, errors), _TOKENIZER_CONTEXT)

    def decode_bytes_batch(self, batch):
        """Decode several id sequences to bytes in one call.

        Equivalent to ``[decode_bytes(ids) for ids in batch]``; an invalid id is
        reported by its position within its own sequence.

        Returns
        -------
        list of bytes
        """
        return _TOKENIZER_NATIVE.tokenizer_public_call('bpe_decode', (self, batch, True, True, 'replace'), _TOKENIZER_CONTEXT)

    def decode_batch(self, batch, errors="replace"):
        """Decode several id sequences to text: ``[decode(ids, errors) for ids in batch]``."""
        return _TOKENIZER_NATIVE.tokenizer_public_call('bpe_decode', (self, batch, False, True, errors), _TOKENIZER_CONTEXT)

    def __repr__(self):
        return _TOKENIZER_NATIVE.tokenizer_public_call('bpe_fact', (self, 'repr'), _TOKENIZER_CONTEXT)


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
        return _TOKENIZER_NATIVE.tokenizer_public_call('trained_init', (self, tokens, merges, stats), _TOKENIZER_CONTEXT)

    @property
    def n_tokens(self):
        """Number of ranks; ``<|endoftext|>`` takes the id after the last one."""
        return _TOKENIZER_NATIVE.tokenizer_public_call('trained_fact', (self, 'n_tokens'), _TOKENIZER_CONTEXT)

    @property
    def n_ties_broken(self):
        """How many merge selections had two or more pairs at the top count."""
        return _TOKENIZER_NATIVE.tokenizer_public_call('trained_fact', (self, 'n_ties_broken'), _TOKENIZER_CONTEXT)

    @property
    def tie_break(self):
        """The rule that settles equal counts, as text."""
        return _TOKENIZER_NATIVE.tokenizer_public_call('trained_fact', (self, 'tie_break'), _TOKENIZER_CONTEXT)

    # OUR format as text: `rank<TAB>hex_of_token_bytes` per line
    # (`tokenizer/train/emit.mojo::render_ranks`).
    def render_ranks(self):
        """The vocabulary as rank-file text: ``rank<TAB>hex_of_token_bytes`` per line."""
        return _TOKENIZER_NATIVE.tokenizer_public_call('trained_render', (self, 0), _TOKENIZER_CONTEXT)

    # A `tokenizer.json` Hugging Face `tokenizers` loads, as text
    # (`tokenizer/train/emit.mojo::render_tokenizer_json`).
    def render_tokenizer_json(self):
        """The vocabulary as Hugging Face ``tokenizer.json`` text, loadable by the ``tokenizers`` library."""
        return _TOKENIZER_NATIVE.tokenizer_public_call('trained_render', (self, 1), _TOKENIZER_CONTEXT)

    def write_ranks(self, path):
        """Write :meth:`render_ranks` to ``path`` and return ``path``."""
        return _TOKENIZER_NATIVE.tokenizer_public_call('trained_write', (self, 0, path), _TOKENIZER_CONTEXT)

    def write_tokenizer_json(self, path):
        """Write :meth:`render_tokenizer_json` to ``path`` and return ``path``."""
        return _TOKENIZER_NATIVE.tokenizer_public_call('trained_write', (self, 1, path), _TOKENIZER_CONTEXT)

    @property
    def identity(self):
        """The :attr:`BpeTokenizer.identity` of this vocabulary, without loading it."""
        return _TOKENIZER_NATIVE.tokenizer_public_call('trained_fact', (self, 'identity'), _TOKENIZER_CONTEXT)

    def tokenizer(self, *, device=None):
        """A :class:`BpeTokenizer` over this vocabulary, without writing a file.

        Parameters
        ----------
        device : {None, 'gpu', 'cpu'}, default None
            As for the :class:`BpeTokenizer` constructor.
        """
        return _TOKENIZER_NATIVE.tokenizer_public_call('trained_tokenizer', (self, device), _TOKENIZER_CONTEXT)

    def __repr__(self):
        return _TOKENIZER_NATIVE.tokenizer_public_call('trained_fact', (self, 'repr'), _TOKENIZER_CONTEXT)


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
# reference `tests/_bpe_trainer.train` is a VERIFICATION oracle only
# (lane/pyglue-text-io, 2026-10-03: Python is glue at run time), held to
# the Mojo trainer's bytes by `pixi run check-bpe-trainer` and the
# tokenizer surface test; `backend="python"` is refused by name.
# `stats["backend"]` records "mojo".
# A build with `-D MOJOLEARN_BPE_TRAINER_SABOTAGE=1` reverses the tie-break
# (the negative control).
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
    device : {None, 'gpu', 'cpu'}, default None
        Where training runs. None follows the numeric mode. 'cpu' runs on the
        CPU and raises ValueError in numeric mode 'fast'. 'gpu' runs on the GPU,
        with the same vocabulary as the CPU, for vocab_size up to 46340, and
        raises ValueError on a CPU-only installation. Other values raise
        ValueError.

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

    def __init__(self, vocab_size=32000, min_frequency=2, backend="auto", *, device=None):
        return _TOKENIZER_NATIVE.tokenizer_public_call('trainer_init', (self, vocab_size, min_frequency, backend, device), _TOKENIZER_CONTEXT)

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
        return _TOKENIZER_NATIVE.tokenizer_public_call('trainer_train', (self, documents), _TOKENIZER_CONTEXT)

    def __repr__(self):
        return _TOKENIZER_NATIVE.tokenizer_public_call('trainer_repr', (self,), _TOKENIZER_CONTEXT)


def _gather_rows(ids_all, out, batch, length, lo, modulus, base):
    """`out[b] = ids_all[lo + (base + b*length) % modulus : + length + 1]`
    for every row b, in one binding call (int32 buffers by address)."""
    return _TOKENIZER_NATIVE.tokenizer_public_call('gather_rows', (ids_all, out, batch, length, lo, modulus, base), _TOKENIZER_CONTEXT)


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


# Import-time binding connection, shared by all public operations. Fast's
# existing optional route is retained; no Python mode lookup runs per call.
import _codecs as _tokenizer_codecs
from ._native_loaded_modules import connect_module as _connect_tokenizer_module
from ._array import Array as _Array
from ._native_classical_bridge import _numpy_free_context
_connect_tokenizer_module("_codecs", _tokenizer_codecs, ("lookup_error",))
_TOKENIZER_NATIVE = _binding()._module
_TOKENIZER_FAST = _fast_binding(connect_all=True)
_TOKENIZER_FAST_ENTRIES = tuple(
    getattr(_TOKENIZER_FAST[0], name, None) for name in
    ("bpe_fast_load", "bpe_encode_batch", "bpe_train", "bpe_trained_sizes", "bpe_trained_copy"))
_TOKENIZER_CONTEXT = (BpeTokenizer, TrainedBpeVocabulary, BpeVocabularyTrainer,
                      _TOKENIZER_NATIVE, _TOKENIZER_FAST, _TOKENIZER_FAST_ENTRIES,
                      vars(_backend), _numpy_free_context(), _gpu_binding())
_TOKENIZER_NATIVE.tokenizer_public_call("connect_bpe", (), _TOKENIZER_CONTEXT)
