# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Tokenize a training corpus once and train a language model on the ids.

The tokenizer is part of the model: it is chosen before the first training
step, and the model's ids mean something only through it. This module
prepares a corpus with a BPE vocabulary, caches the token ids, serves
training batches, and checks that a tokenizer is the one a model was trained
on.

- With no ``vocab``, :func:`prepare` trains a byte-level BPE vocabulary on
  the corpus's training range (never validation or test data), tokenizes the
  whole corpus once and caches the ids.
- With ``vocab=``, your own token map is used and nothing is trained: a
  rank file, an ``(encoder.json, vocab.bpe)`` pair, a
  :class:`~mojolearn.tokenizer.BpeTokenizer` or a ``TrainedBpeVocabulary``.

Cache
-----
Files are written under ``cache_dir`` (default ``$MOJOLEARN_LM_CACHE``, else
``~/.cache/mojolearn/lm``)::

    vocab/<corpus sha16>-v<vocab_size>-f<min_frequency>-s<sample bytes>/
        ranks.tsv, vocabulary.json            (a trained vocabulary)
    vocab/user-<vocabulary sha16>/ranks.tsv, vocabulary.json   (yours)
    tokens/<corpus sha16>-<vocabulary sha16>-d<document bytes>/
        tokens.i32, manifest.json

A rerun with the same corpus and vocabulary reuses the cache after
re-hashing it against its manifest; a mismatch raises ValueError rather than
silently rebuilding. Nothing is written into the package.

Vocabulary identity
-------------------
:meth:`TokenBatches.data_schedule` records the vocabulary's identity (the
SHA-256 of its canonical rank file and its ``n_vocab``). The trainer refuses
a schedule whose ``n_vocab`` differs from its ``vocab_size`` and keeps the
schedule in every checkpoint, and :func:`tokenizer_for` only returns a
tokenizer that matches it, so ids are never decoded with the wrong table.

Documents
---------
BPE never merges across a pre-token boundary, and where a long file is cut
changes the ids at every cut, so the cut is fixed and recorded: a document
never spans the corpus manifest's train / validation / test boundaries, is
at most ``document_bytes`` long, and ends at the last newline at or before
that limit (or exactly at the limit when there is none). Each document is
encoded alone and the ids are concatenated. ``<|endoftext|>`` is neither
inserted nor recognized in the text.

Examples
--------
>>> from mojolearn import lm_corpus, LanguageModelConfig, LanguageModelTrainer
>>> corpus = lm_corpus.prepare("corpus/input.txt")               # train a vocabulary
>>> corpus = lm_corpus.prepare("corpus/input.txt", vocab="ranks.tsv")  # use yours
>>> batches = corpus.batches(batch=1, length=2048)
>>> schedule = batches.data_schedule()
>>> ids = batches.ids(0)
>>> tok = lm_corpus.tokenizer_for(schedule, corpus.vocabulary_path)
"""
# Byte training (ids = raw bytes, vocabulary 256) stays available by explicit
# choice: `tools/lm_train.py --bytes`, whose batches are
# `tools/lm_step_memory_probe.py::CorpusBatches`, unchanged.

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


def default_cache_dir():
    """The default cache directory: ``$MOJOLEARN_LM_CACHE``, else ``~/.cache/mojolearn/lm``."""
    return _NATIVE.lm_corpus_public_call("default_cache_dir", ())


# ---------------------------------------------------------------- the corpus


# The document rule (no document spans a declared range boundary; at most
# `document_bytes` bytes, ending at the last 0x0A at or before the limit) is
# `_cut_documents` in `bindings/_mojolearn_tokenizer_host.mojo`, applied
# inside the one binding call that trains (`bpe_train_corpus`) or encodes
# (`bpe_encode_corpus`); Python never walks the corpus.


# ---------------------------------------------------------------- tokenizing

def _tokenize(data, src, ranges, tok, identity, out, document_bytes, batch_documents, progress=None):
    """Cut, encode and concatenate the corpus in ONE binding call
    (the native tokenizer corpus helper). `batch_documents` is kept for the
    recipe's signature; the binding takes every document at once."""
    return _NATIVE.lm_corpus_public_call("tokenize", (data, src, ranges, tok, identity, out, document_bytes, batch_documents, progress))


class TokenizedCorpus:
    """A prepared corpus: cached token ids and the vocabulary they use.

    Returned by :func:`prepare`.

    Attributes
    ----------
    tokens_dir : pathlib.Path
        Directory holding ``tokens.i32`` (the int32 ids) and ``manifest.json``.
    vocabulary_path : pathlib.Path
        The vocabulary's canonical rank file.
    manifest : dict
        The parsed token manifest.
    """

    def __init__(self, tokens_dir, vocabulary_path):
        return _NATIVE.lm_corpus_public_call("corpus_init", (self, tokens_dir, vocabulary_path))

    @property
    def vocabulary(self):
        """The vocabulary identity record (schema, ``sha256``, ``n_vocab``)."""
        return _NATIVE.lm_corpus_public_call("corpus_vocabulary", (self,))

    @property
    def n_vocab(self):
        """Number of token ids in the vocabulary, ``<|endoftext|>`` included."""
        return _NATIVE.lm_corpus_public_call("corpus_n_vocab", (self,))

    def tokenizer(self):
        """A :class:`~mojolearn.tokenizer.BpeTokenizer` for this corpus's vocabulary."""
        return _NATIVE.lm_corpus_public_call("corpus_tokenizer", (self,))

    def batches(self, batch, length):
        """Training batches over this corpus; see :class:`TokenBatches`."""
        return _NATIVE.lm_corpus_public_call("corpus_batches", (self, batch, length))

    def __repr__(self):
        return _NATIVE.lm_corpus_public_call("corpus_repr", (self,))


# Tokenize `corpus` (a file path) once and return a `TokenizedCorpus`.
#
# `vocab=None` trains a `vocab_size`-rank vocabulary (n_vocab = vocab_size
# + 1 with `<|endoftext|>`) with `BpeVocabularyTrainer` on the first
# `vocab_sample_bytes` of the corpus's TRAIN range. Anything else is the
# caller's own token map (see the module docstring) and nothing is
# trained. Both are cached; a rerun with the same corpus and vocabulary
# reuses the id array after re-hashing it.
def prepare(corpus, *, vocab=None, cache_dir=None, vocab_size=DEFAULT_VOCAB_SIZE,
            min_frequency=DEFAULT_MIN_FREQUENCY, vocab_sample_bytes=DEFAULT_VOCAB_SAMPLE_BYTES,
            document_bytes=DEFAULT_DOCUMENT_BYTES, batch_documents=DEFAULT_BATCH_DOCUMENTS, progress=None):
    """Tokenize a corpus file once and return a :class:`TokenizedCorpus`.

    If a ``manifest.json`` with schema ``mojolearn.byte-lm.corpus.v1`` sits
    beside the corpus, the file is checked against its SHA-256 and length and
    its train / validation / test ranges are used; otherwise the whole file is
    the training range. Results are cached (see the module documentation).

    Parameters
    ----------
    corpus : str or path-like
        Path to the corpus text file.
    vocab : str, path-like, tuple, BpeTokenizer, TrainedBpeVocabulary or None, default None
        None trains a vocabulary of ``vocab_size`` ranks (``n_vocab =
        vocab_size + 1`` with ``<|endoftext|>``) on the first
        ``vocab_sample_bytes`` of the training range. Otherwise your token
        map: a rank file, an ``(encoder.json, vocab.bpe)`` pair, a
        ``BpeTokenizer`` or a ``TrainedBpeVocabulary``; nothing is trained.
        ``"bytes"`` raises ValueError (byte-level training needs no
        vocabulary).
    cache_dir : str, path-like or None, default None
        Cache location; None uses :func:`default_cache_dir`.
    vocab_size : int, default 50256
        Ranks to train (GPT-2's vocabulary size, without ``<|endoftext|>``).
    min_frequency : int, default 2
        Minimum pair frequency for a merge.
    vocab_sample_bytes : int, default 10_000_000
        Bytes of the training range used to train the vocabulary.
    document_bytes : int, default 1048576
        Maximum document length; see the module documentation.
    batch_documents : int, default 16
        Documents tokenized per batch.
    progress : None, default None
        Preparation timings are recorded in the returned manifest. Python
        progress callbacks are not accepted by native preparation.

    Returns
    -------
    TokenizedCorpus

    Raises
    ------
    ValueError
        For a non-positive size argument, a corpus that does not match its
        manifest, or a cache that does not match its manifest.
    """
    return _NATIVE.lm_corpus_public_call("prepare", (corpus, vocab, cache_dir, vocab_size, min_frequency, vocab_sample_bytes, document_bytes, batch_documents, progress))


# ---------------------------------------------------------------- batches

# Token batches from a pinned id array, with `CorpusBatches`' interface
# (`ids(step_index)`, `describe()`), plus `data_schedule()` for the
# trainer.
#
# Step `k` (zero-based), row `b` reads ids
# `[lo + (k*batch*length + b*length) % (hi - lo - length - 1) : + length + 1]`
# where `[lo, hi)` is the TRAIN range in tokens, so a run of any length
# reads no validation or test id. (`CorpusBatches` takes the same form over
# the WHOLE byte file, test range included; it is kept that way so byte
# runs hash as they always have.)
class TokenBatches:
    """Training batches from a tokenized corpus.

    Step ``k`` (zero-based), row ``b`` reads the ``length + 1`` ids starting
    at ``lo + (k * batch * length + b * length) % (hi - lo - length - 1)``,
    where ``[lo, hi)`` is the training range in tokens, so a run of any
    length reads no validation or test id. Targets are the inputs shifted by
    one id. The token file is verified against its manifest when opened.

    Parameters
    ----------
    tokens_dir : str or path-like
        A :class:`TokenizedCorpus` ``tokens_dir``.
    batch : int
        Rows per batch.
    length : int
        Tokens per row (each row holds ``length + 1`` ids).

    Raises
    ------
    ValueError
        If the token file does not match its manifest or the training range
        is shorter than one row.
    """

    def __init__(self, tokens_dir, batch, length):
        return _NATIVE.lm_corpus_public_call("batches_init", (self, tokens_dir, batch, length))

    @property
    def vocabulary(self):
        """The vocabulary identity record (schema, ``sha256``, ``n_vocab``)."""
        return _NATIVE.lm_corpus_public_call("batches_vocabulary", (self,))

    def ids(self, step_index):
        """The batch for training step ``step_index``.

        Returns
        -------
        Array of shape (batch, length + 1), int32
        """
        return _NATIVE.lm_corpus_public_call("batches_ids", (self, step_index))

    # Yield ``(step, ids)`` in order.
    #
    # Each batch is one `tokens_gather_rows` copy out of the already
    # verified mmap, made when the consumer asks for it; there is no host
    # thread (lane/pyglue-text-io, 2026-10-03), so ``depth`` only bounds
    # how many batches may be live and is validated for the old callers.
    # A failure is raised at its exact logical step, before any later
    # batch is made.
    def prefetch(self, start_step, steps, *, depth=2):
        """Iterate over ``(step, ids)`` for ``steps`` steps from ``start_step``.

        Each batch is made when the consumer asks for it, and an error is raised
        at its own step, before any later batch is made.

        Parameters
        ----------
        start_step : int
            First step, at least 0.
        steps : int
            Number of steps, at least 0.
        depth : int, default 2
            Maximum number of batches alive at once; at least 1.

        Returns
        -------
        iterator of (int, Array)
        """
        return _NATIVE.lm_corpus_public_call("batches_prefetch", (self, start_step, steps, depth))

    def _batches(self, start_step, steps):
        return _NATIVE.lm_corpus_public_call("batches_prefetch", (self, start_step, steps, 2))

    def describe(self):
        """A dict describing the token file, its hashes, the training range and the schedule."""
        return _NATIVE.lm_corpus_public_call("batches_describe", (self,))

    def data_schedule(self, **extra):
        """The trainer's ``data_schedule``: which ids, in which order, from which vocabulary.

        Pass it to the language model trainer, which keeps it in every
        checkpoint. Keyword arguments are added to the record.

        Returns
        -------
        dict
        """
        return _NATIVE.lm_corpus_public_call("batches_schedule", (self, extra))


# ---------------------------------------------------------------- identity

def _expected_vocabulary(model):
    """The vocabulary identity a model, checkpoint state, schedule or
    manifest carries, or None when it carries none."""
    return _NATIVE.lm_corpus_public_call("expected_vocabulary", (model,))


def require_vocabulary(model, tokenizer):
    """Check that ``tokenizer`` is the vocabulary ``model`` was trained on.

    Parameters
    ----------
    model : trainer, dict
        A trainer, a checkpoint state, a data schedule or a tokens manifest.
    tokenizer : BpeTokenizer
        The tokenizer to check.

    Returns
    -------
    BpeTokenizer
        ``tokenizer``, unchanged.

    Raises
    ------
    ValueError
        If ``model`` carries no vocabulary identity, or the tokenizer's SHA-256
        or ``n_vocab`` differs from it.
    """
    return _NATIVE.lm_corpus_public_call("require_vocabulary", (model, tokenizer))


def tokenizer_for(model, vocabulary):
    """A tokenizer for ``vocabulary``, checked against the vocabulary ``model`` was trained on.

    Parameters
    ----------
    model : trainer, dict
        A trainer, a checkpoint state, a data schedule or a tokens manifest.
    vocabulary : str, path-like, tuple, BpeTokenizer or TrainedBpeVocabulary
        Any form :func:`prepare` accepts as ``vocab``.

    Returns
    -------
    BpeTokenizer

    Raises
    ------
    ValueError
        If the vocabulary does not match the model's (see
        :func:`require_vocabulary`).
    """
    return _NATIVE.lm_corpus_public_call("tokenizer_for", (model, vocabulary))


class _TokenBatchIterator:
    """Lazy batches whose position and owner are retained by a native handle."""

    def __iter__(self):
        return _NATIVE.lm_corpus_public_call("iterator_iter", (self,))

    def __next__(self):
        return _NATIVE.lm_corpus_public_call("iterator_next", (self,))


# Import-only connection. Public methods select no binding and call no Python
# constructor, filesystem operation, tokenizer helper or scheduling loop.
import mmap as _corpus_mmap
from ._array import Array as _CorpusArray
from ._native_loaded_modules import connect_module as _corpus_connect_module
_corpus_connect_module("mmap", _corpus_mmap, ("mmap",))
_NATIVE = _tokenizer._TOKENIZER_NATIVE
_NATIVE.lm_corpus_public_call("connect", (
    TokenizedCorpus, TokenBatches, _TokenBatchIterator, _CorpusArray, _corpus_mmap.mmap))
