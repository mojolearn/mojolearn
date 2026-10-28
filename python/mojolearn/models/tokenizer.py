# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Load byte-level BPE tokenizers from Hugging Face tokenizer.json files.

Supports GPT-2, Llama 3 and Qwen 2 pre-tokenization patterns, added special
tokens, optional NFC normalization and BOS/EOS templates. Use `encode` and
`decode` for text, or `encode_bytes` and `decode_bytes` for bytes. Vocabulary
files are supplied by the caller.
"""
import unicodedata

from ..tokenizer import _binding

# Module import connects the public package to the compiled extension once.
_NATIVE = _binding()._module

__all__ = ["Tokenizer", "PATTERNS", "pretokenize", "pattern_name"]

_GPT2_TIKTOKEN = "'(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++| ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s"
_GPT2_HF = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
_LLAMA3 = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
_QWEN2 = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

#: pattern name -> the regex spellings that name it (the first is canonical)
PATTERNS = {
    "gpt2": (_GPT2_TIKTOKEN, _GPT2_HF),
    "llama3": (_LLAMA3,),
    "qwen2": (_QWEN2,),
}
_NAME_OF = {spelling: name for name, spellings in PATTERNS.items() for spelling in spellings}


def pattern_name(regex):
    """The name of a known pattern spelling, or None."""
    return _NATIVE.tokenizer_public_call('vocab_pattern', (regex,), _TOKENIZER_CONTEXT)


def _text_bytes(data):
    if isinstance(data, str):
        return data.encode("utf-8")
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    raise TypeError(f"mojolearn.models.Tokenizer: text must be str or bytes-like, got {type(data).__name__}")


def pretokenize(data, pattern):
    """Pre-token boundaries of `data` (bytes) under `pattern` ("gpt2",
    "llama3" or "qwen2"): `m + 1` offsets, first 0 and last `len(data)`.
    The cut is the binding's (`vocab_pretokenize_pattern`)."""
    return _NATIVE.tokenizer_public_call('vocab_cut', (data, pattern), _TOKENIZER_CONTEXT)


# --------------------------------------------------------- the reader


class Tokenizer:
    """See the module header. `tokens` are the rank-ordered token bytes,
    `merges` the `(left, right)` byte pairs, `pattern` a name in
    `PATTERNS`, `added` `{content: id}`."""

    unicode_version = unicodedata.unidata_version

    def __init__(self, tokens, merges, *, pattern, added=None, normalizer=None,
                 bos_ids=(), eos_ids=(), ignore_merges=False, source="<tokens>"):
        return _NATIVE.tokenizer_public_call('vocab_init', (self, tokens, merges, (pattern, added, normalizer, bos_ids, eos_ids, ignore_merges, source)), _TOKENIZER_CONTEXT)

    @classmethod
    def _from_spelled(cls, vocab, merges, *, pattern, added, normalizer, bos_ids, eos_ids, ignore_merges,
                      source):
        return _NATIVE.tokenizer_public_call('vocab_spelled', (cls, vocab, merges, (pattern, added, normalizer, bos_ids, eos_ids, ignore_merges, source)), _TOKENIZER_CONTEXT)

    # ------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, path):
        """`path` is a directory holding `tokenizer.json` (and optionally
        `tokenizer_config.json`), or the `tokenizer.json` itself."""
        return _NATIVE.tokenizer_public_call('vocab_pretrained', (cls, path), _TOKENIZER_CONTEXT)

    @classmethod
    def _from_json(cls, data, path, cfg=None):
        return _NATIVE.tokenizer_public_call('vocab_json', (cls, data, path, cfg), _TOKENIZER_CONTEXT)

    # ------------------------------------------------------------ facts
    @property
    def n_vocab(self):
        """The ranks, extended by the added tokens above them."""
        return _NATIVE.tokenizer_public_call('vocab_fact', (self, 'n_vocab'), _TOKENIZER_CONTEXT)

    @property
    def n_ranks(self):
        return _NATIVE.tokenizer_public_call('vocab_fact', (self, 'n_ranks'), _TOKENIZER_CONTEXT)

    bos_token_id = None
    eos_token_id = None

    # ------------------------------------------------------------ encode
    def pretokenize(self, data):
        """The pre-tokens of `data` (bytes) under this tokenizer's pattern,
        as a list of bytes; the cut before any merge."""
        return _NATIVE.tokenizer_public_call('vocab_pretokenize', (self, data), _TOKENIZER_CONTEXT)

    _bytes = staticmethod(_text_bytes)

    def encode_bytes(self, data, *, allow_special=False):
        """The ids of a byte string. Added tokens are recognized only with
        `allow_special=True`; the segments between them are encoded alone
        (a special token ends the text before it, as the binding's
        `<|endoftext|>` rule has it)."""
        return _NATIVE.tokenizer_public_call('vocab_encode', (self, data, False, allow_special, True), _TOKENIZER_CONTEXT)

    def encode(self, text, *, add_special_tokens=True, allow_special=False):
        """The ids of `text` (str, NFC-normalized when the file says so, then
        UTF-8; or bytes-like), with the template's BOS/EOS ids around them
        when `add_special_tokens` is True."""
        return _NATIVE.tokenizer_public_call('vocab_encode', (self, text, add_special_tokens, allow_special, False), _TOKENIZER_CONTEXT)

    # ------------------------------------------------------------ decode
    def decode_bytes(self, ids):
        """The bytes of an id sequence; an added token renders its content.
        An id outside the vocabulary is refused by value and position."""
        return _NATIVE.tokenizer_public_call('vocab_decode', (self, ids, True, 'replace'), _TOKENIZER_CONTEXT)

    def decode(self, ids, errors="replace"):
        return _NATIVE.tokenizer_public_call('vocab_decode', (self, ids, False, errors), _TOKENIZER_CONTEXT)

    def __repr__(self):
        return _NATIVE.tokenizer_public_call('vocab_fact', (self, 'repr'), _TOKENIZER_CONTEXT)


# Immutable import-time metadata; native execution owns all public operations.
_TOKENIZER_CONTEXT = (Tokenizer, _NATIVE, _NAME_OF, unicodedata.unidata_version,
                      unicodedata.normalize)
_NATIVE.tokenizer_public_call("connect_vocab", (), _TOKENIZER_CONTEXT)
