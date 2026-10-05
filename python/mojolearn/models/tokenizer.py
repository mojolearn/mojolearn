# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`Tokenizer.from_pretrained(path)`: a Hugging Face `tokenizer.json` for the
byte-level BPE families, with the pre-tokenization pattern a PARAMETER
(lane/model-loader, 2026-09-17; DEVIATION 2960).

WHAT THE FILE HOLDS AND WHAT IS TAKEN FROM IT. `model.vocab` (byte-level
spelling -> id) and `model.merges` become a rank table of token bytes, the
same table `BpeTokenizer.from_files` builds from `encoder.json` and
`vocab.bpe`, checked the same way (ids contiguous, every token a single
byte or the result of a merge, merge results rising above both parts, so
that merging by rank IS the merge list). `added_tokens` are the special
tokens; `pre_tokenizer` names the pattern; `normalizer` is null or NFC;
`post_processor` is null, ByteLevel, or a TemplateProcessing whose leading
and trailing special tokens become the BOS/EOS ids `encode` adds.

THE PATTERN IS A PARAMETER (DEVIATION 2960). Byte-level BPE families share
the merge loop and the byte-to-unicode table but not the regex that cuts
text into pre-tokens, and a different cut is a different id sequence:

    gpt2    '(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++| ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s
            (tiktoken's spelling; Hugging Face's `ByteLevel use_regex` spelling
            without possessive quantifiers cuts identically) -- GPT-2, GPT-NeoX,
            Pythia, the Mamba checkpoints
    llama3  (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+
            -- Llama 3.x, SmolLM2 (Llama-3-style digits by threes)
    qwen2   the same with `\\p{N}` (one digit at a time) -- Qwen 2, Qwen 2.5, Qwen 3

WHERE EACH PATTERN RUNS (lane/pyglue-text-io, 2026-10-03). All three
patterns are cut in Mojo, in the tokenizer host binding
(`tokenizer/impl/pretokenize.mojo`: `pretoken_end` for GPT-2,
`pretoken_end_llama` for Llama 3 and Qwen 2, leftmost-first over the seven
alternatives), and every pre-token goes to the binding's merge loop
(`tokenizer/impl/bpe.mojo`); the vocabulary checks, the special-token split
and the decode are `tokenizer/vocab.mojo` and
`tokenizer/impl/vocab_build.mojo`. This module only reads the JSON
configuration, validates it, passes the vocabulary objects to the binding and
returns results. The Python spellings of the cuts that used to live here
are kept as the verification oracle in `_tokenizer_synthetic.py` (GPT-2).

UNICODE CLASSES. `\\p{L}`, `\\p{N}` and `\\s` are the binding's pinned
tables (`tokenizer/impl/unicode_class.mojo`, generated at build time), the
same for every pattern. NFC normalization, when the file asks for it, is
the running Python's `unicodedata.normalize` (`Tokenizer.unicode_version`).

SENTENCEPIECE FAMILIES ARE REFUSED BY NAME. Llama 2, Mistral (v1/v2),
Gemma and Phi-3 ship a SentencePiece model (`tokenizer.model`, or a
`tokenizer.json` with `model.type: Unigram`, `byte_fallback: true`, a
Metaspace pre-tokenizer or `▁` spellings); none of that is byte-level
BPE and this lane does not load it.

SPECIAL TOKENS. Every added token is recognized in the text only with
`allow_special=True` (the existing door's `allow_endoftext` rule); with the
default `False` its characters are ordinary text. `decode` renders an added
token's content. The BOS/EOS the post-processor's template names are added
by `encode(..., add_special_tokens=True)`, the default, as Hugging Face's
`encode` does.
"""
import array
import itertools
import json
import os
import unicodedata

from .._buffer import addr, addr_ro
from ..tokenizer import _binding, _blob, _native, _ro

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
_DIGITS = {"llama3": 3, "qwen2": 1}


def pattern_name(regex):
    """The name of a known pattern spelling, or None."""
    return _NAME_OF.get(regex)


_PATTERN_CODE = {"gpt2": 0, "llama3": 1, "qwen2": 2}


def _text_bytes(data):
    if isinstance(data, str):
        return data.encode("utf-8")
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    raise TypeError(f"mojolearn.models.Tokenizer: text must be str or bytes-like, got {type(data).__name__}")


def pretokenize(data, pattern):
    """Pre-token boundaries of `data` (bytes) under `pattern` ("gpt2",
    "llama3" or "qwen2"): `m + 1` offsets, first 0 and last `len(data)`.
    The cut is the binding's (`tokenizer_pretokenize`)."""
    if pattern not in _PATTERN_CODE:
        raise ValueError(f"mojolearn.models.Tokenizer: unknown pattern {pattern!r}; one of {sorted(PATTERNS)}")
    raw = _text_bytes(data)
    out = array.array("q", bytes(8 * (len(raw) + 1)))
    m = _binding()
    count = int(_native(m.tokenizer_pretokenize, _ro(raw, "text"), len(raw), _PATTERN_CODE[pattern],
                        addr(out, name="bounds"), prefix="mojolearn.models.Tokenizer: "))
    return out[:count].tolist()


# --------------------------------------------------------- the reader

def _refuse(path, what):
    raise ValueError(f"mojolearn.models.Tokenizer: {path}: {what}")


def _pattern_of(pre, path):
    """The pattern name a `pre_tokenizer` entry selects, refusing by name."""
    if pre is None:
        _refuse(path, "pre_tokenizer is null (no cut at all); the byte-level BPE families all cut")
    kind = pre.get("type")
    if kind == "ByteLevel":
        if pre.get("use_regex", True):
            return "gpt2"
        _refuse(path, "pre_tokenizer ByteLevel with use_regex false cuts nothing; not a known family")
    if kind == "Metaspace":
        _refuse(path, "pre_tokenizer Metaspace: a SentencePiece family (Llama 2, Mistral, Gemma, Phi-3), refused by name")
    if kind == "Sequence":
        found = None
        for step in pre.get("pretokenizers", []):  # cpu-route: parses the tokenizer.json pretokenizer config (file input)
            t = step.get("type")
            if t == "Split":
                pat = step.get("pattern", {})
                regex = pat.get("Regex") if isinstance(pat, dict) else None
                if regex is None:
                    _refuse(path, f"pre_tokenizer Split with pattern {pat!r}; only a Regex is a known cut")
                if step.get("behavior") not in ("Isolated", None) or step.get("invert"):
                    _refuse(path, f"pre_tokenizer Split behavior={step.get('behavior')!r} invert={step.get('invert')!r}; only Isolated is a known cut")
                name = pattern_name(regex)
                if name is None:
                    _refuse(path, f"pre_tokenizer regex {regex!r} is not one of the known patterns {sorted(PATTERNS)}; a different cut is a different id sequence, refused by name")
                if found is not None and found != name:
                    _refuse(path, "two Split steps with different patterns")
                found = name
            elif t == "ByteLevel":
                if step.get("use_regex", False):
                    if found is not None and found != "gpt2":
                        _refuse(path, "a Split pattern followed by ByteLevel use_regex true would cut twice")
                    found = found or "gpt2"
            elif t == "Metaspace":
                _refuse(path, "pre_tokenizer Metaspace: a SentencePiece family, refused by name")
            else:
                _refuse(path, f"pre_tokenizer step {t!r} is not a known cut (ByteLevel, Split Regex)")
        if found is None:
            _refuse(path, "pre_tokenizer Sequence names no cut")
        return found
    _refuse(path, f"pre_tokenizer type {kind!r} is not a known cut (ByteLevel, Sequence of Split Regex + ByteLevel)")


class Tokenizer:
    """See the module header. `tokens` are the rank-ordered token bytes,
    `merges` the `(left, right)` byte pairs, `pattern` a name in
    `PATTERNS`, `added` `{content: id}`."""

    unicode_version = unicodedata.unidata_version

    def __init__(self, tokens, merges, *, pattern, added=None, normalizer=None,
                 bos_ids=(), eos_ids=(), ignore_merges=False, source="<tokens>"):
        self._settle(pattern, added, normalizer, bos_ids, eos_ids, source)
        toks = list(tokens)
        pairs = list(merges)
        tblob, tlen = _blob(toks, "every token")
        try:
            if set(map(len, pairs)) - {2}:
                raise TypeError
        except TypeError:
            raise ValueError("mojolearn.models.Tokenizer: every merge must be a pair of bytes") from None
        mparts = list(itertools.chain.from_iterable(pairs))
        mblob, mlen = _blob(mparts, "every merge part")
        sblob, slen, sids = self._special_buffers()
        handle = _native(self._m.vocab_load_bytes,
                         [_ro(tblob, "tokens"), _ro(tlen, "token_lengths"), _ro(mblob, "merges"),
                          _ro(mlen, "merge_lengths"), _ro(sblob, "specials"), _ro(slen, "special_lengths"),
                          _ro(sids, "special_ids")],
                         [len(toks), len(pairs), len(slen), _PATTERN_CODE[pattern], int(bool(ignore_merges))],
                         prefix=f"mojolearn.models.Tokenizer: {source}: ")
        self._adopt(handle)

    def _settle(self, pattern, added, normalizer, bos_ids, eos_ids, source):
        """The configuration: checked here (scalars and the added-token map),
        the vocabulary itself is the binding's."""
        if pattern not in PATTERNS:
            raise ValueError(f"mojolearn.models.Tokenizer: unknown pattern {pattern!r}; one of {sorted(PATTERNS)}")
        if normalizer not in (None, "NFC"):
            raise ValueError(f"mojolearn.models.Tokenizer: normalizer {normalizer!r} is not honored; null or NFC")
        self.pattern = pattern
        self.normalizer = normalizer
        self.source = source
        self.added = dict(added or {})
        for content, i in self.added.items():  # cpu-route: walks the added special tokens (text input)
            if not isinstance(content, str) or not content or type(i) is not int or i < 0:
                raise ValueError(f"mojolearn.models.Tokenizer: added token {content!r} -> {i!r} is not a non-empty string to a non-negative id")
        self._added_by_id = {}
        for content, i in self.added.items():  # cpu-route: walks the added special tokens (text input)
            if i in self._added_by_id:
                raise ValueError(f"mojolearn.models.Tokenizer: added tokens {self._added_by_id[i]!r} and {content!r} share id {i}")
            self._added_by_id[i] = content
        self.bos_ids = tuple(int(i) for i in bos_ids)  # glue: converts the bos id argument
        self.eos_ids = tuple(int(i) for i in eos_ids)  # glue: converts the eos id argument
        self._m = _binding()

    def _special_buffers(self):
        contents = [c.encode("utf-8") for c in self.added]  # cpu-route: encodes the added special tokens (text input)
        blob, lengths = _blob(contents, "every added token")
        return blob, lengths, array.array("q", self.added.values())

    def _adopt(self, handle):
        self._handle = handle
        n_ranks, n_vocab, longest = (int(x) for x in self._m.vocab_info(handle))  # glue: unpacks three vocabulary sizes
        self._n_ranks, self._n_vocab, self._max_token_bytes = n_ranks, n_vocab, longest

    @classmethod
    def _from_spelled(cls, vocab, merges, *, pattern, added, normalizer, bos_ids, eos_ids, ignore_merges,
                      source):
        tok = cls.__new__(cls)
        tok._settle(pattern, added, normalizer, bos_ids, eos_ids, source)
        sblob, slen, sids = tok._special_buffers()
        handle = _native(tok._m.vocab_load_spelled, vocab, merges,
                         [_ro(sblob, "specials"), _ro(slen, "special_lengths"), _ro(sids, "special_ids")],
                         [len(slen), _PATTERN_CODE[pattern], int(bool(ignore_merges))],
                         prefix=f"mojolearn.models.Tokenizer: {source}: ")
        tok._adopt(handle)
        return tok

    # ------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, path):
        """`path` is a directory holding `tokenizer.json` (and optionally
        `tokenizer_config.json`), or the `tokenizer.json` itself."""
        path = os.fspath(path)
        if os.path.isdir(path):
            tj = os.path.join(path, "tokenizer.json")
            if not os.path.isfile(tj):
                if os.path.isfile(os.path.join(path, "tokenizer.model")):
                    raise ValueError(
                        f"mojolearn.models.Tokenizer: {path} holds tokenizer.model and no tokenizer.json: a "
                        "SentencePiece family (Llama 2, Mistral, Gemma, Phi-3), refused by name")
                raise FileNotFoundError(f"mojolearn.models.Tokenizer: {path} holds no tokenizer.json")
            cfg_path = os.path.join(path, "tokenizer_config.json")
        else:
            tj = path
            cfg_path = os.path.join(os.path.dirname(path) or ".", "tokenizer_config.json")
        with open(tj, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        cfg = None
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
        return cls._from_json(data, tj, cfg)

    @classmethod
    def _from_json(cls, data, path, cfg=None):
        if not isinstance(data, dict):
            _refuse(path, "not a JSON object")
        model = data.get("model") or {}
        mtype = model.get("type")
        if mtype != "BPE":
            _refuse(path, f"model.type {mtype!r}" + (": a SentencePiece Unigram model, refused by name" if mtype == "Unigram" else "; only BPE is byte-level BPE"))
        if model.get("byte_fallback"):
            _refuse(path, "model.byte_fallback true: a SentencePiece-style BPE (Llama 2, Mistral), refused by name")
        if model.get("continuing_subword_prefix") or model.get("end_of_word_suffix"):
            _refuse(path, "model.continuing_subword_prefix / end_of_word_suffix: not byte-level BPE")
        dec = data.get("decoder")
        if dec is not None and dec.get("type") != "ByteLevel":
            _refuse(path, f"decoder type {dec.get('type')!r}; ByteLevel is the byte-level BPE decoder")
        norm = data.get("normalizer")
        normalizer = None
        if norm is not None:
            if norm.get("type") == "NFC":
                normalizer = "NFC"
            elif norm.get("type") == "Sequence" and all(s.get("type") == "NFC" for s in norm.get("normalizers", [])) and norm.get("normalizers"):  # cpu-route: parses the tokenizer.json normalizer config (file input)
                normalizer = "NFC"
            else:
                _refuse(path, f"normalizer {norm.get('type')!r} is not honored; null or NFC")
        pattern = _pattern_of(data.get("pre_tokenizer"), path)
        # added tokens
        added = {}
        for entry in data.get("added_tokens", []) or []:  # cpu-route: parses the tokenizer.json added tokens (file input)
            content, i = entry.get("content"), entry.get("id")
            if not isinstance(content, str) or type(i) is not int:
                _refuse(path, f"added token {entry!r} lacks a string content and an int id")
            added[content] = i
        # the vocabulary
        vocab = model.get("vocab")
        if not isinstance(vocab, dict):
            _refuse(path, "model.vocab is not an object")
        # The vocabulary without the added tokens; the binding decodes the
        # spellings, places the ids and checks the merge list.
        base = dict(vocab)
        for content in added:  # cpu-route: parses the tokenizer.json added tokens (file input)
            base.pop(content, None)
        merges = model.get("merges", []) or []
        if not isinstance(merges, list):
            _refuse(path, "model.merges is not a list")
        # the template's specials
        bos_ids, eos_ids = [], []
        post = data.get("post_processor")
        if post is not None:
            kind = post.get("type")
            if kind == "TemplateProcessing":
                single = post.get("single", [])
                specials = post.get("special_tokens", {})
                seen_seq = False
                for item in single:  # cpu-route: parses the tokenizer.json post processor (file input)
                    if "Sequence" in item:
                        seen_seq = True
                    elif "SpecialToken" in item:
                        name = item["SpecialToken"].get("id")
                        ids = specials.get(name, {}).get("ids")
                        if not ids:
                            _refuse(path, f"post_processor names special token {name!r} with no ids")
                        (eos_ids if seen_seq else bos_ids).extend(int(v) for v in ids)  # cpu-route: parses the tokenizer.json post processor ids (file input)
                    else:
                        _refuse(path, f"post_processor template item {item!r} is not Sequence or SpecialToken")
            elif kind not in ("ByteLevel",):
                _refuse(path, f"post_processor type {kind!r} is not honored (null, ByteLevel, TemplateProcessing)")
        if cfg is not None and isinstance(cfg, dict):
            if cfg.get("add_bos_token") is False:
                bos_ids = []
            if cfg.get("add_eos_token") is False:
                eos_ids = []
        tok = cls._from_spelled(base, merges, pattern=pattern, added=added, normalizer=normalizer,
                                bos_ids=bos_ids, eos_ids=eos_ids,
                                ignore_merges=bool(model.get("ignore_merges", False)),
                                source=os.path.abspath(path))
        tok.bos_token_id = tok.bos_ids[0] if tok.bos_ids else None
        tok.eos_token_id = tok.eos_ids[-1] if tok.eos_ids else None
        if cfg is not None and isinstance(cfg, dict):
            for key, attr in (("bos_token", "bos_token_id"), ("eos_token", "eos_token_id")):  # cpu-route: parses the tokenizer config special tokens (file input)
                v = cfg.get(key)
                if isinstance(v, dict):
                    v = v.get("content")
                if isinstance(v, str) and v in added:
                    setattr(tok, attr, added[v])
        return tok

    # ------------------------------------------------------------ facts
    @property
    def n_vocab(self):
        """The ranks, extended by the added tokens above them."""
        return self._n_vocab

    @property
    def n_ranks(self):
        return self._n_ranks

    bos_token_id = None
    eos_token_id = None

    # ------------------------------------------------------------ encode
    def pretokenize(self, data):
        """The pre-tokens of `data` (bytes) under this tokenizer's pattern,
        as a list of bytes; the cut before any merge."""
        raw = _text_bytes(data)
        out = array.array("q", bytes(8 * (len(raw) + 1)))
        count = int(self._m.vocab_pretokenize(self._handle, _ro(raw, "text"), len(raw), addr(out, name="bounds")))
        return [raw[a:b] for a, b in zip(out[:count - 1], out[1:count])]  # cpu-route: slices the text into pretokens at native bounds (text input)

    _bytes = staticmethod(_text_bytes)

    def encode_bytes(self, data, *, allow_special=False):
        """The ids of a byte string. Added tokens are recognized only with
        `allow_special=True`; the segments between them are encoded alone
        (a special token ends the text before it, as the binding's
        `<|endoftext|>` rule has it)."""
        if isinstance(data, str):
            raise TypeError("mojolearn.models.Tokenizer: encode_bytes takes bytes; use encode for str")
        raw = _text_bytes(data)
        if not raw:
            return []
        out = array.array("i", bytes(4 * len(raw)))
        count = int(_native(self._m.vocab_encode, self._handle, addr_ro(raw, name="text"), len(raw),
                            addr(out, name="ids"), len(raw), bool(allow_special),
                            prefix="mojolearn.models.Tokenizer: "))
        return out[:count].tolist()

    def encode(self, text, *, add_special_tokens=True, allow_special=False):
        """The ids of `text` (str, NFC-normalized when the file says so, then
        UTF-8; or bytes-like), with the template's BOS/EOS ids around them
        when `add_special_tokens` is True."""
        if isinstance(text, str) and self.normalizer == "NFC":
            text = unicodedata.normalize("NFC", text)
        ids = self.encode_bytes(_text_bytes(text), allow_special=allow_special)
        if add_special_tokens:
            return list(self.bos_ids) + ids + list(self.eos_ids)
        return ids

    # ------------------------------------------------------------ decode
    def decode_bytes(self, ids):
        """The bytes of an id sequence; an added token renders its content.
        An id outside the vocabulary is refused by value and position."""
        if isinstance(ids, (str, bytes, bytearray)):
            raise TypeError(f"mojolearn.models.Tokenizer: ids must be a sequence of int, got {type(ids).__name__}")
        seq = list(ids)
        if not seq:
            return b""
        cap = len(seq) * self._max_token_bytes
        out = bytearray(cap)
        count = int(_native(self._m.vocab_decode, self._handle, seq, addr(out, name="text"), cap,
                            prefix="mojolearn.models.Tokenizer: "))
        return bytes(out[:count])

    def decode(self, ids, errors="replace"):
        return self.decode_bytes(ids).decode("utf-8", errors)

    def __repr__(self):
        return (f"Tokenizer(pattern={self.pattern!r}, n_ranks={self._n_ranks}, added={len(self.added)}, "
                f"source={self.source!r})")
