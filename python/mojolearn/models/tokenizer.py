# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`Tokenizer.from_pretrained(path)`: a Hugging Face `tokenizer.json` for the
byte-level BPE families, with the pre-tokenization pattern a PARAMETER
(lane/model-loader, 2026-09-17; DEVIATION 2960).

WHAT THE FILE HOLDS AND WHAT IS TAKEN FROM IT. `model.vocab` (byte-level
spelling -> id) and `model.merges` become a rank table of token bytes, the
same table `GPT2Tokenizer.from_files` builds from `encoder.json` and
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

WHERE EACH PATTERN RUNS, AND WHY. The GPT-2 pattern is compiled into the
host binding (`tokenizer/impl/pretokenize.mojo`, hand-rolled; Mojo has no
regex engine) and the `gpt2` tokenizer here encodes through that binding via
`GPT2Tokenizer`, the existing certified door. The Llama 3 and Qwen 2
patterns are NOT in the binding: adding a second hand-rolled pattern is a
Mojo change this lane cannot build or gate (no binding build and no run on
this box), and the binding's `encode` always pre-tokenizes with the GPT-2
pattern first, so handing it Llama-3 pieces would cut them again (`'S`,
`,\\n`, `-hello` are one Llama-3 piece and two or three GPT-2 pieces). So
for `llama3` and `qwen2` the pre-tokenization is implemented HERE, in
Python over the byte codes (`_pretoken_end_llama`, leftmost-first over the
seven alternatives exactly as `pretokenize.mojo` does for GPT-2's), and each
piece goes to the EXISTING Python BPE merge, `_tokenizer_synthetic._bpe`,
the second implementation `pixi run check-tokenizer` holds the Mojo merge
to. It is a per-pre-token Python loop; a prompt is short and this lane
makes no speed claim. Porting the two patterns into `pretokenize.mojo`
beside `pretoken_end` (the "new function with its own cases" the
`NOT_IMPLEMENTED.tsv` row names) is owed to a lane that can build the
binding, and the Python spelling here becomes its oracle.

UNICODE CLASSES. `\\p{L}` and `\\p{N}` are `unicodedata.category`'s L* and
N*, `\\s` the White_Space property (`_tokenizer_synthetic.WHITE_SPACE`, the
list Oniguruma and the Rust `regex` crate both use), the case-insensitive
contractions compare `str.casefold()` of one codepoint. The binding pins a
Unicode version for its tables; this module uses the running Python's
`unicodedata` version, so a codepoint whose category changed between the
two could cut differently. `unicodedata.unidata_version` is exposed as
`Tokenizer.unicode_version` so a record can name it.

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
import json
import os
import unicodedata

from .. import _tokenizer_synthetic as _syn
from ..tokenizer import GPT2Tokenizer, _byte_to_char

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


# ------------------------------------------------------ the Llama-3 cut

def _cp(data, i):
    return _syn._decode_cp(data, i)


def _is_letter(cp):
    return cp >= 0 and unicodedata.category(chr(cp)).startswith("L")


def _is_number(cp):
    return cp >= 0 and unicodedata.category(chr(cp)).startswith("N")


def _is_space(cp):
    return cp >= 0 and _syn._is_space(cp)


def _is_crlf(cp):
    return cp == 0x0D or cp == 0x0A


def _is_other(cp):
    return cp >= 0 and not (_is_space(cp) or _is_letter(cp) or _is_number(cp))


def _run_end(data, start, pred):
    j, n = start, len(data)
    while j < n:
        cp, w = _cp(data, j)
        if cp < 0 or not pred(cp):
            break
        j += w
    return j


def _fold(cp):
    """One codepoint's single-character case fold, or None."""
    if cp < 0:
        return None
    f = chr(cp).casefold()
    return f if len(f) == 1 else None


def _contraction_end_ci(data, i):
    """`(?i:'s|'t|'re|'ve|'m|'ll|'d)` at `i`, or -1."""
    n = len(data)
    if data[i] != 0x27 or i + 1 >= n:
        return -1
    cp1, w1 = _cp(data, i + 1)
    f1 = _fold(cp1)
    if f1 in ("s", "t", "m", "d"):
        return i + 1 + w1
    if f1 in ("r", "v", "l") and i + 1 + w1 < n:
        cp2, w2 = _cp(data, i + 1 + w1)
        f2 = _fold(cp2)
        if (f1, f2) in (("r", "e"), ("v", "e"), ("l", "l")):
            return i + 1 + w1 + w2
    return -1


def _pretoken_end_llama(data, i, max_digits):
    """End of the pre-token at byte `i` under the Llama 3 / Qwen 2 pattern:
    the seven alternatives tried in order, first match wins (the engines
    that compiled these patterns backtrack, so leftmost-first). Always
    > `i`."""
    n = len(data)
    e = _contraction_end_ci(data, i)
    if e > 0:
        return e
    cp0, w0 = _cp(data, i)
    # 2.  [^\r\n\p{L}\p{N}]?\p{L}+
    if _is_letter(cp0):
        return _run_end(data, i, _is_letter)
    if cp0 >= 0 and not _is_crlf(cp0) and not _is_number(cp0) and i + w0 < n:
        cp1, _ = _cp(data, i + w0)
        if _is_letter(cp1):
            return _run_end(data, i + w0, _is_letter)
    # 3.  \p{N}{1,3}  (llama3)  |  \p{N}  (qwen2)
    if _is_number(cp0):
        j, k = i, 0
        while j < n and k < max_digits:
            cp, w = _cp(data, j)
            if not _is_number(cp):
                break
            j += w
            k += 1
        return j
    # 4.   ?[^\s\p{L}\p{N}]+[\r\n]*
    start = i
    if cp0 == 0x20 and i + 1 < n:
        cp1, _ = _cp(data, i + 1)
        if _is_other(cp1):
            start = i + 1
    if start > i or _is_other(cp0):
        j = _run_end(data, start, _is_other)
        if j > start:
            while j < n and (data[j] == 0x0D or data[j] == 0x0A):
                j += 1
            return j
    # 5.  \s*[\r\n]+  -- the run up to and including its last CR or LF
    ws_end = _run_end(data, i, _is_space)
    if ws_end > i:
        last_crlf, j = -1, i
        while j < ws_end:
            cp, w = _cp(data, j)
            if _is_crlf(cp):
                last_crlf = j + w
            j += w
        if last_crlf > 0:
            return last_crlf
        # 6.  \s+(?!\S)  -- the whole run at end of text, else the run minus
        #     its last codepoint when it has two or more
        if ws_end == n:
            return ws_end
        last, j = i, i
        while j < ws_end:
            last = j
            j += _cp(data, j)[1]
        if last > i:
            return last
        # 7.  \s+  -- the single whitespace codepoint before a non-space
        return ws_end
    # an invalid UTF-8 lead byte is its own one-byte pre-token (the binding's rule)
    return i + 1


def pretokenize(data, pattern):
    """Pre-token boundaries of `data` (bytes) under `pattern` ("gpt2",
    "llama3" or "qwen2"): `m + 1` offsets, first 0 and last `len(data)`."""
    if pattern == "gpt2":
        return _syn.pretokenize(data)
    if pattern not in _DIGITS:
        raise ValueError(f"mojolearn.models.Tokenizer: unknown pattern {pattern!r}; one of {sorted(PATTERNS)}")
    digits = _DIGITS[pattern]
    bounds, i, n = [0], 0, len(data)
    while i < n:
        e = _pretoken_end_llama(data, i, digits)
        if e <= i or e > n:
            raise RuntimeError(f"mojolearn.models.Tokenizer: the cut returned {e} at {i}; it must advance and stay in bounds")
        bounds.append(e)
        i = e
    return bounds


# --------------------------------------------------------- the reader

def _refuse(path, what):
    raise ValueError(f"mojolearn.models.Tokenizer: {path}: {what}")


def _validate(tokens, merges, path, ignore_merges):
    """`GPT2Tokenizer.from_files`'s checks over token bytes: the conditions
    under which merging by rank gives the merge list's own result."""
    n = len(tokens)
    index = {}
    for i, t in enumerate(tokens):
        if not t:
            _refuse(path, f"token {i} is empty")
        if t in index:
            _refuse(path, f"token {i} repeats token {index[t]} ({t.hex()})")
        index[t] = i
    missing = [b for b in range(256) if bytes([b]) not in index]
    if missing:
        _refuse(path, f"the vocabulary lacks {len(missing)} of the 256 single-byte tokens (first 0x{missing[0]:02x}); byte-level BPE needs every byte")
    made, last = set(), -1
    for k, (a, b) in enumerate(merges):
        if a not in index or b not in index:
            _refuse(path, f"merge {k} joins {a!r} and {b!r}, not both tokens")
        m = index.get(a + b)
        if m is None:
            _refuse(path, f"merge {k}: the result {(a + b)!r} is not a token")
        if m <= last or m <= index[a] or m <= index[b]:
            _refuse(path, f"merge {k}: result id {m} does not rise above the previous merge ({last}) and both parts; merging by rank would not follow this merge list")
        last = m
        made.add(m)
    loose = [i for i in range(n) if i not in made and len(tokens[i]) != 1]
    if loose and not ignore_merges:
        _refuse(path, f"id {loose[0]} is neither a single byte nor made by a merge (and model.ignore_merges is false, so no whole-word lookup could reach it)")
    return index


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
        for step in pre.get("pretokenizers", []):
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
        if pattern not in PATTERNS:
            raise ValueError(f"mojolearn.models.Tokenizer: unknown pattern {pattern!r}; one of {sorted(PATTERNS)}")
        if normalizer not in (None, "NFC"):
            raise ValueError(f"mojolearn.models.Tokenizer: normalizer {normalizer!r} is not honored; null or NFC")
        self.tokens = [bytes(t) for t in tokens]
        self.merges = [(bytes(a), bytes(b)) for a, b in merges]
        self.pattern = pattern
        self.normalizer = normalizer
        self.source = source
        self._index = _validate(self.tokens, self.merges, source, ignore_merges)
        self.added = dict(added or {})
        for content, i in self.added.items():
            if not isinstance(content, str) or not content or type(i) is not int or i < 0:
                raise ValueError(f"mojolearn.models.Tokenizer: added token {content!r} -> {i!r} is not a non-empty string to a non-negative id")
        self._added_by_id = {}
        for content, i in self.added.items():
            if i in self._added_by_id:
                raise ValueError(f"mojolearn.models.Tokenizer: added tokens {self._added_by_id[i]!r} and {content!r} share id {i}")
            self._added_by_id[i] = content
        self._specials = sorted((c.encode("utf-8"), i) for c, i in self.added.items())
        self._specials.sort(key=lambda p: -len(p[0]))  # longest first at a tie in position
        self.bos_ids = tuple(int(i) for i in bos_ids)
        self.eos_ids = tuple(int(i) for i in eos_ids)
        self._gpt2 = None

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
            elif norm.get("type") == "Sequence" and all(s.get("type") == "NFC" for s in norm.get("normalizers", [])) and norm.get("normalizers"):
                normalizer = "NFC"
            else:
                _refuse(path, f"normalizer {norm.get('type')!r} is not honored; null or NFC")
        pattern = _pattern_of(data.get("pre_tokenizer"), path)
        # added tokens
        added = {}
        for entry in data.get("added_tokens", []) or []:
            content, i = entry.get("content"), entry.get("id")
            if not isinstance(content, str) or type(i) is not int:
                _refuse(path, f"added token {entry!r} lacks a string content and an int id")
            added[content] = i
        # the vocabulary
        vocab = model.get("vocab")
        if not isinstance(vocab, dict):
            _refuse(path, "model.vocab is not an object")
        byte_of = {c: b for b, c in _byte_to_char().items()}
        base = {s: i for s, i in vocab.items() if s not in added}
        n = len(base)
        tokens = [None] * n
        for spelling, i in base.items():
            if type(i) is not int or not 0 <= i < n or tokens[i] is not None:
                _refuse(path, f"id {i!r} of {spelling!r} is not a unique id in [0, {n}) once the added tokens are set aside")
            try:
                tokens[i] = bytes(byte_of[c] for c in spelling)
            except KeyError as exc:
                bad = exc.args[0]
                why = "a SentencePiece spelling (▁), refused by name" if bad == "▁" else "not a byte-level spelling"
                _refuse(path, f"{spelling!r} holds {bad!r}, {why}")
        spelled = {}
        merges = []
        for k, m in enumerate(model.get("merges", []) or []):
            if isinstance(m, str):
                parts = m.split(" ")
            elif isinstance(m, (list, tuple)):
                parts = list(m)
            else:
                parts = []
            if len(parts) != 2:
                _refuse(path, f"merge {k} {m!r} is not a pair")
            pair = []
            for s in parts:
                if s not in spelled:
                    try:
                        spelled[s] = bytes(byte_of[c] for c in s)
                    except KeyError:
                        _refuse(path, f"merge {k} names {s!r}, not a byte-level spelling")
                pair.append(spelled[s])
            merges.append((pair[0], pair[1]))
        # the template's specials
        bos_ids, eos_ids = [], []
        post = data.get("post_processor")
        if post is not None:
            kind = post.get("type")
            if kind == "TemplateProcessing":
                single = post.get("single", [])
                specials = post.get("special_tokens", {})
                seen_seq = False
                for item in single:
                    if "Sequence" in item:
                        seen_seq = True
                    elif "SpecialToken" in item:
                        name = item["SpecialToken"].get("id")
                        ids = specials.get(name, {}).get("ids")
                        if not ids:
                            _refuse(path, f"post_processor names special token {name!r} with no ids")
                        (eos_ids if seen_seq else bos_ids).extend(int(v) for v in ids)
                    else:
                        _refuse(path, f"post_processor template item {item!r} is not Sequence or SpecialToken")
            elif kind not in ("ByteLevel",):
                _refuse(path, f"post_processor type {kind!r} is not honored (null, ByteLevel, TemplateProcessing)")
        if cfg is not None and isinstance(cfg, dict):
            if cfg.get("add_bos_token") is False:
                bos_ids = []
            if cfg.get("add_eos_token") is False:
                eos_ids = []
        tok = cls(tokens, merges, pattern=pattern, added=added, normalizer=normalizer,
                  bos_ids=bos_ids, eos_ids=eos_ids, ignore_merges=bool(model.get("ignore_merges", False)),
                  source=os.path.abspath(path))
        tok.bos_token_id = tok.bos_ids[0] if tok.bos_ids else None
        tok.eos_token_id = tok.eos_ids[-1] if tok.eos_ids else None
        if cfg is not None and isinstance(cfg, dict):
            for key, attr in (("bos_token", "bos_token_id"), ("eos_token", "eos_token_id")):
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
        top = max(self._added_by_id) + 1 if self._added_by_id else 0
        return max(len(self.tokens), top)

    @property
    def n_ranks(self):
        return len(self.tokens)

    bos_token_id = None
    eos_token_id = None

    # ------------------------------------------------------------ encode
    def pretokenize(self, data):
        """The pre-tokens of `data` (bytes) under this tokenizer's pattern,
        as a list of bytes; the cut before any merge."""
        raw = self._bytes(data)
        bounds = pretokenize(raw, self.pattern)
        return [raw[a:b] for a, b in zip(bounds, bounds[1:])]

    @staticmethod
    def _bytes(data):
        if isinstance(data, str):
            return data.encode("utf-8")
        if isinstance(data, (bytes, bytearray, memoryview)):
            return bytes(data)
        raise TypeError(f"mojolearn.models.Tokenizer: text must be str or bytes-like, got {type(data).__name__}")

    def _gpt2_door(self):
        if self._gpt2 is None:
            self._gpt2 = GPT2Tokenizer.from_token_bytes(self.tokens)
        return self._gpt2

    def _split_specials(self, raw):
        """`[(segment_bytes, None) | (b"", id)]` in order."""
        out, i, n = [], 0, len(raw)
        while i < n:
            best_at, best = n, None
            for content, sid in self._specials:
                at = raw.find(content, i, best_at + len(content))
                if at != -1 and (at < best_at or (at == best_at and best is not None and len(content) > len(best[0]))):
                    best_at, best = at, (content, sid)
            if best is None:
                out.append((raw[i:], None))
                break
            if best_at > i:
                out.append((raw[i:best_at], None))
            out.append((b"", best[1]))
            i = best_at + len(best[0])
        return out

    def _encode_ordinary(self, seg, out):
        if not seg:
            return
        if self.pattern == "gpt2":
            out.extend(self._gpt2_door().encode_bytes(seg, False))
            return
        bounds = pretokenize(seg, self.pattern)
        for a, b in zip(bounds, bounds[1:]):
            _syn._bpe(self._index, seg, a, b, out)

    def encode_bytes(self, data, *, allow_special=False):
        """The ids of a byte string. Added tokens are recognized only with
        `allow_special=True`; the segments between them are encoded alone
        (a special token ends the text before it, as the binding's
        `<|endoftext|>` rule has it)."""
        if isinstance(data, str):
            raise TypeError("mojolearn.models.Tokenizer: encode_bytes takes bytes; use encode for str")
        raw = self._bytes(data)
        out = []
        if allow_special and self._specials:
            for seg, sid in self._split_specials(raw):
                if sid is not None:
                    out.append(sid)
                else:
                    self._encode_ordinary(seg, out)
        else:
            self._encode_ordinary(raw, out)
        return out

    def encode(self, text, *, add_special_tokens=True, allow_special=False):
        """The ids of `text` (str, NFC-normalized when the file says so, then
        UTF-8; or bytes-like), with the template's BOS/EOS ids around them
        when `add_special_tokens` is True."""
        if isinstance(text, str) and self.normalizer == "NFC":
            text = unicodedata.normalize("NFC", text)
        ids = self.encode_bytes(self._bytes(text), allow_special=allow_special)
        if add_special_tokens:
            return list(self.bos_ids) + ids + list(self.eos_ids)
        return ids

    # ------------------------------------------------------------ decode
    def decode_bytes(self, ids):
        """The bytes of an id sequence; an added token renders its content.
        An id outside the vocabulary is refused by value and position."""
        out = []
        for k, v in enumerate(ids):
            if isinstance(v, bool) or not hasattr(v, "__index__"):
                raise TypeError(f"mojolearn.models.Tokenizer: ids must be int, got {type(v).__name__} at position {k}")
            i = v.__index__()
            if i in self._added_by_id:
                out.append(self._added_by_id[i].encode("utf-8"))
            elif 0 <= i < len(self.tokens):
                out.append(self.tokens[i])
            else:
                raise ValueError(f"mojolearn.models.Tokenizer: id {i} at position {k} is outside [0, {self.n_vocab}) and no added token")
        return b"".join(out)

    def decode(self, ids, errors="replace"):
        return self.decode_bytes(ids).decode("utf-8", errors)

    def __repr__(self):
        return (f"Tokenizer(pattern={self.pattern!r}, n_ranks={len(self.tokens)}, added={len(self.added)}, "
                f"source={self.source!r})")
