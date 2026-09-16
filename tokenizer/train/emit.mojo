# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE TWO OUTPUT FORMATS a trained vocabulary is written in.

    render_ranks            OURS: `rank<TAB>hex_of_token_bytes`, the file
                            `impl/ranks.mojo` loads and
                            `GPT2Tokenizer.from_ranks_file` reads.
    render_tokenizer_json   `tokenizer.json`, which Hugging Face `tokenizers`
                            loads, so a model published with one of our
                            vocabularies is usable by people who do not use
                            mojolearn.

WHY THE JSON IS HAND-ROLLED AND THE LAYOUT IS FIXED. This file and
`python/mojolearn/_bpe_trainer.py` must produce the SAME BYTES -- that
equality is what `tokenizer/checks/trainer_check.mojo` asserts and the whole
reason there are two implementations. Two JSON libraries agreeing on
formatting is not something to rely on, and Mojo has no JSON writer anyway,
so both sides spell the layout out: one entry per line, no indentation, and
one stated escaping rule.

THE ESCAPING RULE, in full, because both sides implement it:
    `"`   -> `\\"`        `\\`  -> `\\\\`
    a codepoint at or below 0x7E -> itself
    anything above                -> `\\uXXXX`, LOWERCASE hex
A control character cannot occur in a token: the byte-to-unicode bijection
maps every one of the 256 bytes into U+0021..U+00FF or U+0100..U+0143, none
of which is a control. One arriving anyway is a table bug, and `json_ascii`
refuses it by name rather than emitting something a reader would accept.

WHY THE PRE-TOKENIZER IS `Split` AND NOT PLAIN `ByteLevel`. A `tokenizer.json`
can name Hugging Face's own `ByteLevel(use_regex=true)`, which runs ITS GPT-2
regex. Ours is the possessive spelling with `\\s++$` read as END OF HAYSTACK,
and those two readings could disagree on text ending in a newline. Both were
measured to reproduce our splits on every sample tried
(`tools/bpe_trainer_interop.py`), but the `Split` form CARRIES our pattern in
the file instead of depending on that agreement holding in a future release.
The file states the pattern; it does not hope for it.
"""

from tokenizer.impl.byte_unicode import byte_to_codepoint
from tokenizer.train.bpe_train import TrainedVocabulary

comptime HEX_DIGITS = "0123456789abcdef"


def _ascii_table() -> String:
    """The printable ASCII range 0x20..0x7E as a String, built rather than
    typed so a transcription slip is impossible."""
    var b = List[UInt8]()
    for c in range(0x20, 0x7F):
        b.append(UInt8(c))
    var s = String(StringSlice(unsafe_from_utf8=Span(b)))
    # `[[mojo-buffer-freed-at-last-use]]`: the slice views `b`.
    _ = b
    return s^


def _hex_byte(mut out: String, b: Int):
    var d = String(HEX_DIGITS)
    out += String(d[byte = (b >> 4) & 15])
    out += String(d[byte = b & 15])


def _escape_unicode(mut out: String, cp: Int):
    """`\\uXXXX`, lowercase hex, the spelling the Python reference writes."""
    var d = String(HEX_DIGITS)
    out += "\\u"
    out += String(d[byte = (cp >> 12) & 15])
    out += String(d[byte = (cp >> 8) & 15])
    out += String(d[byte = (cp >> 4) & 15])
    out += String(d[byte = cp & 15])


def render_ranks(vocab: TrainedVocabulary) -> String:
    """OUR format: one `rank<TAB>hex` line per token, ascending from 0."""
    var s = String("")
    for id in range(vocab.n_tokens()):
        s += String(id)
        s += "\t"
        var at = vocab.offset[id]
        for k in range(vocab.length[id]):
            _hex_byte(s, Int(vocab.arena[at + k]))
        s += "\n"
    return s^


def json_ascii(text: String) raises -> String:
    """One JSON string literal from ASCII text (the pattern, the special
    token). Refuses a control character or a non-ASCII byte by name: both
    would need an escaping decision this function is not the place to make.
    """
    var ascii = _ascii_table()
    var b = text.as_bytes()
    var out = String('"')
    for i in range(len(b)):
        var c = Int(b[i])
        if c == 0x22:
            out += '\\"'
        elif c == 0x5C:
            out += "\\\\"
        elif c < 0x20:
            raise Error(
                "json_ascii: refusing to escape control character "
                + String(c)
                + " in "
                + text
            )
        elif c <= 0x7E:
            out += String(ascii[byte = c - 0x20])
        else:
            raise Error(
                "json_ascii: byte " + String(c) + " is not ASCII in " + text
            )
    out += '"'
    # `[[mojo-buffer-freed-at-last-use]]`: `b` views `text`.
    _ = text
    return out^


def json_token(
    fwd: List[Int], data: List[UInt8], start: Int, count: Int
) -> String:
    """One token's bytes as a JSON string in the format's printable
    spelling. Goes byte -> codepoint -> escape directly, never through a
    UTF-8 round trip, so the escaping rule is applied to the codepoint the
    bijection names rather than to whatever encoding it landed in."""
    var ascii = _ascii_table()
    var out = String('"')
    for i in range(start, start + count):
        var cp = fwd[Int(data[i])]
        if cp == 0x22:
            out += '\\"'
        elif cp == 0x5C:
            out += "\\\\"
        elif cp <= 0x7E:
            out += String(ascii[byte = cp - 0x20])
        else:
            _escape_unicode(out, cp)
    out += '"'
    return out^


def render_tokenizer_json(
    vocab: TrainedVocabulary, pat_str: String, endoftext: String
) raises -> String:
    """A `tokenizer.json` Hugging Face `tokenizers` loads.

    `endoftext` is an added SPECIAL token at id `n_tokens`, the place the
    GPT-2 format puts it and the place `encoding.mojo::eot_id()` puts it. A
    special token in `added_tokens` is always split out by Hugging Face,
    which corresponds to our `allow_endoftext=True` reading; the
    `allow_endoftext=False` reading, where the thirteen characters are
    ordinary text, has no `tokenizer.json` spelling and
    `tools/bpe_trainer_interop.py` says so rather than papering over it.
    """
    var fwd = byte_to_codepoint()
    var eot_id = vocab.n_tokens()
    var s = String("{\n")
    s += '"version":"1.0",\n'
    s += '"truncation":null,\n'
    s += '"padding":null,\n'
    s += '"added_tokens":[\n'
    s += '{"id":'
    s += String(eot_id)
    s += ',"content":'
    s += json_ascii(endoftext)
    s += ',"single_word":false,"lstrip":false,"rstrip":false,'
    s += '"normalized":false,"special":true}\n'
    s += "],\n"
    s += '"normalizer":null,\n'
    s += '"pre_tokenizer":{"type":"Sequence","pretokenizers":[\n'
    s += '{"type":"Split","pattern":{"Regex":'
    s += json_ascii(pat_str)
    s += '},"behavior":"Isolated","invert":false},\n'
    s += '{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}\n'
    s += "]},\n"
    s += '"post_processor":null,\n'
    s += '"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,'
    s += '"use_regex":false},\n'
    s += '"model":{\n'
    s += '"type":"BPE",\n'
    s += '"dropout":null,\n'
    s += '"unk_token":null,\n'
    s += '"continuing_subword_prefix":null,\n'
    s += '"end_of_word_suffix":null,\n'
    s += '"fuse_unk":false,\n'
    s += '"byte_fallback":false,\n'
    s += '"ignore_merges":false,\n'
    s += '"vocab":{\n'
    var n = vocab.n_tokens()
    for id in range(n):
        s += json_token(fwd, vocab.arena, vocab.offset[id], vocab.length[id])
        s += ":"
        s += String(id)
        if id + 1 < n:
            s += ","
        s += "\n"
    s += "},\n"
    s += '"merges":[\n'
    var m = vocab.n_merges()
    for k in range(m):
        var a = vocab.merge_left[k]
        var b = vocab.merge_right[k]
        s += "["
        s += json_token(fwd, vocab.arena, vocab.offset[a], vocab.length[a])
        s += ","
        s += json_token(fwd, vocab.arena, vocab.offset[b], vocab.length[b])
        s += "]"
        if k + 1 < m:
            s += ","
        s += "\n"
    s += "]\n"
    s += "}\n"
    s += "}\n"
    return s^


def write_text(path: String, text: String) raises:
    with open(path, "w") as f:
        f.write(text)
