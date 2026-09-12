# GPT-2 byte-level BPE tokenizer

`encode` and `decode` for tiktoken 0.14.0's `gpt2` encoding, in Mojo, gated
against 43 cases recorded from tiktoken itself.

```mojo
from tokenizer.encoding import load_gpt2_tokenizer

var tok = load_gpt2_tokenizer()
var ids = tok.encode("hello world", False)     # [31373, 995]
var back = tok.decode(ids)
```

## Verify

```bash
pixi run check-tokenizer
```

43/43 exact id sequences and 43/43 byte-exact decode round trips as of
2026-09-12. The check asserts EXACT id-sequence equality; there is no
tolerance and no skipped case, because an id is an index into someone's
embedding table and a tokenizer that is nearly right is wrong.

## What the property is, and what it is not

This is host-only integer and table work: a byte buffer, three Unicode range
tables, a hash probe and an argmin over small integers. There is no
floating-point arithmetic and no device kernel anywhere in `tokenizer/`, so
cross-vendor bitwise identity is true BY CONSTRUCTION and is not a claim
worth making, and there is no IDENTICAL arm to run. The property that can
actually be wrong, and therefore the only one asserted, is EXACT AGREEMENT
WITH THE REFERENCE IMPLEMENTATION.

## The pieces

| file | what it holds |
| --- | --- |
| `encoding.mojo` | the public surface: `load_gpt2_tokenizer`, `encode`/`encode_bytes`, `decode`/`decode_bytes`, `token_spelling`, and the `<|endoftext|>` rule |
| `impl/byte_unicode.mojo` | GPT-2's byte-to-unicode bijection, built from the recipe: 188 printable Latin-1 fixed points, the other 68 bytes at U+0100..U+0143 |
| `impl/unicode_class.mojo` | `\p{L}`, `\p{N}` and `\s` as binary searches over range tables, plus the UTF-8 decoder |
| `impl/pretokenize.mojo` | the pattern, hand-rolled. Mojo has no regex engine |
| `impl/ranks.mojo` | `data/gpt2_ranks.tsv` as a raw-byte-keyed open-addressing table plus the id -> bytes table `decode` reads |
| `impl/bpe.mojo` | the merge loop: repeatedly merge the adjacent pair of lowest rank |
| `checks/tokenizer_check.mojo` | THE GATE |
| `checks/json_lite.mojo` | just enough JSON to read the fixture. Not a JSON library |
| `tools/gen_unicode_categories.py` | generates `data/unicode_categories.tsv` |

`NOT_IMPLEMENTED.tsv` lists what is deliberately absent: the other
encodings, chat templates, normalization forms, the batch and completion
surfaces, and tiktoken's raise-on-disallowed-special behaviour.

## The pre-tokenizer, which is where the difficulty is

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s
```

Seven alternatives, tried IN ORDER at each position (tiktoken compiles this
with a backtracking engine, so the semantics are leftmost-first, not
leftmost-longest). Three things in it change the output and are easy to get
wrong:

1. **The quantifiers are possessive.** That only bites inside alternative 5,
   `\s++$`: the maximal whitespace run is taken and never given back, so the
   alternative either reaches end of text or fails as a whole.
2. **`\s+(?!\S)` is not possessive and does backtrack**, so it matches a
   whitespace run MINUS ITS LAST CODEPOINT whenever a run of two or more is
   followed by a non-space. That last space then joins the following word
   through alternative 2's optional leading space, which is what makes
   `a  b   c` split as `a`, ` `, ` b`, ` `, ` `, ` c`.
3. **`\p{L}` excludes marks.** A combining acute is not a letter, so
   `e` + U+0301 is two pre-tokens while precomposed U+00E9 is one. The
   fixture's `combining` case is the proof, and it is also why this module
   does not normalize.

`$` is implemented as end of haystack (the Rust `regex` reading, since the
Rust engine produced the fixture), not Python's "also before a final
newline". No fixture case separates the two readings.

The category tables come from `unicodedata` via
`tools/gen_unicode_categories.py` and the file header records the Unicode
version (16.0.0 at the time of writing). `\s` is the Unicode White_Space
PROPERTY, written out from PropList.txt rather than taken from Python, whose
`str.isspace()` also answers True for U+001C..U+001F. A codepoint whose
general category changed between that Unicode version and the one compiled
into tiktoken's `regex` crate is the one way these tables can disagree with
the reference; every such codepoint is outside the fixture.

## What the 43 cases can and cannot catch

Recorded 2026-09-12, three sabotages against the passing gate:

| sabotage | caught by |
| --- | --- |
| merge the HIGHEST ranked pair instead of the lowest | 10 of the 43 id sequences (33/43 still passed) |
| disable alternative 5, so a trailing whitespace run splits through 6 and 7 | THE REACH ARM ONLY -- all 43 id sequences still pass |
| admit uppercase `S` to the contraction set, so `IT'S` splits as `IT`, `'S` | THE REACH ARM ONLY -- all 43 id sequences still pass |

The second and third rows are the reason `check_pattern_reach` exists and
asserts SPLITS, not just ids. Both sabotages move a pre-token boundary
without moving a single id, because BPE never merges across a boundary and
neither `  ` nor `'S` is a token: the wrong split re-merges to the right
ids on exactly these inputs. A gate that only compared ids would have called
both of them correct. Anyone extending the pattern (a second encoding, say)
should assume the same of any new case they add.

## Hand-off

* No Python surface and no `bindings/` entry point: `encoding.mojo` is
  Mojo-only, and nothing in `python/mojolearn/` reaches it.
* Adding an encoding means a new rank table AND a new hand-rolled pattern
  function beside `pretoken_end`, plus its own recorded fixture.
  `impl/ranks.mojo` and `impl/bpe.mojo` are already encoding-agnostic.
