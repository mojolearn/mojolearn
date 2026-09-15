# Byte-level BPE tokenizer (GPT-2 format)

`encode` and `decode` for byte-level BPE in the GPT-2 format, in Mojo, over a
vocabulary the caller supplies. mojolearn ships no vocabulary and tracks no
tokenizer data file (2026-09-15).

```mojo
from tokenizer.encoding import load_gpt2_tokenizer_from

var tok = load_gpt2_tokenizer_from("ranks.tsv")   # rank<TAB>hex lines
var ids = tok.encode("hello world", False)
var back = tok.decode(ids)
```

From Python, `mojolearn.GPT2Tokenizer.from_files(encoder_json, vocab_bpe)`
loads the two files of the GPT-2 format, which users download themselves;
`from_ranks_file(path)` and `from_token_bytes(tokens)` load a rank table.
`GPT2Tokenizer()` with no vocabulary refuses by name.

## Verify

```bash
pixi run check-tokenizer
```

The task writes `build/tokenizer_synthetic/` with
`python/mojolearn/_tokenizer_synthetic.py`: a synthetic vocabulary mojolearn
trains itself (the 256 byte tokens plus 256 merges learned from a
deterministic corpus) and 22 cases whose ids come from that module's second,
pure Python encoder of the same algorithm. The check asserts EXACT id-sequence
equality and byte-exact decode round trips; there is no tolerance and no
skipped case, because an id is an index into an embedding table and a
tokenizer that is nearly right is wrong.

## What the property is, and what it is not

This is host-only integer and table work: a byte buffer, three Unicode range
tables, a hash probe and an argmin over small integers. There is no
floating-point arithmetic and no device kernel anywhere in `tokenizer/`, so
cross-vendor bitwise identity is true BY CONSTRUCTION and is not a claim
worth making, and there is no IDENTICAL arm to run. The property that can
actually be wrong, and therefore the one asserted, is that the algorithm is
the one stated.

## The pieces

| file | what it holds |
| --- | --- |
| `encoding.mojo` | the public surface: `load_gpt2_tokenizer_from`, `encode`/`encode_bytes`, `decode`/`decode_bytes`, `token_spelling`, and the `<|endoftext|>` rule |
| `impl/byte_unicode.mojo` | the format's byte-to-unicode bijection, built from the recipe: 188 printable Latin-1 fixed points, the other 68 bytes at U+0100..U+0143 |
| `impl/unicode_class.mojo` | `\p{L}`, `\p{N}` and `\s` as binary searches over range tables, the pinned Unicode version and table sha256, plus the UTF-8 decoder |
| `impl/pretokenize.mojo` | the pattern, hand-rolled. Mojo has no regex engine |
| `impl/ranks.mojo` | a rank file as a raw-byte-keyed open-addressing table plus the id -> bytes table `decode` reads |
| `impl/bpe.mojo` | the merge loop: repeatedly merge the adjacent pair of lowest rank |
| `checks/tokenizer_check.mojo` | THE GATE |
| `checks/json_lite.mojo` | just enough JSON to read the fixture. Not a JSON library |
| `tools/gen_unicode_table.sh`, `tools/gen_unicode_categories.py` | generate `impl/unicode_table_generated.mojo` (not tracked) at build time |

`NOT_IMPLEMENTED.tsv` lists what is deliberately absent.

## The pre-tokenizer, which is where the difficulty is

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s
```

Seven alternatives, tried IN ORDER at each position (leftmost-first, not
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
   fixture's `combining_mark` case covers it, and it is also why this module
   does not normalize.

`$` is implemented as end of haystack, not "also before a final newline".

## The Unicode classes are generated, not tracked

`bindings/build_host_family.sh tokenizer` and `pixi run check-tokenizer` run
`tools/gen_unicode_table.sh`, which computes the letter and number ranges from
Python's standard `unicodedata` and writes them, with the White_Space ranges,
into `impl/unicode_table_generated.mojo` as a string constant. The Unicode
version (16.0.0) and the table's sha256 are pinned in
`impl/unicode_class.mojo`: the generator refuses any other version or hash
(the script uses a Python whose `unicodedata` is the pin, fetching one with
`pixi exec` when `python3` on PATH is another version), and the loader refuses
a generated module that disagrees with the pin. `\s` is the White_Space
PROPERTY, not Python's `str.isspace()`, which also answers True for
U+001C..U+001F.

## The reach arm

`check_pattern_reach` asserts SPLITS, not just ids. A pre-tokenizer that
dropped alternative 5 or admitted uppercase contractions moves a pre-token
boundary without necessarily moving an id, because BPE never merges across a
boundary and the wrong split can re-merge to the right ids on a given input.
A gate that only compared ids could call such a sabotage correct.

## The Python door (2026-09-14)

`mojolearn.GPT2Tokenizer` (`python/mojolearn/tokenizer.py`) reaches
`encoding.mojo` through the host binding
`bindings/_mojolearn_tokenizer_host.mojo`, built from source with
`bindings/build_tokenizer_host.sh` (a shim over
`bindings/build_host_family.sh`). `python/mojolearn/tests/test_tokenizer_surface.py`
holds that door to the synthetic cases, the rank-file and spelled-file
loaders, a user-supplied GPT-2 vocabulary when `MOJOLEARN_GPT2_ENCODER_JSON`
and `MOJOLEARN_GPT2_VOCAB_BPE` name the files (skipped otherwise), and every
refusal by name. A build with `-D MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1` (ids
written in reverse) must fail it. The lane brief is
`docs/lanes/BRIEF_expose_tokenizer_2026-09-14.md`.

`GPT2Tokenizer.encode_batch(documents, allow_endoftext=False)` (2026-09-15)
passes every document to `gpt2_encode_batch` in ONE call; the binding
encodes each document alone with the same `encode_bytes` call, so each
document's ids equal `encode` on it. `decode_batch` and `decode_bytes_batch`
are a loop over `decode_bytes`. `-D MOJOLEARN_TOKENIZER_BATCH_SABOTAGE=1`
swaps ids across document boundaries inside a batch and must fail the surface
test's batch checks and the `tokenizer` identity lane's batch part.

## Hand-off

* Another pre-tokenizer pattern means a new hand-rolled function beside
  `pretoken_end`, plus its own cases. `impl/ranks.mojo` and `impl/bpe.mojo`
  are already vocabulary-agnostic.
