# Mojolearn native archive Unicode data

The archive literal decoder uses typed Mojo lookups. Python is used only by the
build-time generator. No runtime Python decoder, `unicodedata` import or
CPython name-lookup capsule is used.

`tools/gen_archive_unicode.py` downloads independent version inputs concurrently,
checks each pinned SHA-256, and writes the ignored generated module
`core/archive_unicode_names_generated.mojo`. Cached inputs can be supplied with
`--source-dir`; generation never depends on the generator interpreter's own UCD.
Each name/code point and digit record is shared across versions with a presence
mask. Hangul composition and CJK/Tangut ranges remain algorithmic. Named
sequences are deliberately absent: archive string escapes require one scalar.

| CPython | UCD | Official input directory |
| --- | --- | --- |
| 3.10 | 13.0.0 | [UCD 13](https://www.unicode.org/Public/13.0.0/ucd/) |
| 3.11 | 14.0.0 | [UCD 14](https://www.unicode.org/Public/14.0.0/ucd/) |
| 3.12 | 15.0.0 | [UCD 15](https://www.unicode.org/Public/15.0.0/ucd/) |
| 3.13 | 15.1.0 | [UCD 15.1](https://www.unicode.org/Public/15.1.0/ucd/) |
| 3.14 | 16.0.0 | [UCD 16](https://www.unicode.org/Public/16.0.0/ucd/) |

Every directory supplies `UnicodeData.txt` and `NameAliases.txt`. Exact input
hashes are recorded in the generator's `VERSIONS` table; the data license is
[UNICODE-LICENSE.txt](../licenses/UNICODE-LICENSE.txt). The generated payload has
40,490 shared names/aliases and 888 digit records. Its pinned data digest is
`008b4b54a8a01ae692b2a19f1d16995dcca1c2d112cff162eb74096805ee1179`.
The digest covers names, offsets, digits, CJK ranges and Tangut ranges in that
order. It is a generated-source integrity record, not model identity evidence.

The name API accepts the CPython minor and patch version because the accepted
algorithmic spellings changed in maintenance releases. Ordinary names and
aliases use ASCII case-insensitive comparison in all supported versions.
CPython through 3.13.12 and 3.14.3 requires exact uppercase Hangul/CJK spellings;
its CJK lookup accepts four or five uppercase hexadecimal digits, including
leading zeroes. CPython 3.13.13 and 3.14.4 add case-insensitive Hangul/CJK lookup,
reject leading zeroes in algorithmic hexadecimal names, and support Tangut
lookup. The native version predicate preserves both profiles.

These boundaries were checked against the official tagged implementations:
[3.13.12](https://github.com/python/cpython/blob/v3.13.12/Modules/unicodedata.c),
[3.13.13](https://github.com/python/cpython/blob/v3.13.13/Modules/unicodedata.c),
[3.14.3](https://github.com/python/cpython/blob/v3.14.3/Modules/unicodedata.c),
[3.14.4](https://github.com/python/cpython/blob/v3.14.4/Modules/unicodedata.c),
and the [case-sensitivity fix](https://github.com/python/cpython/commit/ac9e9e2c8fec27bbf06f5ed2178f7a9734eea9a2).
The public release records are the
[3.13 changelog](https://docs.python.org/3.13/whatsnew/changelog.html#python-3-13-13-final)
and [3.14 changelog](https://docs.python.org/3.14/whatsnew/changelog.html#python-3-14-4-final).

`archive_unicode_digit` distinguishes decimal digits (0–9), characters accepted
by `isdigit` but not decimal conversion (-2), and all others (-1). Generated
records use UnicodeData decimal/digit fields and the same version masks.

Generation completed locally from pinned cached official bytes. Compilation,
direct native lookup fixtures and archive roundtrips remain with integration;
no runtime or performance claim follows from generating the table.
