# Native archive reader implementation

Writer/hash execution and the generic archive reader now use native format
implementations in this source branch. Public loading still requires the
orchestrator's affected-group compile/runtime qualification and codec wheel
dependency qualification. Ridge/OLS schema checks and loaded scalar state now
use `core/native_archive_linear.mojo`. Direct MinMaxScaler save/load uses
`core/native_archive_minmax.mojo`; other saved-model families and the generic
host archive dispatch still have schema work, so this is not a claim that
every public load operation is Python-free.

`core/native_archive_codec.mojo` uses public C APIs and owns each decoder's state
and result bytes. A single compressed stream is dependent; independent members
will run through the existing host pool. zlib and bzip2 input/output lengths are
split at `UINT_MAX`, including buffers above 4 GiB. LZMA and Zstandard use native
`size_t` spans. One extra result byte detects directory-length understatement.
No codec invokes CPython, NumPy, Python callbacks, or Python decoders.

The ABIs are LP64, the supported Linux/macOS wheel hosts. Public definitions:

- [zlib manual](https://zlib.net/manual.html), `z_stream` (112 bytes), raw
  `inflateInit2_`/`inflate`/`inflateEnd`.
- [bzip2 manual](https://sourceware.org/bzip2/manual/manual.html), `bz_stream`
  (80 bytes), `BZ2_bzDecompressInit`/`BZ2_bzDecompress`/`BZ2_bzDecompressEnd`.
- [liblzma base header](https://github.com/tukaani-project/xz/blob/v5.6.4/src/liblzma/api/lzma/base.h)
  and [filter header](https://github.com/tukaani-project/xz/blob/v5.6.4/src/liblzma/api/lzma/filter.h),
  `lzma_stream` (136 bytes), `lzma_filter` (16 bytes), raw LZMA1 decoding.
- [Zstandard public header](https://github.com/facebook/zstd/blob/v1.5.6/lib/zstd.h),
  `ZSTD_DStream`, `ZSTD_inBuffer` and `ZSTD_outBuffer` (three native words each).

The existing `SharedSlot`/`OwnedDLHandle` mechanism retains libraries once. Linux
names are `libz.so.1`, `libbz2.so.1.0`, `liblzma.so.5`, `libzstd.so.1`. macOS system
paths are `/usr/lib/libz.1.dylib`, `/usr/lib/libbz2.1.0.dylib`, and
`/usr/lib/liblzma.5.dylib`; Zstandard requests `@rpath/libzstd.1.dylib` through
the existing wheel rpaths.
The macOS SDK provides the first three, not Zstandard. These are discovery
assumptions, **not wheel dependency qualification**. Existing ELF audits walk
linked dependencies; they do not automatically stage these `dlopen` dependencies.
Explicit staging/audit with the existing wheel mechanism is required before
activation. No platform is silently exempted and no codec is removed.

`check-native-archive-codec` has 35 parallel fixtures, including four codecs,
empty and 64 KiB output, independently planted plaintext, trailing bytes,
directory-size mismatch, malformed input, owned output, bounds and overflow.
Compressed fixtures were prepared with the Python standard library at development
time; Zstandard fixtures directly encode the published raw/RLE block format.
The executable itself uses no Python. Zstandard library absence is a dependency
failure, not a skipped/passed fixture.

The parser primitives are in `core/native_archive_zip.mojo` (ZIP/ZIP64 metadata,
duplicate-name precedence and overlap checks), `native_archive_npy.mojo` (NPY
v1/v2/v3 grammar, dtype and layout), `native_archive_unicode.mojo` (strict
UTF-8/UTF-32 and literal escapes), and `native_archive_cp437.mojo` (fixed native
filename mapping). These are exercised by 94 parallel parser fixtures, with
hand-authored NPY bytes and fixed independent stdlib ZIP fixture bytes.

Decoded numeric members preserve their input bits. Stored members borrow the
admitted input owner, compressed members own Mojo byte lists, and typed text
cells hold native scalars/spans. Exact BytesIO input borrows its existing buffer;
native files use the already-connected mmap owner or native readinto when mapping
is unavailable. PACK makes one writable public data copy and uses the shared
native Array layout API. Caller-owned files stay open. Ten parallel direct reader
fixtures cover all five ZIP methods, signed zero/NaN payloads, Unicode/bytes
Fortran layout, CRC refusal, and warning/error ordering.

Name/decimal lookup is supplied by the separate versioned native Unicode data
module; its generation/provenance and interpreter patch profiles are documented
in `ARCHIVE_UNICODE_PROVENANCE_20261110.md`. Literal decode returns native warning
metadata without calling warning hooks. PACK emits those events through the
existing admitted native warning boundary in member order, including warnings
that precede a later dtype/payload refusal.

Direct Ridge/OLS loads bypass the intermediate Python result dictionary. Their
native plan retains saved mode, ancestry checks, exact dtype/shape refusals,
intercept/alpha bits and constructor defaults; PACK creates the public model
and publishes its native fitted-state record. No mean array is invented for
loaded models. The shared generic host-archive map has a compatibility UNPACK
adapter into the same native plan and retains the existing coefficient owner.
Historical numeric/bytes metadata spelling is native too, including NumPy's
legacy print configuration. Thirty-seven parallel native fixtures cover these
schema/refusal/formatting contracts. Scalar spelling follows the published
[NumPy scalar formatting source](https://github.com/numpy/numpy/blob/v2.3.0/numpy/_core/src/multiarray/scalartypes.c.src)
and reuses Mojolearn's existing binary32 formatter.

Filesystem byte decoding and pathlib component spelling now use
`core/native_archive_path.mojo`, including UTF-8 surrogateescape, the distinct
POSIX `//` root and lexical dot removal for pathlib reads. Exact string/byte
filenames keep their prior spelling; no path protocol or codec registry is
called. Forty-two independent native fixtures cover valid and invalid UTF-8,
surrogate/NUL preservation and path component ordering. The Python boundary
retains exact file/path admission and packs the final native Unicode words.
Save reuses the shared native linear mode parser for Unicode whitespace and
ASCII case normalization. Archive type admission uses fixed native identities;
string-subclass storage is checked with `PyType_IsSubtype`, and class names
come from the immutable type descriptor. The obsolete runtime `builtins`
lookups and Python `strip`/`lower` format calls are removed from archive files.
Path-containing error names are unpacked to a typed graph; native Mojo formats
container delimiters, aliases, cycles and file names with an iterative task
stack. Python references only pin input owners and captured native exceptions
at the boundary. Display-order failure selection retains the first exception
even when a later graph node was admitted earlier. Thirty-four direct native
fixtures cover this graph contract, including deep names and file recursion;
the existing public exception/protocol tests remain unchanged.

The MinMaxScaler native plan preserves class-layout admission, ancestry,
schema/refusal order, signed counts, clip metadata and unmodified Float32/64
bits. Save owns metadata as Mojo bytes and borrows the five current public
statistic buffers. Load publishes each writable C-order statistic with one
payload copy, removing the intermediate Python result map and duplicate array
copy. The old dedicated map builder and temporary Python metadata arrays are
deleted. The shared endpoint admission now returns a native pair; its existing
Python-pair wrapper remains for unchanged lifecycle callers. Fifty-four direct
Mojo fixtures cover schema/refusals, native metadata bytes and borrowed views;
one Python seam checks ownership after input close, writable arrays and save
readback. The separate generic `host_archive_minmax` map path is unchanged and
remains migration work. Affected development owner: `runtime_prep`; its source
closure also changes through `minmax_metadata.mojo`. No estimator kernel changes.

Text-cell storage is now allocated once without a serial construction pass.
Each host worker constructs and moves its assigned owning cells directly into
that final allocation. Every slot is initialized even when its decode fails,
so error unwinding can safely destroy the whole list. A worker keeps its first
logical error; the caller scans worker summaries in chunk order. The removed
per-cell error array and serial error scan are no longer proportional overhead
on large text payloads. Numeric members retain their early-return path. Ten
direct native fixtures cover C/F Unicode and byte cells, embedded NULs,
supplementary code points, empty/zero-width input, input-owner release, and
multiple errors whose serialized and logical orders differ. Final owner-size
overflow is refused before allocation, including zero-width text arrays whose
declared count cannot be bounded by payload bytes.

This uses the supported uninitialized List allocation and pointer
`unsafe_write` placement API, which must initialize rather than assign over
owning elements: [Mojo List source](https://github.com/modular/modular/blob/main/mojo/stdlib/std/collections/list.mojo)
and [Mojo 1.0 pointer API migration](https://github.com/modular/modular/blob/main/Mojo/docs/site/releases/v1.0.0.md).
Compilation and runtime coverage remain pending with the orchestrator.

Remaining work is codec wheel dependency qualification and affected public
boundary tests. Existing API/protocol/exception and file ownership tests remain
in place. The old Python-object ZIP,
NPY and Unicode parser/codec helpers and their import-time codec dependencies
are removed with the native reader connection; no callback fallback remains.
