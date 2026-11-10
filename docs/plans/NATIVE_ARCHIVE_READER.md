# Native archive reader implementation

The writer/hash migration is active. The reader is still the existing public
reader while its native replacements are built. The decoder primitives do not
make the public load operation Python-free yet.

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
`/usr/lib/liblzma.5.dylib`; Zstandard currently requests `libzstd.1.dylib`.
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

The next primitives are in `core/native_archive_zip.mojo` (ZIP/ZIP64 metadata,
duplicate-name precedence and overlap checks), `native_archive_npy.mojo` (NPY
v1/v2/v3 grammar, dtype and layout), `native_archive_unicode.mojo` (strict
UTF-8/UTF-32 and literal escapes), and `native_archive_cp437.mojo` (fixed native
filename mapping). These are exercised by 94 parallel parser fixtures, with
hand-authored NPY bytes and fixed independent stdlib ZIP fixture bytes.

Name/decimal lookup is supplied by the separate versioned native Unicode data
module; its generation/provenance and interpreter patch profiles are documented
in `ARCHIVE_UNICODE_PROVENANCE_20261110.md`. Literal decode returns native warning
metadata without calling warning hooks. Boundary warning delivery and exact
exception translation still need integration before activation.

Before reader activation, complete native file-buffer admission, decoded-member
ownership/packing, typed linear schema/state construction, codec wheel dependency
qualification, and boundary warning/error behavior. Existing API/protocol/
exception and file ownership tests remain in place. No legacy reader code is
deleted until its replacement covers these interfaces.
