# Native archive codec packaging

The native archive reader uses public LP64 C APIs for zlib, bzip2, liblzma and
Zstandard. These libraries are opened on first use and their native handles are
cached. They do not appear in a binding's static `DT_NEEDED` or `otool -L`
dependency list, so staging only static dependencies can produce a wheel that
works on the build machine and fails after installation elsewhere.

The existing Linux staging closure now has four explicit roots:
`libz.so.1`, `libbz2.so.1.0`, `liblzma.so.5`, and `libzstd.so.1`. It retains each
library from the pinned pixi environment under its original SONAME, walks each
root's dependencies, and records those bytes and roots in the normal manifest.
The normal auditwheel exclusion list already includes every staged library
name, preserving these names through repair. The loader uses the existing
`$ORIGIN` paths. [Linux dlopen documents the calling object's RUNPATH search](https://man7.org/linux/man-pages/man3/dlopen.3.html).

On macOS, zlib, bzip2 and liblzma continue to use the existing system
`/usr/lib` paths. Zstandard is staged explicitly in the existing `.dylibs`
closure and opened as `@rpath/libzstd.1.dylib` (codec patch `34108ec020`). The
normal `@loader_path` rpaths and final signing apply to it too.
[Apple documents run-path dependent libraries](https://developer.apple.com/library/archive/documentation/DeveloperTools/Conceptual/DynamicLibraries/100-Articles/RunpathDependentLibraries.html).
There is no Python path lookup or new native loader.

The staging graph inspects independent dependency frontiers concurrently;
independent library copies and load-command preparation also run concurrently.
Linux worker counts respect process affinity and cgroup allocation. Graph
results remain in stable discovery order. Redundant macOS signing before the
final load-command edits was removed; each staged dylib is signed after those
edits, as before.

The focused tooling fixtures cover undeclared static edges, transitive codec
dependencies, cycles, missing input libraries, and missing staged libraries
while the build environment still contains them. Root must run those fixtures
and qualify installed-wheel archive decoding with the build environment hidden.
This lane has not run tooling tests, native builds, installed-wheel tests, or
performance measurements. Existing native codec fixture results alone do not
qualify installed-wheel discovery. No kernel or algorithm output bits change.

## Supplemental packaging environment

The compiler SDK is not assumed to contain every codec. In particular, the
current macOS default SDK lacks Zstd, while the existing pinned `pkg`
environment supplies it. Both release staging invocations therefore declare
`--codec-lib` using that packaging environment's library directory. The SDK
remains the first source for codec roots it already supplies; an unused copy
in the supplemental environment does not cause a duplicate-byte refusal.

Transitive dependencies keep the provenance of the environment that needs
them. If both closures actually require the same name from different origins,
staging requires equal bytes; a real conflict refuses assembly. Equal copies
retain SDK provenance. Linux records the codec directory and each selected
source in its normal manifest; macOS records codec source paths and SHA-256
values in the staging log. The shared build-time selection helper is included
in release identities only for source revisions that contain it, preserving
historical identities. No compiler environment is installed or modified.
