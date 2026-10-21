# Native-only release qualification

The standalone native artifact must run without Python installed. This is
stronger than the PyPI binding contract: a compiled CPython buffer adapter can
avoid Python bytecode while still requiring the interpreter ABI. A compile
failure can therefore identify an inseparable binding dependency, a normal
compiler error, or an unsupported target; it does not by itself establish that
Python executed. A successful compile does not establish interpreter independence
either, because Mojo can load Python at execution time.

`packaging/native_only/manifest.json` inventories public names and records the
scope of each standalone smoke. A native kernel or tape witness does not qualify
an entire estimator, its public wrapper, or every configuration. Missing and
blocked entries remain visible and prevent complete release qualification.
Metadata and namespaces are distinguished from executable algorithm surfaces.

Run the local dependency screening with the pinned compiler:

```sh
pixi run python tools/native_only_smoke.py --out /tmp/mojolearn-native-smoke
```

The manual `native-only-smoke.yml` workflow builds the frozen source on Linux,
then executes each binary inside a minimal chroot containing only the tested
executables and their inspected native shared libraries. It copies no Python
interpreter, interpreter library, packages, source tree, shell, or host filesystem
mount. Python is used by the external test/build controller only. The complete
runtime bytes are retained in `native-only-runtime.tar.gz`, alongside source
hashes, compiler identity, library hashes, full logs and per-case results.

Compilation defaults to two independent jobs and a 600-second limit per binary.
Execution defaults to ten seconds per case. CPU smoke cases may run in parallel;
GPU cases run serially. These are correctness and packaging checks, not timing
measurements. Unchanged source closures, toolchains, targets and build settings
reuse content-verified binaries. Timeout, missing output marker, missing coverage,
source mutation during compilation, unresolved imports, CPython dependencies and
nonzero exits remain failures. A selected subset is never a complete release
qualification. Local Mac results are dependency screening, not proof of a Linux
runtime without Python.

The existing public-operation audits remain required. Standalone tests cannot
detect a Python callback in a separate PyPI wrapper they never execute. Native-only
packaging must exercise the same native algorithms as the product and must not
replace them with an easier test implementation. GPU driver isolation and a
complete standalone estimator/model interface remain separate coverage items.

Initial development evidence: the native tape branched-backward executable
compiled and passed locally (8.83 seconds compile, 0.26 seconds execution), with
no Python import in its 90-file source closure or linked interpreter dependency.
This single native-core witness does not qualify a Python-free library release.
Evidence: `~/mojolearn-evidence/core-completion-20261020/native-only-probe/`.
