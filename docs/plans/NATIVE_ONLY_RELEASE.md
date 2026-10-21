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

Retained results on 2026-10-20 now cover all 35 currently written fixtures:
four CPU cases pass on Apple and inside an interpreter-free Linux chroot, and
31 GPU cases pass on Apple. The Linux run is
[39520969549](https://github.com/mojolearn/mojolearn/actions/runs/39520969549).
Its workflow correctly exits nonzero because library-wide coverage remains
incomplete. These are frozen integration-source results, not an assertion that
the current main branch or an installed wheel has the same native closure.

The manifest inventories 603 public exports, including 101 proven aliases.
After aliases are removed, 37 entries have mapped standalone witnesses,
396 still lack fixtures, nine are blocked by their current CPython-bound owner,
and 60 are explicitly nonalgorithm entries. These counts include functions and
helpers and are not counts of distinct estimators. Original compile/runtime
failures are retained alongside repairs: a UMAP test entrypoint import, KDE
borrow origins, a checker false positive, and an IVF oracle that incorrectly
required two different arithmetic folds to return the same bits. The IVF test
now checks the selected fold independently and retains its negative control.

Per-case source/binary hashes, reuse decisions, evidence paths and scope are in
[`native_only_20261020.json`](native_only_20261020.json). Only comment changes
to the two KNN entrypoints reuse nonidentical source bytes; their executable
lines and every transitive source file are unchanged. All other admitted
source closures match byte for byte. The first actual binary/source/log bundle
is verified on the Mac and uploaded to R2 at
`measurements/core-completion/2026-10-20/native-only/first-linux-and-mac-artifacts.tgz`.
Later smoke rounds remain retained on the Mac pending their incremental R2
upload. NVIDIA/AMD standalone runs, installed-wheel checks, and complete native
owner interfaces remain pending. No smoke timing promotes a performance default.
