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

Retained results on 2026-10-20 cover all 41 written fixtures through frozen
integration source `3426e599c`: six CPU cases and 35 GPU cases pass on Apple.
The initial four CPU cases also pass inside an interpreter-free Linux chroot.
The Linux run is
[39520969549](https://github.com/mojolearn/mojolearn/actions/runs/39520969549).
Its workflow correctly exits nonzero because library-wide coverage remains
incomplete. These are frozen integration-source results, not an assertion that
the current main branch or an installed wheel has the same native closure.

AMD additionally passes all 35 GPU fixtures and both the optimizer and
convolution CPU fixtures inside an interpreter-free chroot. All seven optimizer state digests match Apple CPU,
Apple GPU, AMD GPU and the AMD machine's CPU. SVC and SVR output digests match
Apple and AMD GPUs; their independent RBF oracle still uses its stated tolerance.
Four convolution geometry digests also match on all four tested CPU/GPU targets.
GPU execution has dependency screening,
but has not been isolated from an installed interpreter.

The manifest inventories 603 public exports, including 101 proven aliases.
After aliases are removed, 44 entries have mapped standalone witnesses,
389 still lack fixtures, nine are blocked by their current CPython-bound owner,
and 60 are explicitly nonalgorithm entries. These counts include functions and
helpers and are not counts of distinct estimators. Original compile/runtime
failures are retained alongside repairs: a UMAP test entrypoint import, KDE
borrow origins, a checker false positive, and an IVF oracle that incorrectly
required two different arithmetic folds to return the same bits. The IVF test
now checks the selected fold independently and retains its negative control.
Optimizer and convolution gradient negative controls also fail as expected.
A later packager failure copied `tape.mojo` as `Tape.mojo` on a case-insensitive
filesystem. Exact filename resolution repairs that failure without changing the
numerical fixture; the build-tool suite passes all 55 tests.

Per-case source/binary hashes, reuse decisions, evidence paths and scope are in
[`native_only_20261020.json`](native_only_20261020.json). Only comment changes
to the two KNN entrypoints reuse nonidentical source bytes; their executable
lines and every transitive source file are unchanged. All other admitted
source closures match byte for byte. The first actual binary/source/log bundle
is verified on the Mac and uploaded to R2 at
`measurements/core-completion/2026-10-20/native-only/first-linux-and-mac-artifacts.tgz`.
AMD's actual source, binary and native-library bundles are verified on the Mac
and by full R2 readback under `native-only/amd-round1-artifacts.tgz` and
`native-only/amd-incremental-artifacts.tgz`, plus
`native-only/amd-conv-artifacts.tgz`, in that same measurement prefix.
Later Mac smoke rounds, original failures and source snapshots are also verified
by full R2 readback at `native-only/incremental-mac-artifacts.tgz`. NVIDIA
standalone runs, installed-wheel checks, and complete native
owner interfaces remain pending. Later graph edits also need qualification of
their changed source closures. No smoke timing promotes a performance default.

The two SVC/SVR fixtures also landed on main at `23b4f1cc8`, after exact
77-file closure comparison against their qualified source and 19 static gate
tests. That main snapshot contains 37 fixtures and 602 exports; the additional
integration fixtures and public export remain separately scoped. Its reuse
receipt is [`native_only_svm_20261020.json`](native_only_svm_20261020.json).

## The retired Python source guard

Andrew removed the kernel-path/token source guard on 2026-11-10. Its zero count
excluded the binding/provider layers where Python objects and calls remained;
it did not establish Python-free product execution. The checker, push/CI
invocations and checker-only tests are removed, with no replacement source-count gate.

The former baseline files are retained unchanged under
`experiments/retired_python_source_guard_20261110/` as historical evidence, not
as active exceptions. Earlier source-audit reports remain historical records.
The current work is the typed boundary migration in
[`PYTHON_AT_THE_BOUNDARY.md`](PYTHON_AT_THE_BOUNDARY.md), with correctness and
performance established on actual compiled operations.

## The no-compute-in-shells check

A shell file (`bindings/**`, `*pyapi*.mojo`, `*_py.mojo`) is a passthrough: it
unpacks PythonObject arguments into typed values, calls one typed kernel entry
and converts the typed result back. `tools/hooks/no_compute_in_shells.py --tree`
reports, in every shell file, each `for`/`while` line that carries no tag and
each kernel launch (`enqueue_function`, `DeviceContext(`, `parallelize`,
`vectorize`). A loop over argument words of bounded, data-independent length is
tagged `# small-loop(<what>)`, and a driver loop over model structure
`# driver-loop: over <what>`, on the loop line or the line above; launches are
never taggable. The pre-push hook and the second job of
`.github/workflows/no-compute-in-shells.yml` run it with the same main/branch
rule as the other fences.

The hits on main `28b8ed11b` (1,206 in 116 files: 1,185 untagged loops and 21
launches) are listed in `tools/hooks/compute_in_shells_baseline.txt` as
`path:line: code`. A hit is matched by path and code text, so edits above a row
do not break it. The list only shrinks: when the work moves into the kernel
file, `no_compute_in_shells.py --prune-baseline` drops the stale rows.

## The kernel closure compile check

`tools/kernel_closure_check.py` is the compile proof that kernel code needs no
Python. It writes `checks/kernel_closure_main.mojo` (git-ignored), a program
that imports every kernel module, and builds it for the CPU target in a copy of
the tree where every shell that imports or reaches Python is deleted and every
Python import line is removed. A kernel module that uses a Python name, or
imports a Python shell directly or through other modules, fails to compile.
The binary must have no undefined CPython symbol and no libpython dependency,
and it runs once. A static pass first lists kernel imports of Python shells
(failures) and of Python-free helpers that still live under `bindings/`
(to move out). `--emit-only` writes the program and the stripped tree and
prints the program path and the build command. The GitHub workflow
`.github/workflows/kernel-closure.yml` runs it on every push to main and on
dispatch, with a 45-minute compile budget, and keeps the generated program, the
build log and `summary.json`.
