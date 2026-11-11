# Mojolearn toolchain upgrade and simplification

Andrew authorized Mojo 1.2.0 / MAX 26.7.0 on 2026-11-11, followed by removing
obsolete compiler workarounds. This joins the existing Python boundary work in
`preinteg/native-consolidated-20261111` for one combined development build and
merge. It is not a release or a performance claim.

## Changes

- Exact toolchain pins and resolved packages for macOS ARM64, Linux x86-64 and
  Linux ARM64; matching MAX adapter and native package requirements.
- Current GPU namespaces, closure decorator, fixed-size array and origin names,
  memory function names and atomic type arguments. These are API substitutions:
  numerical expressions, kernel routes, allocations and copy sizes are retained.
- IPCA fixtures again use their original nested storage. The six-buffer special
  case is removed because the newer compiler accepts the original reproducer.
- OPTICS captures one outer result pointer and derives interior views inside the
  parallel callback. It keeps its existing parallel work, inputs and oracle.
- No Python fallback, retired source-count guard, vendor-specific reference or
  alternate algorithm is introduced.

The API changes are documented in the official [Mojo 1.1 release notes](https://mojolang.org/releases/v1.1.0/)
and [Mojo 1.2 release notes](https://mojolang.org/releases/v1.2.0/).
The compiler upgrade may change numerical bits; the identity contract remains
agreement between supported vendors within a release.

## Evidence and limits

The earlier source `183c32da489eaa413715617bfe9caf1d8bbfb6a2` compiled all 25 Mac
binding units using Mojo 1.0. Its merge was intentionally held for the combined
upgrade. Those binaries do not qualify the new compiler or source.

The new toolchain resolves and reports `Mojo 1.2.0 (2016d0cd)`. Forty packaging
and adapter metadata controls passed. The frozen tooling suite recorded 183
passes, three expected failures and one failure; the repaired dependency case
then passed its focused rerun. These are controller tests, not native runtime
qualification.

The consolidated Mojo 1.2 build, selected native fixture execution and separate
image call-table proof remain pending in this source checkpoint. The existing
development builder selects all 25 binding units with zero old-toolchain reuse.
Linux/NVIDIA/AMD compilation and measurements remain separate, uncompleted work.
No speed claim follows from removing Python or upgrading the compiler.

Detailed evidence is under `~/mojolearn-evidence/modular-check-20261111/` and
`~/mojolearn-evidence/python-mojo-migration-20261110/`; the finite build controller
records its exact frozen source and merge outcome in
`consolidated-120-<attempt>/integration.json`.

## Work still owed

The Python boundary migration is incomplete: the audited source before the
OneClassSVM addition contains 326 distinct migrated state roles, of which 278
are on the earlier main. These are reviewed roles, not object allocation counts
or a percentage of the whole library. Required public Python exporter pins are
not claimed as removed.

Complete the native call-table rollout family by family after the Ridge/OLS
proof; retain Python only for admission, public ownership and result packing.
Finish the remaining SVM and general input-conversion work, native correctness
test migrations, and runtime Python-crossing measurement. The current Python
profile observer cannot see every direct C-API call and cannot certify a
Python-free operation.

The shared Metal `DeviceContext` failure remains open as Modular #7544; runtime
default arguments remain #7543. GPU fixture groups stay ordered while kernels
remain parallel. `docs/MODULAR_ASKS.md` records the new reproducer results and
the older Linux/NVIDIA/AMD issues still requiring a recheck. No workaround for
the unresolved compiler/runtime defects is authorized by this upgrade.
