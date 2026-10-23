# Ops tools (copies; git is the source of truth)
lq and compile_slot.sh were lost when ~/mojolearn-evidence top-level files were wiped (2026-10-05 ~23:46) and restored from the T9 backup.
Live copies: ~/mojolearn-evidence/lq/lq, ~/mojolearn-evidence/compile_slot.sh, and ~/mojolearn-ops/. Never keep the only copy of a script in a logs or evidence folder.

## Incremental development builds

Internal engineering path: `tools/development_build.py` builds six `identical`
families and seven host families from `bindings/development_groups.json`.
Release scripts keep `bindings/merged_groups.json` and its compact merged files.
The development layout's native compilation, public runtime behavior and
Apple/NVIDIA/AMD identity still require qualification. Controller tests do not
establish numerical equivalence or a compile-time speedup.

Run from an isolated development worktree. Install its pinned default Pixi
environment first. Planning performs no compilation or activation:

```sh
python3 tools/development_build.py plan --sets host --json /path/to/host-plan.json
python3 tools/development_build.py plan --sets identical,host --json /path/to/plan.json
```

Linux GPU planning/builds additionally need one explicit target, for example
`MOJOLEARN_GPU_ARCHS=sm_89` or `MOJOLEARN_GPU_ARCHS=gfx942`. The host plan removes
that GPU target and selects the CPU column. Apple uses the existing compiler
flag classes; it needs no target override. There is no development `fast` lane.

The orchestrator builds on the Mac through the existing semaphore:

```sh
bash ~/mojolearn-evidence/compile_slot.sh \
  python3 tools/development_build.py build --sets identical,host \
  --evidence ~/mojolearn-evidence/development-builds/my-run
```

Repeat that command in the **same worktree and cache location** after editing.
Unchanged groups reuse their retained binaries. Repository and cache paths are
part of the key because they can reach compiler/linker output; moving to a new
worktree intentionally does not claim the old binaries are reusable. Keys also
include transitive sources, generators, actual compiler/stdlib package bytes,
locked dependencies, flags and target. An unresolved import records
`tree-fallback`, widening invalidation rather than guessing it is unrelated.
`--cache` defaults to `~/.cache/mojolearn-development` and must be outside the
source tree. Full logs, plan and per-group receipts remain under `--evidence`.

`--group trees` or `--group host:runtime` limits which missing groups compile.
If another required group is stale or absent, the selected artifacts are kept
in cache but activation returns **3** and preserves the existing loader indexes.
Omit `--group` to build all missing groups. Successful activation stages
immutable files under each set's `.development/<key>/` directory and replaces
its complete loader index atomically. Start a fresh Python process afterward;
already loaded native modules are not hot-reloaded. Remove or relocate explicit
per-binding `.so` overrides first: the controller refuses them because they
would shadow the development index. Use one build controller for the worktree;
do not run a release build there concurrently.

`--jobs N --memory-per-job-gib G` allows parallel groups once `G` is a measured
peak for this source and target. The pool respects available memory and CPU
allocation, including Linux cgroups. `MOJOLEARN_COMPILE_JOBS` controls workers
inside each compiler and defaults to one. Keep the existing Mac semaphore.

To return this worktree to release layout, rebuild the selected complete sets
using the original builder (and the same Linux target when applicable):

```sh
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_PACKAGE_BYTE_LM=1 \
  bash ~/mojolearn-evidence/compile_slot.sh sh bindings/merged_build.sh gpu
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_PACKAGE_BYTE_LM=1 \
  bash ~/mojolearn-evidence/compile_slot.sh \
  env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu sh bindings/merged_build.sh host
```

The complete rebuild replaces all development member mappings. Do not merely
point the index at an old release binary: its sources may have changed. Hidden
development artifacts can remain for reuse; existing release package-data globs
do not include them, and release layout checks refuse a development index.
Final releases still go through the existing release builder and qualification
gates. Plan/build exit **0** means controller success, **1** means failure, and
**3** means an incomplete selected build with no activation; it is not a release
qualification result.
