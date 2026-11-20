# Mojolearn accelerator-target API migration

Mojo 1.2 removed the three vendor `has_*_gpu_accelerator()` functions.
The first consolidated 1.2 build exposed 121 distinct missing-import diagnostics
across its 25 selected Mac identical/host units. The recorded build failed;
this source repair has not been compiled or executed by the lane.

`core/accelerator_target.mojo` exports three compile-time Boolean constants.
Their native initializer helpers check `has_accelerator()` with `comptime if`
before calling `default_accelerator().is_apple_gpu()`, `.is_amd_gpu()` or
`.is_nvidia_gpu()`. This preserves false for accelerator-free CPU builds:
`default_accelerator()` itself requires a configured accelerator. The default
target remains the compiler-selected target, including `--target-accelerator`;
there is no runtime host discovery or target override.

The documented replacement is in the [Mojo 1.2 release notes](https://mojolang.org/releases/v1.2.0/#removed),
and the accelerator precondition is in the
[`default_accelerator` API](https://mojolang.org/docs/std/sys/info/default_accelerator/).
The tagged upstream `mojo/v1.2.0/Mojo/stdlib/std/sys/info.mojo` source confirms
that `has_accelerator()` and `has_amd_rdna_gpu_accelerator()` remain available.
The existing RDNA predicate and its priority remain unchanged.

The mechanical transformation removes 356 obsolete predicate calls in 205
Mojo files and their obsolete imports. Every caller retains its existing Boolean
expression, branch priority, fallback column, threshold, define and default.
The source audit reverses only the declared import/predicate substitutions and
recovers every original file byte for byte; comments and historical references
are untouched. Tuned numerical bodies change only where an existing vendor
guard uses the new compile-time constant. No numerical computation is replaced.

`x_neighbors/gen.py` now emits the same constant import and gate as the checked-in
source. The existing BF16 default-policy test supplies the compile-time Boolean
in its evaluation namespace; its parameters and expected outcomes are unchanged.
No generator, test or compiler was executed. Static Python AST parsing and
source-diff checks are source evidence only.

Root owns the combined retry of all 25 selected units and any native checks.
Accelerator-free CPU, explicit NVIDIA/AMD targets, installed runtime behavior,
performance and cross-vendor identity are not established by this source audit.
The existing one-release identity policy is unchanged.

Evidence: `~/mojolearn-evidence/python-mojo-migration-20261110/accelerator-info-120/`:
`repair-source-audit.json`, `final-audit.log`, full patches and official source
snapshots. Initial ambiguous adoption, missing historical symlink and certificate
fetch failures remain recorded; the successful adoption, transformation and
source audit are separate receipts. Active root build source was not edited.
