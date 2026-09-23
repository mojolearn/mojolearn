# Report to Modular: gfx942 optimized IR differs between identical cold compiles

Draft written 2026-09-23 from `repro_floormod.mojo` in this directory. Not sent yet.

## Summary

Mojo 1.0.0 (ed45d567) emits different optimized LLVM IR for the same
kernel, from byte-identical unoptimized IR, from one cold compile to the
next when the target is `get_gpu_target["mi300x"]()` (gfx942). The same
kernel for `sm_90a` and for the host is stable. The variation is the
association of integer adds around the sign fix-up of `Int` floor
modulus (`i % f`), so it does not move a value, but the emitted code
object's bytes move, which breaks byte-reproducible builds.

## Reproducer

`repro_floormod.mojo`, 60 lines, no third-party code: kernel `k4` calls a
helper with `i % f` in a loop three times and prints the `llvm`,
`llvm-opt` and `asm` emissions of `compile_info` for the mi300x target.
Run it with an empty `MODULAR_HOME` each time (a warm `.mojo_cache`
replays the first compile and hides the effect):

    H=$(mktemp -d); cp $PIXI_ENV/share/max/modular.cfg $H/
    MODULAR_HOME=$H mojo run -j 1 repro_floormod.mojo > out.txt

## Measurements

| host | compiles | unoptimized IR | optimized IR | asm |
|---|---|---|---|---|
| macOS 15, Apple M4 (arm64) | 6 | 1 distinct | 3 distinct | 1 distinct |
| Linux x86-64 (RunPod, EPYC 9575F) | 8 | 1 distinct | 4 distinct | 4 distinct |

The optimized IR differs only here (one of the three sites, from two of the
Mac compiles):

    %113 = sub i64 %110, %112
    %114 = select i1 %.not.i9, i64 0, i64 %.pre-phi
    %115 = add i64 %113, %114

versus

    %113 = select i1 %.not.i9, i64 0, i64 %.pre-phi
    %114 = add nsw i64 %113, %110
    %115 = sub i64 %114, %112

On the Linux host the assembly differs as well (register numbers, two
instructions swapped inside the code object); on the Mac the backend
converged to one assembly. The nvptx path from the same source is
identical in every compile on both hosts.

## What it costs downstream

mojolearn ships one AMD binary per release and compares two builds of one
commit byte for byte. Until we rewrote the affected index arithmetic as
counters and unsigned division (mojolearn 0.8.15), 23 AMD bindings could
not be rebuilt reproducibly; with the rewrite, 23 of 23 are byte-identical
over six cold builds. The rewrite is a workaround for a compiler behavior,
and any future kernel using `Int %` or `//` in a loop reintroduces it.

## Ask

Is there a pass in the AMDGPU pipeline whose iteration order depends on
allocation addresses (the classic cause of run-to-run reassociation), and
is there a flag to make gfx942 codegen deterministic? We can run any
further experiment you name on the reproducer.
