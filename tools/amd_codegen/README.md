# gfx942 codegen reproducibility probes (2026-09-22)

Mojo 1.0.0's AMDGPU optimization pipeline does not always give the same
optimized IR for the same unoptimized IR: integer index arithmetic around a
division/remainder (the sign fix-up of `Int` floor `%` and `//`, and the
adds next to them) comes out associated one way or the other from one cold
compile to the next. The same kernels for sm_90a are stable. No float
operation is involved, but the bytes of the gfx942 binding move.

- `repro_floormod.mojo`: the minimal reproducer, no mojolearn code
  (kernel `k4`: a helper with `i % f` in a loop, called three times).
- `probe_tsa.mojo`, `probe_gemm.mojo`: the Holt-Winters kernels and the
  IDENTICAL GEMM kernel templates at their dispatched instantiations,
  emitted as unoptimized IR, optimized IR and gfx942 asm through
  `std.compile.compile_info`. They compile on any machine (no GPU).
- `co_repro.py`, `ptx_repro.py`, `extract_amdgpu.py`: compare the embedded
  AMDGPU code objects / PTX modules of repeated builds.

A cold compile means an empty MODULAR_HOME (copy `modular.cfg` into a new
directory and run `pixi run env MODULAR_HOME=<dir> mojo build -j 1 ...`);
a warm Mojo cache replays the first compile and hides the variation.
