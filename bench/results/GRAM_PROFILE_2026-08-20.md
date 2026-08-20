# Gram kernel profile attempt, 2026-08-20: what headless Instruments can and cannot see

Two xctrace recordings of the PCA phase binary (Metal System Trace, then
Game Performance), M4, exported via `xctrace export --xpath`.

## Fact 1: the GPU performance-state ramp is real and slow

`gpu-performance-state-intervals`: the device enters the run at **Minimum**
state (~130 ms), steps to **Medium** (~90 ms), and reaches **Maximum** only
~240 ms in, then holds Maximum for the steady-state phase loop. So:

- First-shot numbers are taken partly at Minimum/Medium clock and are NOT
  kernel measurements. This mechanically explains the recorded first-shot
  inflation (gram 49 -> 42.5 steady; kmeans bracket 1: 45/69 vs steady
  27/17) and a chunk of the M4's session-to-session drift.
- The steady-state 44 ms gram bracket runs at Maximum state: clock does NOT
  explain the gap to the ~15 ms traffic floor.

## Fact 2: no limiter counters are reachable headlessly

Metal System Trace records `Counter Set: (null)`, `Shader Timeline:
Disabled`; Game Performance registers exactly ONE counter on this OS ("RT
Unit Active" -- raytracing, useless here). ALU/bandwidth/occupancy counter
sets need a custom template built in the Instruments GUI (interactive
session) or a Metal GPU capture from inside the process. Recorded so nobody
repeats the attempt expecting more.

## The arithmetic the profile forces (limiter by elimination)

At 32x32x4M the kernel does 2*m^2*k = 8.2 GFLOP per call; 44 ms = ~186
GFLOP/s, far under any fp32 ceiling, and traffic (512 MB at ~11.6 GB/s
effective) is far under the ~120 GB/s wall -- at Maximum clock. Neither
FLOP nor DRAM binds. The remaining candidate is the accumulation loop's
SHARED-MEMORY read rate: the current cell ownership is strided singles, so
each FMA consumes ~2 shared reads (ratio 0.5 FMA/read). A register-tile
remap (each thread owns an r x c cell RECTANGLE: r+c shared reads feed
r*c FMAs -- 2x2 gives 1.0, wider gives more) raises the ratio without
touching per-cell accumulation ORDER (each cell still sums k-ascending),
so the FNV bit-identity gates remain enforceable. This is the
"cell-ownership remap" LANE_splitk-interior priced as skip-pending-profile;
the profile now funds it.
