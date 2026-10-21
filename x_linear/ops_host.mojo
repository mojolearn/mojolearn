# x_linear/ops_host.mojo -- the HOST-COLUMN row pool for the x_linear fits
# (2026-10-21). `par_rows` runs a row pass on CPU threads; only the host fit
# modules (x_linear/*_host.mojo, built into the host bindings) import it, so
# no GPU binding reaches CPU threads through x_linear/ops.mojo. The GPU fits
# run on their grid drivers and never call this.
from core.host_parallel import host_parallelize
from x_linear.ops import ROW_CHUNK, seq_rows

def par_rows[F: def(Int, Int) -> None](ref f: F, n: Int, grain: Int = ROW_CHUNK):
    """Runs f(lo, hi) over blocks of at most `grain` units covering [0, n).
    f must write only slots owned by its own units, so the block split
    (and, on the host, the thread count) cannot move a bit. The device runs
    one block."""
    comptime if is_gpu():
        f(0, n)
    else:
        # lane/neural-pass82 (2026-10-01): the blocks run on the host pool
        # (core/host_parallel.mojo, in the caller's floating-point
        # environment), one contiguous group of blocks per physical core.
        # Until this lane they ran one after another on the calling thread,
        # so every x_linear host fit's row passes (GLM, Huber, LogisticCV,
        # quantile, Bayes, ridge LOO, SGD one-vs-rest) were single threaded.
        # Every caller's f writes only its own rows' (or units') slots and
        # per-block scratch (audited again on cgr3-pr86 against the
        # cgr-linear rewrite), so the bits are the serial loop's.
        var blocks = (n + grain - 1) // grain
        var groups = min(blocks, max(1, num_physical_cores()))
        if groups <= 1:
            seq_rows(f, n, grain)
            return
        var per = (blocks + groups - 1) // groups

        def group(g: Int) {imm f, imm n, imm grain, imm per, imm blocks}:
            for bk in range(g * per, min(blocks, (g + 1) * per)):
                f(bk * grain, min(n, (bk + 1) * grain))

        host_parallelize(group, groups)


# ----------------------------------------------------------------- RNG
# splitmix64 (Steele, Lea, Flood 2014): integer only, so every target draws
# the same stream from the same seed. DEVIATION 5004 (IDENTITY_PATHS row 104):
# the shuffle maps a draw to j = draw mod (i + 1), i descending.
