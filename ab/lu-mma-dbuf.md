# LU MMA two-page staging candidate, 2026-10-04

Base c4993c14e42dfeb0f01315b0ec2945ad958b83f8. Source checkpoint bf8a1ad0d.
Define MOJOLEARN_LU_FAST_MMA_DBUF, binding x_decomp; default off. M2 compilation
and M3 evidence are owed; no local compiler, GPU or timing was run.

One variant only: lfm_gemm_sub_kernel specializes KB16 and two shared pages.
Baseline KB32 uses one 17,920-byte page; candidate uses two 9,472-byte pages.
The candidate follows existing AFN ping-pong scheduling: initialize page0,
keep next-window global data in registers, compute current page, stage into
alternate page and fence. KB16 uses half the per-window staging registers;
its single barrier/window does NOT halve total barriers for fixed K relative
to baseline's two barriers per KB32 window. One initial barrier is added.
Potential gains are scheduling/register-pressure effects, not a promised
barrier-count or arithmetic-work reduction. A negative result ends this
variant; no tuning sweep or favorable replay is authorized.

Same 64x64 output tile, 128 threads, ascending 8-step MMA fragments, accumulated
values and final subtraction. Partial pivot selection, row swaps, outer/inner
panel widths, LU solve and zero-pivot handling are unchanged. A/B differ only
in staging. No TSLU/tournament pivot code or candidate ancestry is imported.
No new host numerical computation or single-block routine. FAST+Apple gate
inherits LU_FAST_MMA; IDENTICAL/other vendors and default baseline remain off.

This directly reaches lu-factor and lu-solve synthetic (8192-square), and
other callers of LU_FAST_MMA including LLE. Generic decomp GEMM and Cholesky
are untouched. Inspecting Cholesky confirms a separate ctl_mma_kernel handles
panel solve/update while potrf's outer update is separate again. The prior
triangular-SYRK change was slower (268.77 -> 288.28 ms); do not repeat it or
bundle it into this experiment. LU provides the cleaner two-row first target.

Quality uses tools/lu_fast_mma_quality.py's exact eight systems, including
board8192, odd tails, plain1000/plain2051 with row swaps and zero700. New gate
requires byte-identical LU, pivots and solutions, equal info, finite output,
and each float64 residual <= main without extra tolerance. This is a new
staging candidate, so it does not inherit the old MMA gate's requirement that
board bits differ. Original quality helper and historical evidence unchanged.

Manager M2: A empty; B -D MOJOLEARN_LU_FAST_MMA_DBUF. After staging exact source
and hashes, tools/lu_mma_dbuf_pair.py quality SOURCE w2-lu-dbuf-q-20261004.
Then tools/lu_mma_dbuf_pair.py timing SOURCE w2-lu-dbuf-q-20261004
w2-lu-dbuf-t-20261004 requires the matching PASS receipt. Helpers do no builds,
SSH, queue edits or opponent calls; directories are exclusive, no replay.

Timing worker uses exactly board lu_system(8192), seed7, 64 right-hand sides.
One lu_factor call and one solve(A,B) call per arm, in that fixed order, same
process; no warmup/repetitions. Each prints call, first-read and total times.
Full arrays are read before stopping total timer. Quality is separate and
unscored. Timings include Python/output behavior; no claim of isolated MMA
TFLOP/s. Note the solve follows factor in the same process, so pipeline warm
state is part of this fixed protocol and must accompany any board evidence.
If these data are judged incommensurable with existing board protocol, retain
as experimental evidence rather than silently rerun scored arms.

Promotion would require useful total-time gains, passing strict quality,
review of affected callers, a named _OFF rollback and both M2 builds. No
default or board change is part of this branch.
