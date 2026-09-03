# The size sweep, MI325X, 2026-09-03, sha f3644a86

Three fixture sizes per lane, three alternated rounds each, on a rented
single-tenant MI325X: no laptop governor, no swap, one job on the device.
Ratio is IDENTICAL / FAST, so above 1.0 is what identity costs.

## The table

| lane | S | M | what S said | what M said |
|---|---|---|---|---|
| knn | 6.607 | **24.872** | costly | FAR costlier |
| gemv | 2.274 | **21.413** | costly | FAR costlier |
| nt | **0.610** | **6.067** | identity is FASTER | identity costs 6x |
| gram | 2.115 | 1.834 | costly | costly |
| kmeans | 1.059 | 1.101 | slight cost | slight cost |
| gbdt | **0.782** | **0.731** | identity FASTER | identity FASTER STILL |
| rf | 0.994 | 1.001 | parity | parity |
| et | 1.000 | 1.000 | parity, same bits | parity, same bits |
| dbscan | 0.968 | 0.965 | parity, same bits | parity, same bits |

**A ROW WHOSE TWO MODES RETURN THE SAME BITS STILL COSTS WHAT IT COSTS.**
Where `fast == ident bits`, identity did not change the ANSWER on this
vendor, so the ratio does not price a different result -- but a ratio above
1.0 is still time paid, and what it buys is the OTHER vendors, which a
single-vendor run cannot see. Measured both ways on this fixture family:
dbscan at 100,000 rows is 1.422x on an Apple M4 with identical output bits,
and 0.965 on an MI325X where there is no mode-dependent branch anywhere on
its reached path. So the verdict to record is the BITS and the RATIO
together, never the ratio alone.

On THIS box `dbscan` has no mode-dependent branch anywhere on its reached
path -- same kernel, same grid, same block, same reduction order, no library
call either side -- so its 3% is codegen or noise. That is a statement about
the MI325X and not about the lane: the same lane at the same 100,000 rows
costs 1.422x on an Apple M4. `nt` at tier S and `et` at both tiers are also
same-bits rows, which is the second reason tier S's 0.610 may not be read as
identity being free.

Bits verdict per lane, from each tier's `ratio.tsv` column 14:

dbscan S=fast==ident bits M=fast==ident bits; et S=fast==ident bits M=fast==ident bits; gbdt S=fast!=ident bits M=fast!=ident bits; gemv S=fast!=ident bits M=fast!=ident bits; gram S=fast!=ident bits M=fast!=ident bits; kmeans S=fast!=ident bits M=fast==ident bits; knn S=FAST HASH MOVED across legs (3 distinct) M=FAST HASH MOVED across legs (3 distinct); nt S=fast==ident bits M=fast==ident bits; rf S=fast!=ident bits M=fast!=ident bits

S: kmeans 100k, knn 20k, dbscan 20k, gram 1M, nt 4096, gemv 128, trees 1M.
M: kmeans 500k, knn 100k, dbscan 50k, gram 4M, nt 65536, gemv 2048, trees 1.5M.

## THE FIRST FINDING: a small fixture does not understate the tax, it INVERTS it

`nt` read 0.610 at 4096 rows and 6.067 at 65536 -- not a drift, a SIGN
CHANGE. `gemv` went 2.274 to 21.413 and `knn` 6.607 to 24.872. At the small
sizes the arms are 18 to 80 microseconds, which on a datacenter GPU is
launch overhead with a kernel somewhere inside it.

So "identity is free on the matrix path" was an artifact of the fixture, and
the earlier single-size columns (2026-09-03 MI325X and H100 at tier S) may
not be quoted for those lanes. THE COST OF IDENTITY ON THE MATRIX AND
NEIGHBOR PATHS IS LARGE AND GROWS WITH SIZE. That is the honest headline and
it is worse for the library than what the small fixture said.

## THE SECOND FINDING: gbdt's win is real and grows

0.782 at 1M rows, 0.731 at 1.5M: IDENTICAL is 1.28x then 1.37x FASTER than
FAST, in the opposite direction to every matrix lane, with tight bands and
different output bits, while ALSO paying a software Cephes expf and a
barrier-per-step reduction. Identity is not supposed to be free, so the FAST
arm is what is under suspicion.

**THE FIRST EXPLANATION WAS MEASURED AND IS WRONG.** `replication_for`
multiplies the device's core count, so FAST asks for ~8x the histogram
replicas IDENTICAL's pinned 32 asks for, and the effect tracking core count
(0.731 on 304 CUs, 0.938 on 132 SMs) looked like the signature. DEVIATION
2040 capped it and `tools/fast_replication_ab.sh` measured two FAST builds
alternated on an MI325X at 1,000,000 rows: gbdt 1.001, rf 1.000, et 0.996,
every row bits-equal. The cap is inert on time, and reach is not in doubt --
the binaries differ by 4,096 bytes with different md5s.

TWO CANDIDATES FROM THE SAME AUDIT REMAIN UNTESTED: `partition_stats_chunks`
reads the same pin and gives FAST 304 chunks against 32, and
`reorder_single_pass_for` routes FAST-on-AMD into a decoupled-lookback
partition kernel whose lookback walk is SERIAL ON THREAD 0 with a global
atomic ticket -- a kernel whose own file records its A/B as OWED and which
has never been measured on AMD. Until one of those is measured, why
IDENTICAL wins here is OPEN.

## THE THIRD FINDING WAS A HARNESS BUG, NOT A DEFECT IN THE GUARANTEE

The L tier (dbscan at 100,000 rows) HALTED the sweep, correctly:

    LPRICE dbscan IDENTICAL warmup 100000x8 1.708990246 c8d68ee330223a88
    LPRICE dbscan IDENTICAL 1       100000x8 1.713616972 a8c692fc32c3703b
    HASH-MOVED dbscan IDENTICAL round warmup vs 1

TWO FITS OF ONE FIXTURE IN ONE PROCESS RETURNED DIFFERENT BITS UNDER
IDENTICAL. That is the contract this library exists to keep, broken, and it
is invisible at 20,000 and 50,000 rows where dbscan sat at parity in both
tiers above. Nothing had ever run that lane at 100,000.

It is a REAL DEFECT and not an infrastructure failure: same process, same
input, same build, IDENTICAL mode.

**THE CAUSE IS IN THIS HARNESS.** `dbscan_fit_impl` returns the total
label-propagation PASS COUNT -- its own docstring says so, and
`dbscan/estimator.mojo` carries a note that the value was documented wrong
once already. This lane called it `n_clusters` and folded it into the
identity hash. The fixed point of that propagation is order-invariant, which
is the whole argument for the `Atomic.min` in `sparse/detail/csr.mojo`, but
the NUMBER OF PASSES to reach it is decided by block scheduling.

That explains all three observations without any defect in dbscan. AMD only:
the batch grid is co-resident on 304 CUs and free-running, while the same
grid runs in waves on an M4 and is stable. Larger sizes only: `passes` is
SUMMED OVER BATCHES and the batch count goes 1 -> 2 -> 5 across 20k, 50k and
100k rows, so the chance the sum differs climbs with the fixture. Apple was
stable at the same 100,000 rows in both modes, all four legs
`28b1431d8b9a627f`.

The pass count is out of the hash and printed instead. **CONFIRMATION IS
OWED on an MI325X at 100,000 rows**: if the labels alone hash equal across
two fits, the guarantee was never violated and this entry retracts to a
harness bug. Until that run, the finding stands as UNCONFIRMED in both
directions.

The Apple run also priced the lane at 1.422x with identical output bits,
against 0.965 on the MI325X -- the same lane, the same size, opposite signs
on two vendors. The L tier's remaining lanes did not run,
so gram/nt/gemv/tree numbers at the datacenter step are still owed.

## What this sweep forbids

Quoting any priced ratio without its fixture size beside it. Three lanes
changed by more than 3x between two sizes and one changed sign.
