# sym-ctr A/B requests (lane/apple-fast-sym-ctr, 2026-10-03)

Board lane gbdt-categorical on taxicat (M3 Ultra FAST 37.1 s; the plain symmetric fit on the same rows is 11.25 s).
Every define is FAST + Apple only and default OFF; IDENTICAL compiles main's code. Arm A is FAST with no define, arm B
adds the one define. The read-profile behind each one is docs/apple-fast/notes/sym-ctr.md. Model bits: no define is
meant to move any; a model-hash difference between the arms on the same seed is a bug to report, not a quality trade.

**MOJOLEARN_SYM_CTR_PERM_BATCH** (gbdt/methods/doc_parallel_boosting.mojo, symmetric estimation block). With a CTR
feature the fit runs 4 permutations, and per tree the symmetric path estimated the 3 non-learn permutations serially:
per permutation a fresh N-sized bins buffer, a device partition with its own readback and drain, then a Newton walk
with a drain per evaluation. The arm enqueues the 3 bins + partitions back to back behind one drain and walks the
4 Newton estimations in lock step (the batched helpers `CTR_PERM_BATCH` already uses for non-symmetric trees), with
partitioners held across trees. Expected: the largest effect, since it is the only CTR term that scales with the 500
trees (on the order of 4000 drains and 1500 N-sized allocations per fit removed). Risk: low; same kernels and partitions
per permutation, only the host interleaving changes. Deciding request; run it first.

**MOJOLEARN_CTR_PREP_SHARED** (gbdt/ctrs/fast_prep.mojo `CtrPrepFast`, gbdt/train.mojo fast walk). One CTR prep
context per fit: the binarized target uploaded once, each permutation's estimation order uploaded once, the bin builder
and history calcer scratch allocated once and reused by every (feature, permutation). Without it the fast walk builds
a fresh context per (feature, permutation), as main does. Expected: a few hundred ms of CTR prep (about 20 N-uploads, 40
drains and 120 N-sized allocations fewer). Risk: low; same launches on the same roles. Note: the fast walk itself
(entered under any CTR_* define) also drops main's fresh-order ComputeCurrentBins, whose result is the zero fill, and
re-stages no host codes per permutation.

**MOJOLEARN_CTR_SORT_ONCE** (fast_prep.mojo `freq_ctrs`). The permutation-independent FeatureFreq column is computed
from the last permutation's sorted order rather than from its own identity-order builder, radix sort and calcer (or,
by default, from the host builder). Segment lengths do not depend on the order within a category, so the integer counts
and the column are the same. The segment count is `unique_values + 1` (dense codes), which removes main's full
readback of `bins` to read one element. Expected: modest (one radix sort and about 11 host passes over N per CTR
feature). Risk: low; integer counts.

**MOJOLEARN_CTR_INDEX_FUSED** (fast_prep.mojo `borders_ctrs` device_out, `ctr_borders_from_device`, train.mojo
`_build_cindex_fused`). CTR columns stay on the device: the prior divide writes each column into its own device
buffer, the Uniform grid of each Borders column comes from a device min/max (same Float64 border arithmetic on the
same two floats), and each permutation's compressed index binarizes those buffers in place. Removes 24 N-readbacks,
the host copies, 32 N-uploads and the host flat pack for the CTR columns of every permutation. The FeatureFreq grid
(MinEntropy) still reads its column on the host. Expected: hundreds of ms to about 1 s of prep. Risk: medium (new
cindex builder variant); the index must be bit-identical, and a model-hash difference means a wiring bug.

**MOJOLEARN_CTR_ONEHOT_DEVICE** (fast_prep.mojo `device_dense_codes`, `device_target_histogram`;
models/ctr_value_table.mojo `build_ctr_tables_from_counts`). Every declared categorical column's dense-code pass on
the device: validation, codes, cardinality, denseness and per-code counts in two launches with small readbacks; the
cardinality pre-passes, the one-hot max loop and the apply-table host passes read those results; one-hot columns are
read in place like float columns. Expected: about 1 s of host loops over 4.1M rows x 5 columns (pre-pass, walk,
one-hot max, table histograms). Risk: low; integer atomics. Error messages for invalid codes differ in wording.

**MOJOLEARN_SYM_CTR_ALL** turns on all five; they compose. Expected: the sum of the above, dominated by
SYM_CTR_PERM_BATCH.

## Compile status (2026-10-03; local compiling stopped by Andrew, the M3 peer compiles)

- FAST `-D MOJOLEARN_SYM_CTR_ALL`, first attempt at 13547eaf1: rc=1, two errors, both `fast_prep.mojo` `read_column`
  mut aliasing (`self` and `self.stats`). Fixed in 4e08a1922 (`_read_column` free function). No other errors in
  that log (the parse stopped at those two, so later type errors may remain).
- Not run, compile owed: peer. FAST + SYM_CTR_ALL after the fix; FAST per define (SYM_CTR_PERM_BATCH,
  CTR_PREP_SHARED, CTR_SORT_ONCE, CTR_INDEX_FUSED, CTR_ONEHOT_DEVICE); FAST with all defines off; IDENTICAL once.
- No build has passed (rc=0) on this branch yet.
