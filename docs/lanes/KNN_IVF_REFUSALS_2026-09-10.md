# IVF rank capacity and fused CDNA logical groups

These changes close two source-level refusals in IDENTICAL. They do not
establish physical CDNA bit identity or a new performance result.

IVF's shared exact selector already had a strided rank pass through 1024;
its launcher retained the old 256 check and omitted the now-required rank
capacity parameter. The launcher now uses capacity 256 for existing k and
1024 only for the extended IDENTICAL range. Requests above 1024, or above
the candidate count, remain named refusals. Coarse selection inherits the
same mode-specific capacity. FAST and DETERMINISTIC retain the 256 limit.
Stale IVF references to the missing
`archive/research/UNSUPERVISED_IDENTITY.md` now name the actual
`IDENTITY_PATHS.md` ledger; obsolete owed-item numbering was removed.

The public IVF gate plants 1057 points into four lists, probes all lists,
and checks k = 257 and k = 1024 over two distinct queries against a stable
integer squared-distance sort. Repeated points force ties across original
indices. All 2562 returned indices and raw distance words must match;
k = 1025 must reach the rank-capacity refusal. A separate public gate
probes 257 tied centroids and requires all 1057 candidates plus the lowest
257 original indices. Both gates pass on Apple, as does the full existing
IVF regression suite.

The fused queue keeps its 32-lane sorting topology. IDENTICAL CDNA uses
two aligned logical groups per 64-lane wave: lane IDs modulo 32, XOR
offsets below 32 with explicit masks for each half of the wave, scoped
integer any-votes, and indexed broadcasts from the corresponding physical
base plus source lane. Each cell's floating contraction, feature order,
distance epilogue, and queue comparator are unchanged. Explicit FUSED can
therefore reach this implementation on the 64-lane declaration. AUTO
remains pinned to the existing tiled IDENTICAL path. Other numeric modes
and unsupported widths retain their previous entry refusals.

The fused gate checks query counts 1, 9, and 17 with k = 1, 8, 32, and 64.
It uses different queries in neighboring logical groups, duplicate vectors,
and independent integer-distance ordering. It also checks all 2048 mappings
from 64 virtual lanes to their 32 logical broadcast sources. Native Apple
and Apple with a declared AMD column pass all twelve cases. The latter is
real Metal execution of the scoped implementation plus address emulation,
not physical CDNA execution.

The full identity driver also exercises AUTO's separate tiled small-k
selector. In a simulation of the AMD column, that selector must use its
shared integer tree through `MOJOLEARN_KNN_IDENTICAL_TREE_SELECT`; its
physical 64-lane shuffle cannot run on a 32-lane device. The original Apple
simulation omitted this flag and failed. The corrected simulation passes,
while the explicit fused logical-group implementation remains active. Both
the original failure and corrected result are archived. Actual CDNA
compilation is recorded separately when available; physical device
validation remains **RUN OWED**. A strict compilation flag rejects a
simulated target.

The [Mojo shuffle_idx contract](https://mojolang.org/docs/std/gpu/primitives/warp/shuffle_idx/)
specifies source lane IDs and an explicit participation mask. Keeping
source lane 31 or an any-vote across the full physical wave would mix the
two queues on CDNA. The earlier standalone UInt64 minimum primitive did
not, by itself, close this bitonic queue: its votes, broadcasts, and lane
mapping also needed explicit scope.

Production source commits: IVF `163ff740`, fused `4f201954`.
Gate commits: `1241294b`, `b3dbd110`, and `70c234b5`.
Evidence: `bench/results/knn/2026-09-10-ivf-fused-refusals/`.
Root owns local builds and backend compilation. This lane rented no GPU,
edited no tree source, and reran no opponent benchmark.
