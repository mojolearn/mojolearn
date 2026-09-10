# IVF rank capacity and fused CDNA logical groups

These changes close two source-level refusals in IDENTICAL. They do not
establish physical CDNA bit identity or a new performance result.

IVF's shared exact selector already had a strided rank pass through1024;
its launcher retained the old256 check and omitted the now-required rank
capacity parameter. The launcher now uses256 capacity for existing k and
1024 only for the extended IDENTICAL range. k>1024 and k>candidate-count
remain named refusals. Coarse selection inherits the same mode-specific
capacity. FAST and DETERMINISTIC retain256 admission. Stale IVF references
to the missing archive/research/UNSUPERVISED_IDENTITY.md now name the actual
IDENTITY_PATHS.md ledger; obsolete owed-item numbering was removed.

The new public IVF gate plants1057 points into four lists, probes all lists,
and checks k257 and1024 over two distinct queries against a stable integer
squared-distance sort. Repeated points force ties across original indices.
All2562 returned indices and raw distance words must match; k1025 must reach
the rank-capacity refusal. Apple gate passed. Existing IVF regression gates
remain part of root's final validation.

The fused queue keeps its32-lane sorting topology. IDENTICAL CDNA64 uses two
aligned logical groups per wave: lane IDs modulo32, XOR offsets below32 with
explicit half-wave masks, scoped integer any-votes and indexed broadcasts
from the corresponding physical base+source. Each cell's floating contraction,
feature order, distance epilogue and queue comparator are unchanged. Explicit
FUSED can therefore reach this implementation on the64-lane declaration;
AUTO remains pinned to the existing tiled IDENTICAL path. Other numeric
modes and unsupported widths retain their previous entry refusals.

The fused gate checks query tails1/9/17 and k1/8/32/64, with different
queries in neighboring logical groups, duplicate vectors and independent
integer-distance ordering. It also checks all2048 mappings from64 virtual
lanes to their32 logical broadcast sources. Apple declared-AMD64 gate passes
all twelve cases. This is real Metal execution of the scoped implementation
plus address emulation, not physical CDNA wave64 execution. Actual CDNA
compilation is recorded separately when available; physical device validation
remains RUN OWED. A strict compilation flag rejects a simulated target.

The [Mojo shuffle_idx contract](https://mojolang.org/docs/std/gpu/primitives/warp/shuffle_idx/)
specifies source-lane IDs and an explicit participation mask; simply keeping
source31 or a full-wave any-vote would mix the two queues on physical64.
No claim is made that the earlier standalone UInt64-min primitive alone
closed this bitonic queue: its votes, broadcasts and lane mapping all needed
scoping as well.

Source commits: IVF163ff740, fused4f201954, gates1241294b, comments597fc803.
Root owns local builds and any backend compilation. No GPU was rented by this
lane, no tree source was edited, and no opponent benchmark was rerun.
