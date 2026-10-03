# optics2: OPTICS ordering experiments for FAST on Apple (lane/apple-fast-optics2)

Board lane `optics` (AFC_FAMILY=algos, 10,000-row quad subset, min_samples=10, xi=0.05), istella decides, then taxi.
Profile: docs/apple-fast/notes/optics2.md. Every define is FAST + Apple only; IDENTICAL, the host column and a FAST
build without a define compile main's code. Every variant picks the same point each step (lowest reachability,
lowest index on a tie) and applies the same relaxation, so the ordering, reachability, core distances, predecessors
and labels must equal main FAST's words; any difference is a bug, not noise.

**MOJOLEARN_OPTICS_STEP_BATCH** (tags optics2-sb-*). Mechanism: main's ordering is two launches per step (20,000
launches at 10k rows, ~20 us each: essentially the whole ~412-514 ms fit). This runs 512 steps per launch in 8
co-resident threadgroups of 1024 threads (`optics_batch_kernel`, x_cluster/optics_fast.mojo): each thread owns its rows,
each block publishes its min key per step with a release store and the first warp acquire-spins on the 8 flags.
About n/512 = 20 launches. Expected effect: the largest win of the lane, the fit should drop toward the n^2 passes
(a few tens of ms). Risk: the protocol needs the 8 threadgroups co-resident; a bounded spin sets a fail word and the
fit reruns main's two launches a step (one extra one-word readback after the first launch), so a non-co-resident
grid costs time (slower than main), never a wrong word. A fit much slower than main means the fallback fired.

**MOJOLEARN_OPTICS_FRONTIER_DEVICE** (tags optics2-fd-*). Mechanism: one fused launch per step
(`optics_fused_kernel`): each block relaxes its rows against the step's point and emits its min key over the still
unprocessed rows for the next step, double-buffered partials. n + 2 launches instead of 2n + 1. Expected effect:
roughly half the ordering time. Risk: low (no inter-block sync). It is the conservative alternative to STEP_BATCH;
STEP_BATCH wins when both are set, so ALL does not include it.

**MOJOLEARN_OPTICS_CORE_SQ** (tags optics2-sq-*). Mechanism: no sqrt pass over the 400 MB n x n matrix (euclidean
only): `kth` runs on the squared cells, the n core values are rooted afterwards (sqrt is monotone, so the same
words), and the relaxation roots each cell it reads with `sqrt_cell`'s expression. Expected effect: small, one
0.8 GB pass saved (a few ms at most). Risk: the relaxation now does a sqrt per cell read (n^2 sqrts in total across
steps, same count as the removed pass but spread over the steps).

**MOJOLEARN_OPTICS_LIVEBUF** (tags optics2-lb-*). Mechanism: no memset of slots a kernel fills completely (the 400 MB
distance matrix, the core slot, the ordering buffers), the four outputs laid out in two paired buffers so one
synchronize reads everything back instead of three. Expected effect: small (a 400 MB memset, two syncs, fewer live
buffers). Risk: low; a slot not fully written would surface as a word difference.

**MOJOLEARN_OPTICS2_ALL** (tags optics2-all-*). STEP_BATCH + CORE_SQ + LIVEBUF. Expected: the best number if
STEP_BATCH holds; compare with optics2-sb to see what CORE_SQ and LIVEBUF add on top.

Not built: CORE_TILE (register top-k without the n x n matrix: the ordering needs the resident matrix anyway, and
kth is ~1% of the fit) and EXTRACT_DEVICE (the xi walk is a serial state machine; on the device it would be a
one-thread launch, which the GPU rules forbid; on the host it is well under a millisecond at n = 10,000).

## Compile status (2026-10-03, laptop, compile only)

- FAST + `-D MOJOLEARN_OPTICS2_ALL=1` (STEP_BATCH + CORE_SQ + LIVEBUF): rc=0 at c6f1829c2.
- FAST + STEP_BATCH alone, FRONTIER_DEVICE alone, CORE_SQ alone, LIVEBUF alone: compile owed: peer.
- FAST with no define: compile owed: peer.
- IDENTICAL: compile owed: peer.

Compiling stopped on Andrew's order (compile slots jammed); the M3 peer compiles the rest. FRONTIER_DEVICE is the
one path ALL does not instantiate (`optics_fused_kernel`), so its build is the one most likely to surface an error.
