# Why the non-tree algorithms live in THIS repository

Written 2026-08-19, answering one question Andrew asked: build the other
parallelizable algorithms in a new repository and merge later, or not.

**This plan's work has LANDED.** k-means, brute-force k-NN and random forest
are ported, wired, benchmarked and on the Python surface, and so are a dozen
families this plan never named. What survives is the DECISION and the three
findings that came out of executing it. The sequencing sections are deleted;
`ROADMAP.md` holds what is still owed.

## The decision: NOT a new repository

Three reasons, and the first is decisive.

**The expensive part already exists here and is debugged.** k-means does not
need a histogram; it needs a `DeviceContext`, a launch-geometry policy, a
per-backend capability table, a determinism story, an interleaved benchmark
harness, a machine lock, and packaging. Every one of those is in this tree and
has been through a compiler and a GPU. `checks/kernel_matrix.mojo`,
`checks/numerics.mojo`, `checks/interleaved.mojo`, `launch_probe.mojo` and
`tools/remote_gpu.sh` are the substrate, not the trees. Starting elsewhere
means rebuilding all of it and then reconciling two copies that drifted.

**"Merge later" is where projects go to die.** Three repositories existed
already (mojotrees, this one, bitwise-gbdt) and none had shipped. A fourth is a
fourth CI, a fourth packaging story, a fourth set of licence files, and a merge
that gets deferred until it is too expensive.

**The thing a new repo would protect is protectable more cheaply.** The concern
is real: this tree's standing rule is COPY, DO NOT IMPROVE, and k-means is a
port of nothing, so it would contaminate a controlled experiment. That is a
DIRECTORY problem, not a repository problem. **COPY, DO NOT IMPROVE scopes to
the mirrored code, not to the repository**, and `DERIVATION_MAP.tsv` and
`check_upstream.sh` keep working because they name files, not the directories
above them.

## The one sequencing rule still in force

**Never HNSW.** It is a graph traversal with a serial dependency per hop, close
to the worst case for a wide machine, and FAISS ships it CPU-only for that
reason. Build exact brute force, measure where it stops winning, and only then
consider IVF. See `ROADMAP.md`.

## Three things executing step 1 settled, none of which needed a GPU

1. **`core/` is not tree-shaped.** The fixed-point accumulator transferred from
   histograms to centroid sums with one noun changed in its overflow proof.
   That was the stated purpose of doing k-means first and it is answered.
2. **The upstream is cuVS, not RAFT.** RAFT 26.10 ships no `cluster/` and no
   `neighbors/`; both moved. The mirror is algorithms from cuVS, primitives
   from RAFT.
3. **cuVS's float32 k-means is not float32 on NVIDIA.** Its distance GEMM
   defaults to `CUBLAS_COMPUTE_32F_FAST_TF32`, ten mantissa bits. A second
   incumbent with a device-dependent number system.

And one about k-NN's selection: RAFT's top-k has two implementations, and
`matrix/detail/select_radix.cuh` contains **no warp intrinsics at all** (0
occurrences of `__shfl`, `laneId`, `__ballot`, `__popc`) against 14 in
`select_warpsort.cuh`. It synchronizes with `__syncthreads()` and counts with
CUB block collectives, which is exactly the pair Mojo provides. RAFT's own
learned dispatch prefers warpsort for every k a user actually asks for, so
radix is their second choice.
