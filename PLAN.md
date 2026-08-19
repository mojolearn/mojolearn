# Plan for the non-tree algorithms

Written 2026-08-19. Answers one question Andrew asked: build the other
parallelizable algorithms in a new repository and merge later, or not.

## Recommendation: NOT a new repository

Three reasons, and the first is decisive.

**The expensive part already exists here and is debugged.** k-means does not
need a histogram; it needs a `DeviceContext`, a launch-geometry policy, a
per-backend capability table, a determinism story, an interleaved benchmark
harness, a machine lock, and packaging. Every one of those is in this tree and
has been through a compiler and a GPU. `mojo_only/kernel_matrix.mojo`,
`mojo_only/numerics.mojo`, `mojo_only/interleaved.mojo`, `launch_probe.mojo`
and `tools/remote_gpu.sh` are the substrate, not the trees.

Starting elsewhere means rebuilding all of it and then reconciling two copies
that drifted.

**"Merge later" is where projects go to die.** Three repositories exist
already (mojotrees, this one, bitwise-gbdt) and none has shipped. A fourth is
a fourth CI, a fourth packaging story, a fourth set of licence files, and a
merge that gets deferred until it is too expensive.

**The thing a new repo would protect is protectable more cheaply.** The
concern is real: this tree's standing rule is COPY, DO NOT IMPROVE, and
k-means is a port of nothing, so it would contaminate a controlled experiment.
That is a DIRECTORY problem, not a repository problem.

## The layout that keeps the experiment controlled

    mojolearn/
      core/          shared: device, launch policy, capability matrix,
                     numerics ladder, harness, machine lock
      boosting/
        ported/      CatBoost derivative. COPY, DO NOT IMPROVE applies HERE
                     AND ONLY HERE, unchanged.
        mojo_only/   what CatBoost never had to write
      cluster/       k-means
      neighbors/     k-NN
      ensemble/      random forest

The rule scopes to `boosting/ported/`. `PORTED_MAP.tsv` and `check_upstream.sh`
keep working because they name files, not directories above them. Nothing about
the port's control weakens.

## Order, and the reasoning is dependency rather than preference

**1. k-means. Do this first, and not because it is the most valuable.**

It is the smallest thing that exercises the whole shared substrate: allocate,
launch, reduce, iterate to convergence, produce a result validated against
sklearn. It proves `core/` works on something with no histogram in it, which is
the only way to find out whether `core/` is actually shared or is quietly
tree-shaped.

Distance computation is arithmetic-dense and embarrassingly parallel, which is
the shape a GPU wants, unlike histogram building. So it should also show a
LARGER GPU win than the boosting work, on far less code. A few hundred lines.

**2. k-NN, brute force, exact.**

Shares the distance kernel with k-means, so it is mostly free afterwards.

The claim is unusually strong and worth stating carefully: at laptop scale,
exact brute-force k-NN on the GPU can beat APPROXIMATE k-NN on the CPU,
because brute force is a matrix multiply plus a top-k. We would return the
right neighbours faster than a competitor returns approximate ones, on
hardware they cannot use.

Build brute force, measure where it stops winning, and only then consider IVF.
**Never HNSW**: it is a graph traversal with a serial dependency per hop, which
is close to the worst case for a wide machine, and FAISS ships it CPU-only for
that reason. See `ROADMAP.md`.

**3. Random Forest. Blocked, and the block is real.**

RF is a delta on histograms, binning, split search and partitions. In this tree
those exist but do not yet grow a correct tree on mixed-width data, so building
RF now means building on the thing currently being debugged. It becomes cheap
the moment the port grows a tree on covtype and not before.

Worth the wait: trees are INDEPENDENT in RF, so it parallelizes across trees as
well as within them, which boosting cannot. It is a better GPU fit than the
algorithm this repository is named after.

**4. Bootstrap, permutation tests.** Trivial, slot in anywhere.

**5. Gaussian processes, UMAP, t-SNE.** Later. Both hard. See `ROADMAP.md`.

## Why 1 and 2 are the right things to do RIGHT NOW

They do not touch the tree code. The live blocker in this repository is that
mixed-width trees do not split, bisected down to `leaf_count = 2` versus `1`.
k-means and k-NN need nothing from that path, so they proceed in parallel with
the debugging rather than queueing behind it.

That is the actual argument for starting them now, and it is stronger than
"they would score well".

## Validation, which is not optional here

mojotrees produced six stages that were built, documented and unreachable.
This tree exists partly because of that. So each new algorithm carries the same
discipline from its first commit:

- **Validated against scikit-learn** on the same data, exact where the maths
  permits and within a stated tolerance where it does not. Not against our own
  previous output, which is a ratchet.
- **Reach proved by SABOTAGE, never by a digest.** Corrupt the thing that is
  supposed to matter and watch the result move. On 2026-08-19 an env-path reach
  check passed for mojotrees' packed-bins arm while nothing read the buffer;
  only sabotage caught it.
- **Arms interleaved in one process.** This box drifts 2-3x across time
  windows, so two numbers from two afternoons are not a comparison.
- **`UNWIRED.md` extended.** A row nothing reads is indistinguishable from a
  row something reads.

## What this does not change

The port's experiment still has to conclude. One tree, on covtype, timed,
against mojotrees and LightGBM. Until that number exists, "CatBoost's design is
fast on Metal" is a hypothesis, and no amount of k-means makes it a finding.

## Status, 2026-08-19

**Step 1 is built and is not yet reached.** `cluster/` exists, mirrors cuVS
`2140532c` file for file, and its call graph closes from `fit` down to every
kernel. What does not exist is a `main` that launches it, so by this tree's
own rule nothing in it is ported yet.

Three things this step already settled, none of which needed a GPU:

1. **`core/` is not tree-shaped.** The fixed-point accumulator transferred
   from histograms to centroid sums with one noun changed in its overflow
   proof. That was the stated purpose of doing k-means first and it is
   answered.
2. **The upstream is cuVS, not RAFT.** RAFT 26.10 ships no `cluster/` and no
   `neighbors/`; both moved. The mirror is algorithms from cuVS, primitives
   from RAFT, and that split is cleaner than the one this plan originally
   named.
3. **cuVS's float32 k-means is not float32 on NVIDIA.** Its distance GEMM
   defaults to `CUBLAS_COMPUTE_32F_FAST_TF32`, ten mantissa bits. A second
   incumbent with a device-dependent number system, for `bitwise-gbdt`.

Step 2, brute-force k-NN, gains a concrete answer to the one thing that
looked like a blocker: RAFT's top-k has two implementations and
`matrix/detail/select_radix.cuh` contains **no warp intrinsics at all** (0
occurrences of `__shfl`, `laneId`, `__ballot`, `__popc`), against 14 in
`select_warpsort.cuh`. It synchronizes with `__syncthreads()` and counts with
CUB block collectives, which is exactly the pair Mojo provides. RAFT's own
learned dispatch prefers warpsort for every k a user actually asks for, so
radix is their second choice; it is our only one, and the bar is not
WarpSelect on an NVIDIA card, it is `argpartition` on a CPU, because their
GPU arms do not run on this machine at all. In `PORTED_MAP.tsv` that is a
`replaced` row, the same status `partitions_reduce.mojo` carries.
