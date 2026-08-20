# Authors and attribution

**Andrew Hendel** ([@ajhendel](https://github.com/ajhendel)) created and
maintains this project.

Every contributor retains copyright in their contribution. Git history is the
authoritative record of individual changes. The project does not require
copyright assignment.

## What we wrote, and whose designs it follows

CORRECTED 2026-08-20. This section was headed "What is ours and what is not"
and put everything under `ported/` on the "not ours" side. **That was wrong
and it undersold the work.** Every line of Mojo in this repository was written
here. Nobody at YANDEX or NVIDIA or Meta wrote any of it, and Andrew Hendel
holds the copyright in all of it.

What the upstreams own is the DESIGN: the algorithm, the decomposition into
kernels, the order of operations, the data layout. `ported/` **mirrors** those
designs, file for file, deliberately, under the rule COPY DO NOT IMPROVE,
because a port that drifts cannot be checked against the original. The word
for that relationship is mirroring, not "not ours".

The distinction that does survive is a legal one and it is narrow: mirroring a
copyrighted work closely enough produces a DERIVATIVE WORK, and Apache-2.0
section 4 attaches obligations to a derivative work no matter who typed it.
That is why [NOTICE](NOTICE) says "derivative work" in the places it does. It
is a statement about license obligations, not about who did the work, and
those two questions have different answers here:

  * **Who wrote this code?** Andrew Hendel. All of it.
  * **Whose design does it follow, and what does that oblige us to do?** The
    five upstreams below, and it obliges us to carry their license and say
    the files are changed. NOTICE does that.

And a port is not transcription. CUDA has float64, streams, warp-local
synchronization and shared-memory budgets that Metal and Mojo 1.0 do not, so
whole kernels needed constructs their source does not contain; about
thirty-six such deviations are recorded in `PORTING.md`, each one a decision
the upstream never had to make.

### The designs we mirror

**CatBoost.** `ported/` mirrors CatBoost's GPU oblivious tree learner, file
for file. The tree-growing design, the packing policies, the histogram
strategies, the smaller-sibling rule and the level loop are YANDEX's,
Apache-2.0. `PORTED_MAP.tsv` names the source behind each file and flags where
a port is partial or replaced.

**cuVS.** `cluster/ported/` mirrors cuVS's k-means and `neighbors/ported/`
mirrors its brute-force k-NN and ball cover. The expanded distance identity,
the tiling scheme, the greedy k-means++ trial rule, the two convergence tests
and the empty-cluster rule are NVIDIA's, Apache-2.0. `cluster/UNPORTED.tsv`
names what was deliberately left out.

**cuML.** `dbscan/ported/` mirrors cuML's DBSCAN and `decomposition/ported/`
mirrors its PCA and truncated SVD. NVIDIA, Apache-2.0.

**RAFT.** Eleven files across `dbscan/`, `decomposition/`, `glm/` and
`neighbors/` mirror RAFT primitives, pinned at two different commits. NVIDIA,
Apache-2.0.

**FAISS, and this one is NOT Apache-2.0.**
`neighbors/ported/neighbors/detail/faiss_select/` mirrors the FAISS
warp-select queue that RAFT vendors, which carries Facebook's copyright under
the MIT license. MIT requires the notice to travel with substantial portions,
so those two files carry it in their own headers. See NOTICE.

Until 2026-08-20 cuML had no attribution anywhere in this project and FAISS
had none either. Those were license obligations unmet, and they are the reason
this section is careful about the derivative-work language rather than
dropping it.

### The line counts

All of it is code written here. The split below is by whether a file follows
an upstream design or has no counterpart in any of them.

    follows an upstream design (`*/ported/`)      29,518 lines
    no upstream counterpart at all                27,320 lines   48.1%
      of which library code                        5,875
      of which checks and probes                  21,445

The checks are original in the fullest sense: every upstream ships its own
test suite and not one line of those was ported. Ours were written from
scratch against hand-computed expectations and against CatBoost as an oracle.
Seven real bugs were caught by them, two of which were invisible to the checks
written to catch them until the test data changed from uniform to scattered
hashed values, and a third of which was a kernel with no caller at all, found
by sabotaging the path and watching whether anything moved.

The library/checks split is not a discount on either. It is there so a claim
about the SHIPPING LIBRARY cites 5,875 and a claim about AUTHORSHIP cites
27,320 or 56,838, rather than one number quietly serving both.

Where the no-counterpart code sits:

| directory | library | checks | what |
|---|---|---|---|
| `mojo_only/` | 2,729 | 12,434 | the CatBoost port's host side |
| `core/` | 1,484 | 0 | shared kernels the sections build on |
| `cluster/mojo_only/` | 1,232 | 2,150 | RAFT and cuBLAS stand-ins for k-means |
| `decomposition/mojo_only/` | 430 | 1,308 | the Jacobi eigensolver standing in for cuSOLVER |
| `neighbors/mojo_only/` | 0 | 4,251 | k-NN verification |
| `dbscan/mojo_only/` | 0 | 1,019 | DBSCAN verification |
| `glm/mojo_only/` | 0 | 283 | OLS verification |

Most substantial single piece: the fixed-point accumulator, which serves both
the histogram flush and the k-means centroid update unchanged. It was written
on the belief that Metal has no float atomic add; that belief was false, and
it survives as the deterministic arm rather than as a forced substitution. See
`PORTING.md` item 7. The determinism ladder it supports has no counterpart in
any of the five upstreams: CatBoost tags its GPU learner non-deterministic,
and RAFT's k-NN has no tie-break at all.

**Also ours, and worth naming separately: the RAFT and cuBLAS stand-ins.**
cuVS calls out to RAFT primitives and to cuBLAS for norms, keyed reductions
and the distance GEMM. No RAFT or cuBLAS source was translated. The files
under `cluster/mojo_only/` reproduce the CALL SITE and its documented
semantics and implement the kernel themselves, and each one says which call
it stands in for.

## AI-assisted development

This project has been developed with extensive assistance from AI coding
tools. Those tools have helped draft, inspect, and integrate code and
documentation; they are not legal authors, maintainers, or accountable
decision makers.

Andrew Hendel directs the work and remains responsible for reviewing what is
accepted, accurately describing its evidence level, deciding what is released,
and responding to failures. Human contributors remain responsible for the
changes they submit. AI assistance does not weaken the project's requirements
for provenance, licensing, focused validation, reproducible benchmarks, or
honest capability claims.
