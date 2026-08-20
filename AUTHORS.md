# Authors and attribution

**Andrew Hendel** ([@ajhendel](https://github.com/ajhendel)) created and
maintains this project.

Every contributor retains copyright in their contribution. Git history is the
authoritative record of individual changes. The project does not require
copyright assignment.

## What is ours and what is not

This distinction is the point of the repository layout, so it belongs here
rather than only in [NOTICE](NOTICE).

**Not ours.** Everything under `ported/` implements CatBoost's algorithms and
follows CatBoost's structure, file for file. The tree-growing design, the
packing policies, the histogram strategies, the smaller-sibling rule and the
level loop are YANDEX's work, Apache-2.0, and `PORTED_MAP.tsv` names the
source file behind each of ours. Where a port is partial or replaced, that
column says so.

**Not ours, second upstream.** Everything under `cluster/ported/` implements
cuVS's k-means and follows cuVS's structure, file for file. The expanded
distance identity, the tiling scheme, the greedy k-means++ trial rule, the
two convergence tests and the empty-cluster rule are NVIDIA's work,
Apache-2.0, and `cluster/PORTED_MAP.tsv` names the source file behind each of
ours. `cluster/UNPORTED.tsv` names what was deliberately left out.

**Not ours, third upstream: cuML.** `dbscan/ported/` implements cuML's DBSCAN
and `decomposition/ported/` implements cuML's PCA and truncated SVD, both
following cuML's structure. Apache-2.0, NVIDIA.

**Not ours, fourth upstream: RAFT.** Eleven files across `dbscan/`,
`decomposition/`, `glm/` and `neighbors/` derive from RAFT, pinned at two
different commits. Apache-2.0, NVIDIA.

**Not ours, fifth upstream, and NOT Apache-2.0: FAISS.**
`neighbors/ported/neighbors/detail/faiss_select/` derives from the FAISS
warp-select queue that RAFT vendors, which carries Facebook's copyright under
the MIT license. MIT requires the notice to travel with substantial portions,
so those files carry it in their own headers. See NOTICE.

CORRECTED 2026-08-20: this section said "third upstream" and named RAFT, with
two files. There are five upstreams and RAFT alone has eleven files. cuML had
no attribution anywhere in the project and FAISS had none either, which was a
license obligation unmet rather than a documentation gap. The
`PORTED_MAP.tsv` beside each section names the upstream per row, because
three of the five sections draw from more than one project at once.

**Ours.** The translation itself, which is not mechanical: CUDA has warp-local
synchronization and float atomics that Mojo 1.0 and Metal do not, so several
kernels required a construct their source does not contain. `PORTING.md`
records each such deviation and why.

Everything in these directories is original work. None of it is derived from
any upstream, and that includes the checks: the upstreams all ship their own
test suites, and not one line of those was ported. Ours were written from
scratch against hand-computed expectations and against CatBoost as an oracle.

    original, no upstream counterpart   27,320 lines   48.1% of the repository
    transliterated from the upstreams   29,518 lines

Within the original half, two kinds, because the two answer different
questions and a claim should cite the right one:

| | lines | |
|---|---|---|
| library code | 5,875 | host layer, shared kernels, the stand-ins |
| checks and probes | 21,445 | verification |

Both are ours. The split is not a discount, it is so that a claim about the
SHIPPING LIBRARY cites 5,875 and a claim about AUTHORSHIP cites 27,320,
rather than one number being quietly used for both.

The verification half is the larger one and that is deliberate rather than
embarrassing. Seven real bugs in this project were found by those checks. Two
of them were invisible to the checks written to catch them until the test data
changed from uniform to scattered hashed values, and a third was a kernel
that had no caller at all and was only found by sabotaging the path and
watching whether anything moved. A port whose correctness rests on reading
the original carefully is a port that is wrong; this is the machinery that
made it not wrong.

Where the original code sits:

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
