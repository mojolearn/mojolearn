# Authors and attribution

**Andrew Hendel** ([@ajhendel](https://github.com/ajhendel)) created and
maintains this project.

Every contributor retains copyright in their contribution. Git history is the
authoritative record of individual changes. The project does not require
copyright assignment.

## What we wrote, and whose designs it follows

CORRECTED 2026-08-20. This section was headed "What is ours and what is not"
and put everything under `gbdt/` on the "not ours" side. **That was wrong
and it undersold the work.** Every line of Mojo in this repository was written
here. Nobody at YANDEX or NVIDIA or Meta wrote any of it, and Andrew Hendel
holds the copyright in all of it.

What the upstreams own is the DESIGN: the algorithm, the decomposition into
kernels, the order of operations, the data layout. `gbdt/` **mirrors** those
designs, file for file, deliberately, under the rule COPY DO NOT IMPROVE,
because a port that drifts cannot be checked against the original. The word
for that relationship is mirroring, not "not ours".

The distinction that does survive is a legal one and it is narrow: mirroring a
copyrighted work closely enough produces a DERIVATIVE WORK, and Apache-2.0
section 4 attaches obligations to a derivative work no matter who typed it.
That is why [DERIVATION.md](DERIVATION.md) says "derivative work" in the
places it does. It is a statement about license obligations, not about who did
the work, and those two questions have different answers here:

  * **Who wrote this code?** Andrew Hendel. All of it.
  * **Whose design does it follow, and what does that oblige us to do?** The
    ten upstreams below, and it obliges us to carry their license and say
    the files are changed. [NOTICE](NOTICE) names them, their licence texts
    are in `third_party/LICENSES/`, and [DERIVATION.md](DERIVATION.md) holds
    the record.

And a port is not transcription. CUDA has float64, streams, warp-local
synchronization and shared-memory budgets that Metal and Mojo 1.0 do not, so
whole kernels needed constructs their source does not contain, each one a
decision the upstream never had to make.

CORRECTED 2026-08-31. This paragraph said "about thirty-six such deviations
are recorded in `PORTING.md`", and it was wrong about the number and about
the place. Recounted: **890 distinct numbered `DEVIATION` entries** across
tracked `*.mojo`, `*.md`, `*.tsv`, `*.py` and `*.txt`, of which **791**
appear in the Mojo source itself. They do not live in `PORTING.md`; they live
in per-file `DEVIATION BLOCK`s and in the lane `DERIVATION_MAP.tsv` notes,
and `PORTING.md` references only 28 of them by number. `PORTING.md` has its
own separate numbering, of which there are **147 numbered items**, running to
number 354 with gaps for entries never issued. Both counts are scope
dependent and the scope is stated so it can be re-run: distinct integers
matching `DEVIATION <n>` or a `DEVIATIONS <n>, <n>` list, over the tracked
tree with `build/`, `.pixi/` and `__pycache__` excluded.

### The designs we mirror

**CatBoost.** `gbdt/` mirrors CatBoost's GPU oblivious tree learner, file
for file. The tree-growing design, the packing policies, the histogram
strategies, the smaller-sibling rule and the level loop are YANDEX's,
Apache-2.0. `DERIVATION_MAP.tsv` names the source behind each file and flags where
a port is partial or replaced.

**cuVS.** `cluster/impl/` mirrors cuVS's k-means and `neighbors/impl/`
mirrors its brute-force k-NN and ball cover. The expanded distance identity,
the tiling scheme, the greedy k-means++ trial rule, the two convergence tests
and the empty-cluster rule are NVIDIA's, Apache-2.0. `cluster/NOT_IMPLEMENTED.tsv`
names what was deliberately left out.

**cuML.** `dbscan/impl/` mirrors cuML's DBSCAN and
`decomposition/impl/` mirrors its PCA and truncated SVD, and later lanes
added `glm/impl/`, `neighbors/impl/`, `isolation_forest/impl/`,
`hdbscan/impl/`, `svm/impl/`, `arima/impl/`, `holtwinters/impl/`,
`tsa/impl/` and `extratrees/impl/`. NVIDIA, Apache-2.0.

**RAFT.** CORRECTED 2026-08-31. This said "Eleven files across `dbscan/`,
`decomposition/`, `glm/` and `neighbors/` ... pinned at two different
commits". What corrected it was a recount out of the derivation maps
themselves: across the 28 `DERIVATION_MAP.tsv` files, **54 rows name a RAFT
source in their upstream column, in sixteen lanes**. One of those rows
(`svm/checks/device_select.mojo`) names RAFT only among the calls it stands
in for, leaving 53 derivation rows over 52 distinct files, at **three**
pinned commits, not two. The lanes are `cholesky/`, `core/`, `dbscan/`,
`decomposition/`, `ensemble/`, `extratrees/`, `gemm/`, `glm/`, `hdbscan/`,
`hierarchy/`, `kernel_methods/`, `metrics/`, `neighbors/`, `solver/`,
`spectral/` and `svm/`; `metrics/` alone has thirteen rows, more than the old
sentence claimed in total. NVIDIA, Apache-2.0. The maps are the authority and
this count moves as lanes land; it rose by one during the pass that wrote
this paragraph. The per-lane table is in [DERIVATION.md](DERIVATION.md).

**FAISS, and this one is NOT Apache-2.0.**
`neighbors/impl/neighbors/detail/faiss_select/` mirrors the FAISS
warp-select queue that RAFT vendors, which carries Facebook's copyright under
the MIT license. MIT requires the notice to travel with substantial portions,
so those two files carry Facebook's notice and the MIT permission text in
their own headers, and `third_party/LICENSES/LICENSE.faiss` carries it too.

**HuggingFace transformers.** ADDED 2026-08-31; this section named none of
it, and that was an unmet obligation.
`transformer/impl/transformers/models/llama/modeling_llama.mojo` (3,143
lines) mirrors their `modeling_llama.py` and
`mamba/impl/transformers/models/mamba/modeling_mamba.mojo` (1,419 lines)
mirrors their `modeling_mamba.py`, both at commit `d56c55b`, both inference
only and partial by design. Hugging Face, Apache-2.0.

**state-spaces/mamba.** ADDED 2026-08-31, same reason. `mamba/impl/` is
2,916 lines; its `mamba_ssm/` half mirrors `selective_scan_ref` and
`Mamba.step` at commit `e9594ce`. The recurrence design, the conv window and
the inference cache are Tri Dao's and Albert Gu's, Apache-2.0. `mamba/` had
no `DERIVATION_MAP.tsv` when this paragraph was first drafted, which was a
real gap; one landed on 2026-08-31 and the gap is closed.

**Modular MAX kernels.** ADDED 2026-08-31, and it is a tenth upstream nobody
had counted. `mamba/impl/mamba_ssm/ops/selective_scan_interface.mojo`
takes THE KERNEL SHAPE, and explicitly not the arithmetic, from
`max/kernels/src/state_space/selective_scan.mojo` at commit `10d978e`.
Modular Inc., Apache-2.0 **with LLVM Exceptions**, which is not the plain
Apache-2.0 the rest of this list carries.

**scikit-learn, and this one is NOT Apache-2.0 either.** ADDED 2026-08-31.
Two files mirror scikit-learn arithmetic at commit `77def0e` (1.9.0):
`extratrees/checks/host_splitter.mojo` (1,049 lines), whose own docstring
calls it a transcription of `node_split_random`, and
`extratrees/impl/decisiontree/batched_levelalgo/objectives.mojo` (1,413
lines), which carries their impurity expressions beside cuML's. BSD-3-Clause,
which requires the notice, the conditions and the disclaimer to be retained;
`third_party/LICENSES/LICENSE.scikit_learn` carries all three. **Everything else in this tree that cites
scikit-learn cites it as an ORACLE, not as a design source** -- `mixture/`,
`gaussian_process/`, `resample/`, `kde/`, `kernel_methods/` and
`decomposition/` -- because cuML, cuVS and RAFT implement those algorithms
nowhere at the pinned commits. Reading a reference is not mirroring it.

**NVIDIA cuRAND, which is the one upstream here that is NOT OPEN SOURCE.**
`isolation_forest/impl/curand/curand_kernel.mojo` implements the XORWOW
generator cuML's isolation forest draws from. The algorithm is Marsaglia's
and published (2003); the header that expresses it is NVIDIA proprietary and
is not in this repository. This is recorded as an open question pending a
decision by Andrew Hendel, and it is neither cleared nor conceded. The facts
are in [DERIVATION.md](DERIVATION.md) and
`third_party/LICENSES/LICENSE.curand`.

Until 2026-08-20 cuML had no attribution anywhere in this project and FAISS
had none either; until 2026-08-31 transformers, mamba and scikit-learn had
none. Those were license obligations unmet, and they are the reason this
section is careful about the derivative-work language rather than dropping
it.

### The line counts

All of it is code written here. The split below is by whether a file follows
an upstream design or has no counterpart in any of them.

CORRECTED 2026-08-31, AND THE OLD TABLE WAS NOT SLIGHTLY OFF, IT WAS FROM A
DIFFERENT REPOSITORY. It said:

    follows an upstream design (`*/gbdt/`)      29,518 lines
    no upstream counterpart at all                27,320 lines   48.1%
      of which library code                        5,875
      of which checks and probes                  21,445

Three things were wrong with it. The path `*/gbdt/` never existed and was a
bad find-and-replace of `ported/`, which is itself now `impl/`. The total
it implies, 56,838 lines, was measured when this repository held five lanes;
it now holds twenty-six. And it split a lane count against a whole-tree
count, so the 48.1% was not a percentage of anything. The current figures,
which [DERIVATION.md](DERIVATION.md) states and derives:

    Mojo tracked in this repository                   431,889 lines
    has an upstream file it corresponds to     127,712 to 141,739   30% to 33%
    has none                                   290,150 to 304,177   67% to 70%

The lower bound counts every `impl/` directory plus all of `gbdt/`; the
upper adds `ensemble/`'s cuML-mirroring code less its own original half. Both
bounds are generous to the derivation, so the true original share sits at the
high end. `DERIVATION.md` carries the reasoning.

THE LIBRARY-VERSUS-CHECKS SPLIT IS NOT RESTATED HERE, and that is deliberate.
The old sub-numbers (5,875 and 21,445) were real measurements of the
five-lane tree, but the rule that produced them was never written down, and
inventing a new rule to produce new numbers would put an unverifiable figure
in an attribution file. What survives is the reason the split existed, below.

The checks are original in the fullest sense: every upstream ships its own
test suite and not one line of those was ported. Ours were written from
scratch against hand-computed expectations and against CatBoost as an oracle.
Seven real bugs were caught by them, two of which were invisible to the checks
written to catch them until the test data changed from uniform to scattered
hashed values, and a third of which was a kernel with no caller at all, found
by sabotaging the path and watching whether anything moved.

The library/checks split is not a discount on either. It is there so that a
claim about the SHIPPING LIBRARY and a claim about AUTHORSHIP cite different
numbers, rather than one number quietly serving both. That principle stands
whether or not the numbers behind it have been re-derived.

Where the no-counterpart code sits. The per-directory table below is the
2026-08-20 measurement of the five lanes that existed then and is KEPT AS A
DATED SNAPSHOT, not as a current statement; every directory in it still
exists under these names, and the right-hand column is still true. The
current whole-directory totals are given beside it so the drift is visible.

| directory | library (08-20) | checks (08-20) | total now (08-31) | what |
|---|---|---|---|---|
| `checks/` | 2,729 | 12,434 | 66,477 | the CatBoost port's host side |
| `core/` | 1,484 | 0 | 7,652 | shared kernels the sections build on |
| `cluster/checks/` | 1,232 | 2,150 | 5,296 | RAFT and cuBLAS stand-ins for k-means |
| `decomposition/checks/` | 430 | 1,308 | 4,223 | the Jacobi eigensolver standing in for cuSOLVER |
| `neighbors/checks/` | 0 | 4,251 | 7,360 | k-NN verification |
| `dbscan/checks/` | 0 | 1,019 | 1,520 | DBSCAN verification |
| `glm/checks/` | 0 | 283 | 5,489 | OLS verification |

Those seven directories are 98,017 lines today. They are no longer most of
the no-counterpart code: tracked `*/checks/` across all twenty-six lanes is
246,793 lines.

Most substantial single piece: the fixed-point accumulator, which serves both
the histogram flush and the k-means centroid update unchanged. It was written
on the belief that Metal has no float atomic add; that belief was false, and
it survives as the deterministic arm rather than as a forced substitution. See
`PORTING.md` item 7. The determinism ladder it supports has no counterpart in
any of the ten upstreams: CatBoost tags its GPU learner non-deterministic,
and RAFT's k-NN has no tie-break at all.

**Also ours, and worth naming separately: the RAFT and cuBLAS stand-ins.**
cuVS calls out to RAFT primitives and to cuBLAS for norms, keyed reductions
and the distance GEMM. No RAFT or cuBLAS source was translated. The files
under `cluster/checks/` reproduce the CALL SITE and its documented
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
