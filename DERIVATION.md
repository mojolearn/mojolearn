# Derivation: the record

`NOTICE` is the Apache 2.0 section 4 attribution file and it is deliberately
short, because that is what the upstreams themselves do. This file is the
record behind it: what is original here, what follows someone else's design,
which commit each lane was read at, and the two questions that are open
rather than settled.

**Nothing here is a licence.** The licence texts are in
this file, which names every upstream, its licence and its pinned commit in
one table.
`NOTICE` names the projects. This file explains them. The per-file
correspondence is finer than either: each lane's `DERIVATION_MAP.tsv` maps one
of our files to the upstream file it follows and records whether the port is
transliterated, partial, reimplemented or replaced, and each lane's
`NOT_IMPLEMENTED.tsv` records what was left out on purpose. **Where this file
and a map disagree, the map is right.**

Created 2026-08-31, when `NOTICE` was cut from 392 lines to its licence core.
No sentence was deleted in that pass; the prose moved here.

## What is original to this work

MOVED HERE FROM `NOTICE` 2026-08-31. It was written into `NOTICE` earlier the
same day to fix a real defect, that the file stated derivation without ever
stating proportion. It does not belong in a licence file, and `README.md` and
`CONTRIBUTION.md` carry the argument at more length. It is kept whole.

THIS SECTION WAS ADDED 2026-08-31, AND ITS ABSENCE WAS A DEFECT IN THIS FILE.
Every word below the line was accurate and stays. What was missing was any
statement of PROPORTION, so a reader met "Derived from CatBoost" as the first
heading and reasonably concluded this was a port with some additions. It is
not, and the number is the shortest way to say so.

    Mojo tracked in this repository                   431,889 lines
    has an upstream file it corresponds to     127,712 to 141,739   30% to 33%
    has none                                   290,150 to 304,177   67% to 70%

Measured 2026-08-31 by line count over tracked `*.mojo`. THE RANGE IS HONEST
RATHER THAN EVASIVE, and the first published version of this table was WRONG
because it did not have one. That version said 15.1% derived and 84.9% not,
computed as "everything under a `ported/` directory" -- and `gbdt/`, the
CatBoost mirror this file's first derivation section is about, HAS NO
`ported/` SUBDIRECTORY. It missed 61,888 lines of the most plainly derived
code in the repository. Corrected within the hour, before the release it was
written for reached Zenodo. (The two `ported/` above are QUOTATIONS of that
wrong version. On 2026-08-31 `ported/` became `impl/`, `mojo_only/` became
`checks/`, `PORTED_MAP.tsv` became `DERIVATION_MAP.tsv` and `UNPORTED.tsv`
became `NOT_IMPLEMENTED.tsv`; every live path in this file is the new
spelling and was checked to exist on disk.)

The lower bound counts every `impl/` directory plus all of `gbdt/`. The
upper adds `ensemble/`'s cuML-mirroring code, less its own original half.
The remaining judgement is inside `gbdt/`, where 76 of 149 files appear in no
derivation map and three say NO CATBOOST COUNTERPART in their own headers;
those are counted as derived in both bounds, so both bounds are generous to
the derivation and the true original share is at the high end of the range.

None of the following exists in any upstream this file names, because none of
those projects was ever asked to run anywhere but CUDA:

  * THE DETERMINISM LADDER. Three numeric tiers selected at runtime, whose
    strictest tier produces bit-identical answers on Apple Silicon via Metal,
    NVIDIA via CUDA and AMD via ROCm from ONE source, with the pinned
    arithmetic primitives (`ftz`, `identical_mul_add`, `portable_sqrtf` and
    the rest of `checks/numerics.mojo`) that make it possible.
  * THE METAL BACKEND, and every deviation Metal forced: no threadgroup float
    atomics, no streams, a 32-lane column against CDNA's 64.
  * THE VERIFICATION METHOD. Per-stage identity cards diffed across three
    vendors at pinned commits, host oracles written independently of the
    device code, and the sabotage discipline that requires each gate to be
    shown capable of failing before its pass is believed.
  * THE HOST CONTROL PLANE. Every `estimator.mojo`, the CPython bindings, the
    Python surface and its refusals.
  * THE BIT-IDENTICAL FP32 GEMM CONTRACT, and the transformer and Mamba
    identity work, which are ported from nothing.

## No upstream source is in this repository

MOVED HERE FROM `NOTICE` 2026-08-31, which keeps the first three lines of it
and drops the argument.

THE SHIPPED LIBRARY IS 100% MOJO. There is no CUDA and no C++ in it. Not one
line of CatBoost, cuML, cuVS, RAFT or FAISS source appears anywhere in the
tree, and none of those projects contains any Mojo.

The only C or C++ tracked here is twenty files under `tools/*_oracle/` and
`ensemble/tools/*_oracle/`. Those are TEST HARNESSES: they link an upstream,
or a published reference implementation, to produce reference numbers the
Mojo is checked against. They build the check, not the library. The two that
are somebody else's file verbatim are declared by name further down.

So the sections below do not describe copying, and nothing in this file
should be read that way. They describe implementing a named algorithm in
Mojo against a named reference at a pinned commit, and citing that reference
the way a paper cites prior work. Apache 2.0 section 4 requires the citation
for a derivative work, and the citation is also what makes the originality
claim in the next section checkable instead of merely asserted.

WHERE THE FOLLOWING IS CLOSE AND WHERE IT IS NOT is recorded per file rather
than claimed in bulk. A file-by-file audit on 2026-08-31 of the 98 rows
labelled `transliterated` found the label right on 52 and wrong on 34, and
those 34 are being relabelled. 105 of the 459 files with an upstream
counterpart could not follow the reference even in principle, because the
upstream line they correspond to is a call into CUB, Thrust, cuBLAS,
cuSOLVER, cuRAND or a warp intrinsic that does not exist on this target;
that code was written, not followed. See CONTRIBUTION.md.

## Every attribution fact, in one table

This table exists so that cutting `NOTICE` cannot lose anything. Copyright
holder, licence, the commit each lane was read at, and what derives from it.
Licences are named per row in the table above.

| upstream | copyright | licence | pinned at | what derives |
|---|---|---|---|---|
| CatBoost | Copyright 2017-2026 YANDEX LLC | Apache-2.0 | `54a8143a` | all of `gbdt/`; plus two files redistributed VERBATIM, see below |
| cuVS | Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES | Apache-2.0 | `2140532c`; later lanes `94c2819`, `6ba2ce2` | `cluster/impl/`, `ivf/impl/` |
| cuML | Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES | Apache-2.0 | `00094f7` (branch-25.08); later lanes `265b9da6` (v26.08.00) | `dbscan/`, `decomposition/`, `neighbors/`, `glm/`, `isolation_forest/`, `hdbscan/`, `svm/`, `arima/`, `holtwinters/`, `tsa/`, `extratrees/` |
| RAFT | Copyright (c) NVIDIA CORPORATION & AFFILIATES | Apache-2.0 | `661a3b8`, `9aa17e5`, `ebf92684` | 53 rows over 52 files in 16 lanes, below |
| FAISS, via RAFT's vendored copy | Copyright (c) Facebook, Inc. and its affiliates. | **MIT** | RAFT `9aa17e5` | the two `neighbors/impl/.../faiss_select/` files |
| scikit-learn | Copyright (c) 2007-2026 The scikit-learn developers. All rights reserved. | **BSD-3-Clause** | `77def0ed6e3beab57244885d2a584470e96c103d` (1.9.0) | `extratrees/checks/host_splitter.mojo`, `extratrees/impl/decisiontree/batched_levelalgo/objectives.mojo` |
| HuggingFace transformers | Copyright 2018- The Hugging Face team. All rights reserved. | Apache-2.0 | `d56c55bf564ddb176759eb6ec199442682564916` (5.16.0.dev0) | `transformer/impl/.../modeling_llama.mojo`, `mamba/impl/.../modeling_mamba.mojo` |
| state-spaces/mamba | Copyright 2023 Tri Dao, Albert Gu | Apache-2.0 | `e9594ce1c732d97440f0332fdc43170a2294dbfa` | `mamba/impl/mamba_ssm/` |
| Modular MAX kernels | Copyright (c) 2026, Modular Inc. All rights reserved. | Apache-2.0 **with LLVM Exceptions** | `10d978e3c783ef940d1d30d0a10852b69fe285c8` | kernel shape only in `mamba/impl/mamba_ssm/ops/selective_scan_interface.mojo` |
| NVIDIA cuRAND | Copyright 2010-2014 NVIDIA Corporation | **PROPRIETARY, not open source** | 10.3.10.19 headers | `isolation_forest/impl/curand/`; open question, below |

Two upstream copyright notices travel INSIDE our source, because the upstream
files carry their own beyond the project's:

* `modeling_llama.py`: Copyright 2022 EleutherAI and the HuggingFace Inc.
  team. All rights reserved. This code is based on EleutherAI's GPT-NeoX
  library and the GPT-NeoX and OPT implementations in this library.
* `modeling_mamba.py`: Copyright 2024 state-spaces/mamba org and HuggingFace
  Inc. team.

## Why `NOTICE` is short

Apache 2.0 section 4(d) requires reproducing the attribution notices **from
the NOTICE files** of the works you derive from. Measured 2026-08-31 against
the pinned checkouts, none of them has one:

| upstream | NOTICE file | licence |
|---|---|---|
| CatBoost | none (asserted since 2026-08-19; not re-verified, no checkout on this machine) | Apache-2.0 |
| cuVS | none | Apache-2.0 |
| cuML | none | Apache-2.0 |
| RAFT | none | Apache-2.0 |
| scikit-learn | none | BSD-3-Clause |
| HuggingFace transformers | none | Apache-2.0 |
| state-spaces/mamba | none | Apache-2.0 |
| Modular (MAX kernels) | none | Apache-2.0 with LLVM Exceptions |

So 4(d) is close to vacuous here. What binds is 4(a), give recipients the
licence, discharged by `LICENSE`; 4(b), mark
changed files; and 4(c), retain the notices present in the source form.

**How 4(b) and 4(c) are carried, measured 2026-08-31.** 1,093 tracked files
open with an `SPDX-License-Identifier` line and Andrew Hendel's copyright, and
6 carry a third party's copyright as well, which are the files that need one:
the two FAISS-derived files reproduce Facebook's notice and the MIT permission
text in full. Per-file correspondence to an upstream file and line lives in
each file's own docstring and in the lane `DERIVATION_MAP.tsv`. An earlier
pass had added a separate `Derivative work:` pointer comment to 481 files;
those were removed on 2026-08-31 to match cuML's shape, and no provenance was
lost with them because the docstrings and the maps already carried it.

**For comparison, measured against the pinned checkouts on 2026-08-31.** cuML
derives from scikit-learn, FAISS, UMAP and H2O4GPU and discharges the whole of
it with four bare licence files in `thirdparty/LICENSES/` and nothing else:
0 of its 192 source files carry a non-NVIDIA copyright, its UMAP port
included, which opens with two lines naming only NVIDIA. RAFT: no NOTICE, and
0 of 263 headers, including the FAISS warp-select it vendors. scikit-learn: no
NOTICE, and 5 of 653 `.py` files carry any copyright line at all. This project
attributes at or past that norm, and the LENGTH of the old `NOTICE` was itself
misleading: a reader met eleven pages of derivation language and reasonably
concluded the library was a port with additions. It is not.

## CatBoost

`gbdt/` mirrors CatBoost's GPU oblivious tree learner file for file, under
the rule COPY DO NOT IMPROVE, because a port that drifts cannot be checked
against the original. The tree-growing design, the packing policies, the
histogram strategies, the smaller-sibling rule and the level loop are
YANDEX's. `DERIVATION_MAP.tsv` at the repository root names the source behind
each file.

This section's scope was once written as "everything under `ported/`", which
was backwards: not one of those directories is the CatBoost lane, and
`gbdt/` never had one. The scope is `gbdt/`.

**The two verbatim files.** `tools/permutation_oracle/mersenne64.h` and
`tools/permutation_oracle/mersenne64.cpp` are not a translation. They are
CatBoost's own `util/random/mersenne64.{h,cpp}` at commit 54a8143a,
redistributed under the same Apache License, Version 2.0, and changed in two
ways which section 4(b) requires be recorded: two `util/` includes are
replaced by a local `shim.h` that supplies `ui64` and `ULL`, and the
`IInputStream` constructor is removed because it needs their stream library
and nothing here calls it.

They are not part of the library. They build a test oracle, the check that
`gbdt/data/permutation.mojo` reproduces their random permutation, and they
are included verbatim on purpose: an oracle written from their code is
evidence, and one paraphrased from it is a second guess.

## cuVS

Everything under `cluster/impl/` is translated from CUDA C++ and Cython.
`cluster/DERIVATION_MAP.tsv` maps each file to its cuVS source;
`cluster/NOT_IMPLEMENTED.tsv` records what was deliberately left out.

Files under `cluster/checks/` are **not** derived from cuVS. They replace
calls cuVS makes into RAFT and into NVIDIA cuBLAS. No RAFT or cuBLAS source
was translated: what those files reproduce is the call site and its
documented semantics, and each one names the call it stands in for.

## cuML

Added to `NOTICE` on 2026-08-20, when its absence was found to be an unmet
obligation rather than a formatting gap: cuML is Apache-2.0, sections 4(a)
and 4(b) apply to every derivative work, and eight files here were derivative
works of cuML that `NOTICE` did not name.

At cuML commit `00094f7` (branch-25.08), unless the lane map says otherwise:

    dbscan/impl/dbscan/vertexdeg/algo.mojo
    dbscan/impl/dbscan/corepoints/compute.mojo
    dbscan/impl/dbscan/adjgraph/algo.mojo
    dbscan/impl/dbscan/runner.mojo
    dbscan/impl/dbscan/dbscan.mojo
    dbscan/impl/dbscan/mergelabels/runner.mojo
    decomposition/impl/linalg/detail/pca.mojo
    decomposition/impl/linalg/detail/tsvd.mojo
    neighbors/impl/knn/knn.mojo            (added 2026-08-23, the k-NN
        classifier and regressor)
    neighbors/impl/selection/knn.mojo
    glm/impl/glm/ols.mojo                  (cpp/src/glm/ols.cuh; the
        dispatch and guards)
    glm/impl/glm/ridge.mojo                (added 2026-08-23,
        cpp/src/glm/ridge.cuh: ridgeSolve, ridgeEig, ridgeFit)
    glm/impl/linear_model/qn.mojo          (added 2026-08-23,
        cpp/include/cuml/linear_model/qn.h)
    glm/impl/glm/qn/qn.mojo                (added 2026-08-23, and the
        seven below: cpp/src/glm/qn/, the L-BFGS logistic solver)
    glm/impl/glm/qn/glm_base.mojo
    glm/impl/glm/qn/glm_logistic.mojo
    glm/impl/glm/qn/glm_regularizer.mojo
    glm/impl/glm/qn/qn_solvers.mojo
    glm/impl/glm/qn/qn_linesearch.mojo
    glm/impl/glm/qn/qn_util.mojo
    glm/impl/glm/qn/simple_mat/dense.mojo

Later lanes added more; `isolation_forest/`, `hdbscan/`, `svm/`, `arima/`,
`holtwinters/`, `tsa/` and `extratrees/` all carry cuML rows and several pin
cuML at `265b9da6` (v26.08.00) instead. The maps are the authority.

`decomposition/checks/jacobi_eigh_device.mojo` is marked SUBSTITUTE rather
than derived. It stands in for the `COV_EIG_JACOBI` arm of cuML's
`tsvd.cuh::calEig`, which calls cuSOLVER; no cuML source was translated into
it, and it reproduces the call's documented semantics rather than its code.

## RAFT, and the recount

**Corrected twice.** On 2026-08-20 the `NOTICE` section named two files; that
was raised to eleven derivation rows across four lanes, and the single RAFT
commit it gave was split into the two the lanes were actually pinned to.

**Corrected again 2026-08-31, and the undercount was large.** The sentence
being replaced said "Eleven files across `dbscan/`, `decomposition/`, `glm/`
and `neighbors/`". What corrected it was a recount straight out of the
derivation maps. Across the 28 `DERIVATION_MAP.tsv` files in the tree, **54
rows name a RAFT source in their upstream column, in sixteen lanes, not
four.** One of those 54, `svm/checks/device_select.mojo`, names RAFT only
among the calls it stands in for and its upstream column is `none`, so 53
rows are derivation rows, over 52 distinct files
(`neighbors/impl/matrix/detail/select_warpsort.mojo` has two rows, one per
RAFT header it draws on).

The twelve lanes the section had never named are `cholesky/`, `ensemble/`,
`extratrees/`, `gemm/`, `hdbscan/`, `hierarchy/`, `kernel_methods/`,
`metrics/`, `solver/`, `spectral/`, `svm/` and `core/` (reached through
`resample/`). `metrics/` alone has thirteen rows, more than the whole section
used to claim.

**These counts move as lanes land.** The RAFT row count rose from 53 to 54,
and the map count from 26 to 28, during the single pass that wrote this
section, because other lanes were landing maps at the same time. The counts
below are the measurement of 2026-08-31 and the maps are the authority; the
command is `awk -F'\t' '!/^#/ && NF>1 && tolower($2) ~ /raft/'` over every
`DERIVATION_MAP.tsv`.

| lane | rows | lane | rows | lane | rows |
|---|---|---|---|---|---|
| `metrics/` | 13 | `spectral/` | 7 | `neighbors/` | 6 |
| `dbscan/` | 5 | `solver/` | 4 | `cholesky/` | 3 |
| `glm/` | 3 | `hierarchy/` | 3 | `svm/` | 3 |
| `extratrees/` | 1 | `gemm/` | 1 | `hdbscan/` | 1 |
| `kernel_methods/` | 1 | `decomposition/` | 1 | `core/` (via `resample/`) | 1 |
| `ensemble/` | 1 | | | | |

Counted by the upstream column and not by a grep for the word: `arima/`,
`cluster/`, `holtwinters/`, `ivf/` and `tsa/` also say RAFT in their maps, but
only in the notes, where they are naming a `RAFT_FAIL` macro or a RAFT call
written out at its call site. Those are not derivations and are not counted.

**Three RAFT commits are in use**, not one and not two. Which lane uses which
is in that lane's map header:

| commit | lanes |
|---|---|
| `661a3b8` (branch-25.08) | `dbscan/`, `decomposition/`, `extratrees/`, `gemm/`, `glm/` (svd, matrix), `hdbscan/`, `hierarchy/`, `neighbors/` |
| `9aa17e5` | `glm/` (lstsq) |
| `ebf92684` (v26.08.00) | `cholesky/`, `kernel_methods/`, `metrics/`, `solver/`, `spectral/`, `svm/`, `core/philox.mojo` |

**One pin is in conflict and is flagged rather than quietly picked.**
`neighbors/DERIVATION_MAP.tsv` says "RAFT pinned at 661a3b8". The file list
`NOTICE` carried until today, and the FAISS record below, put the four
`neighbors/` top-k and faiss_select files at `9aa17e5`. Both statements were
made in this project and both are left standing until the lane owner settles
it, because deleting either would be guessing. `upstream/raft` on this machine
is `661a3b8` and `upstream/raft-v26.08.00` is `ebf92684`; there is no
`9aa17e5` checkout here to read the answer off.

The files the old section named, path-corrected. They are a **subset** of the
51:

    at RAFT 661a3b8 (branch-25.08):
      dbscan/impl/neighbors/epsilon_neighborhood.mojo
      dbscan/impl/dbscan/adjgraph/algo.mojo
      dbscan/impl/label/merge_labels.mojo
      dbscan/impl/label/classlabels.mojo
      dbscan/impl/sparse/detail/csr.mojo
      neighbors/impl/label/classlabels.mojo   (added 2026-08-23; the same
          header, the general route for arbitrary int32 class labels)
      decomposition/impl/linalg/detail/pca.mojo   (sign_flip_kernel only,
          from raft matrix/detail/math.cuh:367 signFlipKernel)
      glm/impl/linalg/detail/svd.mojo   (added 2026-08-23; svdEig from
          linalg/detail/svd.cuh)
      glm/impl/matrix/math.mojo         (added 2026-08-23; the matrix
          primitives from matrix/detail/math.cuh and addScalar from
          linalg/detail/add.cuh that ridgeSolve and svdEig call)

    at RAFT 9aa17e5:
      glm/impl/linalg/detail/lstsq.mojo
      neighbors/impl/neighbors/detail/faiss_select/select.mojo
      neighbors/impl/matrix/detail/select_radix.mojo
      neighbors/impl/matrix/detail/select_warpsort.mojo
          (from both select_warpsort.cuh and util/bitonic_sort.cuh)

`dbscan/impl/dbscan/adjgraph/algo.mojo` derives from cuML **and** from
RAFT and is listed under both.

Files that merely **stand in** for a RAFT call, under `core/` and the various
`checks/` directories, are not derived from RAFT: no RAFT source was
translated into them, and each names the call whose semantics it reproduces.

## FAISS, by way of RAFT

Added 2026-08-20. This was the first obligation found in this project that is
not Apache-2.0, and it was the one being missed. RAFT vendors FAISS's
register-resident warp-select queue, and RAFT's copy carries Facebook's
copyright and the MIT license, not NVIDIA's Apache-2.0 grant
(`raft/thirdparty/LICENSES/LICENSE.faiss`).

MIT requires the copyright notice and the permission notice to be included in
all copies or **substantial portions** of the software, and a transliteration
of the algorithm into another language is a substantial portion. The notice
is therefore in `NOTICE` and in the headers of the two files themselves.

    neighbors/impl/neighbors/detail/faiss_select/select.mojo
        <- raft/neighbors/detail/faiss_select/Select.cuh
    neighbors/impl/neighbors/detail/faiss_select/merge_network_warp.mojo
        <- raft/neighbors/detail/faiss_select/{Comparators,
           MergeNetworkUtils, MergeNetworkWarp}.cuh

`neighbors/impl/matrix/detail/select_warpsort.mojo` is **not** covered.
RAFT ships the FAISS warp-select *design* twice, as two unrelated files, and
that one is RAFT's own reimplementation under NVIDIA's copyright and
Apache-2.0. It is a RAFT derivation, not a FAISS one.

**A measurement worth recording, because it was nearly got backwards.** RAFT
`v26.08.00` (`ebf92684`) carries no source file with a Facebook notice; the
only two files naming Facebook are the bare licence files
`thirdparty/LICENSES/LICENSE.faiss` and `LICENSE.pytorch`, and the
`faiss_select/` directory does not exist at that commit at all. But our two
files derive from RAFT at `9aa17e5`/`661a3b8`, and at `661a3b8` the directory
does exist and **eight** source files under it carry
`Copyright (c) Facebook, Inc. and its affiliates.` in their own headers. The
obligation is live at the commit we derived from. A measurement taken at the
newer pin would have retired it wrongly.

## HuggingFace transformers

`transformer/DERIVATION_MAP.tsv` line 3 says in terms that "the
derivative-work language lives in NOTICE and nowhere else". Until 2026-08-31
it lived nowhere, and that was an unmet obligation rather than a formatting
gap.

At `d56c55bf564ddb176759eb6ec199442682564916` (`d56c55b`, version
5.16.0.dev0):

* `transformer/impl/transformers/models/llama/modeling_llama.mojo`, 3,143
  lines, from `src/transformers/models/llama/modeling_llama.py`. One
  Llama-shaped decoder block on the device: RMSNorm, the rotary embedding,
  eager attention, the gated MLP, and the decoder layer that drives them.
  **Partial by design** and inference only: FlashAttention, SDPA, paged
  attention and chunked prefill are out of scope, and the map says why (an
  online softmax's rescale count is an execution-plan quantity, which is a
  summation order the identity contract cannot reach by pinning a fold
  topology). The map is symbol-by-symbol with upstream line ranges.
* `mamba/impl/transformers/models/mamba/modeling_mamba.mojo`, 1,419 lines,
  from `src/transformers/models/mamba/modeling_mamba.py`. One Mamba-1 block:
  `MambaRMSNorm`, the `causal_conv1d_fn` torch fallback, the
  `mamba_selective_scan` torch fallback, `MambaMixer.forward` and
  `MambaBlock.forward`. Inference only. The recurrence itself is not in this
  file; it is the state-spaces/mamba derivation below.

Both files are changed: a different language targeting a different GPU API,
partial, and inference only.

**`mamba/` had no derivation map** when this was first written, and it was
recorded here as a bookkeeping gap: every other derivation lane carried a
`DERIVATION_MAP.tsv` and `mamba/` carried none, its per-file provenance
living only in the file docstrings. **A map landed on 2026-08-31 and the gap
is closed.** That map also revealed a tenth upstream, below.

## Modular MAX kernels

Found 2026-08-31, in the `mamba/` map on the day it landed, and nobody had
counted it. `mamba/impl/mamba_ssm/ops/selective_scan_interface.mojo`
carries a second derivation row against
`max/kernels/src/state_space/selective_scan.mojo` at Modular commit
`10d978e3c783ef940d1d30d0a10852b69fe285c8`, status `reimplemented`.

What is taken is **the kernel shape only, and explicitly not the
arithmetic**: `selective_scan_fwd_gpu` (:74-105), one thread per (batch, dim)
pair, serial over the sequence, `DSTATE = 16` held in registers, outputs
before inputs in the signature. DEVIATION 722 records what of that kernel is
taken and what is refused.

**What is refused is its rounding.** MAX follows the CUDA kernel and forms
`delta * u` then `B * delta_u`; this file follows `selective_scan_ref` and
pairs `(delta * B) * u`. Two sources, and where they disagree the reference
wins. That is why this is a second row on the file rather than a parenthesis
on the first.

The licence is **Apache License 2.0 with LLVM Exceptions**, which is not the
plain Apache-2.0 every other Apache upstream here carries; the exception text
lives at https://llvm.org/LICENSE.txt and the repository's own LICENSE names
it. Copyright (c) 2026, Modular Inc. Modular's repository ships no NOTICE
file either.

Separately from this derivation, MAX and Mojo are Modular trademarks used
under licence, which `NOTICE` records at its foot.

## state-spaces/mamba

`mamba/impl/` is 2,916 lines of Mojo written against a named upstream, and
`NOTICE` named none of it until 2026-08-31.

At `e9594ce1c732d97440f0332fdc43170a2294dbfa` (`e9594ce`):

* `mamba/impl/mamba_ssm/ops/selective_scan_interface.mojo`, 628 lines, from
  `mamba_ssm/ops/selective_scan_interface.py::selective_scan_ref` (:127-193).
  The recurrent core, on the device, forward only. Several upstream options
  (`z`, `delta_bias`, `delta_softplus`, `return_last_state=False`) are
  REFUSED by name rather than implemented, recorded as DEVIATION 723.
* `mamba/impl/mamba_ssm/modules/mamba_simple.mojo`, 848 lines, from
  `mamba_ssm/modules/mamba_simple.py::Mamba.step` (:208-253) and
  `::allocate_inference_cache` (:255-266). The decode path.

One change is worth naming because it is a departure from COPY DO NOT
IMPROVE: upstream's `step` has two arms, a torch fallback and a fused CUDA
kernel, and those two arms do not agree bitwise (the CUDA
`selective_state_update` rounds `B * (delta * u)` where the torch reference
rounds `(delta * B) * u`). This port has one arm. Mirroring the branch would
have mirrored a bitwise fork. That is DEVIATION 732.

## scikit-learn, and what does *not* derive from it

This is the second obligation in this project that is not Apache-2.0, and it
needed reading twice, because most of what cites scikit-learn here does not
derive from it. The two must not be run together.

**What does not derive from it.** `mixture/`, `gaussian_process/` and
`resample/` cite scikit-learn by file and line throughout, and each map
header says in terms that scikit-learn is the SEMANTICS reference and the
ORACLE and is never the design source. cuML, cuVS and RAFT implement none of
those three algorithms at the pinned commits; the maps record the exact
searches that establish that, down to the single case-insensitive hit for
`gaussian_mixture` anywhere in the cuML tree being an xfail entry marking a
test flaky under `cuml.accel`, which is a record that cuML does *not*
accelerate the estimator and falls through to scikit-learn. So there is no
upstream file to transliterate and nothing in those directories is one.
`kde/`, `kernel_methods/` and `decomposition/` cite it the same way, as an
oracle to check against or as the definition of what a parameter means.
Reading a reference is not copying it, and none of those rows is claimed as a
derivative work.

**What does derive from it.** Two files mirror scikit-learn arithmetic
closely enough that the BSD-3 notice condition is engaged. At `77def0e`
(version 1.9.0):

* `extratrees/checks/host_splitter.mojo`, 1,049 lines, from
  `sklearn/tree/_splitter.pyx::node_split_random` (:507-731). Its own
  docstring calls it "a TRANSCRIPTION" and carries a branch-by-branch table
  of where each upstream line went. It sits under `checks/` rather than
  `impl/` because it is a **host oracle**, and that placement records its
  role in the check, not its provenance. Its provenance is its map row and
  `NOTICE`.
* `extratrees/impl/decisiontree/batched_levelalgo/objectives.mojo`, 1,413
  lines, from `sklearn/tree/_criterion.pyx` beside cuML's `objectives.cuh`.
  The scikit-learn part is the impurity and proxy-improvement **expressions**,
  transcribed. The file states that cuML's Gini and scikit-learn's Gini are
  not the same quantity and carries both. Its map row's status is
  `cuML+sklearn` for that reason.

**Open item.** Neither of the two files carries the BSD-3 notice in its own
header today. Both carry `SPDX-License-Identifier: Apache-2.0`, which is
right for the Mojo but says nothing about the upstream.

## NVIDIA cuRAND: the one upstream here that is not open source

**This record neither clears this derivation nor condemns it.** Every other
upstream is licensed in a way that tells us exactly what to do, and doing it
is the whole of the entry. cuRAND is not. This states what is in the
repository, what the upstream licence says, and what has been done to keep
the two apart, and then stops. It is a record, not a legal opinion, and **it
is pending a decision by the author.**

**What the upstream licence says.** The cuRAND device headers are not open
source. `nvidia/curand/include/curand_kernel.h:1-17`, in cuRAND 10.3.10.19
(the `nvidia-curand-cu12` wheel, unpacked read-only at
`upstream/curand-headers`, a sibling directory **outside** this repository),
reads in part:

    Copyright 2010-2014 NVIDIA Corporation.  All rights reserved.

    NOTICE TO LICENSEE:

    The source code and/or documentation ("Licensed Deliverables") are
    subject to NVIDIA intellectual property rights under U.S. and
    international Copyright laws.

    The Licensed Deliverables contained herein are PROPRIETARY and
    CONFIDENTIAL to NVIDIA and are being provided under the terms and
    conditions of a form of NVIDIA software license agreement by and between
    NVIDIA and Licensee ("License Agreement") or electronically accepted by
    Licensee.  Notwithstanding any terms or conditions to the contrary in the
    License Agreement, reproduction or disclosure of the Licensed
    Deliverables to any third party without the express written consent of
    NVIDIA is prohibited.

**What is in this repository.**

* `isolation_forest/impl/curand/curand_kernel.mojo`, 403 lines. It
  implements XORWOW, the generator `curandState` is a typedef for, and it is
  here because cuML's isolation forest calls `curand_init` and `curand` on
  that state and those draws decide which rows and which features every tree
  in the forest sees. It cites the header by line throughout, and its map row
  names the seven upstream functions it follows: `curandStateXORWOW`
  (:150-156, Box-Muller cache fields dropped), `__curand_matvec_inplace<5>`
  (:316-334), `_skipahead_inplace` (:702-719), `_skipahead_sequence_inplace`
  (:721-736), `_curand_init_inplace` / `curand_init` (:800-847), `curand`
  (:863-874) and `_curand_uniform` (`curand_uniform.h:69-71`).
* `core/philox.mojo`, which implements Philox-4x32-10, cuRAND's
  `curandStatePhilox4_32_10_t`, reached through RAFT's `PhiloxGenerator`. Its
  **design source is RAFT**, which is Apache-2.0 and is cited by line, and the
  file is a RAFT derivation. It is named here because the state machine it
  reproduces is cuRAND's, and leaving it out would make this inventory
  incomplete.

**What is not in this repository.**

* **No cuRAND source.** The headers are not vendored. They are read from
  `upstream/curand-headers` at check time, on a machine that has them.
* **Neither of cuRAND's two 32-by-800 precalculated XORWOW tables.**
  `curand_precalc.h` carries them as 51,200 literal constants. DEVIATION 683
  **rebuilds** both from the step function by repeated squaring, in about 130
  integer matrix squarings on the host, and the rebuilt tables are what the
  kernel receives. The check that the rebuild is right
  (`isolation_forest/checks/xorwow_reference.py`) parses the header's
  tables at run time and compares word for word, all 32 sequence matrices and
  all 32 offset matrices; the header's tables themselves are never written
  into this tree. DEVIATION 751 records why the offset table needed its own
  check: cuML always calls `curand_init` with offset 0, so nothing in the
  lane stepped it, and half of DEVIATION 683's claim was unverified until a
  reference was built for it.
* **NVIDIA's own numbers, except four.** The committed TSV fixtures
  (`isolation_forest/checks/xorwow_reference.tsv` and
  `xorwow_offset_reference.tsv`) are **our** generator's output, not NVIDIA's
  tables. `curand_kernel.mojo` contains three distinct hexadecimal literals:
  `0xAAD26B49` and `0xF7DCEFDD`, the two seed-salt constants, and
  `0xFFFFFFFF`, a 32-bit mask. Beside them sit two decimal multipliers,
  1099087573 and 2591861531. **Those four are the whole of what is NVIDIA's
  own in this file**, and upstream's own comments call them "arbitrary
  nonzero values" and "arbitrary odd values" (`curand_kernel.h:779-783`).
  Every other generator constant in the file -- 123456789, 362436069,
  521288629, 88675123, 5783321, 6615241 and the Weyl increment 362437 -- is
  Marsaglia's, published. `core/philox.mojo` carries four more, and those
  four are the published Random123 round multipliers and Weyl constants
  (Salmon, Moraes, Dror and Shaw, SC11), which that file records as BSD-3
  material sitting inside an NVIDIA-proprietary header.

**What is published prior art, independent of NVIDIA.** XORWOW is
Marsaglia's, from "Xorshift RNGs", *Journal of Statistical Software* 8(14),
2003. The generator, its five-word state, its three shifts and its Weyl
sequence are in that paper. What is NVIDIA's is the header that expresses it,
the seed salt above, and the skip-ahead machinery built on the precalculated
tables.

**What is not settled.** Whether a 403-line Mojo implementation of a
published generator, written against a proprietary header and citing it by
line, is a derivative work of that header is a question this project does not
answer and must not pretend to. The four facts above narrow it. They do not
close it. It is written down so the question is visible instead of absent,
and so that whoever decides it decides it on the facts rather than on a
silence.
