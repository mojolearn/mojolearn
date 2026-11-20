# RUN split inventory (2026-11-12)

Planning document for wave 8: move every RUN body out of the Python-importing files in `bindings/` and
`providers/python/` into Python-free `*_run.mojo` modules, then flip the build fence so that only UNPACK/PACK
and publish files may import Python. Read-only analysis of `origin/preinteg/wave7-20261112` at `a6fdeb7dd9`;
nothing was compiled or run.

## Method and evidence

- Script: `~/mojolearn-evidence/briefs-20261112/run-split/inventory.py` (scan, classify; one process per core)
  and `aggregate.py` / `render.py` (per file, per group, fence). Outputs beside it: `units.json` (every top-level
  function and struct with its class), `files.json`, `groups.tsv`, `pystructs.tsv` (every struct holding a
  Python object, ranked by how much code uses it). Rerun: `python3 -I inventory.py <worktree> . && python3 -I
  aggregate.py <worktree> . && python3 -I render.py`.
- Scope: the 908 Mojo files under `bindings/` (740) and `providers/python/` (168) that match the fence pattern of
  `tools/python_free_dirs.sh` (`from python`/`std.python`/`providers.python` import or the `PythonObject` name).
  11,333 top-level units (10,860 `def`, 471 `struct`, 2 `trait`).
- Classes. A unit is Python-touching when its own text names `PythonObject`, `PyObjectPtr`, `Python.`/`Python()`,
  a `py_*` helper, `native_loaded_module`, `GILReleased`, `PythonModuleBuilder`, `raise_python_exception` or a
  CPython call, OR it calls (import-resolved, transitively) a Python-touching function, OR it names a struct that
  holds a Python object (directly or through such a struct).
  - **RUN**: not Python-touching. Also counted RUN (and the file made ENTANGLED): a function that touches Python
    only by naming a Python-holding struct type (111 functions; for example `pls_native._center_scale` on
    `DecompKit`).
  - **MIXED**: Python-touching and doing typed work inline: at least three lines of pointer, device, native-entry
    or typed-arithmetic work, or at least 8 typed statements making up 30 percent or more of its statements
    (control headers, `raise`, error translation and string concatenation are not counted). Sub-kind
    **MIXED-GIL** (79 functions): the only Python in the body is `with GILReleased(Python())`; the split hoists the
    GIL release into the door, as `x_linear/door_run.mojo` did.
  - **UNPACK** / **PACK**: the rest of the Python-touching functions; PACK when it returns a Python object and
    either takes none or builds one, otherwise UNPACK. A public `*_c` door that unpacks, calls RUN and packs
    usually lands in PACK; the UNPACK/PACK split is coarse and only drives the rename list in the fence section.
  - **PUBLISH**: module builders and type registration (`PythonModuleBuilder`, `add_type`, `def_py_*`, `@export`,
    `PyInit_*`). The other functions inside a `_mojolearn_*.mojo` file are classified normally: those files hold
    real doors (for example `_mojolearn_rf.mojo` has 20 RUN/MIXED functions).
  - **TEST-ONLY**: file or function named `test`/`verifier`/`selftest`/`probe`/`smoke` (6; tests live outside).
- File status (only files with RUN or MIXED units have anything to move):
  - **CLEAN**: only RUN units to move; cut-and-move plus import changes.
  - **SPLIT**: at least one MIXED function must be split (RUN body cut out, door keeps UNPACK and PACK) first.
  - **ENTANGLED**: the RUN work uses a Python-holding struct (a "kit" or "program" with Python fields, an
    owner-pinning view), a resident slot holding Python objects, or calls Python per iteration inside a typed loop
    (callback). The construct is named per file below.
  - **BOUNDARY**: nothing to move (all UNPACK/PACK/PUBLISH).
- Group: the `development_groups.json` family whose `bindings/_mojolearn_<unit>.mojo` import closure contains the
  file (identical and host units merged). A file reached by several units goes to the family its name starts with
  (`gp_*` to gp, `causal_*`/`samba_*`/`optimizer_*`/`generation_*` to runtime_neural, `mcls_*`/`normal_*`/`str_*`
  to runtime_family, and so on); the remaining shared helpers are **shared-infra**. **unreached** is
  `bindings/entry/*` (development publish files) and three unreferenced files.

## Model to copy

The doorsplit lanes (`git log --grep=doorsplit`: linear, trees, cnn-xseq, neighbors-cluster) put each RUN module in
the kernel family directory, not in `bindings/`: `x_linear/door_run.mojo` (from `bindings/x_linear_native.mojo`),
`glm/qn_door_run.mojo`, `xtrees/trees_{boost,forest,meta}_run.mojo`, `x_cnn/*_run.mojo`, `sequence/xseq_*_run.mojo`,
`neighbors/neighbors_*_run.mojo`, `x_ann/*_run.mojo`, `cluster/*_run.mojo`, `gbdt/trees_gbdt_*_run.mojo`,
`core/linear_*_run.mojo` (61 `*_run.mojo` files today, 24 of them `checks/native_tests` wrappers). The door in
`bindings/` keeps UNPACK (buffer admission, output allocation), releases the GIL around each RUN call exactly where
it did before, and keeps PACK; the RUN module imports only typed Mojo (EntryRef, entry tables, kernel modules).
Each mechanical lane does the same: `bindings/<stem>.mojo` -> `<kernel dir>/<stem>_run.mojo` (or
`bindings/<stem>_run.mojo` when no kernel directory fits), one RUN file per source file, bits unchanged.

## Totals

| | files | RUN | MIXED (of which GIL-only) | UNPACK | PACK | PUBLISH | TEST-ONLY |
|---|---|---|---|---|---|---|---|
| all 908 files | 908 | 710 | 1,010 (79) | 5,138 | 4,170 | 299 | 6 |

| file status | CLEAN | SPLIT | ENTANGLED | BOUNDARY |
|---|---|---|---|---|
| files | 52 | 196 | 185 | 475 |

Mechanical work (CLEAN + SPLIT): 248 files, 856 RUN/MIXED functions. Entangled: 185 files, 864 RUN/MIXED
functions behind a Python-holding construct; they need a design change before they can move (section
"Entanglements").

## Per group

Lane size by mechanical functions: S up to 15, M 16 to 60, L 61 to 150, XL above 150. "Functions to move" counts
RUN + MIXED in all non-BOUNDARY files (entangled included); "mechanical" only CLEAN + SPLIT files.

| group | files | RUN | MIXED (GIL-only) | UNPACK | PACK | PUBLISH | TEST | CLEAN | SPLIT | ENTANGLED | BOUNDARY | files to touch | functions to move | lines moved | mechanical files (CLEAN+SPLIT) | mechanical functions | lane size |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shared-infra | 167 | 117 | 134 (14) | 842 | 416 | 4 | 3 | 9 | 34 | 38 | 86 | 81 | 251 | 9273 | 43 | 117 | L |
| rf | 2 | 39 | 20 (8) | 37 | 21 | 4 | 0 | 0 | 2 | 0 | 0 | 2 | 59 | 2212 | 2 | 59 | M |
| runtime_family | 63 | 69 | 77 (4) | 444 | 222 | 15 | 0 | 0 | 12 | 21 | 30 | 33 | 146 | 6098 | 12 | 53 | M |
| metrics | 10 | 18 | 37 (0) | 51 | 73 | 4 | 0 | 0 | 5 | 2 | 3 | 7 | 55 | 1756 | 5 | 49 | M |
| embedding | 3 | 39 | 7 (6) | 3 | 20 | 4 | 0 | 0 | 3 | 0 | 0 | 3 | 46 | 1242 | 3 | 46 | M |
| runtime_neural | 34 | 19 | 48 (3) | 309 | 196 | 3 | 0 | 4 | 9 | 9 | 12 | 22 | 67 | 3611 | 13 | 35 | M |
| core | 11 | 16 | 13 (5) | 46 | 154 | 4 | 0 | 2 | 5 | 0 | 4 | 7 | 29 | 769 | 7 | 29 | M |
| byte_lm | 7 | 15 | 15 (0) | 65 | 117 | 5 | 0 | 1 | 3 | 2 | 1 | 6 | 30 | 925 | 4 | 26 | M |
| gbdt | 3 | 16 | 10 (4) | 12 | 25 | 4 | 0 | 0 | 3 | 0 | 0 | 3 | 26 | 1311 | 3 | 26 | M |
| runtime_decomp | 38 | 42 | 29 (5) | 231 | 78 | 4 | 0 | 5 | 8 | 13 | 12 | 26 | 71 | 2930 | 13 | 26 | M |
| ivf | 4 | 13 | 11 (0) | 12 | 14 | 4 | 0 | 0 | 3 | 0 | 1 | 3 | 24 | 635 | 3 | 24 | M |
| runtime_model_selection | 20 | 12 | 46 (21) | 129 | 125 | 1 | 0 | 1 | 1 | 5 | 13 | 7 | 58 | 1790 | 2 | 22 | M |
| gp | 29 | 12 | 45 (0) | 164 | 61 | 4 | 0 | 0 | 6 | 11 | 12 | 17 | 57 | 2720 | 6 | 21 | M |
| runtime_xseq | 22 | 1 | 25 (0) | 239 | 163 | 1 | 0 | 0 | 10 | 2 | 10 | 12 | 26 | 1473 | 10 | 21 | M |
| autograd | 23 | 7 | 16 (0) | 80 | 75 | 0 | 0 | 0 | 6 | 1 | 16 | 7 | 23 | 1647 | 6 | 21 | M |
| runtime_prep | 28 | 56 | 63 (0) | 289 | 133 | 1 | 0 | 0 | 6 | 12 | 10 | 18 | 119 | 4985 | 6 | 20 | M |
| runtime_trees | 37 | 8 | 51 (0) | 267 | 191 | 1 | 0 | 0 | 14 | 4 | 19 | 18 | 59 | 3247 | 14 | 20 | M |
| resample | 6 | 24 | 21 (0) | 20 | 39 | 4 | 0 | 0 | 3 | 2 | 1 | 5 | 45 | 2259 | 3 | 17 | M |
| estimators | 5 | 10 | 5 (0) | 31 | 98 | 4 | 0 | 0 | 2 | 0 | 3 | 2 | 15 | 291 | 2 | 15 | S |
| forest | 6 | 14 | 1 (1) | 9 | 29 | 2 | 0 | 3 | 1 | 0 | 2 | 4 | 15 | 245 | 4 | 15 | S |
| runtime_classical | 24 | 9 | 22 (0) | 171 | 55 | 5 | 0 | 2 | 7 | 4 | 11 | 13 | 31 | 1312 | 9 | 13 | S |
| tokenizer | 32 | 12 | 35 (1) | 181 | 168 | 3 | 0 | 1 | 8 | 9 | 14 | 18 | 47 | 2542 | 9 | 13 | S |
| transformer | 4 | 2 | 21 (0) | 37 | 40 | 4 | 0 | 0 | 2 | 2 | 0 | 4 | 23 | 1004 | 2 | 13 | S |
| linalg | 10 | 4 | 20 (0) | 58 | 67 | 5 | 0 | 1 | 3 | 3 | 3 | 7 | 24 | 1537 | 4 | 12 | S |
| x_sequence | 27 | 10 | 9 (0) | 190 | 116 | 4 | 0 | 2 | 6 | 2 | 17 | 10 | 19 | 763 | 8 | 12 | S |
| trees | 10 | 24 | 22 (4) | 115 | 71 | 4 | 0 | 1 | 1 | 5 | 3 | 7 | 46 | 1416 | 2 | 11 | S |
| mamba | 3 | 5 | 16 (0) | 27 | 96 | 4 | 0 | 0 | 2 | 1 | 0 | 3 | 21 | 685 | 2 | 10 | S |
| svm | 9 | 3 | 7 (0) | 19 | 38 | 4 | 0 | 0 | 2 | 1 | 6 | 3 | 10 | 797 | 2 | 9 | S |
| training | 13 | 7 | 2 (0) | 37 | 77 | 4 | 0 | 3 | 1 | 0 | 9 | 4 | 9 | 102 | 4 | 9 | S |
| kernel_methods | 3 | 3 | 5 (0) | 2 | 23 | 4 | 0 | 0 | 2 | 0 | 1 | 2 | 8 | 333 | 2 | 8 | S |
| runtime_parallel | 21 | 28 | 47 (0) | 200 | 147 | 2 | 0 | 1 | 3 | 10 | 7 | 14 | 75 | 3248 | 4 | 8 | S |
| tsa | 5 | 3 | 4 (0) | 25 | 27 | 4 | 0 | 2 | 1 | 0 | 2 | 3 | 7 | 211 | 3 | 7 | S |
| x_cluster | 10 | 4 | 10 (0) | 61 | 30 | 4 | 0 | 1 | 2 | 2 | 5 | 5 | 14 | 568 | 3 | 7 | S |
| x_linear | 8 | 2 | 7 (0) | 29 | 30 | 8 | 0 | 1 | 4 | 2 | 1 | 7 | 9 | 383 | 5 | 7 | S |
| neural | 5 | 6 | 15 (2) | 86 | 49 | 2 | 0 | 0 | 2 | 2 | 1 | 4 | 21 | 897 | 2 | 6 | S |
| x_trees | 3 | 2 | 4 (0) | 11 | 103 | 5 | 0 | 0 | 1 | 0 | 2 | 1 | 6 | 134 | 1 | 6 | S |
| arima | 4 | 3 | 3 (0) | 1 | 25 | 4 | 0 | 2 | 1 | 1 | 0 | 4 | 6 | 234 | 3 | 5 | S |
| hdbscan | 4 | 2 | 3 (0) | 7 | 14 | 4 | 0 | 0 | 2 | 0 | 2 | 2 | 5 | 265 | 2 | 5 | S |
| x_ann | 4 | 0 | 5 (0) | 10 | 28 | 4 | 0 | 0 | 2 | 0 | 2 | 2 | 5 | 195 | 2 | 5 | S |
| runtime_manifold | 12 | 11 | 12 (0) | 91 | 40 | 1 | 0 | 0 | 2 | 4 | 6 | 6 | 23 | 1961 | 2 | 4 | S |
| x_cnn | 2 | 0 | 4 (0) | 109 | 33 | 4 | 0 | 0 | 1 | 0 | 1 | 1 | 4 | 161 | 1 | 4 | S |
| mixture | 4 | 3 | 8 (0) | 26 | 22 | 4 | 0 | 0 | 1 | 1 | 2 | 2 | 11 | 500 | 1 | 3 | S |
| runtime_byte_trainer | 8 | 7 | 17 (1) | 41 | 49 | 3 | 0 | 2 | 0 | 4 | 2 | 6 | 24 | 1040 | 2 | 3 | S |
| runtime_linear | 9 | 1 | 21 (0) | 118 | 47 | 1 | 0 | 0 | 1 | 2 | 6 | 3 | 22 | 1496 | 1 | 2 | S |
| runtime_sequential | 9 | 1 | 5 (0) | 45 | 32 | 1 | 0 | 1 | 1 | 3 | 4 | 5 | 6 | 312 | 2 | 2 | S |
| solver | 2 | 2 | 0 (0) | 3 | 13 | 4 | 0 | 1 | 0 | 0 | 1 | 1 | 2 | 8 | 1 | 2 | S |
| x_decomp | 22 | 2 | 0 (0) | 35 | 116 | 4 | 0 | 2 | 0 | 0 | 20 | 2 | 2 | 16 | 2 | 2 | S |
| x_metrics | 2 | 2 | 0 (0) | 2 | 13 | 4 | 0 | 2 | 0 | 0 | 0 | 2 | 2 | 16 | 2 | 2 | S |
| embedding_infer | 1 | 1 | 0 (0) | 1 | 6 | 2 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 8 | 1 | 1 | S |
| preprocessing | 3 | 1 | 0 (0) | 5 | 25 | 5 | 0 | 1 | 0 | 0 | 2 | 1 | 1 | 6 | 1 | 1 | S |
| x_neighbors | 9 | 3 | 10 (0) | 87 | 224 | 6 | 0 | 0 | 1 | 4 | 4 | 5 | 13 | 724 | 1 | 1 | S |
| x_prep | 6 | 0 | 1 (0) | 19 | 52 | 4 | 0 | 0 | 1 | 0 | 5 | 1 | 1 | 93 | 1 | 1 | S |
| forecast | 2 | 0 | 0 (0) | 1 | 5 | 2 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | none |
| gp_infer | 1 | 0 | 0 (0) | 1 | 5 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | none |
| hdbscan_infer | 1 | 0 | 0 (0) | 1 | 5 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | none |
| ivf_search | 1 | 0 | 0 (0) | 0 | 5 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | none |
| mixture_infer | 1 | 0 | 0 (0) | 0 | 6 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | none |
| runtime_catalog | 2 | 0 | 1 (0) | 2 | 12 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 96 | 0 | 0 | none |
| tokenizer_fast | 1 | 0 | 0 (0) | 2 | 6 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | none |
| unreached | 93 | 0 | 0 (0) | 2 | 0 | 92 | 3 | 0 | 0 | 0 | 93 | 0 | 0 | 0 | 0 | 0 | none |
| TOTAL | 908 | 710 | 1010 (79) | 5138 | 4170 | 299 | 6 | 52 | 196 | 185 | 475 | 433 | 1720 | 72482 | 248 | 856 |  |

## Proposed mechanical lanes (CLEAN + SPLIT only)

Fourteen lanes of 40 to 75 functions each; every lane owns whole groups, so no file has two owners. Entangled files
in these groups are left alone by these lanes (see the next section).

| lane | groups | mechanical files | mechanical functions |
|---|---|---|---|
| L1 shared-infra A | shared-infra files named `native_*`, `py_*`, `admitted_*`, `base_*` | 13 | 50 |
| L2 shared-infra B | the other shared-infra files (`arena_*`, `array_*`, `host_*`, `pandas_*`, ...) | 30 | 67 |
| L3 forests | rf, forest | 6 | 74 |
| L4 boosting/trees | gbdt, trees, x_trees, runtime_trees | 20 | 63 |
| L5 classical family | runtime_family, runtime_classical | 21 | 66 |
| L6 metrics/estimators | metrics, x_metrics, estimators | 9 | 66 |
| L7 embedding/byte LM | embedding, embedding_infer, byte_lm | 8 | 73 |
| L8 neural | runtime_neural, transformer, mamba, neural, runtime_byte_trainer | 21 | 67 |
| L9 core/autograd | core, autograd, training | 17 | 59 |
| L10 decomposition | runtime_decomp, x_decomp, linalg, ivf | 22 | 64 |
| L11 model selection/prep | runtime_model_selection, resample, runtime_prep, preprocessing, x_prep | 13 | 61 |
| L12 kernels/linear | gp, kernel_methods, svm, mixture, x_linear, runtime_linear, solver | 18 | 52 |
| L13 sequence/tokenizer | runtime_xseq, x_sequence, tsa, arima, tokenizer | 33 | 58 |
| L14 the rest | x_cluster, hdbscan, x_ann, x_neighbors, x_cnn, runtime_manifold, runtime_parallel, runtime_sequential | 17 | 36 |

Every lane: one `*_run.mojo` per source file in the family's kernel directory, the door keeps UNPACK/PACK and the
GIL release, `python_free_dirs.sh` passes (it already checks every `*_run.mojo`), bits unchanged, bindings to
rebuild = the `_mojolearn_<unit>` and `_host` units of its groups. Deleted by each move: nothing but the moved
bodies (no switch; the old in-binding body is the thing removed).

## Entanglements (185 files, 864 functions)

The three biggest, by the RUN code that depends on them (`pystructs.tsv` has all 345 Python-holding structs):

1. **Kits and program builders with Python fields.** `DecompKit` (`bindings/decomp_kit_native.mojo`, 1,203 lines,
   used by 66 functions in 5 files: PLS, CCA, decomposition), `MKit` (`bindings/mani_kit.mojo`, 1,117 lines,
   manifold), `LinKit` (611 lines), and the stage-program builders `EncProg`, `ClsProg`, `MxProg`, `_Prog`
   (`inputs`/`store: PythonObject`; prep encode/classify, metrics). RUN code (`_center_scale`, `_pinv`,
   `_put_proba`, `_kbins_fit`'s program) runs on them. Untangle: a typed kit (buffers, dims, entry refs) for RUN
   and a door-side owner list for the Python inputs; one design lane per kit, then the family lane moves the RUN.
2. **Owner-pinning admitted views used inside RUN.** `ArrayMetadata` (115 users; 2,809 lines of RUN-classified
   code name it), `AdmittedBorrowedBuffer`, `AdmittedX`, `AdmittedTyped`, `BufferView`
   (`bindings/boundary_read.mojo`), `_RPArray` (resample pandas): each carries `owner: PythonObject` beside the
   address and shape. Untangle: RUN takes a typed view (address, shape, strides, dtype); the owner pin stays in
   the door and is named again after the RUN call (AGENTS.md: Mojo frees a local at its last named use). This is
   one shared-infra design lane; it unblocks most entangled files in runtime_prep, runtime_family and gp.
3. **The Python module cache and resident Python slots reaching RUN.** `NativeModules`
   (`_modules: Dict[String, PythonObject]`, passed through 617 functions in 120 files) and SharedSlots that keep
   Python objects across calls (`ParallelMlpBoundary`, `GENERATION_BOUNDARY`, `ShardIdle`/`ShardTrace`,
   `OptimizerInputOwners`, `ContentRecord`). Untangle: the door resolves what it needs from them before RUN; RUN
   signatures drop `mods`; resident state keeps typed handles only.

Also 14 functions call Python once per iteration inside a typed loop (callbacks), for example
`bindings/trees_public_gbdt_fit.mojo:_group_sizes`, `bindings/prep_labels_native.mojo:_int64_codes`,
`bindings/user_objects_native.mojo:_first_seen`, `bindings/native_identity_key.mojo:NativeIdentityKey`; each needs
the thin-unpack shape (one buffer read) before it can move.

## The flipped fence

Proposed rule for `tools/python_free_dirs.sh` (a third check beside the kernel-directory and door checks, same
grep pattern, no baseline): inside `bindings/` and `providers/python/`, a Mojo file that imports Python must match

```bash
boundary_allow=(
    'bindings/_mojolearn*.mojo'   # PUBLISH: module builders and type registration
    'bindings/entry/*.mojo'       # PUBLISH: development entry modules
    'bindings/py_*.mojo'          # shared CPython primitives (py_balanced, py_attr, ...), UNPACK/PACK only
    '*_unpack.mojo'               # UNPACK, and the public door: unpack, typed RUN call, PACK call
    '*_pack.mojo'                 # PACK
)
boundary_spec=(':(glob)bindings/**/*.mojo' ':(glob)providers/python/**/*.mojo')
for a in "${boundary_allow[@]}"; do
    case "$a" in */*) boundary_spec+=(":(exclude,glob)$a") ;; *) boundary_spec+=(":(exclude,glob)**/$a") ;; esac
done
boundary_hits=$(grep_paths "${boundary_spec[@]}")   # non-empty -> exit 1, list the files
```

(the no-git `find` branch needs the same allow patterns as `-path`/`-name` prunes). The existing `*_run.mojo` and
`core/*_entry_table*.mojo` denials stay.

Files that would still violate it after every CLEAN and SPLIT move (717 of 908):

| reason | files |
|---|---|
| ENTANGLED: Python-holding construct used by RUN (named per file below) | 184 |
| holds both UNPACK and PACK functions: split into `<stem>_unpack.mojo` + `<stem>_pack.mojo` | 307 |
| UNPACK only: rename to `<stem>_unpack.mojo` | 138 |
| PACK only (by the coarse rule above): rename to `<stem>_pack.mojo` | 88 |
| already allowed (`_mojolearn_*`, `entry/*`, `py_*`) | 191 |

The renames and splits are 533 files of mechanical churn on top of the RUN moves. Decision for the orchestrator:
either accept that wave (the brief's rule), or add one more allowed pattern `*_door.mojo` for the file holding a
public operation's unpack/run/pack sequence, which turns most of the 307 splits into renames. The fence can only
flip after the entanglement lanes land; until then it would list the 184 entangled files.

## Known misclassifications (spot check of 20 files)

Hand-checked: `bindings/` `base_helpers_released`, `pls_native`, `msel_public_metadata`, `_mojolearn_rf`,
`metrics_exp_rank`, `host_kernel_inference_native`, `prep_encode_native` (two functions), `gp_classifier_native`,
`neural_block_public`, `metrics_exp_curve`, `mani_kit`, `boundary_read`, `decomp_kit_native`, `admitted_family`,
`host_iforest_scalar_repr`, `pandas_arrow_native`, `tsa_arima_native`, `linalg_ops_resample`,
`parallel_mlp_native`, and `providers/python/autograd_program_control_pyapi`. 14 matched; the errors found:

- MIXED is over-counted. Of the 859 functions MIXED only by the typed-statement ratio, the samples suggest roughly
  a third are doors whose typed statements are bookkeeping, not a RUN body (`host_kernel_inference`,
  `admitted_family.admitted_array` stride setup, `autograd_program_control_pyapi.ops_program_decode_py` plan
  edits). Such a SPLIT file is really BOUNDARY; a lane that finds nothing worth cutting leaves it and says so.
- Fixed during the run, recorded so a rerun keeps them: string building (`text += ...`) counted as arithmetic
  (`msel_public_metadata._metadata_repr` was MIXED); parameters of a Python-holding struct type (`mods:
  NativeModules`) were not Python (`neural_block_getstate_binding` was MIXED); GIL-only wrappers
  (`base_helpers_released.*`, `_mojolearn_rf._rf_classifier_fit`) were UNPACK and are now MIXED-GIL.
- `boundary_read.BufferView` is ENTANGLED by the rule (Python owner beside typed reads), but it is the UNPACK view
  type itself; its fix is entanglement 2, not a family lane.
- Not seen: method calls on locals whose Python-holding type is inferred, never spelled (`var kit = make(...)`),
  and calls through methods (`obj.f(...)`) are not resolved by name; such functions may show as RUN or UNPACK
  where they are ENTANGLED or MIXED. Kernel-directory callees are not scanned (the fence already keeps Python out).
- The UNPACK/PACK split is by signature only and should not be used for anything but the rename counts.

## Per file

Files with something to move (CLEAN, SPLIT, ENTANGLED), by group. "fence after moves" is what the flipped rule
would still require of the file.

| file | group | status | RUN | MIXED | UNPACK | PACK | fence after moves | entangling construct / MIXED to split |
|---|---|---|---|---|---|---|---|---|
| `bindings/_mojolearn_arima.mojo` | arima | CLEAN | 2 | 0 | 0 | 18 | ok |  |
| `bindings/_mojolearn_arima_host.mojo` | arima | CLEAN | 1 | 0 | 0 | 7 | ok |  |
| `bindings/arima_host_predict.mojo` | arima | SPLIT | 0 | 2 | 0 | 0 | rename-unpack | split: arima_predict_binding, arima_forecast_binding |
| `providers/python/arima_buffer_pyapi.mojo` | arima | ENTANGLED | 0 | 1 | 1 | 0 | entangled | struct ArimaBuffer holds keep: PythonObject; owner: PythonObject |
| `providers/python/autograd_import_frontend_pyapi.mojo` | autograd | SPLIT | 0 | 1 | 1 | 2 | split-unpack-pack | split: _import |
| `providers/python/autograd_ops_frontend_pyapi.mojo` | autograd | SPLIT | 3 | 1 | 10 | 1 | split-unpack-pack | split: ops_session_call_fields_py |
| `providers/python/autograd_owned_program_public_pyapi.mojo` | autograd | ENTANGLED | 0 | 2 | 7 | 5 | entangled | struct _OwnedProgramConnection holds classes: PythonObject; layouts: PythonObject; struct NativeOwnedProgramRequest holds owner: PythonObject; operation: Python |
| `providers/python/autograd_program_control_pyapi.mojo` | autograd | SPLIT | 0 | 2 | 0 | 0 | rename-unpack | split: ops_program_control_py, ops_program_decode_py |
| `providers/python/autograd_program_pyapi.mojo` | autograd | SPLIT | 3 | 7 | 11 | 4 | split-unpack-pack | split: _array_input_buffer, array_input, _sequence_values, program_input +3 |
| `providers/python/autograd_reader_frontend_pyapi.mojo` | autograd | SPLIT | 0 | 2 | 3 | 8 | split-unpack-pack | split: reader_read, reader_resolve |
| `providers/python/autograd_weight_frontend_pyapi.mojo` | autograd | SPLIT | 1 | 1 | 0 | 3 | rename-pack | split: _record |
| `bindings/_mojolearn_byte_lm.mojo` | byte_lm | SPLIT | 14 | 9 | 14 | 47 | ok | split: _byte_config, byte_lm_parallel_step_binding, byte_lm_parallel_export_binding, byte_lm_model_pool_open_binding +5 |
| `bindings/_mojolearn_byte_lm_host.mojo` | byte_lm | SPLIT | 0 | 1 | 3 | 12 | ok | split: _host_config |
| `bindings/lm_corpus_files.mojo` | byte_lm | ENTANGLED | 0 | 1 | 15 | 7 | entangled | struct CorpusFileConnection holds entries: PythonObject; path_class: PythonObject |
| `bindings/lm_corpus_public_native.mojo` | byte_lm | ENTANGLED | 0 | 3 | 13 | 19 | entangled | struct LmCorpusIteratorHandle holds owner: PythonObject; source: PythonObject |
| `bindings/lm_corpus_scalars.mojo` | byte_lm | SPLIT | 0 | 1 | 2 | 0 | rename-unpack | split: _floating_base |
| `providers/python/byte_lm_host_compat_pyapi.mojo` | byte_lm | CLEAN | 1 | 0 | 9 | 27 | split-unpack-pack |  |
| `bindings/_mojolearn.mojo` | core | SPLIT | 7 | 3 | 21 | 39 | ok | split: kmeans_fit_binding, kmeans_fit_transform_binding, _encode_labels_binding |
| `bindings/_mojolearn_core_host.mojo` | core | SPLIT | 2 | 1 | 6 | 22 | ok | split: kmeans_fit_binding |
| `bindings/array_helpers.mojo` | core | CLEAN | 2 | 0 | 0 | 8 | rename-pack |  |
| `bindings/hotpath_device.mojo` | core | SPLIT | 1 | 6 | 1 | 22 | split-unpack-pack | split: hpdev_try_gather_u64, hpdev_try_encode_labels, hpdev_try_cast_f64_to_f32, hpdev_try_all_finite +2 |
| `bindings/labels_legacy_boundary.mojo` | core | SPLIT | 0 | 1 | 1 | 1 | split-unpack-pack | split: labels_legacy_prepare |
| `bindings/msel_device.mojo` | core | SPLIT | 3 | 2 | 5 | 15 | split-unpack-pack | split: kmeans_fit_fold_binding, pca_fit_fold_binding |
| `bindings/msel_host.mojo` | core | CLEAN | 1 | 0 | 0 | 11 | rename-pack |  |
| `bindings/_mojolearn_embedding.mojo` | embedding | SPLIT | 37 | 5 | 1 | 12 | ok | split: embedding_forward_words, embedding_backward_words, embedding_forward_dev_words, embedding_backward_dev_words +1 |
| `bindings/_mojolearn_embedding_host.mojo` | embedding | SPLIT | 1 | 1 | 1 | 7 | ok | split: embedding_host_backward_words |
| `bindings/embedding_host_forward.mojo` | embedding | SPLIT | 1 | 1 | 1 | 1 | split-unpack-pack | split: embedding_host_forward_words |
| `bindings/_mojolearn_embedding_infer_host.mojo` | embedding_infer | CLEAN | 1 | 0 | 1 | 6 | ok |  |
| `bindings/_mojolearn_estimators.mojo` | estimators | SPLIT | 6 | 2 | 15 | 49 | ok | split: _pca_whiten_apply, pca_fit_owned_binding |
| `bindings/_mojolearn_estimators_host.mojo` | estimators | SPLIT | 4 | 3 | 14 | 42 | ok | split: pca_fit_owned_binding, tsvd_transform_binding, inverse_transform_binding |
| `bindings/forest_export_binding.mojo` | forest | CLEAN | 1 | 0 | 3 | 0 | rename-unpack |  |
| `bindings/forest_host_groves_binding.mojo` | forest | CLEAN | 3 | 0 | 0 | 3 | rename-pack |  |
| `bindings/forest_host_strict_binding.mojo` | forest | SPLIT | 4 | 1 | 0 | 3 | rename-pack | split: forest_strict_predict_host_native |
| `bindings/forest_inference_binding.mojo` | forest | CLEAN | 6 | 0 | 1 | 9 | split-unpack-pack |  |
| `bindings/_mojolearn_gbdt.mojo` | gbdt | SPLIT | 10 | 4 | 5 | 14 | ok | split: _gbdt_locked_insert, gbdt_fit_native, gbdt_fit_two_level_feature_freq_binding, gbdt_fit_ordered_rmse_binding |
| `bindings/_mojolearn_gbdt_host.mojo` | gbdt | SPLIT | 6 | 5 | 5 | 10 | ok | split: _gbdt_fit_pointwise_arm, _gbdt_fit_ordered_arm, gbdt_predict_multi_binding, gbdt_fit_two_level_feature_freq_binding +1 |
| `bindings/gbdt_fit_boundary.mojo` | gbdt | SPLIT | 0 | 1 | 2 | 1 | split-unpack-pack | split: gbdt_fit_params_py |
| `bindings/_mojolearn_gp.mojo` | gp | SPLIT | 0 | 7 | 3 | 16 | ok | split: gpr_fit_binding, gpr_predict_binding, gpr_sample_y_binding, gpc_fit_binding +3 |
| `bindings/_mojolearn_gp_host.mojo` | gp | SPLIT | 0 | 6 | 1 | 9 | ok | split: gpr_fit_binding, gpr_sample_y_binding, gpr_predict_cov_binding, gpc_fit_binding +2 |
| `bindings/gp_classifier_native.mojo` | gp | ENTANGLED | 0 | 6 | 23 | 3 | entangled | struct GPCBinaryPublication holds member: PythonObject; iterations: PythonObject; struct GPCFitPublication holds members: PythonObject; iterations: List[PythonO |
| `bindings/gp_common_native.mojo` | gp | SPLIT | 0 | 3 | 5 | 7 | split-unpack-pack | split: gp_kernel_arrays, gp_kernel_arrays_transport, gp_free |
| `bindings/gp_host_predict.mojo` | gp | SPLIT | 0 | 2 | 0 | 1 | rename-pack | split: gpr_predict_binding, gpc_predict_binding |
| `bindings/gp_kernel_cache_boundary.mojo` | gp | ENTANGLED | 0 | 1 | 6 | 1 | entangled | struct GPKernelKeyItem holds pin: PythonObject; struct GPPreparedKernel holds kernel: PythonObject; context: PythonObject |
| `bindings/gp_kernel_public_native.mojo` | gp | ENTANGLED | 2 | 2 | 16 | 7 | entangled | struct GPKernelSnapshot holds owners: List[PythonObject]; states: List[PythonObject]; struct GPKernelNode holds kind: PythonObject |
| `bindings/gp_label_input_boundary.mojo` | gp | SPLIT | 0 | 1 | 2 | 0 | rename-unpack | split: _gp_label_input |
| `bindings/gp_labels_native.mojo` | gp | ENTANGLED | 2 | 4 | 14 | 5 | entangled | _gp_prepare: typed work on py-struct GPLabelsConnection,_TextLabels; _gp_float_storage_labels: typed work on py-struct _TextLabels |
| `bindings/gp_mixed_label_boundary.mojo` | gp | ENTANGLED | 1 | 1 | 3 | 1 | entangled | struct GPScalarIndex holds values: List[PythonObject] |
| `bindings/gp_native_boundary.mojo` | gp | ENTANGLED | 1 | 0 | 18 | 0 | entangled | struct GPDescriptorMetadata holds provider: PythonObject; struct GPConnection holds provider: PythonObject |
| `bindings/gp_query_x.mojo` | gp | ENTANGLED | 1 | 0 | 3 | 1 | entangled | struct GpQuery holds q: PythonObject; gp_query_refuse: RUN code on Python-holding type GpQuery |
| `bindings/gp_regressor_native.mojo` | gp | ENTANGLED | 1 | 6 | 11 | 4 | entangled | struct GPRNormalizedTargets holds targets: PythonObject |
| `bindings/gp_score_boundary.mojo` | gp | ENTANGLED | 0 | 1 | 2 | 0 | entangled | struct GPRPredictionRoom holds mean: PythonObject; stddev: PythonObject; struct GPCPredictionCodes holds classes: PythonObject |
| `bindings/gp_score_labels_boundary.mojo` | gp | ENTANGLED | 1 | 3 | 3 | 0 | entangled | _gp_score_integer: RUN code on Python-holding type GPMetricLabels,ScoreKeyOwner; _gp_score_prediction: typed work on py-struct GPDecodedScoreLabels |
| `bindings/gp_svm_gp_state.mojo` | gp | ENTANGLED | 2 | 1 | 17 | 0 | entangled | struct GpAdmittedInputs holds items: List[PythonObject] |
| `providers/python/gp_public_lifecycle_pyapi.mojo` | gp | SPLIT | 1 | 1 | 2 | 3 | split-unpack-pack | split: gp_public_operation |
| `bindings/_mojolearn_hdbscan.mojo` | hdbscan | SPLIT | 2 | 2 | 2 | 4 | ok | split: hdbscan_fit_binding, hdbscan_membership_vector_binding |
| `bindings/_mojolearn_hdbscan_host.mojo` | hdbscan | SPLIT | 0 | 1 | 2 | 6 | ok | split: hdbscan_fit_binding |
| `bindings/_mojolearn_ivf.mojo` | ivf | SPLIT | 3 | 4 | 2 | 6 | ok | split: ivf_flat_build_binding, _ivf_search_arrays, ivf_flat_index_prepare_binding, ivf_flat_build_resident_binding |
| `bindings/ivf_host_search.mojo` | ivf | SPLIT | 6 | 4 | 2 | 2 | split-unpack-pack | split: _ivf_host_search_arrays, ivf_flat_extend_binding, ivf_flat_index_prepare_binding, ivf_flat_index_search_binding |
| `bindings/ivf_index_arrays.mojo` | ivf | SPLIT | 4 | 3 | 8 | 0 | rename-unpack | split: ivf_read_index_arrays, ivf_write_index_arrays, ivf_write_extended_arrays |
| `bindings/_mojolearn_kernel_methods.mojo` | kernel_methods | SPLIT | 3 | 2 | 1 | 13 | ok | split: kernel_ridge_fit_ptr_binding, nystroem_transform_binding |
| `bindings/_mojolearn_kernel_methods_host.mojo` | kernel_methods | SPLIT | 0 | 3 | 1 | 8 | ok | split: kernel_ridge_fit_binding, kernel_ridge_predict_binding, nystroem_transform_binding |
| `bindings/_mojolearn_linalg.mojo` | linalg | SPLIT | 0 | 1 | 1 | 19 | ok | split: gemm_fused_binding |
| `bindings/apple_fast_bf16_public.mojo` | linalg | ENTANGLED | 1 | 3 | 6 | 3 | entangled | struct _Boundary holds builtins: PythonObject; array_type: PythonObject; struct _Input holds shape: PythonObject; store: PythonObject |
| `bindings/linalg_ops_buffers.mojo` | linalg | CLEAN | 1 | 0 | 2 | 0 | rename-unpack |  |
| `bindings/linalg_ops_decomp.mojo` | linalg | SPLIT | 1 | 5 | 11 | 0 | rename-unpack | split: lin_svdvals_of, _lin_eigh, _lin_qr_q, _lin_qr +1 |
| `bindings/linalg_ops_kit.mojo` | linalg | ENTANGLED | 0 | 1 | 1 | 0 | entangled | struct KitMatrix holds host: PythonObject; keep: PythonObject; struct LinKit holds b: PythonObject; context: PythonObject |
| `bindings/linalg_ops_lowbit.mojo` | linalg | SPLIT | 0 | 4 | 20 | 13 | split-unpack-pack | split: lin_call, _lin_matmul, _lin_dequantize_int8, _lin_matmul_bf16 |
| `bindings/linalg_ops_resample.mojo` | linalg | ENTANGLED | 1 | 6 | 13 | 4 | entangled | _rs_word: RUN code on Python-holding type AdmittedBuf |
| `bindings/_mojolearn_mamba.mojo` | mamba | SPLIT | 0 | 9 | 4 | 47 | ok | split: mamba1_session_open_binding, mamba2_session_open_binding, mamba3_prefill_session_install_weights_binding, mamba3_prefill_session_forward_owned_binding +5 |
| `bindings/_mojolearn_mamba_host.mojo` | mamba | SPLIT | 0 | 1 | 3 | 27 | ok | split: mamba3_owned_backward_binding |
| `bindings/mamba_public_native.mojo` | mamba | ENTANGLED | 5 | 6 | 20 | 22 | entangled | struct MambaDecodeHandle holds binding: PythonObject; device: PythonObject; mamba_session_decode: typed work on py-struct MambaDecodeHandle,NativeModules |
| `bindings/_mojolearn_metrics.mojo` | metrics | SPLIT | 6 | 13 | 5 | 27 | ok | split: precision_recall_curve_binding, confusion_matrix_binding, spectral_fit_predict_dataset_binding, spectral_fit_predict_graph_binding +9 |
| `bindings/_mojolearn_metrics_host.mojo` | metrics | SPLIT | 3 | 13 | 8 | 26 | ok | split: homogeneity_completeness_v_measure_binding, silhouette_binding, spectral_fit_predict_dataset_binding, spectral_fit_predict_graph_binding +9 |
| `bindings/metrics_native_inputs.mojo` | metrics | SPLIT | 0 | 1 | 2 | 2 | split-unpack-pack | split: _metric_array_flat |
| `bindings/metrics_reg_native.mojo` | metrics | ENTANGLED | 0 | 5 | 14 | 7 | entangled | struct _Prog holds inputs: List[PythonObject]; struct _Reg holds y: PythonObject; p: PythonObject |
| `bindings/p3_labels_metrics_native.mojo` | metrics | SPLIT | 7 | 4 | 8 | 6 | split-unpack-pack | split: _p3_utf8_read_source, _p3_text_codes_source, p3_text_lookup, p3_array_numbers |
| `providers/python/metrics_matrix_input_pyapi.mojo` | metrics | ENTANGLED | 1 | 0 | 1 | 2 | entangled | _metric_f32: RUN code on Python-holding type AdmittedTyped |
| `providers/python/metrics_vector_input_pyapi.mojo` | metrics | SPLIT | 1 | 1 | 3 | 0 | rename-unpack | split: metric_i32_vector |
| `bindings/_mojolearn_mixture.mojo` | mixture | SPLIT | 2 | 1 | 0 | 3 | ok | split: _rebuild_model |
| `bindings/mixture_native.mojo` | mixture | ENTANGLED | 1 | 7 | 26 | 12 | entangled | struct _BgmmConnection holds provider: PythonObject; struct _GmmConnection holds provider: PythonObject |
| `bindings/_mojolearn_neural_host.mojo` | neural | SPLIT | 0 | 4 | 10 | 16 | ok | split: mlp_forward_logits_binding, transformer_forward_fresh_binding, transformer_forward_binding, transformer_decode_step_binding |
| `bindings/neural_block_admission.mojo` | neural | SPLIT | 1 | 1 | 16 | 7 | split-unpack-pack | split: admit_block_f32 |
| `bindings/neural_block_public.mojo` | neural | ENTANGLED | 3 | 6 | 34 | 12 | entangled | struct NativeBlockPlan holds context: PythonObject; entries: PythonObject; struct NativeBlockExecution holds plan: PythonObject; fields: PythonObject |
| `providers/python/neural_descriptor_pyapi.mojo` | neural | ENTANGLED | 2 | 4 | 20 | 10 | entangled | struct NeuralDescriptorProvider holds module: PythonObject; capsule: PythonObject; struct NeuralDescriptorEntries holds entries: List[PythonObject] |
| `bindings/_mojolearn_preprocessing.mojo` | preprocessing | CLEAN | 1 | 0 | 1 | 12 | ok |  |
| `bindings/_mojolearn_resample.mojo` | resample | SPLIT | 6 | 2 | 1 | 10 | ok | split: bootstrap_binding, bootstrap_unpaired_binding |
| `bindings/_mojolearn_resample_host.mojo` | resample | SPLIT | 5 | 3 | 0 | 8 | ok | split: bootstrap_binding, bootstrap_unpaired_binding, permutation_samples_binding |
| `providers/python/resample_pandas_pyapi.mojo` | resample | ENTANGLED | 12 | 13 | 18 | 17 | entangled | struct _RPArray holds owner: PythonObject; dtype: PythonObject; _rp_number: RUN code on Python-holding type _RPArray |
| `providers/python/resample_public_pyapi.mojo` | resample | SPLIT | 0 | 1 | 0 | 2 | rename-pack | split: resample_result_initialize |
| `providers/python/resample_rows_pyapi.mojo` | resample | ENTANGLED | 1 | 2 | 1 | 1 | entangled | struct ResampleRows holds source: PythonObject; sequence: PythonObject; ResampleRows: typed work on py-struct AdmittedBorrowedBuffer,AdmittedBuf |
| `bindings/_mojolearn_rf.mojo` | rf | SPLIT | 25 | 13 | 16 | 9 | ok | split: _forest_out_trees, _rf_classifier_fit, _rf_fit_py, rf_data_session_open_binding +9 |
| `bindings/_mojolearn_rf_host.mojo` | rf | SPLIT | 14 | 7 | 21 | 12 | ok | split: _rf_fit, _rf_fit_py, rf_data_session_open_binding, rf_regressor_fit_session_rows_binding +3 |
| `bindings/byte_lm_host_trainer_native.mojo` | runtime_byte_trainer | ENTANGLED | 0 | 5 | 3 | 12 | entangled | _inference_parameters: typed work on py-struct NativeModules |
| `bindings/byte_lm_session_owner.mojo` | runtime_byte_trainer | CLEAN | 1 | 0 | 0 | 1 | rename-pack |  |
| `bindings/byte_lm_state_admission.mojo` | runtime_byte_trainer | CLEAN | 2 | 0 | 3 | 1 | split-unpack-pack |  |
| `bindings/byte_lm_trainer_native.mojo` | runtime_byte_trainer | ENTANGLED | 3 | 9 | 25 | 27 | entangled | _shape_words: RUN code on Python-holding type _Shape; byte_lm_construct: typed work on py-struct NativeModules |
| `providers/python/byte_lm_metadata_pyapi.mojo` | runtime_byte_trainer | ENTANGLED | 1 | 2 | 6 | 0 | entangled | _shape_of: Python call per iteration inside a typed loop (callback) |
| `providers/python/training_buffer_pyapi.mojo` | runtime_byte_trainer | ENTANGLED | 0 | 1 | 2 | 0 | entangled | training_words32_owned: Python call per iteration inside a typed loop (callback) |
| `bindings/model_catalog_native.mojo` | runtime_catalog | ENTANGLED | 0 | 1 | 2 | 8 | entangled | struct CatalogConnection holds context: PythonObject; _catalog_capture: typed work on py-struct AdmittedBorrowedBuffer |
| `bindings/classical_archive.mojo` | runtime_classical | SPLIT | 0 | 5 | 6 | 5 | split-unpack-pack | split: _archive_path, _archive_array, _archive_unpack_member, _archive_pack_members +1 |
| `bindings/classical_archive_input.mojo` | runtime_classical | ENTANGLED | 0 | 1 | 1 | 1 | entangled | struct ArchiveReadInput holds file: PythonObject; methods: PythonObject; archive_input_unpack: typed work on py-struct ArchiveReadInput |
| `bindings/classical_archive_output.mojo` | runtime_classical | SPLIT | 0 | 1 | 3 | 4 | split-unpack-pack | split: archive_warning_pack |
| `bindings/classical_archive_read.mojo` | runtime_classical | ENTANGLED | 1 | 2 | 12 | 6 | entangled | struct _ArchiveReprFailure holds kind: PythonObject; value: PythonObject |
| `bindings/classical_labels.mojo` | runtime_classical | ENTANGLED | 6 | 7 | 69 | 9 | entangled | struct ClassicalLabels holds classes: PythonObject; codes: PythonObject; struct _PythonTextCodes holds values: PythonObject |
| `bindings/classical_native.mojo` | runtime_classical | CLEAN | 1 | 0 | 0 | 4 | rename-pack |  |
| `bindings/classical_pandas.mojo` | runtime_classical | ENTANGLED | 0 | 1 | 19 | 3 | entangled | struct _ClassicalAdmitTypes holds list_type: PythonObject; tuple_type: PythonObject |
| `bindings/classical_validate.mojo` | runtime_classical | CLEAN | 1 | 0 | 6 | 0 | rename-unpack |  |
| `bindings/linear_score_native.mojo` | runtime_classical | SPLIT | 0 | 1 | 2 | 2 | split-unpack-pack | split: linear_regressor_score |
| `providers/python/classical_family_admission_pyapi.mojo` | runtime_classical | SPLIT | 0 | 1 | 3 | 0 | rename-unpack | split: classical_family_admit |
| `providers/python/linear_fit_boundary.mojo` | runtime_classical | SPLIT | 0 | 1 | 0 | 0 | rename-unpack | split: classical_ridge_fit |
| `providers/python/linear_ols_boundary.mojo` | runtime_classical | SPLIT | 0 | 1 | 0 | 0 | rename-unpack | split: classical_ols_fit |
| `providers/python/linear_predict_boundary.mojo` | runtime_classical | SPLIT | 0 | 1 | 2 | 0 | rename-unpack | split: _linear_predict_boundary |
| `bindings/als_native.mojo` | runtime_decomp | ENTANGLED | 1 | 1 | 3 | 3 | entangled | _one_row: RUN code on Python-holding type DecompKit |
| `bindings/decomp_kit_native.mojo` | runtime_decomp | ENTANGLED | 0 | 1 | 19 | 7 | entangled | struct DecompMatrix holds host: PythonObject   # a float32 Array of rows; struct DecompKit holds entries: PythonObject; context: PythonObject   # the NumPy-free |
| `bindings/decomp_owner_common.mojo` | runtime_decomp | CLEAN | 2 | 0 | 12 | 0 | rename-unpack |  |
| `bindings/decomp_public_lifecycle.mojo` | runtime_decomp | CLEAN | 1 | 0 | 2 | 1 | split-unpack-pack |  |
| `bindings/dict_learning_native.mojo` | runtime_decomp | ENTANGLED | 2 | 1 | 21 | 8 | entangled | _split_sign: RUN code on Python-holding type DecompKit; _normalize: RUN code on Python-holding type DecompKit |
| `bindings/fa_native.mojo` | runtime_decomp | SPLIT | 0 | 2 | 1 | 1 | split-unpack-pack | split: fa_fit, fa_query |
| `bindings/ica_native.mojo` | runtime_decomp | SPLIT | 0 | 2 | 1 | 0 | rename-unpack | split: ica_fit, ica_project |
| `bindings/ipca_descriptor_boundary.mojo` | runtime_decomp | ENTANGLED | 2 | 2 | 6 | 0 | entangled | struct IPCAConnection holds provider: PythonObject |
| `bindings/ipca_native.mojo` | runtime_decomp | SPLIT | 1 | 2 | 13 | 3 | split-unpack-pack | split: ipca_partial_fit, _project |
| `bindings/ipca_owner_boundary.mojo` | runtime_decomp | CLEAN | 1 | 0 | 7 | 3 | split-unpack-pack |  |
| `bindings/jl_native.mojo` | runtime_decomp | CLEAN | 5 | 0 | 0 | 2 | rename-pack |  |
| `bindings/kpca_descriptor_boundary.mojo` | runtime_decomp | ENTANGLED | 3 | 2 | 12 | 1 | entangled | struct KPCANearConnection holds provider: PythonObject; metadata: PythonObject; struct KPCADecompConnection holds provider: PythonObject; reduction_provider: Py |
| `bindings/kpca_native.mojo` | runtime_decomp | SPLIT | 1 | 1 | 12 | 2 | split-unpack-pack | split: kpca_transform |
| `bindings/lda_native.mojo` | runtime_decomp | SPLIT | 0 | 1 | 3 | 3 | split-unpack-pack | split: _run |
| `bindings/linalg_decomp_native.mojo` | runtime_decomp | ENTANGLED | 2 | 0 | 9 | 10 | entangled | _cut: RUN code on Python-holding type DecompKit; _pseudo_solve: RUN code on Python-holding type DecompKit |
| `bindings/mcd_native.mojo` | runtime_decomp | ENTANGLED | 7 | 0 | 11 | 5 | entangled | _mcd_scalar: RUN code on Python-holding type DecompKit; _pinvh: RUN code on Python-holding type DecompKit |
| `bindings/nmf_descriptor_boundary.mojo` | runtime_decomp | ENTANGLED | 0 | 3 | 2 | 0 | entangled | struct NMFConnection holds provider: PythonObject |
| `bindings/nmf_native.mojo` | runtime_decomp | ENTANGLED | 0 | 2 | 12 | 0 | entangled | struct NMFPackedFit holds w: PythonObject; h: PythonObject |
| `bindings/pca_descriptor_boundary.mojo` | runtime_decomp | ENTANGLED | 4 | 3 | 6 | 0 | entangled | struct PCAConnection holds provider: PythonObject |
| `bindings/pca_fit_native.mojo` | runtime_decomp | ENTANGLED | 0 | 1 | 1 | 0 | entangled | struct PCAFitOutput holds keep: PythonObject |
| `bindings/pca_query_native.mojo` | runtime_decomp | SPLIT | 0 | 1 | 0 | 0 | rename-unpack | split: pca_query_operation |
| `bindings/pls_native.mojo` | runtime_decomp | ENTANGLED | 4 | 0 | 16 | 6 | entangled | _captured: RUN code on Python-holding type ArrayMetadata,DecompKit; _center_scale: RUN code on Python-holding type DecompKit |
| `bindings/tsvd_descriptor_boundary.mojo` | runtime_decomp | ENTANGLED | 3 | 1 | 6 | 1 | entangled | struct TSVDConnection holds provider: PythonObject |
| `bindings/tsvd_native.mojo` | runtime_decomp | SPLIT | 1 | 2 | 16 | 2 | split-unpack-pack | split: _exact, tsvd_inverse_transform |
| `providers/python/pca_fit_scalar_pyapi.mojo` | runtime_decomp | CLEAN | 1 | 0 | 7 | 2 | split-unpack-pack |  |
| `providers/python/pca_model_pyapi.mojo` | runtime_decomp | SPLIT | 1 | 1 | 5 | 2 | split-unpack-pack | split: pca_query_admitted |
| `bindings/cfb_agglo.mojo` | runtime_family | SPLIT | 0 | 2 | 8 | 1 | split-unpack-pack | split: _agglo_legacy, _agglo_x |
| `bindings/cfb_sparse_read.mojo` | runtime_family | SPLIT | 0 | 1 | 0 | 0 | rename-unpack | split: cfb_sparse_triples |
| `bindings/cfb_spectral.mojo` | runtime_family | ENTANGLED | 3 | 6 | 23 | 4 | entangled | struct _Coo holds rows: PythonObject; cols: PythonObject; struct _EmbeddingOptionPins holds components: PythonObject; affinity: PythonObject |
| `bindings/family_common.mojo` | runtime_family | ENTANGLED | 2 | 0 | 9 | 5 | entangled | family_snapshot_min: RUN code on Python-holding type ArrayMetadata |
| `bindings/kmeans_descriptor_boundary.mojo` | runtime_family | ENTANGLED | 1 | 2 | 4 | 0 | entangled | struct KMeansConnection holds provider: PythonObject |
| `bindings/kmeans_native.mojo` | runtime_family | SPLIT | 0 | 2 | 8 | 4 | split-unpack-pack | split: _kmeans_fit, _kmeans_inference |
| `bindings/kmeans_owner_boundary.mojo` | runtime_family | ENTANGLED | 0 | 1 | 6 | 1 | entangled | struct KMeansResidentOwner holds provider: PythonObject; centers: PythonObject |
| `bindings/linalg_native.mojo` | runtime_family | SPLIT | 0 | 2 | 4 | 4 | split-unpack-pack | split: _chol_fit, _chol_solve |
| `bindings/mcls_classify.mojo` | runtime_family | ENTANGLED | 5 | 7 | 8 | 5 | entangled | struct McPair holds present: PythonObject; w: PythonObject; mc_wsum_sf64: RUN code on Python-holding type McProg |
| `bindings/mcls_cluster.mojo` | runtime_family | SPLIT | 0 | 5 | 3 | 5 | split-unpack-pack | split: _cluster_codes, _entropy, _silhouette, _trustworthiness +1 |
| `bindings/mcls_common.mojo` | runtime_family | ENTANGLED | 8 | 0 | 48 | 14 | entangled | struct McLabels holds classes: PythonObject; codes: PythonObject; struct MetricIntegerView holds view: PythonObject |
| `bindings/mcls_ranking.mojo` | runtime_family | ENTANGLED | 3 | 4 | 4 | 4 | entangled | mc_fold_stages: RUN code on Python-holding type McProg; mc_stage_average: RUN code on Python-holding type McProg |
| `bindings/metrics_exp_cls.mojo` | runtime_family | ENTANGLED | 3 | 1 | 14 | 13 | entangled | _wsum_sf64: RUN code on Python-holding type MxSums; _cm_stage: RUN code on Python-holding type MxSums |
| `bindings/metrics_exp_cluster.mojo` | runtime_family | SPLIT | 1 | 2 | 6 | 3 | split-unpack-pack | split: _contingency_matrix, _ami |
| `bindings/metrics_exp_common.mojo` | runtime_family | ENTANGLED | 5 | 1 | 43 | 7 | entangled | struct MxProg holds inputs: List[PythonObject]; struct MxArena holds store: PythonObject |
| `bindings/metrics_exp_curve.mojo` | runtime_family | ENTANGLED | 3 | 2 | 6 | 6 | entangled | _fold_stages: RUN code on Python-holding type MxProg; _stage_average: RUN code on Python-holding type MxProg |
| `bindings/metrics_exp_rank.mojo` | runtime_family | ENTANGLED | 2 | 1 | 7 | 6 | entangled | struct _Proba holds codes: PythonObject; a: PythonObject; _put_proba: RUN code on Python-holding type MxProg,_Proba |
| `bindings/neighbors_graph_native.mojo` | runtime_family | ENTANGLED | 0 | 3 | 19 | 16 | entangled | struct _Csr holds indptr: PythonObject; indices: PythonObject |
| `bindings/neighbors_local_native.mojo` | runtime_family | ENTANGLED | 0 | 2 | 33 | 10 | entangled | struct _NcOperands holds query: PythonObject; centroids: PythonObject |
| `bindings/neighbors_semi_native.mojo` | runtime_family | ENTANGLED | 0 | 2 | 24 | 19 | entangled | struct _SemiKfeatCache holds module: PythonObject; table: PythonObject |
| `bindings/normal_dist_helpers.mojo` | runtime_family | SPLIT | 12 | 1 | 0 | 0 | rename-unpack | split: truncnorm_draws_binding |
| `bindings/ocsvm_scalar_boundary.mojo` | runtime_family | ENTANGLED | 0 | 1 | 6 | 2 | entangled | struct OcScalarTypes holds constructor: PythonObject; arrays: List[PythonObject] |
| `bindings/spectral_affinity_boundary.mojo` | runtime_family | ENTANGLED | 0 | 1 | 1 | 0 | entangled | struct SpectralAffinityRoom holds public_array: PythonObject |
| `bindings/str_labels_native.mojo` | runtime_family | SPLIT | 12 | 3 | 5 | 3 | split-unpack-pack | split: _read_source, str_unique_into, str_codes_into |
| `bindings/svc_common.mojo` | runtime_family | SPLIT | 2 | 1 | 18 | 16 | split-unpack-pack | split: svc_r2_sums |
| `bindings/svc_kernel_native.mojo` | runtime_family | ENTANGLED | 2 | 10 | 27 | 17 | entangled | _svm_x_scan: RUN code on Python-holding type ArrayMetadata; _platt: typed work on py-struct ArrayMetadata |
| `bindings/svc_linear_native.mojo` | runtime_family | ENTANGLED | 0 | 3 | 8 | 2 | entangled | _qn_row_weights: typed work on py-struct LinearSvcWeights; _linear_svc_prediction: typed work on py-struct LinearSvcPrediction |
| `bindings/svc_ocsvm_native.mojo` | runtime_family | ENTANGLED | 1 | 3 | 9 | 3 | entangled | struct OcIndexPin holds owner: PythonObject |
| `bindings/svc_score_boundary.mojo` | runtime_family | ENTANGLED | 0 | 4 | 4 | 0 | entangled | struct SvcDecodedScore holds pins: List[PythonObject]; svc_decode_score: typed work on py-struct SvcDecodedScore |
| `providers/python/family_public_lifecycle_pyapi.mojo` | runtime_family | SPLIT | 3 | 1 | 2 | 1 | split-unpack-pack | split: family_lifecycle_begin |
| `providers/python/hdbscan_public_lifecycle_pyapi.mojo` | runtime_family | SPLIT | 1 | 1 | 4 | 5 | split-unpack-pack | split: _inference |
| `providers/python/hierarchy_connectivity_pyapi.mojo` | runtime_family | ENTANGLED | 0 | 1 | 3 | 1 | entangled | struct HierarchyConnectivity holds owner: PythonObject; data: PythonObject; struct _Storage holds keep: PythonObject |
| `providers/python/radius_public_lifecycle_pyapi.mojo` | runtime_family | SPLIT | 0 | 1 | 5 | 2 | split-unpack-pack | split: radius_target |
| `bindings/logistic_native.mojo` | runtime_linear | ENTANGLED | 0 | 2 | 19 | 5 | entangled | struct _QueryX holds x: PythonObject |
| `bindings/logistic_query_boundary.mojo` | runtime_linear | SPLIT | 0 | 2 | 3 | 2 | split-unpack-pack | split: _logistic_width_builtin, logistic_query_width |
| `bindings/x_linear_native.mojo` | runtime_linear | ENTANGLED | 1 | 17 | 63 | 28 | entangled | struct XLinearOps holds images: PythonObject |
| `bindings/mani_decomp_native.mojo` | runtime_manifold | ENTANGLED | 4 | 2 | 16 | 11 | entangled | _lle_smallest: typed work on py-struct MKit |
| `bindings/mani_kit.mojo` | runtime_manifold | ENTANGLED | 2 | 4 | 27 | 11 | entangled | struct MKit holds b: PythonObject; raw: PythonObject; MKit: typed work on py-struct ArrayMetadata,ManifoldDevice |
| `bindings/mani_tsne_native.mojo` | runtime_manifold | SPLIT | 0 | 2 | 3 | 0 | rename-unpack | split: _tsne_pca_start, _tsne_fit |
| `bindings/mani_umap_native.mojo` | runtime_manifold | ENTANGLED | 0 | 3 | 8 | 3 | entangled | _umap_target: Python call per iteration inside a typed loop (callback) |
| `bindings/manifold_errors_native.mojo` | runtime_manifold | SPLIT | 1 | 1 | 1 | 0 | rename-unpack | split: manifold_error |
| `bindings/manifold_lifecycle_native.mojo` | runtime_manifold | ENTANGLED | 4 | 0 | 9 | 1 | entangled | _target_finite: RUN code on Python-holding type ArrayMetadata; _target_finite: typed work on py-struct ArrayMetadata |
| `bindings/msel_common.mojo` | runtime_model_selection | ENTANGLED | 5 | 2 | 41 | 23 | entangled | struct MselGroups holds codes: PythonObject; classes: PythonObject; msel_signed32: RUN code on Python-holding type _Prog |
| `bindings/msel_cv.mojo` | runtime_model_selection | ENTANGLED | 2 | 5 | 26 | 18 | entangled | struct MselFoldRows holds x: PythonObject; y: PythonObject; struct MselScores holds keys: PythonObject; estimators: PythonObject |
| `bindings/msel_helpers_released.mojo` | runtime_model_selection | SPLIT | 0 | 21 | 0 | 0 | rename-unpack | split: msel_arange_i64, msel_count_mask_u8, msel_select_mask_u8_i64, msel_mask_from_indices_u8 +17 |
| `bindings/msel_parallel_wire.mojo` | runtime_model_selection | CLEAN | 1 | 0 | 3 | 9 | split-unpack-pack |  |
| `bindings/msel_parameter_grid.mojo` | runtime_model_selection | ENTANGLED | 0 | 1 | 1 | 1 | entangled | struct ParameterGridRow holds keys: List[PythonObject]; choices: List[PythonObject]; PreparedParameterGrid: typed work on py-struct ParameterGridRow |
| `bindings/msel_search.mojo` | runtime_model_selection | ENTANGLED | 3 | 6 | 14 | 9 | entangled | _group_order: typed work on py-struct MselGroups,_Prog |
| `bindings/msel_split.mojo` | runtime_model_selection | ENTANGLED | 1 | 11 | 14 | 30 | entangled | struct MselSplitter holds model: PythonObject; fields: PythonObject; msel_strat_shuffle_split: typed work on py-struct MselGroups,_Prog |
| `bindings/causal_lm_construct.mojo` | runtime_neural | SPLIT | 1 | 3 | 18 | 17 | split-unpack-pack | split: _model_unpack, model_recurrent_weights_check, _causal_model_read_weights |
| `bindings/causal_lm_public.mojo` | runtime_neural | ENTANGLED | 3 | 10 | 83 | 40 | entangled | struct CausalPublicConnection holds context: PythonObject; struct LegacyGenerationReadPlan holds resident_key: PythonObject; logits: PythonObject |
| `bindings/causal_model_config.mojo` | runtime_neural | ENTANGLED | 0 | 1 | 17 | 20 | entangled | struct ModelConfigConnection holds context: PythonObject |
| `bindings/causal_operation_owner.mojo` | runtime_neural | CLEAN | 2 | 0 | 18 | 13 | split-unpack-pack |  |
| `bindings/generation_buffers.mojo` | runtime_neural | ENTANGLED | 0 | 3 | 8 | 4 | entangled | struct GenerationBoundary holds bi: PythonObject; arrays: PythonObject |
| `bindings/optimizer_container_boundary.mojo` | runtime_neural | CLEAN | 1 | 0 | 1 | 2 | split-unpack-pack |  |
| `bindings/optimizer_descriptor_boundary.mojo` | runtime_neural | ENTANGLED | 0 | 1 | 7 | 0 | entangled | struct OptimizerConnection holds owner: PythonObject; _optimizer_connection: typed work on py-struct NeuralDescriptorProvider,OptimizerConnection |
| `bindings/optimizer_public_door.mojo` | runtime_neural | ENTANGLED | 0 | 5 | 40 | 17 | entangled | struct OptDoor holds context: PythonObject; array_cls: PythonObject; tensor_view: typed work on py-struct OptimizerInputOwners |
| `bindings/optimizer_public_words.mojo` | runtime_neural | CLEAN | 1 | 0 | 5 | 0 | rename-unpack |  |
| `bindings/optimizer_schedule_native.mojo` | runtime_neural | SPLIT | 0 | 1 | 5 | 1 | split-unpack-pack | split: schedule_operation |
| `bindings/samba_checkpoint_native.mojo` | runtime_neural | SPLIT | 3 | 1 | 3 | 6 | split-unpack-pack | split: _checkpoint_read_stream |
| `bindings/samba_construct_native.mojo` | runtime_neural | SPLIT | 0 | 2 | 0 | 2 | rename-pack | split: samba_load_weights, _samba_initialize_parameters |
| `bindings/samba_decode_native.mojo` | runtime_neural | SPLIT | 0 | 4 | 7 | 4 | split-unpack-pack | split: samba_decode_init, samba_decode_step, samba_decode_tokens, samba_decode_sync |
| `bindings/samba_inference_native.mojo` | runtime_neural | SPLIT | 3 | 6 | 6 | 6 | split-unpack-pack | split: _m3_state_call, samba_state_words, _samba_program_call, _samba_inference_ids +2 |
| `bindings/samba_public_native.mojo` | runtime_neural | CLEAN | 1 | 0 | 0 | 2 | rename-pack |  |
| `bindings/samba_state_native.mojo` | runtime_neural | SPLIT | 1 | 2 | 6 | 10 | split-unpack-pack | split: samba_primitive, samba_apply_gradients_from_state |
| `bindings/samba_step_native.mojo` | runtime_neural | ENTANGLED | 0 | 1 | 6 | 3 | entangled | struct SambaMomentOwners holds first: PythonObject; second: PythonObject; _samba_train_step_admitted: typed work on py-struct OptimizerOwner |
| `bindings/samba_training_native.mojo` | runtime_neural | SPLIT | 0 | 2 | 5 | 2 | split-unpack-pack | split: samba_ids, _samba_gradients |
| `providers/python/generation_metadata_pyapi.mojo` | runtime_neural | ENTANGLED | 2 | 2 | 6 | 1 | entangled | struct GenerationWeightSnapshot holds owners: List[PythonObject]; struct GenerationLayerSnapshot holds block: PythonObject; profile: PythonObject |
| `providers/python/generation_session_pyapi.mojo` | runtime_neural | ENTANGLED | 0 | 1 | 4 | 2 | entangled | struct GenerationNativeProvider holds module: PythonObject; capsule: PythonObject; struct GenerationNativeSession holds provider: PythonObject; owner: PythonObj |
| `providers/python/samba_execution_pyapi.mojo` | runtime_neural | ENTANGLED | 1 | 2 | 4 | 1 | entangled | struct SambaLayerPublication holds fields: PythonObject; state: PythonObject; struct AdmittedSambaExecution holds keep: List[PythonObject]; state_keys: List[Pyt |
| `providers/python/samba_resident_pyapi.mojo` | runtime_neural | SPLIT | 0 | 1 | 1 | 2 | split-unpack-pack | split: samba_resident_native_step |
| `bindings/parallel_byte_native.mojo` | runtime_parallel | ENTANGLED | 5 | 10 | 17 | 8 | entangled | struct ParallelByteBoundary holds bi: PythonObject; array_cls: PythonObject; struct ParallelByteOwner holds provider: PythonObject; create: PythonObject |
| `bindings/parallel_causal_native.mojo` | runtime_parallel | ENTANGLED | 1 | 2 | 32 | 22 | entangled | struct ParallelCausalBoundary holds bi: PythonObject; context: PythonObject; struct CausalWorkerRegistry holds plan: PythonObject; top: PythonObject |
| `bindings/parallel_forecast_native.mojo` | runtime_parallel | ENTANGLED | 0 | 4 | 6 | 9 | entangled | struct ParallelForecastBoundary holds arima_cls: PythonObject; hw_cls: PythonObject |
| `bindings/parallel_mlp_native.mojo` | runtime_parallel | ENTANGLED | 3 | 4 | 12 | 5 | entangled | struct ParallelMlpBoundary holds bi: PythonObject; model_cls: PythonObject |
| `bindings/parallel_neighbor_worker.mojo` | runtime_parallel | ENTANGLED | 0 | 1 | 5 | 9 | entangled | struct NeighborSnapshots holds models: PythonObject |
| `bindings/parallel_neural_native.mojo` | runtime_parallel | ENTANGLED | 0 | 2 | 14 | 7 | entangled | struct ParallelNeuralBoundary holds bi: PythonObject; owner_cls: PythonObject |
| `bindings/parallel_shard_common.mojo` | runtime_parallel | ENTANGLED | 5 | 1 | 16 | 12 | entangled | struct ShardTrace holds records: PythonObject; struct ShardIdle holds pools: PythonObject; stored: PythonObject |
| `bindings/parallel_shard_native.mojo` | runtime_parallel | ENTANGLED | 0 | 7 | 17 | 16 | entangled | struct ParallelShardBoundary holds tables: PythonObject |
| `bindings/small_mlp_native.mojo` | runtime_parallel | ENTANGLED | 13 | 8 | 37 | 21 | entangled | struct SmallMlpBoundary holds bi: PythonObject; trainer_cls: PythonObject; struct MlpOperationMetadata holds legacy: PythonObject |
| `bindings/tsa_arima_native.mojo` | runtime_parallel | SPLIT | 0 | 4 | 11 | 3 | split-unpack-pack | split: _exog_array, arima_fit, arima_adopt, arima_predict |
| `bindings/tsa_autoarima_native.mojo` | runtime_parallel | ENTANGLED | 0 | 1 | 2 | 2 | entangled | struct _SearchBlock holds owner: PythonObject |
| `bindings/tsa_public_native.mojo` | runtime_parallel | SPLIT | 0 | 2 | 2 | 3 | split-unpack-pack | split: _series_major, tsa_kpss |
| `providers/python/arima_lifecycle_pyapi.mojo` | runtime_parallel | SPLIT | 0 | 1 | 0 | 3 | rename-pack | split: arima_parameter |
| `providers/python/autoarima_search_pyapi.mojo` | runtime_parallel | CLEAN | 1 | 0 | 10 | 2 | split-unpack-pack |  |
| `bindings/maxabs_native.mojo` | runtime_prep | SPLIT | 0 | 1 | 1 | 2 | split-unpack-pack | split: maxabs_fit |
| `bindings/minmax_native.mojo` | runtime_prep | ENTANGLED | 0 | 6 | 4 | 9 | entangled | minmax_transform: typed work on py-struct NativeModules,ScalerInputSnapshot |
| `bindings/prep_classify_common.mojo` | runtime_prep | ENTANGLED | 7 | 4 | 31 | 3 | entangled | struct XIn holds value: PythonObject; struct CsrIn holds indptr: PythonObject; indices: PythonObject |
| `bindings/prep_classify_native.mojo` | runtime_prep | ENTANGLED | 8 | 2 | 48 | 12 | entangled | struct Counts holds mode: PythonObject; _check_nonnegative: RUN code on Python-holding type ClsProg |
| `bindings/prep_classify_prog.mojo` | runtime_prep | ENTANGLED | 1 | 1 | 1 | 2 | entangled | struct ClsProg holds inputs: PythonObject; inout: PythonObject |
| `bindings/prep_encode_native.mojo` | runtime_prep | ENTANGLED | 18 | 9 | 83 | 28 | entangled | struct EncProg holds inputs: PythonObject; _sel: RUN code on Python-holding type EncProg |
| `bindings/prep_fit_error.mojo` | runtime_prep | SPLIT | 0 | 1 | 0 | 1 | rename-pack | split: prep_fit_error |
| `bindings/prep_iterimp_native.mojo` | runtime_prep | ENTANGLED | 6 | 4 | 19 | 5 | entangled | _ii_rows: RUN code on Python-holding type EncProg; struct _Take holds xo: PythonObject; yo: PythonObject |
| `bindings/prep_labels_native.mojo` | runtime_prep | ENTANGLED | 4 | 9 | 37 | 16 | entangled | struct LabelBuf holds words: PythonObject; _label_load: RUN code on Python-holding type EncProg,LabelBuf |
| `bindings/prep_select_native.mojo` | runtime_prep | ENTANGLED | 1 | 2 | 9 | 6 | entangled | _rfe_rounds: typed work on py-struct ClsProg,XIn; _work: RUN code on Python-holding type EncProg |
| `bindings/prep_stateless_native.mojo` | runtime_prep | SPLIT | 5 | 8 | 10 | 8 | split-unpack-pack | split: _fit, _transform, _col_stats, _variance_fit +4 |
| `bindings/robust_native.mojo` | runtime_prep | SPLIT | 0 | 2 | 4 | 2 | split-unpack-pack | split: robust_fit, robust_transform |
| `bindings/scaler_lifecycle_native.mojo` | runtime_prep | ENTANGLED | 0 | 5 | 10 | 10 | entangled | scaler_transform: typed work on py-struct NativeModules,ScalerInputSnapshot |
| `bindings/scaler_native.mojo` | runtime_prep | ENTANGLED | 0 | 2 | 2 | 3 | entangled | scaler_fit: typed work on py-struct ScalerInputSnapshot |
| `providers/python/prep_arrow_target_pyapi.mojo` | runtime_prep | SPLIT | 0 | 1 | 0 | 0 | rename-unpack | split: prep_arrow_target |
| `providers/python/prep_category_pyapi.mojo` | runtime_prep | SPLIT | 0 | 2 | 2 | 0 | rename-unpack | split: _category_capture, prep_category_array |
| `providers/python/prep_target_pyapi.mojo` | runtime_prep | ENTANGLED | 3 | 1 | 0 | 1 | entangled | _finite: RUN code on Python-holding type ArrayMetadata |
| `providers/python/prep_transform_pyapi.mojo` | runtime_prep | ENTANGLED | 3 | 3 | 3 | 1 | entangled | prep_finite: RUN code on Python-holding type ArrayMetadata |
| `bindings/sequential_admission.mojo` | runtime_sequential | ENTANGLED | 0 | 2 | 14 | 3 | entangled | struct _SequentialValueTypes holds list_type: PythonObject; tuple_type: PythonObject; struct _SequentialValueTypesCache holds contexts: List[PythonObject]; sour |
| `bindings/sequential_arrays.mojo` | runtime_sequential | ENTANGLED | 0 | 1 | 10 | 1 | entangled | struct SequentialInputSnapshot holds keep: PythonObject |
| `bindings/sequential_checkpoint.mojo` | runtime_sequential | CLEAN | 1 | 0 | 2 | 2 | split-unpack-pack |  |
| `bindings/sequential_execution.mojo` | runtime_sequential | SPLIT | 0 | 1 | 0 | 5 | rename-pack | split: sequential_forward |
| `bindings/sequential_inputs.mojo` | runtime_sequential | ENTANGLED | 0 | 1 | 13 | 4 | entangled | struct SequenceBoundaryFrame holds owner: PythonObject; struct SequenceBoundaryResult holds storage: PythonObject |
| `bindings/host_density_hash_native.mojo` | runtime_trees | SPLIT | 0 | 1 | 2 | 3 | split-unpack-pack | split: _density_hash_update |
| `bindings/host_iforest_depth_native.mojo` | runtime_trees | SPLIT | 0 | 1 | 7 | 3 | split-unpack-pack | split: host_iforest_depth_step |
| `bindings/host_iforest_scalar_repr.mojo` | runtime_trees | SPLIT | 0 | 1 | 1 | 1 | split-unpack-pack | split: _if_repr_float |
| `bindings/host_inference_native.mojo` | runtime_trees | SPLIT | 0 | 1 | 1 | 0 | rename-unpack | split: host_inference_operation |
| `bindings/host_kde_bandwidth_native.mojo` | runtime_trees | SPLIT | 0 | 1 | 5 | 0 | rename-unpack | split: kde_temporal_bandwidth |
| `bindings/host_kde_fit_native.mojo` | runtime_trees | SPLIT | 0 | 2 | 3 | 6 | split-unpack-pack | split: _kde_object_storage, _kde_y |
| `bindings/host_knn_predict_native.mojo` | runtime_trees | SPLIT | 0 | 1 | 0 | 2 | rename-pack | split: host_knn_predict |
| `bindings/trees_agnostic_native.mojo` | runtime_trees | ENTANGLED | 3 | 3 | 9 | 8 | entangled | struct _State holds model: PythonObject; bg: PythonObject |
| `bindings/trees_meta_admission.mojo` | runtime_trees | SPLIT | 0 | 1 | 6 | 2 | split-unpack-pack | split: _stamp_into |
| `bindings/trees_public_auxiliary.mojo` | runtime_trees | SPLIT | 0 | 1 | 1 | 5 | split-unpack-pack | split: forest_shape |
| `bindings/trees_public_boost.mojo` | runtime_trees | ENTANGLED | 2 | 10 | 73 | 30 | entangled | struct Session holds native: PythonObject |
| `bindings/trees_public_gbdt.mojo` | runtime_trees | SPLIT | 1 | 4 | 26 | 6 | split-unpack-pack | split: gbdt_predict, gbdt_tree_metadata_op, ordered_rmse_fit, feature_freq_fit |
| `bindings/trees_public_gbdt_fit.mojo` | runtime_trees | ENTANGLED | 2 | 4 | 31 | 25 | entangled | _group_sizes: Python call per iteration inside a typed loop (callback); struct _PairAdmit holds isinstance_: PythonObject; integral_t: PythonObject |
| `bindings/trees_public_meta.mojo` | runtime_trees | ENTANGLED | 0 | 15 | 43 | 51 | entangled | struct Active holds names: PythonObject; ests: PythonObject; struct _CalUnits holds model: PythonObject; xt: PythonObject |
| `bindings/trees_public_mixin_score.mojo` | runtime_trees | SPLIT | 0 | 1 | 2 | 1 | split-unpack-pack | split: gbdt_adapter_score |
| `bindings/trees_public_score.mojo` | runtime_trees | SPLIT | 0 | 2 | 9 | 3 | split-unpack-pack | split: trees_r2, trees_class_codes |
| `bindings/trees_staged_input.mojo` | runtime_trees | SPLIT | 0 | 1 | 1 | 3 | split-unpack-pack | split: trees_stage_members |
| `providers/python/tree_boost_public_lifecycle_pyapi.mojo` | runtime_trees | SPLIT | 0 | 1 | 0 | 1 | rename-pack | split: _boost_classifier_target |
| `bindings/neural_labels_native.mojo` | runtime_xseq | ENTANGLED | 1 | 3 | 1 | 2 | entangled | _flat: typed work on py-struct AdmittedBorrowedBuffer |
| `bindings/xcnn_bn_native.mojo` | runtime_xseq | SPLIT | 0 | 1 | 15 | 8 | split-unpack-pack | split: _bn_forward_admitted |
| `bindings/xcnn_classifier_native.mojo` | runtime_xseq | SPLIT | 0 | 2 | 16 | 10 | split-unpack-pack | split: _build, cnnclf_fit |
| `bindings/xcnn_graph_native.mojo` | runtime_xseq | SPLIT | 0 | 2 | 25 | 26 | split-unpack-pack | split: _linear_forward, _gcn_graph |
| `bindings/xseq_forecast_native.mojo` | runtime_xseq | SPLIT | 0 | 3 | 9 | 11 | split-unpack-pack | split: _croston_fit, _ets_fit, _garch_fit |
| `bindings/xseq_native_common.mojo` | runtime_xseq | SPLIT | 0 | 1 | 23 | 6 | split-unpack-pack | split: XseqStream |
| `bindings/xseq_neural_native.mojo` | runtime_xseq | SPLIT | 0 | 6 | 19 | 8 | split-unpack-pack | split: _rnn_fit, xseq_rnn, _mlp_fit, _mlp_pack +2 |
| `bindings/xseq_prophet_native.mojo` | runtime_xseq | SPLIT | 0 | 3 | 4 | 5 | split-unpack-pack | split: _auto, prophet_fit, prophet_predict |
| `bindings/xseq_score_native.mojo` | runtime_xseq | SPLIT | 0 | 1 | 5 | 2 | split-unpack-pack | split: xseq_regressor_score |
| `providers/python/prophet_buffers_pyapi.mojo` | runtime_xseq | SPLIT | 0 | 1 | 4 | 4 | split-unpack-pack | split: _time_unit |
| `providers/python/recurrent_state_pyapi.mojo` | runtime_xseq | SPLIT | 0 | 1 | 1 | 1 | split-unpack-pack | split: recurrent_state |
| `providers/python/xseq_public_lifecycle_pyapi.mojo` | runtime_xseq | ENTANGLED | 0 | 1 | 7 | 2 | entangled | struct XseqPublicLifecycle holds source: PythonObject; original_arguments: PythonObject |
| `bindings/admitted_call.mojo` | shared-infra | ENTANGLED | 0 | 1 | 3 | 0 | entangled | struct AdmittedInput holds source: PythonObject # the caller's object; owner: PythonObject  # the admitted object that; struct AdmittedCall holds binding: Pytho |
| `bindings/admitted_family.mojo` | shared-infra | SPLIT | 0 | 3 | 13 | 2 | split-unpack-pack | split: admitted_array, admit_family_input, _weight_buffer |
| `bindings/admitted_input.mojo` | shared-infra | ENTANGLED | 9 | 4 | 32 | 9 | entangled | struct AdmittedX holds array: PythonObject; source: PythonObject; struct AdmittedBuf holds keep: PythonObject; owner: PythonObject |
| `bindings/admitted_registry.mojo` | shared-infra | ENTANGLED | 11 | 1 | 10 | 0 | entangled | struct AdmittedRegistry holds items: List[PythonObject]; parents: List[PythonObject]; _registry: RUN code on Python-holding type AdmittedRegistry |
| `bindings/admitted_sequence.mojo` | shared-infra | SPLIT | 0 | 2 | 1 | 1 | split-unpack-pack | split: admit_nd_f32, admit_nd_owned_f32 |
| `bindings/affinity_preference_boundary.mojo` | shared-infra | ENTANGLED | 0 | 1 | 3 | 1 | entangled | struct PreferenceIteratorTypes holds classes: List[PythonObject] |
| `bindings/agglo_native_boundary.mojo` | shared-infra | ENTANGLED | 0 | 1 | 21 | 5 | entangled | struct AggloBindings holds estimators: PythonObject; solver: PythonObject; struct AggloSolverConnection holds provider: PythonObject |
| `bindings/archive_schema_native.mojo` | shared-infra | SPLIT | 0 | 2 | 33 | 20 | split-unpack-pack | split: _schema_whole, _schema_put |
| `bindings/arena_cache_get.mojo` | shared-infra | SPLIT | 1 | 1 | 5 | 5 | split-unpack-pack | split: _integer |
| `bindings/arena_cache_upload.mojo` | shared-infra | SPLIT | 3 | 1 | 5 | 8 | split-unpack-pack | split: _increment |
| `bindings/arena_materialize_native.mojo` | shared-infra | SPLIT | 3 | 2 | 5 | 8 | split-unpack-pack | split: _numpy_copy, _prepare |
| `bindings/arena_prog_native.mojo` | shared-infra | SPLIT | 4 | 3 | 13 | 4 | split-unpack-pack | split: _prep_run_impl, _metrics_download, _metrics_execute |
| `bindings/array_from_buffer_native.mojo` | shared-infra | ENTANGLED | 0 | 1 | 4 | 1 | entangled | array_from_buffer: typed work on py-struct AdmittedBorrowedBuffer |
| `bindings/array_native.mojo` | shared-infra | SPLIT | 9 | 2 | 23 | 39 | split-unpack-pack | split: _reshape, _getitem |
| `bindings/base_helpers_released.mojo` | shared-infra | SPLIT | 0 | 10 | 0 | 0 | rename-unpack | split: base_check_indices_i64, base_bincount_i64, base_reduce_stat, base_reduce_stat_int +6 |
| `bindings/boundary_read.mojo` | shared-infra | ENTANGLED | 4 | 1 | 33 | 4 | entangled | struct BufferView holds owner: PythonObject; view_kind_fits_i64: RUN code on Python-holding type BufferView |
| `bindings/byte_checkpoint_native.mojo` | shared-infra | ENTANGLED | 0 | 1 | 3 | 2 | entangled | struct ByteCheckpointBoundary holds bi: PythonObject; os: PythonObject |
| `bindings/class_weight_boundary.mojo` | shared-infra | SPLIT | 0 | 1 | 1 | 0 | rename-unpack | split: class_weight_dictionary_values |
| `bindings/class_weight_rows_boundary.mojo` | shared-infra | SPLIT | 0 | 1 | 0 | 1 | rename-pack | split: class_weight_plan_admitted |
| `bindings/classifier_public_lifecycle.mojo` | shared-infra | SPLIT | 0 | 1 | 5 | 1 | split-unpack-pack | split: _arrow_string_names |
| `bindings/cpython_table.mojo` | shared-infra | ENTANGLED | 1 | 0 | 1 | 0 | entangled | cpython_table: RUN code on Python-holding type CPythonTable |
| `bindings/dbscan_native_boundary.mojo` | shared-infra | ENTANGLED | 0 | 1 | 13 | 4 | entangled | struct DBSCANConnection holds provider: PythonObject; struct DBSCANReferenceOwner holds provider: PythonObject; refs: PythonObject |
| `bindings/device_rows_door.mojo` | shared-infra | CLEAN | 1 | 0 | 4 | 0 | rename-unpack |  |
| `bindings/holtwinters_host_predict.mojo` | shared-infra | SPLIT | 1 | 2 | 1 | 0 | rename-unpack | split: holtwinters_forecast_binding, holtwinters_predict_binding |
| `bindings/host_archive_native.mojo` | shared-infra | ENTANGLED | 2 | 6 | 32 | 38 | entangled | struct HostArchivePacked holds cls: PythonObject; public_owner: Optional[PythonObject] |
| `bindings/host_helpers.mojo` | shared-infra | CLEAN | 1 | 0 | 0 | 9 | rename-pack |  |
| `bindings/host_inference_preparation.mojo` | shared-infra | SPLIT | 0 | 1 | 0 | 0 | rename-unpack | split: host_inference_pandas_finite |
| `bindings/host_kernel_inference_native.mojo` | shared-infra | SPLIT | 0 | 1 | 3 | 0 | rename-unpack | split: host_kernel_inference |
| `bindings/host_logistic_native.mojo` | shared-infra | ENTANGLED | 0 | 2 | 5 | 1 | entangled | struct _LabelAliases holds types: List[PythonObject]; _host_logistic_decode: typed work on py-struct _LabelAliases |
| `bindings/host_sparse_native.mojo` | shared-infra | ENTANGLED | 1 | 4 | 10 | 2 | entangled | struct _SparseArraySnapshot holds keep: PythonObject; owner: PythonObject |
| `bindings/host_tree_native.mojo` | shared-infra | SPLIT | 0 | 3 | 7 | 5 | split-unpack-pack | split: host_tree_vote, host_tree_labels, host_gbdt_predict |
| `bindings/kernel_methods_native.mojo` | shared-infra | ENTANGLED | 2 | 6 | 14 | 14 | entangled | _km_chi2_input: RUN code on Python-holding type ArrayMetadata; _krr_query_scan: RUN code on Python-holding type ArrayMetadata |
| `bindings/kmeans_variants_boundary.mojo` | shared-infra | ENTANGLED | 0 | 1 | 6 | 0 | entangled | struct KMeansVariantsConnection holds provider: PythonObject; struct VariantsAuxiliaryPins holds owners: List[PythonObject] |
| `bindings/label_input_sources.mojo` | shared-infra | SPLIT | 0 | 1 | 5 | 1 | split-unpack-pack | split: _labels_list_scalar_kind_source |
| `bindings/labels_descriptor_boundary.mojo` | shared-infra | ENTANGLED | 0 | 1 | 10 | 0 | entangled | struct _LabelsProvider holds provider: PythonObject; struct _LabelsPin holds provider: PythonObject |
| `bindings/labels_score_boundary.mojo` | shared-infra | ENTANGLED | 0 | 2 | 5 | 2 | entangled | struct ScoreKeyOwner holds values: List[PythonObject]; pins: List[PythonObject]; _score_keys_unpack: typed work on py-struct ScoreKeyOwner |
| `bindings/logistic_descriptor_boundary.mojo` | shared-infra | ENTANGLED | 0 | 1 | 10 | 0 | entangled | struct LogisticConnection holds owner: PythonObject; logistic_connection: typed work on py-struct LogisticConnection,NeuralDescriptorProvider |
| `bindings/minmax_archive_native.mojo` | shared-infra | SPLIT | 0 | 1 | 3 | 1 | split-unpack-pack | split: minmax_archive_save |
| `bindings/minmax_metadata.mojo` | shared-infra | SPLIT | 0 | 1 | 2 | 1 | split-unpack-pack | split: minmax_range_values |
| `bindings/mode_metadata_native.mojo` | shared-infra | SPLIT | 0 | 1 | 10 | 8 | split-unpack-pack | split: _readback |
| `bindings/msel_inputs.mojo` | shared-infra | SPLIT | 0 | 1 | 6 | 1 | split-unpack-pack | split: _pandas_materialize |
| `bindings/native_array_snapshot.mojo` | shared-infra | ENTANGLED | 0 | 1 | 3 | 1 | entangled | native_array_snapshot: typed work on py-struct AdmittedBorrowedBuffer,NativeModules; struct NativeArrayAdmission holds cls: PythonObject; mro: PythonObject |
| `bindings/native_class_admission.mojo` | shared-infra | CLEAN | 2 | 0 | 9 | 2 | split-unpack-pack |  |
| `bindings/native_codec.mojo` | shared-infra | ENTANGLED | 2 | 1 | 11 | 7 | entangled | struct CodecBoundary holds bi: PythonObject; arrays: PythonObject |
| `bindings/native_identity_key.mojo` | shared-infra | ENTANGLED | 0 | 1 | 0 | 0 | entangled | struct NativeIdentityKey holds objects: List[PythonObject]; list_type: PythonObject; NativeIdentityKey: Python call per iteration inside a typed loop (callback) |
| `bindings/native_idna.mojo` | shared-infra | ENTANGLED | 2 | 1 | 5 | 2 | entangled | struct IdnaBoundary holds bi: PythonObject; normalize: PythonObject |
| `bindings/native_json.mojo` | shared-infra | ENTANGLED | 0 | 1 | 3 | 2 | entangled | struct NativeJSON holds raw: PythonObject; bi: PythonObject |
| `bindings/native_loaded_module.mojo` | shared-infra | ENTANGLED | 1 | 1 | 13 | 4 | entangled | struct _AdmittedModule holds name_key: PythonObject; module: PythonObject; struct NativeModules holds _modules: Dict[String, PythonObject] |
| `bindings/native_model_records.mojo` | shared-infra | ENTANGLED | 14 | 4 | 24 | 3 | entangled | struct StampedRecord holds payload: PythonObject; struct RecordPayloads holds records: Dict[String, PythonObject] |
| `bindings/native_owner_registry.mojo` | shared-infra | CLEAN | 1 | 0 | 4 | 1 | split-unpack-pack |  |
| `bindings/native_posix_transport.mojo` | shared-infra | SPLIT | 8 | 7 | 1 | 0 | rename-unpack | split: posix_sleep, posix_ready, posix_resolve, posix_open +3 |
| `bindings/native_prepared_owner.mojo` | shared-infra | SPLIT | 0 | 1 | 10 | 0 | rename-unpack | split: prepared_class_matches |
| `bindings/native_process_pool.mojo` | shared-infra | ENTANGLED | 1 | 4 | 9 | 7 | entangled | struct ProcessBoundary holds bi: PythonObject; environ: PythonObject |
| `bindings/native_resident_capsule.mojo` | shared-infra | SPLIT | 1 | 1 | 1 | 1 | split-unpack-pack | split: _resident_destroy |
| `bindings/native_scalar_decode.mojo` | shared-infra | SPLIT | 0 | 1 | 3 | 0 | rename-unpack | split: native_scalar_buffer_dtype |
| `bindings/native_scalar_promotion.mojo` | shared-infra | SPLIT | 0 | 1 | 0 | 0 | rename-unpack | split: native_scalar_weak_promotion |
| `bindings/native_transport.mojo` | shared-infra | ENTANGLED | 4 | 2 | 23 | 3 | entangled | struct TransportBoundary holds bi: PythonObject; json_string: PythonObject |
| `bindings/native_unit_threads.mojo` | shared-infra | SPLIT | 3 | 2 | 6 | 0 | rename-unpack | split: native_unit_gate_enter, native_unit_threads_run |
| `bindings/native_worker_inventory.mojo` | shared-infra | ENTANGLED | 2 | 1 | 6 | 0 | entangled | _inventory_utf8: Python call per iteration inside a typed loop (callback) |
| `bindings/optics_fit_boundary.mojo` | shared-infra | ENTANGLED | 0 | 1 | 2 | 0 | entangled | struct OpticsFitConnection holds provider: PythonObject; struct OpticsPackedResult holds core_distances: PythonObject; reachability: PythonObject |
| `bindings/p3_array_arena_records.mojo` | shared-infra | CLEAN | 2 | 0 | 8 | 0 | rename-unpack |  |
| `bindings/pandas_arrow_native.mojo` | shared-infra | SPLIT | 1 | 2 | 5 | 0 | rename-unpack | split: _arrow_typestr, pandas_arrow_ndarray |
| `bindings/pca_exact.mojo` | shared-infra | CLEAN | 2 | 0 | 1 | 0 | rename-unpack |  |
| `bindings/perf2_shared_fitted_field.mojo` | shared-infra | SPLIT | 0 | 1 | 0 | 3 | rename-pack | split: _class_member_names |
| `bindings/public_lifecycle_owner.mojo` | shared-infra | CLEAN | 2 | 0 | 12 | 4 | split-unpack-pack |  |
| `bindings/py_balanced.mojo` | shared-infra | CLEAN | 1 | 0 | 4 | 6 | ok |  |
| `bindings/py_version_tags.mojo` | shared-infra | SPLIT | 4 | 2 | 4 | 0 | ok | split: dict_version_tags, type_version_tags |
| `bindings/standard_archive_native.mojo` | shared-infra | SPLIT | 0 | 1 | 1 | 1 | split-unpack-pack | split: standard_archive_write |
| `bindings/umap_parameters_native.mojo` | shared-infra | ENTANGLED | 1 | 1 | 7 | 3 | entangled | _scalar_word: RUN code on Python-holding type ArrayMetadata |
| `bindings/user_object_flatten.mojo` | shared-infra | ENTANGLED | 0 | 1 | 2 | 1 | entangled | struct UserObjectFlatten holds b: PythonObject; list_type: PythonObject |
| `bindings/user_objects_native.mojo` | shared-infra | ENTANGLED | 3 | 7 | 16 | 11 | entangled | struct _Env holds b: PythonObject; op: PythonObject; _first_seen: Python call per iteration inside a typed loop (callback) |
| `bindings/x_prep_native_inputs.mojo` | shared-infra | SPLIT | 0 | 2 | 4 | 2 | split-unpack-pack | split: _target_pandas, _rows_snapshot |
| `bindings/x_prep_native_owner.mojo` | shared-infra | SPLIT | 2 | 1 | 13 | 7 | split-unpack-pack | split: xprep_run |
| `providers/python/array_metadata_pyapi.mojo` | shared-infra | ENTANGLED | 0 | 1 | 3 | 1 | entangled | struct ArrayMetadata holds keep: PythonObject; owner: PythonObject |
| `providers/python/autoarima_lifecycle_pyapi.mojo` | shared-infra | ENTANGLED | 0 | 2 | 7 | 1 | entangled | autoarima_forecast: Python call per iteration inside a typed loop (callback) |
| `providers/python/compiled_binding_snapshot_pyapi.mojo` | shared-infra | ENTANGLED | 0 | 1 | 2 | 1 | entangled | struct CompiledBindingSnapshot holds provider: PythonObject; module: PythonObject; struct CompiledBindingSnapshots holds modules: List[PythonObject]; factories: |
| `providers/python/kernel_method_descriptor_pyapi.mojo` | shared-infra | SPLIT | 0 | 1 | 1 | 1 | split-unpack-pack | split: kernel_method_execute |
| `providers/python/kernel_public_lifecycle_pyapi.mojo` | shared-infra | ENTANGLED | 1 | 1 | 8 | 2 | entangled | kernel_public_finite: RUN code on Python-holding type ArrayMetadata |
| `providers/python/lowbit_weight_pyapi.mojo` | shared-infra | CLEAN | 1 | 0 | 8 | 5 | split-unpack-pack |  |
| `providers/python/native_execution_pyapi.mojo` | shared-infra | ENTANGLED | 2 | 2 | 1 | 0 | entangled | struct AdmittedNativeExecution holds provider: PythonObject; session_owner: PythonObject; AdmittedNativeExecution: typed work on py-struct NeuralDescriptorProvi |
| `providers/python/operation_lock_pyapi.mojo` | shared-infra | ENTANGLED | 1 | 1 | 3 | 2 | entangled | struct OperationLockBoundary holds cls: PythonObject; getter: PythonObject; struct OperationLockLease holds value: PythonObject; owner: PythonObject |
| `bindings/_mojolearn_solver.mojo` | solver | CLEAN | 2 | 0 | 1 | 5 | ok |  |
| `bindings/_mojolearn_svm.mojo` | svm | SPLIT | 3 | 5 | 6 | 10 | ok | split: svc_fit_binding, svc_predict_binding, svr_fit_binding, svr_predict_binding +1 |
| `bindings/_mojolearn_svm_host.mojo` | svm | SPLIT | 0 | 1 | 4 | 11 | ok | split: iforest_run_binding |
| `bindings/svm_mixture_native.mojo` | svm | ENTANGLED | 0 | 1 | 4 | 3 | entangled | struct QnScoreInput holds source: PythonObject |
| `bindings/_mojolearn_tokenizer_host.mojo` | tokenizer | SPLIT | 1 | 2 | 15 | 24 | ok | split: _table_from_spelled, vocab_init_public_binding |
| `bindings/cross_vendor_native.mojo` | tokenizer | ENTANGLED | 0 | 2 | 19 | 17 | entangled | struct CrossVendorBoundary holds bi: PythonObject; arrays: PythonObject |
| `bindings/embedding_layer_native.mojo` | tokenizer | ENTANGLED | 3 | 6 | 18 | 8 | entangled | struct EmbeddingInput holds owner: PythonObject; EmbeddingInput: typed work on py-struct AdmittedBorrowedBuffer |
| `bindings/full_reset_constructor_native.mojo` | tokenizer | SPLIT | 0 | 1 | 3 | 2 | split-unpack-pack | split: _bind |
| `bindings/full_reset_params_native.mojo` | tokenizer | SPLIT | 0 | 1 | 3 | 6 | split-unpack-pack | split: _release_owned |
| `bindings/numeric_profile_native.mojo` | tokenizer | ENTANGLED | 0 | 1 | 10 | 6 | entangled | struct ProfileAdmission holds context: PythonObject; dicts: List[PythonObject] |
| `bindings/optimizer_device_native.mojo` | tokenizer | ENTANGLED | 0 | 2 | 6 | 5 | entangled | struct DeviceAllocation holds owner: PythonObject  # unpublished public outpu |
| `bindings/optimizer_native_router.mojo` | tokenizer | SPLIT | 0 | 1 | 5 | 7 | split-unpack-pack | split: optimizer_state_array |
| `bindings/optimizer_serialization_boundary.mojo` | tokenizer | CLEAN | 1 | 0 | 0 | 1 | rename-pack |  |
| `bindings/samba_head_native.mojo` | tokenizer | SPLIT | 0 | 1 | 0 | 0 | rename-unpack | split: training_samba_head_loss |
| `bindings/sequence_layers_native.mojo` | tokenizer | ENTANGLED | 2 | 0 | 8 | 6 | entangled | struct LayerInput holds owner: PythonObject; handle: PythonObject; _width_known: RUN code on Python-holding type LayerInput |
| `bindings/sequence_moe_native.mojo` | tokenizer | ENTANGLED | 3 | 0 | 10 | 9 | entangled | _check: RUN code on Python-holding type SequenceFloatBuffer |
| `bindings/sequence_optimizer_native.mojo` | tokenizer | SPLIT | 0 | 1 | 6 | 11 | split-unpack-pack | split: seq_step |
| `bindings/tokenizer_public_native.mojo` | tokenizer | ENTANGLED | 1 | 7 | 34 | 34 | entangled | struct TokenizerPublicConnection holds vocab: PythonObject; bpe: PythonObject; struct TokenizerUtf8Policy holds expected: PythonObject |
| `bindings/training_embedding_native.mojo` | tokenizer | ENTANGLED | 0 | 4 | 3 | 2 | entangled | embedding_indices: typed work on py-struct OptimizerInputOwners |
| `bindings/training_extra_native.mojo` | tokenizer | ENTANGLED | 0 | 3 | 4 | 3 | entangled | training_mlp_sessions: typed work on py-struct OptimizerInputOwners |
| `bindings/training_linear_native.mojo` | tokenizer | SPLIT | 1 | 2 | 1 | 0 | rename-unpack | split: linear_device, training_linear_operation |
| `bindings/training_rms_native.mojo` | tokenizer | SPLIT | 0 | 1 | 1 | 0 | rename-unpack | split: training_rms_operation |
| `bindings/_mojolearn_training.mojo` | training | CLEAN | 2 | 0 | 6 | 33 | ok |  |
| `bindings/residual_dropout_boundary.mojo` | training | SPLIT | 0 | 2 | 1 | 2 | split-unpack-pack | split: residual_dropout_dev_binding, residual_dropout_backward_dev_binding |
| `bindings/residual_dropout_common.mojo` | training | CLEAN | 3 | 0 | 1 | 0 | rename-unpack |  |
| `providers/python/training_samba_afn_pyapi.mojo` | training | CLEAN | 2 | 0 | 2 | 2 | split-unpack-pack |  |
| `bindings/_mojolearn_transformer.mojo` | transformer | SPLIT | 0 | 10 | 3 | 17 | ok | split: transformer_session_forward_fresh_binding, transformer_session_backward_binding, transformer_session_forward_binding, transformer_session_forward_int15_b |
| `bindings/_mojolearn_transformer_host.mojo` | transformer | SPLIT | 0 | 3 | 4 | 12 | ok | split: transformer_forward_fresh_binding, transformer_forward_binding, transformer_decode_step_binding |
| `bindings/transformer_public_native.mojo` | transformer | ENTANGLED | 1 | 2 | 12 | 4 | entangled | struct _GenerationStamp holds key: PythonObject; struct TransformerBuffer holds owner: PythonObject; store: PythonObject |
| `providers/python/transformer_decode_pyapi.mojo` | transformer | ENTANGLED | 1 | 6 | 18 | 7 | entangled | struct TransformerDecodeHandle holds binding: PythonObject; block: PythonObject; struct TransformerDecodeWords holds vendor_entries: List[PythonObject]; mode_en |
| `bindings/_mojolearn_trees.mojo` | trees | SPLIT | 7 | 3 | 7 | 11 | ok | split: _et_export_verify_native, _et_classifier_fit, _et_regressor_fit |
| `bindings/_mojolearn_trees_host.mojo` | trees | ENTANGLED | 5 | 3 | 10 | 12 | ok | et_host_forest_entry_v1: RUN code on Python-holding type EtHostForest |
| `bindings/trees_explainer_native.mojo` | trees | ENTANGLED | 0 | 3 | 10 | 5 | entangled | struct TreeShapOwner holds state: PythonObject; release: PythonObject |
| `bindings/trees_public_forest.mojo` | trees | ENTANGLED | 5 | 8 | 44 | 27 | entangled | struct ForestInput holds native: PythonObject; _fused_labels_capable: RUN code on Python-holding type ForestInput |
| `bindings/trees_public_inputs.mojo` | trees | ENTANGLED | 4 | 2 | 9 | 7 | entangled | struct TreesLabels holds classes: PythonObject; codes: PythonObject; _trees_of: RUN code on Python-holding type ClassicalLabels,TreesLabels |
| `bindings/trees_resident_table.mojo` | trees | ENTANGLED | 2 | 3 | 11 | 1 | entangled | struct TreesResidentEntry holds native: PythonObject; release: PythonObject; _put: typed work on py-struct TreesResidentEntry |
| `bindings/trees_side_records.mojo` | trees | CLEAN | 1 | 0 | 12 | 0 | rename-unpack |  |
| `bindings/_mojolearn_tsa.mojo` | tsa | CLEAN | 2 | 0 | 0 | 10 | ok |  |
| `bindings/tsa_arima_resident.mojo` | tsa | CLEAN | 1 | 0 | 2 | 1 | split-unpack-pack |  |
| `bindings/tsa_hw_native.mojo` | tsa | SPLIT | 0 | 4 | 16 | 10 | split-unpack-pack | split: hw_fit, hw_predict, hw_component, hw_save |
| `bindings/_mojolearn_x_ann.mojo` | x_ann | SPLIT | 0 | 4 | 0 | 10 | ok | split: ivf_pq_build_binding, ivf_pq_build_resident_binding, ivf_sq_build_binding, ivf_rabitq_build_binding |
| `bindings/_mojolearn_x_ann_host.mojo` | x_ann | SPLIT | 0 | 1 | 1 | 14 | ok | split: ivf_sq_build_binding |
| `bindings/cluster_centroid_fit_boundary.mojo` | x_cluster | ENTANGLED | 0 | 1 | 2 | 0 | entangled | struct CentroidFitConnection holds provider: PythonObject; struct CentroidPackedResult holds centers: PythonObject; affinity: PythonObject |
| `bindings/cluster_fit_a_native.mojo` | x_cluster | SPLIT | 0 | 4 | 23 | 5 | split-unpack-pack | split: _meanshift_fit, _affinity_fit, _hdbscan_fit, _hdbscan_prediction_data |
| `bindings/cluster_native.mojo` | x_cluster | SPLIT | 0 | 2 | 3 | 4 | split-unpack-pack | split: _centers_query, cluster_operation |
| `bindings/cluster_resident_table.mojo` | x_cluster | ENTANGLED | 3 | 3 | 10 | 0 | entangled | struct CresHeld holds binding: PythonObject; _table: RUN code on Python-holding type CresTable |
| `providers/python/cluster_public_lifecycle_pyapi.mojo` | x_cluster | CLEAN | 1 | 0 | 2 | 4 | split-unpack-pack |  |
| `bindings/_mojolearn_x_cnn_host.mojo` | x_cnn | SPLIT | 0 | 4 | 44 | 14 | ok | split: _block_forward, conv_block_backward_binding, sgd_binding, adam_binding |
| `bindings/_mojolearn_x_decomp.mojo` | x_decomp | CLEAN | 1 | 0 | 4 | 0 | ok |  |
| `bindings/_mojolearn_x_decomp_host.mojo` | x_decomp | CLEAN | 1 | 0 | 4 | 4 | ok |  |
| `bindings/_mojolearn_x_linear.mojo` | x_linear | CLEAN | 1 | 0 | 4 | 11 | ok |  |
| `bindings/_mojolearn_x_linear_host.mojo` | x_linear | SPLIT | 1 | 1 | 1 | 12 | ok | split: fit_binding |
| `bindings/linear_constructor_native.mojo` | x_linear | SPLIT | 0 | 2 | 0 | 1 | rename-pack | split: _constructor_plan, _initialize |
| `providers/python/linear_entry_boundary.mojo` | x_linear | ENTANGLED | 0 | 1 | 5 | 0 | entangled | struct LinearEntryOwner holds module: PythonObject; struct LinearEntryConnection holds images: PythonObject |
| `providers/python/linear_metadata_boundary.mojo` | x_linear | SPLIT | 0 | 1 | 4 | 5 | split-unpack-pack | split: classical_set_params |
| `providers/python/linear_score_boundary.mojo` | x_linear | SPLIT | 0 | 1 | 1 | 1 | split-unpack-pack | split: classical_linear_score |
| `providers/python/linear_state_boundary.mojo` | x_linear | ENTANGLED | 0 | 1 | 6 | 0 | entangled | struct LinearStateOwner holds coefficients: PythonObject; means: PythonObject; struct LinearStateCall holds owner: PythonObject |
| `bindings/_mojolearn_x_metrics.mojo` | x_metrics | CLEAN | 1 | 0 | 1 | 5 | ok |  |
| `bindings/_mojolearn_x_metrics_host.mojo` | x_metrics | CLEAN | 1 | 0 | 1 | 8 | ok |  |
| `bindings/neighbors_ann_native.mojo` | x_neighbors | SPLIT | 0 | 1 | 8 | 11 | split-unpack-pack | split: _refine |
| `bindings/neighbors_ann_query_native.mojo` | x_neighbors | ENTANGLED | 1 | 3 | 8 | 4 | entangled | struct _StoredArray holds value: PythonObject; owner: PythonObject; _addresses: RUN code on Python-holding type _StoredArray |
| `bindings/neighbors_kde_native.mojo` | x_neighbors | ENTANGLED | 0 | 1 | 12 | 4 | entangled | struct KDEDense holds array: PythonObject; struct KDEResidentLookup holds owner: PythonObject |
| `bindings/neighbors_knn_native.mojo` | x_neighbors | ENTANGLED | 0 | 4 | 41 | 14 | entangled | struct _AdmittedToken holds token: PythonObject |
| `providers/python/neighbors_public_lifecycle_pyapi.mojo` | x_neighbors | ENTANGLED | 2 | 1 | 3 | 3 | entangled | struct NeighborPublicCall holds args: PythonObject; fields: PythonObject |
| `bindings/prep_lda_admission.mojo` | x_prep | SPLIT | 0 | 1 | 8 | 5 | split-unpack-pack | split: lda_native_fields_plain |
| `bindings/sequence_array_native.mojo` | x_sequence | SPLIT | 0 | 1 | 5 | 0 | rename-unpack | split: sequence_array_cast |
| `bindings/sequence_dtype_native.mojo` | x_sequence | SPLIT | 0 | 2 | 6 | 7 | split-unpack-pack | split: sequence_dtype_syntax, sequence_dtype_text |
| `bindings/sequence_string_storage.mojo` | x_sequence | ENTANGLED | 4 | 1 | 4 | 2 | entangled | struct SequenceStringStorage holds module: PythonObject; capsule: PythonObject |
| `providers/python/autograd_device_provider_pyapi.mojo` | x_sequence | SPLIT | 1 | 1 | 1 | 0 | rename-unpack | split: imported_device_provider_run |
| `providers/python/autograd_distributed_pyapi.mojo` | x_sequence | SPLIT | 0 | 1 | 6 | 7 | split-unpack-pack | split: distributed_fit_py |
| `providers/python/autograd_owned_program_dispatch_pyapi.mojo` | x_sequence | CLEAN | 1 | 0 | 2 | 0 | rename-unpack |  |
| `providers/python/autograd_partition_pyapi.mojo` | x_sequence | SPLIT | 0 | 1 | 2 | 9 | split-unpack-pack | split: partition_fit_py |
| `providers/python/autograd_program_device_pyapi.mojo` | x_sequence | SPLIT | 1 | 1 | 0 | 0 | rename-unpack | split: ops_imported_run_device_py |
| `providers/python/autograd_torch_program_pyapi.mojo` | x_sequence | ENTANGLED | 1 | 1 | 6 | 6 | entangled | struct TorchDeviceSnapshot holds owner: PythonObject; struct TorchArtifactBase holds base: PythonObject; modules: PythonObject |
| `providers/python/sequence_seq_tensor_pyapi.mojo` | x_sequence | CLEAN | 2 | 0 | 0 | 6 | rename-pack |  |
| `providers/python/xtrees_api_pyapi.mojo` | x_trees | SPLIT | 2 | 4 | 7 | 88 | split-unpack-pack | split: _xt_owned, accumulate_cols_many_binding, calibrated_proba_binding, ovr_assemble_binding |

Boundary-only files whose name the flipped fence would refuse (`b/` = `bindings/`, `pp/` = `providers/python/`):

- **autograd**, rename-pack (4): `pp/autograd_control_checkpoint_pyapi`, `pp/autograd_execution_profile_pyapi`, `pp/autograd_gradient_diagnostics_pyapi`, `pp/autograd_program_control_grad_pyapi`
- **autograd**, rename-unpack (1): `pp/autograd_host_provider_pyapi`
- **autograd**, split-unpack-pack (11): `pp/autograd_graph_metadata_pyapi`, `pp/autograd_import_lower_pyapi`, `pp/autograd_import_metadata_pyapi`, `pp/autograd_model_checkpoint_pyapi`, `pp/autograd_program_arrays_pyapi`, `pp/autograd_program_owner_pyapi`, `pp/autograd_program_public_dispatch_pyapi`, `pp/autograd_program_save_pyapi`, `pp/autograd_pyapi`, `pp/autograd_python_errors_pyapi`, `pp/autograd_session_frontend_pyapi`
- **byte_lm**, split-unpack-pack (1): `b/lm_corpus_metadata`
- **core**, rename-unpack (1): `b/labels_host_ops`
- **core**, split-unpack-pack (3): `b/hotpath_helpers`, `b/lr_table_helpers`, `pp/core_py2mojo_rows_pyapi`
- **estimators**, rename-pack (1): `b/labeled_reference_resident_binding`
- **estimators**, split-unpack-pack (2): `b/dbscan_core_device`, `b/py2mojo_cluster_est`
- **forecast**, rename-unpack (1): `pp/forecast_owner_pyapi`
- **forest**, rename-unpack (1): `b/forest_fit_boundary`
- **gp**, rename-unpack (10): `b/gp_argument_pins`, `b/gp_class_table_boundary`, `b/gp_label_order_boundary`, `b/gp_labels_boundary`, `b/gp_mapping_boundary`, `b/gp_normalization_boundary`, `b/gp_resident_boundary`, `b/gp_score_label_storage`, `b/gp_state_admission_native`, `b/gp_state_cache_boundary`
- **gp**, split-unpack-pack (2): `b/gp_label_snapshot_native`, `b/gp_svm_records`
- **hdbscan**, rename-pack (1): `b/hdbscan_host_predict`
- **hdbscan**, split-unpack-pack (1): `b/hdbscan_fit_boundary`
- **kernel_methods**, rename-pack (1): `pp/kernel_methods_rbf_resident_pyapi`
- **linalg**, rename-unpack (1): `b/linalg_ops_native`
- **linalg**, split-unpack-pack (1): `pp/linalg_result_pyapi`
- **metrics**, rename-unpack (2): `b/metrics_cls_native`, `b/metrics_plan_admission`
- **metrics**, split-unpack-pack (1): `b/metrics_native_support`
- **mixture**, rename-pack (1): `b/mixture_host_scoring`
- **neural**, split-unpack-pack (1): `b/neural_prefill_native`
- **preprocessing**, split-unpack-pack (1): `pp/preprocessing_input_pyapi`
- **resample**, rename-pack (1): `pp/resample_rows_operation_pyapi`
- **runtime_byte_trainer**, split-unpack-pack (1): `pp/byte_lm_json_checkpoint_pyapi`
- **runtime_classical**, rename-pack (1): `b/classical_error_boundary`
- **runtime_classical**, rename-unpack (5): `b/classical_caller`, `b/classical_host_owner`, `b/classical_warnings`, `pp/linear_execution_boundary`, `pp/linear_parameter_boundary`
- **runtime_classical**, split-unpack-pack (4): `b/classical_archive_linear`, `b/classical_arrays`, `b/classical_attributes`, `b/classical_exceptions`
- **runtime_decomp**, rename-pack (1): `b/pca_archive`
- **runtime_decomp**, rename-unpack (5): `b/decomp_inputs`, `b/tsvd_owner_boundary`, `pp/jl_scalar_pyapi`, `pp/pca_input_pyapi`, `pp/pca_query_cache_pyapi`
- **runtime_decomp**, split-unpack-pack (5): `b/decomp_native`, `b/pca_metadata`, `b/pca_native`, `b/rp_native`, `pp/pca_fit_model_pyapi`
- **runtime_family**, rename-pack (2): `b/gp_native`, `pp/pcs_rng_pyapi`
- **runtime_family**, rename-unpack (10): `b/cluster_fit_b_native`, `b/hdbscan_query_boundary`, `b/lof_provider`, `b/metrics_exp_native`, `b/nearest_centroid_provider`, `b/ocsvm_output_boundary`, `b/ocsvm_provider`, `b/spectral_preparation_boundary`, `b/svr_native_boundary`, `pp/hierarchy_settings_pyapi`
- **runtime_family**, split-unpack-pack (17): `b/cfb_dbscan`, `b/classical_family_door`, `b/family_native`, `b/family_raw_attr`, `b/neighbors_native`, `b/parallel_public_native`, `b/pcs_native_boundary`, `b/spectral_assignment_boundary`, `b/svc_native`, `pp/density_public_lifecycle_pyapi`, `pp/hierarchy_public_lifecycle_pyapi`, `pp/host_estimator_binding_pyapi`, `pp/neighbors_feature_names_pyapi`, `pp/pcs_public_lifecycle_pyapi`, `pp/spectral_lifecycle_pyapi`, `pp/svc_descriptor_pyapi`, `pp/svc_public_state_pyapi`
- **runtime_linear**, rename-unpack (2): `b/logistic_config_boundary`, `b/logistic_state_boundary`
- **runtime_linear**, split-unpack-pack (3): `b/logistic_admission`, `b/qn_boundary`, `b/qn_native`
- **runtime_manifold**, rename-pack (2): `b/manifold_native`, `b/runtime_manifold_entry`
- **runtime_manifold**, split-unpack-pack (3): `b/manifold_scalar_native`, `b/manifold_settings_native`, `b/manifold_storage_native`
- **runtime_model_selection**, rename-pack (2): `b/msel_parallel_witness`, `b/runtime_model_selection_entry`
- **runtime_model_selection**, rename-unpack (3): `b/arena_resident_plan`, `b/msel_helpers_key`, `b/msel_plan_records`
- **runtime_model_selection**, split-unpack-pack (7): `b/model_selection_native`, `b/msel_estimator_native`, `b/msel_json`, `b/msel_public_metadata`, `b/msel_scorer_native`, `b/msel_search_state`, `b/parallel_cv_native`
- **runtime_neural**, rename-pack (1): `b/samba_cpu_inference`
- **runtime_neural**, rename-unpack (5): `b/causal_exception`, `b/optimizer_scalar_boundary`, `b/samba_optimizer_boundary`, `pp/causal_session_handles_pyapi`, `pp/samba_program_handle_pyapi`
- **runtime_neural**, split-unpack-pack (5): `b/causal_checkpoint`, `b/optimizer_admission`, `b/optimizer_owner_boundary`, `b/samba_admission`, `b/samba_config_native`
- **runtime_parallel**, rename-unpack (2): `b/parallel_neural_admission`, `pp/autoarima_owner_pyapi`
- **runtime_parallel**, split-unpack-pack (4): `b/neighbor_query_native`, `b/parallel_plan_door`, `b/parallel_public_result`, `b/small_mlp_items`
- **runtime_prep**, rename-pack (1): `b/labels_native_entry`
- **runtime_prep**, rename-unpack (2): `b/p3_prep_fitted`, `pp/prep_metadata_pyapi`
- **runtime_prep**, split-unpack-pack (6): `b/prep_composition_lifecycle`, `b/prep_context_native`, `b/prep_normalizer_fit`, `b/prep_normalizer_lifecycle`, `b/prep_owner_native`, `b/prep_support_native`
- **runtime_sequential**, rename-pack (1): `b/sequential_native`
- **runtime_sequential**, split-unpack-pack (2): `b/sequential_description`, `b/sequential_weights`
- **runtime_trees**, rename-pack (4): `b/iforest_public_lifecycle`, `b/trees_bagging_members_native`, `b/trees_gbdt_parameters`, `b/trees_public_meta_lifecycle`
- **runtime_trees**, split-unpack-pack (14): `b/classical_trees_door`, `b/host_archive_predict_native`, `b/host_density_archive_native`, `b/host_density_constructor_native`, `b/host_iforest_fit_native`, `b/host_iforest_native`, `b/host_kde_native`, `b/host_projection_native`, `b/trees_boost_admission`, `b/trees_boost_constructor_native`, `b/trees_constructor_native`, `b/trees_gbdt_adapter_lifecycle`, `b/trees_public_lifecycle_native`, `b/trees_public_native`
- **runtime_xseq**, rename-pack (2): `b/xcnn_native`, `b/xseq_native`
- **runtime_xseq**, split-unpack-pack (7): `b/xcnn_common`, `b/xcnn_conv_native`, `b/xcnn_pool_native`, `b/xcnn_tensor_native`, `pp/forecast_initialize_pyapi`, `pp/forecast_matrix_pyapi`, `pp/prophet_lifecycle_pyapi`
- **shared-infra**, rename-pack (15): `b/arena_array_metadata`, `b/device_plan_binding`, `b/host_array_native`, `b/labels_stat_boundary`, `b/native_array_view`, `b/native_buffer_owner`, `b/native_caller_frame`, `b/native_instance_item`, `b/native_mode_recipe`, `b/native_typed_context`, `b/p3_linear_helpers`, `pp/framework_metadata_pyapi`, `pp/hierarchy_cluster_descriptor_pyapi`, `pp/hierarchy_solver_descriptor_pyapi`, `pp/metric_binding_pyapi`
- **shared-infra**, rename-unpack (24): `b/admitted_meta_scope`, `b/archive_unicode_native`, `b/base_helpers_key`, `b/fitted_resident_door`, `b/host_inference_features`, `b/host_inference_state`, `b/kde_admission_boundary`, `b/kmeans_variants_resident_boundary`, `b/kpss_host_test`, `b/label_integer_storage`, `b/label_weights_boundary`, `b/native_buffer_address`, `b/native_identity_type`, `b/native_ops_key_ann`, `b/native_type_module`, `b/p3_guard_records`, `b/p3_lm_tokenizer_cpython`, `b/p3nc_label_lists`, `b/perf3_label_doors_text`, `b/spectral_native_boundary`, `b/spectral_resident_boundary`, `pp/borrowed_buffer_pyapi`, `pp/estimator_lifecycle_pyapi`, `pp/estimator_tabular_boundary_pyapi`
- **shared-infra**, split-unpack-pack (41): `b/arena_buffer_copy`, `b/arena_cache_binding`, `b/arena_cache_release`, `b/arena_cache_stats`, `b/byte_checkpoint_io`, `b/byte_perf_helpers`, `b/cholesky_archive_native`, `b/cpython_borrowed`, `b/cross_vendor_admission`, `b/declared_params_native`, `b/generator_state_native`, `b/host_archive_lifetime`, `b/host_inference_scalars`, `b/host_logistic_query_boundary`, `b/host_svc_native`, `b/host_svr_native`, `b/kde_native_boundary`, `b/kit_call`, `b/label_admission_native`, `b/label_list_words`, `b/label_sequence_source`, `b/label_snapshot_native`, `b/msel_fixed_owner`, `b/native_array_storage`, `b/native_array_storage_admission`, `b/native_numpy_metadata`, `b/native_owner_lifetime`, `b/native_torch_admission`, `b/p3_numpy_types`, `b/p3_samba_caps`, `b/p3nc_fitted_freeze`, `b/pandas_preference_items`, `b/params_metadata_native`, `b/params_setter_native`, `b/public_attributes_native`, `b/sha256_door`, `b/spectral_umap_routes`, `pp/array_layout_boundary`, `pp/estimator_binding_pyapi`, `pp/x_neighbors_admission_pyapi`, `pp/x_neighbors_descriptor_pyapi`
- **svm**, rename-pack (3): `pp/svm_host_scale_gamma_host_pyapi`, `pp/svm_host_svc_ovo_host_pyapi`, `pp/svm_impl_scale_gamma_device_pyapi`
- **svm**, split-unpack-pack (3): `pp/svm_host_svc_proba_pyapi`, `pp/svm_impl_svc_epilogue_pyapi`, `pp/svm_impl_svc_ovo_layout_pyapi`
- **tokenizer**, rename-pack (3): `b/arena_cache_lifecycle`, `b/runtime_sequential_forward`, `b/training_metadata_native`
- **tokenizer**, rename-unpack (2): `b/runtime_classical_unavailable`, `pp/sequence_layer_buffer_pyapi`
- **tokenizer**, split-unpack-pack (9): `b/arena_resident_native`, `b/arena_resident_rows`, `b/arena_rows_native`, `b/backend_public_native`, `b/generator_draw_native`, `b/optimizer_lifecycle`, `b/sequence_device_native`, `b/training_cross_entropy_native`, `b/training_loss_options`
- **training**, rename-pack (3): `b/neural_gemm_boundary`, `b/neural_gemm_boundary_host`, `b/residual_dropout_boundary_host`
- **training**, rename-unpack (2): `b/neural_gemm_boundary_common`, `pp/training_metadata_pyapi`
- **training**, split-unpack-pack (3): `b/optimizer_public_host_pyapi`, `b/optimizer_public_pyapi`, `b/training_scalar_native`
- **trees**, split-unpack-pack (3): `b/trees_lifetime`, `b/trees_public_iforest`, `b/trees_public_state`
- **tsa**, split-unpack-pack (1): `b/tsa_admission`
- **x_ann**, rename-pack (1): `pp/x_ann_resident_pyapi`
- **x_ann**, rename-unpack (1): `pp/x_ann_abi_pyapi`
- **x_cluster**, rename-pack (1): `pp/cluster_constructor_pyapi`
- **x_cluster**, split-unpack-pack (2): `b/cluster_side_records`, `pp/x_cluster_out_pyapi`
- **x_decomp**, rename-pack (8): `pp/x_decomp_graph_device_pyapi`, `pp/x_decomp_lanczos_dev_pyapi`, `pp/x_decomp_lanczos_host_pyapi`, `pp/x_decomp_lle_iter_dev_pyapi`, `pp/x_decomp_mds_dev_pyapi`, `pp/x_decomp_qfix_pyapi`, `pp/x_decomp_s_linalg_fast_pyapi`, `pp/x_decomp_select_dev_pyapi`
- **x_decomp**, rename-unpack (3): `b/decomp_scalar_admission`, `b/decomp_scalar_compare`, `b/decomp_truth_boundary`
- **x_decomp**, split-unpack-pack (9): `pp/x_decomp_api_pyapi`, `pp/x_decomp_decomp_owner_pyapi`, `pp/x_decomp_graph_host_pyapi`, `pp/x_decomp_ipca_owner_pyapi`, `pp/x_decomp_lle_iter_pyapi`, `pp/x_decomp_mds_pyapi`, `pp/x_decomp_resident_pyapi`, `pp/x_decomp_select_ops_pyapi`, `pp/x_decomp_tsvd_owner_pyapi`
- **x_linear**, rename-unpack (1): `b/linear_svc_boundary`
- **x_neighbors**, split-unpack-pack (2): `pp/x_neighbors_dev_pyapi`, `pp/x_neighbors_host_pyapi`
- **x_prep**, rename-pack (1): `pp/x_prep_fastnb_csr_pyapi`
- **x_prep**, rename-unpack (1): `b/prep_entry_cache`
- **x_prep**, split-unpack-pack (1): `b/prep_native_estimator`
- **x_sequence**, rename-pack (4): `pp/autograd_import_resolve_pyapi`, `pp/autograd_model_verify_pyapi`, `pp/autograd_program_public_request_pyapi`, `pp/autograd_session_lifetime_pyapi`
- **x_sequence**, rename-unpack (3): `b/sequence_print_options`, `pp/autograd_owned_program_factory_pyapi`, `pp/autograd_session_public_dispatch_pyapi`
- **x_sequence**, split-unpack-pack (8): `b/sequence_array_warnings`, `pp/autograd_import_reader_pyapi`, `pp/autograd_session_public_request_pyapi`, `pp/sequence_layernorm_dev_pyapi`, `pp/sequence_moe_dev_pyapi`, `pp/sequence_opt_resident_pyapi`, `pp/sequence_pyapi`, `pp/small_mlp_program_pyapi`
