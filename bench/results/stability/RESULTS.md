# Run-to-run determinism: what each tier actually does, per column

The question no other harness in this repository asks. E1/E2/E2U all ask a
CROSS-VENDOR question -- they write cards here so a second machine's cards can
be diffed against them. That inference only exists where a card exists, it
needs two machines, and it says nothing at all about FAST, which is exactly
the arm whose instability the middle tier is sold against.

    tools/repeat_run_stability.py              sequential repeats, one process
    tools/repeat_run_stability.py --concurrent every tier live at once
    tools/e1_bootstrap.sh phase 9              both, on whatever box it runs on

## The headline

| column | fast | deterministic | identical |
|---|---|---|---|
| Apple M4 (Metal) | **MOVED in 8 of 10 attempts** -- 2 or 3 different answers in 24 calls | STABLE 10/10 | STABLE 10/10 |
| NVIDIA RTX 4090 (CUDA 12.8) | **MOVED -- 24 calls returned 24 DIFFERENT answers** | STABLE, 1 answer in 12 calls | STABLE, 1 answer in 12 calls |
| AMD MI325X (ROCm 6.4.1, DigitalOcean) | **MOVED -- 6 different answers in 24 calls**, and the SEQUENTIAL arm moved too: `knn-tied` gave 6 different answers in 6 calls | STABLE, 1 answer in 12 calls | STABLE, 1 answer in 12 calls |

**All three vendors, same verdict: `fast` moves, both upper tiers do not.**
AMD is the only column where the plain SEQUENTIAL arm caught a mover without
the concurrent probe -- `knn-tied`, 6 answers in 6 calls -- where on Apple
that same arm reported STABLE and needed contention between contexts to
expose it. A harness tuned on one vendor under-reports on another.

`fast` moving is `fast` working as specified: it sells speed and makes no
bitwise claim. The point of the table is the other two columns, and the
NVIDIA row is the strongest evidence the middle tier has: **every single call
returned a different answer** where the deterministic build returned one.

The lane is k-NN on a deliberately tie-rich fixture. Ties are the mechanism of
ledger rows 11 and 23 -- which of several equidistant neighbours survives a
merge is decided by the order the blocks won the mutex -- and a uniform
fixture cannot expose it. See [uniform test data hides permutation].

## Sequential arms, NVIDIA RTX 4090, 6 repeats, commit e680bda

| tier | stable | moved | refused |
|---|---|---|---|
| fast | -- | -- | the arm did not run; see THE FAST IMPORT ASYMMETRY below |
| deterministic | 12 | **0** | 4 (`et-clf`, `rf-clf`, `iforest` unbuilt; `gemm-pinned` correctly refuses an identity claim) |
| identical | 13 | **0** | 3 (unbuilt bindings only) |

## AND A CROSS-VENDOR RESULT, FOR FREE

The identical tier's hashes are a fingerprint of the ANSWER, so two columns'
stability runs are also a bit-identity comparison. Apple M4 against NVIDIA
RTX 4090, `identical` tier, through the PYTHON surface:

| | |
|---|---|
| lanes bit-identical on **all three** vendors | **13** |
| divergent | **0** |
| not compared | 3 (`et-clf`, `rf-clf`, `iforest` -- their bindings were not built on either leg) |

Apple M4 (Metal), NVIDIA RTX 4090 (CUDA) and AMD MI325X (HIP), same commit,
through the Python estimator surface.

dbscan, gbdt-depthwise, gbdt-lossguide, gbdt-symmetric, gemm-pinned,
gemm-vendor, kmeans, knn-tied, logistic, ols, pca, ridge, tsvd.

**And the deterministic tier's hashes DIFFER between the columns, which is
the result that shows the tier is not overpaying.** It promises one box
agrees with itself and promises nothing about a second, so 10 of the 12
comparable lanes differ across the three vendors:

| lane | Apple | NVIDIA | AMD |
|---|---|---|---|
| gbdt-depthwise | 757bd55c27047cb0 | 5af5e69c7a51e998 | 55b5854f757cd739 |
| gbdt-lossguide | 82628fff246a204e | d824f0e1683d12b9 | c38591dd2ad30b7f |
| gemm-vendor | 47390974b94e233d | 8e9744e2536e8e2c | e25840533b2ba0cd |
| ols | 8bda051c2e15fb51 | e7456f8e994341c0 | 9a3d2aabdcce9dc2 |
| (10 of 12 comparable lanes differ) | | | |

`gemm-vendor` differing on all three is the specific confirmation that a
deterministic build keeps the vendor matmul, and therefore its speed, rather
than pinning it. Two lanes (`dbscan`, `kmeans`) coincide on all three
columns: that is bit-inertness, not a promise, and must not be read as one.

## THE FAST IMPORT ASYMMETRY (found here, fixed here)

The fast sequential arm produced no table. Traceback:

    ImportError: cannot import name '_mojolearn_trees' from partially
    initialized module 'mojolearn' (most likely due to a circular import)

which names the wrong cause. There is no circular import. The leg built four
of the ten bindings on purpose, and an upper tier gets a `_MissingUpperTier`
stub for each binding that did not build -- the package imports, and the
estimators needing that binding raise BY NAME on use. `select()` returned
early for fast and installed no stubs, so one absent `.so` took the whole
library down at import.

Same partial build, two entirely different outcomes, decided only by which
tier was asked for. Fixed: fast now gets the same stubs. Verified by
sabotage -- hide `_mojolearn_trees.so`, and the package still imports while
`ExtraTreesClassifier(...).fit(...)` raises "numeric_mode='fast' needs
python/mojolearn/_mojolearn_trees.so, which is not built".

## What it took to get one NVIDIA column (three legs)

Recorded because the failures were the harness's, not the silicon's, and the
next person sizing a leg needs them.

1. **The host driver is not the image's to choose.** Driver 570.195.03 (CUDA
   12.8) against MAX's required >= 580; every lane in every tier refused at
   the first kernel launch. Pinning another image does not fix it -- the image
   pins CUDA, the driver belongs to whichever host was scheduled. The phase8
   remote body now points `MODULAR_NVPTX_COMPILER_PATH` at a system `ptxas`.
2. **Pod build time is not Mac build time.** Thirty builds (10 bindings x 3
   tiers) take ~3 minutes locally and ~50 minutes on the pod, which blew the
   54-minute work bound and killed the first measurement arm. Phase 9's build
   set is now `MOJOLEARN_P9_BINDINGS`, four bindings on a leg.
3. Third leg: `bootstrap_exit=0`, twelve builds, both upper arms measured.

Every box was terminated at the end of its work and the termination VERIFIED
against the API (HTTP 404), never assumed.

## Owed

* ~~**AMD**~~ **DONE** -- see the table above. Recorded here because the
  route matters: RunPod's MI300X was created TWICE and never produced an
  ssh endpoint, at 600s and again at 1200s. That is not slow provisioning.
  Both boxes were terminated by the readiness guard and verified gone, and
  the column was taken on a DigitalOcean MI325X through
  `tools/e2_remote_leg.sh` instead. Superseded note:
* ~~The whole reason the tier's promise is still a one-and-a-half
  column result. First attempt 2026-08-29: an MI300X was created and never
  produced an ssh endpoint inside the 600s readiness window, so the leg
  terminated it and exited -- the box billed for ten minutes unarmed, which
  is exactly the window that timeout exists to close. Retried at
  `--ready-timeout 1200`; MI300X provisioning is slow enough that 600s is not
  a safe default for this vendor.
* **The GEMM question.** `gemm-vendor` is STABLE on Apple and on NVIDIA at
  256x4096 @ 4096x128, a wide k chosen to provoke a split-K epilogue. That is
  two columns of evidence that rows 24/27/28/40/41 are cross-vendor class and
  the middle tier keeps their speed. **rocBLAS is now measured too: STABLE at the same shape.** So MAX `matmul`,
  cuBLAS and rocBLAS are each run-to-run reproducible there, ledger rows
  24/27/28/40/41 are cross-vendor class ON EVIDENCE rather than on the
  ledger's say-so, and `linkage_check.mojo:583`'s speculation that "a
  split-K or atomic epilogue may land differently" is answered: not at
  this shape, on any of the three.
* `et-clf`, `rf-clf`, `iforest` on any non-Apple column: their bindings have
  not been built on a leg yet.
