# gbdt-finish lane, 2026-09-11 night

Finishes the gbdt-speed lane: DEVIATIONS 2634 and 2635 were written there and
never timed (2635 never compiled). This lane compiled them, proved identity,
timed them against their own restore arms, and took two further wins, 2636 and
2661.

## The boxes

| pod | GPU | driver | what it did |
|---|---|---|---|
| 6u9iahnyzvxbm8 | NVIDIA H100 80GB HBM3 | 570.195.03 | REFUSED every GPU fit; reaped |
| dk14p0y15w0ig5 | NVIDIA H100 NVL | 580.159.03 | every number below |

Mojo 1.0.0's GPU runtime requires an NVIDIA driver of 580 or newer (CUDA
13.0); on the first pod every fit raised "Your current NVIDIA GPU driver
version is not supported", and handing it the image's CUDA 12.4 `ptxas`
through `MODULAR_NVPTX_COMPILER_PATH` gave `CUDA_ERROR_INVALID_IMAGE`. That
pod still compiled 2634 and 2635 and passed both host checks. `tools/trees_leg.sh`
now takes `TREES_LEG_CUDA_VERSIONS` so a leg can ask for a CUDA 13.0 host;
H100 80GB HBM3 had no such stock, and H100 NVL did.

Rows here are H100 NVL, driver 580.159.03, and are never mixed with the
80GB HBM3 rows of the same night.

## The sets

One source tree, one switch per set, so a before and after pair is the switch
and nothing else.

| set | defines | switches on |
|---|---|---|
| baseline | `-D MOJOLEARN_2634_CTR_PREP_OFF=1 -D MOJOLEARN_2635_LINEAR_BOUNDS=1 -D MOJOLEARN_2636_SERIAL_STAGING=1` | none |
| a2634 | `-D MOJOLEARN_2635_LINEAR_BOUNDS=1 -D MOJOLEARN_2636_SERIAL_STAGING=1` | 2634 |
| both | `-D MOJOLEARN_2636_SERIAL_STAGING=1` | 2634, 2635 |
| all | none | 2634, 2635, 2636 |
| a2661 | `-D MOJOLEARN_2661_NONSYM_GROUP_WIDTH=1` | 2634, 2635, 2636, 2661 |

Pairs: 2634 is baseline to a2634, 2635 is a2634 to both, 2636 is both to all,
2661 is all to a2661, each read from its own phase so the `all` set is never
compared across two heat windows.

## The deviations

- **2634** (`gbdt/train.mojo`, default on): a fit with no `cat_features`
  skips the CTR target prep (MinEntropy target borders, binarized target, CTR
  estimation orders), whose only readers sit in the categorical branch.
- **2635** (`gbdt/grid_creator/binarization.mojo`, default on): the two border
  split candidates' `LowerBound` and `UpperBound` come from binary searches,
  as CatBoost's do, instead of linear walks that rescan long tied runs.
- **2636** (`gbdt/train.mojo`, default on): the host fills of one compressed
  index ring revolution run in parallel, then the revolution's uploads and
  binarize kernels are enqueued in the serial loop's order with the same
  drain. At Istella-S's 220 x 1,000,000 the serial fill is 880 MB of
  single-thread memcpy inside a ~100 ms stage.
- **2661** (`gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo`,
  opt-in): the depthwise and lossguide histogram build binds DEVIATION 2581's
  per-group bit width, which the symmetric driver has had as its IDENTICAL
  default since 2026-09-11 and this call site never passed.

## Identity (H100 NVL, 2026-09-11)

`tools/identity_break.py` over `gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse`,
then `--diff` against baseline:

| set | cells | stable | moved | refused | diff against baseline |
|---|---|---|---|---|---|
| baseline | 36 | 36 | 0 | 0 | - |
| a2634 | 36 | 36 | 0 | 0 | IDENTICAL=36 |
| both | 36 | 36 | 0 | 0 | IDENTICAL=36 |
| all | 36 | 36 | 0 | 0 | IDENTICAL=36 |

`checks/gbdt_sub_byte_identity_check.py` (each cell fitted twice in one
process, and the hash must equal the NVIDIA H100 reference): 16/16 PASS on
baseline, a2634, both and all.

Host checks, with 2635 on and restored, on both pods:
`pixi run check-binarization` and `pixi run check-greedylogsum` pass on both
arms, and the two arms' outputs are equal apart from tcmalloc warnings.

## Speed

TO FILL from `summarize.py` (medians over 3 alternating rounds of 5 timed
fits, our IDENTICAL arm only, 1M rows) and the opponent cells for this tuple.

## Reproducing

On a driver 580 H100, from the lane branch:

    TREES_LEG_CUDA_VERSIONS=13.0 TREES_LEG_STATE=... TREES_LEG_NAME=gbdt-finish \
      sh tools/trees_leg.sh rent --gpu "NVIDIA H100 80GB HBM3" --minutes 60
    # on the pod, from /root/mojolearn:
    sh body_setup.sh            # four sets, both datasets, both host checks
    sh body_ab.sh "baseline a2634 both all" 3 ab
    sh body_phase2.sh 3 5       # a2661, its identity, its A/B, the opponents
    python3 summarize.py /root/trees_out
