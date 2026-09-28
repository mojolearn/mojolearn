# ann-apple3: Apple (Metal) FAST speed, round 3

Lane `ann-apple3`, branch `lane/ann-apple3`, worktree `~/mojolearn-wt/ann-apple3`,
forked from lane/apple3-merged 6856b5f8f (brief:
`~/mojolearn-evidence/apple3_speed_brief.md`). Family: IVF-Flat / PQ / SQ /
RaBitQ, refine, CAGRA, t-SNE (approximate nearest neighbors, classical ML).
cluster/ k-means (the IVF coarse quantizer and the PQ codebook k-means)
belongs to cluster-apple3 and is not edited here.

Bench: `bench/speed/ann_cpu_speed.py` driven by `tools/ann_apple2_ab.sh`
(round 2's script and shapes: IVF family 1M x 28 HIGGS, 1024 lists, 32
probes, 1000 queries, k = 10, 10 k-means iterations; IVF-PQ pq_dim 14, 8
bits; CAGRA 50k; t-SNE 10k, 300 iterations). New in this round: the bench
times a SECOND search of the same queries (`search2_s`, the index resident
and the kernels warm) and prints its digest (`out2`, must equal `out`); the
stage pass also sets `MOJOLEARN_KMEANS_STAGES=1`. Quality:
`bench/speed/ann_fast_quality.py` (recall at 10 against exact search;
trustworthiness and KL for t-SNE), HIGGS and taxi.

## Targets, from the round 2 measurements (M3 Ultra, FAST, final_m3ultra-b_1790613110335)

| cell | FAST s | where the time is (stage pass, ms) |
|---|---|---|
| IVF-PQ fit | 1.74 | codebooks 1085 (14 cluster/ k-means calls), coarse 512, residuals 68, encode 57 |
| t-SNE fit (10k, 300 it) | 0.68 (0.57 with the base driver) | iterations 470 to 620, symmetrize 31, k-NN 20 |
| IVF-SQ fit | 0.60 | coarse 493, encode 57, residuals 16 |
| IVF-Flat fit | 0.59 | not split before this round |
| IVF-RaBitQ fit | 0.52 | coarse 496, encode 17 |
| CAGRA fit (50k) | 0.15 | k-NN + prune 136, reverse merge 12 |
| IVF-Flat search | 0.065 | not split; first search of a process, includes the index preparation |
| refine search (top-40) | 0.059 | select 6.5 per chunk of 200 queries |
| IVF-SQ / PQ / RaBitQ search | 0.040 / 0.034 / 0.023 | per chunk: score 2, select 1.6, probe 0.6; the list-order gather 5 to 8 once per search |
| CAGRA search | 0.022 | search 19 |

"coarse" is `ivf_flat_build`: cluster/'s k-means and predict, plus this
family's host passes (data check, quantizer scale, upload, list layout,
copies). Round 2 never split it. It is split now (marks below).

Left by round 2 (its FINAL): nothing opt-in except the stage marks; the M3
Ultra never ran the round 2 tip (tree join 3634e4645, probe tree 15498e277);
the t-SNE driver regression on the M3 Ultra was removed by going back to the
base file, cause not isolated. py-dn-ann (resident indexes) has no timing at
all: its proof jobs were cancelled unrun.

## Changes

Every change that is not measured yet is OFF in a default build, behind an
opt-in define in `x_ann/switches.mojo` (Andrew's rule, 20:20Z Sep 28). The
A/B script builds an arm as `<commit>+<DEFINE>[+<DEFINE>]`.

| commit | what | tier | default | state |
|---|---|---|---|---|
| 1479e0c53 | stage marks inside `ivf_flat_build`, the x_ann coarse conversion, the PQ codebook loop, the build bindings, the IVF-Flat prepare binding; bench second search; A/B stage pass sets MOJOLEARN_KMEANS_STAGES | both | marks off unless MOJOLEARN_ANN_STAGES | diagnostic |
| fd948718d, then gated | IVF build host passes (below) | both | OFF; `-D MOJOLEARN_ANN3_HOST_PASSES` turns it on | UNMEASURED |
| 171bbcdba, d20eb2129, then gated | index preparation (below) | both | OFF; `-D MOJOLEARN_ANN3_PREPARE` turns it on | UNMEASURED |

| (this commit) | FAST seeding of the PQ codebooks and of the coarse quantizer (`x_ann/kpp_seed.mojo`) | FAST, Apple | OFF; `-D MOJOLEARN_ANN3_PQ_SEED`, `-D MOJOLEARN_ANN3_COARSE_SEED` | UNBUILT, UNMEASURED; moves FAST bits, owes the paired recall check |

`MOJOLEARN_ANN3_PQ_SEED` / `MOJOLEARN_ANN3_COARSE_SEED` (FAST on Apple
only): cluster/'s k-means seeds with scalable k-means|| and, at the ann
shapes, the seeding costs more than the Lloyd iterations (cluster-apple3's
base probe on m4pro-b, 1790626681580, FAST: 1M x 2, k 256: seeding 115 ms of
which the sequential k-means++ over the candidates 46 ms, 20 Lloyd
iterations 75 ms; 1M x 28, k 1024: seeding 698 ms, 10 Lloyd iterations 368
ms). Under the switch this family seeds on the host (k-means++ over a stride
sample, 16 rows per seed) and passes the seeds to cluster/'s k-means as
`INIT_ARRAY`; cluster/ is not edited. cluster-apple3 lists the k-means
seeding rounds as its own target; if its change lands, these switches may
gain little and stay off.

`MOJOLEARN_ANN3_HOST_PASSES`, host code only, no arithmetic changed:
- `ivf/checks/list_layout.mojo::build_list_layout`: each row moved by one memcpy; `with_data=False` leaves `list_data` empty.
- `ivf_flat_build` / `ivf_flat_build_host`: `with_list_data` (default True). The x_ann coarse step passes False: IVF-PQ / SQ / RaBitQ never read the permuted vectors. A traced build always lays them out.
- `ivf_flat_build`: the layout's lists are moved into the index (copied otherwise, 1M x 28 floats among them).
- `ivf_flat_build.mojo::download_f32/u32`: one memcpy (one append per word otherwise).
- `ivf_flat_build.mojo::upload_f32`: on Apple the copy reads the caller's list (comptime `has_apple_gpu_accelerator()`); other vendors keep the host buffer hop.
- `plan_quantizer_scale`: one row pass with a running sum per column (each column adds its rows in the same ascending order: the same Float64).
- `ivf_validate_data`: a first pass without an exit, by bits; the old loop names the offender when there is one.
- x_ann `_coarse`: centres and offsets moved out of the build; labels are the build's own assignment.
- `x_ann/abi.mojo::out_f32/out_i32`, `bindings/ivf_index_arrays.mojo::ivf_write_index_arrays`: memcpy.

`MOJOLEARN_ANN3_PREPARE`, no arithmetic changed:
- `bindings/_mojolearn_ivf.mojo::ivf_flat_index_prepare_binding`: the admitted arrays move into the index (five copies otherwise).
- `IvfFlatDevice`: the host CSR layout (a copy of the index's lists) is made by `ensure_layout` at the first per-query search (filtered, partial storage, traced, k > 32); the one-launch scans never read it.
- `x_ann/resident.mojo`: a resident IVF-PQ / SQ / RaBitQ index gathers its codes (RaBitQ: norms and factors too) into list order once at prepare (`scan_gather_i32/f32`, the launches `ivf_scan_search` makes per search otherwise); an unfiltered search passes the all-ones filter as its own list-order copy. `ivf_scan_search` and the three `*_search_on` take `have_pre, pre_codes, pre_a, pre_b, mask_pre`; the one-shot entries pass False.

## Measurements

### Job 1 (baseline): m4-a, steward 1790626682431, lane/apple3-merged 6856b5f8f against the round 2 tip 96ea1b210

Queued 20:18Z Sep 28. Purpose: the ann family builds and runs on the merged
base in both tiers, its digests against round 2's, the FAST quality numbers
before any change (HIGGS and taxi, seeds 0 to 2). Result: PENDING.

## SHARED CODE touched (for the consolidation's check)

- `ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo` upload/download helpers are imported by `ivf_flat_search.mojo`, `ivf/checks/ivf_check.mojo` and `neighbors/checks/fused_logical32_check.mojo`.
- `x_ann/abi.mojo` and `bindings/ivf_index_arrays.mojo` are compiled into the CPU host bindings too (`_mojolearn_x_ann_host`, `_mojolearn_ivf_host`, `_mojolearn_ivf_search_host`).
- `ivf/` now imports `x_ann/stage_timer.mojo` (the marks).

## Unproven

Both switches until their A/B is recorded here. Nothing unproven is on in a
default build.
