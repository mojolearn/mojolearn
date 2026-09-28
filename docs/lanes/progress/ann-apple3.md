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

| (this commit) | FAST IVF-PQ: the codebook sample's residuals formed on the host, the 1M x 28 residual matrix not downloaded | FAST | OFF; `-D MOJOLEARN_ANN3_PQ_HOST_RESIDUALS` | UNBUILT, UNMEASURED; expected to move no bit (FAST's residual is one subtraction) |

The quality bench has an IVF-Flat row now (`--algos ivf`): the coarse
seeding moves IVF-Flat's FAST results too.

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

### Job 1 (baseline): m4-a (Apple M4), steward 1790626682431, lane/apple3-merged 6856b5f8f against the round 2 tip 96ea1b210

PASS. Raw: `~/mojolearn-evidence/ann-apple3/job1_m4-a_1790626682431.txt`.
Three runs per cell (two reps and the stage pass). The ann family builds and
runs on the merged base in both tiers; every digest equals round 2's in both
tiers. FAST did not move (nothing FAST changed between the two); IDENTICAL
fits are lower on the merged base (cluster-apple2's k-means work).

| cell (s) | IDENTICAL 96ea1b210 | IDENTICAL 6856b5f8f | FAST 96ea1b210 | FAST 6856b5f8f | digests (IDENTICAL; FAST) |
|---|---|---|---|---|---|
| IVF-Flat fit | 3.880/3.672/3.660 | 2.999/3.014/3.003 | 1.466/1.070/1.086 | 1.392/1.077/1.077 | d730b8082a1cdbfb; 9993cf743da23b77 |
| IVF-Flat search | 0.135/0.110/0.110 | 0.161/0.116/0.116 | 0.137/0.107/0.106 | 0.111/0.115/0.112 | |
| IVF-PQ fit | 9.900/9.898/9.896 | 8.445/8.454/8.481 | 2.623/2.660/2.657 | 2.704/2.725/2.707 | 6bb7a6c5fc753846 / 3b1e0c1ae73444eb; 3c7316861c03b6a0 / 4ace8e5f668db63d |
| IVF-PQ search | 0.090/0.084/0.091 | 0.084/0.085/0.090 | 0.118/0.084/0.091 | 0.084/0.084/0.090 | |
| IVF-SQ fit | 3.702/3.714/3.697 | 3.043/3.052/3.037 | 1.105/1.120/1.121 | 1.182/1.121/1.117 | c55eedfcb6459d7b / 1d9c53fd8c13f452; 18180f0ab783f874 / 4cad0bd4e7a37a4d |
| IVF-SQ search | 0.107/0.110/0.113 | 0.108/0.107/0.114 | 0.109/0.108/0.118 | 0.111/0.108/0.119 | |
| IVF-RaBitQ fit | 3.622/3.621/3.647 | 2.955/2.964/2.970 | 1.015/1.029/1.017 | 1.075/1.043/1.034 | a4f2343eb268b0a7 / 056a570709713477; 6b6e86a5d3108cdd / a69552b2898d1871 |
| IVF-RaBitQ search | 0.060/0.050/0.056 | 0.050/0.051/0.056 | 0.064/0.050/0.056 | 0.050/0.050/0.057 | |
| refine search (top-40) | 0.117/0.117/0.122 | 0.115/0.117/0.122 | 0.118/0.118/0.121 | 0.118/0.117/0.125 | 3c73bf8ae59e47e4; 818dcdc7e08c674d |
| CAGRA fit (50k) | 0.783/0.786/0.786 | 0.813/0.793/0.789 | 0.612/0.613/0.602 | 0.614/0.613/0.604 | 54d696296c9c7c8a / 45e435db03654b4e (both tiers) |
| CAGRA search | 0.022 | 0.021 to 0.022 | 0.021 | 0.021 | |
| t-SNE fit (10k, 300 it) | 1.156/1.160/1.141 | 1.206/1.169/1.138 | 0.877/0.879/0.879 | 0.920/0.877/0.876 | 2bb1d3d75ffa1885; ca01838ecfb9306d |

FAST stages at 6856b5f8f on the M4 (ms): IVF-PQ build coarse 1010,
codebooks 1423, encode 187, residuals 68; IVF-SQ coarse 998, encode 57,
residuals 28; IVF-RaBitQ coarse 1002, encode 24; CAGRA k-NN + prune 587,
reverse merge 15; t-SNE iterations 753, symmetrize 66, k-NN 46.

FAST quality at 6856b5f8f, equal to 96ea1b210 in every row (HIGGS, 200k
index, 500 queries, 256 lists, 16 probes, recall at 10 against exact
search, seeds 0 / 1 / 2): IVF-PQ 0.8604 / 0.8524 / 0.8506, IVF-SQ 0.9482 /
0.9470 / 0.9426, IVF-RaBitQ 0.3098 / 0.3114 / 0.2996, CAGRA (3000 rows)
1.0000.

BROKEN ON THE BASE, FIXED (032db3845): `bench/speed/ann_fast_quality.py`
died at its t-SNE row in BOTH arms (`IndexError: mojolearn: too many indices
(3) for shape (1500, 2)`: `TSNE.fit_transform` returns mojolearn's array
type, which refuses `y[:, None, :]`), so job 1 has no t-SNE and no taxi
quality rows. The bench now reads the embedding and the search ids as numpy
arrays.

### Job 2: m3ultra-b, steward 1790627848135, 7c401de42: default against `+MOJOLEARN_ANN3_HOST_PASSES` against `+MOJOLEARN_ANN3_HOST_PASSES+MOJOLEARN_ANN3_PREPARE`

Queued 20:37Z Sep 28. Three arms built from one commit, two reps and the
stage pass (the first stage split inside `ivf_flat_build`, with cluster/'s
k-means marks). Result: PENDING.

## SHARED CODE touched (for the consolidation's check)

- `ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo` upload/download helpers are imported by `ivf_flat_search.mojo`, `ivf/checks/ivf_check.mojo` and `neighbors/checks/fused_logical32_check.mojo`.
- `x_ann/abi.mojo` and `bindings/ivf_index_arrays.mojo` are compiled into the CPU host bindings too (`_mojolearn_x_ann_host`, `_mojolearn_ivf_host`, `_mojolearn_ivf_search_host`).
- `ivf/` now imports `x_ann/stage_timer.mojo` (the marks).

## Unproven

Both switches until their A/B is recorded here. Nothing unproven is on in a
default build.
