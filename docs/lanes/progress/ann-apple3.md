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
| fd948718d, gated c9b127107, on (this commit) | IVF build host passes (below) | both | ON; `-D MOJOLEARN_ANN3_HOST_PASSES_OFF` reverts | MEASURED job 2: equal digests in both tiers, gain |
| 171bbcdba, d20eb2129, gated c9b127107, on (this commit) | index preparation (below) | both | ON; `-D MOJOLEARN_ANN3_PREPARE_OFF` reverts | MEASURED job 2: equal digests in both tiers, gain |
| (this commit) | FAST seeding of the PQ codebooks and of the coarse quantizer (`x_ann/kpp_seed.mojo`) | FAST, Apple | OFF; `-D MOJOLEARN_ANN3_PQ_SEED`, `-D MOJOLEARN_ANN3_COARSE_SEED` | UNBUILT, UNMEASURED; moves FAST bits, owes the paired recall check |

| (this commit) | FAST IVF-PQ: the codebook sample's residuals formed on the host, the 1M x 28 residual matrix not downloaded | FAST | OFF; `-D MOJOLEARN_ANN3_PQ_HOST_RESIDUALS` | UNBUILT, UNMEASURED; expected to move no bit (FAST's residual is one subtraction) |

| (this commit) | FAST t-SNE repulsion at 32 or 64 rows per threadgroup (128 by default) | FAST, Apple | OFF; `-D MOJOLEARN_ANN3_TSNE_RB32` or `-D MOJOLEARN_ANN3_TSNE_RB64` | UNBUILT, UNMEASURED; moves no bit by construction |

| (this commit) | the coarse quantizer's FAST training sample gathered by memcpy | FAST (the sample is FAST's) | OFF; `-D MOJOLEARN_ANN3_TRAINSET_COPY` | UNBUILT, UNMEASURED; plain copies |
| 7a9bb0aeb | IVF-PQ / IVF-SQ builds download their codes straight into the caller's arrays | both | OFF; `-D MOJOLEARN_ANN3_DIRECT_OUT` | UNBUILT, UNMEASURED; plain copies |

t-SNE, why the threadgroup size: FAST iterations are 753 ms on the M4 (job
1) and 470 to 620 ms on the M3 Ultra (round 2), a ratio of 1.2 to 1.6,
where CAGRA's k-NN (391 threadgroups) is 587 against 136 ms, 4.3. At 10,000
rows the repulsion launches 79 threadgroups of 128 rows. Round 2 measured 64
rows per threadgroup on the M4 only (1040 against 1052 ms, flat).

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

### Job 2: m3ultra-b (Apple M3 Ultra), steward 1790627848135, 7c401de42: default against `+MOJOLEARN_ANN3_HOST_PASSES` against `+MOJOLEARN_ANN3_HOST_PASSES+MOJOLEARN_ANN3_PREPARE`

PASS. Three arms built from one commit, arms alternated, three runs per
cell (two reps and the stage pass). EVERY digest is equal in every arm, in
both tiers, and every second search returns the first search's digest. Raw:
`~/mojolearn-evidence/ann-apple3/job2_m3ultra-b_1790627848135.txt`
(tabulated: `job2_tab.txt`, by `tools/ann_apple3_tab.py --stages`).

FAST (s; the first IVF-Flat fit of a process carries a warmup):

| cell | default | + host passes | + host passes + prepare |
|---|---|---|---|
| IVF-Flat fit | 0.980/0.586/0.579 | 0.842/0.489/0.506 | 0.827/0.489/0.484 |
| IVF-Flat first search | 0.067/0.068/0.070 | 0.047/0.047/0.047 | 0.043/0.042/0.042 |
| IVF-Flat second search | 0.0113/0.0114/0.0111 | 0.0107/0.0108/0.0106 | 0.0106/0.0105/0.0109 |
| IVF-PQ fit | 1.758/1.779/1.788 | 1.749/1.729/1.696 | 1.691/1.729/1.752 |
| IVF-PQ first / second search | 0.027 / 0.0200, 0.0196 | 0.027 / 0.0196, 0.0198 | 0.027 / 0.0171, 0.0170 |
| IVF-SQ fit | 0.607/0.612/0.614 | 0.531/0.527/0.515 | 0.516/0.525/0.533 |
| IVF-SQ first / second search | 0.033 / 0.0199, 0.0199 | 0.032 / 0.0200, 0.0200 | 0.033 / 0.0154, 0.0155 |
| IVF-RaBitQ fit | 0.515/0.530/0.529 | 0.459/0.463/0.457 | 0.454/0.466/0.463 |
| IVF-RaBitQ first / second search | 0.016 / 0.0129, 0.0131 | 0.016 / 0.0131, 0.0135 | 0.016 / 0.0124, 0.0126 |
| refine (top-40) first / second search | 0.039 / 0.0315, 0.0315 | 0.039 / 0.0316, 0.0321 | 0.039 / 0.0292, 0.0297 |
| CAGRA fit; first / second search | 0.148; 0.022 / 0.0196 | 0.148; 0.023 / 0.0196 | 0.146; 0.022 / 0.0196 |
| t-SNE fit | 0.541/0.541/0.540 | 0.569/0.538/0.542 | 0.540/0.540/0.543 |

IDENTICAL (s): IVF-Flat fit 1.013 -> 0.903, first search 0.067 -> 0.041;
IVF-PQ fit 3.41 -> 3.20; IVF-SQ fit 1.05 -> 0.95, second search 0.0200 ->
0.0155; IVF-RaBitQ fit 0.97 -> 0.88; refine second search 0.0317 -> 0.0293;
CAGRA and t-SNE flat (not touched).

KEPT, both default on now (`-D MOJOLEARN_ANN3_HOST_PASSES_OFF`,
`-D MOJOLEARN_ANN3_PREPARE_OFF` revert).

Stage split of ONE IVF build, FAST, M3 Ultra (ms; the stage pass drains at
every mark, so the sums run above the timed fits):

| phase of `ivf_flat_build` | default | + host passes |
|---|---|---|
| data check | 18 | 7 |
| training sample + quantizer scale | 25 | 25 to 30 |
| upload | 17 | 11 |
| row norms | 5 | 7 |
| cluster/ k-means (coarse, 262,144 rows, k 1024, 10 iterations) | 368 | 358 |
| cluster/ predict (1M rows) | 16 | 16 |
| downloads | 6 | 2 |
| list layout | 47 | 5 |
| index (copies) | 4 | 0 |
| binding copy out | 23 to 26 | 2 to 3 |

cluster/'s own marks over the FAST stage pass (5 coarse fits and 28
codebook fits): seeding 3596 ms (of it the sequential k-means++ over the
candidates 2705 ms, the k-means|| rounds 503 ms, the candidates' Lloyd 329
ms), the Lloyd iterations of the fits 358 ms. IDENTICAL: seeding 5872 ms
(k-means++ 2757, rounds 2398), Lloyd 1800 ms. So about 90% of the FAST
k-means time of an IVF build is SEEDING, and the k-means is 80 to 95% of
every IVF fit. One PQ codebook fit is 80 ms (14 per IVF-PQ build: 1126 ms of
the 1.73 s fit).

IVF scan, per search of 1000 queries (ms): the list-order gather 4.6 -> 0
with the prepared index; score 8.1, select 8.6, probe 3.1, coarse 1.2.
CAGRA search 19 (one thread per query, 16 threadgroups of 64). t-SNE
iterations 472.

### Job 3: m3ultra-b, steward 1790629592016, c831de064, FAST only, with the quality pass on every arm

Queued 21:06Z Sep 28. Arms, all built from c831de064:
- `after`: the default build (host passes and prepare on);
- B = `+MOJOLEARN_ANN3_PQ_HOST_RESIDUALS+MOJOLEARN_ANN3_DIRECT_OUT+MOJOLEARN_ANN3_TRAINSET_COPY` (expected to move no bit);
- B `+MOJOLEARN_ANN3_PQ_SEED+MOJOLEARN_ANN3_TSNE_RB64`;
- B `+MOJOLEARN_ANN3_PQ_SEED+MOJOLEARN_ANN3_COARSE_SEED+MOJOLEARN_ANN3_TSNE_RB32`.

Quality: HIGGS and taxi, seeds 0 to 4, IVF-Flat / PQ / SQ / RaBitQ / CAGRA
recall at 10 and t-SNE trustworthiness and KL. Result: PENDING.

### NVIDIA: nvc1-0020 (2x A40 shared pod), `tools/ann_apple3_cuda.sh`, the lane's one job

Queued 21:09Z Sep 28 behind seven jobs of other lanes. IDENTICAL only: the
default build against the build with every default-on switch turned off,
HIGGS 1M, and both against the Apple IDENTICAL digests of job 2. It runs
the tree as synced when it starts; the tree is synced again when a switch
becomes default on (while the job is still queued). Result: PENDING.

## SHARED CODE touched (for the consolidation's check)

- `ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo` upload/download helpers are imported by `ivf_flat_search.mojo`, `ivf/checks/ivf_check.mojo` and `neighbors/checks/fused_logical32_check.mojo`.
- `x_ann/abi.mojo` and `bindings/ivf_index_arrays.mojo` are compiled into the CPU host bindings too (`_mojolearn_x_ann_host`, `_mojolearn_ivf_host`, `_mojolearn_ivf_search_host`).
- `ivf/` now imports `x_ann/stage_timer.mojo` (the marks).

## Unproven

- `MOJOLEARN_ANN3_HOST_PASSES` and `MOJOLEARN_ANN3_PREPARE` (default on):
  Metal digests only (m3ultra-b, both tiers). No NVIDIA, AMD or CPU run yet;
  the host code is compiled on every vendor (the Apple upload is comptime
  Apple only). The NaN and non-finite refusal of `ivf_validate_data`'s first
  pass was not exercised on a device.
- Every other switch is OFF in a default build and unmeasured.
