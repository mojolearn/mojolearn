# IVFIndex `metric='euclidean'` (L2SqrtExpanded): the fix, the check and the lane, three columns (2026-09-14)

Branch `fix/ivf-l2sqrt`. The fix is commit fe0f10043; the columns below ran at 76a170dcf, which is
that commit with `lane/kmeans-sqrt-inertia` (DEVIATION 2715, the pinned root in the k-means
min-reduce kernels) merged in. Every column JSON records its commit.

## What was wrong

Through the Python door on the Apple M4, L2SqrtExpanded returned 0.0 for every distance and ids that
were not the nearest rows. `ivf_flat_build` and `ivf_flat_search` filled `row_norm_kernel`'s sqrt
flag from `metric_is_sqrt(metric)`, which is the k-means reduction flag and is true for
L2SqrtExpanded. Every data, centre, query and list norm was rooted, so the expanded distance became
`||q|| + ||y|| - 2 q.y`, negative for most pairs and clamped to zero. cuVS takes squared norms under
both L2 metrics (`ivf_flat_build.cuh:351-357`, `ivf_flat_search.cuh:110-118`).

The fix: `compute_row_norms` has no flag and is always squared; the coarse step and the candidates
are scored and selected on squared distances under both metrics; the root (`identical_sqrt`,
flushed) is applied to the k selected distances by `ivf_common.mojo::postprocess_distances`, where
cuVS takes it (`interleaved_scan_impl.cuh:204`, `ivf_common.cuh:206-221`).

## ivf_check (`ivf/checks/ivf_check.mojo`, new `check_l2_sqrt_is_the_root_of_l2`)

| column | IDENTICAL | `ivf_check.card.a` = `.b` | `ivf_check.sqrt.card.a` = `.b` | where |
|---|---|---|---|---|
| Apple M4 | ALL OK | 1e7c1702fe72 | 1250edd6e8da | this Mac, one core, shared machine (`legs/apple-m4.ivf_check.log`) |
| NVIDIA H100 80GB HBM3, sm_90a | ALL OK | 1e7c1702fe72 | 1250edd6e8da | RunPod pod y970c3l43loo5w, leg `bench/results/e1g/2026-09-15_010314-nvidia-h100-ivf-euclidean-b` (untracked) |
| AMD MI300X, gfx942 | ALL OK | 1e7c1702fe72 | 1250edd6e8da | Hot Aisle 2x MI300X VM, GPU 0 (GPU 1 idle), leg `bench/results/e1g/2026-09-15_010313-amd-mi300x-hotaisle-ivf-euclidean-b` (untracked) |

The L2Expanded card is unchanged from the exposure record (1e7c1702, `bench/results/ivf_embed_km_legs_2026-09-14/README.md`).
The new check writes its own card file, so it does not extend that one; the new L2SqrtExpanded card
is 1250edd6 on all three. The FAST build also reads ALL OK on both boxes (its L2SqrtExpanded lines
are REPORTS by design). `legs/*.summary.txt` are the per-box summaries.

Seen to fail first (Apple M4, one core): the check against the unfixed `ivf/impl` raised
`check_l2_sqrt_is_the_root_of_l2 FAILED (120 of 120 distances at n_probes=3 are exactly 0.0)`,
with (a) 6 of 6 centre norms, and (b), (c) and (d) 119 of 120 slots, first `slot 1 (0x00000000, 1)`
against `(0x3f80ae55, 124)`.

### The H100 card before the k-means root was pinned

At fe0f10043 (without DEVIATION 2715) the H100 read ALL OK and 1e7c1702, but its sqrt card was
bbcb91cd while the M4 and the MI300X read 1250edd6. The H100 stood alone. First divergence
`ivf.quantizer.restart00.iter01.min_dist`, the k-means reduced distance rooted by the stdlib `sqrt`
(approximate on NVIDIA, DEVIATION 258); `sqrt_card_h100_before_vs_after_kmeans_sqrt_fix.txt` is
the trace diff of the two H100 cards. That is the same defect as the 166-lane record's
`kmeans-sqrt/wide` inertia cell. Legs: `bench/results/e1g/2026-09-15_005009-nvidia-h100-ivf-euclidean`
and `bench/results/e1g/2026-09-15_004959-amd-mi300x-hotaisle-ivf-euclidean` (untracked, summaries
in `legs/*.before-kmeans-sqrt-fix.summary.txt`). The `ivf-euclidean` lane was IDENTICAL x3 even
then (the H100 JSON before and after reads `summary: IDENTICAL=18`): its hashes did not reach
that ulp. This fix therefore depends on `lane/kmeans-sqrt-inertia`.

## The lane `ivf-euclidean`

    python3 tools/identity_break.py --diff identity_break.apple-m4.json \
        identity_break.nvidia-nvidia-h100-80gb-hbm3-sm_90a.json identity_break.amd-mi300x-gfx942.json --require-columns 3

`diff_three_columns.txt`, the verdict lines:

    summary: IDENTICAL=18
    summary (infer/model): IDENTICAL=18, N/A=18
    summary (batch): IDENTICAL=18
    require-columns 3 over ['ivf', 'ivf-euclidean']: OK

Every train, infer and batch cell of `ivf` and `ivf-euclidean` on all nine fixtures is IDENTICAL x3;
the model cells are n/a:no-save. Each column alone read `cells=18 stable=18`, `infer: stable=18`,
`batch: stable=18`. `test_ivf_surface` was GREEN (25 checks) on all three.

Seen to fail: `diff_m4_fixed_vs_unfixed_impl.txt` diffs the M4 column against an M4 run of the
same lanes through a binding built from the unfixed `ivf/impl` (door already open):
`summary: DIVERGENT=9, IDENTICAL=9`, every `ivf-euclidean` fixture `parts differ: dist,idx,cand`,
and `ivf` IDENTICAL, as it must be. `test_ivf_surface`'s EUCLIDEAN arm read RED with 4 failures
against that binding.

## What this does not cover

- These columns are not part of the CPU identity gate's recorded columns; IVF has no CPU path.
- The lane probes 4 of 16 lists. A build under each metric is not one index (the quantizer's
  reduction differs), so `ivf` and `ivf-euclidean` are separate lanes, not one lane rooted.
