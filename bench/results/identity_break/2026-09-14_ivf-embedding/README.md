# identity_break lanes `ivf` and `embedding`, three columns (2026-09-14)

The two lanes added when `IVFIndex` and `Embedding` left `_NOT_YET` (lane/expose-ivf-embedding),
run through the Python doors at commit 6c91eb1c3 on nine fixtures with two repeats and the batch
part. Every column JSON records its commit.

| column | JSON | where |
|---|---|---|
| Apple M4 | `identity_break.apple-m4.json` | this Mac, one core, shared machine, bindings built with `bindings/build_ivf.sh` and `build_embedding.sh` |
| NVIDIA H100 80GB HBM3, sm_90a | `identity_break.nvidia-nvidia-h100-80gb-hbm3-sm_90a.json` | RunPod pod 4oxh9h8mkway0c, leg `bench/results/e1g/2026-09-14_234102-nvidia-h100-ivf-embedding` (untracked) |
| AMD MI300X, gfx942 | `identity_break.amd-mi300x-gfx942.json` | Hot Aisle 2x MI300X VM 2fa42f6a, GPU 0, leg `bench/results/e1g/2026-09-14_234941-amd-mi300x-hotaisle-ivf-embedding` (untracked) |

On each box, before the lanes: `test_embedding_surface` GREEN (29 checks), `test_ivf_surface`
GREEN (19 checks), `test_expose_d_manifest` GREEN. Each column alone read `cells=18 stable=18`,
`infer: stable=18`, `batch: stable=18`, `model: n/a=18` (no index or table is saved).

## The diff

    python3 tools/identity_break.py --diff identity_break.apple-m4.json \
        identity_break.nvidia-nvidia-h100-80gb-hbm3-sm_90a.json identity_break.amd-mi300x-gfx942.json --require-columns 3

`diff_three_columns.txt`, the verdict lines:

    summary: IDENTICAL=18
    summary (infer/model): IDENTICAL=18, N/A=18
    summary (batch): IDENTICAL=18
    require-columns 3 over ['embedding', 'ivf']: OK

Every train, infer and batch cell of both lanes on all nine fixtures reads IDENTICAL x3 (54 rows);
the 18 model cells are n/a:no-save on every column. No column stands alone anywhere.

## Seen to fail first (Apple M4, one core)

- `embedding` against a binding built with `-D MOJOLEARN_EMB_SABOTAGE_PAD_ROW_NEG_ZERO=1`: DIVERGENT
  on 9 of 9 train cells, `parts differ: dw; agree: fwd,dw_nopad`.
- `embedding` against `-D MOJOLEARN_EMB_SABOTAGE_FOLD_DESCENDING=1`: REFUSED on both fixtures run,
  `carried dW (t0=171) and unsplit dW differ: 425 bytes of 8192` (base) and 216 of 8192 (hashed).
- `ivf` against `-D MOJOLEARN_IVF_BINDING_SABOTAGE=1` (never shipped; swaps query 0's first two ids):
  DIVERGENT on 9 of 9 train cells, `parts differ: idx; agree: dist,cand`, and 9 of 9 infer cells.
- Inert, recorded: `ivf` against `-D MOJOLEARN_EXPERIMENTAL_KNN_EXACT_CHAIN=1 -D
  MOJOLEARN_KNN_EXACT_CHAIN_SABOTAGE=1` and against `-D MOJOLEARN_537_GEMM_IDENT_SWAP=1` read
  IDENTICAL to the clean column on every fixture run; neither define reaches a bit this lane hashes.

## What the lanes are not

- `ivf` covers `metric='sqeuclidean'` only. `metric='euclidean'` (L2SqrtExpanded) is refused at the
  door: on the M4 its search returned all-zero distances and ids that are not the nearest rows, and
  no check searches it (`ivf/NOT_IMPLEMENTED.tsv`).
- `embedding` runs PLAN_SCAN, the entry's default; PLAN_SORT is clause (d) of `pixi run
  check-embedding` and is not reached through the door.
- These three columns are not part of the CPU identity gate's recorded columns; neither lane has a
  CPU path.
