# Embedding sabotage arms after the five findings were resolved (2026-09-14)

`tools/embedding_sabotage_arm.sh` at lane/expose-ivf-embedding 6c91eb1c3 on three columns. The
first round (`bench/results/ivf_embed_km_legs_2026-09-14/README.md`, at b163b76ba and a64c55281)
left five arms raising or not runnable; 6c91eb1c3's parent baa0b20e5 resolved each in
`embedding/checks/embedding_check.mojo` (and GATHER_CLAMP_OOR's arm in
`embedding/checks/embedding_identical.mojo`). No kernel changed: the clean card is md5
`c7f824c35336bef2a3d0f672a172ef29` on every column, the md5 of the first round.

| column | where | clean | BIT | inert, asserted | raised or not shown |
|---|---|---|---|---|---|
| Apple M4 (`apple-m4-local/`) | this Mac, one core, shared machine, `MOJOLEARN_COMPILE_JOBS=1`, 122 s | PASS | 15 | NO_FLUSH_ACC | none |
| NVIDIA H100 80GB HBM3, sm_90a (`nvidia-h100-sm_90a/`) | RunPod pod 4oxh9h8mkway0c, leg `bench/results/e1g/2026-09-14_234102-nvidia-h100-ivf-embedding` (untracked), 196 s | PASS | 16 | none | none |
| AMD MI300X, gfx942 (`amd-mi300x-gfx942/`) | Hot Aisle 2x MI300X VM 2fa42f6a (GPU 0; GPU 1 ran an idle body), leg `bench/results/e1g/2026-09-14_234941-amd-mi300x-hotaisle-ivf-embedding` (untracked), 227 s | PASS | 16 | none | none |

Each directory holds `verdict.txt`, the clean card's md5, and per arm the `clause (g)`, `device`
and `clause (e)` lines of its log (`<ARM>.verdict-lines.txt`).

## The five findings and what each read on each column

| arm | finding (first round) | resolution | M4 | H100 | MI300X |
|---|---|---|---|---|---|
| EMPTY_ROW_NEG_ZERO | moved `emb.dw_seed` on f_nodup, its predicted-inert case | exact mask: `emb.dw_seed` must move there, the other eight stages (`emb.dw` included) must not | BIT on f_empty; f_nodup moved exactly `emb.dw_seed` (24 cells) | same | same |
| SORT_TIE_REVERSED | moved `emb.perm` on f_dupsame | exact mask: `emb.perm` must move, the other eight must not | BIT on f_order3; f_dupsame moved exactly `emb.perm` (8 cells) | same | same |
| PAD_ROW_NEG_ZERO | first stage `emb.dw_seed`, table said `emb.dw` | first stage is `emb.dw_seed` (the T = 0 backward includes the padding store) | BIT on f_pad, first at `emb.dw_seed`; inert on f_nodup | same | same |
| ACCUM_BY_ADD | clause (e)'s by-add control dead on f_split at t0 = 3 | provably dead there (one contributor after the boundary per straddling row), now its asserted inert case; witness f_tree4 at t0 = 2 | moves 1 cell `0x3f800000 -> 0x3f800001`, 0 of 8 on f_split | same | same |
| GATHER_CLAMP_OOR | no runnable witness | the arm also drops the device id refusal; `device_oor_refusal` is the witness | accepted f_oor_high and f_oor_neg, 12 cells equal the clamped gather; inert on f_nodup | same | same |
| NO_FLUSH_ACC | raised on Apple, where contract 9.3 predicts it inert | a device raw-add probe decides; on a flushing device the arm is asserted inert on every case | probe `-> 0x80000000` (flushed); inert on all 17 cases | probe `-> 0x80400000` (kept); BIT on f_subacc | probe `-> 0x80400000` (kept); BIT on f_subacc |

The clean build's new device refusal step reads, on every column, `f_oor_high REFUSED by name on
the device entry point, output buffer untouched (6 poisoned cells)`.

## Seen to fail

Every BIT line above is the gate failing under a sabotage build, and the device refusal step is
seen both ways (REFUSED on the clean build, ACCEPTED under GATHER_CLAMP_OOR). Not seen to fail:
the exact inert masks' "the other eight stages must not move" half, since no build here moves
`emb.dw` on f_nodup or f_dupsame; the first round's looser masks raising on three columns is the
only failure of that code path on record.

## Not claimed

A flake on the shared M4, recorded: for about ten minutes on 2026-09-14 the clean check, including
the first round's binary that had passed earlier, failed at "the dW POISON did not arrive (0 of 24
cells)" on its first case with 67 MB of memory free on the machine, and passed again unchanged
once memory was released. It is the host, not the code; it is written down because it looked like
a code failure.
