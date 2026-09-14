# IVF-FLAT, Embedding and kernel methods checks on NVIDIA and AMD (2026-09-14)

The owed vendor legs for three lanes that are finished underneath and have no
Python door: `ivf/checks/ivf_check.mojo` (IVFIndex), `embedding/checks/embedding_check.mojo`
(Embedding) and `kernel_methods/checks/km_check.mojo`, the latter's first vendor runs after its
FAST compile fix (834ba18dc) and its NO_EIGEN_CLIP fix (572ccd949). Before 834ba18dc the FAST
build of km_check hit a 25 to 30 minute compile bound on the MI300X and the H100.

Every box built each check from a `git archive` of the commit named, `mojo build -j 8`
(`--target-accelerator` sm_90a or gfx942), ran it under a pty, and brought the per-check cards
home. The Apple M4 column is this Mac, one core, shared machine, `mojo build -j 1`.

## Legs

| leg (raw archive, untracked, under `bench/results/e1g/`) | box | commit | what |
|---|---|---|---|
| `2026-09-14_215153-nvidia-h100-ivf-embed-km` | RunPod NVIDIA H100 80GB HBM3, sm_90a | 171752af4 (main) | km FAST, IVF IDENTICAL, Embedding IDENTICAL, km IDENTICAL, IVF FAST |
| `2026-09-14_215139-amd-mi300x-hotaisle-ivf-embed-km` | Hot Aisle MI300X, gfx942, 13core | 171752af4 (main) | the same body |
| `2026-09-14_221004-nvidia-h100-embed-sab-kmfast` | RunPod H100 | b163b76ba (this branch) | km FAST after the POLY_VIA_POW change, the sixteen Embedding sabotage builds |
| `2026-09-14_221000-amd-mi300x-hotaisle-embed-sab-kmfast` | Hot Aisle MI300X | b163b76ba | the same body |
| `2026-09-14_223034-nvidia-h100-embed-accum-by-add` | RunPod H100 | a64c55281 (this branch) | ACCUM_BY_ADD under clause (e), fixed sabotage script |
| `2026-09-14_223030-amd-mi300x-hotaisle-embed-accum-by-add` | Hot Aisle MI300X | a64c55281 | the same body |

`summaries/` holds each leg's `summary.txt`; `run-lines/` the verdict lines of each run log;
`cards/` the cards; `embedding-sabotage/` the verdict lines of every sabotage run.

## Compile and run seconds

| check | H100 compile / run | MI300X compile / run | M4 (one core) compile / run |
|---|---|---|---|
| km_check FAST (171752af4) | 39 / 2, exit 1 | 38 / 2, exit 1 | n/a |
| km_check FAST (b163b76ba) | 39 / 2, exit 0 | 36 / 3, exit 0 | 28 / 2, exit 0 |
| km_check IDENTICAL | 37 / 2 | 46 / 1 | 23 / 1 |
| ivf_check IDENTICAL | 43 / 2 | 43 / 1 | 26 / 0 |
| ivf_check FAST | 45 / 2 | 46 / 1 | not run |
| embedding_check IDENTICAL | 12 / 2 | 12 / 1 | 7 / 0 |
| the Embedding sabotage script, clean plus sixteen arms | 215 total | 207 total | 108 total |

The 25 to 30 minute km_check FAST compile is gone: 39 seconds on the H100 and 38 on the
MI300X. No restructure is proposed.

## Verdict lines

km_check IDENTICAL, all three columns: `kernel_methods: 18 checks OK [IDENTICAL]`, with
`check_km_sabotages OK [IDENTICAL]: 13 arms ... 11 classified MUST FAIL and shown to move` and
`MUST FAIL  NO_EIGEN_CLIP (linear) on ZERO_ROW: bits moved`.

km_check FAST at 171752af4, H100 and MI300X: `check_km_sabotages FAILED: POLY_VIA_POW moved NO
bit on ANY of the 5 fixtures, at the polynomial kernel`. Under FAST `identical_pow` is `x**p`, the
vendor pow, and on these two columns it returns the repeated product at the integral degree; on
the M4 the same FAST build moves it (`MUST FAIL  POLY_VIA_POW (polynomial) on RBF`). b163b76ba
records it rather than asserting it under FAST on a column where it does not move (IDENTICAL
still asserts it); at b163b76ba both vendors read `RECORDED  POLY_VIA_POW (polynomial): 0 cells
moved on all 5 fixtures`, `check_km_sabotages OK [FAST]: 13 arms ... 9 classified MUST FAIL and
shown to move` and `kernel_methods: 18 checks OK [FAST]`.

ivf_check, all three columns at IDENTICAL and both vendors at FAST: `ivf_check mode=IDENTICAL ALL
OK`, `ivf_check mode=FAST ALL OK`.

embedding_check IDENTICAL, all three columns: `clause (a): PASS, 17 cases, 9/9 stages
bit-identical to the oracle on all 6887 cells, 9/9 card tags`.

## Cards: byte equality across the three columns

| card | Apple M4 | NVIDIA H100 | AMD MI300X |
|---|---|---|---|
| ivf_check IDENTICAL (`ivf_check.card.a` = `.b` on each) | 1e7c1702fe72 | 1e7c1702fe72 | 1e7c1702fe72 |
| km_check IDENTICAL (`km.card.a` = `.b` on each) | 1410222a172e | 1410222a172e | 1410222a172e |
| embedding_check IDENTICAL | c7f824c35336 | c7f824c35336 | c7f824c35336 |

md5 prefixes; the Embedding md5 is the one Apple and AMD recorded on 2026-08-28. The FAST cards
differ by column, as FAST makes no bit claim (IVF FAST a8310dbe on the H100, ff70284a on the
MI300X; km FAST d3e5ff8c, b8010120, cc30ce5e on the H100, MI300X and M4). The M4 IVF and km cards
were produced from the b163b76ba source (the FAST-only POLY_VIA_POW change does not reach an
IDENTICAL build).

## Embedding sabotage

Under a sabotage build the check inverts its own verdict and exits 0 when clause (g) shows the arm
BIT on its witness, first at its own stage, and inert on its predicted control. The script as
prepared required a non-zero exit, so its H100 and MI300X runs at b163b76ba printed "PASSED WITH
THE ARM COMPILED IN" for twelve arms; the logs say BIT for eleven of them. a64c55281 fixes the
script. Per arm, identical on the H100 and the MI300X, and on the M4 except NO_FLUSH_ACC:

| arm | H100 and MI300X | M4 |
|---|---|---|
| FOLD_DESCENDING, FOLD_BALANCED_TREE, SEED_SEEDLESS, SINGLE_RUN_BYPASS, EMPTY_ROW_SKIPPED, FOLD_READS_LAUNCH, RANK_BY_ARRIVAL, PAD_ROW_CONTRIBUTES, GATHER_NO_FLUSH, ACCUM_REFILLS | BIT | BIT |
| NO_FLUSH_ACC | BIT on f_subacc | RAISED, inert on its witness (contract 9.3 predicts inert on Apple; the check raises anyway) |
| EMPTY_ROW_NEG_ZERO | RAISED: bit on f_empty, but moved emb.dw_seed on f_nodup, its predicted-inert control | same |
| SORT_TIE_REVERSED | RAISED: bit on f_order3, but moved emb.perm on f_dupsame, its predicted-inert control | same |
| PAD_ROW_NEG_ZERO | RAISED: moved emb.dw_seed first on f_pad, the contract says its clause writes emb.dw | same |
| ACCUM_BY_ADD | NOT SHOWN: clause (e) of the clean build raises "CLAUSE (e)'s BY-ADD CONTROL IS DEAD" on f_split (a64c55281 legs) | same |
| GATHER_CLAMP_OOR | not runnable: the check refuses it, no runnable witness (contract section 8) | same |

So the gate is shown capable of failing on ten arms on every column and eleven on NVIDIA and AMD.
Five arms are open findings about the check's predictions and witnesses, not about the device:
two predicted-inert masks are wrong, one arm's first stage is not the one the contract names, the
by-add control's fixture does not separate, and the Apple NO_FLUSH_ACC expectation is not encoded.

## What this supports

IVFIndex: the check passes at IDENTICAL and FAST on NVIDIA and AMD and its IDENTICAL card is
byte-identical across Apple, NVIDIA and AMD at one commit. That is the vendor evidence the
`_NOT_YET` entry named as missing. Exposure still needs what the census names for every door: a
binding build, the Python class, an identity_break lane and its three columns.

Embedding: the clause (a) card is byte-identical on three columns and the sabotage arms have run;
the five open arms above should be resolved (or reclassified with reasons) before the gate is
called complete, and exposure needs the same door work as IVF. Neither is exposed in this round.

km_check: first vendor runs after 834ba18dc and 572ccd949 pass at IDENTICAL on both vendors with
the three-column card equal; FAST passes on both at b163b76ba.
