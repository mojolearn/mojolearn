# Embedding owed items closed: eighteen arms, clauses (a) to (f), PLAN_SORT through the door (2026-09-15)

Lane `lane/embedding-owed` at ba4a108bb (parent e2d770ba8). Three columns. In each, `tools/embedding_sabotage_arm.sh`
built the clean check and all eighteen `MOJOLEARN_EMB_SABOTAGE_*` arms. It then built `bindings/build_embedding.sh`
and ran `test_embedding_surface`, `test_expose_d_manifest` and the identity_break lanes `embedding` and
`embedding-sort`.

| column | where | clean (clauses a to f) | BIT | inert, asserted | raised or not shown |
|---|---|---|---|---|---|
| Apple M4 (`apple-m4-local/`) | this Mac, one core, shared machine, `MOJOLEARN_COMPILE_JOBS=1`, 311 s for clean plus 18 arms | PASS | 17 | NO_FLUSH_ACC (device flushes the raw add) | none |
| NVIDIA H100 80GB HBM3, sm_90a (`nvidia-h100-sm_90a/`) | RunPod pod sts3mm92muxxs2, leg `bench/results/e1g/2026-09-15_005746-nvidia-h100-embedding-owed` (untracked), sabotage 391 s | PASS | 18 | none | none |
| AMD Instinct MI325X, gfx942 (`amd-mi325x-gfx942/`) | DigitalOcean droplet 600517613, leg `bench/results/e1g/2026-09-15_005737-amd-mi325x-do-embedding-owed` (untracked), sabotage 240 s | PASS | 18 | none | none |

The clean card is md5 `c7f824c35336bef2a3d0f672a172ef29` on all three columns, the same as on 2026-08-28 and
2026-09-14. No normative kernel changed. Each directory holds `verdict.txt`, `clean.card.md5`,
`clean.verdict-lines.txt` (every clause line of the clean run) and one `<ARM>.verdict-lines.txt` per arm. The
H100 and MI325X directories also hold `leg-summary.txt`.

## The two arms built here

- **FOLD_VIA_GEMM_ONEHOT** routes the backward through `identical_gemm` over a one-hot `[T, V]` matrix. The
  check computes the host GEMM oracle's answer for every cell of every clause (a) case and requires the device
  to match it on all 17 cases. That held on all three columns. The arm moved `emb.dw` on f_hot (`T = 300`,
  P = 3, 2 of 8 cells, first `0xc76c4680 -> 0xc76c4681`) and on f_multiblock (`T = 600`, P = 5, 3 of 3 cells,
  first `0x473c7f78 -> 0x473c7f75`), with the same bits on every column. It also moved one cell of f_subacc
  at `T = 5`, `0x80000000 -> 0x00000000`. There the GEMM's leaf adds the zero products of non-contributing
  positions, and a `+0.0` product launders a `-0.0` chain end (contract 7.1's hole). That is a correction to
  the contract row, which said the arm was inert at every `T <= 128`; the row is fixed.
- **SORT_KEY_ID_ONLY_UNSTABLE** makes PLAN_SORT's bitonic compare pass read the id half of the key only, and
  clause (a) runs PLAN_SORT on that build. It moved `emb.perm` first on f_tree4 (2 of 4 entries) and was inert
  on f_nodup, on all three columns. It moved `emb.perm` on f_dupsame, f_subacc, f_hot, f_multiblock,
  f_wide_v300 and f_d1, and `emb.dw` on f_hot and f_multiblock. Before any build, a Python simulation of the
  network predicted how many `emb.perm` entries move on f_nodup (0), f_dupsame (4), f_tree4 (2), f_subacc (2)
  and f_hot (298), and the device matched every count.

## Clauses (b), (c) and (f), every column

`clean.verdict-lines.txt` on each column reads:
- `clause (b): PASS, launches 2..8 bit-identical to launch 1 on all 69 cells of all 9 stages`
- `clause (c) padding: PASS` and `clause (c) length: PASS`, each with its firing control (3 real positions
  move 2 of 12 cells), plus the KNOWN EXCEPTION reproduced
- `clause (f) host: PASS, 6 refusals`
- `clause (f) device: PASS, 5 device refusals`: NaN and infinity in W (index 4), in dY (index 6), and NaN in
  the carried dW (index 4). Each is refused by name before any launch, and the output is untouched.
- `embedding_check: GREEN, clauses (a), (b), (c), (f), (d), (e) PASS on this column`

**Seen to fail first.** `apple-m4-local/clause_f.before-fix.e2d770ba8.txt` is the same clause on the M4 at
e2d770ba8, before the refusing entry points existed. It raised `DEVIATION 1506 nonfinite refusal remains open`
after measuring 1 nonfinite cell leaking out of the device forward. The `_into` hot-path forward still leaks
that cell, and each column reports it (`raised=False and returned 1 nonfinite cells`), as contract 9.1 now
states.

## PLAN_SORT through `Embedding(plan="sort")`

`test_embedding_surface` is GREEN with 42 checks on the M4, the H100 and the MI325X, including the new PLAN
arm. Sort equals scan bit for bit on plain, padded, carried and one-id (`R = T = 211`) gradients, and bad
plan values are refused by name. The lanes `embedding` and `embedding-sort` read IDENTICAL x3 on all 18 train,
18 held-out infer and 18 batch cells (`bench/results/identity_break/2026-09-15_embedding-sort/diff_three_columns.txt`).
Every `embedding-sort` hash equals the `embedding` hash, and the `embedding` hashes equal the 2026-09-14
record.

**Seen to fail first.** On the M4, a binding built with `-D MOJOLEARN_EMB_SABOTAGE_SORT_KEY_ID_ONLY_UNSTABLE=1`
turned `test_embedding_surface` RED on five PLAN checks
(`apple-m4-local/test_embedding_surface.SORT_KEY_ID_ONLY_UNSTABLE-binding.log`). The `embedding-sort` lane read
REFUSED on 8 of 9 fixtures, naming the pair (`apple-m4-local/identity_break.SORT_KEY_ID_ONLY_UNSTABLE-binding.txt`).
The one fixture it did not refuse is `ties`, where the network leaves the ties in place.

## Not claimed

- No speed number, and no PLAN_SORT dispatch crossover. `plan` defaults to `"scan"`.
- The shipped shape `V = 128256, d = 4096, T = 4096` was not run here.
- The AMD column is an MI325X on DigitalOcean. The 2026-09-14 sabotage round ran on an MI300X.
- No independent reference: both sides of every comparison are ours (contract section 13).
