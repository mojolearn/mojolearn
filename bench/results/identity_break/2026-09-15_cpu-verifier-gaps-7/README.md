# The seven one-device lanes the CPU identity gate did not run (2026-09-15)

lane/cpu-verifier-gaps-7. The lanes are gmm-sample, gmm-random-init-sample,
gp-sample-y, gp-sample-y-normalize, tokenizer, gbdt-categorical-ctr-tables and
gbdt-tensor-ctr-tables. Each had a CPU host function and no manifest entry, so
the full CPU verification (`FULL_CPU_VERIFY`) did not run them. All seven are
covered now; none was refused.

## Metal-saved CTR models (`metal-ctr/`)

The CTR table lanes refuse CPU training by name, so a CPU column LOADS the
Metal-saved model of each fixture. The committed directory
(`../2026-09-15_gbdt-ctr-tables/models/`) held base, ties and odd only, and the
gate runs all nine fixtures. The six others were fitted on the Apple M4 with
the CTR lane's own HEAD Metal build (1386833b4, `_mojolearn.so` and
`_mojolearn_gbdt.so`), one exclusive Metal slot job per fixture, about 25 s per
categorical fixture.

The build was validated on base first (`validate-*.json`): the saved npz bytes
and the train, infer, model and batch hashes equal the committed ones for both
lanes, so the six new models come from the same arithmetic as the three
committed ones. Every new Metal cell is STABLE. The categorical denormal and
denormal_ftz fixtures carry base's train hash, and the tensor lane's two
denormal fixtures share one hash: the denormal values do not move these
models' predictions.

## The gate's covered-lanes path on one x86 CPU (`x86-runpod/`)

RunPod CPU pod 03krlhu4sqjw8e (AMD EPYC 9654, Linux x86-64,
`tools/runpod_cpu_leg.sh`), commit 11c5f2192, 4 shards, DELETED and verified
gone (HTTP 404 and absent from the listing), 229 s billed, $0.0153. The
commands are `cmd.sh`, the gate's own steps.

Production (`cpu.json`, two repeats, all nine fixtures):

- `cpu_column_check.log`: 63 cells, every covered lane STABLE, vendor derived,
  commit matched, all six host bindings read back as the CPU column. 0 failures.
- `diff_four_columns.txt`: `--require-columns 4 --owed-json` against the three
  166-lane GPU columns reads OK with OWED=63 train, OWED=99 infer and model,
  OWED=27 batch (189 parts). No committed GPU record carries these lanes yet
  (the record's tokenizer cells are an older lane revision, so they read as
  absent), so every part is OWED, and `owed_cells.json` lists exactly what the
  next release record owes. Nothing is DIVERGENT.

Sabotage (`cpu-sab.json`, the host set built with the manifest's per-family
defines, one repeat):

- `cpu_sabotage_check.log`: every covered lane still STABLE, a wrong answer
  rather than a refusal. 0 failures.
- `owed_sabotage_check.log`: **189 of 189 owed cell parts moved, 0 failures**.
- `moved_every_part.txt`: every hashed part of every cell moved (moved=189,
  not_moved=0; the 63 n/a parts are the CTR lanes' model parts, the sample
  lanes' batch parts and the no-save model parts).
- `diff_four_columns_sab.txt` exits 1, as it must. It reads ONE-COLUMN rather
  than DIVERGENT for the same reason the production diff reads OWED: no GPU
  record hashes these lanes, so the sabotage arm is their only negative
  control, which is what the owed check above is.

Against the Metal columns (`diff_metal.txt`, `--allow-separate-builds`): 44
train cells IDENTICAL, 44 infer and model, 18 batch. That is every cell a
Metal column carries: the gmm-sample lanes' four fixtures, the gp-sample-y
lanes' nine, and the CTR lanes' nine. ONE-COLUMN elsewhere is an absent Metal
cell (the tokenizer lane, the gmm-sample fixtures no Metal column ran) or a
CTR model part, which a CPU column reports n/a by design: the file is the GPU
column's bytes, not CPU arithmetic.

Spot check of existing lanes on the base fixture (`diff_spot.txt`): gmm,
gmm-random-init and gp read IDENTICAL x4 on train, infer and batch against the
record; gp-normalize-y is OWED (the record does not carry it) and the model
parts are OWED (the GPU columns read n/a:no-save). The diff exits 1 only
because the CPU column ran base alone while the record carries nine fixtures,
so the other eight are short by construction (24 REQUIRE FAIL lines per lane,
8 fixtures x 3 parts). No recorded cell moved.

## Owed

- The NVIDIA and AMD cells of all seven lanes, and the Apple cells of the
  tokenizer lane and of the five gmm-sample fixtures no Metal column ran, to
  the next release record (`x86-runpod/owed_cells.json`).
- No GPU box was rented. The Metal work was the shared M4.
