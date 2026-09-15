# No third-party vocabulary or data, and the tokenizer lane (2026-09-15)

Lane `lane/clean-third-party`. mojolearn ships no GPT-2 table: `GPT2Tokenizer` loads a
vocabulary the user supplies, the Unicode class table is generated at build time, and the
`tokenizer` identity lane uses the synthetic vocabulary mojolearn trains itself
(`python/mojolearn/_tokenizer_synthetic.py`) at `LANE_REVISIONS["tokenizer"] =
"synthetic-vocab-1"`. The tokenizer is host integer code, so one hash per cell holds on every
device class.

## `m4/`: Apple M4, one core, tokenizer binding built from this lane

- `surface.txt`: `test_tokenizer_surface` GREEN, 23 of 23, with GPT-2 files the user
  downloaded named by `MOJOLEARN_GPT2_ENCODER_JSON` and `MOJOLEARN_GPT2_VOCAB_BPE`.
- `surface-no-gpt2-files.txt`: GREEN, 22 of 23, the user-file test skipped.
- `surface-sabotage.txt`: the `-D MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1` build, RED, 10 of 23
  failed.
- `manifest.txt`: `test_tokenizer_manifest` GREEN, 8 checks.
- `check-tokenizer.txt`: `pixi run check-tokenizer` PASS, 22 of 22 exact id sequences and
  byte-exact round trips against the synthetic vocabulary's Python encoder.

## `m4-owed/`: the lane after its round trip became a hashed part

The lane first asserted `decode(encode(x)) == x`, so a sabotaged binary raised and read
REFUSED, which the owed check does not count as a catch. The round trip is now a hashed part.

- `diff.repeat.txt`: two production columns, IDENTICAL on all 9 fixtures (train, infer,
  batch).
- `diff.record.txt`: against the 166-lane record's Apple, NVIDIA and AMD columns with
  `--require-columns 4 --owed-json`. Those columns hashed the lane at an older revision, so
  their cells are not compared: OWED 27 (9 train, 9 infer, 9 batch), exit 0. `owed.json` lists
  them for the next release record.
- `diff.sabotage.txt`: production against the host sabotage column
  (`MOJOLEARN_HOST_ALLOW_SABOTAGE=1`): DIVERGENT on all 9 fixtures for train, infer and batch;
  no cell REFUSED.
- `owed_sabotage_check.txt`: `tools/cpu_identity_gate_check.py owed` on the production and
  sabotage columns: **owed verdict OK, 27 of 27 owed cell parts moved, 0 failures.**
- `owed_check_negative_control.txt`: the same check with the production repeat in the
  sabotage slot: owed verdict FAIL, 0 of 27 moved, 27 failures, so the check can fail.

## `runpod-x86/`: RunPod CPU pod, AMD EPYC 9754, at bc47909fd

Built through `tools/runpod_cpu_leg.sh --build tokenizer --sabotage-build tokenizer` with
`-D MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1`. `generated-head.txt` shows the Unicode table module
the build generated (unicodedata 16.0.0, the pinned sha256).

- `surface.txt`, `manifest.txt`: both GREEN (the user-file test skipped; no GPT-2 files on the
  pod).
- `diff.m4-x86.txt`: the x86 column against the M4 column, IDENTICAL on all 9 fixtures.
- `diff.sabotage.txt`: DIVERGENT on all 9 fixtures for train, infer and batch.
- `diff.record.txt`, `owed.json`, `owed_sabotage_check.txt`: run on the Mac over the pod's
  columns, because the pod ships no `bench/results`. OWED 27 against the 166-lane record;
  **owed verdict OK, 27 of 27 owed cell parts moved, 0 failures.**
- `wheel-listing.txt`, `wheel-third-party-scan.txt`: a test wheel built from the pod checkout
  (`python -m build`) with 101 members, no vocabulary, no `.tsv`, no site-packages and no
  license files of vendored packages.
- `installed-tokenizer.txt`: from the unpacked wheel, `GPT2Tokenizer()` refuses by name and
  the synthetic vocabulary loads (n_vocab 513).
- `installed-identity.txt`, `installed-judge.txt`: the wheel's own harness copy, 9 of 9
  stable; the shipped reference table judges all 27 tokenizer cell parts OWED.
- `verify-lanes-tokenizer.txt`: `python -m mojolearn verify --lanes tokenizer` from the
  installed wheel, exit 0: tokenizer OWED 27, DIVERGENT 0, REFUSED 0.
- `verify-quick.txt`: `python -m mojolearn verify --quick`, exit 0. The test wheel carries
  only the tokenizer host binding, so the other quick lanes and the portable models' batch
  parts refuse by name for their missing bindings; that is the test wheel's scope, not this
  change. The quick depth picks lanes the table can judge, and the tokenizer has no reference
  until the release record.
