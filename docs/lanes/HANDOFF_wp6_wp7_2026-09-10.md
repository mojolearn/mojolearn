# WP6 / WP7 host transfers — September 10, 2026

Scope: non-tree WP6 and WP7. WP8 and tree bindings/estimators belong to
other work. The starting source is d9285a63, including the NumPy-free
integration (36d48b07); the earlier statement that the native converters
were not wired into estimators is superseded by that integration.

## Checked foundation

`bindings/hostptr.mojo` implements DEVIATION 2486: typed non-null pointer
constructors, the hostpass B2 SIMD-8 Float32 copy with a scalar tail, and
Transformer's existing owned-list memcpy read. Counts are elements;
callers validate spans and own lifetimes. No arithmetic or tier dispatch
changes. No production binding calls this module yet.

Host gate: 37,184 checked cells, five pointer/length refusals, plus same-span
copy checks. Covers lengths 0/1/2/7/8/9/15/16/17/31/32/33/2049, offsets 0–7,
untouched boundary sentinels, signed zeros, infinities, subnormals and NaN
payloads. This copy has no numeric-mode dependency. Apple M4 host execution
passed; other hosts remain RUN OWED. This is correctness evidence, not a
performance measurement or estimator surface gate.

```
nice -n 19 pixi run mojo -I . bindings/checks/hostptr_check.mojo
```

## GP candidate, not activated

`bench/results/wp6_hostptr_2026-09-10/gp-candidate-UNQUALIFIED.patch` is a
reviewable first binding migration. It delegates pointer constructors,
changes fit input reads and factor/dual output writes to bulk copies,
changes prediction's quadratic factor/input reads, and removes the unused
prediction-only y_train fill. The empty/negative n_star staging behavior
remains so gpr_predict_host retains its named refusal. The patch is NOT
applied to production source and has NOT passed compilation or a surface
gate. Apply only after capturing the baseline:

```
git apply bench/results/wp6_hostptr_2026-09-10/gp-candidate-UNQUALIFIED.patch
```

The unchanged GP baseline build, using build_gp.sh under the local guard,
was terminated after compressed memory grew by more than 256 MiB. Entry
already had about 6.1 GB swap and 5.7 GB compressed memory on a 16 GiB Mac.
A lower-concurrency retry was refused before launch because memory pressure
was no longer normal. Neither failure is a source/compiler verdict. Limits
were not weakened; no other lane's process was stopped. Logs and guard
telemetry are retained beside the patch.

RUN OWED, with normal memory pressure, before applying the candidate:

```
PATH="$HOME/.pixi/bin:$PATH" MOJOLEARN_COMPILE_JOBS=1 \
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=apple \
nice -n 19 python3 tools/macos_serial_guard.py --seconds 180 --rss-gib 4 \
  --memory-policy user-tiny -- sh bindings/build_gp.sh
PYTHONPATH=python MOJOLEARN_NUMERIC_MODE=identical \
  /tmp/mojolearn-root-validation-20260906/bin/python \
  python/mojolearn/tests/test_gp_surface.py
```

Retain baseline binary and full output bytes, then apply candidate, repeat
build/surface gate, and cmp the output capture. Read gp_numeric_mode and
vendor from each binary. Repeat surface/output gates for other advertised
tiers; NVIDIA/AMD execution is RUN OWED. The n_train=20,000 prediction timing
is RUN OWED: its factor alone is 1.6 GB, so do not launch it under the current
memory pressure. Use an already constructed model to avoid a large fit;
interleave at least five old/new pairs after warmup within a bounded run.
No speed gain, opponent update, or default promotion follows from this work.

## Remaining sequence

After GP admission, migrate byte-LM (three model reads, four captured model/
gradient copies and output writes), Mamba, non-tree SVM, metrics and
preprocessing, then the listed non-tree estimator upload helpers. One binding
migration per commit. Keep owned-state/failure validation intact; inspect
moves separately from byte copies. GBDT/isolation-forest work is excluded.

WP7 remains unimplemented. Source inspection confirms mutable input operands
in kernel_methods/checks/kernel_matrix.mojo::km_kernel_matrix and
gaussian_process/checks/kernels.mojo::gp_kernel_matrix. Removing duplicate
uploads requires checking their callees as well as those signatures;
retaining pointer lifetimes is mandatory. Mixture's E-step precision inputs,
spectral's host-only sum-scale staging, kNN classification's returned index
buffers, and HDBSCAN's pre-download synchronizations still need their own
fingerprint gates. No WP7 deletion is qualified by the host helper check.
