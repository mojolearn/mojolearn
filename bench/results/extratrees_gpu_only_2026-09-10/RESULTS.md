# Extra Trees GPU-only public training — 2026-09-10

Apple M4, Mojo 1.0.0 ed45d567. Public Python constructors/parameter updates
and binding calls refuse CPU training; native unsuffixed convenience fits
now call the existing GPU APIs. Host trainers are explicitly *_reference
oracles with check/benchmark consumers migrated. Host prediction remains a
separate GPU-inference follow-up; this change does not claim full residency.

All three ET extensions built, skipping broad build gates in favor of these
focused checks. `binding_refusal.log`: both raw fit exports reject six invalid
device selectors in each mode (36 cases) before dereferencing null pointers.
`default_before.json` equals `default_after.json`: six GPU classifier/regressor
forest-array and prediction fingerprints are unchanged across the rebuild.

`native-{mode}-fixed.log`: no-context native public entrypoints agree with the
existing context-taking GPU API in every node field/leaf bit, for classification
and regression (four small fits per mode). All exit 0. Original fixture compile
failure from unsupported List construction is retained in `native-fast.log`;
the fix changes fixture literals only. Python refusal checks: 16 passed, retained
in the sibling `../et_gpu_only_2026-09-10/python-refusal.log` artifact.

No remote work, performance claim, full wheel release or cross-device
qualification is part of these checks. Existing GPU APIs and their numerical
algorithms are unchanged. Loading an inference archive does not restore a CPU
training option; explicit CPU constructor/refit requests now fail.
