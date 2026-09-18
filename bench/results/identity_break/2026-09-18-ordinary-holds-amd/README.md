# Ordinary held-route AMD evidence


## Final AMD checkpoint and resource cleanup

All 23 ordinary held routes now have fresh AMD full-property captures: 990
numerical parts agree with current CPU-backed references, zero mismatches.
The final seventeen-route admission changes 153 cells / 774 numerical parts;
1,971 unrelated cells are unchanged. The earlier six-kernel admission accounts
for the other 216 numerical parts. Raw records and strict admission receipts
are in `bench/results/identity_break/2026-09-18-ordinary-holds-amd/`.
Two normalized GP routes initially refused because the minimal build omitted
preprocessing; same-VM recovery built it and captured both successfully. The
original refused columns and original `extra_exit=1` remain preserved.

Twenty-eight historical Apple disagreements remain preserved, not resolved:
26 old `n/a:no-save` model entries and two actual `gp-sample-y/odd` train/infer
hash disagreements. Strict admission checked matching current input/revision
metadata for the retained records, so those two cannot simply be dismissed as
stale fixtures. Their cause remains unexplained and needs a future Apple
investigation/retake; no universal CPU/Apple/NVIDIA closure is claimed.
Five neural reasons now read `unwatched`; all 23 default verification holds
remain (18 classical/kernel candidates plus five neural lanes). NVIDIA and
watched installed CPU replay remain outstanding. Installed development-wheel
replay was deferred under the final usage constraint; no new wheel certificate
or PyPI publication is claimed. Whole loaded-LM AMD evidence was delivered to
the CausalLM lane, which independently compared all 144 parts with CPU/Apple.

AMD VM enc1-gpuvm012 was deleted: DELETE 204, subsequent GET 404 and list absent
at 18:23:16Z. Controller finished, Mac deadman canceled, slot released. The
temporary auto-resume helper PID 42724 was explicitly terminated after checking
its command; no delayed SIGCONT remains. Balance delta was $1.30. No new pods
are authorized. Full fetched evidence survives locally and in this branch.

Bundle regeneration caveat: default `--emit-models` still selects only the four
legacy base models. Do not use it to overwrite the tracked 58-model manifest.
To reproduce the expanded bundle, retain those four entries and copy the exact
54 files from committed AMD `captures-kernels` records, admitting each only
when its file digest equals the table model reference and a numeric batch
reference exists. Preserve each entry's fixture, lane, actual class agreement
and source-record digest. `test_portable_kernel_models.py` checks this payload.
Automating that full regeneration selection is deferred; both builders already
ship the complete tracked bundle. No additional native builds were started.
