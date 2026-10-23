# Verify divergences and refusals, AMD merged verify of main 4a2ef64ad (2026-10-23)

Source: `~/mojolearn-evidence/box-archive/amd-verify-20261023-0257/out/` (MI325X gfx942):
387 verified, 3 DIVERGENT, 2 REFUSED, 26 OWED. Table: `python/mojolearn/verify_reference/table.json`.
Branch `lane/verify-fixes-20261023`.

## Digests per lane (base fixture, `train` part unless noted)

| lane | part | table ref (recorded at) | AMD 661016f7c | AMD 76c5671cb | Apple 76c5671cb | AMD e863a07c5 | AMD 4a2ef64ad verify | AMD 4a2ef64ad record | NVIDIA b43116e5b |
|---|---|---|---|---|---|---|---|---|---|
| logistic-multiclass | train | 09d3df7155c9fe0c (amd 661016f7c, nv f6d0e6eda) | 09d3df71 | e73ef01d | e73ef01d | e73ef01d | e73ef01d | e73ef01d | not landed |
| logistic-multiclass | batch / infer / model | 52e04d35 / e684e696 / 7a548ff1 | same as ref | 23a76c65 / dc3f1070 / 12ac7127 | same as AMD | same | same | same | not landed |
| hf-checkpoint | train (`refusals` sub-part) | 2e4e083eaea46d80 (be0be71e) | 2e4e083e | 01d49c67 (18953450) | 01d49c67 | 01d49c67 | 01d49c67 | 01d49c67 | not landed |
| hf-causal-lm | train (`every` sub-part) | 7f4b353f4b4a5355 (084fccb3) | 7f4b353f | 7f4b353f (084fccb3) | 7f4b353f | 3a5d4daf (8e50bbee); its verify read 586addd4 | 3a5d4daf | 8a4a1c15 (d00f751e) | not landed |

On the AMD box the host column agreed with the device on every cross-checked lane (`batch=agree infer=agree`,
logistic-multiclass included), so the host and device moved together wherever they moved.

## logistic-multiclass: a deliberate fold-order change, re-record

Every part moved (coef and proba sub-parts, batch, infer, model), and it moved the same way on AMD
(76c5671cb, e863a07c5, 4a2ef64ad) and Apple (76c5671cb record; the integ-20261021 Mac verifies; the
7e2ba679a sabotage clean arm): e73ef01d8a478373. The cause is `4b8a8e8f2` (lane debt-block-per-n, 2026-10-21):
"GLM gradient: X^T dZ in the tile order wherever it took one block per cell". It sends every C > 1 gradient
through the tiled order (`glm/impl/qn/glm_base.mojo`, `glm/impl/qn/qn_tiled_rule.mojo`) and moves the host
oracle (`glm/host/qn_oracle.mojo`) with it. Binary logistic (C == 1) verified, as it should. The review
already knew: `docs/plans/PUBLIC_RUNTIME_MIGRATION.md` lists "the recorded `logistic-multiclass`/`hf-checkpoint`
mismatches" as owed.
Action: re-record (ONLY_CHANGED) on every column: apple, nvidia, amd, cpu.

## hf-checkpoint: refusal wording, re-record

Only the `refusals` sub-part moved (be0be71e13c1f100 -> 189534504634264a). `caught` (21ba7c1d) is unchanged,
so every refusal still names its reason. Only the sentences changed. `refusals` hashes the joined messages
of `HF_PLAN_REFUSALS` (`tools/identity_break.py:5371`). Those messages became native in the
causal_model_config migration and were reworded in 400d1ef4a, 43a235dac ("a missing required field says so")
and 76c5671cb (the doubled `TypeError: TypeError:` prefix dropped). The Mac integ checks show the
moves: e6a28132 (8bda94617), then a941a056 (9cb943ea9, 518f63769), then 18953450 (43a235dac). The text
does not depend on the vendor: AMD and Apple both read 18953450. The review had already flagged this lane
(PUBLIC_RUNTIME_MIGRATION.md, above).
Action: re-record (ONLY_CHANGED) on every column.

## hf-causal-lm: use-after-free in the phi3 row slice, kernel-side fix, no re-record

Only `every` moved (all eight HF_ARCHITECTURES forwards). `logits`, `generate`, `step` and `params` (the
Llama row) are unchanged. On AMD, `every` read four different values in four runs of two builds (8e50bbee
twice, d00f751e, and the digest behind 586addd4). The same commit read differently in its verify and
its record. So the digest changes from run to run. It is neither a vendor difference nor a planned bit change.

Cause: `bindings/causal_lm_construct.mojo` `causal_model_read_weights_binding` slices phi3's fused
`qkv_proj` / `gate_up_proj` rows with
`memcpy(..., src=f32_ptr(Int(py=value._addr)) + begin * cols, ...)` and then sets `value = sliced`. The
`value._addr` read is `value`'s last use, and Mojo destroys a local at its last use, so the fused array was
freed before the memcpy read it. The pattern dates from 9d3b9ba18 (2026-10-20). The balanced calls of
db9c30e83 / 528b48840 (2026-10-21, py_balanced) uncovered it: before them, the leaked reference on every
call argument kept the array alive. Once it was freed, the allocator's free-list words (glibc's are
randomized per process) landed in the first rows of the q and gate slices.
Fix: `bindings/causal_lm_construct.mojo:590-600`. A `whole` copy owns the fused tensor through the copy
(`_ = whole^` after it). The bits return to the recorded reference 084fccb3 / 7f4b353f, which was recorded
while the leak still kept the source alive. No re-record. Every vendor and the host column ran the same
loader, so a run after the balanced calls could read wrong bits on any of them.

## Refusals

- `hf-causal-lm-text`: the lane was wrong. `tools/identity_lanes/coverage.py` `_coverage_text_tokenizers`
  now builds the 256-byte vocabulary that byte-level BPE requires. The fixture's 64-byte alphabet sits at
  ids 0..63, which is the model's vocabulary, and the other 192 bytes take ids 64..255. The prompt is spelled
  in the alphabet and the model emits ids below 64 only. The product refusal (`tokenizer/impl/vocab_build.mojo:90`)
  stays.
- `classical-float16-sequences`: the product was wrong. The lane got through OLS, Ridge and both scalers,
  then stopped at `PCA.fit` (`decomposition.py:386` `as_f32_c`): `_buffer._materialize` refused a list whose
  leaves are `numpy.float16`, which subclasses `numbers.Real` but not `float`. scikit-learn's `check_array`
  accepts such a list. `python/mojolearn/_buffer.py` (list branch) now takes numbers.Real leaves through the
  float64 store and numbers.Integral leaves through the int64 store, as the object-dtype branch already did.
  Both stores are exact for half and single values.

## OWED (26)

All 26 are lanes added after the table's records: the 17 coverage-probe lanes in `coverage.py`'s `_batch_decl`,
the 4 weighted-tree batch parts (da14b3ee3), and 5 native-owner and tape lanes (1f1183e21, 19771904e). None
is unexpected. The OWED(14) group, which has no hash at all, holds sequence-device-tensor,
sequence-adafactor-device, sequence-lamb-device and 11 others. That group is not part of the 26.

## Run owed

- Build: `_mojolearn_tokenizer_host` (it carries `causal_lm_construct`) and every merged binding that
  imports `bindings/causal_lm_construct.mojo`, for host, fast and identical. `_buffer.py` is Python only.
- ID check: hf-causal-lm on nv and amd (expect 7f4b353f4b4a5355), plus classical-float16-sequences and
  hf-causal-lm-text (new, owed).
- Re-record, ONLY_CHANGED, on apple, nvidia, amd and cpu: logistic-multiclass, hf-checkpoint, plus the
  two refused lanes now that they run.
