# numpy-free-0.7 merged forward, and what it did not reach

2026-09-10. `numpy-free-0.7` was cut at `fcdcabb3` on 2026-09-07 and sat
244 commits behind `main` while the identical fan-out landed. This records
the merge of `main` into it, what survived, and what is still owed.

## What merged

`main` merged into the branch (no rebase, no history rewrite). 7 files
conflicted out of 61; the other 54 applied clean. The branch's own
contribution is 6,466 insertions and only 761 of those sit in the
conflicted files, so 88 percent of the work merged untouched.

Shipped modules carrying a top-level `import numpy`

| | shipped | tests |
|---|---|---|
| `main` at `b3279609` | 27 | 29 |
| after this merge | 3 | 30 |

Test modules are not a target. NumPy is the oracle a surface test compares
against, so a test importing it is correct.

## The three shipped modules still importing NumPy

All three are code that `main` added or redesigned AFTER the branch was
cut, so the branch never had a chance to convert them.

**`_byte_lm_impl.py`** is a DESIGN conflict, not a merge conflict. The
branch hard-codes `_N = 34944` and a fixed `_OFFSETS` tuple for the
B2/L32/DM32/H4/KV2/FF64/V256 model. `main` generalized the same file to
configurable shapes through `_byte_lm_config.ByteLanguageModelConfig`
(`shape.n_total`, `shape.parameter_shapes`, 35 `shape.` references). The
generalization is the better design and reinstating a hard-coded 34944
would regress it, so this merge takes `main`'s file whole. The NumPy
removal here has to be redone against the config abstraction. This is the
largest single owed item.

**`_training_impl.py`** kept the branch's numpy-free optimizer, clip and
loss path, and inherited four sections `main` added after 2026-09-07 that
are still NumPy-based. Those are the LR schedules (`ConstantLR`,
`WarmupLinearLR`, `WarmupCosineLR` and the exact-rational cosine),
gradient accumulation, the `Generator` RNG, and the
embedding/rms_norm/linear layer helpers. The module docstring claimed
"NumPy is not imported anywhere on this path" and that sentence was true
when it was written and is false after this merge, so it was corrected in
the same commit to name exactly which path is free and which is not.

**`_samba_impl.py`** did not exist when the branch was cut.

## Owed

1. Redo the byte-LM NumPy removal on top of `_byte_lm_config`, not on top
   of `_N`/`_OFFSETS`.
2. Convert the four post-cut sections of `_training_impl.py`.
3. Convert `_samba_impl.py`.
4. DEVIATION 1887 lost its row-tile arm. `main`'s `_arrays.as_f32_colmajor`
   copies large C-order inputs in 256 KB row tiles for cache locality; the
   branch's `_buffer.as_f32_colmajor` does the same ONE copy with the same
   bytes and the same zero-copy borrow for float32 F-order input, but with
   no tiling. Semantics and copy count are unchanged, cache behavior is
   not. RUN OWED, a fit-time measurement at the 1M-2M row floor on a
   large row-major float64 input, before this is called free.
5. `_verify.py` and `transformer.py` import NumPy lazily behind a guard
   with an install hint. That is the intended optional-diagnostic shape
   and is not an owed conversion.

## Not run

Nothing was executed beyond `compileall`. No surface test, no build, no
device work. RUN OWED for the whole merged tree.
