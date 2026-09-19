# Empty array NumPy interoperability fix

Six AMD radius fixtures refused before producing their requested model hashes.
The retained failure log names `TypeError: int() ... not Array`. On the same
Python 3.10.12 / NumPy 2.2.6 host, an empty int64 Array exported pointer zero
and reproduced this exception; a nonempty Array converted successfully.

Commit 1aa581748cc09e14982baf36e81b509ffe49f1ea supplies a stable non-null
address only for empty array-interface exports. Kernel addresses, array storage,
and all 123 native/runtime wheel files are unchanged. The private candidate
and parent SHA are in array-candidate-audit.json; neither is published 0.8.8.

Fifteen empty shape/dtype conversion cases passed on the AMD host using an
isolated import of the patched module, without changing the installed package
used by the ongoing capture. The new local pytest regression passed (15 empty
cases plus a nonempty zero-copy write check). A broader existing test could
not run in that worktree because its native conversion binding was absent;
no native rebuild or numerical suite was launched. Hardware rerun targets
only the six refused fixtures; the three matching fixtures remain retained.

The installed fixed candidate now completes all six formerly refused AMD
fixtures (two repeats each). All eighteen captured train/infer/model hashes
match the existing references; six of these close the missing model witnesses.
See `amd-array-fix-par-queries-radius-two.json` and
`array-fix-hardware-comparison.json`. No reference digest changed.
