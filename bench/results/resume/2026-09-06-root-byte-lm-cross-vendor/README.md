# Qualified bidirectional NVIDIA/AMD byte-LM training and resume

Both independent final all-raw comparator runs exited 0 with identity and
learning admitted, effective missing-moments controls and no missing prerequisites:
[NVIDIA → AMD](nv-to-amd-resume128-comparison.json) and
[AMD → NVIDIA](amd-to-nv-resume128-comparison.json).
[Combined admission and pinned record hashes](bidirectional-admission.json).

The two-block **34,944-parameter FP32** model trains on pinned real text.
All 128 continuous steps match across NVIDIA/AMD, and both fresh-process
foreign step-64 checkpoint continuations reproduce their same-device
continuous trajectories through step 128. Full raw parameters, gradients,
optimizer moments, flags/counters, losses, held-out batches and checkpoint
bytes are checked. Held-out loss falls **5.5413 → 2.8436**. Both independently
guarded first-step FP64 gradient/AdamW references pass.

This is a bounded two-vendor result. Metal, other shapes/architectures,
generation quality and actual step cost remain open. The new working-tree
immutable-bytes loader is outside this unchanged numerical inventory.
Root alone ran every test/model/comparator; remote jobs used two-core affinity
and ran serially. Both root comparator runs had a 60-second deadline and
1 GiB sampled RSS ceiling, with peaks below 47 MiB. All rentals were deleted
and verified absent before the final local comparisons.

## First direction and earlier diagnostic provenance


Root's [full raw comparator](nv-to-amd-resume128-comparison.json) exited 0:
`QUALIFIED_BOUNDED_CAPTURE`, identity and learning admitted, no missing
prerequisites. It compared both full 128-step trajectories, the actual NVIDIA
64-step checkpoint, AMD continuation through step 128, all required successful
guard receipts, and independently guarded first-step FP64 gradient/AdamW
oracles on both vendors.

The two-block model has 34,944 FP32 parameters and uses pinned real text.
Held-out loss fell **5.5412986278533936 → 2.8436418771743774**. Parameters,
gradients, optimizer moments, flags/counters and losses match byte-for-byte at
every compared step; held-out token/loss bytes and final checkpoints also match.
The missing-moments control preserves forward/backward inputs but changes
step-65 post-update parameters and both moments, as required.

That first record closes **NVIDIA → AMD**; the separate reverse record above
now closes **AMD → NVIDIA**. Metal and broader scope remain separate.
The exact retained numerical inventory is unchanged from `d921eade`; transport
snapshot `d5893e2a` supplies system-Python setup on AMD. The new working-tree
immutable-bytes loader is not part of this certificate.

Root alone ran the comparator after both rentals were deleted and verified
absent, with two-thread limits, a 60-second deadline, a 1 GiB sampled RSS ceiling
and 47,008 KiB peak observed RSS. [Command](nv-to-amd-resume128-command.json),
[root guard](nv-to-amd-resume128-root-guard.json),
[NVIDIA head evidence](../2026-09-06-root-byte-lm-resume-nvidia/README.md),
[AMD continuation evidence](../2026-09-06-root-byte-lm-resume-amd/README.md).

## Earlier continuous-only diagnostic (retained)


Root compared the complete retained NVIDIA run2 and AMD run3 at frozen
`d921eade0da5454e39e0cd103b9a1799286b41b9`. Both passed single-vendor
admission: independent first-step FP64 gradient/AdamW oracle and controls,
then 128 training steps on pinned real text. The model has two blocks and
34,944 FP32 parameters.

[Comparison](continuous128-comparison.json) found every retained raw step
array equal across all 128 steps, plus heldout token/loss bytes and final
checkpoint bytes. Both heldout losses fall 5.5412986278533936 to
2.8436418771743774. These are direct raw-file comparisons, not just hashes.

The comparator intentionally exited 1 with AGREEMENT_ONLY_DIAGNOSTIC and
identity_admitted=false because head64, foreign resume128 and effective
missing-moments control captures are absent. This establishes continuous
trajectory agreement, not the complete training/resume protocol. Apple,
training-step cost and general architectures remain outside this result.

Root alone executed the file comparator with two-thread environment limits
and a 60-second deadline. Both GPU rentals were already deleted/verified gone.
