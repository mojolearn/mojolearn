# Neural gradient-buffer pooling — two H100s

Cloud qualification passed on RunPod `smqlvlvt7exixd`, 2026-09-14, using two
H100 80GB HBM3 GPUs (81559 MiB each), driver 580.126.09 and Mojo/MAX 26.5.
No local builds or tests were run. This receipt qualifies gradient-buffer
capacity and the recorded numerical fixtures, not complete-model capacity,
throughput scaling, or every vendor/configuration.

The final production and fault builds preserve the original per-column
accumulation tree. Thirty shape/accumulation cases match the original path
bitwise, including cancellation, signed zero and subnormals. Invalid admission
and nonfinite inputs refuse without publishing output. Post-compute failures
on both owners leave the caller unchanged; subsequent valid calls recover
exactly. Sixty optimizer configurations over three steps pass, as do MLP,
Samba and clipped attention/dropout Samba ordered-replay checks. Complete MLP,
Samba and optimizer JSON receipts also match the earlier RTX 5090 references;
this does not constitute execution of the new implementation on RTX 5090.

## Measured capacity

The same 16-microbatch fixture contains 1,500,000,001 gradient columns and
96,000,000,064 bytes of input. One H100 refuses an actual CUDA allocation with
all output canaries unchanged. Two H100s pass with every one of the
1,500,000,001 output values checked bitwise against the original reduction of
a repeating prime-length fixture. Sampled peak memory is 69393 MiB per GPU.
The full output SHA256 is
`6cded5c6e714aa73487d33819fcc44f394587e4118d047852e386fb8916e4c7f`.
This is a gradient-buffer component gate, not training a 1.5B-parameter model.
The host retains the complete input and output.

## Evidence and failed attempts

- `out/final/` contains production/fault build logs, numerical receipts,
  capacity JSON/logs/memory samples, and the final successful comparison.
- `out/reference-5090/` preserves the exact comparison references.
- `out/jobs/` retains job scripts, logs and return codes, including failures.
- `out/source-final.tgz` is the final corrected source snapshot, SHA256
  `c61831b8afcaa1bef68dcff188e5231e573f891a09bd2eee2d7d4a0a74bf6343`.
  Its embedded historical commit is not a substitute for this archive hash.
- The initial gate used a mistaken cancellation expectation at 2**24. Original
  and pooled outputs agreed; the expected zero was wrong. The corrected
  2**26 witness distinguishes the original balanced tree from a serial fold.
  The initial source and failed logs remain in `out/`.
- The one-device capacity process used the frozen capacity script. Its host
  allocation suffered transparent-huge-page compaction before completing the
  expected GPU refusal. The two-device process used the preserved
  `out/final/capacity-no-thp.py`, setting `NUMPY_MADVISE_HUGEPAGE=0` before
  importing NumPy. Sizes, data and arithmetic are unchanged. This host-only
  setup adjustment is now in the checked-in gate. Reported elapsed time omits
  host setup and supports no throughput claim. Cgroup OOM counters stayed zero.

The pod is retained for the separately qualified clipping work; this receipt
makes no claim that it was terminated after this batch.
