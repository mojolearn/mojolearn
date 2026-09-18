# CPU proof follow-up (2026-09-18)

coverage.json partitions the 234 harness routes: 161 default public CPU
routes, 23 ordinary pending, and 50 parallel (18 with logical CPU paths and
32 requiring GPUs). The two public_host_only_routes are INCLUDED in 161;
they are not two extra gaps. These are route counts, not distinct algorithms.

The ARM64 release clean sweep passed; classical saved-model expectations
failed on nine stale UMAP fixtures. The sabotage run also failed: its byte-LM
loader was not opted into deliberate faulty native binaries, and the two
reference-sharded neighbor lanes discarded actual numerical mismatches as
REFUSED. The latter must retain wrong measured hashes and report DIVERGENT,
never STABLE, while genuine worker errors must remain failures. Retained
excerpt identifies both problems; full external log is arm64-certification.log.

Reporting regression tests use mocked computations; they do not constitute a
fresh native sabotage run. No release qualification or publication is claimed.

Fresh targeted native confirmation now passes: clean ties fixture twice for
both neighbor routes and byte-LM inference; intentionally faulty core bindings
retain repeatable DIVERGENT bytes, and byte-LM catches an RLPAIR_MOVED mismatch.
The strict sabotage-column validator passes. See native-control-receipt.json.
This closes the demonstrated local reporting/wiring bugs, not full release
recertification. Missing infer/model/batch entries after a training oracle
failure are unexecuted probes, not qualified inference evidence.
