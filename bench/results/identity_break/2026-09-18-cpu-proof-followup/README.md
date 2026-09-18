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
