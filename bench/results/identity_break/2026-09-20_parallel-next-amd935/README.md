# Complete AMD parallel column after causal batch cleanup

Python/harness source `935b6f9046ac59b12d467a13c2ef4700e635bd5c` completed all 14 requested lanes × nine fixtures with full parts and one repeat. All 459 numeric parts matched existing references: 267 CPU and 192 Apple comparisons, zero findings. The column is complete and the guarded command exited 0.

Native binaries are the exact published 0.8.9 Linux wheel, SHA256 `b2f7856e5959a518ce0ebc23af0240e9092a9f3a5256faf77cc389565915638f`, native source `819a47ae48166e91951f54f54e64ee173658e32a`. They were downloaded from PyPI, hash-verified, and staged beside the newer Python source without rebuilding or relabeling them. Adjacent provenance preserves the public URL and every staged binary digest. This is reference evidence, not qualification of a new wheel or physical two-GPU execution.

The preceding source149 run stopped after 112 cells at its 12 GiB process-group RSS cap; that incomplete diagnostic record remains separate. This fresh fixed-source run completed under the same cap, peaking at 7,019,106,304 RSS bytes (6.54 GiB) and 4,428,996,608 VRAM bytes. The causal-model batch cleanup hook therefore passed a full mixed-lane hardware trial without increasing the guard limit. Runtime was about 323 seconds.

DigitalOcean MI325X rental 602247558 was automatically destroyed after fetch: DELETE HTTP204, followed by verified GET HTTP404 at 2026-09-20T21:54:46Z. No rental was retained idle.
