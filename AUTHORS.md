# Authors and attribution

**Andrew Hendel** ([@ajhendel](https://github.com/ajhendel)) created and
maintains this project.

Every contributor retains copyright in their contribution. Git history is the
authoritative record of individual changes. The project does not require
copyright assignment.

## What is ours and what is not

This distinction is the point of the repository layout, so it belongs here
rather than only in [NOTICE](NOTICE).

**Not ours.** Everything under `ported/` implements CatBoost's algorithms and
follows CatBoost's structure, file for file. The tree-growing design, the
packing policies, the histogram strategies, the smaller-sibling rule and the
level loop are YANDEX's work, Apache-2.0, and `PORTED_MAP.tsv` names the
source file behind each of ours. Where a port is partial or replaced, that
column says so.

**Ours.** The translation itself, which is not mechanical: CUDA has warp-local
synchronization and float atomics that Mojo 1.0 and Metal do not, so several
kernels required a construct their source does not contain. `PORTING.md`
records each such deviation and why. Everything under `mojo_only/` is work
CatBoost never had to do, most substantially the fixed-point accumulator that
exists because Metal has no float atomic add.

## AI-assisted development

This project has been developed with extensive assistance from AI coding
tools. Those tools have helped draft, inspect, and integrate code and
documentation; they are not legal authors, maintainers, or accountable
decision makers.

Andrew Hendel directs the work and remains responsible for reviewing what is
accepted, accurately describing its evidence level, deciding what is released,
and responding to failures. Human contributors remain responsible for the
changes they submit. AI assistance does not weaken the project's requirements
for provenance, licensing, focused validation, reproducible benchmarks, or
honest capability claims.
