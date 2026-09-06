# Embedding

GPU embedding lookup and gradient primitives with a cross-vendor reproducibility path.

`IDENTICAL_EMBEDDING_CONTRACT.md` is the normative specification for index handling, duplicate
updates, accumulation order, padding behavior, and error semantics. Implementation changes should
be evaluated against that contract rather than historical prose.

FAST implementations may use a different schedule only when their behavior remains inside the
documented mode boundary and repeated benchmarks demonstrate an actual speed advantage.
