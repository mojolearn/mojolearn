# mojolearn 0.8.30

Published macOS arm64, Linux core, NVIDIA and AMD wheels from main at `6667bf7c49d537c9c88a31a08532d37a47dae2b2`.

The release reuses the completed checks of merged lanes at the user's explicit request. No fresh runtime verification or wheel smoke was run. All four published file hashes match the release manifest. See release-provenance.json for the merge records and publication workflow.

Rebuilt decomposition bindings include the merged LLE iterative route and bounded dense SVD work. The release reuses 108 unchanged Mac bindings and 240 unchanged Linux bindings from 0.8.29. The explicit build-only option only changes post-pack verification; its compilation equivalence record is attached.
