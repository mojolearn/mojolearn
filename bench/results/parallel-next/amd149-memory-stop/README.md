# Incomplete AMD trial retained for diagnosis

Python/harness source `14944378ff9d7a733cd39779c6bdb24b7d041e76`, exact published 0.8.9 Linux native wheel (source819, SHA256 in provenance). The 12 GiB process-group RSS guard stopped the run after 112 of 126 requested cells, during `par-causal-lm/denormal`. Its last RSS sample was 13,774,925,824 bytes. The checkpoint is `complete:false`, is not admissible reference evidence, and is not relabeled as a finished subset.

All 440 numeric parts recorded before the stop matched current references (248 CPU and 192 Apple comparisons). The guard failure led to finding a real harness cleanup omission: the causal-model batch probe did not register its model close hook. The fixed-source full rerun is separate evidence.

Rental 602246436 was deleted and absence verified by HTTP404 at 2026-09-20T21:43:01Z. A contemplated resume was stopped during preflight before creating another rental. The cleanup validation instead starts a fresh full column from source935.
