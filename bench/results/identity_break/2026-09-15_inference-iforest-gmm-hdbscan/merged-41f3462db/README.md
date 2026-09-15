# x86 confirmation on the soft clustering merge (41f3462db)

One RunPod CPU pod (8 vCPU, pod 6an2rrr4m4zx12, 493 s billed, $0.0329, delete verified by GET 404),
host bindings built there at x86-64-v3 from 41f3462db through the R2 binding cache (five working-tree
evidence files were uncommitted and shipped with the source; no code was). Raw leg directory untracked,
under ~/mojolearn-evidence/inference-neighbors-density/.

| file | verdict |
|---|---|
| `check_x86_host.2026-09-15-apple-m4-neighbors-density.txt` | through a host directory holding only the five wheel bindings this lane uses: `gate verdict IDENTICAL (54 fixtures, 4 GPU columns, exit 0)`, 162 identity hashes EQUAL to committed infer cells, 54 ABSENT |
| `check_x86_host.2026-09-15-apple-m4-iforest-gmm-hdbscan.txt` | `gate verdict IDENTICAL (16 fixtures, 4 GPU columns, exit 0)`, 36 EQUAL, 18 N/A, 10 ABSENT (the re-recorded hdbscan models included) |
| `check_x86_sabotage.2026-09-15-apple-m4-neighbors-density.txt` | `-D MOJOLEARN_HOST_SABOTAGE=1`: `gate verdict EXPECTED MISMATCH SEEN`; the four fixtures that stay EQUAL are knn-cosine, knn-rbc, radius and radius-manhattan on ties, as on the M4 |
| `check_x86_sabotage.2026-09-15-apple-m4-iforest-gmm-hdbscan.txt` | DID NOT RUN (exit 2): the leg built no sabotage copy of mixture_infer or hdbscan_infer, so the first gmm model could not load. Owed to the next leg |
| `cpu-x86.iforest-gmm-hdbscan.merged.json`, `diff.iforest-gmm.txt` | iforest and gmm against the 166-lane record cut to base, ties and dupes: `summary: IDENTICAL=12`, `summary (infer/model): IDENTICAL=12, OWED=12`, `summary (batch): IDENTICAL=12`, require-columns 4 OK (12 OWED) |
| `diff.hdbscan.txt` | hdbscan against the membership lane's Metal column with the record's NVIDIA and AMD columns: `summary: IDENTICAL=6`, infer/model OWED=12, batch OWED=6, require-columns 4 OK (18 OWED) |
| `diff.hdbscan-vs-membership-cpu-x86.txt` | against the membership lane's own x86 CPU column: `summary: IDENTICAL=6`, infer/model IDENTICAL=6, batch IDENTICAL=6 |
| `cpu-x86.iforest-gmm-hdbscan.merged.sabotage.json`, `diff.sabotage.txt` | `summary: DIVERGENT=18`, `summary (infer/model): DIVERGENT=18, ONE-COLUMN=18`, `summary (batch): DIVERGENT=18`; no cell left IDENTICAL |
| `test_host_surface.txt` | 131 passed, 1 failed: `test_recordings_and_columns_exist`, because a leg ships only the bench paths it names; it passes on the M4 |
