# The AMD predict recording, PARTIAL: 3 of 36 (2026-09-16)

`lane/classical-host-recordings`, same box and lease as
`../2026-09-16-amd-kmeans`. **This directory is deliberately NOT in
`host_surface.CLASSICAL_RECORDED`**, and the name says why: it holds
`dbscan` on `base`, `ties` and `hashed` and nothing else, where the NVIDIA
recording of the same four lanes holds 36.

`record` did not fail a comparison. It DIED, on the fourth fixture:

    Exception: At max/mojo/max/gpu/host/device_context.mojo:4073:35:
    HIP call failed: hipErrorOutOfMemory (out of memory)
      density.py:325 dbscan_fit_core   <- DBSCAN(eps=0.9, min_samples=5,
                                          prediction_data=True).fit(X[:6000, :4])

on a card with 192 GB, fitting 6000 rows of four columns.

What the three fixtures that DID record are worth, on this box: `check` reads
`gate verdict IDENTICAL (3 fixtures, exit 0)` and the predict-only sabotage
arm reads `EXPECTED MISMATCH SEEN (3 fixtures, exit 0)` under
`--every-fixture`. They are real cells, on a vendor that had none. They are
not coverage of the four predict lanes and must not be quoted as such.

See `../../identity_break/2026-09-16_amd-mi300x/README.md` for what the
identity column says about the same failure, which is the more informative
half: the first four DBSCAN fits in the process succeeded and every one after
them raised, on every lane and every fixture, including shapes that had just
worked.
