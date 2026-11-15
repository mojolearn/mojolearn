# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""How many training rows the k-means++ seeding reads (`kpp_seed_rows`),
for the device seeding in x_ann/kpp_seed_device.mojo."""
# TOMBSTONE: MOJOLEARN_ANN3_PQ_SEED (unmeasured opt-in arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-14-cluster; code recoverable at fa411f5da5.
# Restore: git apply experiments/removed/MOJOLEARN_ANN3_PQ_SEED.patch; record in docs/TOMBSTONES.md.
# (Sweep 2026-11-15: the host k-means++ seeding `kpp_seed`, its lane width KPP_W and the HostRng import removed: its only caller was the always-False PQ_FAST_SEED branch of x_ann/ivf_pq_device.mojo.)


def kpp_seed_rows(n: Int, k: Int, rows_per_seed: Int) -> Int:
    """How many training rows the seeding reads."""
    var ns = rows_per_seed * k
    return ns if ns < n else n

