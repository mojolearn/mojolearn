# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LANE cpu2-l7-metrics (2026-10-04): op 61 `rank_epi`, the ranking and probability tails (ovo / ovr averages, roc_auc and average precision averages, d2 log loss and Brier denominators, dcg discounts, ndcg) and op 62 cl_epi_unit, the clustering centroid tail,
on the device in soft binary64 (x_metrics/tail.mojo helpers). STUB: filled by the lane."""
from x_metrics.common import FP, IP, p


def rank_epi_unit(t: Int, f: FP, q: IP):
    pass


def cl_epi_unit(t: Int, f: FP, q: IP):
    pass
