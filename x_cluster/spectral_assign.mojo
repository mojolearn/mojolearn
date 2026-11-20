# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SpectralClustering's assign_labels codes. Both assignments run in
float-float on every column: 'discretize' and 'cluster_qr' are
x_cluster/spectral_df64.mojo (device driver x_cluster/spectral_df64_device.mojo,
host column x_cluster/host/spectral_df64_host.mojo). Reference: scikit-learn
`sklearn/cluster/_spectral.py::discretize` (Yu and Shi 2003) and
`cluster_qr` (Damle, Minden and Ying 2019).
"""

comptime ASSIGN_DISCRETIZE = 0
comptime ASSIGN_CLUSTER_QR = 1


# TOMBSTONE: `cluster_qr_labels`, `jacobi_svd`, `_polar`, `_dot_cols` (assign_labels='cluster_qr' in binary64 on the
# host: the n x k pivoted QR and the k x k Jacobi SVD, the fit's last host step) deleted 2026-11-09 by lane
# m3-linalg-and-housekeeping; code recoverable at 0ce88411d3. The float-float form on every column is
# x_cluster/spectral_df64.mojo (`cq_*` bodies; `cluster_qr_df64_device`, `cluster_qr_df64_host`).
# Restore: git apply experiments/removed/CLUSTER_QR_BINARY64_HOST.patch


# TOMBSTONE: `discretize_labels` (assign_labels='discretize' in binary64 on the host: n x k loops, the fit's last host
# step) deleted 2026-11-09 by lane n9b-device-svd-qr; code recoverable at 5b5490960d. The float-float form on every
# column is x_cluster/spectral_df64.mojo (device driver x_cluster/spectral_df64_device.mojo).
# Restore: git apply experiments/removed/N9B_SPECTRAL_DISCRETIZE.patch
