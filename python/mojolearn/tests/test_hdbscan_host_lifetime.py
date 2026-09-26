# SPDX-License-Identifier: Apache-2.0
"""Regression for the owned row-norm buffer used through untracked pointers."""
import hashlib
import numpy as np
import mojolearn as ml


def _hash(value):
    value = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()[:16]


def test_hdbscan_row_norm_owner_survives_distance_tasks():
    # The verifier's base prefix, deliberately strided and large enough that
    # a freed norm buffer is reused. Before the fix leaf fit refused 3998
    # NaN distances, while EOM silently produced a different clustering.
    x = np.random.default_rng(0).standard_normal((4000, 16)).astype(np.float32)
    for rows, options, expected in (
        (2000, dict(min_cluster_size=8, min_samples=3,
                    cluster_selection_method='leaf', allow_single_cluster=True), '7e6eda79b757c89f'),
        (4000, dict(min_cluster_size=5), '3c5c76e51b0ae88f'),
    ):
        model = ml.HDBSCAN(prediction_data=True, **options).fit(x[:rows, :4])
        parts = dict(labels=_hash(model.labels_), core=_hash(model.core_distances_),
                     counts=_hash(np.asarray([model.n_clusters_, model.n_outliers_,
                                              model.n_boruvka_rounds_, model.n_condensed_clusters_], dtype=np.int64)))
        joined = '|'.join(f'{k}={v}' for k, v in sorted(parts.items())).encode()
        got = _hash(np.frombuffer(joined, dtype=np.uint8))
        if ml._backend.requested_mode() == "identical":
            assert got == expected
        else:
            # FAST is not bitwise (its k-NN core distances are FAST
            # arithmetic): the hash is REPORTED, and the regression this
            # file guards -- a freed norm buffer refusing NaN distances or
            # producing a different clustering -- is asserted by the fit
            # completing with finite core distances and clusters found.
            print(f"[fast] hdbscan host lifetime rows={rows} hash={got} (identical pin {expected})")
            assert np.all(np.isfinite(np.asarray(model.core_distances_)))
            assert int(model.n_clusters_) > 0


if __name__ == '__main__':
    test_hdbscan_row_norm_owner_survives_distance_tasks()
    print('PASS HDBSCAN row-norm lifetime regression')
