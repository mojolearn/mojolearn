# SPDX-License-Identifier: Apache-2.0
"""RUN piece of providers/python/pca_model_pyapi.mojo: the names of the
fields whose value the PCA query decodes. Typed Mojo only, no Python import
(tools/python_free_dirs.sh); moved unchanged (lane rs-decomposition,
2026-11-12)."""


def _pca_query_decoded(name: String) -> Bool:
    # The six fields whose value (not only type) the query decodes.
    return (name == "_mojolearn_fitted" or name == "_mojolearn_fitting" or name == "n_samples_"
            or name == "n_features_in_" or name == "n_components_" or name == "whiten")
