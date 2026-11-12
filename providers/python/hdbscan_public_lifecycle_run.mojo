# SPDX-License-Identifier: Apache-2.0
# RUN part of providers/python/hdbscan_public_lifecycle_pyapi.mojo: typed
# Mojo only (no Python imports).


def _query_slot(method: String) -> String:
    return "cluster_hdbscan_" + method + "_v2"
