# SPDX-License-Identifier: Apache-2.0
"""RUN piece of the decomposition public lifecycle
(bindings/decomp_public_lifecycle.mojo): the operation-to-method table.
Typed Mojo only, no Python import (tools/python_free_dirs.sh); moved
unchanged (lane rs-decomposition, 2026-11-12)."""


def decomp_public_method(operation: String) -> String:
    """The public method a door operation serves ("" for a private one: the
    random projection's components_ slot, get_covariance, perplexity, ...)."""
    if (operation == "fit" or operation == "partial_fit" or operation == "fit_transform"
            or operation == "fit_predict" or operation == "transform" or operation == "inverse_transform"
            or operation == "score" or operation == "score_samples" or operation == "decision_function"
            or operation == "predict"):
        return operation
    if operation == "dl_fit" or operation == "mbdl_fit" or operation == "spca_fit" or operation == "coder_fit":
        return "fit"
    if (operation == "dl_fit_transform" or operation == "mbdl_fit_transform" or operation == "spca_fit_transform"
            or operation == "coder_fit_transform"):
        return "fit_transform"
    if operation == "coding_transform" or operation == "spca_transform" or operation == "coder_transform":
        return "transform"
    if operation == "coding_inverse" or operation == "spca_inverse" or operation == "coder_inverse":
        return "inverse_transform"
    if operation == "elliptic_score":
        return "score"
    return ""
