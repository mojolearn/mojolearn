# SPDX-License-Identifier: Apache-2.0
# The compiled owner of the fits of DBSCAN (python/mojolearn/density.py), AgglomerativeClustering (_hierarchy_impl.py), SpectralClustering and SpectralEmbedding / spectral_embedding (_spectral_impl.py): one native call per public method,
# reached from bindings/family_native.mojo for every `family_cfb_*` operation.
from std.python import Python, PythonObject
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.cfb_dbscan_door import cfb_dbscan_operation
from bindings.cfb_agglo_door import cfb_agglo_operation
from bindings.cfb_spectral_door import cfb_spectral_operation
from bindings.operation_names_cfb import (
    CFB_OPN_FAMILY_XC_DBSCAN_PREDICT, CFB_OPN_FAMILY_CFB_AGGLO_SETSTATE, CFB_OPN_FAMILY_CFB_COO_TRIPLES)


def cluster_fit_b_unknown_operation(operation: String) raises:
    """A `family_cfb_*` name the operation table does not hold (the shells
    pass table names only): the text the name chain raised, chosen by the
    same prefixes (each sub-owner's last line named the operation)."""
    if operation.startswith("family_cfb_dbscan_"):
        raise Error("RuntimeError: unknown DBSCAN operation " + operation)
    if operation.startswith("family_cfb_agglo_"):
        raise Error("RuntimeError: unknown AgglomerativeClustering operation " + operation)
    if operation.startswith("family_cfb_sc_") or operation.startswith("family_cfb_se_"):
        raise Error("RuntimeError: unknown spectral operation " + operation)
    raise Error("RuntimeError: unknown family operation " + operation)


def cluster_fit_b_operation(model: PythonObject, operation: Int, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    """`operation` is the leaf code of bindings/operation_names_cfb.mojo,
    numbered by sub-owner (lane bw2-cluster): one range test picks the owner
    (it was three prefix tests and three name compares per call). The x_cluster
    DBSCAN predict reaches here as CFB_OPN_FAMILY_XC_DBSCAN_PREDICT."""
    classical_connect_numpy(context)
    if operation <= CFB_OPN_FAMILY_XC_DBSCAN_PREDICT:
        return cfb_dbscan_operation(model, operation, args, context)
    if operation <= CFB_OPN_FAMILY_CFB_AGGLO_SETSTATE:
        return cfb_agglo_operation(model, operation, args, context)
    if operation <= CFB_OPN_FAMILY_CFB_COO_TRIPLES:
        return cfb_spectral_operation(model, operation, args, context)
    raise Error("RuntimeError: unknown family operation code " + String(operation))
