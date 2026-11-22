# SPDX-License-Identifier: Apache-2.0
# The `family_mani_*` operation names of the classical owner's code table
# (bindings/native_operation_codes.mojo reads this file once per process, at
# table fill; lane bw2-core, 2026-11-20). A leaf: no Python, no imports of an
# owner.
#
# CODED (lane bw2-manifold, 2026-11-20): each name has a leaf code, numbered by
# sub-owner (the decomp four 0-10, TSNE 11-12, UMAP 13-15) so the manifold
# owner picks TSNE or UMAP with one range test and each owner switches on the
# code. The owner sits behind its own compiled entry
# (bindings/runtime_manifold_entry_pack.mojo): the family door forwards the
# name, and that entry resolves the str to its code with ONE dictionary lookup
# (bindings/native_operation_names_unpack.mojo); an unknown `family_mani_*`
# name is not in the table and keeps its String, for its unchanged error text
# only (`manifold_unknown_operation`). A constant is
# `MANI_OPN_<operation name, upper case>`, the form
# tools/boundary_keys_plain.py reads back as the name.

comptime MANI_OPN_FAMILY_MANI_CMDS_FIT = 0
comptime MANI_OPN_FAMILY_MANI_CMDS_FIT_TRANSFORM = 1
comptime MANI_OPN_FAMILY_MANI_ISO_FIT = 2
comptime MANI_OPN_FAMILY_MANI_ISO_FIT_TRANSFORM = 3
comptime MANI_OPN_FAMILY_MANI_ISO_RECONSTRUCTION_ERROR = 4
comptime MANI_OPN_FAMILY_MANI_ISO_TRANSFORM = 5
comptime MANI_OPN_FAMILY_MANI_LLE_FIT = 6
comptime MANI_OPN_FAMILY_MANI_LLE_FIT_TRANSFORM = 7
comptime MANI_OPN_FAMILY_MANI_LLE_TRANSFORM = 8
comptime MANI_OPN_FAMILY_MANI_MDS_FIT = 9
comptime MANI_OPN_FAMILY_MANI_MDS_FIT_TRANSFORM = 10
comptime MANI_OPN_FAMILY_MANI_TSNE_FIT = 11
comptime MANI_OPN_FAMILY_MANI_TSNE_FIT_TRANSFORM = 12
comptime MANI_OPN_FAMILY_MANI_UMAP_FIT = 13
comptime MANI_OPN_FAMILY_MANI_UMAP_FIT_TRANSFORM = 14
comptime MANI_OPN_FAMILY_MANI_UMAP_TRANSFORM = 15
# `vendor_used` of the decomp four (Isomap, LocallyLinearEmbedding, MDS,
# ClassicalMDS): metadata, no input, served before any lifecycle step.
comptime MANI_OPN_FAMILY_MANI_VENDOR_USED = 16


def operation_names_mani(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_mani_cmds_fit")
    leaves.append(MANI_OPN_FAMILY_MANI_CMDS_FIT)
    names.append("family_mani_cmds_fit_transform")
    leaves.append(MANI_OPN_FAMILY_MANI_CMDS_FIT_TRANSFORM)
    names.append("family_mani_iso_fit")
    leaves.append(MANI_OPN_FAMILY_MANI_ISO_FIT)
    names.append("family_mani_iso_fit_transform")
    leaves.append(MANI_OPN_FAMILY_MANI_ISO_FIT_TRANSFORM)
    names.append("family_mani_iso_reconstruction_error")
    leaves.append(MANI_OPN_FAMILY_MANI_ISO_RECONSTRUCTION_ERROR)
    names.append("family_mani_iso_transform")
    leaves.append(MANI_OPN_FAMILY_MANI_ISO_TRANSFORM)
    names.append("family_mani_lle_fit")
    leaves.append(MANI_OPN_FAMILY_MANI_LLE_FIT)
    names.append("family_mani_lle_fit_transform")
    leaves.append(MANI_OPN_FAMILY_MANI_LLE_FIT_TRANSFORM)
    names.append("family_mani_lle_transform")
    leaves.append(MANI_OPN_FAMILY_MANI_LLE_TRANSFORM)
    names.append("family_mani_mds_fit")
    leaves.append(MANI_OPN_FAMILY_MANI_MDS_FIT)
    names.append("family_mani_mds_fit_transform")
    leaves.append(MANI_OPN_FAMILY_MANI_MDS_FIT_TRANSFORM)
    names.append("family_mani_tsne_fit")
    leaves.append(MANI_OPN_FAMILY_MANI_TSNE_FIT)
    names.append("family_mani_tsne_fit_transform")
    leaves.append(MANI_OPN_FAMILY_MANI_TSNE_FIT_TRANSFORM)
    names.append("family_mani_umap_fit")
    leaves.append(MANI_OPN_FAMILY_MANI_UMAP_FIT)
    names.append("family_mani_umap_fit_transform")
    leaves.append(MANI_OPN_FAMILY_MANI_UMAP_FIT_TRANSFORM)
    names.append("family_mani_umap_transform")
    leaves.append(MANI_OPN_FAMILY_MANI_UMAP_TRANSFORM)
    names.append("family_mani_vendor_used")
    leaves.append(MANI_OPN_FAMILY_MANI_VENDOR_USED)
