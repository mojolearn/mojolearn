# SPDX-License-Identifier: Apache-2.0
# The `family_chol_*` operation names of the classical owner's code table
# (bindings/native_operation_codes.mojo reads this file once per process, at
# table fill; lane bw2-core, 2026-11-20). A leaf: no Python, no imports of an
# owner.
#
# CODED (lane bw2-linalg, 2026-11-20; the form of bindings/operation_names_gp.mojo):
# each name has a leaf code, numbered by sub-owner (public state, binding metadata, fit/solve, parallel) so a prefix test
# of the owner's chain is one range test and each owner switches on the code.
# The family door resolves a caller's str to the code with ONE dictionary
# lookup (bindings/native_operation_names_unpack.mojo) and never builds the
# name; an unknown name of these prefixes is not in the table and keeps its
# String, for its unchanged error text only (`linalg_unknown_operation`).
# A constant is `CHOL_OPN_<operation name, upper case>`, the form
# tools/boundary_keys_plain.py reads back as the name. Add every name a caller
# can pass: a name missing here is refused as unknown.

comptime CHOL_OPN_FAMILY_CHOL_INITIALIZE = 0
comptime CHOL_OPN_FAMILY_CHOL_LOGDET = 1
comptime CHOL_OPN_FAMILY_CHOL_BINDING = 2
comptime CHOL_OPN_FAMILY_CHOL_DOOR = 3
comptime CHOL_OPN_FAMILY_CHOL_VENDOR = 4
comptime CHOL_OPN_FAMILY_CHOL_PROFILE_JITTER = 5
comptime CHOL_OPN_FAMILY_CHOL_FIT = 6
comptime CHOL_OPN_FAMILY_CHOL_SOLVE = 7
comptime CHOL_OPN_FAMILY_CHOL_PAR_FIT = 8
comptime CHOL_OPN_FAMILY_CHOL_PAR_SOLVE = 9


def operation_names_chol(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_chol_initialize")
    leaves.append(CHOL_OPN_FAMILY_CHOL_INITIALIZE)
    names.append("family_chol_logdet")
    leaves.append(CHOL_OPN_FAMILY_CHOL_LOGDET)
    names.append("family_chol_binding")
    leaves.append(CHOL_OPN_FAMILY_CHOL_BINDING)
    names.append("family_chol_door")
    leaves.append(CHOL_OPN_FAMILY_CHOL_DOOR)
    names.append("family_chol_vendor")
    leaves.append(CHOL_OPN_FAMILY_CHOL_VENDOR)
    names.append("family_chol_profile_jitter")
    leaves.append(CHOL_OPN_FAMILY_CHOL_PROFILE_JITTER)
    names.append("family_chol_fit")
    leaves.append(CHOL_OPN_FAMILY_CHOL_FIT)
    names.append("family_chol_solve")
    leaves.append(CHOL_OPN_FAMILY_CHOL_SOLVE)
    names.append("family_chol_par_fit")
    leaves.append(CHOL_OPN_FAMILY_CHOL_PAR_FIT)
    names.append("family_chol_par_solve")
    leaves.append(CHOL_OPN_FAMILY_CHOL_PAR_SOLVE)
