# SPDX-License-Identifier: Apache-2.0
# The `family_low_*` operation names of the classical owner's code table
# (bindings/native_operation_codes.mojo reads this file once per process, at
# table fill; lane bw2-core, 2026-11-20). A leaf: no Python, no imports of an
# owner.
#
# CODED (lane bw2-linalg, 2026-11-20; the form of bindings/operation_names_gp.mojo):
# each name has a leaf code, numbered by sub-owner (queries, init, conversions, dict owners) so a prefix test
# of the owner's chain is one range test and each owner switches on the code.
# The family door resolves a caller's str to the code with ONE dictionary
# lookup (bindings/native_operation_names_unpack.mojo) and never builds the
# name; an unknown name of these prefixes is not in the table and keeps its
# String, for its unchanged error text only (`lowbit_weight_unknown_operation`).
# A constant is `LOW_OPN_<operation name, upper case>`, the form
# tools/boundary_keys_plain.py reads back as the name. Add every name a caller
# can pass: a name missing here is refused as unknown.

comptime LOW_OPN_FAMILY_LOW_IS_PACKED = 0
comptime LOW_OPN_FAMILY_LOW_FORMAT_OF = 1
comptime LOW_OPN_FAMILY_LOW_REPR = 2
comptime LOW_OPN_FAMILY_LOW_BF16_INIT = 3
comptime LOW_OPN_FAMILY_LOW_INT8_INIT = 4
comptime LOW_OPN_FAMILY_LOW_MATERIALIZE_ONE = 5
comptime LOW_OPN_FAMILY_LOW_WIDEN_BF16 = 6
comptime LOW_OPN_FAMILY_LOW_PACK_ONE = 7
comptime LOW_OPN_FAMILY_LOW_UNPACK = 8
comptime LOW_OPN_FAMILY_LOW_PACK = 9


def operation_names_low(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_low_is_packed")
    leaves.append(LOW_OPN_FAMILY_LOW_IS_PACKED)
    names.append("family_low_format_of")
    leaves.append(LOW_OPN_FAMILY_LOW_FORMAT_OF)
    names.append("family_low_repr")
    leaves.append(LOW_OPN_FAMILY_LOW_REPR)
    names.append("family_low_bf16_init")
    leaves.append(LOW_OPN_FAMILY_LOW_BF16_INIT)
    names.append("family_low_int8_init")
    leaves.append(LOW_OPN_FAMILY_LOW_INT8_INIT)
    names.append("family_low_materialize_one")
    leaves.append(LOW_OPN_FAMILY_LOW_MATERIALIZE_ONE)
    names.append("family_low_widen_bf16")
    leaves.append(LOW_OPN_FAMILY_LOW_WIDEN_BF16)
    names.append("family_low_pack_one")
    leaves.append(LOW_OPN_FAMILY_LOW_PACK_ONE)
    names.append("family_low_unpack")
    leaves.append(LOW_OPN_FAMILY_LOW_UNPACK)
    names.append("family_low_pack")
    leaves.append(LOW_OPN_FAMILY_LOW_PACK)
