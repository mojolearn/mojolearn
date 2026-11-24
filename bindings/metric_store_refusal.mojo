# SPDX-License-Identifier: Apache-2.0
"""The refusal words of a GPU metric image whose resident store cannot be
reached (lane vdf-xm-wire2, 2026-11-24): the x_prep transport of its tier and
vendor is not connected or does not match. Shared by the x_metrics program
runner (bindings/x_metrics_resident_door.mojo `metric_store_ref`) and the
kernel metrics' slots (bindings/metrics_slot_door.mojo `McSlots`); no imports,
so either owner can read it. The selector refuses, it never widens: before,
both swallowed the table refusal and took the host route."""


def metric_store_refusal(family: String, mode: Int, vendor: Int, cause: String) -> String:
    """`RuntimeError: <family>: ...` naming the image's tier and vendor and
    the table's own refusal (`cause`)."""
    return ("RuntimeError: " + family + ": the " + ("identical" if mode == 1 else "fast") + " GPU image (vendor "
            + String(vendor) + ") keeps device-resident inputs in the x_prep store, but the x_prep image of the "
            + "same tier and vendor is not connected or does not match (" + cause + "); install mojolearn "
            + "binaries of one build")
