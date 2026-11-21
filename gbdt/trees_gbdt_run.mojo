# SPDX-License-Identifier: Apache-2.0
"""RUN of the GradientBoosting predict door (bindings/trees_public_gbdt.mojo):
the decisions that read only typed values. The door's UNPACK reads the
estimator's Python fields and the connected binding's entries; these
functions decide on plain values. No Python import:
`tools/python_free_dirs.sh` covers this directory.

Lane doorsplit-trees (2026-11-11): moved from the door. Each function is a
per-call scalar decision or a small per-process cache; there is no
independent work to split across cores.
"""


# The per-binary vendor and numeric-mode cache (`GbdtBinaryFacts`,
# `gbdt_known_vendor` / `gbdt_remember_vendor` / `gbdt_known_mode` /
# `gbdt_mode_of_code`) is gone (lane dw-trees-boost, 2026-11-21): the GBDT
# door reads both words from the resolved native entry of the image
# (`entry_ref_vendor`, `entry_ref_mode`, core/entry_table.mojo), so no door
# calls the binary's `gbdt_vendor` / `gbdt_numeric_mode` Python entries.


def gbdt_mode_checked(clean: String) raises -> String:
    """`_backend.binding`'s check of a stripped, lower-cased mode."""
    if clean != "fast" and clean != "identical":
        raise Error("numeric_mode must be fast, identical or None")
    return clean


def gbdt_text_has_ctr(text: String) -> Bool:
    """`ensemble._has_ctr_records` of a model text: a CTR table or tensor
    CTR registry record."""
    var names: List[String] = ["ctr_columns", "ctr_table", "tensor_ctr_registry"]
    for i in range(3):  # small-loop(names): the three fixed record names
        if ("\n" + names[i] + " ") in text or text.startswith(names[i] + " "):
            return True
    return False


def _type_or_value(error: Error) -> Bool:
    """A refusal Python's `except (TypeError, ValueError)` catches."""
    var message = String(error)
    return not (message.startswith("RuntimeError:") or message.startswith("ImportError:")
                or message.startswith("AttributeError:") or message.startswith("IndexError:")
                or message.startswith("NotImplementedError:") or message.startswith("OverflowError:"))
