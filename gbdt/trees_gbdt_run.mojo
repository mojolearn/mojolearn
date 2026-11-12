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
from core.shared_slot import SharedSlot


struct GbdtBinaryFacts(Defaultable, Movable):
    """Compile-time answers of a loaded GBDT binary (its vendor and numeric
    mode), keyed on the address of the compiled entry that gives them. The
    entries are bound functions a connected binding table holds for the
    life of the process (an extension module is never unloaded), so the
    address names one binary. p3-trees: every predict called the compiled
    entry (and str()) again for the same constant."""
    var vendor: Dict[Int, String]
    var mode: Dict[Int, String]

    def __init__(out self):
        self.vendor = Dict[Int, String]()
        self.mode = Dict[Int, String]()


comptime GBDT_BINARY_FACTS = SharedSlot[GbdtBinaryFacts, "MojolearnTreesGbdtBinaryFactsV1"]


def gbdt_known_vendor(key: Int) raises -> Optional[String]:
    """The vendor answer cached for the binary whose entry is at `key`."""
    var facts = GBDT_BINARY_FACTS.get_or_create_ptr()
    if key in facts[].vendor:
        return Optional(facts[].vendor[key])
    return None


def gbdt_remember_vendor(key: Int, answer: String) raises:
    var facts = GBDT_BINARY_FACTS.get_or_create_ptr()
    facts[].vendor[key] = answer


def gbdt_known_mode(key: Int) raises -> Optional[String]:
    """The numeric mode cached for the binary whose entry is at `key`."""
    var facts = GBDT_BINARY_FACTS.get_or_create_ptr()
    if key in facts[].mode:
        return Optional(facts[].mode[key])
    return None


def gbdt_mode_of_code(key: Int, code: Int) raises -> String:
    """`numeric_mode_used()`'s answer from the binary's mode word, cached
    for that binary; an unknown word refuses."""
    var facts = GBDT_BINARY_FACTS.get_or_create_ptr()
    if code == 0:
        facts[].mode[key] = "fast"
        return "fast"
    if code == 1:
        facts[].mode[key] = "identical"
        return "identical"
    raise Error("RuntimeError: mojolearn: _mojolearn_gbdt returned unknown numeric mode " + String(code))


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
